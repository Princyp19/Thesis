"""
train_sft.py — Conditional SFT training for SMARTS-GPT (Phase 1).

Trains SMILES -> unique SMARTS fragment set generation, starting from the
pretrained PubChem-GPT checkpoint. Saves pi_theta_0.pt (trained SFT policy)
and pi_ref.pt (frozen copy, REBEL KL anchor) once validation SMARTS validity
first exceeds 90%.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, '/home/pyq02mab/Thesis/pretraining')
sys.path.insert(0, '/home/pyq02mab/Thesis/sft')
from smarts_gpt_model import SmartsTokenizer, SPECIAL_TOKENS, SMILES_NS, tokenize_smiles  # noqa: E402
from sft_model import load_pretrained_for_sft  # noqa: E402
from sft_dataset import (  # noqa: E402
    SFTDataset, smiles_from_record, smiles_to_prefix_tokens, build_prefix_ids, PREFIX_MODES,
    prop_vocab, PROP_MASK,
)

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("WARNING: RDKit not available, validity metrics will be skipped")


THESIS_ROOT = '/home/pyq02mab/Thesis'


def git_commit():
    """Short git commit hash of the repo, or 'unknown' if unavailable."""
    import subprocess
    try:
        return subprocess.check_output(
            ['git', '-C', THESIS_ROOT, 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def warmup_tag(data_path):
    """Short tag identifying a warmup dataset, e.g. sft_warmup_all_20k.jsonl -> all20k."""
    base = os.path.basename(data_path)
    stem = base.replace('sft_warmup_', '').replace('.jsonl', '')
    aliases = {'filtered': 'filt20', 'v2_fixed': 'v2f40',
               'all_20k': 'all20k', 'all_50k': 'all50k', 'v3_max80': 'v3max80'}
    return aliases.get(stem, stem)


def write_train_config(args, output_dir):
    """Record what this checkpoint was trained on, next to the weights (survives log deletion)."""
    config = {
        'warmup_dataset': os.path.abspath(args.data),
        'warmup_tag': warmup_tag(args.data),
        'base_checkpoint': os.path.abspath(args.checkpoint),
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'max_len': args.max_len,
        'warmup_steps': args.warmup_steps,
        'prefix_mode': args.prefix_mode,
        'seed': 42,
        'slurm_job_id': os.environ.get('SLURM_JOB_ID', 'local'),
        'git_commit': git_commit(),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(output_dir, 'train_config.json'), 'w') as f:
        json.dump(config, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description='Conditional SFT training for SMARTS-GPT')
    p.add_argument('--data', type=str,
                   default=f'{THESIS_ROOT}/data/sft/sft_warmup_filtered.jsonl')
    p.add_argument('--checkpoint', type=str,
                   default=f'{THESIS_ROOT}/pretraining/checkpoints/pubchem_gpt/best_checkpoint.pt')
    p.add_argument('--vocab', type=str,
                   default=f'{THESIS_ROOT}/pretraining/checkpoints/pubchem_gpt/tokenizer.json')
    p.add_argument('--output-dir', type=str, default=f'{THESIS_ROOT}/sft/checkpoints')
    p.add_argument('--prefix-mode', type=str, default='smiles', choices=list(PREFIX_MODES),
                   help="Conditioning-prefix representation. 'smiles' = <SMI>-namespaced "
                        "SMILES tokens (legacy, needs a vocab extension). 'smarts' = the "
                        "whole molecule written as SMARTS in the EXISTING fragment vocab, "
                        "so the prefix already contains the exact atom primitives "
                        "('[C&H2&R]') the model must emit, plus full connectivity. "
                        "Recorded in the checkpoint; generation reads it back.")
    p.add_argument('--prop-col', type=str, default='exp',
                   help="Record key holding the property, used ONLY by "
                        "--prefix-mode smarts_prop. Ignored otherwise.")
    p.add_argument('--max-len', type=int, default=512)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--min-lr', type=float, default=1e-5)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--val-split', type=float, default=0.05)
    p.add_argument('--valid-rate-threshold', type=float, default=0.90)
    p.add_argument('--n-eval-samples', type=int, default=50)
    p.add_argument('--warmup-steps', type=int, default=200)
    p.add_argument('--max-new-tokens', type=int, default=400)
    return p.parse_args()


def masked_ce_loss(logits, targets, loss_mask):
    vocab_size = logits.size(-1)
    per_tok_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size), targets.reshape(-1), reduction='none'
    )
    loss_mask = loss_mask.reshape(-1)
    denom = loss_mask.sum().clamp(min=1)
    return (per_tok_loss * loss_mask).sum() / denom


@torch.no_grad()
def generate_fragments(model, tokenizer, sep_id, frag_sep_id, smiles, device,
                        max_new_tokens=400, temperature=1.0, top_k=40,
                        prefix_mode='smiles', min_unique=0):
    """Autoregressively generate a fragment set conditioned on a SMILES string.

    prefix_mode must match how the checkpoint was TRAINED; generate_smarts.py
    reads it back off the checkpoint so the two cannot drift apart.
    """
    model.eval()
    # prop is deliberately NOT passed: 'smarts_prop' falls back to <PROP>MASK.
    # Generating with the true property present would let the model encode the
    # answer in its fragments, which the GCM reads back -- the evaluation would
    # measure label round-tripping rather than chemistry.
    body = build_prefix_ids(smiles, tokenizer, prefix_mode)
    if body is None:
        return []

    prefix_ids = [SPECIAL_TOKENS['<BOS>']] + body + [sep_id]

    idx = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    eos_id = SPECIAL_TOKENS['<EOS>']
    pad_id = SPECIAL_TOKENS['<PAD>']
    bos_id = SPECIAL_TOKENS['<BOS>']
    unk_id = SPECIAL_TOKENS['<UNK>']

    # SMILES-namespace tokens are input-only; forbid the decoder from emitting
    # them as fragment tokens.
    smiles_ids = [i for t, i in tokenizer.token2id.items() if t.startswith(SMILES_NS)]
    # Property tokens are input-only too; never emit them as fragment tokens.
    smiles_ids += [i for t, i in tokenizer.token2id.items() if t.startswith('<PROP>')]

    # min_unique: suppress EOS until this many DISTINCT fragments exist. Repeats are
    # emitted normally and simply don't count (rejection sampling) -- blocking
    # FRAG_SEP instead forces off-distribution continuations and measurably halves
    # the natural stop (163.6 -> 67.9 emissions) while wrecking faithfulness.
    #
    # The real ceiling is the CONTEXT WINDOW, not this parameter: generation slides
    # idx[:, -model.max_len:], so past ~239 emissions the molecule prefix scrolls out
    # and the model writes blind. max_new_tokens is what enforces that -- keep it
    # under (model.max_len - len(prefix_ids)).
    emitted, cur_tokens = set(), []
    generated = []
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.max_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        logits[:, pad_id] = float('-inf')
        logits[:, bos_id] = float('-inf')
        logits[:, unk_id] = float('-inf')
        if smiles_ids:
            logits[:, smiles_ids] = float('-inf')

        if top_k > 0:
            k = min(top_k, logits.size(-1))
            kth = torch.topk(logits, k)[0][:, -1, None]
            logits = logits.masked_fill(logits < kth, float('-inf'))

        if min_unique > 0 and len(emitted) < min_unique:
            logits[:, eos_id] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)

        tok_id = next_tok.item()
        if tok_id == eos_id:
            break
        if tok_id == frag_sep_id:
            if cur_tokens:
                emitted.add(tuple(cur_tokens))
            cur_tokens = []
        else:
            cur_tokens.append(tok_id)
        generated.append(tok_id)
        idx = torch.cat([idx, next_tok], dim=1)

    # Split generated tail on FRAG_SEP into individual fragment token-id lists.
    fragments = []
    current = []
    for tok_id in generated:
        if tok_id == frag_sep_id:
            if current:
                fragments.append(current)
            current = []
        else:
            current.append(tok_id)
    if current:
        fragments.append(current)

    return [tokenizer.decode(f, skip_special=True) for f in fragments]


def evaluate_validity(model, tokenizer, sep_id, frag_sep_id, smiles_list, device, n_samples,
                       max_new_tokens=400, prefix_mode='smiles'):
    if not HAS_RDKIT:
        return None
    n_valid = 0
    n_total = 0
    for smiles in smiles_list[:n_samples]:
        frags = generate_fragments(model, tokenizer, sep_id, frag_sep_id, smiles, device,
                                    max_new_tokens=max_new_tokens, prefix_mode=prefix_mode)
        for frag in frags:
            n_total += 1
            if Chem.MolFromSmarts(frag) is not None:
                n_valid += 1
    if n_total == 0:
        return 0.0
    return n_valid / n_total


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    write_train_config(args, args.output_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if device == 'cpu' and os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is set but torch.cuda.is_available() is False — "
            "GPU allocated but not usable (bad node/driver?). Aborting instead of "
            "silently running on CPU."
        )

    records = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # 'smarts' prefixes reuse the fragment vocab verbatim, so there is nothing to
    # add; only the '<SMI>'-namespaced mode needs the vocab extended.
    if args.prefix_mode == 'smiles':
        smiles_tokens = sorted({
            t for rec in records for t in tokenize_smiles(smiles_from_record(rec))
        })
        print(f"SMILES conditioning vocab: {len(smiles_tokens)} unique tokens")
    else:
        smiles_tokens = None
        print("Prefix mode 'smarts': whole-molecule SMARTS prefix, no vocab extension")
    extra_tokens = prop_vocab() if args.prefix_mode == 'smarts_prop' else None
    if extra_tokens:
        print(f"Prefix mode 'smarts_prop': +{len(extra_tokens)} property tokens "
              f"(binned logD, plus {PROP_MASK} for inference)")

    base_tokenizer = SmartsTokenizer.load(args.vocab)
    model, tokenizer, sep_id, frag_sep_id, hparams = load_pretrained_for_sft(
        args.checkpoint, base_tokenizer, smiles_tokens=smiles_tokens,
        extra_tokens=extra_tokens, max_len=args.max_len, device=device
    )
    model.to(device)
    print(f"SEP_ID={sep_id}  FRAG_SEP_ID={frag_sep_id}  vocab_size={tokenizer.vocab_size}")

    tokenizer.save(os.path.join(args.output_dir, 'tokenizer.json'))

    dataset = SFTDataset(records, tokenizer, sep_id, frag_sep_id, max_len=args.max_len,
                         prefix_mode=args.prefix_mode,
                         prop_col=args.prop_col if args.prefix_mode == 'smarts_prop' else None)
    n_val = max(200, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {n_train}  Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    val_smiles = [smiles_from_record(rec) for rec in records[-args.n_eval_samples:]]

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    total_steps = args.epochs * len(train_loader)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        min_ratio = args.min_lr / args.lr
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    saved_convergence_ckpt = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        t0 = time.time()

        for x, y, loss_mask in train_loader:
            x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
            logits, _ = model(x)
            loss = masked_ce_loss(logits, y, loss_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, loss_mask in val_loader:
                x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
                logits, _ = model(x)
                val_losses.append(masked_ce_loss(logits, y, loss_mask).item())

        val_valid_rate = evaluate_validity(
            model, tokenizer, sep_id, frag_sep_id, val_smiles, device, args.n_eval_samples,
            max_new_tokens=args.max_new_tokens, prefix_mode=args.prefix_mode
        )

        dt = time.time() - t0
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_valid_rate={val_valid_rate if val_valid_rate is not None else 'N/A'} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | {dt:.1f}s")

        if val_valid_rate is not None and val_valid_rate > args.valid_rate_threshold:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_valid_rate': val_valid_rate,
                'hparams': hparams,
                'sep_id': sep_id,
                'frag_sep_id': frag_sep_id,
                'prefix_mode': args.prefix_mode,
                'tokenizer_path': os.path.join(args.output_dir, 'tokenizer.json'),
            }

            # Unique per-epoch snapshot, never overwritten, for comparing checkpoints later.
            snapshot_name = f"pi_theta_epoch{epoch:03d}_valid{val_valid_rate:.4f}_loss{val_loss:.4f}.pt"
            torch.save(ckpt, os.path.join(args.output_dir, snapshot_name))

            # pi_theta_0.pt always points at the latest epoch above threshold.
            torch.save(ckpt, os.path.join(args.output_dir, 'pi_theta_0.pt'))

            if not saved_convergence_ckpt:
                # pi_ref.pt is the frozen REBEL KL anchor: saved once, at first
                # convergence, and never touched again after this.
                torch.save(ckpt, os.path.join(args.output_dir, 'pi_ref.pt'))
                print(f"  Convergence reached (val_valid_rate={val_valid_rate:.1%} > "
                      f"{args.valid_rate_threshold:.0%}). Saved pi_ref.pt (frozen REBEL KL anchor) "
                      f"to {args.output_dir}")
                saved_convergence_ckpt = True

            print(f"  Saved {snapshot_name} and updated pi_theta_0.pt to epoch {epoch} "
                  f"(val_valid_rate={val_valid_rate:.1%}, val_loss={val_loss:.4f})")

    if not saved_convergence_ckpt:
        print(f"WARNING: training finished without val_valid_rate exceeding "
              f"{args.valid_rate_threshold:.0%}. No pi_theta_0.pt/pi_ref.pt saved.")


if __name__ == '__main__':
    main()
