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
from smarts_gpt_model import SmartsTokenizer, SPECIAL_TOKENS  # noqa: E402
from sft_model import load_pretrained_for_sft  # noqa: E402
from sft_dataset import SFTDataset, smiles_to_atom_tokens  # noqa: E402

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("WARNING: RDKit not available, validity metrics will be skipped")


THESIS_ROOT = '/home/pyq02mab/Thesis'


def parse_args():
    p = argparse.ArgumentParser(description='Conditional SFT training for SMARTS-GPT')
    p.add_argument('--data', type=str,
                   default=f'{THESIS_ROOT}/data/sft/sft_warmup_filtered.jsonl')
    p.add_argument('--checkpoint', type=str,
                   default=f'{THESIS_ROOT}/pretraining/checkpoints/pubchem_gpt/best_checkpoint.pt')
    p.add_argument('--vocab', type=str,
                   default=f'{THESIS_ROOT}/pretraining/checkpoints/pubchem_gpt/tokenizer.json')
    p.add_argument('--output-dir', type=str, default=f'{THESIS_ROOT}/sft/checkpoints')
    p.add_argument('--max-len', type=int, default=256)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--min-lr', type=float, default=1e-5)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--val-split', type=float, default=0.05)
    p.add_argument('--valid-rate-threshold', type=float, default=0.90)
    p.add_argument('--n-eval-samples', type=int, default=200)
    p.add_argument('--warmup-steps', type=int, default=200)
    p.add_argument('--max-new-tokens', type=int, default=150)
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
                        max_new_tokens=120, temperature=1.0, top_k=40):
    """Autoregressively generate a fragment set conditioned on a SMILES string."""
    model.eval()
    atom_tokens = smiles_to_atom_tokens(smiles, tokenizer)
    if atom_tokens is None:
        return []

    prefix_ids = [SPECIAL_TOKENS['<BOS>']]
    prefix_ids += [tokenizer.token2id.get(t, SPECIAL_TOKENS['<UNK>']) for t in atom_tokens]
    prefix_ids += [sep_id]

    idx = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    eos_id = SPECIAL_TOKENS['<EOS>']
    pad_id = SPECIAL_TOKENS['<PAD>']
    bos_id = SPECIAL_TOKENS['<BOS>']
    unk_id = SPECIAL_TOKENS['<UNK>']

    generated = []
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.max_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        logits[:, pad_id] = float('-inf')
        logits[:, bos_id] = float('-inf')
        logits[:, unk_id] = float('-inf')

        if top_k > 0:
            k = min(top_k, logits.size(-1))
            kth = torch.topk(logits, k)[0][:, -1, None]
            logits = logits.masked_fill(logits < kth, float('-inf'))

        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)

        tok_id = next_tok.item()
        if tok_id == eos_id:
            break
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
                       max_new_tokens=150):
    if not HAS_RDKIT:
        return None
    n_valid = 0
    n_total = 0
    for smiles in smiles_list[:n_samples]:
        frags = generate_fragments(model, tokenizer, sep_id, frag_sep_id, smiles, device,
                                    max_new_tokens=max_new_tokens)
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    base_tokenizer = SmartsTokenizer.load(args.vocab)
    model, tokenizer, sep_id, frag_sep_id, hparams = load_pretrained_for_sft(
        args.checkpoint, base_tokenizer, max_len=args.max_len, device=device
    )
    model.to(device)
    print(f"SEP_ID={sep_id}  FRAG_SEP_ID={frag_sep_id}  vocab_size={tokenizer.vocab_size}")

    tokenizer.save(os.path.join(args.output_dir, 'tokenizer.json'))

    dataset = SFTDataset.from_jsonl(args.data, tokenizer, sep_id, frag_sep_id, max_len=args.max_len)
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

    val_smiles = []
    with open(args.data) as f:
        for line in f:
            val_smiles.append(json.loads(line)['smiles'])
    val_smiles = val_smiles[-args.n_eval_samples:]

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
            max_new_tokens=args.max_new_tokens
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
