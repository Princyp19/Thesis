"""
generate_smarts.py — Generate SMARTS fragments from SMILES using a trained
SFT checkpoint (pi_theta_0.pt / pi_ref.pt).

Two modes:
  --smiles "<one SMILES>"   Single molecule, printed to stdout.
  --input-csv <path>        Batch mode: one JSONL record per row, written to
                             --output in the same schema as
                             data/sft/sft_warmup_filtered.jsonl:
                             {"id"?, "smiles", "fragments", "n_valid", "n_total"}

Usage:
    python generate_smarts.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
    python generate_smarts.py --smiles "CCO" --checkpoint sft/checkpoints/job_22579555/pi_theta_0.pt \
        --n-samples 5
    python generate_smarts.py --input-csv data/sft/Lipophilicity.csv \
        --output sft/results/lipophilicity_smarts.jsonl
"""

import argparse
import csv
import json
import os
import sys
import time

import torch

sys.path.insert(0, '/home/pyq02mab/Thesis/pretraining')
sys.path.insert(0, '/home/pyq02mab/Thesis/sft')
from smarts_gpt_model import SmartsTokenizer, SmartsGPT  # noqa: E402
from train_sft import generate_fragments, git_commit  # noqa: E402

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

THESIS_ROOT = '/home/pyq02mab/Thesis'


def parse_args():
    p = argparse.ArgumentParser(description='Generate SMARTS fragments from SMILES')
    p.add_argument('--smiles', type=str, default=None,
                   help='Single SMILES string (mutually exclusive with --input-csv)')
    p.add_argument('--input-csv', type=str, default=None,
                   help='Batch mode: CSV file with a SMILES column')
    p.add_argument('--output', type=str, default=None,
                   help='Batch mode: JSONL output path (required with --input-csv)')
    p.add_argument('--smiles-col', type=str, default='smiles')
    p.add_argument('--limit', type=int, default=None, help='Batch mode: only process first N rows')
    p.add_argument('--log-every', type=int, default=100, help='Batch mode: progress log interval')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--tokenizer', type=str, default=None,
                   help='Defaults to tokenizer_path stored inside the checkpoint')
    p.add_argument('--n-samples', type=int, default=1,
                   help='Number of samples to draw per molecule. In batch mode, >1 '
                        'enables consensus decoding (see --min-consensus).')
    p.add_argument('--min-consensus', type=int, default=1,
                   help='Batch mode: keep only fragments that appear in at least this '
                        'many of the --n-samples draws for a molecule (default: 1, i.e. '
                        'union of all samples). Set e.g. 2 or 3 to filter out one-off '
                        'fragments the model is not confident about.')
    p.add_argument('--max-new-tokens', type=int, default=400)
    p.add_argument('--min-unique', type=int, default=0,
                   help='Suppress EOS until this many DISTINCT fragments have been '
                        'emitted (rejection sampling; repeats are kept but do not count). '
                        '0 = off. Capped in practice by --max-new-tokens, which must stay '
                        'under model.max_len minus the prefix or the molecule scrolls out '
                        'of context and the model generates blind.')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-k', type=int, default=40)
    args = p.parse_args()
    if args.smiles is None and args.input_csv is None:
        p.error('one of --smiles or --input-csv is required')
    if args.input_csv is not None and args.output is None:
        p.error('--output is required with --input-csv')
    if args.min_consensus > args.n_samples:
        p.error(f'--min-consensus ({args.min_consensus}) cannot exceed --n-samples ({args.n_samples})')
    return args


def load_sft_checkpoint(checkpoint_path, tokenizer_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hp = ckpt['hparams']

    tok_path = tokenizer_path or ckpt['tokenizer_path']
    tokenizer = SmartsTokenizer.load(tok_path)

    model = SmartsGPT(
        vocab_size=hp['vocab_size'],
        n_layer=hp['n_layer'],
        n_head=hp['n_head'],
        n_embd=hp['n_embd'],
        max_len=hp['max_len'],
        dropout=hp['dropout'],
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # Checkpoints predating the prefix-mode switch were all trained with the
    # '<SMI>'-namespaced SMILES prefix, so default to that.
    prefix_mode = ckpt.get('prefix_mode', 'smiles')
    print(f"Prefix mode: {prefix_mode}"
          f"{' (default — not recorded in checkpoint)' if 'prefix_mode' not in ckpt else ''}")

    return (model, tokenizer, ckpt['sep_id'], ckpt['frag_sep_id'],
            ckpt.get('val_valid_rate'), prefix_mode)


def run_single(args, model, tokenizer, sep_id, frag_sep_id):
    print(f"SMILES: {args.smiles}\n")
    for i in range(args.n_samples):
        frags = generate_fragments(
            model, tokenizer, sep_id, frag_sep_id, args.smiles, args.device,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
            prefix_mode=args.prefix_mode,
            min_unique=args.min_unique
        )
        print(f"Sample {i + 1}: {len(frags)} fragment(s)")
        for frag in frags:
            if HAS_RDKIT:
                valid = Chem.MolFromSmarts(frag) is not None
                print(f"  [{'VALID' if valid else 'INVALID'}] {frag}")
            else:
                print(f"  {frag}")
        print()


def consensus_fragments(model, tokenizer, sep_id, frag_sep_id, smiles, args):
    """Draw args.n_samples fragment sets for one molecule and keep the fragments
    that appear in at least args.min_consensus of them.

    Each sample contributes a fragment at most once (set semantics), so the
    consensus count is "number of samples that proposed this fragment", not raw
    multiplicity. With n_samples=1/min_consensus=1 this reduces to a single draw.
    Returned fragments are sorted for reproducibility.
    """
    from collections import Counter
    votes = Counter()
    for _ in range(args.n_samples):
        frags = generate_fragments(
            model, tokenizer, sep_id, frag_sep_id, smiles, args.device,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_k=args.top_k, prefix_mode=args.prefix_mode,
            min_unique=args.min_unique
        )
        for frag in set(frags):
            votes[frag] += 1
    return sorted(f for f, c in votes.items() if c >= args.min_consensus)


def run_batch(args, model, tokenizer, sep_id, frag_sep_id):
    with open(args.input_csv, newline='') as f:
        rows = list(csv.DictReader(f))
    if args.limit is not None:
        rows = rows[:args.limit]
    print(f"Read {len(rows)} rows from {args.input_csv}")
    print(f"Sampling: n_samples={args.n_samples}, min_consensus={args.min_consensus}")

    n_valid_total = 0
    n_frag_total = 0
    n_unparseable = 0
    t0 = time.time()

    with open(args.output, 'w') as out_f:
        for i, row in enumerate(rows):
            smiles = row[args.smiles_col]
            frags = consensus_fragments(
                model, tokenizer, sep_id, frag_sep_id, smiles, args,
            )

            if not frags:
                n_unparseable += 1

            record = dict(row)
            record['fragments'] = frags
            if HAS_RDKIT:
                n_valid = sum(1 for frag in frags if Chem.MolFromSmarts(frag) is not None)
                n_valid_total += n_valid
                n_frag_total += len(frags)
                record['n_valid'] = n_valid
                record['n_total'] = len(frags)

            out_f.write(json.dumps(record) + '\n')
            out_f.flush()  # results survive a time-limit kill; progress visible while running

            if (i + 1) % args.log_every == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(rows) - (i + 1)) / rate / 60
                print(f"  {i + 1}/{len(rows)} molecules processed "
                      f"({rate:.2f} mol/s, ETA {eta:.0f} min)", flush=True)

    print(f"\nDone in {time.time() - t0:.1f}s -> {args.output}")
    print(f"Unparseable SMILES (no fragments generated): {n_unparseable}/{len(rows)}")
    if HAS_RDKIT and n_frag_total > 0:
        print(f"Overall fragment validity: {n_valid_total}/{n_frag_total} "
              f"({n_valid_total / n_frag_total:.1%})")

    # Provenance sidecar: records what produced this .jsonl, next to it (survives log deletion).
    meta = {
        'output': os.path.abspath(args.output),
        'input_csv': os.path.abspath(args.input_csv),
        'checkpoint': os.path.abspath(args.checkpoint),
        'prefix_mode': args.prefix_mode,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'n_samples': args.n_samples,
        'min_consensus': args.min_consensus,
        'max_new_tokens': args.max_new_tokens,
        'n_molecules': len(rows),
        'git_commit': git_commit(),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(args.output + '.meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Provenance -> {args.output}.meta.json")


def main():
    args = parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, tokenizer, sep_id, frag_sep_id, val_valid_rate, prefix_mode = load_sft_checkpoint(
        args.checkpoint, args.tokenizer, args.device
    )
    # Taken from the checkpoint, never from a flag: a prefix that disagrees with
    # training silently produces garbage rather than failing.
    args.prefix_mode = prefix_mode
    print(f"Loaded checkpoint: {args.checkpoint}")
    if val_valid_rate is not None:
        print(f"  (reported val_valid_rate at save time: {val_valid_rate:.1%})")
    print(f"Device: {args.device}")

    if args.input_csv is not None:
        run_batch(args, model, tokenizer, sep_id, frag_sep_id)
    else:
        run_single(args, model, tokenizer, sep_id, frag_sep_id)


if __name__ == '__main__':
    main()
