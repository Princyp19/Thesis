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
import sys
import time

import torch

sys.path.insert(0, '/home/pyq02mab/Thesis/pretraining')
sys.path.insert(0, '/home/pyq02mab/Thesis/sft')
from smarts_gpt_model import SmartsTokenizer, SmartsGPT  # noqa: E402
from train_sft import generate_fragments  # noqa: E402

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
    p.add_argument('--checkpoint', type=str,
                   default=f'{THESIS_ROOT}/sft/checkpoints/job_22580112/pi_theta_epoch005_valid0.9987_loss0.2859.pt')
    p.add_argument('--tokenizer', type=str, default=None,
                   help='Defaults to tokenizer_path stored inside the checkpoint')
    p.add_argument('--n-samples', type=int, default=1,
                   help='Single-SMILES mode: number of samples to draw')
    p.add_argument('--max-new-tokens', type=int, default=150)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-k', type=int, default=40)
    args = p.parse_args()
    if args.smiles is None and args.input_csv is None:
        p.error('one of --smiles or --input-csv is required')
    if args.input_csv is not None and args.output is None:
        p.error('--output is required with --input-csv')
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

    return model, tokenizer, ckpt['sep_id'], ckpt['frag_sep_id'], ckpt.get('val_valid_rate')


def run_single(args, model, tokenizer, sep_id, frag_sep_id):
    print(f"SMILES: {args.smiles}\n")
    for i in range(args.n_samples):
        frags = generate_fragments(
            model, tokenizer, sep_id, frag_sep_id, args.smiles, args.device,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
        )
        print(f"Sample {i + 1}: {len(frags)} fragment(s)")
        for frag in frags:
            if HAS_RDKIT:
                valid = Chem.MolFromSmarts(frag) is not None
                print(f"  [{'VALID' if valid else 'INVALID'}] {frag}")
            else:
                print(f"  {frag}")
        print()


def run_batch(args, model, tokenizer, sep_id, frag_sep_id):
    with open(args.input_csv, newline='') as f:
        rows = list(csv.DictReader(f))
    if args.limit is not None:
        rows = rows[:args.limit]
    print(f"Read {len(rows)} rows from {args.input_csv}")

    n_valid_total = 0
    n_frag_total = 0
    n_unparseable = 0
    t0 = time.time()

    with open(args.output, 'w') as out_f:
        for i, row in enumerate(rows):
            smiles = row[args.smiles_col]
            frags = generate_fragments(
                model, tokenizer, sep_id, frag_sep_id, smiles, args.device,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                top_k=args.top_k,
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

            if (i + 1) % args.log_every == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i + 1}/{len(rows)} molecules processed ({rate:.1f} mol/s)")

    print(f"\nDone in {time.time() - t0:.1f}s -> {args.output}")
    print(f"Unparseable SMILES (no fragments generated): {n_unparseable}/{len(rows)}")
    if HAS_RDKIT and n_frag_total > 0:
        print(f"Overall fragment validity: {n_valid_total}/{n_frag_total} "
              f"({n_valid_total / n_frag_total:.1%})")


def main():
    args = parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, tokenizer, sep_id, frag_sep_id, val_valid_rate = load_sft_checkpoint(
        args.checkpoint, args.tokenizer, args.device
    )
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
