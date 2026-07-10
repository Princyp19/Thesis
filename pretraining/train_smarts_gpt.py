"""
train_smarts_gpt.py — Train SMARTS-GPT from scratch

Usage:
    python train_smarts_gpt.py --data_dir ./data --output_dir ./checkpoints

What this script does:
  1. Loads all .npy files from data_dir
  2. Builds tokenizer from scratch
  3. Trains a small GPT decoder
  4. Saves best checkpoint (by validation loss)
  5. Prints validity metrics every N epochs

Tuned for your dataset:
  - 532k-735k SMARTS patterns
  - 176 unique tokens
  - Max sequence length ~13 tokens
"""

import argparse
import json
import os
import time
import glob
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from smarts_gpt_model import SmartsGPT, SmartsTokenizer, SmartsDataset, SPECIAL_TOKENS

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("WARNING: RDKit not available, validity metrics will be skipped")


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Train SMARTS-GPT from scratch')

    # Data
    p.add_argument('--data_dir', type=str, required=True,
                   help='Directory with train.csv / test.csv / valid.csv (or .npy)')
    p.add_argument('--smarts_column', type=str, default=None,
                   help='CSV column name containing SMARTS (auto-detected if not set)')
    p.add_argument('--output_dir', type=str, default='./checkpoints_smarts_gpt',
                   help='Where to save checkpoints and tokenizer')
    p.add_argument('--max_len', type=int, default=32,
                   help='Max token sequence length (your data max is 13, so 32 is safe)')

    # Model
    p.add_argument('--n_layer', type=int, default=6,
                   help='Number of transformer layers (6 is good for 532k dataset)')
    p.add_argument('--n_head', type=int, default=8,
                   help='Number of attention heads')
    p.add_argument('--n_embd', type=int, default=256,
                   help='Embedding dimension (256 → ~5M params)')
    p.add_argument('--dropout', type=float, default=0.1)

    # Training
    p.add_argument('--batch_size', type=int, default=512,
                   help='Batch size (can be large since sequences are short)')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--max_epochs', type=int, default=50)
    p.add_argument('--val_split', type=float, default=0.05,
                   help='Fraction of data for validation')
    p.add_argument('--patience', type=int, default=10,
                   help='Early stopping patience (epochs)')

    # Evaluation
    p.add_argument('--eval_every', type=int, default=5,
                   help='Evaluate generation quality every N epochs')
    p.add_argument('--n_eval_samples', type=int, default=500,
                   help='Number of samples to generate for validity eval')

    # Generation
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top_k', type=int, default=40)
    p.add_argument('--top_p', type=float, default=0.92)

    return p.parse_args()


# ============================================================================
# DATA LOADING
# ============================================================================

def detect_smarts_column(df):
    """
    Auto-detect which column contains SMARTS strings.
    Tries common names first, then sniffs by content.
    """
    # Common column names to try (case-insensitive)
    candidates = ['smarts', 'SMARTS', 'pattern', 'fragment',
                  'smiles', 'SMILES', 'molecule', 'mol']

    for col in candidates:
        if col in df.columns:
            print(f"  Auto-detected SMARTS column: '{col}'")
            return col

    # Fallback: find first string column that looks like SMARTS
    # (contains brackets or bond chars)
    import re
    smarts_hint = re.compile(r'\[.+\]|[=#\-@]')
    for col in df.columns:
        sample = df[col].dropna().head(10).astype(str)
        if sample.apply(lambda x: bool(smarts_hint.search(x))).mean() > 0.5:
            print(f"  Auto-detected SMARTS column by content: '{col}'")
            return col

    # Last resort: use first column
    print(f"  WARNING: Could not auto-detect SMARTS column. "
          f"Using first column: '{df.columns[0]}'")
    print(f"  Available columns: {list(df.columns)}")
    print(f"  To fix: set --smarts_column <column_name>")
    return df.columns[0]


def load_all_smarts(data_dir, smarts_column=None):
    """
    Load SMARTS from train.csv / test.csv / valid.csv (or any .csv / .npy).
    
    Handles both formats:
      CSV: column auto-detected or specified via smarts_column
      NPY: flat array of strings
    """
    import pandas as pd

    all_smarts = []

    # --- CSV files (your format: train.csv, test.csv, valid.csv) ---
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    # Skip data_summary.csv - it's metadata not training data
    csv_files = [f for f in csv_files
                 if os.path.basename(f) not in ('data_summary.csv',)]

    for f in sorted(csv_files):
        df = pd.read_csv(f)
        print(f"\n  File: {os.path.basename(f)} | "
              f"shape: {df.shape} | columns: {list(df.columns)}")

        col = smarts_column if smarts_column else detect_smarts_column(df)
        smarts_list = df[col].dropna().astype(str).tolist()
        smarts_list = [s.strip() for s in smarts_list if len(s.strip()) > 0]
        all_smarts.extend(smarts_list)
        print(f"  Loaded {len(smarts_list):,} patterns")

    # --- NPY files (fallback / legacy) ---
    npy_files = glob.glob(os.path.join(data_dir, '*.npy'))
    for f in sorted(npy_files):
        data = np.load(f, allow_pickle=True)
        smarts_list = [str(s).strip() for s in data
                       if len(str(s).strip()) > 0]
        all_smarts.extend(smarts_list)
        print(f"  Loaded {len(smarts_list):,} patterns from {os.path.basename(f)}")

    if not all_smarts:
        raise FileNotFoundError(
            f"No CSV or NPY files found in {data_dir}\n"
            f"Expected: train.csv, test.csv, valid.csv (or .npy files)"
        )

    # Deduplicate
    before = len(all_smarts)
    all_smarts = list(dict.fromkeys(all_smarts))  # preserve order, remove dupes
    if before != len(all_smarts):
        print(f"\n  Deduplication: {before:,} → {len(all_smarts):,} "
              f"(removed {before - len(all_smarts):,} duplicates)")

    print(f"\n  ✓ Total unique SMARTS: {len(all_smarts):,}")
    return all_smarts


# ============================================================================
# EVALUATION
# ============================================================================

def compute_validity_metrics(model, tokenizer, n_samples=500, device='cpu',
                              temperature=1.0, top_k=40, top_p=0.92,
                              train_set=None):
    """
    Generate N samples and compute:
      validity   = % passing RDKit SMARTS validation
      uniqueness = % unique among generated
      novelty    = % not in training set
    """
    generated = []
    for _ in range(n_samples):
        smarts = model.generate(tokenizer, max_new_tokens=30,
                                temperature=temperature, top_k=top_k,
                                top_p=top_p, device=device)
        generated.append(smarts)

    # Uniqueness
    unique = list(set(generated))
    uniqueness = len(unique) / len(generated)

    # Validity
    if HAS_RDKIT:
        valid = [s for s in unique if Chem.MolFromSmarts(s) is not None]
        validity = len(valid) / len(generated)
    else:
        valid = unique
        validity = None

    # Novelty (vs training set)
    if train_set is not None:
        train_set_lookup = set(train_set)
        novel = [s for s in unique if s not in train_set_lookup]
        novelty = len(novel) / max(len(unique), 1)
    else:
        novelty = None

    return {
        'validity': validity,
        'uniqueness': uniqueness,
        'novelty': novelty,
        'n_generated': len(generated),
        'n_unique': len(unique),
        'n_valid': len(valid) if HAS_RDKIT else None,
        'examples': valid[:5] if valid else generated[:5],
    }


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    print("\n=== Loading Data ===")
    all_smarts = load_all_smarts(args.data_dir, smarts_column=args.smarts_column)

    # -------------------------------------------------------------------------
    # 2. Build tokenizer
    # -------------------------------------------------------------------------
    print("\n=== Building Tokenizer ===")
    tokenizer = SmartsTokenizer()
    tokenizer.build_vocab(all_smarts)

    # -------------------------------------------------------------------------
    # 3. Create output dir and save tokenizer
    # -------------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer_path = os.path.join(args.output_dir, 'tokenizer.json')
    tokenizer.save(tokenizer_path)

    # -------------------------------------------------------------------------
    # 4. Create dataset
    # -------------------------------------------------------------------------
    print("\n=== Creating Dataset ===")
    dataset = SmartsDataset(all_smarts, tokenizer, max_len=args.max_len)

    n_val = max(1000, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # -------------------------------------------------------------------------
    # 5. Build model
    # -------------------------------------------------------------------------
    print("\n=== Building Model ===")
    model = SmartsGPT(
        vocab_size=tokenizer.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        max_len=args.max_len,
        dropout=args.dropout,
    ).to(device)

    # -------------------------------------------------------------------------
    # 6. Optimizer + scheduler
    # -------------------------------------------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01, betas=(0.9, 0.95))

    # Cosine decay with warmup
    def lr_lambda(step):
        warmup_steps = 500
        total_steps = args.max_epochs * len(train_loader)
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # -------------------------------------------------------------------------
    # 7. Training
    # -------------------------------------------------------------------------
    print("\n=== Training ===")
    print(f"Epochs: {args.max_epochs} | Batch: {args.batch_size} | "
          f"LR: {args.lr} | Patience: {args.patience}")

    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_ppl': [], 'val_ppl': [],
        'generation_quality': [],
    }
    global_step = 0

    for epoch in range(1, args.max_epochs + 1):
        # --- Train ---
        model.train()
        train_losses = []
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())
            global_step += 1

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_ppl = np.exp(train_loss)
        val_ppl = np.exp(val_loss)
        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_ppl'].append(train_ppl)
        history['val_ppl'].append(val_ppl)

        print(f"Epoch {epoch:3d}/{args.max_epochs} | "
              f"train_loss={train_loss:.4f} ppl={train_ppl:.2f} | "
              f"val_loss={val_loss:.4f} ppl={val_ppl:.2f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        # ----------------------------------------------------------------
        # Save epoch checkpoint every epoch (always)
        # Filename encodes epoch + val_ppl for easy comparison
        # ----------------------------------------------------------------
        epoch_ckpt_path = os.path.join(
            args.output_dir,
            f"checkpoint_epoch{epoch:03d}_ppl{val_ppl:.2f}.pt"
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_ppl': val_ppl,
            'hparams': {
                'vocab_size': tokenizer.vocab_size,
                'n_layer': args.n_layer,
                'n_head': args.n_head,
                'n_embd': args.n_embd,
                'max_len': args.max_len,
                'dropout': args.dropout,
            },
        }, epoch_ckpt_path)

        # ----------------------------------------------------------------
        # Track best checkpoint separately
        # ----------------------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_ckpt_path = os.path.join(args.output_dir, 'best_checkpoint.pt')
            # Copy by re-saving with optimizer state for resuming
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_ppl': val_ppl,
                'hparams': {
                    'vocab_size': tokenizer.vocab_size,
                    'n_layer': args.n_layer,
                    'n_head': args.n_head,
                    'n_embd': args.n_embd,
                    'max_len': args.max_len,
                    'dropout': args.dropout,
                },
            }, best_ckpt_path)
            print(f"  ✓ best_checkpoint.pt updated (epoch={epoch}, val_ppl={val_ppl:.2f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

        # ----------------------------------------------------------------
        # Generation quality eval every eval_every epochs
        # Novelty/validity metrics printed but checkpoint already saved above
        # ----------------------------------------------------------------
        if epoch % args.eval_every == 0:
            print(f"\n  --- Generation Quality @ Epoch {epoch} ---")
            metrics = compute_validity_metrics(
                model, tokenizer, n_samples=args.n_eval_samples,
                device=device, temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p,
                train_set=all_smarts,
            )
            history['generation_quality'].append({'epoch': epoch, 'metrics': metrics})

            if metrics['validity'] is not None:
                print(f"  Validity:    {metrics['validity']:.1%}")
            print(f"  Uniqueness:  {metrics['uniqueness']:.1%}")
            if metrics['novelty'] is not None:
                print(f"  Novelty:     {metrics['novelty']:.1%}")
            print(f"  Examples:    {metrics['examples'][:3]}")

            # Re-save this epoch checkpoint with novelty info appended to name
            nov = metrics['novelty']
            if nov is not None:
                named_path = os.path.join(
                    args.output_dir,
                    f"checkpoint_epoch{epoch:03d}_ppl{val_ppl:.2f}_nov{nov:.2f}.pt"
                )
                import shutil
                shutil.copy2(epoch_ckpt_path, named_path)
                os.remove(epoch_ckpt_path)  # replace with named version
                print(f"  Saved: {os.path.basename(named_path)}")
            else:
                print(f"  Saved: {os.path.basename(epoch_ckpt_path)}")
            print()


    # -------------------------------------------------------------------------
    # 8. Save training history
    # -------------------------------------------------------------------------
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    print(f"\nTraining history saved → {history_path}")

    # -------------------------------------------------------------------------
    # 9. Final generation eval on best checkpoint
    # -------------------------------------------------------------------------
    print("\n=== Final Evaluation (Best Checkpoint) ===")
    ckpt = torch.load(os.path.join(args.output_dir, 'best_checkpoint.pt'),
                      map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    metrics = compute_validity_metrics(
        model, tokenizer, n_samples=1000, device=device,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        train_set=all_smarts,
    )

    print(f"\nFinal Results (best checkpoint, epoch {ckpt['epoch']}):")
    if metrics['validity'] is not None:
        print(f"  Validity:    {metrics['validity']:.1%}")
    print(f"  Uniqueness:  {metrics['uniqueness']:.1%}")
    if metrics['novelty'] is not None:
        print(f"  Novelty:     {metrics['novelty']:.1%}")
    print(f"\nSample generated SMARTS:")
    for s in metrics['examples']:
        print(f"  {s}")

    print(f"\n✓ Done. Outputs in: {args.output_dir}/")
    print(f"  best_checkpoint.pt")
    print(f"  tokenizer.json")
    print(f"  training_history.json")


if __name__ == '__main__':
    args = parse_args()
    train(args)