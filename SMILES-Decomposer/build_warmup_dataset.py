"""
Build a (SMILES, fragment_list) dataset for SMARTS-GPT warm-up.

Output: a JSONL file where each line is
    {"smiles": <str>, "fragments": [<smarts>, <smarts>, ...]}

Why per-molecule fragment lists (not flat pairs):
  - During SFT, sample ONE fragment per epoch per molecule.
  - This gives the model exposure to many fragments across epochs without
    blowing up dataset size or biasing toward molecules with many fragments.
  - Also lets you do scaffold splitting on the molecule level cleanly.

Also produces a quality report.
"""
import argparse
import json
import random
import time
from collections import Counter
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')
from utils_v2 import fragments_from_smiles, select_fragments_for_sft

import re

TOK_RE = re.compile(r'(\[[^\]]+\]|Br|Cl|Si|Se|@@|>>|->)|(.)', re.DOTALL)

def tokenize(s):
    return [m.group() for m in TOK_RE.finditer(s)]

def load_vocab(vocab_path):
    with open(vocab_path) as f:
        data = json.load(f)
    if "token2id" in data:
        data = data["token2id"]
    vocab = set(data.keys())
    assert len(vocab) > 150, f"Vocab loaded only {len(vocab)} tokens — check file"
    return vocab


def scaffold_of(smiles):
    """Bemis–Murcko scaffold SMILES (empty string for acyclic)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaf, canonical=True)


def scaffold_split(smiles_list, frac_train=0.8, frac_val=0.1, seed=0):
    """Bemis-Murcko scaffold split: largest scaffolds go to train, smallest to test."""
    import random
    rng = random.Random(seed)
    by_scaffold = {}
    for smi in smiles_list:
        scaf = scaffold_of(smi) or smi   # singletons get their own group
        by_scaffold.setdefault(scaf, []).append(smi)

    # Sort groups by size descending (Chemprop convention)
    groups = sorted(by_scaffold.values(), key=lambda g: -len(g))
    n = len(smiles_list)
    n_train = int(frac_train * n)
    n_val = int(frac_val * n)

    train, val, test = [], [], []
    for grp in groups:
        if len(train) + len(grp) <= n_train:
            train.extend(grp)
        elif len(val) + len(grp) <= n_val:
            val.extend(grp)
        else:
            test.extend(grp)
    return train, val, test


def build_dataset(
    smiles_list,
    max_atoms: int = 5,
    min_atoms: int = 2,
    allowed_atoms: list = None,
    output_path: str = "warmup_pairs.jsonl",
    max_frags: int = 20,
    known_vocab: set = None,
):
    """Decompose each SMILES, write JSONL, return quality stats."""
    rows = []
    failed = []
    n_atoms_per_mol = []
    n_frags_per_mol = []
    global_frag_counter = Counter()
    smarts_lengths = []

    t0 = time.time()
    for smi in tqdm(smiles_list):
        try:
            pairs = fragments_from_smiles(smi, max_atoms, min_atoms, allowed_atoms)
            if not pairs:
                failed.append((smi, "no fragments"))
                continue
            mol = Chem.MolFromSmiles(smi)
            smarts_only = select_fragments_for_sft(
                pairs,
                max_fragments=max_frags,
                known_vocab=known_vocab,
                tokenize_fn=tokenize,
            )
            if not smarts_only:
                failed.append((smi, "all fragments OOV"))
                continue
            rows.append({"smiles": smi, "fragments": smarts_only})
            n_atoms_per_mol.append(mol.GetNumAtoms())
            n_frags_per_mol.append(len(smarts_only))
            smarts_lengths.extend(len(s) for s in smarts_only)
            global_frag_counter.update(smarts_only)
        except Exception as e:
            failed.append((smi, str(e)))

    elapsed = time.time() - t0

    # Write JSONL
    with open(output_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Stats
    def pct(xs, p):
        xs = sorted(xs)
        return xs[int(len(xs) * p / 100)]

    n_unique_global = len(global_frag_counter)
    n_total_emissions = sum(global_frag_counter.values())
    singletons = sum(1 for c in global_frag_counter.values() if c == 1)

    print()
    print("=" * 60)
    print(f"Dataset stats (n={len(rows)} molecules, {elapsed:.1f}s)")
    print("=" * 60)
    print(f"  Failed: {len(failed)}")
    print(f"  Atoms/molecule:     median={pct(n_atoms_per_mol,50)}, "
          f"p90={pct(n_atoms_per_mol,90)}, max={max(n_atoms_per_mol)}")
    print(f"  Fragments/molecule: median={pct(n_frags_per_mol,50)}, "
          f"p90={pct(n_frags_per_mol,90)}, max={max(n_frags_per_mol)}")
    print(f"  SMARTS length:      median={pct(smarts_lengths,50)} chars, "
          f"p90={pct(smarts_lengths,90)}, max={max(smarts_lengths)}")
    print(f"  Global unique fragments: {n_unique_global}")
    print(f"  Total emissions:         {n_total_emissions}")
    print(f"  Avg occurrences/frag:    {n_total_emissions/n_unique_global:.2f}")
    print(f"  Singletons (occ=1):      {singletons} ({100*singletons/n_unique_global:.1f}%)")
    print(f"\nOutput: {output_path}")
    return rows, failed


def build_dataset_csv(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    max_atoms: int = 5,
    min_atoms: int = 2,
    allowed_atoms: list = None,
    output_path: str = "warmup_pairs.jsonl",
    max_frags: int = 20,
    known_vocab: set = None,
):
    """Like build_dataset but reads a CSV and preserves all existing columns.

    Each JSONL record contains every original CSV column plus a 'fragments' key.
    """
    # df = pd.read_csv(csv_path)
    extra_cols = [c for c in df.columns if c != smiles_col]

    rows = []
    failed = []
    n_atoms_per_mol = []
    n_frags_per_mol = []
    global_frag_counter = Counter()
    smarts_lengths = []

    t0 = time.time()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        smi = row[smiles_col]
        if not isinstance(smi, str) or not smi.strip():
            failed.append((smi, "empty smiles"))
            continue
        try:
            pairs = fragments_from_smiles(smi, max_atoms, min_atoms, allowed_atoms)
            if not pairs:
                failed.append((smi, "no fragments"))
                continue
            mol = Chem.MolFromSmiles(smi)
            smarts_only = select_fragments_for_sft(
                pairs,
                max_fragments=max_frags,
                known_vocab=known_vocab,
                tokenize_fn=tokenize,
            )
            if not smarts_only:
                failed.append((smi, "all fragments OOV"))
                continue
            record = {smiles_col: smi}
            for col in extra_cols:
                record[col] = row[col]
            record["fragments"] = smarts_only
            rows.append(record)
            n_atoms_per_mol.append(mol.GetNumAtoms())
            n_frags_per_mol.append(len(smarts_only))
            smarts_lengths.extend(len(s) for s in smarts_only)
            global_frag_counter.update(smarts_only)
        except Exception as e:
            failed.append((smi, str(e)))

    elapsed = time.time() - t0

    with open(output_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    def pct(xs, p):
        xs = sorted(xs)
        return xs[int(len(xs) * p / 100)]

    n_unique_global = len(global_frag_counter)
    n_total_emissions = sum(global_frag_counter.values())
    singletons = sum(1 for c in global_frag_counter.values() if c == 1)

    print()
    print("=" * 60)
    print(f"Dataset stats (n={len(rows)} molecules, {elapsed:.1f}s)")
    print("=" * 60)
    print(f"  Failed: {len(failed)}")
    print(f"  Atoms/molecule:     median={pct(n_atoms_per_mol,50)}, "
          f"p90={pct(n_atoms_per_mol,90)}, max={max(n_atoms_per_mol)}")
    print(f"  Fragments/molecule: median={pct(n_frags_per_mol,50)}, "
          f"p90={pct(n_frags_per_mol,90)}, max={max(n_frags_per_mol)}")
    print(f"  SMARTS length:      median={pct(smarts_lengths,50)} chars, "
          f"p90={pct(smarts_lengths,90)}, max={max(smarts_lengths)}")
    print(f"  Global unique fragments: {n_unique_global}")
    print(f"  Total emissions:         {n_total_emissions}")
    print(f"  Avg occurrences/frag:    {n_total_emissions/n_unique_global:.2f}")
    print(f"  Singletons (occ=1):      {singletons} ({100*singletons/n_unique_global:.1f}%)")
    print(f"\nOutput: {output_path}")
    return rows, failed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV with a 'smiles' column")
    p.add_argument("--smiles-col", default="SMILES")
    p.add_argument("--max-atoms", type=int, default=5)
    p.add_argument("--min-atoms", type=int, default=2)
    p.add_argument("--allowed-atoms", default="C,N,O,Cl,S,F,Br,I,P,Si,B",
                   help="Comma-separated list; molecules with other atoms are skipped")
    p.add_argument("--out-prefix", default="sft_warmup")
    p.add_argument("--scaffold-split", action="store_true",
                   help="Produce train/val/test JSONL via Bemis-Murcko split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vocab", default=None,
                   help="Path to vocab.json — filters out OOV-containing fragments")
    p.add_argument("--max-frags", type=int,
                   help="Max fragments per molecule (cap, shortest first)")
    p.add_argument("--sample-size", type=int, default=None,
                   help="Randomly subsample this many molecules (seeded by --seed)")
    args = p.parse_args()

    df = pd.read_csv(args.csv, sep=",")
    smiles_list = df[args.smiles_col].dropna().drop_duplicates().tolist()
    print(f"Loaded {len(smiles_list)} unique SMILES from {args.csv}")

    if args.sample_size is not None and args.sample_size < len(df):
        df = df.sample(n=args.sample_size, random_state=args.seed)
        print(f"Subsampled to {len(df)} molecules (seed={args.seed})")

    allowed = [a.strip() for a in args.allowed_atoms.split(",")] if args.allowed_atoms else None

    # Load vocab for OOV filtering
    known_vocab = load_vocab(args.vocab) if args.vocab else None

    if args.scaffold_split:
        train, val, test = scaffold_split(smiles_list, seed=args.seed)
        print(f"Scaffold split: train={len(train)}, val={len(val)}, test={len(test)}")
        for name, lst in [("train", train), ("val", val), ("test", test)]:
            print(f"\n--- {name} ---")
            build_dataset(lst, args.max_atoms, args.min_atoms, allowed,
                          f"{args.out_prefix}_{name}.jsonl",
                          max_frags=args.max_frags, known_vocab=known_vocab)
    else:
        # build_dataset(smiles_list, args.max_atoms, args.min_atoms, allowed,
        #               f"{args.out_prefix}.jsonl",
        #               max_frags=args.max_frags, known_vocab=known_vocab)

        rows, failed = build_dataset_csv(
            df=df,
            smiles_col=args.smiles_col,
            allowed_atoms=allowed,
            output_path=f"{args.out_prefix}.jsonl",
            max_frags=args.max_frags, known_vocab=known_vocab)



if __name__ == "__main__":
    main()
