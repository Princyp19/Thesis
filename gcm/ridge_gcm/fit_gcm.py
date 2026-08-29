"""
fit_gcm.py — Classical additive Group-Contribution-Method (GCM) baseline.

Fits a linear (Ridge) model on pre-computed SMARTS fragment counts to predict
a molecular property (e.g. lipophilicity). Uses a Bemis-Murcko scaffold split
so that train/val/test contain disjoint scaffolds, and evaluates with RMSE,
R^2, and Spearman correlation.

Usage:
    python fit_gcm.py \
        --dataset    /path/to/Lipophilicity_with_frags.csv \
        --frags-col  fragments \
        --experiment v1 \
        --output-dir gcm/results/
"""

import argparse
import ast
import json
import os
import subprocess
import time
from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd

from gcm_reward import FrozenGCM


def git_commit():
    """Short git commit hash, or 'unknown' if unavailable."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def parse_args():
    p = argparse.ArgumentParser(description='Fit an additive GCM (Ridge on fragment counts)')
    p.add_argument('--dataset', type=str, required=True,
                    help='Path to dataset file (.csv or .jsonl)')
    p.add_argument('--smiles-col', type=str, default='smiles',
                    help="SMILES column name (default: 'smiles')")
    p.add_argument('--target-col', type=str, default='exp',
                    help="Target property column name (default: 'exp')")
    p.add_argument('--frags-col', type=str, required=True,
                    help='Column containing pre-computed SMARTS fragment lists')
    p.add_argument('--min-freq', type=int, default=3,
                    help='Min fragment frequency (distinct training molecules) for vocab inclusion (default: 3)')
    p.add_argument('--max-frag-atoms', type=int, default=0,
                    help='If >0, drop any fragment with more than this many atoms before '
                         'fitting. Large fragments are mostly singletons (non-reusable) and '
                         'hurt coverage; capping concentrates the vocab on reusable pieces. '
                         '0 = no cap (default).')
    p.add_argument('--output-dir', type=str, default='gcm/results/',
                    help='Output directory (default: gcm/results/)')
    p.add_argument('--experiment', type=str, required=True,
                    help='Short name for this run, e.g. v1, v2, sft (used in output filenames)')
    p.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 42). NOTE: with --split legacy this is '
                         'IGNORED -- that splitter sorts scaffold groups largest-first and '
                         'is fully deterministic, so every seed gives the same partition. '
                         'Only --split paper actually varies with it.')
    p.add_argument('--count-mode', choices=['presence', 'nu'], default='presence',
                    help="'presence' = original binary features. 'nu' = SIMPOL.1 "
                         "multiplicity (Pankow & Asher 2008 Eq.1), counted from the "
                         "molecule by RDKit substructure matching. Opt-in: leaving it "
                         "unset reproduces every recorded result exactly.")
    p.add_argument('--split', choices=['legacy', 'paper'], default='legacy',
                    help="'legacy' = the original largest-first 80/10/10 split, kept so every "
                         "previously recorded result stays reproducible. 'paper' = "
                         "GROVER/GODE scaffold_balanced 80/10/10, seeded -- the only one "
                         "comparable to published Lipophilicity RMSE, and the only one that "
                         "supports the 3-seed mean+/-std the baselines report.")
    return p.parse_args()


def scaffold_split(smiles_list, frac_train=0.8, frac_val=0.1, seed=42):
    scaffolds = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            # RDKit renamed includeChiralAtoms -> includeChirality across versions.
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChiralAtoms=False)
            except TypeError:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        else:
            scaffold = ""
        scaffolds[scaffold].append(idx)

    # Deterministic tie-breaking: sort scaffold groups by (size desc, scaffold smiles)
    # so that groups of equal size land in a stable, seed-independent order.
    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    n = len(smiles_list)
    train_cut = frac_train * n
    val_cut = (frac_train + frac_val) * n

    train_idx, val_idx, test_idx = [], [], []
    for group in scaffold_sets:
        if len(train_idx) + len(group) <= train_cut:
            train_idx.extend(group)
        elif len(train_idx) + len(val_idx) + len(group) <= val_cut:
            val_idx.extend(group)
        else:
            test_idx.extend(group)
    return train_idx, val_idx, test_idx


def parse_fragment_field(raw):
    """Parse a fragments cell into a list of SMARTS strings.

    Accepts Python list literals, JSON arrays, or semicolon-separated strings.
    Returns [] on missing/unparseable values.
    """
    if raw is None:
        return []
    if isinstance(raw, float) and np.isnan(raw):
        return []
    if isinstance(raw, list):
        return raw
    s = str(raw).strip()
    if not s:
        return []

    # Try JSON array first, then Python literal, then semicolon-separated.
    for loader in (json.loads, ast.literal_eval):
        try:
            val = loader(s)
            if isinstance(val, (list, tuple)):
                return list(val)
        except (ValueError, SyntaxError):
            pass

    if ';' in s:
        return [frag.strip() for frag in s.split(';') if frag.strip()]

    # Single fragment with no delimiter.
    return [s]


def validate_fragments(frag_lists):
    """Discard fragments that RDKit cannot parse as SMARTS. Returns cleaned lists."""
    smarts_cache = {}
    cleaned = []
    n_with_valid = 0
    for frags in frag_lists:
        valid = []
        for frag in frags:
            if frag not in smarts_cache:
                smarts_cache[frag] = Chem.MolFromSmarts(frag) is not None
            if smarts_cache[frag]:
                valid.append(frag)
        cleaned.append(valid)
        if valid:
            n_with_valid += 1
    return cleaned, n_with_valid


def filter_by_size(frag_lists, max_atoms):
    """Drop fragments with more than max_atoms atoms. Returns (cleaned lists, n_dropped).

    In this pipeline every atom is written as a bracketed SMARTS primitive
    (e.g. '[C&H2&R]'), so the atom count is exactly the number of '[' characters.
    max_atoms <= 0 disables the filter.
    """
    if max_atoms <= 0:
        return frag_lists, 0
    cleaned = []
    n_dropped = 0
    for frags in frag_lists:
        kept = []
        for frag in frags:
            if frag.count('[') <= max_atoms:
                kept.append(frag)
            else:
                n_dropped += 1
        cleaned.append(kept)
    return cleaned, n_dropped


def build_vocab(train_frag_lists, min_freq):
    """Vocab = fragments appearing in >= min_freq distinct training molecules."""
    mol_counts = defaultdict(int)
    for frags in train_frag_lists:
        for frag in set(frags):
            mol_counts[frag] += 1
    vocab = sorted(f for f, c in mol_counts.items() if c >= min_freq)
    return vocab


_PATT_CACHE = {}


def build_count_matrix(frag_lists, vocab, smiles_list=None, mode='presence'):
    """Feature matrix. 'presence' is the original behaviour; 'nu' is SIMPOL.1 form.

    'presence': X[j,i] = occurrences of fragment i in molecule j's LIST. Because
        both the GT files and generate_smarts.py output are deduplicated, every
        fragment appears exactly once -- so this is effectively binary.

    'nu': X[j,i] = nu_k, the number of times fragment i actually MATCHES molecule
        j (Pankow & Asher 2008, Eq. 1: log Z = b0 + sum_k nu_k b_k). Multiplicity
        comes from the molecule via RDKit, not from the list, so a generator
        cannot inflate it by emitting repeats.

        Two behaviour changes worth knowing. (1) A fragment that does NOT match
        its molecule scores 0, so unfaithful generated fragments silently drop
        out -- an implicit faithfulness filter, which may help or may remove
        signal. (2) Unlike SIMPOL.1's ~30 near-disjoint groups, our 2-5 atom
        SMARTS overlap heavily (205 per ~25-atom molecule), so nu_k is a count
        over an over-complete basis rather than a decomposition.
    """
    f2i = {f: i for i, f in enumerate(vocab)}
    X = np.zeros((len(frag_lists), len(vocab)), dtype=np.float32)

    if mode == 'presence':
        for j, frags in enumerate(frag_lists):
            for frag in frags:
                i = f2i.get(frag)
                if i is not None:
                    X[j, i] += 1
        return X

    if smiles_list is None:
        raise ValueError("mode='nu' needs smiles_list to count substructure matches")
    # Only fragments the molecule was decomposed into are matched (~205 per
    # molecule, not the full ~26k vocab), which keeps this minutes not hours.
    for j, frags in enumerate(frag_lists):
        mol = Chem.MolFromSmiles(smiles_list[j])
        if mol is None:
            continue
        for frag in set(frags):
            i = f2i.get(frag)
            if i is None:
                continue
            if frag not in _PATT_CACHE:
                _PATT_CACHE[frag] = Chem.MolFromSmarts(frag)
            p = _PATT_CACHE[frag]
            if p is None:
                continue
            X[j, i] = len(mol.GetSubstructMatches(p, uniquify=True, maxMatches=500))
    return X


def coverage_and_avg(frag_lists, vocab):
    vocab_set = set(vocab)
    coverages = []
    frag_counts = []
    for frags in frag_lists:
        frag_counts.append(len(frags))
        if frags:
            coverages.append(sum(1 for f in frags if f in vocab_set) / len(frags))
        else:
            coverages.append(0.0)
    mean_coverage = float(np.mean(coverages)) if coverages else 0.0
    avg_frags = float(np.mean(frag_counts)) if frag_counts else 0.0
    return mean_coverage, avg_frags


def load_dataset(path):
    """Load a CSV or JSON-Lines dataset into a DataFrame based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.jsonl', '.json'):
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records)
    return pd.read_csv(path)


def evaluate(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    rho, _ = spearmanr(y_true, y_pred)
    return rmse, r2, float(rho)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    df = load_dataset(args.dataset)
    smiles_list = df[args.smiles_col].tolist()
    y_all = df[args.target_col].to_numpy(dtype=np.float64)

    raw_frags = df[args.frags_col].tolist()
    parsed_frags = [parse_fragment_field(r) for r in raw_frags]
    frag_lists, n_with_valid = validate_fragments(parsed_frags)
    print(f'Fragments: {n_with_valid}/{len(frag_lists)} molecules have >=1 valid SMARTS fragment')

    frag_lists, n_dropped = filter_by_size(frag_lists, args.max_frag_atoms)
    if args.max_frag_atoms > 0:
        print(f'Size filter: dropped {n_dropped} fragments with > {args.max_frag_atoms} atoms')

    if args.split == 'paper':
        import sys
        sys.path.insert(0, '/home/pyq02mab/Thesis/gcm')
        from splits import scaffold_balanced
        train_idx, val_idx, test_idx = scaffold_balanced(smiles_list, seed=args.seed)
        print(f'Split: paper scaffold_balanced 80/10/10 seed={args.seed}')
    else:
        train_idx, val_idx, test_idx = scaffold_split(smiles_list, seed=args.seed)
        print('Split: legacy largest-first 80/10/10 (seed ignored — deterministic)')

    train_frags = [frag_lists[i] for i in train_idx]
    val_frags = [frag_lists[i] for i in val_idx]
    test_frags = [frag_lists[i] for i in test_idx]

    y_train = y_all[train_idx]
    y_val = y_all[val_idx]
    y_test = y_all[test_idx]

    vocab = build_vocab(train_frags, args.min_freq)
    print(f'Vocabulary: {len(vocab)} fragments (min_freq={args.min_freq})')

    tr_smi = [smiles_list[i] for i in train_idx]
    va_smi = [smiles_list[i] for i in val_idx]
    te_smi = [smiles_list[i] for i in test_idx]
    X_train = build_count_matrix(train_frags, vocab, tr_smi, args.count_mode)
    X_val = build_count_matrix(val_frags, vocab, va_smi, args.count_mode)
    X_test = build_count_matrix(test_frags, vocab, te_smi, args.count_mode)
    if args.count_mode == 'nu':
        nz = X_train[X_train > 0]
        print(f'nu counts: mean={nz.mean():.2f} max={X_train.max():.0f} '
              f'(presence mode would make all of these 1.0)')

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    gcm = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')
    gcm.fit(X_train, y_train)

    val_pred = gcm.predict(X_val)
    test_pred = gcm.predict(X_test)

    val_rmse, val_r2, val_spearman = evaluate(y_val, val_pred)
    test_rmse, test_r2, test_spearman = evaluate(y_test, test_pred)

    coverage_val, avg_frags_val = coverage_and_avg(val_frags, vocab)
    coverage_test, avg_frags_test = coverage_and_avg(test_frags, vocab)
    avg_frags_per_mol = float(np.mean([avg_frags_val, avg_frags_test]))

    metrics = {
        'experiment': args.experiment,
        'dataset': 'lipophilicity',
        'split': 'bemis_murcko',
        # WHICH bemis-murcko splitter: the two are near-inverses in difficulty, so a
        # result is meaningless without it. 'paper' also makes `seed` meaningful.
        'count_mode': args.count_mode,
        'split_variant': args.split,
        'split_seed': args.seed if args.split == 'paper' else None,
        # Provenance: what produced this result (survives log deletion).
        'input_dataset': os.path.abspath(args.dataset),
        'frags_col': args.frags_col,
        'min_freq': args.min_freq,
        'git_commit': git_commit(),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'alpha': float(gcm.alpha_),
        'max_frag_atoms': args.max_frag_atoms,
        'n_fragments': len(vocab),
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'val_spearman': val_spearman,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_spearman': test_spearman,
        'coverage_val': coverage_val,
        'coverage_test': coverage_test,
        'avg_frags_per_mol': avg_frags_per_mol,
    }

    out_dir = os.path.join(args.output_dir, args.experiment)
    os.makedirs(out_dir, exist_ok=True)

    frozen = FrozenGCM(
        coefficients=gcm.coef_,
        bias=gcm.intercept_,
        fragment_vocab=vocab,
        experiment=args.experiment,
        alpha=float(gcm.alpha_),
        metrics=metrics,
    )
    pkl_path = os.path.join(out_dir, f'gcm_lipo_{args.experiment}_frozen.pkl')
    frozen.save(pkl_path)

    json_path = os.path.join(out_dir, f'gcm_lipo_{args.experiment}_results.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    top50 = frozen.top_fragments(n=50)
    top50_path = os.path.join(out_dir, f'gcm_lipo_{args.experiment}_top50.csv')
    top50_df = pd.DataFrame(top50, columns=['fragment', 'coefficient'])
    top50_df.to_csv(top50_path, index=False)

    print()
    print(f'Experiment : {args.experiment}')
    print(f'Split      : bemis_murcko (train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})')
    print(f'Fragments  : {len(vocab)} (min_freq={args.min_freq})')
    print(f'Alpha      : {gcm.alpha_}')
    print(f'Val  RMSE  : {val_rmse:.4f}   R²: {val_r2:.4f}   Spearman: {val_spearman:.4f}')
    print(f'Test RMSE  : {test_rmse:.4f}   R²: {test_r2:.4f}   Spearman: {test_spearman:.4f}')
    print(f'Coverage   : val={coverage_val * 100:.1f}%  test={coverage_test * 100:.1f}%')
    print()
    print(f'Saved -> {pkl_path}')
    print(f'Saved -> {json_path}')
    print(f'Saved -> {top50_path}')


if __name__ == '__main__':
    main()
