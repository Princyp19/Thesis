"""
score_with_frozen_ridge.py — Score a set of fragment lists through a *frozen*
classical (ridge) GCM.

Loads a frozen GCM (coefficients + vocab fixed) and evaluates a dataset of
fragment lists against it. This lets you feed model-generated SMARTS through a
GCM that was fit on ground-truth fragments, so only the fragments change and the
reward model is held constant.

Example:
    python score_with_frozen_ridge.py \
        --frozen  ../results/ridge/gt_uncapped_all/gcm_lipo_gt_uncapped_all_frozen.pkl \
        --dataset ../../sft/results/lipo_smarts_generated_uncapped.jsonl \
        --frags-col fragments --target-col exp
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

from gcm_reward import FrozenGCM
from fit_gcm import load_dataset, parse_fragment_field


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--frozen', required=True, help='Path to frozen GCM .pkl')
    p.add_argument('--dataset', required=True, help='Dataset with fragment lists (.jsonl/.csv)')
    p.add_argument('--frags-col', default='fragments')
    p.add_argument('--target-col', default='exp')
    p.add_argument('--out', default=None, help='Optional path to write metrics JSON')
    return p.parse_args()


def main():
    args = parse_args()
    gcm = FrozenGCM.load(args.frozen)

    df = load_dataset(args.dataset)
    y_true = df[args.target_col].to_numpy(dtype=np.float64)
    frag_lists = [parse_fragment_field(r) for r in df[args.frags_col].tolist()]

    y_pred = np.array([gcm.predict(frags) for frags in frag_lists])
    coverages = np.array([gcm.coverage(frags) for frags in frag_lists])

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    rho, _ = spearmanr(y_true, y_pred)

    metrics = {
        'frozen_experiment': gcm.experiment,
        'frozen_alpha': gcm.alpha,
        'frozen_vocab_size': len(gcm.fragment_vocab),
        'dataset': os.path.basename(args.dataset),
        'n_molecules': len(y_true),
        'rmse': rmse,
        'r2': r2,
        'spearman': float(rho),
        'coverage_mean': float(coverages.mean()),
        'avg_frags_per_mol': float(np.mean([len(f) for f in frag_lists])),
    }
    print(json.dumps(metrics, indent=2))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'\nSaved -> {args.out}')


if __name__ == '__main__':
    main()
