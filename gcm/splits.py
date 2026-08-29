"""
splits.py — the benchmark-comparable scaffold split.

Why this exists
---------------
Three different scaffold splitters were in use, and none of them matched the
protocol the baseline papers report against:

  fit_gcm.scaffold_split                80/10/10, largest groups -> TRAIN,
                                        but `seed` was accepted and never used,
                                        so every seed returned the same split.
  regression_from_embeddings            90/10,    largest groups -> TEST.
  build_warmup_dataset.scaffold_split   a third copy.

The first two are near-inverses: their test sets overlap 0/420. That is why the
additive GCM's ground-truth RMSE (0.8427, reward split) looks worse than ridge's
(0.7956, fit_gcm split) even though its R2 is far better (0.6154 vs 0.5333) --
the reward split's test set carries std 1.359 against fit_gcm's 1.164.

`scaffold_balanced` below reproduces GROVER's split_data(), which GODE
(arXiv:2306.01631) vendors and invokes as `--split_type scaffold_balanced` with
the default sizes (0.8, 0.1, 0.1), over three random seeds, reporting mean+/-std.
Reproducing it is what makes an RMSE here comparable to their Table 2.

The rule: scaffold groups larger than half the val or test budget are "big" and
are placed first (so they land in train, keeping rare scaffolds out of test);
the rest are shuffled. Both classes are shuffled with `seed`, which is what
makes multi-seed reporting possible at all.
"""
from collections import defaultdict
import random

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def murcko(smiles):
    """Bemis-Murcko scaffold SMILES, '' if RDKit cannot parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ''
    # RDKit renamed includeChiralAtoms -> includeChirality across versions.
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChiralAtoms=False)
    except TypeError:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_balanced(smiles_list, sizes=(0.8, 0.1, 0.1), seed=0):
    """GROVER/chemprop `scaffold_balanced`. Returns (train_idx, val_idx, test_idx).

    Unlike a strict largest-first sort this is seed-dependent, so three seeds give
    three genuinely different splits -- the protocol the baselines report.
    """
    if abs(sum(sizes) - 1.0) > 1e-6:
        raise ValueError(f'sizes must sum to 1, got {sizes}')

    scaffolds = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaffolds[murcko(smi)].append(i)

    n = len(smiles_list)
    train_size, val_size, test_size = (sizes[0] * n, sizes[1] * n, sizes[2] * n)

    # "Big" = a group large enough that dropping it into val or test would eat
    # over half that budget. Those go to train, so test stays scaffold-diverse.
    big, small = [], []
    for group in scaffolds.values():
        if len(group) > val_size / 2 or len(group) > test_size / 2:
            big.append(group)
        else:
            small.append(group)

    rng = random.Random(seed)
    rng.shuffle(big)
    rng.shuffle(small)

    train, val, test = [], [], []
    for group in big + small:
        if len(train) + len(group) <= train_size:
            train += group
        elif len(val) + len(group) <= val_size:
            val += group
        else:
            test += group
    return train, val, test
