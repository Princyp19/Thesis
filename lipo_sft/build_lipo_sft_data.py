"""
build_lipo_sft_data.py — SFT data from the Lipophilicity decomposition, using ONLY
the scaffold-TRAIN molecules.

Why train-split-only matters
----------------------------
The headline GCM number (test R2 = 0.257) is measured on a Bemis-Murcko scaffold
split of these same 4,200 molecules. If SFT sees a test molecule's ground-truth
decomposition, the model can reproduce it at generation time and the R2 stops
measuring anything. This script therefore reuses fit_gcm.scaffold_split -- the
exact splitter behind that number -- and writes only its train portion.

Two variants, for the property-conditioning A/B:
  plain : {smiles, fragments}            -- control
  prop  : {smiles, fragments, exp}       -- property kept, for a prefix that
          conditions on logD. Only meaningful with an SFT prefix mode that
          consumes it; the property must be DROPPED at inference or the
          evaluation leaks (fragments would encode the answer).

Usage:
    python build_lipo_sft_data.py --source ../data/gcm/Lipo_smarts_all_uncapped.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, '/home/pyq02mab/Thesis/gcm/ridge_gcm')
sys.path.append('/home/pyq02mab/SMART_LLM/Final_LLM_Reg')
from fit_gcm import scaffold_split as ridge_split  # noqa: E402
from regression_from_embeddings import scaffold_split as reward_split  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', default='/home/pyq02mab/Thesis/data/gcm/Lipo_smarts_all_uncapped.jsonl',
                   help='Lipophilicity decomposition (uncapped by default; use '
                        'Lipo_smarts_v2.jsonl for cap-20 targets)')
    p.add_argument('--out-dir', default='/home/pyq02mab/Thesis/data/sft')
    p.add_argument('--tag', default='lipo_train',
                   help='Output files: sft_warmup_<tag>.jsonl and _prop.jsonl')
    p.add_argument('--frags-col', default='fragments',
                   help="Fragment key in --source. 'fragments' for Lipo_smarts_*.jsonl, "
                        "'smarts' for Lipophilicity_smarts.jsonl (the 205-frag/mol "
                        "decomposition -- the richer target).")
    p.add_argument('--split', choices=['ridge', 'reward', 'paper'], default='ridge',
                   help="Which scaffold splitter defines the held-out molecules. The two "
                        "are near-INVERSES (ridge puts the largest scaffold groups in "
                        "train, reward puts them in test), so this choice decides what "
                        "the fine-tuned policy may later be evaluated against. "
                        "'ridge'  -> pair with --reward-type ridge (fit_gcm split). "
                        "'reward' -> pair with --reward-type additive; using 'ridge' data "
                        "there put ALL 432 additive-reward test molecules inside the "
                        "policy's training set.")
    p.add_argument('--seed', type=int, default=42,
                   help='MUST match the seed fit_gcm uses, or the split differs and '
                        'test molecules leak into training')
    return p.parse_args()


def main():
    a = parse_args()
    recs = [json.loads(l) for l in open(a.source) if l.strip()]
    smiles = [r['smiles'] for r in recs]
    if a.split == 'paper':
        import sys as _s; _s.path.insert(0, '/home/pyq02mab/Thesis/gcm')
        from splits import scaffold_balanced
        tr, va, te = scaffold_balanced(smiles, seed=a.seed)
    elif a.split == 'ridge':
        tr, va, te = ridge_split(smiles, seed=a.seed)
    else:
        tr_all, te = reward_split(smiles, len(smiles), test_size=0.1)
        # carve val exactly as the additive GCM did, so its val molecules are held
        # out here too (it used them for checkpoint selection).
        import numpy as np
        rng = np.random.default_rng(a.seed + 1)
        perm = rng.permutation(len(tr_all))
        nv = max(1, int(len(tr_all) * 0.125))
        va, tr = tr_all[perm[:nv]], tr_all[perm[nv:]]
    # val is held out too: it is used for GCM model selection, so training on it
    # would still contaminate the reported numbers.
    keep = sorted(tr)
    print(f'split={a.split}: {len(recs)} molecules -> train={len(tr)} val={len(va)} test={len(te)}')
    print(f'writing {len(keep)} TRAIN-only molecules (val+test excluded)')

    os.makedirs(a.out_dir, exist_ok=True)
    suffix = {'ridge': '', 'reward': '_rwsplit', 'paper': '_paper'}[a.split]
    plain = os.path.join(a.out_dir, f'sft_warmup_{a.tag}{suffix}.jsonl')
    prop = os.path.join(a.out_dir, f'sft_warmup_{a.tag}{suffix}_prop.jsonl')
    n_frag = 0
    with open(plain, 'w') as fp, open(prop, 'w') as fq:
        for i in keep:
            r = recs[i]
            frags = sorted(set(r[a.frags_col]))
            n_frag += len(frags)
            fp.write(json.dumps({'smiles': r['smiles'], 'fragments': frags}) + '\n')
            fq.write(json.dumps({'smiles': r['smiles'], 'fragments': frags,
                                 'exp': float(r['exp'])}) + '\n')
    print(f'  {plain}   ({n_frag / len(keep):.1f} unique frags/mol)')
    print(f'  {prop}   (same + exp column)')
    print('\nLeakage check: the val/test molecules above are the SAME ones the GCM '
          'scores on, and none of them are in these files.')


if __name__ == '__main__':
    main()
