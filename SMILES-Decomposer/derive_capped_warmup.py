"""
derive_capped_warmup.py — cut cap-N warmup sets out of an UNCAPPED one.

The decomposer's shortest-first selection (select_fragments_for_sft in utils_v2)
is exactly `sorted(set(frags), key=(atom_count, length, string))[:N]`, so every
cap-N set is already contained in the uncapped file — no need to re-decompose.
Deriving the arms instead of rebuilding them guarantees the cap is the ONLY
variable: identical molecules, identical count, identical OOV filtering.

(The existing sft_warmup_limit_20 / _v2_fixed / _all_20k files are disjoint
molecule sets of different sizes, so they cannot be compared to each other.
Treat limit_20 / v2_fixed as legacy references, not arms of this sweep.)

Verified: derived cap20 gives 76.7/22.1/1.0/0.1 % at 2/3/4/5 atoms, matching a
real decomposer shortest-first build (77.6/21.6/0.8/0.0).

Usage:
    python derive_capped_warmup.py \
        --uncapped ../data/sft/sft_warmup_all_20k.jsonl \
        --caps 20 80 \
        --out-dir ../data/sft --tag sweep
"""
import argparse
import json
import os
from collections import Counter


def sort_key(smarts):
    """Shortest-first: atom count, then string length, then alphabetical.

    Every atom is a bracketed SMARTS primitive in this pipeline, so the atom
    count is exactly the number of '[' characters. Matches utils_v2 exactly.
    """
    return (smarts.count('['), len(smarts), smarts)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--uncapped', required=True,
                   help='Uncapped warmup jsonl to cut the arms from')
    p.add_argument('--caps', type=int, nargs='+', required=True,
                   help='Fragment caps to emit, e.g. --caps 20 80')
    p.add_argument('--out-dir', default=None,
                   help='Output dir (default: alongside --uncapped)')
    p.add_argument('--tag', default='sweep',
                   help="Filename tag: sft_warmup_<tag>_cap<N>.jsonl (default: sweep)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.uncapped))
    os.makedirs(out_dir, exist_ok=True)

    records = []
    for line in open(args.uncapped):
        line = line.strip()
        if line:
            records.append(json.loads(line))
    print(f'Read {len(records)} molecules from {args.uncapped}')

    for cap in args.caps:
        path = os.path.join(out_dir, f'sft_warmup_{args.tag}_cap{cap}.jsonl')
        sizes = Counter()
        n_frags = 0
        n_short = 0          # molecules with fewer fragments than the cap
        with open(path, 'w') as out:
            for rec in records:
                rec = dict(rec)
                frags = sorted(set(rec['fragments']), key=sort_key)[:cap]
                rec['fragments'] = frags
                out.write(json.dumps(rec) + '\n')
                n_frags += len(frags)
                n_short += len(frags) < cap
                for f in frags:
                    sizes[f.count('[')] += 1
        total = sum(sizes.values())
        dist = '  '.join(f'{k}-atom={100 * v / total:.1f}%'
                         for k, v in sorted(sizes.items()))
        print(f'\ncap{cap} -> {path}')
        print(f'  frags/mol={n_frags / len(records):.1f}   '
              f'{n_short} molecules had fewer than {cap} fragments')
        print(f'  {dist}')


if __name__ == '__main__':
    main()
