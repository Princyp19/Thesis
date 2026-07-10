"""
Audit the decomposer for quality and uniqueness issues.

Checks performed:
  1. SMARTS parseability (RDKit can parse output)
  2. Match-against-source (each SMARTS actually matches the originating SMILES)
  3. Textual vs semantic uniqueness (canonical-canonical match)
  4. Cross-canonicalization stability (running canonicalize twice should be a no-op)
  5. Fragment size and atom-type distribution
  6. Time per molecule
"""
import time
import random
from collections import Counter, defaultdict
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')   # silence RDKit warnings

from utils import fragments_from_smiles, _canonicalize_smarts

# A representative sample of Lipophilicity-style SMILES (varied sizes, ring systems, heteroatoms)
SAMPLE_SMILES = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",                               # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                             # ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",                            # caffeine
    "Clc1ccc2[nH]c(=O)[nH]c2c1",                               # benzimidazolone, Cl
    "CCN(CC)C(=O)C1=CC=C(C=C1)N",                              # procainamide-ish
    "OC(=O)c1ccccc1O",                                         # salicylic acid
    "CN(C)CCOC(C)c1ccccc1",                                    # diphenhydramine-ish
    "CC(=O)Nc1ccc(O)cc1",                                      # paracetamol
    "OC1=CC=C(C=C1)CCN",                                       # tyramine
    "c1ccc(cc1)CCN",                                           # phenethylamine
    "FC(F)(F)c1ccc(cc1)C(=O)N",                                # trifluoromethyl benzamide
    "CC(C)(C)c1ccc(O)cc1",                                     # 4-tert-butylphenol
    "O=S(=O)(c1ccc(N)cc1)N",                                   # sulfanilamide
    "Cc1ccc2nc(N)sc2c1",                                       # 6-methyl-benzothiazol-2-amine
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",                   # glucose-like, stereo
    "P(=O)(O)(O)OCC",                                          # ethyl phosphate
]


def audit_one(smiles, max_atoms=5):
    """Run the decomposer on one SMILES and check quality of outputs."""
    t0 = time.time()
    frags = fragments_from_smiles(smiles, max_atoms_in_smarts=max_atoms)
    elapsed = time.time() - t0

    mol = Chem.MolFromSmiles(smiles)
    report = {
        "smiles": smiles,
        "n_atoms": mol.GetNumAtoms(),
        "n_frags": len(frags),
        "time_s": elapsed,
        "parseable": 0,
        "matches_source": 0,
        "canonical_stable": 0,
        "semantic_dupes": 0,
        "size_hist": Counter(),
    }

    # Build a semantic-equivalence map: two SMARTS are semantically equal if their
    # match sets on the parent molecule are identical
    semantic_groups = defaultdict(list)

    for s in frags:
        # 1) Parseable?
        patt = Chem.MolFromSmarts(s)
        if patt is None:
            continue
        report["parseable"] += 1

        # 2) Matches source molecule?
        matches = mol.GetSubstructMatches(patt, useChirality=False)
        if matches:
            report["matches_source"] += 1
        report["size_hist"][patt.GetNumAtoms()] += 1

        # 3) Canonical-form stable under second round-trip?
        re_canon = _canonicalize_smarts(s)
        if re_canon == s:
            report["canonical_stable"] += 1

        # 4) Semantic grouping by match-set signature on parent
        match_sig = frozenset(frozenset(m) for m in matches)
        semantic_groups[match_sig].append(s)

    # Count semantic duplicates: textually distinct SMARTS with identical match sets
    for sig, group in semantic_groups.items():
        if len(group) > 1:
            report["semantic_dupes"] += len(group) - 1

    report["sample_dupes"] = [g for g in semantic_groups.values() if len(g) > 1][:3]
    return report


def main():
    print(f"{'SMILES':<50} {'n':>3} {'frags':>5} {'parseable':>9} {'match':>5} {'stable':>6} {'semdup':>6} {'time':>6}")
    print("-" * 100)
    totals = Counter()
    all_size_hist = Counter()
    semdup_examples = []

    for smi in SAMPLE_SMILES:
        try:
            r = audit_one(smi, max_atoms=5)
        except Exception as e:
            print(f"FAILED on {smi}: {e}")
            continue
        print(f"{smi[:50]:<50} {r['n_atoms']:>3} {r['n_frags']:>5} "
              f"{r['parseable']:>9} {r['matches_source']:>5} "
              f"{r['canonical_stable']:>6} {r['semantic_dupes']:>6} {r['time_s']:>5.2f}s")
        totals["frags"] += r["n_frags"]
        totals["parseable"] += r["parseable"]
        totals["matches"] += r["matches_source"]
        totals["stable"] += r["canonical_stable"]
        totals["semdup"] += r["semantic_dupes"]
        all_size_hist.update(r["size_hist"])
        if r["sample_dupes"]:
            semdup_examples.append((smi, r["sample_dupes"]))

    print("-" * 100)
    print(f"TOTAL fragments: {totals['frags']}")
    print(f"  Parseable:       {totals['parseable']:>5} ({100*totals['parseable']/max(1,totals['frags']):.1f}%)")
    print(f"  Match source:    {totals['matches']:>5} ({100*totals['matches']/max(1,totals['frags']):.1f}%)")
    print(f"  Canon stable:    {totals['stable']:>5} ({100*totals['stable']/max(1,totals['frags']):.1f}%)")
    print(f"  Semantic dupes:  {totals['semdup']:>5} ({100*totals['semdup']/max(1,totals['frags']):.1f}%)")
    print()
    print("Fragment size distribution (heavy atoms):")
    for sz in sorted(all_size_hist):
        print(f"  {sz}: {all_size_hist[sz]}")
    print()
    print("Examples of semantic duplicates (textually different, same match-set):")
    for smi, dupes in semdup_examples[:3]:
        print(f"\n  Parent SMILES: {smi}")
        for group in dupes[:2]:
            print(f"    Group ({len(group)} variants of the same substructure):")
            for s in group:
                print(f"      {s}")


if __name__ == "__main__":
    main()
