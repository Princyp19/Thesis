"""
Decomposer v2 — same idea as utils.py, but:
  - dedup by match-set signature on the parent (kills semantic duplicates)
  - clears atom map numbers after use
  - optional possible_atom_list filter actually works
  - random sampling takes a seed and never crashes on small molecules
  - returns rich tuples so you can inspect / build datasets cleanly
"""
import random
from itertools import combinations
import re
from rdkit import Chem


def _atom_smarts(atom: Chem.Atom) -> str:
    """Bracketed SMARTS for `atom` with H-count, ring membership, formal charge."""
    sym = atom.GetSymbol().lower() if atom.GetIsAromatic() else atom.GetSymbol()
    parts = [sym, f"H{atom.GetTotalNumHs()}"]

    if atom.IsInRing():
        ring_info = atom.GetOwningMol().GetRingInfo()
        n_rings = sum(1 for ring in ring_info.AtomRings() if atom.GetIdx() in ring)
        parts.append("R" if n_rings == 1 else f"R{n_rings}")   # R for single-ring, R2/R3/... for fused

    q = atom.GetFormalCharge()
    if q:
        parts.append(("+" if q > 0 else "-") + (str(abs(q)) if abs(q) > 1 else ""))

    return f"[{';'.join(parts)}]"


def _connected(indices, nbrs):
    if len(indices) == 1:
        return True
    start = next(iter(indices))
    seen, stack = {start}, [start]
    while stack:
        a = stack.pop()
        for b in nbrs[a]:
            if b in indices and b not in seen:
                seen.add(b); stack.append(b)
    return len(seen) == len(indices)


_MAP_RE = re.compile(r"\[[^\]]+:(\d+)\]")


def _smarts_for_atom_set(mol, aset):
    """Generate a SMARTS string for the connected atom subset `aset` of mol."""
    # Tag atoms with map numbers so we can find them in the output SMARTS
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    for idx in aset:
        mol.GetAtomWithIdx(idx).SetAtomMapNum(idx + 1)

    bonds = [b.GetIdx() for b in mol.GetBonds()
             if {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} <= aset]
    raw = Chem.MolFragmentToSmarts(mol, list(aset), bonds, isomericSmarts=True)

    def repl(m):
        idx = int(m.group(1)) - 1
        return _atom_smarts(mol.GetAtomWithIdx(idx))

    cooked = _MAP_RE.sub(repl, raw)
    cooked = cooked.replace(';', '&')

    # Clean up: clear map numbers so the caller's mol isn't polluted
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)

    return cooked


def fragments_from_smiles(
    smiles: str,
    max_atoms: int = 5,
    min_atoms: int = 1,
    allowed_atoms: list = None,
) -> list[tuple[str, frozenset]]:
    """
    Enumerate connected SMARTS fragments of `smiles` with min_atoms ≤ k ≤ max_atoms.
    Returns a list of (smarts_string, match_set_signature) tuples.

    `allowed_atoms`: if given, skip the molecule entirely if it contains any
                     atom outside this list.
    `match_set_signature` is a frozenset of frozensets of parent-atom-indices.
    Two fragments with the same signature are semantically identical and should
    not both appear in a fine-tuning dataset.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")

    if allowed_atoms is not None:
        syms = {a.GetSymbol() for a in mol.GetAtoms()}
        if not syms.issubset(set(allowed_atoms)):
            return []

    n = mol.GetNumAtoms()
    nbrs = {i: [nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors()]
            for i in range(n)}

    # Map: signature -> (smarts, source_atom_set). First-seen wins.
    by_sig = {}

    for k in range(min_atoms, min(max_atoms, n) + 1):
        for combo in combinations(range(n), k):
            aset = set(combo)
            if not _connected(aset, nbrs):
                continue

            smarts = _smarts_for_atom_set(mol, aset)
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                continue

            matches = mol.GetSubstructMatches(patt, useChirality=False, uniquify=True)
            if not matches:
                continue
            sig = frozenset(frozenset(m) for m in matches)

            # Prefer shorter SMARTS for ties — easier for the model to learn
            if sig not in by_sig or len(smarts) < len(by_sig[sig][0]):
                by_sig[sig] = (smarts, frozenset(aset))

    return [(s, sig) for sig, (s, _) in by_sig.items()]


def sample_fragments_for_smiles(
    smiles: str,
    n_samples: int = 5,
    max_atoms: int = 5,
    min_atoms: int = 2,           # skip single-atom trivia like [C;H3]
    allowed_atoms: list = None,
    rng: random.Random = None,
) -> list[str]:
    """
    Return up to `n_samples` unique fragments from `smiles`.
    Returns FEWER than n_samples if the molecule has fewer unique fragments
    (no crash). Deterministic given an `rng`.
    """
    rng = rng or random.Random(0)
    frags = fragments_from_smiles(smiles, max_atoms, min_atoms, allowed_atoms)
    smarts_only = [s for s, _ in frags]
    if len(smarts_only) <= n_samples:
        return sorted(smarts_only)
    return sorted(rng.sample(smarts_only, n_samples))


def is_vocab_compatible(smarts: str, known_vocab: set, tokenize_fn) -> bool:
    """Return True if every token in this SMARTS string exists in the vocabulary."""
    return all(tok in known_vocab for tok in tokenize_fn(smarts))


def select_fragments_for_sft(
    fragments: list,
    max_fragments: int = 20,
    known_vocab: set = None,
    tokenize_fn=None,
) -> list:
    """
    From a list of (smarts_string, match_set) tuples,
    return up to max_fragments SMARTS strings:
      1. Filter out any fragment containing OOV tokens
      2. Deduplicate by SMARTS string
      3. Sort shortest first (more general patterns first)
      4. Cap at max_fragments

    No chemistry filtering — REBEL decides what's meaningful.
    """
    # Extract SMARTS strings only
    smarts_list = [s for s, _ in fragments]

    # Deduplicate
    smarts_list = list(set(smarts_list))

    # Filter OOV if vocab provided
    if known_vocab is not None and tokenize_fn is not None:
        smarts_list = [
            s for s in smarts_list
            if is_vocab_compatible(s, known_vocab, tokenize_fn)
        ]

    # Sort: shortest first (most general), then alphabetical for determinism
    smarts_list.sort(key=lambda s: (s.count("["), len(s), s))

    return smarts_list[:max_fragments]