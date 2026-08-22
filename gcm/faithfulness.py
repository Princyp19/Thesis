"""
faithfulness.py — does a generated SMARTS fragment actually occur in the molecule
it was generated for?

Neither GCM checks this: the ridge model counts fragments by vocabulary lookup and
the neural model embeds the fragment string, so a fragment that does not match its
molecule is still scored. Measured match rates on generated output run 19-75%
(ground truth is 100% by construction), so an RL policy optimising an unguarded
reward can raise its score with fragments that do not describe the molecule at all.

`unfaithful_frac` is the penalty term for that: the fraction of a fragment set that
does NOT substructure-match the molecule. Reward-model agnostic, so the ridge and
neural rewards can share it. Everything is cached — REBEL scores the same
(smiles, fragment) pairs many times over a run.
"""

from rdkit import Chem, RDLogger

# Generated SMARTS is frequently malformed; RDKit logs every parse failure to
# stderr, which would drown the training log. We count failures ourselves instead.
RDLogger.DisableLog('rdApp.*')

_patt_cache = {}
_mol_cache = {}
_match_cache = {}


def _patt(smarts):
    """Parsed SMARTS pattern, or None if unparseable (cached)."""
    if smarts not in _patt_cache:
        try:
            _patt_cache[smarts] = Chem.MolFromSmarts(smarts)
        except Exception:
            _patt_cache[smarts] = None
    return _patt_cache[smarts]


def _mol(smiles):
    """Parsed molecule, or None if unparseable (cached)."""
    if smiles not in _mol_cache:
        try:
            _mol_cache[smiles] = Chem.MolFromSmiles(smiles)
        except Exception:
            _mol_cache[smiles] = None
    return _mol_cache[smiles]


def matches(smiles, smarts):
    """True if `smarts` substructure-matches `smiles`.

    Unparseable fragments count as NOT matching — they are malformed output and
    should be penalised, not excused.
    """
    key = (smiles, smarts)
    if key not in _match_cache:
        mol, patt = _mol(smiles), _patt(smarts)
        if mol is None or patt is None:
            _match_cache[key] = False
        else:
            try:
                _match_cache[key] = bool(mol.HasSubstructMatch(patt))
            except Exception:
                _match_cache[key] = False
    return _match_cache[key]


def faithful_frac(smiles, fragments):
    """Fraction of `fragments` that actually occur in `smiles` (1.0 = all match).

    An empty fragment set scores 0.0: it describes nothing.
    """
    if not fragments:
        return 0.0
    return sum(1 for f in fragments if matches(smiles, f)) / len(fragments)


def unfaithful_frac(smiles, fragments):
    """Fraction of `fragments` that do NOT occur in `smiles` — the penalty term.

    An empty fragment set scores 1.0 (fully unfaithful), so a policy cannot dodge
    the penalty by emitting nothing.
    """
    if not fragments:
        return 1.0
    return 1.0 - faithful_frac(smiles, fragments)
