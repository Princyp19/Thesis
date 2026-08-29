"""
sft_dataset.py — Conditional SFT dataset for SMARTS-GPT.

Each training example is one molecule:
    <BOS> smiles_prefix_tokens <SEP> frag_1 <FRAG_SEP> frag_2 <FRAG_SEP> ... frag_N <EOS> <PAD>...

The SMILES conditioning prefix is encoded with a real atom-level SMILES
tokenizer (see tokenize_smiles), preserving the full molecular graph — bonds,
branches, and ring closures — instead of the old lossy per-atom SMARTS-primitive
"bag of atoms". SMILES tokens are namespaced (SMILES_NS + raw, e.g. '<SMI>C') so
they never alias the SMARTS output vocab; their embeddings are learned during SFT.

Loss is computed only on the fragment-set target (+ FRAG_SEP + EOS), never on
the conditioning prefix. The mask for that is returned per-example as
`loss_mask`, aligned with the shifted target `y` for causal LM training.
"""

import json
import random
import sys

import torch
from torch.utils.data import Dataset
from rdkit import Chem

sys.path.insert(0, '/home/pyq02mab/Thesis/pretraining')
sys.path.insert(0, '/home/pyq02mab/Thesis/SMILES-Decomposer')
from smarts_gpt_model import SPECIAL_TOKENS, SMILES_NS, tokenize_smiles  # noqa: E402
# Same writer the decomposer uses for the fragment targets. Importing it rather
# than reimplementing is load-bearing: the prefix only helps if its atom
# primitives are byte-identical to the ones the model must emit.
from utils_v2 import _smarts_for_atom_set  # noqa: E402


def smiles_from_record(rec):
    """Return a record's SMILES, matching the key under any casing.

    Decomposer output names this key after the source CSV column, so it may be
    'smiles' or 'SMILES'. Only the key is case-insensitive — the SMILES value is
    case-sensitive (lowercase = aromatic, uppercase = aliphatic) and never altered.
    """
    for key in rec:
        if key.lower() == 'smiles':
            return rec[key]
    raise KeyError(
        f"record has no 'smiles' key under any casing; found keys: {list(rec)}"
    )


def smiles_to_prefix_tokens(smiles):
    """SMILES -> list of namespaced SMILES tokens ('<SMI>' + raw), or None if
    RDKit can't parse it. Tokens unseen at train time map to <UNK> at encode time."""
    if Chem.MolFromSmiles(smiles) is None:
        return None
    return [SMILES_NS + t for t in tokenize_smiles(smiles)]


def molecule_to_smarts_prefix_tokens(smiles):
    """SMILES -> the WHOLE molecule written as SMARTS, tokenized in the fragment
    vocabulary. Returns None if RDKit can't parse it.

    Motivation: with the '<SMI>'-namespaced prefix the model must *infer* each
    atom's H-count and ring membership from raw SMILES, and that is measurably
    where it fails -- 95-99% of hallucinated fragments use an atom primitive the
    molecule does not contain, while connectivity errors are ~0%. Those features
    are deterministic given the molecule, so hand them over instead: this prefix
    carries the exact primitives the model must emit ('[C&H2&R]') AND the full
    connectivity (bonds, branches, ring closures), in the existing SMARTS vocab.
    No namespace and no vocabulary extension are needed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    whole = _smarts_for_atom_set(mol, set(range(mol.GetNumAtoms())))
    if not whole:
        return None
    return [whole]          # tokenized by the SMARTS tokenizer at encode time


PROP_NS = '<PROP>'
PROP_MASK = PROP_NS + 'MASK'
PROP_BIN = 0.5


def prop_token(value):
    """Continuous property -> a discrete token, binned to PROP_BIN.

    A transformer cannot consume a float, so logD is bucketed. 0.5 over the
    observed -1.5..4.5 range gives ~13 tokens: coarse enough that each is seen
    often, fine enough to carry real information.
    """
    if value is None:
        return PROP_MASK
    return f'{PROP_NS}{PROP_BIN * round(float(value) / PROP_BIN):+.1f}'


def prop_vocab(lo=-4.0, hi=8.0):
    """Every property token that could occur, so the vocab is fixed up front and
    an unseen test value never maps to <UNK>. Range is deliberately wider than
    Lipophilicity's -1.5..4.5."""
    n = int(round((hi - lo) / PROP_BIN))
    return [PROP_MASK] + [f'{PROP_NS}{lo + i * PROP_BIN:+.1f}' for i in range(n + 1)]


PREFIX_MODES = ('smiles', 'smarts', 'smarts_prop')


def build_prefix_ids(smiles, tokenizer, prefix_mode, prop=None):
    """Conditioning-prefix token ids for one molecule, or None if unparseable.

    'smiles'      -> '<SMI>'-namespaced SMILES tokens (legacy; per-token lookup)
    'smarts'      -> whole-molecule SMARTS, encoded with the SMARTS tokenizer
    'smarts_prop' -> a property token, then the whole-molecule SMARTS.

    WARNING for 'smarts_prop': the property MUST be masked at inference
    (prop=None -> <PROP>MASK). Generating with the true value present lets the
    model encode the answer in its fragment choice, which the GCM then reads
    back -- R2 would rise without the fragments being any more meaningful, and
    predicting a molecule would require already knowing its label.
    """
    if prefix_mode == 'smarts_prop':
        toks = molecule_to_smarts_prefix_tokens(smiles)
        if toks is None:
            return None
        pid = tokenizer.token2id.get(prop_token(prop))
        if pid is None:                       # vocab lacks property tokens
            pid = tokenizer.token2id.get(PROP_MASK, SPECIAL_TOKENS['<UNK>'])
        return [pid] + tokenizer.encode(toks[0], add_special=False)
    if prefix_mode == 'smarts':
        toks = molecule_to_smarts_prefix_tokens(smiles)
        if toks is None:
            return None
        return tokenizer.encode(toks[0], add_special=False)
    toks = smiles_to_prefix_tokens(smiles)
    if toks is None:
        return None
    return [tokenizer.token2id.get(t, SPECIAL_TOKENS['<UNK>']) for t in toks]


class SFTDataset(Dataset):
    """
    Conditional SFT dataset: SMILES (as atom-primitive tokens) -> unique SMARTS fragment set.
    """

    def __init__(self, records, tokenizer, sep_id, frag_sep_id, max_len=512, seed=42,
                 prefix_mode='smiles', prop_col=None):
        if prefix_mode not in PREFIX_MODES:
            raise ValueError(f'prefix_mode must be one of {PREFIX_MODES}, got {prefix_mode!r}')
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.prefix_mode = prefix_mode
        self.samples = []

        pad_id = SPECIAL_TOKENS['<PAD>']
        bos_id = SPECIAL_TOKENS['<BOS>']
        eos_id = SPECIAL_TOKENS['<EOS>']

        skipped_unparseable = 0
        skipped_too_long = 0
        truncated = 0
        rng = random.Random(seed)

        for rec in records:
            prefix_ids = build_prefix_ids(smiles_from_record(rec), tokenizer, prefix_mode,
                                          prop=rec.get(prop_col) if prop_col else None)
            if prefix_ids is None:
                skipped_unparseable += 1
                continue

            # Shuffled, not shortest-first: with size-stratified fragment sets
            # (2-5 atoms) a fixed ascending-length order always places the
            # informative larger fragments last, so a model that predicts EOS
            # early (common on long targets before full convergence) would
            # systematically never be scored/trained on them. A fixed shuffle
            # per molecule spreads sizes evenly across sequence positions.
            fragments = list(rec['fragments'])
            rng.shuffle(fragments)

            # Truncate to fit rather than dropping the record: an over-long
            # molecule keeps as many fragments as the budget allows instead of
            # vanishing from the dataset entirely. What gets dropped otherwise
            # is not a random subset — it is systematically the largest,
            # most fragment-rich molecules.
            prefix_len = 1 + len(prefix_ids) + 1          # <BOS> + prefix + <SEP>
            budget = max_len - prefix_len - 1             # reserve one slot for <EOS>
            if budget <= 0:
                skipped_too_long += 1
                continue

            target_ids = []
            n_kept = 0
            for frag in fragments:
                frag_ids = tokenizer.encode(frag, add_special=False)
                cost = len(frag_ids) + (1 if n_kept > 0 else 0)   # + <FRAG_SEP>
                if len(target_ids) + cost > budget:
                    # Stop at the first fragment that does not fit rather than
                    # skipping it to try shorter ones later: `fragments` is
                    # already shuffled, so keeping a prefix of that shuffle is a
                    # size-unbiased sample. Packing by size would bias the
                    # retained set toward short fragments.
                    truncated += 1
                    break
                if n_kept > 0:
                    target_ids.append(frag_sep_id)
                target_ids.extend(frag_ids)
                n_kept += 1

            if n_kept == 0:
                skipped_too_long += 1
                continue
            target_ids.append(eos_id)

            ids = [bos_id] + prefix_ids + [sep_id] + target_ids

            target_start = 1 + len(prefix_ids) + 1  # position of first target token
            target_mask = [0] * len(ids) + [0] * (max_len - len(ids))
            for i in range(target_start, len(ids)):
                target_mask[i] = 1

            padded_ids = ids + [pad_id] * (max_len - len(ids))

            self.samples.append((padded_ids, target_mask))

        if skipped_unparseable > 0:
            print(f"  Skipped {skipped_unparseable} records with unparseable SMILES")
        if skipped_too_long > 0:
            print(f"  Skipped {skipped_too_long} records whose SMILES prefix alone "
                  f"exceeds max_len={max_len}")
        if truncated > 0:
            print(f"  Truncated {truncated} records to fit max_len={max_len} "
                  f"({100 * truncated / max(1, len(self.samples)):.1f}% of kept records)")
        print(f"  SFTDataset: {len(self.samples)} examples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, target_mask = self.samples[idx]
        ids = torch.tensor(ids, dtype=torch.long)
        target_mask = torch.tensor(target_mask, dtype=torch.float)

        x = ids[:-1]
        y = ids[1:]
        loss_mask = target_mask[1:]
        return x, y, loss_mask

    @classmethod
    def from_jsonl(cls, path, tokenizer, sep_id, frag_sep_id, max_len=512, seed=42,
                   prefix_mode='smiles'):
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return cls(records, tokenizer, sep_id, frag_sep_id, max_len=max_len, seed=seed,
                   prefix_mode=prefix_mode)
