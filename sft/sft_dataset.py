"""
sft_dataset.py — Conditional SFT dataset for SMARTS-GPT.

Each training example is one molecule:
    <BOS> atom_prefix_tokens <SEP> frag_1 <FRAG_SEP> frag_2 <FRAG_SEP> ... frag_N <EOS> <PAD>...

The SMILES conditioning prefix is encoded as per-atom SMARTS primitive tokens
(e.g. "[C&H3]", "[c&H1&R]") rather than raw SMILES characters, since the
pretrained vocab (built from SMARTS fragments only) has no bare atom-letter
tokens. This keeps every prefix token inside the pretrained embedding space.

Loss is computed only on the fragment-set target (+ FRAG_SEP + EOS), never on
the conditioning prefix. The mask for that is returned per-example as
`loss_mask`, aligned with the shifted target `y` for causal LM training.
"""

import json
import sys

import torch
from torch.utils.data import Dataset
from rdkit import Chem

sys.path.insert(0, '/home/pyq02mab/Thesis/pretraining')
from smarts_gpt_model import SPECIAL_TOKENS  # noqa: E402


def _atom_token_candidates(atom, ring_count):
    """Yield candidate SMARTS primitive tokens for one atom, most specific first."""
    if atom.GetAtomicNum() == 14:
        sym = '#14'  # vocab encodes Silicon by atomic number, not symbol
    else:
        sym = atom.GetSymbol()
        if atom.GetIsAromatic():
            sym = sym.lower()

    h = atom.GetTotalNumHs()
    chg = atom.GetFormalCharge()

    if chg > 0:
        chg_variants = ([f'&+{chg}'] if chg > 1 else []) + ['&+', '']
    elif chg < 0:
        chg_variants = ([f'&-{abs(chg)}'] if abs(chg) > 1 else []) + ['&-', '']
    else:
        chg_variants = ['']

    if ring_count >= 2:
        ring_variants = [f'&R{ring_count}', '&R', '']
    elif ring_count == 1:
        ring_variants = ['&R', '']
    else:
        ring_variants = ['']

    for rv in ring_variants:
        for cv in chg_variants:
            yield f'[{sym}&H{h}{rv}{cv}]'


def smiles_to_atom_tokens(smiles, tokenizer):
    """SMILES -> list of per-atom SMARTS primitive tokens, or None if RDKit can't parse it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    ring_info = mol.GetRingInfo()
    tokens = []
    for atom in mol.GetAtoms():
        ring_count = ring_info.NumAtomRings(atom.GetIdx())
        tok = '<UNK>'
        for cand in _atom_token_candidates(atom, ring_count):
            if cand in tokenizer.token2id:
                tok = cand
                break
        tokens.append(tok)
    return tokens


class SFTDataset(Dataset):
    """
    Conditional SFT dataset: SMILES (as atom-primitive tokens) -> unique SMARTS fragment set.
    """

    def __init__(self, records, tokenizer, sep_id, frag_sep_id, max_len=256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        pad_id = SPECIAL_TOKENS['<PAD>']
        bos_id = SPECIAL_TOKENS['<BOS>']
        eos_id = SPECIAL_TOKENS['<EOS>']

        skipped_unparseable = 0
        skipped_too_long = 0

        for rec in records:
            atom_tokens = smiles_to_atom_tokens(rec['smiles'], tokenizer)
            if atom_tokens is None:
                skipped_unparseable += 1
                continue

            fragments = sorted(
                rec['fragments'],
                key=lambda f: (len(tokenizer.tokenize(f)), f),
            )

            prefix_ids = [tokenizer.token2id.get(t, SPECIAL_TOKENS['<UNK>']) for t in atom_tokens]

            target_ids = []
            for i, frag in enumerate(fragments):
                if i > 0:
                    target_ids.append(frag_sep_id)
                target_ids.extend(tokenizer.encode(frag, add_special=False))
            target_ids.append(eos_id)

            ids = [bos_id] + prefix_ids + [sep_id] + target_ids

            if len(ids) > max_len:
                skipped_too_long += 1
                continue

            target_start = 1 + len(prefix_ids) + 1  # position of first target token
            target_mask = [0] * len(ids) + [0] * (max_len - len(ids))
            for i in range(target_start, len(ids)):
                target_mask[i] = 1

            padded_ids = ids + [pad_id] * (max_len - len(ids))

            self.samples.append((padded_ids, target_mask))

        if skipped_unparseable > 0:
            print(f"  Skipped {skipped_unparseable} records with unparseable SMILES")
        if skipped_too_long > 0:
            print(f"  Skipped {skipped_too_long} records exceeding max_len={max_len}")
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
    def from_jsonl(cls, path, tokenizer, sep_id, frag_sep_id, max_len=256):
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return cls(records, tokenizer, sep_id, frag_sep_id, max_len=max_len)
