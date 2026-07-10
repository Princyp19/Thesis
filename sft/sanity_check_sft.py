"""
sanity_check_sft.py — Quick standalone validation of the SFT pipeline before a full run.

- Loads 100 examples from the data file
- Runs one forward pass and computes masked loss
- Explicitly logs one example's loss_mask to confirm the conditioning prefix is zeroed
- Generates 5 fragment sets for aspirin and checks each fragment with RDKit
"""

import json
import sys

import torch

sys.path.insert(0, '/home/pyq02mab/Thesis/pretraining')
sys.path.insert(0, '/home/pyq02mab/Thesis/sft')
from smarts_gpt_model import SmartsTokenizer  # noqa: E402
from sft_model import load_pretrained_for_sft  # noqa: E402
from sft_dataset import SFTDataset  # noqa: E402
from train_sft import masked_ce_loss, generate_fragments  # noqa: E402

from rdkit import Chem

THESIS_ROOT = '/home/pyq02mab/Thesis'
CHECKPOINT = f'{THESIS_ROOT}/pretraining/checkpoints/pubchem_gpt/checkpoint_epoch010_ppl3.75_nov0.38.pt'
VOCAB = f'{THESIS_ROOT}/pretraining/checkpoints/pubchem_gpt/tokenizer.json'
DATA = f'{THESIS_ROOT}/data/sft/sft_warmup_filtered.jsonl'
ASPIRIN = 'CC(=O)Oc1ccccc1C(=O)O'


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    base_tokenizer = SmartsTokenizer.load(VOCAB)
    model, tokenizer, sep_id, frag_sep_id, hparams = load_pretrained_for_sft(
        CHECKPOINT, base_tokenizer, max_len=256, device=device
    )
    model.to(device)
    print(f"Model loaded. vocab_size={tokenizer.vocab_size} max_len={model.max_len} "
          f"SEP_ID={sep_id} FRAG_SEP_ID={frag_sep_id}")

    records = []
    with open(DATA) as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            records.append(json.loads(line))

    dataset = SFTDataset(records, tokenizer, sep_id, frag_sep_id, max_len=256)

    x, y, loss_mask = dataset[0]
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)
    loss_mask = loss_mask.unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(x)
        loss = masked_ce_loss(logits, y, loss_mask)
    print(f"\nForward pass OK. logits shape={tuple(logits.shape)}  masked_loss={loss.item():.4f}")

    print("\n--- loss_mask check for example 0 (aligned with y = ids[1:]) ---")
    print(f"y tokens:      {y[0].tolist()[:40]} ...")
    print(f"loss_mask:     {[int(v) for v in loss_mask[0].tolist()][:40]} ...")
    n_zero = int((loss_mask[0] == 0).sum().item())
    n_one = int((loss_mask[0] == 1).sum().item())
    print(f"loss_mask summary: {n_zero} positions masked out (prefix/SEP/PAD), "
          f"{n_one} positions weighted (fragment-set target + FRAG_SEP + EOS)")

    print(f"\n--- Generating 5 fragment sets for aspirin: {ASPIRIN} ---")
    for i in range(5):
        frags = generate_fragments(model, tokenizer, sep_id, frag_sep_id, ASPIRIN, device)
        print(f"\nSample {i+1}: {len(frags)} fragments generated")
        for frag in frags:
            valid = Chem.MolFromSmarts(frag) is not None
            print(f"  [{'VALID' if valid else 'INVALID'}] {frag}")


if __name__ == '__main__':
    main()
