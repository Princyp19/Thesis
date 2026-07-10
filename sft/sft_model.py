"""
sft_model.py — Adapt a pretrained SmartsGPT checkpoint for conditional SFT.

Adds two new special tokens (<SEP>, <FRAG_SEP>) and resizes positional
embeddings from max_len=32 to max_len=256, while preserving all pretrained
weights. Never mutates the source checkpoint or the caller's tokenizer object.
"""

import copy
import sys

import torch

sys.path.insert(0, '/home/pyq02mab/Thesis/pretraining')
from smarts_gpt_model import SmartsGPT, SPECIAL_TOKENS  # noqa: E402


def add_special_tokens(tokenizer):
    """Return a copy of tokenizer with <SEP> and <FRAG_SEP> appended. Does not mutate the input."""
    tok = copy.deepcopy(tokenizer)

    sep_id = tok.vocab_size
    tok.token2id['<SEP>'] = sep_id
    tok.id2token[sep_id] = '<SEP>'

    frag_sep_id = tok.vocab_size + 1
    tok.token2id['<FRAG_SEP>'] = frag_sep_id
    tok.id2token[frag_sep_id] = '<FRAG_SEP>'

    tok.vocab_size += 2
    return tok, sep_id, frag_sep_id


def load_pretrained_for_sft(checkpoint_path, tokenizer, max_len=256, device='cpu'):
    """
    Load a pretrained SmartsGPT checkpoint, add <SEP>/<FRAG_SEP>, and resize
    positional embeddings to max_len. Returns (model, tokenizer_with_new_tokens, sep_id, frag_sep_id).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hp = ckpt['hparams']
    old_state = ckpt['model_state_dict']
    old_max_len = hp['max_len']

    tok, sep_id, frag_sep_id = add_special_tokens(tokenizer)

    model = SmartsGPT(
        vocab_size=tok.vocab_size,
        n_layer=hp['n_layer'],
        n_head=hp['n_head'],
        n_embd=hp['n_embd'],
        max_len=max_len,
        dropout=hp['dropout'],
    )

    bos_id = SPECIAL_TOKENS['<BOS>']
    eos_id = SPECIAL_TOKENS['<EOS>']
    old_vocab_size = hp['vocab_size']

    with torch.no_grad():
        # tok_emb is tied to lm_head, so this single copy updates both.
        model.tok_emb.weight.data[:old_vocab_size] = old_state['tok_emb.weight']
        new_tok_embedding = (
            old_state['tok_emb.weight'][bos_id] + old_state['tok_emb.weight'][eos_id]
        ) / 2.0
        model.tok_emb.weight.data[sep_id] = new_tok_embedding
        model.tok_emb.weight.data[frag_sep_id] = new_tok_embedding

        model.pos_emb.weight.data[:old_max_len] = old_state['pos_emb.weight']
        for i in range(old_max_len, max_len):
            model.pos_emb.weight.data[i] = old_state['pos_emb.weight'][old_max_len - 1]

    skip_keys = {'tok_emb.weight', 'pos_emb.weight', 'lm_head.weight'}
    remaining_state = {
        k: v for k, v in old_state.items()
        if k not in skip_keys and not k.endswith('.mask')
    }
    missing, unexpected = model.load_state_dict(remaining_state, strict=False)

    new_hparams = dict(hp)
    new_hparams['vocab_size'] = tok.vocab_size
    new_hparams['max_len'] = max_len

    return model, tok, sep_id, frag_sep_id, new_hparams
