"""
sft_model.py — Adapt a pretrained SmartsGPT checkpoint for conditional SFT.

Adds two new special tokens (<SEP>, <FRAG_SEP>) and resizes positional
embeddings from max_len=32 to max_len=256, while preserving all pretrained
weights. Never mutates the source checkpoint or the caller's tokenizer object.
"""

import copy
import math
import sys

import torch

sys.path.insert(0, '/home/pyq02mab/Thesis/pretraining')
from smarts_gpt_model import SmartsGPT, SPECIAL_TOKENS, SMILES_NS  # noqa: E402


def add_special_tokens(tokenizer, smiles_tokens=None, extra_tokens=None):
    """Return a copy of tokenizer with <SEP>, <FRAG_SEP>, and (optionally) the
    namespaced SMILES conditioning vocab appended. Does not mutate the input.

    SMILES tokens are stored as SMILES_NS + raw (e.g. '<SMI>C') so an input
    SMILES 'C' never aliases the SMARTS output token 'C'. They come last in the
    id order, so their embedding rows stay freshly initialized (learned during
    SFT) while all pretrained rows keep their weights.
    """
    tok = copy.deepcopy(tokenizer)

    sep_id = tok.vocab_size
    tok.token2id['<SEP>'] = sep_id
    tok.id2token[sep_id] = '<SEP>'

    frag_sep_id = tok.vocab_size + 1
    tok.token2id['<FRAG_SEP>'] = frag_sep_id
    tok.id2token[frag_sep_id] = '<FRAG_SEP>'

    tok.vocab_size += 2

    for raw in (smiles_tokens or []):
        key = SMILES_NS + raw
        if key not in tok.token2id:
            i = tok.vocab_size
            tok.token2id[key] = i
            tok.id2token[i] = key
            tok.vocab_size += 1

    # extra_tokens are appended VERBATIM (no namespace). Used for the optional
    # property-conditioning prefix, which already carries its own '<PROP>' tag.
    # Ignored unless explicitly passed, so existing callers are unaffected.
    for key in (extra_tokens or []):
        if key not in tok.token2id:
            i = tok.vocab_size
            tok.token2id[key] = i
            tok.id2token[i] = key
            tok.vocab_size += 1

    return tok, sep_id, frag_sep_id


def load_pretrained_for_sft(checkpoint_path, tokenizer, smiles_tokens=None,
                            extra_tokens=None,
                             max_len=256, device='cpu'):
    """
    Load a pretrained SmartsGPT checkpoint, add <SEP>/<FRAG_SEP> and the
    namespaced SMILES conditioning vocab, and resize positional embeddings to
    max_len. Returns (model, tokenizer_with_new_tokens, sep_id, frag_sep_id, hparams).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hp = ckpt['hparams']
    old_state = ckpt['model_state_dict']
    old_max_len = hp['max_len']

    tok, sep_id, frag_sep_id = add_special_tokens(tokenizer, smiles_tokens=smiles_tokens,
                                                 extra_tokens=extra_tokens)

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
        # Positions beyond the pretrained window (old_max_len is typically 32)
        # previously all got an identical copy of pos_emb[old_max_len - 1],
        # leaving the model unable to distinguish position 100 from 3000 at
        # init. Use sinusoidal encodings instead so every new position starts
        # distinct, scaled to the magnitude of the pretrained embeddings so the
        # transition at old_max_len is not a discontinuity.
        n_new = max_len - old_max_len
        if n_new > 0:
            n_embd = model.pos_emb.weight.shape[1]
            pos = torch.arange(old_max_len, max_len, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, n_embd, 2, dtype=torch.float32)
                * (-math.log(10000.0) / n_embd)
            )
            sinusoid = torch.zeros(n_new, n_embd)
            sinusoid[:, 0::2] = torch.sin(pos * div)
            sinusoid[:, 1::2] = torch.cos(pos * div)[:, :n_embd // 2]
            # .item(): old_state may live on GPU (map_location=device) while
            # sinusoid is built on CPU, and in-place mul across devices errors.
            sinusoid *= old_state['pos_emb.weight'].std().item()
            model.pos_emb.weight.data[old_max_len:] = sinusoid

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
