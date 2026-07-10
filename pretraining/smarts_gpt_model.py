"""
SMARTS-GPT: A small decoder-only transformer for SMARTS generation.

Architecture facts for your dataset:
  - Vocab size:     ~200 tokens (all bracket atoms + operators)
  - Max seq length: 32 tokens (your data max is 13, padded for BOS/EOS + safety)
  - Model size:     ~2-5M parameters (right-sized for 532k-735k samples)
  - Training obj:   next-token prediction (causal LM)

This is directly PPO-compatible because generation is autoregressive.
"""

import math
import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path


# ============================================================================
# SMARTS TOKENIZER
# ============================================================================

SMARTS_PATTERN = re.compile(
    r'(\[[^\]]+\]'   # bracket atoms: [#6], [C&H3], [N&H0&R], etc.
    r'|Br|Cl|Si|Se'  # two-char organic atoms
    r'|@@|>>|->)'    # multi-char operators
    r'|(.)'          # everything else: single char ( ) - = # : . 1 2 3 etc.
)

SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<BOS>': 1,
    '<EOS>': 2,
    '<UNK>': 3,
}


class SmartsTokenizer:
    """
    Atom-level SMARTS tokenizer.
    Each bracket group [C&H3] is a single token, not split into chars.
    """

    def __init__(self):
        self.token2id = dict(SPECIAL_TOKENS)
        self.id2token = {v: k for k, v in SPECIAL_TOKENS.items()}
        self.vocab_size = len(SPECIAL_TOKENS)

    def tokenize(self, smarts: str):
        """Split SMARTS string into atom-level tokens."""
        tokens = []
        for m in SMARTS_PATTERN.finditer(smarts):
            tok = m.group(0)
            tokens.append(tok)
        return tokens

    def build_vocab(self, smarts_list):
        """Build vocabulary from a list of SMARTS strings."""
        all_tokens = set()
        for smarts in smarts_list:
            all_tokens.update(self.tokenize(smarts))

        for tok in sorted(all_tokens):
            if tok not in self.token2id:
                idx = len(self.token2id)
                self.token2id[tok] = idx
                self.id2token[idx] = tok

        self.vocab_size = len(self.token2id)
        print(f"Vocabulary built: {self.vocab_size} tokens "
              f"(4 special + {self.vocab_size - 4} SMARTS tokens)")
        return self

    def encode(self, smarts: str, add_special=True):
        """SMARTS string → list of token IDs."""
        tokens = self.tokenize(smarts)
        ids = [self.token2id.get(t, SPECIAL_TOKENS['<UNK>']) for t in tokens]
        if add_special:
            ids = [SPECIAL_TOKENS['<BOS>']] + ids + [SPECIAL_TOKENS['<EOS>']]
        return ids

    def decode(self, ids, skip_special=True):
        """List of token IDs → SMARTS string."""
        tokens = []
        for i in ids:
            tok = self.id2token.get(i, '<UNK>')
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return ''.join(tokens)

    def save(self, path):
        data = {
            'token2id': self.token2id,
            'id2token': {str(k): v for k, v in self.id2token.items()},
            'vocab_size': self.vocab_size,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Tokenizer saved → {path}")

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        tok = cls()
        tok.token2id = data['token2id']
        tok.id2token = {int(k): v for k, v in data['id2token'].items()}
        tok.vocab_size = data['vocab_size']
        return tok


# ============================================================================
# DATASET
# ============================================================================

class SmartsDataset(Dataset):
    """
    Dataset for causal language modeling on SMARTS strings.
    Input:  <BOS> [#6] - [O&H1]
    Target:        [#6] - [O&H1] <EOS>
    """

    def __init__(self, smarts_list, tokenizer, max_len=32):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        skipped = 0
        for smarts in smarts_list:
            ids = tokenizer.encode(smarts, add_special=True)
            if len(ids) <= max_len:
                # Pad to max_len
                padded = ids + [SPECIAL_TOKENS['<PAD>']] * (max_len - len(ids))
                self.samples.append(padded)
            else:
                skipped += 1

        if skipped > 0:
            print(f"  Skipped {skipped} sequences longer than max_len={max_len}")
        print(f"  Dataset: {len(self.samples)} sequences")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = torch.tensor(self.samples[idx], dtype=torch.long)
        # Causal LM: input is all tokens except last, target is all tokens except first
        x = ids[:-1]
        y = ids[1:]
        return x, y


# ============================================================================
# MODEL: Decoder-only Transformer (GPT-style)
# ============================================================================

class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, dropout=0.1, max_len=32):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal mask
        mask = torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).split(self.n_embd, dim=2)
        q, k, v = [t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                   for t in qkv]

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class TransformerBlock(nn.Module):

    def __init__(self, n_embd, n_head, ffn_mult=4, dropout=0.1, max_len=32):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, max_len)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, ffn_mult * n_embd),
            nn.GELU(),
            nn.Linear(ffn_mult * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SmartsGPT(nn.Module):
    """
    Decoder-only transformer for SMARTS generation.

    Default config for 532k-735k dataset:
      n_layer=6, n_head=8, n_embd=256  →  ~5M params
    """

    def __init__(self, vocab_size, n_layer=6, n_head=8, n_embd=256,
                 max_len=32, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd,
                                    padding_idx=SPECIAL_TOKENS['<PAD>'])
        self.pos_emb = nn.Embedding(max_len, n_embd)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head, dropout=dropout, max_len=max_len)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying
        self.tok_emb.weight = self.lm_head.weight

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"SmartsGPT: {n_params/1e6:.2f}M parameters | "
              f"vocab={vocab_size} | layers={n_layer} | "
              f"heads={n_head} | embd={n_embd} | max_len={max_len}")

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)

        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Ignore PAD tokens in loss
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=SPECIAL_TOKENS['<PAD>']
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, tokenizer, max_new_tokens=30, temperature=1.0,
                 top_k=40, top_p=0.92, device='cpu'):
        """
        Autoregressive generation from <BOS>.
        Returns decoded SMARTS string.
        """
        self.eval()
        ids = [SPECIAL_TOKENS['<BOS>']]
        idx = torch.tensor([ids], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            # Crop to max_len
            idx_cond = idx[:, -self.max_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Remove special tokens from sampling (except EOS)
            logits[:, SPECIAL_TOKENS['<PAD>']] = float('-inf')
            logits[:, SPECIAL_TOKENS['<BOS>']] = float('-inf')
            logits[:, SPECIAL_TOKENS['<UNK>']] = float('-inf')

            # Top-k
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, top_k)[0][:, -1, None]
                logits = logits.masked_fill(logits < kth, float('-inf'))

            # Top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float('-inf')
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)

            if next_tok.item() == SPECIAL_TOKENS['<EOS>']:
                break

            idx = torch.cat([idx, next_tok], dim=1)

        generated_ids = idx[0].tolist()[1:]  # Remove BOS
        return tokenizer.decode(generated_ids)
