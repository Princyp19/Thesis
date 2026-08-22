"""
train_additive_gcm.py — end-to-end neural GCM, with the SMARTS encoder TRAINABLE.

Why this exists
---------------
The frozen pipeline (smarts_gpt_embed.py -> regression_from_embeddings.py) tops
out at scaffold R2 ~= 0.05 on generated fragments, while a plain ridge GCM on the
same fragments reaches 0.257. The bottleneck is measurable: a fragment's frozen
SmartsGPT embedding predicts its fitted ridge coefficient at only R2 = 0.054, so
the encoder simply does not represent property-relevant fragment information --
it was pretrained for next-token prediction and never sees the property task.
Here the encoder receives gradients.

Two heads, selected with --head:

  additive   contribution_i = g(encode(frag_i));  y_hat = sum_i contribution_i + b
             A group-contribution model whose coefficient is PREDICTED from
             fragment structure instead of looked up in a table. Ridge is the
             special case where g is a lookup. This is the only head that yields
             a per-fragment contribution in logD units, which is the point of a
             GCM -- and being additive over UNIQUE fragments it is structurally
             immune to the padding/duplication hack that inflates AttnPool+RF by
             ~11.7%.

  attnpool   y_hat = MLP(LayerNorm(sum_i softmax(a_i) * h_i))
             The existing architecture, reproduced here so the two can be
             compared under an identical loop. NOT decomposable per fragment:
             attention weights are unitless and the MLP is nonlinear, so this
             head cannot produce a GCM no matter how accurate it is.

--encoder-lr 0 freezes the encoder, giving the 2x2 (head x frozen/unfrozen) that
separates "does unfreezing help?" from "what does additivity cost?".

Usage
-----
    python train_additive_gcm.py \
        --jsonl ../../sft/results/lipo_smarts_smaprefix_all_50k.jsonl \
        --head additive --encoder-lr 1e-5 --head-lr 1e-3 \
        --output ../results/nn/addgcm_all50k
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

REF = '/home/pyq02mab/SMART_LLM/Final_LLM_Reg'
sys.path.insert(0, REF)
sys.path.insert(0, '/home/pyq02mab/Thesis/pretraining')
from regression_from_embeddings import scaffold_split, random_split, _metrics  # noqa: E402
sys.path.insert(0, '/home/pyq02mab/Thesis/gcm')
from splits import scaffold_balanced  # noqa: E402
from smarts_gpt_model import SmartsGPT, SmartsTokenizer, SPECIAL_TOKENS  # noqa: E402


# ---------------------------------------------------------------- data

def load_records(path, smiles_col, frags_col, target_col):
    """One record per molecule: (smiles, UNIQUE sorted fragments, target)."""
    recs = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        smi = next(d[k] for k in d if k.lower() == smiles_col)
        y = d[target_col]
        if y is None or y == '':
            continue
        # Unique: duplicates carry no chemical information, and summing over them
        # is exactly the padding hack we want the reward to be immune to.
        frags = sorted(set(d[frags_col]))
        if not frags:
            continue
        recs.append((smi, frags, float(y)))
    return recs


def build_frag_table(recs, tokenizer, max_len):
    """Global fragment -> row of a padded (F, max_len) id table, encoded once."""
    pad = SPECIAL_TOKENS['<PAD>']
    vocab = sorted({f for _, fl, _ in recs for f in fl})
    f2i = {f: i for i, f in enumerate(vocab)}
    table = np.full((len(vocab), max_len), pad, dtype=np.int64)
    for f, i in f2i.items():
        ids = tokenizer.encode(f, add_special=True)[:max_len]
        table[i, :len(ids)] = ids
    return vocab, f2i, torch.tensor(table)


# ---------------------------------------------------------------- model

def encode_batch(enc, idx, pad_id):
    """(B, T) token ids -> (B, D) mean-pooled hidden states, WITH gradients.

    Mirrors smarts_gpt_embed.embed_fragment exactly, but batched and without the
    .cpu()/detach that made the original inference-only.
    """
    mask = (idx != pad_id).float()
    pos = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0)
    x = enc.drop(enc.tok_emb(idx) + enc.pos_emb(pos))
    x = enc.blocks(x)
    x = enc.ln_f(x)
    m = mask.unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp(min=1e-9)


class AdditiveHead(nn.Module):
    """contribution_i = g(h_i);  y = sum_i contribution_i + bias."""

    def __init__(self, d, hidden=(128,), dropout=0.1):
        """`hidden` is the width of each hidden layer of g, so depth is tunable.

        Depth changes only how a fragment's contribution is COMPUTED, never that
        the molecule prediction is a plain sum over fragments -- y = sum g(h_i) + b
        still holds at any depth, so per-fragment attribution survives unchanged.
        """
        super().__init__()
        if isinstance(hidden, int):
            hidden = (hidden,)
        layers, cur = [], d
        for h in hidden:
            layers += [nn.Linear(cur, h), nn.ReLU(), nn.Dropout(dropout)]
            cur = h
        layers.append(nn.Linear(cur, 1))
        self.g = nn.Sequential(*layers)
        self.bias = nn.Parameter(torch.zeros(1))

    def contributions(self, h):
        """(n_frags, D) -> (n_frags,) per-fragment contribution in target units."""
        return self.g(h).squeeze(-1)

    def forward(self, h):
        return self.contributions(h).sum() + self.bias.squeeze()


EXCEL_MAX_COLS = 16384


def write_contribution_workbook(path, recs, keep, vocab, contrib, b0, meta,
                                matrix_cols=2000, sheets=None):
    """The GCM deliverable as a workbook: how each molecule's property is built up
    out of its fragments.

    The additive head guarantees  y_hat = b0 + sum_{f in frags(mol)} g(encode(f))
    exactly under .eval(), so these rows are not an approximation or an
    attribution method -- they ARE the model. Sheets:

      About         provenance + the additivity residual, so a stale workbook is
                    self-evident
      Summary       one row per molecule; bias + sum_contributions = y_pred
      Matrix        molecules x fragments, cells = contribution. Excel allows only
                    16,384 columns against a vocabulary of tens of thousands, so
                    the `matrix_cols` most frequent fragments get their own column
                    and the remainder are pooled into `other_fragments` -- every
                    row still sums to y_pred, nothing is silently dropped.
      PerMolecule   long form, one row per (molecule, fragment). Schema follows
                    Final_LLM_Reg/gcm_regression.py's gcm_contributions.csv
                    (smiles, smarts_fragment, frag_index, n_frags,
                    contribution_ai, y_pred, y_true) so the two are directly
                    comparable, with two deliberate differences:
                      * bias_b0 is a column, because this head has an intercept
                        and that one has none -- b0 + sum(contribution_ai)
                        = y_pred, and without the column the rows would appear
                        not to close.
                      * NO epsilon filter. The reference drops rows with
                        |a_i| < eps, which is why its CSV misses ~60% of each
                        molecule's fragments and misses y_pred by up to 0.76
                        logD. Every fragment is written here, so each molecule
                        sums exactly.
                    attn_weight has no analogue: this head has no attention.
      FragmentTable the global lookup, the SIMPOL b_k analogue, with occurrence
                    counts
    """
    import pandas as pd

    # Summary is always COMPUTED (the additivity check is built on it) but only
    # written when asked for. Dropping Matrix is the large file-size saving.
    want = set(sheets) if sheets else {'About', 'Summary', 'Matrix',
                                       'PerMolecule', 'FragmentTable'}
    c = dict(zip(vocab, contrib))
    rows = [r for r in recs if keep is None or r[0] in keep]

    summary, long_rows = [], []
    for smi, frags, y in rows:
        v = np.array([c[f] for f in frags])
        s = float(v.sum())
        summary.append({'smiles': smi, 'y_true': y, 'y_pred': b0 + s,
                        'error': (b0 + s) - y, 'bias_b0': b0,
                        'sum_contributions': s, 'n_fragments': len(frags),
                        'sum_positive': float(v[v > 0].sum()),
                        'sum_negative': float(v[v < 0].sum())})
        for rank, i in enumerate(np.argsort(-np.abs(v)), 1):
            long_rows.append({'smiles': smi, 'smarts_fragment': frags[i],
                              'frag_index': int(i), 'n_frags': len(frags),
                              'contribution_ai': float(v[i]),
                              'rank_in_molecule': rank, 'bias_b0': b0,
                              'y_pred': b0 + s, 'y_true': y})
    df_sum = pd.DataFrame(summary)
    df_long = pd.DataFrame(long_rows)

    df_mat, matrix_note = None, 'disabled'
    if matrix_cols > 0 and len(rows) and 'Matrix' in want:
        freq = df_long.groupby('smarts_fragment').size().sort_values(ascending=False)
        ncol = int(min(matrix_cols, EXCEL_MAX_COLS - 8, len(freq)))
        colset = {f: j for j, f in enumerate(freq.index[:ncol])}
        M = np.zeros((len(rows), ncol), dtype=np.float32)
        other = np.zeros(len(rows))
        for i, (smi, frags, y) in enumerate(rows):
            for f in frags:
                j = colset.get(f)
                if j is None:
                    other[i] += c[f]
                else:
                    M[i, j] = c[f]
        df_mat = pd.DataFrame(M, columns=list(colset))
        df_mat.insert(0, 'smiles', [r[0] for r in rows])
        df_mat.insert(1, 'y_true', [r[2] for r in rows])
        df_mat.insert(2, 'y_pred', df_sum['y_pred'].values)
        df_mat.insert(3, 'bias_b0', b0)
        df_mat.insert(4, 'other_fragments', other)
        named = float(np.mean([sum(f in colset for f in fl) / len(fl)
                               for _, fl, _ in rows]))
        matrix_note = (f'{ncol} fragment columns; {100 * named:.1f}% of per-molecule '
                       f'fragments named individually, rest pooled in other_fragments')

    occ = df_long.groupby('smarts_fragment').size() if len(df_long) else {}
    df_frag = pd.DataFrame({'fragment': vocab, 'contribution': contrib})
    df_frag['n_molecules'] = df_frag['fragment'].map(occ).fillna(0).astype(int)
    df_frag = df_frag.reindex(df_frag['contribution'].abs()
                              .sort_values(ascending=False).index)

    about = pd.DataFrame({'key': list(meta) + ['matrix', 'molecules', 'vocab_size',
                                              'bias_b0'],
                          'value': [str(v) for v in meta.values()]
                                   + [matrix_note, len(rows), len(vocab), b0]})
    with pd.ExcelWriter(path, engine='openpyxl') as w:
        if 'About' in want:
            about.to_excel(w, sheet_name='About', index=False)
        if 'Summary' in want:
            df_sum.to_excel(w, sheet_name='Summary', index=False)
        if df_mat is not None:
            df_mat.to_excel(w, sheet_name='Matrix', index=False)
        if 'PerMolecule' in want:
            df_long.to_excel(w, sheet_name='PerMolecule', index=False)
        if 'FragmentTable' in want:
            df_frag.to_excel(w, sheet_name='FragmentTable', index=False)
    written = [s for s in ('About', 'Summary', 'Matrix', 'PerMolecule',
                           'FragmentTable')
               if s in want and (s != 'Matrix' or df_mat is not None)]
    return df_sum, f'{matrix_note}; sheets: {", ".join(written)}'


class AttnPoolHead(nn.Module):
    """Reproduces AttnDirectModel: softmax pooling + nonlinear MLP. Not additive."""

    def __init__(self, d, attn_hidden=256, mlp_dims=(256, 256, 128, 64), dropout=0.3):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(d, attn_hidden), nn.Tanh(),
                                  nn.Linear(attn_hidden, 1, bias=False))
        self.norm = nn.LayerNorm(d)
        layers, cur = [], d
        for hh in mlp_dims:
            layers += [nn.Linear(cur, hh), nn.ReLU(), nn.Dropout(dropout)]
            cur = hh
        layers.append(nn.Linear(cur, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, h):
        w = torch.softmax(self.attn(h), dim=0)
        return self.mlp(self.norm((h * w).sum(0)).unsqueeze(0)).squeeze()


# ---------------------------------------------------------------- args

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--jsonl', required=True)
    p.add_argument('--smiles-col', default='smiles')
    p.add_argument('--frags-col', default='fragments')
    p.add_argument('--target-col', default='exp')
    p.add_argument('--encoder-ckpt',
                   default=f'{REF}/checkpoints_pubchem_gpt/checkpoint_epoch010_ppl3.75_nov0.38.pt')
    p.add_argument('--encoder-tok',
                   default=f'{REF}/checkpoints_pubchem_gpt/tokenizer.json')
    p.add_argument('--head', choices=['additive', 'attnpool'], default='additive')
    p.add_argument('--encoder-lr', type=float, default=1e-5,
                   help='0 = FREEZE the encoder (reproduces the precomputed-embedding setting)')
    p.add_argument('--head-lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=16, help='molecules per step')
    p.add_argument('--split', choices=['scaffold', 'random', 'paper'], default='scaffold',
                   help="'scaffold' = the 90/10 reward split (largest groups -> TEST); keep "
                        "this for anything that must stay consistent with the frozen reward "
                        "model. 'paper' = GROVER/GODE scaffold_balanced, 80/10/10, seeded -- "
                        "the ONLY one comparable to published Lipophilicity RMSE. The two are "
                        "near-inverses (test sets overlap 0/420), so they are not "
                        "interchangeable and results must state which was used.")
    p.add_argument('--test-size', type=float, default=0.1)
    p.add_argument('--val-size', type=float, default=0.125)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--head-dims', default='128',
                   help="Comma-separated hidden widths of the per-fragment head g, e.g. "
                        "'128' (default, 1 layer), '256,128', '256,256,128'. Additivity is "
                        "unaffected at any depth -- y = sum g(h_i) + b -- so the fragment "
                        "contribution table stays valid. Only 'additive' uses this.")
    p.add_argument('--dropout', type=float, default=0.1,
                   help='Dropout inside the per-fragment head g. The GT/paper-split fit '
                        'reaches train RMSE 0.651 against test 0.819, so this model is '
                        'capacity-rich and generalisation-poor: raising this is more likely '
                        'to help than adding layers.')
    p.add_argument('--frag-dropout', type=float, default=0.0,
                   help="Randomly drop this share of each TRAINING molecule's fragments "
                        "per step, rescaling survivors by 1/(1-p). 0.0 (default) is the "
                        "existing behaviour exactly. Motivation: dropping the smallest "
                        "25%% of a molecule's fragments at inference takes test RMSE from "
                        "0.72 to 1.48, so the model is brittle to incomplete fragment "
                        "sets -- which is precisely what a generation policy emits. "
                        "Additive head only.")
    p.add_argument('--seed', type=int, default=42,
                   help='Default for both --split-seed and --init-seed, so omitting '
                        'those reproduces every earlier run bit-for-bit.')
    p.add_argument('--split-seed', type=int, default=None,
                   help='Seed for the train/val/test split ONLY. Vary this over 0,1,2 '
                        'for the GROVER/GODE multi-seed protocol -- each value is a '
                        'different test set, so RMSEs across split seeds are not '
                        'paired and must be reported as mean+/-std, never ensembled.')
    p.add_argument('--init-seed', type=int, default=None,
                   help='Seed for weight init and batch order ONLY, holding the split '
                        'fixed. Runs that share a --split-seed predict the SAME '
                        'molecules, so their test_predictions.npz can be averaged -- '
                        'which is what that file was saved for.')
    p.add_argument('--output', required=True)
    p.add_argument('--contrib-scope', default='test', choices=['test', 'all', 'none'],
                   help="Molecules in molecule_contributions.xlsx. 'test' is the "
                        "held-out set the reported RMSE was computed on; 'all' "
                        "includes train/val, which are FIT molecules and must not "
                        "be read as evidence of generalisation.")
    p.add_argument('--contrib-matrix-cols', type=int, default=2000,
                   help='Fragment columns on the Matrix sheet (0 skips that sheet). '
                        'Excel hard-caps at 16,384 columns.')
    p.add_argument('--contrib-sheets',
                   default='About,Summary,Matrix,PerMolecule,FragmentTable',
                   help='Which sheets to write. Matrix dominates the file size, so '
                        'About,PerMolecule,FragmentTable is the compact form.')
    return p.parse_args()


# ---------------------------------------------------------------- main

def main():
    a = parse_args()
    # One --seed used to drive the split, the weight init AND the batch order at
    # once, which made the two variance sources inseparable: three "seeds" gave
    # three near-disjoint test sets (overlap 23-39 of 420), so a change in RMSE
    # could not be attributed to the model rather than to test-set difficulty,
    # and test_predictions.npz could not be ensembled as intended. Both default
    # to --seed, so any run that does not pass them is unchanged.
    split_seed = a.seed if a.split_seed is None else a.split_seed
    init_seed = a.seed if a.init_seed is None else a.init_seed
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frozen = a.encoder_lr == 0

    recs = load_records(a.jsonl, a.smiles_col.lower(), a.frags_col, a.target_col)
    y = np.array([r[2] for r in recs], dtype=np.float32)
    smiles = [r[0] for r in recs]
    print(f'{len(recs)} molecules | unique frags/mol mean='
          f'{np.mean([len(r[1]) for r in recs]):.1f} | target {y.min():.2f}..{y.max():.2f}')

    ck = torch.load(a.encoder_ckpt, map_location=dev, weights_only=False)
    hp = ck['hparams'] if 'hparams' in ck else ck
    tok = SmartsTokenizer.load(a.encoder_tok)
    enc = SmartsGPT(vocab_size=hp['vocab_size'], n_layer=hp['n_layer'], n_head=hp['n_head'],
                    n_embd=hp['n_embd'], max_len=hp['max_len'], dropout=hp['dropout']).to(dev)
    enc.load_state_dict(ck['model_state_dict'])
    D, PAD = hp['n_embd'], SPECIAL_TOKENS['<PAD>']
    print(f"encoder: d={D} max_len={hp['max_len']} | {'FROZEN' if frozen else f'trainable @ lr={a.encoder_lr}'}")

    vocab, f2i, table = build_frag_table(recs, tok, hp['max_len'])
    table = table.to(dev)
    mol_idx = [torch.tensor([f2i[f] for f in fl], dtype=torch.long, device=dev) for _, fl, _ in recs]
    print(f'distinct fragments: {len(vocab)}')

    dims = tuple(int(x) for x in a.head_dims.split(',') if x.strip())
    head = (AdditiveHead(D, hidden=dims, dropout=a.dropout) if a.head == 'additive'
            else AttnPoolHead(D, dropout=a.dropout)).to(dev)
    if a.head == 'additive':
        print(f'head g: {D} -> ' + ' -> '.join(str(x) for x in dims) + ' -> 1'
              f'  ({sum(p_.numel() for p_ in head.parameters()):,} params)')
    groups = [{'params': head.parameters(), 'lr': a.head_lr}]
    if frozen:
        enc.eval()
        for p_ in enc.parameters():
            p_.requires_grad_(False)
    else:
        groups.append({'params': enc.parameters(), 'lr': a.encoder_lr})
    opt = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    N = len(recs)
    if a.split == 'paper':
        # Already returns train/val/test at 80/10/10, so do NOT carve val again.
        tr, va, te = (np.array(x) for x in scaffold_balanced(smiles, seed=split_seed))
    else:
        if a.split == 'scaffold':
            tr, te = scaffold_split(smiles, N, test_size=a.test_size)
        else:
            tr, te = random_split(N, test_size=a.test_size, seed=split_seed)
        rng = np.random.default_rng(split_seed + 1)
        perm = rng.permutation(len(tr))
        nv = max(1, int(len(tr) * a.val_size))
        va, tr = tr[perm[:nv]], tr[perm[nv:]]
    print(f'split={a.split}  train={len(tr)} val={len(va)} test={len(te)}  '
          f'split_seed={split_seed} init_seed={init_seed}')

    yt = torch.tensor(y, device=dev)

    fdrop = a.frag_dropout if a.head == 'additive' else 0.0
    if a.frag_dropout and a.head != 'additive':
        print('WARNING: --frag-dropout applies only to the additive head; ignoring.')

    def run(idxs, train=False):
        """Forward a set of molecules; dedupe fragments across the batch so each
        unique fragment is encoded once per step.

        With --frag-dropout p, each TRAINING molecule keeps a random (1-p) share of
        its fragments and the surviving contributions are divided by (1-p). The
        rescale is what makes this correct rather than a bias: E[sum of kept
        contributions / (1-p)] = sum over all fragments, so the model still targets
        the true y and eval (no dropout, no rescale) matches training in
        expectation. Dropping WITHOUT the rescale would train the model to reach y
        from a fraction of the molecule and then overshoot on the full set.

        The bias is added once, outside the rescale -- it is a molecule-level
        intercept, not a fragment contribution.
        """
        preds = []
        allf = torch.cat([mol_idx[i] for i in idxs])
        uniq, inv = torch.unique(allf, return_inverse=True)
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            H = encode_batch(enc, table[uniq], PAD)          # (U, D)
            pos = 0
            for i in idxs:
                n = len(mol_idx[i])
                h = H[inv[pos:pos + n]]
                if train and fdrop > 0:
                    keep = torch.rand(n, device=h.device) >= fdrop
                    if not bool(keep.any()):     # never leave a molecule empty
                        keep[torch.randint(n, (1,), device=h.device)] = True
                    preds.append(head.contributions(h[keep]).sum() / (1.0 - fdrop)
                                 + head.bias.squeeze())
                else:
                    preds.append(head(h))
                pos += n
        return torch.stack(preds)

    best, best_state = float('inf'), None
    srng = np.random.default_rng(init_seed)
    for ep in range(1, a.epochs + 1):
        head.train()
        if not frozen:
            enc.train()
        order = srng.permutation(len(tr))
        tot = 0.0
        for s in range(0, len(order), a.batch_size):
            idxs = tr[order[s:s + a.batch_size]]
            opt.zero_grad()
            loss = ((run(idxs, train=True) - yt[idxs]) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p_ for g in groups for p_ in g['params'] if p_.requires_grad], 1.0)
            opt.step()
            tot += loss.item() * len(idxs)
        sched.step()

        head.eval()
        enc.eval()
        vp = np.concatenate([run(va[s:s + 64]).cpu().numpy() for s in range(0, len(va), 64)])
        vr = float(np.sqrt(np.mean((y[va] - vp) ** 2)))
        if vr < best:
            best = vr
            best_state = ({k: v.detach().clone() for k, v in head.state_dict().items()},
                          {k: v.detach().clone() for k, v in enc.state_dict().items()})
        print(f'  epoch {ep:3d}/{a.epochs}  train_mse={tot / len(tr):.4f}  val_rmse={vr:.4f}  best={best:.4f}')

    if best_state:
        head.load_state_dict(best_state[0])
        enc.load_state_dict(best_state[1])
    head.eval()
    enc.eval()
    tp = np.concatenate([run(te[s:s + 64]).cpu().numpy() for s in range(0, len(te), 64)])
    m = _metrics(y[te], tp)
    # Per-molecule test predictions, so seed-ensembling (and any paired significance
    # test on RMSE) is a post-hoc step instead of a re-run. Molecule order is the
    # test index order, which is reproducible from the split + seed.
    os.makedirs(a.output, exist_ok=True)
    np.savez(os.path.join(a.output, 'test_predictions.npz'),
             y_true=y[te], y_pred=tp, test_idx=np.asarray(te),
             smiles=np.array([smiles[i] for i in te]))
    print(f"\nTEST  R2={m['R2']}  RMSE={m['RMSE']}  MAE={m['MAE']}  Spearman={m['Spearman']}")

    os.makedirs(a.output, exist_ok=True)
    torch.save({'head_state_dict': head.state_dict(), 'encoder_state_dict': enc.state_dict(),
                'head_type': a.head, 'embd_dim': D, 'metrics': m,
                'head_dims': list(dims),      # so consumers rebuild the right shape
                'encoder_lr': a.encoder_lr, 'frozen': frozen},
               os.path.join(a.output, 'additive_gcm.pt'))
    json.dump({'metrics': m, 'head': a.head, 'encoder_lr': a.encoder_lr,
               'frozen': frozen, 'head_lr': a.head_lr, 'split': a.split,
               # Both seeds, so a result can never be mistaken for a different
               # test set -- runs are only ensemblable when split_seed matches.
               'split_seed': split_seed, 'init_seed': init_seed,
               'head_dims': list(dims), 'dropout': a.dropout,
               'frag_dropout': a.frag_dropout, 'weight_decay': a.weight_decay,
               'epochs': a.epochs, 'input': os.path.abspath(a.jsonl),
               'n_molecules': N, 'n_fragments': len(vocab),
               'sizes': {'train': len(tr), 'val': len(va), 'test': len(te)}},
              open(os.path.join(a.output, 'results.json'), 'w'), indent=2)

    # The GCM deliverable: a contribution per fragment, in target units.
    if a.head == 'additive':
        with torch.no_grad():
            contrib = np.concatenate([
                head.contributions(encode_batch(enc, table[s:s + 512], PAD)).cpu().numpy()
                for s in range(0, len(vocab), 512)])
        np.save(os.path.join(a.output, 'fragment_contributions.npy'), contrib)
        with open(os.path.join(a.output, 'fragment_contributions.csv'), 'w') as f:
            f.write('fragment,contribution\n')
            for i in np.argsort(-np.abs(contrib)):
                f.write(f'"{vocab[i]}",{contrib[i]:.6f}\n')
        print(f'per-fragment contributions -> {a.output}/fragment_contributions.csv')

        # Per-molecule view of the same numbers: how each prediction is built up
        # out of its fragments. Written here rather than post-hoc so it can never
        # disagree with the checkpoint that produced the reported metrics.
        # Reporting only. Everything the run is judged on -- metrics, checkpoint,
        # test_predictions.npz, results.json -- is already on disk above, so this
        # is wrapped: a workbook failure (openpyxl missing, disk full, a 16k-column
        # overflow) must never turn a finished training run into a failed job.
        if a.contrib_scope != 'none':
            try:
                b0 = float(head.bias.detach().cpu().squeeze())
                keep = None if a.contrib_scope == 'all' else {smiles[i] for i in te}
                xlsx = os.path.join(a.output, 'molecule_contributions.xlsx')
                df_sum, note = write_contribution_workbook(
                    xlsx, recs, keep, vocab, contrib, b0,
                    {'run': os.path.abspath(a.output),
                     'input_jsonl': os.path.abspath(a.jsonl),
                     'frags_col': a.frags_col, 'split_variant': a.split,
                     'split_seed': split_seed, 'init_seed': init_seed,
                     'scope': a.contrib_scope, 'head_dims': list(dims),
                     'test_RMSE': m['RMSE'], 'test_R2': m['R2']},
                    matrix_cols=a.contrib_matrix_cols,
                    sheets=[s.strip() for s in a.contrib_sheets.split(',') if s.strip()])
                print(f'per-molecule contributions -> {xlsx}  ({note})')
                # The workbook claims y_pred = b0 + sum(contributions). Verify it
                # against the predictions the metrics above were computed from; a
                # table that drifts from the model must not be presented as one.
                pred = dict(zip(df_sum['smiles'], df_sum['y_pred']))
                d = np.array([tp[k] - pred[smiles[i]]
                              for k, i in enumerate(te) if smiles[i] in pred])
                print(f'  additivity check: max |diff| vs test predictions = '
                      f'{np.abs(d).max():.2e}' if len(d) else '  additivity check: n/a')
            except Exception as e:
                print(f'WARNING: contribution workbook failed ({type(e).__name__}: {e}). '
                      f'Training results above are unaffected and already saved.')
    print('saved ->', a.output)


if __name__ == '__main__':
    main()
