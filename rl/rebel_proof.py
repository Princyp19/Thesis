"""
rebel_proof.py — Minimal 1-iteration REBEL proof-of-concept.

Goal: show that ONE REBEL regression step raises the reward of the SFT policy on
held-out molecules — i.e. that RL can push generated SMARTS toward more
predictive fragments.

Defaults target the current best pair: the uncapped SMARTS-prefix policy
(85.8% faithful) rewarded by the ADDITIVE neural GCM (scaffold R2=0.35, vs 0.23
for ridge on the same split). The additive reward is used because it scores
fragments it has never seen -- a ridge lookup returns 0 for those, which would
train the policy to stop inventing fragments, the opposite of the goal.

Reward going up is NOT sufficient evidence: the reward is a model and RL will
exploit it. Faithfulness (whether generated fragments actually occur in the
molecule) is reported before/after on the SAME held-out molecules and is not part
of the objective, so it is the independent check for reward hacking.

REBEL objective for a pair (y, y') on the same molecule x:

    loss = [ (1/eta) * ( (lnπ_θ(y|x) - lnπ_ref(y|x))
                       - (lnπ_θ(y'|x) - lnπ_ref(y'|x)) )
             - ( r(x,y) - r(x,y') ) ]^2

where π_ref = π_θt is the policy snapshot at the start of the iteration (for a
single iteration this is just the initial SFT policy), and
r(x,y) = -|logD_true - GCM_predict(fragments(y))|  (negative absolute error).

This is intentionally small: train on a few hundred scaffold-TRAIN molecules,
evaluate reward on held-out scaffold-TEST molecules before vs after the update.
"""

import argparse
import copy
import datetime
import json
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error

# Final_LLM_Reg is APPENDED, not inserted: it ships its own smarts_gpt_model
# without SMILES_NS, and putting it ahead of Thesis/pretraining shadows the real
# one and breaks sft_dataset's import.
sys.path.append('/home/pyq02mab/SMART_LLM/Final_LLM_Reg')
sys.path.insert(0, '/home/pyq02mab/Thesis/gcm/ridge_gcm')
sys.path.insert(0, '/home/pyq02mab/Thesis/gcm/nn_gcm')
sys.path.insert(0, '/home/pyq02mab/Thesis/gcm')
sys.path.insert(0, '/home/pyq02mab/Thesis/sft')
sys.path.insert(0, '/home/pyq02mab/Thesis/pretraining')
from smarts_gpt_model import SmartsTokenizer, SmartsGPT, SPECIAL_TOKENS  # noqa: E402
from sft_dataset import build_prefix_ids  # noqa: E402
from faithfulness import faithful_frac  # noqa: E402
from gcm_reward import FrozenGCM  # noqa: E402
from fit_gcm import scaffold_split as ridge_scaffold_split  # noqa: E402
sys.path.insert(0, '/home/pyq02mab/Thesis/gcm')   # literal: THESIS_ROOT is defined below
from splits import scaffold_balanced  # noqa: E402
# The additive GCM was fit with THIS splitter; fit_gcm's is nearly its inverse
# (it puts the largest scaffold groups in train, this one puts them in test), so
# using the wrong one would put REBEL's held-out molecules inside the reward
# model's training set.
from regression_from_embeddings import scaffold_split as reward_scaffold_split  # noqa: E402

THESIS_ROOT = '/home/pyq02mab/Thesis'
EOS = SPECIAL_TOKENS['<EOS>']
PAD = SPECIAL_TOKENS['<PAD>']
BOS = SPECIAL_TOKENS['<BOS>']
UNK = SPECIAL_TOKENS['<UNK>']


def parse_args():
    p = argparse.ArgumentParser(description='Minimal 1-iteration REBEL proof')
    p.add_argument('--checkpoint', type=str,
                   default=f'{THESIS_ROOT}/sft/checkpoints/smaprefix_all_50k_j23015580/'
                           f'pi_theta_epoch010_valid0.9984_loss0.5341.pt',
                   help='SFT policy. Default = the uncapped SMARTS-prefix policy '
                        '(85.8%% faithful, ridge R2=0.257 — the best base policy).')
    p.add_argument('--reward-type', choices=['additive', 'ridge'], default='additive',
                   help="'additive' = the neural GCM (scaffold R2=0.35, beats ridge's "
                        "0.23 on the same split, scores NOVEL fragments, duplicate-safe). "
                        "'ridge' = frozen lookup GCM; it returns 0 for any fragment "
                        "outside its vocabulary, so RL against it teaches the policy to "
                        "stop inventing fragments — keep it as a control only.")
    p.add_argument('--reward-ckpt', type=str,
                   default=f'{THESIS_ROOT}/gcm/results/nn/smaprefix_all_50k_additive_enc5e-5/additive_gcm.pt',
                   help='additive_gcm.pt (for --reward-type additive)')
    p.add_argument('--reward-pkl', type=str,
                   default=f'{THESIS_ROOT}/gcm/results/ridge/gt_cap20_old20/gcm_lipo_gt_cap20_old20_frozen.pkl',
                   help='frozen ridge .pkl (for --reward-type ridge)')
    p.add_argument('--encoder-ckpt', type=str,
                   default='/home/pyq02mab/SMART_LLM/Final_LLM_Reg/checkpoints_pubchem_gpt/checkpoint_epoch010_ppl3.75_nov0.38.pt')
    p.add_argument('--encoder-tok', type=str,
                   default='/home/pyq02mab/SMART_LLM/Final_LLM_Reg/checkpoints_pubchem_gpt/tokenizer.json')
    p.add_argument('--dataset', type=str,
                   default=f'{THESIS_ROOT}/data/gcm/Lipo_smarts_v2.jsonl',
                   help='JSONL with smiles + exp (true logD) columns')
    p.add_argument('--n-train-mols', type=int, default=300)
    p.add_argument('--n-test-mols', type=int, default=200)
    p.add_argument('--n-inner-steps', type=int, default=150,
                   help='Gradient steps solving the REBEL least-squares regression')
    p.add_argument('--batch-pairs', type=int, default=32)
    p.add_argument('--eta', type=float, default=1.0,
                   help='REBEL 1/eta reward-gap scale')
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--temperature', type=float, default=0.4,
                   help='0.4 is the measured best setting for faithfulness across '
                        'every checkpoint (e.g. 74.5%% -> 80.9%% on filt20).')
    p.add_argument('--top-k', type=int, default=40)
    p.add_argument('--max-new-tokens', type=int, default=2900,
                   help='Target-only budget. The uncapped policy needs ~1537 tokens '
                        '(p95 2846); 400 would truncate 97%% of completions. Use 256 '
                        'for a cap20 policy, 800 for cap80.')
    p.add_argument('--reward-clip', type=float, default=100.0,
                   help='Clip |logD error| to this before negating, so a single '
                        'wildly-mispredicted molecule cannot dominate the regression '
                        '(default 100 = effectively off; set ~4 for the scaled run)')
    p.add_argument('--eval-samples', type=int, default=4,
                   help='Completions sampled per test molecule when estimating '
                        'E[reward]. 0 = greedy/argmax (measures the mode, NOT what '
                        'REBEL optimises).')
    p.add_argument('--min-gap', type=float, default=0.0,
                   help='Drop training pairs whose |reward gap| is below this. Tiny '
                        'gaps are mostly GCM noise rather than real quality '
                        'differences (default 0 = keep all).')
    p.add_argument('--val-pair-frac', type=float, default=0.2,
                   help='Fraction of pairs held out to measure whether the inner '
                        'regression is ACTUALLY converging. Minibatch loss is '
                        'confounded by batch composition (pairs differ in Δr²); this '
                        'fixed set is not.')
    p.add_argument('--allow-cpu', action='store_true',
                   help='Permit running on CPU even when a GPU was allocated but is '
                        'unusable. Without this the run aborts rather than silently '
                        'crawling on CPU.')
    p.add_argument('--reward-form', choices=['error', 'value'], default='error',
                   help="'error' = -|logD_true - GCM(frags)|, per-molecule accuracy. Six "
                        "hyperparameter configurations all returned noise with this form, "
                        "because its direction is molecule-specific and cancels. "
                        "'value' = mean per-fragment |contribution| from the additive GCM: "
                        "a dataset-level notion of informativeness that is consistent "
                        "across molecules and fully controlled by the policy. REQUIRES "
                        "--reward-type additive (ridge has no per-fragment contributions "
                        "and returns 0 for novel fragments). Never reads y_true, so it "
                        "cannot leak -- watch the reported R2 to see whether optimising "
                        "description quality actually improves prediction.")
    p.add_argument('--faith-penalty', type=float, default=0.0,
                   help='Subtract beta*(1 - faithful_frac) from the reward. 0 = off '
                        '(accuracy only, the formulation that swept to noise). The '
                        'accuracy term rewards lucky draws; this term rewards something '
                        'the policy actually controls. A beta sweep on the 3-arm data '
                        'put the sign flip near 2.0 (reward preferring MORE faithful '
                        'sets rather than less). Reward is in logD units.')
    p.add_argument('--gate', choices=['auto', 'on', 'off'], default='auto',
                   help="Reject an iteration's update when the held-out pair slice does "
                        "not improve. 'auto' = on for T=1, off for T>1. The only run "
                        "that ever improved reward (23042651) was GATED; ungated T=100 "
                        "(23043424) degraded monotonically. But the slice must be big "
                        "enough to judge — at 32 mols/iter it is only ~6 pairs, which "
                        "rejected everything and produced a no-op run (23042656). Raise "
                        "--mols-per-iter alongside 'on'.")
    p.add_argument('--n-iterations', type=int, default=1,
                   help='REBEL outer iterations T (Algorithm 1). Each collects a FRESH '
                        'on-policy batch and takes a small step. Theorem 1 regret is '
                        'O(sqrt(1/T)+...), so T matters; T=1 is a proof-of-concept only. '
                        'cap20 cost ~0.7 min/iter, so T=100 is ~1h.')
    p.add_argument('--mols-per-iter', type=int, default=32,
                   help='Molecules sampled per iteration (|D_t| in the paper = 32). '
                        '0 = use all --n-train-mols (the old single-shot behaviour).')
    p.add_argument('--eval-every', type=int, default=20,
                   help='Evaluate on held-out test every K iterations. Per-iteration Δ '
                        'is far below one sem, so only the TREND is informative.')
    p.add_argument('--n-samples', type=int, default=5,
                   help='Draws per molecule during rollout; the pair is best-of-N vs '
                        'worst-of-N (REBEL paper Remark 2). N=2 reproduces the old '
                        'behaviour, where 27%% of gaps fell under 0.1 — almost no signal. '
                        'Cost is linear in N, gap gain is large.')
    p.add_argument('--check-every', type=int, default=5,
                   help='Held-out loss check interval. The 5h run 23031488 checked every '
                        '25 steps, so ~100 steps of damage happened before --patience '
                        'could fire.')
    p.add_argument('--patience', type=int, default=4,
                   help='Abort the inner regression after this many consecutive '
                        'held-out checks above the STARTING loss. Run 23031488 spent '
                        '5h in that state for a null result. 0 = never abort.')
    p.add_argument('--split', choices=['auto', 'paper'], default='auto',
                   help="'auto' picks the splitter the reward was fitted under (additive -> "
                        "90/10 reward split, ridge -> fit_gcm 80/10/10), which is right for "
                        "measuring REBEL itself. 'paper' uses GROVER/GODE scaffold_balanced "
                        "80/10/10 -- required for any RMSE quoted against published "
                        "baselines, and only valid if the reward model was ALSO fitted with "
                        "--split paper at the same seed.")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save-dir', default='',
                   help='Where to write the RL policy. Empty = derive from the slurm '
                        'job id under rl/checkpoints/. Two files are written at every '
                        'evaluation: pi_theta_rebel.pt (latest, so a walltime kill still '
                        'leaves something) and pi_theta_rebel_best.pt (highest mean '
                        'reward — the ungated arm degrades, so the last iterate is not '
                        'necessarily the best one).')
    return p.parse_args()


class AdditiveGCMReward:
    """The additive neural GCM behind FrozenGCM's `.predict(fragment_set)` API.

    y_hat = sum_i g(encode(frag_i)) + bias, over UNIQUE fragments. Three
    properties matter here that the ridge reward does not have:
      * a fragment the model has never seen still gets a real contribution
        (ridge looks it up and returns 0, which would train the policy to STOP
        inventing fragments -- the opposite of the thesis),
      * summing over unique fragments makes it immune to the duplicate/padding
        hack that inflates AttnPool+RF by ~11.7%,
      * every prediction decomposes into per-fragment contributions.
    """

    def __init__(self, ckpt_path, encoder_ckpt, encoder_tok, device):
        from train_additive_gcm import AdditiveHead, encode_batch
        self._encode_batch = encode_batch
        st = torch.load(ckpt_path, map_location=device, weights_only=False)
        if st.get('head_type') != 'additive':
            raise SystemExit(f"{ckpt_path} is head_type={st.get('head_type')!r}; only "
                             "'additive' exposes per-fragment contributions and is "
                             "duplicate-safe. Re-train with --head additive.")
        base = torch.load(encoder_ckpt, map_location=device, weights_only=False)
        hp = base['hparams']
        self.tok = SmartsTokenizer.load(encoder_tok)
        self.enc = SmartsGPT(vocab_size=hp['vocab_size'], n_layer=hp['n_layer'],
                             n_head=hp['n_head'], n_embd=hp['n_embd'],
                             max_len=hp['max_len'], dropout=hp['dropout']).to(device)
        self.enc.load_state_dict(st['encoder_state_dict'])
        self.enc.eval()
        # Rebuild the head at the SHAPE IT WAS TRAINED AT. Older checkpoints predate
        # 'head_dims', so infer it from the Linear weights: g.0, g.3, ... are the
        # hidden layers and g.<last> is the 1-unit output.
        dims = st.get('head_dims')
        if dims is None:
            ws = sorted((int(k.split('.')[1]), v.shape[0])
                        for k, v in st['head_state_dict'].items()
                        if k.startswith('g.') and k.endswith('.weight'))
            dims = [o for _, o in ws[:-1]] or [128]
        self.head = AdditiveHead(st['embd_dim'], hidden=tuple(dims)).to(device)
        print(f'Reward head g: {st["embd_dim"]} -> '
              + ' -> '.join(map(str, dims)) + ' -> 1')
        self.head.load_state_dict(st['head_state_dict'])
        self.head.eval()
        self.device, self.max_len = device, hp['max_len']
        self.metrics = st.get('metrics', {})
        self.fragment_vocab = []          # no vocabulary: any SMARTS is scorable
        self._cache = {}

    def _ids(self, frag):
        if frag not in self._cache:
            row = np.full(self.max_len, SPECIAL_TOKENS['<PAD>'], dtype=np.int64)
            e = self.tok.encode(frag, add_special=True)[:self.max_len]
            row[:len(e)] = e
            self._cache[frag] = row
        return self._cache[frag]

    @torch.no_grad()
    def predict(self, fragment_set):
        uniq = sorted(set(fragment_set))
        if not uniq:
            return 0.0
        idx = torch.tensor(np.stack([self._ids(f) for f in uniq]), device=self.device)
        h = self._encode_batch(self.enc, idx, SPECIAL_TOKENS['<PAD>'])
        return float(self.head(h))

    @torch.no_grad()
    def contributions(self, fragment_set):
        """Per-fragment contribution in logD units — the GCM deliverable."""
        uniq = sorted(set(fragment_set))
        if not uniq:
            return {}
        idx = torch.tensor(np.stack([self._ids(f) for f in uniq]), device=self.device)
        h = self._encode_batch(self.enc, idx, SPECIAL_TOKENS['<PAD>'])
        return dict(zip(uniq, self.head.contributions(h).cpu().numpy().tolist()))

    def coverage(self, fragment_set):
        return 1.0                        # no vocabulary, so nothing is ever a miss

    def top_fragments(self, n=20):
        return []                         # contributions are per-molecule, not global


def load_policy(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    hp = ckpt['hparams']
    tok = SmartsTokenizer.load(ckpt['tokenizer_path'])
    model = SmartsGPT(vocab_size=hp['vocab_size'], n_layer=hp['n_layer'],
                      n_head=hp['n_head'], n_embd=hp['n_embd'],
                      max_len=hp['max_len'], dropout=hp['dropout'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    # Read the conditioning representation off the checkpoint rather than a flag:
    # a prefix that disagrees with training does not error, it silently produces
    # garbage. Checkpoints predating the switch used the '<SMI>' SMILES prefix.
    prefix_mode = ckpt.get('prefix_mode', 'smiles')
    print(f"Policy prefix mode: {prefix_mode}"
          f"{' (default — not recorded in checkpoint)' if 'prefix_mode' not in ckpt else ''}")
    # Hand back the source dict too: saving the RL policy means re-emitting these
    # exact keys with new weights, so generate_smarts.py loads it unchanged.
    return model, tok, ckpt['sep_id'], ckpt['frag_sep_id'], prefix_mode, ckpt


def save_policy(model, src_ckpt, args, save_dir, iteration, ev, filename):
    """Write the RL policy in the SFT checkpoint format, plus provenance.

    Called at every evaluation, not only at the end: a 60-hour run killed by the
    walltime would otherwise leave nothing, and the point of the run is to produce
    a policy that can be generated from. Two files are kept — the latest, and the
    best by mean reward — because the ungated arm is expected to degrade, so the
    final weights are not necessarily the ones worth keeping.
    """
    os.makedirs(save_dir, exist_ok=True)
    out = {k: v for k, v in src_ckpt.items() if k != 'model_state_dict'}
    out['model_state_dict'] = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
    out['rebel_iteration'] = iteration
    path = os.path.join(save_dir, filename)
    torch.save(out, path)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                         cwd=THESIS_ROOT, text=True).strip()
    except Exception:
        commit = 'unknown'
    meta = {'source_policy': os.path.abspath(args.checkpoint),
            'reward_type': args.reward_type,
            'reward_ckpt': os.path.abspath(args.reward_ckpt),
            'iterations_done': iteration, 'n_iterations': args.n_iterations,
            'mols_per_iter': args.mols_per_iter, 'n_inner_steps': args.n_inner_steps,
            'lr': args.lr, 'eta': args.eta, 'faith_penalty': args.faith_penalty,
            # reward_form was absent here, and the sbatch never forwarded the flag,
            # so runs could not be told apart after the fact -- 23307322 looked
            # like a new experiment but was the known-dead 'error' arm.
            'reward_form': args.reward_form, 'n_samples': args.n_samples,
            'eval_samples': args.eval_samples, 'split': args.split,
            'gate': args.gate, 'seed': args.seed, 'temperature': args.temperature,
            'prefix_mode': src_ckpt.get('prefix_mode', 'smiles'),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID', ''),
            'git_commit': commit,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'eval': {k: ev[k] for k in ('mean', 'r2', 'rmse', 'faithful', 'n_frags')}}
    json.dump(meta, open(path + '.meta.json', 'w'), indent=2)


def make_prefix(smiles, tok, sep_id, prefix_mode):
    """BOS + conditioning prefix + SEP, or None if RDKit cannot parse the SMILES."""
    body = build_prefix_ids(smiles, tok, prefix_mode)
    if body is None:
        return None
    return [BOS] + body + [sep_id]


@torch.no_grad()
def sample_completion(model, tok, prefix_ids, frag_sep_id, device,
                      temperature, top_k, max_new_tokens, greedy=False):
    """Sample one completion. Returns (gen_ids, ended_eos, fragments)."""
    model.eval()
    idx = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    gen = []
    ended = False
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.max_len:])
        logits = logits[:, -1, :] / temperature
        logits[:, PAD] = float('-inf')
        logits[:, BOS] = float('-inf')
        logits[:, UNK] = float('-inf')
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            kth = torch.topk(logits, k)[0][:, -1, None]
            logits = logits.masked_fill(logits < kth, float('-inf'))
        if greedy:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        t = nxt.item()
        if t == EOS:
            ended = True
            break
        gen.append(t)
        idx = torch.cat([idx, nxt], dim=1)

    # split on frag_sep into fragments
    frags, cur = [], []
    for t in gen:
        if t == frag_sep_id:
            if cur:
                frags.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        frags.append(cur)
    frag_strs = [tok.decode(f, skip_special=True) for f in frags]
    return gen, ended, frag_strs


@torch.no_grad()
def sample_completions_batch(model, tok, prefix_ids, frag_sep_id, device,
                             temperature, top_k, max_new_tokens, n, greedy=False):
    """n completions from the SAME prefix, generated in parallel.

    Semantically identical to calling sample_completion() n times, but one forward
    pass per step instead of n. Generation was the entire cost of a REBEL run --
    64 molecules x 5 samples x ~2500 tokens, one sequence at a time on a GPU that
    fits the whole batch -- so this is ~n-fold wall-clock for free. No padding is
    needed because every row starts from the same prefix.

    Rows that emit EOS are frozen: they stop contributing tokens but stay in the
    batch (fed PAD) so the tensor remains rectangular.
    """
    model.eval()
    idx = torch.tensor([list(prefix_ids)] * n, dtype=torch.long, device=device)
    gens = [[] for _ in range(n)]
    ended = [False] * n
    live = torch.ones(n, dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.max_len:])
        logits = logits[:, -1, :] / temperature
        logits[:, PAD] = float('-inf')
        logits[:, BOS] = float('-inf')
        logits[:, UNK] = float('-inf')
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            kth = torch.topk(logits, k)[0][:, -1, None]
            logits = logits.masked_fill(logits < kth, float('-inf'))
        if greedy:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        toks = nxt.squeeze(1).tolist()
        for i, t in enumerate(toks):
            if not live[i]:
                continue
            if t == EOS:
                ended[i] = True
                live[i] = False
            else:
                gens[i].append(t)
        if not bool(live.any()):
            break
        # finished rows are fed PAD so shapes stay rectangular; their logits are
        # discarded above, so what they emit never matters.
        nxt = torch.where(live.unsqueeze(1), nxt,
                          torch.full_like(nxt, PAD))
        idx = torch.cat([idx, nxt], dim=1)

    out = []
    for i in range(n):
        frags, cur = [], []
        for t in gens[i]:
            if t == frag_sep_id:
                if cur:
                    frags.append(cur)
                cur = []
            else:
                cur.append(t)
        if cur:
            frags.append(cur)
        out.append((gens[i], ended[i],
                    [tok.decode(f, skip_special=True) for f in frags]))
    return out


def seq_logprob(model, prefix_ids, gen_ids, ended, device):
    """Sum of token log-probs of the GENERATED part (gen tokens + EOS if ended),
    under `model`, with gradient. Teacher-forces the full sequence."""
    seq = list(prefix_ids) + list(gen_ids) + ([EOS] if ended else [])
    if len(seq) < 2:
        return torch.zeros((), device=device)
    seq = seq[:model.max_len]
    x = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
    tgt = torch.tensor(seq[1:], dtype=torch.long, device=device)
    logits, _ = model(x)                     # [1, T-1, V]
    logp = F.log_softmax(logits[0], dim=-1)  # [T-1, V]
    tok_logp = logp[torch.arange(len(tgt)), tgt]
    # score only positions that predict a GENERATED token: target index >= len(prefix)
    start = len(prefix_ids) - 1              # first t whose target is gen_ids[0]
    return tok_logp[start:].sum()


def reward(gcm, frags, y_true, clip=100.0, smiles=None, faith_penalty=0.0,
           form='error'):
    """-|logD_true - GCM(frags)|, optionally minus a faithfulness penalty.

    The accuracy term alone is only PARTLY controllable by the policy: best-of-N
    picks whichever completion's prediction happened to land nearest y_true, so it
    selects for a lucky draw as much as for a better fragment set. That is the
    leading explanation for why every accuracy-only sweep came back as noise
    (jobs 23046317-22: six configs, all within 1 sem).

    Faithfulness -- whether the emitted fragments actually occur in the molecule --
    IS fully controllable by the policy, so it gives REBEL something learnable.
    It is also the metric the reward currently cannot track: within-policy the
    reward preferred the more faithful of two samples only 47.6% of the time
    (chance = 50%).
    """
    if form == 'value':
        # Dataset-level notion of quality, in the spirit of the reward in the prior
        # GCM/RL thesis (Humeedat 2025), where reward is the change in the GCM's
        # error over the WHOLE validation set -- so "better" means the same thing
        # on every molecule.
        #
        # Our -|error| reward is per-molecule, and that is measurably why it fails:
        # the preferred sample differs from the rejected one by ~0 on every
        # attribute the policy controls (faithfulness +0.002, fragment count +0.28),
        # and 52% of winners over-predict while 48% under-predict, so the update
        # directions cancel across molecules.
        #
        # Here each fragment contributes its |contribution to logD| ONLY IF it
        # actually occurs in the molecule. Faithfulness enters as a GATE, not as a
        # competing penalty, which removes the weighting problem: measured on real
        # output, |contribution| and a faithfulness penalty differ in spread by
        # 5-22x, so any beta large enough to stop hallucination drowns the
        # informativeness signal. Gating instead of penalising needs no beta at all,
        # and a hallucinated fragment scores 0 however large its contribution --
        # closing the hole where the reward preferred a hallucinated hydroxyl
        # (+0.778) over genuinely present fragments (+0.042).
        #
        # SCALE: x25 puts reward gaps in the same range as the error form (~0.6),
        # so the tuned lr/eta stay valid. Two independently trained reward models
        # order candidate sets the same way 86.9% of the time (the error form
        # managed 98%, but reproducibly measured something unlearnable).
        # y_true is never read, so this cannot leak.
        uniq = sorted(set(frags))
        if not uniq:
            return -1.0                        # an empty set describes nothing
        c = gcm.contributions(uniq)
        if not c:
            return -1.0
        tot = sum(abs(v) * (1.0 if faithful_frac(smiles, [f]) > 0 else 0.0)
                  for f, v in c.items()) if smiles is not None else \
              sum(abs(v) for v in c.values())
        return 25.0 * tot / len(uniq)
    err = min(abs(y_true - gcm.predict(frags)), clip)
    r = -err
    if faith_penalty > 0 and smiles is not None:
        # empty fragment set counts as fully unfaithful, so emitting nothing is
        # never a way to dodge the penalty
        ff = faithful_frac(smiles, frags) if frags else 0.0
        r -= faith_penalty * (1.0 - ff)
    return r


def eval_mean_reward(model, tok, sep_id, frag_sep_id, mols, gcm, device, args, prefix_mode):
    """Estimate E_{y~pi}[reward] on `mols`.

    REBEL optimises the SAMPLING distribution, so evaluating with greedy/argmax
    decoding measures the mode and can stay flat even when the sampled
    distribution genuinely improves. So by default we draw --eval-samples
    completions per molecule at the training temperature and average.
    (--eval-samples 0 falls back to greedy.)

    Returns a dict with:
      per_mol : per-molecule mean reward, ALIGNED across calls so before/after can
                be compared PAIRED. Molecule difficulty dominates the spread of
                reward and cancels in a paired difference, which is ~3x tighter
                than treating the two evals as independent samples.
      mean, sem, hicoef, r2 (frozen-GCM R2 on generated fragments — reported for
      comparability with the thesis metric, but it is NOISIER than reward
      (headroom/sem 4.8 vs 6.0) so it is not the significance test).
    """
    greedy = args.eval_samples <= 0
    k = 1 if greedy else args.eval_samples
    # Common random numbers: same seed before and after the update so the
    # before/after comparison isn't swamped by sampling noise.
    torch.manual_seed(args.seed + 12345)
    per_mol, hicoef, y_true, y_pred, faith, nfrag = [], [], [], [], [], []
    top = {f for f, _ in gcm.top_fragments(n=50)}
    for smi, y in mols:
        prefix = make_prefix(smi, tok, sep_id, prefix_mode)
        if prefix is None:
            continue
        rs, hs, ps, fs, ns = [], [], [], [], []
        draws = sample_completions_batch(
            model, tok, prefix, frag_sep_id, device,
            args.temperature, args.top_k, args.max_new_tokens, k, greedy=greedy)
        for _, _, frags in draws:
            rs.append(reward(gcm, frags, y, args.reward_clip, smiles=smi,
                             faith_penalty=args.faith_penalty, form=args.reward_form))
            ps.append(gcm.predict(frags))
            hs.append(sum(1 for f in frags if f in top))
            # Faithfulness is the guard against reward hacking: the reward is a
            # model, and "reward went up" is NOT evidence the policy improved.
            # If reward rises while faithfulness falls, RL found a hole.
            fs.append(faithful_frac(smi, frags) if frags else 0.0)
            ns.append(len(set(frags)))
        per_mol.append(float(np.mean(rs)))
        hicoef.append(float(np.mean(hs)))
        faith.append(float(np.mean(fs)))
        nfrag.append(float(np.mean(ns)))
        y_true.append(y)
        y_pred.append(float(np.mean(ps)))
    per_mol = np.asarray(per_mol)
    sem = float(np.std(per_mol) / np.sqrt(max(1, len(per_mol))))
    return {
        'per_mol': per_mol,
        'mean': float(per_mol.mean()),
        'sem': sem,
        'hicoef': float(np.mean(hicoef)),
        'faithful': float(np.mean(faith)),
        'faithful_sem': float(np.std(faith) / np.sqrt(max(1, len(faith)))),
        'n_frags': float(np.mean(nfrag)),
        'r2': float(r2_score(np.array(y_true), np.array(y_pred))),
        # RMSE in logD units. R2 alone is not comparable across runs whose test
        # sets differ in variance (R2 = 1 - SSE/SStot), and RMSE is the number the
        # GCM tables elsewhere report, so log both.
        'rmse': float(np.sqrt(mean_squared_error(np.array(y_true), np.array(y_pred)))),
        # Kept per-molecule and ALIGNED (same order, same skips) so RMSE gets the
        # same paired treatment as reward. RMSE is the thesis's headline metric,
        # and a before/after RMSE with no uncertainty cannot be reported.
        'y_true': np.asarray(y_true, dtype=float),
        'y_pred': np.asarray(y_pred, dtype=float),
    }


def collect_pairs(model, tok, sep_id, frag_sep_id, mols, gcm, device, args,
                  prefix_mode, verbose=True):
    """Best-of-N vs worst-of-N pairs for one REBEL iteration.

    REBEL paper (Gao et al. 2024) Remark 2: "Setting mu to be the best-of-N of
    pi_t makes mu cover higher quality comparator policies. Selecting nu_t as the
    worst-of-N of pi_t still ensures coverage to pi_t while at the same time
    INCREASING the reward gap r(x,y) - r(x,y')."  With N=2 both draws are typical
    samples and the gap is small; measured on cap20, N=2 left 20% of pairs under
    |dr|=0.1 while best-of-5 left ZERO and doubled the median gap.
    """
    pairs = []
    n_draw = max(2, args.n_samples)
    for smi, y in mols:
        prefix = make_prefix(smi, tok, sep_id, prefix_mode)
        if prefix is None:
            continue
        draws = []
        for g, e, f in sample_completions_batch(
                model, tok, prefix, frag_sep_id, device,
                args.temperature, args.top_k, args.max_new_tokens, n_draw):
            if f:
                draws.append((reward(gcm, f, y, args.reward_clip, smiles=smi,
                                     faith_penalty=args.faith_penalty,
                                     form=args.reward_form), g, e))
        if len(draws) < 2:
            continue
        draws.sort(key=lambda d: d[0])
        (r_lo, g_lo, e_lo), (r_hi, g_hi, e_hi) = draws[0], draws[-1]
        dr = r_hi - r_lo
        if abs(dr) < 1e-6:
            continue
        # y = best-of-N (higher reward), y' = worst-of-N, so dr is positive.
        pairs.append((prefix, g_hi, e_hi, g_lo, e_lo, dr))
    if verbose and pairs:
        gaps = np.abs([p[5] for p in pairs])
        print(f'  collected {len(pairs)} pairs | |Δr| mean={gaps.mean():.3f} '
              f'median={np.median(gaps):.3f} <0.1={100*np.mean(gaps<0.1):.0f}%')
    return pairs


def solve_regression(model, ref, pairs, args, rng, device, verbose=True, gate=True):
    """One REBEL least-squares solve. Returns (start_loss, best_loss, n_steps).

    Held out a slice of pairs because the minibatch loss is confounded by which
    pairs land in the batch (they differ in dr^2) and so cannot distinguish
    "learning" from "easy batch". The solve is only working if held-out loss
    drops BELOW its starting value -- at init pi_theta == pi_ref, so pred_gap is
    0 and the starting value is exactly mean(dr^2).
    """
    n_val = max(1, int(len(pairs) * args.val_pair_frac))
    perm = rng.permutation(len(pairs))
    val_pairs = [pairs[i] for i in perm[:n_val]]
    train_pairs = [pairs[i] for i in perm[n_val:]]
    if not train_pairs:
        return None, None, 0

    def pair_loss(pair, grad=True):
        prefix, g1, e1, g2, e2, dr = pair
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            lp1 = seq_logprob(model, prefix, g1, e1, device)
            lp2 = seq_logprob(model, prefix, g2, e2, device)
        with torch.no_grad():
            r1 = seq_logprob(ref, prefix, g1, e1, device)
            r2 = seq_logprob(ref, prefix, g2, e2, device)
        return (((lp1 - r1) - (lp2 - r2)) / args.eta - dr) ** 2

    @torch.no_grad()
    def val_loss():
        return float(np.mean([pair_loss(p, grad=False).item() for p in val_pairs]))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # eval() (NOT train()) during the regression: dropout=0.1 makes ln pi_theta
    # stochastic while pi_ref is scored in eval mode, injecting ~18 nats of noise
    # into a signal whose real gaps are ~0.7. Gradients still flow.
    model.eval()
    v0 = val_loss()
    best_v, best_state, worse_streak = v0, None, 0
    steps_done = 0
    for step in range(args.n_inner_steps):
        batch = [train_pairs[i] for i in rng.choice(
            len(train_pairs), min(args.batch_pairs, len(train_pairs)), replace=False)]
        loss = torch.stack([pair_loss(p) for p in batch]).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        steps_done = step + 1
        is_last = (step + 1) == args.n_inner_steps
        if (step + 1) % args.check_every == 0 or step == 0 or is_last:
            vl = val_loss()
            if verbose:
                print(f'    step {step+1:3d} | batch {loss.item():.4f} | held-out {vl:.4f} '
                      f'({"OK" if vl < v0 else "worse"})', flush=True)
            if vl < best_v:
                best_v, worse_streak = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif vl > v0:
                worse_streak += 1
                if args.patience > 0 and worse_streak >= args.patience:
                    if verbose:
                        print(f'    early stop @ {step+1} (held-out above start x{worse_streak})')
                    break
    # Keep the best iterate, not whatever the last step left behind: batch loss
    # falls while held-out rises, so the final weights are usually not the best.
    if best_state is not None:
        model.load_state_dict(best_state)
    elif gate:
        # single-iteration proof: refuse a step the held-out slice rejects
        model.load_state_dict({k: v.detach().clone() for k, v in ref.state_dict().items()})
    # multi-iteration: keep the step. REBEL's trust region comes from eta/lr, not
    # from a 6-pair validation gate; gating every iteration meant the policy was
    # reset to pi_ref 100 times and never moved at all (job 23042656).
    return v0, best_v, steps_done


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cpu' and os.environ.get('CUDA_VISIBLE_DEVICES') and not args.allow_cpu:
        raise RuntimeError(
            'CUDA_VISIBLE_DEVICES is set but torch.cuda.is_available() is False — '
            'a GPU was allocated but is not usable (bad node/driver?). Aborting '
            'instead of silently running on CPU for hours. Re-submit to get a '
            'different node, or pass --allow-cpu to run on CPU anyway.'
        )

    model, tok, sep_id, frag_sep_id, prefix_mode, src_ckpt = load_policy(
        args.checkpoint, device)
    save_dir = args.save_dir or os.path.join(
        THESIS_ROOT, 'rl', 'checkpoints',
        f"rebel_g{args.gate}_j{os.environ.get('SLURM_JOB_ID', 'local')}")
    best_reward = -float('inf')
    ref = copy.deepcopy(model)          # π_θt snapshot (frozen)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    if args.reward_type == 'additive':
        gcm = AdditiveGCMReward(args.reward_ckpt, args.encoder_ckpt,
                                args.encoder_tok, device)
        print(f"Reward: additive neural GCM  test R2={gcm.metrics.get('R2')}  "
              f"(no vocabulary — novel fragments are scored, duplicates ignored)")
    else:
        gcm = FrozenGCM.load(args.reward_pkl)
        print(f'Reward: frozen ridge, {len(gcm.fragment_vocab)} fragments, '
              f"test R2={gcm.metrics.get('test_r2'):.3f}")
        print('  WARNING: ridge returns 0 for any fragment outside its vocabulary, '
              'so this reward penalises novelty. Control runs only.')

    # molecules + true logD
    rows = [json.loads(l) for l in open(args.dataset) if l.strip()]
    smiles = [r['smiles'] for r in rows]
    y_all = [float(r['exp']) for r in rows]

    # Split with the SAME splitter the reward was fit under, otherwise REBEL's
    # held-out molecules land inside the reward model's training set and the
    # before/after comparison is measured where the reward is optimistically
    # accurate. The two splitters are close to inverses of each other.
    if args.split == 'paper':
        # Benchmark-comparable protocol (GROVER/GODE scaffold_balanced 80/10/10).
        # val is held out from RL training too: the GCM selects its checkpoint on
        # it, so training the policy there would contaminate the reported number.
        tr_idx, _va, te_idx = scaffold_balanced(smiles, seed=args.seed)
        print(f'Split: paper scaffold_balanced 80/10/10 seed={args.seed} '
              f'(train={len(tr_idx)}, test={len(te_idx)})')
        print('  NOTE: the reward model must have been fitted with --split paper at the '
              'SAME seed, or these test molecules sit inside its training set.')
    elif args.reward_type == 'additive':
        tr_idx, te_idx = reward_scaffold_split(smiles, len(smiles), test_size=0.1)
        print(f'Split: reward-consistent scaffold (train={len(tr_idx)}, test={len(te_idx)})')
    else:
        tr_idx, _va, te_idx = ridge_scaffold_split(smiles, seed=args.seed)
        print(f'Split: fit_gcm scaffold (train={len(tr_idx)}, test={len(te_idx)})')
    rng = np.random.RandomState(args.seed)
    tr_sel = rng.choice(tr_idx, min(args.n_train_mols, len(tr_idx)), replace=False)
    te_sel = te_idx[:args.n_test_mols]
    train_mols = [(smiles[i], y_all[i]) for i in tr_sel]
    test_mols = [(smiles[i], y_all[i]) for i in te_sel]
    print(f'Train mols: {len(train_mols)}  Test mols: {len(test_mols)}')

    # ---- pre-update evaluation on held-out test ----
    mode = 'greedy' if args.eval_samples <= 0 else f'{args.eval_samples} samples/mol @ T={args.temperature}'
    print(f'\nEval mode: {mode}')
    pre = eval_mean_reward(model, tok, sep_id, frag_sep_id, test_mols,
                           gcm, device, args, prefix_mode)
    print(f"[BEFORE] test mean reward = {pre['mean']:.4f} (+/-{pre['sem']:.4f} sem) | "
          f"R2 = {pre['r2']:+.4f} | RMSE = {pre['rmse']:.4f} | faithful = {100*pre['faithful']:.1f}% "
          f"(+/-{100*pre['faithful_sem']:.1f}) | frags/mol = {pre['n_frags']:.1f}")

    # ---- REBEL iterations ----
    # Algorithm 1 of the paper is: for t = 0..T-1, collect a SMALL fresh on-policy
    # batch D_t and solve the regression. Their |D_t| = 32. Running one iteration
    # with 800 pairs and 300 gradient steps (what this script used to do) is not
    # REBEL -- steps 2..300 optimise against a policy that no longer exists, and
    # held-out loss rose in every such run. Theorem 1's regret is O(sqrt(1/T) + ...),
    # so T matters; a single iteration was never going to be conclusive.
    history = [(0, pre)]
    n_solves = n_improved = 0
    n_iters = max(1, args.n_iterations)
    per_iter = args.mols_per_iter if args.mols_per_iter > 0 else len(train_mols)
    if n_iters > 1:
        print(f'\nRunning {n_iters} REBEL iterations, {per_iter} molecules each '
              f'(best-of-{args.n_samples}), {args.n_inner_steps} inner steps, '
              f'eval every {args.eval_every}')
    for t in range(n_iters):
        # pi_ref = pi_theta_t, refreshed EVERY iteration (the KL anchor is the
        # current policy, not the original SFT model).
        ref = copy.deepcopy(model)
        ref.eval()
        for p_ in ref.parameters():
            p_.requires_grad_(False)
        sel = rng.choice(len(train_mols), min(per_iter, len(train_mols)), replace=False)
        mols_t = [train_mols[i] for i in sel]
        verbose = n_iters == 1 or (t + 1) % args.eval_every == 0 or t == 0
        if n_iters > 1 and verbose:
            print(f'\n--- iteration {t+1}/{n_iters} ---')
        pairs = collect_pairs(model, tok, sep_id, frag_sep_id, mols_t, gcm,
                              device, args, prefix_mode, verbose=verbose)
        if len(pairs) < 2:
            print(f'  iteration {t+1}: fewer than 2 usable pairs — skipping.')
            continue
        if args.min_gap > 0:
            pairs = [p for p in pairs if abs(p[5]) >= args.min_gap]
            if len(pairs) < 2:
                continue
        gate = (n_iters == 1) if args.gate == 'auto' else (args.gate == 'on')
        v0, bv, ns = solve_regression(model, ref, pairs, args, rng, device,
                                      verbose=verbose, gate=gate)
        n_solves += 1
        if v0 is not None and bv < v0:
            n_improved += 1
        if verbose and v0 is not None:
            tag = 'improved' if bv < v0 else 'NO improvement'
            print(f'  solve: held-out {v0:.4f} -> {bv:.4f} ({tag}) in {ns} steps')
        if (t + 1) % args.eval_every == 0 or t == n_iters - 1:
            ev = eval_mean_reward(model, tok, sep_id, frag_sep_id, test_mols,
                                  gcm, device, args, prefix_mode)
            history.append((t + 1, ev))
            print(f"  [iter {t+1:3d}] reward {ev['mean']:+.4f} | R2 {ev['r2']:+.4f} | "
                  f"RMSE {ev['rmse']:.4f} | faithful {100*ev['faithful']:.1f}% | "
                  f"frags/mol {ev['n_frags']:.1f}",
                  flush=True)
            save_policy(model, src_ckpt, args, save_dir, t + 1, ev, 'pi_theta_rebel.pt')
            tag = 'latest'
            if ev['mean'] > best_reward:
                best_reward = ev['mean']
                save_policy(model, src_ckpt, args, save_dir, t + 1, ev,
                            'pi_theta_rebel_best.pt')
                tag = 'latest + BEST'
            print(f'    saved policy ({tag}) -> {save_dir}', flush=True)

    # ---- post-update evaluation ----
    # The loop already evaluates at t == n_iters-1, so reuse it. Re-running cost a
    # full extra eval per job (5 min on cap20, 90 min on uncapped).
    if len(history) > 1:
        post = history[-1][1]
    else:
        post = eval_mean_reward(model, tok, sep_id, frag_sep_id, test_mols,
                                gcm, device, args, prefix_mode)
        # Single-iteration runs never enter the in-loop save path above.
        save_policy(model, src_ckpt, args, save_dir, n_iters, post, 'pi_theta_rebel.pt')
        if post['mean'] > best_reward:
            save_policy(model, src_ckpt, args, save_dir, n_iters, post,
                        'pi_theta_rebel_best.pt')
    print(f"\n[AFTER]  test mean reward = {post['mean']:.4f} (+/-{post['sem']:.4f} sem) | "
          f"R2 = {post['r2']:+.4f} | RMSE = {post['rmse']:.4f} | faithful = {100*post['faithful']:.1f}% "
          f"(+/-{100*post['faithful_sem']:.1f}) | frags/mol = {post['n_frags']:.1f}")

    # Reward going up is NOT evidence the policy improved -- the reward is a model
    # and RL exploits whatever it can. Faithfulness is measured on the SAME
    # held-out molecules and is not part of the objective, so it is the
    # independent check: reward up + faithfulness down means RL found a hole.
    df = post['faithful'] - pre['faithful']
    print(f"[GUARD]  faithfulness change = {100*df:+.2f} pp   "
          f"fragments/mol change = {post['n_frags'] - pre['n_frags']:+.2f}")
    if df < -0.02:
        print("  *** WARNING: faithfulness DROPPED >2pp while optimising reward. "
              "Treat any reward gain as reward hacking, not improvement. ***")

    # PAIRED comparison: same molecules before and after, so per-molecule difficulty
    # (which dominates the spread of reward) cancels. This is ~3x tighter than
    # sqrt(pre_sem^2 + post_sem^2), which wrongly assumes the two evals are independent.
    d = post['per_mol'] - pre['per_mol']
    delta = float(d.mean())
    delta_sem = float(np.std(d, ddof=1) / np.sqrt(len(d)))
    unpaired_sem = float(np.sqrt(pre['sem'] ** 2 + post['sem'] ** 2))
    rho = float(np.corrcoef(pre['per_mol'], post['per_mol'])[0, 1])

    if n_solves:
        print(f'\n  solve success: {n_improved}/{n_solves} iterations improved held-out '
              f'loss ({100*n_improved/n_solves:.0f}%)  <- the prerequisite; reward cannot '
              f'be judged until this is high')
    if len(history) > 2:
        print('\n--- reward trend across iterations ---')
        print(f"  {'iter':>6s} {'reward':>9s} {'R2':>8s} {'RMSE':>8s} {'faithful':>9s} {'frags/mol':>10s}")
        for it, ev in history:
            print(f"  {it:6d} {ev['mean']:+9.4f} {ev['r2']:+8.4f} {ev['rmse']:8.4f} "
                  f"{100*ev['faithful']:8.1f}% {ev['n_frags']:10.1f}")

    print(f'\n================ REBEL ({n_iters} iteration{"s" if n_iters>1 else ""}) ================')
    print(f'  eval mode        : {mode}')
    print(f"  test mean reward : {pre['mean']:.4f} -> {post['mean']:.4f}")
    print(f'  PAIRED Δ         : {delta:+.4f} +/- {delta_sem:.4f} sem   (n={len(d)}, ρ={rho:.3f})')
    print(f'                     [unpaired sem would be {unpaired_sem:.4f} — '
          f'{unpaired_sem/max(delta_sem,1e-9):.1f}x looser]')
    # Paired ΔRMSE, reported two ways.
    #
    # (a) PAIRED BOOTSTRAP -- the primary statistic, and what the model-comparison
    #     literature recommends when two models are scored on the SAME test set.
    #     Resample molecule INDICES once per replicate and recompute both RMSEs on
    #     that identical resample, so molecule difficulty cancels within every
    #     replicate; only the model difference drives the spread. Independence-based
    #     SEs on an RMSE difference are overstated because the two RMSEs are
    #     positively correlated. It also makes no linearisation assumption, which
    #     matters here: n=100 test molecules is squarely in the range where
    #     CLT-based intervals are unreliable.
    #
    # (b) DELTA METHOD -- kept for continuity with earlier runs and the report.
    #     Pairs per-molecule squared error, then maps ΔMSE to ΔRMSE through
    #     d(sqrt(m))/dm = 1/(2*sqrt(m)) about the BEFORE value. Valid only while
    #     ΔMSE << MSE; if the two disagree, trust the bootstrap.
    e_pre = (pre['y_true'] - pre['y_pred']) ** 2
    e_post = (post['y_true'] - post['y_pred']) ** 2
    de = e_post - e_pre
    dmse, dmse_sem = float(de.mean()), float(np.std(de, ddof=1) / np.sqrt(len(de)))
    scale = 1.0 / (2.0 * max(pre['rmse'], 1e-9))
    print(f"  frozen-GCM RMSE  : {pre['rmse']:.4f} -> {post['rmse']:.4f}  "
          f"(Δ {post['rmse'] - pre['rmse']:+.4f} logD)")

    n_boot, rs = 10000, np.random.default_rng(0)
    idx = rs.integers(0, len(de), size=(n_boot, len(de)))
    boot = (np.sqrt(e_post[idx].mean(1)) - np.sqrt(e_pre[idx].mean(1)))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # Two-sided bootstrap p: how often the resampled difference falls on the
    # opposite side of zero from the observed one.
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    sig = 'SIGNIFICANT (95% CI excludes 0)' if lo > 0 or hi < 0 else 'not significant'
    print(f"    PAIRED ΔRMSE   : {post['rmse'] - pre['rmse']:+.4f} logD  "
          f"[95% CI {lo:+.4f}, {hi:+.4f}]  p={p:.3f}  ({n_boot} paired bootstrap) — "
          f"{sig}   (negative = RMSE improved)")
    print(f"      delta method : {dmse * scale:+.4f} +/- {dmse_sem * scale:.4f} sem  "
          f"({'agrees' if (dmse * scale < 0) == (post['rmse'] < pre['rmse']) else 'DISAGREES'} "
          f"in sign with the bootstrap)")
    print(f"  frozen-GCM R2    : {pre['r2']:+.4f} -> {post['r2']:+.4f}  (informational; "
          f'noisier than reward)')
    print(f"  high-coef frags  : {pre['hicoef']:.2f} -> {post['hicoef']:.2f}  "
          f"(Δ {post['hicoef']-pre['hicoef']:+.2f})")
    print('  (reward = -|logD error|; higher = better. Positive Δ = REBEL helped.)')
    print(f'  reference: ceiling (ground-truth frags) ~= -0.739, floor (no frags) ~= -1.656')
    print(f'  policy saved     : {save_dir}')
    print(f'                     pi_theta_rebel.pt (iter {history[-1][0]}) | '
          f'pi_theta_rebel_best.pt (reward {best_reward:+.4f})')
    if abs(delta) < 2 * delta_sem:
        print(f'  VERDICT: INCONCLUSIVE — |Δ| is within 2 paired sem ({2*delta_sem:.4f}).')
    elif delta > 0:
        print('  VERDICT: IMPROVED — Δ exceeds 2 paired sem.')
    else:
        print('  VERDICT: REGRESSED — Δ exceeds 2 paired sem in the wrong direction.')


if __name__ == '__main__':
    main()
