# SMARTS Fragment Generation for Interpretable Property Prediction

A language model is trained to **generate** SMARTS substructure fragments for a
molecule, and an additive group-contribution model (GCM) predicts lipophilicity
(logD) from those fragments alone. Because the prediction is a sum of per-fragment
contributions, every prediction decomposes into a chemically readable explanation —
something graph neural networks do not provide.

Performance is measured as **test RMSE on MoleculeNet Lipophilicity**, under the
same scaffold-split protocol as the published graph-neural baselines, so the
interpretability is bought against a directly comparable number. Reinforcement
learning (REBEL) is the lever for improving the generated fragments.

---

## Scoreboard

Lipophilicity test RMSE (lower is better). Literature values as reported by their
authors; ours measured under the protocol in [How RMSE is measured](#how-rmse-is-measured).

| Method                                                        | RMSE                     |                                                          |
| ------------------------------------------------------------- | ------------------------ | -------------------------------------------------------- |
| MSHG-MAE                                                      | 0.465                    | ⚠️ different split protocol — not directly comparable |
| KANO_CMPNN                                                    | 0.641                    | strongest comparable baseline                            |
| KANO_GTrans                                                   | 0.651                    |                                                          |
| MPNN                                                          | 0.672 ± 0.051           |                                                          |
| DMPNN                                                         | 0.683 ± 0.016           |                                                          |
| GCN                                                           | 0.712 ± 0.049           |                                                          |
| Hu et al.                                                     | 0.739 ± 0.003           |                                                          |
| GODE                                                          | 0.743 ± 0.043           |                                                          |
| MolCLR_GTrans                                                 | 0.767 ± 0.064           |                                                          |
| MolCLR                                                        | 0.789 ± 0.009           |                                                          |
| **Ours — additive neural GCM, ground-truth fragments** | **0.799 ± 0.034** | best single run 0.715                                    |
| GIN                                                           | 0.850 ± 0.071           |                                                          |
| KGE_NFM w/ MolKG                                              | 0.877 ± 0.071           |                                                          |
| GROVER_Large                                                  | 0.890 ± 0.050           |                                                          |
| **Ours — additive neural GCM, generated fragments**    | **0.895 ± 0.054** | best single run 0.816                                    |
| SchNet                                                        | 0.909 ± 0.098           |                                                          |
| Ours — ridge GCM, generated fragments                        | 1.004                    | earlier, additive-ridge baseline                         |
| MGCN                                                          | 1.113 ± 0.041           |                                                          |

Full metrics: `baselines_rmse.xlsx` and `gcm_results_comparison.xlsx`.

### Status

REBEL: **15/20 iterations, RMSE 0.878** (policy eval on 300 molecules with sampled
completions — a different protocol from the table above, not directly comparable).

---

## Pipeline

```
                    SMILES (Lipophilicity, 4200 molecules)
                              │
              SMILES-Decomposer│  rule-based decomposition
                              ▼
                    ground-truth SMARTS fragments
                              │
                          SFT │  teach a policy to emit fragments
                              │  given the molecule as a SMARTS prefix
                              ▼
                     generated SMARTS fragments
                              │
        additive neural GCM   │  SmartsGPT encoder → per-fragment
                              │  contribution → sum
                              ▼
                            logD  →  RMSE
                              │
                        REBEL │  reward = GCM accuracy
                              └──────────► back to the policy
```

**Two gaps define the work:**

| Gap              | From → To     | Owner                                                            |
| ---------------- | -------------- | ---------------------------------------------------------------- |
| Fragment quality | 0.895 → 0.799 | REBEL — make generated fragments as informative as ground truth |
| Model capacity   | 0.799 → lower  | the GCM — a better predictor over the same fragments            |

Any experiment should say which gap it attacks.

---

## How RMSE is measured

Comparability with the baseline table depends entirely on this, so it is fixed:

- **Dataset:** MoleculeNet Lipophilicity, 4,200 molecules, target logD.
- **Split:** `scaffold_balanced` 80/10/10 (`gcm/splits.py`) — the Bemis–Murcko
  balanced scaffold protocol used by GROVER and GODE.
- **Repeats:** 3 split seeds (`SPLIT_SEED` 0,1,2) × 5 init seeds (`INIT_SEED`).
  Reported value is the mean over inits within a split, then mean ± std across
  splits.
- **Validation** is used for checkpoint selection only, never for reporting.
- **Never quote a single run.** Init spread within one fixed split is 0.03–0.11
  RMSE — larger than most architecture differences we measured.

Two other splitters exist in the codebase (`fit_gcm.scaffold_split` and the
reward-side splitter) and are close to inverses of each other. A policy and the
GCM that rewards it must use the *same* splitter at the *same* seed, or the
policy's held-out molecules sit inside the reward model's training set.

---

## Repo map

```
SMILES-Decomposer/   rule-based SMILES → SMARTS fragment decomposition,
                     warmup-dataset builders (build_warmup_dataset.py,
                     derive_capped_warmup.py)
pretraining/         SmartsGPT: the PubChem-pretrained fragment LM used as the
                     encoder everywhere downstream (smarts_gpt_model.py)
sft/                 supervised fine-tuning of the generator + sampling
                     (train_sft.py, generate_smarts.py) → results/*.jsonl
lipo_sft/            builds the Lipophilicity-only SFT set, train-split-only, so
                     the generator never sees a test molecule's decomposition
gcm/
  splits.py          scaffold_balanced — the canonical split
  faithfulness.py    do emitted fragments actually match the molecule?
  ridge_gcm/         additive ridge GCM (fit_gcm.py) — the earlier baseline
  nn_gcm/            additive neural GCM (train_additive_gcm.py) — current
  results/ridge/     21 ridge experiments
  results/nn/        57 neural experiments, each self-describing via results.json
rl/                  REBEL (rebel_proof.py) — RL against the GCM reward
data/                decompositions and SFT warmup sets (not distributed)
yliterature/         reference papers
```

---

## Running

Entry points, in pipeline order:

| Stage                 | Submit                                     | Key environment variables                                                         |
| --------------------- | ------------------------------------------ | --------------------------------------------------------------------------------- |
| Decompose             | `SMILES-Decomposer/run_decompose.sbatch` |                                                                                   |
| Pretrain encoder      | `pretraining/train_smarts_gpt.py`        |                                                                                   |
| SFT the generator     | `sft/run_sft.sbatch`                     | warmup set,`PREFIX_MODE`                                                        |
| Lipophilicity SFT set | `lipo_sft/run_lipo_sft.sbatch`           | `--split paper`                                                                 |
| Generate fragments    | `sft/run_generate_smarts.sbatch`         | `CHECKPOINT`, `TEMPERATURE`, `N_SAMPLES`                                    |
| Ridge GCM             | `gcm/ridge_gcm/run_ridge_gcm.sbatch`     | dataset                                                                           |
| Neural GCM            | `gcm/nn_gcm/run_additive_gcm.sbatch`     | `JSONL`, `HEAD`, `ENCODER_LR`, `HEAD_DIMS`, `SPLIT_SEED`, `INIT_SEED` |
| REBEL                 | `rl/run_rebel_proof.sbatch`              | `CHECKPOINT`, `REWARD_TYPE`, `REWARD_CKPT`, `SPLIT`                       |

---

## Conventions

**Naming.** SFT checkpoints are `sft/checkpoints/<warmup-tag>_j<slurm-job-id>/`.
GCM results are `gcm/results/{ridge,nn}/<source>_<detail>/`, where `gt_*` is
ground-truth decomposition and `gen_*` is model-generated; the neural grid adds
`_s<split-seed>i<init-seed>_d<head-dims>`.

**Provenance.** Every artifact records what produced it, so no result depends on a
surviving job log:

| Artifact            | Sidecar                    | Records                                       |
| ------------------- | -------------------------- | --------------------------------------------- |
| SFT checkpoint      | `train_config.json`      | warmup set, hyperparameters, git commit       |
| Generated fragments | `<file>.jsonl.meta.json` | source checkpoint, temperature, sampling      |
| GCM result          | `results.json`           | input dataset, head, seeds, metrics           |
| REBEL checkpoint    | `<file>.pt.meta.json`    | source policy, reward model, iterations, eval |

Writers for these live inside the training code, never in a post-hoc script.
