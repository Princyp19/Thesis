#!/bin/bash
#SBATCH --job-name=rebel_proof
#SBATCH --partition=gpuidle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rebel_%j.out
#SBATCH --error=logs/rebel_%j.err

# ---------------------------------------------------------------------------
# One REBEL iteration: does an RL step actually improve the SFT policy?
#
# Defaults pair the best base policy (uncapped SMARTS-prefix, 85.8% faithful,
# ridge R2=0.257) with the ADDITIVE neural GCM reward (scaffold R2=0.35 vs 0.23
# for ridge on the same split). Everything is env-overridable:
#
#   sbatch --exclude=gpu003 run_rebel_proof.sbatch                      # defaults
#   REWARD_TYPE=ridge sbatch --exclude=gpu003 run_rebel_proof.sbatch    # control
#   CHECKPOINT=<cap20.pt> MAX_NEW_TOKENS=256 sbatch ... run_rebel_proof.sbatch
#
# READING THE RESULT — reward going up is NOT enough. The reward is a model and
# RL exploits models. Three things must hold TOGETHER:
#   1. reward-gap diagnostic: if most |dr| are tiny, the regression is fitting
#      reward-model noise rather than fragment quality. Stop if so.
#   2. [AFTER] mean reward > [BEFORE] by more than the paired sem.
#   3. [GUARD] faithfulness must NOT drop. It is measured on the same held-out
#      molecules and is not part of the objective, so a reward gain with a
#      faithfulness loss means RL found a hole in the reward.
# ---------------------------------------------------------------------------

THESIS=/home/pyq02mab/Thesis
PYTHON=/home/pyq02mab/.conda/envs/smi-ted-env/bin/python
WORKDIR=$THESIS/rl

CHECKPOINT=${CHECKPOINT:-$THESIS/sft/checkpoints/smaprefix_all_50k_j23015580/pi_theta_epoch010_valid0.9984_loss0.5341.pt}
REWARD_TYPE=${REWARD_TYPE:-additive}
REWARD_CKPT=${REWARD_CKPT:-$THESIS/gcm/results/nn/smaprefix_all_50k_additive_enc5e-5/additive_gcm.pt}
REWARD_PKL=${REWARD_PKL:-$THESIS/gcm/results/ridge/gt_cap20_old20/gcm_lipo_gt_cap20_old20_frozen.pkl}
DATASET=${DATASET:-$THESIS/data/gcm/Lipo_smarts_v2.jsonl}

# --- settings carried over from the earlier tuned runs (do not "clean up") ---
# lr 1e-6: 1e-5 overshot badly — log-ratios moved ~1-2 nats when the targets need
#   ~0.3-0.6, so held-out loss ROSE above its starting point instead of falling.
# eval-samples 4: measure E[reward] under the SAMPLING distribution, which is what
#   REBEL optimises. The first run used greedy decoding, which measures the mode
#   and stayed flat regardless of the update.
# reward-clip 4.0: stops one wildly-mispredicted molecule dominating the
#   regression. NOTE this was calibrated against the RIDGE reward's error scale;
#   the additive GCM may need a different value — check the gap diagnostic.
LR=${LR:-1e-6}
EVAL_SAMPLES=${EVAL_SAMPLES:-4}
REWARD_CLIP=${REWARD_CLIP:-4.0}
N_TRAIN_MOLS=${N_TRAIN_MOLS:-800}
N_TEST_MOLS=${N_TEST_MOLS:-300}
N_INNER_STEPS=${N_INNER_STEPS:-300}
BATCH_PAIRS=${BATCH_PAIRS:-32}
ETA=${ETA:-1.0}
MIN_GAP=${MIN_GAP:-0.0}
# Abort the inner solve if held-out loss stays above its starting value: run
# 23031488 burned 5h in exactly that state (batch loss fell, held-out doubled).
PATIENCE=${PATIENCE:-4}
# Best-of-N vs worst-of-N pairing (REBEL paper Remark 2): ranks N draws and pairs
# the extremes, which maximises the reward gap per unit of generation.
N_SAMPLES=${N_SAMPLES:-5}
CHECK_EVERY=${CHECK_EVERY:-5}
# REBEL Algorithm 1: T outer iterations, each on a SMALL fresh on-policy batch
# (paper |D_t|=32). T=1 with 800 pairs x 300 steps is NOT REBEL and diverged in
# every run. cap20 costs ~0.7 min/iteration, so T=100 is about an hour.
N_ITERATIONS=${N_ITERATIONS:-1}
MOLS_PER_ITER=${MOLS_PER_ITER:-32}
EVAL_EVERY=${EVAL_EVERY:-20}
GATE=${GATE:-auto}
# Adds beta*(1-faithful) penalty: gives REBEL a term the policy CONTROLS, unlike
# accuracy which best-of-N selects for partly by luck. 0 = accuracy only.
FAITH_PENALTY=${FAITH_PENALTY:-0.0}
VAL_PAIR_FRAC=${VAL_PAIR_FRAC:-0.2}
# Reward form. 'error' = -|y_true - GCM(frags)|: its sign is molecule-specific
# (the reward model over-predicts 43% of molecules and under-predicts 57%), so
# best-of-N asks for "predict lower" on some and "predict higher" on others over
# one shared vocabulary and the updates cancel -- six configurations returned
# noise (23046317-22) and so did 23307322. This flag existed in rebel_proof.py
# but was NEVER forwarded here, so every run silently used 'error'.
# 'value' = mean per-fragment |contribution|, gated on faithfulness: one
# direction for every molecule, fully policy-controlled, never reads y_true.
# Requires REWARD_TYPE=additive (ridge has no per-fragment contributions).
REWARD_FORM=${REWARD_FORM:-error}

# Temperature 0.4 is the measured best for faithfulness on every checkpoint
# (74.5% -> 80.9% on filt20). The earlier runs used 1.0, before that was known.
TEMPERATURE=${TEMPERATURE:-0.4}
TOP_K=${TOP_K:-40}
# Generation budget MUST match the policy: 256 for cap20, 800 for cap80, 2900 for
# uncapped. Too small silently truncates completions part-way through the set.
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2900}
SPLIT=${SPLIT:-auto}
SEED=${SEED:-0}

mkdir -p "$WORKDIR/logs"
cd "$WORKDIR"

echo "=============================="
echo "Job ID    : $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
echo "Policy    : $(basename $(dirname $CHECKPOINT))/$(basename $CHECKPOINT)"
echo "Reward    : $REWARD_TYPE $([ "$REWARD_TYPE" = additive ] && basename $(dirname $REWARD_CKPT))"
echo "Sampling  : T=$TEMPERATURE  max_new=$MAX_NEW_TOKENS  eval_samples=$EVAL_SAMPLES"
echo "Reward fm : $REWARD_FORM   faith_penalty=$FAITH_PENALTY   seed=$SEED  split=$SPLIT"
echo "Molecules : train=$N_TRAIN_MOLS  test=$N_TEST_MOLS   lr=$LR  clip=$REWARD_CLIP"
echo "Start     : $(date)"
echo "=============================="

# Abort rather than crawl: rollout + 300 inner steps on CPU would take days, and
# gpu003 has handed out CUDA_VISIBLE_DEVICES with device_count()==0.
$PYTHON -c 'import torch,sys; sys.exit(0 if torch.cuda.device_count() else 1)' \
    || { echo "ERROR: no usable GPU on $SLURMD_NODENAME — resubmit elsewhere."; exit 1; }

[ -f "$CHECKPOINT" ] || { echo "ERROR: policy $CHECKPOINT not found."; exit 1; }
if [ "$REWARD_TYPE" = "additive" ]; then
    [ -f "$REWARD_CKPT" ] || { echo "ERROR: reward $REWARD_CKPT not found."; exit 1; }
fi

# -u = unbuffered so progress shows live in the log
$PYTHON -u rebel_proof.py \
    --checkpoint     "$CHECKPOINT" \
    --reward-type    "$REWARD_TYPE" \
    --reward-ckpt    "$REWARD_CKPT" \
    --reward-pkl     "$REWARD_PKL" \
    --dataset        "$DATASET" \
    --n-train-mols   $N_TRAIN_MOLS \
    --n-test-mols    $N_TEST_MOLS \
    --n-inner-steps  $N_INNER_STEPS \
    --batch-pairs    $BATCH_PAIRS \
    --eta            $ETA \
    --lr             $LR \
    --temperature    $TEMPERATURE \
    --top-k          $TOP_K \
    --max-new-tokens $MAX_NEW_TOKENS \
    --reward-clip    $REWARD_CLIP \
    --eval-samples   $EVAL_SAMPLES \
    --min-gap        $MIN_GAP \
    --val-pair-frac  $VAL_PAIR_FRAC \
    --patience       $PATIENCE \
    --n-samples      $N_SAMPLES \
    --check-every    $CHECK_EVERY \
    --n-iterations   $N_ITERATIONS \
    --mols-per-iter  $MOLS_PER_ITER \
    --eval-every     $EVAL_EVERY \
    --gate           $GATE \
    --split          "$SPLIT" \
    --seed           $SEED \
    --reward-form    "$REWARD_FORM" \
    --faith-penalty  $FAITH_PENALTY || { echo "ERROR: rebel_proof.py failed."; exit 1; }

echo "=============================="
echo "Done : $(date)"
echo "Check in order: reward-gap diagnostic -> [BEFORE]/[AFTER] -> [GUARD]"
echo "=============================="
