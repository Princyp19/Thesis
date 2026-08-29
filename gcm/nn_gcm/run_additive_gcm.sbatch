#!/bin/bash
#SBATCH --job-name=add_gcm
#SBATCH --partition=gpuidle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/addgcm_%j.out
#SBATCH --error=logs/addgcm_%j.err

# ---------------------------------------------------------------------------
# End-to-end neural GCM with a TRAINABLE SMARTS encoder.
#
# Runs one cell of the 2x2 (head x frozen/unfrozen). Everything is overridable
# from the environment, so all four cells use this one file:
#
#   cd /home/pyq02mab/Thesis/gcm/nn_gcm
#   J=$PWD/../../sft/results/lipo_smarts_smaprefix_all_50k.jsonl
#
#   HEAD=additive ENCODER_LR=0    JSONL=$J sbatch --exclude=gpu003 run_additive_gcm.sbatch
#   HEAD=attnpool ENCODER_LR=0    JSONL=$J sbatch --exclude=gpu003 run_additive_gcm.sbatch
#   HEAD=additive ENCODER_LR=1e-5 JSONL=$J sbatch --exclude=gpu003 run_additive_gcm.sbatch
#   HEAD=attnpool ENCODER_LR=1e-5 JSONL=$J sbatch --exclude=gpu003 run_additive_gcm.sbatch
#
# ENCODER_LR=0 freezes the encoder (reproduces the precomputed-embedding
# setting), so the 2x2 separates "does unfreezing help?" from "what does
# additivity cost?".
#
# Only HEAD=additive yields a per-fragment contribution table
# (fragment_contributions.csv) -- attnpool pools then applies a nonlinear MLP,
# so its output cannot be decomposed per fragment at any accuracy.
#
# Reference points on the SAME generated fragments (scaffold split):
#   ridge GCM                          R2 = 0.257
#   AttnPool+RF on frozen embeddings   R2 = 0.050
#   additive head, frozen, 2 epochs    R2 = 0.241   (smoke test, cap20)
# ---------------------------------------------------------------------------

THESIS=/home/pyq02mab/Thesis
PYTHON=/home/pyq02mab/.conda/envs/smi-ted-env/bin/python
WORK=$THESIS/gcm/nn_gcm

JSONL=${JSONL:-$THESIS/sft/results/lipo_smarts_smaprefix_all_50k.jsonl}
FRAGS_COL=${FRAGS_COL:-fragments}
HEAD=${HEAD:-additive}
ENCODER_LR=${ENCODER_LR:-1e-5}
HEAD_LR=${HEAD_LR:-1e-3}
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-8}
SPLIT=${SPLIT:-scaffold}
SEED=${SEED:-42}
# SPLIT_SEED picks the test set (vary 0,1,2 -> mean+/-std, the GODE protocol).
# INIT_SEED picks the weights/batch order at a FIXED test set (vary 0..4 and
# average their test_predictions.npz -> the ensemble). Both default to SEED, so
# omitting them reproduces earlier runs exactly.
SPLIT_SEED=${SPLIT_SEED:-$SEED}
INIT_SEED=${INIT_SEED:-$SEED}
HEAD_DIMS=${HEAD_DIMS:-128}
DROPOUT=${DROPOUT:-0.1}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
# Per-molecule contribution workbook (additive head only).
#   scope=test -> the held-out molecules the reported RMSE is computed on
#   scope=all  -> all molecules, so FragmentTable's n_molecules counts the WHOLE
#                 dataset; train/val rows are FIT molecules, not generalisation
CONTRIB_SCOPE=${CONTRIB_SCOPE:-test}
CONTRIB_MATRIX_COLS=${CONTRIB_MATRIX_COLS:-2000}
CONTRIB_SHEETS=${CONTRIB_SHEETS:-About,Summary,Matrix,PerMolecule,FragmentTable}

# Tag encodes the cell so the four runs never overwrite each other.
if [ "$ENCODER_LR" = "0" ]; then ENCTAG=frozen; else ENCTAG=enc${ENCODER_LR}; fi
SRC=$(basename "$JSONL" .jsonl | sed 's/^lipo_smarts_//')
# Seed/split/dims MUST be in the tag: without them, seeds 1 and 2 of the same
# config write to ONE directory and clobber each other (this is how the epoch007
# generation file was corrupted by two concurrent jobs). BOTH seeds now appear
# (s<split>i<init>) for the same reason -- five init seeds of one split are five
# concurrent jobs that would otherwise share a directory.
TAG=${TAG:-${SRC}_${HEAD}_${ENCTAG}_${SPLIT}_s${SPLIT_SEED}i${INIT_SEED}_d${HEAD_DIMS//,/-}}
OUT=$THESIS/gcm/results/nn/$TAG

mkdir -p "$WORK/logs" "$OUT"
cd "$WORK"

echo "=============================="
echo "Job $SLURM_JOB_ID  on $SLURMD_NODENAME  ($(date))"
echo "Data      : $(basename $JSONL)"
echo "Head      : $HEAD"
echo "Encoder LR: $ENCODER_LR  $([ "$ENCODER_LR" = "0" ] && echo '(FROZEN)')"
echo "Head LR   : $HEAD_LR   epochs=$EPOCHS  batch=$BATCH_SIZE  split=$SPLIT"
echo "Seeds     : split=$SPLIT_SEED (test set)  init=$INIT_SEED (weights/batch order)"
echo "Output    : $OUT"
echo "=============================="

# Fail fast on a node whose GPU isn't actually usable (gpu003 has handed out
# CUDA_VISIBLE_DEVICES with device_count()==0). Unfrozen training on CPU would
# take days rather than hours, so this is worth aborting for.
$PYTHON -c 'import torch,sys; sys.exit(0 if torch.cuda.device_count() else 1)' \
    || { echo "ERROR: no usable GPU on $SLURMD_NODENAME — resubmit elsewhere."; exit 1; }

[ -f "$JSONL" ] || { echo "ERROR: $JSONL not found."; exit 1; }

$PYTHON -u train_additive_gcm.py \
    --jsonl       "$JSONL" \
    --frags-col   "$FRAGS_COL" \
    --head        "$HEAD" \
    --encoder-lr  "$ENCODER_LR" \
    --head-lr     "$HEAD_LR" \
    --epochs      $EPOCHS \
    --batch-size  $BATCH_SIZE \
    --split       "$SPLIT" \
    --split-seed  $SPLIT_SEED \
    --init-seed   $INIT_SEED \
    --head-dims   "$HEAD_DIMS" \
    --dropout     $DROPOUT \
    --weight-decay $WEIGHT_DECAY \
    --contrib-scope       "$CONTRIB_SCOPE" \
    --contrib-matrix-cols $CONTRIB_MATRIX_COLS \
    --contrib-sheets      "$CONTRIB_SHEETS" \
    --output      "$OUT" || { echo "ERROR: train_additive_gcm.py failed."; exit 1; }

echo ""
echo "=============================="
echo "Done $(date)"
echo "Results -> $OUT/results.json"
[ "$HEAD" = "additive" ] && echo "Contributions -> $OUT/fragment_contributions.csv"
echo "=============================="
