#!/bin/bash
#SBATCH --job-name=sft_smarts_gpt
#SBATCH --partition=gpuidle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err

# ---------------------------------------------------------------------------
# CAP SWEEP — the three arms are derived from ONE uncapped file by
# derive_capped_warmup.py, so they share identical molecules (19,839) and differ
# only in the fragment cap. Set DATA/MAX_LEN/BATCH_SIZE together from this table;
# MAX_LEN below the p95 silently truncates targets (train_sft reports how many).
#
#   arm       DATA                            frags/mol  p95 tok  MAX_LEN  BATCH warmup
#   cap20     sft_warmup_sweep_cap20.jsonl         20       256      512    128     200
#   cap80     sft_warmup_sweep_cap80.jsonl         78       800     1024     48     500
#   uncapped  sft_warmup_all_20k.jsonl            168      2845     3072      4     5500
#
# p95/MAX_LEN measured with the 'smarts' prefix. cap20 and cap80 fit entirely
# (longest is 474 / 832); uncapped truncates ~4% of molecules even at 3072.
#
# (uncapped at batch 4 is what the earlier all20k runs used — larger OOMs.)
# ---------------------------------------------------------------------------

# Config
DATA=/home/pyq02mab/Thesis/data/sft/sft_warmup_50k_cap20.jsonl
CHECKPOINT=/home/pyq02mab/Thesis/pretraining/checkpoints/pubchem_gpt/checkpoint_epoch010_ppl3.75_nov0.38.pt
VOCAB=/home/pyq02mab/Thesis/pretraining/checkpoints/pubchem_gpt/tokenizer.json
WORKDIR=/home/pyq02mab/Thesis/sft
PYTHON=/home/pyq02mab/.conda/envs/smi-ted-env/bin/python

# Conditioning prefix. 'smarts' = whole molecule written as SMARTS in the
# existing fragment vocab, so the prefix already carries the exact atom
# primitives ('[C&H2&R]') the model must emit plus full connectivity — 95-99% of
# hallucinated fragments were using a primitive absent from the molecule.
# 'smiles' = legacy <SMI>-namespaced prefix. Recorded in the checkpoint;
# generate_smarts.py reads it back, so generation cannot drift from training.
PREFIX_MODE="smarts"

# Naming convention: checkpoint dir = <tag>_j<jobid> so the folder name alone
# tells you which run it came from. Put the ARM and the PREFIX_MODE in the tag —
# the warmup filename alone does not say which prefix trained the model.
TAG="smaprefix_50k_cap20"
OUTPUT_DIR=$WORKDIR/checkpoints/${TAG}_j${SLURM_JOB_ID}

EPOCHS=15
LR=1.5e-4
BATCH_SIZE=128
MAX_LEN=512
WARMUP_STEPS=200
MAX_NEW_TOKENS=256

mkdir -p "$WORKDIR/logs" "$OUTPUT_DIR"
cd "$WORKDIR"

echo "=============================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPU         : $CUDA_VISIBLE_DEVICES"
echo "Data        : $(basename $DATA)"
echo "Prefix mode : $PREFIX_MODE"
echo "max_len     : $MAX_LEN   batch: $BATCH_SIZE"
echo "Start       : $(date)"
echo "=============================="

$PYTHON train_sft.py \
    --data          "$DATA" \
    --checkpoint    "$CHECKPOINT" \
    --vocab         "$VOCAB" \
    --output-dir    "$OUTPUT_DIR" \
    --prefix-mode   "$PREFIX_MODE" \
    --epochs        $EPOCHS \
    --lr            $LR \
    --batch-size    $BATCH_SIZE \
    --max-len       $MAX_LEN \
    --warmup-steps  $WARMUP_STEPS \
    --max-new-tokens $MAX_NEW_TOKENS

if [ $? -ne 0 ]; then echo "ERROR: train_sft.py failed."; exit 1; fi

echo ""
echo "=============================="
echo "Done : $(date)"
echo "Checkpoints -> $OUTPUT_DIR/pi_theta_0.pt, $OUTPUT_DIR/pi_ref.pt"
echo "=============================="
