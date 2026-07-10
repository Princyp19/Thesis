#!/bin/bash
#SBATCH --job-name=sft_smarts_gpt
#SBATCH --partition=downtime-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err

# Config
DATA=/home/pyq02mab/Thesis/data/sft/sft_warmup_filtered.jsonl
CHECKPOINT=/home/pyq02mab/Thesis/pretraining/checkpoints/pubchem_gpt/checkpoint_epoch010_ppl3.75_nov0.38.pt
VOCAB=/home/pyq02mab/Thesis/pretraining/checkpoints/pubchem_gpt/tokenizer.json
WORKDIR=/home/pyq02mab/Thesis/sft
OUTPUT_DIR=$WORKDIR/checkpoints/job_${SLURM_JOB_ID}
PYTHON=/home/pyq02mab/.conda/envs/smi-ted-env/bin/python

EPOCHS=5
LR=1e-4
BATCH_SIZE=256
MAX_LEN=256
WARMUP_STEPS=200
MAX_NEW_TOKENS=150

mkdir -p "$WORKDIR/logs" "$OUTPUT_DIR"
cd "$WORKDIR"

echo "=============================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "GPU      : $CUDA_VISIBLE_DEVICES"
echo "Start    : $(date)"
echo "=============================="

$PYTHON train_sft.py \
    --data          "$DATA" \
    --checkpoint    "$CHECKPOINT" \
    --vocab         "$VOCAB" \
    --output-dir    "$OUTPUT_DIR" \
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
