#!/bin/bash
#SBATCH --job-name=gcm_fit
#SBATCH --partition=gpuidle
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=gcm/logs/gcm_%j.log

# CPU-only job — Ridge regression needs no GPU.

PYTHON=/home/pyq02mab/.conda/envs/smi-ted-env/bin/python
WORKDIR=/home/pyq02mab/Thesis/gcm

mkdir -p "$WORKDIR/logs" "$WORKDIR/results"
cd "$WORKDIR"

$PYTHON fit_gcm.py \
  --dataset /home/pyq02mab/Thesis/sft/results/lipophilicity_smarts_sft_test.jsonl \
  --frags-col fragments \
  --experiment sft_generated_smarts \
  --output-dir results/

