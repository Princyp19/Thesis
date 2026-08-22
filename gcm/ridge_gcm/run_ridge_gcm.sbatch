#!/bin/bash
#SBATCH --job-name=gcm_fit
#SBATCH --partition=gpuidle
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=gcm/logs/gcm_%j.log

# CPU-only job — Ridge regression needs no GPU.

PYTHON=/home/pyq02mab/.conda/envs/smi-ted-env/bin/python
WORKDIR=/home/pyq02mab/Thesis/gcm/ridge_gcm

# Naming convention: results land in gcm/results/ridge/<experiment>/.
# Experiment names follow <source>_<detail>, e.g. gt_cap20, gen_all20k_uncapped.
mkdir -p /home/pyq02mab/Thesis/gcm/logs /home/pyq02mab/Thesis/gcm/results/ridge
cd "$WORKDIR"

$PYTHON fit_gcm.py \
--dataset /home/pyq02mab/Thesis/sft/results/lipo_smarts_generated_uncapped.jsonl \
--smiles-col smiles \
--target-col exp \
--frags-col fragments \
--experiment gen_all20k_uncapped \
--output-dir ../results/ridge/

  # --dataset /home/pyq02mab/Thesis/data/gcm/Lipo_smarts_v2.1.jsonl \
  # --frags-col fragments \
  # --experiment gt_cap40 \
  # --output-dir ../results/ridge/

