#!/bin/bash
#SBATCH --job-name=gen_smarts
#SBATCH --partition=gpuidle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gen_%j.out
#SBATCH --error=logs/gen_%j.err

# Config
CHECKPOINT=${CHECKPOINT:-/home/pyq02mab/Thesis/sft/checkpoints/smaprefix_all_50k_j23015580/pi_theta_epoch007_valid0.9993_loss0.5444.pt} #/home/pyq02mab/Thesis/sft/checkpoints/smitok_all20k_j22869097/pi_theta_epoch010_valid0.9990_loss0.5975.pt
TOKENIZER=/home/pyq02mab/Thesis/sft/checkpoints/smaprefix_all_50k_j23015580/tokenizer.json
INPUT_CSV=/home/pyq02mab/Thesis/data/smiles_dataset/Lipophilicity.csv
SMILES_COL=smiles
WORKDIR=/home/pyq02mab/Thesis/sft
OUTPUT=${OUTPUT:-$WORKDIR/results/lipo_smarts_smaprefix_all_50k_epoch007.jsonl}
PYTHON=/home/pyq02mab/.conda/envs/smi-ted-env/bin/python

MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2900}
TEMPERATURE=${TEMPERATURE:-0.4}
TOP_K=40

# Consensus decoding (fix #2): draw N_SAMPLES per molecule, keep only fragments
# proposed in >= MIN_CONSENSUS of them. N_SAMPLES=1/MIN_CONSENSUS=1 == old behaviour.
MIN_UNIQUE=${MIN_UNIQUE:-0}
N_SAMPLES=${N_SAMPLES:-1}
MIN_CONSENSUS=1

[ -e "$OUTPUT" ] && { echo "ERROR: $OUTPUT already exists — refusing to overwrite (concurrent writes corrupted a previous run)."; exit 1; }
mkdir -p "$WORKDIR/logs" "$WORKDIR/results"
cd "$WORKDIR"

echo "=============================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPU        : $CUDA_VISIBLE_DEVICES"
echo "Checkpoint : $CHECKPOINT"
echo "Start      : $(date)"
echo "=============================="

$PYTHON -u generate_smarts.py \
    --input-csv      "$INPUT_CSV" \
    --smiles-col     "$SMILES_COL" \
    --checkpoint     "$CHECKPOINT" \
    --tokenizer      "$TOKENIZER" \
    --output         "$OUTPUT" \
    --max-new-tokens $MAX_NEW_TOKENS \
    --min-unique   $MIN_UNIQUE \
    --temperature    $TEMPERATURE \
    --top-k          $TOP_K \
    --n-samples      $N_SAMPLES \
    --min-consensus  $MIN_CONSENSUS \
    --log-every      50

if [ $? -ne 0 ]; then echo "ERROR: generate_smarts.py failed."; exit 1; fi

echo ""
echo "=============================="
echo "Done : $(date)"
echo "Output -> $OUTPUT"
echo "=============================="
