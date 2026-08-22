#!/bin/bash
#SBATCH --job-name=decompose
#SBATCH --partition=uds-hub
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120
#SBATCH --output=SMILES-Decomposer/logs/decompose_%j.out
#SBATCH --error=SMILES-Decomposer/logs/decompose_%j.err

# CPU-only RDKit job — no GPU needed.
#
# Partition note: uds-hub has PreemptMode=OFF and a 21-day limit. Do NOT move
# this to gpuidle/b200idle — those are PreemptMode=REQUEUE, which is what kills
# long runs partway through. epyc-256 / haswell-64 (30-day) are fine too.
#
# The job is restart-safe: build_warmup_dataset.py streams each molecule to the
# output JSONL and skips SMILES already present on resume. If this job is killed,
# preempted, or hits the walltime, just resubmit the SAME command and it picks up
# where it stopped. Pass --no-resume to force a clean rebuild.

# Config
PYTHON=/home/pyq02mab/.conda/envs/smi-ted-env/bin/python
WORKDIR=/home/pyq02mab/Thesis
INPUT_CSV=$WORKDIR/data/sft/Lipophilicity.csv
SMILES_COL=smiles
OUT_PREFIX=$WORKDIR/data/gcm/Lipo_smarts_all_uncapped

MAX_FRAGS=1000
MAX_ATOMS=5
MIN_ATOMS=2
# SAMPLE_SIZE=50000
SEED=42

mkdir -p "$WORKDIR/SMILES-Decomposer/logs" "$WORKDIR/data/sft"
cd "$WORKDIR/SMILES-Decomposer"

echo "=============================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Restart #  : ${SLURM_RESTART_COUNT:-0}"
echo "Input      : $INPUT_CSV"
echo "Output     : ${OUT_PREFIX}.jsonl"
echo "Max frags  : $MAX_FRAGS"
echo "Start      : $(date)"
echo "=============================="

if [ -f "${OUT_PREFIX}.jsonl" ]; then
    echo "Existing output: $(wc -l < ${OUT_PREFIX}.jsonl) molecules — will resume."
fi

# On preemption/walltime, Slurm sends USR1 120s early. The output JSONL is
# already flushed incrementally, so we just requeue and let resume do the work.
trap 'echo "Signal received at $(date) — requeueing $SLURM_JOB_ID"; \
      scontrol requeue $SLURM_JOB_ID; exit 15' USR1

srun $PYTHON build_warmup_dataset.py \
    --csv         "$INPUT_CSV" \
    --smiles-col  "$SMILES_COL" \
    --out-prefix  "$OUT_PREFIX" \
    --max-frags   $MAX_FRAGS \
    --max-atoms   $MAX_ATOMS \
    --min-atoms   $MIN_ATOMS \
    # --sample-size $SAMPLE_SIZE \
    --seed        $SEED &

wait $!
STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "ERROR: build_warmup_dataset.py exited $STATUS"
    exit $STATUS
fi

echo ""
echo "=============================="
echo "Done : $(date)"
echo "Output -> ${OUT_PREFIX}.jsonl ($(wc -l < ${OUT_PREFIX}.jsonl) molecules)"
echo "=============================="
