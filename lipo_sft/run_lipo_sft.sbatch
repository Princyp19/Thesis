#!/bin/bash
#SBATCH --job-name=lipo_sft
#SBATCH --partition=gpuidle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/lipo_sft_%j.out
#SBATCH --error=logs/lipo_sft_%j.err

# ---------------------------------------------------------------------------
# Domain fine-tuning on Lipophilicity molecules.
#
# Data: build_lipo_sft_data.py writes ONLY the scaffold-TRAIN molecules (3,359 of
# 4,199). The val/test molecules are the ones the GCM is scored on, so training on
# them would make the reported R2 meaningless. Verified zero overlap.
#
# BASE CHECKPOINT — start from the SFT model, not the raw pre-trained one:
#   3,359 molecules is far too few to learn molecule->fragments from scratch. The
#   existing SFT checkpoint already does that task at 85.8% faithfulness on these
#   very molecules; this run is domain ADAPTATION, not task learning.
#   Set BASE=pubchem for the control that shows the PubChem SFT stage mattered.
#
#   BASE=sft      (default) smaprefix_all_50k_j23015580 ep010
#   BASE=pubchem            pretraining/checkpoints/pubchem_gpt ep010
#
# Usage:
#   sbatch --exclude=gpu003 run_lipo_sft.sbatch
#   BASE=pubchem sbatch --exclude=gpu003 run_lipo_sft.sbatch
# ---------------------------------------------------------------------------

THESIS=/home/pyq02mab/Thesis
PYTHON=/home/pyq02mab/.conda/envs/smi-ted-env/bin/python
WORKDIR=$THESIS/sft            # train_sft.py lives here

DATA=${DATA:-$THESIS/data/sft/sft_warmup_lipo_train.jsonl}
BASE=${BASE:-sft}
if [ "$BASE" = "sft" ]; then
    CKPT=$THESIS/sft/checkpoints/smaprefix_all_50k_j23015580/pi_theta_epoch010_valid0.9984_loss0.5341.pt
    VOCAB=$THESIS/sft/checkpoints/smaprefix_all_50k_j23015580/tokenizer.json
    TAGBASE=fromsft
else
    CKPT=$THESIS/pretraining/checkpoints/pubchem_gpt/checkpoint_epoch010_ppl3.75_nov0.38.pt
    VOCAB=$THESIS/pretraining/checkpoints/pubchem_gpt/tokenizer.json
    TAGBASE=frompubchem
fi

PREFIX_MODE=${PREFIX_MODE:-smarts}
# Fine-tuning, not training from scratch: fewer epochs and a lower LR than the
# 50k runs (which used 1.5e-4 for 10 epochs on 15x more data). Too much here
# overwrites what the PubChem SFT stage learned.
EPOCHS=${EPOCHS:-5}
LR=${LR:-5e-5}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_LEN=${MAX_LEN:-3072}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2900}
WARMUP_STEPS=${WARMUP_STEPS:-400}

TAG=${TAG:-lipo_${TAGBASE}}
OUTPUT_DIR=$THESIS/sft/checkpoints/${TAG}_j${SLURM_JOB_ID}

mkdir -p "$THESIS/lipo_sft/logs" "$OUTPUT_DIR"
cd "$WORKDIR"

echo "=============================="
echo "Job $SLURM_JOB_ID  Node: $SLURMD_NODENAME  ($(date))"
echo "Data        : $(basename $DATA)"
echo "Base        : $BASE -> $(basename $(dirname $CKPT))/$(basename $CKPT)"
echo "Prefix mode : $PREFIX_MODE"
echo "epochs=$EPOCHS lr=$LR batch=$BATCH_SIZE max_len=$MAX_LEN"
echo "Output      : $OUTPUT_DIR"
echo "=============================="

$PYTHON -c 'import torch,sys; sys.exit(0 if torch.cuda.device_count() else 1)' \
    || { echo "ERROR: no usable GPU on $SLURMD_NODENAME — resubmit elsewhere."; exit 1; }
[ -f "$DATA" ] || { echo "ERROR: $DATA not found — run build_lipo_sft_data.py first."; exit 1; }

$PYTHON -u train_sft.py \
    --data           "$DATA" \
    --checkpoint     "$CKPT" \
    --vocab          "$VOCAB" \
    --output-dir     "$OUTPUT_DIR" \
    --prefix-mode    "$PREFIX_MODE" \
    --epochs         $EPOCHS \
    --lr             $LR \
    --batch-size     $BATCH_SIZE \
    --max-len        $MAX_LEN \
    --warmup-steps   $WARMUP_STEPS \
    --max-new-tokens $MAX_NEW_TOKENS || { echo "ERROR: train_sft.py failed."; exit 1; }

echo ""
echo "=============================="
echo "Done : $(date)"
echo "Checkpoints -> $OUTPUT_DIR"
echo "Next: generate with sft/run_generate_smarts.sbatch (temp 0.4, N=1,"
echo "      max_new=2900), then fit the GCM and compare against 0.257."
echo "=============================="
