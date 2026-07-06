#!/bin/bash
#SBATCH --job-name=puredl_ct
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.log
#SBATCH --error=logs/slurm_%j.err

# ============================================================
#  SLURM Job Script — PureDL Full Pipeline (32GB GPU)
#  Usage: sbatch configs/cluster_32gb.slurm
#
#  Runs all 3 steps sequentially:
#    Step 1: gVXR projection generation (all 5 STLs x 4 variations)
#    Step 2: FDK ground truth + dataset building
#    Step 3: PureDL training
# ============================================================

set -e
echo "Job started on $(hostname) at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "GPU assigned: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Activate your conda environment (change name if different)
source activate ct_pipeline || conda activate ct_pipeline

REPO=/path/to/gVXRsimulation_PureDLReconstruction_Pipeline  # <-- SET THIS
cd "$REPO"
mkdir -p logs

# --------------------------------------------------
# Load cluster config
# --------------------------------------------------
source configs/cluster_32gb.sh

# --------------------------------------------------
# STEP 1: Generate gVXR Projections (all objects)
# --------------------------------------------------
echo "========================================"
echo "STEP 1: gVXR Projection Generation"
echo "========================================"
python DATACREATION/generate_datasets.py

# --------------------------------------------------
# STEP 2: FDK Reconstruction + Dataset Building
#         (run for each generated sample directory)
# --------------------------------------------------
echo "========================================"
echo "STEP 2: FDK Reference + Dataset Building"
echo "========================================"
python scripts/run_batch_pipeline.py \
    --downsample-factor $DOWNSAMPLE_FACTOR \
    --detector-count $DETECTOR_COUNT \
    --image-size $IMAGE_SIZE \
    --sparse-step $SPARSE_STEP \
    --slice-stride $SLICE_STRIDE

# --------------------------------------------------
# STEP 3: PureDL Training
# --------------------------------------------------
echo "========================================"
echo "STEP 3: Training PureDL Model"
echo "========================================"
python scripts/pure_dl/02_train_pure_dl.py \
    --dataset-path outputs/batch_datasets \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning-rate $LEARNING_RATE \
    --val-fraction $VAL_FRACTION

echo "========================================"
echo "Pipeline completed at $(date)"
echo "========================================"
