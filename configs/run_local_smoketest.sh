#!/bin/bash
# ============================================================
#  LOCAL SMOKE TEST — RTX 3050 4GB VRAM
#  Runs ONE object at reduced resolution to verify the full
#  pipeline works end-to-end before submitting to cluster.
#
#  Usage: bash configs/run_local_smoketest.sh
# ============================================================

set -e
source configs/local_rtx3050.sh

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

echo "================================================================"
echo "LOCAL SMOKE TEST — RTX 3050 4GB"
echo "Image size: ${IMAGE_SIZE}x${IMAGE_SIZE} | Detector: $DETECTOR_COUNT"
echo "Batch size: $BATCH_SIZE | Epochs: $EPOCHS"
echo "================================================================"

# STEP 1: One object only
echo ""
echo "--- Step 1: gVXR Projection (whistle, Ti, standard noise) ---"
python DATACREATION/gvxr_projection_script.py \
    --stl "$GVXR_STL" \
    --output_dir data/smoketest_whistle \
    --material "$GVXR_MATERIAL" \
    --i0 $GVXR_I0 \
    --gaussian-std $GVXR_GAUSSIAN_STD \
    --scan-method centered

# STEP 2: FDK reconstruction on the generated sample
echo ""
echo "--- Step 2: FDK Reconstruction ---"
python scripts/classical_reconstruction/reconstruct_fdk.py \
    --sample-dir data/smoketest_whistle_centered \
    --downsample-factor $DOWNSAMPLE_FACTOR \
    --output-dir outputs/smoketest_fdk

# STEP 3: Build training dataset
echo ""
echo "--- Step 3: Build Dataset ---"
python scripts/data_preparation/01_build_dataset.py \
    --sample-dir data/smoketest_whistle_centered \
    --target-volume outputs/smoketest_fdk/fdk_volume.tif \
    --output-path outputs/smoketest_dataset.npz \
    --downsample-factor $DOWNSAMPLE_FACTOR \
    --sparse-step $SPARSE_STEP \
    --detector-count $DETECTOR_COUNT \
    --image-size $IMAGE_SIZE \
    --slice-stride $SLICE_STRIDE

# STEP 4: Training
echo ""
echo "--- Step 4: PureDL Training (${EPOCHS} epochs) ---"
python scripts/pure_dl/02_train_pure_dl.py \
    --dataset-path outputs/smoketest_dataset.npz \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning-rate $LEARNING_RATE \
    --val-fraction $VAL_FRACTION \
    --scan-method centered

echo ""
echo "================================================================"
echo "Smoke test complete! Check outputs/pure_dl_training_centered/"
echo "================================================================"
