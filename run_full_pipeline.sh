#!/bin/bash
set -e

# Setup conda activation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct_pipeline

echo "================================================================"
echo "Step 1: Generating Projections using gVXR"
echo "================================================================"
python -u DATACREATION/generate_datasets.py

echo "================================================================"
echo "Step 2: Classical Reconstruction & Dataset Generation"
echo "================================================================"
python -u scripts/run_batch_pipeline.py

echo "================================================================"
echo "Step 3: Training Pure DL Model on Generated Dataset"
echo "================================================================"
python -u scripts/pure_dl/02_train_pure_dl.py --dataset-path outputs/batch_datasets

echo "================================================================"
echo "Step 4: DL Inference — Reconstruct 3D Volume with Trained Model"
echo "================================================================"
# Find the first valid sample directory (contains settings.cto)
SAMPLE_DIR=$(find data/ -name "settings.cto" -maxdepth 2 2>/dev/null | head -1 | xargs -I{} dirname {})
if [ -z "$SAMPLE_DIR" ]; then
    echo "No sample directory found in data/ — skipping inference."
else
    echo "Using sample dir: $SAMPLE_DIR"
    mkdir -p outputs/dl_reconstruction
    python -u scripts/pure_dl/03_inference.py \
        --model-path outputs/pure_dl_training_centered/best_model_centered.pt \
        --sample-dir "$SAMPLE_DIR" \
        --output-path outputs/dl_reconstruction/dl_volume.tif \
        --batch-size 8
    echo "Reconstruction saved to outputs/dl_reconstruction/dl_volume.tif"
fi

echo "================================================================"
echo "Pipeline completed successfully!"
echo "================================================================"
