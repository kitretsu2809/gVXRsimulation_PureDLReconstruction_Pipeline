#!/bin/bash
set -e

echo "================================================================"
echo "Step 1: Generating Projections using gVXR"
echo "================================================================"
conda run -n ct_pipeline python DATACREATION/generate_datasets.py

echo "================================================================"
echo "Step 2: Classical Reconstruction & Dataset Generation"
echo "================================================================"
conda run -n ct_pipeline python scripts/run_batch_pipeline.py

echo "================================================================"
echo "Step 3: Training Pure DL Model on Generated Dataset"
echo "================================================================"
conda run -n ct_pipeline python scripts/pure_dl/02_train_pure_dl.py --dataset-path outputs/batch_datasets

echo "================================================================"
echo "Pipeline completed successfully!"
echo "================================================================"
