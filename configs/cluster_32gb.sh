#!/bin/bash
# ============================================================
#  CONFIG: HPC Cluster — 32GB VRAM (A100 / V100 / RTX 6000)
#
#  This is the full-quality configuration for publishable results.
#  Run ALL 5 STL objects x ALL 4 material/noise variations.
#  Recommended scheduler: SLURM (see cluster_32gb.slurm)
# ============================================================

# --- STEP 2: Dataset Building (01_build_dataset.py) ---
export DOWNSAMPLE_FACTOR=1        # NO downsampling — use full 1800x1800 detector
export SPARSE_STEP=4              # Keep every 4th projection (90 views from 360)
export DETECTOR_COUNT=512         # Full-resolution sinogram columns
export IMAGE_SIZE=512             # High-resolution 512x512 CT slices
export SLICE_STRIDE=1             # Use EVERY Z-slice (maximum training data)

# --- STEP 3: Training (02_train_pure_dl.py) ---
export BATCH_SIZE=8               # 32GB VRAM supports batch=8 at 512x512
export EPOCHS=100                 # Full training run
export LEARNING_RATE=5e-4         # Slightly lower LR for larger batches
export VAL_FRACTION=0.15

echo "Cluster 32GB config loaded."
echo "Expected training time: ~6-12 hours for 100 epochs on 512x512"
echo "Expected VRAM usage: ~26GB peak at batch=8"
