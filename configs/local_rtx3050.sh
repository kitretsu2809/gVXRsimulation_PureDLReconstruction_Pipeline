#!/bin/bash
# ============================================================
#  CONFIG: Local machine — RTX 3050 4GB VRAM (your laptop)
#  AMD Ryzen 9 6900HS | 30GB RAM | 4GB VRAM
#
#  WARNING: gVXR projection generation will work fine (CPU+GPU).
#  Training will work but SLOWLY — small image/detector size,
#  batch_size=1 to avoid OOM. Treat this as a smoke-test only.
#  Do NOT expect publishable results from this config.
# ============================================================

# --- STEP 1: gVXR Projection Generation ---
# These are fixed in gvxr_projection_script.py (1800x1800 detector).
# The raw TIFFs will be ~2.5GB per object per variation.
# Reduce STL count if disk space is tight (currently 91GB free).
# Recommend running only 1 object + 1 variation for a local smoke test:
export GVXR_STL="DATACREATION/STL/super-mini-whistle-by-prntmkr.stl"
export GVXR_MATERIAL="Ti"
export GVXR_I0=50000
export GVXR_GAUSSIAN_STD=10

# --- STEP 2: Dataset Building (01_build_dataset.py) ---
export DOWNSAMPLE_FACTOR=4        # 4x downsample: 1800 -> 450 detector cols (fits 4GB VRAM)
export SPARSE_STEP=4              # Use every 4th projection (90 views from 360)
export DETECTOR_COUNT=128         # Resize sinogram to 128 detector columns
export IMAGE_SIZE=128             # Reconstruct 128x128 slices (tiny but trainable)
export SLICE_STRIDE=4             # Use every 4th Z-slice (reduces dataset size ~4x)

# --- STEP 3: Training (02_train_pure_dl.py) ---
export BATCH_SIZE=1               # CRITICAL: 4GB VRAM requires batch=1
export EPOCHS=5                   # Just enough to verify training works
export LEARNING_RATE=1e-3
export VAL_FRACTION=0.2

echo "Local RTX 3050 config loaded."
echo "Expected training time: ~2-4 hours for 5 epochs on 128x128"
echo "Expected VRAM usage: ~3.2GB peak"
