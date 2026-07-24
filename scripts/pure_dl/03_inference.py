#!/usr/bin/env python3
"""
Pure DL Inference Script.
Loads a trained PureDLPipeline checkpoint and reconstructs a 3D .tif volume 
from raw projection data slice-by-slice.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ct_recon.data_loader import load_sample
from ct_recon.geometry import parse_geometry
from ct_recon.paths import OUTPUTS_DIR, resolve_repo_path
from ct_recon.reconstruct_fdk_astra import convert_to_attenuation, downsample_projection_stack
from ct_recon.sparse_ct_reconstruction import resize_2d_array
from ct_recon.pure_dl_net import PureDLPipeline

def compute_row_range(zmin: int, zmax: int, downsample_factor: int) -> tuple[int, int]:
    row_start = max(0, int(np.floor(zmin / downsample_factor)))
    row_stop = int(np.ceil(zmax / downsample_factor))
    return row_start, row_stop

def main():
    parser = argparse.ArgumentParser(description="Run Pure DL Inference to generate 3D volume.")
    parser.add_argument("--model-path", required=True, help="Path to the trained .pt model checkpoint.")
    parser.add_argument("--sample-dir", required=True, help="Path to the raw dataset folder (containing .tiff and settings.cto).")
    parser.add_argument("--output-path", default=str(OUTPUTS_DIR / "pure_dl_reconstruction.tif"), help="Path to save the output .tif volume.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of slices to process at once.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model Checkpoint and Metadata
    model_path = resolve_repo_path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    metadata = checkpoint["metadata"]
    
    # 2. Instantiate Network
    target_image_size = metadata["image_size"]
    print(f"Initializing PureDLPipeline for {target_image_size}x{target_image_size} inference...")
    model = PureDLPipeline(target_image_size=target_image_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 3. Load Raw Data
    sample_dir = resolve_repo_path(args.sample_dir)
    print(f"Loading raw projections from: {sample_dir}")
    sample = load_sample(sample_dir)
    geometry = parse_geometry(sample_dir / "settings.cto")

    print(f"Downsampling and converting to attenuation...")
    projections_ds = downsample_projection_stack(sample.projections, metadata["downsample_factor"])
    
    # CRITICAL MATH FIX: We must load the pre-calculated scales to undo the 16-bit TIFF normalization.
    # Otherwise, the network receives botched exposure values during inference!
    scales_path = sample_dir / "sino_scales.npy"
    if not scales_path.exists():
        print(f"WARNING: {scales_path} not found! Exposure may be mathematically incorrect.")
    attenuation = convert_to_attenuation(projections_ds, scales_path=scales_path)

    dense_angle_count = int(attenuation.shape[0])
    sparse_indices = np.arange(0, dense_angle_count, metadata["sparse_step"], dtype=np.int32)
    
    row_start, row_stop = compute_row_range(geometry.zmin, geometry.zmax, metadata["downsample_factor"])
    
    # Check boundaries
    max_rows = attenuation.shape[1]
    row_start = max(0, row_start)
    row_stop = min(max_rows - 1, row_stop)
    num_slices = row_stop - row_start + 1
    
    print(f"Geometry dictates {num_slices} slices (rows {row_start} to {row_stop}).")

    # 4. Prepare Output Volume Array
    output_volume = np.zeros((num_slices, target_image_size, target_image_size), dtype=np.float32)
    
    image_scale = max(metadata["image_max"] - metadata["image_min"], 1e-6)
    
    # 5. Run Inference Slice-by-Slice in Batches
    batch_sinograms = []
    batch_indices = []
    
    def process_batch(sinos, idxs):
        if not sinos: return
        
        # Stack to batch tensor [B, 1, Angles, Detectors]
        batch_tensor = torch.tensor(np.stack(sinos, axis=0)).unsqueeze(1).to(device)
        
        # The network might expect the interpolated dense shape?
        # Let's check 02_train_pure_dl.py behavior. Wait, PureDLPipeline consumes whatever shape is passed, 
        # but in training we interpolated the sparse sinogram to match the dense sinogram target size!
        dense_shape = (dense_angle_count, metadata["detector_count"])
        if batch_tensor.shape[2:] != dense_shape:
            batch_tensor = F.interpolate(
                batch_tensor,
                size=dense_shape,
                mode='bilinear',
                align_corners=False
            )
            
        with torch.no_grad():
            final_image, _, _ = model(batch_tensor)
            
        # Denormalize output
        predictions = final_image.squeeze(1).cpu().numpy()
        predictions = (predictions * image_scale) + metadata["image_min"]
        
        # Save to volume
        for b_idx, vol_idx in enumerate(idxs):
            output_volume[vol_idx, :, :] = predictions[b_idx]
            
    print("Beginning Deep Learning Reconstruction...")
    
    for vol_idx, detector_row in enumerate(range(row_start, row_stop + 1)):
        # Extract dense slice, then make it sparse
        dense_sinogram = attenuation[:, detector_row, :]
        sparse_sinogram = dense_sinogram[sparse_indices]
        
        # Resize it exactly as training did
        sparse_sinogram_resized = resize_2d_array(sparse_sinogram, (len(sparse_indices), metadata["detector_count"]))
        
        # Normalize
        normalized_sino = np.clip(sparse_sinogram_resized / metadata["sinogram_scale"], 0.0, None).astype(np.float32)
        
        batch_sinograms.append(normalized_sino)
        batch_indices.append(vol_idx)
        
        if len(batch_sinograms) >= args.batch_size:
            process_batch(batch_sinograms, batch_indices)
            print(f"Processed slices up to {vol_idx+1}/{num_slices}")
            batch_sinograms = []
            batch_indices = []
            
    # Process remaining
    if batch_sinograms:
        process_batch(batch_sinograms, batch_indices)
        print(f"Processed slices up to {num_slices}/{num_slices}")

    # 6. Save Volume
    output_path = resolve_repo_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving fully reconstructed 3D volume to: {output_path}")
    tifffile.imwrite(output_path, output_volume)
    
    # Save contrast-normalized 3-axis preview PNG automatically
    try:
        import matplotlib.pyplot as plt
        preview_path = output_path.parent / f"{output_path.stem}_preview.png"
        z_mid = output_volume.shape[0] // 2
        y_mid = output_volume.shape[1] // 2
        x_mid = output_volume.shape[2] // 2
        
        v_min, v_max = np.percentile(output_volume, 1), np.percentile(output_volume, 99)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(output_volume[z_mid], cmap='gray', vmin=v_min, vmax=v_max)
        axes[0].set_title(f"Axial Slice (Z={z_mid})")
        
        axes[1].imshow(output_volume[:, y_mid, :], cmap='gray', aspect='auto', vmin=v_min, vmax=v_max)
        axes[1].set_title(f"Coronal Slice (Y={y_mid})")
        
        axes[2].imshow(output_volume[:, :, x_mid], cmap='gray', aspect='auto', vmin=v_min, vmax=v_max)
        axes[2].set_title(f"Sagittal Slice (X={x_mid})")
        
        plt.tight_layout()
        fig.savefig(preview_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved 3-axis contrast preview to: {preview_path}")
    except Exception as e:
        print(f"Note: Could not save PNG preview: {e}")

    print("Inference Complete!")

if __name__ == "__main__":
    main()
