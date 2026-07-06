from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ct_recon.data_loader import load_sample
from ct_recon.geometry import parse_geometry
from ct_recon.paths import OUTPUTS_DIR, SAMPLE_DIR, resolve_repo_path
from ct_recon.reconstruct_fdk_astra import convert_to_attenuation, downsample_projection_stack
from ct_recon.sparse_ct_reconstruction import SparseSinogramDatasetMetadata, resize_2d_array


def resolve_target_volume_path(path_hint: str | None) -> Path:
    candidates = []
    if path_hint:
        candidates.append(resolve_repo_path(path_hint))
    candidates.extend(
        [
            OUTPUTS_DIR / "fdk_full_ds2" / "fdk_volume.tif",
            OUTPUTS_DIR / "fdk_astra" / "fdk_volume.tif",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No target volume found. Generate an FDK reference first or pass --target-volume.")


def compute_row_range(zmin: int, zmax: int, downsample_factor: int) -> tuple[int, int]:
    row_start = max(0, int(np.floor(zmin / downsample_factor)))
    row_stop = int(np.ceil(zmax / downsample_factor))
    return row_start, row_stop


def main():
    parser = argparse.ArgumentParser(description="Build sparse-sinogram to target-slice training data.")
    parser.add_argument("--sample-dir", default=str(SAMPLE_DIR))
    parser.add_argument("--target-volume", default=None, help="Optional path to a high-quality target volume.")
    parser.add_argument("--output-path", default=str(OUTPUTS_DIR / "sparse_sinogram_dataset.npz"))
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--sparse-step", type=int, default=4, help="Keep every Nth projection for the sparse input.")
    parser.add_argument("--detector-count", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--slice-stride", type=int, default=1, help="Keep every Nth slice from the target volume.")
    args = parser.parse_args()

    if args.sparse_step <= 0:
        raise ValueError("--sparse-step must be positive")
    if args.slice_stride <= 0:
        raise ValueError("--slice-stride must be positive")

    sample_dir = resolve_repo_path(args.sample_dir)
    sample = load_sample(sample_dir)
    geometry = parse_geometry(sample_dir / "settings.cto")
    target_volume_path = resolve_target_volume_path(args.target_volume)
    target_volume = tifffile.imread(target_volume_path).astype(np.float32)

    projections_ds = downsample_projection_stack(sample.projections, args.downsample_factor)
    attenuation = convert_to_attenuation(projections_ds)

    dense_angle_count = int(attenuation.shape[0])
    sparse_indices = np.arange(0, dense_angle_count, args.sparse_step, dtype=np.int32)
    row_start, row_stop = compute_row_range(geometry.zmin, geometry.zmax, args.downsample_factor)

    # Adjust row range to match actual target volume dimensions
    # Different samples have different z-ranges, so we use target volume as reference
    actual_slices = target_volume.shape[0]
    expected_slices = row_stop - row_start + 1
    
    if actual_slices != expected_slices:
        print(f"Note: Target volume has {actual_slices} slices, expected {expected_slices}")
        print(f"Adjusting row range from [{row_start}, {row_stop}] to use available data...")
        # Use whichever is smaller to avoid index errors
        usable_slices = min(actual_slices, expected_slices)
        row_stop = row_start + usable_slices - 1
        print(f"Using rows {row_start} to {row_stop} ({usable_slices} slices)")

    image_min = float(target_volume.min())
    image_max = float(target_volume.max())
    image_scale = max(image_max - image_min, 1e-6)
    sinogram_scale = float(np.percentile(attenuation, 99.5))
    sinogram_scale = max(sinogram_scale, 1e-6)

    selected_slice_indices = np.arange(0, target_volume.shape[0], args.slice_stride, dtype=np.int32)
    input_sinograms = []
    target_sinograms = []
    target_images = []

    for slice_idx in selected_slice_indices:
        detector_row = row_start + int(slice_idx)
        dense_sinogram = attenuation[:, detector_row, :]
        sparse_sinogram = dense_sinogram[sparse_indices]

        dense_sinogram_resized = resize_2d_array(dense_sinogram, (dense_angle_count, args.detector_count))
        sparse_sinogram_resized = resize_2d_array(sparse_sinogram, (len(sparse_indices), args.detector_count))
        target_image = resize_2d_array(target_volume[slice_idx], (args.image_size, args.image_size))

        input_sinograms.append(np.clip(sparse_sinogram_resized / sinogram_scale, 0.0, None).astype(np.float32))
        target_sinograms.append(np.clip(dense_sinogram_resized / sinogram_scale, 0.0, None).astype(np.float32))
        target_images.append(np.clip((target_image - image_min) / image_scale, 0.0, 1.0).astype(np.float32))

    output_path = resolve_repo_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = SparseSinogramDatasetMetadata(
        sparse_step=int(args.sparse_step),
        dense_angle_count=int(dense_angle_count),
        sparse_angle_count=int(len(sparse_indices)),
        detector_count=int(args.detector_count),
        image_size=int(args.image_size),
        downsample_factor=int(args.downsample_factor),
        row_start=int(row_start),
        row_stop=int(row_stop),
        slice_count=int(len(selected_slice_indices)),
        sinogram_scale=float(sinogram_scale),
        image_min=float(image_min),
        image_max=float(image_max),
        target_volume_path=str(target_volume_path),
        settings_path=str(sample_dir / "settings.cto"),
    )

    np.savez_compressed(
        output_path,
        input_sinograms=np.stack(input_sinograms, axis=0).astype(np.float32),
        target_sinograms=np.stack(target_sinograms, axis=0).astype(np.float32),
        target_images=np.stack(target_images, axis=0).astype(np.float32),
        selected_slice_indices=selected_slice_indices.astype(np.int32),
        sparse_projection_indices=sparse_indices.astype(np.int32),
        metadata_json=json.dumps(metadata.__dict__),
    )
    print(output_path)


if __name__ == "__main__":
    main()
