from __future__ import annotations

from pathlib import Path

import astra
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from .geometry import geometry_for_projection_count, parse_geometry
from .paths import OUTPUTS_DIR, SAMPLE_DIR, resolve_repo_path
from .simulate_degradation import DegradedProjectionData, make_full_projection_dataset


def block_average_2d(image: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return image
    height, width = image.shape
    cropped_h = (height // factor) * factor
    cropped_w = (width // factor) * factor
    image = image[:cropped_h, :cropped_w]
    return image.reshape(cropped_h // factor, factor, cropped_w // factor, factor).mean(axis=(1, 3))


def downsample_projection_stack(projections: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return projections.astype(np.float32, copy=False)
    reduced = [block_average_2d(proj, factor) for proj in projections]
    return np.stack(reduced, axis=0).astype(np.float32, copy=False)


def convert_to_attenuation(projections: np.ndarray, scales_path: Path | None = None, air_percentile: float = 99.9) -> np.ndarray:
    projections = projections.astype(np.float32, copy=False)
    
    # If sino_scales.npy is provided, these are pre-calculated attenuation values normalized to 16-bit
    if scales_path is not None and scales_path.exists():
        scales = np.load(scales_path)
        # scales has shape (num_projections,)
        # projections has shape (num_projections, rows, cols)
        scales = scales.reshape(-1, 1, 1)
        return (projections / 65535.0) * scales
        
    # Otherwise, these are raw transmission intensities
    air_level = np.percentile(projections, air_percentile, axis=(1, 2), keepdims=True).astype(np.float32)
    air_level = np.maximum(air_level, 1.0)
    normalized = np.clip(projections / air_level, 1e-6, 1.0)
    attenuation = -np.log(normalized)
    return attenuation.astype(np.float32, copy=False)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    min_value = float(image.min())
    max_value = float(image.max())
    if np.isclose(max_value, min_value):
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_value) / (max_value - min_value)


def build_cone_geometry(
    geometry,
    angles_rad: np.ndarray,
    detector_rows: int,
    detector_cols: int,
    detector_pixel_mm: float,
):
    proj_geom = astra.create_proj_geom(
        "cone",
        detector_pixel_mm,
        detector_pixel_mm,
        detector_rows,
        detector_cols,
        angles_rad.astype(np.float32, copy=False),
        geometry.source_to_object_mm,
        geometry.source_to_detector_mm - geometry.source_to_object_mm,
    )

    # 1. Calculate horizontal shift robustly using pure physical units (mm)
    original_center_x_mm = ((geometry.detector_cols - 1) / 2.0) * geometry.detector_pixel_size_mm
    cor_x_mm = geometry.center_of_rotation_px * geometry.detector_pixel_size_mm
    shift_x_mm = cor_x_mm - original_center_x_mm
    
    # Convert back to downsampled ASTRA pixels
    shift_x_px = shift_x_mm / detector_pixel_mm

    # 2. Protect against missing/zero vertical center which causes the "squashing"
    if not hasattr(geometry, 'vertical_center_px') or geometry.vertical_center_px == 0:
        shift_y_px = 0.0
    else:
        original_center_y_mm = ((geometry.detector_rows - 1) / 2.0) * geometry.detector_pixel_size_mm
        vertical_center_y_mm = geometry.vertical_center_px * geometry.detector_pixel_size_mm
        shift_y_mm = vertical_center_y_mm - original_center_y_mm
        shift_y_px = shift_y_mm / detector_pixel_mm

    proj_geom = astra.geom_postalignment(proj_geom, (shift_x_px, shift_y_px))
    return proj_geom


def build_volume_geometry(detector_rows: int, detector_cols: int, voxel_size_mm: float):
    return astra.create_vol_geom(
        detector_cols,
        detector_cols,
        detector_rows,
        -detector_cols * voxel_size_mm / 2.0,
        detector_cols * voxel_size_mm / 2.0,
        -detector_cols * voxel_size_mm / 2.0,
        detector_cols * voxel_size_mm / 2.0,
        -detector_rows * voxel_size_mm / 2.0,
        detector_rows * voxel_size_mm / 2.0,
    )


def crop_valid_z(volume: np.ndarray, zmin: int, zmax: int, downsample_factor: int) -> tuple[np.ndarray, tuple[int, int]]:
    zmin_ds = max(0, int(np.floor(zmin / downsample_factor)))
    zmax_ds = min(volume.shape[0] - 1, int(np.ceil(zmax / downsample_factor)))
    cropped = volume[zmin_ds : zmax_ds + 1]
    return cropped, (zmin_ds, zmax_ds)


def save_preview(volume: np.ndarray, voxel_size_mm: float, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    z_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    x_mid = volume.shape[2] // 2

    axial = normalize_image(volume[z_mid])
    coronal = normalize_image(volume[:, y_mid, :])
    sagittal = normalize_image(volume[:, :, x_mid])

    x_extent = volume.shape[2] * voxel_size_mm / 2.0
    y_extent = volume.shape[1] * voxel_size_mm / 2.0
    z_extent = volume.shape[0] * voxel_size_mm / 2.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ensure it looks like this:
    axes[0].imshow(axial, cmap="gray", extent=[-x_extent, x_extent, y_extent, -y_extent], aspect="equal")
    axes[0].set_title(f"Axial z={z_mid}")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    
    axes[1].imshow(coronal, cmap="gray", extent=[-x_extent, x_extent, z_extent, -z_extent], aspect="equal")
    axes[1].set_title(f"Coronal y={y_mid}")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("z (mm)")

    axes[2].imshow(sagittal, cmap="gray", extent=[-y_extent, y_extent, z_extent, -z_extent], aspect="equal")
    axes[2].set_title(f"Sagittal x={x_mid}")
    axes[2].set_xlabel("y (mm)")
    axes[2].set_ylabel("z (mm)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def reconstruct_volume_from_projection_dataset(
    dataset: DegradedProjectionData,
    sample_dir: str | Path = SAMPLE_DIR,
    downsample_factor: int = 2,
) -> tuple[np.ndarray, dict[str, float | int | tuple[int, ...]]]:
    sample_dir = resolve_repo_path(sample_dir)
    geometry = parse_geometry(sample_dir / "settings.cto")
    geometry = geometry_for_projection_count(geometry, dataset.projections.shape[0])

    projections_ds = downsample_projection_stack(dataset.projections, downsample_factor)
    scales_path = sample_dir / "sino_scales.npy"
    attenuation = convert_to_attenuation(projections_ds, scales_path=scales_path)

    detector_rows = attenuation.shape[1]
    detector_cols = attenuation.shape[2]

    # --- THE BULLETPROOF SCALING FIX ---
    # 1. Calculate the true physical width of the original detector from the CTO metadata
    original_physical_width_mm = geometry.detector_cols * geometry.detector_pixel_size_mm
    
    # 2. Force the new pixel size to maintain that exact physical width across the new column count
    detector_pixel_mm = original_physical_width_mm / detector_cols
    # -----------------------------------

    voxel_size_mm = detector_pixel_mm * (geometry.source_to_object_mm / geometry.source_to_detector_mm)

    proj_geom = build_cone_geometry(
        geometry,
        dataset.angles_rad,
        detector_rows,
        detector_cols,
        detector_pixel_mm,
    )
    vol_geom = build_volume_geometry(detector_rows, detector_cols, voxel_size_mm)

    projection_data = np.transpose(attenuation, (1, 0, 2)).astype(np.float32, copy=False)

    proj_id = astra.data3d.create("-sino", proj_geom, projection_data)
    vol_id = astra.data3d.create("-vol", vol_geom)
    alg_id = None

    try:
        cfg = astra.astra_dict("FDK_CUDA")
        cfg["ProjectionDataId"] = proj_id
        cfg["ReconstructionDataId"] = vol_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        reconstructed = astra.data3d.get(vol_id).astype(np.float32, copy=False)
    finally:
        if alg_id is not None:
            astra.algorithm.delete(alg_id)
        astra.data3d.delete(vol_id)
        astra.data3d.delete(proj_id)

    # Calculate effective downsample ratio for the Z-crop
    total_ratio = geometry.detector_cols / detector_cols
    cropped, (zmin_ds, zmax_ds) = crop_valid_z(reconstructed, geometry.zmin, geometry.zmax, total_ratio)
    
    info: dict[str, float | int | tuple[int, ...]] = {
        "input_projections": int(dataset.projections.shape[0]),
        "projection_shape_downsampled": tuple(int(v) for v in attenuation.shape),
        "voxel_size_mm": float(voxel_size_mm),
        "cropped_z_range_downsampled": (int(zmin_ds), int(zmax_ds)),
        "output_volume_shape": tuple(int(v) for v in cropped.shape),
    }
    return cropped, info


def save_reconstruction_outputs(
    volume: np.ndarray,
    info: dict[str, float | int | tuple[int, ...]],
    output_dir: str | Path,
    prefix: str,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    volume_path = output_dir / f"{prefix}_volume.tif"
    preview_path = output_dir / f"{prefix}_preview.png"
    meta_path = output_dir / f"{prefix}_info.txt"

    tifffile.imwrite(volume_path, volume.astype(np.float32))
    voxel_size_mm = info["voxel_size_mm"]
    if not isinstance(voxel_size_mm, (int, float)):
        raise TypeError("info['voxel_size_mm'] must be a numeric value")
    save_preview(volume, voxel_size_mm=float(voxel_size_mm), output_path=preview_path)
    meta_path.write_text(
        "\n".join(f"{key}={value}" for key, value in info.items()),
        encoding="utf-8",
    )

    return {
        "volume_path": volume_path,
        "preview_path": preview_path,
        "meta_path": meta_path,
    }


def run_fdk_reconstruction(
    sample_dir: str | Path = SAMPLE_DIR,
    downsample_factor: int = 2,
    output_dir: str | Path = OUTPUTS_DIR / "fdk_astra",
) -> dict[str, Path]:
    dataset = make_full_projection_dataset(sample_dir)
    volume, info = reconstruct_volume_from_projection_dataset(
        dataset=dataset,
        sample_dir=sample_dir,
        downsample_factor=downsample_factor,
    )
    return save_reconstruction_outputs(volume, info, output_dir=resolve_repo_path(output_dir), prefix="fdk")


if __name__ == "__main__":
    outputs = run_fdk_reconstruction()
    for key, value in outputs.items():
        print(f"{key}: {value}")


##- this uses all projections, but spatially downsamples detector rows/cols by 2 to fit GPU memory
##  - if you want, next I can try a chunked higher-resolution version to push beyond 500 x 500 x 441 without losing all projections
