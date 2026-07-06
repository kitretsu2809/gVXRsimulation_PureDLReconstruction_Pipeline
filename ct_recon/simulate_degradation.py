from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .data_loader import CTScanData, load_sample
from .geometry import geometry_for_projection_count, parse_geometry
from .paths import OUTPUTS_DIR, SAMPLE_DIR, resolve_repo_path


@dataclass
class DegradedProjectionData:
    name: str
    projections: np.ndarray
    angles_rad: np.ndarray
    kept_projection_indices: np.ndarray
    metadata: dict[str, Any]


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    min_value = float(image.min())
    max_value = float(image.max())
    if np.isclose(min_value, max_value):
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_value) / (max_value - min_value)


def make_full_projection_dataset(sample_dir: str | Path = SAMPLE_DIR) -> DegradedProjectionData:
    sample_dir = resolve_repo_path(sample_dir)
    data = load_sample(sample_dir)
    geometry = parse_geometry(sample_dir / "settings.cto")
    geometry = geometry_for_projection_count(geometry, data.projections.shape[0])

    return DegradedProjectionData(
        name="full",
        projections=data.projections.astype(np.float32, copy=False),
        angles_rad=geometry.angles_rad.astype(np.float32, copy=False),
        kept_projection_indices=np.arange(data.projections.shape[0], dtype=np.int32),
        metadata={
            "sample_dir": str(sample_dir),
            "input_projection_count": int(data.projections.shape[0]),
            "degradation_type": "full",
        },
    )


def sparse_view_subset(
    projections: np.ndarray,
    angles_rad: np.ndarray,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if step <= 0:
        raise ValueError("step must be positive")
    indices = np.arange(0, projections.shape[0], step, dtype=np.int32)
    return projections[indices], angles_rad[indices], indices


def limited_angle_subset(
    projections: np.ndarray,
    angles_rad: np.ndarray,
    start_deg: float,
    stop_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_rad = np.deg2rad(start_deg)
    stop_rad = np.deg2rad(stop_deg)
    wrapped = np.mod(angles_rad, 2.0 * np.pi)
    start_rad = float(np.mod(start_rad, 2.0 * np.pi))
    stop_rad = float(np.mod(stop_rad, 2.0 * np.pi))

    if start_rad <= stop_rad:
        mask = (wrapped >= start_rad) & (wrapped <= stop_rad)
    else:
        mask = (wrapped >= start_rad) | (wrapped <= stop_rad)

    indices = np.flatnonzero(mask).astype(np.int32)
    if len(indices) == 0:
        raise ValueError("limited-angle selection kept zero projections")
    return projections[indices], angles_rad[indices], indices


def add_gaussian_noise(
    projections: np.ndarray,
    sigma_fraction: float,
    seed: int = 0,
) -> np.ndarray:
    if sigma_fraction < 0:
        raise ValueError("sigma_fraction must be non-negative")
    rng = np.random.default_rng(seed)
    sigma = float(np.mean(projections) * sigma_fraction)
    noisy = projections.astype(np.float32, copy=False) + rng.normal(0.0, sigma, size=projections.shape).astype(np.float32)
    return np.clip(noisy, 1.0, None)


def add_poisson_noise(
    projections: np.ndarray,
    photon_fraction: float,
    seed: int = 0,
) -> np.ndarray:
    if photon_fraction <= 0:
        raise ValueError("photon_fraction must be positive")
    rng = np.random.default_rng(seed)
    peak = float(np.max(projections))
    scaled = np.clip(projections / peak, 1e-6, 1.0)
    photon_budget = max(1.0, 65535.0 * photon_fraction)
    counts = rng.poisson(scaled * photon_budget).astype(np.float32)
    restored = np.clip(counts / photon_budget, 1e-6, None) * peak
    return restored.astype(np.float32, copy=False)


def create_sparse_view_dataset(sample_dir: str | Path = SAMPLE_DIR, step: int = 4) -> DegradedProjectionData:
    base = make_full_projection_dataset(sample_dir)
    projections, angles_rad, indices = sparse_view_subset(base.projections, base.angles_rad, step=step)
    return DegradedProjectionData(
        name=f"sparse_view_step_{step}",
        projections=projections,
        angles_rad=angles_rad,
        kept_projection_indices=indices,
        metadata={
            **base.metadata,
            "degradation_type": "sparse_view",
            "step": int(step),
            "output_projection_count": int(len(indices)),
        },
    )


def create_limited_angle_dataset(
    sample_dir: str | Path = SAMPLE_DIR,
    start_deg: float = 0.0,
    stop_deg: float = 180.0,
) -> DegradedProjectionData:
    base = make_full_projection_dataset(sample_dir)
    projections, angles_rad, indices = limited_angle_subset(
        base.projections,
        base.angles_rad,
        start_deg=start_deg,
        stop_deg=stop_deg,
    )
    return DegradedProjectionData(
        name=f"limited_angle_{int(start_deg)}_{int(stop_deg)}",
        projections=projections,
        angles_rad=angles_rad,
        kept_projection_indices=indices,
        metadata={
            **base.metadata,
            "degradation_type": "limited_angle",
            "start_deg": float(start_deg),
            "stop_deg": float(stop_deg),
            "output_projection_count": int(len(indices)),
        },
    )


def create_noisy_dataset(
    sample_dir: str | Path = SAMPLE_DIR,
    mode: str = "poisson",
    level: float = 0.25,
    seed: int = 0,
) -> DegradedProjectionData:
    base = make_full_projection_dataset(sample_dir)

    if mode == "poisson":
        noisy = add_poisson_noise(base.projections, photon_fraction=level, seed=seed)
    elif mode == "gaussian":
        noisy = add_gaussian_noise(base.projections, sigma_fraction=level, seed=seed)
    else:
        raise ValueError("mode must be 'poisson' or 'gaussian'")

    return DegradedProjectionData(
        name=f"{mode}_noise_{level:g}",
        projections=noisy,
        angles_rad=base.angles_rad,
        kept_projection_indices=base.kept_projection_indices,
        metadata={
            **base.metadata,
            "degradation_type": "noise",
            "noise_mode": mode,
            "noise_level": float(level),
            "seed": int(seed),
            "output_projection_count": int(noisy.shape[0]),
        },
    )


def save_projection_dataset(dataset: DegradedProjectionData, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir) / dataset.name
    output_dir.mkdir(parents=True, exist_ok=True)

    projections_path = output_dir / "projections.npz"
    metadata_path = output_dir / "metadata.json"
    preview_path = output_dir / "preview.png"

    np.savez_compressed(
        projections_path,
        projections=dataset.projections.astype(np.float32),
        angles_rad=dataset.angles_rad.astype(np.float32),
        kept_projection_indices=dataset.kept_projection_indices.astype(np.int32),
    )

    metadata_path.write_text(json.dumps(dataset.metadata, indent=2), encoding="utf-8")
    save_dataset_preview(dataset, preview_path)

    return {
        "dataset_dir": output_dir,
        "projections_path": projections_path,
        "metadata_path": metadata_path,
        "preview_path": preview_path,
    }


def save_dataset_preview(dataset: DegradedProjectionData, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mid_projection = dataset.projections.shape[0] // 2
    mid_row = dataset.projections.shape[1] // 2

    projection_image = normalize_image(dataset.projections[mid_projection])
    sinogram = normalize_image(dataset.projections[:, mid_row, :].T)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(projection_image, cmap="gray")
    axes[0].set_title(f"{dataset.name}: projection {mid_projection}")
    axes[0].axis("off")

    axes[1].imshow(sinogram, cmap="gray", aspect="auto")
    axes[1].set_title(f"{dataset.name}: sinogram row {mid_row}")
    axes[1].set_xlabel("projection index")
    axes[1].set_ylabel("detector column")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_default_degradation_sets(sample_dir: str | Path = SAMPLE_DIR) -> list[DegradedProjectionData]:
    return [
        create_sparse_view_dataset(sample_dir=sample_dir, step=2),
        create_sparse_view_dataset(sample_dir=sample_dir, step=4),
        create_limited_angle_dataset(sample_dir=sample_dir, start_deg=0.0, stop_deg=180.0),
        create_limited_angle_dataset(sample_dir=sample_dir, start_deg=30.0, stop_deg=210.0),
        create_noisy_dataset(sample_dir=sample_dir, mode="poisson", level=0.5, seed=0),
        create_noisy_dataset(sample_dir=sample_dir, mode="poisson", level=0.25, seed=0),
    ]


def build_and_save_default_degradation_sets(
    sample_dir: str | Path = SAMPLE_DIR,
    output_dir: str | Path = OUTPUTS_DIR / "degradations",
) -> list[dict[str, Path]]:
    datasets = build_default_degradation_sets(sample_dir=sample_dir)
    output_dir = resolve_repo_path(output_dir)
    return [save_projection_dataset(dataset, output_dir) for dataset in datasets]


if __name__ == "__main__":
    outputs = build_and_save_default_degradation_sets()
    for item in outputs:
        print(item["dataset_dir"])
