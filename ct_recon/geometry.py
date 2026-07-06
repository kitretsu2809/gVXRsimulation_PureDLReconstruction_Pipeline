from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .data_loader import load_cto_settings
from .paths import SAMPLE_DIR


@dataclass
class CTGeometry:
    settings_path: Path
    projections: int
    angle_range_deg: float
    angles_deg: np.ndarray
    angles_rad: np.ndarray
    detector_rows: int
    detector_cols: int
    detector_pixel_size_mm: float
    source_to_object_mm: float
    source_to_detector_mm: float
    center_of_rotation_px: float
    vertical_center_px: float
    recon_rows: int
    recon_cols: int
    zmin: int
    zmax: int
    direction: int


def _require(section: dict[str, Any], key: str) -> Any:
    if key not in section:
        raise KeyError(f"Missing required key '{key}'")
    return section[key]


def parse_geometry(settings_path: str | Path) -> CTGeometry:
    settings_path = Path(settings_path)
    settings = load_cto_settings(settings_path)

    detector = settings["Detector settings"]
    scan = settings["CT scan settings"]
    recon = settings["CT reconstruction settings"]

    projections = int(_require(scan, "projections"))
    angle_range_deg = float(_require(scan, "angle range"))
    direction = int(_require(recon, "direction"))

    start_angle_deg = 0.0
    stop_angle_deg = start_angle_deg + direction * angle_range_deg
    angles_deg = np.linspace(start_angle_deg, stop_angle_deg, projections, endpoint=False, dtype=np.float32)
    angles_rad = np.deg2rad(angles_deg).astype(np.float32)

    detector_rows = int(_require(recon, "rows"))
    detector_cols = int(_require(recon, "columns"))
    detector_pixel_size_mm = float(_require(detector, "pixel size"))

    return CTGeometry(
        settings_path=settings_path,
        projections=projections,
        angle_range_deg=angle_range_deg,
        angles_deg=angles_deg,
        angles_rad=angles_rad,
        detector_rows=detector_rows,
        detector_cols=detector_cols,
        detector_pixel_size_mm=detector_pixel_size_mm,
        source_to_object_mm=float(scan.get("SOD", recon.get("SOD"))),
        source_to_detector_mm=float(scan.get("SDD", recon.get("SDD"))),
        center_of_rotation_px=float(_require(recon, "COR")),
        vertical_center_px=float(_require(recon, "vertical center")),
        recon_rows=int(_require(recon, "rows")),
        recon_cols=int(_require(recon, "columns")),
        zmin=int(_require(recon, "zmin")),
        zmax=int(_require(recon, "zmax")),
        direction=direction,
    )


def geometry_for_projection_count(geometry: CTGeometry, projection_count: int) -> CTGeometry:
    if projection_count <= 0:
        raise ValueError("projection_count must be positive")

    start_angle_deg = 0.0
    stop_angle_deg = start_angle_deg + geometry.direction * geometry.angle_range_deg
    angles_deg = np.linspace(
        start_angle_deg,
        stop_angle_deg,
        projection_count,
        endpoint=False,
        dtype=np.float32,
    )
    angles_rad = np.deg2rad(angles_deg).astype(np.float32)

    return replace(
        geometry,
        projections=projection_count,
        angles_deg=angles_deg,
        angles_rad=angles_rad,
    )


if __name__ == "__main__":
    geometry = parse_geometry(SAMPLE_DIR / "settings.cto")
    print(geometry)
