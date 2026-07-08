from __future__ import annotations

from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
EXPORTS_DIR = REPO_ROOT / "exports"
DOCS_DIR = REPO_ROOT / "docs"
REFERENCES_DIR = REPO_ROOT / "references"

SAMPLE_DIRS = {}
if DATA_DIR.exists():
    for d in DATA_DIR.iterdir():
        if d.is_dir() and (d / "settings.cto").exists():
            SAMPLE_DIRS[d.name] = d

SAMPLE_DIR = list(SAMPLE_DIRS.values())[0] if SAMPLE_DIRS else None

def set_sample(sample_name: str) -> None:
    global SAMPLE_DIR
    if sample_name not in SAMPLE_DIRS:
        raise ValueError(f"Sample {sample_name} not found.")
    SAMPLE_DIR = SAMPLE_DIRS[sample_name]


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate
