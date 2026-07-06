from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ct_recon.reconstruct_fdk_astra import run_fdk_reconstruction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FDK reconstruction.")
    parser.add_argument("--sample-dir", required=True, help="Path to the sample directory (e.g. data/FINAL30_standard_auto)")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--output-dir", required=True, help="Path to the output directory")
    args = parser.parse_args()

    print(f"Starting classical FDK reconstruction for folder: {args.sample_dir}")
    print(f"Output will be saved to: {args.output_dir}")
    
    outputs = run_fdk_reconstruction(
        sample_dir=args.sample_dir,
        downsample_factor=args.downsample,
        output_dir=args.output_dir,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")
