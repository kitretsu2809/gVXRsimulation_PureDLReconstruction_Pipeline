#!/usr/bin/env python3
"""
Batch processing orchestrator for Pure DL datasets.
This script scans the data/ directories for simulated datasets (e.g. from gVXR),
runs the classical FDK reconstruction to generate ground-truth targets,
and builds individual .npz training files for each dataset.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Batch processing orchestrator for Pure DL datasets.")
    parser.add_argument("--sparse-step", type=int, default=1, help="Downsample factor for projections. 1 = All data, >1 = Sparse View.")
    parser.add_argument("--downsample", type=int, default=2, help="Spatial downsampling factor for detector rows/cols (default: 2).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    
    if not data_dir.exists():
        print(f"Data directory not found. Looked in {data_dir}. Please ensure you have datasets in the 'data/' folder.")
        return
        
    print(f"Scanning {data_dir} for valid datasets...")
    
    datasets = []
    for d in data_dir.iterdir():
        if d.is_dir() and (d / "settings.cto").exists():
            datasets.append(d)
            
    if not datasets:
        print("No valid datasets found. Make sure folders contain 'settings.cto'.")
        return
        
    print(f"Found {len(datasets)} valid datasets to process.")
    
    batch_datasets_dir = repo_root / "outputs" / "batch_datasets"
    batch_datasets_dir.mkdir(parents=True, exist_ok=True)
    
    fdk_script = repo_root / "scripts" / "classical_reconstruction" / "reconstruct_fdk.py"
    build_script = repo_root / "scripts" / "data_preparation" / "01_build_dataset.py"
    
    for dataset_path in datasets:
        dataset_name = dataset_path.name
        print(f"\n{'='*50}\nProcessing Dataset: {dataset_name}\n{'='*50}")
        
        # 1. Check/Run FDK Baseline
        fdk_out_dir = repo_root / "outputs" / f"fdk_astra_{dataset_name}"
        fdk_vol_path = fdk_out_dir / "fdk_volume.tif"
        
        if not fdk_vol_path.exists():
            print(f"[{dataset_name}] Target volume missing. Generating Classical FDK...")
            cmd = [
                sys.executable, str(fdk_script),
                "--sample-dir", str(dataset_path),
                "--downsample", str(args.downsample),
                "--output-dir", str(fdk_out_dir)
            ]
            subprocess.run(cmd, check=True)
        else:
            print(f"[{dataset_name}] Classical FDK volume already exists.")
            
        # 2. Build .npz Dataset
        sparse_suffix = f"_sparse_{args.sparse_step}" if args.sparse_step > 1 else ""
        npz_out_path = batch_datasets_dir / f"{dataset_name}{sparse_suffix}.npz"
        if not npz_out_path.exists():
            print(f"[{dataset_name}] Building Pure DL training dataset (.npz)...")
            cmd = [
                sys.executable, str(build_script),
                "--sample-dir", str(dataset_path),
                "--target-volume", str(fdk_vol_path),
                "--image-size", "1024",
                "--downsample-factor", str(args.downsample),
                "--output-path", str(npz_out_path),
                "--sparse-step", str(args.sparse_step)
            ]
            subprocess.run(cmd, check=True)
        else:
            print(f"[{dataset_name}] Dataset .npz already built.")
            
    print("\nBatch Processing Complete!")
    print(f"All processed training sets are located in: {batch_datasets_dir}")
    print("You can now train the Pure DL model using this folder.")

if __name__ == "__main__":
    main()
