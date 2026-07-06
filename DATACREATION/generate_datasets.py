#!/usr/bin/env python3
import os
import glob
import sys
import subprocess
from pathlib import Path

def generate_datasets():
    # Central output data directory (Dynamic for portability)
    script_dir = Path(__file__).resolve().parent
    output_base_dir = script_dir.parent / "data"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    output_base_dir_str = str(output_base_dir)
    
    # Locate all STLs
    stl_dir = script_dir / "STL"
    stl_files = glob.glob(str(stl_dir / "*.stl"))
    if not stl_files:
        print("No STL files found in the STL/ directory.")
        return
        
    print(f"Found {len(stl_files)} STL files: {stl_files}")
    
    # Define our dataset variations
    variations = [
        {
            "suffix": "standard",
            "material": "Ti",
            "i0": 50000.0,
            "gaussian_std": 10.0,
        },
        {
            "suffix": "dense_high_noise",
            "material": "Ti",
            "i0": 10000.0,
            "gaussian_std": 20.0,
        },
        {
            "suffix": "perfect_no_noise",
            "material": "Ti",
            "i0": 50000.0, # ignored practically when we want no noise, but handled in gvxr via clipping? Wait, gvxr script always applies noise. 
            "gaussian_std": 0.0,
        },
        {
            "suffix": "aluminum_low_density",
            "material": "Al",
            "i0": 50000.0,
            "gaussian_std": 10.0,
        }
        # Add hundreds more variations here as needed!
    ]
    
    for stl_path in stl_files:
        stl_name = Path(stl_path).stem
        print(f"\n{'='*50}\nProcessing STL: {stl_name}\n{'='*50}")
        
        for var in variations:
            dataset_name = f"{stl_name}_{var['suffix']}"
            dataset_out_dir = os.path.join(output_base_dir_str, dataset_name)
            
            # Since we pass --scan-method auto, gvxr appends _auto to the output directory
            dataset_out_dir_actual = f"{dataset_out_dir}_auto"
            
            print(f"  -> Generating variation: {var['suffix']}")
            
            # Skip if already fully generated (settings.cto is written at the very end)
            if os.path.exists(dataset_out_dir_actual) and os.path.exists(os.path.join(dataset_out_dir_actual, "settings.cto")):
                print(f"     Already fully generated at {dataset_out_dir_actual}, skipping.")
                continue
                
            try:
                cmd = [
                    sys.executable, str(script_dir / "gvxr_projection_script.py"),
                    "--stl", stl_path,
                    "--output_dir", dataset_out_dir,
                    "--material", var["material"],
                    "--i0", str(var["i0"]),
                    "--gaussian-std", str(var["gaussian_std"]),
                    "--scan-method", "auto"
                ]
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"     Error generating {dataset_name}: Subprocess failed with exit code {e.returncode}")
            except Exception as e:
                print(f"     Error generating {dataset_name}: {e}")

if __name__ == "__main__":
    generate_datasets()
