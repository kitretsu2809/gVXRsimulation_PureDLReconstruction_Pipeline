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
    
    # Dataset variations — covers a wide range of densities and noise levels
    # so the trained model generalises to many real-world materials.
    # gVXR uses real NIST X-ray attenuation coefficients per element, so
    # changing the element string automatically gives physically correct contrast.
    variations = [
        # ---- Standard reference (Titanium, 4.51 g/cm³) ----
        {"suffix": "Ti_standard",      "material": "Ti", "i0": 50000.0, "gaussian_std": 10.0},

        # ---- Lightweight metals ----
        {"suffix": "Mg_lightweight",   "material": "Mg", "i0": 50000.0, "gaussian_std": 10.0},  # Magnesium  1.74 g/cm³  — aerospace
        {"suffix": "Al_low_density",   "material": "Al", "i0": 50000.0, "gaussian_std": 10.0},  # Aluminium  2.70 g/cm³  — structural

        # ---- Medium-density structural metals ----
        {"suffix": "Fe_iron",          "material": "Fe", "i0": 50000.0, "gaussian_std": 10.0},  # Iron       7.87 g/cm³  — steel
        {"suffix": "Ni_nickel",        "material": "Ni", "i0": 50000.0, "gaussian_std": 10.0},  # Nickel     8.90 g/cm³  — superalloys
        {"suffix": "Cu_copper",        "material": "Cu", "i0": 50000.0, "gaussian_std": 10.0},  # Copper     8.96 g/cm³  — electronics/pipes

        # ---- High-density metals ----
        {"suffix": "Pb_lead",          "material": "Pb", "i0": 50000.0, "gaussian_std": 10.0},  # Lead      11.34 g/cm³  — shielding
        {"suffix": "W_tungsten",       "material": "W",  "i0": 50000.0, "gaussian_std": 10.0},  # Tungsten  19.30 g/cm³  — hardmetals/tooling

        # ---- Noise stress tests (Titanium base) ----
        {"suffix": "Ti_high_noise",    "material": "Ti", "i0": 10000.0, "gaussian_std": 20.0},  # Low photon count — noisy detector
        {"suffix": "Ti_no_noise",      "material": "Ti", "i0": 50000.0, "gaussian_std":  0.0},  # Perfect reference — no noise
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
