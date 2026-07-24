#!/usr/bin/env python3
"""
Sequential STL Training Pipeline
=================================
Trains the Pure DL model one STL at a time to keep disk usage flat.

For each STL it:
  1. Runs gVXR to generate projections for every material variation
  2. Runs classical FDK reconstruction to get the reference CT volume
  3. Builds a .npz training dataset
  4. Trains (or resumes) the Pure DL model with an auto-decayed learning rate
  5. Runs DL inference to produce a reconstructed 3D volume (kept as output)
  6. DELETES all generated data for that STL (projections, FDK volume, .npz)
  7. Saves pipeline state so new STLs can be added and training resumed later

Usage
-----
  # First time (train all STLs from scratch):
  python scripts/sequential_train_pipeline.py

  # Add new .stl files to DATACREATION/STL/ then just run again:
  python scripts/sequential_train_pipeline.py

  # Preview what would happen without running anything:
  python scripts/sequential_train_pipeline.py --dry-run

  # Override epochs or scan method:
  python scripts/sequential_train_pipeline.py --epochs 10 --scan-method centered

  # Quick local test (1 STL, 1 material, 2 epochs — finishes in minutes):
  python scripts/sequential_train_pipeline.py --quick-test

  # Training only (fast) — run inference manually later when needed:
  python scripts/sequential_train_pipeline.py

  # Training + inference after each STL (slower, but saves a reconstruction):
  python scripts/sequential_train_pipeline.py --run-inference
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ct_recon.paths import OUTPUTS_DIR

# ---------------------------------------------------------------------------
# State file — persists between runs so new STLs can be added later
# ---------------------------------------------------------------------------
STATE_FILE = OUTPUTS_DIR / "seq_pipeline_state.json"

# ---------------------------------------------------------------------------
# Learning rate schedule
# Starts at LR_INITIAL, multiplied by LR_DECAY each round, never below LR_FLOOR
# Round 0 (first STL) : 1e-3   (fresh training)
# Round 1              : 6e-4
# Round 2              : 3.6e-4
# Round 3              : 2.2e-4
# Round 4              : 1.3e-4
# Round 5+             : 5e-5  (floor — fine-tuning plateau)
# ---------------------------------------------------------------------------
LR_INITIAL = 1e-3
LR_DECAY   = 0.6
LR_FLOOR   = 5e-5

# ---------------------------------------------------------------------------
# Material / noise variations — each STL is simulated with all of these.
# Uses real NIST X-ray attenuation data inside gVXR per element.
# Covers a wide density range so the model generalises to many materials.
# ---------------------------------------------------------------------------
MATERIAL_VARIATIONS = [
    # ---- Standard reference ----
    {"suffix": "Ti_standard",      "material": "Ti", "i0": 50000.0, "gaussian_std": 10.0},  # 4.51 g/cm³
    # ---- Lightweight metal ----
    {"suffix": "Al_low_density",   "material": "Al", "i0": 50000.0, "gaussian_std": 10.0},  # 2.70 g/cm³
    # ---- Heavy structural metal ----
    {"suffix": "Cu_copper",        "material": "Cu", "i0": 50000.0, "gaussian_std": 10.0},  # 8.96 g/cm³
    # ---- Ultra-dense metal ----
    {"suffix": "W_tungsten",       "material": "W",  "i0": 50000.0, "gaussian_std": 10.0},  # 19.3 g/cm³
    # ---- Noise stress test ----
    {"suffix": "Ti_high_noise",    "material": "Ti", "i0": 10000.0, "gaussian_std": 20.0},  # Low photon count
]

# Script paths (relative to repo root)
_GVXR_SCRIPT      = REPO_ROOT / "DATACREATION"   / "gvxr_projection_script.py"
_FDK_SCRIPT       = REPO_ROOT / "scripts" / "classical_reconstruction" / "reconstruct_fdk.py"
_BUILD_SCRIPT     = REPO_ROOT / "scripts" / "data_preparation"         / "01_build_dataset.py"
_TRAIN_SCRIPT     = REPO_ROOT / "scripts" / "pure_dl"                  / "02_train_pure_dl.py"
_INFERENCE_SCRIPT = REPO_ROOT / "scripts" / "pure_dl"                  / "03_inference.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_state() -> dict:
    """Load pipeline state from disk (or return fresh state)."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("[WARN] State file is corrupted — starting fresh.")
    return {
        "completed_stls": [],
        "round": 0,
        "last_checkpoint": None,
        "last_lr": LR_INITIAL,
    }


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def compute_lr(round_idx: int) -> float:
    """Return the learning rate for a given round index (0-based)."""
    lr = LR_INITIAL * (LR_DECAY ** round_idx)
    return max(lr, LR_FLOOR)


def run_step(cmd: list, dry_run: bool, label: str) -> None:
    """Print and optionally execute a subprocess command."""
    cmd_str = " ".join(str(c) for c in cmd)
    prefix = "[DRY-RUN] " if dry_run else ""
    print(f"\n{prefix}▶  {label}")
    print(f"   {cmd_str}")
    if not dry_run:
        subprocess.run([str(c) for c in cmd], check=True)


def delete_dir(path: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"   [DRY-RUN] Would delete directory: {path}")
        return
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
        print(f"   🗑  Deleted: {path}")


def delete_file(path: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"   [DRY-RUN] Would delete file: {path}")
        return
    if path.exists():
        path.unlink()
        print(f"   🗑  Deleted: {path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential STL Training Pipeline — trains one STL at a time "
                    "and deletes generated data to keep disk usage flat.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stl-dir",
        default=str(REPO_ROOT / "DATACREATION" / "STL"),
        help="Directory containing .stl files (default: DATACREATION/STL/)",
    )
    parser.add_argument(
        "--scan-method",
        choices=["centered", "offset", "auto"],
        default="auto",
        help="gVXR scan geometry (default: auto — chooses based on object size)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Training epochs per STL round (5-10 recommended, default: 8)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size (default: 2)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=2,
        help="Spatial downsample factor for FDK & dataset (default: 2)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Target image size for training slices (default: 256)",
    )
    parser.add_argument(
        "--sparse-step",
        type=int,
        default=1,
        help="Projection sampling step (default: 1 — uses ALL 360 projection angles for full dense reconstruction)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of slices to use for validation (default: 0.2)",
    )
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="After training each STL, run DL inference to produce a 3D reconstruction volume "
             "BEFORE deleting the projection data. Off by default to keep training fast. "
             "Output always overwrites outputs/dl_reconstruction/dl_volume.tif (no disk accumulation). "
             "INPUT: Ti_standard projection data of the current STL (the only data available before cleanup).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print every step without running anything — useful for testing on lab computer before committing real compute.",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a fast end-to-end test: uses only the first STL, only the Ti_standard "
             "material variation, and 2 training epochs. Finishes in minutes. "
             "Use this to verify the full pipeline works before a real run.",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Only generate projections, FDK volumes, and .npz datasets. "
             "Skip training and inference entirely. Use this to prepare data "
             "on a machine with OpenGL (laptop) and train elsewhere (lab/Colab).",
    )
    args = parser.parse_args()

    # --quick-test overrides: 1 STL, 1 material, 2 epochs, low resolution
    if args.quick_test:
        print("\n⚡  QUICK-TEST MODE: 1 STL · Ti_standard only · 2 epochs · Low-Res (128x128)")
        args.epochs = 2
        args.downsample = 4
        args.image_size = 128
        material_variations = [MATERIAL_VARIATIONS[0]]  # Ti_standard only
    else:
        material_variations = MATERIAL_VARIATIONS

    stl_dir = Path(args.stl_dir)
    all_stls = sorted(stl_dir.glob("*.stl"))
    if not all_stls:
        print(f"❌  No .stl files found in {stl_dir}")
        sys.exit(1)

    state = load_state()
    completed = set(state["completed_stls"])
    pending   = [s for s in all_stls if s.name not in completed]

    # ------------------------------------------------------------------
    print(f"\n{'='*62}")
    print(f"  Sequential STL Training Pipeline")
    print(f"{'='*62}")
    print(f"  STL directory : {stl_dir}")
    print(f"  Total STLs    : {len(all_stls)}")
    print(f"  Already done  : {len(completed)}")
    print(f"  Pending       : {len(pending)}")
    print(f"  State file    : {STATE_FILE}")
    if args.dry_run:
        print(f"\n  *** DRY-RUN MODE — nothing will be executed or deleted ***")
    print(f"{'='*62}\n")

    if not pending:
        print("✅  All STL files already trained. Drop new .stl files into")
        print(f"   {stl_dir}  and run this script again to extend training.")
        return

    data_dir = REPO_ROOT / "data"

    for stl_path in pending:
        stl_name  = stl_path.stem
        round_idx = state["round"]
        lr        = compute_lr(round_idx)

        print(f"\n{'#'*62}")
        print(f"#  STL : {stl_name}")
        print(f"#  Round {round_idx + 1}   |   Learning Rate: {lr:.2e}   |   Epochs: {args.epochs}")
        print(f"{'#'*62}")

        dirs_to_delete  = []   # collected during this round, deleted at end
        files_to_delete = []
        npz_files       = []
        # Track first variation's data dir for inference (we use Ti_standard as reference)
        inference_sample_dir = None

        batch_dir = OUTPUTS_DIR / f"seq_batch_{stl_name}"
        existing_npz = list(batch_dir.glob("*.npz")) if batch_dir.exists() else []

        if existing_npz:
            print(f"\n   ⏭  Found {len(existing_npz)} pre-built .npz dataset(s) in {batch_dir}")
            print(f"   ⚡  Skipping gVXR simulation & FDK reconstruction — jumping straight to training!")
        else:
            # ----------------------------------------------------------------
            # Step 1 + 2 + 3  (per material variation)
            # ----------------------------------------------------------------
            for var in material_variations:
                tag          = var["suffix"]
                dataset_name = f"{stl_name}_{tag}"
                out_dir      = data_dir / dataset_name

                # ---- 1. gVXR projection simulation ----
                # gVXR script appends _<scan_method> to the output dir
                actual_data_dir = Path(f"{out_dir}_{args.scan_method}")

                if actual_data_dir.exists() and (actual_data_dir / "settings.cto").exists():
                    print(f"\n   ⏭  Skipping gVXR for {tag} — projections already exist at {actual_data_dir}")
                else:
                    run_step(
                        [
                            sys.executable, _GVXR_SCRIPT,
                            "--stl",           stl_path,
                            "--output_dir",    out_dir,
                            "--material",      var["material"],
                            "--i0",            var["i0"],
                            "--gaussian-std",  var["gaussian_std"],
                            "--scan-method",   args.scan_method,
                        ],
                        dry_run=args.dry_run,
                        label=f"[{stl_name}] gVXR simulation — {tag} ({var['material']}, {var['i0']:.0f} photons)",
                    )

                dirs_to_delete.append(actual_data_dir)
                # Use the first variation as the inference reference sample
                if inference_sample_dir is None:
                    inference_sample_dir = actual_data_dir

                # ---- 2. FDK reconstruction (reference volume) ----
                fdk_out_dir  = OUTPUTS_DIR / f"fdk_{stl_name}_{tag}"
                fdk_vol_path = fdk_out_dir / "fdk_volume.tif"

                if fdk_vol_path.exists():
                    print(f"\n   ⏭  Skipping FDK for {tag} — volume already exists at {fdk_vol_path}")
                else:
                    run_step(
                        [
                            sys.executable, _FDK_SCRIPT,
                            "--sample-dir", actual_data_dir,
                            "--downsample", args.downsample,
                            "--output-dir", fdk_out_dir,
                        ],
                        dry_run=args.dry_run,
                        label=f"[{stl_name}] FDK reconstruction — {tag}",
                    )
                dirs_to_delete.append(fdk_out_dir)

                # ---- 3. Build .npz dataset ----
                npz_path = OUTPUTS_DIR / f"seq_{stl_name}_{tag}.npz"
                if npz_path.exists():
                    print(f"\n   ⏭  Skipping dataset build for {tag} — {npz_path.name} already exists")
                else:
                    run_step(
                        [
                            sys.executable, _BUILD_SCRIPT,
                            "--sample-dir",       actual_data_dir,
                            "--target-volume",    fdk_vol_path,
                            "--output-path",      npz_path,
                            "--downsample-factor",args.downsample,
                            "--image-size",       args.image_size,
                            "--sparse-step",      args.sparse_step,
                        ],
                        dry_run=args.dry_run,
                        label=f"[{stl_name}] Build dataset — {tag}",
                    )
                npz_files.append(npz_path)
                files_to_delete.append(npz_path)

                # ---- 4. Immediate Cleanup (Save Storage!) ----
                # Projections and FDK volumes are massive (5GB+ per variation).
                # We must delete them immediately to prevent filling the user's hard drive.
                # But only delete if the .npz has been successfully created.
                if npz_path.exists():
                    if args.run_inference and actual_data_dir == inference_sample_dir:
                        # Keep the first variation's projections around for inference later
                        print(f"   [INFO] Keeping {actual_data_dir.name} projections for inference later.")
                        # We can still delete its FDK volume since inference only needs projections
                        delete_dir(fdk_out_dir, args.dry_run)
                    else:
                        delete_dir(actual_data_dir, args.dry_run)
                        delete_dir(fdk_out_dir, args.dry_run)

            # ----------------------------------------------------------------
            # Step 4 — Move all .npz files into a batch folder for multi-file
            #           training, then call 02_train_pure_dl.py
            # ----------------------------------------------------------------
            if not args.dry_run:
                batch_dir.mkdir(parents=True, exist_ok=True)
                for npz in npz_files:
                    if npz.exists():
                        npz.rename(batch_dir / npz.name)
            else:
                print(f"\n[DRY-RUN] Would create {batch_dir} and move {len(npz_files)} .npz files into it")
        # NOTE: Do NOT add batch_dir to dirs_to_delete so the dataset remains preserved for future training runs!

        # Build training command
        if args.data_only:
            print(f"\n   ℹ️  --data-only mode: skipping training. Datasets saved in {batch_dir}")
        else:
            checkpoint = state.get("last_checkpoint")
            train_cmd = [
                sys.executable, _TRAIN_SCRIPT,
                "--dataset-path",   batch_dir,
                "--epochs",         args.epochs,
                "--batch-size",     args.batch_size,
                "--learning-rate",  f"{lr:.6f}",
                "--val-fraction",   args.val_fraction,
                "--scan-method",    "centered",      # always use centered for the model
            ]
            if checkpoint and (args.dry_run or Path(checkpoint).exists()):
                train_cmd += ["--resume-checkpoint", checkpoint]
                resume_label = f"resume from {Path(checkpoint).name}"
            else:
                resume_label = "fresh training"

            run_step(
                train_cmd,
                dry_run=args.dry_run,
                label=f"[{stl_name}] Train model — {resume_label}, lr={lr:.2e}, epochs={args.epochs}",
            )

        # ----------------------------------------------------------------
        # Step 5 — Inference (OPTIONAL — only runs if --run-inference is set)
        # ----------------------------------------------------------------
        if args.data_only:
            print(f"\n   ℹ️  --data-only mode: skipping inference and cleanup.")
            print(f"       Datasets preserved in: {batch_dir}")
        else:
            if args.run_inference:
                best_ckpt     = OUTPUTS_DIR / "pure_dl_training_centered" / "best_model_centered.pt"
                dl_recon_dir  = OUTPUTS_DIR / "dl_reconstruction"
                dl_recon_path = dl_recon_dir / "dl_volume.tif"
                if not args.dry_run:
                    dl_recon_dir.mkdir(parents=True, exist_ok=True)
                    # Delete old reconstruction before writing new one
                    if dl_recon_path.exists():
                        dl_recon_path.unlink()
                        print(f"   🗑  Deleted old reconstruction: {dl_recon_path}")
                if inference_sample_dir is not None:
                    run_step(
                        [
                            sys.executable, _INFERENCE_SCRIPT,
                            "--model-path",  best_ckpt,
                            "--sample-dir",  inference_sample_dir,
                            "--output-path", dl_recon_path,
                            "--batch-size",  8,
                        ],
                        dry_run=args.dry_run,
                        label=f"[{stl_name}] DL Inference → {dl_recon_path}  (input: Ti_standard projections)",
                    )
                    if not args.dry_run and dl_recon_path.exists():
                        print(f"   📦  Reconstruction saved (overwrote previous): {dl_recon_path}")
                else:
                    print(f"   [WARN] No inference_sample_dir found — skipping inference for {stl_name}")
            else:
                print(f"\n   ℹ️  Inference skipped (use --run-inference to enable). "
                      f"To run inference manually later, regenerate projections first.")

            # ----------------------------------------------------------------
            # Step 6 — Cleanup: delete all generated data for this STL
            # ----------------------------------------------------------------
            print(f"\n{'─'*62}")
            print(f"  🧹  Cleaning up data for {stl_name} ...")
            for d in dirs_to_delete:
                delete_dir(d, args.dry_run)
            for f in files_to_delete:
                delete_file(f, args.dry_run)

        # ----------------------------------------------------------------
        # Step 7 — Save state
        # ----------------------------------------------------------------
        best_ckpt = OUTPUTS_DIR / "pure_dl_training_centered" / "best_model_centered.pt"
        state["completed_stls"].append(stl_path.name)
        state["round"]           = round_idx + 1
        state["last_lr"]         = lr
        state["last_checkpoint"] = str(best_ckpt) if (best_ckpt.exists() or args.dry_run) else state.get("last_checkpoint")

        if not args.dry_run:
            save_state(state)

        print(f"\n{'─'*62}")
        if args.dry_run:
            print(f"  [DRY-RUN] Would mark {stl_path.name} complete.")
            print(f"  [DRY-RUN] Next round would use lr={compute_lr(round_idx + 1):.2e}")
        else:
            print(f"  ✅  [{stl_name}] Complete.  State saved → {STATE_FILE}")
            print(f"      Next round LR will be: {compute_lr(round_idx + 1):.2e}")

    # ----------------------------------------------------------------
    print(f"\n{'='*62}")
    if args.dry_run:
        print(f"  [DRY-RUN] Pipeline preview complete.  {len(pending)} STL(s) would be trained.")
    else:
        print(f"  ✅  Sequential pipeline complete!  {len(pending)} STL(s) trained.")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
