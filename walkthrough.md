# End-to-End Pure DL Pipeline Guide

This guide walks you through the entire process of simulating physical CT data using gVXR, preparing the sparse datasets, training the Pure DL architecture, and finally running inference on new data!

## Prerequisites
Open a terminal and activate your Conda environment:
```bash
cd /home/kitretsu/Desktop/PureDL_gVXR_Pipeline
conda activate ct_pipeline
```
*(If you need to set up the environment, simply run `pip install -r requirements.txt`)*

---

## Step 1: Generate Physical Projections (The Simulator)

> [!WARNING]
> **Windows Requirement for gVXR**
> Because gVXR requires specific NVIDIA driver support for OpenGL/CUDA interoperability that is currently broken on Linux, **this data creation step must be run on a Windows machine.**
> 
> **Windows Setup:**
> 1. Open a Command Prompt or PowerShell on your Windows machine.
> 2. Install the gVXR engine and its dependencies via pip: 
>    `pip install gvxrPython3 numpy scikit-image trimesh`
> 3. Run the generation script, then transfer the resulting `data/` folder back to your Linux machine for training!

Before we can train the AI, we need data. The `generate_datasets.py` script automatically scans the `DATACREATION/STL/` folder for 3D CAD models and uses the **gVXR** physics engine to simulate realistic X-ray projections.

```cmd
:: On your Windows Machine, run the physics simulator
python DATACREATION\generate_datasets.py
```

**What happens:**
- The engine simulates a real cone-beam CT scanner.
- It generates 360-degree noisy projections (Sinograms) for different scenarios (e.g. titanium, aluminum, low dose).
- The raw output TIFFs are saved in the `data/` directory.

---

## Step 2: Build the Datasets (Sparse Data Mode)
We need to package the raw TIFF projections into `.npz` files for PyTorch. 

```bash
# Automatically generate sparse ground-truth targets and .npz datasets
python scripts/run_batch_pipeline.py --sparse-step 8
```

**What happens:**
1. Runs Classical FDK Reconstruction on the perfect baseline to generate a high-quality 3D "Answer Key".
2. Extracts sparse sinograms from the noisy runs (e.g. only 45 angles).
3. Packages them as `sample_name_sparse_8.npz` inside the `outputs/batch_datasets/` folder.

---

## Step 3: Train the Pure DL Model
Now that the data is structured, we can train the 3-Stage Pure DL Pipeline to learn the physics mapping.

```bash
# Train the model on the sparse dataset
python scripts/pure_dl/02_train_pure_dl.py \
    --dataset-path outputs/batch_datasets \
    --output-dir outputs/pure_dl_sparse_8 \
    --epochs 50 \
    --batch-size 2
```

> [!TIP]
> Notice that `--dataset-path` points directly to the `outputs/batch_datasets` folder instead of a specific file. The training script will automatically discover all the generated `.npz` files inside and combine them for massive-scale training!

**What happens:**
- **Stage 1** cleans the sparse sinograms.
- **Stage 2** stretches and guesses the physical mapping into a 2D image grid.
- **Stage 3** polishes the resulting image against the classical "Answer Key".
- The best performing model weights are saved in `outputs/pure_dl_sparse_8/best_model_centered.pt`.

---

## Step 4: Run Inference (Reconstruct a 3D Volume!)
Once the model is trained, you can use it to reconstruct a full 3D volume from a raw folder of projections (even if that folder only contains a sparse set of angles).

```bash
# Reconstruct a 3D volume using the trained AI
python scripts/pure_dl/03_inference.py \
    --model-path outputs/pure_dl_sparse_8/best_model_centered.pt \
    --sample-dir data/sample_1 \
    --output-path outputs/final_reconstruction.tif
```

**What happens:**
1. The script loads your trained `.pt` weights.
2. It slices the raw projection data horizontally (row by row).
3. It passes each sparse sinogram through the AI to generate a clean, reconstructed 2D slice.
4. It stacks all the 2D slices together and saves them as a single, fully reconstructed 3D `.tif` volume!
