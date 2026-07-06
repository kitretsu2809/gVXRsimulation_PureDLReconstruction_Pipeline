# gVXRsimulation PureDL Reconstruction Pipeline

This project provides a comprehensive, end-to-end pipeline for generating simulated Computed Tomography (CT) projection data using gVXR and reconstructing it into 2D CT slices using a novel **Pure Deep Learning (PureDL)** architecture. This approach circumvents classical physics-based reconstruction algorithms (like FDK or ASTRA-based methods) during inference, mapping sensor domain data directly to the image domain.

## Pure Deep Learning Architecture

The core of this project is the `PureDLPipeline` (located in `ct_recon/pure_dl_net.py`), an innovative 3-stage neural network architecture that entirely replaces standard analytical reconstruction steps with learned transformations. 

### Stage 1: SinogramUNet (Sensor Domain Rectification)
- **Purpose**: Cleans up noise and artifacts in the raw sinogram (sensor domain) before the transformation.
- **Architecture**: A standard U-Net architecture adapted for 2D sinogram data (Angles × Detectors). It acts as an initial filter to stabilize the input.

### Stage 2: ConvolutionalDomainTransform (Physics Replacement)
- **Purpose**: Acts as the "Domain Transformer". It maps the 2D Sinogram into a 2D Image slice (W × H), fundamentally bypassing traditional Radon transform inversion algorithms.
- **Architecture**: A fully convolutional network that maintains resolution independence. 
  - It spatially interpolates the sinogram tensor to match the target image grid.
  - Utilizes a bottleneck structure with **dilated convolutions** to force a massive receptive field. This allows the network to learn the global "unscrambling" of the Radon transform entirely from data.

### Stage 3: ImageUNet (Image Domain Enhancement)
- **Purpose**: Refines the output of the Domain Transform, sharpening industrial edges and restoring high-frequency details.
- **Architecture**: Another U-Net operating on the image domain, taking the rough spatially-transformed output and polishing it into the final high-quality CT slice.

## Execution Pipeline

The full pipeline is automated through batch scripts (`run_full_pipeline.sh` and `run_full_pipeline.bat`), divided into three main steps:

### Step 1: Generating Projections
```bash
python DATACREATION/generate_datasets.py
```
Uses **gVXR** to simulate X-ray projections of 3D STL models (located in `DATACREATION/STL/`). It simulates realistic physics, taking into account sensor configurations and object geometries.

### Step 2: Classical Reconstruction & Dataset Generation
```bash
python scripts/run_batch_pipeline.py
```
This step handles data preparation and optionally generates classical reconstructions (FDK/ASTRA) to serve as ground-truth targets or baselines for training the PureDL model.

### Step 3: Training the Pure DL Model
```bash
python scripts/pure_dl/02_train_pure_dl.py --dataset-path outputs/batch_datasets
```
Trains the `PureDLPipeline` on the generated batch datasets, learning the intricate mapping from noisy projections to clean CT slices.

## Requirements
- PyTorch
- gVXR
- ASTRA Toolbox (for ground truth generation and baseline comparison)

Please see `requirements.txt` for the full list of dependencies.
