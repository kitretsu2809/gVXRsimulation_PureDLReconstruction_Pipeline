# 🚀 Kaggle Training Guide for Pure DL CT Reconstruction

Since you have an Nvidia T4 (16GB VRAM) available for free on Kaggle, we can train this model significantly faster than on your laptop! We can increase the batch size from 2 to **16 or 32**.

## 1. Prepare your files
I am currently generating your dataset locally on your laptop in the background. Once it finishes, do the following:
1. Open your project folder (`gVXRsimulation_PureDLReconstruction_Pipeline`).
2. Select these specific folders/files and add them to a single `.zip` file (e.g., `ct_training.zip`):
   - `ct_recon/` (Folder)
   - `scripts/` (Folder)
   - `outputs/batch_datasets/` (Folder - contains your `.npz` data)

## 2. Upload to Kaggle
1. Go to [Kaggle](https://www.kaggle.com/) and create a free account if you haven't already.
2. Click **Create -> New Notebook**.
3. On the right-side panel, find **Session Options** (the 3 dots or arrow) and change the **Accelerator** from `None` to **GPU T4 x2** or **GPU P100**.
4. In the top-right corner, click **Add Input -> Upload Data**, name it something like `ct-dataset`, and upload your `ct_training.zip` file.

## 3. Run the Training
In your Kaggle Notebook, paste the following code into the first cell and hit the **Play** button:

```python
import os
import shutil

# 1. Copy files from your uploaded dataset into the working directory
# Note: Replace 'ct-dataset' below with whatever you named your dataset in Kaggle!
dataset_path = "/kaggle/input/ct-dataset"
working_dir = "/kaggle/working"

for item in os.listdir(dataset_path):
    s = os.path.join(dataset_path, item)
    d = os.path.join(working_dir, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

# 2. Run the training script!
# Because Kaggle gives you 16GB of VRAM, we can safely boost the batch size to 16 for blazing fast training!
!python scripts/pure_dl/02_train_pure_dl.py --dataset-path /kaggle/working/outputs/batch_datasets --epochs 50 --batch-size 16
```

## 4. Download the Trained Model
When the cell finishes running, it will have saved a file called `best_model_centered.pt`. 
1. Look at the right-side panel under **Output** -> `/kaggle/working/outputs/pure_dl_training_centered/`.
2. Find `best_model_centered.pt`, click the three dots, and select **Download**.
3. Place this file back in your laptop's `outputs/pure_dl_training_centered/` folder.

## 5. Reconstruct on your Laptop
Finally, on your laptop, open the GUI (`python gui/pipeline_gui.py`), ensure the settings match what you trained on (Resolution 256), and click **Export 3D CAD File (STL/OBJ)** or **Open DL Reconstruction** to generate your CAD file from the newly trained weights!
