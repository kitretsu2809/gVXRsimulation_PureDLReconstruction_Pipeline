# Step 1 — add your new .stl to DATACREATION/STL/ then regenerate projections
python DATACREATION/generate_datasets.py

# Step 2 — rebuild the .npz dataset (includes all old + new data)
python scripts/data_preparation/01_build_dataset.py

# Step 3 — resume training from your existing best model
python scripts/pure_dl/02_train_pure_dl.py \
    --resume-checkpoint outputs/pure_dl_training_centered/best_model_centered.pt \
    --learning-rate 1e-4 \
    --epochs 10
