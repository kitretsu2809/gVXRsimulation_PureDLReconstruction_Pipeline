@echo off

echo ================================================================
echo Step 1: Generating Projections using gVXR
echo ================================================================
conda run -n ct_pipeline python DATACREATION\generate_datasets.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo ================================================================
echo Step 2: Classical Reconstruction ^& Dataset Generation
echo ================================================================
conda run -n ct_pipeline python scripts\run_batch_pipeline.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo ================================================================
echo Step 3: Training Pure DL Model on Generated Dataset
echo ================================================================
conda run -n ct_pipeline python scripts\pure_dl\02_train_pure_dl.py --dataset-path outputs\batch_datasets
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo ================================================================
echo Pipeline completed successfully!
echo ================================================================
