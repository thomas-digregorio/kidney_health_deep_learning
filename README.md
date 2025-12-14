# Kidney Function Deep Learning Project
This project implements GPU-accelerated Machine Learning models (using RAPIDS cuDF and XGBoost) to predict kidney health metrics from high-performance Parquet data, with experiment tracking via Weights & Biases.

It contains two primary models:
1.  **CKD Classifier**: Predicts binary `CKD_Status` (Chronic Kidney Disease).
2.  **GFR Regressor**: Predicts continuous `GFR` (Glomerular Filtration Rate).

## Project Structure

```text
├── .env                    # Environment variables (W&B API Key)
├── data/
│   ├── raw/                # Original CSV data
│   └── processed/          # High-performance Parquet format
├── environment.yaml        # Conda environment specification
├── notebooks/              # Jupyter notebooks for EDA and Analysis
│   ├── 01_eda_datashader.ipynb    # GPU-accelerated visualization
│   └── 02_analysis_post_hoc.ipynb # Model analysis (SHAP, plots)
├── src/                    # Source code
│   ├── common.py           # Shared data loading & preprocessing utils
│   ├── ingest.py           # Data conversion script (CSV -> Parquet)
│   ├── train_ckd.py        # Training script for Classification
│   └── train_gfr.py        # Training script for Regression
├── sweep_ckd.yaml          # W&B Sweep Config for Classifier
├── sweep_gfr.yaml          # W&B Sweep Config for Regressor
└── WSL_SETUP.md            # Detailed WSL2 setup guide
```

## Requirements & Setup

**Core Requirement**: This project uses RAPIDS (`cuml`, `cudf`), which **requires NVIDIA GPU and Linux/WSL2**. It will **NOT** run on native Windows.

### 1. Prerequisites (Windows Users)
Ensure you have **WSL2** installed with an Ubuntu instance.
See [WSL_SETUP.md](./WSL_SETUP.md) for detailed instructions.

### 2. Install Environment
Run the following inside your WSL terminal:

```bash
# Update conda
conda update conda

# Create environment from file
conda env create -f environment.yaml

# Activate environment
conda activate kidney_dl
```

### 3. Data Preparation
Convert the raw CSV data into high-performance Parquet format:

```bash
python src/ingest.py
```

## Training

To train the models with default hyperparameters:

**CKD Classification:**
```bash
python src/train_ckd.py
```

**GFR Regression:**
```bash
python src/train_gfr.py
```

These scripts automatically log metrics to [Weights & Biases](https://wandb.ai/).

## Hyperparameter Sweeps

This project uses Weights & Biases Sweeps for Bayesian Hyperparameter Optimization.

**1. Initialize Sweep:**
```bash
wandb sweep sweep_ckd.yaml
# OR
wandb sweep sweep_gfr.yaml
```
*This command will output an alphanumeric SWEEP_ID.*

**2. Start Agent:**
```bash
wandb agent <your-entity>/kidney-health/<SWEEP_ID> --count 20
```

## Notebooks

To run the analysis notebooks, ensure you start Jupyter Lab from within the WSL environment:

```bash
jupyter lab
```

*   **01_eda_datashader.ipynb**: Visualizes the dataset using `datashader` for handling large (or potentially large) datasets on GPU.
*   **02_analysis_post_hoc.ipynb**: Loads trained models (`model_ckd.json`, `model_gfr.json`) and performs SHAP analysis to explain model predictions.

## Design Notes

*   **Modularization**: Training logic is split into separate scripts (`train_ckd.py`, `train_gfr.py`) to avoid W&B config conflicts and allow independent tuning.
*   **Data Leakage Prevention**: When training the CKD Classifier, the `GFR` regressor target is rigorously excluded from the feature set, as CKD status is mathematically derived from GFR.
