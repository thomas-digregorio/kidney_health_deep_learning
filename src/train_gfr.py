import sys
from typing import Any
import wandb
import xgboost as xgb
from wandb.integration.xgboost import WandbCallback
# Use cuML metrics for GPU acceleration
from cuml.metrics import mean_squared_error
# Enforce cuML
try:
    from cuml.model_selection import train_test_split
except ImportError:
    print("cuML not found (env check failed). Exiting.")
    sys.exit(1)

import cudf
from common import load_data, preprocess_data, validate_os, PROCESSED_DATA_DIR

def train_gfr_model(
    X_train: cudf.DataFrame, 
    y_train: cudf.Series, 
    X_test: cudf.DataFrame, 
    y_test: cudf.Series,
    config: Any
) -> xgb.Booster:
    """Train the Regression Model (GFR)."""
    print("Training Regression Model (GFR)...")
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'device': 'cuda',
        'learning_rate': config.learning_rate,
        'max_depth': config.max_depth,
        'subsample': config.subsample,
        'eval_metric': ['rmse']
    }
    
    bst = xgb.train(
        params, 
        dtrain, 
        num_boost_round=100, 
        evals=[(dtest, "test")],
        early_stopping_rounds=10,
        callbacks=[WandbCallback(log_model=True, log_feature_importance=False)]
    )
    
    preds = bst.predict(dtest)
    rmse = float(mean_squared_error(y_test, preds, squared=False))
    
    wandb.log({"gfr_rmse_final": rmse})
    return bst

def main() -> None:
    validate_os()
    
    # Setup WandB
    wandb.init(project="kidney-health", job_type="train-regressor", config={
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8
    })
    config = wandb.config
    
    # Pipeline
    df = load_data(PROCESSED_DATA_DIR)
    X, _, y_gfr = preprocess_data(df) # Ignore ckd
    
    X_train, X_test, y_gfr_train, y_gfr_test = train_test_split(X, y_gfr, test_size=0.2, random_state=42)
    
    # Train
    bst_gfr = train_gfr_model(X_train, y_gfr_train, X_test, y_gfr_test, config)
    
    # Save
    bst_gfr.save_model("model_gfr.json")
    print("GFR Model saved.")

if __name__ == "__main__":
    main()
