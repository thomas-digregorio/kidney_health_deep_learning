import sys
from typing import Any
import wandb
import xgboost as xgb
from wandb.integration.xgboost import WandbCallback
# Use cuML metrics for GPU acceleration
from cuml.metrics import accuracy_score, roc_auc_score
# Enforce cuML
try:
    from cuml.model_selection import train_test_split
except ImportError:
    print("cuML not found (env check failed). Exiting.")
    sys.exit(1)

import cudf
from common import load_data, preprocess_data, validate_os, PROCESSED_DATA_DIR

def train_ckd_model(
    X_train: cudf.DataFrame, 
    y_train: cudf.Series, 
    X_test: cudf.DataFrame, 
    y_test: cudf.Series,
    config: Any
) -> xgb.Booster:
    """Train the Classification Model (CKD Status)."""
    print("Training Classification Model (CKD)...")
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'device': 'cuda',
        'learning_rate': config.learning_rate,
        'max_depth': config.max_depth,
        'subsample': config.subsample,
        'eval_metric': ['auc', 'logloss']
    }
    
    bst = xgb.train(
        params, 
        dtrain, 
        num_boost_round=100, 
        evals=[(dtest, "test")],
        early_stopping_rounds=10,
        # Standard callback works fine in single-model script
        callbacks=[WandbCallback(log_model=True, log_feature_importance=False)] 
    )
    
    # Evaluation
    preds_prob = bst.predict(dtest)
    preds_binary = (preds_prob > 0.5).astype('int32')
    
    accuracy = float(accuracy_score(y_test, preds_binary))
    auc = float(roc_auc_score(y_test, preds_prob))
    
    wandb.log({"ckd_accuracy_final": accuracy, "ckd_auc_final": auc})
    return bst

def main() -> None:
    validate_os()
    
    # Setup WandB
    wandb.init(project="kidney-health", job_type="train-classifier", config={
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8
    })
    config = wandb.config
    
    # Pipeline
    df = load_data(PROCESSED_DATA_DIR)
    X, y_ckd, _ = preprocess_data(df) 
    # We ignore the GFR target (_) here. 
    # NOTE: GFR is also dropped from 'X' in preprocess_data() to prevent Data Leakage, 
    # since CKD_Status is defined by GFR (CKD if GFR < 60).
    
    X_train, X_test, y_ckd_train, y_ckd_test = train_test_split(X, y_ckd, test_size=0.2, random_state=42)
    
    # Train
    bst_ckd = train_ckd_model(X_train, y_ckd_train, X_test, y_ckd_test, config)
    
    # Save
    bst_ckd.save_model("model_ckd.json")
    print("CKD Model saved.")

if __name__ == "__main__":
    main()
