import sys
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from pathlib import Path

# --- Mock Setup ---
# We mock GPU libraries globally before importing src modules to prevent ImportError or runtime failures.
# This allows testing the *logic* of the pipelines on non-GPU/Windows machines.

@pytest.fixture(scope="module", autouse=True)
def mock_gpu_environment():
    """
    Patches sys.modules to replace 'cudf', 'cuml', 'xgboost' with CPU-compatible mocks
    before any source code is imported.
    """
    # 1. Alias cudf to pandas
    # This allows us to use real DataFrames for logic testing
    sys.modules["cudf"] = pd

    # 2. Mock cuml (doesn't exist on CPU)
    # We mock specific metrics and splitters used in the code
    cuml_mock = MagicMock()
    # Mock metrics to return dummy values
    cuml_mock.metrics.accuracy_score = lambda y_true, y_pred: 0.95
    cuml_mock.metrics.roc_auc_score = lambda y_true, y_pred: 0.85
    cuml_mock.metrics.mean_squared_error = lambda y_true, y_pred, squared: 0.5
    
    # Mock train_test_split to just split the input data naively
    def fake_split(*arrays, **kwargs):
        # Returns (X_train, X_test, y_train, y_test)
        # We just return the input array duplicated for both train/test
        return arrays[0], arrays[0], arrays[1], arrays[1]
    
    cuml_mock.model_selection.train_test_split = fake_split
    
    sys.modules["cuml"] = cuml_mock
    sys.modules["cuml.metrics"] = cuml_mock.metrics
    sys.modules["cuml.model_selection"] = cuml_mock.model_selection

    # 3. Mock XGBoost
    # We use a MagicMock to avoid strict GPU param validation
    xgb_mock = MagicMock()
    xgb_mock.DMatrix = MagicMock()
    
    # Mock the Booster object returned by train
    booster_mock = MagicMock()
    # Mock predict to return a numpy array of correct length
    booster_mock.predict.side_effect = lambda data: np.zeros(2) 
    xgb_mock.train.return_value = booster_mock
    
    sys.modules["xgboost"] = xgb_mock

    # 4. Mock WandB
    wandb_mock = MagicMock()
    wandb_mock.integration.xgboost.WandbCallback = MagicMock
    sys.modules["wandb"] = wandb_mock
    sys.modules["wandb.integration.xgboost"] = wandb_mock.integration.xgboost

    yield

# --- Tests ---

def test_preprocess_logic():
    """
    Test that preprocess_data correctly splits features (X) and targets (y_ckd, y_gfr),
    and handles categorical encoding logic.
    """
    # Import inside test to rely on mocks
    from common import preprocess_data

    # Create a Pandas DataFrame (acting as cuDF)
    # Columns: 2 features + 2 targets
    df = pd.DataFrame({
        "Age": [25, 60],
        "Gender": ["Male", "Female"], # Object, should be encoded
        "CKD_Status": [0, 1],         # Target 1
        "GFR": [90.5, 45.2]           # Target 2
    })

    X, y_ckd, y_gfr = preprocess_data(df)

    # 1. Verify Targets are removed from X
    assert "CKD_Status" not in X.columns
    assert "GFR" not in X.columns
    
    # 2. Verify Features remain
    assert "Age" in X.columns
    assert "Gender" in X.columns
    
    # 3. Verify Target Extraction
    assert len(y_ckd) == 2
    assert y_ckd.iloc[1] == 1
    assert y_gfr.iloc[0] == 90.5
    
    # 4. Verify Categorical Encoding
    # 'Gender' was ["Male", "Female"], should now be numeric codes
    assert np.issubdtype(X["Gender"].dtype, np.number)
    
    # 5. Verify Float32 Cast
    assert X["Age"].dtype == "float32"

@patch("ingest.kagglehub.dataset_download")
@patch("ingest.shutil.copyfile")
def test_ingest_download_flow(mock_copy, mock_download, tmp_path):
    """Verify that download_data calls the correct APIs and moves files."""
    from ingest import download_data
    
    # Setup: Create a fake cache directory with a CSV
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "data.csv").touch()
    
    # Mock return of kagglehub to point to our fake cache
    mock_download.return_value = str(cache_dir)
    
    dest_dir = tmp_path / "raw"
    
    # Run
    download_data("dataset/name", dest_dir)
    
    # Assert
    mock_download.assert_called_once()
    mock_copy.assert_called() 
    # Check that copyfile was called with the correct source/dest logic
    # (Checking strictly args if needed, but existence is good for now)

@patch("train_gfr.wandb")
def test_train_gfr_execution(mock_wandb):
    """Verify train_gfr_model calls XGBoost training with correct params."""
    from train_gfr import train_gfr_model
    import xgboost as xgb
    
    # Setup dummy data (using pandas objects, handled by mock_gpu_environment)
    df = pd.DataFrame({"feat1": [1.0, 2.0], "feat2": [3.0, 4.0]})
    # Fix: Ensure types are float32 as expected by pipeline
    df = df.astype("float32")
    y = pd.Series([10.0, 20.0])
    
    # Config mock
    config = MagicMock()
    config.learning_rate = 0.01
    config.max_depth = 3
    config.subsample = 0.8

    # Run
    bst = train_gfr_model(df, y, df, y, config)
    
    # Assert
    xgb.train.assert_called_once()
    
    # Inspect arguments passed to xgboost.train
    call_args = xgb.train.call_args
    params = call_args[0][0] # First arg is params dict
    
    assert params["objective"] == "reg:squarederror"
    assert params["learning_rate"] == 0.01
    assert params["eval_metric"] == ["rmse"]
    
    # Verify return value
    assert bst is not None
