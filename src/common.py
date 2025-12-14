import sys
from pathlib import Path
from typing import Tuple
import cudf
import xgboost as xgb
from dotenv import load_dotenv

# Load secrets once here or in main scripts
load_dotenv()

PROCESSED_DATA_DIR = Path("data/processed")

def validate_os() -> None:
    """Ensure script is running in a supported environment (WSL2/Linux)."""
    if sys.platform.startswith("win"):
        print("\n CRITICAL ERROR: RAPIDS (cuDF) is NOT supported on native Windows.")
        print(" You MUST run this script inside WSL2 (Windows Subsystem for Linux).")
        print(" See WSL_SETUP.md for instructions.\n")
        sys.exit(1)

def load_data(data_dir: Path) -> cudf.DataFrame:
    """Load the first Parquet file found in the data directory."""
    print("Loading data...")
    if not data_dir.exists():
         raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    files = list(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet data found in {data_dir}")
        
    data_path = files[0]
    return cudf.read_parquet(data_path)

def preprocess_data(df: cudf.DataFrame) -> Tuple[cudf.DataFrame, cudf.Series, cudf.Series]:
    """Split dataframe into features and targets, handling categorical encoding."""
    # Simple label encoding for strings
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes
        
    target_ckd = "CKD_Status"
    target_gfr = "GFR"
    
    X = df.drop(columns=[target_ckd, target_gfr])
    y_ckd = df[target_ckd]
    y_gfr = df[target_gfr]
    
    # Cast to float32 for XGBoost GPU efficiency
    X = X.astype("float32")
    
    return X, y_ckd, y_gfr
