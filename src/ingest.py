import os
import glob
import shutil
import sys
from pathlib import Path
from dotenv import load_dotenv
import cudf
import kagglehub

# Load secrets first
load_dotenv()

# Constants
DATASET_NAME = "miadul/kidney-function-health-dataset"
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def validate_os() -> None:
    """Ensure script is running in a supported environment (WSL2/Linux)."""
    if sys.platform.startswith("win"):
        print("\n🛑 CRITICAL ERROR: RAPIDS (cuDF) is NOT supported on native Windows.")
        print("👉 You MUST run this script inside WSL2 (Windows Subsystem for Linux).")
        print("   See WSL_SETUP.md for instructions.\n")
        sys.exit(1)

def ensure_directories(raw_path: Path, processed_path: Path) -> None:
    """Create necessary data directories."""
    raw_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

def download_data(dataset_name: str, destination: Path) -> Path:
    """Download dataset using kagglehub and move to destination."""
    print(f"Downloading {dataset_name}...")
    try:
        # Download to cache
        cache_path = kagglehub.dataset_download(dataset_name)
        print(f"Dataset downloaded to cache at: {cache_path}")
        
        # Move files
        copied_files = []
        # Ensure destination exists
        destination.mkdir(parents=True, exist_ok=True)
        
        for file_path in glob.glob(os.path.join(cache_path, "*.csv")):
            file_name = Path(file_path).name
            dest_file = destination / file_name
            
            # Remove if exists to avoid 'Operation not permitted' on overwrite sometimes
            if dest_file.exists():
                try:
                    os.remove(dest_file)
                except OSError:
                    pass

            print(f"Copying {file_name} -> {dest_file}...")
            # Use copyfile to avoid metadata/permission preservation issues on /mnt/c
            shutil.copyfile(file_path, dest_file)
            copied_files.append(file_name)
            
        print(f"Files copied to {destination}: {copied_files}")
        return Path(cache_path)

    except Exception as e:
        print(f"Failed to download data: {e}")
        # In a real app, we might raise e or return None, but for script simplicity we exit
        sys.exit(1)

def convert_to_parquet(raw_path: Path, processed_path: Path) -> None:
    """Convert all CSVs in raw_path to Parquet in processed_path using cuDF."""
    csv_files = list(raw_path.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {raw_path}!")
        return

    for csv_file in csv_files:
        print(f"Processing {csv_file.name} with cuDF...")
        try:
            df = cudf.read_csv(csv_file)
            parquet_path = processed_path / csv_file.with_suffix(".parquet").name
            
            # Write to parquet
            df.to_parquet(parquet_path)
            print(f"Saved to {parquet_path}")
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")

def main() -> None:
    validate_os()
    
    # Check Credentials
    if not os.getenv("KAGGLE_USERNAME"):
         print("WARNING: KAGGLE_USERNAME is missing from .env")

    ensure_directories(RAW_DIR, PROCESSED_DIR)
    download_data(DATASET_NAME, RAW_DIR)
    convert_to_parquet(RAW_DIR, PROCESSED_DIR)

if __name__ == "__main__":
    main()
