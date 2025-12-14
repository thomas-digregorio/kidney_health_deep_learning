import pytest
import sys
from pathlib import Path

# Add src to path so we can import modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def mock_parquet_data(tmp_path):
    """Create a mock parquet file for testing."""
    # We need cudf or pandas. Since environment has cudf, try to use it.
    try:
        import cudf
        import numpy as np
        
        df = cudf.DataFrame({
            "Age": np.random.randint(20, 80, 100),
            "Gender": np.random.choice(["Male", "Female"], 100),
            "GFR": np.random.uniform(15, 120, 100).astype("float32"),
            "CKD_Status": np.random.randint(0, 2, 100),
        })
        
        path = tmp_path / "dataset.parquet"
        df.to_parquet(path)
        return path
    except ImportError:
        pytest.skip("cuDF not installed or not finding GPU, skipping GPU tests")
