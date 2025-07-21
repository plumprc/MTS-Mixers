import pytest
import tempfile
import shutil
import os
from pathlib import Path
import pandas as pd
import numpy as np
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_data():
    """Generate sample time series data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='H')
    data = pd.DataFrame({
        'date': dates,
        'value1': np.random.randn(100).cumsum(),
        'value2': np.random.randn(100).cumsum() * 2,
        'value3': np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100) * 0.1
    })
    return data


@pytest.fixture
def sample_tensor():
    """Generate sample tensor data for model testing."""
    if not HAS_TORCH:
        pytest.skip("torch not available")
    torch.manual_seed(42)
    batch_size = 32
    seq_len = 96
    n_features = 7
    return torch.randn(batch_size, seq_len, n_features)


@pytest.fixture
def mock_config():
    """Provide a mock configuration object for testing."""
    class MockConfig:
        def __init__(self):
            self.seq_len = 96
            self.pred_len = 24
            self.d_model = 512
            self.n_heads = 8
            self.e_layers = 2
            self.d_ff = 2048
            self.dropout = 0.1
            self.activation = 'gelu'
            self.batch_size = 32
            self.learning_rate = 0.001
            self.device = 'cpu'
            self.use_gpu = False
            self.num_workers = 0
            
    return MockConfig()


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for data loader testing."""
    class MockDataset:
        def __init__(self, size=1000):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, index):
            np.random.seed(index)
            seq_x = np.random.randn(96, 7).astype(np.float32)
            seq_y = np.random.randn(24, 7).astype(np.float32)
            seq_x_mark = np.random.randn(96, 4).astype(np.float32)
            seq_y_mark = np.random.randn(24, 4).astype(np.float32)
            return seq_x, seq_y, seq_x_mark, seq_y_mark
            
    return MockDataset()


@pytest.fixture
def sample_csv_file(temp_dir, sample_data):
    """Create a sample CSV file for testing data loading."""
    csv_path = temp_dir / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    if HAS_TORCH:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)


@pytest.fixture
def device():
    """Provide the appropriate device for testing."""
    if not HAS_TORCH:
        pytest.skip("torch not available")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_model_config():
    """Provide a small model configuration for faster testing."""
    class SmallConfig:
        def __init__(self):
            self.seq_len = 24
            self.pred_len = 12
            self.d_model = 64
            self.n_heads = 4
            self.e_layers = 1
            self.d_ff = 128
            self.dropout = 0.1
            self.activation = 'relu'
            self.enc_in = 7
            self.c_out = 7
            self.factor = 1
            self.output_attention = False
            
    return SmallConfig()


@pytest.fixture
def mock_experiment_args():
    """Provide mock arguments for experiment testing."""
    class Args:
        def __init__(self):
            self.is_training = 1
            self.model_id = 'test_model'
            self.model = 'Transformer'
            self.data = 'custom'
            self.root_path = './data/'
            self.data_path = 'test.csv'
            self.checkpoints = './checkpoints/'
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 24
            self.enc_in = 7
            self.dec_in = 7
            self.c_out = 7
            self.d_model = 512
            self.n_heads = 8
            self.e_layers = 2
            self.d_layers = 1
            self.d_ff = 2048
            self.factor = 3
            self.distil = True
            self.dropout = 0.05
            self.embed = 'timeF'
            self.freq = 'h'
            self.activation = 'gelu'
            self.output_attention = False
            self.num_workers = 0
            self.itr = 1
            self.train_epochs = 1
            self.batch_size = 32
            self.patience = 3
            self.learning_rate = 0.0001
            self.des = 'test'
            self.loss = 'mse'
            self.lradj = 'type1'
            self.use_amp = False
            self.use_gpu = False
            self.gpu = 0
            self.devices = '0'
            
    return Args()