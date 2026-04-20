"""
Pytest configuration and common fixtures for spatial_adapter tests.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from spatial_adapter.data import generate_combined_synthetic_data
from spatial_adapter.models.spatial_basis_learner import SpatialBasisLearner
from spatial_adapter.models.trend_model import TrendModel


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    locations = np.linspace(-3, 3, 10)

    cat_features, cont_features, targets = generate_combined_synthetic_data(
        location=locations,
        n_samples=100,
        noise_std=0.1,
        eigenvalue=4.0,
        global_mean=50.0,
        seed=42,
    )

    return {
        "cat_features": cat_features,
        "cont_features": cont_features,
        "targets": targets,
        "locations": locations,
    }


@pytest.fixture
def trend_model(sample_data):
    """Create a TrendModel instance for testing."""
    p_dim = sample_data["cont_features"].shape[-1]
    n_locations = sample_data["locations"].shape[0]

    model = TrendModel(
        num_continuous_features=p_dim,
        hidden_layer_sizes=[64, 32],
        n_locations=n_locations,
        init_weight=None,
        init_bias=None,
        freeze_init=False,
        dropout_rate=0.1,
    )

    return model


@pytest.fixture
def spatial_basis_learner(sample_data):
    """Create a SpatialBasisLearner instance for testing."""
    n_locations = sample_data["locations"].shape[0]
    latent_dim = 3

    model = SpatialBasisLearner(
        num_locations=n_locations,
        latent_dim=latent_dim,
        pca_init=None,
    )

    return model


@pytest.fixture
def sample_batch(sample_data):
    """Create a sample batch for testing."""
    batch_size = 4
    cont_features = torch.from_numpy(sample_data["cont_features"][:batch_size]).float()
    targets = torch.from_numpy(sample_data["targets"][:batch_size]).float()

    return {
        "cont_features": cont_features,
        "targets": targets,
    }


@pytest.fixture
def device():
    """Get the device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


@pytest.fixture
def sample_config():
    """Create a sample configuration dictionary for testing."""
    return {
        "seed": 42,
        "n_time_steps": 50,
        "n_locations": 10,
        "noise_std": 0.1,
        "eigenvalue": 4.0,
        "batch_size": 16,
        "lr_mu": 1e-2,
        "max_iters": 100,
        "rho": 1,
        "latent_dim": 2,
        "ckpt_dir": "test_ckpts",
        "phi_every": 5,
        "phi_freeze": 50,
        "min_outer": 10,
        "hidden_layer_sizes": [32, 16],
        "w_ols": torch.randn(5),
        "b_ols": 1.0,
        "dropout_rate": 0.1,
    }


@pytest.fixture
def sample_train_loader(sample_data):
    """Create a sample DataLoader for testing."""
    return DataLoader(sample_data["train"], batch_size=8, shuffle=True)


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield
    # Clear any cached tensors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Markers for different test types
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


# Skip GPU tests if CUDA is not available
def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if CUDA is not available."""
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
