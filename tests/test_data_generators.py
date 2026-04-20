"""
Unit tests for spatial_adapter.data.generators module.
"""

import numpy as np

from spatial_adapter.data.generators import (
    generate_combined_synthetic_data,
    generate_time_synthetic_data,
)


class TestGenerateCombinedSyntheticData:
    """Test generate_combined_synthetic_data function."""

    def test_basic_generation(self):
        """Test basic data generation."""
        location = np.linspace(-3, 3, 10)
        n_samples = 100
        noise_std = 0.1
        eigenvalue = 4.0
        seed = 42

        cat_feat, cont_feat, targets = generate_combined_synthetic_data(
            location, n_samples, noise_std, eigenvalue, seed=seed
        )

        # Check shapes
        assert cat_feat.shape == (
            n_samples,
            len(location),
            0,
        )  # No categorical features
        assert cont_feat.shape == (n_samples, len(location), 3)  # 3 continuous features
        assert targets.shape == (n_samples, len(location))

        # Check data types
        assert cat_feat.dtype == np.int64
        assert cont_feat.dtype == np.float64
        assert targets.dtype == np.float64

        # Check categorical features are all zeros
        assert np.all(cat_feat == 0)

        # Check continuous features have reasonable values
        assert np.all(np.isfinite(cont_feat))
        assert np.all(np.isfinite(targets))

    def test_deterministic_generation(self):
        """Test that generation is deterministic with same seed."""
        location = np.linspace(-3, 3, 5)
        seed = 42

        cat1, cont1, targets1 = generate_combined_synthetic_data(
            location, n_samples=10, noise_std=1.0, eigenvalue=0.5, seed=seed
        )
        cat2, cont2, targets2 = generate_combined_synthetic_data(
            location, n_samples=10, noise_std=1.0, eigenvalue=0.5, seed=seed
        )

        np.testing.assert_array_equal(cat1, cat2)
        np.testing.assert_array_equal(cont1, cont2)
        np.testing.assert_array_equal(targets1, targets2)

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        location = np.linspace(-3, 3, 5)

        cat1, cont1, targets1 = generate_combined_synthetic_data(
            location, n_samples=10, noise_std=1.0, eigenvalue=0.5, seed=42
        )
        cat2, cont2, targets2 = generate_combined_synthetic_data(
            location, n_samples=10, noise_std=1.0, eigenvalue=0.5, seed=43
        )

        # Should be different (very unlikely to be identical)
        assert not np.array_equal(targets1, targets2)


class TestGenerateTimeSyntheticData:
    """Test generate_time_synthetic_data function."""

    def test_basic_generation(self):
        """Test basic temporal data generation."""
        locs = np.linspace(-3, 3, 10)
        n_time_steps = 100
        noise_std = 0.1
        eigenvalue = 4.0
        seed = 42

        cat, cont, y = generate_time_synthetic_data(
            locs, n_time_steps, noise_std, eigenvalue, seed=seed
        )

        # Check shapes
        assert cat.shape == (n_time_steps, len(locs), 0)  # No categorical features
        assert cont.shape == (n_time_steps, len(locs), 5)  # temp/wind/hum/pres/poll
        assert y.shape == (n_time_steps, len(locs))

        # Check data types
        assert cat.dtype == np.int64
        assert cont.dtype == np.float64
        assert y.dtype == np.float64

        # Check categorical features are all zeros
        assert np.all(cat == 0)

        # Check continuous features have reasonable values
        assert np.all(np.isfinite(cont))
        assert np.all(np.isfinite(y))

    def test_deterministic_generation(self):
        """Test that generation is deterministic with same seed."""
        locs = np.linspace(-3, 3, 5)
        seed = 42

        cat1, cont1, y1 = generate_time_synthetic_data(
            locs, n_time_steps=10, noise_std=1.0, eigenvalue=0.5, seed=seed
        )
        cat2, cont2, y2 = generate_time_synthetic_data(
            locs, n_time_steps=10, noise_std=1.0, eigenvalue=0.5, seed=seed
        )

        np.testing.assert_array_equal(cat1, cat2)
        np.testing.assert_array_equal(cont1, cont2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        locs = np.linspace(-3, 3, 5)

        cat1, cont1, y1 = generate_time_synthetic_data(
            locs, n_time_steps=10, noise_std=1.0, eigenvalue=0.5, seed=42
        )
        cat2, cont2, y2 = generate_time_synthetic_data(
            locs, n_time_steps=10, noise_std=1.0, eigenvalue=0.5, seed=43
        )

        # Should be different (very unlikely to be identical)
        assert not np.array_equal(y1, y2)
