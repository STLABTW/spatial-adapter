"""Unit tests for spatial_adapter.tuning.model_cache.ModelCache."""

import torch
import torch.nn as nn

from spatial_adapter.tuning import ModelCache


class TestModelCache:
    def test_empty_on_init(self):
        cache = ModelCache()
        assert cache.size() == 0
        assert cache.keys() == []

    def test_store_and_size(self):
        cache = ModelCache()
        cache.store(1.0, 0.1, {"w": 1}, {"b": 2})
        assert cache.size() == 1
        cache.store(2.0, 0.2, {"w": 3}, {"b": 4})
        assert cache.size() == 2

    def test_keys(self):
        cache = ModelCache()
        cache.store(1.0, 0.5, {}, {})
        cache.store(2.0, 0.5, {}, {})
        assert set(cache.keys()) == {(1.0, 0.5), (2.0, 0.5)}

    def test_clear(self):
        cache = ModelCache()
        cache.store(1.0, 1.0, {}, {})
        cache.clear()
        assert cache.size() == 0

    def test_load_nearest_empty_cache_noop(self):
        cache = ModelCache()
        trend = nn.Linear(3, 1)
        basis = nn.Linear(5, 2)
        old_w = trend.weight.clone()
        cache.load_nearest(trend, basis, 1.0, 1.0)
        assert torch.equal(trend.weight, old_w)  # unchanged

    def test_load_nearest_single_entry(self):
        import copy

        cache = ModelCache()
        trend = nn.Linear(3, 1)
        basis = nn.Linear(5, 2)
        # deepcopy: state_dict() shares storage with params, so in-place
        # modifications would corrupt the cache without a copy.
        cache.store(
            1.0,
            1.0,
            copy.deepcopy(trend.state_dict()),
            copy.deepcopy(basis.state_dict()),
        )

        with torch.no_grad():
            trend.weight.fill_(999.0)
            basis.weight.fill_(999.0)

        cache.load_nearest(trend, basis, 1.0, 1.0)
        assert trend.weight.abs().max().item() < 10  # restored from cache, not 999

    def test_load_nearest_picks_closest(self):
        import copy

        cache = ModelCache()
        t1 = nn.Linear(2, 1)
        b1 = nn.Linear(3, 1)

        with torch.no_grad():
            t1.weight.fill_(1.0)
        cache.store(
            1.0, 1.0, copy.deepcopy(t1.state_dict()), copy.deepcopy(b1.state_dict())
        )
        with torch.no_grad():
            t1.weight.fill_(100.0)
        cache.store(
            100.0, 100.0, copy.deepcopy(t1.state_dict()), copy.deepcopy(b1.state_dict())
        )

        with torch.no_grad():
            t1.weight.fill_(0.0)
        cache.load_nearest(t1, b1, 1.1, 0.9)
        assert t1.weight.mean().item() < 50  # closer to 1.0 entry

    def test_log_space_distance(self):
        """Entries differing by a factor of 10 should be "closer" in
        log-space than entries differing by an additive 10."""
        cache = ModelCache()
        t = nn.Linear(1, 1)
        b = nn.Linear(1, 1)
        cache.store(1.0, 1.0, t.state_dict(), b.state_dict())
        cache.store(1000.0, 1000.0, t.state_dict(), b.state_dict())

        # Query tau=(10, 10) — log-closer to (1,1) than to (1000,1000)
        with torch.no_grad():
            t.weight.fill_(0.0)
        cache.load_nearest(t, b, 10.0, 10.0)
        # Should load (1,1) entry (log10(10)-log10(1))^2=1 vs (log10(10)-log10(1000))^2=4
