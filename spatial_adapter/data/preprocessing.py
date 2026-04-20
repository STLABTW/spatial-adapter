"""Data preprocessing: scaling, train/val/test splits, and tensor conversion."""

from typing import Any, Dict, Tuple

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataPreprocessor:
    """Feature and target scaler with inverse-transform support."""

    def __init__(
        self,
        feature_scaler_type: str = "standard",
        target_scaler_type: str = "standard",
        fit_on_train_only: bool = True,
    ):
        self.feature_scaler_type = feature_scaler_type
        self.target_scaler_type = target_scaler_type
        self.fit_on_train_only = fit_on_train_only

        self.feature_scaler = self._make_scaler(feature_scaler_type)
        self.target_scaler = self._make_scaler(target_scaler_type)
        self.is_fitted = False
        self.scaler_stats: Dict[str, Any] = {}

    @staticmethod
    def _make_scaler(name: str):
        if name == "standard":
            return StandardScaler()
        if name == "minmax":
            return MinMaxScaler()
        raise ValueError(f"Unknown scaler type: {name}")

    def fit_scalers(self, cont_features: np.ndarray, targets: np.ndarray) -> None:
        """Fit on (T, N, F) features and (T, N) targets."""
        T, N, F = cont_features.shape
        self.feature_scaler.fit(cont_features.reshape(-1, F))
        self.target_scaler.fit(targets.reshape(-1, 1))
        self.scaler_stats = {
            k: getattr(self.feature_scaler, k, None)
            for k in ("mean_", "scale_", "min_", "max_")
        }
        self.scaler_stats.update(
            {
                f"target_{k}": getattr(self.target_scaler, k, None)
                for k in ("mean_", "scale_", "min_", "max_")
            }
        )
        self.is_fitted = True

    def transform_data(
        self,
        cont_features: np.ndarray,
        targets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale (T, N, F) features and (T, N) targets."""
        if not self.is_fitted:
            raise ValueError("Scalers must be fitted before transforming")
        T, N, F = cont_features.shape
        cont_scaled = self.feature_scaler.transform(
            cont_features.reshape(-1, F)
        ).reshape(T, N, F)
        target_scaled = self.target_scaler.transform(targets.reshape(-1, 1)).reshape(
            T, N
        )
        return cont_scaled, target_scaled

    def inverse_transform_targets(self, targets_scaled: np.ndarray) -> np.ndarray:
        """Map scaled targets back to original scale (preserves shape)."""
        if not self.is_fitted:
            raise ValueError("Scalers must be fitted before inverse transforming")
        shape = targets_scaled.shape
        return self.target_scaler.inverse_transform(
            targets_scaled.reshape(-1, 1)
        ).reshape(shape)

    def get_scaler_info(self) -> Dict[str, Any]:
        if not self.is_fitted:
            return {"error": "Scalers not fitted yet"}
        return self.scaler_stats


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def _split_indices(T: int, train_ratio: float, val_ratio: float):
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))
    return train_end, val_end


def _to_dataset(cat, cont, targets):
    return torch.utils.data.TensorDataset(
        torch.from_numpy(cat).long(),
        torch.from_numpy(cont).float(),
        torch.from_numpy(targets).float(),
    )


def prepare_all(
    cat_features: np.ndarray,
    cont_features: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[
    torch.utils.data.TensorDataset,
    torch.utils.data.TensorDataset,
    torch.utils.data.TensorDataset,
]:
    """Split into train/val/test TensorDatasets (no scaling)."""
    T = cat_features.shape[0]
    t1, t2 = _split_indices(T, train_ratio, val_ratio)
    return (
        _to_dataset(cat_features[:t1], cont_features[:t1], targets[:t1]),
        _to_dataset(cat_features[t1:t2], cont_features[t1:t2], targets[t1:t2]),
        _to_dataset(cat_features[t2:], cont_features[t2:], targets[t2:]),
    )


def prepare_all_with_scaling(
    cat_features: np.ndarray,
    cont_features: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    feature_scaler_type: str = "standard",
    target_scaler_type: str = "standard",
    fit_on_train_only: bool = True,
) -> Tuple[
    torch.utils.data.TensorDataset,
    torch.utils.data.TensorDataset,
    torch.utils.data.TensorDataset,
    DataPreprocessor,
]:
    """Split + scale.  Returns (train_ds, val_ds, test_ds, preprocessor)."""
    T = cat_features.shape[0]
    t1, t2 = _split_indices(T, train_ratio, val_ratio)

    preprocessor = DataPreprocessor(
        feature_scaler_type=feature_scaler_type,
        target_scaler_type=target_scaler_type,
        fit_on_train_only=fit_on_train_only,
    )
    preprocessor.fit_scalers(cont_features[:t1], targets[:t1])

    train_c, train_t = preprocessor.transform_data(cont_features[:t1], targets[:t1])
    val_c, val_t = preprocessor.transform_data(cont_features[t1:t2], targets[t1:t2])
    test_c, test_t = preprocessor.transform_data(cont_features[t2:], targets[t2:])

    return (
        _to_dataset(cat_features[:t1], train_c, train_t),
        _to_dataset(cat_features[t1:t2], val_c, val_t),
        _to_dataset(cat_features[t2:], test_c, test_t),
        preprocessor,
    )


def denormalize_predictions(
    predictions_scaled: np.ndarray,
    preprocessor: DataPreprocessor,
) -> np.ndarray:
    """Inverse-scale predictions back to original units."""
    return preprocessor.inverse_transform_targets(predictions_scaled)
