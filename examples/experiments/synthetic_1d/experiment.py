"""
Synthetic 1D experiment — subclass of BaseExperiment.

Stage 1: OLS linear trend (frozen).
Stage 2: Spatial Adapter on residuals.
Evaluation: reconstruction RMSE, MAE, R², CovFrob against known ground truth.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from examples.experiments.base import BaseExperiment, DataSplit
from spatial_adapter.data.generators import generate_time_synthetic_data
from spatial_adapter.data.preprocessing import prepare_all_with_scaling
from spatial_adapter.metrics import compute_metrics, cov_frob_observed
from spatial_adapter.models.spatial_adapter import SpatialAdapter
from spatial_adapter.models.trend_model import TrendModel
from spatial_adapter.utils.experiment_helpers import compute_ols_coefficients


class Synthetic1DExperiment(BaseExperiment):
    """Synthetic 1D benchmark with OLS first stage."""

    def load_data(self, seed: int) -> DataSplit:
        cfg = self.data_cfg
        locs = np.linspace(-3, 3, cfg["n_locations"])

        cat_features, cont_features, targets = generate_time_synthetic_data(
            locs=locs,
            n_time_steps=cfg["n_time_steps"],
            noise_std=cfg["noise_std"],
            eigenvalue=cfg["eigenvalue"],
            eta_rho=cfg.get("eta_rho", 0.8),
            f_rho=cfg.get("f_rho", 0.6),
            global_mean=cfg.get("global_mean", 50.0),
            feature_noise_std=cfg.get("feature_noise_std", 0.1),
            seed=seed,
        )

        train_ds, val_ds, test_ds, preprocessor = prepare_all_with_scaling(
            cat_features=cat_features,
            cont_features=cont_features,
            targets=targets,
            train_ratio=cfg.get("train_ratio", 0.7),
            val_ratio=cfg.get("val_ratio", 0.15),
            feature_scaler_type="standard",
            target_scaler_type="standard",
            fit_on_train_only=True,
        )

        batch_size = self.adapter_config.training.batch_size
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        _, train_X, train_y = train_ds.tensors
        _, val_X, val_y = val_ds.tensors
        _, test_X, test_y = test_ds.tensors

        return DataSplit(
            train_loader=train_loader,
            train_cont=train_X,
            train_y=train_y,
            val_cont=val_X,
            val_y=val_y,
            test_cont=test_X,
            test_y=test_y,
            locs=locs,
            n_locations=cfg["n_locations"],
            p_dim=train_X.shape[-1],
            metadata={
                "preprocessor": preprocessor,
                "eigenvalue": cfg["eigenvalue"],
            },
        )

    def build_first_stage(self, data: DataSplit, seed: int) -> torch.nn.Module:
        """Build OLS trend model (frozen linear layer)."""
        w_ols, b_ols = compute_ols_coefficients(
            data.train_cont,
            data.train_y,
            device=self.device,
        )
        trend = TrendModel(
            num_continuous_features=data.p_dim,
            hidden_layer_sizes=[],
            n_locations=data.n_locations,
            init_weight=w_ols,
            init_bias=b_ols,
            freeze_init=True,
            dropout_rate=0.0,
        ).to(self.device)
        return trend

    def evaluate(
        self,
        trainer: SpatialAdapter,
        data: DataSplit,
        model_name: str,
    ) -> dict:
        """Evaluate reconstruction quality + covariance recovery."""
        trainer.trend.eval()
        trainer.basis.eval()

        with torch.no_grad():
            test_cont = data.test_cont.to(self.device)
            test_y = data.test_y.to(self.device)

            mu = trainer.trend(test_cont)
            R = test_y - mu
            spatial = (R @ trainer.basis.basis) @ trainer.basis.basis.T
            y_pred = mu + spatial

        rmse, mae, r2 = compute_metrics(test_y, y_pred)
        metrics = {
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "r2": round(r2, 6),
        }

        # CovFrob against ground truth
        try:
            cf = cov_frob_observed(
                test_y.cpu().numpy(),
                y_pred.detach().cpu().numpy(),
            )
            metrics["covfrob"] = round(cf, 6)
        except Exception:
            pass

        return metrics
