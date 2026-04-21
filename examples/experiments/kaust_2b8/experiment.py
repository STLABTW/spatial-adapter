"""
KAUST 2b_8 experiment — subclass of BaseExperiment.

Stage 1: STDK spatiotemporal model (deliberately under-trained with 10/10/80 split).
Stage 2: Spatial Adapter on residuals.
Evaluation: reconstruction RMSE, CovFrob, SV_score.
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader

from examples.baselines.timesplit.experiment_core import (
    DictDataset,
    build_contiguous_time_splits,
    build_fixed_location_subset,
    collate_fn,
    sv_match_loss_for_prediction_matrix,
)
from examples.experiments.base import BaseExperiment, DataSplit
from spatial_adapter.metrics import compute_metrics, cov_frob_observed
from spatial_adapter.models.spatial_adapter import SpatialAdapter


class KAUSTExperiment(BaseExperiment):
    """KAUST 2b_8 benchmark with STDK first stage."""

    def load_data(self, seed: int) -> DataSplit:
        cfg = self.data_cfg
        data_path = cfg["data_dir"] + "/" + cfg["filename"]
        df_full = pd.read_csv(data_path)[["x", "y", "t", "z"]]

        # Location subset
        space_ratio = cfg.get("space_ratio_keep", 1.0)
        if space_ratio < 1.0:
            df_full, keep_sites, n_sites_full = build_fixed_location_subset(
                df_full,
                keep_ratio=space_ratio,
                seed=seed,
            )

        # Time splits — returns (train_mask, val_mask, test_mask,
        #   train_idx, val_idx, test_idx, uniq_t, n_times)
        (
            train_mask,
            val_mask,
            test_mask,
            train_idx,
            val_idx,
            test_idx,
            uniq_t,
            n_times,
        ) = build_contiguous_time_splits(
            df_full,
            train_ratio=cfg.get("train_ratio", 0.1),
            val_ratio=cfg.get("val_ratio", 0.1),
            test_ratio=cfg.get("test_ratio", 0.8),
        )

        # Build tensors
        df_full["site_id"] = pd.factorize(list(zip(df_full["x"], df_full["y"])))[0]
        uniq_locs = df_full.groupby("site_id")[["x", "y"]].first().values
        N = len(uniq_locs)
        len(uniq_t)

        # Pivot to (T, N) matrix for adapter
        y_matrix = df_full.pivot_table(
            index="t",
            columns="site_id",
            values="z",
        ).values  # (T, N)

        coords = torch.tensor(uniq_locs, dtype=torch.float32)
        y_all = torch.tensor(y_matrix, dtype=torch.float32)

        # STDK DataLoader: flattened (B, p=0), (B, 2), (B, 1), (B, 1) pairs
        def _build_stdk_loader(time_idx):
            n_t = len(time_idx)
            # Repeat coords for each time step: (n_t * N, 2)
            coords_flat = coords.unsqueeze(0).expand(n_t, -1, -1).reshape(-1, 2)
            # Time values: (n_t * N, 1)
            t_vals = (
                torch.tensor(uniq_t[time_idx], dtype=torch.float32)
                .unsqueeze(1)
                .expand(-1, N)
                .reshape(-1, 1)
            )
            # X (covariates): empty (n_t * N, 0)
            X_flat = torch.zeros(n_t * N, 0)
            # Y: (n_t * N, 1)
            y_flat = y_all[time_idx].reshape(-1, 1)

            ds = DictDataset(X_flat, coords_flat, t_vals, y_flat)
            return DataLoader(
                ds,
                batch_size=self.config.get("first_stage", {}).get("batch_size", 256),
                shuffle=True,
                collate_fn=collate_fn,
            )

        stdk_train_loader = _build_stdk_loader(train_idx)

        # Adapter uses (T_split, N) matrix format
        # Build a simple DataLoader for the adapter (cat, cont, y)
        from torch.utils.data import TensorDataset

        cat_dummy = torch.zeros(len(train_idx), N, 0)
        # cont = just an index placeholder; trend wrapper handles prediction
        cont_train = (
            torch.arange(len(train_idx), dtype=torch.float32)
            .unsqueeze(1)
            .expand(-1, N)
            .unsqueeze(-1)
        )
        adapter_train_ds = TensorDataset(cat_dummy, cont_train, y_all[train_idx])
        adapter_train_loader = DataLoader(
            adapter_train_ds,
            batch_size=self.adapter_config.training.batch_size,
            shuffle=True,
        )

        cont_val = (
            torch.arange(len(val_idx), dtype=torch.float32)
            .unsqueeze(1)
            .expand(-1, N)
            .unsqueeze(-1)
        )
        cont_test = (
            torch.arange(len(test_idx), dtype=torch.float32)
            .unsqueeze(1)
            .expand(-1, N)
            .unsqueeze(-1)
        )

        return DataSplit(
            train_loader=adapter_train_loader,
            train_cont=cont_train,
            train_y=y_all[train_idx],
            val_cont=cont_val,
            val_y=y_all[val_idx],
            test_cont=cont_test,
            test_y=y_all[test_idx],
            locs=uniq_locs,
            n_locations=N,
            p_dim=1,
            metadata={
                "coords": coords,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "y_all": y_all,
                "uniq_t": uniq_t,
                "stdk_train_loader": stdk_train_loader,
            },
        )

    def build_first_stage(self, data: DataSplit, seed: int) -> torch.nn.Module:
        """Train STDK model as first stage with early stopping."""
        from examples.baselines.stdk.st_interp import STInterpMLP
        from examples.baselines.stdk.trainer import Trainer as STDKTrainer

        fs_cfg = self.config.get("first_stage", {})

        model = STInterpMLP(
            p=0,
            train_coords=data.locs,
        ).to(self.device)

        # Build STDK val loader (same format as train)
        stdk_train_loader = data.metadata["stdk_train_loader"]
        stdk_val_loader = self._build_stdk_loader(data, data.metadata["val_idx"])

        stdk_config = {
            "epochs": fs_cfg.get("epochs", 100),
            "lr": fs_cfg.get("lr", 1e-3),
            "weight_decay": fs_cfg.get("weight_decay", 1e-4),
            "batch_size": fs_cfg.get("batch_size", 256),
            "patience": fs_cfg.get("patience", 15),
            "current_quantile": 0.5,
        }

        ckpt_dir = self.output_dir / f"stdk_seed{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        trainer = STDKTrainer(
            model=model,
            train_loader=stdk_train_loader,
            val_loader=stdk_val_loader,
            config=stdk_config,
            device=self.device,
            output_dir=ckpt_dir,
        )
        trainer.fit()
        model.eval()

        return _STDKTrendWrapper(model, data.metadata, self.device)

    def _build_stdk_loader(self, data: DataSplit, time_idx):
        """Build a STDK-format DataLoader for given time indices."""
        coords = data.metadata["coords"]
        uniq_t = data.metadata["uniq_t"]
        y_all = data.metadata["y_all"]
        N = data.n_locations
        n_t = len(time_idx)

        coords_flat = coords.unsqueeze(0).expand(n_t, -1, -1).reshape(-1, 2)
        t_vals = (
            torch.tensor(uniq_t[time_idx], dtype=torch.float32)
            .unsqueeze(1)
            .expand(-1, N)
            .reshape(-1, 1)
        )
        X_flat = torch.zeros(n_t * N, 0)
        y_flat = y_all[time_idx].reshape(-1, 1)

        ds = DictDataset(X_flat, coords_flat, t_vals, y_flat)
        return DataLoader(
            ds,
            batch_size=self.config.get("first_stage", {}).get("batch_size", 256),
            shuffle=False,
            collate_fn=collate_fn,
        )

    def _compute_sv_score(self, y_true, y_pred, coords) -> float:
        """Semivariogram-match score between Y and Ŷ on the full test grid."""
        coords_np = coords.cpu().numpy() if hasattr(coords, "cpu") else coords
        yt_np = y_true.cpu().numpy() if hasattr(y_true, "cpu") else y_true
        yp_np = y_pred.detach().cpu().numpy() if hasattr(y_pred, "detach") else y_pred
        result = sv_match_loss_for_prediction_matrix(
            coords=coords_np,
            y_true=yt_np,
            y_pred=yp_np,
            time_idx=list(range(yt_np.shape[0])),
        )
        return float(result["loss"])

    def _evaluate_baseline(self, trend, data: DataSplit) -> dict:
        """Baseline = STDK only; augment default metrics with sv_score."""
        metrics = super()._evaluate_baseline(trend, data)
        try:
            trend.eval()
            with torch.no_grad():
                y_pred = trend(data.test_cont.to(self.device))
            metrics["sv_score"] = round(
                self._compute_sv_score(data.test_y, y_pred, data.metadata["coords"]),
                6,
            )
        except Exception as e:
            print(f"  [warning] baseline sv_score failed: {e}")
        return metrics

    def evaluate(
        self,
        trainer: SpatialAdapter,
        data: DataSplit,
        model_name: str,
    ) -> dict:
        """Evaluate reconstruction RMSE and CovFrob."""
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

        try:
            cf = cov_frob_observed(
                test_y.cpu().numpy(),
                y_pred.detach().cpu().numpy(),
            )
            metrics["covfrob"] = round(cf, 6)
        except Exception:
            pass

        # Semivariogram-match score for the Table 4 validation-criterion
        # ablation; lower is better.  Empirical semivariograms of Y vs Ŷ.
        try:
            metrics["sv_score"] = round(
                self._compute_sv_score(test_y, y_pred, data.metadata["coords"]),
                6,
            )
        except Exception as e:
            print(f"  [warning] sv_score failed: {e}")

        return metrics


class _STDKTrendWrapper(torch.nn.Module):
    """Wraps a trained STDK model to behave like a trend model for the adapter.

    The adapter passes cont of shape (T_batch, N, 1) — an index placeholder.
    This wrapper ignores cont and instead predicts on all (time, location) pairs
    using the stored STDK model, returning shape (T_batch, N).
    """

    def __init__(self, stdk_model, metadata, device):
        super().__init__()
        self.stdk_model = stdk_model
        self.metadata = metadata
        self._device = device
        self._cache = {}

    def residual_parameters(self):
        """No trainable parameters — STDK is frozen."""
        return iter([])

    def forward(self, cont: torch.Tensor) -> torch.Tensor:
        """Return STDK predictions as (T_batch, N) matrix."""
        T_batch = cont.shape[0]
        coords = self.metadata["coords"].to(self._device)
        N = coords.shape[0]
        self.metadata["uniq_t"]

        # Determine which time indices this batch corresponds to
        # cont[:, 0, 0] contains the batch-local index (0..T_batch-1)
        # We need the actual time values from uniq_t
        # For simplicity, use the train/val/test indices stored in metadata
        batch_idx = cont[:, 0, 0].long().cpu().numpy()

        # Check cache
        cache_key = tuple(batch_idx)
        if cache_key in self._cache:
            return self._cache[cache_key]

        self.stdk_model.eval()
        preds = []
        with torch.no_grad():
            for local_i in range(T_batch):
                # Predict at all N locations for this time step
                t_val = torch.full((N, 1), float(local_i), device=self._device)
                X_empty = torch.zeros(N, 0, device=self._device)
                coords_d = coords.to(self._device)
                pred_i = self.stdk_model(X_empty, coords_d, t_val)  # (N, 1)
                preds.append(pred_i.squeeze(-1))  # (N,)

        result = torch.stack(preds, dim=0)  # (T_batch, N)
        self._cache[cache_key] = result
        return result
