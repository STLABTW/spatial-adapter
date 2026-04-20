"""
Weather2K experiments — subclasses of BaseExperiment.

Two variants:
  - Weather2KTimesplitExperiment: time-split reconstruction (80/10/10 and 10/10/80)
  - Weather2KHoldoutExperiment: spatial holdout prediction at unseen stations

Stage 1: STDK spatiotemporal model.
Stage 2: Spatial Adapter on residuals.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from examples.baselines.timesplit.experiment_core import (
    DictDataset,
    build_contiguous_time_splits,
    build_fixed_location_subset,
    collate_fn,
    load_weather2k_as_long_df,
)
from examples.experiments.base import BaseExperiment, DataSplit
from spatial_adapter.metrics import compute_metrics, cov_frob_observed
from spatial_adapter.models.spatial_adapter import SpatialNeuralAdapter


class Weather2KTimesplitExperiment(BaseExperiment):
    """Weather2K time-split benchmark with STDK first stage."""

    def load_data(self, seed: int) -> DataSplit:
        cfg = self.data_cfg

        df_full = load_weather2k_as_long_df(
            npy_path=cfg["npy_path"],
            target_var_idx=cfg.get("target_var_idx", 0),
            lat_idx=cfg.get("lat_idx", 0),
            lon_idx=cfg.get("lon_idx", 1),
            t_keep=cfg.get("t_keep", 1000),
        )

        # Location subset
        df_full, keep_sites, n_sites_full = build_fixed_location_subset(
            df_full,
            keep_ratio=cfg.get("space_ratio_keep", 0.1),
            seed=seed,
        )

        # Time splits — returns (train_mask, val_mask, test_mask,
        #   train_idx, val_idx, test_idx, uniq_t, n_times)
        (
            _,
            _,
            _,
            train_idx,
            val_idx,
            test_idx,
            uniq_t,
            n_times,
        ) = build_contiguous_time_splits(
            df_full,
            train_ratio=cfg.get("train_ratio", 0.8),
            val_ratio=cfg.get("val_ratio", 0.1),
            test_ratio=cfg.get("test_ratio", 0.1),
        )

        # Build tensors
        uniq_locs = df_full[["x", "y"]].drop_duplicates().sort_values(["x", "y"]).values
        N = len(uniq_locs)
        uniq_t = np.sort(df_full["t"].unique())
        T = len(uniq_t)

        df_full["site_id"] = pd.factorize(list(zip(df_full["x"], df_full["y"])))[0]
        y_matrix = df_full.pivot_table(
            index="t",
            columns="site_id",
            values="z",
        ).values

        coords = torch.tensor(uniq_locs, dtype=torch.float32)
        t_all = torch.arange(T, dtype=torch.float32).unsqueeze(1)
        t_all.expand(T, N).unsqueeze(-1)
        y_all = torch.tensor(y_matrix, dtype=torch.float32)

        # Adapter train loader: (T, N) matrix format
        from torch.utils.data import TensorDataset

        cat_dummy = torch.zeros(len(train_idx), N, 0)
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

        # STDK train loader: flattened (B, 2), (B, 1) pair format
        def _build_stdk_loader(time_idx, shuffle=True):
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
                batch_size=self.config.get("first_stage", {}).get("batch_size", 512),
                shuffle=shuffle,
                collate_fn=collate_fn,
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
                "stdk_train_loader": _build_stdk_loader(train_idx),
                "stdk_val_loader": _build_stdk_loader(val_idx, shuffle=False),
            },
        )

    def build_first_stage(self, data: DataSplit, seed: int) -> torch.nn.Module:
        """Train STDK model as first stage with early stopping."""
        from examples.baselines.stdk.st_interp import STInterpMLP
        from examples.baselines.stdk.trainer import Trainer as STDKTrainer
        from examples.experiments.kaust_2b8.experiment import _STDKTrendWrapper

        fs_cfg = self.config.get("first_stage", {})

        model = STInterpMLP(
            p=0,
            train_coords=data.locs,
        ).to(self.device)

        stdk_config = {
            "epochs": fs_cfg.get("epochs", 350),
            "lr": fs_cfg.get("lr", 1e-3),
            "weight_decay": fs_cfg.get("weight_decay", 1e-4),
            "batch_size": fs_cfg.get("batch_size", 512),
            "patience": fs_cfg.get("patience", 30),
            "current_quantile": 0.5,
        }

        ckpt_dir = self.output_dir / f"stdk_seed{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        trainer = STDKTrainer(
            model=model,
            train_loader=data.metadata["stdk_train_loader"],
            val_loader=data.metadata["stdk_val_loader"],
            config=stdk_config,
            device=self.device,
            output_dir=ckpt_dir,
        )
        trainer.fit()
        model.eval()

        return _STDKTrendWrapper(model, data.metadata, self.device)

    def evaluate(
        self,
        trainer: SpatialNeuralAdapter,
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

        return metrics


class Weather2KHoldoutExperiment(Weather2KTimesplitExperiment):
    """Weather2K with held-out stations for spatial prediction (kriging)."""

    def load_data(self, seed: int) -> DataSplit:
        """Load data with station holdout for spatial prediction."""

        cfg = self.data_cfg
        data = super().load_data(seed)

        # Further split stations into observed / held-out
        heldout_ratio = cfg.get("heldout_station_ratio", 0.2)
        N = data.n_locations
        rng = np.random.RandomState(seed)
        perm = rng.permutation(N)
        n_heldout = int(np.round(heldout_ratio * N))
        heldout_idx = np.sort(perm[:n_heldout])
        observed_idx = np.sort(perm[n_heldout:])

        data.metadata["observed_idx"] = observed_idx
        data.metadata["heldout_idx"] = heldout_idx

        return data

    def evaluate(
        self,
        trainer: SpatialNeuralAdapter,
        data: DataSplit,
        model_name: str,
    ) -> dict:
        """Evaluate spatial prediction at held-out stations.

        Returns RMSE, MPIW, and CP against the continuous target at held-out
        stations (paper Table 6).  MPIW / CP use the closed-form plug-in
        variance
        $v(\\mathbf s)=\\sum_k \\phi_k(\\mathbf s)^2\\,\\hat\\Lambda_k+\\hat\\sigma^2$
        evaluated from the training-station residual covariance.
        """
        metrics = super().evaluate(trainer, data, model_name)

        heldout_idx = data.metadata.get("heldout_idx")
        if heldout_idx is None or len(heldout_idx) == 0:
            return metrics

        trainer.trend.eval()
        trainer.basis.eval()

        with torch.no_grad():
            test_y = data.test_y.to(self.device)
            mu = trainer.trend(data.test_cont.to(self.device))
            y_pred_heldout = mu[:, heldout_idx]
            y_true_heldout = test_y[:, heldout_idx]

            rmse_ho, _, _ = compute_metrics(y_true_heldout, y_pred_heldout)
            metrics["rmse_heldout"] = round(rmse_ho, 6)

            # --- Plug-in interval: v(s) = sum_k phi_k(s)^2 Lambda_k + sigma^2 ---
            try:
                Phi = trainer.basis.basis  # (N, K)
                train_cont = data.train_cont.to(self.device)
                train_y = data.train_y.to(self.device)
                mu_train = trainer.trend(train_cont)
                R_train = train_y - mu_train
                S = (R_train.T @ R_train) / R_train.shape[0]
                PhiTS = Phi.T @ S @ Phi
                eigvals = torch.linalg.eigvalsh(PhiTS)
                sigma2 = max(
                    1e-6,
                    (torch.trace(S).item() - eigvals.sum().item())
                    / (data.n_locations - self.latent_dim),
                )
                Lambda = torch.clamp(eigvals - sigma2, min=0.0)
                v_s = (Phi**2) @ Lambda + sigma2  # (N,)

                sqrt_v_ho = torch.sqrt(v_s[heldout_idx]).unsqueeze(0)  # (1, n_ho)
                y_lo = y_pred_heldout - 1.96 * sqrt_v_ho
                y_hi = y_pred_heldout + 1.96 * sqrt_v_ho

                mpiw = (y_hi - y_lo).mean().item()
                metrics["mpiw"] = round(mpiw, 6)

                covered = ((y_true_heldout >= y_lo) & (y_true_heldout <= y_hi)).float()
                metrics["cp"] = round(covered.mean().item() * 100.0, 4)
            except Exception as e:
                print(f"  [warning] holdout MPIW/CP failed: {e}")

        return metrics
