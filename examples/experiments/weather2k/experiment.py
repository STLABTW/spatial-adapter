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
from spatial_adapter.models.spatial_adapter import SpatialAdapter


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

        return metrics


class Weather2KHoldoutExperiment(Weather2KTimesplitExperiment):
    """Weather2K with held-out stations for spatial prediction (kriging).

    Contrary to ``Weather2KTimesplitExperiment``, held-out stations are
    *actually* removed from the adapter and STDK training data; their
    Y values are only used to evaluate held-out-station metrics at
    test time via the conditional-kriging predictor of paper §4.1.
    """

    def load_data(self, seed: int) -> DataSplit:
        from torch.utils.data import TensorDataset

        cfg = self.data_cfg
        data = super().load_data(seed)

        # Split stations into observed (training) vs held-out (eval only).
        # Use a distinct RNG offset from the 10% spatial subsample so the
        # two random choices are independent.
        heldout_ratio = cfg.get("heldout_station_ratio", 0.2)
        N_full = data.n_locations
        rng = np.random.RandomState(seed + 33333)
        perm = rng.permutation(N_full)
        n_heldout = int(np.round(heldout_ratio * N_full))
        heldout_idx = np.sort(perm[:n_heldout])
        observed_idx = np.sort(perm[n_heldout:])

        # Save full-N references BEFORE we overwrite DataSplit fields.
        locs_full = data.locs  # (N_full, 2) numpy
        y_all_full = data.metadata["y_all"]  # (T, N_full) tensor
        uniq_t = data.metadata["uniq_t"]
        test_idx = data.metadata["test_idx"]
        train_idx = data.metadata["train_idx"]
        val_idx = data.metadata["val_idx"]

        # ----- Restrict adapter-side DataSplit to observed stations -----
        data.train_y = data.train_y[:, observed_idx].contiguous()
        data.val_y = data.val_y[:, observed_idx].contiguous()
        data.test_y = data.test_y[:, observed_idx].contiguous()
        data.train_cont = data.train_cont[:, observed_idx, :].contiguous()
        data.val_cont = data.val_cont[:, observed_idx, :].contiguous()
        data.test_cont = data.test_cont[:, observed_idx, :].contiguous()
        data.locs = locs_full[observed_idx]
        data.n_locations = len(observed_idx)

        # Rebuild the adapter train loader on the observed subset
        # (ADMM's Φ / Z / U dimensions all follow data.n_locations).
        cat_dummy = torch.zeros(data.train_y.shape[0], data.n_locations, 0)
        data.train_loader = DataLoader(
            TensorDataset(cat_dummy, data.train_cont, data.train_y),
            batch_size=self.adapter_config.training.batch_size,
            shuffle=True,
        )

        # Rebuild STDK loaders on observed stations only, so the backbone
        # is likewise blind to held-out Y during training.
        coords_obs = torch.tensor(data.locs, dtype=torch.float32)
        y_all_obs = y_all_full[:, observed_idx].contiguous()
        N_obs = data.n_locations

        def _build_stdk_loader(time_idx, shuffle=True):
            n_t = len(time_idx)
            coords_flat = coords_obs.unsqueeze(0).expand(n_t, -1, -1).reshape(-1, 2)
            t_vals = (
                torch.tensor(uniq_t[time_idx], dtype=torch.float32)
                .unsqueeze(1)
                .expand(-1, N_obs)
                .reshape(-1, 1)
            )
            X_flat = torch.zeros(n_t * N_obs, 0)
            y_flat = y_all_obs[time_idx].reshape(-1, 1)
            ds = DictDataset(X_flat, coords_flat, t_vals, y_flat)
            return DataLoader(
                ds,
                batch_size=self.config.get("first_stage", {}).get("batch_size", 512),
                shuffle=shuffle,
                collate_fn=collate_fn,
            )

        data.metadata["stdk_train_loader"] = _build_stdk_loader(train_idx)
        data.metadata["stdk_val_loader"] = _build_stdk_loader(val_idx, shuffle=False)
        data.metadata["coords"] = coords_obs
        data.metadata["y_all"] = y_all_obs

        # Held-out bookkeeping — never used during training, only at evaluate().
        data.metadata["observed_idx"] = observed_idx
        data.metadata["heldout_idx"] = heldout_idx
        data.metadata["locs_full"] = locs_full
        data.metadata["locs_heldout"] = locs_full[heldout_idx]
        data.metadata["test_y_heldout"] = y_all_full[test_idx][:, heldout_idx].float()
        # Held-out Y at val times — used as the calibration set for post-hoc
        # conformal PI.  Matches the test distribution (held-out space × unseen
        # time), and is disjoint from both training (masked) and Optuna
        # hyperparameter selection (which uses val_y at OBSERVED stations).
        data.metadata["val_y_heldout"] = y_all_full[val_idx][:, heldout_idx].float()

        return data

    def evaluate(
        self,
        trainer: SpatialAdapter,
        data: DataSplit,
        model_name: str,
    ) -> dict:
        """Evaluate spatial prediction at held-out stations using paper §4.1.

        Held-out stations are *not* part of training data (masked in
        ``load_data``); their Y values are used only to score the
        conditional-kriging predictor built from training-station
        residuals.  PI is reported in two flavours:

          * plug-in Gaussian (z=1.96) — eq:cgi special case
          * post-hoc calibrated (q̂_α) — eq:calibrated-q, val-set calibrated
        """
        from spatial_adapter.cpp_extensions import spatial_utils as _su
        from spatial_adapter.prediction import (
            calibrate_q,
            conditional_covariance,
            conditional_score,
        )

        metrics = super().evaluate(trainer, data, model_name)

        heldout_idx = data.metadata.get("heldout_idx")
        if heldout_idx is None or len(heldout_idx) == 0:
            return metrics

        trainer.trend.eval()
        trainer.basis.eval()

        try:
            # ── STDK predictions at BOTH observed and held-out stations at
            # test times, computed directly on the underlying STDK model so
            # we aren't limited to the trend wrapper's observed-only coords.
            stdk = trainer.trend.stdk_model
            coords_obs = data.metadata["coords"].to(self.device)  # (N_obs, 2)
            coords_ho = torch.tensor(
                data.metadata["locs_heldout"], dtype=torch.float32
            ).to(
                self.device
            )  # (N_ho, 2)
            uniq_t = data.metadata["uniq_t"]
            test_idx = data.metadata["test_idx"]
            t_test_vals = torch.tensor(uniq_t[test_idx], dtype=torch.float32)

            N_obs = coords_obs.shape[0]
            coords_ho.shape[0]
            T_test = len(test_idx)

            def _stdk_predict(coords, t_vals):
                """Predict STDK at (coords, t_vals) pairs per time step."""
                n = coords.shape[0]
                T = len(t_vals)
                out = torch.empty(T, n, device=self.device)
                with torch.no_grad():
                    for i in range(T):
                        t_i = torch.full((n, 1), float(t_vals[i]), device=self.device)
                        out[i] = stdk(
                            torch.zeros(n, 0, device=self.device),
                            coords.to(self.device),
                            t_i,
                        ).squeeze(-1)
                return out

            mu_obs_test = _stdk_predict(coords_obs, t_test_vals)  # (T_test, N_obs)
            mu_ho_test = _stdk_predict(coords_ho, t_test_vals)  # (T_test, N_ho)

            # Residuals on training set → estimate Λ (K,) and σ²
            with torch.no_grad():
                Phi_obs = trainer.basis.basis.detach()  # (N_obs, K)
                mu_train = trainer.trend(data.train_cont.to(self.device))
                R_train = data.train_y.to(self.device) - mu_train  # (T_train, N_obs)
                S = (R_train.T @ R_train) / R_train.shape[0]  # (N_obs, N_obs)
                eigvals = torch.linalg.eigvalsh(Phi_obs.T @ S @ Phi_obs)
                sigma2 = max(
                    1e-6,
                    (torch.trace(S).item() - eigvals.sum().item())
                    / (N_obs - self.latent_dim),
                )
                Lambda = torch.clamp(eigvals - sigma2, min=0.0).cpu().numpy()  # (K,)

            # TPS-interpolate basis to held-out (paper line 1724)
            Phi_obs_np = Phi_obs.detach().cpu().numpy().astype(np.float64)
            Phi_ho_np = _su.interpolate_eigenfunction(
                data.metadata["locs_heldout"].astype(np.float64),
                data.locs.astype(np.float64),
                Phi_obs_np,
            )  # (N_ho, K)

            # Residuals at test times (observed stations only — held-out
            # residuals are what we're predicting).
            with torch.no_grad():
                R_test_obs = (
                    (data.test_y.to(self.device) - mu_obs_test)
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )  # (T_test, N_obs)

            # Per-time conditional mean + variance at each held-out station.
            # All training stations are observable at each test time, so
            # Λ_cond is shared across j; α_j depends on the per-time residual.
            Lambda_diag = np.diag(Lambda)
            Lambda_cond = conditional_covariance(
                Lambda_diag, Phi_obs_np, sigma2
            )  # (K, K)

            # α_j for every test time j, stacked → (T_test, K)
            alpha_all = np.stack(
                [
                    conditional_score(Lambda_cond, Phi_obs_np, R_test_obs[j], sigma2)
                    for j in range(T_test)
                ],
                axis=0,
            )

            # Conditional predictor + variance at held-out stations.
            # v̂(s*) = φ̂(s*)^T Λ_cond φ̂(s*) + σ²
            y_pred_heldout = (
                mu_ho_test.cpu().numpy() + alpha_all @ Phi_ho_np.T
            )  # (T_test, N_ho)
            v_ho = (
                np.einsum("ik,kl,il->i", Phi_ho_np, Lambda_cond, Phi_ho_np) + sigma2
            )  # (N_ho,)
            sqrt_v_ho = np.sqrt(np.maximum(v_ho, 1e-12))  # (N_ho,)

            y_true_heldout = data.metadata["test_y_heldout"].numpy()  # (T_test, N_ho)

            # RMSE at held-out
            rmse_ho = float(np.sqrt(np.mean((y_true_heldout - y_pred_heldout) ** 2)))
            metrics["rmse_heldout"] = round(rmse_ho, 6)

            # Plug-in Gaussian PI (uncalibrated)
            y_lo = y_pred_heldout - 1.96 * sqrt_v_ho[None, :]
            y_hi = y_pred_heldout + 1.96 * sqrt_v_ho[None, :]
            metrics["mpiw"] = round(float(np.mean(y_hi - y_lo)), 6)
            metrics["cp"] = round(
                float(
                    np.mean((y_true_heldout >= y_lo) & (y_true_heldout <= y_hi)) * 100.0
                ),
                4,
            )

            # Post-hoc calibrated PI (paper eq:calibrated-q).
            # Calibration set = HELD-OUT stations × VAL times (same spatial
            # OOD regime as test, different time window).  Held-out Y at val
            # times is never used in training (masked in load_data) nor in
            # hyperparameter selection (Optuna minimises RMSE on OBSERVED val
            # stations), so this calibration is leak-free yet distribution-
            # matched to the test query point.
            val_y_heldout = data.metadata["val_y_heldout"].numpy()  # (T_val, N_ho)
            t_val_vals = torch.tensor(
                uniq_t[data.metadata["val_idx"]], dtype=torch.float32
            )
            mu_obs_val = _stdk_predict(coords_obs, t_val_vals)  # (T_val, N_obs)
            mu_ho_val = _stdk_predict(coords_ho, t_val_vals)  # (T_val, N_ho)

            R_val_obs = (
                (data.val_y.to(self.device) - mu_obs_val)
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            T_val = R_val_obs.shape[0]
            alpha_val = np.stack(
                [
                    conditional_score(Lambda_cond, Phi_obs_np, R_val_obs[j], sigma2)
                    for j in range(T_val)
                ],
                axis=0,
            )  # (T_val, K)
            # Predict Y at held-out × val times using the same conditional-
            # kriging pipeline as test-time prediction.
            y_pred_val_ho = mu_ho_val.cpu().numpy() + alpha_val @ Phi_ho_np.T

            q_hat = calibrate_q(
                y_cal=val_y_heldout.ravel(),
                eta_hat_cal=y_pred_val_ho.ravel(),
                pred_var_cal=np.broadcast_to(
                    v_ho[None, :], y_pred_val_ho.shape
                ).ravel(),
                alpha=0.05,
            )
            metrics["q_hat"] = round(float(q_hat), 4)

            y_lo_cal = y_pred_heldout - q_hat * sqrt_v_ho[None, :]
            y_hi_cal = y_pred_heldout + q_hat * sqrt_v_ho[None, :]
            metrics["mpiw_calibrated"] = round(float(np.mean(y_hi_cal - y_lo_cal)), 6)
            metrics["cp_calibrated"] = round(
                float(
                    np.mean((y_true_heldout >= y_lo_cal) & (y_true_heldout <= y_hi_cal))
                    * 100.0
                ),
                4,
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"  [warning] holdout conditional-kriging eval failed: {e}")

        return metrics
