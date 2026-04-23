##  config
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
WORK_ROOT = THIS_FILE.parents[4]
SCRIPT_DIR = THIS_FILE.parent

ROOT = str(REPO_ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

torch.set_default_dtype(torch.float32)
optuna.logging.set_verbosity(optuna.logging.WARNING)

from examples.baselines.stdk.st_interp import create_model
from examples.baselines.timesplit.experiment_core import (
    DictDataset,
    build_contiguous_time_splits,
    build_fixed_location_subset,
    build_standard_bin_edges,
    build_stdk_model_config,
    collate_fn,
    fit_adapter_reconstruct_all_times,
    new_trend_basis,
    predict_all_simple,
    rmse_pooled,
    seed_everything,
    sv_match_loss_for_prediction_matrix,
    train_simple_loop,
)
from spatial_adapter.models.spatial_adapter import (
    ADMMConfig,
    BasisConfig,
    SpatialAdapterConfig,
    TrainingConfig,
)

# Global settings & dirs
SEED = 123

WEATHER2K_NPY = WORK_ROOT / "data" / "weather2k.npy"
TARGET_VAR_IDX = 4
LAT_IDX = 0
LON_IDX = 1
T_KEEP = 1000

EPOCHS = 350
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-5

SPACE_RATIO_KEEP = 0.1

TRAIN_RATIO_TIME = 0.8
VAL_RATIO_TIME = 0.1
TEST_RATIO_TIME = 0.1

GNA_BATCH_SIZE = 64
N_TRIALS = 50
TAU_MIN = 1e-8
TAU_MAX = 1e4

K_FIXED = 40
N_RUNS_FIXED_K = 100

SEMIVAR_WEIGHTED = True
SEMIVAR_NORMALIZED = False
SEMIVAR_ESTIMATOR = "matheron"

TUNING_TARGET = "rmse"

if TUNING_TARGET not in {"rmse", "covfrob", "sv_score"}:
    raise ValueError("TUNING_TARGET must be one of {'rmse', 'covfrob', 'sv_score'}")

RESULT_DIR = SCRIPT_DIR / "weather2k"
REPEAT_DIR = RESULT_DIR / (
    f"{TUNING_TARGET}_tuning"
    f"/var{TARGET_VAR_IDX}_tkeep{T_KEEP}"
    f"_k_{K_FIXED}_fixedspace{SPACE_RATIO_KEEP}"
    f"_time_train{TRAIN_RATIO_TIME}_val{VAL_RATIO_TIME}_test{TEST_RATIO_TIME}"
)
REPEAT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = REPEAT_DIR / f"k_{K_FIXED}_repeat_runs_summary.csv"

PRED_DIR = REPEAT_DIR / "saved_predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

TRIAL_DIR = REPEAT_DIR / "trial_results"
TRIAL_DIR.mkdir(parents=True, exist_ok=True)

PHI_DIR = REPEAT_DIR / "saved_phi"
PHI_DIR.mkdir(parents=True, exist_ok=True)

ALL_TRIAL_CSV = TRIAL_DIR / "all_trial_results.csv"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device, flush=True)

seed_everything(SEED)

config = SpatialAdapterConfig(
    admm=ADMMConfig(
        rho=1.0,
        dual_momentum=0.2,
        max_iters=3000,
        min_outer=20,
        tol=1e-4,
    ),
    training=TrainingConfig(
        lr_mu=1e-2,
        batch_size=GNA_BATCH_SIZE,
        pretrain_epochs=5,
    ),
    basis=BasisConfig(
        phi_every=5,
        phi_freeze=200,
    ),
)


# Data helpers
def load_weather2k_as_long_df(
    npy_path,
    target_var_idx,
    lat_idx,
    lon_idx,
    t_keep=None,
    normalize_xy=True,
):
    arr = np.load(str(npy_path)).astype(np.float32)

    if arr.ndim != 3:
        raise ValueError(f"Expected arr.ndim == 3 (S, V, T), got shape={arr.shape}")

    n_sites, n_vars, n_times_full = arr.shape

    if not (
        0 <= lat_idx < n_vars and 0 <= lon_idx < n_vars and 0 <= target_var_idx < n_vars
    ):
        raise ValueError(
            f"Bad index: lat_idx={lat_idx}, lon_idx={lon_idx}, "
            f"target_var_idx={target_var_idx}, n_vars={n_vars}"
        )

    lat = arr[:, lat_idx, 0].astype(np.float32)
    lon = arr[:, lon_idx, 0].astype(np.float32)
    z_full = arr[:, target_var_idx, :].astype(np.float32)

    if t_keep is not None:
        if t_keep <= 0 or t_keep > n_times_full:
            raise ValueError(f"t_keep must be in [1, {n_times_full}], got {t_keep}")
        z_use = z_full[:, -t_keep:]
        n_times = t_keep
    else:
        z_use = z_full
        n_times = n_times_full

    if normalize_xy:
        lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
        lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
        x = ((lon - lon_min) / (lon_max - lon_min + 1e-12)).astype(np.float32)
        y = ((lat - lat_min) / (lat_max - lat_min + 1e-12)).astype(np.float32)
    else:
        x = lon.astype(np.float32)
        y = lat.astype(np.float32)

    t = np.arange(n_times, dtype=np.int64)

    xx = np.repeat(x, n_times)
    yy = np.repeat(y, n_times)
    tt = np.tile(t, n_sites)
    zz = z_use.reshape(-1).astype(np.float32)

    df = pd.DataFrame(
        {
            "x": xx,
            "y": yy,
            "t": tt,
            "z": zz,
        }
    )

    z_np = df["z"].to_numpy(np.float32)
    ok = np.isfinite(z_np)
    df = df.loc[ok].reset_index(drop=True)

    return df


# Metric helpers
def empirical_cov(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)

    if field.ndim != 2:
        raise ValueError(f"field must be 2D, got shape={field.shape}")
    if field.shape[0] <= 1:
        raise ValueError("field must have at least 2 rows to compute sample covariance")

    field_centered = field - np.mean(field, axis=0, keepdims=True)
    cov = (field_centered.T @ field_centered) / (field.shape[0] - 1)
    cov = 0.5 * (cov + cov.T)
    return cov


def cov_frob_observed(y_true_field: np.ndarray, y_pred_field: np.ndarray) -> float:
    sigma_obs = empirical_cov(y_true_field)
    sigma_pred = empirical_cov(y_pred_field)

    return float(
        np.linalg.norm(sigma_pred - sigma_obs, ord="fro")
        / (np.linalg.norm(sigma_obs, ord="fro") + 1e-12)
    )


# Objective helpers
def choose_objective_value(val_rmse, covfrob_reg_val, sv_loss_val):
    if TUNING_TARGET == "rmse":
        return float(val_rmse)
    if TUNING_TARGET == "covfrob":
        return float(covfrob_reg_val)
    if TUNING_TARGET == "sv_score":
        return float(sv_loss_val)
    raise ValueError(f"Unknown TUNING_TARGET: {TUNING_TARGET}")


def objective_label():
    if TUNING_TARGET == "rmse":
        return "best_val_rmse"
    if TUNING_TARGET == "covfrob":
        return "best_val_covfrob"
    if TUNING_TARGET == "sv_score":
        return "best_val_sv_score"
    raise ValueError(f"Unknown TUNING_TARGET: {TUNING_TARGET}")


def objective_best_by_column():
    if TUNING_TARGET == "rmse":
        return "val_rmse_raw"
    if TUNING_TARGET == "covfrob":
        return "covfrob_reg_val"
    if TUNING_TARGET == "sv_score":
        return "sv_loss_val"
    raise ValueError(f"Unknown TUNING_TARGET: {TUNING_TARGET}")


## seed runs
def run_once_fixed_k(run_seed: int):
    seed_everything(run_seed)

    df_full = load_weather2k_as_long_df(
        npy_path=WEATHER2K_NPY,
        target_var_idx=TARGET_VAR_IDX,
        lat_idx=LAT_IDX,
        lon_idx=LON_IDX,
        t_keep=T_KEEP,
        normalize_xy=True,
    )

    df_run, keep_sites_run, n_sites_full_run = build_fixed_location_subset(
        df=df_full,
        keep_ratio=SPACE_RATIO_KEEP,
        seed=run_seed + 11111,
    )

    df_run["t_norm"] = (
        (df_run["t"] - df_run["t"].min())
        / (df_run["t"].max() - df_run["t"].min() + 1e-12)
    ).astype(np.float32)

    (
        train_mask_flat_run,
        val_mask_flat_run,
        test_mask_flat_run,
        train_time_idx_run,
        val_time_idx_run,
        test_time_idx_run,
        uniq_t_run,
        n_times_run,
    ) = build_contiguous_time_splits(
        df=df_run,
        train_ratio=TRAIN_RATIO_TIME,
        val_ratio=VAL_RATIO_TIME,
        test_ratio=TEST_RATIO_TIME,
    )

    df_run["z_raw"] = df_run["z"].astype(np.float32)

    z_train_raw = df_run.loc[train_mask_flat_run, "z_raw"].to_numpy(np.float32)
    z_mean_run = float(np.mean(z_train_raw))
    z_sd_run = float(np.std(z_train_raw, ddof=0))
    if z_sd_run < 1e-12:
        z_sd_run = 1.0

    df_run["z"] = (
        (df_run["z_raw"].to_numpy(np.float32) - z_mean_run) / (z_sd_run + 1e-12)
    ).astype(np.float32)

    def to_raw(arr_std: np.ndarray) -> np.ndarray:
        return arr_std * z_sd_run + z_mean_run

    coords_all_run = df_run[["x", "y"]].to_numpy(np.float32)
    t_all_run = df_run["t_norm"].to_numpy(np.float32).reshape(-1, 1)
    y_all_run = df_run["z"].to_numpy(np.float32).reshape(-1, 1)
    X_all_run = np.empty((df_run.shape[0], 0), dtype=np.float32)

    X_train_run = X_all_run[train_mask_flat_run]
    coords_train_run = coords_all_run[train_mask_flat_run]
    t_train_run = t_all_run[train_mask_flat_run]
    y_train_run = y_all_run[train_mask_flat_run]

    train_dataset_run = DictDataset(
        torch.from_numpy(X_train_run),
        torch.from_numpy(coords_train_run),
        torch.from_numpy(t_train_run),
        torch.from_numpy(y_train_run),
    )

    g = torch.Generator()
    g.manual_seed(run_seed + 1000)

    train_loader_run = DataLoader(
        train_dataset_run,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    stdk_config = build_stdk_model_config(
        EPOCHS=EPOCHS,
        LR=LR,
        WEIGHT_DECAY=WEIGHT_DECAY,
        BATCH_SIZE=BATCH_SIZE,
    )

    stdk_run = create_model(
        stdk_config,
        train_coords=coords_train_run,
    ).to(device)

    stdk_run = train_simple_loop(
        model=stdk_run,
        train_loader=train_loader_run,
        device=device,
        config=stdk_config,
    )

    y_hat_all_run = predict_all_simple(
        model=stdk_run,
        X=torch.from_numpy(X_all_run),
        coords=torch.from_numpy(coords_all_run),
        t=torch.from_numpy(t_all_run),
        batch_size=BATCH_SIZE,
        device=device,
    )

    locs_run, inv_loc_run = np.unique(coords_all_run, axis=0, return_inverse=True)
    t_to_idx_run = {t: i for i, t in enumerate(uniq_t_run)}

    T_run = len(uniq_t_run)
    N_run = len(locs_run)

    t_idx_run = np.array([t_to_idx_run[t] for t in df_run["t"].to_numpy()])
    s_idx_run = inv_loc_run

    y_stdk_run = np.full((T_run, N_run), np.nan, np.float32)
    y_true_run = np.full((T_run, N_run), np.nan, np.float32)

    y_stdk_run[t_idx_run, s_idx_run] = y_hat_all_run
    y_true_run[t_idx_run, s_idx_run] = df_run["z"].to_numpy(np.float32)

    y_stdk_raw_run = to_raw(y_stdk_run)
    y_true_raw_run = to_raw(y_true_run)

    residual_true_run = y_true_run - y_stdk_run

    time_feat_run = (
        (uniq_t_run - uniq_t_run.min()) / (uniq_t_run.max() - uniq_t_run.min() + 1e-12)
    ).astype(np.float32)

    cont_all_run = (
        torch.from_numpy(time_feat_run)
        .float()
        .unsqueeze(1)
        .repeat(1, N_run)
        .unsqueeze(-1)
    )

    train_cont_train_run = cont_all_run[train_time_idx_run]
    train_y_train_run = torch.from_numpy(
        residual_true_run[train_time_idx_run, :]
    ).float()
    train_idx_train_run = torch.arange(len(train_time_idx_run), dtype=torch.long)

    gna_loader_train_run = DataLoader(
        TensorDataset(train_idx_train_run, train_cont_train_run, train_y_train_run),
        batch_size=min(GNA_BATCH_SIZE, len(train_time_idx_run)),
        shuffle=True,
        drop_last=False,
    )

    residual_true_tensor_all_run = torch.from_numpy(residual_true_run).float()

    seed_phi_dir = PHI_DIR / f"seed_{run_seed}"
    seed_phi_dir.mkdir(parents=True, exist_ok=True)

    semivar_bin_edges = build_standard_bin_edges(locs_run)
    full_time_idx_run = np.arange(T_run, dtype=int)

    def rmse_std(y_pred_std: np.ndarray, time_idx: np.ndarray) -> float:
        mask = np.isfinite(y_true_run[time_idx, :]) & np.isfinite(
            y_pred_std[time_idx, :]
        )
        return rmse_pooled(y_true_run[time_idx, :], y_pred_std[time_idx, :], mask)

    def rmse_raw(y_pred_std: np.ndarray, time_idx: np.ndarray) -> float:
        y_pred_raw = to_raw(y_pred_std[time_idx, :])
        mask = np.isfinite(y_true_raw_run[time_idx, :]) & np.isfinite(y_pred_raw)
        return rmse_pooled(y_true_raw_run[time_idx, :], y_pred_raw, mask)

    def covfrob_std(y_pred_std: np.ndarray, time_idx: np.ndarray) -> float:
        return cov_frob_observed(
            y_true_run[time_idx, :],
            y_pred_std[time_idx, :],
        )

    def covfrob_raw(y_pred_std: np.ndarray, time_idx: np.ndarray) -> float:
        return cov_frob_observed(
            y_true_raw_run[time_idx, :],
            to_raw(y_pred_std[time_idx, :]),
        )

    def sv_loss_std(y_pred_std: np.ndarray, time_idx: np.ndarray) -> float:
        out = sv_match_loss_for_prediction_matrix(
            coords=locs_run,
            y_true=y_true_run,
            y_pred=y_pred_std,
            time_idx=time_idx,
            bin_edges=semivar_bin_edges,
            estimator=SEMIVAR_ESTIMATOR,
            weighted=SEMIVAR_WEIGHTED,
            normalized=SEMIVAR_NORMALIZED,
        )
        return float(out["loss"])

    def sv_loss_raw(y_pred_std: np.ndarray, time_idx: np.ndarray) -> float:
        out = sv_match_loss_for_prediction_matrix(
            coords=locs_run,
            y_true=y_true_raw_run,
            y_pred=to_raw(y_pred_std),
            time_idx=time_idx,
            bin_edges=semivar_bin_edges,
            estimator=SEMIVAR_ESTIMATOR,
            weighted=SEMIVAR_WEIGHTED,
            normalized=SEMIVAR_NORMALIZED,
        )
        return float(out["loss"])

    writer_unreg = SummaryWriter(
        str(REPEAT_DIR / "logs_unreg" / f"seed_{run_seed}" / "unreg_tau1_0_tau2_0")
    )

    trend_unreg, basis_unreg = new_trend_basis(N_run, K_FIXED)
    fit_unreg = fit_adapter_reconstruct_all_times(
        tag="unreg_tau1_0_tau2_0",
        tau1=0.0,
        tau2=0.0,
        trend=trend_unreg,
        basis=basis_unreg,
        train_loader=gna_loader_train_run,
        val_cont=train_cont_train_run,
        val_y=train_y_train_run,
        locs=locs_run,
        config=config,
        device=device,
        cont_all=cont_all_run,
        residual_true_tensor_all=residual_true_tensor_all_run,
        writer=writer_unreg,
    )
    writer_unreg.close()

    residual_full_unreg_run = fit_unreg["pred_all"]
    diag_train_unreg_run = fit_unreg["diag_train"]
    diag_all_unreg_run = fit_unreg["diag_all"]
    phi_unreg_run = fit_unreg["phi"]

    np.savez_compressed(
        seed_phi_dir / "unreg_phi.npz",
        phi=phi_unreg_run.astype(np.float32),
        tau1=np.array([0.0], dtype=np.float64),
        tau2=np.array([0.0], dtype=np.float64),
        seed=np.array([run_seed], dtype=np.int32),
    )

    y_final_unreg_run = y_stdk_run + residual_full_unreg_run
    y_final_unreg_raw_run = to_raw(y_final_unreg_run)

    trial_rows = []
    trial_cache = {}

    def objective(trial: optuna.Trial):
        tau1 = trial.suggest_float("tau1", TAU_MIN, TAU_MAX, log=True)
        tau2 = trial.suggest_float("tau2", TAU_MIN, TAU_MAX, log=True)

        tag = f"trial_{trial.number:03d}_tau1_{tau1:.3e}_tau2_{tau2:.3e}"
        writer_trial = SummaryWriter(
            str(REPEAT_DIR / "logs_reg" / f"seed_{run_seed}" / tag)
        )

        trend_trial, basis_trial = new_trend_basis(N_run, K_FIXED)
        fit_trial = fit_adapter_reconstruct_all_times(
            tag=tag,
            tau1=tau1,
            tau2=tau2,
            trend=trend_trial,
            basis=basis_trial,
            train_loader=gna_loader_train_run,
            val_cont=train_cont_train_run,
            val_y=train_y_train_run,
            locs=locs_run,
            config=config,
            device=device,
            cont_all=cont_all_run,
            residual_true_tensor_all=residual_true_tensor_all_run,
            writer=writer_trial,
        )
        writer_trial.close()

        residual_full_trial = fit_trial["pred_all"]
        diag_train_trial = fit_trial["diag_train"]
        diag_all_trial = fit_trial["diag_all"]
        phi_trial = fit_trial["phi"]

        y_final_trial = y_stdk_run + residual_full_trial

        train_rmse_std = rmse_std(y_final_trial, train_time_idx_run)
        val_rmse_std = rmse_std(y_final_trial, val_time_idx_run)
        test_rmse_std = rmse_std(y_final_trial, test_time_idx_run)
        full_rmse_std = rmse_std(y_final_trial, full_time_idx_run)

        train_rmse_raw = rmse_raw(y_final_trial, train_time_idx_run)
        val_rmse_raw = rmse_raw(y_final_trial, val_time_idx_run)
        test_rmse_raw = rmse_raw(y_final_trial, test_time_idx_run)
        full_rmse_raw = rmse_raw(y_final_trial, full_time_idx_run)

        covfrob_reg_train = covfrob_std(y_final_trial, train_time_idx_run)
        covfrob_reg_val = covfrob_std(y_final_trial, val_time_idx_run)
        covfrob_reg_test = covfrob_std(y_final_trial, test_time_idx_run)
        covfrob_reg_full = covfrob_std(y_final_trial, full_time_idx_run)

        covfrob_reg_train_raw = covfrob_raw(y_final_trial, train_time_idx_run)
        covfrob_reg_val_raw = covfrob_raw(y_final_trial, val_time_idx_run)
        covfrob_reg_test_raw = covfrob_raw(y_final_trial, test_time_idx_run)
        covfrob_reg_full_raw = covfrob_raw(y_final_trial, full_time_idx_run)

        sv_loss_train = sv_loss_std(y_final_trial, train_time_idx_run)
        sv_loss_val = sv_loss_std(y_final_trial, val_time_idx_run)
        sv_loss_test = sv_loss_std(y_final_trial, test_time_idx_run)
        sv_loss_full = sv_loss_std(y_final_trial, full_time_idx_run)

        sv_loss_train_raw = sv_loss_raw(y_final_trial, train_time_idx_run)
        sv_loss_val_raw = sv_loss_raw(y_final_trial, val_time_idx_run)
        sv_loss_test_raw = sv_loss_raw(y_final_trial, test_time_idx_run)
        sv_loss_full_raw = sv_loss_raw(y_final_trial, full_time_idx_run)

        objective_value = choose_objective_value(
            val_rmse=val_rmse_raw,
            covfrob_reg_val=covfrob_reg_val,
            sv_loss_val=sv_loss_val,
        )

        phi_path = seed_phi_dir / f"trial_{trial.number:03d}_phi.npz"
        np.savez_compressed(
            phi_path,
            phi=phi_trial.astype(np.float32),
            tau1=np.array([tau1], dtype=np.float64),
            tau2=np.array([tau2], dtype=np.float64),
            seed=np.array([run_seed], dtype=np.int32),
            trial=np.array([trial.number], dtype=np.int32),
        )

        row = {
            "seed": int(run_seed),
            "trial": int(trial.number),
            "tau1": float(tau1),
            "tau2": float(tau2),
            "log10_tau1": float(np.log10(tau1)),
            "log10_tau2": float(np.log10(tau2)),
            "objective_value": float(objective_value),
            "train_rmse": float(train_rmse_std),
            "val_rmse": float(val_rmse_std),
            "test_rmse": float(test_rmse_std),
            "full_rmse": float(full_rmse_std),
            "train_rmse_raw": float(train_rmse_raw),
            "val_rmse_raw": float(val_rmse_raw),
            "test_rmse_raw": float(test_rmse_raw),
            "full_rmse_raw": float(full_rmse_raw),
            "sv_loss_train": float(sv_loss_train),
            "sv_loss_val": float(sv_loss_val),
            "sv_loss_test": float(sv_loss_test),
            "sv_loss_full": float(sv_loss_full),
            "sv_loss_train_raw": float(sv_loss_train_raw),
            "sv_loss_val_raw": float(sv_loss_val_raw),
            "sv_loss_test_raw": float(sv_loss_test_raw),
            "sv_loss_full_raw": float(sv_loss_full_raw),
            "covfrob_reg_train": float(covfrob_reg_train),
            "covfrob_reg_val": float(covfrob_reg_val),
            "covfrob_reg_test": float(covfrob_reg_test),
            "covfrob_reg_full": float(covfrob_reg_full),
            "covfrob_reg_train_raw": float(covfrob_reg_train_raw),
            "covfrob_reg_val_raw": float(covfrob_reg_val_raw),
            "covfrob_reg_test_raw": float(covfrob_reg_test_raw),
            "covfrob_reg_full_raw": float(covfrob_reg_full_raw),
            "recon_mse_train": float(diag_train_trial["recon_mse"]),
            "smooth_penalty_train": float(diag_train_trial["smooth_penalty"]),
            "l1_penalty_train": float(diag_train_trial["l1_penalty"]),
            "total_surrogate_train": float(diag_train_trial["total_surrogate"]),
            "smooth_penalty_per_entry_train": float(
                diag_train_trial["smooth_penalty_per_entry"]
            ),
            "l1_penalty_per_entry_train": float(
                diag_train_trial["l1_penalty_per_entry"]
            ),
            "smooth_over_recon_train": float(diag_train_trial["smooth_over_recon"]),
            "l1_over_recon_train": float(diag_train_trial["l1_over_recon"]),
            "recon_mse_all": float(diag_all_trial["recon_mse"]),
            "smooth_penalty_all": float(diag_all_trial["smooth_penalty"]),
            "l1_penalty_all": float(diag_all_trial["l1_penalty"]),
            "total_surrogate_all": float(diag_all_trial["total_surrogate"]),
            "smooth_penalty_per_entry_all": float(
                diag_all_trial["smooth_penalty_per_entry"]
            ),
            "l1_penalty_per_entry_all": float(diag_all_trial["l1_penalty_per_entry"]),
            "smooth_over_recon_all": float(diag_all_trial["smooth_over_recon"]),
            "l1_over_recon_all": float(diag_all_trial["l1_over_recon"]),
            "phi_path": str(phi_path),
        }

        trial_rows.append(row)
        trial_cache[int(trial.number)] = {
            "tau1": float(tau1),
            "tau2": float(tau2),
            "objective_value": float(objective_value),
            "pred_all": residual_full_trial.astype(np.float32),
            "phi": phi_trial.astype(np.float32),
            "diag_train": diag_train_trial,
            "diag_all": diag_all_trial,
        }

        print(
            f"[seed {run_seed}] trial {trial.number + 1:03d}/{N_TRIALS:03d} | "
            f"target={TUNING_TARGET} | "
            f"tau1={tau1:.3e} | tau2={tau2:.3e} | "
            f"train_rmse={train_rmse_std:.6f} | "
            f"val_rmse={val_rmse_std:.6f} | "
            f"sv_loss_val={sv_loss_val:.6f} | "
            f"covfrob_val={covfrob_reg_val:.6f} | "
            f"objective={objective_value:.6f}",
            flush=True,
        )

        return float(objective_value)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    trial_df_run = pd.DataFrame(trial_rows)
    trial_csv_run = TRIAL_DIR / f"seed_{run_seed}_trials.csv"
    trial_df_run.to_csv(trial_csv_run, index=False)

    best_trial_no = int(study.best_trial.number)
    best_trial = trial_cache[best_trial_no]

    best_tau1_run = float(best_trial["tau1"])
    best_tau2_run = float(best_trial["tau2"])
    best_objective_run = float(best_trial["objective_value"])
    residual_full_reg_best_run = best_trial["pred_all"]
    phi_reg_best_run = best_trial["phi"]
    diag_train_reg_best_run = best_trial["diag_train"]
    diag_all_reg_best_run = best_trial["diag_all"]

    np.savez_compressed(
        seed_phi_dir / "reg_best_phi.npz",
        phi=phi_reg_best_run.astype(np.float32),
        tau1=np.array([best_tau1_run], dtype=np.float64),
        tau2=np.array([best_tau2_run], dtype=np.float64),
        seed=np.array([run_seed], dtype=np.int32),
        trial=np.array([best_trial_no], dtype=np.int32),
    )

    y_final_reg_best_run = y_stdk_run + residual_full_reg_best_run
    y_final_reg_best_raw_run = to_raw(y_final_reg_best_run)

    def collect_metrics(prefix: str, y_pred_std: np.ndarray):
        return {
            f"{prefix}_rmse_train": rmse_std(y_pred_std, train_time_idx_run),
            f"{prefix}_rmse_val": rmse_std(y_pred_std, val_time_idx_run),
            f"{prefix}_rmse_test": rmse_std(y_pred_std, test_time_idx_run),
            f"{prefix}_rmse_full": rmse_std(y_pred_std, full_time_idx_run),
            f"{prefix}_rmse_train_raw": rmse_raw(y_pred_std, train_time_idx_run),
            f"{prefix}_rmse_val_raw": rmse_raw(y_pred_std, val_time_idx_run),
            f"{prefix}_rmse_test_raw": rmse_raw(y_pred_std, test_time_idx_run),
            f"{prefix}_rmse_full_raw": rmse_raw(y_pred_std, full_time_idx_run),
            f"{prefix}_covfrob_train": covfrob_std(y_pred_std, train_time_idx_run),
            f"{prefix}_covfrob_val": covfrob_std(y_pred_std, val_time_idx_run),
            f"{prefix}_covfrob_test": covfrob_std(y_pred_std, test_time_idx_run),
            f"{prefix}_covfrob_full": covfrob_std(y_pred_std, full_time_idx_run),
            f"{prefix}_covfrob_train_raw": covfrob_raw(y_pred_std, train_time_idx_run),
            f"{prefix}_covfrob_val_raw": covfrob_raw(y_pred_std, val_time_idx_run),
            f"{prefix}_covfrob_test_raw": covfrob_raw(y_pred_std, test_time_idx_run),
            f"{prefix}_covfrob_full_raw": covfrob_raw(y_pred_std, full_time_idx_run),
            f"{prefix}_sv_loss_train": sv_loss_std(y_pred_std, train_time_idx_run),
            f"{prefix}_sv_loss_val": sv_loss_std(y_pred_std, val_time_idx_run),
            f"{prefix}_sv_loss_test": sv_loss_std(y_pred_std, test_time_idx_run),
            f"{prefix}_sv_loss_full": sv_loss_std(y_pred_std, full_time_idx_run),
            f"{prefix}_sv_loss_train_raw": sv_loss_raw(y_pred_std, train_time_idx_run),
            f"{prefix}_sv_loss_val_raw": sv_loss_raw(y_pred_std, val_time_idx_run),
            f"{prefix}_sv_loss_test_raw": sv_loss_raw(y_pred_std, test_time_idx_run),
            f"{prefix}_sv_loss_full_raw": sv_loss_raw(y_pred_std, full_time_idx_run),
        }

    summary_row = {
        "seed": int(run_seed),
        "n_sites_full": int(n_sites_full_run),
        "n_sites_keep": int(len(keep_sites_run)),
        "n_times": int(n_times_run),
        "n_train_times": int(len(train_time_idx_run)),
        "n_val_times": int(len(val_time_idx_run)),
        "n_test_times": int(len(test_time_idx_run)),
        "z_mean_train_raw": float(z_mean_run),
        "z_sd_train_raw": float(z_sd_run),
        "best_tau1": float(best_tau1_run),
        "best_tau2": float(best_tau2_run),
        "best_trial": int(best_trial_no),
        objective_label(): float(best_objective_run),
        "unreg_recon_mse_train": float(diag_train_unreg_run["recon_mse"]),
        "unreg_smooth_penalty_train": float(diag_train_unreg_run["smooth_penalty"]),
        "unreg_l1_penalty_train": float(diag_train_unreg_run["l1_penalty"]),
        "unreg_total_surrogate_train": float(diag_train_unreg_run["total_surrogate"]),
        "unreg_smooth_penalty_per_entry_train": float(
            diag_train_unreg_run["smooth_penalty_per_entry"]
        ),
        "unreg_l1_penalty_per_entry_train": float(
            diag_train_unreg_run["l1_penalty_per_entry"]
        ),
        "unreg_smooth_over_recon_train": float(
            diag_train_unreg_run["smooth_over_recon"]
        ),
        "unreg_l1_over_recon_train": float(diag_train_unreg_run["l1_over_recon"]),
        "unreg_recon_mse_all": float(diag_all_unreg_run["recon_mse"]),
        "unreg_smooth_penalty_all": float(diag_all_unreg_run["smooth_penalty"]),
        "unreg_l1_penalty_all": float(diag_all_unreg_run["l1_penalty"]),
        "unreg_total_surrogate_all": float(diag_all_unreg_run["total_surrogate"]),
        "unreg_smooth_penalty_per_entry_all": float(
            diag_all_unreg_run["smooth_penalty_per_entry"]
        ),
        "unreg_l1_penalty_per_entry_all": float(
            diag_all_unreg_run["l1_penalty_per_entry"]
        ),
        "unreg_smooth_over_recon_all": float(diag_all_unreg_run["smooth_over_recon"]),
        "unreg_l1_over_recon_all": float(diag_all_unreg_run["l1_over_recon"]),
        "reg_best_recon_mse_train": float(diag_train_reg_best_run["recon_mse"]),
        "reg_best_smooth_penalty_train": float(
            diag_train_reg_best_run["smooth_penalty"]
        ),
        "reg_best_l1_penalty_train": float(diag_train_reg_best_run["l1_penalty"]),
        "reg_best_total_surrogate_train": float(
            diag_train_reg_best_run["total_surrogate"]
        ),
        "reg_best_smooth_penalty_per_entry_train": float(
            diag_train_reg_best_run["smooth_penalty_per_entry"]
        ),
        "reg_best_l1_penalty_per_entry_train": float(
            diag_train_reg_best_run["l1_penalty_per_entry"]
        ),
        "reg_best_smooth_over_recon_train": float(
            diag_train_reg_best_run["smooth_over_recon"]
        ),
        "reg_best_l1_over_recon_train": float(diag_train_reg_best_run["l1_over_recon"]),
        "reg_best_recon_mse_all": float(diag_all_reg_best_run["recon_mse"]),
        "reg_best_smooth_penalty_all": float(diag_all_reg_best_run["smooth_penalty"]),
        "reg_best_l1_penalty_all": float(diag_all_reg_best_run["l1_penalty"]),
        "reg_best_total_surrogate_all": float(diag_all_reg_best_run["total_surrogate"]),
        "reg_best_smooth_penalty_per_entry_all": float(
            diag_all_reg_best_run["smooth_penalty_per_entry"]
        ),
        "reg_best_l1_penalty_per_entry_all": float(
            diag_all_reg_best_run["l1_penalty_per_entry"]
        ),
        "reg_best_smooth_over_recon_all": float(
            diag_all_reg_best_run["smooth_over_recon"]
        ),
        "reg_best_l1_over_recon_all": float(diag_all_reg_best_run["l1_over_recon"]),
    }

    summary_row.update(collect_metrics("stdk", y_stdk_run))
    summary_row.update(collect_metrics("unreg", y_final_unreg_run))
    summary_row.update(collect_metrics("reg_best", y_final_reg_best_run))

    np.savez_compressed(
        PRED_DIR / f"seed_{run_seed}.npz",
        seed=np.array([run_seed], dtype=np.int32),
        keep_sites_run=np.asarray(keep_sites_run, dtype=np.int32),
        n_sites_full_run=np.array([n_sites_full_run], dtype=np.int32),
        uniq_t_run=np.asarray(uniq_t_run, dtype=np.float32),
        train_time_idx_run=np.asarray(train_time_idx_run, dtype=np.int32),
        val_time_idx_run=np.asarray(val_time_idx_run, dtype=np.int32),
        test_time_idx_run=np.asarray(test_time_idx_run, dtype=np.int32),
        locs_run=locs_run.astype(np.float32),
        y_true_run=y_true_run.astype(np.float32),
        y_true_raw_run=y_true_raw_run.astype(np.float32),
        y_stdk_run=y_stdk_run.astype(np.float32),
        y_stdk_raw_run=y_stdk_raw_run.astype(np.float32),
        y_final_unreg_run=y_final_unreg_run.astype(np.float32),
        y_final_unreg_raw_run=y_final_unreg_raw_run.astype(np.float32),
        y_final_reg_best_run=y_final_reg_best_run.astype(np.float32),
        y_final_reg_best_raw_run=y_final_reg_best_raw_run.astype(np.float32),
        residual_full_unreg_run=residual_full_unreg_run.astype(np.float32),
        residual_full_reg_best_run=residual_full_reg_best_run.astype(np.float32),
        z_mean_run=np.array([z_mean_run], dtype=np.float32),
        z_sd_run=np.array([z_sd_run], dtype=np.float32),
        best_tau1=np.array([best_tau1_run], dtype=np.float64),
        best_tau2=np.array([best_tau2_run], dtype=np.float64),
        best_trial=np.array([best_trial_no], dtype=np.int32),
    )

    print(
        f"[seed {run_seed}] best trial={best_trial_no} | "
        f"tau1={best_tau1_run:.3e} | tau2={best_tau2_run:.3e} | "
        f"test_rmse={summary_row['reg_best_rmse_test']:.6f} | "
        f"test_covfrob={summary_row['reg_best_covfrob_test']:.6f} | "
        f"test_sv={summary_row['reg_best_sv_loss_test']:.6f}",
        flush=True,
    )

    return summary_row


## run
if __name__ == "__main__":
    rows_fixed_k = []

    for r in range(N_RUNS_FIXED_K):
        run_seed = SEED + r * 1000
        print(
            f"\n===== FIXED K RUN {r + 1}/{N_RUNS_FIXED_K} | "
            f"seed={run_seed} | target={TUNING_TARGET} | "
            f"space_keep={SPACE_RATIO_KEEP} | "
            f"time_train={TRAIN_RATIO_TIME} | "
            f"time_val={VAL_RATIO_TIME} | "
            f"time_test={TEST_RATIO_TIME} =====",
            flush=True,
        )
        rows_fixed_k.append(run_once_fixed_k(run_seed))

    results_fixed_k_df = pd.DataFrame(rows_fixed_k)
    results_fixed_k_df.to_csv(SUMMARY_CSV, index=False)

    trial_files = sorted(TRIAL_DIR.glob("seed_*_trials.csv"))
    if len(trial_files) > 0:
        all_trial_df = pd.concat(
            [pd.read_csv(f) for f in trial_files], ignore_index=True
        )
        all_trial_df.to_csv(ALL_TRIAL_CSV, index=False)
        n_trial_rows = len(all_trial_df)
    else:
        n_trial_rows = 0

    print("\n=== Summary saved ===", flush=True)
    print("SUMMARY_CSV:", SUMMARY_CSV, flush=True)
    print("ALL_TRIAL_CSV:", ALL_TRIAL_CSV, flush=True)
    print("n_summary_rows:", len(results_fixed_k_df), flush=True)
    print("n_trial_rows:", n_trial_rows, flush=True)
