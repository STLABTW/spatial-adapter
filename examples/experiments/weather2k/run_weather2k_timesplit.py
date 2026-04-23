import argparse
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

torch.set_default_dtype(torch.float32)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load_dependencies():
    global ADMMConfig, BasisConfig, SpatialAdapterConfig, TrainingConfig
    global create_model
    global DictDataset, build_contiguous_time_splits, build_fixed_location_subset
    global build_standard_bin_edges, build_stdk_model_config, collate_fn
    global cov_frob_observed, fit_adapter_reconstruct_all_times
    global load_weather2k_as_long_df, new_trend_basis, predict_all_simple
    global rmse_pooled, seed_everything, sv_match_loss_for_prediction_matrix
    global train_simple_loop

    from examples.baselines.stdk.st_interp import create_model
    from examples.baselines.timesplit.experiment_core import (
        DictDataset,
        build_contiguous_time_splits,
        build_fixed_location_subset,
        build_standard_bin_edges,
        build_stdk_model_config,
        collate_fn,
        cov_frob_observed,
        fit_adapter_reconstruct_all_times,
        load_weather2k_as_long_df,
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


def parse_args(default_overrides=None):
    default_overrides = {} if default_overrides is None else dict(default_overrides)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weather2k-npy",
        type=Path,
        default=default_overrides.get(
            "weather2k_npy", Path("/home/wangxc1117/Weather2K/weather2k.npy")
        ),
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=default_overrides.get(
            "result_dir", THIS_FILE.parent / "outputs" / "weather2k_timesplit"
        ),
    )
    parser.add_argument(
        "--target-var-idx", type=int, default=default_overrides.get("target_var_idx", 4)
    )
    parser.add_argument(
        "--lat-idx", type=int, default=default_overrides.get("lat_idx", 0)
    )
    parser.add_argument(
        "--lon-idx", type=int, default=default_overrides.get("lon_idx", 1)
    )
    parser.add_argument(
        "--t-keep", type=int, default=default_overrides.get("t_keep", 1000)
    )
    parser.add_argument("--seed", type=int, default=default_overrides.get("seed", 123))
    parser.add_argument(
        "--epochs", type=int, default=default_overrides.get("epochs", 350)
    )
    parser.add_argument(
        "--batch-size", type=int, default=default_overrides.get("batch_size", 512)
    )
    parser.add_argument("--lr", type=float, default=default_overrides.get("lr", 1e-3))
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=default_overrides.get("weight_decay", 1e-5),
    )
    parser.add_argument(
        "--space-ratio-keep",
        type=float,
        default=default_overrides.get("space_ratio_keep", 0.1),
    )
    parser.add_argument(
        "--train-ratio-time",
        type=float,
        default=default_overrides.get("train_ratio_time", 0.8),
    )
    parser.add_argument(
        "--val-ratio-time",
        type=float,
        default=default_overrides.get("val_ratio_time", 0.1),
    )
    parser.add_argument(
        "--test-ratio-time",
        type=float,
        default=default_overrides.get("test_ratio_time", 0.1),
    )
    parser.add_argument(
        "--gna-batch-size",
        type=int,
        default=default_overrides.get("gna_batch_size", 64),
    )
    parser.add_argument(
        "--n-trials", type=int, default=default_overrides.get("n_trials", 50)
    )
    parser.add_argument(
        "--tau-min", type=float, default=default_overrides.get("tau_min", 1e-8)
    )
    parser.add_argument(
        "--tau-max", type=float, default=default_overrides.get("tau_max", 1e4)
    )
    parser.add_argument(
        "--k-fixed", type=int, default=default_overrides.get("k_fixed", 40)
    )
    parser.add_argument(
        "--n-runs-fixed-k",
        type=int,
        default=default_overrides.get("n_runs_fixed_k", 100),
    )
    parser.add_argument(
        "--tuning-target",
        choices=["rmse", "covfrob", "sv_score"],
        default=default_overrides.get("tuning_target", "rmse"),
    )
    parser.add_argument(
        "--semivar-estimator",
        default=default_overrides.get("semivar_estimator", "matheron"),
    )
    parser.add_argument(
        "--semivar-weighted",
        action="store_true",
        default=default_overrides.get("semivar_weighted", True),
    )
    parser.add_argument(
        "--semivar-not-weighted", dest="semivar_weighted", action="store_false"
    )
    parser.add_argument("--semivar-normalized", action="store_true", default=False)
    parser.add_argument("--skip-final-table", action="store_true", default=False)
    return parser.parse_args()


def build_adapter_config(gna_batch_size):
    return SpatialAdapterConfig(
        admm=ADMMConfig(
            rho=1.0,
            dual_momentum=0.2,
            max_iters=3000,
            min_outer=20,
            tol=1e-4,
        ),
        training=TrainingConfig(
            lr_mu=1e-2,
            batch_size=gna_batch_size,
            pretrain_epochs=5,
        ),
        basis=BasisConfig(
            phi_every=5,
            phi_freeze=200,
        ),
    )


def build_paths(args):
    repeat_dir = args.result_dir / (
        f"{args.tuning_target}_tuning"
        f"/var{args.target_var_idx}_tkeep{args.t_keep}"
        f"_k_{args.k_fixed}_fixedspace{args.space_ratio_keep}"
        f"_time_train{args.train_ratio_time}_val{args.val_ratio_time}_test{args.test_ratio_time}"
    )
    pred_dir = repeat_dir / "saved_predictions"
    trial_dir = repeat_dir / "trial_results"
    phi_dir = repeat_dir / "saved_phi"

    repeat_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    trial_dir.mkdir(parents=True, exist_ok=True)
    phi_dir.mkdir(parents=True, exist_ok=True)

    return {
        "repeat_dir": repeat_dir,
        "summary_csv": repeat_dir / f"k_{args.k_fixed}_repeat_runs_summary.csv",
        "all_trial_csv": trial_dir / "all_trial_results.csv",
        "pred_dir": pred_dir,
        "trial_dir": trial_dir,
        "phi_dir": phi_dir,
    }


def choose_objective_value(tuning_target, val_rmse, covfrob_reg_val, sv_loss_val):
    if tuning_target == "rmse":
        return float(val_rmse)
    if tuning_target == "covfrob":
        return float(covfrob_reg_val)
    if tuning_target == "sv_score":
        return float(sv_loss_val)
    raise ValueError(f"Unknown TUNING_TARGET: {tuning_target}")


def objective_label(tuning_target):
    if tuning_target == "rmse":
        return "best_val_rmse"
    if tuning_target == "covfrob":
        return "best_val_covfrob"
    if tuning_target == "sv_score":
        return "best_val_sv_score"
    raise ValueError(f"Unknown TUNING_TARGET: {tuning_target}")


def split_label(args):
    return f"{int(round(args.train_ratio_time * 100))}_{int(round(args.val_ratio_time * 100))}_{int(round(args.test_ratio_time * 100))}"


def rmse_recalc(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return (
        float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))
        if np.any(mask)
        else float("nan")
    )


def mae_recalc(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return (
        float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
        if np.any(mask)
        else float("nan")
    )


def r2_recalc(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")


def mean_se(series):
    s = pd.Series(series).dropna().astype(float)
    return float(s.mean()), float(s.sem())


def fmt_mean_se(series, digits=6, scale=1.0):
    mean, se = mean_se(pd.Series(series) * scale)
    return f"{mean:.{digits}f} ({se:.{digits}f})"


def write_final_table(args, paths):
    summary_csv = paths["summary_csv"]
    pred_dir = paths["pred_dir"]
    if not summary_csv.exists() or not pred_dir.exists():
        return

    summary_df = pd.read_csv(summary_csv)
    npz_files = sorted(pred_dir.glob("seed_*.npz"))
    if not npz_files:
        return

    rows = []
    for fp in npz_files:
        data = np.load(fp)
        seed = int(data["seed"][0])
        test_idx = data["test_time_idx_run"]
        y_true = data["y_true_raw_run"]
        y_stdk = data["y_stdk_raw_run"]
        y_unreg = data["y_final_unreg_raw_run"]
        y_reg = data["y_final_reg_best_raw_run"]
        rows.append(
            {
                "seed": seed,
                "stdk_rmse_test_raw_recalc": rmse_recalc(
                    y_true[test_idx], y_stdk[test_idx]
                ),
                "unreg_rmse_test_raw_recalc": rmse_recalc(
                    y_true[test_idx], y_unreg[test_idx]
                ),
                "reg_best_rmse_test_raw_recalc": rmse_recalc(
                    y_true[test_idx], y_reg[test_idx]
                ),
                "stdk_mae_test": mae_recalc(y_true[test_idx], y_stdk[test_idx]),
                "unreg_mae_test": mae_recalc(y_true[test_idx], y_unreg[test_idx]),
                "reg_best_mae_test": mae_recalc(y_true[test_idx], y_reg[test_idx]),
                "stdk_r2_test": r2_recalc(y_true[test_idx], y_stdk[test_idx]),
                "unreg_r2_test": r2_recalc(y_true[test_idx], y_unreg[test_idx]),
                "reg_best_r2_test": r2_recalc(y_true[test_idx], y_reg[test_idx]),
                "stdk_covfrob_test_raw_recalc": cov_frob_observed(
                    y_true[test_idx], y_stdk[test_idx]
                ),
                "unreg_covfrob_test_raw_recalc": cov_frob_observed(
                    y_true[test_idx], y_unreg[test_idx]
                ),
                "reg_best_covfrob_test_raw_recalc": cov_frob_observed(
                    y_true[test_idx], y_reg[test_idx]
                ),
            }
        )

    extra_df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    merged_df = (
        pd.merge(summary_df, extra_df, on="seed", how="inner")
        .sort_values("seed")
        .reset_index(drop=True)
    )

    current_split_label = split_label(args).replace("_", "/")
    result_table = pd.DataFrame(
        {
            "Split": [current_split_label, current_split_label, current_split_label],
            "Model": ["STDK", "Adapter (unreg.)", "Adapter (reg.)"],
            "RMSE": [
                fmt_mean_se(merged_df["stdk_rmse_test_raw"]),
                fmt_mean_se(merged_df["unreg_rmse_test_raw"]),
                fmt_mean_se(merged_df["reg_best_rmse_test_raw"]),
            ],
            "MAE": [
                fmt_mean_se(merged_df["stdk_mae_test"]),
                fmt_mean_se(merged_df["unreg_mae_test"]),
                fmt_mean_se(merged_df["reg_best_mae_test"]),
            ],
            "R2_percent": [
                fmt_mean_se(merged_df["stdk_r2_test"], scale=100.0),
                fmt_mean_se(merged_df["unreg_r2_test"], scale=100.0),
                fmt_mean_se(merged_df["reg_best_r2_test"], scale=100.0),
            ],
            "CovFrob": [
                fmt_mean_se(merged_df["stdk_covfrob_test"]),
                fmt_mean_se(merged_df["unreg_covfrob_test"]),
                fmt_mean_se(merged_df["reg_best_covfrob_test"]),
            ],
        }
    )

    out_csv = paths["repeat_dir"] / f"weather2k_{split_label(args)}_table_values.csv"
    result_table.to_csv(out_csv, index=False)
    print("\n=== Final table values ===", flush=True)
    print(result_table.to_string(index=False), flush=True)
    print("FINAL_TABLE_CSV:", out_csv, flush=True)


def run_once_fixed_k(args, paths, device, adapter_config, run_seed):
    seed_everything(run_seed)

    df_full = load_weather2k_as_long_df(
        npy_path=args.weather2k_npy,
        target_var_idx=args.target_var_idx,
        lat_idx=args.lat_idx,
        lon_idx=args.lon_idx,
        t_keep=args.t_keep,
        normalize_xy=True,
    )

    df_run, keep_sites_run, n_sites_full_run = build_fixed_location_subset(
        df=df_full,
        keep_ratio=args.space_ratio_keep,
        seed=run_seed + 11111,
    )

    df_run["t_norm"] = (
        (df_run["t"] - df_run["t"].min())
        / (df_run["t"].max() - df_run["t"].min() + 1e-12)
    ).astype(np.float32)

    (
        train_mask_flat_run,
        _val_mask_flat_run,
        _test_mask_flat_run,
        train_time_idx_run,
        val_time_idx_run,
        test_time_idx_run,
        uniq_t_run,
        n_times_run,
    ) = build_contiguous_time_splits(
        df=df_run,
        train_ratio=args.train_ratio_time,
        val_ratio=args.val_ratio_time,
        test_ratio=args.test_ratio_time,
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

    def to_raw(arr_std):
        return arr_std * z_sd_run + z_mean_run

    coords_all_run = df_run[["x", "y"]].to_numpy(np.float32)
    t_all_run = df_run["t_norm"].to_numpy(np.float32).reshape(-1, 1)
    y_all_run = df_run["z"].to_numpy(np.float32).reshape(-1, 1)
    X_all_run = np.empty((df_run.shape[0], 0), dtype=np.float32)

    train_dataset_run = DictDataset(
        torch.from_numpy(X_all_run[train_mask_flat_run]),
        torch.from_numpy(coords_all_run[train_mask_flat_run]),
        torch.from_numpy(t_all_run[train_mask_flat_run]),
        torch.from_numpy(y_all_run[train_mask_flat_run]),
    )

    generator = torch.Generator()
    generator.manual_seed(run_seed + 1000)

    train_loader_run = DataLoader(
        train_dataset_run,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    stdk_config = build_stdk_model_config(
        EPOCHS=args.epochs,
        LR=args.lr,
        WEIGHT_DECAY=args.weight_decay,
        BATCH_SIZE=args.batch_size,
    )
    stdk_run = create_model(
        stdk_config, train_coords=coords_all_run[train_mask_flat_run]
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
        batch_size=args.batch_size,
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
        batch_size=min(args.gna_batch_size, len(train_time_idx_run)),
        shuffle=True,
        drop_last=False,
    )

    residual_true_tensor_all_run = torch.from_numpy(residual_true_run).float()
    seed_phi_dir = paths["phi_dir"] / f"seed_{run_seed}"
    seed_phi_dir.mkdir(parents=True, exist_ok=True)
    semivar_bin_edges = build_standard_bin_edges(locs_run)
    full_time_idx_run = np.arange(T_run, dtype=int)

    def rmse_std(y_pred_std, time_idx):
        mask = np.isfinite(y_true_run[time_idx, :]) & np.isfinite(
            y_pred_std[time_idx, :]
        )
        return rmse_pooled(y_true_run[time_idx, :], y_pred_std[time_idx, :], mask)

    def rmse_raw(y_pred_std, time_idx):
        y_pred_raw = to_raw(y_pred_std[time_idx, :])
        mask = np.isfinite(y_true_raw_run[time_idx, :]) & np.isfinite(y_pred_raw)
        return rmse_pooled(y_true_raw_run[time_idx, :], y_pred_raw, mask)

    def covfrob_std(y_pred_std, time_idx):
        return cov_frob_observed(y_true_run[time_idx, :], y_pred_std[time_idx, :])

    def covfrob_raw(y_pred_std, time_idx):
        return cov_frob_observed(
            y_true_raw_run[time_idx, :], to_raw(y_pred_std[time_idx, :])
        )

    def sv_loss_std(y_pred_std, time_idx):
        out = sv_match_loss_for_prediction_matrix(
            coords=locs_run,
            y_true=y_true_run,
            y_pred=y_pred_std,
            time_idx=time_idx,
            bin_edges=semivar_bin_edges,
            estimator=args.semivar_estimator,
            weighted=args.semivar_weighted,
            normalized=args.semivar_normalized,
        )
        return float(out["loss"])

    def sv_loss_raw(y_pred_std, time_idx):
        out = sv_match_loss_for_prediction_matrix(
            coords=locs_run,
            y_true=y_true_raw_run,
            y_pred=to_raw(y_pred_std),
            time_idx=time_idx,
            bin_edges=semivar_bin_edges,
            estimator=args.semivar_estimator,
            weighted=args.semivar_weighted,
            normalized=args.semivar_normalized,
        )
        return float(out["loss"])

    writer_unreg = SummaryWriter(
        str(
            paths["repeat_dir"]
            / "logs_unreg"
            / f"seed_{run_seed}"
            / "unreg_tau1_0_tau2_0"
        )
    )
    trend_unreg, basis_unreg = new_trend_basis(N_run, args.k_fixed)
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
        config=adapter_config,
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

    def objective(trial):
        tau1 = trial.suggest_float("tau1", args.tau_min, args.tau_max, log=True)
        tau2 = trial.suggest_float("tau2", args.tau_min, args.tau_max, log=True)
        tag = f"trial_{trial.number:03d}_tau1_{tau1:.3e}_tau2_{tau2:.3e}"
        writer_trial = SummaryWriter(
            str(paths["repeat_dir"] / "logs_reg" / f"seed_{run_seed}" / tag)
        )

        trend_trial, basis_trial = new_trend_basis(N_run, args.k_fixed)
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
            config=adapter_config,
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
            args.tuning_target,
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

        trial_rows.append(
            {
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
                "l1_penalty_per_entry_all": float(
                    diag_all_trial["l1_penalty_per_entry"]
                ),
                "smooth_over_recon_all": float(diag_all_trial["smooth_over_recon"]),
                "l1_over_recon_all": float(diag_all_trial["l1_over_recon"]),
                "phi_path": str(phi_path),
            }
        )

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
            f"[seed {run_seed}] trial {trial.number + 1:03d}/{args.n_trials:03d} | "
            f"target={args.tuning_target} | tau1={tau1:.3e} | tau2={tau2:.3e} | "
            f"val_rmse={val_rmse_std:.6f} | objective={objective_value:.6f}",
            flush=True,
        )
        return float(objective_value)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    trial_df_run = pd.DataFrame(trial_rows)
    trial_df_run.to_csv(paths["trial_dir"] / f"seed_{run_seed}_trials.csv", index=False)

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

    def collect_metrics(prefix, y_pred_std):
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
        objective_label(args.tuning_target): float(best_objective_run),
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
        paths["pred_dir"] / f"seed_{run_seed}.npz",
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
        f"test_rmse={summary_row['reg_best_rmse_test_raw']:.6f}",
        flush=True,
    )
    return summary_row


def main(default_overrides=None):
    if any(arg in sys.argv[1:] for arg in ("-h", "--help")):
        parse_args(default_overrides=default_overrides)
        return
    _load_dependencies()
    args = parse_args(default_overrides=default_overrides)
    seed_everything(args.seed)
    paths = build_paths(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter_config = build_adapter_config(args.gna_batch_size)

    rows = []
    for run_idx in range(args.n_runs_fixed_k):
        run_seed = args.seed + run_idx * 1000
        print(
            f"\n===== FIXED K RUN {run_idx + 1}/{args.n_runs_fixed_k} | "
            f"seed={run_seed} | target={args.tuning_target} | "
            f"time=({args.train_ratio_time}, {args.val_ratio_time}, {args.test_ratio_time}) =====",
            flush=True,
        )
        rows.append(run_once_fixed_k(args, paths, device, adapter_config, run_seed))

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(paths["summary_csv"], index=False)

    trial_files = sorted(paths["trial_dir"].glob("seed_*_trials.csv"))
    if trial_files:
        all_trial_df = pd.concat(
            [pd.read_csv(path) for path in trial_files], ignore_index=True
        )
        all_trial_df.to_csv(paths["all_trial_csv"], index=False)
    else:
        pd.DataFrame().to_csv(paths["all_trial_csv"], index=False)

    print("\n=== Summary saved ===", flush=True)
    print("SUMMARY_CSV:", paths["summary_csv"], flush=True)
    print("ALL_TRIAL_CSV:", paths["all_trial_csv"], flush=True)
    if not args.skip_final_table:
        write_final_table(args, paths)


if __name__ == "__main__":
    main()
