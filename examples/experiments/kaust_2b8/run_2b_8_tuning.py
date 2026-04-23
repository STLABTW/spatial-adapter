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
    global cov_frob_from_field, fit_adapter_reconstruct_all_times
    global new_trend_basis, predict_all_simple, rmse_on_time_subset
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
        cov_frob_from_field,
        fit_adapter_reconstruct_all_times,
        new_trend_basis,
        predict_all_simple,
        rmse_on_time_subset,
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "/home/wangxc1117/study-DeepKriging/Space-Time.DeepKriging/simulation_2b-8/data"
        ),
    )
    parser.add_argument("--full-file", default="2b_8.csv")
    parser.add_argument(
        "--result-dir", type=Path, default=THIS_FILE.parent / "outputs" / "2b_8"
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--space-ratio-keep", type=float, default=0.1)
    parser.add_argument("--train-ratio-time", type=float, default=0.1)
    parser.add_argument("--val-ratio-time", type=float, default=0.1)
    parser.add_argument("--test-ratio-time", type=float, default=0.8)
    parser.add_argument("--gna-batch-size", type=int, default=64)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--tau-min", type=float, default=1e-8)
    parser.add_argument("--tau-max", type=float, default=1e4)
    parser.add_argument("--k-fixed", type=int, default=80)
    parser.add_argument("--n-runs-fixed-k", type=int, default=100)
    parser.add_argument(
        "--tuning-target", choices=["rmse", "covfrob", "sv_score"], default="covfrob"
    )
    parser.add_argument("--semivar-estimator", default="matheron")
    parser.add_argument("--semivar-weighted", action="store_true", default=True)
    parser.add_argument(
        "--semivar-not-weighted", dest="semivar_weighted", action="store_false"
    )
    parser.add_argument("--semivar-normalized", action="store_true", default=False)
    return parser.parse_args()


def build_adapter_config(gna_batch_size):
    return SpatialAdapterConfig(
        admm=ADMMConfig(
            rho=1.0, dual_momentum=0.2, max_iters=3000, min_outer=20, tol=1e-4
        ),
        training=TrainingConfig(
            lr_mu=1e-2, batch_size=gna_batch_size, pretrain_epochs=5
        ),
        basis=BasisConfig(phi_every=5, phi_freeze=200),
    )


def build_paths(args):
    repeat_dir = args.result_dir / (
        f"{args.tuning_target}_tuning"
        f"/k_{args.k_fixed}_fixedspace{args.space_ratio_keep}"
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


def run_once_fixed_k(args, paths, device, adapter_config, run_seed):
    seed_everything(run_seed)
    df_full = pd.read_csv(args.data_dir / args.full_file)[["x", "y", "t", "z"]]
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
        _n_times_run,
    ) = build_contiguous_time_splits(
        df=df_run,
        train_ratio=args.train_ratio_time,
        val_ratio=args.val_ratio_time,
        test_ratio=args.test_ratio_time,
    )

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
    stdk_run = train_simple_loop(stdk_run, train_loader_run, device, stdk_config)
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
    val_rmse_unreg_run = rmse_on_time_subset(
        y_stdk_run, y_true_run, residual_full_unreg_run, val_time_idx_run
    )
    covfrob_stdk_train_base = cov_frob_from_field(
        y_stdk_run[train_time_idx_run, :], locs_run
    )
    covfrob_unreg_train_base = cov_frob_from_field(
        y_final_unreg_run[train_time_idx_run, :], locs_run
    )
    covfrob_stdk_val_base = cov_frob_from_field(
        y_stdk_run[val_time_idx_run, :], locs_run
    )
    covfrob_unreg_val_base = cov_frob_from_field(
        y_final_unreg_run[val_time_idx_run, :], locs_run
    )
    covfrob_stdk_test_base = cov_frob_from_field(
        y_stdk_run[test_time_idx_run, :], locs_run
    )
    covfrob_unreg_test_base = cov_frob_from_field(
        y_final_unreg_run[test_time_idx_run, :], locs_run
    )
    float(
        sv_match_loss_for_prediction_matrix(
            locs_run,
            y_true_run,
            y_stdk_run,
            train_time_idx_run,
            semivar_bin_edges,
            args.semivar_estimator,
            args.semivar_weighted,
            args.semivar_normalized,
        )["loss"]
    )
    float(
        sv_match_loss_for_prediction_matrix(
            locs_run,
            y_true_run,
            y_final_unreg_run,
            train_time_idx_run,
            semivar_bin_edges,
            args.semivar_estimator,
            args.semivar_weighted,
            args.semivar_normalized,
        )["loss"]
    )
    float(
        sv_match_loss_for_prediction_matrix(
            locs_run,
            y_true_run,
            y_stdk_run,
            val_time_idx_run,
            semivar_bin_edges,
            args.semivar_estimator,
            args.semivar_weighted,
            args.semivar_normalized,
        )["loss"]
    )
    float(
        sv_match_loss_for_prediction_matrix(
            locs_run,
            y_true_run,
            y_final_unreg_run,
            val_time_idx_run,
            semivar_bin_edges,
            args.semivar_estimator,
            args.semivar_weighted,
            args.semivar_normalized,
        )["loss"]
    )
    float(
        sv_match_loss_for_prediction_matrix(
            locs_run,
            y_true_run,
            y_stdk_run,
            test_time_idx_run,
            semivar_bin_edges,
            args.semivar_estimator,
            args.semivar_weighted,
            args.semivar_normalized,
        )["loss"]
    )
    float(
        sv_match_loss_for_prediction_matrix(
            locs_run,
            y_true_run,
            y_final_unreg_run,
            test_time_idx_run,
            semivar_bin_edges,
            args.semivar_estimator,
            args.semivar_weighted,
            args.semivar_normalized,
        )["loss"]
    )

    trial_cache = {}
    trial_rows = []

    def objective_run(trial):
        tau1 = trial.suggest_float("tau1", args.tau_min, args.tau_max, log=True)
        tau2 = trial.suggest_float("tau2", args.tau_min, args.tau_max, log=True)
        writer_reg = SummaryWriter(
            str(
                paths["repeat_dir"]
                / "logs_reg"
                / f"seed_{run_seed}"
                / f"reg_trial_{trial.number:03d}_tau1_{tau1:.2e}_tau2_{tau2:.2e}"
            )
        )
        trend_trial, basis_trial = new_trend_basis(N_run, args.k_fixed)
        fit_reg = fit_adapter_reconstruct_all_times(
            tag=f"reg_trial_{trial.number:03d}_tau1_{tau1:.2e}_tau2_{tau2:.2e}",
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
            writer=writer_reg,
        )
        writer_reg.close()

        pred_all = fit_reg["pred_all"]
        diag_train = fit_reg["diag_train"]
        diag_all = fit_reg["diag_all"]
        phi_trial = fit_reg["phi"]
        train_rmse = rmse_on_time_subset(
            y_stdk_run, y_true_run, pred_all, train_time_idx_run
        )
        val_rmse = rmse_on_time_subset(
            y_stdk_run, y_true_run, pred_all, val_time_idx_run
        )
        y_final_reg = y_stdk_run + pred_all
        covfrob_reg_train = cov_frob_from_field(
            y_final_reg[train_time_idx_run, :], locs_run
        )
        covfrob_reg_val = cov_frob_from_field(
            y_final_reg[val_time_idx_run, :], locs_run
        )
        covfrob_reg_test = cov_frob_from_field(
            y_final_reg[test_time_idx_run, :], locs_run
        )
        sv_loss_train = float(
            sv_match_loss_for_prediction_matrix(
                locs_run,
                y_true_run,
                y_final_reg,
                train_time_idx_run,
                semivar_bin_edges,
                args.semivar_estimator,
                args.semivar_weighted,
                args.semivar_normalized,
            )["loss"]
        )
        sv_loss_val = float(
            sv_match_loss_for_prediction_matrix(
                locs_run,
                y_true_run,
                y_final_reg,
                val_time_idx_run,
                semivar_bin_edges,
                args.semivar_estimator,
                args.semivar_weighted,
                args.semivar_normalized,
            )["loss"]
        )
        sv_loss_test = float(
            sv_match_loss_for_prediction_matrix(
                locs_run,
                y_true_run,
                y_final_reg,
                test_time_idx_run,
                semivar_bin_edges,
                args.semivar_estimator,
                args.semivar_weighted,
                args.semivar_normalized,
            )["loss"]
        )
        objective_value = choose_objective_value(
            args.tuning_target, val_rmse, covfrob_reg_val, sv_loss_val
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
        trial_cache[trial.number] = {
            "tau1": float(tau1),
            "tau2": float(tau2),
            "pred_all": pred_all,
            "diag_train": diag_train,
            "diag_all": diag_all,
            "phi_path": str(phi_path),
            "objective_value": float(objective_value),
        }
        trial_rows.append(
            {
                "seed": int(run_seed),
                "trial": int(trial.number),
                "tau1": float(tau1),
                "tau2": float(tau2),
                "log10_tau1": float(np.log10(tau1)),
                "log10_tau2": float(np.log10(tau2)),
                "train_rmse": float(train_rmse),
                "val_rmse": float(val_rmse),
                "sv_loss_train": float(sv_loss_train),
                "sv_loss_val": float(sv_loss_val),
                "sv_loss_test": float(sv_loss_test),
                "objective_value": float(objective_value),
                "covfrob_stdk_train": float(covfrob_stdk_train_base),
                "covfrob_unreg_train": float(covfrob_unreg_train_base),
                "covfrob_reg_train": float(covfrob_reg_train),
                "covfrob_stdk_val": float(covfrob_stdk_val_base),
                "covfrob_unreg_val": float(covfrob_unreg_val_base),
                "covfrob_reg_val": float(covfrob_reg_val),
                "covfrob_stdk_test": float(covfrob_stdk_test_base),
                "covfrob_unreg_test": float(covfrob_unreg_test_base),
                "covfrob_reg_test": float(covfrob_reg_test),
                "cov_gain_reg_train": float(
                    covfrob_stdk_train_base - covfrob_reg_train
                ),
                "cov_gain_reg_val": float(covfrob_stdk_val_base - covfrob_reg_val),
                "cov_gain_reg_test": float(covfrob_stdk_test_base - covfrob_reg_test),
                "recon_mse_train": float(diag_train["recon_mse"]),
                "smooth_penalty_train": float(diag_train["smooth_penalty"]),
                "l1_penalty_train": float(diag_train["l1_penalty"]),
                "total_surrogate_train": float(diag_train["total_surrogate"]),
                "n_locations": int(diag_train["n_locations"]),
                "k_basis": int(diag_train["k_basis"]),
                "smooth_penalty_per_entry_train": float(
                    diag_train["smooth_penalty_per_entry"]
                ),
                "l1_penalty_per_entry_train": float(diag_train["l1_penalty_per_entry"]),
                "smooth_over_recon_train": float(diag_train["smooth_over_recon"]),
                "l1_over_recon_train": float(diag_train["l1_over_recon"]),
                "recon_mse_all": float(diag_all["recon_mse"]),
                "smooth_penalty_all": float(diag_all["smooth_penalty"]),
                "l1_penalty_all": float(diag_all["l1_penalty"]),
                "total_surrogate_all": float(diag_all["total_surrogate"]),
                "smooth_penalty_per_entry_all": float(
                    diag_all["smooth_penalty_per_entry"]
                ),
                "l1_penalty_per_entry_all": float(diag_all["l1_penalty_per_entry"]),
                "smooth_over_recon_all": float(diag_all["smooth_over_recon"]),
                "l1_over_recon_all": float(diag_all["l1_over_recon"]),
                "phi_path": str(phi_path),
            }
        )
        return objective_value

    study_run = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=run_seed + 2026)
    )
    study_run.optimize(objective_run, n_trials=args.n_trials, n_jobs=1)
    trial_df = pd.DataFrame(trial_rows).sort_values("trial").reset_index(drop=True)
    trial_df.to_csv(paths["trial_dir"] / f"seed_{run_seed}_trials.csv", index=False)

    best_run = study_run.best_trial
    tau1_best_run = float(best_run.params["tau1"])
    tau2_best_run = float(best_run.params["tau2"])
    best_objective_value_run = float(best_run.value)
    residual_full_reg_best_run = trial_cache[best_run.number]["pred_all"]
    diag_train_reg_best_run = trial_cache[best_run.number]["diag_train"]
    diag_all_reg_best_run = trial_cache[best_run.number]["diag_all"]
    phi_reg_best_run = np.load(trial_cache[best_run.number]["phi_path"])["phi"].astype(
        np.float32
    )

    np.savez_compressed(
        seed_phi_dir / "reg_best_phi.npz",
        phi=phi_reg_best_run.astype(np.float32),
        tau1=np.array([tau1_best_run], dtype=np.float64),
        tau2=np.array([tau2_best_run], dtype=np.float64),
        seed=np.array([run_seed], dtype=np.int32),
        trial=np.array([best_run.number], dtype=np.int32),
    )

    y_final_reg_best_run = y_stdk_run + residual_full_reg_best_run
    train_mask_run = np.zeros((T_run, N_run), dtype=bool)
    train_mask_run[train_time_idx_run, :] = True
    val_mask_run = np.zeros((T_run, N_run), dtype=bool)
    val_mask_run[val_time_idx_run, :] = True
    test_mask_run = np.zeros((T_run, N_run), dtype=bool)
    test_mask_run[test_time_idx_run, :] = True
    full_mask_run = np.ones((T_run, N_run), dtype=bool)

    stdk_full_run = rmse_pooled(y_true_run, y_stdk_run, full_mask_run)
    unreg_full_run = rmse_pooled(y_true_run, y_final_unreg_run, full_mask_run)
    reg_best_full_run = rmse_pooled(y_true_run, y_final_reg_best_run, full_mask_run)
    stdk_train_run = rmse_pooled(y_true_run, y_stdk_run, train_mask_run)
    unreg_train_run = rmse_pooled(y_true_run, y_final_unreg_run, train_mask_run)
    reg_best_train_run = rmse_pooled(y_true_run, y_final_reg_best_run, train_mask_run)
    stdk_val_run = rmse_pooled(y_true_run, y_stdk_run, val_mask_run)
    unreg_val_run = rmse_pooled(y_true_run, y_final_unreg_run, val_mask_run)
    reg_best_val_run = rmse_pooled(y_true_run, y_final_reg_best_run, val_mask_run)
    stdk_test_run = rmse_pooled(y_true_run, y_stdk_run, test_mask_run)
    unreg_test_run = rmse_pooled(y_true_run, y_final_unreg_run, test_mask_run)
    reg_best_test_run = rmse_pooled(y_true_run, y_final_reg_best_run, test_mask_run)
    covfrob_stdk_full_run = cov_frob_from_field(y_stdk_run, locs_run)
    covfrob_unreg_full_run = cov_frob_from_field(y_final_unreg_run, locs_run)
    covfrob_reg_best_full_run = cov_frob_from_field(y_final_reg_best_run, locs_run)
    covfrob_stdk_train_run = cov_frob_from_field(
        y_stdk_run[train_time_idx_run, :], locs_run
    )
    covfrob_unreg_train_run = cov_frob_from_field(
        y_final_unreg_run[train_time_idx_run, :], locs_run
    )
    covfrob_reg_best_train_run = cov_frob_from_field(
        y_final_reg_best_run[train_time_idx_run, :], locs_run
    )
    covfrob_stdk_val_run = cov_frob_from_field(
        y_stdk_run[val_time_idx_run, :], locs_run
    )
    covfrob_unreg_val_run = cov_frob_from_field(
        y_final_unreg_run[val_time_idx_run, :], locs_run
    )
    covfrob_reg_best_val_run = cov_frob_from_field(
        y_final_reg_best_run[val_time_idx_run, :], locs_run
    )
    covfrob_stdk_test_run = cov_frob_from_field(
        y_stdk_run[test_time_idx_run, :], locs_run
    )
    covfrob_unreg_test_run = cov_frob_from_field(
        y_final_unreg_run[test_time_idx_run, :], locs_run
    )
    covfrob_reg_best_test_run = cov_frob_from_field(
        y_final_reg_best_run[test_time_idx_run, :], locs_run
    )
    sv_loss_stdk_train_run = float(
        sv_match_loss_for_prediction_matrix(
            locs_run,
            y_true_run,
            y_stdk_run,
            train_time_idx_run,
            semivar_bin_edges,
            args.semivar_estimator,
            args.semivar_weighted,
            args.semivar_normalized,
        )["loss"]
    )
    sv_loss_unreg_train_run = float(
        sv_match_loss_for_prediction_matrix(
            locs_run,
            y_true_run,
            y_final_unreg_run,
            train_time_idx_run,
            semivar_bin_edges,
            args.semivar_estimator,
            args.semivar_weighted,
            args.semivar_normalized,
        )["loss"]
    )
    sv_loss_reg_best_train_run = float(
        sv_match_loss_for_prediction_matrix(
            locs_run,
            y_true_run,
            y_final_reg_best_run,
            train_time_idx_run,
            semivar_bin_edges,
            args.semivar_estimator,
            args.semivar_weighted,
            args.semivar_normalized,
        )["loss"]
    )
    sv_loss_stdk_val_run = float(
        sv_match_loss_for_prediction_matrix(
            locs_run,
            y_true_run,
            y_stdk_run,
            val_time_idx_run,
            semivar_bin_edges,
            args.semivar_estimator,
            args.semivar_weighted,
            args.semivar_normalized,
        )["loss"]
    )
    sv_loss_unreg_val_run = float(
        sv_match_loss_for_prediction_matrix(
            locs_run,
            y_true_run,
            y_final_unreg_run,
            val_time_idx_run,
            semivar_bin_edges,
            args.semivar_estimator,
            args.semivar_weighted,
            args.semivar_normalized,
        )["loss"]
    )
    sv_loss_reg_best_val_run = float(
        sv_match_loss_for_prediction_matrix(
            locs_run,
            y_true_run,
            y_final_reg_best_run,
            val_time_idx_run,
            semivar_bin_edges,
            args.semivar_estimator,
            args.semivar_weighted,
            args.semivar_normalized,
        )["loss"]
    )
    sv_loss_stdk_test_run = float(
        sv_match_loss_for_prediction_matrix(
            locs_run,
            y_true_run,
            y_stdk_run,
            test_time_idx_run,
            semivar_bin_edges,
            args.semivar_estimator,
            args.semivar_weighted,
            args.semivar_normalized,
        )["loss"]
    )
    sv_loss_unreg_test_run = float(
        sv_match_loss_for_prediction_matrix(
            locs_run,
            y_true_run,
            y_final_unreg_run,
            test_time_idx_run,
            semivar_bin_edges,
            args.semivar_estimator,
            args.semivar_weighted,
            args.semivar_normalized,
        )["loss"]
    )
    sv_loss_reg_best_test_run = float(
        sv_match_loss_for_prediction_matrix(
            locs_run,
            y_true_run,
            y_final_reg_best_run,
            test_time_idx_run,
            semivar_bin_edges,
            args.semivar_estimator,
            args.semivar_weighted,
            args.semivar_normalized,
        )["loss"]
    )

    np.savez_compressed(
        paths["pred_dir"] / f"seed_{run_seed}.npz",
        y_true=y_true_run.astype(np.float32),
        y_stdk=y_stdk_run.astype(np.float32),
        y_unreg=y_final_unreg_run.astype(np.float32),
        y_reg_best=y_final_reg_best_run.astype(np.float32),
        phi_unreg=phi_unreg_run.astype(np.float32),
        phi_reg_best=phi_reg_best_run.astype(np.float32),
        train_mask=train_mask_run.astype(bool),
        val_mask=val_mask_run.astype(bool),
        test_mask=test_mask_run.astype(bool),
        full_mask=full_mask_run.astype(bool),
        locs=locs_run.astype(np.float32),
        keep_sites=keep_sites_run.astype(np.int32),
        train_time_idx=train_time_idx_run.astype(np.int32),
        val_time_idx=val_time_idx_run.astype(np.int32),
        test_time_idx=test_time_idx_run.astype(np.int32),
        n_locations=np.array([N_run], dtype=np.int32),
        k_basis=np.array([args.k_fixed], dtype=np.int32),
        tau1_best=np.array([tau1_best_run], dtype=np.float64),
        tau2_best=np.array([tau2_best_run], dtype=np.float64),
        best_objective_value=np.array([best_objective_value_run], dtype=np.float64),
        sv_loss_stdk_train=np.array([sv_loss_stdk_train_run], dtype=np.float64),
        sv_loss_unreg_train=np.array([sv_loss_unreg_train_run], dtype=np.float64),
        sv_loss_reg_best_train=np.array([sv_loss_reg_best_train_run], dtype=np.float64),
        sv_loss_stdk_val=np.array([sv_loss_stdk_val_run], dtype=np.float64),
        sv_loss_unreg_val=np.array([sv_loss_unreg_val_run], dtype=np.float64),
        sv_loss_reg_best_val=np.array([sv_loss_reg_best_val_run], dtype=np.float64),
        sv_loss_stdk_test=np.array([sv_loss_stdk_test_run], dtype=np.float64),
        sv_loss_unreg_test=np.array([sv_loss_unreg_test_run], dtype=np.float64),
        sv_loss_reg_best_test=np.array([sv_loss_reg_best_test_run], dtype=np.float64),
    )

    return {
        "seed": int(run_seed),
        "tuning_target": args.tuning_target,
        "n_sites_full": int(n_sites_full_run),
        "n_sites_kept": int(len(keep_sites_run)),
        "tau1_best": float(tau1_best_run),
        "tau2_best": float(tau2_best_run),
        "best_objective_value": float(best_objective_value_run),
        "val_rmse_unreg": float(val_rmse_unreg_run),
        "stdk_full": float(stdk_full_run),
        "unreg_full": float(unreg_full_run),
        "reg_best_full": float(reg_best_full_run),
        "stdk_train": float(stdk_train_run),
        "unreg_train": float(unreg_train_run),
        "reg_best_train": float(reg_best_train_run),
        "stdk_val": float(stdk_val_run),
        "unreg_val": float(unreg_val_run),
        "reg_best_val": float(reg_best_val_run),
        "stdk_test": float(stdk_test_run),
        "unreg_test": float(unreg_test_run),
        "reg_best_test": float(reg_best_test_run),
        "covfrob_stdk_full": float(covfrob_stdk_full_run),
        "covfrob_unreg_full": float(covfrob_unreg_full_run),
        "covfrob_reg_best_full": float(covfrob_reg_best_full_run),
        "covfrob_stdk_train": float(covfrob_stdk_train_run),
        "covfrob_unreg_train": float(covfrob_unreg_train_run),
        "covfrob_reg_best_train": float(covfrob_reg_best_train_run),
        "covfrob_stdk_val": float(covfrob_stdk_val_run),
        "covfrob_unreg_val": float(covfrob_unreg_val_run),
        "covfrob_reg_best_val": float(covfrob_reg_best_val_run),
        "covfrob_stdk_test": float(covfrob_stdk_test_run),
        "covfrob_unreg_test": float(covfrob_unreg_test_run),
        "covfrob_reg_best_test": float(covfrob_reg_best_test_run),
        "sv_loss_stdk_train": float(sv_loss_stdk_train_run),
        "sv_loss_unreg_train": float(sv_loss_unreg_train_run),
        "sv_loss_reg_best_train": float(sv_loss_reg_best_train_run),
        "sv_loss_stdk_val": float(sv_loss_stdk_val_run),
        "sv_loss_unreg_val": float(sv_loss_unreg_val_run),
        "sv_loss_reg_best_val": float(sv_loss_reg_best_val_run),
        "sv_loss_stdk_test": float(sv_loss_stdk_test_run),
        "sv_loss_unreg_test": float(sv_loss_unreg_test_run),
        "sv_loss_reg_best_test": float(sv_loss_reg_best_test_run),
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


def main():
    if any(arg in sys.argv[1:] for arg in ("-h", "--help")):
        parse_args()
        return
    _load_dependencies()
    args = parse_args()
    paths = build_paths(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter_config = build_adapter_config(args.gna_batch_size)
    rows = []
    for r in range(args.n_runs_fixed_k):
        run_seed = args.seed + r * 1000
        print(
            f"\n===== FIXED K RUN {r + 1}/{args.n_runs_fixed_k} | seed={run_seed} | "
            f"target={args.tuning_target} =====",
            flush=True,
        )
        rows.append(run_once_fixed_k(args, paths, device, adapter_config, run_seed))
    pd.DataFrame(rows).to_csv(paths["summary_csv"], index=False)
    trial_files = sorted(paths["trial_dir"].glob("seed_*_trials.csv"))
    if trial_files:
        all_trial_df = pd.concat(
            [pd.read_csv(path) for path in trial_files], ignore_index=True
        )
        all_trial_df.to_csv(paths["all_trial_csv"], index=False)
    else:
        pd.DataFrame().to_csv(paths["all_trial_csv"], index=False)
    print("SUMMARY_CSV:", paths["summary_csv"], flush=True)
    print("ALL_TRIAL_CSV:", paths["all_trial_csv"], flush=True)


if __name__ == "__main__":
    main()
