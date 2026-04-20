import random
from pathlib import Path

import gstools as gs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import gamma, kv
from torch.utils.data import DataLoader, Dataset

from spatial_adapter.metrics import rmse_pooled
from spatial_adapter.models.spatial_basis_learner import SpatialBasisLearner
from spatial_adapter.models.spatial_adapter import SpatialNeuralAdapter
from spatial_adapter.models.trend_model import TrendModel

# STDK was relocated out of the spatial_adapter package to
# examples/baselines/stdk/ so that the library itself has no baseline
# dependencies.  Use a relative import here so this file stays
# location-agnostic: as long as ``examples.baselines.timesplit`` and
# ``examples.baselines.stdk`` are siblings in the same package tree,
# the import resolves correctly regardless of how the package is
# rooted on sys.path.
from ..stdk.losses import quantile_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_times_contiguous(n_times, train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError("Ratios must sum to 1.")

    n_train = int(np.round(train_ratio * n_times))
    n_val = int(np.round(val_ratio * n_times))

    n_train = max(1, min(n_train, n_times - 2))
    n_val = max(1, min(n_val, n_times - n_train - 1))
    n_times - n_train - n_val

    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_times)

    return train_idx, val_idx, test_idx


def build_fixed_location_subset(df, keep_ratio, seed):
    df = df.copy()
    site_pairs = pd.Series(list(zip(df["x"], df["y"])))
    df["site_id"] = pd.factorize(site_pairs)[0]
    n_sites = df["site_id"].nunique()

    rs = np.random.RandomState(seed)
    perm = rs.permutation(n_sites)

    n_keep = int(np.round(keep_ratio * n_sites))
    n_keep = max(1, min(n_keep, n_sites))
    keep_sites = np.sort(perm[:n_keep])

    df_keep = df[df["site_id"].isin(keep_sites)].copy().reset_index(drop=True)
    return df_keep, keep_sites, n_sites


def build_contiguous_time_splits(df, train_ratio, val_ratio, test_ratio):
    uniq_t = np.sort(df["t"].unique())
    n_times = len(uniq_t)

    train_idx, val_idx, test_idx = split_times_contiguous(
        n_times,
        train_ratio,
        val_ratio,
        test_ratio,
    )

    return (
        df["t"].isin(uniq_t[train_idx]).to_numpy(),
        df["t"].isin(uniq_t[val_idx]).to_numpy(),
        df["t"].isin(uniq_t[test_idx]).to_numpy(),
        train_idx,
        val_idx,
        test_idx,
        uniq_t,
        n_times,
    )


def get_subset_time_idx(train_idx, val_idx, test_idx, subset):
    if subset == "train":
        return np.asarray(train_idx, dtype=int)
    if subset == "val":
        return np.asarray(val_idx, dtype=int)
    if subset == "test":
        return np.asarray(test_idx, dtype=int)
    raise ValueError(f"Unknown subset: {subset}")


def rmse_on_time_subset(y_stdk, y_true, pred_all, time_idx):
    y_final = y_stdk[time_idx, :] + pred_all[time_idx, :]
    y_true_sub = y_true[time_idx, :]
    mask = np.ones_like(y_true_sub, dtype=bool)
    return rmse_pooled(y_true_sub, y_final, mask)


def mean_sd_repeat(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(np.mean(x)), 0.0
    return float(np.mean(x)), float(np.std(x, ddof=1))


def fmt_pm(x):
    mu, sd = mean_sd_repeat(x)
    if np.isnan(mu):
        return "nan ± nan"
    return f"{mu:.6f} ± {sd:.6f}"


def matern_corr(r, nu):
    r = np.asarray(r, dtype=float)
    out = np.zeros_like(r)
    out[r < 1e-12] = 1.0
    nz = r >= 1e-12
    rr = r[nz]
    coef = 1.0 / (2.0 ** (nu - 1.0) * gamma(nu))
    out[nz] = coef * (rr**nu) * kv(nu, rr)
    return out


def true_spatial_cov_2b8(coords):
    coords = np.asarray(coords, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    return matern_corr(dists / 0.08, 1.0)


def empirical_spatial_cov(y_pred):
    y_pred = np.asarray(y_pred, dtype=float)
    if y_pred.ndim != 2:
        raise ValueError("y_pred must be 2D with shape (n_time, n_locations)")
    if y_pred.shape[0] < 2:
        return np.full((y_pred.shape[1], y_pred.shape[1]), np.nan, dtype=float)
    yc = y_pred - np.mean(y_pred, axis=0, keepdims=True)
    return (yc.T @ yc) / (y_pred.shape[0] - 1)


def cov_frob_from_field(y_pred, coords):
    sigma_hat = empirical_spatial_cov(y_pred)
    sigma_true = true_spatial_cov_2b8(coords)

    if not np.all(np.isfinite(sigma_hat)):
        return float("nan")

    denom = np.linalg.norm(sigma_true)
    if denom < 1e-12:
        return float("nan")

    return float(np.linalg.norm(sigma_hat - sigma_true) / denom)


def subset_by_time_idx(y, time_idx):
    y = np.asarray(y)
    time_idx = np.asarray(time_idx, dtype=int)
    return y[time_idx, :]


def build_standard_bin_edges(coords):
    coords = np.asarray(coords, dtype=np.float64)
    pos = tuple(coords[:, j] for j in range(coords.shape[1]))
    return gs.variogram.standard_bins(pos=pos)


def semivariogram_replicates(
    coords,
    Y,
    bin_edges=None,
    estimator="matheron",
    return_counts=True,
):
    coords = np.asarray(coords, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if Y.ndim != 2:
        raise ValueError("Y must be 2D with shape (n_time, n_locations)")

    n_time, p = Y.shape
    if coords.shape[0] != p:
        raise ValueError("coords.shape[0] must match Y.shape[1]")

    pos = tuple(coords[:, j] for j in range(coords.shape[1]))

    out = gs.vario_estimate(
        pos=pos,
        field=Y,
        bin_edges=bin_edges,
        estimator=estimator,
        return_counts=return_counts,
    )

    if return_counts:
        lags, gamma_hat, counts = out
        return lags, gamma_hat, counts

    lags, gamma_hat = out
    return lags, gamma_hat, None


def semivariogram_match_loss_replicates(
    coords,
    Y_true,
    Y_pred,
    bin_edges=None,
    estimator="matheron",
    weighted=True,
    normalized=False,
    eps=1e-8,
):
    lags_true, gamma_true, counts_true = semivariogram_replicates(
        coords=coords,
        Y=Y_true,
        bin_edges=bin_edges,
        estimator=estimator,
        return_counts=True,
    )
    lags_pred, gamma_pred, counts_pred = semivariogram_replicates(
        coords=coords,
        Y=Y_pred,
        bin_edges=bin_edges,
        estimator=estimator,
        return_counts=True,
    )

    mask = np.isfinite(gamma_true) & np.isfinite(gamma_pred)

    if counts_true is not None and counts_pred is not None:
        counts = np.minimum(counts_true, counts_pred)
        mask = mask & (counts > 0)
    else:
        counts = np.ones_like(gamma_true, dtype=float)

    if not np.any(mask):
        return {
            "loss": float("nan"),
            "lags": np.array([], dtype=float),
            "gamma_true": np.array([], dtype=float),
            "gamma_pred": np.array([], dtype=float),
            "counts": np.array([], dtype=float),
        }

    if normalized:
        diff = (gamma_pred[mask] - gamma_true[mask]) / (gamma_true[mask] + eps)
    else:
        diff = gamma_pred[mask] - gamma_true[mask]

    if weighted:
        denom = np.sum(counts[mask])
        loss = np.sum(counts[mask] * diff**2) / max(denom, eps)
    else:
        loss = np.mean(diff**2)

    return {
        "loss": float(loss),
        "lags": lags_true[mask],
        "gamma_true": gamma_true[mask],
        "gamma_pred": gamma_pred[mask],
        "counts": counts[mask],
    }


def sv_match_loss_for_prediction_matrix(
    coords,
    y_true,
    y_pred,
    time_idx,
    bin_edges=None,
    estimator="matheron",
    weighted=True,
    normalized=False,
    eps=1e-8,
):
    y_true_sub = subset_by_time_idx(y_true, time_idx)
    y_pred_sub = subset_by_time_idx(y_pred, time_idx)

    if bin_edges is None:
        bin_edges = build_standard_bin_edges(coords)

    return semivariogram_match_loss_replicates(
        coords=coords,
        Y_true=y_true_sub,
        Y_pred=y_pred_sub,
        bin_edges=bin_edges,
        estimator=estimator,
        weighted=weighted,
        normalized=normalized,
        eps=eps,
    )


def sv_match_loss_on_final_prediction(
    coords,
    y_true,
    y_stdk,
    pred_all,
    time_idx,
    bin_edges=None,
    estimator="matheron",
    weighted=True,
    normalized=False,
    eps=1e-8,
):
    y_pred_final = y_stdk + pred_all
    return sv_match_loss_for_prediction_matrix(
        coords=coords,
        y_true=y_true,
        y_pred=y_pred_final,
        time_idx=time_idx,
        bin_edges=bin_edges,
        estimator=estimator,
        weighted=weighted,
        normalized=normalized,
        eps=eps,
    )


class DictDataset(Dataset):
    def __init__(self, X, coords, t, y):
        self.X = X
        self.coords = coords
        self.t = t
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return {
            "X": self.X[idx],
            "coords": self.coords[idx],
            "t": self.t[idx],
            "y": self.y[idx],
        }


def collate_fn(batch):
    return {
        "X": torch.stack([b["X"] for b in batch]),
        "coords": torch.stack([b["coords"] for b in batch]),
        "t": torch.stack([b["t"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
    }


def build_stdk_model_config(EPOCHS, LR, WEIGHT_DECAY, BATCH_SIZE):
    return {
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "current_quantile": 0.5,
    }


def train_simple_loop(model, train_loader, device, config):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    for _ in range(config["epochs"]):
        for batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(
                batch["X"].to(device),
                batch["coords"].to(device),
                batch["t"].to(device),
            )
            loss = quantile_loss(
                y_pred,
                batch["y"].to(device),
                config["current_quantile"],
            )
            loss.backward()
            optimizer.step()

    return model


def predict_all_simple(model, X, coords, t, batch_size, device):
    dataset = DictDataset(X, coords, t, torch.zeros((len(X), 1)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for batch in loader:
            preds.append(
                model(
                    batch["X"].to(device),
                    batch["coords"].to(device),
                    batch["t"].to(device),
                )
                .cpu()
                .numpy()
            )

    return np.concatenate(preds).reshape(-1)


def new_trend_basis(n_locations: int, k_basis: int):
    trend = TrendModel(
        num_continuous_features=1,
        hidden_layer_sizes=[],
        n_locations=n_locations,
        dropout_rate=0.0,
    )
    basis = SpatialBasisLearner(n_locations, k_basis)
    return trend, basis


def adapter_diagnostics(trainer, cont: torch.Tensor, y_true: torch.Tensor, device):
    with torch.no_grad():
        mu = trainer.trend(cont.to(device))
        residual = y_true.to(device) - mu
        Phi = trainer.basis.basis
        Omega = trainer.omega

        coeff = residual @ Phi
        residual_hat = coeff @ Phi.T

        recon_mse = torch.mean((residual - residual_hat) ** 2).item()
        smooth_penalty = (trainer.tau1 * torch.sum(Phi * (Omega @ Phi))).item()
        l1_penalty = (trainer.tau2 * torch.sum(torch.abs(Phi))).item()
        total_surrogate = recon_mse + smooth_penalty + l1_penalty

        n_locations, k_basis = Phi.shape
        nk = float(n_locations * k_basis)

        smooth_penalty_per_entry = smooth_penalty / nk
        l1_penalty_per_entry = l1_penalty / nk

        eps = 1e-12
        smooth_over_recon = smooth_penalty_per_entry / max(recon_mse, eps)
        l1_over_recon = l1_penalty_per_entry / max(recon_mse, eps)

    return {
        "recon_mse": float(recon_mse),
        "smooth_penalty": float(smooth_penalty),
        "l1_penalty": float(l1_penalty),
        "total_surrogate": float(total_surrogate),
        "n_locations": int(n_locations),
        "k_basis": int(k_basis),
        "smooth_penalty_per_entry": float(smooth_penalty_per_entry),
        "l1_penalty_per_entry": float(l1_penalty_per_entry),
        "smooth_over_recon": float(smooth_over_recon),
        "l1_over_recon": float(l1_over_recon),
    }


def fit_adapter_reconstruct_all_times(
    *,
    tag,
    tau1,
    tau2,
    trend,
    basis,
    train_loader,
    val_cont,
    val_y,
    locs,
    config,
    device,
    cont_all,
    residual_true_tensor_all,
    writer=None,
):
    trainer = SpatialNeuralAdapter(
        trend=trend,
        basis=basis,
        train_loader=train_loader,
        val_cont=val_cont,
        val_y=val_y,
        locs=locs.astype(np.float32),
        config=config,
        device=device,
        writer=writer,
        tau1=float(tau1),
        tau2=float(tau2),
    )

    trainer.pretrain_trend()
    trainer.init_basis_dense()
    trainer.run()

    with torch.no_grad():
        pred_all = (
            trainer.reconstruct(
                cont_all.to(device),
                residual_true_tensor_all.to(device),
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        phi_np = trainer.basis.basis.detach().cpu().numpy().astype(np.float32)

    diag_train = adapter_diagnostics(trainer, val_cont, val_y, device)
    diag_all = adapter_diagnostics(trainer, cont_all, residual_true_tensor_all, device)

    return {
        "tag": tag,
        "tau1": float(tau1),
        "tau2": float(tau2),
        "pred_all": pred_all,
        "diag_train": diag_train,
        "diag_all": diag_all,
        "phi": phi_np,
        "trainer": trainer,
    }


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

    df = pd.DataFrame({"x": xx, "y": yy, "t": tt, "z": zz})
    ok = np.isfinite(df["z"].to_numpy(np.float32))
    return df.loc[ok].reset_index(drop=True)


def split_observed_heldout_sites(df, heldout_ratio, seed):
    df = df.copy()
    df["site_id"] = pd.factorize(list(zip(df["x"], df["y"])))[0]
    uniq_sites = np.sort(df["site_id"].unique())
    n_sites = len(uniq_sites)

    rs = np.random.RandomState(seed)
    perm = rs.permutation(n_sites)

    n_heldout = int(np.round(heldout_ratio * n_sites))
    n_heldout = max(1, min(n_heldout, n_sites - 1))

    heldout_sites = np.sort(perm[:n_heldout])
    observed_sites = np.sort(np.setdiff1d(uniq_sites, heldout_sites))

    df_obs = df[df["site_id"].isin(observed_sites)].copy().reset_index(drop=True)
    df_held = df[df["site_id"].isin(heldout_sites)].copy().reset_index(drop=True)
    return df_obs, df_held, observed_sites, heldout_sites


def build_field_matrix_from_df_and_pred(df, y_pred_flat):
    coords_all = df[["x", "y"]].to_numpy(np.float32)
    uniq_t = np.sort(df["t"].unique())
    locs, inv_loc = np.unique(coords_all, axis=0, return_inverse=True)
    t_to_idx = {t: i for i, t in enumerate(uniq_t)}

    T = len(uniq_t)
    N = len(locs)
    t_idx = np.array([t_to_idx[t] for t in df["t"].to_numpy()])
    s_idx = inv_loc

    y_pred_mat = np.full((T, N), np.nan, np.float32)
    y_true_mat = np.full((T, N), np.nan, np.float32)
    y_pred_mat[t_idx, s_idx] = y_pred_flat.astype(np.float32)
    y_true_mat[t_idx, s_idx] = df["z"].to_numpy(np.float32)
    return y_true_mat, y_pred_mat, locs, uniq_t


# ──────────────────────────────────────────────────────────────────────
# Trial heatmap plotting (tau1/tau2 grid search diagnostics)
# ──────────────────────────────────────────────────────────────────────

_TRIAL_POINT_SIZE = 50
_TRIAL_POINT_ALPHA = 0.85
_TRIAL_COLOR_COLS = ("val_rmse", "train_rmse", "smooth_penalty_train")
_TRIAL_COLOR_LABELS = {
    "val_rmse": "Validation RMSE",
    "train_rmse": "Train RMSE",
    "smooth_penalty_train": "Smooth Penalty (Train)",
}


def plot_trial_maps(
    df,
    output_dir: Path,
    plot_title_suffix: str,
    file_suffix: str,
    color_cols=_TRIAL_COLOR_COLS,
    color_labels=_TRIAL_COLOR_LABELS,
):
    """Scatter tau1/tau2 Optuna trials coloured by a metric; save linear + log plots.

    Expects `df` to have columns: tau1, tau2, val_rmse, and optionally
    train_rmse / smooth_penalty_train. `log10_tau1` / `log10_tau2` are
    computed on the fly if missing.
    """
    if df.empty:
        print(f"Skip {file_suffix}: empty dataframe.")
        return

    df = df.copy()
    if "log10_tau1" not in df.columns:
        df["log10_tau1"] = np.log10(df["tau1"])
    if "log10_tau2" not in df.columns:
        df["log10_tau2"] = np.log10(df["tau2"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_idx = df["val_rmse"].idxmin()

    print(f"\n=== Best trial info {plot_title_suffix} ===")
    best_cols = ["seed", "trial", "tau1", "tau2", "log10_tau1", "log10_tau2"]
    for c in ("train_rmse", "val_rmse", "smooth_penalty_train"):
        if c in df.columns:
            best_cols.append(c)
    print(df.loc[[best_idx], [c for c in best_cols if c in df.columns]].to_string(index=False))

    for color_col in color_cols:
        if color_col not in df.columns:
            print(f"Skip {color_col}: column not found in {file_suffix}.")
            continue

        color_label = color_labels.get(color_col, color_col)
        vmin, vmax = df[color_col].min(), df[color_col].max()

        for scale, x_col, y_col in (
            ("linear", "tau1", "tau2"),
            ("log", "log10_tau1", "log10_tau2"),
        ):
            fig, ax = plt.subplots(figsize=(7, 6))
            sc = ax.scatter(
                df[x_col], df[y_col],
                c=df[color_col],
                s=_TRIAL_POINT_SIZE, alpha=_TRIAL_POINT_ALPHA,
                vmin=vmin, vmax=vmax,
            )
            ax.scatter(
                df.loc[best_idx, x_col], df.loc[best_idx, y_col],
                s=180, facecolors="none", edgecolors="red",
                linewidths=2, label="Best trial",
            )
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{color_label} - {scale} scale {plot_title_suffix}")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend()
            fig.colorbar(sc, ax=ax).set_label(color_label)

            save_path = output_dir / f"{color_col}_{scale}_{file_suffix}.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {scale} plot: {save_path}")
