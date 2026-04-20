# Data Directory

This directory holds external datasets used by the experiments. Data files are
not tracked by git (see `.gitignore`).

## Download

```bash
bash data/download_data.sh
```

Requires the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) with a valid
API key at `~/.kaggle/kaggle.json`.

## Datasets

| Dataset | Source | Target path |
|---------|--------|-------------|
| KAUST 2b_8 | [Kaggle competition](https://www.kaggle.com/competitions/2022-kaust-ss-competition-2b) | `data/kaust/2b_8.csv` |
| Weather2K | [Zhu et al. (2023)](https://github.com/bycnfz/weather2k) | `data/weather2k/weather2k.npy` |
| GWHD | [Kaggle competition](https://www.kaggle.com/competitions/global-wheat-detection) | `data/gwhd/train.csv` + `data/gwhd/train/` |

### GWHD

After downloading, the data handler (`spatial_adapter/data/gwhd.py`)
automatically extracts backbone features and caches them to `.pt` files in
`cache_dir`. Subsequent runs load from cache without re-extracting.

### Weather2K

The `.npy` file is not directly downloadable via script. Prepare it from the
raw Weather2K repository and copy to `data/weather2k/weather2k.npy`.
