"""
Unit tests for spatial_adapter.data.gwhd loader helpers.

Exercises pure-function paths (bbox parsing, patch labelling, grid location
helpers, CSV loading) and the cache-hit branch of get_gwhd_dataloader_and_val
without requiring real GWHD imagery or downloading backbone weights.
"""

import numpy as np
import pytest
import torch
from PIL import Image

torchvision = pytest.importorskip("torchvision")

from spatial_adapter.data.gwhd import (  # noqa: E402
    GRID_H,
    GRID_W,
    PATCH_SIZE,
    _bboxes_to_patch_labels,
    _build_splits,
    _extract_patches,
    _parse_bbox_string,
    build_backbone,
    build_patch_labels,
    extract_features_for_dataset,
    get_gwhd_dataloader_and_val,
    get_patch_locations,
    load_gwhd_annotations,
)


# ──────────────────────────────────────────────────────────────────────
# _parse_bbox_string
# ──────────────────────────────────────────────────────────────────────


class TestParseBboxString:
    def test_empty_sentinel_inputs_return_empty_list(self):
        for s in ("", "[]", "no_box"):
            assert _parse_bbox_string(s) == []

    def test_nan_returns_empty_list(self):
        assert _parse_bbox_string(float("nan")) == []

    def test_gwhd2021_semicolon_format(self):
        """xmin ymin xmax ymax ; ..."""
        s = "0 0 64 64;128 128 200 200"
        out = _parse_bbox_string(s)
        assert out == [[0.0, 0.0, 64.0, 64.0], [128.0, 128.0, 200.0, 200.0]]

    def test_grouped_kaggle_format_in_xywh(self):
        """Kaggle grouped '[x,y,w,h];[x,y,w,h]' → converted to xyxy."""
        s = "[10, 20, 30, 40];[100, 200, 10, 10]"
        out = _parse_bbox_string(s)
        assert out == [[10.0, 20.0, 40.0, 60.0], [100.0, 200.0, 110.0, 210.0]]

    def test_single_kaggle_format(self):
        """Single '[xmin, ymin, w, h]' (no semicolon, no leading digit)."""
        s = " [5, 10, 20, 30] "
        out = _parse_bbox_string(s)
        assert out == [[5.0, 10.0, 25.0, 40.0]]

    def test_malformed_part_is_skipped(self):
        s = "0 0 64 64;garbage"  # second part has no 4 tokens
        out = _parse_bbox_string(s)
        assert out == [[0.0, 0.0, 64.0, 64.0]]

    def test_fully_unparseable_returns_empty_list(self):
        # No semicolon, not digit-leading, ast.literal_eval fails
        assert _parse_bbox_string("xyz") == []

    def test_kaggle_literal_eval_fails_falls_back_to_space_split(self):
        """'[malformed, ...' → literal_eval raises; falls through to space-split branch."""
        out = _parse_bbox_string("[not 4 numeric tokens here")
        # No space-split match either (wrong token count), empty
        assert out == []


# ──────────────────────────────────────────────────────────────────────
# _bboxes_to_patch_labels
# ──────────────────────────────────────────────────────────────────────


class TestBboxesToPatchLabels:
    def test_empty_bboxes_gives_all_zero_label(self):
        labels = _bboxes_to_patch_labels([])
        assert labels.shape == (GRID_H * GRID_W,)
        assert labels.sum() == 0

    def test_single_bbox_in_one_cell(self):
        """A small bbox within patch (0,0) should set exactly one cell."""
        labels = _bboxes_to_patch_labels([[0.0, 0.0, 10.0, 10.0]])
        assert labels[0] == 1.0
        assert labels.sum() == 1.0

    def test_bbox_spanning_multiple_cells(self):
        """A bbox spanning grid cells (0,0)-(1,1) should set 4 cells."""
        labels = _bboxes_to_patch_labels([[0, 0, PATCH_SIZE + 1, PATCH_SIZE + 1]])
        assert labels.sum() == 4.0

    def test_bbox_beyond_grid_is_clipped(self):
        """Bbox going beyond image is clipped to grid bounds, not index-error."""
        labels = _bboxes_to_patch_labels([[-100, -100, 10000, 10000]])
        assert labels.sum() == GRID_H * GRID_W  # every cell set


# ──────────────────────────────────────────────────────────────────────
# get_patch_locations
# ──────────────────────────────────────────────────────────────────────


class TestGetPatchLocations:
    def test_shape_and_range(self):
        locs = get_patch_locations()
        assert locs.shape == (GRID_H * GRID_W, 2)
        assert locs.min() == 0.0
        assert locs.max() == 1.0

    def test_corner_points(self):
        locs = get_patch_locations()
        # First point = (0, 0); last point = (1, 1)
        np.testing.assert_allclose(locs[0], [0.0, 0.0])
        np.testing.assert_allclose(locs[-1], [1.0, 1.0])


# ──────────────────────────────────────────────────────────────────────
# _extract_patches
# ──────────────────────────────────────────────────────────────────────


class TestExtractPatches:
    def test_output_shape(self):
        img = Image.new("RGB", (1024, 1024), color=(128, 128, 128))
        patches = _extract_patches(img)
        assert patches.shape == (GRID_H * GRID_W, 3, PATCH_SIZE, PATCH_SIZE)

    def test_grayscale_input_is_converted_to_rgb(self):
        img = Image.new("L", (1024, 1024), color=128)
        patches = _extract_patches(img)
        assert patches.shape[1] == 3


# ──────────────────────────────────────────────────────────────────────
# load_gwhd_annotations
# ──────────────────────────────────────────────────────────────────────


class TestLoadGwhdAnnotations:
    def test_kaggle_schema_image_id(self, tmp_path):
        csv = tmp_path / "ann.csv"
        # bbox cells are quoted because they contain commas
        csv.write_text(
            'image_id,bbox\n'
            'img1.jpg,"[0, 0, 10, 10]"\n'
            'img2,"[20, 20, 5, 5]"\n'
        )
        df = load_gwhd_annotations(str(csv))
        assert list(df.columns) >= ["image_id", "bbox"]
        assert "img1" in df["image_id"].tolist()  # .jpg stripped

    def test_gwhd2021_schema_image_name(self, tmp_path):
        csv = tmp_path / "ann.csv"
        csv.write_text("image_name,BoxesString\nfoo.png,0 0 64 64\n")
        df = load_gwhd_annotations(str(csv))
        # image_name → image_id, BoxesString → bbox
        assert "image_id" in df.columns
        assert "bbox" in df.columns
        assert df["image_id"].iloc[0] == "foo"

    def test_missing_id_column_raises(self, tmp_path):
        csv = tmp_path / "bad.csv"
        csv.write_text("other,bbox\nx,[0,0,1,1]\n")
        with pytest.raises(ValueError, match="image_name"):
            load_gwhd_annotations(str(csv))


# ──────────────────────────────────────────────────────────────────────
# build_patch_labels
# ──────────────────────────────────────────────────────────────────────


class TestBuildPatchLabels:
    def test_one_row_per_image(self, tmp_path):
        import pandas as pd

        df = pd.DataFrame(
            {"image_id": ["a", "b"], "bbox": ["0 0 64 64", ""]}
        )
        ids, labels = build_patch_labels(df)
        assert ids == ["a", "b"]
        assert labels.shape == (2, GRID_H * GRID_W)
        assert labels[0].sum() > 0
        assert labels[1].sum() == 0

    def test_multiple_rows_per_image_are_merged(self):
        import pandas as pd

        df = pd.DataFrame(
            {"image_id": ["a", "a"], "bbox": ["0 0 64 64", "128 128 200 200"]}
        )
        ids, labels = build_patch_labels(df)
        assert ids == ["a"]
        # Two non-overlapping bboxes → label count > each alone
        assert labels[0].sum() >= 2


# ──────────────────────────────────────────────────────────────────────
# build_backbone — unknown backbone branch
# ──────────────────────────────────────────────────────────────────────


class TestBuildBackbone:
    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown backbone"):
            build_backbone("not_a_real_backbone", device=torch.device("cpu"))


# ──────────────────────────────────────────────────────────────────────
# _build_splits — direct call (no image/backbone needed)
# ──────────────────────────────────────────────────────────────────────


class TestBuildSplits:
    def test_shapes_and_dataloader(self):
        T, N, p = 12, GRID_H * GRID_W, 8
        features = torch.randn(T, N, p)
        labels = (torch.rand(T, N) > 0.5).float()
        locs = get_patch_locations()
        train_idx = np.arange(0, 8)
        val_idx = np.arange(8, 10)
        test_idx = np.arange(10, 12)

        loader, val_cont, val_y, test_cont, test_y, locs_out = _build_splits(
            features, labels, locs, train_idx, val_idx, test_idx, batch_size=4
        )
        assert val_cont.shape == (2, N, p)
        assert val_y.shape == (2, N)
        assert test_cont.shape == (2, N, p)
        assert test_y.shape == (2, N)
        assert locs_out is locs
        # Iterate the loader to ensure it yields the expected schema
        cat, cont, y = next(iter(loader))
        assert cat.shape == (4, N, 0)
        assert cont.shape == (4, N, p)
        assert y.shape == (4, N)


# ──────────────────────────────────────────────────────────────────────
# extract_features_for_dataset — with a tiny CPU backbone + temp jpg
# ──────────────────────────────────────────────────────────────────────


class _TinyBackbone(torch.nn.Module):
    """Deterministic backbone: average over spatial dims → (B, 3)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(-2, -1))  # (B, 3, H, W) → (B, 3)


class TestExtractFeaturesForDataset:
    def test_produces_expected_shape(self, tmp_path):
        device = torch.device("cpu")
        backbone = _TinyBackbone().to(device)

        # Write one 1024x1024 jpg
        img = Image.new("RGB", (1024, 1024), color=(50, 100, 150))
        img_path = tmp_path / "img1.jpg"
        img.save(img_path)

        features = extract_features_for_dataset(
            image_dir=str(tmp_path),
            image_ids=["img1"],
            backbone=backbone,
            device=device,
            batch_size=128,
            feature_dim=3,
        )
        assert features.shape == (1, GRID_H * GRID_W, 3)

    def test_auto_detects_feature_dim(self, tmp_path):
        device = torch.device("cpu")
        backbone = _TinyBackbone().to(device)
        img = Image.new("RGB", (1024, 1024), color=(0, 0, 0))
        (tmp_path / "img1.jpg").write_bytes(b"")  # ensure the path exists
        img.save(tmp_path / "img1.jpg")

        features = extract_features_for_dataset(
            image_dir=str(tmp_path),
            image_ids=["img1"],
            backbone=backbone,
            device=device,
            batch_size=64,
            feature_dim=None,  # auto-detect triggers the dummy forward
        )
        assert features.shape == (1, GRID_H * GRID_W, 3)

    def test_png_fallback_path(self, tmp_path):
        """When .jpg missing, the code falls back to .png."""
        device = torch.device("cpu")
        backbone = _TinyBackbone().to(device)
        Image.new("RGB", (1024, 1024), color=(200, 0, 0)).save(
            tmp_path / "imgA.png"
        )

        features = extract_features_for_dataset(
            image_dir=str(tmp_path),
            image_ids=["imgA"],
            backbone=backbone,
            device=device,
            batch_size=64,
            feature_dim=3,
        )
        assert features.shape == (1, GRID_H * GRID_W, 3)


# ──────────────────────────────────────────────────────────────────────
# get_gwhd_dataloader_and_val — cache-hit path (skips backbone load)
# ──────────────────────────────────────────────────────────────────────


class TestGetGwhdDataloaderCacheHit:
    def test_loads_from_cache_without_touching_images(self, tmp_path):
        """If a cache file exists, the loader returns from it without
        loading CSV / images / backbone."""
        T, N, p = 10, GRID_H * GRID_W, 4
        features = torch.randn(T, N, p)
        labels = (torch.rand(T, N) > 0.5).float()
        locs = get_patch_locations()

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_path = cache_dir / "gwhd_resnet152.pt"
        torch.save(
            {"features": features, "labels": labels, "locs": locs}, cache_path
        )

        loader, val_cont, val_y, test_cont, test_y, out_locs = (
            get_gwhd_dataloader_and_val(
                csv_path="/nonexistent.csv",  # must not be read under cache hit
                image_dir="/nonexistent",
                backbone_name="resnet152",
                device=torch.device("cpu"),
                train_ratio=0.6,
                val_ratio=0.2,
                batch_size=4,
                seed=0,
                cache_dir=str(cache_dir),
            )
        )
        # Train window has 6 samples (0.6 × 10)
        n_train = int(T * 0.6)
        n_val = int(T * 0.2)
        assert val_cont.shape[0] == n_val
        assert test_cont.shape[0] == T - n_train - n_val
        assert out_locs.shape == locs.shape
