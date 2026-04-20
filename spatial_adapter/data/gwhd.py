"""
Global Wheat Head Detection (GWHD) dataset loader for the spatial adapter.

Converts detection annotations into a patch-level binary classification task:
  - Each 1024x1024 image is divided into a 16x16 grid of 64x64 patches (N=256).
  - A patch is labelled positive (1) if any ground-truth bbox overlaps it.
  - Per-patch features are extracted from a frozen backbone.

The output format matches the adapter interface:
  train_loader: DataLoader with (cat, cont, targets)
      cat:     (B, N, 0)  -- empty categorical placeholder
      cont:    (B, N, p)  -- backbone features per patch
      targets: (B, N)     -- binary labels
  val_cont, val_y, test_cont, test_y, locs
"""

import ast
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Patch-grid helpers
# ---------------------------------------------------------------------------

GRID_H, GRID_W = 16, 16
PATCH_SIZE = 64  # 1024 / 16


def _parse_bbox_string(bbox_str: str) -> List[List[float]]:
    """
    Parse bounding boxes from GWHD format.

    Supports two formats:
      1. GWHD 2021 BoxesString: "xmin ymin xmax ymax;xmin ymin xmax ymax;..."
         Each box is space-separated, boxes separated by semicolons.
         Returns [[xmin, ymin, xmax, ymax], ...] (already in xyxy format).
      2. Kaggle competition bbox: "[xmin, ymin, w, h]" (JSON-like list).
         Returns [[xmin, ymin, xmax, ymax], ...] (converted to xyxy).
    """
    if pd.isna(bbox_str) or bbox_str.strip() in ("", "[]", "no_box"):
        return []

    bbox_str = bbox_str.strip()

    # Format 1: GWHD 2021 semicolon-separated "xmin ymin xmax ymax"
    # Also handles grouped Kaggle format: "[x,y,w,h];[x,y,w,h];..."
    if ";" in bbox_str or (bbox_str[0].isdigit()):
        boxes = []
        for part in bbox_str.split(";"):
            part = part.strip()
            if not part:
                continue
            # Try JSON-like first (Kaggle: "[x, y, w, h]")
            if part.startswith("["):
                try:
                    parsed = ast.literal_eval(part)
                    if isinstance(parsed, list) and len(parsed) == 4:
                        xmin, ymin, w, h = [float(v) for v in parsed]
                        boxes.append([xmin, ymin, xmin + w, ymin + h])
                        continue
                except (ValueError, SyntaxError):
                    pass
            # Space-separated "xmin ymin xmax ymax"
            vals = part.split()
            if len(vals) == 4:
                boxes.append([float(v) for v in vals])
        return boxes

    # Format 2: single JSON-like "[xmin, ymin, w, h]"
    try:
        parsed = ast.literal_eval(bbox_str)
    except (ValueError, SyntaxError):
        return []
    if (
        isinstance(parsed, list)
        and len(parsed) == 4
        and isinstance(parsed[0], (int, float))
    ):
        xmin, ymin, w, h = parsed
        return [[xmin, ymin, xmin + w, ymin + h]]
    return []


def _bboxes_to_patch_labels(
    bboxes: List[List[float]],
    img_h: int = 1024,
    img_w: int = 1024,
) -> np.ndarray:
    """
    Convert bounding boxes to a binary (GRID_H, GRID_W) label map.
    bbox format: [xmin, ymin, xmax, ymax] (xyxy).
    """
    labels = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for box in bboxes:
        xmin, ymin, xmax, ymax = box
        # Map to grid indices
        c0 = max(0, int(xmin // PATCH_SIZE))
        c1 = min(GRID_W - 1, int(xmax // PATCH_SIZE))
        r0 = max(0, int(ymin // PATCH_SIZE))
        r1 = min(GRID_H - 1, int(ymax // PATCH_SIZE))
        labels[r0 : r1 + 1, c0 : c1 + 1] = 1.0
    return labels.ravel()  # (256,)


# ---------------------------------------------------------------------------
# Patch locations (normalised 2-D grid coordinates)
# ---------------------------------------------------------------------------


def get_patch_locations() -> np.ndarray:
    """Return (N, 2) array of normalised patch-centre coordinates in [0, 1]."""
    rows = np.arange(GRID_H) / (GRID_H - 1)
    cols = np.arange(GRID_W) / (GRID_W - 1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return np.stack([rr.ravel(), cc.ravel()], axis=1).astype(np.float64)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _extract_patches(img: Image.Image) -> torch.Tensor:
    """Split a 1024x1024 PIL image into (256, 3, 64, 64) tensor of patches."""
    to_tensor = transforms.ToTensor()
    img_t = to_tensor(img.convert("RGB"))  # (3, 1024, 1024)
    # unfold into grid
    patches = img_t.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)
    # patches shape: (3, 16, 16, 64, 64)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()  # (16, 16, 3, 64, 64)
    return patches.view(GRID_H * GRID_W, 3, PATCH_SIZE, PATCH_SIZE)


@torch.no_grad()
def extract_features_for_dataset(
    image_dir: str,
    image_ids: List[str],
    backbone: nn.Module,
    device: torch.device,
    batch_size: int = 64,
    feature_dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Extract per-patch features for all images.

    Args:
        image_dir: directory containing .jpg images.
        image_ids: list of image id strings (no extension).
        backbone: frozen feature extractor mapping (B, 3, 64, 64) -> (B, p).
        device: torch device.
        batch_size: patches processed per forward pass.
        feature_dim: expected output dim p (auto-detected if None).

    Returns:
        features: (T, N, p) float32 tensor on CPU.
    """
    backbone.eval()
    N = GRID_H * GRID_W  # 256

    # Auto-detect feature_dim
    if feature_dim is None:
        dummy = torch.randn(1, 3, PATCH_SIZE, PATCH_SIZE, device=device)
        feature_dim = backbone(dummy).shape[-1]

    all_features = []
    for img_id in image_ids:
        img_path = os.path.join(image_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, f"{img_id}.png")
        img = Image.open(img_path).resize((1024, 1024))
        patches = _extract_patches(img)  # (256, 3, 64, 64)

        feats = []
        for i in range(0, N, batch_size):
            batch = patches[i : i + batch_size].to(device)
            out = backbone(batch)
            if out.dim() > 2:
                out = (
                    out.mean(dim=(-2, -1))
                    if out.dim() == 4
                    else out.view(out.size(0), -1)
                )
            feats.append(out.cpu())
        all_features.append(torch.cat(feats, dim=0))  # (256, p)

    return torch.stack(all_features, dim=0)  # (T, 256, p)


@torch.no_grad()
def extract_features_for_dataset_sam(
    image_dir: str,
    image_ids: List[str],
    encoder: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract per-patch features using SAM's full-image encoder.

    SAM processes the full 1024×1024 image and outputs a (B, 256, 64, 64)
    spatial feature map.  Our 16×16 patch grid maps to 4×4 blocks of the
    64×64 feature map, so we average-pool each block to get (256, 256)
    per-patch features per image.

    Returns:
        features: (T, 256, 256) float32 tensor on CPU.
    """
    encoder.eval()
    resize = transforms.Resize((1024, 1024), antialias=True)
    to_tensor = transforms.ToTensor()

    all_features = []
    for img_id in image_ids:
        img_path = os.path.join(image_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, f"{img_id}.png")
        img = Image.open(img_path).convert("RGB")
        img_t = resize(to_tensor(img)).unsqueeze(0).to(device)  # (1, 3, 1024, 1024)

        feat_map = encoder(img_t)  # (1, 256, 64, 64)
        # Reshape to 16×16 grid of 4×4 blocks, then avg-pool each block
        # (1, 256, 64, 64) -> (1, 256, 16, 4, 16, 4)
        feat_map = feat_map.view(1, 256, GRID_H, 4, GRID_W, 4)
        patch_feats = feat_map.mean(dim=(3, 5))  # (1, 256, 16, 16)
        patch_feats = patch_feats.view(256, GRID_H * GRID_W).permute(1, 0)  # (256, 256)
        all_features.append(patch_feats.cpu())

    return torch.stack(all_features, dim=0)  # (T, 256, 256)


# ---------------------------------------------------------------------------
# Backbone builders
# ---------------------------------------------------------------------------


def build_backbone(name: str, device: torch.device) -> Tuple[nn.Module, int]:
    """
    Build a frozen feature extractor from a backbone name.

    Returns:
        (backbone_module, feature_dim)
    """
    import torchvision.models as tv_models

    if name == "resnet152":
        model = tv_models.resnet152(weights=tv_models.ResNet152_Weights.IMAGENET1K_V1)
        # Remove the final fc layer; output of avgpool is (B, 2048)
        modules = list(model.children())[:-1]
        backbone = nn.Sequential(*modules, nn.Flatten())
        feature_dim = 2048

    elif name == "convnext_tiny":
        model = tv_models.convnext_tiny(
            weights=tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )
        # features + avgpool, skip classifier
        backbone = nn.Sequential(model.features, model.avgpool, nn.Flatten())
        feature_dim = 768

    elif name == "vit_b_16":
        model = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.IMAGENET1K_V1)
        # ViT needs 224x224 input; we'll add resize in the transform
        class ViTFeatureExtractor(nn.Module):
            def __init__(self, vit):
                super().__init__()
                self.vit = vit
                self.resize = transforms.Resize((224, 224), antialias=True)

            def forward(self, x):
                x = self.resize(x)
                # Use encoder output, not classification head
                x = self.vit._process_input(x)
                n = x.shape[0]
                batch_class_token = self.vit.class_token.expand(n, -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)
                x = self.vit.encoder(x)
                return x[:, 0]  # CLS token

        backbone = ViTFeatureExtractor(model)
        feature_dim = 768

    elif name == "sam_vit_h":
        import os

        from segment_anything import sam_model_registry

        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        ckpt_path = os.path.join(repo_root, "sam_vit_h_4b8939.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"SAM ViT-H checkpoint not found at {ckpt_path}. "
                "Download from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            )

        sam = sam_model_registry["vit_h"](checkpoint=ckpt_path)
        image_encoder = sam.image_encoder  # ViT-H encoder

        # SAM is a full-image encoder, not a patch-level one.
        # Feature extraction for SAM is handled separately in
        # extract_features_for_dataset_sam() below.
        # We return the raw image encoder here; feature_dim = 256.
        backbone = image_encoder
        feature_dim = 256

    else:
        raise ValueError(f"Unknown backbone: {name}")

    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    return backbone, feature_dim


# ---------------------------------------------------------------------------
# High-level loader
# ---------------------------------------------------------------------------


def load_gwhd_annotations(csv_path: str) -> pd.DataFrame:
    """Load and parse GWHD CSV annotations."""
    df = pd.read_csv(csv_path)
    # Standardise column names: GWHD 2021 uses "image_name" and "BoxesString"
    if "image_name" in df.columns:
        df = df.rename(columns={"image_name": "image_id", "BoxesString": "bbox"})
    elif "image_id" not in df.columns:
        raise ValueError(
            f"CSV must have 'image_name' or 'image_id' column. Found: {df.columns.tolist()}"
        )
    if "bbox" not in df.columns and "BoxesString" in df.columns:
        df = df.rename(columns={"BoxesString": "bbox"})
    # Strip extensions from image_id if present
    df["image_id"] = (
        df["image_id"].astype(str).str.replace(r"\.(jpg|png)$", "", regex=True)
    )
    return df


def build_patch_labels(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    """
    Build per-image patch labels from annotation DataFrame.

    Each image has one row with all bboxes in the 'bbox' column
    (GWHD format: semicolon-separated "xmin ymin xmax ymax" strings).

    Returns:
        image_ids: list of unique image ids
        labels: (T, N) binary array
    """
    # Group by image_id in case there are multiple rows per image
    grouped = (
        df.groupby("image_id")["bbox"]
        .apply(lambda x: ";".join(str(v) for v in x if pd.notna(v)))
        .reset_index()
    )
    image_ids = grouped["image_id"].tolist()
    labels = []
    for _, row in grouped.iterrows():
        all_bboxes = _parse_bbox_string(str(row["bbox"]))
        labels.append(_bboxes_to_patch_labels(all_bboxes))
    return image_ids, np.stack(labels, axis=0)


def get_gwhd_dataloader_and_val(
    csv_path: str,
    image_dir: str,
    backbone_name: str,
    device: torch.device,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    seed: int = 42,
    max_images: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[
    DataLoader, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray
]:
    """
    End-to-end loader: parse annotations -> extract features -> split -> return.

    Returns:
        train_loader, val_cont, val_y, test_cont, test_y, locs
    """
    # Check for cached features
    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        suffix = f"_{max_images}" if max_images else ""
        cache_path = os.path.join(cache_dir, f"gwhd_{backbone_name}{suffix}.pt")
        if os.path.exists(cache_path):
            cached = torch.load(cache_path, weights_only=False)
            features = cached["features"]
            labels_all = cached["labels"]
            locs = cached["locs"]
            T = features.shape[0]
            rng = np.random.default_rng(seed)
            perm = rng.permutation(T)
            n_train = int(T * train_ratio)
            n_val = int(T * val_ratio)
            train_idx = perm[:n_train]
            val_idx = perm[n_train : n_train + n_val]
            test_idx = perm[n_train + n_val :]
            return _build_splits(
                features, labels_all, locs, train_idx, val_idx, test_idx, batch_size
            )

    # Parse annotations
    df = load_gwhd_annotations(csv_path)
    image_ids, labels_all = build_patch_labels(df)

    if max_images is not None:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(image_ids), min(max_images, len(image_ids)), replace=False)
        sel.sort()
        image_ids = [image_ids[i] for i in sel]
        labels_all = labels_all[sel]

    # Extract features
    backbone, feature_dim = build_backbone(backbone_name, device)
    if backbone_name == "sam_vit_h":
        features = extract_features_for_dataset_sam(
            image_dir, image_ids, backbone, device
        )
    else:
        features = extract_features_for_dataset(
            image_dir,
            image_ids,
            backbone,
            device,
            batch_size=64,
            feature_dim=feature_dim,
        )
    labels_all = torch.from_numpy(labels_all).float()
    locs = get_patch_locations()

    # Cache
    if cache_path is not None:
        torch.save(
            {"features": features, "labels": labels_all, "locs": locs}, cache_path
        )

    # Split
    T = features.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(T)
    n_train = int(T * train_ratio)
    n_val = int(T * val_ratio)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    return _build_splits(
        features, labels_all, locs, train_idx, val_idx, test_idx, batch_size
    )


def _build_splits(
    features: torch.Tensor,
    labels: torch.Tensor,
    locs: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int,
) -> Tuple[
    DataLoader, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray
]:
    """Build DataLoader and validation/test tensors from indices."""
    train_cont = features[train_idx]
    train_y = labels[train_idx]
    val_cont = features[val_idx]
    val_y = labels[val_idx]
    test_cont = features[test_idx]
    test_y = labels[test_idx]

    N = features.shape[1]
    train_cat = torch.zeros(len(train_idx), N, 0, dtype=torch.long)
    dataset = TensorDataset(train_cat, train_cont, train_y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_cont, val_y, test_cont, test_y, locs
