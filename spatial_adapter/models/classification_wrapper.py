"""Classification wrapper: (B, N, p) → (B, N) logits for binary tasks."""

from typing import List, Optional

import torch
import torch.nn as nn


class ClassificationWrapper(nn.Module):
    """Adapter trend for classification.

    Optionally wraps a frozen backbone: (B, N, C, H, W) → backbone → (B, N, p) → head → (B, N).
    """

    def __init__(
        self,
        feature_dim: int,
        n_locations: int = 1,
        hidden_dims: Optional[List[int]] = None,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_locations = n_locations
        self.backbone = backbone

        if hidden_dims:
            layers = []
            in_d = feature_dim
            for h in hidden_dims:
                layers.extend([nn.Linear(in_d, h), nn.GELU(), nn.Dropout(0.1)])
                in_d = h
            layers.append(nn.Linear(in_d, 1))
            self.head = nn.Sequential(*layers)
        else:
            self.head = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, p) or (B, N, C, H, W) → (B, N) logits."""
        if x.dim() == 5 and self.backbone is not None:
            B, N, C, H, W = x.shape
            feat = self.backbone(x.view(B * N, C, H, W))
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            x = feat.view(B, N, -1)
        B, N, p = x.shape
        return self.head(x.reshape(-1, p)).view(B, N)

    def residual_parameters(self) -> List[torch.Tensor]:
        """Trainable parameters for the adapter's theta-step."""
        return [p for p in self.parameters() if p.requires_grad]
