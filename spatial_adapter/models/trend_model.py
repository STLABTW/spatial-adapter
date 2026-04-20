"""Trend model: frozen OLS linear + optional residual MLP."""

from typing import Optional

import torch
import torch.nn as nn


class TrendModel(nn.Module):
    """Frozen OLS linear layer + zero-init residual MLP on top."""

    def __init__(
        self,
        num_continuous_features: int,
        hidden_layer_sizes: list[int],
        n_locations: int,
        init_weight: Optional[torch.Tensor] = None,
        init_bias: Optional[float] = None,
        freeze_init: bool = True,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        F = num_continuous_features

        self.init_lin = nn.Linear(F, 1, bias=True)
        if init_weight is not None and init_bias is not None:
            with torch.no_grad():
                w = init_weight.mean(dim=0) if init_weight.dim() == 2 else init_weight
                self.init_lin.weight.copy_(w.view(1, F))
                self.init_lin.bias.copy_(torch.tensor([init_bias]))
            if freeze_init:
                for p in self.init_lin.parameters():
                    p.requires_grad = False

        if hidden_layer_sizes:
            blocks = []
            for i, h in enumerate(hidden_layer_sizes):
                in_d = F if i == 0 else hidden_layer_sizes[i - 1]
                blocks.extend(
                    [
                        nn.Linear(in_d, h, bias=True),
                        nn.LayerNorm(h),
                        nn.GELU(),
                        nn.Dropout(dropout_rate),
                    ]
                )
            self.res_blocks = nn.Sequential(*blocks)
            self.res_out = nn.Linear(hidden_layer_sizes[-1], 1)
        else:
            self.res_blocks = None
            self.res_out = None

        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.init_lin:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, F) → (B, N) predictions."""
        B, P, F = x.shape
        flat = x.view(-1, F)
        out = self.init_lin(flat)
        if self.res_blocks is not None and self.res_out is not None:
            out = out + self.res_out(self.res_blocks(flat))
        return out.view(B, -1)

    def residual_parameters(self):
        """Trainable parameters (used by the adapter's theta-step)."""
        return [p for p in self.parameters() if p.requires_grad]
