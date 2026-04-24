#!/usr/bin/env python3
"""Parameter-efficiency summary for the four GWHD backbones.

Reports, per backbone:
    1. Frozen first-stage total = vision backbone + Stage-1 classification head
       (both frozen during Stage 2).
    2. Stage-2 added trainable  = residual trend correction mu_net + learned
       spatial basis Phi.
    3. Ratio = added / frozen.

This fills the parameter-efficiency summary table referenced in Appendix L
(Wheat Head setup details) of the paper.

Usage:
    python examples/experiments/wheat_head/count_params.py
    python examples/experiments/wheat_head/count_params.py --tex
    python examples/experiments/wheat_head/count_params.py --skip-instantiate
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import torch
from torch import nn

from spatial_adapter.models import ClassificationWrapper, SpatialBasisLearner


# --------------------------------------------------------------------------- #
# Per-backbone specification (matches paper Appendix J / Table 5 + §4.2).
# feature_dim: output of the frozen feature extractor.
# K: rank used downstream (paper's rank-selection rule at tau_var = 0.9).
# literature_backbone_params: fallback if we cannot instantiate (used only for
# SAM ViT-H when the checkpoint is absent). Values are from the upstream model
# cards: torchvision ResNet-152 / ConvNeXt-Tiny / ViT-B/16 (IMAGENET1K_V1) and
# Kirillov et al. (2023) SAM ViT-H image encoder.
# --------------------------------------------------------------------------- #
@dataclass
class BackboneSpec:
    key: str
    display: str
    feature_dim: int
    K: int
    literature_backbone_params: int


SPECS = [
    BackboneSpec("resnet152", "ResNet-152", 2048, 153, 60_192_808),
    BackboneSpec("convnext_tiny", "ConvNeXt-T", 768, 155, 28_589_128),
    BackboneSpec("vit_b_16", "ViT-B/16", 768, 155, 86_567_656),
    BackboneSpec("sam_vit_h", "SAM (ViT-H)", 256, 155, 637_000_000),
]

N_LOCATIONS = 256  # 16 x 16 patch grid for GWHD.
HEAD_HIDDEN = [256, 128]  # ClassificationWrapper hidden dims (Stage 1).
MU_HIDDEN = 64  # TwoStageTrend.mu_net hidden width (Stage 2).


def _count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _human(n: int) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return str(n)


def _mu_net(feature_dim: int) -> nn.Sequential:
    """Reproduces TwoStageTrend.mu_net from
    examples/experiments/wheat_head/wheat_head_classification.py.
    """
    return nn.Sequential(
        nn.Linear(feature_dim, MU_HIDDEN),
        nn.GELU(),
        nn.Linear(MU_HIDDEN, 1),
    )


def _count_backbone(spec: BackboneSpec, skip_instantiate: bool) -> tuple[int, str]:
    """Return (params, source) — source is 'instantiated' or 'literature'."""
    if skip_instantiate:
        return spec.literature_backbone_params, "literature"
    try:
        from spatial_adapter.data.gwhd import build_backbone

        bb, _ = build_backbone(spec.key, torch.device("cpu"))
        return _count(bb), "instantiated"
    except Exception as e:  # noqa: BLE001
        print(
            f"[warn] {spec.key}: failed to instantiate ({e.__class__.__name__}: {e});"
            " falling back to literature value.",
            file=sys.stderr,
        )
        return spec.literature_backbone_params, "literature"


def _summarise(spec: BackboneSpec, skip_instantiate: bool) -> dict:
    backbone_params, source = _count_backbone(spec, skip_instantiate)

    head = ClassificationWrapper(
        feature_dim=spec.feature_dim,
        n_locations=N_LOCATIONS,
        hidden_dims=HEAD_HIDDEN,
    )
    head_params = _count(head)

    mu_params = _count(_mu_net(spec.feature_dim))
    basis_params = _count(SpatialBasisLearner(N_LOCATIONS, spec.K))

    frozen_total = backbone_params + head_params
    added_total = mu_params + basis_params

    return {
        "spec": spec,
        "source": source,
        "backbone": backbone_params,
        "head": head_params,
        "frozen_total": frozen_total,
        "mu_net": mu_params,
        "basis": basis_params,
        "added_total": added_total,
        "ratio": added_total / frozen_total,
    }


def _print_plain(rows: list[dict]) -> None:
    header = (
        f"{'Backbone':<14} {'Frozen total':>14} {'Stage-2 added':>14} "
        f"{'Ratio':>10} {'Breakdown (backbone / head | mu / basis)':<50}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        s = r["spec"]
        note = "" if r["source"] == "instantiated" else "*"
        print(
            f"{s.display:<14}"
            f" {_human(r['frozen_total']):>12}{note:<2}"
            f" {_human(r['added_total']):>14}"
            f" {r['ratio'] * 100:>8.3f}%"
            f"   {_human(r['backbone']):>8} / {_human(r['head']):>8}"
            f" | {_human(r['mu_net']):>6} / {_human(r['basis']):>6}"
        )
    if any(r["source"] == "literature" for r in rows):
        print("\n* literature value used (backbone could not be instantiated locally).")


def _print_tex(rows: list[dict]) -> None:
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(
        r"  \caption{GWHD parameter-efficiency summary.  "
        r"\emph{Frozen first stage} collects the pretrained vision backbone "
        r"and the Stage-1 classification head (both frozen during Stage~2).  "
        r"\emph{Stage-2 added trainable} is the residual trend correction "
        r"$\mu_{\text{net}}$ plus the learned spatial basis "
        r"$\widehat{\bm\Phi}\in\mathbb R^{N\times K}$.  "
        r"The adapter adds a fraction of a percent of the frozen first-stage "
        r"parameter count across all four backbones.}"
    )
    print(r"  \label{tab:wheat-param-efficiency}")
    print(r"  \begin{tabular}{@{}lrrrr@{}}")
    print(r"    \toprule")
    print(
        r"    \textbf{Backbone} & \textbf{Frozen total} "
        r"& \textbf{Stage-2 added} & \textbf{Ratio} "
        r"& \textbf{Breakdown (backbone / head \textbar{} $\mu_{\text{net}}$ / $\Phi$)} \\"
    )
    print(r"    \midrule")
    for r in rows:
        s = r["spec"]
        note = "" if r["source"] == "instantiated" else r"$^{\dagger}$"
        bd = (
            f"{_human(r['backbone'])} / {_human(r['head'])}"
            rf" \textbar{{}} {_human(r['mu_net'])} / {_human(r['basis'])}"
        )
        print(
            f"    {s.display}{note} & {_human(r['frozen_total'])}"
            f" & {_human(r['added_total'])}"
            f" & {r['ratio'] * 100:.3f}\\%"
            f" & {bd} \\\\"
        )
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    if any(r["source"] == "literature" for r in rows):
        print(
            r"  \vspace{0.25em}"
            "\n  {\\footnotesize $^{\\dagger}$ literature value (SAM ViT-H image "
            r"encoder, Kirillov et al.\ 2023) used when the checkpoint is "
            r"absent from the local environment.}"
        )
    print(r"\end{table}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tex", action="store_true", help="Emit a LaTeX table for the appendix."
    )
    parser.add_argument(
        "--skip-instantiate",
        action="store_true",
        help="Do not build backbones; use literature values only (offline mode).",
    )
    args = parser.parse_args()

    rows = [_summarise(s, args.skip_instantiate) for s in SPECS]
    if args.tex:
        _print_tex(rows)
    else:
        _print_plain(rows)


if __name__ == "__main__":
    main()
