"""GatedBridge — Eq.16-17 of the paper.

Soft routing: a sigmoid gate blends a linear projection with a learned
baseline vector. This gives the bridge a continuous on/off per-dimension,
which the MI diagnostic can later inspect to see which dimensions are actively
used for conditioning.

    r  = σ(Wg · hA + bg)          ∈ (0,1)^out_dim   (Eq.16 gate)
    u  = r ⊙ (W · hA + b)
       + (1-r) ⊙ c                                    (Eq.17 blend)

where c is a learned baseline parameter (nn.Parameter).

The gate tensor `r` is returned as the second element of `_transform` so the
base class can pass it through as ``BridgeOutput.gate`` when telemetry is
requested.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import TensorBridge


class GatedBridge(TensorBridge):
    """Eq.16-17 gated bridge: soft blend of projection and learned baseline.

    Args:
        in_dim: Upstream representation dimensionality (dA).
        out_dim: Downstream representation dimensionality (dB).
        dropout: Bridge dropout probability (Stabilizer 5), forwarded to base.
        layernorm: Apply output LayerNorm (Stabilizer 1), forwarded to base.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        dropout: float = 0.0,
        layernorm: bool = True,
    ) -> None:
        super().__init__(in_dim, out_dim, dropout=dropout, layernorm=layernorm)
        # Main projection: hA → out_dim
        self.proj = nn.Linear(in_dim, out_dim, bias=True)
        # Gate projection: hA → out_dim, sigmoid-activated
        self.gate_proj = nn.Linear(in_dim, out_dim, bias=True)
        # Learned baseline c (Eq.17): constant mixture component
        self.baseline = nn.Parameter(torch.zeros(out_dim))

    def _transform(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """Map hA → r ⊙ proj(hA) + (1-r) ⊙ c, returning (u, r)."""
        r = torch.sigmoid(self.gate_proj(h))          # Eq.16
        u = r * self.proj(h) + (1.0 - r) * self.baseline  # Eq.17
        return u, r
