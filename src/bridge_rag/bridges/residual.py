"""ResidualBridge — Eq.14 of the paper (stability-first).

A residual adapter that preserves the identity at initialisation, giving
gradients a direct highway through the bridge and preventing the earliest
training steps from disrupting the upstream representation.

    uAB = hA + α · Adapter(hA)                        (Eq.14)

where α is a learned scalar initialised to 1e-3 (so uAB ≈ hA at step 0) and
Adapter is a small two-layer MLP: in_dim → in_dim → in_dim with GELU.

CONSTRAINT: ResidualBridge requires in_dim == out_dim. A residual skip is only
valid when the upstream and downstream representations share the same space. If
dimensions differ, use BottleneckBridge or AttentionBridge instead.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import TensorBridge


class ResidualBridge(TensorBridge):
    """Eq.14 residual bridge: hA + α·Adapter(hA).

    Args:
        in_dim: Upstream representation dimensionality. Must equal ``out_dim``.
        out_dim: Downstream representation dimensionality. Must equal ``in_dim``.
        dropout: Bridge dropout probability (Stabilizer 5), forwarded to base.
        layernorm: Apply output LayerNorm (Stabilizer 1), forwarded to base.

    Raises:
        AssertionError: If ``in_dim != out_dim``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        dropout: float = 0.0,
        layernorm: bool = True,
    ) -> None:
        assert in_dim == out_dim, (
            f"ResidualBridge requires in_dim == out_dim, got {in_dim} != {out_dim}. "
            "Choose BottleneckBridge or AttentionBridge when dimensions differ."
        )
        super().__init__(in_dim, out_dim, dropout=dropout, layernorm=layernorm)

        # Learned scale initialised small so the bridge starts near identity
        self.alpha = nn.Parameter(torch.full((), 1e-3))

        # Small adapter MLP: in_dim → in_dim with GELU in between
        self.adapter = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=True),
            nn.GELU(),
            nn.Linear(in_dim, in_dim, bias=True),
        )

    def _transform(self, h: Tensor) -> tuple[Tensor, None]:
        """Return hA + α·Adapter(hA), no gate."""
        return h + self.alpha * self.adapter(h), None
