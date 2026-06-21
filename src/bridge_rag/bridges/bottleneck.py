"""BottleneckBridge — Eq.15 of the paper.

Two-layer MLP that compresses the upstream representation through a narrow
bottleneck (`db << min(dA, dB)`), then expands to the downstream dimension.

    uAB = W2 · σ(W1 · hA)

where σ is GELU. The base class applies the output LayerNorm (Stabilizer 1)
and dropout (Stabilizer 5) after `_transform` returns. This is the DEFAULT
bridge family — preferred unless there is a concrete reason to prefer gated or
attention routing.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import TensorBridge


class BottleneckBridge(TensorBridge):
    """Eq.15 bottleneck bridge: W2(σ(W1(hA))).

    Args:
        in_dim: Upstream representation dimensionality (dA).
        out_dim: Downstream representation dimensionality (dB).
        bottleneck_width: Hidden size of the bottleneck layer. Defaults to
            ``max(1, min(in_dim, out_dim) // 4)`` so the neck is genuinely
            narrow unless the caller overrides it.
        dropout: Bridge dropout probability (Stabilizer 5), forwarded to base.
        layernorm: Apply output LayerNorm (Stabilizer 1), forwarded to base.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bottleneck_width: int | None = None,
        dropout: float = 0.0,
        layernorm: bool = True,
    ) -> None:
        super().__init__(in_dim, out_dim, dropout=dropout, layernorm=layernorm)
        bw = bottleneck_width if bottleneck_width is not None else max(1, min(in_dim, out_dim) // 4)
        self.bottleneck_width = bw
        # W1: compress in_dim → bw
        self.w1 = nn.Linear(in_dim, bw, bias=True)
        # W2: expand bw → out_dim
        self.w2 = nn.Linear(bw, out_dim, bias=True)
        self.act = nn.GELU()

    def _transform(self, h: Tensor) -> tuple[Tensor, None]:
        """Map hA → W2(σ(W1(hA))), no gate."""
        return self.w2(self.act(self.w1(h))), None
