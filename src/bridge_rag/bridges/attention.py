"""AttentionBridge — Eq.18-20 of the paper.

The production sidecar interface. Realizes multi-head self-attention over the
upstream representation, then projects to the downstream dimension.

    Q = h · WQ,  K = h · WK,  V = h · WV          (Eq.18 projections)
    A = softmax(Q Kᵀ / √dk)                         (Eq.19 attention weights)
    u = Wo · (A · V)                                 (Eq.20 output projection)

Handles both 2-D inputs (B, in_dim) — treated as a single-token sequence —
and 3-D inputs (B, T, in_dim), returning matching shape with last dim out_dim.
Dropout (Stabilizer 5) and output LayerNorm (Stabilizer 1) are applied by the
base class after `_transform` returns.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import TensorBridge


class AttentionBridge(TensorBridge):
    """Eq.18-20 multi-head self-attention bridge.

    Args:
        in_dim: Upstream representation dimensionality (dA / dk before split).
        out_dim: Downstream representation dimensionality (dB).
        num_heads: Number of attention heads. Must divide ``in_dim``.
        dropout: Bridge dropout probability (Stabilizer 5), forwarded to base.
        layernorm: Apply output LayerNorm (Stabilizer 1), forwarded to base.

    Raises:
        ValueError: If ``num_heads`` does not divide ``in_dim``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.0,
        layernorm: bool = True,
    ) -> None:
        if in_dim % num_heads != 0:
            raise ValueError(
                f"AttentionBridge: in_dim ({in_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        super().__init__(in_dim, out_dim, dropout=dropout, layernorm=layernorm)
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Q, K, V projections (Eq.18) — no bias, matching standard attention
        self.wq = nn.Linear(in_dim, in_dim, bias=False)
        self.wk = nn.Linear(in_dim, in_dim, bias=False)
        self.wv = nn.Linear(in_dim, in_dim, bias=False)
        # Output projection Wo (Eq.20): in_dim → out_dim
        self.wo = nn.Linear(in_dim, out_dim, bias=True)

    def _transform(self, h: Tensor) -> tuple[Tensor, None]:
        """Multi-head self-attention over h, then project to out_dim.

        Accepts h of shape (..., in_dim) or (..., T, in_dim).
        Returns (u, None) where u has shape (..., out_dim) or (..., T, out_dim).
        """
        squeeze = h.dim() == 1 or (h.dim() >= 2 and h.shape[-1] == self.in_dim and h.dim() == 2 and h.ndim == 2)

        # Normalise to (..., T, in_dim) — at least 2-D
        if h.dim() < 2:
            # unlikely given base contract, but guard anyway
            h = h.unsqueeze(0)

        # 2-D (B, in_dim) → treat as (B, 1, in_dim) single token
        added_seq = False
        if h.dim() == 2:
            h = h.unsqueeze(1)   # (B, 1, in_dim)
            added_seq = True

        # h is now (..., T, in_dim)
        *batch, T, _ = h.shape
        B = 1
        for d in batch:
            B *= d
        h_flat = h.reshape(B, T, self.in_dim)   # (B, T, in_dim)

        # Q, K, V projections — Eq.18
        Q = self.wq(h_flat)   # (B, T, in_dim)
        K = self.wk(h_flat)
        V = self.wv(h_flat)

        # Split heads: (B, T, in_dim) → (B, H, T, head_dim)
        def split_heads(x: Tensor) -> Tensor:
            b, t, d = x.shape
            return x.reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(Q)   # (B, H, T, head_dim)
        K = split_heads(K)
        V = split_heads(V)

        # Scaled dot-product attention — Eq.19
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)   # (B, H, T, head_dim)

        # Merge heads: (B, H, T, head_dim) → (B, T, in_dim)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.in_dim)

        # Output projection — Eq.20
        u = self.wo(attn_out)   # (B, T, out_dim)

        # Restore original batch shape
        u = u.reshape(*batch, T, self.out_dim)

        # Remove the sequence dimension we added for single-token inputs
        if added_seq:
            u = u.squeeze(-2)   # (..., out_dim)

        return u, None
