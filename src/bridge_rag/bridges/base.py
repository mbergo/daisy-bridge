"""TensorBridge — the trainable `Tensor -> Tensor` interface (Eq. 14-20).

A bridge is an interface contract, not glue (Section 2.2). The *pure tensor*
families (bottleneck, gated, attention, residual) all satisfy this ABC. The
production span extractor is deliberately NOT a TensorBridge — it is a model
with a latency contract, and lives under `sidecar/`.

Two stabilizers are baked into the base class because the paper lists them as
non-optional (Section 5.2):

- Stabilizer 1, LayerNorm at every bridge output. Controls the activation
  distribution so a downstream shift cannot blow up the upstream input. Bounds
  the per-link Jacobian gain in the Eq.13 chain.
- Stabilizer 5, bridge dropout during training. Prevents stage co-adaptation;
  forces each stage to be independently useful.

Subclasses implement `_transform`; the base wraps it with the contract checks,
the LayerNorm, the dropout, and the telemetry the trainer needs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from ..contracts.shape import check_finite, check_tensor
from ..types import BridgeOutput


class TensorBridge(nn.Module, ABC):
    """Base class for the learnable intermediate bridges.

    Args:
        in_dim:  dimensionality of the upstream representation (dA).
        out_dim: dimensionality the downstream stage expects (dB).
        dropout: bridge dropout probability (Stabilizer 5).
        layernorm: apply output LayerNorm (Stabilizer 1). Default on; turning it
            off is a deliberate, tested choice, never an accident.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        dropout: float = 0.0,
        layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm = nn.LayerNorm(out_dim) if layernorm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    @abstractmethod
    def _transform(self, h: Tensor) -> tuple[Tensor, Tensor | None]:
        """Map upstream `h` (..., in_dim) to bridged `u` (..., out_dim).

        Returns `(u, gate)` where `gate` is the routing tensor for gated
        bridges (Eq.17) or None. The base class applies norm + dropout to `u`.
        """

    def forward(self, h: Tensor, *, return_telemetry: bool = False) -> Tensor | BridgeOutput:
        check_tensor(h, name=f"{type(self).__name__}.in", last_dim=self.in_dim)
        raw, gate = self._transform(h)
        pre = float(raw.detach().std()) if return_telemetry else None
        u = self.norm(raw)
        u = self.dropout(u)
        check_tensor(u, name=f"{type(self).__name__}.out", last_dim=self.out_dim)
        check_finite(u, name=f"{type(self).__name__}.out")
        if not return_telemetry:
            return u
        return BridgeOutput(
            value=u,
            pre_norm=pre,
            post_norm=float(u.detach().std()),
            gate=gate,
        )

    @torch.no_grad()
    def jacobian_spectral_norm(self, h: Tensor, *, iters: int = 5) -> float:
        """Power-iteration estimate of the output/input Jacobian spectral norm.

        This is the quantity that, multiplied six times (Eq.13), decides whether
        training is stable. > 1 across all links => explosion risk; near 1 (what
        LayerNorm + residual buy you) => well-conditioned. Diagnostic only.
        """
        h = h.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            out = self.forward(h)
            assert isinstance(out, Tensor)
            v = torch.randn_like(out)
            v = v / (v.norm() + 1e-12)
            sigma = 0.0
            for _ in range(iters):
                (jvp,) = torch.autograd.grad(out, h, grad_outputs=v, retain_graph=True)
                sigma = float(jvp.norm())
                v = jvp / (jvp.norm() + 1e-12)
                out2 = self.forward(h)
                assert isinstance(out2, Tensor)
                out = out2
        return sigma
