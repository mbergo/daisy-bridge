"""The pure differentiable composition (Eq. 6) — the training graph.

Eq.6 is `ŷ = fC(gBC(fB(gAB(fA(x)))))`. Two faces of it:

- SERVING is discrete: `gAB` selects spans (argmax, non-differentiable), the
  generator emits tokens. That path is the async orchestrator.
- TRAINING is continuous: the *representation-level* composition flows the
  embedding `hA` through the two learnable TensorBridges and a generation head,
  fully differentiable, so the Eq.13 six-Jacobian gradient and the Eq.11 MI
  probes have something to attach to.

This module is the second face: a clean `nn.Module` holding `gAB` and `gBC` as
TensorBridges (`fA` frozen upstream, `fB` is identity). It is what the trainer
optimizes and what the stability/information invariants are measured on. It does
NOT import a generator backend or any serving concern — that separation keeps
the gradient story testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from ..bridges.base import TensorBridge
from ..config import Settings, get_settings


@dataclass(slots=True)
class CompositionOutput:
    """Differentiable composition result + the bridge representations.

    `u_ab` and `u_bc` are exposed because the regularizer (Eq.8) penalizes them
    and the MI diagnostic measures `I(U;X)`/`I(U;Y)` on them.
    """

    logits: Tensor  # (..., vocab) generation head output
    u_ab: Tensor  # bridge AB representation (the sidecar's continuous twin)
    u_bc: Tensor  # bridge BC representation (generator conditioning)


class BridgeComposition(nn.Module):
    """`fB(gBC(gAB(.)))` as a differentiable module; `fA` is upstream+frozen.

    Args:
        bridge_ab: the AB TensorBridge (attention-style in production).
        bridge_bc: the BC TensorBridge (gated routing in production).
        gen_head: maps the BC representation to generation logits. A linear head
            stands in for the QLoRA-adapted generator in the differentiable
            training graph; the real generator runs in serving.
    """

    def __init__(
        self,
        bridge_ab: TensorBridge,
        bridge_bc: TensorBridge,
        gen_head: nn.Module,
    ) -> None:
        super().__init__()
        self.bridge_ab = bridge_ab
        self.bridge_bc = bridge_bc
        self.gen_head = gen_head

    def forward(self, h_a: Tensor) -> CompositionOutput:
        u_ab = self.bridge_ab(h_a)  # gAB
        h_b = u_ab  # fB: identity, the needle
        u_bc = self.bridge_bc(h_b)  # gBC
        logits = self.gen_head(u_bc)  # fC (differentiable stand-in)
        return CompositionOutput(logits=logits, u_ab=u_ab, u_bc=u_bc)

    def named_stage_parameters(self) -> dict[str, list[nn.Parameter]]:
        """Parameters grouped by stage for LR partitioning + grad probes.

        `fA` is absent on purpose (frozen). Keys match the LR knobs in Settings.
        """
        return {
            "gab": list(self.bridge_ab.parameters()),
            "gbc": list(self.bridge_bc.parameters()),
            "fc": list(self.gen_head.parameters()),
        }


def build_composition(
    *,
    settings: Optional[Settings] = None,
    out_vocab: int = 4096,
) -> BridgeComposition:
    """Construct the composition for the active profile using the bridge factory.

    The factory is imported lazily so this Layer-1 module does not hard-depend on
    the concrete bridge implementations (which land in the parallel fan-out).
    """
    settings = settings or get_settings()
    profile = settings.profile
    from ..bridges.factory import build_bridge  # lazy: concrete families

    bridge_ab = build_bridge(
        settings.bridge_ab,
        in_dim=profile.embed_dim,
        out_dim=profile.bottleneck_dim,
        dropout=settings.dropout_gab,
    )
    bridge_bc = build_bridge(
        settings.bridge_bc,
        in_dim=profile.bottleneck_dim,
        out_dim=profile.gen_hidden_dim,
        dropout=settings.dropout_gbc,
    )
    gen_head = nn.Linear(profile.gen_hidden_dim, out_vocab)
    return BridgeComposition(bridge_ab, bridge_bc, gen_head)
