"""Bridge regularization losses — Eq.8, 9, 10 of the paper.

The information bottleneck objective (Eq.11) is:

    min  I(U;X) - β·I(U;Y)

``-β·I(U;Y)`` is approximated by task loss and handled by the trainer.
``I(U;X)`` is approximated here by a closed-form tractable surrogate Ω(U),
giving the total training loss (Eq.8):

    L = L_task + λAB·Ω(uAB) + λBC·Ω(uBC)

Three surrogate choices (``Regularizer`` enum):

- ``L2``  (Eq.9): mean ||u||₂²  — energy / activation-magnitude control.
- ``L1``  (Eq.10): mean ||u||₁  — sparsity / feature selectivity.
- ``VIB`` : KL(N(μ,σ²) ‖ N(0,I)) closed form, treating u as the mean vector
  and assuming unit variance (σ²=1).  This gives the upper bound
  ``0.5 · mean(u²)`` on I(U;X) from the variational information bottleneck
  (Alemi et al., 2017).  The VIB KL in full form is
  ``0.5 · (σ² + μ² - 1 - log σ²)``; with σ²=1 this reduces to
  ``0.5 · μ²``, i.e. ``0.5 · mean(u²)``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..config import Regularizer


def omega(u: Tensor, kind: Regularizer) -> Tensor:
    """Compute the compression surrogate Ω(U) as a scalar tensor.

    Args:
        u: The bridge representation, arbitrary shape (..., d).
        kind: Which surrogate to use (L1, L2, or VIB).

    Returns:
        A scalar tensor suitable for inclusion in the backward graph.
    """
    if kind is Regularizer.L2:
        # Eq.9: mean ||u||₂² — sum of squared activations over all elements
        return u.pow(2).mean()

    if kind is Regularizer.L1:
        # Eq.10: mean ||u||₁ — mean absolute activation
        return u.abs().mean()

    if kind is Regularizer.VIB:
        # VIB upper bound on I(U;X): KL(N(u,1) ‖ N(0,1)) = 0.5·mean(u²)
        # Full VIB KL = 0.5·(σ² + μ² - 1 - log σ²); with σ²=1 → 0.5·μ²
        return 0.5 * u.pow(2).mean()

    raise ValueError(f"Unknown Regularizer: {kind!r}")


def bridge_regularization(
    u_ab: Tensor,
    u_bc: Tensor,
    *,
    lambda_ab: float,
    lambda_bc: float,
    kind: Regularizer,
) -> Tensor:
    """Eq.8 regularization term: λAB·Ω(uAB) + λBC·Ω(uBC).

    Args:
        u_ab: Bridge AB representation tensor.
        u_bc: Bridge BC representation tensor.
        lambda_ab: Regularization weight for the AB bridge.
        lambda_bc: Regularization weight for the BC bridge.
        kind: Compression surrogate to apply to both bridges.

    Returns:
        A scalar tensor: the weighted sum of both surrogate penalties.
    """
    return lambda_ab * omega(u_ab, kind) + lambda_bc * omega(u_bc, kind)


def total_loss(
    task_loss: Tensor,
    u_ab: Tensor,
    u_bc: Tensor,
    *,
    lambda_ab: float,
    lambda_bc: float,
    kind: Regularizer,
) -> Tensor:
    """Eq.8 full training loss: L_task + λAB·Ω(uAB) + λBC·Ω(uBC).

    Args:
        task_loss: The primary task loss (e.g. cross-entropy on generation).
        u_ab: Bridge AB representation tensor.
        u_bc: Bridge BC representation tensor.
        lambda_ab: Regularization weight for the AB bridge.
        lambda_bc: Regularization weight for the BC bridge.
        kind: Compression surrogate (L1, L2, or VIB).

    Returns:
        A scalar tensor: task loss plus the bridge regularization penalty.
    """
    return task_loss + bridge_regularization(
        u_ab, u_bc, lambda_ab=lambda_ab, lambda_bc=lambda_bc, kind=kind
    )
