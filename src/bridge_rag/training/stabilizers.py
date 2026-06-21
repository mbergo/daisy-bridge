"""Composable per-step stabilizer helpers (Stabilizers 1–5, runtime guards).

These are called *every* optimiser step, not only on detected spikes.
``apply_stabilizers`` is the single choreographer: record norms, clip, log.

Stabilizer mapping to paper sections:
  - Stabilizer 1: LayerNorm — baked into TensorBridge base; not here.
  - Stabilizer 2: residual connections — baked into TensorBridge subclasses.
  - Stabilizer 3: bridge dropout — ``set_bridge_dropout`` sets p at init; the
    base class carries the Dropout module and activates it in training mode.
  - Stabilizer 4: LR partitioning — ``optimizer.py``; not here.
  - Stabilizer 5: gradient clipping — ``clip_gradients`` + ``apply_stabilizers``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    # Avoided at import time so this module loads without torch installed.
    import torch
    import torch.nn as nn
    from ..pipeline.composition import BridgeComposition
    from ..contracts.invariants import GradNormProbe

logger = logging.getLogger(__name__)


def clip_gradients(
    parameters: Iterable["nn.Parameter"],
    max_norm: float,
) -> float:
    """Clip gradient norms in-place and return the pre-clip total norm.

    Must be called on **every** optimiser step, not only when spikes are
    detected. Clipping unconditionally keeps the Eq.13 Jacobian chain
    well-conditioned even before a spike manifests in the probe window.

    Args:
        parameters: Iterable of ``nn.Parameter`` objects whose ``.grad``
            tensors will be clipped.
        max_norm: Maximum allowable total L2 gradient norm.

    Returns:
        The total gradient norm *before* clipping (useful for logging and the
        ``GradNormProbe`` record).
    """
    import torch.nn.utils as utils

    pre_norm = float(utils.clip_grad_norm_(parameters, max_norm))
    return pre_norm


def set_bridge_dropout(
    composition: "BridgeComposition",
    *,
    p_gab: float,
    p_gbc: float,
) -> None:
    """Adjust dropout probability on the AB and BC bridges in-place.

    ``TensorBridge`` exposes ``self.dropout`` as an ``nn.Dropout`` module (see
    ``bridges/base.py``). This helper lets the trainer reconfigure dropout
    between training phases without rebuilding the composition.

    Args:
        composition: The differentiable bridge composition.
        p_gab: New dropout probability for the AB bridge (span-extractor seam).
        p_gbc: New dropout probability for the BC bridge (generator conditioning).
    """
    composition.bridge_ab.dropout.p = p_gab
    composition.bridge_bc.dropout.p = p_gbc
    logger.debug(
        "Bridge dropout updated: gAB p=%.3f  gBC p=%.3f", p_gab, p_gbc
    )


def apply_stabilizers(
    composition: "BridgeComposition",
    optimizer: "torch.optim.Optimizer",
    *,
    settings: "Settings",  # noqa: F821 – forward ref for type checker only
    grad_probe: "GradNormProbe",
) -> dict:
    """Per-step stability hook — call after ``loss.backward()``, before ``optimizer.step()``.

    Sequence:
    1. Compute per-stage gradient norms via ``grad_group_norms``.
    2. Record each norm into ``grad_probe``.
    3. Clip all trainable gradients to ``settings.grad_clip_norm``.
    4. Query ``grad_probe.spikes()``; log a WARNING for each spiking group.

    Args:
        composition: The differentiable bridge module (owns ``named_stage_parameters``).
        optimizer: The current optimiser (only its ``param_groups`` are read for
            the all-params iterator used by clipping).
        settings: Runtime settings; ``grad_clip_norm`` is the only field read here.
        grad_probe: Running spike detector; mutated in-place by ``record()``.

    Returns:
        ``{"grad_norms": dict[str, float], "spikes": list[str], "clipped_from": float}``

    Note:
        This function is the runtime guard against the Eq.13 collapse described
        in the paper. A spike warning is informational — training continues. The
        caller may choose to take additional action (e.g., reduce LR) but
        crashing on a spike would abort otherwise-recoverable runs.
    """
    from ..contracts.invariants import grad_group_norms

    named_groups = composition.named_stage_parameters()
    norms: dict[str, float] = grad_group_norms(named_groups)

    for group_name, norm_value in norms.items():
        grad_probe.record(group_name, norm_value)

    # Gather all trainable parameters from the optimizer's param groups.
    all_params = [
        p
        for group in optimizer.param_groups
        for p in group["params"]
    ]
    pre_norm = clip_gradients(all_params, settings.grad_clip_norm)

    spiking = grad_probe.spikes()
    if spiking:
        logger.warning(
            "Gradient spike detected in stage(s): %s  "
            "(pre-clip norm=%.4f, clip_norm=%.4f). "
            "LR partitioning and clipping remain active — training continues.",
            ", ".join(spiking),
            pre_norm,
            settings.grad_clip_norm,
        )

    return {
        "grad_norms": norms,
        "spikes": spiking,
        "clipped_from": pre_norm,
    }
