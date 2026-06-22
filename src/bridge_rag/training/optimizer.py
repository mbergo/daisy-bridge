"""LR-partitioned optimizer (Stabilizer 4).

The paper's central stability insight: downstream noise from the generation
head must not destabilise the upstream span-extraction representations. The
mechanism is purely in the learning-rate schedule: each stage trains at a
rate calibrated to its position in the Eq.13 Jacobian chain.

   fA (frozen)  <  gAB (1e-4)  <  gBC (1e-3)  <  fC (1e-4)

`fA` is upstream-pretrained and frozen; it never appears in the optimizer.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def build_partitioned_optimizer(
    composition: "BridgeComposition",  # noqa: F821 – forward ref, torch not needed at import
    settings: "Settings",  # noqa: F821
    *,
    fa_params: Optional[list] = None,
) -> "torch.optim.AdamW":  # noqa: F821
    """Build an AdamW with per-stage learning rates.

    Args:
        composition: The differentiable bridge composition whose ``named_stage_parameters``
            method provides the per-stage parameter lists.
        settings: Runtime settings carrying ``lr_gab``, ``lr_gbc``, ``lr_fc``,
            and ``lr_fa``.
        fa_params: Optional list of ``nn.Parameter`` objects for the ``fA``
            embedder. Pass these when the caller has unfrozen the embedder for
            full fine-tuning; they receive ``settings.lr_fa``. Absent means fA
            is frozen (the standard training regime).

    Returns:
        Configured ``torch.optim.AdamW`` with one param-group per stage.

    Note:
        LR partitioning is the *primary* lever against Eq.13 collapse —
        keep the ratios intact. Changing any single LR without adjusting
        the others breaks the chain stability guarantee.
    """
    import torch.optim as optim

    stage_params = composition.named_stage_parameters()

    param_groups: list[dict] = [
        {
            "name": "gab",
            "params": stage_params["gab"],
            "lr": settings.lr_gab,
        },
        {
            "name": "gbc",
            "params": stage_params["gbc"],
            "lr": settings.lr_gbc,
        },
        {
            "name": "fc",
            "params": stage_params["fc"],
            "lr": settings.lr_fc,
        },
    ]

    if fa_params is not None:
        param_groups.append(
            {
                "name": "fa",
                "params": fa_params,
                "lr": settings.lr_fa,
            }
        )

    optimizer = optim.AdamW(param_groups)

    group_summary = ", ".join(
        f"{g['name']}@{g['lr']:.2e}" for g in param_groups
    )
    logger.debug("Partitioned AdamW built: %s", group_summary)

    return optimizer


def describe_groups(optimizer: "torch.optim.AdamW") -> list[dict]:  # noqa: F821
    """Return a human-readable summary of each param-group.

    Used by tests that assert the four LR groups exist with the correct rates.

    Args:
        optimizer: A partitioned AdamW built by :func:`build_partitioned_optimizer`.

    Returns:
        List of dicts with keys ``name``, ``lr``, and ``num_params``.
    """
    result: list[dict] = []
    for group in optimizer.param_groups:
        result.append(
            {
                "name": group.get("name", ""),
                "lr": group["lr"],
                "num_params": sum(p.numel() for p in group["params"]),
            }
        )
    return result
