"""Training-time invariants — the statistical clauses of a bridge contract.

These cannot run on a single forward pass: stability and information are
properties of a distribution, evaluated over a batch or a window. The trainer
calls them on a schedule (not every step on the hot path).

Two invariants matter, both from the paper:

- STABILITY (Eq.13): the end-to-end gradient is a product of six Jacobians; one
  spike collapses the run. We probe per-stage gradient norms and flag spikes.
  This is the observable behind "LR partitioning is the primary lever".

- INFORMATION (Eq.11): the bridge should compress input (low I(U;X)) while
  preserving answer signal (high I(U;Y)). We expose hooks to record the MI
  estimates produced by the diagnostic estimator — reporting only, never
  backprop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


@dataclass(slots=True)
class GradNormProbe:
    """Tracks per-parameter-group gradient norms to catch Eq.13 spikes.

    A spike is a value exceeding `spike_factor` times the running median for
    that group. Returning the offending groups lets the trainer halt or clip
    harder before the collapse propagates.
    """

    spike_factor: float = 8.0
    window: int = 50
    _history: dict[str, list[float]] = field(default_factory=dict)

    def record(self, group: str, norm: float) -> None:
        hist = self._history.setdefault(group, [])
        hist.append(float(norm))
        if len(hist) > self.window:
            del hist[0]

    def _median(self, values: list[float]) -> float:
        s = sorted(values)
        n = len(s)
        if n == 0:
            return 0.0
        mid = n // 2
        return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])

    def spikes(self) -> list[str]:
        out = []
        for group, hist in self._history.items():
            if len(hist) < 2:
                continue
            med = self._median(hist[:-1])
            if med > 0 and hist[-1] > self.spike_factor * med:
                out.append(group)
        return out


def grad_group_norms(named_groups: dict[str, Iterable[Any]]) -> dict[str, float]:
    """Compute the L2 grad norm of each named parameter group.

    `named_groups` maps a stage name (fA/gAB/fC/gBC) to its parameters. Used by
    the trainer to feed `GradNormProbe`. Torch is imported lazily so this module
    stays import-light for the non-training code paths.
    """
    import torch

    norms: dict[str, float] = {}
    for name, params in named_groups.items():
        total = 0.0
        for p in params:
            g = getattr(p, "grad", None)
            if g is not None:
                total += float(torch.linalg.vector_norm(g.detach()) ** 2)
        norms[name] = total**0.5
    return norms


@dataclass(slots=True)
class MIReport:
    """A diagnostic record of the information-bottleneck objective (Eq.11).

    `i_ux` should fall (compression) and `i_uy` should hold/rise (signal
    preserved) as training proceeds. `objective = i_ux - beta * i_uy` is the
    quantity Eq.11 minimizes — reported, not optimized (the surrogate loss is).
    """

    i_ux: float
    i_uy: float
    beta: float = 1.0

    @property
    def objective(self) -> float:
        return self.i_ux - self.beta * self.i_uy

    @property
    def compression_ratio(self) -> Optional[float]:
        if self.i_ux <= 0:
            return None
        return self.i_uy / self.i_ux
