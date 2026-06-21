"""Latency budget as a typed contract — two enforcement modes.

The paper's `<0.3ms` sidecar budget is real but GPU-fused-kernel territory; you
cannot reproduce it on a dev CPU, and `assert elapsed < 0.3ms` in pytest is
flaky garbage. So the budget splits in two:

1. STRUCTURAL (deterministic, runs in CI everywhere): the things that *protect*
   the budget in code review — bounded input, bounded output, no synchronous
   I/O on the hot path, no Python-level per-token loop. These are properties of
   the code, not the hardware.

2. EMPIRICAL (environment-relative, runs in the benchmark script / prod
   healthcheck): p99 wall-clock against a *per-profile* threshold. Dev asserts
   50ms, prod asserts 0.3ms. CI only ever checks the dev number.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator


class BudgetViolation(AssertionError):
    """Raised when an empirical p99 exceeds the profile budget."""


@dataclass(frozen=True, slots=True)
class StructuralCheck:
    """One deterministic property the hot path must satisfy."""

    name: str
    predicate: Callable[[], bool]
    detail: str = ""

    def evaluate(self) -> tuple[bool, str]:
        ok = bool(self.predicate())
        return ok, "" if ok else f"structural check failed: {self.name} ({self.detail})"


@dataclass(frozen=True, slots=True)
class LatencyContract:
    """The budget clause of a bridge contract.

    `p99_ms` is environment-relative — pass the *current profile's* number.
    `structural` is environment-independent and is what unit tests enforce.
    """

    name: str
    p99_ms: float
    structural: tuple[StructuralCheck, ...] = field(default_factory=tuple)

    def check_structural(self) -> None:
        failures = []
        for chk in self.structural:
            ok, msg = chk.evaluate()
            if not ok:
                failures.append(msg)
        if failures:
            raise AssertionError(
                f"{self.name}: structural budget violated:\n  " + "\n  ".join(failures)
            )

    def check_empirical(self, samples_ms: list[float]) -> float:
        """Assert p99 of measured samples is within budget. Returns the p99."""
        if not samples_ms:
            raise ValueError("no latency samples")
        ordered = sorted(samples_ms)
        idx = min(len(ordered) - 1, int(round(0.99 * (len(ordered) - 1))))
        p99 = ordered[idx]
        if p99 > self.p99_ms:
            raise BudgetViolation(
                f"{self.name}: p99 {p99:.3f}ms exceeds budget {self.p99_ms:.3f}ms "
                f"(n={len(samples_ms)})"
            )
        return p99


@dataclass(slots=True)
class Timer:
    """Accumulates wall-clock samples for empirical budget checks."""

    samples_ms: list[float] = field(default_factory=list)

    @contextmanager
    def measure(self) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.samples_ms.append((time.perf_counter() - t0) * 1000.0)

    @property
    def last_ms(self) -> float:
        return self.samples_ms[-1] if self.samples_ms else 0.0


def sidecar_contract(
    p99_ms: float,
    *,
    max_candidates: int,
    max_spans: int,
    get_candidate_count: Callable[[], int],
    get_span_count: Callable[[], int],
) -> LatencyContract:
    """Build the sidecar's latency contract for the current profile.

    The structural checks bind the input/output sizes that actually drive the
    budget: candidate count (sidecar input) and span count (sidecar output).
    """
    return LatencyContract(
        name="sidecar(gAB)",
        p99_ms=p99_ms,
        structural=(
            StructuralCheck(
                name="bounded_candidates",
                predicate=lambda: get_candidate_count() <= max_candidates,
                detail=f"<= {max_candidates}",
            ),
            StructuralCheck(
                name="bounded_spans",
                predicate=lambda: get_span_count() <= max_spans,
                detail=f"<= {max_spans}",
            ),
        ),
    )
