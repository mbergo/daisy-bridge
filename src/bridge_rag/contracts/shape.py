"""Cheap, runtime shape/dtype/count contracts — safe on the hot path.

A bridge is an interface CONTRACT (Section 2.2). The *shape* clause of that
contract is the part that can be checked in O(1) on every forward pass without
blowing the latency budget: tensor rank, dimensionality, dtype, and the span
count bound. The statistical clauses (stability, information) live in
`invariants.py` and run on a schedule in the trainer, never here.
"""

from __future__ import annotations

from typing import Any, Sequence


class ContractViolation(ValueError):
    """Raised when a value violates a bridge interface contract."""


def _shape(t: Any) -> tuple[int, ...]:
    return tuple(t.shape)


def check_tensor(
    t: Any,
    *,
    name: str,
    last_dim: int | None = None,
    rank: int | None = None,
    dtype: Any | None = None,
) -> Any:
    """Validate a tensor against the cheap clauses of a contract.

    Returns the tensor unchanged so it can wrap an expression inline:
        u = check_tensor(bridge(h), name="uAB", last_dim=db)
    """
    shape = _shape(t)
    if rank is not None and len(shape) != rank:
        raise ContractViolation(
            f"{name}: expected rank {rank}, got shape {shape}"
        )
    if last_dim is not None and (len(shape) == 0 or shape[-1] != last_dim):
        raise ContractViolation(
            f"{name}: expected last dim {last_dim}, got shape {shape}"
        )
    if dtype is not None and getattr(t, "dtype", None) != dtype:
        raise ContractViolation(
            f"{name}: expected dtype {dtype}, got {getattr(t, 'dtype', None)}"
        )
    return t


def check_span_count(spans: Sequence[Any], *, max_spans: int, name: str = "spans") -> None:
    """The output cap from the budget contract: bounded span count.

    An unbounded span set would defeat the bottleneck (Eq.11) — the whole point
    is that ~100-300 tokens reach the generator, not the haystack.
    """
    n = len(spans)
    if n > max_spans:
        raise ContractViolation(
            f"{name}: span count {n} exceeds contract cap {max_spans}"
        )


def check_finite(t: Any, *, name: str) -> Any:
    """Guard against the NaN/Inf that an Eq.13 gradient spike produces.

    Cheap enough for the bridge output check; catches a collapsing run early
    instead of letting poison propagate down the 6-Jacobian chain.
    """
    isfinite = getattr(t, "isfinite", None)
    if isfinite is not None and not bool(isfinite().all()):
        raise ContractViolation(f"{name}: contains NaN or Inf")
    return t
