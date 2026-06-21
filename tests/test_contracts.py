"""Shape and budget contracts — the cheap hot-path + the latency clause."""

from __future__ import annotations

import pytest
import torch

from bridge_rag.contracts.budget import (
    BudgetViolation,
    LatencyContract,
    StructuralCheck,
)
from bridge_rag.contracts.shape import (
    ContractViolation,
    check_finite,
    check_span_count,
    check_tensor,
)


def test_check_tensor_last_dim() -> None:
    t = torch.randn(3, 8)
    assert check_tensor(t, name="x", last_dim=8) is t
    with pytest.raises(ContractViolation):
        check_tensor(t, name="x", last_dim=7)


def test_check_finite_catches_nan() -> None:
    bad = torch.tensor([1.0, float("nan")])
    with pytest.raises(ContractViolation):
        check_finite(bad, name="x")


def test_check_span_count_cap() -> None:
    with pytest.raises(ContractViolation):
        check_span_count([1, 2, 3], max_spans=2)


def test_structural_budget_pass_and_fail() -> None:
    flag = {"ok": True}
    contract = LatencyContract(
        name="t",
        p99_ms=1.0,
        structural=(
            StructuralCheck("flag", predicate=lambda: flag["ok"], detail="must be true"),
        ),
    )
    contract.check_structural()  # passes
    flag["ok"] = False
    with pytest.raises(AssertionError):
        contract.check_structural()


def test_empirical_p99_enforced() -> None:
    contract = LatencyContract(name="t", p99_ms=10.0)
    contract.check_empirical([1.0, 2.0, 3.0])  # within budget
    with pytest.raises(BudgetViolation):
        contract.check_empirical([100.0] * 50)
