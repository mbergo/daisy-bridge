"""Information-bottleneck loss: closed-form surrogate + MINE diagnostic."""

from __future__ import annotations

import pytest
import torch

from bridge_rag.config import Regularizer
from bridge_rag.contracts.invariants import MIReport
from bridge_rag.losses.mi_estimator import estimate_mi_report, infonce_mi
from bridge_rag.losses.regularization import bridge_regularization, omega, total_loss


def test_omega_l1_rewards_sparsity() -> None:
    dense = torch.ones(10, 8)
    sparse = torch.zeros(10, 8)
    sparse[:, 0] = 1.0
    assert float(omega(sparse, Regularizer.L1)) < float(omega(dense, Regularizer.L1))


def test_omega_l2_is_energy() -> None:
    u = torch.full((4, 4), 2.0)
    # mean of squares == 4
    assert float(omega(u, Regularizer.L2)) == 4.0


def test_omega_vib_nonnegative() -> None:
    assert float(omega(torch.randn(8, 8), Regularizer.VIB)) >= 0.0


def test_total_loss_adds_regularizer() -> None:
    u_ab = torch.randn(4, 8)
    u_bc = torch.randn(4, 8)
    task = torch.tensor(1.0)
    reg = bridge_regularization(
        u_ab, u_bc, lambda_ab=1e-3, lambda_bc=1e-3, kind=Regularizer.L1
    )
    tot = total_loss(task, u_ab, u_bc, lambda_ab=1e-3, lambda_bc=1e-3, kind=Regularizer.L1)
    assert float(tot) == pytest.approx(float(task) + float(reg), rel=1e-6)


def test_infonce_mi_finite() -> None:
    x = torch.randn(16, 8)
    z = x + 0.01 * torch.randn(16, 8)  # strongly coupled
    mi = float(infonce_mi(x, z))
    assert mi == mi  # not NaN


def test_estimate_mi_report_type() -> None:
    u = torch.randn(16, 8)
    rep = estimate_mi_report(u, torch.randn(16, 8), torch.randn(16, 8), beta=1.0)
    assert isinstance(rep, MIReport)
    assert hasattr(rep, "objective")
