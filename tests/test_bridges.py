"""Bridge family contracts: shapes, dim guards, gate telemetry, finiteness."""

from __future__ import annotations

import pytest
import torch

from bridge_rag.bridges.factory import build_bridge
from bridge_rag.config import BridgeKind
from bridge_rag.contracts.shape import ContractViolation
from bridge_rag.types import BridgeOutput


@pytest.mark.parametrize(
    "kind", [BridgeKind.BOTTLENECK, BridgeKind.GATED, BridgeKind.ATTENTION]
)
def test_bridge_maps_to_out_dim(kind: BridgeKind) -> None:
    bridge = build_bridge(kind, in_dim=32, out_dim=16)
    out = bridge(torch.randn(8, 32))
    assert out.shape == (8, 16)
    assert torch.isfinite(out).all()


def test_residual_requires_equal_dims() -> None:
    with pytest.raises(ValueError):
        build_bridge(BridgeKind.RESIDUAL, in_dim=32, out_dim=16)


def test_residual_is_near_identity_at_init() -> None:
    # alpha initialized small => uAB ~= LN(hA), i.e. identity preserved.
    bridge = build_bridge(BridgeKind.RESIDUAL, in_dim=16, out_dim=16, layernorm=False)
    h = torch.randn(4, 16)
    out = bridge(h)
    assert torch.allclose(out, h, atol=1e-1)


def test_gate_telemetry_exposed() -> None:
    bridge = build_bridge(BridgeKind.GATED, in_dim=16, out_dim=8)
    out = bridge(torch.randn(2, 16), return_telemetry=True)
    assert isinstance(out, BridgeOutput)
    assert out.gate is not None
    assert out.gate.shape[-1] == 8
    assert ((out.gate >= 0) & (out.gate <= 1)).all()  # sigmoid range


def test_layernorm_controls_distribution() -> None:
    # Stabilizer 1: output std ~= 1 after LayerNorm regardless of input scale.
    bridge = build_bridge(BridgeKind.BOTTLENECK, in_dim=64, out_dim=32)
    out = bridge(torch.randn(128, 64) * 50.0, return_telemetry=True)
    assert isinstance(out, BridgeOutput)
    assert out.post_norm == pytest.approx(1.0, abs=0.25)


def test_contract_rejects_wrong_in_dim() -> None:
    bridge = build_bridge(BridgeKind.BOTTLENECK, in_dim=16, out_dim=8)
    with pytest.raises(ContractViolation):
        bridge(torch.randn(4, 17))


def test_jacobian_spectral_norm_bounded() -> None:
    # Near-1 conditioning is what keeps the six-Jacobian product (Eq.13) stable.
    bridge = build_bridge(BridgeKind.RESIDUAL, in_dim=16, out_dim=16)
    sigma = bridge.jacobian_spectral_norm(torch.randn(4, 16))
    assert sigma == pytest.approx(1.0, abs=2.0)
