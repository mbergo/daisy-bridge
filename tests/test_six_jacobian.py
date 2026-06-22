"""The literal Eq.13 six: fA unfrozen, fB parametric, the product tracked.

The default composition follows the paper's freeze-fA recipe (four factors to a
gAB param). This suite exercises the *other* half of the paper: the full
six-factor gradient chain, where the gradient reaches a perception parameter
through all six Jacobians.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from bridge_rag.config import Settings
from bridge_rag.pipeline.composition import FullBridgeComposition, build_full_composition
from bridge_rag.training.optimizer import build_full_partitioned_optimizer, describe_groups
from bridge_rag.training.stabilizers import apply_stabilizers
from bridge_rag.contracts.invariants import GradNormProbe


def _settings() -> Settings:
    return Settings()


def test_full_composition_has_five_stages() -> None:
    comp = build_full_composition(settings=_settings(), out_vocab=64)
    groups = comp.named_stage_parameters()
    assert set(groups) == {"fa", "gab", "fb", "gbc", "fc"}
    # fA and fB carry real parameters now — not frozen, not identity.
    assert len(groups["fa"]) > 0
    assert len(groups["fb"]) > 0


def test_full_optimizer_partitions_all_five() -> None:
    s = _settings()
    comp = build_full_composition(settings=s, out_vocab=64)
    opt = build_full_partitioned_optimizer(comp, s)
    rates = {g["name"]: g["lr"] for g in describe_groups(opt)}
    assert set(rates) == {"fa", "gab", "fb", "gbc", "fc"}
    assert rates["fa"] == s.lr_fa
    assert rates["fb"] == s.lr_fb
    # Upstream perception still learns slowest — noise-decoupling preserved.
    assert rates["fa"] < rates["gab"]
    assert rates["fa"] < rates["gbc"]


def test_gradient_reaches_perception_through_six_jacobians() -> None:
    # The whole point: dL/dθA exists. If fA were frozen, this grad would be None.
    s = _settings()
    comp = build_full_composition(settings=s, out_vocab=64)
    x = torch.randn(4, s.profile.embed_dim)
    out = comp(x)
    targets = torch.randint(0, 64, (4,))
    loss = F.cross_entropy(out.logits, targets)
    loss.backward()

    fa_param = comp.named_stage_parameters()["fa"][0]
    fb_param = comp.named_stage_parameters()["fb"][0]
    assert fa_param.grad is not None
    assert float(fa_param.grad.norm()) > 0.0  # the 6th factor is live
    assert fb_param.grad is not None  # fB is a real differentiable stage


def test_six_jacobian_product_is_finite_and_bounded() -> None:
    comp = build_full_composition(settings=_settings(), out_vocab=64)
    x = torch.randn(4, comp.perception_head[0].in_features)
    sigma = comp.six_jacobian_spectral_norm(x)
    assert sigma == sigma  # not NaN
    assert sigma < 1e4  # LayerNorm + residual keep the product from exploding


def test_full_regime_step_runs_without_spike() -> None:
    s = _settings()
    comp = build_full_composition(settings=s, out_vocab=64)
    opt = build_full_partitioned_optimizer(comp, s)
    probe = GradNormProbe()
    x = torch.randn(4, s.profile.embed_dim)
    targets = torch.randint(0, 64, (4,))
    for _ in range(3):
        opt.zero_grad()
        out = comp(x)
        loss = F.cross_entropy(out.logits, targets)
        loss.backward()
        apply_stabilizers(comp, opt, settings=s, grad_probe=probe)
        opt.step()
    assert probe.spikes() == []
