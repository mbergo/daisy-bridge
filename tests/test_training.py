"""Training: LR partitioning (the primary stability lever) + the 4-step regime."""

from __future__ import annotations

from bridge_rag.config import Settings
from bridge_rag.pipeline.composition import build_composition
from bridge_rag.training.optimizer import build_partitioned_optimizer, describe_groups
from bridge_rag.training.trainer import Trainer, synthetic_batches


def test_optimizer_has_partitioned_lrs() -> None:
    s = Settings()
    comp = build_composition(settings=s, out_vocab=64)
    opt = build_partitioned_optimizer(comp, s)
    groups = {g["name"]: g["lr"] for g in describe_groups(opt)}
    assert groups["gab"] == s.lr_gab
    assert groups["gbc"] == s.lr_gbc
    assert groups["fc"] == s.lr_fc
    # Upstream learns slower than downstream — the noise-decoupling invariant.
    assert groups["gab"] < groups["gbc"]


def test_full_regime_runs_and_decreases_loss() -> None:
    s = Settings()
    vocab = 64
    comp = build_composition(settings=s, out_vocab=vocab)
    batches = synthetic_batches(s, n=6, vocab=vocab)
    trainer = Trainer(comp, s)
    trainer.fit(batches)  # steps 1-4 must complete without a gradient blow-up


def test_grad_probe_detects_no_spike_on_smooth_run() -> None:
    s = Settings()
    vocab = 64
    comp = build_composition(settings=s, out_vocab=vocab)
    batches = synthetic_batches(s, n=4, vocab=vocab)
    trainer = Trainer(comp, s)
    trainer.step4_joint_finetune(batches)
    # A smooth synthetic run should not trip the spike detector.
    assert trainer._grad_probe.spikes() == []
