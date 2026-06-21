"""Stage-wise then joint training regime (Steps 1-4 of the paper).

The four steps implement the curriculum from Section 5.3:

  Step 1 — fA frozen (no-op here; fA is not in the composition).
  Step 2 — train gAB only (span-extraction sidecar, KD supervised).
  Step 3 — train gBC + fC only (generation conditioning, QLoRA stage).
  Step 4 — joint fine-tune with LR partitioning protecting upstream stages.

``Trainer`` is framework-light: it owns a composition and a settings object,
delegates optimiser construction to ``optimizer.py``, and calls
``apply_stabilizers`` every step for the gradient-norm runtime guard.

``synthetic_batches`` generates random CPU tensors so the full four-step
regime can be smoke-tested without any real data or GPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    import torch
    from torch import Tensor
    from ..pipeline.composition import BridgeComposition
    from ..config import Settings

logger = logging.getLogger(__name__)

# How often (in steps) to run the MI diagnostic when enabled.
_MI_DIAG_INTERVAL = 10


@dataclass
class TrainConfig:
    """Hyper-parameters for a single training run.

    Defaults are intentionally tiny so the full regime fits on a laptop CPU in
    seconds — useful for smoke tests and quick iteration.
    """

    epochs_per_step: int = 1
    batch_size: int = 4
    log_every_n_steps: int = 5
    mi_diag_interval: int = _MI_DIAG_INTERVAL


def _freeze(params: list) -> None:
    """Disable gradient computation for a parameter list in-place."""
    for p in params:
        p.requires_grad_(False)


def _unfreeze(params: list) -> None:
    """Re-enable gradient computation for a parameter list in-place."""
    for p in params:
        p.requires_grad_(True)


class Trainer:
    """Orchestrates the four-step training curriculum on ``BridgeComposition``.

    Args:
        composition: The differentiable bridge module to train.
        settings: Runtime settings (LRs, clip norm, regularizer knobs, …).
        train_config: Optional fine-grained training hyper-parameters.
            Defaults to ``TrainConfig()`` (small values for dev/smoke runs).
    """

    def __init__(
        self,
        composition: "BridgeComposition",
        settings: "Settings",
        train_config: TrainConfig | None = None,
    ) -> None:
        self._composition = composition
        self._settings = settings
        self._train_config = train_config or TrainConfig()
        self._global_step = 0

        from ..contracts.invariants import GradNormProbe

        self._grad_probe = GradNormProbe()

    # ------------------------------------------------------------------
    # Step-level public interface
    # ------------------------------------------------------------------

    def step1_freeze_perception(self) -> None:
        """Document that fA is pretrained and frozen — no-op on the composition.

        ``BridgeComposition`` does not contain the fA embedder; it receives the
        already-computed ``h_a`` embedding as input. This method exists to make
        the four-step curriculum explicit in code and logs.
        """
        logger.info(
            "Step 1: fA (perception embedder) is pretrained and frozen. "
            "It is upstream of this composition and does not appear in any "
            "param group. No parameter changes applied."
        )

    def step2_train_sidecar(self, batches: list[tuple["Tensor", "Tensor"]]) -> None:
        """Step 2: train gAB only; gBC and fC are frozen.

        This is the span-extraction sidecar phase, supervised by knowledge
        distillation from the teacher (see ``distill.py``). Only the AB bridge
        parameters receive gradient updates.

        Args:
            batches: Sequence of ``(h_a, targets)`` tuples.
        """
        logger.info("Step 2: training gAB sidecar bridge (gBC and fC frozen).")
        stage_params = self._composition.named_stage_parameters()
        _freeze(stage_params["gbc"])
        _freeze(stage_params["fc"])
        _unfreeze(stage_params["gab"])

        import torch.optim as optim

        optimizer = optim.AdamW(
            [{"params": stage_params["gab"], "lr": self._settings.lr_gab}]
        )
        self._run_epochs(batches, optimizer, label="step2")

        # Restore grad computation for subsequent steps.
        _unfreeze(stage_params["gbc"])
        _unfreeze(stage_params["fc"])

    def step3_train_generation(self, batches: list[tuple["Tensor", "Tensor"]]) -> None:
        """Step 3: train gBC + fC; gAB is frozen.

        This mirrors the QLoRA-on-span-format stage in the paper. The upstream
        AB bridge representation is treated as fixed; only the generation
        conditioning bridge and head are updated.

        Args:
            batches: Sequence of ``(h_a, targets)`` tuples.
        """
        logger.info("Step 3: training gBC + fC (gAB frozen, QLoRA stage).")
        stage_params = self._composition.named_stage_parameters()
        _freeze(stage_params["gab"])
        _unfreeze(stage_params["gbc"])
        _unfreeze(stage_params["fc"])

        import torch.optim as optim

        optimizer = optim.AdamW(
            [
                {"params": stage_params["gbc"], "lr": self._settings.lr_gbc},
                {"params": stage_params["fc"], "lr": self._settings.lr_fc},
            ]
        )
        self._run_epochs(batches, optimizer, label="step3")

        _unfreeze(stage_params["gab"])

    def step4_joint_finetune(self, batches: list[tuple["Tensor", "Tensor"]]) -> None:
        """Step 4: unfreeze all stages and train with the partitioned optimizer.

        LR partitioning (Stabilizer 4) takes over as the protection mechanism:
        gAB trains at a lower rate than gBC so downstream noise cannot
        destabilise the upstream span-extraction representation.

        Args:
            batches: Sequence of ``(h_a, targets)`` tuples.
        """
        logger.info(
            "Step 4: joint fine-tuning with partitioned AdamW "
            "(LR partitioning is the primary Eq.13 stability lever)."
        )
        stage_params = self._composition.named_stage_parameters()
        for params in stage_params.values():
            _unfreeze(params)

        from .optimizer import build_partitioned_optimizer

        optimizer = build_partitioned_optimizer(self._composition, self._settings)
        self._run_epochs(batches, optimizer, label="step4")

    def fit(self, batches: list[tuple["Tensor", "Tensor"]]) -> None:
        """Execute the full four-step curriculum in order.

        Args:
            batches: Shared batch list passed to each step.
        """
        self.step1_freeze_perception()
        self.step2_train_sidecar(batches)
        self.step3_train_generation(batches)
        self.step4_joint_finetune(batches)
        logger.info("Training curriculum complete (%d total steps).", self._global_step)

    # ------------------------------------------------------------------
    # Internal training loop
    # ------------------------------------------------------------------

    def _run_epochs(
        self,
        batches: list[tuple["Tensor", "Tensor"]],
        optimizer: "torch.optim.Optimizer",
        *,
        label: str,
    ) -> None:
        """Inner loop: iterate over batches for ``epochs_per_step`` epochs.

        Each step:
          1. Forward the composition.
          2. Compute task loss (cross-entropy) + regularization.
          3. Backward.
          4. ``apply_stabilizers`` (grad norm recording, clipping, spike warn).
          5. ``optimizer.step()`` then ``zero_grad``.
          6. Optionally run MI diagnostic under ``torch.no_grad()``.
        """
        import torch
        import torch.nn.functional as F

        cfg = self._train_config
        for epoch in range(cfg.epochs_per_step):
            for step_idx, (h_a, targets) in enumerate(batches):
                optimizer.zero_grad()

                out = self._composition(h_a)

                # Task loss: cross-entropy of logits against integer targets.
                # logits shape: (batch, vocab); targets shape: (batch,)
                task_loss = F.cross_entropy(out.logits, targets)

                # Regularization loss — lazy import to decouple from the
                # parallel-written losses.regularization module.
                reg_loss = self._compute_regularized_loss(
                    task_loss, out.u_ab, out.u_bc
                )

                reg_loss.backward()

                from .stabilizers import apply_stabilizers

                stab_info = apply_stabilizers(
                    self._composition,
                    optimizer,
                    settings=self._settings,
                    grad_probe=self._grad_probe,
                )

                optimizer.step()
                self._global_step += 1

                if self._global_step % cfg.log_every_n_steps == 0:
                    logger.debug(
                        "[%s] epoch=%d step=%d loss=%.4f pre_clip_norm=%.4f spikes=%s",
                        label,
                        epoch,
                        step_idx,
                        float(reg_loss.detach()),
                        stab_info["clipped_from"],
                        stab_info["spikes"] or "none",
                    )

                if (
                    self._settings.enable_mine_diagnostic
                    and self._global_step % cfg.mi_diag_interval == 0
                ):
                    self._run_mi_diagnostic(h_a, out.u_ab, out.u_bc, targets)

    def _compute_regularized_loss(
        self,
        task_loss: "Tensor",
        u_ab: "Tensor",
        u_bc: "Tensor",
    ) -> "Tensor":
        """Combine task loss with the stage regularizer (Eq.8).

        Lazy-imports ``losses.regularization.total_loss`` to avoid a hard
        module-load dependency on the parallel-written regularization module at
        import time.
        """
        # Lazy import: losses.regularization is written in a parallel work stream
        # and may not be present in early integration; the import error surfaces
        # at call time rather than at module load.
        from ..losses.regularization import total_loss  # type: ignore[import]

        return total_loss(
            task_loss,
            u_ab,
            u_bc,
            lambda_ab=self._settings.lambda_ab,
            lambda_bc=self._settings.lambda_bc,
            kind=self._settings.regularizer,
        )

    def _run_mi_diagnostic(
        self,
        h_a: "Tensor",
        u_ab: "Tensor",
        u_bc: "Tensor",
        targets: "Tensor",
    ) -> None:
        """Run the MI estimator diagnostic under no_grad — never on the backward path.

        The MI report (Eq.11) is logged at DEBUG level. It is purely diagnostic:
        ``i_ux`` should decrease (compression) and ``i_uy`` should hold or rise
        (signal preserved). Adding it to the training loss would break the
        surrogate-loss design in Eq.8.
        """
        import torch

        # Lazy import: mi_estimator is written in a parallel work stream.
        from ..losses.mi_estimator import estimate_mi_report  # type: ignore[import]

        with torch.no_grad():
            report = estimate_mi_report(
                u_ab,
                h_a,
                targets,
                beta=self._settings.beta,
            )
        logger.debug(
            "MI diagnostic (step=%d): I(U;X)=%.4f  I(U;Y)=%.4f  "
            "objective=%.4f  compression_ratio=%s",
            self._global_step,
            report.i_ux,
            report.i_uy,
            report.objective,
            f"{report.compression_ratio:.4f}" if report.compression_ratio is not None else "n/a",
        )


# ---------------------------------------------------------------------------
# Smoke-test helper
# ---------------------------------------------------------------------------


def synthetic_batches(
    settings: "Settings",
    n: int = 8,
    *,
    vocab: int = 4096,
) -> list[tuple["Tensor", "Tensor"]]:
    """Generate random CPU batches for smoke-running the full training regime.

    Each batch is ``(h_a, targets)`` where:
      - ``h_a``: shape ``(batch_size, embed_dim)`` float32 random normal.
      - ``targets``: shape ``(batch_size,)`` random integer class indices in
        ``[0, vocab_size)``.  ``vocab_size`` is inferred from the composition's
        generation head output dimension (default 4096).

    Args:
        settings: Runtime settings; ``profile.embed_dim`` sets the embedding
            dimension; ``profile.gen_hidden_dim`` is used as a stand-in for
            vocab if a composition isn't available.
        n: Number of batches to generate.

    Returns:
        List of ``(h_a, targets)`` tuples, all on CPU.
    """
    import torch

    profile = settings.profile
    embed_dim = profile.embed_dim
    batch_size = 4

    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(n):
        h_a = torch.randn(batch_size, embed_dim)
        targets = torch.randint(0, vocab, (batch_size,))
        batches.append((h_a, targets))
    return batches
