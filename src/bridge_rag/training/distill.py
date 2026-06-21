"""Knowledge-distillation interface: Qwen3-8B teacher -> BGE-M3 student (Step 2).

Production path:
  - Teacher: Qwen3-8B (``profile.teacher_model``) produces span+reasoning
    supervision logits over the sidecar output vocabulary.
  - Student: BGE-M3 0.6B (``profile.sidecar_model``) is distilled under the
    KD loss below and must hit the <0.3 ms sidecar latency budget.
  - Budget: the student must satisfy ``profile.sidecar_budget_ms``. In prod
    that is 0.3 ms; in dev it is 50 ms (no assertion in CI against the prod
    number — see config.py).

Dev path (teacher_model == sidecar_model):
  - No heavy model load. ``distill`` logs that teacher==student and returns
    zero loss immediately. This keeps the dev E2E smoke-test honest about the
    code path without pulling Qwen3-8B onto a laptop.

KD loss formula (Hinton 2015, adapted for span logits):

    L_KD = alpha * KL(softmax(s/T) || softmax(t/T)) * T^2
         + (1 - alpha) * CE(s, hard_targets)

where:
  - ``s`` are student logits, ``t`` are teacher logits.
  - ``T`` is the softening temperature (``DistillConfig.temperature``).
  - ``alpha`` is the KD weight (``DistillConfig.alpha``).
  - ``T^2`` restores gradient magnitude (standard Hinton correction).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from ..config import ModelProfile

logger = logging.getLogger(__name__)


@dataclass
class DistillConfig:
    """Hyper-parameters for the KD distillation step.

    Args:
        temperature: Softening temperature T. Higher values produce softer
            probability distributions, exposing more of the teacher's
            inter-class similarity structure to the student.
        alpha: Weight on the KD (soft-target) loss term. ``1 - alpha`` is
            applied to the hard cross-entropy term.
    """

    temperature: float = 4.0
    alpha: float = 0.7


class SpanDistiller:
    """Distils span-extraction supervision from a teacher into a student bridge.

    Args:
        profile: The active ``ModelProfile``; used to detect the dev no-op path
            (``teacher_model == sidecar_model``) and to log latency budgets.
        config: KD hyper-parameters. Defaults to ``DistillConfig()``.
    """

    def __init__(
        self,
        profile: "ModelProfile",
        config: DistillConfig | None = None,
    ) -> None:
        self._profile = profile
        self._config = config or DistillConfig()
        self._is_identity = (profile.teacher_model == profile.sidecar_model)

        if self._is_identity:
            logger.info(
                "SpanDistiller: teacher_model == sidecar_model ('%s'). "
                "Running in identity (no-op) mode — distillation loss is zero. "
                "Production distillation requires teacher_model='%s' "
                "and student within sidecar_budget_ms=%.1f ms.",
                profile.teacher_model,
                "Qwen/Qwen3-8B",
                profile.sidecar_budget_ms,
            )

    def distill(
        self,
        student: "torch.nn.Module",  # noqa: F821
        teacher: "torch.nn.Module",  # noqa: F821
        batches: list[tuple["Tensor", "Tensor"]],
    ) -> dict:
        """Run KD training of student toward teacher over the provided batches.

        In dev mode (teacher == student model name), the method skips all
        heavy computation and returns immediately with zero loss.

        In prod mode, for each batch:
          1. Teacher forward under ``torch.no_grad()`` to obtain soft targets.
          2. Student forward to obtain logits.
          3. Compute ``L_KD = alpha * KL * T^2 + (1-alpha) * CE``.
          4. Backward + step on student parameters.

        Args:
            student: The student nn.Module (BGE-M3 bridge in production).
            teacher: The teacher nn.Module (Qwen3-8B in production).
            batches: Sequence of ``(inputs, hard_targets)`` tuples.

        Returns:
            ``{"mean_kd_loss": float, "steps": int, "identity_mode": bool}``
        """
        if self._is_identity:
            logger.debug(
                "SpanDistiller.distill: identity mode — skipping %d batches.",
                len(batches),
            )
            return {
                "mean_kd_loss": 0.0,
                "steps": 0,
                "identity_mode": True,
            }

        # Lazy import: avoid pulling torch at module load time.
        import torch
        import torch.nn.functional as F  # noqa: N812

        cfg = self._config
        T = cfg.temperature
        alpha = cfg.alpha

        optimizer = torch.optim.AdamW(student.parameters())
        total_kd_loss = 0.0
        steps = 0

        for inputs, hard_targets in batches:
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits: Tensor = teacher(inputs)

            student_logits: Tensor = student(inputs)

            # Soft KD loss — KL divergence with temperature scaling.
            soft_student = F.log_softmax(student_logits / T, dim=-1)
            soft_teacher = F.softmax(teacher_logits / T, dim=-1)
            kd_loss = (
                F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)
            )

            # Hard target cross-entropy.
            hard_loss = F.cross_entropy(student_logits, hard_targets)

            loss = alpha * kd_loss + (1.0 - alpha) * hard_loss
            loss.backward()
            optimizer.step()

            total_kd_loss += float(loss.detach())
            steps += 1

            logger.debug(
                "Distill step=%d  kd_loss=%.4f  hard_loss=%.4f  total=%.4f",
                steps,
                float(kd_loss.detach()),
                float(hard_loss.detach()),
                float(loss.detach()),
            )

        mean_loss = total_kd_loss / steps if steps else 0.0
        logger.info(
            "Distillation complete: %d steps, mean KD loss=%.4f",
            steps,
            mean_loss,
        )
        return {
            "mean_kd_loss": mean_loss,
            "steps": steps,
            "identity_mode": False,
        }


def build_distiller(
    profile: "ModelProfile",
    config: DistillConfig | None = None,
) -> SpanDistiller:
    """Construct a ``SpanDistiller`` for the given profile.

    Args:
        profile: The active ``ModelProfile`` (controls dev vs. prod path).
        config: Optional KD hyper-parameters.

    Returns:
        A ``SpanDistiller`` ready to call ``.distill(student, teacher, batches)``.

    Note:
        Production path distils Qwen3-8B span+reasoning supervision into the
        BGE-M3 0.6B student under the <0.3 ms latency budget
        (``profile.sidecar_budget_ms``). The dev path (teacher == sidecar model)
        is a verified no-op that keeps the smoke-test fast.
    """
    return SpanDistiller(profile=profile, config=config)
