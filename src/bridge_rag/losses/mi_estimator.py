"""Mutual-information diagnostics — DIAGNOSTIC ONLY, never on the backward path.

WARNING: This module is for EVALUATION / REPORTING ONLY.
Do NOT add any output of this module to a training loss. All public entry
points should be called under ``torch.no_grad()`` in the trainer's eval loop.
The trainer uses these estimates to populate ``MIReport`` in the invariant
probes (Eq.11 monitoring), NOT to drive gradient descent.

Two estimators are provided:

- ``MINE`` (trainable): Mutual Information Neural Estimator (Belghazi et al.,
  2018). Uses the Donsker-Varadhan representation. Has its own parameters —
  useful when a smooth, lower-variance estimate over many batches is needed,
  but requires separate optimisation of the statistics network T.

- ``infonce_mi`` (parameter-free): InfoNCE lower bound (van den Oord et al.,
  2018). Treats each sample as a query and all other samples in the batch as
  negatives. Lower variance than MINE for moderate batch sizes; RECOMMENDED as
  the default diagnostic.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..contracts.invariants import MIReport

logger = logging.getLogger(__name__)


class MINE(nn.Module):
    """Mutual Information Neural Estimator (Donsker-Varadhan bound).

    Trains a statistics network T(x, z) and estimates MI as:

        I(X; Z) ≈ E[T(x, z)] - log E[e^{T(x, z')}]

    where z' is a shuffled copy of z (marginal samples / negatives).

    DIAGNOSTIC ONLY — the parameters of this network should NOT be part of
    the main model's optimizer. Maintain a separate MINE optimizer or use it
    purely in ``torch.no_grad()`` mode with a pre-trained T.

    Args:
        x_dim: Dimensionality of the X variable.
        z_dim: Dimensionality of the Z variable.
        hidden_dim: Hidden layer size of the statistics network T.
    """

    def __init__(self, x_dim: int, z_dim: int, *, hidden_dim: int = 128) -> None:
        super().__init__()
        self.T = nn.Sequential(
            nn.Linear(x_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        """Concatenate (x, z) and score with T. Returns raw logit."""
        return self.T(torch.cat([x, z], dim=-1)).squeeze(-1)

    def mutual_information(self, x: Tensor, z: Tensor) -> Tensor:
        """Donsker-Varadhan MI estimate over the batch.

        Args:
            x: Paired samples, shape (B, x_dim).
            z: Paired samples, shape (B, z_dim).

        Returns:
            Scalar tensor: DV lower bound on I(X; Z).
        """
        # Joint term: E[T(x, z)] over paired samples
        t_joint = self.forward(x, z)
        joint_term = t_joint.mean()

        # Marginal term: E[e^{T(x, z')}] with z' shuffled along the batch dim
        idx = torch.randperm(z.shape[0], device=z.device)
        z_shuffle = z[idx]
        t_marginal = self.forward(x, z_shuffle)
        # Stable log-sum-exp: log mean(e^t) = logsumexp(t) - log(B)
        marginal_term = torch.logsumexp(t_marginal, dim=0) - torch.log(
            torch.tensor(float(z.shape[0]), device=z.device)
        )

        return joint_term - marginal_term


def infonce_mi(x: Tensor, z: Tensor, *, temperature: float = 0.1) -> Tensor:
    """InfoNCE lower bound on I(X; Z) — parameter-free, RECOMMENDED.

    Treats each sample i as a query against a batch of N keys. The bound is:

        I(X; Z) ≥ log(N) - L_NCE

    where L_NCE is the average cross-entropy loss of identifying the paired
    (x_i, z_i) among all N candidates.

    Args:
        x: First variable, shape (B, dx). Projected to a similarity space via
           cosine similarity with z.
        z: Second variable, shape (B, dz). Must have the same batch size as x.
           If dx != dz the raw dot product will be replaced by a normalised
           cosine similarity after projecting both to a common space via L2
           normalisation only.
        temperature: Softmax temperature τ (lower → sharper discrimination).

    Returns:
        Scalar tensor: InfoNCE MI lower bound (nats).

    Note:
        For best results use a batch size ≥ 64. Smaller batches cause the
        estimate to be noisy because there are few negatives.
    """
    B = x.shape[0]
    if B < 2:
        logger.warning("infonce_mi: batch size %d is too small for a reliable estimate.", B)
        return torch.zeros(1, device=x.device, dtype=x.dtype).squeeze()

    # L2-normalise both representations before dot-product scoring.
    # This avoids a learned linear projection and keeps the estimator
    # parameter-free while still capturing alignment.
    x_norm = F.normalize(x.reshape(B, -1).float(), dim=-1)  # (B, dx_flat)
    z_norm = F.normalize(z.reshape(B, -1).float(), dim=-1)  # (B, dz_flat)

    # If feature dims differ we cannot directly dot-product; use a common
    # dimension by truncating to the smaller. The estimate is still valid
    # (it lower-bounds the true MI of the truncated projections).
    dx, dz = x_norm.shape[-1], z_norm.shape[-1]
    if dx != dz:
        dim = min(dx, dz)
        x_norm = x_norm[..., :dim]
        z_norm = z_norm[..., :dim]

    # Similarity matrix: logits[i,j] = cos(x_i, z_j) / τ
    logits = torch.matmul(x_norm, z_norm.T) / temperature  # (B, B)

    # Positive pairs are on the diagonal
    targets = torch.arange(B, device=x.device)
    loss = F.cross_entropy(logits, targets)

    # InfoNCE bound: log(N) - L_NCE  (in nats)
    import math
    mi_estimate = math.log(B) - loss.item()
    return torch.tensor(mi_estimate, device=x.device, dtype=x.dtype)


def estimate_mi_report(
    u: Tensor,
    x_in: Tensor,
    y_out: Tensor,
    *,
    beta: float = 1.0,
) -> MIReport:
    """Compute I(U;X) and I(U;Y) and wrap them in an MIReport.

    DIAGNOSTIC ONLY — call this inside ``torch.no_grad()``.

    Uses ``infonce_mi`` for both estimates. I(U;X) should fall (compression)
    and I(U;Y) should hold/rise (signal preserved) as training progresses.

    Args:
        u: Bridge representation U, shape (B, du).
        x_in: Upstream input X, shape (B, dx).
        y_out: Downstream target / output Y, shape (B, dy).
        beta: β coefficient matching the IB objective (Eq.11).

    Returns:
        ``MIReport`` with ``i_ux``, ``i_uy``, and ``beta`` populated.
    """
    i_ux_t = infonce_mi(u, x_in)
    i_uy_t = infonce_mi(u, y_out)
    i_ux = float(i_ux_t.item())
    i_uy = float(i_uy_t.item())
    logger.debug("MI diagnostic: I(U;X)=%.4f  I(U;Y)=%.4f  β=%.4f", i_ux, i_uy, beta)
    return MIReport(i_ux=i_ux, i_uy=i_uy, beta=beta)
