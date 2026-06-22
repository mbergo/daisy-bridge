"""Bridge factory — maps BridgeKind to the concrete TensorBridge class.

Imported lazily by pipeline/composition.py so the concrete bridge families do
not create a hard import-time dependency from the pipeline layer down into the
implementation layer. The factory is the only module that knows all four
families by name.
"""

from __future__ import annotations

from ..config import BridgeKind
from .attention import AttentionBridge
from .base import TensorBridge
from .bottleneck import BottleneckBridge
from .gated import GatedBridge
from .residual import ResidualBridge

_KIND_TO_CLASS: dict[BridgeKind, type[TensorBridge]] = {
    BridgeKind.BOTTLENECK: BottleneckBridge,
    BridgeKind.GATED: GatedBridge,
    BridgeKind.ATTENTION: AttentionBridge,
    BridgeKind.RESIDUAL: ResidualBridge,
}


def build_bridge(
    kind: BridgeKind,
    *,
    in_dim: int,
    out_dim: int,
    dropout: float = 0.0,
    **kw: object,
) -> TensorBridge:
    """Instantiate the TensorBridge family identified by ``kind``.

    Args:
        kind: Which bridge family to construct.
        in_dim: Upstream representation dimensionality.
        out_dim: Downstream representation dimensionality.
        dropout: Bridge dropout probability (Stabilizer 5).
        **kw: Extra keyword arguments forwarded verbatim to the bridge
            constructor (e.g. ``bottleneck_width``, ``num_heads``).

    Returns:
        A freshly initialised ``TensorBridge`` instance.

    Raises:
        ValueError: If ``kind`` is ``RESIDUAL`` and ``in_dim != out_dim``.
    """
    if kind is BridgeKind.RESIDUAL and in_dim != out_dim:
        raise ValueError(
            f"BridgeKind.RESIDUAL requires in_dim == out_dim "
            f"(got in_dim={in_dim}, out_dim={out_dim}). "
            "Choose BridgeKind.BOTTLENECK or BridgeKind.ATTENTION when "
            "the upstream and downstream dimensions differ."
        )
    cls = _KIND_TO_CLASS[kind]
    return cls(in_dim, out_dim, dropout=dropout, **kw)
