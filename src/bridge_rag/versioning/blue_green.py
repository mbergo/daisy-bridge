"""Blue/green deployment for co-adapted (SIDECAR_Vn, QLORA_Vn) version pairs.

Section 8.3 of the paper: the sidecar span extractor and the generator's QLoRA
adapter are trained together. Serving an unpaired combination degrades quality in
a way that is invisible to per-component health checks. This module enforces
pairing at load time — the only permitted traffic-switch path is
``load_pair -> stage_green -> validate_green -> atomic_switch``.

An unpaired ``BridgeVersion`` never becomes ``current()``.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Optional

from ..types import BridgeVersion

logger = logging.getLogger(__name__)

_VERSION_RE = re.compile(r"^SIDECAR_V(\d+)$", re.ASCII)
_QLORA_RE = re.compile(r"^QLORA_V(\d+)$", re.ASCII)


class UnpairedVersionError(RuntimeError):
    """Raised when a sidecar tag and a QLoRA tag do not correspond to the same Vn.

    Example: ``SIDECAR_V2`` must pair with ``QLORA_V2``.  Loading
    ``SIDECAR_V2`` + ``QLORA_V3`` raises this error immediately rather than
    silently degrading retrieval-generation alignment.
    """


def _parse_version_number(sidecar_tag: str, qlora_tag: str) -> tuple[int, int]:
    """Extract the integer *n* from ``SIDECAR_Vn`` and ``QLORA_Vn``.

    Args:
        sidecar_tag: e.g. ``"SIDECAR_V2"``.
        qlora_tag:   e.g. ``"QLORA_V2"``.

    Returns:
        A ``(sidecar_n, qlora_n)`` integer pair.

    Raises:
        UnpairedVersionError: If either tag does not match the expected format.
    """
    sm = _VERSION_RE.match(sidecar_tag)
    if sm is None:
        raise UnpairedVersionError(
            f"sidecar_tag {sidecar_tag!r} does not match SIDECAR_Vn format. "
            "Expected e.g. 'SIDECAR_V2'."
        )
    qm = _QLORA_RE.match(qlora_tag)
    if qm is None:
        raise UnpairedVersionError(
            f"qlora_tag {qlora_tag!r} does not match QLORA_Vn format. "
            "Expected e.g. 'QLORA_V2'."
        )
    return int(sm.group(1)), int(qm.group(1))


class VersionedDeployment:
    """Manages a blue (live) and optional green (staged) ``BridgeVersion`` pair.

    Traffic model: blue receives 100% of requests; green receives 0% while under
    validation.  ``atomic_switch`` promotes green to blue in a single reference
    swap (lock-protected) so there is never a window where ``current()`` is
    undefined.

    All methods are thread-safe.

    Args:
        initial_sidecar_tag: Tag for the initial blue sidecar, e.g. ``"SIDECAR_V1"``.
        initial_qlora_tag:   Tag for the initial blue QLoRA, e.g. ``"QLORA_V1"``.

    Raises:
        UnpairedVersionError: If the initial tags are not a valid pair.
    """

    def __init__(
        self,
        initial_sidecar_tag: str = "SIDECAR_V1",
        initial_qlora_tag: str = "QLORA_V1",
    ) -> None:
        self._lock = threading.Lock()
        self._blue: BridgeVersion = self.load_pair(initial_sidecar_tag, initial_qlora_tag)
        self._green: Optional[BridgeVersion] = None
        self._green_validated: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def load_pair(sidecar_tag: str, qlora_tag: str) -> BridgeVersion:
        """Construct a ``BridgeVersion`` and hard-fail if tags are not paired.

        Pairing rule: the integer *n* in ``SIDECAR_Vn`` must equal the integer
        *n* in ``QLORA_Vn``.

        Args:
            sidecar_tag: e.g. ``"SIDECAR_V2"``.
            qlora_tag:   e.g. ``"QLORA_V2"``.

        Returns:
            A validated ``BridgeVersion`` instance.

        Raises:
            UnpairedVersionError: If the version numbers do not match or either
                tag has an invalid format.
        """
        sn, qn = _parse_version_number(sidecar_tag, qlora_tag)
        if sn != qn:
            raise UnpairedVersionError(
                f"Version mismatch: {sidecar_tag!r} (n={sn}) paired with "
                f"{qlora_tag!r} (n={qn}). Sidecar and QLoRA must share the "
                "same version number because they are co-adapted."
            )
        version = BridgeVersion(sidecar=sidecar_tag, qlora=qlora_tag)
        logger.info("load_pair: validated %s", version.tag)
        return version

    def stage_green(self, version: BridgeVersion) -> None:
        """Install *version* as the green (staged, 0%-traffic) deployment.

        The version must have been produced by ``load_pair`` (i.e. already
        validated for internal pairing).  This does NOT promote it to live
        traffic — call ``atomic_switch`` after ``validate_green`` passes.

        Args:
            version: A ``BridgeVersion`` to stage. Use ``load_pair`` to
                construct it so the pairing invariant is enforced before staging.
        """
        with self._lock:
            self._green = version
            self._green_validated = False
            logger.info(
                "stage_green: staged %s (not yet validated or promoted)", version.tag
            )

    def validate_green(self) -> bool:
        """Run validation checks on the green deployment.

        Currently performs structural validation (confirms the green version is
        loaded and internally paired).  In production this hook should be
        extended to run shadow traffic evaluation, canary metric checks, and
        faithfulness probes before returning ``True``.

        Returns:
            ``True`` if green is loaded and passes all checks; ``False``
            otherwise (including when no green is staged).
        """
        with self._lock:
            if self._green is None:
                logger.warning("validate_green: no green version staged")
                return False

            # Structural check: re-validate the pairing invariant on the
            # already-staged object (defence in depth).
            try:
                sn, qn = _parse_version_number(self._green.sidecar, self._green.qlora)
                if sn != qn:
                    logger.error(
                        "validate_green: green version %s failed pairing check",
                        self._green.tag,
                    )
                    return False
            except UnpairedVersionError as exc:
                logger.error("validate_green: structural failure — %s", exc)
                return False

            self._green_validated = True
            logger.info("validate_green: %s passed", self._green.tag)
            return True

    def atomic_switch(self) -> None:
        """Promote green to blue atomically (single reference swap).

        This is the only permitted path to changing the live version. The swap
        is protected by a lock so ``current()`` never returns ``None``.

        Raises:
            RuntimeError: If no green is staged or green has not been validated
                via ``validate_green``.
        """
        with self._lock:
            if self._green is None:
                raise RuntimeError(
                    "atomic_switch: no green version staged. "
                    "Call stage_green() first."
                )
            if not self._green_validated:
                raise RuntimeError(
                    f"atomic_switch: green version {self._green.tag} has not "
                    "been validated. Call validate_green() and confirm it "
                    "returns True before switching."
                )
            old_tag = self._blue.tag
            self._blue = self._green
            self._green = None
            self._green_validated = False
            logger.info(
                "atomic_switch: promoted %s (was %s)", self._blue.tag, old_tag
            )

    def current(self) -> BridgeVersion:
        """Return the currently live (blue) ``BridgeVersion``.

        Always returns a valid, paired version.  Never ``None``.
        """
        with self._lock:
            return self._blue

    @property
    def green(self) -> Optional[BridgeVersion]:
        """The staged green version, or ``None`` if none is staged."""
        with self._lock:
            return self._green

    @property
    def green_validated(self) -> bool:
        """Whether the current green has passed ``validate_green``."""
        with self._lock:
            return self._green_validated
