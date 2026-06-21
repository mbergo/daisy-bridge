"""Bridge versioning: co-adapted (sidecar, qlora) pairs only (Section 8.3)."""

from __future__ import annotations

import pytest

from bridge_rag.versioning.blue_green import UnpairedVersionError, VersionedDeployment


def test_paired_load_succeeds() -> None:
    dep = VersionedDeployment()
    v = dep.load_pair("SIDECAR_V1", "QLORA_V1")
    assert v.tag == "SIDECAR_V1+QLORA_V1"


def test_unpaired_load_hard_fails() -> None:
    dep = VersionedDeployment()
    with pytest.raises(UnpairedVersionError):
        dep.load_pair("SIDECAR_V2", "QLORA_V1")


def test_atomic_switch_only_after_validation() -> None:
    dep = VersionedDeployment()  # blue defaults to SIDECAR_V1+QLORA_V1
    dep.stage_green(dep.load_pair("SIDECAR_V2", "QLORA_V2"))
    assert dep.validate_green() is True
    dep.atomic_switch()
    assert dep.current().tag == "SIDECAR_V2+QLORA_V2"
