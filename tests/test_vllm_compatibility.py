from pathlib import Path
from typing import cast

from test_elf_dependencies import _write_elf

from uv_torch_compass.candidate_failures import FrameworkCompatibilityStatus
from uv_torch_compass.candidate_lock import LockedPackage
from uv_torch_compass.vllm_compatibility import (
    catalog_requirement,
    decide_vllm_compatibility,
    dependency_advisories,
    framework_catalog_metadata,
    inspect_vllm_native_libraries,
)


def _package(name: str, version: str) -> LockedPackage:
    return LockedPackage(name, version, "https://pypi.org/simple", (), "registry")


def test_catalog_restricts_reviewed_vllm_wheels_to_their_cuda_variant() -> None:
    vllm = _package("vllm", "0.6.0")
    requirement = catalog_requirement(vllm)

    rejected = decide_vllm_compatibility(
        "cu124",
        catalog=requirement,
        inspected=None,
    )
    accepted = decide_vllm_compatibility(
        "cu121",
        catalog=requirement,
        inspected=None,
    )

    assert rejected.status is FrameworkCompatibilityStatus.INCOMPATIBLE
    assert "requires cu121" in rejected.summary
    assert accepted.status is FrameworkCompatibilityStatus.COMPATIBLE


def test_catalog_reports_transformers_major_drift() -> None:
    vllm = _package("vllm", "0.6.0")

    advisories = dependency_advisories(
        vllm,
        (vllm, _package("transformers", "5.14.1"), _package("torch", "2.4.0")),
    )

    assert advisories[0][1].version == "5.14.1"
    assert "4.44.2" in advisories[0][0].suggestions[0]


def test_inspects_vllm_cuda_soname_without_importing_it(tmp_path: Path) -> None:
    library = tmp_path / "lib/python/site-packages/vllm/_C.abi3.so"
    library.parent.mkdir(parents=True)
    _write_elf(library, ("libcudart.so.13", "libtorch.so"))

    requirement = inspect_vllm_native_libraries(
        tmp_path,
        version="0.26.0",
        source_url="https://pypi.org/simple",
    )

    assert requirement.required_cuda_major == 13
    assert requirement.needed_libraries == ("libcudart.so.13", "libtorch.so")


def test_framework_catalog_exposes_reviewed_provenance() -> None:
    metadata = framework_catalog_metadata()
    sources = cast(list[str], metadata["sources"])
    entries = cast(list[dict[str, object]], metadata["entries"])

    assert metadata["reviewed_date"] == "2026-08-01"
    assert len(sources) == 2
    assert {entry["version"] for entry in entries} == {
        "0.6.0",
        "0.26.0",
    }
