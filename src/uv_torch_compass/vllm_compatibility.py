"""Evaluate vLLM wheel and dependency compatibility without network access."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from uv_torch_compass.candidate_failures import (
    FrameworkBinaryRequirement,
    FrameworkCompatibilityDecision,
    FrameworkCompatibilityEvidence,
    FrameworkCompatibilityStatus,
)
from uv_torch_compass.candidate_lock import LockedPackage
from uv_torch_compass.elf_dependencies import read_elf_needed
from uv_torch_compass.errors import ProbeError

CATALOG_REVIEWED_DATE = "2026-08-01"
CATALOG_SOURCE_URLS = (
    "https://docs.vllm.ai/en/v0.6.0/getting_started/installation.html",
    "https://pypi.org/pypi/vllm/0.26.0/json",
)

_CUDA_LIBRARY_PATTERN = re.compile(r"(?:^|/)libcudart\.so\.(?P<major>[0-9]+)$")
_MAX_NATIVE_LIBRARIES = 512
_OFFICIAL_PYPI_HOSTS = frozenset({"pypi.org", "files.pythonhosted.org"})


@dataclass(frozen=True, slots=True)
class DependencyAdvisory:
    """Describe one catalogued dependency range known to break a framework."""

    package: str
    incompatible: SpecifierSet
    summary: str
    suggestions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VllmCatalogEntry:
    """Record reviewed compatibility facts for an official vLLM wheel."""

    version: Version
    cuda_variant: str
    cuda_major: int
    source_url: str
    advisories: tuple[DependencyAdvisory, ...] = ()


_CATALOG = (
    VllmCatalogEntry(
        Version("0.6.0"),
        "cu121",
        12,
        CATALOG_SOURCE_URLS[0],
        (
            DependencyAdvisory(
                "transformers",
                SpecifierSet(">=5"),
                "vLLM 0.6.0 resolved with Transformers 5, which requires Torch "
                "APIs unavailable in its pinned Torch 2.4 runtime.",
                (
                    "Constrain transformers to 4.44.2 and rerun plan.",
                    "Upgrade vLLM if a current Transformers release is required.",
                ),
            ),
        ),
    ),
    VllmCatalogEntry(
        Version("0.26.0"),
        "cu130",
        13,
        CATALOG_SOURCE_URLS[1],
    ),
)


def catalog_entry(package: LockedPackage) -> VllmCatalogEntry | None:
    """Return reviewed facts for an official PyPI vLLM distribution."""
    if package.name != "vllm" or package.source_kind != "registry":
        return None
    if urlsplit(package.source_url).hostname not in _OFFICIAL_PYPI_HOSTS:
        return None
    try:
        version = Version(package.version)
    except InvalidVersion:
        return None
    return next((entry for entry in _CATALOG if entry.version == version), None)


def catalog_requirement(package: LockedPackage) -> FrameworkBinaryRequirement | None:
    """Return a reviewed CUDA requirement when the official catalog matches."""
    entry = catalog_entry(package)
    if entry is None:
        return None
    return FrameworkBinaryRequirement(
        "vllm",
        package.version,
        required_cuda_variant=entry.cuda_variant,
        required_cuda_major=entry.cuda_major,
        evidence=FrameworkCompatibilityEvidence.CATALOG,
        source_url=package.source_url,
    )


def inspect_vllm_native_libraries(
    environment: Path,
    *,
    version: str,
    source_url: str,
) -> FrameworkBinaryRequirement:
    """Inspect installed vLLM ELF files without importing the package.

    Raises:
        ProbeError: If native files are unsafe, malformed, or contradictory.
    """
    root = environment.resolve(strict=True)
    native_files = sorted(root.rglob("vllm/**/*.so"))
    if len(native_files) > _MAX_NATIVE_LIBRARIES:
        raise ProbeError("vLLM wheel contains too many native libraries")
    needed: list[str] = []
    for native_file in native_files:
        needed.extend(read_elf_needed(native_file, root=root))
    libraries = tuple(dict.fromkeys(needed))
    cuda_majors = {
        int(match.group("major"))
        for library in libraries
        if (match := _CUDA_LIBRARY_PATTERN.search(library)) is not None
    }
    if len(cuda_majors) > 1:
        raise ProbeError("vLLM wheel requires conflicting CUDA runtime majors")
    cuda_major = next(iter(cuda_majors), None)
    return FrameworkBinaryRequirement(
        "vllm",
        version,
        required_cuda_major=cuda_major,
        needed_libraries=libraries,
        evidence=(
            FrameworkCompatibilityEvidence.ELF
            if cuda_major is not None
            else FrameworkCompatibilityEvidence.UNKNOWN
        ),
        source_url=source_url,
    )


def decide_vllm_compatibility(
    candidate_backend: str,
    *,
    catalog: FrameworkBinaryRequirement | None,
    inspected: FrameworkBinaryRequirement | None,
) -> FrameworkCompatibilityDecision:
    """Decide whether reviewed and inspected vLLM requirements allow a backend."""
    if candidate_backend == "cpu":
        requirement = catalog or inspected
        if requirement is not None and (
            requirement.required_cuda_variant is not None
            or requirement.required_cuda_major is not None
        ):
            return FrameworkCompatibilityDecision(
                FrameworkCompatibilityStatus.INCOMPATIBLE,
                candidate_backend,
                f"vLLM {requirement.version} requires CUDA but candidate cpu does not",
                requirement,
            )
        return FrameworkCompatibilityDecision(
            FrameworkCompatibilityStatus.UNKNOWN,
            candidate_backend,
            "vLLM did not declare a verifiable CPU or CUDA wheel variant",
            requirement,
        )

    candidate_variant, candidate_major = _backend_identity(candidate_backend)
    if catalog is not None and catalog.required_cuda_variant != candidate_variant:
        return FrameworkCompatibilityDecision(
            FrameworkCompatibilityStatus.INCOMPATIBLE,
            candidate_backend,
            f"vLLM {catalog.version} requires {catalog.required_cuda_variant}, but "
            f"candidate {candidate_backend} provides {_display_cuda(candidate_backend)}",
            catalog,
        )
    if (
        catalog is not None
        and inspected is not None
        and inspected.required_cuda_major is not None
        and catalog.required_cuda_major != inspected.required_cuda_major
    ):
        return FrameworkCompatibilityDecision(
            FrameworkCompatibilityStatus.INCOMPATIBLE,
            candidate_backend,
            "the reviewed vLLM catalog conflicts with the inspected wheel CUDA major",
            inspected,
        )
    requirement = catalog or inspected
    if requirement is None:
        return FrameworkCompatibilityDecision(
            FrameworkCompatibilityStatus.UNKNOWN,
            candidate_backend,
            "vLLM wheel did not expose a verifiable CUDA runtime requirement",
        )
    required_major = requirement.required_cuda_major
    if required_major is not None and required_major != candidate_major:
        return FrameworkCompatibilityDecision(
            FrameworkCompatibilityStatus.INCOMPATIBLE,
            candidate_backend,
            f"vLLM {requirement.version} requires CUDA {required_major}, but candidate "
            f"{candidate_backend} provides {_display_cuda(candidate_backend)}",
            requirement,
        )
    if catalog is not None or required_major is not None:
        return FrameworkCompatibilityDecision(
            FrameworkCompatibilityStatus.COMPATIBLE,
            candidate_backend,
            f"vLLM {requirement.version} matches candidate {candidate_backend}",
            requirement,
        )
    return FrameworkCompatibilityDecision(
        FrameworkCompatibilityStatus.UNKNOWN,
        candidate_backend,
        "vLLM wheel did not expose a verifiable CUDA runtime requirement",
        requirement,
    )


def dependency_advisories(
    vllm: LockedPackage,
    packages: tuple[LockedPackage, ...],
) -> tuple[tuple[DependencyAdvisory, LockedPackage], ...]:
    """Return catalogued incompatibilities present in a resolved graph."""
    entry = catalog_entry(vllm)
    if entry is None:
        return ()
    by_name = {package.name: package for package in packages}
    matches: list[tuple[DependencyAdvisory, LockedPackage]] = []
    for advisory in entry.advisories:
        package = by_name.get(advisory.package)
        if package is None:
            continue
        try:
            version = Version(package.version)
        except InvalidVersion:
            continue
        if advisory.incompatible.contains(version, prereleases=True):
            matches.append((advisory, package))
    return tuple(matches)


def _backend_identity(backend: str) -> tuple[str, int]:
    match = re.fullmatch(r"cu(?P<digits>[0-9]{2,3})", backend)
    if match is None:
        raise ProbeError(f"invalid CUDA backend {backend!r}")
    digits = match.group("digits")
    major_text = digits[:-1]
    return backend, int(major_text)


def _display_cuda(backend: str) -> str:
    digits = backend.removeprefix("cu")
    return f"CUDA {int(digits[:-1])}.{digits[-1]}"
