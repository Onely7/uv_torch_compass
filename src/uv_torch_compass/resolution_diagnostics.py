"""Interpret redacted uv resolver failures as stable domain diagnostics."""

from __future__ import annotations

import re
from collections.abc import Sequence

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from uv_torch_compass.domain import (
    PYTORCH_PACKAGES,
    BackendCandidate,
    FailedIndex,
    FailedPackage,
    ResolutionFailure,
    ResolutionFailureKind,
)
from uv_torch_compass.redaction import redact

_MAX_DIAGNOSTIC_CHARACTERS = 65_536
_MAX_LINE_CHARACTERS = 2_000
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_DEPENDENCY = re.compile(
    r"(?P<dependent>[A-Za-z0-9_.-]+)"
    r"(?P<dependent_spec>(?:===|==|~=|!=|<=|>=|<|>)[^,\s]+)?"
    r"\s+depends on\s+"
    r"(?P<requirement>[A-Za-z0-9_.-]+(?:\[[^\]]+\])?"
    r"(?:(?:===|==|~=|!=|<=|>=|<|>)[^,\s]+)?)",
    re.IGNORECASE,
)
_NO_VERSION = re.compile(
    r"(?:no version of|only)\s+"
    r"(?P<requirement>[A-Za-z0-9_.-]+"
    r"(?:(?:===|==|~=|!=|<=|>=|<|>)[^,\s]+)?)",
    re.IGNORECASE,
)
_DISTRIBUTION = re.compile(
    r"Distribution\s+[`'](?P<requirement>[A-Za-z0-9_.-]+"
    r"(?:(?:===|==)[^@\s`']+)?)",
    re.IGNORECASE,
)
_FOUND_ON_INDEX = re.compile(
    r"(?P<package>[A-Za-z0-9_.-]+)\s+was found on\s+"
    r"(?P<url>https?://[^\s,]+)",
    re.IGNORECASE,
)
_PLATFORM = re.compile(
    r"(?:You're on|current platform(?: is)?)[^\n]*?"
    r"`(?P<platform>[^`\n]+)`",
    re.IGNORECASE,
)


def interpret_uv_failure(
    output: str,
    *,
    candidate: BackendCandidate,
    dependency_roots: Sequence[str],
) -> ResolutionFailure:
    """Convert uv text into a bounded, redacted diagnostic.

    Args:
        output: Combined uv stdout and stderr.
        candidate: Backend whose official index was used for PyTorch.
        dependency_roots: Selected requirements supplied to uv.

    Returns:
        A structured failure. Unknown formats produce an explicit fallback.
    """
    clean = _normalize_output(output)
    lowered = clean.lower()
    dependency = _dependency_context(clean, dependency_roots)
    package = dependency[0] or _package_from_failure(clean)
    required_by = dependency[1]
    platform = _match_group(_PLATFORM, clean, "platform")
    index = _index_for(package, clean, candidate)

    if _contains_any(lowered, ("401", "403", "unauthorized", "forbidden")):
        return _failure(
            ResolutionFailureKind.AUTHENTICATION,
            "The package index rejected authentication.",
            package,
            required_by,
            index,
            platform,
            ("Check the index credentials available to uv.",),
        )
    if _contains_any(
        lowered,
        (
            "failed to connect",
            "connection error",
            "connection timed out",
            "dns",
            "network",
        ),
    ):
        return _failure(
            ResolutionFailureKind.NETWORK,
            "uv could not reach a required package index.",
            package,
            required_by,
            index,
            platform,
            ("Check network, proxy, certificate, and index availability.",),
        )
    if _contains_any(lowered, ("failed to build", "build backend returned an error")):
        return _failure(
            ResolutionFailureKind.BUILD_FAILURE,
            "A required distribution could not be built.",
            package,
            required_by,
            index,
            platform,
            ("Inspect the private log for the redacted build backend output.",),
        )
    if _contains_any(
        lowered,
        (
            "doesn't have a source distribution or wheel",
            "no wheel",
            "only has wheels for",
        ),
    ):
        return _failure(
            ResolutionFailureKind.WHEEL_UNAVAILABLE,
            "The selected package version has no installable wheel for this platform.",
            package,
            required_by,
            index,
            platform,
            ("Select a package version that publishes a wheel for this platform.",),
        )
    if _NO_VERSION.search(clean) or (
        package is not None and "not at the requested version" in lowered
    ):
        suggestions = _distribution_suggestions(package, candidate)
        return _failure(
            ResolutionFailureKind.NO_COMPATIBLE_DISTRIBUTION,
            "The required package build is unavailable from this index.",
            package,
            required_by,
            index,
            platform,
            suggestions,
        )
    if _contains_any(
        lowered,
        ("requirements are unsatisfiable", "are incompatible", "depends on"),
    ):
        return _failure(
            ResolutionFailureKind.DEPENDENCY_CONFLICT,
            "The selected dependency requirements cannot be resolved together.",
            package,
            required_by,
            index,
            platform,
            ("Align or narrow the conflicting dependency requirements.",),
        )
    return _failure(
        ResolutionFailureKind.UNKNOWN,
        "uv did not provide a recognized structured failure reason.",
        package,
        required_by,
        index,
        platform,
        ("Inspect the private redacted log for the complete uv diagnostic.",),
    )


def runtime_failure(summary: str) -> ResolutionFailure:
    """Create a runtime-validation failure without parsing resolver output."""
    return ResolutionFailure(
        ResolutionFailureKind.RUNTIME_VALIDATION,
        summary,
        suggestions=("Inspect the private log for the failed runtime check.",),
    )


def _normalize_output(output: str) -> str:
    without_ansi = _ANSI_ESCAPE.sub("", output)
    printable = "".join(
        character
        for character in without_ansi
        if character in "\n\t" or ord(character) >= 32
    )
    clean = redact(printable)
    lines = (line[:_MAX_LINE_CHARACTERS].strip() for line in clean.splitlines())
    return "\n".join(line for line in lines if line)[:_MAX_DIAGNOSTIC_CHARACTERS]


def _dependency_context(
    output: str, roots: Sequence[str]
) -> tuple[FailedPackage | None, tuple[str, ...]]:
    match = _DEPENDENCY.search(output)
    if match is None:
        return None, ()
    requirement = _parse_requirement(match.group("requirement"))
    if requirement is None:
        return None, ()
    dependent = canonicalize_name(match.group("dependent"))
    root = next(
        (value for value in roots if _requirement_name(value) == dependent),
        None,
    )
    dependent_text = root or (
        match.group("dependent") + (match.group("dependent_spec") or "")
    )
    package = _failed_package(requirement)
    return package, (dependent_text, str(requirement))


def _package_from_failure(output: str) -> FailedPackage | None:
    for pattern in (_DISTRIBUTION, _NO_VERSION):
        match = pattern.search(output)
        if match is None:
            continue
        requirement = _parse_requirement(match.group("requirement"))
        if requirement is not None:
            return _failed_package(requirement)
    return None


def _failed_package(requirement: Requirement) -> FailedPackage:
    exact_versions = [
        specifier.version
        for specifier in requirement.specifier
        if specifier.operator in {"==", "==="} and "*" not in specifier.version
    ]
    return FailedPackage(
        str(canonicalize_name(requirement.name)),
        exact_versions[0] if len(exact_versions) == 1 else None,
        str(requirement),
    )


def _index_for(
    package: FailedPackage | None,
    output: str,
    candidate: BackendCandidate,
) -> FailedIndex | None:
    if package is not None and package.name in PYTORCH_PACKAGES:
        return FailedIndex(candidate.index_name, candidate.index_url)
    match = _FOUND_ON_INDEX.search(output)
    if match is None:
        return None
    return FailedIndex("", redact(match.group("url")).rstrip(".,"))


def _distribution_suggestions(
    package: FailedPackage | None, candidate: BackendCandidate
) -> tuple[str, ...]:
    suggestions = [
        "Select a dependency version compatible with a published PyTorch build.",
        "Update the NVIDIA driver and rerun plan if a newer CUDA backend is required.",
    ]
    if candidate.is_cuda:
        suggestions.append(
            "Explicitly select --backend cpu if CPU execution is acceptable."
        )
    if package is not None and package.name not in PYTORCH_PACKAGES:
        suggestions[0] = "Select a version published on the required package index."
    return tuple(suggestions)


def _parse_requirement(value: str) -> Requirement | None:
    try:
        return Requirement(value.rstrip(".;"))
    except InvalidRequirement:
        return None


def _requirement_name(value: str) -> str | None:
    requirement = _parse_requirement(value)
    return str(canonicalize_name(requirement.name)) if requirement is not None else None


def _match_group(pattern: re.Pattern[str], value: str, group: str) -> str | None:
    match = pattern.search(value)
    return match.group(group) if match is not None else None


def _contains_any(value: str, needles: Sequence[str]) -> bool:
    return any(needle in value for needle in needles)


def _failure(
    kind: ResolutionFailureKind,
    summary: str,
    package: FailedPackage | None,
    required_by: tuple[str, ...],
    index: FailedIndex | None,
    platform: str | None,
    suggestions: tuple[str, ...],
) -> ResolutionFailure:
    return ResolutionFailure(
        kind,
        summary,
        package,
        required_by,
        index,
        platform,
        suggestions,
    )
