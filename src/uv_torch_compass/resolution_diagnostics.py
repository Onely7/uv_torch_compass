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
_BUILD_PACKAGE = re.compile(
    r"(?:Failed to build|build backend.*for)\s+[`']"
    r"(?P<requirement>[A-Za-z0-9_.-]+(?:(?:===|==)[^`\s']+)?)",
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
_REGISTRY_MISSING = re.compile(
    r"(?P<requirement>[A-Za-z0-9_.-]+"
    r"(?:(?:===|==|~=|!=|<=|>=|<|>)[^,\s]+)?)"
    r"\s+was not found in the package registry",
    re.IGNORECASE,
)
_WHEEL_PLATFORM = re.compile(r"`([^`\n]+)`")
_NON_PACKAGE_WORDS = frozenset(
    {"a", "an", "the", "this", "that", "these", "those", "following"}
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
    explicit_package = _package_from_failure(clean)
    dependency = _dependency_context(clean, dependency_roots, explicit_package)
    package = explicit_package or dependency[0]
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
        failure = _failure(
            ResolutionFailureKind.WHEEL_UNAVAILABLE,
            "The selected package version has no installable wheel for this platform.",
            package,
            required_by,
            index,
            platform,
            ("Select a package version that publishes a wheel for this platform.",),
        )
        return ResolutionFailure(
            failure.kind,
            failure.summary,
            failure.package,
            failure.required_by,
            failure.index,
            failure.platform,
            failure.suggestions,
            available_wheel_platforms=_wheel_platforms(clean),
        )
    if (
        _NO_VERSION.search(clean)
        or _REGISTRY_MISSING.search(clean)
        or (package is not None and "not at the requested version" in lowered)
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


def timeout_failure(stage: str) -> ResolutionFailure:
    """Create a timeout diagnostic for one candidate operation stage."""
    return ResolutionFailure(
        ResolutionFailureKind.TIMEOUT,
        f"The candidate {stage} command exceeded its timeout.",
        suggestions=(
            "Retry after checking index availability and the configured timeout.",
        ),
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
    output: str,
    roots: Sequence[str],
    target: FailedPackage | None,
) -> tuple[FailedPackage | None, tuple[str, ...]]:
    matches = list(_DEPENDENCY.finditer(output))
    if not matches:
        return None, ()
    edges: list[tuple[str, str, Requirement]] = []
    for match in matches:
        requirement = _parse_requirement(match.group("requirement"))
        if requirement is None:
            continue
        dependent = str(canonicalize_name(match.group("dependent")))
        dependent_text = match.group("dependent") + (
            match.group("dependent_spec") or ""
        )
        edges.append((dependent, dependent_text, requirement))
    if not edges:
        return None, ()

    target_name = target.name if target is not None else None
    root_by_name = {
        name: value for value in roots if (name := _requirement_name(value)) is not None
    }
    for root_name, root_text in root_by_name.items():
        path = _dependency_path(edges, root_name, target_name)
        if path:
            return _failed_package(path[-1]), (root_text, *(str(item) for item in path))

    dependent, dependent_text, requirement = edges[0]
    del dependent
    return _failed_package(requirement), (
        root_by_name.get(_requirement_name(dependent_text), dependent_text),
        str(requirement),
    )


def _dependency_path(
    edges: Sequence[tuple[str, str, Requirement]],
    root: str,
    target: str | None,
) -> tuple[Requirement, ...]:
    adjacency: dict[str, list[Requirement]] = {}
    for dependent, _text, requirement in edges:
        adjacency.setdefault(dependent, []).append(requirement)
    pending: list[tuple[str, tuple[Requirement, ...]]] = [(root, ())]
    visited: set[str] = set()
    while pending:
        package, path = pending.pop(0)
        if package in visited:
            continue
        visited.add(package)
        for requirement in adjacency.get(package, []):
            child = str(canonicalize_name(requirement.name))
            child_path = (*path, requirement)
            if target is None or child == target:
                return child_path
            pending.append((child, child_path))
    return ()


def _package_from_failure(output: str) -> FailedPackage | None:
    for pattern in (_DISTRIBUTION, _BUILD_PACKAGE, _NO_VERSION, _REGISTRY_MISSING):
        match = pattern.search(output)
        if match is None:
            continue
        requirement = _parse_requirement(match.group("requirement"))
        if requirement is not None:
            return _failed_package(requirement)
    index_match = _FOUND_ON_INDEX.search(output)
    if index_match is not None:
        return FailedPackage(str(canonicalize_name(index_match.group("package"))))
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
        requirement = Requirement(value.rstrip(".;"))
    except InvalidRequirement:
        return None
    if str(canonicalize_name(requirement.name)) in _NON_PACKAGE_WORDS:
        return None
    return requirement


def _requirement_name(value: str) -> str | None:
    requirement = _parse_requirement(value)
    return str(canonicalize_name(requirement.name)) if requirement is not None else None


def _match_group(pattern: re.Pattern[str], value: str, group: str) -> str | None:
    match = pattern.search(value)
    return match.group(group) if match is not None else None


def _contains_any(value: str, needles: Sequence[str]) -> bool:
    return any(needle in value for needle in needles)


def _wheel_platforms(output: str) -> tuple[str, ...]:
    marker = "only has wheels for the following platforms:"
    lowered = output.lower()
    position = lowered.find(marker)
    if position < 0:
        return ()
    section = output[position + len(marker) : position + len(marker) + 2_000]
    return tuple(dict.fromkeys(_WHEEL_PLATFORM.findall(section)))[:20]


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
