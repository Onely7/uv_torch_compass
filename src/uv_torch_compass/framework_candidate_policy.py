"""Narrow backend candidates using reviewed direct framework requirements."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import tomlkit
from packaging.version import Version
from tomlkit.exceptions import ParseError

from uv_torch_compass.domain import ProjectRequirements
from uv_torch_compass.errors import ConfigurationError
from uv_torch_compass.vllm_compatibility import catalog_backend_for_version


@dataclass(frozen=True, slots=True)
class FrameworkCandidateConstraint:
    """Describe a reviewed backend constraint known before dependency locking."""

    package: str
    requested: str
    resolved_version: str
    required_backend: str


@dataclass(frozen=True, slots=True)
class FrameworkVersionSelection:
    """Record a verified framework version selected from a broader request."""

    package: str
    requested: str
    resolved_version: str
    rejected_versions: tuple[str, ...]


def direct_vllm_candidate_constraint(
    requirements: ProjectRequirements,
    pyproject: Path,
) -> FrameworkCandidateConstraint | None:
    """Return a safe catalog constraint for one exact official vLLM requirement.

    Custom vLLM sources are deliberately excluded because facts reviewed for an
    official wheel cannot be transferred to a repackaged or locally built wheel.

    Raises:
        ConfigurationError: If the target source table cannot be parsed safely.
    """
    selected = requirements.requirement_for("vllm")
    if len(selected) != 1 or selected[0].url is not None:
        return None
    specifiers = tuple(selected[0].specifier)
    if (
        len(specifiers) != 1
        or specifiers[0].operator not in {"==", "==="}
        or "*" in specifiers[0].version
    ):
        return None
    if _has_custom_vllm_source(pyproject):
        return None
    version = str(Version(specifiers[0].version))
    backend = catalog_backend_for_version(version)
    if backend is None:
        return None
    return FrameworkCandidateConstraint(
        "vllm",
        str(selected[0]),
        version,
        backend,
    )


def vllm_version_search_request(
    requirements: ProjectRequirements,
    pyproject: Path,
) -> str | None:
    """Return a direct vLLM range eligible for bounded candidate-only search."""
    selected = requirements.requirement_for("vllm")
    if len(selected) != 1 or selected[0].url is not None:
        return None
    specifiers = tuple(selected[0].specifier)
    exact = any(
        item.operator in {"==", "==="} and "*" not in item.version
        for item in specifiers
    )
    if exact or _has_custom_vllm_source(pyproject):
        return None
    return str(selected[0])


def _has_custom_vllm_source(pyproject: Path) -> bool:
    try:
        document = tomlkit.parse(pyproject.read_text(encoding="utf-8"))
    except (OSError, ParseError) as exc:
        raise ConfigurationError(
            f"failed to inspect vLLM source policy: {exc}"
        ) from exc
    tool = document.get("tool", {})
    if not isinstance(tool, Mapping):
        raise ConfigurationError("[tool] must be a table")
    uv = tool.get("uv", {})
    if not isinstance(uv, Mapping):
        raise ConfigurationError("[tool.uv] must be a table")
    sources = uv.get("sources", {})
    if not isinstance(sources, Mapping):
        raise ConfigurationError("[tool.uv].sources must be a table")
    return "vllm" in sources
