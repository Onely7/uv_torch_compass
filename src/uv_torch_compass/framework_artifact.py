"""Inspect locked framework wheels before installing their dependency graphs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from uv_torch_compass.candidate_failures import (
    FrameworkBinaryRequirement,
    FrameworkCompatibilityDecision,
)
from uv_torch_compass.candidate_resolution import CandidateResolution
from uv_torch_compass.command_runner import CommandResult
from uv_torch_compass.uv_commands import UvCommandClient
from uv_torch_compass.vllm_compatibility import (
    catalog_requirement,
    decide_vllm_compatibility,
    inspect_vllm_native_libraries,
)

_REQUIRED_CAPABILITIES = frozenset({"--only-install-package", "--no-build-package"})


@dataclass(frozen=True, slots=True)
class FrameworkArtifactPreflight:
    """Describe the artifact evidence collected before a full installation."""

    status: str
    decision: FrameworkCompatibilityDecision | None
    requirement: FrameworkBinaryRequirement | None
    detail: str = ""


@dataclass(slots=True)
class FrameworkArtifactInspector:
    """Inspect each immutable vLLM artifact at most once per command run."""

    uv: UvCommandClient
    temporary_root: Path
    python: Path
    capabilities: frozenset[str]
    _cache: dict[tuple[object, ...], FrameworkBinaryRequirement] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def inspect(
        self,
        resolution: CandidateResolution,
        project_dir: Path,
    ) -> tuple[FrameworkArtifactPreflight, CommandResult | None]:
        """Inspect vLLM and compare its CUDA requirement with the candidate.

        Returns:
            The compatibility result and optional uv extraction command result.

        Raises:
            CommandTimeoutError: If selective wheel extraction times out.
            ProbeError: If a wheel is malformed or contradicts reviewed facts.
        """
        vllm = resolution.lock.package("vllm")
        if vllm is None:
            return FrameworkArtifactPreflight("not-present", None, None), None
        reviewed = catalog_requirement(vllm)
        reviewed_decision = decide_vllm_compatibility(
            resolution.backend.value,
            catalog=reviewed,
            inspected=None,
        )
        if not reviewed_decision.allowed:
            return (
                FrameworkArtifactPreflight(
                    "catalog-rejected",
                    reviewed_decision,
                    reviewed,
                    reviewed_decision.summary,
                ),
                None,
            )
        if not _REQUIRED_CAPABILITIES.issubset(self.capabilities):
            return (
                FrameworkArtifactPreflight(
                    "unsupported-uv",
                    reviewed_decision if reviewed is not None else None,
                    reviewed,
                    "the active uv does not support selective wheel extraction",
                ),
                None,
            )
        if vllm.source_kind != "registry" or not vllm.wheels:
            return (
                FrameworkArtifactPreflight(
                    "not-a-registry-wheel",
                    reviewed_decision if reviewed is not None else None,
                    reviewed,
                    "vLLM did not resolve to an inspectable registry wheel",
                ),
                None,
            )

        key = _artifact_key(resolution, vllm.version, vllm.source_url, vllm.wheels)
        inspected = self._cache.get(key)
        command_result: CommandResult | None = None
        if inspected is None:
            environment = self.temporary_root / (
                "framework-artifact-" + _key_digest(key)
            )
            command_result = self.uv.sync_locked_package(
                environment,
                project_dir,
                self.python,
                "vllm",
            )
            if command_result.returncode != 0:
                return (
                    FrameworkArtifactPreflight(
                        "wheel-unavailable",
                        reviewed_decision if reviewed is not None else None,
                        reviewed,
                        "the locked vLLM wheel could not be extracted for inspection",
                    ),
                    command_result,
                )
            inspected = inspect_vllm_native_libraries(
                environment,
                version=vllm.version,
                source_url=vllm.source_url,
            )
            self._cache[key] = inspected

        decision = decide_vllm_compatibility(
            resolution.backend.value,
            catalog=reviewed,
            inspected=inspected,
        )
        requirement = reviewed or inspected
        return (
            FrameworkArtifactPreflight(
                "inspected",
                decision,
                requirement,
                decision.summary,
            ),
            command_result,
        )


def _artifact_key(
    resolution: CandidateResolution,
    version: str,
    source_url: str,
    artifacts: tuple[object, ...],
) -> tuple[object, ...]:
    return (
        version,
        source_url,
        tuple(repr(artifact) for artifact in artifacts),
        resolution.environment.implementation_name,
        resolution.environment.python_minor,
        resolution.environment.sys_platform,
        resolution.environment.platform_machine,
    )


def _key_digest(key: tuple[object, ...]) -> str:
    return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:16]
