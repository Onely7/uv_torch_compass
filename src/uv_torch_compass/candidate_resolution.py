"""Represent the dependency resolution completed before candidate installation."""

from __future__ import annotations

from dataclasses import dataclass, replace

from uv_torch_compass.candidate_environment import CandidateExecutionEnvironment
from uv_torch_compass.candidate_lock import CandidateLockSnapshot, LockedPackage
from uv_torch_compass.domain import BackendCandidate, ResolutionFailure

_FRAMEWORK_DIAGNOSTIC_PACKAGES = frozenset(
    {
        "torch",
        "torchvision",
        "torchaudio",
        "vllm",
        "transformers",
        "xformers",
        "vllm-flash-attn",
    }
)


@dataclass(frozen=True, slots=True)
class CandidateResolution:
    """Describe one backend's resolved graph before side-effectful installation."""

    backend: BackendCandidate
    environment: CandidateExecutionEnvironment
    lock: CandidateLockSnapshot

    @property
    def pytorch_packages(self) -> tuple[LockedPackage, ...]:
        """Return resolved PyTorch ecosystem package identities."""
        return self.lock.pytorch_packages

    @property
    def framework_packages(self) -> tuple[LockedPackage, ...]:
        """Return packages useful when explaining framework compatibility."""
        return tuple(
            package
            for package in self.lock.packages
            if package.name in _FRAMEWORK_DIAGNOSTIC_PACKAGES
        )

    def enrich_failure(self, failure: ResolutionFailure) -> ResolutionFailure:
        """Attach dependency paths and the concrete platform when lock data permits."""
        package = failure.package
        if package is None:
            return replace(
                failure,
                platform=failure.platform or self.environment.platform_label,
            )
        paths = tuple(
            tuple(self._display_package(name) for name in path)
            for path in self.lock.dependency_paths(package.name)
        )
        required_by = failure.required_by
        if paths and not required_by:
            required_by = paths[0]
        return replace(
            failure,
            required_by=required_by,
            platform=failure.platform or self.environment.platform_label,
            dependency_paths=paths,
        )

    def dependency_paths(self, package: str) -> tuple[tuple[str, ...], ...]:
        """Return versioned dependency paths from the synthetic project root."""
        return tuple(
            tuple(self._display_package(name) for name in path)
            for path in self.lock.dependency_paths(package)
        )

    def _display_package(self, name: str) -> str:
        package = self.lock.package(name)
        if name == self.lock.project_name:
            return "project"
        return f"{name}=={package.version}" if package is not None else name
