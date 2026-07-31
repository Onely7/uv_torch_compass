"""Model one candidate dependency graph independently of its uv transport."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from packaging.utils import canonicalize_name

from uv_torch_compass.domain import PYTORCH_PACKAGES


class ResolutionEvidenceSource(str, Enum):
    """Identify the uv boundary that supplied a candidate dependency graph."""

    LOCKFILE = "lockfile"
    WORKSPACE_METADATA = "workspace-metadata"


@dataclass(frozen=True, slots=True)
class WheelArtifact:
    """Identify one immutable wheel artifact reported by uv."""

    url: str
    hash: str
    size: int


@dataclass(frozen=True, slots=True)
class CandidatePackage:
    """Describe one resolved distribution and its direct dependency names."""

    name: str
    version: str
    source_url: str
    dependencies: tuple[str, ...]
    source_kind: str = "registry"
    wheels: tuple[WheelArtifact, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidateDependencyGraph:
    """Expose validated package identities and paths for one candidate."""

    project_name: str
    packages: tuple[CandidatePackage, ...]
    evidence_source: ResolutionEvidenceSource = ResolutionEvidenceSource.LOCKFILE

    def package(self, name: str) -> CandidatePackage | None:
        """Return the uniquely resolved package with the requested name."""
        normalized = str(canonicalize_name(name))
        return next(
            (package for package in self.packages if package.name == normalized),
            None,
        )

    @property
    def pytorch_packages(self) -> tuple[CandidatePackage, ...]:
        """Return resolved PyTorch ecosystem distributions."""
        return tuple(
            package for package in self.packages if package.name in PYTORCH_PACKAGES
        )

    def dependency_paths(self, target: str) -> tuple[tuple[str, ...], ...]:
        """Return all simple paths from the candidate project to one package."""
        normalized_target = str(canonicalize_name(target))
        graph = {package.name: package.dependencies for package in self.packages}
        paths: list[tuple[str, ...]] = []
        pending: deque[tuple[str, tuple[str, ...]]] = deque(
            [(self.project_name, (self.project_name,))]
        )
        while pending:
            package, path = pending.popleft()
            for dependency in graph.get(package, ()):
                if dependency in path:
                    continue
                child_path = (*path, dependency)
                if dependency == normalized_target:
                    paths.append(child_path)
                    continue
                pending.append((dependency, child_path))
        return tuple(paths)


# These aliases preserve the internal contract while callers migrate to names
# that describe the domain rather than the original TOML representation.
LockedArtifact = WheelArtifact
LockedPackage = CandidatePackage
CandidateLockSnapshot = CandidateDependencyGraph
