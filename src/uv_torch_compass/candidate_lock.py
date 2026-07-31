"""Parse bounded uv candidate lockfiles into an immutable dependency graph."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from packaging.utils import canonicalize_name

try:
    import tomllib  # ty: ignore[unresolved-import]
except ModuleNotFoundError:  # pragma: no cover - exercised by Python 3.10 CI.
    import tomli as tomllib  # ty: ignore[unresolved-import]

from uv_torch_compass.domain import PYTORCH_PACKAGES
from uv_torch_compass.errors import ProbeError
from uv_torch_compass.redaction import redact

_MAX_LOCK_BYTES = 32 * 1024 * 1024
_MAX_ARTIFACTS_PER_PACKAGE = 512


@dataclass(frozen=True, slots=True)
class LockedArtifact:
    """Identify one immutable wheel artifact recorded by uv."""

    url: str
    hash: str
    size: int


@dataclass(frozen=True, slots=True)
class LockedPackage:
    """Describe one resolved distribution and its direct dependency names."""

    name: str
    version: str
    source_url: str
    dependencies: tuple[str, ...]
    source_kind: str = "registry"
    wheels: tuple[LockedArtifact, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidateLockSnapshot:
    """Expose validated package identities and paths from one candidate lock."""

    project_name: str
    packages: tuple[LockedPackage, ...]

    def package(self, name: str) -> LockedPackage | None:
        """Return the uniquely resolved package with the requested name."""
        normalized = str(canonicalize_name(name))
        return next(
            (package for package in self.packages if package.name == normalized),
            None,
        )

    @property
    def pytorch_packages(self) -> tuple[LockedPackage, ...]:
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


def read_candidate_lock(
    path: Path,
    *,
    project_name: str = "uv-torch-compass-candidate",
) -> CandidateLockSnapshot:
    """Read a uv lockfile without trusting unbounded or ambiguous package data.

    Args:
        path: Lockfile generated inside the disposable candidate project.
        project_name: Synthetic root package used to build dependency paths.

    Returns:
        A validated dependency graph with credential-redacted source URLs.

    Raises:
        ProbeError: If the lockfile is unsafe, malformed, or ambiguous.
    """
    normalized_project = str(canonicalize_name(project_name))
    try:
        if path.is_symlink() or not path.is_file():
            raise ProbeError("candidate lockfile is not a regular file")
        if path.stat().st_size > _MAX_LOCK_BYTES:
            raise ProbeError("candidate lockfile exceeds the 32 MiB size limit")
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ProbeError(f"failed to read candidate lockfile: {exc}") from exc

    raw_packages = document.get("package")
    if not isinstance(raw_packages, list):
        raise ProbeError("candidate lockfile has no package array")
    packages = tuple(_parse_package(item) for item in raw_packages)
    names = [package.name for package in packages]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ProbeError(
            "candidate lockfile contains ambiguous package identities: "
            + ", ".join(duplicates)
        )
    if normalized_project not in names:
        raise ProbeError("candidate lockfile does not contain its project package")
    known = set(names)
    missing = sorted(
        {
            dependency
            for package in packages
            for dependency in package.dependencies
            if dependency not in known
        }
    )
    if missing:
        raise ProbeError(
            "candidate lockfile references missing packages: " + ", ".join(missing)
        )
    return CandidateLockSnapshot(normalized_project, packages)


def _parse_package(value: object) -> LockedPackage:
    if not isinstance(value, Mapping):
        raise ProbeError("candidate lockfile contains a non-table package")
    name = value.get("name")
    version = value.get("version")
    if not isinstance(name, str) or not name:
        raise ProbeError("candidate lockfile package has an invalid name")
    if not isinstance(version, str) or not version:
        raise ProbeError(f"candidate lockfile package {name!r} has no version")
    dependencies = value.get("dependencies", [])
    if not isinstance(dependencies, list):
        raise ProbeError(
            f"candidate lockfile package {name!r} has invalid dependencies"
        )
    source = value.get("source", {})
    if not isinstance(source, Mapping):
        raise ProbeError(f"candidate lockfile package {name!r} has an invalid source")
    source_kind, source_url = _source_identity(cast(Mapping[str, object], source))
    if not isinstance(source_url, str):
        raise ProbeError(
            f"candidate lockfile package {name!r} has an invalid source identity"
        )
    raw_wheels = value.get("wheels", [])
    if not isinstance(raw_wheels, list):
        raise ProbeError(f"candidate lockfile package {name!r} has invalid wheels")
    if len(raw_wheels) > _MAX_ARTIFACTS_PER_PACKAGE:
        raise ProbeError(
            f"candidate lockfile package {name!r} has too many wheel artifacts"
        )
    return LockedPackage(
        str(canonicalize_name(name)),
        version,
        redact(source_url),
        tuple(_dependency_name(item, package=name) for item in dependencies),
        source_kind,
        tuple(_parse_artifact(item, package=name) for item in raw_wheels),
    )


def _source_identity(source: Mapping[str, object]) -> tuple[str, object]:
    for kind in ("registry", "url", "git", "path", "editable", "virtual"):
        if kind in source:
            return kind, source[kind]
    return "unknown", ""


def _parse_artifact(value: object, *, package: str) -> LockedArtifact:
    if not isinstance(value, Mapping):
        raise ProbeError(
            f"candidate lockfile package {package!r} has a non-table wheel artifact"
        )
    url = value.get("url")
    hash_value = value.get("hash")
    size = value.get("size")
    if not isinstance(url, str) or not url:
        raise ProbeError(
            f"candidate lockfile package {package!r} has an invalid wheel URL"
        )
    if not isinstance(hash_value, str) or not hash_value.startswith("sha256:"):
        raise ProbeError(
            f"candidate lockfile package {package!r} has an invalid wheel hash"
        )
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ProbeError(
            f"candidate lockfile package {package!r} has an invalid wheel size"
        )
    return LockedArtifact(redact(url), hash_value, size)


def _dependency_name(value: object, *, package: str) -> str:
    if not isinstance(value, Mapping):
        raise ProbeError(
            f"candidate lockfile package {package!r} has a non-table dependency"
        )
    name = cast(Mapping[str, Any], value).get("name")
    if not isinstance(name, str) or not name:
        raise ProbeError(
            f"candidate lockfile package {package!r} has an invalid dependency"
        )
    return str(canonicalize_name(name))
