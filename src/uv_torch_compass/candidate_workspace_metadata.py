"""Parse bounded uv workspace metadata into a candidate dependency graph."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from packaging.utils import canonicalize_name

from uv_torch_compass.candidate_metadata import (
    CandidateDependencyGraph,
    CandidatePackage,
    ResolutionEvidenceSource,
    WheelArtifact,
)
from uv_torch_compass.errors import LockMetadataError
from uv_torch_compass.redaction import redact

_MAX_METADATA_CHARACTERS = 2 * 1024 * 1024
_MAX_PACKAGES = 10_000
_MAX_DEPENDENCIES_PER_PACKAGE = 2_048
_MAX_ARTIFACTS_PER_PACKAGE = 512


def read_candidate_workspace_metadata(
    raw_output: str,
    *,
    project_name: str = "uv-torch-compass-candidate",
) -> CandidateDependencyGraph:
    """Read the candidate graph exported by ``uv workspace metadata``.

    Args:
        raw_output: JSON emitted by uv for the already locked candidate project.
        project_name: Synthetic project package expected in the resolution.

    Returns:
        A validated graph whose URLs have credentials removed.

    Raises:
        LockMetadataError: If the JSON is oversized, malformed, or ambiguous.
    """
    if len(raw_output) > _MAX_METADATA_CHARACTERS:
        raise LockMetadataError("candidate workspace metadata exceeds the size limit")
    try:
        document = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise LockMetadataError(
            "candidate workspace metadata is not valid JSON"
        ) from exc
    if not isinstance(document, Mapping):
        raise LockMetadataError("candidate workspace metadata must be an object")
    _validate_schema(document.get("schema"))
    raw_resolution = document.get("resolution")
    if not isinstance(raw_resolution, Mapping):
        raise LockMetadataError("candidate workspace metadata omitted its resolution")
    if len(raw_resolution) > _MAX_PACKAGES:
        raise LockMetadataError("candidate workspace metadata has too many packages")

    package_rows = _package_rows(cast(Mapping[object, object], raw_resolution))
    packages = tuple(
        _parse_package(package_id, value, package_rows)
        for package_id, value in package_rows.items()
    )
    normalized_project = str(canonicalize_name(project_name))
    names = [package.name for package in packages]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise LockMetadataError(
            "candidate workspace metadata contains ambiguous package identities: "
            + ", ".join(duplicates)
        )
    if normalized_project not in names:
        raise LockMetadataError(
            "candidate workspace metadata omitted its project package"
        )
    return CandidateDependencyGraph(
        normalized_project,
        packages,
        ResolutionEvidenceSource.WORKSPACE_METADATA,
    )


def _validate_schema(value: object) -> None:
    if not isinstance(value, Mapping) or value.get("version") != "preview":
        raise LockMetadataError("candidate workspace metadata schema is unsupported")


def _package_rows(
    resolution: Mapping[object, object],
) -> dict[str, Mapping[str, object]]:
    rows: dict[str, Mapping[str, object]] = {}
    for raw_id, raw_value in resolution.items():
        if not isinstance(raw_id, str) or not raw_id:
            raise LockMetadataError("candidate workspace metadata has an invalid ID")
        if not isinstance(raw_value, Mapping):
            raise LockMetadataError(
                "candidate workspace metadata has a non-object package"
            )
        value = cast(Mapping[str, object], raw_value)
        if value.get("kind") != "package":
            continue
        rows[raw_id] = value
    return rows


def _parse_package(
    package_id: str,
    value: Mapping[str, object],
    package_rows: Mapping[str, Mapping[str, object]],
) -> CandidatePackage:
    name = value.get("name")
    version = value.get("version")
    if not isinstance(name, str) or not name:
        raise LockMetadataError("candidate workspace metadata has an invalid name")
    if not isinstance(version, str) or not version:
        raise LockMetadataError(
            f"candidate workspace metadata package {name!r} has no version"
        )
    source = value.get("source")
    if not isinstance(source, Mapping):
        raise LockMetadataError(
            f"candidate workspace metadata package {name!r} has an invalid source"
        )
    source_kind, source_url = _source_identity(cast(Mapping[str, object], source))
    dependencies = _dependencies(value.get("dependencies", []), name, package_rows)
    wheels = _wheels(value.get("wheels", []), name)
    return CandidatePackage(
        str(canonicalize_name(name)),
        version,
        redact(source_url),
        dependencies,
        source_kind,
        wheels,
        package_id,
    )


def _source_identity(source: Mapping[str, object]) -> tuple[str, str]:
    for kind in ("registry", "url", "git", "path", "editable", "virtual"):
        if kind not in source:
            continue
        value = source[kind]
        if isinstance(value, str):
            return kind, value
        if isinstance(value, Mapping):
            nested_url = value.get("url")
            if isinstance(nested_url, str):
                return kind, nested_url
        raise LockMetadataError("candidate workspace metadata has an invalid source")
    return "unknown", ""


def _dependencies(
    value: object,
    package: str,
    package_rows: Mapping[str, Mapping[str, object]],
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise LockMetadataError(
            f"candidate workspace metadata package {package!r} has invalid dependencies"
        )
    if len(value) > _MAX_DEPENDENCIES_PER_PACKAGE:
        raise LockMetadataError(
            f"candidate workspace metadata package {package!r} has too many dependencies"
        )
    dependencies: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise LockMetadataError(
                f"candidate workspace metadata package {package!r} has an invalid dependency"
            )
        dependency_id = item.get("id")
        if not isinstance(dependency_id, str):
            raise LockMetadataError(
                f"candidate workspace metadata package {package!r} has an invalid dependency"
            )
        dependency = package_rows.get(dependency_id)
        if dependency is None or not isinstance(dependency.get("name"), str):
            raise LockMetadataError(
                f"candidate workspace metadata package {package!r} references a missing package"
            )
        dependencies.append(str(canonicalize_name(cast(str, dependency["name"]))))
    return tuple(dependencies)


def _wheels(value: object, package: str) -> tuple[WheelArtifact, ...]:
    if not isinstance(value, list):
        raise LockMetadataError(
            f"candidate workspace metadata package {package!r} has invalid wheels"
        )
    if len(value) > _MAX_ARTIFACTS_PER_PACKAGE:
        raise LockMetadataError(
            f"candidate workspace metadata package {package!r} has too many wheels"
        )
    return tuple(_wheel(item, package) for item in value)


def _wheel(value: object, package: str) -> WheelArtifact:
    if not isinstance(value, Mapping):
        raise LockMetadataError(
            f"candidate workspace metadata package {package!r} has an invalid wheel"
        )
    url = value.get("url")
    if not isinstance(url, str):
        raise LockMetadataError(
            f"candidate workspace metadata package {package!r} has an invalid wheel"
        )
    hashes = value.get("hashes", {})
    if not isinstance(hashes, Mapping):
        raise LockMetadataError(
            f"candidate workspace metadata package {package!r} has invalid hashes"
        )
    sha256 = hashes.get("sha256", "")
    if not isinstance(sha256, str):
        raise LockMetadataError(
            f"candidate workspace metadata package {package!r} has an invalid hash"
        )
    size = value.get("size")
    if size is not None and (
        not isinstance(size, int) or isinstance(size, bool) or size <= 0
    ):
        raise LockMetadataError(
            f"candidate workspace metadata package {package!r} has an invalid wheel size"
        )
    return WheelArtifact(redact(url), sha256, size)
