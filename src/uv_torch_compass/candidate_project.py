"""Render isolated candidate projects with target source semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import tomlkit
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from tomlkit.exceptions import ParseError
from tomlkit.items import AoT

from uv_torch_compass.domain import PYTORCH_PACKAGES, BackendCandidate, Channel
from uv_torch_compass.errors import ConfigurationError

_COPIED_UV_KEYS = {
    "allow-insecure-host",
    "config-settings",
    "config-settings-package",
    "constraint-dependencies",
    "dependency-metadata",
    "exclude-newer",
    "exclude-newer-package",
    "fork-strategy",
    "index-strategy",
    "keyring-provider",
    "no-binary",
    "no-binary-package",
    "no-build",
    "no-build-package",
    "override-dependencies",
    "prerelease",
    "resolution",
}


def render_candidate_project(
    target: Path,
    *,
    destination: Path,
    requirements: Sequence[str],
    candidate: BackendCandidate,
    workspace_members: Mapping[str, Path],
) -> Path:
    """Write a temporary project that preserves relevant target source policy.

    Relative path and workspace sources are converted to absolute path sources
    because the isolated project lives outside the target workspace. The
    returned project is disposable and never shares a lockfile or environment
    with the target.

    Args:
        target: Project metadata whose resolution policy is authoritative.
        destination: Empty temporary directory for candidate state.
        requirements: Selected dependency roots for the candidate.
        candidate: Official PyTorch index being verified.
        workspace_members: Normalized workspace package names and paths.

    Returns:
        Path to the generated ``pyproject.toml``.

    Raises:
        ConfigurationError: If source metadata cannot be copied safely.
    """
    try:
        source_document = tomlkit.parse(target.read_text(encoding="utf-8"))
    except (OSError, ParseError) as exc:
        raise ConfigurationError(
            f"failed to read candidate source policy: {exc}"
        ) from exc

    destination.mkdir(parents=True, exist_ok=False)
    document = tomlkit.document()
    project = tomlkit.table()
    project["name"] = "uv-torch-compass-candidate"
    project["version"] = "0"
    project["requires-python"] = _requires_python(source_document)
    project["dependencies"] = _candidate_requirements(requirements)
    document["project"] = project

    source_uv = _source_uv_table(source_document)
    tool = tomlkit.table()
    uv = tomlkit.table()
    for key in sorted(_COPIED_UV_KEYS):
        if key in source_uv:
            uv[key] = deepcopy(source_uv[key])
    if candidate.channel is Channel.NIGHTLY:
        uv["prerelease"] = "allow"

    indexes = _copy_indexes(source_uv)
    _add_candidate_index(indexes, candidate)
    uv["index"] = indexes

    source_entries = _copy_selected_sources(
        source_uv,
        requirements,
        target.parent,
        workspace_members,
    )
    for package in _candidate_pytorch_packages(requirements):
        value = tomlkit.inline_table()
        value["index"] = candidate.index_name
        source_entries[package] = value
    uv["sources"] = source_entries
    tool["uv"] = uv
    document["tool"] = tool

    path = destination / "pyproject.toml"
    try:
        path.write_text(tomlkit.dumps(document), encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"failed to write candidate project: {exc}") from exc
    return path


def _requires_python(document: Mapping[str, object]) -> str:
    project = document.get("project")
    if not isinstance(project, Mapping):
        return ">=3.10"
    value = project.get("requires-python", ">=3.10")
    if not isinstance(value, str):
        raise ConfigurationError("[project].requires-python must be a string")
    return value


def _candidate_requirements(requirements: Sequence[str]) -> list[str]:
    values = list(dict.fromkeys(requirements))
    names = {_requirement_name(raw) for raw in values}
    # A direct anchor makes uv.sources authoritative even when a framework is
    # the dependency that constrains torch.
    if "torch" not in names:
        values.append("torch")
    return values


def _source_uv_table(document: Mapping[str, object]) -> Mapping[str, object]:
    tool = document.get("tool", {})
    if not isinstance(tool, Mapping):
        raise ConfigurationError("[tool] must be a table")
    uv = tool.get("uv", {})
    if not isinstance(uv, Mapping):
        raise ConfigurationError("[tool.uv] must be a table")
    return cast(Mapping[str, object], uv)


def _copy_indexes(uv: Mapping[str, object]) -> AoT:
    raw = uv.get("index")
    if raw is None:
        return tomlkit.aot()
    if not isinstance(raw, list):
        raise ConfigurationError("[tool.uv].index must be an array of tables")
    indexes = tomlkit.aot()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ConfigurationError("[tool.uv].index contains a non-table entry")
        table = tomlkit.table()
        for key, value in item.items():
            table[str(key)] = deepcopy(value)
        indexes.append(table)
    return indexes


def _add_candidate_index(indexes: AoT, candidate: BackendCandidate) -> None:
    for index in indexes:
        if index.get("name") != candidate.index_name:
            continue
        if index.get("url") != candidate.index_url:
            raise ConfigurationError(
                f"index {candidate.index_name!r} already uses a non-official URL"
            )
        index["explicit"] = True
        return
    table = tomlkit.table()
    table["name"] = candidate.index_name
    table["url"] = candidate.index_url
    table["explicit"] = True
    indexes.append(table)


def _copy_selected_sources(
    uv: Mapping[str, object],
    requirements: Sequence[str],
    project_dir: Path,
    workspace_members: Mapping[str, Path],
) -> Any:
    raw = uv.get("sources", {})
    if not isinstance(raw, Mapping):
        raise ConfigurationError("[tool.uv].sources must be a table")
    selected = {_requirement_name(requirement) for requirement in requirements}
    sources = tomlkit.table()
    for raw_name, value in raw.items():
        package = str(canonicalize_name(str(raw_name)))
        if package not in selected or package in PYTORCH_PACKAGES:
            continue
        sources[str(raw_name)] = _portable_source(
            value,
            package=package,
            project_dir=project_dir,
            workspace_members=workspace_members,
        )
    return sources


def _portable_source(
    value: object,
    *,
    package: str,
    project_dir: Path,
    workspace_members: Mapping[str, Path],
) -> object:
    if isinstance(value, list):
        return [
            _portable_source(
                item,
                package=package,
                project_dir=project_dir,
                workspace_members=workspace_members,
            )
            for item in value
        ]
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"source for {package} must be a table or array")
    result = tomlkit.inline_table()
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if key == "workspace" and raw_value is True:
            member = workspace_members.get(package)
            if member is None:
                raise ConfigurationError(
                    f"workspace source {package!r} was not found in uv metadata"
                )
            result["path"] = str(member)
            continue
        if key == "path":
            if not isinstance(raw_value, str) or not raw_value:
                raise ConfigurationError(f"path source for {package} is invalid")
            source_path = Path(raw_value).expanduser()
            if not source_path.is_absolute():
                source_path = project_dir / source_path
            result["path"] = str(source_path.resolve())
            continue
        result[key] = deepcopy(raw_value)
    return result


def _candidate_pytorch_packages(requirements: Sequence[str]) -> tuple[str, ...]:
    packages = {
        _requirement_name(requirement)
        for requirement in requirements
        if _requirement_name(requirement) in PYTORCH_PACKAGES
    }
    packages.add("torch")
    return tuple(sorted(packages))


def _requirement_name(raw: str) -> str:
    try:
        return str(canonicalize_name(Requirement(raw).name))
    except InvalidRequirement as exc:
        raise ConfigurationError(f"invalid candidate requirement {raw!r}") from exc
