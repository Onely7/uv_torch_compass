"""Resolve standalone projects and uv workspace members."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib  # ty: ignore[unresolved-import]
except ModuleNotFoundError:  # pragma: no cover - exercised by Python 3.10 CI.
    import tomli as tomllib  # ty: ignore[unresolved-import]

from uv_torch_compass.errors import CommandError, ConfigurationError
from uv_torch_compass.uv_commands import UvCommandClient


@dataclass(frozen=True, slots=True)
class WorkspaceContext:
    """Identify the target project and shared transaction root."""

    project_dir: Path
    workspace_root: Path
    package: str | None
    lockfile: Path
    is_workspace: bool
    members: tuple[tuple[str, Path], ...] = ()


def resolve_workspace(pyproject: Path, uv: UvCommandClient) -> WorkspaceContext:
    """Resolve workspace ownership using uv's own member discovery.

    Raises:
        CommandError: If workspace discovery is required but unsupported.
        ConfigurationError: If the target is not a member of the discovered workspace.
    """
    project_dir = pyproject.parent.resolve()
    package = _read_package_name(pyproject)
    result = uv.workspace_metadata(project_dir)
    if result.returncode != 0:
        if _workspace_ancestor(project_dir) is not None:
            raise CommandError(
                "this uv version cannot inspect the containing workspace; update uv"
            )
        return WorkspaceContext(
            project_dir=project_dir,
            workspace_root=project_dir,
            package=package,
            lockfile=project_dir / "uv.lock",
            is_workspace=False,
            members=(),
        )

    metadata = _parse_metadata(result.stdout)
    workspace_root = _metadata_path(metadata, "workspace_root")
    members: tuple[tuple[Path, str], ...] = ()
    is_workspace = workspace_root != project_dir or _declares_workspace(
        workspace_root / "pyproject.toml"
    )
    if is_workspace:
        members = _metadata_members(metadata)
        member_paths = {path for path, _name in members}
        if project_dir not in member_paths and project_dir != workspace_root:
            raise ConfigurationError(
                f"{project_dir} is not a member of workspace {workspace_root}"
            )
        member_name = next(
            (name for path, name in members if path == project_dir), None
        )
        if project_dir != workspace_root and package is None:
            raise ConfigurationError("a workspace member must define [project].name")
        if (
            project_dir != workspace_root
            and member_name is not None
            and package != member_name
        ):
            raise ConfigurationError(
                f"[project].name {package!r} does not match uv workspace member "
                f"{member_name!r}"
            )

    return WorkspaceContext(
        project_dir=project_dir,
        workspace_root=workspace_root,
        package=package if is_workspace and project_dir != workspace_root else None,
        lockfile=workspace_root / "uv.lock",
        is_workspace=is_workspace,
        members=tuple((name, path) for path, name in members),
    )


def _read_package_name(pyproject: Path) -> str | None:
    try:
        with pyproject.open("rb") as stream:
            document = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"failed to read {pyproject}: {exc}") from exc
    project = document.get("project")
    if not isinstance(project, dict):
        return None
    name = project.get("name")
    if name is None:
        return None
    if not isinstance(name, str) or not name.strip():
        raise ConfigurationError("[project].name must be a non-empty string")
    return name.strip()


def _workspace_ancestor(project_dir: Path) -> Path | None:
    for directory in (project_dir, *project_dir.parents):
        pyproject = directory / "pyproject.toml"
        if _declares_workspace(pyproject):
            return directory
    return None


def _declares_workspace(pyproject: Path) -> bool:
    if not pyproject.is_file():
        return False
    try:
        with pyproject.open("rb") as stream:
            document = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError):
        return False
    tool = document.get("tool")
    if not isinstance(tool, dict):
        return False
    uv = tool.get("uv")
    return isinstance(uv, dict) and isinstance(uv.get("workspace"), dict)


def _parse_metadata(output: str) -> Mapping[str, Any]:
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        raise CommandError("uv workspace metadata returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise CommandError("uv workspace metadata must return a JSON object")
    return value


def _metadata_path(metadata: Mapping[str, Any], key: str) -> Path:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CommandError(f"uv workspace metadata omitted {key}")
    return Path(value).expanduser().resolve()


def _metadata_members(metadata: Mapping[str, Any]) -> tuple[tuple[Path, str], ...]:
    values = metadata.get("members")
    if not isinstance(values, list):
        raise CommandError("uv workspace metadata omitted members")
    members: list[tuple[Path, str]] = []
    for value in values:
        if not isinstance(value, dict):
            raise CommandError("uv workspace metadata contains an invalid member")
        path = value.get("path")
        name = value.get("name")
        if not isinstance(path, str) or not path.strip():
            raise CommandError("uv workspace member omitted path")
        if not isinstance(name, str) or not name.strip():
            raise CommandError("uv workspace member omitted name")
        members.append((Path(path).expanduser().resolve(), name.strip()))
    return tuple(members)
