"""Read dependency scopes and render safe PyTorch source updates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableSequence
from pathlib import Path
from typing import Any, cast

import tomlkit
from packaging.markers import InvalidMarker, Marker, default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from tomlkit.exceptions import ParseError
from tomlkit.items import AoT

try:
    import tomllib  # ty: ignore[unresolved-import]
except ModuleNotFoundError:  # pragma: no cover - exercised by Python 3.10 CI.
    import tomli as tomllib  # ty: ignore[unresolved-import]

from uv_torch_compass.domain import (
    PYTORCH_PACKAGES,
    BackendCandidate,
    Channel,
    ProjectRequirements,
    Scope,
    ScopedRequirement,
)
from uv_torch_compass.errors import ConfigurationError, ProjectUpdateError

_OFFICIAL_INDEX_PREFIX = "https://download.pytorch.org/whl/"
_LINUX_MARKER = "sys_platform == 'linux'"
_NON_LINUX_MARKER = "sys_platform != 'linux'"


def read_project_requirements(
    pyproject: Path,
    *,
    extras: tuple[str, ...],
    groups: tuple[str, ...],
    overrides: tuple[str, ...],
) -> ProjectRequirements:
    """Read selected scopes and build the effective probe requirements.

    Args:
        pyproject: Target project metadata.
        extras: Optional dependency names included in verification.
        groups: Dependency group names included in verification.
        overrides: Complete PyTorch requirements applied to base dependencies.

    Returns:
        Parsed requirements with scope ownership retained.

    Raises:
        ConfigurationError: If metadata, scopes, or requirements are invalid.
    """
    document = _read_toml(pyproject)
    project = document.get("project")
    if not isinstance(project, dict):
        raise ConfigurationError("[project] table was not found")
    requires_python = project.get("requires-python", ">=3.10")
    if not isinstance(requires_python, str):
        raise ConfigurationError("[project].requires-python must be a string")
    try:
        SpecifierSet(requires_python)
    except InvalidSpecifier as exc:
        raise ConfigurationError(
            f"invalid requires-python value: {requires_python!r}"
        ) from exc

    base = _string_array(project.get("dependencies", []), "[project].dependencies")
    optional = project.get("optional-dependencies", {})
    if not isinstance(optional, dict):
        raise ConfigurationError("[project.optional-dependencies] must be a table")
    dependency_groups = document.get("dependency-groups", {})
    if not isinstance(dependency_groups, dict):
        raise ConfigurationError("[dependency-groups] must be a table")

    missing_extras = [name for name in extras if name not in optional]
    missing_groups = [name for name in groups if name not in dependency_groups]
    if missing_extras:
        raise ConfigurationError("unknown extras: " + ", ".join(missing_extras))
    if missing_groups:
        raise ConfigurationError(
            "unknown dependency groups: " + ", ".join(missing_groups)
        )

    base_scope = Scope("base")
    scope_values: list[tuple[Scope, list[str]]] = [(base_scope, list(base))]
    scope_values.extend(
        (
            Scope("extra", name),
            _string_array(optional[name], f"[project.optional-dependencies].{name}"),
        )
        for name in extras
    )
    scope_values.extend(
        (
            Scope("group", name),
            _group_requirements(name, dependency_groups, ()),
        )
        for name in groups
    )

    override_by_package: dict[str, str] = {
        str(canonicalize_name(Requirement(raw).name)): raw for raw in overrides
    }
    scope_values[0] = (
        base_scope,
        _apply_requirement_overrides(scope_values[0][1], override_by_package),
    )

    selected: list[ScopedRequirement] = []
    for scope, values in scope_values:
        parsed = [_scoped_requirement(scope, raw) for raw in values]
        packages = {item.package for item in parsed}
        if (
            packages.intersection({"torchvision", "torchaudio"})
            and "torch" not in packages
        ):
            parsed.append(_scoped_requirement(scope, "torch"))
        selected.extend(parsed)

    if not any(item.package in PYTORCH_PACKAGES for item in selected):
        raise ConfigurationError(
            "no torch, torchvision, or torchaudio requirement was selected"
        )

    all_pytorch = _read_all_pytorch(project, optional, dependency_groups)
    _reject_conflicting_exact_versions((*selected, *all_pytorch))
    return ProjectRequirements(
        requires_python=requires_python,
        python_file_value=_read_python_version_file(pyproject.parent),
        selected=tuple(selected),
        all_pytorch=tuple(all_pytorch),
        selected_scopes=tuple(scope for scope, _ in scope_values),
    )


def render_project_configuration(
    pyproject: Path,
    *,
    requirements: ProjectRequirements,
    overrides: tuple[str, ...],
    backend: BackendCandidate,
    numpy_lt2_required: bool,
) -> tuple[str, tuple[str, ...]]:
    """Render a comment-preserving project update without writing it.

    Returns:
        Updated TOML and concise descriptions of semantic changes.

    Raises:
        ProjectUpdateError: If the TOML shape cannot be changed safely.
    """
    try:
        document = tomlkit.parse(pyproject.read_text(encoding="utf-8"))
        project = document.get("project")
        if project is None:
            raise ProjectUpdateError("[project] table was not found")
        changes: list[str] = []
        override_by_package: dict[str, str] = {
            str(canonicalize_name(Requirement(raw).name)): raw for raw in overrides
        }
        base = _toml_array(project, "dependencies")
        if override_by_package:
            _replace_array_values(
                base, _apply_requirement_overrides(base, override_by_package)
            )
            changes.append("updated base PyTorch requirements")

        for scope in requirements.selected_scopes:
            values = _scope_array(document, scope)
            packages = _dependency_names(values)
            effective_packages = {
                item.package for item in requirements.selected if item.scope == scope
            }
            if (
                effective_packages.intersection({"torchvision", "torchaudio"})
                and "torch" not in packages
            ):
                values.append("torch")
                changes.append(f"added torch to {scope.label}")
            if numpy_lt2_required and effective_packages.intersection(PYTORCH_PACKAGES):
                _ensure_numpy_lt2(values, scope)
                changes.append(f"added Linux NumPy constraint to {scope.label}")

        packages = {
            item.package
            for item in requirements.selected
            if item.package in PYTORCH_PACKAGES
        }
        tool = _ensure_table(document, "tool")
        uv = _ensure_table(tool, "uv")
        sources = _ensure_table(uv, "sources")
        for package in sorted(packages):
            sources[package] = _linux_source_value(sources.get(package), backend)
        if packages:
            changes.append("configured the verified Linux PyTorch index")

        indexes = uv.get("index")
        if indexes is None:
            indexes = tomlkit.aot()
            uv["index"] = indexes
        if not isinstance(indexes, AoT):
            raise ProjectUpdateError("tool.uv.index must be an array of tables")
        _ensure_verified_index(indexes, backend)
        _remove_unreferenced_official_indexes(indexes, sources, backend.index_name)
        return tomlkit.dumps(document), tuple(dict.fromkeys(changes))
    except ProjectUpdateError:
        raise
    except (OSError, ParseError, TypeError, ValueError) as exc:
        raise ProjectUpdateError(f"failed to update {pyproject}: {exc}") from exc


def read_configured_backend(
    pyproject: Path, packages: Iterable[str]
) -> BackendCandidate:
    """Read the one official Linux backend shared by selected packages.

    Raises:
        ConfigurationError: If sources are missing, inconsistent, or unofficial.
    """
    document = _read_toml(pyproject)
    tool = document.get("tool", {})
    uv = tool.get("uv", {}) if isinstance(tool, dict) else {}
    sources = uv.get("sources", {}) if isinstance(uv, dict) else {}
    indexes = uv.get("index", []) if isinstance(uv, dict) else []
    if not isinstance(sources, dict) or not isinstance(indexes, list):
        raise ConfigurationError("tool.uv sources or index configuration is invalid")
    index_urls = {
        item.get("name"): item.get("url")
        for item in indexes
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and isinstance(item.get("url"), str)
    }
    selected_names: set[str] = set()
    linux_environment = cast(dict[str, str], dict(default_environment()))
    linux_environment.update({"sys_platform": "linux", "platform_system": "Linux"})
    for package in packages:
        raw = sources.get(package)
        candidates = raw if isinstance(raw, list) else [raw]
        matching: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, dict) or not isinstance(
                candidate.get("index"), str
            ):
                continue
            marker = candidate.get("marker")
            try:
                applies = marker is None or Marker(str(marker)).evaluate(
                    linux_environment
                )
            except InvalidMarker as exc:
                raise ConfigurationError(
                    f"invalid source marker for {package}: {marker!r}"
                ) from exc
            if applies:
                matching.append(candidate["index"])
        if len(matching) != 1:
            raise ConfigurationError(
                f"{package} must have exactly one Linux source, found {len(matching)}"
            )
        selected_names.add(matching[0])
    if len(selected_names) != 1:
        raise ConfigurationError(
            "selected PyTorch packages use different Linux indexes"
        )
    index_name = next(iter(selected_names))
    index_url = index_urls.get(index_name)
    if not isinstance(index_url, str) or not index_url.startswith(
        _OFFICIAL_INDEX_PREFIX
    ):
        raise ConfigurationError(
            f"index {index_name!r} is not an official PyTorch index"
        )
    path = index_url.removeprefix(_OFFICIAL_INDEX_PREFIX).strip("/")
    channel = Channel.NIGHTLY if path.startswith("nightly/") else Channel.STABLE
    backend = path.removeprefix("nightly/")
    candidate = BackendCandidate(backend, channel)
    if candidate.index_name != index_name or candidate.index_url != index_url:
        raise ConfigurationError(
            f"index {index_name!r} does not match its official URL {index_url!r}"
        )
    return candidate


def _read_toml(pyproject: Path) -> dict[str, Any]:
    try:
        with pyproject.open("rb") as stream:
            return tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"failed to read {pyproject}: {exc}") from exc


def _string_array(raw: object, label: str) -> list[str]:
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ConfigurationError(f"{label} must be an array of strings")
    return cast(list[str], list(raw))


def _group_requirements(
    name: str, groups: Mapping[str, object], parents: tuple[str, ...]
) -> list[str]:
    if name in parents:
        chain = " -> ".join((*parents, name))
        raise ConfigurationError(f"dependency group include cycle: {chain}")
    raw = groups.get(name)
    if not isinstance(raw, list):
        raise ConfigurationError(f"[dependency-groups].{name} must be an array")
    values: list[str] = []
    for item in raw:
        if isinstance(item, str):
            values.append(item)
            continue
        if isinstance(item, dict) and set(item) == {"include-group"}:
            typed_item = cast(dict[str, object], item)
            included = typed_item["include-group"]
            if not isinstance(included, str) or included not in groups:
                raise ConfigurationError(
                    f"dependency group {name!r} includes an unknown group"
                )
            values.extend(_group_requirements(included, groups, (*parents, name)))
            continue
        raise ConfigurationError(
            f"[dependency-groups].{name} contains an unsupported entry"
        )
    return values


def _scoped_requirement(scope: Scope, raw: str) -> ScopedRequirement:
    try:
        return ScopedRequirement(scope, raw)
    except ConfigurationError:
        raise


def _apply_requirement_overrides(
    values: Iterable[object], overrides: Mapping[str, str]
) -> list[str]:
    remaining = dict(overrides)
    updated: list[str] = []
    for raw in values:
        text = str(raw)
        try:
            package = canonicalize_name(Requirement(text).name)
        except InvalidRequirement:
            updated.append(text)
            continue
        updated.append(remaining.pop(package, text))
    updated.extend(remaining.values())
    return updated


def _read_all_pytorch(
    project: Mapping[str, object],
    optional: Mapping[str, object],
    groups: Mapping[str, object],
) -> list[ScopedRequirement]:
    values: list[ScopedRequirement] = []
    scope_arrays: list[tuple[Scope, list[str]]] = [
        (
            Scope("base"),
            _string_array(project.get("dependencies", []), "[project].dependencies"),
        )
    ]
    scope_arrays.extend(
        (
            Scope("extra", name),
            _string_array(raw, f"[project.optional-dependencies].{name}"),
        )
        for name, raw in optional.items()
    )
    scope_arrays.extend(
        (Scope("group", name), _group_requirements(name, groups, ())) for name in groups
    )
    for scope, dependencies in scope_arrays:
        for raw in dependencies:
            try:
                parsed = Requirement(raw)
            except InvalidRequirement:
                continue
            if canonicalize_name(parsed.name) in PYTORCH_PACKAGES:
                values.append(ScopedRequirement(scope, raw))
    return values


def _reject_conflicting_exact_versions(
    requirements: Iterable[ScopedRequirement],
) -> None:
    exact: dict[str, list[tuple[ScopedRequirement, set[str]]]] = {}
    for item in requirements:
        if item.package not in PYTORCH_PACKAGES:
            continue
        versions = {
            specifier.version
            for specifier in item.requirement.specifier
            if specifier.operator in {"==", "==="} and "*" not in specifier.version
        }
        if len(versions) > 1:
            raise ConfigurationError(
                f"conflicting exact versions in {item.scope.label}: {item.raw!r}"
            )
        if versions:
            exact.setdefault(item.package, []).append((item, versions))
    conflicts: list[str] = []
    for package, entries in exact.items():
        for position, (left, left_versions) in enumerate(entries):
            for right, right_versions in entries[position + 1 :]:
                if left_versions != right_versions and _markers_overlap(left, right):
                    conflicts.append(package)
                    break
            if package in conflicts:
                break
    if conflicts:
        raise ConfigurationError(
            "conflicting exact versions across dependency scopes: "
            + ", ".join(sorted(conflicts))
        )


def _markers_overlap(left: ScopedRequirement, right: ScopedRequirement) -> bool:
    for minor in range(10, 15):
        for implementation_name, platform_implementation in (
            ("cpython", "CPython"),
            ("pypy", "PyPy"),
        ):
            environment = cast(dict[str, str], dict(default_environment()))
            environment.update(
                {
                    "python_version": f"3.{minor}",
                    "python_full_version": f"3.{minor}.0",
                    "implementation_name": implementation_name,
                    "platform_python_implementation": platform_implementation,
                    "sys_platform": "linux",
                    "platform_system": "Linux",
                }
            )
            if _marker_applies(left, environment) and _marker_applies(
                right, environment
            ):
                return True
    return False


def _marker_applies(
    requirement: ScopedRequirement, environment: Mapping[str, str]
) -> bool:
    marker = requirement.requirement.marker
    return marker is None or marker.evaluate(environment)


def _read_python_version_file(project_dir: Path) -> str:
    path = project_dir / ".python-version"
    if not path.is_file():
        return ""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                return value
    except OSError as exc:
        raise ConfigurationError(f"failed to read {path}: {exc}") from exc
    return ""


def _toml_array(table: Any, key: str) -> MutableSequence[object]:
    value = table.get(key)
    if value is None:
        value = tomlkit.array().multiline(True)
        table[key] = value
    if not isinstance(value, MutableSequence):
        raise ProjectUpdateError(f"{key} must be an array")
    return cast(MutableSequence[object], value)


def _scope_array(document: Any, scope: Scope) -> MutableSequence[object]:
    project = document["project"]
    if scope.kind == "base":
        return _toml_array(project, "dependencies")
    if scope.kind == "extra":
        optional = project.get("optional-dependencies")
        if optional is None or scope.name not in optional:
            raise ProjectUpdateError(f"optional dependency {scope.name!r} disappeared")
        return _toml_array(optional, scope.name)
    groups = document.get("dependency-groups")
    if groups is None or scope.name not in groups:
        raise ProjectUpdateError(f"dependency group {scope.name!r} disappeared")
    return _toml_array(groups, scope.name)


def _replace_array_values(
    target: MutableSequence[object], values: Iterable[str]
) -> None:
    del target[:]
    target.extend(values)


def _dependency_names(values: Iterable[object]) -> set[str]:
    names: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        try:
            names.add(canonicalize_name(Requirement(value).name))
        except InvalidRequirement:
            continue
    return names


def _ensure_numpy_lt2(values: MutableSequence[object], scope: Scope) -> None:
    linux_environment = cast(dict[str, str], dict(default_environment()))
    linux_environment.update({"sys_platform": "linux", "platform_system": "Linux"})
    for value in values:
        if not isinstance(value, str):
            continue
        try:
            requirement = Requirement(value)
        except InvalidRequirement:
            continue
        if canonicalize_name(requirement.name) != "numpy":
            continue
        applies = requirement.marker is None or requirement.marker.evaluate(
            linux_environment
        )
        if not applies:
            continue
        if any(
            spec.operator in {">", ">=", "~=", "==", "==="}
            and spec.version.lstrip("=").startswith("2")
            for spec in requirement.specifier
        ):
            raise ProjectUpdateError(
                f"numpy requirement in {scope.label} conflicts with numpy<2"
            )
        if any(
            spec.operator == "<" and spec.version == "2"
            for spec in requirement.specifier
        ):
            return
    values.append("numpy<2; sys_platform == 'linux'")


def _ensure_table(parent: Any, key: str) -> Any:
    value = parent.get(key)
    if value is None:
        value = tomlkit.table()
        parent[key] = value
    if not isinstance(value, Mapping):
        raise ProjectUpdateError(f"{key} must be a table")
    return value


def _linux_source_value(existing: object, backend: BackendCandidate) -> object:
    candidates = list(existing) if isinstance(existing, list) else [existing]
    preserved: list[object] = []
    for candidate in candidates:
        if candidate is None:
            continue
        if not isinstance(candidate, Mapping):
            raise ProjectUpdateError("tool.uv.sources entries must be tables")
        marker_value = candidate.get("marker")
        if marker_value is not None and not isinstance(marker_value, str):
            raise ProjectUpdateError("source marker must be a string")
        marker = marker_value or ""
        if marker and _is_exact_linux_marker(marker):
            continue
        guarded = _guard_non_linux(marker)
        copied = tomlkit.inline_table()
        for key, value in candidate.items():
            if not isinstance(key, str):
                raise ProjectUpdateError("source table keys must be strings")
            copied[key] = value
        copied["marker"] = guarded
        preserved.append(copied)

    source = tomlkit.inline_table()
    source["index"] = backend.index_name
    source["marker"] = _LINUX_MARKER
    preserved.append(source)
    result = tomlkit.array().multiline(True)
    for item in preserved:
        result.append(item)
    return result


def _is_exact_linux_marker(marker: str) -> bool:
    try:
        return str(Marker(marker)) == str(Marker(_LINUX_MARKER))
    except InvalidMarker as exc:
        raise ProjectUpdateError(f"invalid source marker {marker!r}") from exc


def _guard_non_linux(marker: str) -> str:
    if not marker:
        return _NON_LINUX_MARKER
    try:
        normalized = str(Marker(marker))
    except InvalidMarker as exc:
        raise ProjectUpdateError(f"invalid source marker {marker!r}") from exc
    if _NON_LINUX_MARKER in normalized.replace('"', "'"):
        return marker
    return f"({marker}) and {_NON_LINUX_MARKER}"


def _ensure_verified_index(indexes: AoT, backend: BackendCandidate) -> None:
    for entry in indexes:
        if entry.get("name") != backend.index_name:
            continue
        existing_url = entry.get("url")
        if existing_url != backend.index_url:
            raise ProjectUpdateError(
                f"index {backend.index_name!r} already uses a non-matching URL"
            )
        entry["explicit"] = True
        return
    entry = tomlkit.table()
    entry["name"] = backend.index_name
    entry["url"] = backend.index_url
    entry["explicit"] = True
    indexes.append(entry)


def _remove_unreferenced_official_indexes(
    indexes: AoT, sources: Mapping[str, object], selected_name: str
) -> None:
    referenced = _referenced_indexes(sources)
    for position in reversed(range(len(indexes))):
        entry = indexes[position]
        name = entry.get("name")
        url = entry.get("url")
        if (
            isinstance(name, str)
            and isinstance(url, str)
            and url.startswith(_OFFICIAL_INDEX_PREFIX)
            and name != selected_name
            and name not in referenced
        ):
            indexes.pop(position)


def _referenced_indexes(sources: Mapping[str, object]) -> set[str]:
    names: set[str] = set()
    for value in sources.values():
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                index = candidate.get("index")
                if isinstance(index, str):
                    names.add(index)
    return names
