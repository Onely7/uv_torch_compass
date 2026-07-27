"""Resolve command-line, environment, and project configuration layers."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar, cast

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

try:
    import tomllib  # ty: ignore[unresolved-import]
except ModuleNotFoundError:  # pragma: no cover - exercised by Python 3.10 CI.
    import tomli as tomllib  # ty: ignore[unresolved-import]

from uv_torch_compass.cuda_compatibility import CompatibilityPolicy
from uv_torch_compass.domain import (
    BackendRequest,
    Channel,
    GpuDevice,
    Operation,
    OutputFormat,
    ProbeProfile,
    RunOptions,
)
from uv_torch_compass.errors import ConfigurationError

_TOOL_KEYS = {
    "python",
    "backend",
    "channel",
    "extras",
    "groups",
    "cuda-device",
    "cuda-compatibility",
    "probe-profile",
    "link-mode",
    "log-dir",
    "timeout",
    "output-format",
    "state",
}
_LINK_MODES = {"clone", "copy", "hardlink", "symlink"}
_ENV_PREFIX = "UV_TORCH_COMPASS_"
_EnumSetting = TypeVar(
    "_EnumSetting", Channel, OutputFormat, ProbeProfile, CompatibilityPolicy
)


def resolve_options(
    namespace: Namespace,
    *,
    environ: Mapping[str, str],
    cwd: Path,
) -> RunOptions:
    """Resolve one immutable command configuration.

    CLI values override namespaced environment values, project settings, and
    defaults in that order.

    Args:
        namespace: Values parsed by the public argument parser.
        environ: Calling-process environment.
        cwd: Base directory for relative target paths.

    Returns:
        Validated options ready for infrastructure construction.

    Raises:
        ConfigurationError: If any configuration layer is invalid.
    """
    operation = Operation(namespace.operation)
    pyproject = _resolve_path(
        _first(
            getattr(namespace, "pyproject", None),
            environ.get(f"{_ENV_PREFIX}PYPROJECT"),
            "pyproject.toml",
        ),
        cwd,
    )
    if pyproject.name != "pyproject.toml":
        raise ConfigurationError("--pyproject must point to a pyproject.toml file")
    project_settings = _read_project_settings(pyproject)
    project_dir = pyproject.parent

    python_request = _optional_string_setting(
        "python",
        getattr(namespace, "python_request", None),
        environ.get(f"{_ENV_PREFIX}PYTHON"),
        project_settings,
    )
    backend = BackendRequest.parse(
        _string_setting(
            "backend",
            getattr(namespace, "backend", None),
            environ.get(f"{_ENV_PREFIX}BACKEND"),
            project_settings,
            default="auto",
        )
    )
    channel = _enum_setting(
        Channel,
        "channel",
        getattr(namespace, "channel", None),
        environ.get(f"{_ENV_PREFIX}CHANNEL"),
        project_settings,
        default="stable",
    )
    cuda_compatibility = _enum_setting(
        CompatibilityPolicy,
        "cuda-compatibility",
        getattr(namespace, "cuda_compatibility", None),
        environ.get(f"{_ENV_PREFIX}CUDA_COMPATIBILITY"),
        project_settings,
        default="strict",
    )
    probe_profile = _enum_setting(
        ProbeProfile,
        "probe-profile",
        getattr(namespace, "probe_profile", None),
        environ.get(f"{_ENV_PREFIX}PROBE_PROFILE"),
        project_settings,
        default="standard",
    )
    output_format = _enum_setting(
        OutputFormat,
        "output-format",
        getattr(namespace, "output_format", None),
        environ.get(f"{_ENV_PREFIX}OUTPUT_FORMAT"),
        project_settings,
        default="text",
    )
    extras = _list_setting(
        "extras",
        getattr(namespace, "extras", None),
        environ.get(f"{_ENV_PREFIX}EXTRAS"),
        project_settings,
    )
    groups = _list_setting(
        "groups",
        getattr(namespace, "groups", None),
        environ.get(f"{_ENV_PREFIX}GROUPS"),
        project_settings,
    )
    cuda_device_text = _optional_string_setting(
        "cuda-device",
        getattr(namespace, "cuda_device", None),
        environ.get(f"{_ENV_PREFIX}CUDA_DEVICE"),
        project_settings,
    )
    link_mode = _string_setting(
        "link-mode",
        getattr(namespace, "link_mode", None),
        environ.get(f"{_ENV_PREFIX}LINK_MODE"),
        project_settings,
        default="copy",
    )
    if link_mode not in _LINK_MODES:
        raise ConfigurationError(
            f"invalid link mode {link_mode!r}; expected one of "
            + ", ".join(sorted(_LINK_MODES))
        )

    log_dir_text = _string_setting(
        "log-dir",
        getattr(namespace, "log_dir", None),
        environ.get(f"{_ENV_PREFIX}LOG_DIR"),
        project_settings,
        default=".uv-torch-compass/logs",
    )
    timeout = _integer_setting(
        "timeout",
        getattr(namespace, "timeout", None),
        environ.get(f"{_ENV_PREFIX}TIMEOUT"),
        project_settings,
        default=1800,
    )
    report_file_text = _optional_external_string(
        "report-file",
        getattr(namespace, "report_file", None),
        environ.get(f"{_ENV_PREFIX}REPORT_FILE"),
    )

    overrides = tuple(
        normalized
        for package, raw in (
            (
                "torch",
                _optional_external_string(
                    "torch",
                    getattr(namespace, "torch_requirement", None),
                    environ.get(f"{_ENV_PREFIX}TORCH"),
                ),
            ),
            (
                "torchvision",
                _optional_external_string(
                    "torchvision",
                    getattr(namespace, "torchvision_requirement", None),
                    environ.get(f"{_ENV_PREFIX}TORCHVISION"),
                ),
            ),
            (
                "torchaudio",
                _optional_external_string(
                    "torchaudio",
                    getattr(namespace, "torchaudio_requirement", None),
                    environ.get(f"{_ENV_PREFIX}TORCHAUDIO"),
                ),
            ),
        )
        if (normalized := _normalize_requirement_override(package, raw))
    )

    return RunOptions(
        operation=operation,
        pyproject=pyproject,
        python_request=python_request,
        requirement_overrides=overrides,
        backend=backend,
        channel=channel,
        cuda_compatibility=cuda_compatibility,
        probe_profile=probe_profile,
        extras=extras,
        groups=groups,
        cuda_device=GpuDevice(cuda_device_text) if cuda_device_text else None,
        link_mode=link_mode,
        log_dir=_resolve_path(log_dir_text, project_dir),
        timeout_seconds=timeout,
        output_format=output_format,
        report_file=(
            _resolve_path(report_file_text, project_dir) if report_file_text else None
        ),
    )


def _read_project_settings(pyproject: Path) -> dict[str, Any]:
    if not pyproject.is_file():
        raise ConfigurationError(f"{pyproject} was not found")
    try:
        with pyproject.open("rb") as stream:
            document = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"failed to read {pyproject}: {exc}") from exc

    tool = document.get("tool", {})
    if not isinstance(tool, dict):
        raise ConfigurationError("[tool] must be a table")
    settings = tool.get("uv-torch-compass", {})
    if not isinstance(settings, dict):
        raise ConfigurationError("[tool.uv-torch-compass] must be a table")
    typed_settings = cast(dict[str, Any], settings)
    unknown = sorted(set(typed_settings) - _TOOL_KEYS)
    if unknown:
        raise ConfigurationError(
            "unknown [tool.uv-torch-compass] keys: " + ", ".join(unknown)
        )
    state = typed_settings.get("state")
    if state is not None and not isinstance(state, dict):
        raise ConfigurationError("[tool.uv-torch-compass.state] must be a table")
    return {key: value for key, value in typed_settings.items() if key != "state"}


def _first(*values: str | None) -> str:
    return next((value for value in values if value is not None), "")


def _string_setting(
    key: str,
    cli_value: str | None,
    environment_value: str | None,
    project: Mapping[str, Any],
    *,
    default: str,
) -> str:
    raw = cli_value if cli_value is not None else environment_value
    if raw is None:
        raw = project.get(key, default)
    if not isinstance(raw, str):
        raise ConfigurationError(f"{key} must be a string")
    value = raw.strip()
    if not value and default:
        raise ConfigurationError(f"{key} must not be empty")
    return value


def _optional_string_setting(
    key: str,
    cli_value: str | None,
    environment_value: str | None,
    project: Mapping[str, Any],
) -> str:
    if cli_value is not None:
        raw: object | None = cli_value
    elif environment_value is not None:
        raw = environment_value
    elif key in project:
        raw = project[key]
    else:
        return ""
    if not isinstance(raw, str):
        raise ConfigurationError(f"{key} must be a string")
    value = raw.strip()
    if not value:
        raise ConfigurationError(f"{key} must not be empty")
    return value


def _optional_external_string(
    key: str, cli_value: str | None, environment_value: str | None
) -> str:
    raw = cli_value if cli_value is not None else environment_value
    if raw is None:
        return ""
    value = raw.strip()
    if not value:
        raise ConfigurationError(f"{key} must not be empty")
    return value


def _enum_setting(
    enum_type: type[_EnumSetting],
    key: str,
    cli_value: str | None,
    environment_value: str | None,
    project: Mapping[str, Any],
    *,
    default: str,
) -> _EnumSetting:
    raw = _string_setting(
        key, cli_value, environment_value, project, default=default
    ).lower()
    try:
        return enum_type(raw)
    except ValueError as exc:
        choices = ", ".join(member.value for member in enum_type)
        raise ConfigurationError(
            f"invalid {key} {raw!r}; expected one of {choices}"
        ) from exc


def _integer_setting(
    key: str,
    cli_value: int | None,
    environment_value: str | None,
    project: Mapping[str, Any],
    *,
    default: int,
) -> int:
    raw: object = cli_value if cli_value is not None else environment_value
    if raw is None:
        raw = project.get(key, default)
    if isinstance(raw, bool):
        raise ConfigurationError(f"{key} must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{key} must be an integer") from exc
    if value <= 0:
        raise ConfigurationError(f"{key} must be greater than zero")
    return value


def _list_setting(
    key: str,
    cli_value: Sequence[str] | None,
    environment_value: str | None,
    project: Mapping[str, Any],
) -> tuple[str, ...]:
    raw: object
    if cli_value:
        raw = list(cli_value)
    elif environment_value is not None:
        raw = environment_value.split(",")
    else:
        raw = project.get(key, [])
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ConfigurationError(f"{key} must be an array of strings")
    values: list[str] = []
    seen: set[str] = set()
    for item in raw:
        value = item.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return tuple(values)


def _normalize_requirement_override(package: str, raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value[0] in "<>=!~":
        value = f"{package}{value}"
    try:
        requirement = Requirement(value)
    except InvalidRequirement as exc:
        raise ConfigurationError(
            f"invalid {package} requirement override: {raw!r}"
        ) from exc
    if canonicalize_name(requirement.name) != package:
        raise ConfigurationError(f"{package} override must name {package}")
    return str(requirement)


def _resolve_path(raw_path: str, base: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()
