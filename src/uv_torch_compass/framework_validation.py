"""Define and validate explicit or automatically detected framework results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

from uv_torch_compass.candidate_failures import (
    BoundedExceptionReport,
    FrameworkPackageVersion,
    FrameworkProbeTrigger,
    TracebackFrame,
)
from uv_torch_compass.domain import FrameworkProbe
from uv_torch_compass.errors import ProbeError

_MAX_EXCEPTION_MESSAGE = 4096
_MAX_TRACEBACK_FRAMES = 12


@dataclass(frozen=True, slots=True)
class FrameworkValidation:
    """Describe one framework's import, native, and platform checks."""

    framework: FrameworkProbe
    status: str
    version: str
    import_test: str
    native_extension_test: str
    platform_test: str
    platform: str
    error: str = ""
    trigger: FrameworkProbeTrigger = FrameworkProbeTrigger.EXPLICIT
    exception: BoundedExceptionReport | None = None
    packages: tuple[FrameworkPackageVersion, ...] = ()


def parse_framework_probe(
    output: str,
    expected: tuple[FrameworkProbe, ...],
    *,
    allow_automatic: bool = False,
    require_success: bool = True,
) -> tuple[FrameworkValidation, ...]:
    """Parse a framework probe and enforce the requested validation contract.

    Args:
        output: Probe output whose final non-empty line must be one JSON object.
        expected: Frameworks explicitly requested by the caller.
        allow_automatic: Whether installed frameworks may add probe results.
        require_success: Whether failed validations should raise ``ProbeError``.

    Raises:
        ProbeError: If output is malformed, incomplete, or reports a required failure.
    """
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        raise ProbeError("framework probe produced no result")
    try:
        document = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ProbeError("framework probe did not end with valid JSON") from exc
    if not isinstance(document, dict) or document.get("schema_version") not in {1, 2}:
        raise ProbeError("framework probe schema version is unsupported")
    raw_results = document.get("results")
    if not isinstance(raw_results, list):
        raise ProbeError("framework probe results must be an array")

    results = tuple(_parse_result(raw) for raw in raw_results)
    by_framework = {result.framework: result for result in results}
    if not set(expected).issubset(by_framework) or (
        not allow_automatic and set(by_framework) != set(expected)
    ):
        raise ProbeError("framework probe did not return exactly the requested results")
    failed = [result for result in results if result.status != "PASS"]
    if require_success and failed:
        details = "; ".join(
            f"{result.framework.value}: {result.error or 'validation failed'}"
            for result in failed
        )
        raise ProbeError(f"framework validation failed: {details}")
    order = tuple(dict.fromkeys((*expected, *by_framework)))
    return tuple(by_framework[item] for item in order)


def framework_validation_document(
    results: tuple[FrameworkValidation, ...],
) -> list[dict[str, object]]:
    """Return stable JSON-compatible framework validation records."""
    return [
        {
            "framework": result.framework.value,
            "status": result.status,
            "version": result.version,
            "import_test": result.import_test,
            "native_extension_test": result.native_extension_test,
            "platform_test": result.platform_test,
            "platform": result.platform,
            "error": result.error,
            "trigger": result.trigger.value,
            "exception": _exception_document(result.exception),
            "packages": [
                {
                    "name": package.name,
                    "version": package.version,
                    "source_url": package.source_url,
                }
                for package in result.packages
            ],
        }
        for result in results
    ]


def _parse_result(raw: object) -> FrameworkValidation:
    if not isinstance(raw, dict):
        raise ProbeError("framework probe result must be an object")
    required_keys = (
        "framework",
        "status",
        "version",
        "import_test",
        "native_extension_test",
        "platform_test",
        "platform",
        "error",
        "trigger",
    )
    values = {key: raw.get(key) for key in required_keys}
    if not all(isinstance(value, str) for value in values.values()):
        raise ProbeError("framework probe result contains a non-string field")
    try:
        framework = FrameworkProbe(cast(str, values["framework"]))
    except ValueError as exc:
        raise ProbeError("framework probe returned an unknown framework") from exc
    try:
        trigger = FrameworkProbeTrigger(cast(str, values["trigger"]))
    except ValueError as exc:
        raise ProbeError("framework probe returned an invalid trigger") from exc
    exception = _parse_exception(raw.get("exception"))
    packages = _parse_packages(raw.get("packages", []))
    return FrameworkValidation(
        framework,
        cast(str, values["status"]),
        cast(str, values["version"]),
        cast(str, values["import_test"]),
        cast(str, values["native_extension_test"]),
        cast(str, values["platform_test"]),
        cast(str, values["platform"]),
        cast(str, values["error"])[:_MAX_EXCEPTION_MESSAGE],
        trigger,
        exception,
        packages,
    )


def _parse_exception(raw: object) -> BoundedExceptionReport | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ProbeError("framework exception must be an object")
    exception_type = raw.get("type")
    message = raw.get("message")
    if not isinstance(exception_type, str) or not isinstance(message, str):
        raise ProbeError("framework exception has invalid identity fields")
    raw_frames = raw.get("frames", [])
    if not isinstance(raw_frames, list) or len(raw_frames) > _MAX_TRACEBACK_FRAMES:
        raise ProbeError("framework exception has invalid traceback frames")
    frames: list[TracebackFrame] = []
    for raw_frame in raw_frames:
        if not isinstance(raw_frame, dict):
            raise ProbeError("framework traceback frame must be an object")
        module = raw_frame.get("module")
        filename = raw_frame.get("filename")
        function = raw_frame.get("function")
        line_number = raw_frame.get("line_number")
        if (
            not isinstance(module, str)
            or not isinstance(filename, str)
            or not isinstance(function, str)
            or not isinstance(line_number, int)
            or isinstance(line_number, bool)
            or line_number < 0
        ):
            raise ProbeError("framework traceback frame contains an invalid field")
        frames.append(TracebackFrame(module, filename, function, line_number))
    optional = {
        key: raw.get(key)
        for key in (
            "missing_symbol",
            "missing_module",
            "consumer_package",
            "provider_package",
        )
    }
    if any(
        value is not None and not isinstance(value, str) for value in optional.values()
    ):
        raise ProbeError("framework exception contains an invalid optional field")
    return BoundedExceptionReport(
        exception_type,
        message[:_MAX_EXCEPTION_MESSAGE],
        tuple(frames),
        cast(str | None, optional["missing_symbol"]),
        cast(str | None, optional["missing_module"]),
        cast(str | None, optional["consumer_package"]),
        cast(str | None, optional["provider_package"]),
    )


def _parse_packages(raw: object) -> tuple[FrameworkPackageVersion, ...]:
    if not isinstance(raw, list):
        raise ProbeError("framework package versions must be an array")
    packages: list[FrameworkPackageVersion] = []
    for value in raw:
        if not isinstance(value, dict):
            raise ProbeError("framework package version must be an object")
        name = value.get("name")
        version = value.get("version")
        source_url = value.get("source_url", "")
        if not all(isinstance(item, str) for item in (name, version, source_url)):
            raise ProbeError("framework package version contains an invalid field")
        packages.append(
            FrameworkPackageVersion(
                cast(str, name),
                cast(str, version),
                cast(str, source_url),
            )
        )
    return tuple(packages)


def _exception_document(
    exception: BoundedExceptionReport | None,
) -> dict[str, object] | None:
    if exception is None:
        return None
    return {
        "type": exception.exception_type,
        "message": exception.message,
        "missing_symbol": exception.missing_symbol,
        "missing_module": exception.missing_module,
        "consumer_package": exception.consumer_package,
        "provider_package": exception.provider_package,
        "frames": [
            {
                "module": frame.module,
                "filename": frame.filename,
                "function": frame.function,
                "line_number": frame.line_number,
            }
            for frame in exception.frames
        ],
    }
