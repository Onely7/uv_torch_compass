"""Define and validate explicit or automatically detected framework results."""

from __future__ import annotations

import json
from dataclasses import dataclass

from uv_torch_compass.domain import FrameworkProbe
from uv_torch_compass.errors import ProbeError


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
    trigger: str = "explicit"


def parse_framework_probe(
    output: str,
    expected: tuple[FrameworkProbe, ...],
    *,
    allow_automatic: bool = False,
) -> tuple[FrameworkValidation, ...]:
    """Parse a framework probe and enforce the requested validation contract.

    Raises:
        ProbeError: If output is malformed, incomplete, or reports a failure.
    """
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        raise ProbeError("framework probe produced no result")
    try:
        document = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ProbeError("framework probe did not end with valid JSON") from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ProbeError("framework probe schema version is unsupported")
    raw_results = document.get("results")
    if not isinstance(raw_results, list):
        raise ProbeError("framework probe results must be an array")

    results: list[FrameworkValidation] = []
    for raw in raw_results:
        if not isinstance(raw, dict):
            raise ProbeError("framework probe result must be an object")
        values = {
            key: raw.get(key)
            for key in (
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
        }
        if not all(isinstance(value, str) for value in values.values()):
            raise ProbeError("framework probe result contains a non-string field")
        try:
            framework = FrameworkProbe(values["framework"])
        except ValueError as exc:
            raise ProbeError("framework probe returned an unknown framework") from exc
        if values["trigger"] not in {"explicit", "automatic"}:
            raise ProbeError("framework probe returned an invalid trigger")
        results.append(
            FrameworkValidation(
                framework,
                values["status"],
                values["version"],
                values["import_test"],
                values["native_extension_test"],
                values["platform_test"],
                values["platform"],
                values["error"],
                values["trigger"],
            )
        )

    by_framework = {result.framework: result for result in results}
    if not set(expected).issubset(by_framework) or (
        not allow_automatic and set(by_framework) != set(expected)
    ):
        raise ProbeError("framework probe did not return exactly the requested results")
    failed = [result for result in results if result.status != "PASS"]
    if failed:
        details = "; ".join(
            f"{result.framework.value}: {result.error or 'validation failed'}"
            for result in failed
        )
        raise ProbeError(f"framework validation failed: {details}")
    order = tuple(dict.fromkeys((*expected, *by_framework)))
    return tuple(by_framework[item] for item in order)


def framework_validation_document(
    results: tuple[FrameworkValidation, ...],
) -> list[dict[str, str]]:
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
            "trigger": result.trigger,
        }
        for result in results
    ]
