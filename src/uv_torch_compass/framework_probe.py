"""Probe selected framework integrations without starting user workloads."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import re
import traceback
from collections.abc import Sequence
from pathlib import Path

_DIAGNOSTIC_PACKAGES = (
    "vllm",
    "transformers",
    "xformers",
    "vllm-flash-attn",
    "torch",
    "torchvision",
    "torchaudio",
)
_MISSING_SYMBOL = re.compile(r"cannot import name ['\"](?P<name>[^'\"]+)['\"]")
_MISSING_MODULE = re.compile(r"No module named ['\"](?P<name>[^'\"]+)['\"]")
_MAX_MESSAGE = 4096
_MAX_TRACEBACK = 16_384
_MAX_TRACEBACK_FRAMES = 12


def main(arguments: Sequence[str] | None = None) -> int:
    """Run bounded framework checks and emit one JSON result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", action="append", choices=("vllm",), default=[])
    parser.add_argument("--auto-detect", action="store_true")
    parser.add_argument("--expected-backend", required=True)
    namespace = parser.parse_args(arguments)
    explicit = tuple(dict.fromkeys(namespace.framework))
    automatic = (
        ("vllm",) if namespace.auto_detect and _distribution_installed("vllm") else ()
    )
    frameworks = tuple(dict.fromkeys((*explicit, *automatic)))
    results: list[dict[str, object]] = []
    for framework in frameworks:
        if framework == "vllm":
            result = _probe_vllm(namespace.expected_backend)
            result["trigger"] = "explicit" if framework in explicit else "automatic"
            results.append(result)
    print(json.dumps({"schema_version": 2, "results": results}, sort_keys=True))
    return 0 if all(result["status"] == "PASS" for result in results) else 1


def _probe_vllm(expected_backend: str) -> dict[str, object]:
    result: dict[str, object] = {
        "framework": "vllm",
        "status": "FAIL",
        "version": "not-installed",
        "import_test": "FAIL",
        "native_extension_test": "FAIL",
        "platform_test": "FAIL",
        "platform": "unknown",
        "error": "",
        "trigger": "explicit",
        "exception": None,
        "packages": _package_versions(),
    }
    try:
        result["version"] = importlib.metadata.version("vllm")
        importlib.import_module("vllm")
        result["import_test"] = "PASS"
        if importlib.util.find_spec("vllm._C") is None:
            raise RuntimeError("vllm native extension vllm._C was not found")
        importlib.import_module("vllm._C")
        result["native_extension_test"] = "PASS"

        platforms = importlib.import_module("vllm.platforms")
        current = getattr(platforms, "current_platform", None)
        if current is None:
            raise RuntimeError("vllm did not expose current_platform")
        result["platform"] = type(current).__name__
        predicate_name = "is_cpu" if expected_backend == "cpu" else "is_cuda"
        predicate = getattr(current, predicate_name, None)
        if not callable(predicate) or not predicate():
            raise RuntimeError(
                f"vllm platform does not match backend {expected_backend}"
            )
        result["platform_test"] = "PASS"
        result["status"] = "PASS"
    except Exception as exc:  # The isolated process reports third-party failures.
        result["error"] = f"{type(exc).__name__}: {exc}"[:_MAX_MESSAGE]
        result["exception"] = _exception_report(exc)
        result["traceback"] = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )[-_MAX_TRACEBACK:]
    return result


def _exception_report(exc: Exception) -> dict[str, object]:
    extracted = traceback.extract_tb(exc.__traceback__)[-_MAX_TRACEBACK_FRAMES:]
    frames = [
        {
            "module": _frame_module(frame.filename),
            "filename": Path(frame.filename).name,
            "function": frame.name,
            "line_number": max(frame.lineno or 0, 0),
        }
        for frame in extracted
    ]
    message = str(exc)[:_MAX_MESSAGE]
    missing_symbol_match = _MISSING_SYMBOL.search(message)
    missing_module_match = _MISSING_MODULE.search(message)
    consumer = next(
        (
            module.split(".", 1)[0]
            for frame in reversed(extracted)
            if (module := _frame_module(frame.filename))
        ),
        None,
    )
    provider = None
    if missing_module_match is not None:
        provider = missing_module_match.group("name").split(".", 1)[0]
    elif " from " in message:
        provider = message.rsplit(" from ", 1)[-1].strip(" '\"").split(".", 1)[0]
    return {
        "type": type(exc).__name__,
        "message": message,
        "frames": frames,
        "missing_symbol": (
            missing_symbol_match.group("name") if missing_symbol_match else None
        ),
        "missing_module": (
            missing_module_match.group("name") if missing_module_match else None
        ),
        "consumer_package": consumer,
        "provider_package": provider,
    }


def _frame_module(filename: str) -> str:
    parts = Path(filename).parts
    for anchor in ("site-packages", "dist-packages"):
        if anchor in parts:
            index = parts.index(anchor) + 1
            if index < len(parts):
                return parts[index].removesuffix(".py").replace("-", "_")
    return ""


def _package_versions() -> list[dict[str, str]]:
    packages: list[dict[str, str]] = []
    for name in _DIAGNOSTIC_PACKAGES:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        packages.append({"name": name, "version": version, "source_url": ""})
    return packages


def _distribution_installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
