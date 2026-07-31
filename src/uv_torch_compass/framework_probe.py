"""Probe selected framework integrations without starting user workloads."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
from collections.abc import Sequence


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
    results = []
    for framework in frameworks:
        if framework == "vllm":
            result = _probe_vllm(namespace.expected_backend)
            result["trigger"] = "explicit" if framework in explicit else "automatic"
            results.append(result)
    print(json.dumps({"schema_version": 1, "results": results}, sort_keys=True))
    return 0 if all(result["status"] == "PASS" for result in results) else 1


def _probe_vllm(expected_backend: str) -> dict[str, str]:
    result = {
        "framework": "vllm",
        "status": "FAIL",
        "version": "not-installed",
        "import_test": "FAIL",
        "native_extension_test": "FAIL",
        "platform_test": "FAIL",
        "platform": "unknown",
        "error": "",
        "trigger": "explicit",
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
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _distribution_installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
