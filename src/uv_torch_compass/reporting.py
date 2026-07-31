"""Emit redacted progress logs and stable machine-readable command results."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from uv_torch_compass.candidate_failures import (
    FrameworkFailure,
    FrameworkFailureKind,
    ResolutionFailure,
    ResolutionFailureKind,
    ToolFailureKind,
    ToolValidationFailure,
)
from uv_torch_compass.domain import CommandOutcome, OutputFormat, RunOptions
from uv_torch_compass.errors import ProjectUpdateError, ReportError
from uv_torch_compass.redaction import redact
from uv_torch_compass.safe_transaction import atomic_write_private
from uv_torch_compass.workspace import WorkspaceContext


@dataclass(slots=True)
class CommandReporter:
    """Write progress to the terminal and a private per-run log."""

    options: RunOptions
    version: str
    warning_messages: list[str] = field(default_factory=list)
    _started_at: float = field(default_factory=time.monotonic, init=False)
    _log_path: Path | None = field(default=None, init=False)
    _stream: TextIO | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> CommandReporter:
        """Create a unique mode-0600 log file.

        Raises:
            ReportError: If the log directory or file cannot be created safely.
        """
        try:
            self.options.log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
            for suffix in range(1000):
                candidate = self.options.log_dir / (
                    f"{self.options.operation.value}-{stamp}-{os.getpid()}-{suffix}.log"
                )
                try:
                    descriptor = os.open(
                        candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                    )
                except FileExistsError:
                    continue
                self._log_path = candidate
                self._stream = os.fdopen(descriptor, "w", encoding="utf-8")
                break
            if self._stream is None:
                raise ReportError("could not allocate a unique log filename")
        except OSError as exc:
            raise ReportError(f"could not create a private log: {exc}") from exc
        self.detail(
            f"uv-torch-compass {self.version}\n"
            f"operation={self.options.operation.value}\n"
            f"target={self.options.pyproject}\n"
        )
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Flush and close the private log stream."""
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    @property
    def log_path(self) -> Path:
        """Return the allocated log path after entering the context."""
        if self._log_path is None:
            raise ReportError("reporter has not opened its log")
        return self._log_path

    @property
    def elapsed_seconds(self) -> float:
        """Return elapsed monotonic runtime."""
        return time.monotonic() - self._started_at

    def phase(self, name: str, message: str) -> None:
        """Report the start of one domain-level operation phase."""
        self._emit(name.upper(), message)

    def info(self, message: str) -> None:
        """Report supporting information."""
        self._emit("INFO", message)

    def ok(self, message: str) -> None:
        """Report a successful check or side effect."""
        self._emit("OK", message)

    def warn(self, message: str) -> None:
        """Record and display a non-fatal warning."""
        clean = redact(message)
        self.warning_messages.append(clean)
        self._emit("WARN", clean, force_stderr=True)

    def fail(self, message: str) -> None:
        """Report a fatal command failure."""
        self._emit("FAIL", message, force_stderr=True)

    def detail(self, content: str) -> None:
        """Append redacted command output to the log only."""
        if not content or self._stream is None:
            return
        clean = redact(content)
        self._stream.write(clean)
        if not clean.endswith("\n"):
            self._stream.write("\n")
        self._stream.flush()

    def emit_final(
        self,
        outcome: CommandOutcome,
        workspace: WorkspaceContext | None,
        *,
        exit_code: int,
        error: str = "",
    ) -> None:
        """Persist an optional report before emitting the final terminal result.

        Raises:
            ReportError: If report persistence fails. A completed apply remains
                applied because report I/O is outside the project transaction.
        """
        document = _redact_document(
            _result_document(
                self.options,
                workspace,
                outcome,
                log_path=self.log_path,
                elapsed_seconds=self.elapsed_seconds,
                exit_code=exit_code,
                error=error,
                reporter_warnings=tuple(self.warning_messages),
            )
        )
        operation_state = document["operation_state"]
        operation_state["applied"] = outcome.applied
        operation_state["report_written"] = False
        if self.options.report_file is not None:
            operation_state["report_written"] = True
            serialized_report = json.dumps(document, indent=2, sort_keys=True) + "\n"
            try:
                atomic_write_private(self.options.report_file, serialized_report)
            except ProjectUpdateError as exc:
                operation_state["report_written"] = False
                message = f"could not write report {self.options.report_file}: {exc}"
                failure_document = dict(document)
                failure_document["status"] = "failed"
                failure_document["exit_code"] = 1
                failure_document["errors"] = [
                    *document.get("errors", []),
                    message,
                ]
                raise ReportError(
                    message,
                    applied=outcome.applied,
                    document=failure_document,
                ) from exc
        serialized = json.dumps(document, indent=2, sort_keys=True) + "\n"
        if self.options.output_format is OutputFormat.JSON:
            print(serialized, end="", file=sys.stdout, flush=True)
        else:
            self._emit_text_summary(document)

    def _emit(self, level: str, message: str, *, force_stderr: bool = False) -> None:
        clean = redact(message)
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} [{level}] {clean}"
        terminal = (
            sys.stderr
            if force_stderr or self.options.output_format is OutputFormat.JSON
            else sys.stdout
        )
        print(line, file=terminal, flush=True)
        if self._stream is not None:
            print(line, file=self._stream, flush=True)

    def _emit_text_summary(self, document: dict[str, Any]) -> None:
        print("", file=sys.stdout)
        print(f"Result: {document['status']}", file=sys.stdout)
        if document["selected_backend"]:
            print(f"Backend: {document['selected_backend']}", file=sys.stdout)
            print(f"Index: {document['selected_index']}", file=sys.stdout)
            compatibility = document["validation"].get("compatibility", {})
            if compatibility:
                print(
                    "CUDA compatibility: "
                    f"{compatibility.get('level', 'unknown')} "
                    f"({compatibility.get('reason', '')})",
                    file=sys.stdout,
                )
        skipped = [
            attempt
            for attempt in document["candidate_attempts"]
            if attempt["status"] == "skipped"
        ]
        if skipped:
            print("Skipped candidates:", file=sys.stdout)
            displayed = skipped[:5]
            for attempt in displayed:
                print(
                    f"  - {attempt['backend']}: {attempt['reason']}",
                    file=sys.stdout,
                )
            if len(skipped) > len(displayed):
                print(
                    f"  - ... and {len(skipped) - len(displayed)} more "
                    "(see the JSON report or private log)",
                    file=sys.stdout,
                )
        failed = [
            attempt
            for attempt in document["candidate_attempts"]
            if attempt["status"] == "failed"
        ]
        if failed:
            print("Failed candidates:", file=sys.stdout)
            _print_failed_attempts(failed)
            blocking = document.get("blocking_summary")
            if isinstance(blocking, dict) and blocking.get("summary"):
                print(f"Blocking summary: {blocking['summary']}", file=sys.stdout)
        print(f"Applied: {'yes' if document['applied'] else 'no'}", file=sys.stdout)
        if document["changes"]:
            print("Changes:", file=sys.stdout)
            for change in document["changes"]:
                print(f"  - {change}", file=sys.stdout)
        if document["planned_diff"]:
            print("\nPlanned pyproject.toml diff:", file=sys.stdout)
            print(document["planned_diff"], file=sys.stdout)
        if document["errors"]:
            print(f"Error: {document['errors'][0]}", file=sys.stderr)
        print(f"Log: {document['log_file']}", file=sys.stdout)


def _result_document(
    options: RunOptions,
    workspace: WorkspaceContext | None,
    outcome: CommandOutcome,
    *,
    log_path: Path,
    elapsed_seconds: float,
    exit_code: int,
    error: str,
    reporter_warnings: tuple[str, ...],
) -> dict[str, Any]:
    runtime = outcome.runtime
    selected_backend = runtime.backend.value if runtime is not None else ""
    selected_index = runtime.backend.index_url if runtime is not None else ""
    python = outcome.metadata.get("python", {})
    selected_gpu = outcome.metadata.get("gpu")
    runtime_document: dict[str, Any] = {}
    if runtime is not None:
        runtime_document = {
            "schema_version": runtime.schema_version,
            "torch": runtime.torch_version,
            "torchvision": runtime.torchvision_version,
            "torchaudio": runtime.torchaudio_version,
            "numpy": runtime.numpy_version,
            "cuda_runtime": runtime.cuda_runtime,
            "runtime_component_version": runtime.runtime_component_version,
            "gpu_name": runtime.gpu_name,
            "gpu_device_capability": runtime.gpu_device_capability,
            "compiled_architectures": list(runtime.compiled_architectures),
            "native_architecture_test": runtime.native_architecture_test,
            "cuda_test": runtime.cuda_test,
            "cublas_test": runtime.cublas_test,
            "cudnn_test": runtime.cudnn_test,
            "numpy_bridge_test": runtime.numpy_bridge_test,
            "torchvision_test": runtime.torchvision_test,
            "torchaudio_test": runtime.torchaudio_test,
            "compile_test": runtime.compile_test,
            "probe_profile": runtime.probe_profile,
        }
        if outcome.compatibility is not None:
            runtime_document["compatibility"] = {
                "level": outcome.compatibility.level.value,
                "minimum_driver": outcome.compatibility.minimum_driver,
                "reason": outcome.compatibility.reason,
            }
    return {
        "schema_version": 8,
        "operation": options.operation.value,
        "status": outcome.status,
        "exit_code": exit_code,
        "applied": outcome.applied,
        "target": str(options.pyproject),
        "workspace": str(workspace.workspace_root) if workspace else "",
        "package": workspace.package if workspace else None,
        "request": {
            "python": options.python_request,
            "requirements": list(options.requirement_overrides),
            "backend": (options.backend.concrete_value or options.backend.kind.value),
            "channel": options.channel.value,
            "cuda_compatibility": options.cuda_compatibility.value,
            "probe_profile": options.probe_profile.value,
            "framework_probes": [probe.value for probe in options.framework_probes],
            "extras": list(options.extras),
            "groups": list(options.groups),
            "cuda_device": options.cuda_device.value if options.cuda_device else None,
        },
        "python": python,
        "candidate_attempts": [
            {
                "backend": attempt.backend,
                "stage": attempt.stage,
                "status": attempt.status,
                "reason": redact(attempt.reason),
                "compatibility": attempt.compatibility,
                "failure": _failure_document(attempt.failure),
                "resolution": _resolution_document(attempt.resolution),
                "framework_compatibility": _framework_compatibility_document(
                    attempt.framework_compatibility
                ),
                "framework": {
                    "requested": list(attempt.framework_requests),
                    "resolved": _resolved_framework_versions(attempt.resolution),
                },
                "phases": _phase_document(attempt.stage, attempt.status),
            }
            for attempt in outcome.attempts
        ],
        "blocking_summary": _blocking_summary(outcome),
        "failure_category": _failure_category(outcome),
        "selected_backend": selected_backend,
        "selected_index": selected_index,
        "selected_gpu": selected_gpu,
        "resolved_packages": (
            {
                "torch": runtime.torch_version,
                "torchvision": runtime.torchvision_version,
                "torchaudio": runtime.torchaudio_version,
                "numpy": runtime.numpy_version,
            }
            if runtime is not None
            else {}
        ),
        "dependency_roots": outcome.metadata.get("dependency_roots", []),
        "source_anchors": outcome.metadata.get("source_anchors", []),
        "required_environment": outcome.metadata.get("required_environment", ""),
        "probe_contract": outcome.metadata.get("probe_contract", {}),
        "framework_validation": outcome.metadata.get("framework_validation", []),
        "framework_version_selection": outcome.metadata.get(
            "framework_version_selection"
        ),
        "operation_state": outcome.metadata.get(
            "operation_state",
            {"applied": outcome.applied, "report_written": False},
        ),
        "environment_policy": outcome.metadata.get("environment_policy", {}),
        "validation": runtime_document,
        "changes": list(outcome.changes),
        "backups": [str(path) for path in outcome.backups],
        "warnings": list(dict.fromkeys((*outcome.warnings, *reporter_warnings))),
        "errors": [error] if error else [],
        "planned_diff": redact(outcome.planned_diff),
        "metadata": outcome.metadata,
        "timing": {"elapsed_seconds": round(elapsed_seconds, 3)},
        "log_file": str(log_path),
    }


def _failure_document(failure: Any) -> dict[str, Any] | None:
    if failure is None:
        return None
    if isinstance(failure, ToolValidationFailure):
        return {
            "kind": failure.kind.value,
            "summary": failure.summary,
            "suggestions": list(failure.suggestions),
            "backend_independent": True,
        }
    package = (
        {
            "name": failure.package.name,
            "version": failure.package.version,
            "requirement": failure.package.requirement,
        }
        if failure.package is not None
        else None
    )
    if isinstance(failure, FrameworkFailure):
        return {
            "kind": failure.kind.value,
            "summary": failure.summary,
            "framework": failure.framework,
            "framework_version": failure.framework_version,
            "package": package,
            "dependency_paths": [list(path) for path in failure.dependency_paths],
            "binary_requirement": _binary_requirement_document(
                failure.binary_requirement
            ),
            "exception": _bounded_exception_document(failure.exception),
            "packages": [
                {
                    "name": item.name,
                    "version": item.version,
                    "source_url": item.source_url,
                }
                for item in failure.packages
            ],
            "suggestions": list(failure.suggestions),
            "backend_independent": failure.backend_independent,
        }
    if not isinstance(failure, ResolutionFailure):
        return None
    index = (
        {"name": failure.index.name, "url": failure.index.url}
        if failure.index is not None
        else None
    )
    return {
        "kind": failure.kind.value,
        "summary": failure.summary,
        "package": package,
        "required_by": list(failure.required_by),
        "index": index,
        "platform": failure.platform,
        "suggestions": list(failure.suggestions),
        "dependency_paths": [list(path) for path in failure.dependency_paths],
        "available_wheel_platforms": list(failure.available_wheel_platforms),
        "backend_independent": False,
    }


def _resolution_document(resolution: Any) -> dict[str, Any] | None:
    if resolution is None:
        return None
    return {
        "status": "resolved",
        "environment": {
            "implementation": resolution.environment.implementation_name,
            "python_version": resolution.environment.python_version,
            "python_minor": resolution.environment.python_minor,
            "sys_platform": resolution.environment.sys_platform,
            "platform_machine": resolution.environment.platform_machine,
            "required_marker": resolution.environment.required_environment_marker,
        },
        "pytorch": {
            package.name: {
                "version": package.version,
                "index": package.source_url,
            }
            for package in resolution.pytorch_packages
        },
        "framework_packages": {
            package.name: {
                "version": package.version,
                "index": package.source_url,
            }
            for package in resolution.framework_packages
        },
        "package_count": len(resolution.lock.packages),
        "evidence_source": resolution.lock.evidence_source.value,
        "lock_schema": (
            {
                "version": resolution.lock.lock_schema.version,
                "revision": resolution.lock.lock_schema.revision,
            }
            if resolution.lock.lock_schema is not None
            else None
        ),
    }


def _resolved_framework_versions(resolution: Any) -> dict[str, str]:
    if resolution is None:
        return {}
    return {package.name: package.version for package in resolution.framework_packages}


def _phase_document(terminal_stage: str, status: str) -> dict[str, str]:
    stages = ("lock", "artifact", "install", "runtime", "framework")
    if status == "skipped":
        return dict.fromkeys(stages, "not-run")
    terminal_position = stages.index(terminal_stage)
    return {
        stage: (
            "passed"
            if position < terminal_position
            else status
            if position == terminal_position
            else "not-run"
        )
        for position, stage in enumerate(stages)
    }


def _blocking_summary(outcome: CommandOutcome) -> dict[str, Any] | None:
    failed = [attempt for attempt in outcome.attempts if attempt.failure is not None]
    if not failed:
        return None
    resolved_builds: list[dict[str, Any]] = []
    blockers: dict[tuple[object, ...], dict[str, Any]] = {}
    suggestions: list[str] = []
    for attempt in failed:
        resolution = attempt.resolution
        if resolution is not None:
            build = {
                "backend": attempt.backend,
                "index": resolution.backend.index_url,
                "packages": {
                    package.name: package.version
                    for package in resolution.pytorch_packages
                },
            }
            if build not in resolved_builds:
                resolved_builds.append(build)
        failure = attempt.failure
        if failure is None:
            continue
        package = getattr(failure, "package", None)
        package_name = package.name if package is not None else None
        package_version = package.version if package is not None else None
        platform = failure.platform if isinstance(failure, ResolutionFailure) else None
        dependency_paths = getattr(failure, "dependency_paths", ())
        fingerprint = (
            failure.kind.value,
            package_name,
            package_version,
            platform,
            dependency_paths,
        )
        blocker = blockers.setdefault(
            fingerprint,
            {
                "kind": failure.kind.value,
                "package": package_name,
                "version": package_version,
                "platform": platform,
                "dependency_paths": [list(path) for path in dependency_paths],
                "candidates": [],
                "backend_independent": (
                    failure.backend_independent
                    if isinstance(failure, FrameworkFailure)
                    else False
                ),
            },
        )
        blocker["candidates"].append(attempt.backend)
        suggestions.extend(failure.suggestions)
    summary = (
        "Compatible PyTorch builds were resolved, but a later candidate phase failed."
        if resolved_builds
        else "No candidate resolved the selected dependency graph."
    )
    return {
        "summary": summary,
        "pytorch_builds_found": resolved_builds,
        "common_blockers": list(blockers.values()),
        "suggestions": list(dict.fromkeys(suggestions)),
    }


def _failure_category(outcome: CommandOutcome) -> str | None:
    failures = tuple(
        attempt.failure for attempt in outcome.attempts if attempt.failure is not None
    )
    if not failures:
        return None
    if any(
        isinstance(failure, ToolValidationFailure)
        and failure.kind is ToolFailureKind.UNSUPPORTED_LOCK_SCHEMA
        for failure in failures
    ):
        return "lock-schema-unsupported"
    if any(isinstance(failure, ToolValidationFailure) for failure in failures):
        return "tool-validation-error"
    if any(
        isinstance(failure, FrameworkFailure)
        and failure.kind is FrameworkFailureKind.API_INCOMPATIBILITY
        for failure in failures
    ):
        return "framework-api-incompatible"
    if any(
        isinstance(failure, FrameworkFailure)
        and failure.kind is FrameworkFailureKind.CUDA_ABI
        for failure in failures
    ):
        return "framework-cuda-incompatible"
    if any(isinstance(failure, FrameworkFailure) for failure in failures):
        return "framework-validation-failed"
    if any(
        isinstance(failure, ResolutionFailure)
        and failure.kind is ResolutionFailureKind.RUNTIME_VALIDATION
        for failure in failures
    ):
        return "runtime-validation-failed"
    return "dependency-unsatisfiable"


def _framework_compatibility_document(decision: Any) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "status": decision.status.value,
        "candidate_backend": decision.candidate_backend,
        "summary": decision.summary,
        "requirement": _binary_requirement_document(decision.requirement),
    }


def _binary_requirement_document(requirement: Any) -> dict[str, Any] | None:
    if requirement is None:
        return None
    return {
        "framework": requirement.framework,
        "version": requirement.version,
        "required_cuda_variant": requirement.required_cuda_variant,
        "required_cuda_major": requirement.required_cuda_major,
        "needed_libraries": list(requirement.needed_libraries),
        "evidence": requirement.evidence.value,
        "source_url": requirement.source_url,
    }


def _bounded_exception_document(exception: Any) -> dict[str, Any] | None:
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


def _print_failed_attempt(attempt: dict[str, Any]) -> None:
    failure = attempt.get("failure")
    if not isinstance(failure, dict):
        print(f"  - {attempt['backend']}: {attempt['reason']}", file=sys.stdout)
        return
    print(
        f"  - {attempt['backend']}: {failure['summary']}",
        file=sys.stdout,
    )
    resolution = attempt.get("resolution")
    if isinstance(resolution, dict) and resolution.get("pytorch"):
        packages = ", ".join(
            f"{name}=={details['version']}"
            for name, details in resolution["pytorch"].items()
        )
        print(f"    Resolved PyTorch: {packages}", file=sys.stdout)
    package = failure.get("package")
    if isinstance(package, dict):
        requirement = package.get("requirement") or package.get("name")
        print(f"    Package: {requirement}", file=sys.stdout)
    required_by = failure.get("required_by")
    if required_by:
        print(f"    Required by: {' -> '.join(required_by)}", file=sys.stdout)
    dependency_paths = failure.get("dependency_paths", [])
    if dependency_paths:
        print(
            f"    Dependency path: {' -> '.join(dependency_paths[0])}",
            file=sys.stdout,
        )
    index = failure.get("index")
    if isinstance(index, dict):
        label = index.get("name") or "package index"
        print(f"    Index: {label} ({index.get('url', '')})", file=sys.stdout)
    if failure.get("platform"):
        print(f"    Platform: {failure['platform']}", file=sys.stdout)
    if failure.get("available_wheel_platforms"):
        print(
            "    Available wheels: " + ", ".join(failure["available_wheel_platforms"]),
            file=sys.stdout,
        )
    requirement = failure.get("binary_requirement")
    if isinstance(requirement, dict):
        required = requirement.get("required_cuda_variant") or (
            f"CUDA {requirement['required_cuda_major']}"
            if requirement.get("required_cuda_major") is not None
            else "unknown CUDA variant"
        )
        print(
            f"    Framework requirement: {required} "
            f"({requirement.get('evidence', 'unknown')})",
            file=sys.stdout,
        )
    exception = failure.get("exception")
    if isinstance(exception, dict) and exception.get("message"):
        print(
            f"    Exception: {exception.get('type', 'Error')}: {exception['message']}",
            file=sys.stdout,
        )
    for suggestion in failure.get("suggestions", []):
        print(f"    Suggestion: {suggestion}", file=sys.stdout)


def _print_failed_attempts(attempts: list[dict[str, Any]]) -> None:
    unavailable: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for attempt in attempts:
        failure = attempt.get("failure")
        package = failure.get("package") if isinstance(failure, dict) else None
        if (
            isinstance(failure, dict)
            and failure.get("kind") == "no-compatible-distribution"
            and isinstance(package, dict)
            and package.get("name") in {"torch", "torchvision", "torchaudio"}
        ):
            unavailable.append(attempt)
        else:
            remaining.append(attempt)
    if len(unavailable) > 1:
        backends = ", ".join(
            dict.fromkeys(str(attempt["backend"]) for attempt in unavailable)
        )
        print(
            "  - PyTorch builds were unavailable from: " + backends,
            file=sys.stdout,
        )
        print(
            "    See the JSON report or private log for each requirement and index.",
            file=sys.stdout,
        )
    else:
        remaining.extend(unavailable)
    for attempt in remaining:
        _print_failed_attempt(attempt)


def _redact_document(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [_redact_document(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_document(item) for key, item in value.items()}
    return value
