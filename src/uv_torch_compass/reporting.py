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
            for attempt in skipped:
                print(
                    f"  - {attempt['backend']}: {attempt['reason']}",
                    file=sys.stdout,
                )
        failed = [
            attempt
            for attempt in document["candidate_attempts"]
            if attempt["status"] == "failed"
        ]
        if failed:
            print("Failed candidates:", file=sys.stdout)
            for attempt in failed:
                _print_failed_attempt(attempt)
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
        "schema_version": 5,
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
            }
            for attempt in outcome.attempts
        ],
        "resolution_failure": (
            _aggregate_resolution_failure(outcome)
            if outcome.status == "failed"
            else None
        ),
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
        "operation_state": outcome.metadata.get(
            "operation_state",
            {"applied": outcome.applied, "report_written": False},
        ),
        "environment_policy": outcome.metadata.get("environment_policy", {}),
        "candidate_failure_summary": _aggregate_resolution_failure(outcome),
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
    package = (
        {
            "name": failure.package.name,
            "version": failure.package.version,
            "requirement": failure.package.requirement,
        }
        if failure.package is not None
        else None
    )
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
    }


def _aggregate_resolution_failure(outcome: CommandOutcome) -> dict[str, Any] | None:
    failures = [
        attempt.failure for attempt in outcome.attempts if attempt.failure is not None
    ]
    if not failures:
        return None
    packages = list(
        dict.fromkeys(
            failure.package.name for failure in failures if failure.package is not None
        )
    )
    indexes = list(
        dict.fromkeys(
            failure.index.name or failure.index.url
            for failure in failures
            if failure.index is not None
        )
    )
    suggestions = list(
        dict.fromkeys(
            suggestion for failure in failures for suggestion in failure.suggestions
        )
    )
    return {
        "summary": "No candidate satisfied the selected dependency graph.",
        "packages": packages,
        "indexes": indexes,
        "suggestions": suggestions,
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
    package = failure.get("package")
    if isinstance(package, dict):
        requirement = package.get("requirement") or package.get("name")
        print(f"    Package: {requirement}", file=sys.stdout)
    required_by = failure.get("required_by")
    if required_by:
        print(f"    Required by: {' -> '.join(required_by)}", file=sys.stdout)
    index = failure.get("index")
    if isinstance(index, dict):
        label = index.get("name") or "package index"
        print(f"    Index: {label} ({index.get('url', '')})", file=sys.stdout)
    if failure.get("platform"):
        print(f"    Platform: {failure['platform']}", file=sys.stdout)
    for suggestion in failure.get("suggestions", []):
        print(f"    Suggestion: {suggestion}", file=sys.stdout)


def _redact_document(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [_redact_document(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_document(item) for key, item in value.items()}
    return value
