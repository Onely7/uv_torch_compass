"""Execute external commands through an explicit, shell-free boundary."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from uv_torch_compass.errors import (
    CommandError,
    CommandTimeoutError,
    TerminationRequested,
)

_CONTROL_ENVIRONMENT = {
    "CONDA_PREFIX",
    "UV_FROZEN",
    "UV_LOCKED",
    "UV_NO_SYNC",
    "UV_PROJECT",
    "UV_PYTHON",
    "UV_PYTHON_PREFERENCE",
    "UV_TORCH_BACKEND",
    "UV_WORKING_DIR",
    "VIRTUAL_ENV",
}


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Capture an external command result without imposing success policy."""

    returncode: int
    stdout: str
    stderr: str


class ProcessRunner(Protocol):
    """Execute commands at the infrastructure boundary."""

    def run(
        self,
        arguments: Sequence[str | Path],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        """Run one command and capture its output."""


@dataclass(frozen=True, slots=True)
class SubprocessRunner:
    """Execute argument-vector commands without involving a shell."""

    def run(
        self,
        arguments: Sequence[str | Path],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        """Run one command and return decoded output.

        Args:
            arguments: Executable followed by individual, unsplit arguments.
            cwd: Working directory required by the external tool.
            env: Complete child environment, or the current environment when absent.
            timeout_seconds: Maximum runtime, or no deadline when omitted.

        Returns:
            The process exit status and captured streams.

        Raises:
            CommandError: If no executable is supplied or process creation fails.
        """
        if not arguments:
            raise CommandError("cannot execute an empty command")

        executable = Path(arguments[0])
        if not executable.is_absolute():
            raise CommandError(f"executable path must be absolute: {executable}")

        try:
            process = subprocess.Popen(
                [str(argument) for argument in arguments],
                cwd=cwd,
                env=dict(env) if env is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=True,
            )
        except OSError as exc:
            raise CommandError(f"could not execute {executable}: {exc}") from exc

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _stop_process_group(process)
            raise CommandTimeoutError(
                f"command timed out after {timeout_seconds:g}s: {executable}"
            ) from exc
        except (KeyboardInterrupt, TerminationRequested):
            _stop_process_group(process)
            raise

        return CommandResult(process.returncode, stdout, stderr)


def resolve_executable(name: str) -> Path:
    """Resolve an executable through PATH once.

    Raises:
        CommandError: If the executable is unavailable.
    """
    resolved = shutil.which(name)
    if resolved is None:
        raise CommandError(f"{name} was not found in PATH")
    return Path(resolved).resolve()


def sanitized_environment(
    environ: Mapping[str, str], *, overrides: Mapping[str, str] | None = None
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Remove workflow-control variables while preserving connectivity settings.

    Args:
        environ: Parent environment to filter.
        overrides: Trusted values that the application must set explicitly.

    Returns:
        The child environment and sorted names removed from the parent.
    """
    removed = tuple(
        sorted(
            key
            for key in environ
            if key in _CONTROL_ENVIRONMENT or key.startswith("UV_TORCH_COMPASS_")
        )
    )
    child = {key: value for key, value in environ.items() if key not in removed}
    if overrides:
        child.update(overrides)
    return child, removed


def _stop_process_group(process: subprocess.Popen[str]) -> None:
    """Terminate a child process group and kill it after a short grace period."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            process.wait()
