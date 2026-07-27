"""Translate application operations into bounded uv command invocations."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from uv_torch_compass.command_runner import (
    CommandResult,
    ProcessRunner,
    resolve_executable,
    sanitized_environment,
)
from uv_torch_compass.domain import BackendCandidate, Channel

_BACKEND_HELP_PATTERN = re.compile(r"\b(?:auto|cpu|cu[0-9]{2,3})\b", re.ASCII)


@dataclass(frozen=True, slots=True)
class UvCommandClient:
    """Expose uv operations without leaking command assembly into workflows."""

    executable: Path
    runner: ProcessRunner
    link_mode: str
    heavy_timeout_seconds: int
    diagnostic_timeout_seconds: int = 30
    environment: Mapping[str, str] = field(default_factory=lambda: os.environ.copy())

    @classmethod
    def discover(
        cls,
        runner: ProcessRunner,
        *,
        link_mode: str,
        timeout_seconds: int,
        environment: Mapping[str, str] | None = None,
    ) -> UvCommandClient:
        """Create a client for the uv executable visible in PATH."""
        return cls(
            executable=resolve_executable("uv"),
            runner=runner,
            link_mode=link_mode,
            heavy_timeout_seconds=timeout_seconds,
            environment=os.environ.copy() if environment is None else dict(environment),
        )

    @property
    def removed_environment_names(self) -> tuple[str, ...]:
        """Return control-variable names excluded from child commands."""
        _, removed = self._child_environment()
        return removed

    def version(self) -> CommandResult:
        """Return uv version information."""
        return self._diagnostic(["--version"])

    def available_torch_backends(self) -> tuple[str, ...]:
        """Return CPU and CUDA backend identifiers advertised by this uv build."""
        result = self._diagnostic(["pip", "install", "--help"])
        if result.returncode != 0 or "--torch-backend" not in result.stdout:
            return ()
        values: list[str] = []
        seen: set[str] = set()
        for match in _BACKEND_HELP_PATTERN.finditer(result.stdout):
            value = match.group(0)
            if value not in seen:
                seen.add(value)
                values.append(value)
        return tuple(values)

    def python_find(self, request: str, *, project_dir: Path) -> CommandResult:
        """Find a system interpreter without selecting an unrelated virtualenv."""
        return self._diagnostic(
            ["python", "find", "--system", request], cwd=project_dir
        )

    def python_install(self, request: str, *, project_dir: Path) -> CommandResult:
        """Install an interpreter satisfying a uv Python request."""
        return self._heavy(["python", "install", request], cwd=project_dir)

    def create_venv(self, path: Path, python: Path, *, cwd: Path) -> CommandResult:
        """Create an isolated candidate environment without project configuration."""
        return self._heavy(
            ["--no-config", "venv", str(path), "--python", str(python)], cwd=cwd
        )

    def install_candidate(
        self,
        venv: Path,
        requirements: Sequence[str],
        candidate: BackendCandidate,
        *,
        dry_run: bool = False,
    ) -> CommandResult:
        """Resolve or install selected requirements for one backend candidate."""
        arguments: list[str | Path] = [
            "--no-config",
            "pip",
            "install",
            "--python",
            str(venv),
            *requirements,
        ]
        if candidate.channel is Channel.NIGHTLY:
            arguments.extend(["--index", candidate.index_url, "--prerelease", "allow"])
        else:
            arguments.extend(["--torch-backend", candidate.value])
        if dry_run:
            arguments.append("--dry-run")
        return self._heavy(arguments)

    def install_numpy_lt2(self, venv: Path) -> CommandResult:
        """Constrain NumPy after a candidate demonstrates bridge incompatibility."""
        return self._heavy(
            ["--no-config", "pip", "install", "--python", str(venv), "numpy<2"]
        )

    def workspace_metadata(self, project_dir: Path) -> CommandResult:
        """Return read-only JSON metadata for the containing uv workspace."""
        return self._diagnostic(["workspace", "metadata", "--dry-run"], cwd=project_dir)

    def lock(self, project_dir: Path, python: Path) -> CommandResult:
        """Update the project or workspace lockfile with a concrete interpreter."""
        return self._heavy(
            ["lock", "--project", str(project_dir), "--python", str(python)],
            cwd=project_dir,
        )

    def check_lock(self, project_dir: Path) -> CommandResult:
        """Check lockfile freshness without modifying it."""
        return self._heavy(
            ["lock", "--project", str(project_dir), "--check"], cwd=project_dir
        )

    def sync(
        self,
        project_dir: Path,
        python: Path,
        *,
        package: str | None,
        extras: Sequence[str],
        groups: Sequence[str],
        check: bool = False,
        dry_run: bool = False,
    ) -> CommandResult:
        """Synchronize or check exactly the selected project package and scopes."""
        arguments: list[str | Path] = [
            "sync",
            "--project",
            str(project_dir),
            "--python",
            str(python),
            "--locked",
        ]
        if package:
            arguments.extend(["--package", package])
        arguments.extend(_scope_arguments(extras, groups))
        if check:
            arguments.append("--check")
            return self._heavy(arguments, cwd=project_dir)
        if dry_run:
            arguments.append("--dry-run")
        return self._heavy(arguments, cwd=project_dir)

    def run_project_python(
        self,
        project_dir: Path,
        python: Path,
        script_arguments: Sequence[str | Path],
        *,
        package: str | None,
        extras: Sequence[str],
        groups: Sequence[str],
        cuda_device: str | None,
    ) -> CommandResult:
        """Run a probe without allowing uv to mutate the target environment."""
        arguments: list[str | Path] = [
            "run",
            "--project",
            str(project_dir),
            "--python",
            str(python),
            "--locked",
            "--no-sync",
        ]
        if package:
            arguments.extend(["--package", package])
        arguments.extend(_scope_arguments(extras, groups))
        arguments.extend(["python", *script_arguments])
        overrides = {"CUDA_VISIBLE_DEVICES": cuda_device} if cuda_device else None
        return self._heavy(arguments, cwd=project_dir, overrides=overrides)

    def _diagnostic(
        self,
        arguments: Sequence[str | Path],
        *,
        cwd: Path | None = None,
        overrides: Mapping[str, str] | None = None,
    ) -> CommandResult:
        return self._run(
            arguments,
            cwd=cwd,
            timeout_seconds=self.diagnostic_timeout_seconds,
            overrides=overrides,
        )

    def _heavy(
        self,
        arguments: Sequence[str | Path],
        *,
        cwd: Path | None = None,
        overrides: Mapping[str, str] | None = None,
    ) -> CommandResult:
        return self._run(
            arguments,
            cwd=cwd,
            timeout_seconds=self.heavy_timeout_seconds,
            overrides=overrides,
        )

    def _run(
        self,
        arguments: Sequence[str | Path],
        *,
        cwd: Path | None,
        timeout_seconds: int,
        overrides: Mapping[str, str] | None = None,
    ) -> CommandResult:
        child_env, _ = self._child_environment(overrides)
        return self.runner.run(
            [self.executable, *arguments],
            cwd=cwd,
            env=child_env,
            timeout_seconds=timeout_seconds,
        )

    def _child_environment(
        self, overrides: Mapping[str, str] | None = None
    ) -> tuple[dict[str, str], tuple[str, ...]]:
        trusted = {
            "UV_LINK_MODE": self.link_mode,
            "UV_NO_PROGRESS": "1",
            "UV_COLOR": "never",
        }
        if overrides:
            trusted.update(overrides)
        return sanitized_environment(self.environment, overrides=trusted)


def _scope_arguments(extras: Sequence[str], groups: Sequence[str]) -> list[str | Path]:
    arguments: list[str | Path] = []
    for extra in extras:
        arguments.extend(["--extra", extra])
    for group in groups:
        arguments.extend(["--group", group])
    return arguments
