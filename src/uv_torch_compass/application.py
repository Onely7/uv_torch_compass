"""Orchestrate inspect, resolve, verify, apply, and restore phases."""

from __future__ import annotations

import difflib
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path

from uv_torch_compass.backend_selection import build_candidate_plan
from uv_torch_compass.candidate_probe import CandidateProbeService, ProbeOutcome
from uv_torch_compass.command_runner import CommandResult, ProcessRunner
from uv_torch_compass.domain import (
    BackendKind,
    CommandOutcome,
    GpuDevice,
    Operation,
    ProjectRequirements,
    RunOptions,
    RuntimeReport,
)
from uv_torch_compass.errors import (
    CommandError,
    ExternalModificationError,
    ProjectUpdateError,
)
from uv_torch_compass.nvidia import NvidiaInspector, NvidiaSnapshot
from uv_torch_compass.project_metadata import (
    read_configured_backend,
    read_project_requirements,
    render_project_configuration,
)
from uv_torch_compass.python_selection import PythonSelector, ResolvedPython
from uv_torch_compass.reporting import CommandReporter
from uv_torch_compass.safe_transaction import (
    FileSnapshot,
    SafeProjectTransaction,
    WorkspaceAdvisoryLock,
)
from uv_torch_compass.uv_commands import UvCommandClient
from uv_torch_compass.workspace import WorkspaceContext, resolve_workspace


@dataclass(frozen=True, slots=True)
class ApplicationResult:
    """Return a command outcome together with its resolved workspace."""

    outcome: CommandOutcome
    workspace: WorkspaceContext


@dataclass(frozen=True, slots=True)
class _TargetState:
    pyproject: FileSnapshot
    lockfile: FileSnapshot

    @classmethod
    def capture(cls, workspace: WorkspaceContext) -> _TargetState:
        return cls(
            FileSnapshot.capture(workspace.project_dir / "pyproject.toml"),
            FileSnapshot.capture(workspace.lockfile),
        )

    def require_unchanged(self, operation: Operation) -> None:
        for expected in (self.pyproject, self.lockfile):
            current = FileSnapshot.capture(expected.path)
            if (current.existed, current.digest) != (
                expected.existed,
                expected.digest,
            ):
                raise ExternalModificationError(
                    f"{expected.path} changed while {operation.value} was running"
                )


@dataclass(slots=True)
class CompassApplication:
    """Run one command while keeping domain policy separate from infrastructure."""

    options: RunOptions
    runner: ProcessRunner
    reporter: CommandReporter
    uv: UvCommandClient

    def run(self) -> ApplicationResult:
        """Dispatch the selected side-effect contract.

        Raises:
            CompassError: If inspection, verification, mutation, or recovery fails.
        """
        self._preflight()
        self.reporter.phase("inspect", "reading the target project and workspace")
        workspace = resolve_workspace(self.options.pyproject, self.uv)
        initial_state = _TargetState.capture(workspace)
        requirements = read_project_requirements(
            self.options.pyproject,
            extras=self.options.extras,
            groups=self.options.groups,
            overrides=self.options.requirement_overrides,
        )
        resolved_python = PythonSelector(self.uv, self.runner).resolve(
            explicit_request=self.options.python_request,
            requirements=requirements,
            project_dir=workspace.project_dir,
        )
        for warning in resolved_python.warnings:
            self.reporter.warn(warning)
        requirements = requirements.for_interpreter(
            resolved_python.version,
            resolved_python.implementation_name,
            resolved_python.platform_implementation,
        )
        self.reporter.info(
            f"using Python {resolved_python.version} at {resolved_python.executable}"
        )
        if self.uv.removed_environment_names:
            self.reporter.info(
                "ignored control environment variables: "
                + ", ".join(self.uv.removed_environment_names)
            )

        if self.options.operation is Operation.CHECK:
            outcome = self._check(
                workspace, requirements, resolved_python, initial_state
            )
        else:
            outcome = self._plan_or_apply(
                workspace, requirements, resolved_python, initial_state
            )
        return ApplicationResult(outcome, workspace)

    def _preflight(self) -> None:
        if platform.system() != "Linux":
            raise CommandError("apply, plan, and check currently support Linux only")
        version = self.uv.version()
        _require_success(version, "failed to run uv --version")
        if not self.uv.available_torch_backends():
            raise CommandError(
                "this uv version does not support PyTorch backend selection; update uv"
            )
        self.reporter.info(version.stdout.strip())

    def _plan_or_apply(
        self,
        workspace: WorkspaceContext,
        requirements: ProjectRequirements,
        python: ResolvedPython,
        initial_state: _TargetState,
    ) -> CommandOutcome:
        self.reporter.phase("resolve", "building backend candidates")
        nvidia, gpu_warnings = self._inspect_gpu(
            require_cuda=self.options.backend.kind
            in {BackendKind.CUDA, BackendKind.CONCRETE}
        )
        for warning in gpu_warnings:
            self.reporter.warn(warning)
        advertised = self.uv.available_torch_backends()
        candidate_plan = build_candidate_plan(
            self.options.backend,
            channel=self.options.channel,
            advertised_backends=advertised,
            nvidia=nvidia,
        )
        for warning in candidate_plan.warnings:
            self.reporter.warn(warning)
        gpu_selector = nvidia.selected.uuid if nvidia is not None else None

        self.reporter.phase("verify", "testing candidates in isolated environments")
        with tempfile.TemporaryDirectory(prefix="uv-torch-compass-") as temporary:
            probe = CandidateProbeService(
                uv=self.uv,
                runner=self.runner,
                reporter=self.reporter,
                temporary_root=Path(temporary),
                project_python=python.executable,
                requirements=requirements,
                runtime_probe=Path(__file__).with_name("runtime_probe.py").resolve(),
                cuda_device=gpu_selector,
            )
            verified = probe.find_working_candidate(candidate_plan.candidates)

        initial_state.require_unchanged(self.options.operation)
        try:
            original = initial_state.pyproject.content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProjectUpdateError(
                f"{self.options.pyproject} is not valid UTF-8"
            ) from exc
        updated, changes = render_project_configuration(
            self.options.pyproject,
            requirements=requirements,
            overrides=self.options.requirement_overrides,
            backend=verified.runtime.backend,
            numpy_lt2_required=verified.numpy_lt2_required,
        )
        planned_diff = _unified_diff(self.options.pyproject, original, updated)
        metadata = _metadata(python, nvidia)
        warnings = (*gpu_warnings, *candidate_plan.warnings)
        if self.options.operation is Operation.PLAN:
            initial_state.require_unchanged(self.options.operation)
            return CommandOutcome(
                status="planned",
                applied=False,
                runtime=verified.runtime,
                attempts=verified.attempts,
                changes=changes,
                warnings=warnings,
                planned_diff=planned_diff,
                metadata=metadata,
            )
        return self._apply(
            workspace,
            requirements,
            python,
            verified,
            updated,
            changes,
            original,
            gpu_selector,
            warnings,
            metadata,
        )

    def _apply(
        self,
        workspace: WorkspaceContext,
        requirements: ProjectRequirements,
        python: ResolvedPython,
        verified: ProbeOutcome,
        updated: str,
        changes: tuple[str, ...],
        original: str,
        gpu_selector: str | None,
        warnings: tuple[str, ...],
        metadata: dict[str, object],
    ) -> CommandOutcome:
        self.reporter.phase("apply", "updating and synchronizing the target project")
        lock_path = workspace.workspace_root / ".uv-torch-compass.lock"
        transaction: SafeProjectTransaction | None = None
        with WorkspaceAdvisoryLock(lock_path):
            if self.options.pyproject.read_text(encoding="utf-8") != original:
                raise ExternalModificationError(
                    f"{self.options.pyproject} changed before the transaction started"
                )
            try:
                transaction = SafeProjectTransaction.create(
                    self.options.pyproject, workspace.lockfile
                )
                transaction.write_pyproject(updated)
                try:
                    lock_result = self.uv.lock(workspace.project_dir, python.executable)
                finally:
                    # uv may update the lockfile before timing out or receiving a signal.
                    # Record that state so rollback can distinguish it from editor changes.
                    transaction.accept_lockfile_change()
                _require_success(lock_result, "uv lock failed", self.reporter)
                sync_result = self.uv.sync(
                    workspace.project_dir,
                    python.executable,
                    package=workspace.package,
                    extras=self.options.extras,
                    groups=self.options.groups,
                )
                _require_success(sync_result, "uv sync failed", self.reporter)
                final = self._run_project_probe(
                    workspace,
                    requirements,
                    python,
                    verified.runtime.backend,
                    gpu_selector,
                )
            except BaseException as exc:
                if transaction is not None:
                    self._restore(workspace, python, transaction, exc)
                raise
        status = (
            "success_with_warnings" if self.reporter.warning_messages else "success"
        )
        return CommandOutcome(
            status=status,
            applied=True,
            runtime=final,
            attempts=verified.attempts,
            changes=changes,
            backups=transaction.backups if transaction is not None else (),
            warnings=warnings,
            metadata=metadata,
        )

    def _check(
        self,
        workspace: WorkspaceContext,
        requirements: ProjectRequirements,
        python: ResolvedPython,
        initial_state: _TargetState,
    ) -> CommandOutcome:
        self.reporter.phase(
            "verify", "checking current metadata, lock, and environment"
        )
        packages = {
            item.package
            for item in requirements.selected
            if item.package in {"torch", "torchvision", "torchaudio"}
        }
        configured = read_configured_backend(self.options.pyproject, packages)
        nvidia, gpu_warnings = self._inspect_gpu(require_cuda=configured.is_cuda)
        for warning in gpu_warnings:
            self.reporter.warn(warning)
        gpu_selector = nvidia.selected.uuid if nvidia is not None else None
        _require_success(
            self.uv.check_lock(workspace.project_dir),
            "uv lock is missing or stale",
            self.reporter,
        )
        _require_success(
            self.uv.sync(
                workspace.project_dir,
                python.executable,
                package=workspace.package,
                extras=self.options.extras,
                groups=self.options.groups,
                check=True,
            ),
            "project environment is not synchronized",
            self.reporter,
        )
        runtime = self._run_project_probe(
            workspace, requirements, python, configured, gpu_selector
        )
        initial_state.require_unchanged(self.options.operation)
        return CommandOutcome(
            status="valid",
            applied=False,
            runtime=runtime,
            warnings=gpu_warnings,
            metadata=_metadata(python, nvidia),
        )

    def _inspect_gpu(
        self, *, require_cuda: bool
    ) -> tuple[NvidiaSnapshot | None, tuple[str, ...]]:
        requested_device, devices_hidden = _requested_cuda_device(self.options)
        if devices_hidden:
            if require_cuda:
                raise CommandError(
                    "CUDA_VISIBLE_DEVICES hides all GPUs but CUDA is required"
                )
            return None, ("CUDA_VISIBLE_DEVICES hides all GPUs; CPU remains available",)
        inspector = NvidiaInspector.discover(self.runner)
        if inspector is None:
            if require_cuda:
                raise CommandError("nvidia-smi was not found but CUDA is required")
            return None, ("nvidia-smi was not found; CPU remains available",)
        try:
            snapshot = inspector.inspect(requested_device)
        except CommandError as exc:
            if require_cuda:
                raise
            return None, (f"NVIDIA inspection failed; CPU remains available: {exc}",)
        self.reporter.info(
            f"selected NVIDIA device {snapshot.selected.index}: {snapshot.selected.name}"
        )
        return snapshot, ()

    def _run_project_probe(
        self,
        workspace: WorkspaceContext,
        requirements: ProjectRequirements,
        python: ResolvedPython,
        expected_backend,
        gpu_selector: str | None,
    ) -> RuntimeReport:
        arguments: list[str | Path] = [
            Path(__file__).with_name("runtime_probe.py").resolve(),
            "--expected-backend",
            expected_backend.value,
        ]
        if requirements.has_package("torchvision"):
            arguments.append("--validate-torchvision")
        if requirements.has_package("torchaudio"):
            arguments.append("--validate-torchaudio")
        result = self.uv.run_project_python(
            workspace.project_dir,
            python.executable,
            arguments,
            package=workspace.package,
            extras=self.options.extras,
            groups=self.options.groups,
            cuda_device=gpu_selector,
        )
        _require_success(result, "final runtime validation failed", self.reporter)
        report = RuntimeReport.from_output(
            result.stdout, channel=expected_backend.channel
        )
        report.validate_requirements(requirements)
        if report.backend.value != expected_backend.value:
            raise CommandError(
                f"final runtime reported {report.backend.value}, expected "
                f"{expected_backend.value}"
            )
        return report

    def _restore(
        self,
        workspace: WorkspaceContext,
        python: ResolvedPython,
        transaction: SafeProjectTransaction,
        original_error: BaseException,
    ) -> None:
        self.reporter.phase("restore", "restoring project files after failure")
        try:
            transaction.restore()
            if transaction.lockfile_snapshot.existed:
                recovery = self.uv.sync(
                    workspace.project_dir,
                    python.executable,
                    package=workspace.package,
                    extras=self.options.extras,
                    groups=self.options.groups,
                )
                if recovery.returncode != 0:
                    self.reporter.warn(
                        "project files were restored, but environment recovery failed"
                    )
                return

            recovery_lock = self.uv.lock(workspace.project_dir, python.executable)
            transaction.accept_lockfile_change()
            if recovery_lock.returncode == 0:
                recovery_sync = self.uv.sync(
                    workspace.project_dir,
                    python.executable,
                    package=workspace.package,
                    extras=self.options.extras,
                    groups=self.options.groups,
                )
                if recovery_sync.returncode != 0:
                    self.reporter.warn(
                        "files were restored, but the original environment was not recovered"
                    )
            else:
                self.reporter.warn("could not create a temporary recovery lockfile")
            transaction.restore()
        except ProjectUpdateError as restore_error:
            raise ProjectUpdateError(
                f"operation failed and safe restoration also failed: {restore_error}"
            ) from original_error


def _require_success(
    result: CommandResult,
    message: str,
    reporter: CommandReporter | None = None,
) -> None:
    if reporter is not None:
        reporter.detail(result.stdout + result.stderr)
    if result.returncode == 0:
        return
    diagnostic = next(
        (
            line.strip()
            for line in reversed((result.stderr or result.stdout).splitlines())
            if line.strip()
        ),
        "",
    )
    raise CommandError(f"{message}: {diagnostic}" if diagnostic else message)


def _unified_diff(path: Path, original: str, updated: str) -> str:
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
            n=0,
        )
    )


def _metadata(
    python: ResolvedPython, nvidia: NvidiaSnapshot | None
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "python": {
            "request": python.request,
            "version": python.version,
            "implementation_name": python.implementation_name,
            "implementation": python.platform_implementation,
            "executable": str(python.executable),
        }
    }
    if nvidia is not None:
        metadata["gpu"] = {
            "index": nvidia.selected.index,
            "uuid": nvidia.selected.uuid,
            "name": nvidia.selected.name,
            "driver": nvidia.selected.driver_version,
            "cuda_max": nvidia.driver_cuda_max,
        }
    return metadata


def _requested_cuda_device(options: RunOptions) -> tuple[str | None, bool]:
    if options.cuda_device is not None:
        return options.cuda_device.value, False
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None, False
    first = raw.split(",", 1)[0].strip()
    if not first or first == "-1":
        return None, True
    return GpuDevice(first).value, False
