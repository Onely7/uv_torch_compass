import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from uv_torch_compass.application import CompassApplication, _requested_cuda_device
from uv_torch_compass.command_runner import CommandResult, ProcessRunner
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
from uv_torch_compass.errors import CommandError, ExternalModificationError
from uv_torch_compass.reporting import CommandReporter
from uv_torch_compass.uv_commands import UvCommandClient


def _runtime_json(backend: str = "cpu") -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "backend": backend,
            "torch_version": "2.7.0",
            "torchvision_version": "not-installed",
            "torchaudio_version": "not-installed",
            "numpy_version": "2.2.0",
            "cuda_runtime": "none" if backend == "cpu" else "12.8",
            "gpu_name": "none" if backend == "cpu" else "Fake GPU",
            "cuda_test": "NOT_APPLICABLE" if backend == "cpu" else "PASS",
            "numpy_bridge_test": "PASS",
            "torchvision_test": "NOT_REQUESTED",
            "torchaudio_test": "NOT_REQUESTED",
        }
    )


class FakeRunner:
    def __init__(self, *, numpy_failure: bool = False) -> None:
        self.numpy_failure = numpy_failure
        self.probe_calls = 0

    def run(
        self, arguments, *, cwd=None, env=None, timeout_seconds=None
    ) -> CommandResult:
        del cwd, env, timeout_seconds
        rendered = [str(argument) for argument in arguments]
        if "platform.python_version" in rendered[-1]:
            return CommandResult(
                0,
                '{"version": "3.12.13", "implementation_name": "cpython", '
                '"platform_implementation": "CPython"}\n',
                "",
            )
        if rendered[0].endswith("/bin/python") and rendered[1].endswith(
            "runtime_probe.py"
        ):
            self.probe_calls += 1
            if self.numpy_failure and self.probe_calls == 1:
                return CommandResult(22, "", "NUMPY_BRIDGE_FAILED: incompatible\n")
            return CommandResult(0, _runtime_json() + "\n", "")
        return CommandResult(0, "", "")


class FakeUv:
    def __init__(
        self,
        project_dir: Path,
        *,
        fail_sync_once: bool = False,
        mutate_during_probe: Path | None = None,
    ) -> None:
        self.project_dir = project_dir
        self.fail_sync_once = fail_sync_once
        self.mutate_during_probe = mutate_during_probe
        self.sync_calls = 0
        self.numpy_installs = 0
        self.removed_environment_names: tuple[str, ...] = ()
        self.diagnostic_timeout_seconds = 30
        self.heavy_timeout_seconds = 1800

    def version(self) -> CommandResult:
        return CommandResult(0, "uv 0.11.28\n", "")

    def available_torch_backends(self) -> tuple[str, ...]:
        return ("auto", "cpu", "cu128")

    def workspace_metadata(self, project_dir: Path) -> CommandResult:
        return CommandResult(
            0,
            json.dumps(
                {
                    "workspace_root": str(project_dir),
                    "members": [{"name": "target", "path": str(project_dir)}],
                }
            ),
            "",
        )

    def python_find(self, request: str, *, project_dir: Path) -> CommandResult:
        del request, project_dir
        return CommandResult(0, f"{Path(sys.executable).resolve()}\n", "")

    def python_install(self, request: str, *, project_dir: Path) -> CommandResult:
        del request, project_dir
        return CommandResult(0, "", "")

    def create_venv(self, path: Path, python: Path, *, cwd: Path) -> CommandResult:
        del python, cwd
        (path / "bin").mkdir(parents=True)
        return CommandResult(0, "", "")

    def install_candidate(
        self, venv: Path, requirements, candidate, *, dry_run: bool = False
    ) -> CommandResult:
        del venv, requirements, candidate, dry_run
        return CommandResult(0, "installed", "")

    def install_numpy_lt2(self, venv: Path) -> CommandResult:
        del venv
        self.numpy_installs += 1
        return CommandResult(0, "installed", "")

    def lock(self, project_dir: Path, python: Path) -> CommandResult:
        del python
        (project_dir / "uv.lock").write_text("generated lock", encoding="utf-8")
        return CommandResult(0, "locked", "")

    def check_lock(self, project_dir: Path) -> CommandResult:
        return CommandResult(
            0 if (project_dir / "uv.lock").exists() else 1, "", "missing lock"
        )

    def sync(
        self,
        project_dir: Path,
        python: Path,
        *,
        package,
        extras,
        groups,
        check: bool = False,
    ) -> CommandResult:
        del project_dir, python, package, extras, groups, check
        self.sync_calls += 1
        if self.fail_sync_once and self.sync_calls == 1:
            return CommandResult(1, "", "sync failed")
        return CommandResult(0, "synced", "")

    def run_project_python(
        self,
        project_dir: Path,
        python: Path,
        script_arguments,
        *,
        package,
        extras,
        groups,
        cuda_device,
    ) -> CommandResult:
        del (
            project_dir,
            python,
            script_arguments,
            package,
            extras,
            groups,
            cuda_device,
        )
        if self.mutate_during_probe is not None:
            self.mutate_during_probe.write_text("external edit", encoding="utf-8")
            self.mutate_during_probe = None
        return CommandResult(0, _runtime_json() + "\n", "")


def _project(tmp_path: Path) -> Path:
    pyproject = tmp_path / "target project" / "pyproject.toml"
    pyproject.parent.mkdir()
    pyproject.write_text(
        "[project]\n"
        'name = "target"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.10,<3.15"\n'
        'dependencies = ["torch>=2.5"]\n',
        encoding="utf-8",
    )
    return pyproject


def _options(pyproject: Path, operation: Operation) -> RunOptions:
    return RunOptions(
        operation=operation,
        pyproject=pyproject,
        python_request="",
        requirement_overrides=(),
        backend=BackendRequest.parse("cpu"),
        channel=Channel.STABLE,
        cuda_compatibility=CompatibilityPolicy.STRICT,
        probe_profile=ProbeProfile.STANDARD,
        extras=(),
        groups=(),
        cuda_device=cast(GpuDevice | None, None),
        link_mode="copy",
        log_dir=pyproject.parent / "logs",
        timeout_seconds=1800,
        output_format=OutputFormat.JSON,
        report_file=None,
    )


def _run(
    monkeypatch,
    options: RunOptions,
    runner: FakeRunner,
    uv: FakeUv,
):
    monkeypatch.setattr("uv_torch_compass.application.platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "uv_torch_compass.application.NvidiaInspector.discover", lambda _runner: None
    )
    reporter = CommandReporter(options, "0.1.0")
    with reporter:
        result = CompassApplication(
            options,
            cast(ProcessRunner, runner),
            reporter,
            cast(UvCommandClient, uv),
        ).run()
    return result, reporter


def test_plan_verifies_candidate_without_changing_target(
    tmp_path: Path, monkeypatch
) -> None:
    pyproject = _project(tmp_path)
    original = pyproject.read_text(encoding="utf-8")

    result, _ = _run(
        monkeypatch,
        _options(pyproject, Operation.PLAN),
        FakeRunner(),
        FakeUv(pyproject.parent),
    )

    assert result.outcome.status == "planned"
    assert not result.outcome.applied
    assert "pytorch-cpu" in result.outcome.planned_diff
    assert pyproject.read_text(encoding="utf-8") == original
    assert not (pyproject.parent / "uv.lock").exists()


def test_apply_updates_locks_syncs_and_validates(tmp_path: Path, monkeypatch) -> None:
    pyproject = _project(tmp_path)
    runner = FakeRunner(numpy_failure=True)
    uv = FakeUv(pyproject.parent)

    result, _ = _run(monkeypatch, _options(pyproject, Operation.APPLY), runner, uv)

    assert result.outcome.applied
    assert result.outcome.status == "success_with_warnings"
    assert "pytorch-cpu" in pyproject.read_text(encoding="utf-8")
    assert (pyproject.parent / "uv.lock").is_file()
    assert uv.numpy_installs == 1
    assert all(path.is_file() for path in result.outcome.backups)


def test_sync_failure_restores_project_and_existing_lock(
    tmp_path: Path, monkeypatch
) -> None:
    pyproject = _project(tmp_path)
    original = pyproject.read_text(encoding="utf-8")
    lockfile = pyproject.parent / "uv.lock"
    lockfile.write_text("original lock", encoding="utf-8")
    uv = FakeUv(pyproject.parent, fail_sync_once=True)
    options = _options(pyproject, Operation.APPLY)
    monkeypatch.setattr("uv_torch_compass.application.platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "uv_torch_compass.application.NvidiaInspector.discover", lambda _runner: None
    )
    reporter = CommandReporter(options, "0.1.0")

    with reporter, pytest.raises(CommandError, match="uv sync failed"):
        CompassApplication(
            options,
            cast(ProcessRunner, FakeRunner()),
            reporter,
            cast(UvCommandClient, uv),
        ).run()

    assert pyproject.read_text(encoding="utf-8") == original
    assert lockfile.read_text(encoding="utf-8") == "original lock"
    assert uv.sync_calls == 2


def test_sync_failure_restores_original_absence_of_lockfile(
    tmp_path: Path, monkeypatch
) -> None:
    pyproject = _project(tmp_path)
    original = pyproject.read_text(encoding="utf-8")
    uv = FakeUv(pyproject.parent, fail_sync_once=True)
    options = _options(pyproject, Operation.APPLY)
    monkeypatch.setattr("uv_torch_compass.application.platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "uv_torch_compass.application.NvidiaInspector.discover", lambda _runner: None
    )
    reporter = CommandReporter(options, "0.1.0")

    with reporter, pytest.raises(CommandError, match="uv sync failed"):
        CompassApplication(
            options,
            cast(ProcessRunner, FakeRunner()),
            reporter,
            cast(UvCommandClient, uv),
        ).run()

    assert pyproject.read_text(encoding="utf-8") == original
    assert not (pyproject.parent / "uv.lock").exists()
    assert uv.sync_calls == 2


def test_check_validates_applied_state_without_changes(
    tmp_path: Path, monkeypatch
) -> None:
    pyproject = _project(tmp_path)
    apply_result, _ = _run(
        monkeypatch,
        _options(pyproject, Operation.APPLY),
        FakeRunner(),
        FakeUv(pyproject.parent),
    )
    assert apply_result.outcome.applied
    original = pyproject.read_bytes()
    original_lock = (pyproject.parent / "uv.lock").read_bytes()

    result, _ = _run(
        monkeypatch,
        _options(pyproject, Operation.CHECK),
        FakeRunner(),
        FakeUv(pyproject.parent),
    )

    assert result.outcome.status == "valid"
    assert pyproject.read_bytes() == original
    assert (pyproject.parent / "uv.lock").read_bytes() == original_lock


def test_commands_reject_non_linux_and_cuda_without_nvidia(
    tmp_path: Path, monkeypatch
) -> None:
    pyproject = _project(tmp_path)
    options = _options(pyproject, Operation.PLAN)
    reporter = CommandReporter(options, "0.1.0")
    application = CompassApplication(
        options,
        cast(ProcessRunner, FakeRunner()),
        reporter,
        cast(UvCommandClient, FakeUv(pyproject.parent)),
    )
    monkeypatch.setattr(
        "uv_torch_compass.application.platform.system", lambda: "Darwin"
    )
    with reporter, pytest.raises(CommandError, match="Linux"):
        application.run()

    cuda_options = replace(options, backend=BackendRequest.parse("cuda"))
    reporter = CommandReporter(cuda_options, "0.1.0")
    monkeypatch.setattr("uv_torch_compass.application.platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "uv_torch_compass.application.NvidiaInspector.discover", lambda _runner: None
    )
    with reporter, pytest.raises(CommandError, match="nvidia-smi"):
        CompassApplication(
            cuda_options,
            cast(ProcessRunner, FakeRunner()),
            reporter,
            cast(UvCommandClient, FakeUv(pyproject.parent)),
        ).run()


def test_check_rejects_missing_or_stale_lock(tmp_path: Path, monkeypatch) -> None:
    pyproject = _project(tmp_path)
    apply_result, _ = _run(
        monkeypatch,
        _options(pyproject, Operation.APPLY),
        FakeRunner(),
        FakeUv(pyproject.parent),
    )
    assert apply_result.outcome.applied
    (pyproject.parent / "uv.lock").unlink()
    options = _options(pyproject, Operation.CHECK)
    monkeypatch.setattr("uv_torch_compass.application.platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "uv_torch_compass.application.NvidiaInspector.discover", lambda _runner: None
    )
    reporter = CommandReporter(options, "0.1.0")

    with reporter, pytest.raises(CommandError, match="missing or stale"):
        CompassApplication(
            options,
            cast(ProcessRunner, FakeRunner()),
            reporter,
            cast(UvCommandClient, FakeUv(pyproject.parent)),
        ).run()


def test_check_invalidates_result_after_external_lock_edit(
    tmp_path: Path, monkeypatch
) -> None:
    pyproject = _project(tmp_path)
    apply_result, _ = _run(
        monkeypatch,
        _options(pyproject, Operation.APPLY),
        FakeRunner(),
        FakeUv(pyproject.parent),
    )
    assert apply_result.outcome.applied
    lockfile = pyproject.parent / "uv.lock"
    options = _options(pyproject, Operation.CHECK)
    monkeypatch.setattr("uv_torch_compass.application.platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "uv_torch_compass.application.NvidiaInspector.discover", lambda _runner: None
    )
    reporter = CommandReporter(options, "0.1.0")

    with (
        reporter,
        pytest.raises(ExternalModificationError, match="changed while check"),
    ):
        CompassApplication(
            options,
            cast(ProcessRunner, FakeRunner()),
            reporter,
            cast(
                UvCommandClient,
                FakeUv(pyproject.parent, mutate_during_probe=lockfile),
            ),
        ).run()


def test_existing_cuda_visibility_selects_first_device_or_hides_all(
    tmp_path: Path, monkeypatch
) -> None:
    options = _options(_project(tmp_path), Operation.PLAN)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-BBB,0")
    assert _requested_cuda_device(options) == ("GPU-BBB", False)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert _requested_cuda_device(options) == (None, True)

    explicit = replace(options, cuda_device=GpuDevice("1"))
    assert _requested_cuda_device(explicit) == ("1", False)
