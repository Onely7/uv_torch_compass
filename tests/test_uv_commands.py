from pathlib import Path

from uv_torch_compass.command_runner import CommandResult
from uv_torch_compass.uv_commands import UvCommandClient


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[
            tuple[list[str], Path | None, dict[str, str], float | None]
        ] = []
        self.help_output = (
            "--torch-backend <TORCH_BACKEND> "
            "[possible values: auto, cpu, cu130, cu128, rocm6.2, xpu]"
        )

    def run(
        self, arguments, *, cwd=None, env=None, timeout_seconds=None
    ) -> CommandResult:
        rendered = [str(argument) for argument in arguments]
        self.calls.append((rendered, cwd, dict(env or {}), timeout_seconds))
        if rendered[-3:] == ["pip", "install", "--help"]:
            return CommandResult(0, self.help_output, "")
        return CommandResult(0, "ok\n", "")


def _client(runner: RecordingRunner) -> UvCommandClient:
    return UvCommandClient(
        Path("/usr/bin/uv"),
        runner,
        "copy",
        1800,
        environment={
            "UV_PROJECT": "/unsafe",
            "UV_INDEX": "https://index.invalid/simple",
        },
    )


def test_client_discovers_backend_values_and_sanitizes_environment() -> None:
    runner = RecordingRunner()
    client = _client(runner)

    assert client.available_torch_backends() == ("auto", "cpu", "cu130", "cu128")
    assert client.removed_environment_names == ("UV_PROJECT",)
    arguments, _, environment, timeout = runner.calls[0]
    assert arguments == ["/usr/bin/uv", "pip", "install", "--help"]
    assert environment["UV_INDEX"] == "https://index.invalid/simple"
    assert "UV_PROJECT" not in environment
    assert timeout == 30


def test_client_builds_isolated_candidate_sync_command(tmp_path: Path) -> None:
    runner = RecordingRunner()
    client = _client(runner)
    venv = tmp_path / "venv"
    project = tmp_path / "candidate"
    python = Path("/usr/bin/python3")

    client.sync_candidate(venv, project, python)

    arguments, cwd, environment, timeout = runner.calls[-1]
    assert arguments == [
        "/usr/bin/uv",
        "sync",
        "--project",
        str(project),
        "--python",
        str(python),
        "--no-install-project",
        "--no-dev",
    ]
    assert cwd == project
    assert environment["UV_PROJECT_ENVIRONMENT"] == str(venv)
    assert timeout == 1800


def test_client_builds_workspace_and_project_commands(tmp_path: Path) -> None:
    runner = RecordingRunner()
    client = _client(runner)
    python = Path("/usr/bin/python3")

    client.version()
    client.python_find("3.12", project_dir=tmp_path)
    client.python_install("3.12", project_dir=tmp_path)
    client.install_numpy_lt2(tmp_path / "venv")
    client.workspace_metadata(tmp_path)
    client.lock(tmp_path, python)
    client.check_lock(tmp_path)
    client.sync(
        tmp_path,
        python,
        package="member",
        extras=("vision",),
        groups=("training",),
    )
    client.sync(
        tmp_path,
        python,
        package=None,
        extras=(),
        groups=(),
        check=True,
    )
    client.run_project_python(
        tmp_path,
        python,
        (Path("/probe.py"),),
        package="member",
        extras=("vision",),
        groups=("training",),
        cuda_device="GPU-123",
    )

    rendered_calls = [call[0] for call in runner.calls]
    sync = next(
        call for call in rendered_calls if call[1] == "sync" and "member" in call
    )
    assert "--locked" in sync
    assert sync[sync.index("--package") + 1] == "member"
    assert sync[sync.index("--extra") + 1] == "vision"
    run = next(call for call in rendered_calls if call[1] == "run")
    assert "--no-sync" in run
    assert runner.calls[-1][2]["CUDA_VISIBLE_DEVICES"] == "GPU-123"
    assert runner.calls[-1][3] == 1800

    project_scale_timeouts = [
        timeout
        for arguments, _, _, timeout in runner.calls
        if arguments[1] in {"lock", "sync", "run"}
    ]
    assert project_scale_timeouts
    assert set(project_scale_timeouts) == {1800}
