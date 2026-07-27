import sys
from pathlib import Path

import pytest

from uv_torch_compass.command_runner import (
    SubprocessRunner,
    sanitized_environment,
)
from uv_torch_compass.errors import (
    CommandError,
    CommandTimeoutError,
    TerminationRequested,
)


def test_runner_requires_absolute_executable() -> None:
    with pytest.raises(CommandError):
        SubprocessRunner().run(["python", "--version"])


def test_runner_preserves_arguments_with_spaces(tmp_path: Path) -> None:
    result = SubprocessRunner().run(
        [Path(sys.executable).resolve(), "-c", "import sys; print(sys.argv[1])", "a b"],
        cwd=tmp_path,
        timeout_seconds=5,
    )

    assert result.returncode == 0
    assert result.stdout == "a b\n"


def test_runner_terminates_a_timed_out_process_group() -> None:
    with pytest.raises(CommandTimeoutError):
        SubprocessRunner().run(
            [Path(sys.executable).resolve(), "-c", "import time; time.sleep(10)"],
            timeout_seconds=0.01,
        )


def test_environment_filter_removes_control_values_but_keeps_connectivity() -> None:
    environment, removed = sanitized_environment(
        {
            "VIRTUAL_ENV": "/unsafe",
            "UV_PROJECT": "/other",
            "UV_TORCH_COMPASS_BACKEND": "cpu",
            "HTTPS_PROXY": "https://proxy.invalid",
            "UV_INDEX": "https://user:secret@example.invalid/simple",
        },
        overrides={"UV_LINK_MODE": "copy"},
    )

    assert set(removed) == {
        "UV_PROJECT",
        "UV_TORCH_COMPASS_BACKEND",
        "VIRTUAL_ENV",
    }
    assert environment["HTTPS_PROXY"] == "https://proxy.invalid"
    assert environment["UV_INDEX"].startswith("https://user:")
    assert environment["UV_LINK_MODE"] == "copy"


def test_project_environment_currently_survives_environment_sanitization() -> None:
    environment, removed = sanitized_environment(
        {"PATH": "/bin", "UV_PROJECT_ENVIRONMENT": "/project-environment"}
    )

    assert environment["UV_PROJECT_ENVIRONMENT"] == "/project-environment"
    assert "UV_PROJECT_ENVIRONMENT" not in removed


def test_runner_stops_process_group_after_termination_request(monkeypatch) -> None:
    stopped: list[object] = []

    class InterruptedProcess:
        def communicate(self, *, timeout=None):
            del timeout
            raise TerminationRequested("stop")

    process = InterruptedProcess()
    monkeypatch.setattr(
        "uv_torch_compass.command_runner.subprocess.Popen",
        lambda *_arguments, **_options: process,
    )
    monkeypatch.setattr(
        "uv_torch_compass.command_runner._stop_process_group", stopped.append
    )

    with pytest.raises(TerminationRequested):
        SubprocessRunner().run([Path("/bin/tool")], timeout_seconds=5)

    assert stopped == [process]
