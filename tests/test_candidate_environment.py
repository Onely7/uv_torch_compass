import pytest

from uv_torch_compass.candidate_environment import CandidateExecutionEnvironment
from uv_torch_compass.errors import ConfigurationError


def test_environment_limits_resolution_to_selected_linux_interpreter() -> None:
    environment = CandidateExecutionEnvironment(
        "3.12.12",
        "cpython",
        "linux",
        "x86_64",
    )

    assert environment.python_minor == "3.12"
    assert environment.requires_python == ">=3.12,<3.13"
    assert environment.required_environment_marker == (
        "sys_platform == 'linux' and platform_machine == 'x86_64'"
    )
    assert environment.resolution_environment_marker == (
        "implementation_name == 'cpython' and python_version == '3.12' and "
        "sys_platform == 'linux' and platform_machine == 'x86_64'"
    )
    assert environment.platform_label == "linux-x86_64"


@pytest.mark.parametrize(
    ("version", "implementation", "system", "machine"),
    [
        ("not-a-version", "cpython", "linux", "x86_64"),
        ("3", "cpython", "linux", "x86_64"),
        ("3.12", "cpython", "darwin", "x86_64"),
        ("3.12", "cpython; unsafe", "linux", "x86_64"),
        ("3.12", "cpython", "linux", ""),
    ],
)
def test_environment_rejects_invalid_marker_values(
    version: str,
    implementation: str,
    system: str,
    machine: str,
) -> None:
    with pytest.raises(ConfigurationError):
        CandidateExecutionEnvironment(version, implementation, system, machine)
