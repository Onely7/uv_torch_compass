import json
import sys
from pathlib import Path
from typing import cast

import pytest

from uv_torch_compass.command_runner import CommandResult, ProcessRunner
from uv_torch_compass.domain import ProjectRequirements, Scope, ScopedRequirement
from uv_torch_compass.errors import ConfigurationError
from uv_torch_compass.python_selection import PythonSelector
from uv_torch_compass.uv_commands import UvCommandClient


class FakeUv:
    def __init__(self) -> None:
        self.requests: list[str] = []
        self.installs = 0

    def python_find(self, request: str, *, project_dir: Path) -> CommandResult:
        del project_dir
        self.requests.append(request)
        return CommandResult(0, f"{Path(sys.executable).resolve()}\n", "")

    def python_install(self, request: str, *, project_dir: Path) -> CommandResult:
        del request, project_dir
        self.installs += 1
        return CommandResult(0, "", "")


class VersionRunner:
    def __init__(self, versions: list[str]) -> None:
        self.versions = versions

    def run(self, arguments, *, cwd=None, env=None, timeout_seconds=None):
        del arguments, cwd, env, timeout_seconds
        return CommandResult(
            0,
            json.dumps(
                {
                    "version": self.versions.pop(0),
                    "implementation_name": "cpython",
                    "platform_implementation": "CPython",
                    "sys_platform": "linux",
                    "platform_machine": "x86_64",
                }
            ),
            "",
        )


def _requirements(python_file: str, specifier: str) -> ProjectRequirements:
    scope = Scope("base")
    requirement = ScopedRequirement(scope, "torch")
    return ProjectRequirements(
        specifier,
        python_file,
        (requirement,),
        (requirement,),
        (scope,),
    )


def test_python_file_request_falls_back_when_resolved_version_is_incompatible(
    tmp_path: Path,
) -> None:
    uv = FakeUv()
    selector = PythonSelector(
        cast(UvCommandClient, uv),
        cast(ProcessRunner, VersionRunner(["3.9.20", "3.12.13"])),
    )

    selected = selector.resolve(
        explicit_request="",
        requirements=_requirements("cpython3.9", ">=3.10,<3.15"),
        project_dir=tmp_path,
    )

    assert uv.requests == ["cpython3.9", ">=3.10,<3.15"]
    assert selected.version == "3.12.13"
    assert selected.implementation_name == "cpython"
    assert selected.sys_platform == "linux"
    assert selected.platform_machine == "x86_64"
    assert ".python-version" in selected.warnings[0]


def test_explicit_incompatible_python_is_rejected(tmp_path: Path) -> None:
    selector = PythonSelector(
        cast(UvCommandClient, FakeUv()),
        cast(ProcessRunner, VersionRunner(["3.9.20"])),
    )

    with pytest.raises(ConfigurationError, match="does not satisfy"):
        selector.resolve(
            explicit_request="/custom/python",
            requirements=_requirements("", ">=3.10"),
            project_dir=tmp_path,
        )
