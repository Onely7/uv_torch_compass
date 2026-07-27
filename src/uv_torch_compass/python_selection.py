"""Resolve one concrete Python interpreter for project and probe operations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from uv_torch_compass.command_runner import ProcessRunner, sanitized_environment
from uv_torch_compass.domain import ProjectRequirements, has_upper_bound
from uv_torch_compass.errors import CommandError, ConfigurationError
from uv_torch_compass.uv_commands import UvCommandClient


@dataclass(frozen=True, slots=True)
class ResolvedPython:
    """Describe the interpreter selected for one target project."""

    request: str
    executable: Path
    version: str
    implementation_name: str
    platform_implementation: str
    sys_platform: str
    platform_machine: str
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PythonSelector:
    """Resolve uv Python requests and enforce the project's version contract."""

    uv: UvCommandClient
    runner: ProcessRunner

    def resolve(
        self,
        *,
        explicit_request: str,
        requirements: ProjectRequirements,
        project_dir: Path,
    ) -> ResolvedPython:
        """Resolve and validate the highest-priority usable Python request.

        Raises:
            CommandError: If uv cannot resolve or install a usable interpreter.
            ConfigurationError: If the resolved version violates requires-python.
        """
        warnings: list[str] = []
        requests: list[str] = []
        if explicit_request:
            requests.append(explicit_request)
        elif requirements.python_file_value:
            requests.extend(
                [requirements.python_file_value, requirements.requires_python]
            )
        else:
            requests.append(requirements.requires_python)

        for position, request in enumerate(dict.fromkeys(requests)):
            executable = self._resolve_request(request, project_dir)
            (
                version,
                implementation_name,
                platform_implementation,
                sys_platform,
                platform_machine,
            ) = self._inspect_version(executable)
            if Version(version) in SpecifierSet(requirements.requires_python):
                if not has_upper_bound(requirements.requires_python):
                    warnings.append(
                        f"requires-python {requirements.requires_python!r} has no upper "
                        f"bound; only Python {version} was verified"
                    )
                return ResolvedPython(
                    request,
                    executable,
                    version,
                    implementation_name,
                    platform_implementation,
                    sys_platform,
                    platform_machine,
                    tuple(warnings),
                )
            if explicit_request or position == len(requests) - 1:
                raise ConfigurationError(
                    f"Python {version} does not satisfy "
                    f"requires-python {requirements.requires_python}"
                )
            warnings.append(
                f".python-version request {request!r} resolves to Python {version}, "
                "which does not satisfy requires-python; using the project specifier"
            )
        raise CommandError("no Python request could be resolved")

    def _resolve_request(self, request: str, project_dir: Path) -> Path:
        found = self.uv.python_find(request, project_dir=project_dir)
        if found.returncode != 0 or not found.stdout.strip():
            installed = self.uv.python_install(request, project_dir=project_dir)
            if installed.returncode != 0:
                raise CommandError(
                    _command_failure(
                        f"could not install Python {request!r}", installed.stderr
                    )
                )
            found = self.uv.python_find(request, project_dir=project_dir)
        if found.returncode != 0:
            raise CommandError(
                _command_failure(f"could not resolve Python {request!r}", found.stderr)
            )
        path_text = next(
            (line.strip() for line in found.stdout.splitlines() if line.strip()), ""
        )
        if not path_text:
            raise CommandError(f"uv returned no Python path for {request!r}")
        path = Path(path_text).expanduser().resolve()
        if not path.is_file():
            raise CommandError(f"resolved Python path does not exist: {path}")
        return path

    def _inspect_version(
        self,
        executable: Path,
    ) -> tuple[str, str, str, str, str]:
        environment, _ = sanitized_environment(os.environ)
        result = self.runner.run(
            [
                executable,
                "-c",
                "import json, platform, sys; "
                "print(json.dumps({'version': platform.python_version(), "
                "'implementation_name': sys.implementation.name, "
                "'platform_implementation': platform.python_implementation(), "
                "'sys_platform': sys.platform, "
                "'platform_machine': platform.machine()}))",
            ],
            env=environment,
            timeout_seconds=30,
        )
        if result.returncode != 0:
            raise CommandError("failed to inspect the resolved Python interpreter")
        try:
            value = json.loads(result.stdout)
            version = value["version"]
            implementation_name = value["implementation_name"]
            platform_implementation = value["platform_implementation"]
            sys_platform = value["sys_platform"]
            platform_machine = value["platform_machine"]
            if not all(
                isinstance(item, str) and item
                for item in (
                    version,
                    implementation_name,
                    platform_implementation,
                    sys_platform,
                    platform_machine,
                )
            ):
                raise TypeError
            Version(version)
        except (json.JSONDecodeError, KeyError, TypeError, InvalidVersion) as exc:
            raise CommandError("resolved Python returned an invalid version") from exc
        return (
            version,
            implementation_name,
            platform_implementation,
            sys_platform,
            platform_machine,
        )


def _command_failure(message: str, output: str) -> str:
    diagnostic = next(
        (line.strip() for line in reversed(output.splitlines()) if line.strip()), ""
    )
    return f"{message}: {diagnostic}" if diagnostic else message
