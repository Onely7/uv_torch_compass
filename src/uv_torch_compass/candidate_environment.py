"""Model the concrete interpreter environment used for candidate resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass

from packaging.version import InvalidVersion, Version

from uv_torch_compass.errors import ConfigurationError

_MARKER_VALUE = re.compile(r"[A-Za-z0-9_.-]+", re.ASCII)


@dataclass(frozen=True, slots=True)
class CandidateExecutionEnvironment:
    """Describe the one Linux runtime a disposable candidate must support."""

    python_version: str
    implementation_name: str
    sys_platform: str
    platform_machine: str

    def __post_init__(self) -> None:
        """Reject values that cannot form a safe, concrete marker environment."""
        try:
            parsed = Version(self.python_version)
        except InvalidVersion as exc:
            raise ConfigurationError(
                f"invalid candidate Python version {self.python_version!r}"
            ) from exc
        if len(parsed.release) < 2:
            raise ConfigurationError("candidate Python version requires a minor value")
        if self.sys_platform != "linux":
            raise ConfigurationError("candidate execution currently requires Linux")
        for label, value in (
            ("implementation", self.implementation_name),
            ("machine", self.platform_machine),
        ):
            if not value or not _MARKER_VALUE.fullmatch(value):
                raise ConfigurationError(f"invalid candidate {label} {value!r}")

    @property
    def python_minor(self) -> str:
        """Return the selected major and minor Python version."""
        release = Version(self.python_version).release
        return f"{release[0]}.{release[1]}"

    @property
    def requires_python(self) -> str:
        """Return a PEP 440 range limited to the selected Python minor."""
        release = Version(self.python_version).release
        return f">={release[0]}.{release[1]},<{release[0]}.{release[1] + 1}"

    @property
    def required_environment_marker(self) -> str:
        """Return the platform marker used for uv wheel availability checks."""
        return (
            f"sys_platform == '{self.sys_platform}' and "
            f"platform_machine == '{self.platform_machine}'"
        )

    @property
    def resolution_environment_marker(self) -> str:
        """Return the marker restricting uv resolution to this interpreter."""
        return (
            f"implementation_name == '{self.implementation_name}' and "
            f"python_version == '{self.python_minor}' and "
            f"{self.required_environment_marker}"
        )

    @property
    def platform_label(self) -> str:
        """Return a compact platform label for diagnostics."""
        return f"{self.sys_platform}-{self.platform_machine}"
