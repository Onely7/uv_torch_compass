"""Describe the concrete Linux platform that uv must be able to install."""

from __future__ import annotations

import platform
from dataclasses import dataclass

from uv_torch_compass.errors import ConfigurationError

_MACHINES = {
    "amd64": "x86_64",
    "x86_64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
}


@dataclass(frozen=True, slots=True)
class RequiredEnvironment:
    """Represent a marker that must have installable distributions."""

    marker: str

    @classmethod
    def current_linux(cls) -> RequiredEnvironment:
        """Return the marker for the running Linux machine.

        Raises:
            ConfigurationError: If the machine architecture is unsupported.
        """
        machine = _MACHINES.get(platform.machine().lower())
        if machine is None:
            raise ConfigurationError(
                f"unsupported Linux machine architecture {platform.machine()!r}"
            )
        return cls(f"sys_platform == 'linux' and platform_machine == '{machine}'")
