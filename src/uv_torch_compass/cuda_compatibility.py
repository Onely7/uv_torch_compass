"""Model CUDA runtime and NVIDIA driver versions for compatibility decisions."""

from __future__ import annotations

from dataclasses import dataclass

from packaging.version import InvalidVersion, Version

from uv_torch_compass.errors import ConfigurationError


@dataclass(frozen=True, slots=True, order=True)
class CudaRuntimeVersion:
    """Represent a CUDA runtime major and minor version."""

    major: int
    minor: int

    @classmethod
    def from_backend(cls, backend: str) -> CudaRuntimeVersion:
        """Parse a concrete ``cuNNN`` backend identifier.

        Raises:
            ConfigurationError: If the identifier has no numeric CUDA version.
        """
        digits = backend.removeprefix("cu")
        if not backend.startswith("cu") or len(digits) < 2 or not digits.isdigit():
            raise ConfigurationError(f"invalid CUDA backend {backend!r}")
        return cls(int(digits[:-1]), int(digits[-1]))

    @classmethod
    def parse(cls, value: str) -> CudaRuntimeVersion:
        """Parse a dotted CUDA runtime version.

        Raises:
            ConfigurationError: If the value does not contain a major and minor.
        """
        parts = value.split(".", 2)
        if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
            raise ConfigurationError(f"invalid CUDA runtime version {value!r}")
        return cls(int(parts[0]), int(parts[1]))

    @property
    def backend(self) -> str:
        """Return the corresponding concrete backend identifier."""
        return f"cu{self.major}{self.minor}"

    def __str__(self) -> str:
        """Return the dotted runtime version."""
        return f"{self.major}.{self.minor}"


@dataclass(frozen=True, slots=True)
class NvidiaDriverVersion:
    """Represent one normalized NVIDIA driver version."""

    value: Version

    @classmethod
    def parse(cls, value: str) -> NvidiaDriverVersion:
        """Parse an NVIDIA driver version.

        Raises:
            ConfigurationError: If the reported version is invalid.
        """
        try:
            return cls(Version(value))
        except InvalidVersion as exc:
            raise ConfigurationError(
                f"invalid NVIDIA driver version {value!r}"
            ) from exc

    def __str__(self) -> str:
        """Return the normalized driver version."""
        return str(self.value)


def backend_within_reported_cuda_maximum(backend: str, reported_maximum: str) -> bool:
    """Return whether a backend does not exceed the nvidia-smi CUDA maximum.

    Malformed external maximum values remain permissive here to preserve the
    legacy behavior. The strict policy added later replaces this fail-open rule.
    """
    if not reported_maximum or not backend.startswith("cu"):
        return True
    try:
        backend_version = CudaRuntimeVersion.from_backend(backend)
        maximum_version = CudaRuntimeVersion.parse(reported_maximum)
    except ConfigurationError:
        return True
    return backend_version <= maximum_version
