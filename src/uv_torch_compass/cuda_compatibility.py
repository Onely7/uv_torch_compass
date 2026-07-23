"""Decide whether NVIDIA drivers can safely run CUDA-backed PyTorch wheels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from packaging.version import InvalidVersion, Version

from uv_torch_compass.errors import ConfigurationError


class CompatibilityPolicy(str, Enum):
    """Select strict driver support or explicit CUDA minor compatibility."""

    STRICT = "strict"
    MINOR = "minor"


class CompatibilityLevel(str, Enum):
    """Classify how a driver can execute a CUDA runtime."""

    STRICT = "strict"
    MINOR = "minor"
    UNSUPPORTED = "unsupported"


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
        """Parse a dotted CUDA runtime or component version.

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


@dataclass(frozen=True, slots=True)
class CompatibilityDecision:
    """Explain whether and how a driver may execute one CUDA backend."""

    level: CompatibilityLevel
    minimum_driver: str
    reason: str

    @property
    def allowed(self) -> bool:
        """Return whether the candidate may proceed to installation."""
        return self.level is not CompatibilityLevel.UNSUPPORTED


@dataclass(frozen=True, slots=True)
class _CatalogEntry:
    full_support_driver: Version
    minor_family_driver: Version


# These conservative boundaries are taken from NVIDIA CUDA Toolkit release
# notes. Keeping them local makes an unknown future runtime fail closed instead
# of silently changing policy through a network response.
_CATALOG = {
    "cu118": _CatalogEntry(Version("520.61.05"), Version("450.80.02")),
    "cu120": _CatalogEntry(Version("525.60.13"), Version("525.60.13")),
    "cu121": _CatalogEntry(Version("530.30.02"), Version("525.60.13")),
    "cu122": _CatalogEntry(Version("535.104.05"), Version("525.60.13")),
    "cu123": _CatalogEntry(Version("545.23.08"), Version("525.60.13")),
    "cu124": _CatalogEntry(Version("550.54.15"), Version("525.60.13")),
    "cu125": _CatalogEntry(Version("555.42.06"), Version("525.60.13")),
    "cu126": _CatalogEntry(Version("560.35.05"), Version("525.60.13")),
    "cu128": _CatalogEntry(Version("570.124.06"), Version("525.60.13")),
    "cu129": _CatalogEntry(Version("575.57.08"), Version("525.60.13")),
    "cu130": _CatalogEntry(Version("580.82.07"), Version("580.65.06")),
}


def known_cuda_backends() -> tuple[str, ...]:
    """Return catalogued CUDA backends in newest-first order."""
    return tuple(
        sorted(
            _CATALOG,
            key=lambda value: CudaRuntimeVersion.from_backend(value),
            reverse=True,
        )
    )


def decide_compatibility(
    backend: str,
    *,
    driver_version: str,
    reported_cuda_maximum: str,
    policy: CompatibilityPolicy,
) -> CompatibilityDecision:
    """Classify one concrete backend against an NVIDIA driver.

    Args:
        backend: Official PyTorch CUDA backend identifier.
        driver_version: Version returned for the selected GPU by ``nvidia-smi``.
        reported_cuda_maximum: CUDA maximum printed by ``nvidia-smi``.
        policy: Whether minor-version compatibility is explicitly allowed.

    Returns:
        A decision containing the classification and user-facing reason.
    """
    entry = _CATALOG.get(backend)
    if entry is None:
        return CompatibilityDecision(
            CompatibilityLevel.UNSUPPORTED,
            "",
            f"{backend} is not present in the bundled compatibility catalog",
        )
    try:
        runtime = CudaRuntimeVersion.from_backend(backend)
        maximum = CudaRuntimeVersion.parse(reported_cuda_maximum)
        driver = NvidiaDriverVersion.parse(driver_version)
    except ConfigurationError as exc:
        return CompatibilityDecision(
            CompatibilityLevel.UNSUPPORTED,
            str(entry.full_support_driver),
            str(exc),
        )

    if runtime <= maximum and driver.value >= entry.full_support_driver:
        return CompatibilityDecision(
            CompatibilityLevel.STRICT,
            str(entry.full_support_driver),
            (
                f"{backend} is fully supported by driver {driver}; "
                f"nvidia-smi reports CUDA {maximum}"
            ),
        )

    if policy is CompatibilityPolicy.STRICT:
        return CompatibilityDecision(
            CompatibilityLevel.UNSUPPORTED,
            str(entry.full_support_driver),
            (
                f"{backend} requires driver {entry.full_support_driver} or newer "
                f"for strict compatibility; found {driver} with CUDA {maximum}"
            ),
        )

    next_major_driver = Version("525") if runtime.major == 11 else Version("580")
    same_family = runtime.major == maximum.major
    below_next_major = runtime.major >= 13 or driver.value < next_major_driver
    if same_family and below_next_major and driver.value >= entry.minor_family_driver:
        return CompatibilityDecision(
            CompatibilityLevel.MINOR,
            str(entry.minor_family_driver),
            (
                f"{backend} uses CUDA minor-version compatibility with driver "
                f"{driver}; features requiring a newer driver may fail"
            ),
        )

    return CompatibilityDecision(
        CompatibilityLevel.UNSUPPORTED,
        str(entry.minor_family_driver),
        (
            f"{backend} is outside the CUDA family supported by driver {driver} "
            f"with CUDA {maximum}"
        ),
    )


def validate_runtime_identity(
    backend: str, *, cuda_runtime: str, runtime_component: str
) -> None:
    """Ensure the installed runtime and component match the selected backend.

    Raises:
        ConfigurationError: If an unknown or mismatched runtime was installed.
    """
    if backend not in _CATALOG:
        raise ConfigurationError(
            f"{backend} is not present in the bundled compatibility catalog"
        )
    runtime = CudaRuntimeVersion.parse(cuda_runtime)
    component = CudaRuntimeVersion.parse(runtime_component)
    expected = CudaRuntimeVersion.from_backend(backend)
    if runtime != expected or component != expected:
        raise ConfigurationError(
            f"{backend} resolved CUDA runtime {runtime} and component {component}"
        )
