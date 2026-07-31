"""Define immutable diagnostics for candidate and framework failures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ResolutionFailureKind(str, Enum):
    """Classify a resolver or runtime failure at a stable public boundary."""

    NO_COMPATIBLE_DISTRIBUTION = "no-compatible-distribution"
    DEPENDENCY_CONFLICT = "dependency-conflict"
    WHEEL_UNAVAILABLE = "wheel-unavailable"
    BUILD_FAILURE = "build-failure"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    TIMEOUT = "timeout"
    RUNTIME_VALIDATION = "runtime-validation"
    UNKNOWN = "unknown"


class FrameworkFailureKind(str, Enum):
    """Classify a failure found while validating a framework integration."""

    CUDA_ABI = "framework-cuda-abi"
    API_INCOMPATIBILITY = "framework-api-incompatibility"
    BINARY_INCOMPATIBILITY = "framework-binary-incompatibility"
    IMPORT = "framework-import"
    NATIVE_EXTENSION = "framework-native-extension"
    PLATFORM = "framework-platform"
    METADATA = "framework-metadata"


class FrameworkProbeTrigger(str, Enum):
    """Record why a framework probe was included in a validation run."""

    AUTOMATIC = "automatic"
    EXPLICIT = "explicit"


class FrameworkCompatibilityEvidence(str, Enum):
    """Identify the evidence used to decide framework binary compatibility."""

    CATALOG = "catalog"
    ELF = "elf"
    METADATA = "metadata"
    RUNTIME = "runtime"
    UNKNOWN = "unknown"


class FrameworkCompatibilityStatus(str, Enum):
    """Describe whether one framework artifact can use a backend candidate."""

    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FailedPackage:
    """Identify a package constraint implicated by a candidate failure."""

    name: str
    version: str | None = None
    requirement: str | None = None


@dataclass(frozen=True, slots=True)
class FailedIndex:
    """Identify the package index implicated by a candidate failure."""

    name: str
    url: str


@dataclass(frozen=True, slots=True)
class ResolutionFailure:
    """Describe an actionable, redacted resolver or runtime failure."""

    kind: ResolutionFailureKind
    summary: str
    package: FailedPackage | None = None
    required_by: tuple[str, ...] = ()
    index: FailedIndex | None = None
    platform: str | None = None
    suggestions: tuple[str, ...] = ()
    dependency_paths: tuple[tuple[str, ...], ...] = ()
    available_wheel_platforms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TracebackFrame:
    """Describe one bounded traceback frame without exposing an absolute path."""

    module: str
    filename: str
    function: str
    line_number: int


@dataclass(frozen=True, slots=True)
class BoundedExceptionReport:
    """Contain a redacted exception and a bounded set of traceback frames."""

    exception_type: str
    message: str
    frames: tuple[TracebackFrame, ...] = ()
    missing_symbol: str | None = None
    missing_module: str | None = None
    consumer_package: str | None = None
    provider_package: str | None = None


@dataclass(frozen=True, slots=True)
class FrameworkPackageVersion:
    """Identify a framework-related package resolved for one candidate."""

    name: str
    version: str
    source_url: str = ""


@dataclass(frozen=True, slots=True)
class FrameworkBinaryRequirement:
    """Describe CUDA libraries required by one prebuilt framework artifact."""

    framework: str
    version: str
    required_cuda_variant: str | None = None
    required_cuda_major: int | None = None
    needed_libraries: tuple[str, ...] = ()
    evidence: FrameworkCompatibilityEvidence = FrameworkCompatibilityEvidence.UNKNOWN
    source_url: str = ""


@dataclass(frozen=True, slots=True)
class FrameworkCompatibilityDecision:
    """Explain whether a framework artifact matches a backend candidate."""

    status: FrameworkCompatibilityStatus
    candidate_backend: str
    summary: str
    requirement: FrameworkBinaryRequirement | None = None

    @property
    def allowed(self) -> bool:
        """Return whether validation may continue with this candidate."""
        return self.status is not FrameworkCompatibilityStatus.INCOMPATIBLE


@dataclass(frozen=True, slots=True)
class FrameworkFailure:
    """Describe a structured framework compatibility or validation failure."""

    kind: FrameworkFailureKind
    summary: str
    framework: str
    framework_version: str
    package: FailedPackage | None = None
    dependency_paths: tuple[tuple[str, ...], ...] = ()
    binary_requirement: FrameworkBinaryRequirement | None = None
    exception: BoundedExceptionReport | None = None
    packages: tuple[FrameworkPackageVersion, ...] = ()
    suggestions: tuple[str, ...] = ()
    backend_independent: bool = False


CandidateFailure = ResolutionFailure | FrameworkFailure
