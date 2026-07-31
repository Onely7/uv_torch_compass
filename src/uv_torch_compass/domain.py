"""Domain values for command selection, probing, and result reporting."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from uv_torch_compass.candidate_failures import (
    CandidateFailure,
    FailedIndex,
    FailedPackage,
    FrameworkCompatibilityDecision,
    ResolutionFailure,
    ResolutionFailureKind,
)
from uv_torch_compass.cuda_compatibility import (
    CompatibilityDecision,
    CompatibilityPolicy,
)
from uv_torch_compass.errors import ConfigurationError, ProbeError

if TYPE_CHECKING:
    from uv_torch_compass.candidate_resolution import CandidateResolution

__all__ = [
    "FailedIndex",
    "FailedPackage",
    "ResolutionFailure",
    "ResolutionFailureKind",
]

_CUDA_BACKEND_PATTERN = re.compile(r"cu[0-9]{2,3}", re.ASCII)
_GPU_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9_.:-]+", re.ASCII)
PYTORCH_PACKAGES = ("torch", "torchvision", "torchaudio")


class Operation(str, Enum):
    """Identify the side-effect contract selected by a subcommand."""

    APPLY = "apply"
    PLAN = "plan"
    CHECK = "check"


class Channel(str, Enum):
    """Select the official stable or nightly PyTorch wheel channel."""

    STABLE = "stable"
    NIGHTLY = "nightly"


class OutputFormat(str, Enum):
    """Select human-readable or machine-readable command output."""

    TEXT = "text"
    JSON = "json"


class ProbeProfile(str, Enum):
    """Select standard runtime checks or optional compiler validation."""

    STANDARD = "standard"
    COMPILE = "compile"


class FrameworkProbe(str, Enum):
    """Identify a supported framework integration check."""

    VLLM = "vllm"


class BackendKind(str, Enum):
    """Select automatic, CPU-only, CUDA-only, or a concrete CUDA policy."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    CONCRETE = "concrete"


@dataclass(frozen=True, slots=True)
class BackendRequest:
    """Represent a validated backend selection policy."""

    kind: BackendKind
    concrete_value: str = ""

    def __post_init__(self) -> None:
        """Reject concrete and non-concrete states that disagree."""
        has_concrete_value = bool(self.concrete_value)
        if (self.kind is BackendKind.CONCRETE) != has_concrete_value:
            raise ConfigurationError("a concrete backend requires one CUDA identifier")
        if has_concrete_value and not _CUDA_BACKEND_PATTERN.fullmatch(
            self.concrete_value
        ):
            raise ConfigurationError(
                f"invalid concrete CUDA backend {self.concrete_value!r}"
            )

    @classmethod
    def parse(cls, raw_value: str) -> BackendRequest:
        """Parse a public backend option.

        Raises:
            ConfigurationError: If the value is not auto, cpu, cuda, or cuNNN.
        """
        value = raw_value.strip().lower()
        if value in {"auto", "cpu", "cuda"}:
            return cls(BackendKind(value))
        if _CUDA_BACKEND_PATTERN.fullmatch(value):
            return cls(BackendKind.CONCRETE, value)
        raise ConfigurationError(
            f"unsupported backend {raw_value!r}; expected auto, cpu, cuda, or cuNNN"
        )


@dataclass(frozen=True, slots=True)
class BackendCandidate:
    """Represent one backend that can be installed and runtime-tested."""

    value: str
    channel: Channel = Channel.STABLE

    def __post_init__(self) -> None:
        """Ensure candidates are auto, CPU, or concrete NVIDIA CUDA builds."""
        if self.value not in {"auto", "cpu"} and not _CUDA_BACKEND_PATTERN.fullmatch(
            self.value
        ):
            raise ConfigurationError(f"invalid backend candidate {self.value!r}")
        if self.channel is Channel.NIGHTLY and self.value == "auto":
            raise ConfigurationError("nightly candidates must use a concrete backend")

    @property
    def is_cuda(self) -> bool:
        """Return whether this is a concrete NVIDIA CUDA build."""
        return self.value.startswith("cu")

    @property
    def is_concrete(self) -> bool:
        """Return whether this candidate maps to one official index."""
        return self.value != "auto"

    @property
    def index_name(self) -> str:
        """Return the uv index name used in project metadata.

        Raises:
            ConfigurationError: If automatic selection has not resolved yet.
        """
        if not self.is_concrete:
            raise ConfigurationError("the auto backend has no concrete index")
        prefix = "pytorch-nightly" if self.channel is Channel.NIGHTLY else "pytorch"
        return f"{prefix}-{self.value}"

    @property
    def index_url(self) -> str:
        """Return the official index URL for this concrete candidate.

        Raises:
            ConfigurationError: If automatic selection has not resolved yet.
        """
        if not self.is_concrete:
            raise ConfigurationError("the auto backend has no concrete index")
        channel_path = "nightly/" if self.channel is Channel.NIGHTLY else ""
        return f"https://download.pytorch.org/whl/{channel_path}{self.value}"


@dataclass(frozen=True, slots=True)
class Scope:
    """Identify base, optional, or dependency-group requirements."""

    kind: str
    name: str = ""

    def __post_init__(self) -> None:
        """Reject unnamed non-base scopes and named base scopes."""
        if self.kind not in {"base", "extra", "group"}:
            raise ConfigurationError(f"unsupported dependency scope {self.kind!r}")
        if (self.kind == "base") == bool(self.name):
            raise ConfigurationError("only extra and group scopes require a name")

    @property
    def label(self) -> str:
        """Return a stable human- and machine-readable scope label."""
        return self.kind if self.kind == "base" else f"{self.kind}:{self.name}"


@dataclass(frozen=True, slots=True)
class ScopedRequirement:
    """Keep a parsed requirement associated with its declaration scope."""

    scope: Scope
    raw: str
    requirement: Requirement = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Parse the requirement once so later policies share one interpretation."""
        try:
            parsed = Requirement(self.raw)
        except InvalidRequirement as exc:
            raise ConfigurationError(
                f"invalid requirement in {self.scope.label}: {self.raw!r}"
            ) from exc
        package = canonicalize_name(parsed.name)
        if package in PYTORCH_PACKAGES and parsed.url is not None:
            raise ConfigurationError(
                f"direct URL requirements are not supported for {package}"
            )
        object.__setattr__(self, "requirement", parsed)

    @property
    def package(self) -> str:
        """Return the normalized package name."""
        return canonicalize_name(self.requirement.name)


@dataclass(frozen=True, slots=True)
class ProjectRequirements:
    """Describe selected Python and PyTorch requirements across project scopes."""

    requires_python: str
    python_file_value: str
    selected: tuple[ScopedRequirement, ...]
    all_pytorch: tuple[ScopedRequirement, ...]
    selected_scopes: tuple[Scope, ...]
    marker_environment: tuple[tuple[str, str], ...] = ()

    @property
    def probe_requirements(self) -> tuple[str, ...]:
        """Return all selected roots installed in candidate environments."""
        # Import locally to keep the foundational domain module independent of
        # the policy object that consumes its requirement values.
        from uv_torch_compass.dependency_roots import SelectedDependencyRoots

        return SelectedDependencyRoots(self.selected).candidate_requirements

    def has_package(self, package: str) -> bool:
        """Return whether the selected scopes contain a direct package requirement."""
        normalized = canonicalize_name(package)
        return any(item.package == normalized for item in self.selected)

    def requirement_for(self, package: str) -> tuple[Requirement, ...]:
        """Return parsed requirements for one package in selected scopes."""
        normalized = canonicalize_name(package)
        return tuple(
            item.requirement for item in self.selected if item.package == normalized
        )

    def for_interpreter(
        self,
        version: str,
        implementation_name: str,
        platform_implementation: str,
    ) -> ProjectRequirements:
        """Return requirements whose markers apply to the resolved Linux runtime."""
        parsed_version = Version(version)
        environment = cast(dict[str, str], dict(default_environment()))
        environment.update(
            {
                "python_version": ".".join(
                    str(part) for part in parsed_version.release[:2]
                ),
                "python_full_version": str(parsed_version),
                "implementation_name": implementation_name,
                "platform_python_implementation": platform_implementation,
                "sys_platform": "linux",
                "platform_system": "Linux",
            }
        )
        selected = tuple(
            item
            for item in self.selected
            if item.requirement.marker is None
            or item.requirement.marker.evaluate(environment)
        )
        if not selected:
            raise ConfigurationError(
                "no selected dependency requirement applies to the resolved interpreter"
            )
        return ProjectRequirements(
            self.requires_python,
            self.python_file_value,
            selected,
            self.all_pytorch,
            self.selected_scopes,
            tuple(sorted(environment.items())),
        )

    def environment(self) -> dict[str, str]:
        """Return the resolved marker environment, or a Linux default."""
        if self.marker_environment:
            return dict(self.marker_environment)
        environment = cast(dict[str, str], dict(default_environment()))
        environment.update({"sys_platform": "linux", "platform_system": "Linux"})
        return environment


@dataclass(frozen=True, slots=True)
class GpuDevice:
    """Represent an optional nvidia-smi index or UUID selector."""

    value: str

    def __post_init__(self) -> None:
        """Reject shell metacharacters and empty identifiers at the CLI boundary."""
        if not self.value or not _GPU_IDENTIFIER_PATTERN.fullmatch(self.value):
            raise ConfigurationError(f"invalid CUDA device {self.value!r}")


@dataclass(frozen=True, slots=True)
class RunOptions:
    """Contain fully resolved, immutable command settings."""

    operation: Operation
    pyproject: Path
    python_request: str
    requirement_overrides: tuple[str, ...]
    backend: BackendRequest
    channel: Channel
    cuda_compatibility: CompatibilityPolicy
    probe_profile: ProbeProfile
    extras: tuple[str, ...]
    groups: tuple[str, ...]
    cuda_device: GpuDevice | None
    link_mode: str
    log_dir: Path
    timeout_seconds: int
    output_format: OutputFormat
    report_file: Path | None
    framework_probes: tuple[FrameworkProbe, ...] = ()

    def __post_init__(self) -> None:
        """Enforce options that must be valid before infrastructure starts."""
        if self.pyproject.name != "pyproject.toml":
            raise ConfigurationError("--pyproject must point to a pyproject.toml file")
        if self.timeout_seconds <= 0:
            raise ConfigurationError("timeout must be greater than zero")
        if self.link_mode not in {"clone", "copy", "hardlink", "symlink"}:
            raise ConfigurationError(f"invalid link mode {self.link_mode!r}")
        if self.operation is Operation.CHECK and self.requirement_overrides:
            raise ConfigurationError("check does not accept dependency overrides")


@dataclass(frozen=True, slots=True)
class RuntimeReport:
    """Describe versions and device behavior verified by a probe interpreter."""

    schema_version: int
    backend: BackendCandidate
    torch_version: str
    torchvision_version: str
    torchaudio_version: str
    numpy_version: str
    cuda_runtime: str
    gpu_name: str
    cuda_test: str
    numpy_bridge_test: str
    torchvision_test: str
    torchaudio_test: str
    runtime_component_version: str = "not-installed"
    gpu_device_capability: str = "none"
    compiled_architectures: tuple[str, ...] = ()
    native_architecture_test: str = "NOT_APPLICABLE"
    cublas_test: str = "NOT_APPLICABLE"
    cudnn_test: str = "NOT_APPLICABLE"
    compile_test: str = "NOT_REQUESTED"
    probe_profile: str = "standard"

    @classmethod
    def from_output(
        cls, output: str, *, channel: Channel = Channel.STABLE
    ) -> RuntimeReport:
        """Parse and validate the final JSON line from a runtime probe.

        Raises:
            ProbeError: If output is missing, malformed, or violates the schema.
        """
        lines = [line for line in output.splitlines() if line.strip()]
        if not lines:
            raise ProbeError("runtime probe produced no result")
        try:
            value = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise ProbeError("runtime probe did not end with valid JSON") from exc
        if not isinstance(value, dict):
            raise ProbeError("runtime probe JSON must be an object")
        if value.get("schema_version") != 2:
            raise ProbeError("runtime probe schema version is unsupported")

        string_fields = (
            "backend",
            "torch_version",
            "torchvision_version",
            "torchaudio_version",
            "numpy_version",
            "cuda_runtime",
            "gpu_name",
            "cuda_test",
            "numpy_bridge_test",
            "torchvision_test",
            "torchaudio_test",
            "runtime_component_version",
            "gpu_device_capability",
            "native_architecture_test",
            "cublas_test",
            "cudnn_test",
            "compile_test",
            "probe_profile",
        )
        invalid = [
            field for field in string_fields if not isinstance(value.get(field), str)
        ]
        if invalid:
            raise ProbeError(f"runtime probe has invalid fields: {', '.join(invalid)}")
        architectures = value.get("compiled_architectures")
        if not isinstance(architectures, list) or not all(
            isinstance(item, str) for item in architectures
        ):
            raise ProbeError(
                "runtime probe compiled_architectures must be an array of strings"
            )
        try:
            backend = BackendCandidate(value["backend"], channel)
        except ConfigurationError as exc:
            raise ProbeError(str(exc)) from exc
        return cls(
            schema_version=2,
            backend=backend,
            torch_version=value["torch_version"],
            torchvision_version=value["torchvision_version"],
            torchaudio_version=value["torchaudio_version"],
            numpy_version=value["numpy_version"],
            cuda_runtime=value["cuda_runtime"],
            gpu_name=value["gpu_name"],
            cuda_test=value["cuda_test"],
            numpy_bridge_test=value["numpy_bridge_test"],
            torchvision_test=value["torchvision_test"],
            torchaudio_test=value["torchaudio_test"],
            runtime_component_version=value["runtime_component_version"],
            gpu_device_capability=value["gpu_device_capability"],
            compiled_architectures=tuple(architectures),
            native_architecture_test=value["native_architecture_test"],
            cublas_test=value["cublas_test"],
            cudnn_test=value["cudnn_test"],
            compile_test=value["compile_test"],
            probe_profile=value["probe_profile"],
        )

    def validate_requirements(self, requirements: ProjectRequirements) -> None:
        """Confirm reported package versions satisfy selected direct requirements.

        Raises:
            ProbeError: If a reported version is invalid or violates a requirement.
        """
        reported = {
            "torch": self.torch_version,
            "torchvision": self.torchvision_version,
            "torchaudio": self.torchaudio_version,
        }
        for package, version_text in reported.items():
            package_requirements = requirements.requirement_for(package)
            if not package_requirements:
                continue
            if version_text == "not-installed":
                raise ProbeError(f"{package} was required but not installed")
            try:
                version = Version(version_text.split("+", 1)[0])
            except InvalidVersion as exc:
                raise ProbeError(
                    f"runtime probe returned invalid {package} version {version_text!r}"
                ) from exc
            for requirement in package_requirements:
                allow_prereleases = self.backend.channel is Channel.NIGHTLY
                if requirement.specifier and not requirement.specifier.contains(
                    version, prereleases=allow_prereleases
                ):
                    raise ProbeError(
                        f"{package} {version} does not satisfy {requirement.specifier}"
                    )

    def validate_probe_results(
        self,
        requirements: ProjectRequirements,
        *,
        expected_profile: ProbeProfile,
        require_native_architecture: bool,
        expected_packages: frozenset[str] | None = None,
    ) -> None:
        """Confirm every reported check has the result required by this run.

        Raises:
            ProbeError: If the probe profile, device identity, or a check result
                does not match the requested validation.
        """
        if self.probe_profile != expected_profile.value:
            raise ProbeError(
                f"runtime probe reported profile {self.probe_profile!r}, expected "
                f"{expected_profile.value!r}"
            )
        packages = (
            expected_packages
            if expected_packages is not None
            else frozenset(
                package
                for package in PYTORCH_PACKAGES
                if requirements.has_package(package)
            )
        )
        expected = {
            "numpy_bridge_test": "PASS",
            "torchvision_test": (
                "PASS" if "torchvision" in packages else "NOT_REQUESTED"
            ),
            "torchaudio_test": (
                "PASS" if "torchaudio" in packages else "NOT_REQUESTED"
            ),
            "compile_test": (
                "PASS" if expected_profile is ProbeProfile.COMPILE else "NOT_REQUESTED"
            ),
        }
        if self.backend.is_cuda:
            expected.update(
                cuda_test="PASS",
                cublas_test="PASS",
                cudnn_test="PASS",
            )
            if self.gpu_name == "none" or not re.fullmatch(
                r"[0-9]+\.[0-9]+", self.gpu_device_capability
            ):
                raise ProbeError("runtime probe returned invalid CUDA device details")
            if not self.compiled_architectures:
                raise ProbeError(
                    "runtime probe returned no compiled CUDA architectures"
                )
            allowed_native = (
                {"PASS"}
                if require_native_architecture
                else {
                    "PASS",
                    "PTX_ONLY",
                }
            )
            if self.native_architecture_test not in allowed_native:
                raise ProbeError(
                    "runtime probe did not verify the required native architecture"
                )
        else:
            expected.update(
                cuda_test="NOT_APPLICABLE",
                cublas_test="NOT_APPLICABLE",
                cudnn_test="NOT_APPLICABLE",
                native_architecture_test="NOT_APPLICABLE",
            )
            if (
                self.cuda_runtime != "none"
                or self.runtime_component_version != "not-installed"
                or self.gpu_name != "none"
                or self.gpu_device_capability != "none"
                or self.compiled_architectures
            ):
                raise ProbeError("CPU probe reported unexpected CUDA runtime details")

        for field_name, expected_value in expected.items():
            actual = getattr(self, field_name)
            if actual != expected_value:
                raise ProbeError(
                    f"runtime probe returned {field_name}={actual!r}, expected "
                    f"{expected_value!r}"
                )


@dataclass(frozen=True, slots=True)
class CandidateAttempt:
    """Record one backend candidate result without retaining secret command data."""

    backend: str
    stage: str
    status: str
    reason: str
    compatibility: str
    failure: CandidateFailure | None = None
    resolution: CandidateResolution | None = None
    framework_compatibility: FrameworkCompatibilityDecision | None = None
    framework_requests: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    """Return a completed operation and its serializable result fields."""

    status: str
    applied: bool
    runtime: RuntimeReport | None
    compatibility: CompatibilityDecision | None = None
    attempts: tuple[CandidateAttempt, ...] = ()
    changes: tuple[str, ...] = ()
    backups: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = ()
    planned_diff: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def has_upper_bound(specifier: str) -> bool:
    """Return whether a Python specifier contains a finite upper bound."""
    return any(item.operator in {"<", "<="} for item in SpecifierSet(specifier))
