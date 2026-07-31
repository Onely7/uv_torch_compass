"""Define immutable contracts for candidate and runtime verification."""

from __future__ import annotations

from dataclasses import dataclass

from uv_torch_compass.candidate_resolution import CandidateResolution
from uv_torch_compass.cuda_compatibility import CompatibilityDecision
from uv_torch_compass.domain import (
    CandidateAttempt,
    ProbeProfile,
    ResolutionFailure,
    RuntimeReport,
)
from uv_torch_compass.framework_validation import FrameworkValidation
from uv_torch_compass.source_ownership import ManagedSourceAnchor


@dataclass(frozen=True, slots=True)
class ProbeContract:
    """Describe packages selected for execution and result validation.

    Installed packages control both executed checks and expected results so
    direct and transitive companion dependencies share one contract.
    """

    installed_pytorch: frozenset[str]
    expected_pytorch: frozenset[str]
    profile: ProbeProfile

    def validates(self, package: str) -> bool:
        """Return whether the runtime process must validate an installed package."""
        return package in self.installed_pytorch

    @classmethod
    def for_installed_packages(
        cls,
        packages: frozenset[str],
        profile: ProbeProfile,
    ) -> ProbeContract:
        """Build one consistent execution and validation contract."""
        return cls(packages, packages, profile)


@dataclass(frozen=True, slots=True)
class ProbeOutcome:
    """Contain the first verified runtime and any required NumPy repair."""

    runtime: RuntimeReport
    compatibility: CompatibilityDecision
    numpy_lt2_required: bool
    attempts: tuple[CandidateAttempt, ...]
    installed_pytorch: frozenset[str] = frozenset()
    source_anchors: tuple[ManagedSourceAnchor, ...] = ()
    framework_validation: tuple[FrameworkValidation, ...] = ()
    resolution: CandidateResolution | None = None


@dataclass(frozen=True, slots=True)
class CandidateProbeResult:
    """Represent either a verified candidate or a rejected candidate."""

    outcome: ProbeOutcome | None
    failure: ResolutionFailure | None
    resolution: CandidateResolution | None = None
    stage: str = "runtime"

    def __post_init__(self) -> None:
        """Require exactly one success outcome or failure reason."""
        if (self.outcome is None) == (self.failure is None):
            raise ValueError("candidate result requires either an outcome or a failure")
        if self.stage not in {"lock", "install", "runtime", "framework"}:
            raise ValueError(f"unsupported candidate result stage {self.stage!r}")

    @classmethod
    def passed(cls, outcome: ProbeOutcome) -> CandidateProbeResult:
        """Create a successful candidate result."""
        return cls(outcome, None, outcome.resolution, "runtime")

    @classmethod
    def failed(
        cls,
        failure: ResolutionFailure,
        resolution: CandidateResolution | None = None,
        *,
        stage: str = "runtime",
    ) -> CandidateProbeResult:
        """Create a failed candidate result."""
        return cls(None, failure, resolution, stage)
