"""Build backend candidates and verify each in an isolated environment."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from uv_torch_compass.candidate_environment import CandidateExecutionEnvironment
from uv_torch_compass.candidate_failures import (
    FailedPackage,
    FrameworkCompatibilityDecision,
    FrameworkFailure,
    FrameworkFailureKind,
)
from uv_torch_compass.candidate_lock import read_candidate_lock
from uv_torch_compass.candidate_metadata import CandidateDependencyGraph
from uv_torch_compass.candidate_project import render_candidate_project
from uv_torch_compass.candidate_resolution import CandidateResolution
from uv_torch_compass.candidate_workspace_metadata import (
    read_candidate_workspace_metadata,
)
from uv_torch_compass.command_runner import ProcessRunner, sanitized_environment
from uv_torch_compass.cuda_compatibility import (
    CompatibilityDecision,
    CompatibilityLevel,
    CompatibilityPolicy,
    validate_runtime_identity,
)
from uv_torch_compass.domain import (
    PYTORCH_PACKAGES,
    BackendCandidate,
    CandidateAttempt,
    FailedIndex,
    FrameworkProbe,
    ProbeProfile,
    ProjectRequirements,
    ResolutionFailure,
    ResolutionFailureKind,
    RuntimeReport,
)
from uv_torch_compass.errors import (
    CandidateResolutionError,
    CommandTimeoutError,
    ConfigurationError,
    ProbeError,
)
from uv_torch_compass.framework_artifact import FrameworkArtifactInspector
from uv_torch_compass.framework_diagnostics import (
    artifact_failure,
    catalog_dependency_failure,
    validation_failure,
)
from uv_torch_compass.framework_validation import (
    FrameworkValidation,
    framework_validation_document,
    parse_framework_probe,
)
from uv_torch_compass.index_url import canonical_official_pytorch_url
from uv_torch_compass.installed_metadata import (
    InstalledDistribution,
    read_installed_distributions,
)
from uv_torch_compass.nvidia import NvidiaSnapshot
from uv_torch_compass.probe_contract import (
    CandidateProbeResult,
    ProbeContract,
    ProbeOutcome,
)
from uv_torch_compass.reporting import CommandReporter
from uv_torch_compass.resolution_diagnostics import (
    interpret_uv_failure,
    runtime_failure,
    timeout_failure,
)
from uv_torch_compass.source_ownership import derive_lock_source_anchors
from uv_torch_compass.uv_commands import UvCommandClient


@dataclass(slots=True)
class CandidateProbeService:
    """Install and runtime-test candidates until one passes all requirements."""

    uv: UvCommandClient
    runner: ProcessRunner
    reporter: CommandReporter
    temporary_root: Path
    project_python: Path
    requirements: ProjectRequirements
    runtime_probe: Path
    cuda_device: str | None
    nvidia: NvidiaSnapshot | None
    compatibility_policy: CompatibilityPolicy
    probe_profile: ProbeProfile
    target_pyproject: Path
    execution_environment: CandidateExecutionEnvironment
    workspace_members: tuple[tuple[str, Path], ...] = ()
    framework_probes: tuple[FrameworkProbe, ...] = ()
    _artifact_inspector: FrameworkArtifactInspector | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _artifact_fallback_warned: bool = field(default=False, init=False, repr=False)
    _metadata_fallback_warned: bool = field(default=False, init=False, repr=False)

    def find_working_candidate(
        self,
        candidates: tuple[BackendCandidate, ...],
        *,
        prior_attempts: tuple[CandidateAttempt, ...] = (),
    ) -> ProbeOutcome:
        """Return the first verified candidate.

        Raises:
            CommandError: If every candidate fails installation or runtime checks.
        """
        attempts = list(prior_attempts)
        required_backend: str | None = None
        required_cuda_major: int | None = None
        for candidate in candidates:
            skip_reason = _framework_candidate_skip_reason(
                candidate,
                required_backend=required_backend,
                required_cuda_major=required_cuda_major,
            )
            if skip_reason is not None:
                attempts.append(
                    CandidateAttempt(
                        candidate.value,
                        "artifact",
                        "skipped",
                        skip_reason,
                        "incompatible",
                    )
                )
                continue
            result = self._probe_candidate(candidate)
            if result.outcome is None:
                failure = result.failure
                if failure is None:
                    raise ProbeError("candidate failure did not include a diagnostic")
                attempts.append(
                    CandidateAttempt(
                        candidate.value,
                        result.stage,
                        "failed",
                        failure.summary,
                        self._compatibility_for(candidate).level.value,
                        failure,
                        result.resolution,
                        result.framework_compatibility,
                    )
                )
                if isinstance(failure, FrameworkFailure):
                    requirement = failure.binary_requirement
                    if failure.backend_independent:
                        attempts.extend(
                            _skipped_framework_attempts(
                                candidates,
                                candidate,
                                "the same backend-independent framework failure "
                                "would affect this candidate",
                            )
                        )
                        break
                    if requirement is not None:
                        required_backend = requirement.required_cuda_variant
                        required_cuda_major = requirement.required_cuda_major
                continue
            outcome = result.outcome
            attempts.append(
                CandidateAttempt(
                    candidate.value,
                    ("framework" if outcome.framework_validation else "runtime"),
                    "passed",
                    f"resolved as {outcome.runtime.backend.value}",
                    outcome.compatibility.level.value,
                    resolution=outcome.resolution,
                    framework_compatibility=outcome.framework_compatibility,
                )
            )
            if outcome.compatibility.level is CompatibilityLevel.MINOR:
                self.reporter.warn(outcome.compatibility.reason)
            return ProbeOutcome(
                outcome.runtime,
                outcome.compatibility,
                outcome.numpy_lt2_required,
                tuple(attempts),
                outcome.installed_pytorch,
                outcome.source_anchors,
                outcome.framework_validation,
                outcome.resolution,
                outcome.framework_compatibility,
            )
        attempted = ", ".join(candidate.value for candidate in candidates)
        if any(attempt.resolution is not None for attempt in attempts):
            message = (
                "compatible PyTorch builds were resolved, but the selected "
                "dependency graph could not be fully installed and validated"
            )
        else:
            message = "no usable PyTorch backend was found"
        raise CandidateResolutionError(
            f"{message}; attempted: {attempted}",
            tuple(attempts),
        )

    def _probe_candidate(self, candidate: BackendCandidate) -> CandidateProbeResult:
        candidate_root = Path(
            tempfile.mkdtemp(
                prefix=f"candidate-{candidate.value}-", dir=self.temporary_root
            )
        )
        venv = candidate_root / "environment"
        self.reporter.info(f"testing backend candidate {candidate.value}")
        prepared = self._resolve_candidate(candidate, candidate_root)
        if isinstance(prepared, CandidateProbeResult):
            return prepared
        resolution, project_dir = prepared
        artifact = self._preflight_framework_artifact(
            resolution,
            project_dir,
            candidate,
        )
        if isinstance(artifact, CandidateProbeResult):
            return artifact
        try:
            installed = self.uv.sync_locked_candidate(
                venv,
                project_dir,
                self.project_python,
            )
        except CommandTimeoutError:
            return CandidateProbeResult.failed(
                timeout_failure("installation"),
                resolution,
                stage="install",
            )
        self.reporter.detail(installed.stdout + installed.stderr)
        if installed.returncode != 0:
            self.reporter.warn(f"candidate {candidate.value}: installation failed")
            return CandidateProbeResult.failed(
                resolution.enrich_failure(
                    interpret_uv_failure(
                        installed.stdout + installed.stderr,
                        candidate=candidate,
                        dependency_roots=self.requirements.probe_requirements,
                    )
                ),
                resolution,
                stage="install",
            )

        try:
            installed_distributions = read_installed_distributions(venv)
            installed_pytorch = frozenset(
                distribution.name
                for distribution in installed_distributions
                if distribution.name in {"torch", "torchvision", "torchaudio"}
            )
        except ProbeError as exc:
            self.reporter.warn(f"candidate {candidate.value}: {exc}")
            return CandidateProbeResult.failed(
                runtime_failure("Installed package metadata could not be validated."),
                resolution,
                stage="install",
            )
        if "torch" not in installed_pytorch:
            self.reporter.warn(
                f"candidate {candidate.value}: selected dependencies do not install torch"
            )
            return CandidateProbeResult.failed(
                runtime_failure("The selected dependencies did not install torch."),
                resolution,
                stage="install",
            )

        contract = ProbeContract.for_installed_packages(
            installed_pytorch,
            self.probe_profile,
        )
        validation = self._run_probe(venv, candidate, contract)
        if validation.returncode == 0:
            outcome = self._parse_success(
                validation.stdout,
                candidate,
                False,
                contract,
                resolution,
            )
            return self._finalize_candidate(
                venv,
                candidate,
                outcome,
                resolution,
                installed_distributions,
                artifact,
            )
        self.reporter.detail(validation.stdout + validation.stderr)
        if "NUMPY_BRIDGE_FAILED" not in validation.stderr:
            self.reporter.warn(
                f"candidate {candidate.value}: runtime validation failed"
            )
            return CandidateProbeResult.failed(
                runtime_failure("The candidate runtime validation failed."),
                resolution,
                stage="runtime",
            )

        self.reporter.warn(
            f"candidate {candidate.value}: retrying the NumPy bridge with numpy<2"
        )
        repaired = self.uv.install_numpy_lt2(venv)
        self.reporter.detail(repaired.stdout + repaired.stderr)
        if repaired.returncode != 0:
            return CandidateProbeResult.failed(
                runtime_failure(
                    "The NumPy compatibility repair could not be installed."
                ),
                resolution,
                stage="install",
            )
        validation = self._run_probe(venv, candidate, contract)
        self.reporter.detail(validation.stdout + validation.stderr)
        if validation.returncode != 0:
            return CandidateProbeResult.failed(
                runtime_failure(
                    "The candidate failed after the NumPy compatibility repair."
                ),
                resolution,
                stage="runtime",
            )
        outcome = self._parse_success(
            validation.stdout,
            candidate,
            True,
            contract,
            resolution,
        )
        return self._finalize_candidate(
            venv,
            candidate,
            outcome,
            resolution,
            installed_distributions,
            artifact,
        )

    def _preflight_framework_artifact(
        self,
        resolution: CandidateResolution,
        project_dir: Path,
        candidate: BackendCandidate,
    ) -> FrameworkCompatibilityDecision | None | CandidateProbeResult:
        """Inspect vLLM wheel requirements before installing its dependency graph."""
        inspector = self._framework_artifact_inspector()
        try:
            preflight, command = inspector.inspect(resolution, project_dir)
        except CommandTimeoutError:
            return CandidateProbeResult.failed(
                timeout_failure("framework artifact inspection"),
                resolution,
                stage="artifact",
            )
        except ProbeError as exc:
            failure = FrameworkFailure(
                FrameworkFailureKind.METADATA,
                f"vLLM artifact inspection failed safely: {exc}",
                "vllm",
                _locked_version(resolution, "vllm"),
                FailedPackage("vllm", _locked_version(resolution, "vllm"), "vllm"),
                resolution.dependency_paths("vllm"),
                packages=(),
                suggestions=(
                    "Verify that the locked vLLM wheel is complete and from a trusted source.",
                ),
            )
            return CandidateProbeResult.failed(failure, resolution, stage="artifact")
        if command is not None:
            self.reporter.detail(command.stdout + command.stderr)
        if (
            preflight.status in {"unsupported-uv", "wheel-unavailable"}
            and not self._artifact_fallback_warned
        ):
            self.reporter.warn(
                f"candidate {candidate.value}: {preflight.detail}; "
                "continuing with full runtime validation"
            )
            self._artifact_fallback_warned = True
        decision = preflight.decision
        if decision is not None and not decision.allowed:
            failure = artifact_failure(resolution, decision)
            self.reporter.warn(f"candidate {candidate.value}: {failure.summary}")
            return CandidateProbeResult.failed(
                failure,
                resolution,
                stage="artifact",
                framework_compatibility=decision,
            )
        advisory_failure = catalog_dependency_failure(resolution)
        if advisory_failure is not None:
            self.reporter.warn(
                f"candidate {candidate.value}: {advisory_failure.summary}"
            )
            return CandidateProbeResult.failed(
                advisory_failure,
                resolution,
                stage="artifact",
                framework_compatibility=decision,
            )
        return decision

    def _framework_artifact_inspector(self) -> FrameworkArtifactInspector:
        """Return the run-scoped artifact inspector and cached uv capabilities."""
        if self._artifact_inspector is None:
            capability_method = getattr(self.uv, "sync_capabilities", None)
            capabilities = (
                capability_method() if callable(capability_method) else frozenset()
            )
            self._artifact_inspector = FrameworkArtifactInspector(
                self.uv,
                self.temporary_root,
                self.project_python,
                capabilities,
            )
        return self._artifact_inspector

    def _resolve_candidate(
        self,
        candidate: BackendCandidate,
        candidate_root: Path,
    ) -> tuple[CandidateResolution, Path] | CandidateProbeResult:
        requirements = list(self.requirements.probe_requirements)
        anchored = {
            package
            for raw_requirement in requirements
            if (package := _requirement_name(raw_requirement)) in PYTORCH_PACKAGES
        }
        anchored.add("torch")
        for iteration in range(len(PYTORCH_PACKAGES) + 1):
            try:
                project = render_candidate_project(
                    self.target_pyproject,
                    destination=candidate_root / f"project-{iteration}",
                    requirements=requirements,
                    candidate=candidate,
                    environment=self.execution_environment,
                    workspace_members=dict(self.workspace_members),
                )
                locked = self.uv.lock_candidate(project.parent, self.project_python)
            except CommandTimeoutError:
                return CandidateProbeResult.failed(
                    timeout_failure("dependency resolution"),
                    stage="lock",
                )
            except ConfigurationError as exc:
                self.reporter.warn(f"candidate {candidate.value}: {exc}")
                return CandidateProbeResult.failed(
                    runtime_failure(
                        "The candidate source policy could not be prepared."
                    ),
                    stage="lock",
                )
            self.reporter.detail(locked.stdout + locked.stderr)
            if locked.returncode != 0:
                failure = interpret_uv_failure(
                    locked.stdout + locked.stderr,
                    candidate=candidate,
                    dependency_roots=self.requirements.probe_requirements,
                )
                missing_anchor = (
                    failure.package.name
                    if failure.package is not None
                    and failure.package.name in PYTORCH_PACKAGES
                    and failure.package.name not in anchored
                    else None
                )
                if missing_anchor is not None:
                    anchored.add(missing_anchor)
                    requirements.append(missing_anchor)
                    self.reporter.info(
                        f"candidate {candidate.value}: anchoring transitive "
                        f"{missing_anchor} to {candidate.index_name}"
                    )
                    continue
                self.reporter.warn(
                    f"candidate {candidate.value}: dependency resolution failed"
                )
                return CandidateProbeResult.failed(failure, stage="lock")
            try:
                resolution = CandidateResolution(
                    candidate,
                    self.execution_environment,
                    self._candidate_dependency_graph(project.parent),
                )
            except ProbeError as exc:
                self.reporter.warn(f"candidate {candidate.value}: {exc}")
                return CandidateProbeResult.failed(
                    runtime_failure("The candidate lockfile could not be validated."),
                    stage="lock",
                )
            mismatched = tuple(
                package
                for package in resolution.pytorch_packages
                if canonical_official_pytorch_url(package.source_url)
                != candidate.index_url
            )
            new_anchors = tuple(
                package.name for package in mismatched if package.name not in anchored
            )
            if new_anchors:
                for package in new_anchors:
                    anchored.add(package)
                    requirements.append(package)
                    self.reporter.info(
                        f"candidate {candidate.value}: anchoring transitive "
                        f"{package} to {candidate.index_name}"
                    )
                continue
            if mismatched:
                package = mismatched[0]
                failure = ResolutionFailure(
                    ResolutionFailureKind.DEPENDENCY_CONFLICT,
                    "A PyTorch package resolved from an unexpected package index.",
                    FailedPackage(package.name, package.version, package.name),
                    index=FailedIndex(candidate.index_name, candidate.index_url),
                    platform=self.execution_environment.platform_label,
                    suggestions=(
                        "Remove conflicting source or override rules for this package.",
                    ),
                )
                return CandidateProbeResult.failed(
                    failure,
                    resolution,
                    stage="lock",
                )
            return resolution, project.parent
        return CandidateProbeResult.failed(
            runtime_failure("Transitive PyTorch source anchoring did not converge."),
            stage="lock",
        )

    def _candidate_dependency_graph(
        self,
        project_dir: Path,
    ) -> CandidateDependencyGraph:
        """Read a candidate graph through uv's JSON boundary when available."""
        metadata_method = getattr(self.uv, "workspace_metadata", None)
        if callable(metadata_method):
            result = metadata_method(project_dir)
            self.reporter.detail(result.stdout + result.stderr)
            if result.returncode == 0:
                return read_candidate_workspace_metadata(result.stdout)
            if not self._metadata_fallback_warned:
                self.reporter.warn(
                    "uv workspace metadata resolution is unavailable; using the "
                    "validated lockfile fallback"
                )
                self._metadata_fallback_warned = True
        return read_candidate_lock(project_dir / "uv.lock")

    def _finalize_candidate(
        self,
        venv: Path,
        candidate: BackendCandidate,
        outcome: ProbeOutcome | None,
        resolution: CandidateResolution,
        installed_distributions: tuple[InstalledDistribution, ...],
        framework_compatibility: FrameworkCompatibilityDecision | None,
    ) -> CandidateProbeResult:
        if outcome is None:
            return CandidateProbeResult.failed(
                runtime_failure("The candidate runtime validation failed."),
                resolution,
                stage="runtime",
            )
        framework_probes = self._framework_probes(
            venv,
            candidate,
            resolution,
            installed_distributions,
        )
        if isinstance(framework_probes, CandidateProbeResult):
            return framework_probes
        return CandidateProbeResult.passed(
            ProbeOutcome(
                outcome.runtime,
                outcome.compatibility,
                outcome.numpy_lt2_required,
                outcome.attempts,
                outcome.installed_pytorch,
                outcome.source_anchors,
                framework_probes,
                resolution,
                framework_compatibility,
            )
        )

    def _framework_probes(
        self,
        venv: Path,
        candidate: BackendCandidate,
        resolution: CandidateResolution,
        installed_distributions: tuple[InstalledDistribution, ...],
    ) -> tuple[FrameworkValidation, ...] | CandidateProbeResult:
        installed = {distribution.name for distribution in installed_distributions}
        automatic = (
            frozenset({FrameworkProbe.VLLM})
            if "vllm" in installed and FrameworkProbe.VLLM not in self.framework_probes
            else frozenset()
        )
        requested = tuple(dict.fromkeys((*self.framework_probes, *automatic)))
        if not requested:
            return ()
        arguments: list[str | Path] = [
            venv / "bin" / "python",
            Path(__file__).with_name("framework_probe.py").resolve(),
            "--expected-backend",
            candidate.value,
        ]
        for framework in self.framework_probes:
            arguments.extend(["--framework", framework.value])
        if automatic:
            arguments.append("--auto-detect")
        environment, _ = sanitized_environment(os.environ)
        try:
            result = self.runner.run(
                arguments,
                env=environment,
                timeout_seconds=self.uv.diagnostic_timeout_seconds,
            )
        except CommandTimeoutError:
            return CandidateProbeResult.failed(
                timeout_failure("framework validation"),
                resolution,
                stage="framework",
            )
        self.reporter.detail(result.stdout + result.stderr)
        try:
            validations = parse_framework_probe(
                result.stdout,
                self.framework_probes,
                allow_automatic=True,
                require_success=False,
            )
        except ProbeError as exc:
            self.reporter.warn(f"candidate {candidate.value}: {exc}")
            return CandidateProbeResult.failed(
                runtime_failure(str(exc)),
                resolution,
                stage="framework",
            )
        failed = next(
            (validation for validation in validations if validation.status != "PASS"),
            None,
        )
        if failed is not None or result.returncode != 0:
            if failed is None:
                return CandidateProbeResult.failed(
                    runtime_failure(
                        "The framework probe exited without a failure result."
                    ),
                    resolution,
                    stage="framework",
                )
            return CandidateProbeResult.failed(
                validation_failure(failed, resolution),
                resolution,
                stage="framework",
            )
        self.reporter.detail(str(framework_validation_document(validations)))
        return validations

    def _run_probe(
        self,
        venv: Path,
        candidate: BackendCandidate,
        contract: ProbeContract,
    ):
        expected = candidate.value if candidate.is_concrete else None
        arguments: list[str | Path] = [venv / "bin" / "python", self.runtime_probe]
        arguments.extend(["--probe-profile", self.probe_profile.value])
        if expected:
            arguments.extend(["--expected-backend", expected])
        if (
            candidate.is_cuda
            and self._compatibility_for(candidate).level is CompatibilityLevel.MINOR
        ):
            arguments.append("--require-native-architecture")
        if contract.validates("torchvision"):
            arguments.append("--validate-torchvision")
        if contract.validates("torchaudio"):
            arguments.append("--validate-torchaudio")
        overrides = (
            {"CUDA_VISIBLE_DEVICES": self.cuda_device} if self.cuda_device else None
        )
        environment, _ = sanitized_environment(os.environ, overrides=overrides)
        return self.runner.run(
            arguments,
            env=environment,
            timeout_seconds=self.uv.heavy_timeout_seconds,
        )

    def _parse_success(
        self,
        output: str,
        candidate: BackendCandidate,
        numpy_lt2_required: bool,
        contract: ProbeContract,
        resolution: CandidateResolution,
    ) -> ProbeOutcome | None:
        try:
            report = RuntimeReport.from_output(output, channel=candidate.channel)
            report.validate_requirements(self.requirements)
            if report.backend.is_cuda:
                validate_runtime_identity(
                    report.backend.value,
                    cuda_runtime=report.cuda_runtime,
                    runtime_component=report.runtime_component_version,
                )
        except (ConfigurationError, ProbeError) as exc:
            self.reporter.warn(f"candidate {candidate.value}: {exc}")
            return None
        if candidate.is_concrete and report.backend.value != candidate.value:
            self.reporter.warn(
                f"candidate {candidate.value}: runtime reported {report.backend.value}"
            )
            return None
        compatibility = self._compatibility_for(report.backend)
        if not compatibility.allowed:
            self.reporter.warn(f"candidate {candidate.value}: {compatibility.reason}")
            return None
        try:
            report.validate_probe_results(
                self.requirements,
                expected_profile=contract.profile,
                require_native_architecture=(
                    compatibility.level is CompatibilityLevel.MINOR
                ),
                expected_packages=contract.expected_pytorch,
            )
        except ProbeError as exc:
            self.reporter.warn(f"candidate {candidate.value}: {exc}")
            return None
        return ProbeOutcome(
            report,
            compatibility,
            numpy_lt2_required,
            (),
            contract.installed_pytorch,
            derive_lock_source_anchors(resolution.lock, self.requirements),
            (),
            resolution,
        )

    def _compatibility_for(self, candidate: BackendCandidate) -> CompatibilityDecision:
        if not candidate.is_cuda:
            return CompatibilityDecision(
                CompatibilityLevel.STRICT,
                "",
                "CPU backend does not require an NVIDIA CUDA driver",
            )
        if self.nvidia is None:
            return CompatibilityDecision(
                CompatibilityLevel.UNSUPPORTED,
                "",
                "CUDA backend requires a visible NVIDIA GPU",
            )
        return self.nvidia.compatibility_for(candidate.value, self.compatibility_policy)


def _requirement_name(raw_value: str) -> str:
    try:
        return str(canonicalize_name(Requirement(raw_value).name))
    except InvalidRequirement:
        return ""


def _locked_version(resolution: CandidateResolution, package: str) -> str:
    locked = resolution.lock.package(package)
    return locked.version if locked is not None else "unknown"


def _framework_candidate_skip_reason(
    candidate: BackendCandidate,
    *,
    required_backend: str | None,
    required_cuda_major: int | None,
) -> str | None:
    if required_backend is not None and candidate.value != required_backend:
        return f"vLLM requires {required_backend}; this candidate was not installed"
    if required_cuda_major is None:
        return None
    if not candidate.is_cuda or _cuda_major(candidate.value) != required_cuda_major:
        return f"vLLM requires CUDA {required_cuda_major}; this candidate was not installed"
    return None


def _cuda_major(backend: str) -> int | None:
    digits = backend.removeprefix("cu")
    return int(digits[:-1]) if digits.isdigit() and len(digits) in {2, 3} else None


def _skipped_framework_attempts(
    candidates: tuple[BackendCandidate, ...],
    current: BackendCandidate,
    reason: str,
) -> tuple[CandidateAttempt, ...]:
    found_current = False
    skipped: list[CandidateAttempt] = []
    for candidate in candidates:
        if not found_current:
            if candidate is current:
                found_current = True
            continue
        skipped.append(
            CandidateAttempt(
                candidate.value,
                "framework",
                "skipped",
                reason,
                "not-tested",
            )
        )
    return tuple(skipped)
