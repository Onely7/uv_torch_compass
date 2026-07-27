"""Build backend candidates and verify each in an isolated environment."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from uv_torch_compass.command_runner import ProcessRunner, sanitized_environment
from uv_torch_compass.cuda_compatibility import (
    CompatibilityDecision,
    CompatibilityLevel,
    CompatibilityPolicy,
    validate_runtime_identity,
)
from uv_torch_compass.domain import (
    BackendCandidate,
    CandidateAttempt,
    ProbeProfile,
    ProjectRequirements,
    RuntimeReport,
)
from uv_torch_compass.errors import (
    CandidateResolutionError,
    CommandTimeoutError,
    ConfigurationError,
    ProbeError,
)
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
from uv_torch_compass.source_ownership import derive_managed_source_anchors
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
        for candidate in candidates:
            result = self._probe_candidate(candidate)
            if result.outcome is None:
                failure = result.failure
                if failure is None:
                    raise ProbeError("candidate failure did not include a diagnostic")
                attempts.append(
                    CandidateAttempt(
                        candidate.value,
                        (
                            "runtime"
                            if failure.kind.value == "runtime-validation"
                            else "install"
                        ),
                        "failed",
                        failure.summary,
                        self._compatibility_for(candidate).level.value,
                        failure,
                    )
                )
                continue
            outcome = result.outcome
            attempts.append(
                CandidateAttempt(
                    candidate.value,
                    "runtime",
                    "passed",
                    f"resolved as {outcome.runtime.backend.value}",
                    outcome.compatibility.level.value,
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
            )
        attempted = ", ".join(candidate.value for candidate in candidates)
        raise CandidateResolutionError(
            f"no usable PyTorch backend was found; attempted: {attempted}",
            tuple(attempts),
        )

    def _probe_candidate(self, candidate: BackendCandidate) -> CandidateProbeResult:
        venv = Path(
            tempfile.mkdtemp(
                prefix=f"candidate-{candidate.value}-", dir=self.temporary_root
            )
        )
        self.reporter.info(f"testing backend candidate {candidate.value}")
        try:
            created = self.uv.create_venv(
                venv, self.project_python, cwd=self.temporary_root
            )
        except CommandTimeoutError:
            return CandidateProbeResult.failed(timeout_failure("environment creation"))
        self.reporter.detail(created.stdout + created.stderr)
        if created.returncode != 0:
            self.reporter.warn(f"candidate {candidate.value}: venv creation failed")
            return CandidateProbeResult.failed(
                runtime_failure(
                    "The candidate virtual environment could not be created."
                )
            )
        try:
            installed = self.uv.install_candidate(
                venv, self.requirements.probe_requirements, candidate
            )
        except CommandTimeoutError:
            return CandidateProbeResult.failed(timeout_failure("installation"))
        self.reporter.detail(installed.stdout + installed.stderr)
        if installed.returncode != 0:
            self.reporter.warn(f"candidate {candidate.value}: installation failed")
            return CandidateProbeResult.failed(
                interpret_uv_failure(
                    installed.stdout + installed.stderr,
                    candidate=candidate,
                    dependency_roots=self.requirements.probe_requirements,
                )
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
                runtime_failure("Installed package metadata could not be validated.")
            )
        if "torch" not in installed_pytorch:
            self.reporter.warn(
                f"candidate {candidate.value}: selected dependencies do not install torch"
            )
            return CandidateProbeResult.failed(
                runtime_failure("The selected dependencies did not install torch.")
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
                installed_distributions,
            )
            return (
                CandidateProbeResult.passed(outcome)
                if outcome is not None
                else CandidateProbeResult.failed(
                    runtime_failure("The candidate runtime validation failed.")
                )
            )
        self.reporter.detail(validation.stdout + validation.stderr)
        if "NUMPY_BRIDGE_FAILED" not in validation.stderr:
            self.reporter.warn(
                f"candidate {candidate.value}: runtime validation failed"
            )
            return CandidateProbeResult.failed(
                runtime_failure("The candidate runtime validation failed.")
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
                )
            )
        validation = self._run_probe(venv, candidate, contract)
        self.reporter.detail(validation.stdout + validation.stderr)
        if validation.returncode != 0:
            return CandidateProbeResult.failed(
                runtime_failure(
                    "The candidate failed after the NumPy compatibility repair."
                )
            )
        outcome = self._parse_success(
            validation.stdout,
            candidate,
            True,
            contract,
            installed_distributions,
        )
        return (
            CandidateProbeResult.passed(outcome)
            if outcome is not None
            else CandidateProbeResult.failed(
                runtime_failure("The candidate runtime validation failed.")
            )
        )

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
        installed_distributions: tuple[InstalledDistribution, ...],
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
            derive_managed_source_anchors(
                installed_distributions,
                self.requirements,
            ),
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
