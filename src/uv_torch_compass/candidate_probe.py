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
from uv_torch_compass.errors import CommandError, ConfigurationError, ProbeError
from uv_torch_compass.nvidia import NvidiaSnapshot
from uv_torch_compass.reporting import CommandReporter
from uv_torch_compass.uv_commands import UvCommandClient


@dataclass(frozen=True, slots=True)
class ProbeOutcome:
    """Contain the first verified runtime and any required NumPy repair."""

    runtime: RuntimeReport
    compatibility: CompatibilityDecision
    numpy_lt2_required: bool
    attempts: tuple[CandidateAttempt, ...]


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
            outcome = self._probe_candidate(candidate)
            if outcome is None:
                attempts.append(
                    CandidateAttempt(
                        candidate.value,
                        "runtime",
                        "failed",
                        "candidate installation or runtime validation failed",
                        self._compatibility_for(candidate).level.value,
                    )
                )
                continue
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
            )
        attempted = ", ".join(candidate.value for candidate in candidates)
        raise CommandError(
            f"no usable PyTorch backend was found; attempted: {attempted}"
        )

    def _probe_candidate(self, candidate: BackendCandidate) -> ProbeOutcome | None:
        venv = Path(
            tempfile.mkdtemp(
                prefix=f"candidate-{candidate.value}-", dir=self.temporary_root
            )
        )
        self.reporter.info(f"testing backend candidate {candidate.value}")
        created = self.uv.create_venv(
            venv, self.project_python, cwd=self.temporary_root
        )
        self.reporter.detail(created.stdout + created.stderr)
        if created.returncode != 0:
            self.reporter.warn(f"candidate {candidate.value}: venv creation failed")
            return None
        installed = self.uv.install_candidate(
            venv, self.requirements.probe_requirements, candidate
        )
        self.reporter.detail(installed.stdout + installed.stderr)
        if installed.returncode != 0:
            self.reporter.warn(f"candidate {candidate.value}: installation failed")
            return None

        validation = self._run_probe(venv, candidate)
        if validation.returncode == 0:
            return self._parse_success(validation.stdout, candidate, False)
        self.reporter.detail(validation.stdout + validation.stderr)
        if "NUMPY_BRIDGE_FAILED" not in validation.stderr:
            self.reporter.warn(
                f"candidate {candidate.value}: runtime validation failed"
            )
            return None

        self.reporter.warn(
            f"candidate {candidate.value}: retrying the NumPy bridge with numpy<2"
        )
        repaired = self.uv.install_numpy_lt2(venv)
        self.reporter.detail(repaired.stdout + repaired.stderr)
        if repaired.returncode != 0:
            return None
        validation = self._run_probe(venv, candidate)
        self.reporter.detail(validation.stdout + validation.stderr)
        if validation.returncode != 0:
            return None
        return self._parse_success(validation.stdout, candidate, True)

    def _run_probe(self, venv: Path, candidate: BackendCandidate):
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
        if self.requirements.has_package("torchvision"):
            arguments.append("--validate-torchvision")
        if self.requirements.has_package("torchaudio"):
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
    ) -> ProbeOutcome | None:
        try:
            report = RuntimeReport.from_output(output, channel=candidate.channel)
            report.validate_requirements(self.requirements)
            if report.probe_profile != self.probe_profile.value:
                raise ProbeError(
                    f"runtime probe reported profile {report.probe_profile!r}"
                )
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
        return ProbeOutcome(report, compatibility, numpy_lt2_required, ())

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
