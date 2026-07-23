"""Build backend candidates and verify each in an isolated environment."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from uv_torch_compass.command_runner import ProcessRunner, sanitized_environment
from uv_torch_compass.domain import (
    BackendCandidate,
    BackendKind,
    BackendRequest,
    CandidateAttempt,
    Channel,
    ProjectRequirements,
    RuntimeReport,
)
from uv_torch_compass.errors import CommandError, ProbeError
from uv_torch_compass.nvidia import NvidiaSnapshot
from uv_torch_compass.reporting import CommandReporter
from uv_torch_compass.uv_commands import UvCommandClient

_CURATED_CUDA_BACKENDS = (
    "cu130",
    "cu129",
    "cu128",
    "cu126",
    "cu125",
    "cu124",
    "cu123",
    "cu122",
    "cu121",
    "cu120",
    "cu118",
)


@dataclass(frozen=True, slots=True)
class CandidatePlan:
    """Contain ordered candidates and non-fatal discovery diagnostics."""

    candidates: tuple[BackendCandidate, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProbeOutcome:
    """Contain the first verified runtime and any required NumPy repair."""

    runtime: RuntimeReport
    numpy_lt2_required: bool
    attempts: tuple[CandidateAttempt, ...]


def build_candidate_plan(
    request: BackendRequest,
    *,
    channel: Channel,
    advertised_backends: tuple[str, ...],
    nvidia: NvidiaSnapshot | None,
) -> CandidatePlan:
    """Build a deterministic candidate sequence from policy and host capability.

    Raises:
        CommandError: If CUDA is required but no usable CUDA candidate exists.
    """
    warnings: list[str] = []
    advertised_cuda = tuple(
        sorted(
            (value for value in advertised_backends if value.startswith("cu")),
            key=_cuda_sort_key,
            reverse=True,
        )
    )
    if not advertised_cuda:
        advertised_cuda = _CURATED_CUDA_BACKENDS
        warnings.append(
            "uv did not advertise CUDA backends; using the curated fallback list"
        )
    compatible_cuda = tuple(
        value
        for value in advertised_cuda
        if nvidia is not None and nvidia.supports_backend(value)
    )

    if request.kind is BackendKind.CPU:
        values = ("cpu",)
    elif request.kind is BackendKind.CONCRETE:
        if nvidia is None:
            raise CommandError("a concrete CUDA backend requires a visible NVIDIA GPU")
        if not nvidia.supports_backend(request.concrete_value):
            raise CommandError(
                f"backend {request.concrete_value} exceeds the CUDA version "
                "supported by the visible NVIDIA driver"
            )
        values = (request.concrete_value,)
    elif request.kind is BackendKind.CUDA:
        if nvidia is None:
            raise CommandError("--backend cuda requires a visible NVIDIA GPU")
        if not compatible_cuda:
            raise CommandError("no CUDA backend is compatible with the visible driver")
        values = compatible_cuda
    elif channel is Channel.NIGHTLY:
        values = (*compatible_cuda, "cpu") if nvidia is not None else ("cpu",)
    else:
        values = (
            ("auto", *compatible_cuda, "cpu") if nvidia is not None else ("auto", "cpu")
        )

    candidates = tuple(
        BackendCandidate(value, channel)
        for value in dict.fromkeys(values)
        if not (channel is Channel.NIGHTLY and value == "auto")
    )
    return CandidatePlan(candidates, tuple(warnings))


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

    def find_working_candidate(
        self, candidates: tuple[BackendCandidate, ...]
    ) -> ProbeOutcome:
        """Return the first verified candidate.

        Raises:
            CommandError: If every candidate fails installation or runtime checks.
        """
        attempts: list[CandidateAttempt] = []
        for candidate in candidates:
            outcome = self._probe_candidate(candidate)
            if outcome is None:
                attempts.append(CandidateAttempt(candidate.value, False, "failed"))
                continue
            attempts.append(
                CandidateAttempt(
                    candidate.value,
                    True,
                    f"resolved as {outcome.runtime.backend.value}",
                )
            )
            return ProbeOutcome(
                outcome.runtime, outcome.numpy_lt2_required, tuple(attempts)
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
        if expected:
            arguments.extend(["--expected-backend", expected])
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
        except ProbeError as exc:
            self.reporter.warn(f"candidate {candidate.value}: {exc}")
            return None
        if candidate.is_concrete and report.backend.value != candidate.value:
            self.reporter.warn(
                f"candidate {candidate.value}: runtime reported {report.backend.value}"
            )
            return None
        return ProbeOutcome(report, numpy_lt2_required, ())


def _cuda_sort_key(value: str) -> int:
    try:
        return int(value.removeprefix("cu"))
    except ValueError:
        return -1
