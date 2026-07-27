"""Represent the dependency resolution completed before candidate installation."""

from __future__ import annotations

from dataclasses import dataclass

from uv_torch_compass.candidate_environment import CandidateExecutionEnvironment
from uv_torch_compass.candidate_lock import CandidateLockSnapshot, LockedPackage
from uv_torch_compass.domain import BackendCandidate


@dataclass(frozen=True, slots=True)
class CandidateResolution:
    """Describe one backend's resolved graph before side-effectful installation."""

    backend: BackendCandidate
    environment: CandidateExecutionEnvironment
    lock: CandidateLockSnapshot

    @property
    def pytorch_packages(self) -> tuple[LockedPackage, ...]:
        """Return resolved PyTorch ecosystem package identities."""
        return self.lock.pytorch_packages
