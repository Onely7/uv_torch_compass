"""Build ordered PyTorch backend candidates from host capability and policy."""

from __future__ import annotations

from dataclasses import dataclass

from uv_torch_compass.cuda_compatibility import (
    CompatibilityPolicy,
    known_cuda_backends,
)
from uv_torch_compass.domain import (
    BackendCandidate,
    BackendKind,
    BackendRequest,
    CandidateAttempt,
    Channel,
)
from uv_torch_compass.errors import CommandError
from uv_torch_compass.nvidia import NvidiaSnapshot


@dataclass(frozen=True, slots=True)
class CandidatePlan:
    """Contain ordered candidates and non-fatal discovery diagnostics."""

    candidates: tuple[BackendCandidate, ...]
    skipped: tuple[CandidateAttempt, ...]
    warnings: tuple[str, ...]


def build_candidate_plan(
    request: BackendRequest,
    *,
    channel: Channel,
    advertised_backends: tuple[str, ...],
    nvidia: NvidiaSnapshot | None,
    compatibility_policy: CompatibilityPolicy,
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
        advertised_cuda = known_cuda_backends()
        warnings.append(
            "uv did not advertise CUDA backends; using the curated fallback list"
        )
    compatible_cuda: list[str] = []
    skipped: list[CandidateAttempt] = []
    if nvidia is not None:
        for value in advertised_cuda:
            decision = nvidia.compatibility_for(value, compatibility_policy)
            if decision.allowed:
                compatible_cuda.append(value)
                continue
            skipped.append(
                CandidateAttempt(
                    value,
                    "policy",
                    "skipped",
                    decision.reason,
                    decision.level.value,
                )
            )

    if request.kind is BackendKind.CPU:
        values = ("cpu",)
    elif request.kind is BackendKind.CONCRETE:
        if nvidia is None:
            raise CommandError("a concrete CUDA backend requires a visible NVIDIA GPU")
        decision = nvidia.compatibility_for(
            request.concrete_value, compatibility_policy
        )
        if not decision.allowed:
            raise CommandError(
                f"backend {request.concrete_value} is not allowed by "
                f"{compatibility_policy.value} compatibility: {decision.reason}"
            )
        values = (request.concrete_value,)
    elif request.kind is BackendKind.CUDA:
        if nvidia is None:
            raise CommandError("--backend cuda requires a visible NVIDIA GPU")
        if not compatible_cuda:
            raise CommandError(_no_cuda_candidate_message(compatibility_policy))
        values = tuple(compatible_cuda)
    elif nvidia is None:
        values = ("cpu",)
    else:
        if not compatible_cuda:
            raise CommandError(_no_cuda_candidate_message(compatibility_policy))
        values = tuple(compatible_cuda)

    candidates = tuple(
        BackendCandidate(value, channel) for value in dict.fromkeys(values)
    )
    return CandidatePlan(candidates, tuple(skipped), tuple(warnings))


def _cuda_sort_key(value: str) -> int:
    try:
        return int(value.removeprefix("cu"))
    except ValueError:
        return -1


def _no_cuda_candidate_message(policy: CompatibilityPolicy) -> str:
    return (
        f"no CUDA backend satisfies {policy.value} compatibility; update the NVIDIA "
        "driver, relax the PyTorch requirement, select --backend cpu, or explicitly "
        "use --cuda-compatibility minor"
    )
