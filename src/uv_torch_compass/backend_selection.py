"""Build ordered PyTorch backend candidates from host capability and policy."""

from __future__ import annotations

from dataclasses import dataclass

from uv_torch_compass.domain import (
    BackendCandidate,
    BackendKind,
    BackendRequest,
    Channel,
)
from uv_torch_compass.errors import CommandError
from uv_torch_compass.nvidia import NvidiaSnapshot

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


def _cuda_sort_key(value: str) -> int:
    try:
        return int(value.removeprefix("cu"))
    except ValueError:
        return -1
