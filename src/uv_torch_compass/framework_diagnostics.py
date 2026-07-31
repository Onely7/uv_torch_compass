"""Translate framework artifact and runtime evidence into stable diagnostics."""

from __future__ import annotations

import re

from uv_torch_compass.candidate_failures import (
    BoundedExceptionReport,
    FailedPackage,
    FrameworkCompatibilityDecision,
    FrameworkFailure,
    FrameworkFailureKind,
    FrameworkPackageVersion,
)
from uv_torch_compass.candidate_resolution import CandidateResolution
from uv_torch_compass.framework_validation import FrameworkValidation
from uv_torch_compass.vllm_compatibility import dependency_advisories

_CUDA_LIBRARY = re.compile(r"libcudart\.so\.(?P<major>[0-9]+)")


def framework_versions(
    resolution: CandidateResolution,
) -> tuple[FrameworkPackageVersion, ...]:
    """Return resolved framework package versions needed by public diagnostics."""
    return tuple(
        FrameworkPackageVersion(package.name, package.version, package.source_url)
        for package in resolution.framework_packages
    )


def artifact_failure(
    resolution: CandidateResolution,
    decision: FrameworkCompatibilityDecision,
) -> FrameworkFailure:
    """Build a failure for a vLLM artifact rejected before full installation."""
    vllm = resolution.lock.package("vllm")
    version = vllm.version if vllm is not None else "unknown"
    requirement = decision.requirement
    suggestions = _cuda_suggestions(
        requirement.required_cuda_variant if requirement else None
    )
    return FrameworkFailure(
        FrameworkFailureKind.CUDA_ABI,
        decision.summary,
        "vllm",
        version,
        FailedPackage("vllm", version, f"vllm=={version}"),
        resolution.dependency_paths("vllm"),
        requirement,
        packages=framework_versions(resolution),
        suggestions=suggestions,
    )


def catalog_dependency_failure(
    resolution: CandidateResolution,
) -> FrameworkFailure | None:
    """Return a reviewed framework API incompatibility in the locked graph."""
    vllm = resolution.lock.package("vllm")
    if vllm is None:
        return None
    matches = dependency_advisories(vllm, resolution.framework_packages)
    if not matches:
        return None
    advisory, package = matches[0]
    paths = tuple(
        dict.fromkeys(
            (
                *resolution.dependency_paths(package.name),
                *resolution.dependency_paths("torch"),
            )
        )
    )
    return FrameworkFailure(
        FrameworkFailureKind.API_INCOMPATIBILITY,
        advisory.summary,
        "vllm",
        vllm.version,
        FailedPackage(
            package.name, package.version, f"{package.name}=={package.version}"
        ),
        paths,
        packages=framework_versions(resolution),
        suggestions=advisory.suggestions,
        backend_independent=True,
    )


def validation_failure(
    validation: FrameworkValidation,
    resolution: CandidateResolution,
) -> FrameworkFailure:
    """Classify one failed framework runtime result without guessing versions."""
    exception = validation.exception or _legacy_exception(validation.error)
    message = exception.message if exception is not None else validation.error
    kind, backend_independent = _classify(validation, exception, message)
    package_name = (
        exception.consumer_package
        if exception is not None and exception.consumer_package
        else validation.framework.value
    )
    locked = resolution.lock.package(package_name)
    package_version = locked.version if locked is not None else None
    paths = resolution.dependency_paths(package_name)
    suggestions: tuple[str, ...] = ()
    reviewed = catalog_dependency_failure(resolution)
    if kind is FrameworkFailureKind.API_INCOMPATIBILITY and reviewed is not None:
        suggestions = reviewed.suggestions
        paths = reviewed.dependency_paths or paths
    return FrameworkFailure(
        kind,
        _summary(kind, validation, message),
        validation.framework.value,
        validation.version,
        FailedPackage(package_name, package_version, package_name),
        paths,
        exception=exception,
        packages=validation.packages or framework_versions(resolution),
        suggestions=suggestions,
        backend_independent=backend_independent,
    )


def _classify(
    validation: FrameworkValidation,
    exception: BoundedExceptionReport | None,
    message: str,
) -> tuple[FrameworkFailureKind, bool]:
    if "DTensor" in message and (
        "torch.distributed.tensor" in message
        or (exception is not None and exception.missing_symbol == "DTensor")
    ):
        return FrameworkFailureKind.API_INCOMPATIBILITY, True
    if _CUDA_LIBRARY.search(message):
        return FrameworkFailureKind.CUDA_ABI, False
    if "undefined symbol" in message.lower():
        return FrameworkFailureKind.BINARY_INCOMPATIBILITY, False
    if validation.native_extension_test == "FAIL" and validation.import_test == "PASS":
        return FrameworkFailureKind.NATIVE_EXTENSION, False
    if validation.platform_test == "FAIL" and validation.import_test == "PASS":
        return FrameworkFailureKind.PLATFORM, False
    if exception is not None and exception.exception_type in {
        "ImportError",
        "ModuleNotFoundError",
    }:
        return FrameworkFailureKind.IMPORT, True
    return FrameworkFailureKind.IMPORT, False


def _summary(
    kind: FrameworkFailureKind,
    validation: FrameworkValidation,
    message: str,
) -> str:
    if kind is FrameworkFailureKind.API_INCOMPATIBILITY:
        return f"{validation.framework.value} failed because resolved Python APIs are incompatible: {message}"
    if kind is FrameworkFailureKind.CUDA_ABI:
        return f"{validation.framework.value} requires an incompatible CUDA runtime library: {message}"
    if kind is FrameworkFailureKind.BINARY_INCOMPATIBILITY:
        return f"{validation.framework.value} native libraries are binary-incompatible: {message}"
    if kind is FrameworkFailureKind.PLATFORM:
        return f"{validation.framework.value} selected an incompatible execution platform: {message}"
    return (
        f"{validation.framework.value} validation failed: {message or 'unknown error'}"
    )


def _legacy_exception(error: str) -> BoundedExceptionReport | None:
    if not error:
        return None
    exception_type, separator, message = error.partition(":")
    return BoundedExceptionReport(
        exception_type.strip() if separator else "FrameworkError",
        message.strip() if separator else error,
        missing_symbol="DTensor" if "DTensor" in error else None,
        provider_package="torch" if "torch" in error or "DTensor" in error else None,
    )


def _cuda_suggestions(required_variant: str | None) -> tuple[str, ...]:
    suggestions = ["Select a vLLM version built for an allowed CUDA backend."]
    if required_variant:
        suggestions.append(
            f"Use --backend {required_variant} after confirming the NVIDIA driver supports it."
        )
    suggestions.append(
        "Build vLLM from source for the selected PyTorch and CUDA stack."
    )
    return tuple(suggestions)
