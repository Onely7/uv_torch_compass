import pytest

from uv_torch_compass.cuda_compatibility import (
    CompatibilityLevel,
    CompatibilityPolicy,
    CudaRuntimeVersion,
    NvidiaDriverVersion,
    decide_compatibility,
    known_cuda_backends,
    validate_runtime_identity,
)
from uv_torch_compass.errors import ConfigurationError


@pytest.mark.parametrize(
    ("backend", "driver", "maximum", "policy", "expected"),
    [
        (
            "cu124",
            "550.100",
            "12.4",
            CompatibilityPolicy.STRICT,
            CompatibilityLevel.STRICT,
        ),
        (
            "cu129",
            "550.100",
            "12.4",
            CompatibilityPolicy.STRICT,
            CompatibilityLevel.UNSUPPORTED,
        ),
        (
            "cu129",
            "550.100",
            "12.4",
            CompatibilityPolicy.MINOR,
            CompatibilityLevel.MINOR,
        ),
        (
            "cu129",
            "575.57.08",
            "12.9",
            CompatibilityPolicy.STRICT,
            CompatibilityLevel.STRICT,
        ),
        (
            "cu130",
            "575.57.08",
            "12.9",
            CompatibilityPolicy.MINOR,
            CompatibilityLevel.UNSUPPORTED,
        ),
    ],
)
def test_driver_compatibility_boundaries(
    backend: str,
    driver: str,
    maximum: str,
    policy: CompatibilityPolicy,
    expected: CompatibilityLevel,
) -> None:
    decision = decide_compatibility(
        backend,
        driver_version=driver,
        reported_cuda_maximum=maximum,
        policy=policy,
    )

    assert decision.level is expected
    assert decision.allowed is (expected is not CompatibilityLevel.UNSUPPORTED)


def test_unknown_or_malformed_versions_fail_closed() -> None:
    unknown = decide_compatibility(
        "cu132",
        driver_version="600.1",
        reported_cuda_maximum="13.2",
        policy=CompatibilityPolicy.MINOR,
    )
    malformed = decide_compatibility(
        "cu129",
        driver_version="not-a-version",
        reported_cuda_maximum="12.9",
        policy=CompatibilityPolicy.STRICT,
    )

    assert unknown.level is CompatibilityLevel.UNSUPPORTED
    assert malformed.level is CompatibilityLevel.UNSUPPORTED


def test_cuda_value_objects_validate_and_normalize() -> None:
    runtime = CudaRuntimeVersion.from_backend("cu129")

    assert runtime == CudaRuntimeVersion.parse("12.9.79")
    assert runtime.backend == "cu129"
    assert str(runtime) == "12.9"
    assert str(NvidiaDriverVersion.parse("550.100")) == "550.100"
    assert known_cuda_backends()[0] == "cu130"
    with pytest.raises(ConfigurationError):
        CudaRuntimeVersion.from_backend("auto")
    with pytest.raises(ConfigurationError):
        CudaRuntimeVersion.parse("twelve")
    with pytest.raises(ConfigurationError):
        NvidiaDriverVersion.parse("invalid driver")


def test_runtime_identity_requires_known_matching_component() -> None:
    validate_runtime_identity("cu129", cuda_runtime="12.9", runtime_component="12.9.79")

    with pytest.raises(ConfigurationError, match="resolved CUDA runtime"):
        validate_runtime_identity(
            "cu129", cuda_runtime="12.9", runtime_component="12.8.90"
        )
    with pytest.raises(ConfigurationError, match="catalog"):
        validate_runtime_identity(
            "cu132", cuda_runtime="13.2", runtime_component="13.2.1"
        )
