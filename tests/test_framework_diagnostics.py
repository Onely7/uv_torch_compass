import pytest

from uv_torch_compass.candidate_environment import CandidateExecutionEnvironment
from uv_torch_compass.candidate_failures import (
    BoundedExceptionReport,
    FrameworkCompatibilityDecision,
    FrameworkCompatibilityStatus,
    FrameworkFailureKind,
    FrameworkProbeTrigger,
)
from uv_torch_compass.candidate_lock import CandidateLockSnapshot, LockedPackage
from uv_torch_compass.candidate_resolution import CandidateResolution
from uv_torch_compass.domain import BackendCandidate, FrameworkProbe
from uv_torch_compass.framework_diagnostics import (
    artifact_failure,
    catalog_dependency_failure,
    validation_failure,
)
from uv_torch_compass.framework_validation import FrameworkValidation
from uv_torch_compass.vllm_compatibility import catalog_requirement


def _resolution() -> CandidateResolution:
    packages = (
        LockedPackage("uv-torch-compass-candidate", "0", "", ("vllm",), "virtual"),
        LockedPackage(
            "vllm",
            "0.6.0",
            "https://pypi.org/simple",
            ("torch", "transformers"),
        ),
        LockedPackage("transformers", "5.14.1", "https://pypi.org/simple", ("torch",)),
        LockedPackage("torch", "2.4.0", "https://download.pytorch.org/whl/cu121", ()),
    )
    return CandidateResolution(
        BackendCandidate("cu121"),
        CandidateExecutionEnvironment("3.12.1", "cpython", "linux", "x86_64"),
        CandidateLockSnapshot("uv-torch-compass-candidate", packages),
    )


def test_catalog_dependency_failure_reports_paths_and_constraint() -> None:
    failure = catalog_dependency_failure(_resolution())

    assert failure is not None
    assert failure.kind is FrameworkFailureKind.API_INCOMPATIBILITY
    assert failure.backend_independent
    assert any("transformers==5.14.1" in path for path in failure.dependency_paths)
    assert "4.44.2" in failure.suggestions[0]


def test_artifact_failure_retains_catalog_requirement() -> None:
    resolution = _resolution()
    vllm = resolution.lock.package("vllm")
    assert vllm is not None
    requirement = catalog_requirement(vllm)
    decision = FrameworkCompatibilityDecision(
        FrameworkCompatibilityStatus.INCOMPATIBLE,
        "cu129",
        "vLLM 0.6.0 requires cu121",
        requirement,
    )

    failure = artifact_failure(resolution, decision)

    assert failure.kind is FrameworkFailureKind.CUDA_ABI
    assert failure.binary_requirement is requirement
    assert any("--backend cu121" in item for item in failure.suggestions)


@pytest.mark.parametrize(
    ("error", "import_test", "native_test", "platform_test", "expected"),
    [
        (
            "ImportError: cannot import name 'DTensor' from torch.distributed.tensor",
            "FAIL",
            "FAIL",
            "FAIL",
            FrameworkFailureKind.API_INCOMPATIBILITY,
        ),
        (
            "ImportError: libcudart.so.13: cannot open shared object file",
            "FAIL",
            "FAIL",
            "FAIL",
            FrameworkFailureKind.CUDA_ABI,
        ),
        (
            "ImportError: undefined symbol: _ZN5torch",
            "FAIL",
            "FAIL",
            "FAIL",
            FrameworkFailureKind.BINARY_INCOMPATIBILITY,
        ),
        (
            "RuntimeError: CUDA platform mismatch",
            "PASS",
            "PASS",
            "FAIL",
            FrameworkFailureKind.PLATFORM,
        ),
        (
            "RuntimeError: native extension missing",
            "PASS",
            "FAIL",
            "FAIL",
            FrameworkFailureKind.NATIVE_EXTENSION,
        ),
        (
            "ModuleNotFoundError: No module named 'example'",
            "FAIL",
            "FAIL",
            "FAIL",
            FrameworkFailureKind.IMPORT,
        ),
    ],
)
def test_runtime_framework_failures_are_classified(
    error: str,
    import_test: str,
    native_test: str,
    platform_test: str,
    expected: FrameworkFailureKind,
) -> None:
    exception_type, _, message = error.partition(":")
    validation = FrameworkValidation(
        FrameworkProbe.VLLM,
        "FAIL",
        "0.6.0",
        import_test,
        native_test,
        platform_test,
        "CudaPlatform",
        error,
        FrameworkProbeTrigger.AUTOMATIC,
        BoundedExceptionReport(exception_type, message.strip()),
    )

    failure = validation_failure(validation, _resolution())

    assert failure.kind is expected
    assert failure.exception is not None
