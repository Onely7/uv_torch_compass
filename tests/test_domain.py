import json

import pytest

from uv_torch_compass.domain import (
    BackendCandidate,
    BackendKind,
    BackendRequest,
    Channel,
    ProbeProfile,
    ProjectRequirements,
    RuntimeReport,
    Scope,
    ScopedRequirement,
    has_upper_bound,
)
from uv_torch_compass.errors import ConfigurationError, ProbeError


@pytest.mark.parametrize(
    "raw, kind, concrete",
    [
        ("AUTO", BackendKind.AUTO, ""),
        ("cpu", BackendKind.CPU, ""),
        ("cuda", BackendKind.CUDA, ""),
        ("CU128", BackendKind.CONCRETE, "cu128"),
    ],
)
def test_backend_request_normalizes_public_values(raw, kind, concrete) -> None:
    request = BackendRequest.parse(raw)
    assert request.kind is kind
    assert request.concrete_value == concrete


@pytest.mark.parametrize("raw", ["", "auto-cpu", "rocm6.2", "xpu", "cu12x"])
def test_backend_request_rejects_unsupported_values(raw: str) -> None:
    with pytest.raises(ConfigurationError):
        BackendRequest.parse(raw)


def test_nightly_candidate_uses_official_nightly_index() -> None:
    candidate = BackendCandidate("cu128", Channel.NIGHTLY)
    assert candidate.index_name == "pytorch-nightly-cu128"
    assert candidate.index_url == "https://download.pytorch.org/whl/nightly/cu128"


def test_candidate_requirements_include_all_selected_roots() -> None:
    scope = Scope("base")
    requirements = ProjectRequirements(
        ">=3.10",
        "",
        (
            ScopedRequirement(scope, "vllm==0.19.1"),
            ScopedRequirement(scope, "torch"),
            ScopedRequirement(scope, "torchvision"),
        ),
        (),
        (scope,),
    )

    assert requirements.probe_requirements == (
        "vllm==0.19.1",
        "torch",
        "torchvision",
    )


def test_runtime_report_validates_schema_and_requirement_versions() -> None:
    output = json.dumps(
        {
            "schema_version": 2,
            "backend": "cpu",
            "torch_version": "2.7.0+cpu",
            "torchvision_version": "not-installed",
            "torchaudio_version": "not-installed",
            "numpy_version": "2.2.0",
            "cuda_runtime": "none",
            "runtime_component_version": "not-installed",
            "gpu_name": "none",
            "gpu_device_capability": "none",
            "compiled_architectures": [],
            "native_architecture_test": "NOT_APPLICABLE",
            "cuda_test": "NOT_APPLICABLE",
            "cublas_test": "NOT_APPLICABLE",
            "cudnn_test": "NOT_APPLICABLE",
            "numpy_bridge_test": "PASS",
            "torchvision_test": "NOT_REQUESTED",
            "torchaudio_test": "NOT_REQUESTED",
            "compile_test": "NOT_REQUESTED",
            "probe_profile": "standard",
        }
    )
    report = RuntimeReport.from_output(output)
    requirements = ProjectRequirements(
        ">=3.10",
        "",
        (ScopedRequirement(Scope("base"), "torch>=2.6,<3"),),
        (),
        (Scope("base"),),
    )

    report.validate_requirements(requirements)
    report.validate_probe_results(
        requirements,
        expected_profile=ProbeProfile.STANDARD,
        require_native_architecture=False,
    )
    assert report.backend.value == "cpu"


def test_runtime_report_rejects_wrong_schema() -> None:
    with pytest.raises(ProbeError, match="schema"):
        RuntimeReport.from_output('{"schema_version": 1}')


def test_runtime_report_rejects_invalid_compiled_architectures() -> None:
    invalid = {
        "schema_version": 2,
        "backend": "cpu",
        "torch_version": "2.7.0",
        "torchvision_version": "not-installed",
        "torchaudio_version": "not-installed",
        "numpy_version": "2.2.0",
        "cuda_runtime": "none",
        "runtime_component_version": "not-installed",
        "gpu_name": "none",
        "gpu_device_capability": "none",
        "compiled_architectures": "sm_89",
        "native_architecture_test": "NOT_APPLICABLE",
        "cuda_test": "NOT_APPLICABLE",
        "cublas_test": "NOT_APPLICABLE",
        "cudnn_test": "NOT_APPLICABLE",
        "numpy_bridge_test": "PASS",
        "torchvision_test": "NOT_REQUESTED",
        "torchaudio_test": "NOT_REQUESTED",
        "compile_test": "NOT_REQUESTED",
        "probe_profile": "standard",
    }

    with pytest.raises(ProbeError, match="compiled_architectures"):
        RuntimeReport.from_output(json.dumps(invalid))


def test_runtime_report_rejects_spoofed_validation_result() -> None:
    report = RuntimeReport(
        2,
        BackendCandidate("cu128"),
        "2.7.0",
        "not-installed",
        "not-installed",
        "2.2.0",
        "12.8",
        "Fake GPU",
        "PASS",
        "PASS",
        "NOT_REQUESTED",
        "NOT_REQUESTED",
        "12.8.90",
        "8.9",
        ("sm_89",),
        "PASS",
        "FAIL",
        "PASS",
        "NOT_REQUESTED",
        "standard",
    )
    requirements = ProjectRequirements(
        ">=3.10",
        "",
        (ScopedRequirement(Scope("base"), "torch>=2.6"),),
        (),
        (Scope("base"),),
    )

    with pytest.raises(ProbeError, match="cublas_test"):
        report.validate_probe_results(
            requirements,
            expected_profile=ProbeProfile.STANDARD,
            require_native_architecture=False,
        )


@pytest.mark.parametrize(
    "specifier, expected",
    [("~=3.10", False), (">=3.10,<3.15", True), (">=3.10,<=3.14", True)],
)
def test_upper_bound_detection_is_semantic(specifier: str, expected: bool) -> None:
    assert has_upper_bound(specifier) is expected


def test_requirements_filter_markers_for_resolved_interpreter() -> None:
    scope = Scope("base")
    requirements = ProjectRequirements(
        ">=3.10,<3.15",
        "",
        (
            ScopedRequirement(scope, 'torch>=2.6; python_version >= "3.12"'),
            ScopedRequirement(scope, 'torchvision; implementation_name == "cpython"'),
            ScopedRequirement(scope, 'torchaudio; implementation_name == "pypy"'),
        ),
        (),
        (scope,),
    )

    selected = requirements.for_interpreter("3.12.4", "cpython", "CPython")

    assert [item.package for item in selected.selected] == ["torch", "torchvision"]


def test_direct_pytorch_url_and_inapplicable_markers_are_rejected() -> None:
    scope = Scope("base")
    with pytest.raises(ConfigurationError, match="direct URL"):
        ScopedRequirement(scope, "torch @ https://example.invalid/torch.whl")

    requirements = ProjectRequirements(
        ">=3.10",
        "",
        (ScopedRequirement(scope, 'torch; python_version < "3.10"'),),
        (),
        (scope,),
    )
    with pytest.raises(ConfigurationError, match="no selected"):
        requirements.for_interpreter("3.12.0", "cpython", "CPython")
