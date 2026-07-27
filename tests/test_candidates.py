import json
from pathlib import Path
from typing import cast

import pytest

from uv_torch_compass.backend_selection import build_candidate_plan
from uv_torch_compass.candidate_probe import CandidateProbeService
from uv_torch_compass.command_runner import CommandResult, ProcessRunner
from uv_torch_compass.cuda_compatibility import CompatibilityPolicy
from uv_torch_compass.domain import (
    BackendCandidate,
    BackendRequest,
    Channel,
    ProbeProfile,
    ProjectRequirements,
    Scope,
    ScopedRequirement,
)
from uv_torch_compass.errors import (
    CandidateResolutionError,
    CommandError,
    CommandTimeoutError,
)
from uv_torch_compass.nvidia import NvidiaDevice, NvidiaSnapshot
from uv_torch_compass.reporting import CommandReporter
from uv_torch_compass.uv_commands import UvCommandClient


def _nvidia(cuda_max: str = "12.8") -> NvidiaSnapshot:
    device = NvidiaDevice("0", "GPU-123", "Test GPU", "570.124.06")
    return NvidiaSnapshot((device,), cuda_max, device)


def test_auto_orders_strict_concrete_cuda_without_cpu_fallback() -> None:
    plan = build_candidate_plan(
        BackendRequest.parse("auto"),
        channel=Channel.STABLE,
        advertised_backends=("auto", "cpu", "cu130", "cu128", "cu121", "rocm6.2"),
        nvidia=_nvidia(),
        compatibility_policy=CompatibilityPolicy.STRICT,
    )

    assert [candidate.value for candidate in plan.candidates] == [
        "cu128",
        "cu121",
    ]
    assert plan.skipped[0].backend == "cu130"


def test_driver_550_selects_cu124_without_installing_newer_backends() -> None:
    device = NvidiaDevice("0", "GPU-123", "Ada GPU", "550.100")
    plan = build_candidate_plan(
        BackendRequest.parse("auto"),
        channel=Channel.STABLE,
        advertised_backends=("auto", "cpu", "cu129", "cu128", "cu124", "cu121"),
        nvidia=NvidiaSnapshot((device,), "12.4", device),
        compatibility_policy=CompatibilityPolicy.STRICT,
    )

    assert [candidate.value for candidate in plan.candidates] == ["cu124", "cu121"]
    assert [attempt.backend for attempt in plan.skipped] == ["cu129", "cu128"]


def test_minor_policy_allows_newer_same_family_backend() -> None:
    device = NvidiaDevice("0", "GPU-123", "Ada GPU", "550.100")
    plan = build_candidate_plan(
        BackendRequest.parse("auto"),
        channel=Channel.STABLE,
        advertised_backends=("auto", "cpu", "cu129", "cu124"),
        nvidia=NvidiaSnapshot((device,), "12.4", device),
        compatibility_policy=CompatibilityPolicy.MINOR,
    )

    assert [candidate.value for candidate in plan.candidates] == ["cu129", "cu124"]


def test_probe_rejects_runtime_without_an_allowed_driver(
    tmp_path: Path,
) -> None:
    """Characterize the legacy gap between uv auto and concrete filtering."""
    reporter = ProbeReporter()
    service = _service(
        tmp_path,
        ProbeUv(),
        ProbeRunner([CommandResult(0, _report("cu129"), "")]),
        reporter,
    )

    with pytest.raises(CommandError, match="no usable"):
        service.find_working_candidate((BackendCandidate("auto"),))


def test_nightly_auto_uses_strict_concrete_candidates_only() -> None:
    plan = build_candidate_plan(
        BackendRequest.parse("auto"),
        channel=Channel.NIGHTLY,
        advertised_backends=("auto", "cpu", "cu128"),
        nvidia=_nvidia(),
        compatibility_policy=CompatibilityPolicy.STRICT,
    )
    assert [candidate.value for candidate in plan.candidates] == ["cu128"]


def test_cuda_policy_requires_visible_nvidia() -> None:
    with pytest.raises(CommandError, match="visible NVIDIA"):
        build_candidate_plan(
            BackendRequest.parse("cuda"),
            channel=Channel.STABLE,
            advertised_backends=("cpu", "cu128"),
            nvidia=None,
            compatibility_policy=CompatibilityPolicy.STRICT,
        )


def test_auto_with_visible_gpu_does_not_fall_back_to_cpu() -> None:
    device = NvidiaDevice("0", "GPU-123", "Old GPU", "520.1")
    with pytest.raises(CommandError, match="select --backend cpu"):
        build_candidate_plan(
            BackendRequest.parse("auto"),
            channel=Channel.STABLE,
            advertised_backends=("auto", "cpu", "cu129"),
            nvidia=NvidiaSnapshot((device,), "11.8", device),
            compatibility_policy=CompatibilityPolicy.STRICT,
        )


def test_candidate_policies_cover_cpu_concrete_and_curated_fallback() -> None:
    cpu = build_candidate_plan(
        BackendRequest.parse("cpu"),
        channel=Channel.STABLE,
        advertised_backends=(),
        nvidia=None,
        compatibility_policy=CompatibilityPolicy.STRICT,
    )
    concrete = build_candidate_plan(
        BackendRequest.parse("cu128"),
        channel=Channel.STABLE,
        advertised_backends=("cpu", "cu128"),
        nvidia=_nvidia(),
        compatibility_policy=CompatibilityPolicy.STRICT,
    )

    assert [item.value for item in cpu.candidates] == ["cpu"]
    assert cpu.warnings
    assert [item.value for item in concrete.candidates] == ["cu128"]
    with pytest.raises(CommandError, match="no CUDA backend"):
        build_candidate_plan(
            BackendRequest.parse("cuda"),
            channel=Channel.STABLE,
            advertised_backends=("cu130",),
            nvidia=_nvidia(),
            compatibility_policy=CompatibilityPolicy.STRICT,
        )
    with pytest.raises(CommandError, match="not allowed"):
        build_candidate_plan(
            BackendRequest.parse("cu130"),
            channel=Channel.STABLE,
            advertised_backends=("cu130",),
            nvidia=_nvidia(),
            compatibility_policy=CompatibilityPolicy.STRICT,
        )


def _report(backend: str = "cpu") -> str:
    return json.dumps(
        {
            "schema_version": 2,
            "backend": backend,
            "torch_version": "2.7.0",
            "torchvision_version": "not-installed",
            "torchaudio_version": "not-installed",
            "numpy_version": "1.26.4",
            "cuda_runtime": "none" if backend == "cpu" else "12.8",
            "runtime_component_version": (
                "not-installed" if backend == "cpu" else "12.8.90"
            ),
            "gpu_name": "none" if backend == "cpu" else "Fake GPU",
            "gpu_device_capability": "none" if backend == "cpu" else "8.9",
            "compiled_architectures": [] if backend == "cpu" else ["sm_89"],
            "native_architecture_test": (
                "NOT_APPLICABLE" if backend == "cpu" else "PASS"
            ),
            "cuda_test": "NOT_APPLICABLE" if backend == "cpu" else "PASS",
            "cublas_test": "NOT_APPLICABLE" if backend == "cpu" else "PASS",
            "cudnn_test": "NOT_APPLICABLE" if backend == "cpu" else "PASS",
            "numpy_bridge_test": "PASS",
            "torchvision_test": "NOT_REQUESTED",
            "torchaudio_test": "NOT_REQUESTED",
            "compile_test": "NOT_REQUESTED",
            "probe_profile": "standard",
        }
    )


class ProbeUv:
    diagnostic_timeout_seconds = 30
    heavy_timeout_seconds = 1800

    def __init__(
        self,
        *,
        create_codes: list[int] | None = None,
        install_codes: list[int] | None = None,
        numpy_code: int = 0,
        install_error: str = "install failed",
    ) -> None:
        self.create_codes = create_codes or [0]
        self.install_codes = install_codes or [0]
        self.numpy_code = numpy_code
        self.install_error = install_error

    def create_venv(self, path: Path, python: Path, *, cwd: Path) -> CommandResult:
        del path, python, cwd
        return CommandResult(self.create_codes.pop(0), "", "venv failed")

    def install_candidate(
        self, path: Path, requirements, candidate, *, dry_run: bool = False
    ) -> CommandResult:
        del requirements, candidate, dry_run
        metadata = path / "lib/python/site-packages/torch-2.7.0.dist-info/METADATA"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text("Name: torch\nVersion: 2.7.0\n", encoding="utf-8")
        return CommandResult(self.install_codes.pop(0), "", self.install_error)

    def install_numpy_lt2(self, path: Path) -> CommandResult:
        del path
        return CommandResult(self.numpy_code, "", "repair failed")


class ProbeRunner:
    def __init__(self, results: list[CommandResult]) -> None:
        self.results = results

    def run(self, arguments, *, cwd=None, env=None, timeout_seconds=None):
        del arguments, cwd, env, timeout_seconds
        return self.results.pop(0)


class ProbeReporter:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def info(self, message: str) -> None:
        del message

    def detail(self, message: str) -> None:
        del message

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def _service(
    tmp_path: Path, uv: ProbeUv, runner: ProbeRunner, reporter: ProbeReporter
) -> CandidateProbeService:
    scope = Scope("base")
    requirements = ProjectRequirements(
        ">=3.10",
        "",
        (ScopedRequirement(scope, "torch>=2.6"),),
        (),
        (scope,),
    )
    return CandidateProbeService(
        cast(UvCommandClient, uv),
        cast(ProcessRunner, runner),
        cast(CommandReporter, reporter),
        tmp_path,
        Path("/usr/bin/python3"),
        requirements,
        Path("/probe.py"),
        None,
        None,
        CompatibilityPolicy.STRICT,
        ProbeProfile.STANDARD,
    )


def test_probe_retries_numpy_and_records_success(tmp_path: Path) -> None:
    reporter = ProbeReporter()
    service = _service(
        tmp_path,
        ProbeUv(),
        ProbeRunner(
            [
                CommandResult(22, "", "NUMPY_BRIDGE_FAILED: incompatible"),
                CommandResult(0, _report(), ""),
            ]
        ),
        reporter,
    )

    outcome = service.find_working_candidate((BackendCandidate("cpu"),))

    assert outcome.numpy_lt2_required
    assert outcome.attempts[0].status == "passed"
    assert "retrying" in reporter.warnings[0]


def test_probe_rejects_failed_setup_and_invalid_runtime(tmp_path: Path) -> None:
    reporter = ProbeReporter()
    service = _service(
        tmp_path,
        ProbeUv(create_codes=[1, 0, 0], install_codes=[1, 0]),
        ProbeRunner([CommandResult(0, "not-json", "")]),
        reporter,
    )

    with pytest.raises(CommandError, match="no usable"):
        service.find_working_candidate(
            (
                BackendCandidate("cu128"),
                BackendCandidate("cu121"),
                BackendCandidate("cpu"),
            )
        )

    assert any("venv creation" in warning for warning in reporter.warnings)
    assert any("installation" in warning for warning in reporter.warnings)
    assert any("valid JSON" in warning for warning in reporter.warnings)


def test_install_failure_is_currently_reported_without_package_context(
    tmp_path: Path,
) -> None:
    reporter = ProbeReporter()
    service = _service(
        tmp_path,
        ProbeUv(install_codes=[1]),
        ProbeRunner([]),
        reporter,
    )

    with pytest.raises(
        CommandError,
        match=r"no usable PyTorch backend was found; attempted: cu121",
    ):
        service.find_working_candidate((BackendCandidate("cu121"),))

    assert reporter.warnings == ["candidate cu121: installation failed"]


def test_install_failure_preserves_structured_resolution_context(
    tmp_path: Path,
) -> None:
    service = _service(
        tmp_path,
        ProbeUv(
            install_codes=[1],
            install_error=(
                "Because vllm==0.25.0 depends on torch==2.10.0 and "
                "no version of torch==2.10.0 is available"
            ),
        ),
        ProbeRunner([]),
        ProbeReporter(),
    )
    service.requirements = ProjectRequirements(
        ">=3.10",
        "",
        (ScopedRequirement(Scope("base"), "vllm>=0.25.0"),),
        (),
        (Scope("base"),),
    )

    with pytest.raises(CandidateResolutionError) as captured:
        service.find_working_candidate((BackendCandidate("cu121"),))

    attempts = captured.value.attempts
    assert len(attempts) == 1
    failure = attempts[0].failure
    assert failure is not None
    assert failure.package is not None
    assert failure.package.name == "torch"
    assert failure.index is not None
    assert failure.index.name == "pytorch-cu121"


def test_install_timeout_is_preserved_as_candidate_diagnostic(
    tmp_path: Path,
) -> None:
    class TimeoutUv(ProbeUv):
        def install_candidate(
            self, path: Path, requirements, candidate, *, dry_run: bool = False
        ) -> CommandResult:
            del path, requirements, candidate, dry_run
            raise CommandTimeoutError("timed out")

    service = _service(
        tmp_path,
        TimeoutUv(),
        ProbeRunner([]),
        ProbeReporter(),
    )

    with pytest.raises(CandidateResolutionError) as captured:
        service.find_working_candidate((BackendCandidate("cpu"),))

    failure = captured.value.attempts[0].failure
    assert failure is not None
    assert failure.kind.value == "timeout"


def test_probe_does_not_retry_non_numpy_or_failed_repair(tmp_path: Path) -> None:
    reporter = ProbeReporter()
    non_numpy = _service(
        tmp_path,
        ProbeUv(),
        ProbeRunner([CommandResult(21, "", "CUDA_RUNTIME_FAILED")]),
        reporter,
    )
    with pytest.raises(CommandError):
        non_numpy.find_working_candidate((BackendCandidate("cu128"),))

    failed_repair = _service(
        tmp_path,
        ProbeUv(numpy_code=1),
        ProbeRunner([CommandResult(22, "", "NUMPY_BRIDGE_FAILED")]),
        reporter,
    )
    with pytest.raises(CommandError):
        failed_repair.find_working_candidate((BackendCandidate("cpu"),))
