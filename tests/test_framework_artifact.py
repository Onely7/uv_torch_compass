from pathlib import Path
from typing import cast

from test_elf_dependencies import _write_elf

from uv_torch_compass.candidate_environment import CandidateExecutionEnvironment
from uv_torch_compass.candidate_lock import (
    CandidateLockSnapshot,
    LockedArtifact,
    LockedPackage,
)
from uv_torch_compass.candidate_resolution import CandidateResolution
from uv_torch_compass.command_runner import CommandResult
from uv_torch_compass.domain import BackendCandidate
from uv_torch_compass.framework_artifact import FrameworkArtifactInspector
from uv_torch_compass.uv_commands import UvCommandClient


class ArtifactUv:
    def __init__(self, result_code: int = 0, needed: str = "libcudart.so.12") -> None:
        self.result_code = result_code
        self.needed = needed
        self.calls = 0

    def sync_locked_package(
        self,
        environment: Path,
        project_dir: Path,
        python: Path,
        package: str,
    ) -> CommandResult:
        del project_dir, python
        self.calls += 1
        if self.result_code == 0:
            native = environment / "lib/python/site-packages/vllm/_C.so"
            native.parent.mkdir(parents=True, exist_ok=True)
            _write_elf(native, (self.needed, "libtorch.so"))
        return CommandResult(self.result_code, "", "wheel extraction failed")


def _resolution(
    backend: str,
    *,
    version: str = "0.19.1",
    source_url: str = "https://pypi.org/simple",
    source_kind: str = "registry",
    with_wheel: bool = True,
) -> CandidateResolution:
    wheel = (
        LockedArtifact(
            f"https://files.pythonhosted.org/vllm-{version}.whl",
            "sha256:" + "1" * 64,
            100,
        ),
    )
    return CandidateResolution(
        BackendCandidate(backend),
        CandidateExecutionEnvironment("3.12.12", "cpython", "linux", "x86_64"),
        CandidateLockSnapshot(
            "uv-torch-compass-candidate",
            (
                LockedPackage(
                    "uv-torch-compass-candidate",
                    "0",
                    "",
                    ("vllm",),
                    "virtual",
                ),
                LockedPackage(
                    "vllm",
                    version,
                    source_url,
                    (),
                    source_kind,
                    wheel if with_wheel else (),
                ),
            ),
        ),
    )


def _inspector(
    tmp_path: Path, uv: ArtifactUv, capabilities=frozenset()
) -> FrameworkArtifactInspector:
    return FrameworkArtifactInspector(
        cast(UvCommandClient, uv),
        tmp_path,
        Path("/usr/bin/python3"),
        capabilities,
    )


def test_catalog_rejects_before_selective_wheel_install(tmp_path: Path) -> None:
    uv = ArtifactUv()
    inspector = _inspector(
        tmp_path,
        uv,
        frozenset({"--only-install-package", "--no-build-package"}),
    )

    preflight, command = inspector.inspect(
        _resolution("cu129", version="0.26.0"), tmp_path
    )

    assert preflight.status == "catalog-rejected"
    assert preflight.requirement is not None
    assert preflight.requirement.required_cuda_variant == "cu130"
    assert command is None
    assert uv.calls == 0


def test_missing_uv_capability_and_custom_source_fall_back(tmp_path: Path) -> None:
    unsupported, _ = _inspector(tmp_path, ArtifactUv()).inspect(
        _resolution("cu128"), tmp_path
    )
    custom, _ = _inspector(
        tmp_path,
        ArtifactUv(),
        frozenset({"--only-install-package", "--no-build-package"}),
    ).inspect(
        _resolution(
            "cu128",
            source_url="https://packages.example.invalid/simple",
            source_kind="git",
            with_wheel=False,
        ),
        tmp_path,
    )

    assert unsupported.status == "unsupported-uv"
    assert custom.status == "not-a-registry-wheel"


def test_selective_extraction_failure_falls_back_to_full_validation(
    tmp_path: Path,
) -> None:
    inspector = _inspector(
        tmp_path,
        ArtifactUv(result_code=1),
        frozenset({"--only-install-package", "--no-build-package"}),
    )

    preflight, command = inspector.inspect(_resolution("cu128"), tmp_path)

    assert preflight.status == "wheel-unavailable"
    assert command is not None and command.returncode == 1


def test_inspected_wheel_is_cached_and_reused_across_candidates(tmp_path: Path) -> None:
    uv = ArtifactUv()
    inspector = _inspector(
        tmp_path,
        uv,
        frozenset({"--only-install-package", "--no-build-package"}),
    )

    first, _ = inspector.inspect(_resolution("cu128"), tmp_path)
    second, command = inspector.inspect(_resolution("cu126"), tmp_path)

    assert first.decision is not None and first.decision.allowed
    assert second.decision is not None and second.decision.allowed
    assert command is None
    assert uv.calls == 1


def test_catalog_and_elf_conflict_fails_closed(tmp_path: Path) -> None:
    inspector = _inspector(
        tmp_path,
        ArtifactUv(needed="libcudart.so.13"),
        frozenset({"--only-install-package", "--no-build-package"}),
    )

    preflight, _ = inspector.inspect(_resolution("cu121", version="0.6.0"), tmp_path)

    assert preflight.decision is not None
    assert not preflight.decision.allowed
    assert "conflicts" in preflight.decision.summary
