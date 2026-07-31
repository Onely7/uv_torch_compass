from pathlib import Path

import pytest

from uv_torch_compass.candidate_lock import read_candidate_lock
from uv_torch_compass.errors import ProbeError


def _write_lock(path: Path, packages: str) -> None:
    path.write_text(
        f'version = 1\nrequires-python = ">=3.12,<3.13"\n{packages}',
        encoding="utf-8",
    )


def test_lock_snapshot_records_pytorch_sources_and_dependency_paths(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "uv.lock"
    _write_lock(
        lock,
        """
[[package]]
name = "uv-torch-compass-candidate"
version = "0"
source = { virtual = "." }
dependencies = [{ name = "vllm" }, { name = "torch" }]

[[package]]
name = "vllm"
version = "0.19.1"
source = { registry = "https://pypi.org/simple" }
dependencies = [{ name = "torch" }, { name = "xgrammar" }]

[[package]]
name = "torch"
version = "2.10.0+cu126"
source = { registry = "https://download.pytorch.org/whl/cu126" }

[[package]]
name = "xgrammar"
version = "0.2.3"
source = { registry = "https://pypi.org/simple" }
""",
    )

    snapshot = read_candidate_lock(lock)

    torch = snapshot.package("Torch")
    assert torch is not None
    assert torch.version == "2.10.0+cu126"
    assert [package.name for package in snapshot.pytorch_packages] == ["torch"]
    assert snapshot.dependency_paths("xgrammar") == (
        ("uv-torch-compass-candidate", "vllm", "xgrammar"),
    )


@pytest.mark.parametrize(
    "packages",
    [
        "",
        """
[[package]]
name = "uv-torch-compass-candidate"
version = "0"
dependencies = [{ name = "missing" }]
""",
        """
[[package]]
name = "uv-torch-compass-candidate"
version = "0"
[[package]]
name = "duplicate"
version = "1"
[[package]]
name = "duplicate"
version = "2"
""",
    ],
)
def test_lock_snapshot_rejects_incomplete_or_ambiguous_graphs(
    tmp_path: Path,
    packages: str,
) -> None:
    lock = tmp_path / "uv.lock"
    _write_lock(lock, packages)

    with pytest.raises(ProbeError):
        read_candidate_lock(lock)


def test_lock_snapshot_rejects_oversized_input(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_bytes(b"x" * (32 * 1024 * 1024 + 1))

    with pytest.raises(ProbeError, match="32 MiB"):
        read_candidate_lock(lock)
