from pathlib import Path

import pytest

from uv_torch_compass.candidate_lock import read_candidate_lock
from uv_torch_compass.errors import ProbeError

_LOCK_FIXTURES = Path(__file__).parent / "fixtures" / "locks"


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


def test_lock_snapshot_accepts_wheel_without_size(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "uv.lock"
    _write_lock(
        lock,
        """
[[package]]
name = "uv-torch-compass-candidate"
version = "0"
dependencies = [{ name = "torch" }]

[[package]]
name = "torch"
version = "2.4.0+cu121"
source = { registry = "https://download.pytorch.org/whl/cu121" }
wheels = [
    { url = "https://download.pytorch.org/whl/cu121/torch.whl", hash = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" },
]
""",
    )

    snapshot = read_candidate_lock(lock)

    torch = snapshot.package("torch")
    assert torch is not None
    assert torch.wheels[0].size is None
    assert snapshot.lock_schema is not None
    assert snapshot.lock_schema.version == 1


@pytest.mark.parametrize(
    "filename, backend",
    [
        ("uv-0.9.28-cu121.lock", "cu121"),
        ("uv-0.11.28-cu126.lock", "cu126"),
        ("uv-0.11.28-cu128.lock", "cu128"),
        ("uv-0.11.28-cu129.lock", "cu129"),
    ],
)
def test_lock_fixtures_accept_official_wheels_without_size(
    filename: str,
    backend: str,
) -> None:
    snapshot = read_candidate_lock(_LOCK_FIXTURES / filename)

    torch = snapshot.package("torch")
    assert torch is not None
    assert torch.version.endswith(f"+{backend}")
    assert torch.source_url.endswith(f"/whl/{backend}")
    assert torch.wheels[0].size is None


@pytest.mark.parametrize("size", ["unknown", True, 0, -1])
def test_lock_snapshot_rejects_invalid_present_wheel_size(
    tmp_path: Path,
    size: object,
) -> None:
    lock = tmp_path / "uv.lock"
    rendered_size = f'"{size}"' if isinstance(size, str) else str(size).lower()
    _write_lock(
        lock,
        f"""
[[package]]
name = "uv-torch-compass-candidate"
version = "0"
dependencies = [{{ name = "torch" }}]

[[package]]
name = "torch"
version = "2.4.0+cu121"
source = {{ registry = "https://download.pytorch.org/whl/cu121" }}
wheels = [
    {{ url = "https://download.pytorch.org/whl/cu121/torch.whl", hash = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", size = {rendered_size} }},
]
""",
    )

    with pytest.raises(ProbeError, match="invalid wheel size"):
        read_candidate_lock(lock)


def test_lock_snapshot_validates_schema_and_revision(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text(
        'version = 1\nrevision = 3\n[[package]]\nname = "uv-torch-compass-candidate"\nversion = "0"\n',
        encoding="utf-8",
    )

    snapshot = read_candidate_lock(lock)

    assert snapshot.lock_schema is not None
    assert snapshot.lock_schema.revision == 3

    lock.write_text(
        'version = 2\n[[package]]\nname = "uv-torch-compass-candidate"\nversion = "0"\n',
        encoding="utf-8",
    )
    with pytest.raises(ProbeError, match="schema 2"):
        read_candidate_lock(lock)
