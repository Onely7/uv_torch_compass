from pathlib import Path

import pytest

from uv_torch_compass.errors import ProbeError
from uv_torch_compass.installed_metadata import read_installed_distributions


def test_reads_installed_distribution_identity(tmp_path: Path) -> None:
    metadata = tmp_path / "lib/python/site-packages/vllm-0.19.1.dist-info/METADATA"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("Name: vLLM\nVersion: 0.19.1\n", encoding="utf-8")

    distributions = read_installed_distributions(tmp_path)

    assert [(item.name, item.version) for item in distributions] == [("vllm", "0.19.1")]


def test_reads_installed_dependency_metadata(tmp_path: Path) -> None:
    metadata = tmp_path / "lib/python/site-packages/vllm-0.19.1.dist-info/METADATA"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        "Name: vLLM\n"
        "Version: 0.19.1\n"
        "Requires-Dist: torch==2.10.0\n"
        "Requires-Dist: torchvision; extra == 'vision'\n",
        encoding="utf-8",
    )

    (distribution,) = read_installed_distributions(tmp_path)

    assert distribution.requires_dist == (
        "torch==2.10.0",
        "torchvision; extra == 'vision'",
    )


def test_rejects_incomplete_installed_metadata(tmp_path: Path) -> None:
    metadata = tmp_path / "lib/python/site-packages/broken.dist-info/METADATA"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("Name: broken\n", encoding="utf-8")

    with pytest.raises(ProbeError, match="incomplete"):
        read_installed_distributions(tmp_path)


def test_rejects_oversized_installed_metadata(tmp_path: Path) -> None:
    metadata = tmp_path / "lib/python/site-packages/huge.dist-info/METADATA"
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(b"x" * (4 * 1024 * 1024 + 1))

    with pytest.raises(ProbeError, match="size limit"):
        read_installed_distributions(tmp_path)
