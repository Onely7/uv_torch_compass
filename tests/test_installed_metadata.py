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


def test_rejects_incomplete_installed_metadata(tmp_path: Path) -> None:
    metadata = tmp_path / "lib/python/site-packages/broken.dist-info/METADATA"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("Name: broken\n", encoding="utf-8")

    with pytest.raises(ProbeError, match="incomplete"):
        read_installed_distributions(tmp_path)
