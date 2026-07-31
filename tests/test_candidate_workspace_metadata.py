import json

import pytest

from uv_torch_compass.candidate_metadata import ResolutionEvidenceSource
from uv_torch_compass.candidate_workspace_metadata import (
    read_candidate_workspace_metadata,
)
from uv_torch_compass.errors import LockMetadataError


def _document() -> dict[str, object]:
    root_id = "uv-torch-compass-candidate==0@virtual+."
    torch_id = "torch==2.4.0+cu121@registry+https://download.pytorch.org/whl/cu121"
    vllm_id = "vllm==0.6.0@registry+https://pypi.org/simple"
    return {
        "schema": {"version": "preview"},
        "resolution": {
            root_id: {
                "name": "uv-torch-compass-candidate",
                "version": "0",
                "source": {"virtual": "."},
                "kind": "package",
                "dependencies": [{"id": vllm_id}, {"id": torch_id}],
            },
            vllm_id: {
                "name": "vllm",
                "version": "0.6.0",
                "source": {"registry": {"url": "https://pypi.org/simple"}},
                "kind": "package",
                "dependencies": [{"id": torch_id}],
                "wheels": [
                    {
                        "url": "https://files.pythonhosted.org/vllm.whl",
                        "hashes": {"sha256": "b" * 64},
                        "size": 100,
                    }
                ],
            },
            torch_id: {
                "name": "torch",
                "version": "2.4.0+cu121",
                "source": {
                    "registry": {"url": "https://download.pytorch.org/whl/cu121"}
                },
                "kind": "package",
                "dependencies": [],
                "wheels": [
                    {
                        "url": "https://download.pytorch.org/whl/cu121/torch.whl",
                        "hashes": {"sha256": "a" * 64},
                    }
                ],
            },
        },
    }


def test_reads_workspace_resolution_with_optional_wheel_size() -> None:
    graph = read_candidate_workspace_metadata(json.dumps(_document()))

    assert graph.evidence_source is ResolutionEvidenceSource.WORKSPACE_METADATA
    assert graph.dependency_paths("torch") == (
        ("uv-torch-compass-candidate", "torch"),
        ("uv-torch-compass-candidate", "vllm", "torch"),
    )
    torch = graph.package("torch")
    assert torch is not None
    assert torch.package_id.startswith("torch==2.4.0")
    assert torch.wheels[0].size is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document.update({"schema": {"version": "2"}}),
        lambda document: document.update({"resolution": []}),
        lambda document: document["resolution"].pop(
            "uv-torch-compass-candidate==0@virtual+."
        ),
    ],
)
def test_rejects_unsupported_or_incomplete_workspace_metadata(mutation) -> None:
    document = _document()
    mutation(document)

    with pytest.raises(LockMetadataError):
        read_candidate_workspace_metadata(json.dumps(document))
