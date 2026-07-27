from pathlib import Path
from textwrap import dedent

import pytest
import tomlkit

from uv_torch_compass.domain import BackendCandidate, Channel
from uv_torch_compass.errors import ConfigurationError, ProjectUpdateError
from uv_torch_compass.project_metadata import (
    read_configured_backend,
    read_project_requirements,
    render_project_configuration,
)


def _write(path: Path, content: str) -> None:
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def test_reads_base_extra_group_and_adds_effective_torch(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    _write(
        pyproject,
        """
        [project]
        name = "target"
        version = "0.1.0"
        requires-python = ">=3.10,<3.15"
        dependencies = ["numpy>=1.26"]

        [project.optional-dependencies]
        vision = ["torchvision>=0.21"]

        [dependency-groups]
        audio = ["torchaudio>=2.6"]
        training = [{ include-group = "audio" }]
        """,
    )

    requirements = read_project_requirements(
        pyproject, extras=("vision",), groups=("training",), overrides=()
    )

    selected = [(item.scope.label, item.package) for item in requirements.selected]
    assert ("extra:vision", "torch") in selected
    assert ("group:training", "torch") in selected
    assert requirements.has_package("torchvision")
    assert requirements.has_package("torchaudio")

    content, _ = render_project_configuration(
        pyproject,
        requirements=requirements,
        overrides=(),
        backend=BackendCandidate("cpu"),
        numpy_lt2_required=True,
    )
    document = tomlkit.parse(content).unwrap()
    assert "torch" in document["dependency-groups"]["training"]
    assert (
        "numpy<2; sys_platform == 'linux'" in document["dependency-groups"]["training"]
    )


def test_transitive_only_pytorch_project_adds_managed_source_anchor(
    tmp_path: Path,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    _write(
        pyproject,
        """
        [project]
        name = "target"
        version = "0.1.0"
        dependencies = ["vllm==0.19.1"]
        """,
    )

    requirements = read_project_requirements(
        pyproject,
        extras=(),
        groups=(),
        overrides=(),
    )
    content, changes = render_project_configuration(
        pyproject,
        requirements=requirements,
        overrides=(),
        backend=BackendCandidate("cu129"),
        numpy_lt2_required=False,
        source_packages=frozenset({"torch"}),
    )

    document = tomlkit.parse(content).unwrap()
    assert document["project"]["dependencies"] == ["vllm==0.19.1", "torch"]
    assert document["tool"]["uv"]["sources"]["torch"]
    assert document["tool"]["uv-torch-compass"]["state"]["managed-source-anchors"] == [
        "torch"
    ]
    assert "added managed PyTorch source anchors: torch" in changes


def test_render_preserves_comments_guards_old_sources_and_is_idempotent(
    tmp_path: Path,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    _write(
        pyproject,
        """
        # retained comment
        [project]
        name = "target"
        version = "0.1.0"
        dependencies = ["torchvision>=0.21", "torchaudio>=2.6"]

        [tool.uv.sources]
        torch = [{ index = "old", marker = "python_version < '3.11'" }]

        [[tool.uv.index]]
        name = "old"
        url = "https://example.invalid/simple"
        explicit = true
        """,
    )
    requirements = read_project_requirements(
        pyproject, extras=(), groups=(), overrides=("torch>=2.6",)
    )
    backend = BackendCandidate("cu128")

    first, changes = render_project_configuration(
        pyproject,
        requirements=requirements,
        overrides=("torch>=2.6",),
        backend=backend,
        numpy_lt2_required=True,
    )
    pyproject.write_text(first, encoding="utf-8")
    second, _ = render_project_configuration(
        pyproject,
        requirements=read_project_requirements(
            pyproject, extras=(), groups=(), overrides=()
        ),
        overrides=(),
        backend=backend,
        numpy_lt2_required=True,
    )

    document = tomlkit.parse(first).unwrap()
    assert first == second
    assert "# retained comment" in first
    assert "sys_platform != 'linux'" in first
    assert document["project"]["dependencies"].count("torch>=2.6") == 1
    assert "numpy<2; sys_platform == 'linux'" in document["project"]["dependencies"]
    assert set(document["tool"]["uv"]["sources"]) == {
        "torch",
        "torchvision",
        "torchaudio",
    }
    assert "configured the verified Linux PyTorch index" in changes
    assert (
        read_configured_backend(pyproject, ("torch", "torchvision", "torchaudio"))
        == backend
    )


def test_nonofficial_same_name_index_is_not_overwritten(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    _write(
        pyproject,
        """
        [project]
        name = "target"
        version = "0.1.0"
        dependencies = ["torch"]

        [[tool.uv.index]]
        name = "pytorch-cpu"
        url = "https://example.invalid/fake"
        """,
    )
    requirements = read_project_requirements(
        pyproject, extras=(), groups=(), overrides=()
    )

    with pytest.raises(ProjectUpdateError, match="non-matching"):
        render_project_configuration(
            pyproject,
            requirements=requirements,
            overrides=(),
            backend=BackendCandidate("cpu"),
            numpy_lt2_required=False,
        )


def test_unknown_scope_fails_before_render(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    _write(
        pyproject,
        """
        [project]
        name = "target"
        version = "0.1.0"
        dependencies = ["torch"]
        """,
    )

    with pytest.raises(ConfigurationError, match="unknown extras"):
        read_project_requirements(
            pyproject, extras=("missing",), groups=(), overrides=()
        )


def test_nightly_configuration_round_trips(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    _write(
        pyproject,
        """
        [project]
        name = "target"
        version = "0.1.0"
        dependencies = ["torch"]
        """,
    )
    requirements = read_project_requirements(
        pyproject, extras=(), groups=(), overrides=()
    )
    backend = BackendCandidate("cpu", Channel.NIGHTLY)
    content, _ = render_project_configuration(
        pyproject,
        requirements=requirements,
        overrides=(),
        backend=backend,
        numpy_lt2_required=False,
    )
    pyproject.write_text(content, encoding="utf-8")
    assert read_configured_backend(pyproject, ("torch",)) == backend


def test_exact_versions_may_differ_across_disjoint_python_markers(
    tmp_path: Path,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    _write(
        pyproject,
        """
        [project]
        name = "target"
        version = "0.1.0"
        dependencies = [
            "torch==2.5; python_version < '3.12'",
            "torch==2.6; python_version >= '3.12'",
        ]
        """,
    )

    requirements = read_project_requirements(
        pyproject, extras=(), groups=(), overrides=()
    )

    selected = requirements.for_interpreter("3.12.4", "cpython", "CPython")
    assert [str(item.requirement.specifier) for item in selected.selected] == ["==2.6"]


def test_overlapping_exact_versions_are_rejected(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    _write(
        pyproject,
        """
        [project]
        name = "target"
        version = "0.1.0"
        dependencies = [
            "torch==2.5; python_version >= '3.11'",
            "torch==2.6; python_version >= '3.12'",
        ]
        """,
    )

    with pytest.raises(ConfigurationError, match="conflicting exact"):
        read_project_requirements(pyproject, extras=(), groups=(), overrides=())
