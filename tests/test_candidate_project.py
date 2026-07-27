from pathlib import Path
from textwrap import dedent

import pytest

try:
    import tomllib  # ty: ignore[unresolved-import]
except ModuleNotFoundError:  # pragma: no cover - exercised by Python 3.10 CI.
    import tomli as tomllib  # ty: ignore[unresolved-import]

from uv_torch_compass.candidate_environment import CandidateExecutionEnvironment
from uv_torch_compass.candidate_project import render_candidate_project
from uv_torch_compass.domain import BackendCandidate, Channel
from uv_torch_compass.errors import ConfigurationError

_ENVIRONMENT = CandidateExecutionEnvironment("3.12.12", "cpython", "linux", "x86_64")


def _write(path: Path, content: str) -> None:
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def _render(
    tmp_path: Path,
    content: str,
    *,
    requirements: tuple[str, ...] = ("vllm==0.19.1",),
    candidate: BackendCandidate | None = None,
    members: dict[str, Path] | None = None,
) -> dict:
    target = tmp_path / "pyproject.toml"
    _write(target, content)
    destination = tmp_path / "candidate"
    rendered = render_candidate_project(
        target,
        destination=destination,
        requirements=requirements,
        candidate=candidate or BackendCandidate("cu128"),
        environment=_ENVIRONMENT,
        workspace_members=members or {},
    )
    with rendered.open("rb") as stream:
        return tomllib.load(stream)


def test_preserves_resolution_policy_and_makes_sources_portable(
    tmp_path: Path,
) -> None:
    local = tmp_path / "packages" / "local"
    member = tmp_path / "packages" / "member"
    document = _render(
        tmp_path,
        """
        [project]
        name = "target"
        version = "1"
        requires-python = ">=3.12"

        [tool.uv]
        constraint-dependencies = ["numpy<3"]
        override-dependencies = ["idna>=3"]
        resolution = "highest"

        [tool.uv.sources]
        local-package = { path = "packages/local", editable = true }
        workspace-package = { workspace = true }
        unused = { git = "https://example.invalid/unused.git" }
        torch = { index = "old-pytorch" }

        [[tool.uv.index]]
        name = "private"
        url = "https://packages.example.invalid/simple"
        explicit = true
        """,
        requirements=(
            "local-package",
            "workspace-package",
            "vllm==0.19.1",
            "torchvision",
        ),
        members={"workspace-package": member},
    )

    assert document["project"]["requires-python"] == ">=3.12,<3.13"
    assert document["project"]["dependencies"] == [
        "local-package",
        "workspace-package",
        "vllm==0.19.1",
        "torchvision",
        "torch",
    ]
    uv = document["tool"]["uv"]
    assert uv["constraint-dependencies"] == ["numpy<3"]
    assert uv["override-dependencies"] == ["idna>=3"]
    assert uv["resolution"] == "highest"
    assert uv["sources"]["local-package"]["path"] == str(local)
    assert uv["sources"]["local-package"]["editable"] is True
    assert uv["sources"]["workspace-package"]["path"] == str(member)
    assert "unused" not in uv["sources"]
    assert uv["sources"]["torch"] == {"index": "pytorch-cu128"}
    assert uv["sources"]["torchvision"] == {"index": "pytorch-cu128"}
    assert [index["name"] for index in uv["index"]] == [
        "private",
        "pytorch-cu128",
    ]


def test_candidate_limits_python_and_declares_concrete_environment_policy(
    tmp_path: Path,
) -> None:
    document = _render(
        tmp_path,
        """
        [project]
        name = "target"
        version = "1"
        requires-python = ">=3.12"
        """,
    )

    assert document["project"]["requires-python"] == ">=3.12,<3.13"
    assert document["tool"]["uv"]["environments"] == [
        _ENVIRONMENT.resolution_environment_marker
    ]
    assert document["tool"]["uv"]["required-environments"] == [
        _ENVIRONMENT.required_environment_marker
    ]


def test_nightly_candidate_allows_prereleases_and_reuses_official_index(
    tmp_path: Path,
) -> None:
    document = _render(
        tmp_path,
        """
        [project]
        name = "target"
        version = "1"

        [[tool.uv.index]]
        name = "pytorch-nightly-cu128"
        url = "https://download.pytorch.org/whl/nightly/cu128"
        """,
        candidate=BackendCandidate("cu128", Channel.NIGHTLY),
    )

    uv = document["tool"]["uv"]
    assert uv["prerelease"] == "allow"
    assert len(uv["index"]) == 1
    assert uv["index"][0]["explicit"] is True
    assert uv["sources"]["torch"]["index"] == "pytorch-nightly-cu128"


def test_preserves_marker_specific_git_and_url_sources(tmp_path: Path) -> None:
    document = _render(
        tmp_path,
        """
        [project]
        name = "target"
        version = "1"

        [tool.uv.sources]
        framework = [
          { git = "https://example.invalid/framework.git", tag = "v1", marker = "python_version < '3.13'" },
          { url = "https://example.invalid/framework.whl", marker = "python_version >= '3.13'" },
        ]
        """,
        requirements=("framework[server]>=1",),
    )

    sources = document["tool"]["uv"]["sources"]["framework"]
    assert sources[0]["tag"] == "v1"
    assert sources[1]["url"].endswith("framework.whl")


@pytest.mark.parametrize(
    "content, message",
    [
        ("not = valid", "failed to read"),
        (
            """
            [project]
            name = "target"
            version = "1"
            requires-python = []
            """,
            "requires-python",
        ),
        (
            """
            [project]
            name = "target"
            version = "1"
            [tool]
            uv = "invalid"
            """,
            "tool.uv",
        ),
        (
            """
            [project]
            name = "target"
            version = "1"
            [tool.uv]
            index = "invalid"
            """,
            "index",
        ),
        (
            """
            [project]
            name = "target"
            version = "1"
            [[tool.uv.index]]
            name = "pytorch-cu128"
            url = "https://example.invalid/simple"
            """,
            "non-official",
        ),
        (
            """
            [project]
            name = "target"
            version = "1"
            [tool.uv]
            sources = "invalid"
            """,
            "sources",
        ),
        (
            """
            [project]
            name = "target"
            version = "1"
            [tool.uv.sources]
            vllm = "invalid"
            """,
            "table or array",
        ),
        (
            """
            [project]
            name = "target"
            version = "1"
            [tool.uv.sources]
            vllm = { workspace = true }
            """,
            "workspace source",
        ),
        (
            """
            [project]
            name = "target"
            version = "1"
            [tool.uv.sources]
            vllm = { path = "" }
            """,
            "path source",
        ),
    ],
)
def test_rejects_unsafe_source_shapes(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    target = tmp_path / "pyproject.toml"
    _write(target, content)

    with pytest.raises(ConfigurationError, match=message):
        render_candidate_project(
            target,
            destination=tmp_path / "candidate",
            requirements=("vllm",),
            candidate=BackendCandidate("cu128"),
            environment=_ENVIRONMENT,
            workspace_members={},
        )


def test_rejects_invalid_candidate_requirement(tmp_path: Path) -> None:
    target = tmp_path / "pyproject.toml"
    _write(target, "[project]\nname = 'target'\nversion = '1'")

    with pytest.raises(ConfigurationError, match="invalid candidate requirement"):
        render_candidate_project(
            target,
            destination=tmp_path / "candidate",
            requirements=("not a valid requirement !!!",),
            candidate=BackendCandidate("cpu"),
            environment=_ENVIRONMENT,
            workspace_members={},
        )
