import json
from pathlib import Path
from typing import cast

import pytest

try:
    import tomllib  # ty: ignore[unresolved-import]
except ModuleNotFoundError:  # pragma: no cover - exercised by Python 3.10 CI.
    import tomli as tomllib  # ty: ignore[unresolved-import]

from uv_torch_compass.command_runner import CommandResult
from uv_torch_compass.errors import CommandError, ConfigurationError
from uv_torch_compass.uv_commands import UvCommandClient
from uv_torch_compass.workspace import resolve_workspace


class FakeWorkspaceUv:
    def __init__(
        self,
        root: Path,
        members: tuple[Path, ...] = (),
        *,
        fail_directory: bool = False,
    ) -> None:
        self.root = root
        self.members = members
        self.fail_directory = fail_directory

    def workspace_metadata(self, project_dir: Path) -> CommandResult:
        del project_dir
        if self.fail_directory:
            return CommandResult(1, "", "unsupported")
        members = [
            {"name": _name_for(path), "path": str(path)} for path in self.members
        ]
        return CommandResult(
            0,
            json.dumps({"workspace_root": str(self.root), "members": members}),
            "",
        )


def _name_for(path: Path) -> str:
    with (path / "pyproject.toml").open("rb") as stream:
        document = tomllib.load(stream)
    return document.get("project", {}).get("name", "unknown")


def _write_project(path: Path, *, name: str | None = "target", workspace=False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    project = ""
    if name is not None:
        project = f'[project]\nname = "{name}"\nversion = "0.1.0"\n'
    workspace_table = (
        '\n[tool.uv.workspace]\nmembers = ["packages/*"]\n' if workspace else ""
    )
    path.write_text(project + workspace_table, encoding="utf-8")


def test_resolves_standalone_project(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    _write_project(pyproject)

    context = resolve_workspace(
        pyproject, cast(UvCommandClient, FakeWorkspaceUv(tmp_path))
    )

    assert not context.is_workspace
    assert context.workspace_root == tmp_path
    assert context.package is None
    assert context.lockfile == tmp_path / "uv.lock"


def test_resolves_named_workspace_member_and_shared_lock(tmp_path: Path) -> None:
    root_project = tmp_path / "pyproject.toml"
    member = tmp_path / "packages" / "vision"
    _write_project(root_project, name=None, workspace=True)
    _write_project(member / "pyproject.toml", name="vision")

    context = resolve_workspace(
        member / "pyproject.toml",
        cast(UvCommandClient, FakeWorkspaceUv(tmp_path, (member,))),
    )

    assert context.is_workspace
    assert context.package == "vision"
    assert context.lockfile == tmp_path / "uv.lock"


def test_workspace_capability_failure_is_not_treated_as_standalone(
    tmp_path: Path,
) -> None:
    root_project = tmp_path / "pyproject.toml"
    member = tmp_path / "packages" / "vision"
    _write_project(root_project, name=None, workspace=True)
    _write_project(member / "pyproject.toml")

    with pytest.raises(CommandError, match="update uv"):
        resolve_workspace(
            member / "pyproject.toml",
            cast(UvCommandClient, FakeWorkspaceUv(tmp_path, fail_directory=True)),
        )


def test_workspace_rejects_nonmember_and_unnamed_member(tmp_path: Path) -> None:
    root_project = tmp_path / "pyproject.toml"
    member = tmp_path / "packages" / "vision"
    _write_project(root_project, name=None, workspace=True)
    _write_project(member / "pyproject.toml", name=None)

    with pytest.raises(ConfigurationError, match="not a member"):
        resolve_workspace(
            member / "pyproject.toml",
            cast(UvCommandClient, FakeWorkspaceUv(tmp_path, ())),
        )
    with pytest.raises(ConfigurationError, match="must define"):
        resolve_workspace(
            member / "pyproject.toml",
            cast(UvCommandClient, FakeWorkspaceUv(tmp_path, (member,))),
        )


class RawMetadataUv:
    def __init__(self, output: str) -> None:
        self.output = output

    def workspace_metadata(self, project_dir: Path) -> CommandResult:
        del project_dir
        return CommandResult(0, self.output, "")


@pytest.mark.parametrize(
    "output, message",
    [
        ("not-json", "invalid JSON"),
        ("[]", "JSON object"),
        ("{}", "workspace_root"),
    ],
)
def test_workspace_rejects_invalid_metadata(
    tmp_path: Path, output: str, message: str
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    _write_project(pyproject)

    with pytest.raises(CommandError, match=message):
        resolve_workspace(
            pyproject,
            cast(UvCommandClient, RawMetadataUv(output)),
        )


@pytest.mark.parametrize(
    "members, message",
    [
        ("invalid", "omitted members"),
        (["invalid"], "invalid member"),
        ([{"name": "target"}], "omitted path"),
        ([{"path": "/member"}], "omitted name"),
    ],
)
def test_workspace_rejects_invalid_member_metadata(
    tmp_path: Path, members: object, message: str
) -> None:
    root = tmp_path / "root"
    member = root / "packages" / "target"
    _write_project(root / "pyproject.toml", name=None, workspace=True)
    _write_project(member / "pyproject.toml")
    output = json.dumps({"workspace_root": str(root), "members": members})

    with pytest.raises(CommandError, match=message):
        resolve_workspace(
            member / "pyproject.toml",
            cast(UvCommandClient, RawMetadataUv(output)),
        )


def test_workspace_rejects_member_name_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "root"
    member = root / "packages" / "target"
    _write_project(root / "pyproject.toml", name=None, workspace=True)
    _write_project(member / "pyproject.toml", name="declared")
    output = json.dumps(
        {
            "workspace_root": str(root),
            "members": [{"name": "metadata", "path": str(member)}],
        }
    )

    with pytest.raises(ConfigurationError, match="does not match"):
        resolve_workspace(
            member / "pyproject.toml",
            cast(UvCommandClient, RawMetadataUv(output)),
        )
