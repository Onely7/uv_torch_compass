from datetime import datetime, timezone
from pathlib import Path

import pytest

from uv_torch_compass.errors import (
    ConcurrentRunError,
    ExternalModificationError,
    ProjectUpdateError,
)
from uv_torch_compass.safe_transaction import (
    SafeProjectTransaction,
    WorkspaceAdvisoryLock,
)


def test_transaction_restores_existing_files_atomically(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    lockfile = tmp_path / "uv.lock"
    pyproject.write_text("original project", encoding="utf-8")
    lockfile.write_text("original lock", encoding="utf-8")
    transaction = SafeProjectTransaction.create(
        pyproject,
        lockfile,
        now=lambda: datetime(2026, 7, 23, tzinfo=timezone.utc),
    )
    transaction.write_pyproject("changed")
    lockfile.write_text("new lock", encoding="utf-8")
    transaction.accept_lockfile_change()

    transaction.restore()

    assert pyproject.read_text(encoding="utf-8") == "original project"
    assert lockfile.read_text(encoding="utf-8") == "original lock"
    assert all(path.is_file() for path in transaction.backups)


def test_transaction_restores_absent_lockfile(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    lockfile = tmp_path / "uv.lock"
    pyproject.write_text("original", encoding="utf-8")
    transaction = SafeProjectTransaction.create(pyproject, lockfile)
    lockfile.write_text("generated", encoding="utf-8")
    transaction.accept_lockfile_change()

    transaction.restore()

    assert not lockfile.exists()


def test_transaction_does_not_overwrite_external_changes(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    lockfile = tmp_path / "uv.lock"
    pyproject.write_text("original", encoding="utf-8")
    transaction = SafeProjectTransaction.create(pyproject, lockfile)
    transaction.write_pyproject("tool change")
    pyproject.write_text("editor change", encoding="utf-8")

    with pytest.raises(ExternalModificationError):
        transaction.restore()

    assert pyproject.read_text(encoding="utf-8") == "editor change"


def test_workspace_lock_rejects_a_second_owner(tmp_path: Path) -> None:
    path = tmp_path / ".uv-torch-compass.lock"
    with (
        WorkspaceAdvisoryLock(path),
        pytest.raises(ConcurrentRunError),
        WorkspaceAdvisoryLock(path),
    ):
        pass


def test_transaction_rejects_symlink_targets(tmp_path: Path) -> None:
    real = tmp_path / "real.toml"
    real.write_text("[project]\n", encoding="utf-8")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.symlink_to(real)

    with pytest.raises(ProjectUpdateError, match="symlink"):
        SafeProjectTransaction.create(pyproject, tmp_path / "uv.lock")


def test_backup_collision_does_not_remove_another_writers_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("original", encoding="utf-8")
    collision = tmp_path / "pyproject.toml.bak.20260723-000000"
    from uv_torch_compass import safe_transaction

    original_write = safe_transaction._write_new_file
    first = True

    def race(path: Path, content: bytes, mode: int) -> None:
        nonlocal first
        if first:
            first = False
            path.write_text("other writer", encoding="utf-8")
            raise FileExistsError(path)
        original_write(path, content, mode)

    monkeypatch.setattr(safe_transaction, "_write_new_file", race)

    transaction = SafeProjectTransaction.create(
        pyproject,
        tmp_path / "uv.lock",
        now=lambda: datetime(2026, 7, 23, tzinfo=timezone.utc),
    )

    assert collision.read_text(encoding="utf-8") == "other writer"
    assert transaction.pyproject_backup.name.endswith(".1")
