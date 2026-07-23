from datetime import datetime, timezone
from pathlib import Path

import pytest

from uv_torch_compass.errors import ConcurrentRunError, ExternalModificationError
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
