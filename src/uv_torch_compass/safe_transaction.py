"""Protect project updates with backups, hashes, and an advisory workspace lock."""

from __future__ import annotations

import fcntl
import hashlib
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from uv_torch_compass.errors import (
    ConcurrentRunError,
    ExternalModificationError,
    ProjectUpdateError,
)


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    """Retain exact file content and permissions for restoration."""

    path: Path
    existed: bool
    content: bytes
    mode: int
    digest: str

    @classmethod
    def capture(cls, path: Path) -> FileSnapshot:
        """Capture a file or its absence.

        Raises:
            ProjectUpdateError: If an existing file cannot be read.
        """
        if not path.exists():
            return cls(path, False, b"", 0o644, _digest(b""))
        if not path.is_file():
            raise ProjectUpdateError(f"transaction target is not a file: {path}")
        try:
            content = path.read_bytes()
            mode = path.stat().st_mode & 0o777
        except OSError as exc:
            raise ProjectUpdateError(f"failed to snapshot {path}: {exc}") from exc
        return cls(path, True, content, mode, _digest(content))


@dataclass(slots=True)
class WorkspaceAdvisoryLock:
    """Prevent two mutating tool processes from updating one workspace."""

    path: Path
    _descriptor: int | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> WorkspaceAdvisoryLock:
        """Acquire the non-blocking lock and record the owning process ID.

        Raises:
            ConcurrentRunError: If another process owns the workspace lock.
            ProjectUpdateError: If the lock file cannot be opened.
        """
        try:
            descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                os.close(descriptor)
                raise ConcurrentRunError(
                    f"another uv-torch-compass process is updating {self.path.parent}"
                ) from exc
            os.ftruncate(descriptor, 0)
            os.write(descriptor, f"pid={os.getpid()}\n".encode())
            os.fsync(descriptor)
        except ConcurrentRunError:
            raise
        except OSError as exc:
            raise ProjectUpdateError(
                f"failed to acquire workspace lock {self.path}: {exc}"
            ) from exc
        self._descriptor = descriptor
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Release the advisory lock while leaving its reusable inode in place."""
        if self._descriptor is None:
            return
        fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        os.close(self._descriptor)
        self._descriptor = None


@dataclass(slots=True)
class SafeProjectTransaction:
    """Apply and restore one pyproject and its project/workspace lockfile."""

    pyproject_snapshot: FileSnapshot
    lockfile_snapshot: FileSnapshot
    pyproject_backup: Path
    lockfile_backup: Path | None
    _expected_states: dict[Path, tuple[bool, str]]

    @classmethod
    def create(
        cls,
        pyproject: Path,
        lockfile: Path,
        *,
        now: Callable[[], datetime] = lambda: datetime.now().astimezone(),
    ) -> SafeProjectTransaction:
        """Capture and durably back up all files before mutation.

        Raises:
            ProjectUpdateError: If snapshots or backups cannot be created.
        """
        pyproject_snapshot = FileSnapshot.capture(pyproject)
        if not pyproject_snapshot.existed:
            raise ProjectUpdateError(f"{pyproject} does not exist")
        lockfile_snapshot = FileSnapshot.capture(lockfile)
        stamp = now().strftime("%Y%m%d-%H%M%S")
        pyproject_backup = _unused_backup_path(pyproject, stamp)
        lockfile_backup = (
            _unused_backup_path(lockfile, stamp) if lockfile_snapshot.existed else None
        )
        try:
            _write_new_file(
                pyproject_backup,
                pyproject_snapshot.content,
                pyproject_snapshot.mode,
            )
            if lockfile_backup is not None:
                _write_new_file(
                    lockfile_backup,
                    lockfile_snapshot.content,
                    lockfile_snapshot.mode,
                )
        except OSError as exc:
            pyproject_backup.unlink(missing_ok=True)
            if lockfile_backup is not None:
                lockfile_backup.unlink(missing_ok=True)
            raise ProjectUpdateError(f"failed to back up project files: {exc}") from exc
        return cls(
            pyproject_snapshot=pyproject_snapshot,
            lockfile_snapshot=lockfile_snapshot,
            pyproject_backup=pyproject_backup,
            lockfile_backup=lockfile_backup,
            _expected_states={
                pyproject: (pyproject_snapshot.existed, pyproject_snapshot.digest),
                lockfile: (lockfile_snapshot.existed, lockfile_snapshot.digest),
            },
        )

    @property
    def backups(self) -> tuple[Path, ...]:
        """Return durable backup paths created for this transaction."""
        values = [self.pyproject_backup]
        if self.lockfile_backup is not None:
            values.append(self.lockfile_backup)
        return tuple(values)

    def write_pyproject(self, content: str) -> None:
        """Replace pyproject only if it still matches the captured state.

        Raises:
            ExternalModificationError: If another process changed the file.
            ProjectUpdateError: If the atomic write fails.
        """
        path = self.pyproject_snapshot.path
        self._require_expected(path)
        _atomic_replace(path, content.encode(), self.pyproject_snapshot.mode)
        self._expected_states[path] = (True, _digest(content.encode()))

    def accept_lockfile_change(self) -> None:
        """Record the lockfile state produced by the successful uv lock command."""
        snapshot = FileSnapshot.capture(self.lockfile_snapshot.path)
        self._expected_states[snapshot.path] = (snapshot.existed, snapshot.digest)

    def restore(self) -> None:
        """Restore files without overwriting unrecognized external changes.

        Raises:
            ExternalModificationError: If a tracked file changed unexpectedly.
            ProjectUpdateError: If restoration cannot complete.
        """
        self._require_expected(self.pyproject_snapshot.path)
        self._require_expected(self.lockfile_snapshot.path)
        _restore_snapshot(self.pyproject_snapshot)
        _restore_snapshot(self.lockfile_snapshot)
        self._expected_states[self.pyproject_snapshot.path] = (
            self.pyproject_snapshot.existed,
            self.pyproject_snapshot.digest,
        )
        self._expected_states[self.lockfile_snapshot.path] = (
            self.lockfile_snapshot.existed,
            self.lockfile_snapshot.digest,
        )

    def _require_expected(self, path: Path) -> None:
        current = FileSnapshot.capture(path)
        expected = self._expected_states[path]
        if (current.existed, current.digest) != expected:
            raise ExternalModificationError(
                f"{path} changed outside uv-torch-compass; backup retained at "
                f"{self.pyproject_backup}"
            )


def _restore_snapshot(snapshot: FileSnapshot) -> None:
    if snapshot.existed:
        _atomic_replace(snapshot.path, snapshot.content, snapshot.mode)
        return
    try:
        snapshot.path.unlink(missing_ok=True)
        _fsync_directory(snapshot.path.parent)
    except OSError as exc:
        raise ProjectUpdateError(f"failed to remove {snapshot.path}: {exc}") from exc


def _atomic_replace(path: Path, content: bytes, mode: int) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        temporary.replace(path)
        _fsync_directory(path.parent)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ProjectUpdateError(f"failed to replace {path} atomically: {exc}") from exc


def atomic_write_private(path: Path, content: str) -> None:
    """Atomically write a private text artifact with mode 0600."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_replace(path, content.encode(), 0o600)


def _write_new_file(path: Path, content: bytes, mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unused_backup_path(path: Path, stamp: str) -> Path:
    candidate = path.with_name(f"{path.name}.bak.{stamp}")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.bak.{stamp}.{suffix}")
        suffix += 1
    return candidate


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
