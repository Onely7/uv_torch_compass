"""Track target files across read-only and mutating operations."""

from __future__ import annotations

from dataclasses import dataclass

from uv_torch_compass.domain import CommandOutcome, Operation
from uv_torch_compass.errors import ExternalModificationError
from uv_torch_compass.safe_transaction import FileSnapshot
from uv_torch_compass.workspace import WorkspaceContext


@dataclass(frozen=True, slots=True)
class ApplicationResult:
    """Return a command outcome together with its resolved workspace."""

    outcome: CommandOutcome
    workspace: WorkspaceContext


@dataclass(frozen=True, slots=True)
class TargetState:
    """Capture project and lockfile content before an operation starts."""

    pyproject: FileSnapshot
    lockfile: FileSnapshot

    @classmethod
    def capture(cls, workspace: WorkspaceContext) -> TargetState:
        """Capture the target member metadata and shared lockfile."""
        return cls(
            FileSnapshot.capture(workspace.project_dir / "pyproject.toml"),
            FileSnapshot.capture(workspace.lockfile),
        )

    def require_unchanged(self, operation: Operation) -> None:
        """Reject a result when either tracked file changed externally.

        Raises:
            ExternalModificationError: If target content no longer matches the
                initial snapshots.
        """
        for expected in (self.pyproject, self.lockfile):
            current = FileSnapshot.capture(expected.path)
            if (current.existed, current.digest) != (
                expected.existed,
                expected.digest,
            ):
                raise ExternalModificationError(
                    f"{expected.path} changed while {operation.value} was running"
                )
