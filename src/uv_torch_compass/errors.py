"""Application-specific exceptions exposed at internal layer boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uv_torch_compass.domain import CandidateAttempt


class CompassError(Exception):
    """Base class for recoverable uv-torch-compass failures."""


class ConfigurationError(CompassError):
    """Indicate invalid CLI, environment, or project configuration."""


class CommandError(CompassError):
    """Indicate that a required external command could not be executed."""


class CandidateResolutionError(CommandError):
    """Preserve all candidate attempts when no backend can be resolved."""

    def __init__(self, message: str, attempts: tuple[CandidateAttempt, ...]) -> None:
        """Store immutable diagnostic attempts for the CLI reporting boundary."""
        super().__init__(message)
        self.attempts = attempts


class CommandTimeoutError(CommandError):
    """Indicate that an external command exceeded its configured deadline."""


class ProbeError(CompassError):
    """Indicate that a backend runtime probe returned unusable data."""


class ProjectUpdateError(CompassError):
    """Indicate that project files could not be updated or restored safely."""


class ConcurrentRunError(ProjectUpdateError):
    """Indicate another mutating command already owns the workspace lock."""


class ExternalModificationError(ProjectUpdateError):
    """Indicate that a tracked project file changed outside the transaction."""


class ReportError(CompassError):
    """Indicate that a log or machine-readable report could not be written."""

    def __init__(
        self,
        message: str,
        *,
        applied: bool = False,
        document: dict[str, object] | None = None,
    ) -> None:
        """Retain terminal state when report persistence fails after an apply."""
        super().__init__(message)
        self.applied = applied
        self.document = document


class TerminationRequested(CompassError):
    """Indicate that SIGTERM requested rollback and command termination."""
