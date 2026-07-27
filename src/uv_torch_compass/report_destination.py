"""Validate machine-readable report destinations before project mutation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from uv_torch_compass.domain import RunOptions
from uv_torch_compass.errors import ReportError
from uv_torch_compass.workspace import WorkspaceContext


def preflight_report_destination(
    options: RunOptions,
    workspace: WorkspaceContext,
    *,
    log_path: Path,
) -> None:
    """Reject unsafe report paths and verify the destination is writable.

    The check creates and removes a private sibling file. It does not modify an
    existing report.

    Raises:
        ReportError: If the path overlaps project state, is a symlink, or cannot
            be prepared for an atomic write.
    """
    report_path = options.report_file
    if report_path is None:
        return
    resolved = report_path.resolve(strict=False)
    protected = {
        options.pyproject.resolve(strict=False),
        workspace.lockfile.resolve(strict=False),
        log_path.resolve(strict=False),
        (workspace.workspace_root / ".uv-torch-compass.lock").resolve(strict=False),
    }
    if resolved in protected or _looks_like_backup(report_path, options, workspace):
        raise ReportError(
            f"report path overlaps protected project state: {report_path}"
        )
    if report_path.is_symlink():
        raise ReportError(f"report path must not be a symlink: {report_path}")

    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{report_path.name}.preflight.",
            dir=report_path.parent,
        )
        os.fchmod(descriptor, 0o600)
        os.close(descriptor)
        Path(temporary_name).unlink()
    except OSError as exc:
        raise ReportError(
            f"report destination is not writable: {report_path}: {exc}"
        ) from exc


def _looks_like_backup(
    report_path: Path,
    options: RunOptions,
    workspace: WorkspaceContext,
) -> bool:
    names = (options.pyproject.name, workspace.lockfile.name)
    return any(report_path.name.startswith(f"{name}.bak.") for name in names)
