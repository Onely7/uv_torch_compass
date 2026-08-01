"""Command-line interface for applying, planning, and checking PyTorch indexes."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import traceback
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import FrameType

from uv_torch_compass import __version__
from uv_torch_compass.application import CompassApplication
from uv_torch_compass.command_runner import SubprocessRunner
from uv_torch_compass.configuration import resolve_options
from uv_torch_compass.domain import CommandOutcome
from uv_torch_compass.errors import (
    CandidateResolutionError,
    CompassError,
    ConfigurationError,
    ReportError,
    TerminationRequested,
)
from uv_torch_compass.reporting import CommandReporter
from uv_torch_compass.uv_commands import UvCommandClient

_LINK_MODES = ("clone", "copy", "hardlink", "symlink")


def create_parser() -> argparse.ArgumentParser:
    """Create the public command-oriented argument parser."""
    parser = argparse.ArgumentParser(
        prog="uv-torch-compass",
        description=(
            "Find an official PyTorch index that works on this Linux machine, "
            "then plan, apply, or check the project configuration."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    subcommands = parser.add_subparsers(dest="operation", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    _add_shared_options(shared)
    selection = argparse.ArgumentParser(add_help=False, parents=[shared])
    _add_selection_options(selection)

    subcommands.add_parser(
        "apply",
        parents=[selection],
        help="verify a candidate and update the target project",
        description="Verify a candidate, update pyproject.toml, lock, and synchronize.",
    )
    subcommands.add_parser(
        "plan",
        parents=[selection],
        help="verify a candidate and show the change without applying it",
        description="Verify a candidate and print a proposed diff without target changes.",
    )
    subcommands.add_parser(
        "check",
        parents=[shared],
        help="validate the current configuration without changing it",
        description="Check metadata, lock freshness, synchronization, and runtime behavior.",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Run one public subcommand and translate failures to stable exit codes.

    Args:
        arguments: Command arguments excluding the executable name.

    Returns:
        Zero for success, one for operational failure, or argparse's two for
        invalid command syntax.
    """
    raw_arguments = list(arguments) if arguments is not None else sys.argv[1:]
    namespace = create_parser().parse_args(raw_arguments)
    try:
        options = resolve_options(
            namespace, environ=os.environ, cwd=Path.cwd().resolve()
        )
    except ConfigurationError as exc:
        _emit_early_failure(namespace, str(exc))
        return 1

    reporter = CommandReporter(options, __version__)
    try:
        with reporter, _termination_guard():
            runner = SubprocessRunner()
            uv = UvCommandClient.discover(
                runner,
                link_mode=options.link_mode,
                timeout_seconds=options.timeout_seconds,
                environment=os.environ,
            )
            try:
                result = CompassApplication(options, runner, reporter, uv).run()
            except KeyboardInterrupt:
                outcome = CommandOutcome("failed", False, None)
                reporter.fail(
                    "operation interrupted; project restoration was attempted"
                )
                reporter.emit_final(
                    outcome, None, exit_code=1, error="operation interrupted"
                )
                return 1
            except CandidateResolutionError as exc:
                outcome = CommandOutcome(
                    "failed",
                    False,
                    None,
                    attempts=exc.attempts,
                )
                reporter.fail(str(exc))
                reporter.emit_final(outcome, None, exit_code=1, error=str(exc))
                return 1
            except CompassError as exc:
                outcome = CommandOutcome("failed", False, None)
                reporter.fail(str(exc))
                reporter.emit_final(outcome, None, exit_code=1, error=str(exc))
                return 1
            except Exception as exc:  # Application boundary retains unexpected context.
                reporter.detail(traceback.format_exc())
                message = f"unexpected {type(exc).__name__}: {exc}"
                reporter.fail(message)
                reporter.emit_final(
                    CommandOutcome("failed", False, None),
                    None,
                    exit_code=1,
                    error=message,
                )
                return 1
            reporter.emit_final(result.outcome, result.workspace, exit_code=0)
            return 0
    except ReportError as exc:
        if exc.document is not None:
            print(
                json.dumps(exc.document, indent=2, sort_keys=True),
                file=sys.stderr,
            )
        else:
            print(f"uv-torch-compass: {exc}", file=sys.stderr)
        return 1


def _add_shared_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pyproject", metavar="PATH")
    parser.add_argument("--extra", dest="extras", action="append", metavar="NAME")
    parser.add_argument("--group", dest="groups", action="append", metavar="NAME")
    parser.add_argument("--cuda-device", metavar="INDEX_OR_UUID")
    parser.add_argument("--cuda-compatibility", choices=("strict", "minor"))
    parser.add_argument("--probe-profile", choices=("standard", "compile"))
    parser.add_argument(
        "--framework-probe",
        dest="framework_probes",
        action="append",
        choices=("vllm",),
        metavar="NAME",
    )
    parser.add_argument("--log-dir", metavar="PATH")
    parser.add_argument("--timeout", type=int, metavar="SECONDS")
    parser.add_argument("--output-format", choices=("text", "json"))
    parser.add_argument("--report-file", metavar="PATH")


def _add_selection_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--python", dest="python_request", metavar="REQUEST")
    parser.add_argument("--torch", dest="torch_requirement", metavar="REQUIREMENT")
    parser.add_argument(
        "--torchvision", dest="torchvision_requirement", metavar="REQUIREMENT"
    )
    parser.add_argument(
        "--torchaudio", dest="torchaudio_requirement", metavar="REQUIREMENT"
    )
    parser.add_argument("--backend", metavar="BACKEND")
    parser.add_argument("--channel", choices=("stable", "nightly"))
    parser.add_argument("--link-mode", choices=_LINK_MODES)


def _emit_early_failure(namespace: argparse.Namespace, message: str) -> None:
    if getattr(namespace, "output_format", None) == "json":
        document = {
            "schema_version": 8,
            "operation": getattr(namespace, "operation", ""),
            "status": "failed",
            "exit_code": 1,
            "applied": False,
            "target": getattr(namespace, "pyproject", "") or "",
            "workspace": "",
            "request": {},
            "python": {},
            "candidate_attempts": [],
            "blocking_summary": None,
            "failure_category": "configuration-error",
            "selected_backend": "",
            "selected_index": "",
            "selected_gpu": None,
            "resolved_packages": {},
            "dependency_roots": [],
            "source_anchors": [],
            "required_environment": "",
            "validation": {},
            "framework_validation": [],
            "framework_version_selection": None,
            "probe_contract": {},
            "operation_state": {
                "applied": False,
                "report_written": False,
            },
            "environment_policy": {},
            "changes": [],
            "backups": [],
            "warnings": [],
            "errors": [message],
            "timing": {"elapsed_seconds": 0.0},
        }
        print(json.dumps(document, sort_keys=True))
        return
    print(f"uv-torch-compass: {message}", file=sys.stderr)


@contextmanager
def _termination_guard() -> Iterator[None]:
    """Convert SIGTERM into a recoverable exception so transactions can restore."""
    previous = signal.getsignal(signal.SIGTERM)

    def request_termination(_number: int, _frame: FrameType | None) -> None:
        raise TerminationRequested("termination requested")

    signal.signal(signal.SIGTERM, request_termination)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
