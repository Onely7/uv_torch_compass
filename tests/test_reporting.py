import json
import stat
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from uv_torch_compass.candidate_environment import CandidateExecutionEnvironment
from uv_torch_compass.candidate_lock import CandidateLockSnapshot, LockedPackage
from uv_torch_compass.candidate_resolution import CandidateResolution
from uv_torch_compass.cuda_compatibility import CompatibilityPolicy
from uv_torch_compass.domain import (
    BackendCandidate,
    BackendRequest,
    CandidateAttempt,
    Channel,
    CommandOutcome,
    FailedIndex,
    FailedPackage,
    Operation,
    OutputFormat,
    ProbeProfile,
    ResolutionFailure,
    ResolutionFailureKind,
    RunOptions,
    RuntimeReport,
)
from uv_torch_compass.errors import ProjectUpdateError, ReportError
from uv_torch_compass.report_destination import preflight_report_destination
from uv_torch_compass.reporting import CommandReporter, redact
from uv_torch_compass.workspace import WorkspaceContext


def test_redaction_removes_url_and_header_credentials() -> None:
    value = (
        "https://user:password@example.invalid/simple?token=secret\n"
        "Authorization: Bearer abc123\n"
        "UV_INDEX_TOKEN=top-secret\n"
        'Cookie: session=cookie-secret\n{"access_token": "json-secret"}\n'
        "--api-key option-secret\n"
    )
    redacted = redact(value)
    assert "password" not in redacted
    assert "secret" not in redacted
    assert "abc123" not in redacted
    assert "top-secret" not in redacted
    assert "cookie-secret" not in redacted
    assert "json-secret" not in redacted
    assert "option-secret" not in redacted
    assert "<redacted>" in redacted


def test_json_report_is_single_document_and_private(tmp_path: Path, capsys) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[project]\n", encoding="utf-8")
    report_file = tmp_path / "artifacts" / "result.json"
    options = RunOptions(
        operation=Operation.PLAN,
        pyproject=pyproject,
        python_request="",
        requirement_overrides=(),
        backend=BackendRequest.parse("cpu"),
        channel=Channel.STABLE,
        cuda_compatibility=CompatibilityPolicy.STRICT,
        probe_profile=ProbeProfile.STANDARD,
        extras=(),
        groups=(),
        cuda_device=None,
        link_mode="copy",
        log_dir=tmp_path / "logs",
        timeout_seconds=1800,
        output_format=OutputFormat.JSON,
        report_file=report_file,
    )
    reporter = CommandReporter(options, "0.1.0")
    resolution = CandidateResolution(
        BackendCandidate("cpu"),
        CandidateExecutionEnvironment("3.12.12", "cpython", "linux", "x86_64"),
        CandidateLockSnapshot(
            "uv-torch-compass-candidate",
            (
                LockedPackage(
                    "uv-torch-compass-candidate",
                    "0",
                    "",
                    ("torch",),
                ),
                LockedPackage(
                    "torch",
                    "2.7.0",
                    "https://download.pytorch.org/whl/cpu",
                    (),
                ),
            ),
        ),
    )
    with reporter:
        reporter.phase("inspect", "https://user:secret@example.invalid/simple")
        reporter.warn("warning")
        reporter.emit_final(
            CommandOutcome(
                "planned",
                False,
                None,
                attempts=(
                    CandidateAttempt(
                        "cpu",
                        "runtime",
                        "passed",
                        "resolved as cpu",
                        "strict",
                        resolution=resolution,
                    ),
                ),
                changes=("one change",),
            ),
            None,
            exit_code=0,
        )

    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert document["schema_version"] == 6
    assert document["status"] == "planned"
    assert document["changes"] == ["one change"]
    attempt = document["candidate_attempts"][0]
    assert attempt["resolution"]["pytorch"]["torch"]["version"] == "2.7.0"
    assert attempt["phases"] == {
        "lock": "passed",
        "install": "passed",
        "runtime": "passed",
        "framework": "not-run",
    }
    assert "secret" not in captured.err
    assert json.loads(report_file.read_text(encoding="utf-8")) == document
    assert stat.S_IMODE(report_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(reporter.log_path.stat().st_mode) == 0o600


def _text_options(tmp_path: Path, report_file: Path | None = None) -> RunOptions:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[project]\n", encoding="utf-8")
    return RunOptions(
        operation=Operation.APPLY,
        pyproject=pyproject,
        python_request="3.12",
        requirement_overrides=(),
        backend=BackendRequest.parse("cpu"),
        channel=Channel.STABLE,
        cuda_compatibility=CompatibilityPolicy.STRICT,
        probe_profile=ProbeProfile.STANDARD,
        extras=("vision",),
        groups=("training",),
        cuda_device=None,
        link_mode="copy",
        log_dir=tmp_path / "logs",
        timeout_seconds=1800,
        output_format=OutputFormat.TEXT,
        report_file=report_file,
    )


def test_text_report_prints_selection_diff_error_and_workspace(
    tmp_path: Path, capsys
) -> None:
    options = _text_options(tmp_path)
    runtime = RuntimeReport(
        2,
        BackendCandidate("cpu"),
        "2.7.0",
        "0.22.0",
        "not-installed",
        "2.2.0",
        "none",
        "none",
        "NOT_APPLICABLE",
        "PASS",
        "PASS",
        "NOT_REQUESTED",
    )
    workspace = WorkspaceContext(tmp_path, tmp_path, None, tmp_path / "uv.lock", False)
    reporter = CommandReporter(options, "0.1.0")
    with reporter:
        reporter.ok("verified")
        reporter.emit_final(
            CommandOutcome(
                "failed",
                False,
                runtime,
                planned_diff="@@ planned @@",
                changes=("changed source",),
            ),
            workspace,
            exit_code=1,
            error="failed safely",
        )

    captured = capsys.readouterr()
    assert "Backend: cpu" in captured.out
    assert "Planned pyproject.toml diff" in captured.out
    assert "failed safely" in captured.err


def test_text_report_explains_failed_candidate(tmp_path: Path, capsys) -> None:
    options = _text_options(tmp_path)
    failure = ResolutionFailure(
        ResolutionFailureKind.NO_COMPATIBLE_DISTRIBUTION,
        "The required package build is unavailable from this index.",
        FailedPackage("torch", "2.10.0", "torch==2.10.0"),
        ("vllm>=0.25.0", "torch==2.10.0"),
        FailedIndex(
            "pytorch-cu121",
            "https://download.pytorch.org/whl/cu121",
        ),
        "linux-x86_64",
        ("Select a compatible vLLM version.",),
    )
    reporter = CommandReporter(options, "0.1.0")

    with reporter:
        reporter.emit_final(
            CommandOutcome(
                "failed",
                False,
                None,
                attempts=(
                    CandidateAttempt(
                        "cu121",
                        "install",
                        "failed",
                        failure.summary,
                        "strict",
                        failure,
                    ),
                ),
            ),
            None,
            exit_code=1,
            error="no usable backend",
        )

    captured = capsys.readouterr()
    assert "Package: torch==2.10.0" in captured.out
    assert "Required by: vllm>=0.25.0 -> torch==2.10.0" in captured.out
    assert "Index: pytorch-cu121" in captured.out
    assert "Suggestion: Select a compatible vLLM version." in captured.out


def test_text_report_limits_long_skipped_candidate_lists(
    tmp_path: Path, capsys
) -> None:
    reporter = CommandReporter(_text_options(tmp_path), "0.1.0")
    attempts = tuple(
        CandidateAttempt(
            f"cu{index}",
            "policy",
            "skipped",
            f"candidate {index} is unsupported",
            "unsupported",
        )
        for index in range(7)
    )

    with reporter:
        reporter.emit_final(
            CommandOutcome("failed", False, None, attempts=attempts),
            None,
            exit_code=1,
            error="no usable backend",
        )

    captured = capsys.readouterr()
    assert "cu4: candidate 4 is unsupported" in captured.out
    assert "cu5: candidate 5 is unsupported" not in captured.out
    assert "... and 2 more" in captured.out


def test_reporter_requires_context_and_wraps_atomic_write_failure(
    tmp_path: Path, monkeypatch
) -> None:
    report_file = tmp_path / "report.json"
    reporter = CommandReporter(_text_options(tmp_path, report_file), "0.1.0")
    with pytest.raises(ReportError, match="opened"):
        _ = reporter.log_path

    def fail_write(_path: Path, _content: str) -> None:
        raise ProjectUpdateError("disk failure")

    monkeypatch.setattr("uv_torch_compass.reporting.atomic_write_private", fail_write)
    with (
        reporter,
        pytest.raises(ReportError, match="could not write report") as captured,
    ):
        reporter.emit_final(
            CommandOutcome("success", True, None),
            None,
            exit_code=0,
        )

    assert captured.value.applied is True
    assert captured.value.document is not None
    assert captured.value.document["applied"] is True
    assert captured.value.document["exit_code"] == 1
    operation_state = captured.value.document["operation_state"]
    assert isinstance(operation_state, dict)
    assert cast(dict[str, object], operation_state)["report_written"] is False


def test_report_destination_rejects_the_target_pyproject(
    tmp_path: Path,
) -> None:
    options = _text_options(tmp_path)
    options = replace(options, report_file=options.pyproject)
    workspace = WorkspaceContext(
        tmp_path,
        tmp_path,
        None,
        tmp_path / "uv.lock",
        False,
    )

    with pytest.raises(ReportError, match="protected"):
        preflight_report_destination(
            options,
            workspace,
            log_path=tmp_path / "logs/run.log",
        )
