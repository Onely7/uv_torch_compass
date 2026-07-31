import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from uv_torch_compass.cli import create_parser, main
from uv_torch_compass.configuration import resolve_options
from uv_torch_compass.cuda_compatibility import CompatibilityPolicy
from uv_torch_compass.domain import (
    BackendCandidate,
    BackendKind,
    CandidateAttempt,
    Channel,
    CommandOutcome,
    FailedIndex,
    FailedPackage,
    FrameworkProbe,
    Operation,
    OutputFormat,
    ProbeProfile,
    ResolutionFailure,
    ResolutionFailureKind,
    RuntimeReport,
)
from uv_torch_compass.errors import (
    CandidateResolutionError,
    CommandError,
    ConfigurationError,
)
from uv_torch_compass.workspace import WorkspaceContext


def _write_project(path: Path, tool_settings: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "[project]\n"
            'name = "target"\n'
            'version = "0.1.0"\n'
            'requires-python = ">=3.10"\n'
            'dependencies = ["torch>=2.5"]\n'
            f"{tool_settings}"
        ),
        encoding="utf-8",
    )


def test_parser_exposes_apply_plan_and_check() -> None:
    parser = create_parser()

    assert parser.parse_args(["apply"]).operation == "apply"
    assert parser.parse_args(["plan"]).operation == "plan"
    assert parser.parse_args(["check"]).operation == "check"


def test_parser_rejects_legacy_flat_invocation() -> None:
    with pytest.raises(SystemExit) as captured:
        create_parser().parse_args(["--backend", "cpu"])

    assert captured.value.code == 2


def test_cli_overrides_namespaced_environment_and_project_settings(
    tmp_path: Path,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    _write_project(
        pyproject,
        "\n[tool.uv-torch-compass]\n"
        'backend = "auto"\nchannel = "stable"\nextras = ["table"]\n'
        'cuda-compatibility = "strict"\nprobe-profile = "standard"\n',
    )
    namespace = create_parser().parse_args(
        [
            "apply",
            "--backend",
            "cuda",
            "--channel",
            "nightly",
            "--extra",
            "cli",
            "--cuda-compatibility",
            "minor",
            "--probe-profile",
            "compile",
            "--framework-probe",
            "vllm",
            "--torch",
            ">=2.7",
        ]
    )

    options = resolve_options(
        namespace,
        environ={
            "UV_TORCH_COMPASS_BACKEND": "cpu",
            "UV_TORCH_COMPASS_EXTRAS": "environment",
        },
        cwd=tmp_path,
    )

    assert options.operation is Operation.APPLY
    assert options.backend.kind is BackendKind.CUDA
    assert options.channel is Channel.NIGHTLY
    assert options.extras == ("cli",)
    assert options.cuda_compatibility is CompatibilityPolicy.MINOR
    assert options.probe_profile is ProbeProfile.COMPILE
    assert options.framework_probes == (FrameworkProbe.VLLM,)
    assert options.requirement_overrides == ("torch>=2.7",)


def test_environment_lists_are_trimmed_and_deduplicated(tmp_path: Path) -> None:
    _write_project(tmp_path / "pyproject.toml")
    options = resolve_options(
        create_parser().parse_args(["plan"]),
        environ={
            "UV_TORCH_COMPASS_EXTRAS": "vision, audio,vision,,",
            "UV_TORCH_COMPASS_GROUPS": "training, training",
            "UV_TORCH_COMPASS_OUTPUT_FORMAT": "json",
            "UV_TORCH_COMPASS_CUDA_COMPATIBILITY": "minor",
            "UV_TORCH_COMPASS_PROBE_PROFILE": "compile",
            "UV_TORCH_COMPASS_FRAMEWORK_PROBES": "vllm,vllm",
        },
        cwd=tmp_path,
    )

    assert options.extras == ("vision", "audio")
    assert options.groups == ("training",)
    assert options.output_format is OutputFormat.JSON
    assert options.cuda_compatibility is CompatibilityPolicy.MINOR
    assert options.probe_profile is ProbeProfile.COMPILE
    assert options.framework_probes == (FrameworkProbe.VLLM,)


def test_new_cuda_settings_default_to_strict_standard(tmp_path: Path) -> None:
    _write_project(tmp_path / "pyproject.toml")

    options = resolve_options(
        create_parser().parse_args(["check"]), environ={}, cwd=tmp_path
    )

    assert options.cuda_compatibility is CompatibilityPolicy.STRICT
    assert options.probe_profile is ProbeProfile.STANDARD
    assert options.framework_probes == ()


def test_unknown_project_setting_is_rejected(tmp_path: Path) -> None:
    _write_project(
        tmp_path / "pyproject.toml",
        "\n[tool.uv-torch-compass]\nunknown = true\n",
    )

    with pytest.raises(ConfigurationError, match="unknown"):
        resolve_options(create_parser().parse_args(["plan"]), environ={}, cwd=tmp_path)


def test_pyproject_must_use_standard_filename(tmp_path: Path) -> None:
    _write_project(tmp_path / "custom.toml")

    with pytest.raises(ConfigurationError, match=r"pyproject\.toml"):
        resolve_options(
            create_parser().parse_args(
                ["plan", "--pyproject", str(tmp_path / "custom.toml")]
            ),
            environ={},
            cwd=tmp_path,
        )


def test_main_emits_json_for_configuration_failure(tmp_path: Path, capsys) -> None:
    _write_project(
        tmp_path / "pyproject.toml",
        "\n[tool.uv-torch-compass]\ntimeout = 0\n",
    )

    assert (
        main(
            [
                "plan",
                "--pyproject",
                str(tmp_path / "pyproject.toml"),
                "--output-format",
                "json",
            ]
        )
        == 1
    )
    document = json.loads(capsys.readouterr().out)
    assert document["schema_version"] == 7
    assert document["status"] == "failed"
    assert document["exit_code"] == 1
    assert document["errors"]


def _runtime_report() -> RuntimeReport:
    return RuntimeReport(
        2,
        BackendCandidate("cpu"),
        "2.7.0",
        "not-installed",
        "not-installed",
        "2.2.0",
        "none",
        "none",
        "NOT_APPLICABLE",
        "PASS",
        "NOT_REQUESTED",
        "NOT_REQUESTED",
    )


def test_main_emits_successful_json_document(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    _write_project(pyproject)
    workspace = WorkspaceContext(tmp_path, tmp_path, None, tmp_path / "uv.lock", False)

    class SuccessfulApplication:
        def __init__(self, *_arguments) -> None:
            pass

        def run(self):
            return SimpleNamespace(
                outcome=CommandOutcome("planned", False, _runtime_report()),
                workspace=workspace,
            )

    monkeypatch.setattr(
        "uv_torch_compass.cli.CompassApplication", SuccessfulApplication
    )
    monkeypatch.setattr(
        "uv_torch_compass.cli.UvCommandClient.discover",
        lambda *_arguments, **_keywords: object(),
    )

    status = main(
        [
            "plan",
            "--pyproject",
            str(pyproject),
            "--output-format",
            "json",
        ]
    )

    document = json.loads(capsys.readouterr().out)
    assert status == 0
    assert document["status"] == "planned"
    assert document["selected_backend"] == "cpu"
    assert document["request"]["cuda_compatibility"] == "strict"
    assert document["request"]["probe_profile"] == "standard"


@pytest.mark.parametrize(
    "failure, expected",
    [
        (ConfigurationError("known failure"), "known failure"),
        (RuntimeError("unexpected failure"), "unexpected RuntimeError"),
        (KeyboardInterrupt(), "operation interrupted"),
    ],
)
def test_main_reports_operational_failures(
    tmp_path: Path, capsys, monkeypatch, failure: BaseException, expected: str
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    _write_project(pyproject)

    class FailingApplication:
        def __init__(self, *_arguments) -> None:
            pass

        def run(self):
            raise failure

    monkeypatch.setattr("uv_torch_compass.cli.CompassApplication", FailingApplication)
    monkeypatch.setattr(
        "uv_torch_compass.cli.UvCommandClient.discover",
        lambda *_arguments, **_keywords: object(),
    )

    status = main(
        [
            "plan",
            "--pyproject",
            str(pyproject),
            "--output-format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert expected in json.loads(captured.out)["errors"][0]


def test_generic_command_failure_has_no_candidate_attempts(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    _write_project(pyproject)

    class FailingApplication:
        def __init__(self, *_arguments) -> None:
            pass

        def run(self):
            raise CommandError("no usable PyTorch backend was found; attempted: cu121")

    monkeypatch.setattr("uv_torch_compass.cli.CompassApplication", FailingApplication)
    monkeypatch.setattr(
        "uv_torch_compass.cli.UvCommandClient.discover",
        lambda *_arguments, **_keywords: object(),
    )

    status = main(
        [
            "plan",
            "--pyproject",
            str(pyproject),
            "--output-format",
            "json",
        ]
    )

    document = json.loads(capsys.readouterr().out)
    assert status == 1
    assert document["schema_version"] == 7
    assert document["candidate_attempts"] == []


def test_candidate_resolution_failure_reaches_json_boundary(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    _write_project(pyproject)
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
    attempt = CandidateAttempt(
        "cu121",
        "install",
        "failed",
        failure.summary,
        "strict",
        failure,
    )

    class FailingApplication:
        def __init__(self, *_arguments) -> None:
            pass

        def run(self):
            raise CandidateResolutionError("no usable backend", (attempt,))

    monkeypatch.setattr("uv_torch_compass.cli.CompassApplication", FailingApplication)
    monkeypatch.setattr(
        "uv_torch_compass.cli.UvCommandClient.discover",
        lambda *_arguments, **_keywords: object(),
    )

    status = main(
        [
            "plan",
            "--pyproject",
            str(pyproject),
            "--output-format",
            "json",
        ]
    )

    document = json.loads(capsys.readouterr().out)
    diagnostic = document["candidate_attempts"][0]["failure"]
    assert status == 1
    assert diagnostic["package"]["name"] == "torch"
    assert diagnostic["required_by"][0] == "vllm>=0.25.0"
    assert diagnostic["index"]["name"] == "pytorch-cu121"
    assert document["blocking_summary"]["common_blockers"][0]["package"] == "torch"


def test_main_reports_log_creation_failure(tmp_path: Path, capsys) -> None:
    pyproject = tmp_path / "pyproject.toml"
    _write_project(pyproject)
    blocked_log_dir = tmp_path / "not-a-directory"
    blocked_log_dir.write_text("file", encoding="utf-8")

    status = main(
        [
            "plan",
            "--pyproject",
            str(pyproject),
            "--log-dir",
            str(blocked_log_dir),
        ]
    )

    assert status == 1
    assert "private log" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments, environment, message",
    [
        (["plan", "--python", ""], {}, "python must not be empty"),
        (["plan"], {"UV_TORCH_COMPASS_TORCH": ""}, "torch must not be empty"),
        (
            ["plan", "--report-file", ""],
            {},
            "report-file must not be empty",
        ),
    ],
)
def test_explicit_empty_settings_are_rejected(
    tmp_path: Path,
    arguments: list[str],
    environment: dict[str, str],
    message: str,
) -> None:
    _write_project(tmp_path / "pyproject.toml")

    with pytest.raises(ConfigurationError, match=message):
        resolve_options(
            create_parser().parse_args(arguments),
            environ=environment,
            cwd=tmp_path,
        )
