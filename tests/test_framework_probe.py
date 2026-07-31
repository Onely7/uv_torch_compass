import json
from types import SimpleNamespace

from uv_torch_compass import framework_probe


class CudaPlatform:
    def is_cuda(self) -> bool:
        return True

    def is_cpu(self) -> bool:
        return False


def test_vllm_probe_validates_import_native_extension_and_platform(
    monkeypatch,
) -> None:
    platforms = SimpleNamespace(current_platform=CudaPlatform())
    monkeypatch.setattr(
        framework_probe.importlib.metadata, "version", lambda _: "0.19.1"
    )
    monkeypatch.setattr(
        framework_probe.importlib.util,
        "find_spec",
        lambda name: object() if name == "vllm._C" else None,
    )
    monkeypatch.setattr(
        framework_probe.importlib,
        "import_module",
        lambda name: platforms if name == "vllm.platforms" else object(),
    )

    result = framework_probe._probe_vllm("cu128")

    assert result["status"] == "PASS"
    assert result["version"] == "0.19.1"
    assert result["platform"] == "CudaPlatform"
    assert result["trigger"] == "explicit"


def test_vllm_probe_reports_bounded_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        framework_probe.importlib.metadata, "version", lambda _: "0.19.1"
    )
    monkeypatch.setattr(framework_probe.importlib, "import_module", lambda _: object())
    monkeypatch.setattr(framework_probe.importlib.util, "find_spec", lambda _: None)

    result = framework_probe._probe_vllm("cpu")

    assert result["status"] == "FAIL"
    assert "native extension" in result["error"]


def test_main_emits_one_json_document(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        framework_probe,
        "_probe_vllm",
        lambda _: {
            "framework": "vllm",
            "status": "PASS",
            "version": "1",
            "import_test": "PASS",
            "native_extension_test": "PASS",
            "platform_test": "PASS",
            "platform": "CudaPlatform",
            "error": "",
        },
    )

    status = framework_probe.main(
        ["--framework", "vllm", "--expected-backend", "cu128"]
    )

    document = json.loads(capsys.readouterr().out)
    assert status == 0
    assert document["schema_version"] == 2
    assert document["results"][0]["framework"] == "vllm"
    assert document["results"][0]["trigger"] == "explicit"


def test_main_auto_detects_installed_vllm(monkeypatch, capsys) -> None:
    monkeypatch.setattr(framework_probe, "_distribution_installed", lambda _: True)
    monkeypatch.setattr(
        framework_probe,
        "_probe_vllm",
        lambda _: {
            "framework": "vllm",
            "status": "PASS",
            "version": "1",
            "import_test": "PASS",
            "native_extension_test": "PASS",
            "platform_test": "PASS",
            "platform": "CudaPlatform",
            "error": "",
            "trigger": "explicit",
        },
    )

    status = framework_probe.main(["--auto-detect", "--expected-backend", "cu128"])

    document = json.loads(capsys.readouterr().out)
    assert status == 0
    assert document["results"][0]["trigger"] == "automatic"


def test_main_fails_when_framework_check_fails(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        framework_probe,
        "_probe_vllm",
        lambda _: {"status": "FAIL"},
    )

    status = framework_probe.main(["--framework", "vllm", "--expected-backend", "cpu"])

    assert status == 1
    assert json.loads(capsys.readouterr().out)["results"][0]["status"] == "FAIL"
