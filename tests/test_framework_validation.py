import json

import pytest

from uv_torch_compass.domain import FrameworkProbe
from uv_torch_compass.errors import ProbeError
from uv_torch_compass.framework_validation import (
    FrameworkValidation,
    framework_validation_document,
    parse_framework_probe,
)


def _output(**overrides: str) -> str:
    result = {
        "framework": "vllm",
        "status": "PASS",
        "version": "0.19.1",
        "import_test": "PASS",
        "native_extension_test": "PASS",
        "platform_test": "PASS",
        "platform": "CudaPlatform",
        "error": "",
        "trigger": "explicit",
    }
    result.update(overrides)
    return json.dumps({"schema_version": 1, "results": [result]})


def test_parses_requested_framework_validation() -> None:
    validations = parse_framework_probe(_output(), (FrameworkProbe.VLLM,))

    assert validations == (
        FrameworkValidation(
            FrameworkProbe.VLLM,
            "PASS",
            "0.19.1",
            "PASS",
            "PASS",
            "PASS",
            "CudaPlatform",
            "",
            "explicit",
        ),
    )
    assert framework_validation_document(validations)[0]["framework"] == "vllm"


@pytest.mark.parametrize(
    "output, message",
    [
        ("", "no result"),
        ("not-json", "valid JSON"),
        ('{"schema_version": 2}', "schema"),
        ('{"schema_version": 1, "results": {}}', "array"),
        ('{"schema_version": 1, "results": ["invalid"]}', "object"),
        (
            '{"schema_version": 1, "results": [{"framework": 1}]}',
            "non-string",
        ),
        (
            _output(framework="unknown"),
            "unknown framework",
        ),
        (
            _output(trigger="unknown"),
            "invalid trigger",
        ),
        (
            '{"schema_version": 1, "results": []}',
            "exactly",
        ),
        (
            _output(status="FAIL", error="native extension missing"),
            "native extension missing",
        ),
    ],
)
def test_rejects_invalid_or_failed_framework_results(
    output: str,
    message: str,
) -> None:
    with pytest.raises(ProbeError, match=message):
        parse_framework_probe(output, (FrameworkProbe.VLLM,))


def test_duplicate_result_does_not_satisfy_distinct_contract() -> None:
    document = json.loads(_output())
    document["results"].append(document["results"][0])

    validations = parse_framework_probe(
        json.dumps(document),
        (FrameworkProbe.VLLM,),
    )

    assert len(validations) == 1


def test_accepts_automatic_framework_results_when_enabled() -> None:
    validations = parse_framework_probe(
        _output(trigger="automatic"),
        (),
        allow_automatic=True,
    )

    assert validations[0].trigger == "automatic"
