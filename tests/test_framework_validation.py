import json
from typing import cast

import pytest

from uv_torch_compass.candidate_failures import FrameworkProbeTrigger
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
            FrameworkProbeTrigger.EXPLICIT,
        ),
    )
    assert framework_validation_document(validations)[0]["framework"] == "vllm"


@pytest.mark.parametrize(
    "output, message",
    [
        ("", "no result"),
        ("not-json", "valid JSON"),
        ('{"schema_version": 3}', "schema"),
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


def test_schema_two_parses_bounded_exception_and_package_versions() -> None:
    output = json.dumps(
        {
            "schema_version": 2,
            "results": [
                {
                    "framework": "vllm",
                    "status": "FAIL",
                    "version": "0.6.0",
                    "import_test": "FAIL",
                    "native_extension_test": "FAIL",
                    "platform_test": "FAIL",
                    "platform": "unknown",
                    "error": "ImportError: cannot import DTensor",
                    "trigger": "automatic",
                    "exception": {
                        "type": "ImportError",
                        "message": "cannot import DTensor",
                        "missing_symbol": "DTensor",
                        "missing_module": None,
                        "consumer_package": "transformers",
                        "provider_package": "torch",
                        "frames": [
                            {
                                "module": "transformers",
                                "filename": "modeling_utils.py",
                                "function": "<module>",
                                "line_number": 10,
                            }
                        ],
                    },
                    "packages": [
                        {
                            "name": "transformers",
                            "version": "5.14.1",
                            "source_url": "https://pypi.org/simple",
                        }
                    ],
                }
            ],
        }
    )

    results = parse_framework_probe(
        output,
        (),
        allow_automatic=True,
        require_success=False,
    )
    document = framework_validation_document(results)

    assert results[0].exception is not None
    assert results[0].exception.frames[0].filename == "modeling_utils.py"
    assert results[0].packages[0].name == "transformers"
    exception = cast(dict[str, object], document[0]["exception"])
    packages = cast(list[dict[str, object]], document[0]["packages"])
    assert exception["missing_symbol"] == "DTensor"
    assert packages[0]["version"] == "5.14.1"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("exception", [], "exception must be an object"),
        ("packages", {}, "package versions must be an array"),
    ],
)
def test_schema_two_rejects_invalid_nested_values(
    field: str, value: object, message: str
) -> None:
    raw = json.loads(_output())
    raw["schema_version"] = 2
    raw["results"][0][field] = value

    with pytest.raises(ProbeError, match=message):
        parse_framework_probe(json.dumps(raw), (FrameworkProbe.VLLM,))
