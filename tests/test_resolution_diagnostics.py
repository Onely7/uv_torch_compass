from pathlib import Path

import pytest

from uv_torch_compass.domain import (
    BackendCandidate,
    ResolutionFailureKind,
)
from uv_torch_compass.resolution_diagnostics import interpret_uv_failure

_FIXTURES = Path(__file__).parent / "fixtures" / "uv_failures"


def _fixture(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


def test_reports_transitive_torch_requirement_and_candidate_index() -> None:
    failure = interpret_uv_failure(
        _fixture("transitive_torch_unavailable.txt"),
        candidate=BackendCandidate("cu121"),
        dependency_roots=("vllm>=0.25.0",),
    )

    assert failure.kind is ResolutionFailureKind.NO_COMPATIBLE_DISTRIBUTION
    assert failure.package is not None
    assert failure.package.name == "torch"
    assert failure.package.version == "2.10.0"
    assert failure.package.requirement == "torch==2.10.0"
    assert failure.required_by == ("vllm>=0.25.0", "torch==2.10.0")
    assert failure.index is not None
    assert failure.index.name == "pytorch-cu121"
    assert failure.index.url == "https://download.pytorch.org/whl/cu121"


def test_reports_unavailable_wheel_and_platform() -> None:
    failure = interpret_uv_failure(
        _fixture("wheel_unavailable.txt"),
        candidate=BackendCandidate("cu121"),
        dependency_roots=("vllm",),
    )

    assert failure.kind is ResolutionFailureKind.WHEEL_UNAVAILABLE
    assert failure.package is not None
    assert failure.package.name == "xgrammar"
    assert failure.package.version == "0.2.4"
    assert failure.platform == "manylinux_2_39_x86_64"
    assert failure.available_wheel_platforms == (
        "manylinux_2_28_aarch64",
        "macosx_11_0_arm64",
    )


def test_reports_dependency_conflict() -> None:
    failure = interpret_uv_failure(
        _fixture("dependency_conflict.txt"),
        candidate=BackendCandidate("cpu"),
        dependency_roots=("project-a", "project-b"),
    )

    assert failure.kind is ResolutionFailureKind.DEPENDENCY_CONFLICT
    assert failure.package is not None
    assert failure.package.name == "numpy"


def test_reports_multihop_dependency_path_and_registry_absence() -> None:
    failure = interpret_uv_failure(
        "Because vllm==0.19.1 depends on engine==1 and engine==1 depends on "
        "torch==2.10.0 and torch==2.10.0 was not found in the package registry, "
        "the requirements are unsatisfiable.",
        candidate=BackendCandidate("cu126"),
        dependency_roots=("vllm==0.19.1",),
    )

    assert failure.kind is ResolutionFailureKind.NO_COMPATIBLE_DISTRIBUTION
    assert failure.package is not None
    assert failure.package.requirement == "torch==2.10.0"
    assert failure.required_by == (
        "vllm==0.19.1",
        "engine==1",
        "torch==2.10.0",
    )
    assert failure.index is not None
    assert failure.index.name == "pytorch-cu126"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("HTTP 401 Unauthorized", ResolutionFailureKind.AUTHENTICATION),
        ("failed to connect: DNS failure", ResolutionFailureKind.NETWORK),
        ("Failed to build `legacy==1`", ResolutionFailureKind.BUILD_FAILURE),
        ("unrecognized resolver prose", ResolutionFailureKind.UNKNOWN),
        (
            "packaging was found on https://test.pypi.org/simple, but not at "
            "the requested version",
            ResolutionFailureKind.NO_COMPATIBLE_DISTRIBUTION,
        ),
    ],
)
def test_classifies_other_failure_boundaries(
    message: str, expected: ResolutionFailureKind
) -> None:
    failure = interpret_uv_failure(
        message,
        candidate=BackendCandidate("cpu"),
        dependency_roots=("example",),
    )

    assert failure.kind is expected


def test_redacts_and_bounds_untrusted_uv_output() -> None:
    output = (
        "\x1b[31mAuthorization: Bearer visible-secret\x1b[0m\n"
        "https://user:password@example.invalid/simple?token=query-secret\n"
        + "x"
        * 100_000
    )

    failure = interpret_uv_failure(
        output,
        candidate=BackendCandidate("cpu"),
        dependency_roots=("example",),
    )

    serialized = repr(failure)
    assert "visible-secret" not in serialized
    assert "password" not in serialized
    assert "query-secret" not in serialized


def test_article_is_not_accepted_as_an_opaque_uv_package_name() -> None:
    """Do not mistake English prose for a distribution identity."""
    failure = interpret_uv_failure(
        "Because no version of the package can be used, the requirements "
        "are unsatisfiable.",
        candidate=BackendCandidate("cpu"),
        dependency_roots=("example",),
    )

    assert failure.package is None


def test_article_from_index_prose_is_not_accepted_as_a_package_name() -> None:
    failure = interpret_uv_failure(
        "the was found on https://pypi.org/simple, but resolution still failed",
        candidate=BackendCandidate("cpu"),
        dependency_roots=("example",),
    )

    assert failure.package is None
