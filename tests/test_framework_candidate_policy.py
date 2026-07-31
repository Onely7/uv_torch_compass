from pathlib import Path

from uv_torch_compass.domain import ProjectRequirements, Scope, ScopedRequirement
from uv_torch_compass.framework_candidate_policy import (
    direct_vllm_candidate_constraint,
)


def _requirements(raw: str) -> ProjectRequirements:
    scope = Scope("base")
    return ProjectRequirements(
        ">=3.12",
        "",
        (ScopedRequirement(scope, raw),),
        (),
        (scope,),
    )


def test_exact_official_vllm_uses_reviewed_backend(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "example"\nversion = "0"\n', encoding="utf-8"
    )

    constraint = direct_vllm_candidate_constraint(
        _requirements("vllm==0.6.0"),
        pyproject,
    )

    assert constraint is not None
    assert constraint.required_backend == "cu121"
    assert constraint.resolved_version == "0.6.0"


def test_range_or_custom_source_is_not_prefiltered(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "example"\nversion = "0"\n'
        '[tool.uv.sources]\nvllm = { path = "../vllm" }\n',
        encoding="utf-8",
    )

    assert (
        direct_vllm_candidate_constraint(_requirements("vllm>=0.6"), pyproject) is None
    )
    assert (
        direct_vllm_candidate_constraint(_requirements("vllm==0.6.0"), pyproject)
        is None
    )
