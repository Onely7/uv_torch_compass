from uv_torch_compass.domain import ProjectRequirements, Scope, ScopedRequirement
from uv_torch_compass.installed_metadata import InstalledDistribution
from uv_torch_compass.source_ownership import (
    ManagedSourceAnchor,
    derive_managed_source_anchors,
)


def _requirements(*items: ScopedRequirement) -> ProjectRequirements:
    scopes = tuple(dict.fromkeys(item.scope for item in items))
    return ProjectRequirements(
        ">=3.10",
        "",
        items,
        (),
        scopes,
    )


def test_maps_transitive_torch_to_the_introducing_extra() -> None:
    extra = Scope("extra", "serve")
    requirements = _requirements(ScopedRequirement(extra, "vllm[cuda]>=0.19"))
    distributions = (
        InstalledDistribution(
            "vllm",
            "0.19.1",
            (
                "torch==2.10.0",
                "ignored; extra == 'other'",
            ),
        ),
        InstalledDistribution("torch", "2.10.0"),
    )

    anchors = derive_managed_source_anchors(distributions, requirements)

    assert anchors == (ManagedSourceAnchor("torch", extra),)


def test_base_scope_wins_when_multiple_roots_reach_same_package() -> None:
    base = Scope("base")
    group = Scope("group", "training")
    requirements = _requirements(
        ScopedRequirement(base, "vllm"),
        ScopedRequirement(group, "accelerator"),
    )
    distributions = (
        InstalledDistribution("vllm", "1", ("shared-runtime",)),
        InstalledDistribution("accelerator", "1", ("shared-runtime",)),
        InstalledDistribution("shared-runtime", "1", ("torch",)),
        InstalledDistribution("torch", "2.10"),
    )

    anchors = derive_managed_source_anchors(distributions, requirements)

    assert anchors == (ManagedSourceAnchor("torch", base),)


def test_incomplete_metadata_keeps_anchors_in_selected_scopes() -> None:
    group = Scope("group", "training")
    requirements = _requirements(ScopedRequirement(group, "opaque-framework"))
    distributions = (
        InstalledDistribution("opaque-framework", "1"),
        InstalledDistribution("torch", "2.10"),
    )

    anchors = derive_managed_source_anchors(distributions, requirements)

    assert anchors == (ManagedSourceAnchor("torch", group),)
