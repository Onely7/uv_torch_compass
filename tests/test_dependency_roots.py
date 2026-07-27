from uv_torch_compass.dependency_roots import SelectedDependencyRoots
from uv_torch_compass.domain import Scope, ScopedRequirement


def test_legacy_probe_view_preserves_order_and_deduplicates() -> None:
    scope = Scope("base")
    roots = SelectedDependencyRoots(
        (
            ScopedRequirement(scope, "vllm==0.19.1"),
            ScopedRequirement(scope, "torch>=2"),
            ScopedRequirement(scope, "numpy"),
            ScopedRequirement(scope, "torch>=2"),
        )
    )

    assert roots.legacy_probe_requirements == ("torch>=2", "numpy")
