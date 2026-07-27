"""Model dependency roots selected for candidate resolution."""

from __future__ import annotations

from dataclasses import dataclass

from uv_torch_compass.domain import PYTORCH_PACKAGES, ScopedRequirement


@dataclass(frozen=True, slots=True)
class SelectedDependencyRoots:
    """Provide stable views over requirements selected from project scopes."""

    requirements: tuple[ScopedRequirement, ...]

    @property
    def candidate_requirements(self) -> tuple[str, ...]:
        """Return every selected root for complete candidate resolution."""
        values: list[str] = []
        seen: set[str] = set()
        for item in self.requirements:
            normalized = str(item.requirement)
            if normalized not in seen:
                seen.add(normalized)
                values.append(normalized)
        return tuple(values)

    @property
    def legacy_probe_requirements(self) -> tuple[str, ...]:
        """Return the requirements accepted by the legacy candidate probe."""
        values: list[str] = []
        seen: set[str] = set()
        for item in self.requirements:
            if item.package not in {*PYTORCH_PACKAGES, "numpy"}:
                continue
            normalized = str(item.requirement)
            if normalized not in seen:
                seen.add(normalized)
                values.append(normalized)
        return tuple(values)
