"""Map transitive PyTorch packages back to selected dependency scopes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from packaging.markers import UndefinedEnvironmentName
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from uv_torch_compass.domain import PYTORCH_PACKAGES, ProjectRequirements, Scope
from uv_torch_compass.installed_metadata import InstalledDistribution


@dataclass(frozen=True, slots=True)
class ManagedSourceAnchor:
    """Identify a tool-managed direct requirement and its owning scope."""

    package: str
    scope: Scope


def derive_managed_source_anchors(
    distributions: tuple[InstalledDistribution, ...],
    requirements: ProjectRequirements,
) -> tuple[ManagedSourceAnchor, ...]:
    """Find scopes that introduce transitive PyTorch packages.

    Installed ``Requires-Dist`` metadata is the authoritative dependency graph.
    If metadata is incomplete, selected root scopes are used as a fail-safe so
    an anchor is not silently moved into base dependencies.
    """
    graph = {distribution.name: distribution for distribution in distributions}
    installed_pytorch = set(PYTORCH_PACKAGES).intersection(graph)
    direct_pytorch = {
        item.package
        for item in requirements.selected
        if item.package in PYTORCH_PACKAGES
    }
    transitive = installed_pytorch.difference(direct_pytorch)
    owners: dict[str, set[Scope]] = {package: set() for package in transitive}

    for root in requirements.selected:
        reachable = _reachable_packages(
            root.package,
            tuple(str(extra) for extra in root.requirement.extras),
            graph,
            requirements.environment(),
        )
        for package in transitive.intersection(reachable):
            owners[package].add(root.scope)

    fallback_scopes = {item.scope for item in requirements.selected}
    anchors: list[ManagedSourceAnchor] = []
    for package in sorted(transitive):
        scopes = owners[package] or fallback_scopes
        base = next((scope for scope in scopes if scope.kind == "base"), None)
        selected_scopes = (
            (base,)
            if base is not None
            else tuple(sorted(scopes, key=lambda scope: scope.label))
        )
        anchors.extend(
            ManagedSourceAnchor(package, scope)
            for scope in selected_scopes
            if scope is not None
        )
    return tuple(anchors)


def _reachable_packages(
    root: str,
    root_extras: tuple[str, ...],
    graph: dict[str, InstalledDistribution],
    environment: dict[str, str],
) -> frozenset[str]:
    discovered: set[str] = set()
    pending = deque([(root, root_extras)])
    while pending:
        package, active_extras = pending.popleft()
        if package in discovered:
            continue
        discovered.add(package)
        distribution = graph.get(package)
        if distribution is None:
            continue
        for raw_requirement in distribution.requires_dist:
            try:
                requirement = Requirement(raw_requirement)
            except InvalidRequirement:
                continue
            if not _marker_applies(requirement, environment, active_extras):
                continue
            pending.append(
                (
                    str(canonicalize_name(requirement.name)),
                    tuple(str(extra) for extra in requirement.extras),
                )
            )
    return frozenset(discovered)


def _marker_applies(
    requirement: Requirement,
    environment: dict[str, str],
    active_extras: tuple[str, ...],
) -> bool:
    marker = requirement.marker
    if marker is None:
        return True
    extras = active_extras or ("",)
    try:
        return any(marker.evaluate({**environment, "extra": extra}) for extra in extras)
    except UndefinedEnvironmentName:
        return False
