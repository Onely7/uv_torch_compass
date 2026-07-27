"""Inspect installed distribution metadata without importing third-party code."""

from __future__ import annotations

from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path

from packaging.utils import canonicalize_name

from uv_torch_compass.errors import ProbeError


@dataclass(frozen=True, slots=True)
class InstalledDistribution:
    """Describe one distribution found in an isolated candidate environment."""

    name: str
    version: str


def read_installed_distributions(venv: Path) -> tuple[InstalledDistribution, ...]:
    """Read package names and versions from bounded ``dist-info`` metadata.

    Args:
        venv: Candidate environment whose metadata should be inspected.

    Returns:
        Distributions ordered by normalized package name.

    Raises:
        ProbeError: If installed metadata is malformed or escapes the environment.
    """
    root = venv.resolve()
    distributions: dict[str, InstalledDistribution] = {}
    for metadata_path in root.rglob("*.dist-info/METADATA"):
        resolved = metadata_path.resolve()
        if not resolved.is_relative_to(root):
            raise ProbeError("installed metadata escaped the candidate environment")
        try:
            message = BytesParser(policy=compat32).parsebytes(resolved.read_bytes())
        except OSError as exc:
            raise ProbeError(f"failed to read installed metadata: {exc}") from exc
        raw_name = message.get("Name")
        raw_version = message.get("Version")
        if not raw_name or not raw_version:
            raise ProbeError(f"incomplete installed metadata at {resolved}")
        name = canonicalize_name(raw_name)
        distributions[name] = InstalledDistribution(name, raw_version)
    return tuple(distributions[name] for name in sorted(distributions))
