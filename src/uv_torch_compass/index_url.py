"""Validate and canonicalize official PyTorch index URLs."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

_PATH = re.compile(r"/whl/(?:(?:nightly)/)?(?:cpu|cu[0-9]{2,3})")


def canonical_official_pytorch_url(value: str) -> str | None:
    """Return a canonical official URL, or ``None`` for any unsafe variant."""
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme.lower() != "https"
        or parsed.hostname != "download.pytorch.org"
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or parsed.query
        or parsed.fragment
    ):
        return None
    path = parsed.path.rstrip("/")
    if _PATH.fullmatch(path) is None:
        return None
    return f"https://download.pytorch.org{path}"
