"""Remove common credential forms from externally supplied diagnostic text."""

from __future__ import annotations

import re

_URL_USERINFO = re.compile(r"(https?://)[^/@\s]+@", re.IGNORECASE)
_SECRET_QUERY = re.compile(
    r"([?&](?:token|key|password|secret|signature|credential)=)[^&\s]+",
    re.IGNORECASE,
)
_AUTHORIZATION = re.compile(
    r"(?im)^(authorization|proxy-authorization|cookie|set-cookie|x-api-key):\s*.+$"
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?im)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY|AUTH)"
    r"[A-Z0-9_]*)\s*([=:])\s*([^\s]+)"
)
_JSON_SECRET = re.compile(
    r'(?i)("(?:access[_-]?token|refresh[_-]?token|api[_-]?key|password|secret)"'
    r"\s*:\s*)"
    r'"[^"]*"'
)
_OPTION_SECRET = re.compile(
    r"(?i)(--(?:token|password|secret|api-key|credential)(?:=|\s+))\S+"
)


def redact(value: str) -> str:
    """Remove common URL, header, and variable credential forms."""
    redacted = _URL_USERINFO.sub(r"\1<redacted>@", value)
    redacted = _SECRET_QUERY.sub(r"\1<redacted>", redacted)
    redacted = _AUTHORIZATION.sub(r"\1: <redacted>", redacted)
    redacted = _SECRET_ASSIGNMENT.sub(r"\1\2<redacted>", redacted)
    redacted = _JSON_SECRET.sub(r'\1"<redacted>"', redacted)
    return _OPTION_SECRET.sub(r"\1<redacted>", redacted)
