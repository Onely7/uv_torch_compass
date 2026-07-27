import platform

import pytest

from uv_torch_compass.errors import ConfigurationError
from uv_torch_compass.platform_requirement import RequiredEnvironment


def test_normalizes_current_linux_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")

    required = RequiredEnvironment.current_linux()

    assert required.marker == (
        "sys_platform == 'linux' and platform_machine == 'x86_64'"
    )


def test_rejects_unknown_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "machine", lambda: "mips")

    with pytest.raises(ConfigurationError, match="unsupported"):
        RequiredEnvironment.current_linux()
