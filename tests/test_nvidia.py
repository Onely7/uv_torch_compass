from pathlib import Path

import pytest

from uv_torch_compass.command_runner import CommandResult
from uv_torch_compass.errors import CommandError, ConfigurationError
from uv_torch_compass.nvidia import NvidiaInspector, re_search_cuda


class FakeRunner:
    def __init__(
        self,
        *,
        query_code: int = 0,
        summary_code: int = 0,
        summary: str = "Driver Version: 570.1 CUDA Version: 12.8",
    ) -> None:
        self.query_code = query_code
        self.summary_code = summary_code
        self.summary = summary

    def run(self, arguments, *, cwd=None, env=None, timeout_seconds=None):
        del cwd, env, timeout_seconds
        if "--query-gpu=index,uuid,name,driver_version" in arguments:
            return CommandResult(
                self.query_code,
                "0, GPU-AAA, First GPU, 570.1\n1, GPU-BBB, Second GPU, 570.1\n",
                "query failed" if self.query_code else "",
            )
        return CommandResult(
            self.summary_code,
            self.summary,
            "summary failed" if self.summary_code else "",
        )


def test_inspector_selects_index_or_uuid_and_filters_cuda() -> None:
    inspector = NvidiaInspector(Path("/usr/bin/nvidia-smi"), FakeRunner())
    by_index = inspector.inspect("1")
    by_uuid = inspector.inspect("GPU-AAA")

    assert by_index.selected.name == "Second GPU"
    assert by_uuid.selected.index == "0"
    assert by_uuid.supports_backend("cu128")
    assert not by_uuid.supports_backend("cu130")


def test_inspector_rejects_missing_device_and_failed_commands() -> None:
    inspector = NvidiaInspector(Path("/usr/bin/nvidia-smi"), FakeRunner())
    with pytest.raises(ConfigurationError, match="not visible"):
        inspector.inspect("GPU-MISSING")
    with pytest.raises(CommandError, match="device query"):
        NvidiaInspector(Path("/usr/bin/nvidia-smi"), FakeRunner(query_code=1)).inspect(
            None
        )
    with pytest.raises(CommandError, match="summary"):
        NvidiaInspector(
            Path("/usr/bin/nvidia-smi"), FakeRunner(summary_code=1)
        ).inspect(None)
    with pytest.raises(CommandError, match="CUDA version"):
        NvidiaInspector(
            Path("/usr/bin/nvidia-smi"), FakeRunner(summary="unparseable")
        ).inspect(None)


def test_cuda_summary_parser_handles_missing_value() -> None:
    assert re_search_cuda("CUDA Version: 12.8") == "12.8"
    assert re_search_cuda("no CUDA data") == ""
