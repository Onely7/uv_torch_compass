"""Inspect NVIDIA hardware visible to the current process."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from uv_torch_compass.command_runner import ProcessRunner
from uv_torch_compass.cuda_compatibility import (
    CompatibilityDecision,
    CompatibilityPolicy,
    decide_compatibility,
)
from uv_torch_compass.errors import CommandError, ConfigurationError


def re_search_cuda(output: str) -> str:
    """Extract the maximum CUDA version displayed by nvidia-smi."""
    match = re.search(r"CUDA Version:\s*([0-9.]+)", output)
    return match.group(1) if match else ""


def _first_line(output: str) -> str:
    return next((line.strip() for line in output.splitlines() if line.strip()), "")


@dataclass(frozen=True, slots=True)
class NvidiaDevice:
    """Describe one NVIDIA device visible to nvidia-smi."""

    index: str
    uuid: str
    name: str
    driver_version: str
    memory_free_mib: int = 0


@dataclass(frozen=True, slots=True)
class NvidiaSnapshot:
    """Describe visible NVIDIA devices and the driver's maximum CUDA version."""

    devices: tuple[NvidiaDevice, ...]
    driver_cuda_max: str
    selected: NvidiaDevice

    def compatibility_for(
        self, backend: str, policy: CompatibilityPolicy
    ) -> CompatibilityDecision:
        """Classify a backend against the selected GPU's driver."""
        return decide_compatibility(
            backend,
            driver_version=self.selected.driver_version,
            reported_cuda_maximum=self.driver_cuda_max,
            policy=policy,
        )


@dataclass(frozen=True, slots=True)
class NvidiaInspector:
    """Inspect NVIDIA devices through a bounded nvidia-smi subprocess."""

    executable: Path
    runner: ProcessRunner
    timeout_seconds: int = 30

    @classmethod
    def discover(cls, runner: ProcessRunner) -> NvidiaInspector | None:
        """Create an inspector when nvidia-smi is present in PATH."""
        resolved = shutil.which("nvidia-smi")
        return None if resolved is None else cls(Path(resolved).resolve(), runner)

    def inspect(self, requested_device: str | None) -> NvidiaSnapshot:
        """Return visible devices and select an index or UUID.

        Raises:
            CommandError: If nvidia-smi fails or returns unusable data.
            ConfigurationError: If the requested device is not visible.
        """
        query = self.runner.run(
            [
                self.executable,
                "--query-gpu=index,uuid,name,driver_version,memory.free",
                "--format=csv,noheader,nounits",
            ],
            timeout_seconds=self.timeout_seconds,
        )
        if query.returncode != 0:
            diagnostic = _first_line(query.stderr or query.stdout)
            suffix = f": {diagnostic}" if diagnostic else ""
            raise CommandError(f"nvidia-smi device query failed{suffix}")
        devices = _parse_devices(query.stdout)
        if not devices:
            raise CommandError("nvidia-smi reported no visible NVIDIA devices")

        selected = max(devices, key=lambda device: device.memory_free_mib)
        if requested_device is not None:
            matches = [
                device
                for device in devices
                if requested_device in {device.index, device.uuid}
            ]
            if not matches:
                raise ConfigurationError(
                    f"CUDA device {requested_device!r} is not visible to nvidia-smi"
                )
            selected = matches[0]

        summary = self.runner.run(
            [self.executable], timeout_seconds=self.timeout_seconds
        )
        if summary.returncode != 0:
            diagnostic = _first_line(summary.stderr or summary.stdout)
            suffix = f": {diagnostic}" if diagnostic else ""
            raise CommandError(f"nvidia-smi summary failed{suffix}")
        driver_cuda_max = re_search_cuda(summary.stdout)
        if not driver_cuda_max:
            raise CommandError("nvidia-smi summary did not contain a CUDA version")
        return NvidiaSnapshot(devices, driver_cuda_max, selected)


def _parse_devices(output: str) -> tuple[NvidiaDevice, ...]:
    devices: list[NvidiaDevice] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",", 4)]
        if len(values) != 5 or not all(values):
            raise CommandError(f"nvidia-smi returned an invalid device row: {line!r}")
        try:
            memory_free_mib = int(values[4])
        except ValueError as exc:
            raise CommandError(
                f"nvidia-smi returned invalid free memory: {values[4]!r}"
            ) from exc
        devices.append(
            NvidiaDevice(
                index=values[0],
                uuid=values[1],
                name=values[2],
                driver_version=values[3],
                memory_free_mib=memory_free_mib,
            )
        )
    return tuple(devices)
