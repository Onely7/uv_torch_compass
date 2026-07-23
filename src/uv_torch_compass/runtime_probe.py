"""Validate PyTorch inside an isolated or synchronized target environment.

The probe is ordinary source because a different Python interpreter executes it.
Keeping the boundary visible lets static analysis inspect every runtime check.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence


class RuntimeValidationError(Exception):
    """Carry a stable process status and diagnostic marker for probe failures."""

    def __init__(self, message: str, *, exit_code: int) -> None:
        """Initialize a validation failure visible to the parent workflow."""
        super().__init__(message)
        self.exit_code = exit_code


def main(arguments: Sequence[str] | None = None) -> int:
    """Run package, tensor, device, and NumPy checks and emit one JSON report.

    Args:
        arguments: Command arguments excluding the executable name.

    Returns:
        Zero after all checks pass, or a stable nonzero validation code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-torchvision", action="store_true")
    parser.add_argument("--validate-torchaudio", action="store_true")
    parser.add_argument("--expected-backend")
    options = parser.parse_args(arguments)
    try:
        report = _validate_runtime(
            validate_torchvision=options.validate_torchvision,
            validate_torchaudio=options.validate_torchaudio,
            expected_backend=options.expected_backend,
        )
    except RuntimeValidationError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    print(json.dumps(report, sort_keys=True))
    return 0


def _validate_runtime(
    *,
    validate_torchvision: bool,
    validate_torchaudio: bool = False,
    expected_backend: str | None,
) -> dict[str, str | int]:
    # These imports intentionally resolve in the target environment rather than
    # the environment that hosts uv-torch-compass.
    import numpy as np  # ty: ignore[unresolved-import]
    import torch  # ty: ignore[unresolved-import]

    torchvision_version = "not-installed"
    torchvision_test = "NOT_REQUESTED"
    if validate_torchvision:
        import torchvision  # ty: ignore[unresolved-import]

        torchvision_version = torchvision.__version__
        try:
            boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
            scores = torch.tensor([1.0])
            selected = torchvision.ops.nms(boxes, scores, 0.5)
            if selected.tolist() != [0]:
                raise ValueError("torchvision.ops.nms returned unexpected indexes")
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeValidationError(
                f"TORCHVISION_FAILED: {type(exc).__name__}: {exc}", exit_code=23
            ) from exc
        torchvision_test = "PASS"

    torchaudio_version = "not-installed"
    torchaudio_test = "NOT_REQUESTED"
    if validate_torchaudio:
        import torchaudio  # ty: ignore[unresolved-import]

        torchaudio_version = torchaudio.__version__
        try:
            waveform = torch.tensor([[0.25, -0.25]])
            gained = torchaudio.functional.gain(waveform, 0.0)
            if gained.tolist() != waveform.tolist():
                raise ValueError("torchaudio gain changed a zero-decibel waveform")
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeValidationError(
                f"TORCHAUDIO_FAILED: {type(exc).__name__}: {exc}", exit_code=24
            ) from exc
        torchaudio_test = "PASS"

    cuda_runtime = torch.version.cuda
    hip_runtime = getattr(torch.version, "hip", None)
    if cuda_runtime is not None:
        backend = "cu" + str(cuda_runtime).replace(".", "")
    elif hip_runtime is not None:
        raise RuntimeValidationError("ROCm is not supported by this tool", exit_code=19)
    else:
        backend = "cpu"

    if expected_backend is not None and backend != expected_backend:
        raise RuntimeValidationError(
            f"expected {expected_backend}, found {backend}", exit_code=30
        )

    cpu_tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    cpu_result = (cpu_tensor @ cpu_tensor).sum().item()
    if cpu_result != 3920.0:
        raise RuntimeValidationError(
            f"CPU tensor calculation returned {cpu_result!r}", exit_code=18
        )

    try:
        numpy_array = torch.tensor([1.0, 2.0, 3.0]).numpy()
        if not isinstance(numpy_array, np.ndarray):
            raise TypeError("Tensor.numpy() did not return numpy.ndarray")
        if numpy_array.tolist() != [1.0, 2.0, 3.0]:
            raise ValueError("Tensor to NumPy values did not round-trip")
        source_array = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        restored_tensor = torch.from_numpy(source_array)
        if restored_tensor.tolist() != [4.0, 5.0, 6.0]:
            raise ValueError("NumPy to Tensor values did not round-trip")
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeValidationError(
            f"NUMPY_BRIDGE_FAILED: {type(exc).__name__}: {exc}", exit_code=22
        ) from exc

    gpu_name = "none"
    cuda_test = "NOT_APPLICABLE"
    if backend.startswith("cu"):
        if not torch.cuda.is_available():
            code = 31 if expected_backend is not None else 20
            raise RuntimeValidationError(
                "CUDA build is installed but CUDA is unavailable", exit_code=code
            )
        try:
            gpu_tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")
            doubled = gpu_tensor * 2
            torch.cuda.synchronize(0)
            if doubled.cpu().tolist() != [2.0, 4.0, 6.0]:
                raise ValueError("CUDA values were incorrect after transfer to CPU")
            if doubled.cpu().numpy().tolist() != [2.0, 4.0, 6.0]:
                raise ValueError("CUDA tensor could not cross the NumPy bridge")
            gpu_name = torch.cuda.get_device_name(0)
            cuda_test = "PASS"
        except (RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeValidationError(
                f"CUDA_RUNTIME_FAILED: {type(exc).__name__}: {exc}", exit_code=21
            ) from exc

    return {
        "schema_version": 1,
        "backend": backend,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision_version,
        "torchaudio_version": torchaudio_version,
        "numpy_version": np.__version__,
        "cuda_runtime": str(cuda_runtime or "none"),
        "gpu_name": gpu_name,
        "cuda_test": cuda_test,
        "numpy_bridge_test": "PASS",
        "torchvision_test": torchvision_test,
        "torchaudio_test": torchaudio_test,
    }


if __name__ == "__main__":
    raise SystemExit(main())
