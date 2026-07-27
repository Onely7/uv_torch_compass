"""Validate PyTorch inside an isolated or synchronized target environment.

The probe is ordinary source because a different Python interpreter executes it.
Keeping the boundary visible lets static analysis inspect every runtime check.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from collections.abc import Sequence
from typing import Any


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
    parser.add_argument(
        "--probe-profile", choices=("standard", "compile"), default="standard"
    )
    parser.add_argument("--require-native-architecture", action="store_true")
    options = parser.parse_args(arguments)
    try:
        report = _validate_runtime(
            validate_torchvision=options.validate_torchvision,
            validate_torchaudio=options.validate_torchaudio,
            expected_backend=options.expected_backend,
            probe_profile=options.probe_profile,
            require_native_architecture=options.require_native_architecture,
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
    probe_profile: str = "standard",
    require_native_architecture: bool = False,
) -> dict[str, Any]:
    # These imports intentionally resolve in the target environment rather than
    # the environment that hosts uv-torch-compass.
    import numpy as np  # ty: ignore[unresolved-import]
    import torch  # ty: ignore[unresolved-import]

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

    _validate_cpu_and_numpy(torch, np)
    device = "cuda:0" if backend.startswith("cu") else "cpu"
    cuda_results = _validate_cuda(
        torch,
        backend=backend,
        require_native_architecture=require_native_architecture,
    )
    torchvision_version, torchvision_test = _validate_torchvision(
        torch,
        validate=validate_torchvision or _is_distribution_installed("torchvision"),
        device=device,
    )
    torchaudio_version, torchaudio_test = _validate_torchaudio(
        torch,
        validate=validate_torchaudio or _is_distribution_installed("torchaudio"),
        device=device,
    )
    compile_test = _validate_compile(torch, profile=probe_profile, device=device)

    return {
        "schema_version": 2,
        "backend": backend,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision_version,
        "torchaudio_version": torchaudio_version,
        "numpy_version": np.__version__,
        "cuda_runtime": str(cuda_runtime or "none"),
        "runtime_component_version": _runtime_component_version(cuda_runtime),
        "gpu_name": cuda_results["gpu_name"],
        "gpu_device_capability": cuda_results["gpu_device_capability"],
        "compiled_architectures": cuda_results["compiled_architectures"],
        "native_architecture_test": cuda_results["native_architecture_test"],
        "cuda_test": cuda_results["cuda_test"],
        "cublas_test": cuda_results["cublas_test"],
        "cudnn_test": cuda_results["cudnn_test"],
        "numpy_bridge_test": "PASS",
        "torchvision_test": torchvision_test,
        "torchaudio_test": torchaudio_test,
        "compile_test": compile_test,
        "probe_profile": probe_profile,
    }


def _is_distribution_installed(package: str) -> bool:
    try:
        importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _validate_cpu_and_numpy(torch: Any, np: Any) -> None:
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


def _validate_cuda(
    torch: Any,
    *,
    backend: str,
    require_native_architecture: bool,
) -> dict[str, str | list[str]]:
    if not backend.startswith("cu"):
        return {
            "gpu_name": "none",
            "gpu_device_capability": "none",
            "compiled_architectures": [],
            "native_architecture_test": "NOT_APPLICABLE",
            "cuda_test": "NOT_APPLICABLE",
            "cublas_test": "NOT_APPLICABLE",
            "cudnn_test": "NOT_APPLICABLE",
        }
    if not torch.cuda.is_available():
        raise RuntimeValidationError(
            "CUDA build is installed but CUDA is unavailable", exit_code=20
        )

    try:
        gpu_tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")
        doubled = gpu_tensor * 2
        torch.cuda.synchronize(0)
        if doubled.cpu().tolist() != [2.0, 4.0, 6.0]:
            raise ValueError("CUDA values were incorrect after transfer to CPU")
        if doubled.cpu().numpy().tolist() != [2.0, 4.0, 6.0]:
            raise ValueError("CUDA tensor could not cross the NumPy bridge")
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeValidationError(
            f"CUDA_RUNTIME_FAILED: {type(exc).__name__}: {exc}", exit_code=21
        ) from exc

    try:
        left = torch.ones((2, 2), device="cuda:0")
        product = left @ left
        torch.cuda.synchronize(0)
        if product.cpu().tolist() != [[2.0, 2.0], [2.0, 2.0]]:
            raise ValueError("cuBLAS matrix product returned unexpected values")
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeValidationError(
            f"CUBLAS_FAILED: {type(exc).__name__}: {exc}", exit_code=25
        ) from exc

    try:
        cudnn_version = torch.backends.cudnn.version()
        if not isinstance(cudnn_version, int) or cudnn_version <= 0:
            raise ValueError("cuDNN version is unavailable")
        image = torch.ones((1, 1, 3, 3), device="cuda:0")
        kernel = torch.ones((1, 1, 2, 2), device="cuda:0")
        convolution = torch.nn.functional.conv2d(image, kernel)
        torch.cuda.synchronize(0)
        expected = [[[[4.0, 4.0], [4.0, 4.0]]]]
        if convolution.cpu().tolist() != expected:
            raise ValueError("cuDNN convolution returned unexpected values")
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeValidationError(
            f"CUDNN_FAILED: {type(exc).__name__}: {exc}", exit_code=26
        ) from exc

    capability = torch.cuda.get_device_capability(0)
    if (
        not isinstance(capability, tuple)
        or len(capability) != 2
        or not all(isinstance(part, int) for part in capability)
    ):
        raise RuntimeValidationError(
            "CUDA_ARCHITECTURE_FAILED: invalid device capability", exit_code=28
        )
    architectures = list(torch.cuda.get_arch_list())
    native_architecture = f"sm_{capability[0]}{capability[1]}"
    native_test = "PASS" if native_architecture in architectures else "PTX_ONLY"
    if require_native_architecture and native_test != "PASS":
        raise RuntimeValidationError(
            f"CUDA_ARCHITECTURE_FAILED: {native_architecture} is not native "
            "in this PyTorch build",
            exit_code=28,
        )

    return {
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "gpu_device_capability": f"{capability[0]}.{capability[1]}",
        "compiled_architectures": architectures,
        "native_architecture_test": native_test,
        "cuda_test": "PASS",
        "cublas_test": "PASS",
        "cudnn_test": "PASS",
    }


def _validate_torchvision(
    torch: Any, *, validate: bool, device: str
) -> tuple[str, str]:
    if not validate:
        return "not-installed", "NOT_REQUESTED"
    import torchvision  # ty: ignore[unresolved-import]

    try:
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device)
        scores = torch.tensor([1.0], device=device)
        selected = torchvision.ops.nms(boxes, scores, 0.5)
        if selected.cpu().tolist() != [0]:
            raise ValueError("torchvision.ops.nms returned unexpected indexes")
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeValidationError(
            f"TORCHVISION_FAILED: {type(exc).__name__}: {exc}", exit_code=23
        ) from exc
    return str(torchvision.__version__), "PASS"


def _validate_torchaudio(torch: Any, *, validate: bool, device: str) -> tuple[str, str]:
    if not validate:
        return "not-installed", "NOT_REQUESTED"
    import torchaudio  # ty: ignore[unresolved-import]

    try:
        waveform = torch.tensor([[0.25, -0.25]], device=device)
        gained = torchaudio.functional.gain(waveform, 0.0)
        if gained.cpu().tolist() != waveform.cpu().tolist():
            raise ValueError("torchaudio gain changed a zero-decibel waveform")
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeValidationError(
            f"TORCHAUDIO_FAILED: {type(exc).__name__}: {exc}", exit_code=24
        ) from exc
    return str(torchaudio.__version__), "PASS"


def _validate_compile(torch: Any, *, profile: str, device: str) -> str:
    if profile != "compile":
        return "NOT_REQUESTED"
    try:
        compiled = torch.compile(lambda value: value * 2, fullgraph=True)
        result = compiled(torch.tensor([1.0, 2.0], device=device))
        if device == "cuda:0":
            torch.cuda.synchronize(0)
        if result.cpu().tolist() != [2.0, 4.0]:
            raise ValueError("compiled function returned unexpected values")
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeValidationError(
            f"COMPILE_FAILED: {type(exc).__name__}: {exc}", exit_code=27
        ) from exc
    return "PASS"


def _runtime_component_version(cuda_runtime: object) -> str:
    if cuda_runtime is None:
        return "not-installed"
    major = str(cuda_runtime).split(".", 1)[0]
    distributions = (
        f"nvidia-cuda-runtime-cu{major}",
        "nvidia-cuda-runtime",
        "cuda-runtime",
    )
    for distribution in distributions:
        if version := _installed_distribution_version(distribution):
            return version
    return "not-installed"


def _installed_distribution_version(distribution: str) -> str | None:
    """Return an installed distribution version when metadata is available."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
