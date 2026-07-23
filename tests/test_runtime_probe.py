import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from uv_torch_compass import runtime_probe


class FakeArray:
    def __init__(self, values) -> None:
        self.values = list(values)

    def tolist(self):
        return self.values


class FakeScalar:
    def item(self) -> float:
        return 3920.0


class FakeTensor:
    def __init__(self, values=None, *, broken_numpy: bool = False) -> None:
        self.values = list(values or [])
        self.broken_numpy = broken_numpy

    def reshape(self, *_shape):
        return self

    def __matmul__(self, _other):
        if self.values and isinstance(self.values[0], list):
            return FakeTensor([[2.0, 2.0], [2.0, 2.0]])
        return self

    def sum(self):
        return FakeScalar()

    def numpy(self):
        if self.broken_numpy:
            raise RuntimeError("NumPy is unavailable")
        return FakeArray(self.values)

    def tolist(self):
        return self.values

    def __mul__(self, multiplier):
        return FakeTensor([value * multiplier for value in self.values])

    def cpu(self):
        return self


def _install_fake_runtime(
    monkeypatch,
    *,
    cuda_runtime=None,
    hip_runtime=None,
    cuda_available=True,
    broken_numpy=False,
) -> None:
    numpy = cast(Any, ModuleType("numpy"))
    numpy.ndarray = FakeArray
    numpy.float32 = "float32"
    numpy.__version__ = "2.2.0"
    numpy.array = lambda values, dtype=None: FakeArray(values)

    torch = cast(Any, ModuleType("torch"))
    torch.__version__ = "2.7.0"
    torch.float32 = "float32"
    torch.version = SimpleNamespace(cuda=cuda_runtime, hip=hip_runtime)
    torch.arange = lambda *_args, **_kwargs: FakeTensor()
    torch.tensor = lambda values, **_kwargs: FakeTensor(
        values, broken_numpy=broken_numpy
    )
    torch.from_numpy = lambda array: FakeTensor(array.values)
    torch.ones = lambda shape, **_kwargs: FakeTensor(
        [[1.0, 1.0], [1.0, 1.0]] if shape == (2, 2) else [1.0]
    )
    torch.backends = SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 9200))
    torch.nn = SimpleNamespace(
        functional=SimpleNamespace(
            conv2d=lambda *_args: FakeTensor([[[[4.0, 4.0], [4.0, 4.0]]]])
        )
    )
    torch.compile = lambda function, **_kwargs: function
    torch.cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        synchronize=lambda _index=0: None,
        get_device_name=lambda _index: "Fake GPU",
        get_device_capability=lambda _index: (8, 9),
        get_arch_list=lambda: ["sm_89", "compute_90"],
    )

    torchvision = cast(Any, ModuleType("torchvision"))
    torchvision.__version__ = "0.22.0"
    torchvision.ops = SimpleNamespace(nms=lambda *_args: FakeTensor([0]))
    torchaudio = cast(Any, ModuleType("torchaudio"))
    torchaudio.__version__ = "2.7.0"
    torchaudio.functional = SimpleNamespace(gain=lambda waveform, _gain: waveform)
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)
    monkeypatch.setattr(
        runtime_probe.importlib.metadata,
        "version",
        lambda _distribution: "12.8.90",
    )


def test_runtime_reports_cpu_and_optional_packages(monkeypatch) -> None:
    _install_fake_runtime(monkeypatch)

    report = runtime_probe._validate_runtime(
        validate_torchvision=True,
        validate_torchaudio=True,
        expected_backend="cpu",
    )

    assert report["schema_version"] == 2
    assert report["backend"] == "cpu"
    assert report["torchvision_test"] == "PASS"
    assert report["torchaudio_test"] == "PASS"


def test_runtime_reports_cuda_execution(monkeypatch) -> None:
    _install_fake_runtime(monkeypatch, cuda_runtime="12.8")

    report = runtime_probe._validate_runtime(
        validate_torchvision=False,
        validate_torchaudio=False,
        expected_backend="cu128",
    )

    assert report["backend"] == "cu128"
    assert report["gpu_name"] == "Fake GPU"
    assert report["cuda_test"] == "PASS"
    assert report["cublas_test"] == "PASS"
    assert report["cudnn_test"] == "PASS"
    assert report["gpu_device_capability"] == "8.9"
    assert report["runtime_component_version"] == "12.8.90"


@pytest.mark.parametrize(
    "runtime_options, expected_code",
    [
        ({"hip_runtime": "6.2"}, 19),
        ({"cuda_runtime": "12.8", "cuda_available": False}, 20),
        ({"broken_numpy": True}, 22),
    ],
)
def test_runtime_rejects_unsupported_or_broken_environments(
    monkeypatch, runtime_options: dict[str, Any], expected_code: int
) -> None:
    _install_fake_runtime(monkeypatch, **runtime_options)

    with pytest.raises(runtime_probe.RuntimeValidationError) as captured:
        runtime_probe._validate_runtime(
            validate_torchvision=False,
            validate_torchaudio=False,
            expected_backend=None,
        )

    assert captured.value.exit_code == expected_code


def test_runtime_main_emits_json_or_stable_error(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        runtime_probe,
        "_validate_runtime",
        lambda **_options: {"schema_version": 2, "backend": "cpu"},
    )
    assert runtime_probe.main(["--expected-backend", "cpu"]) == 0
    assert '"backend": "cpu"' in capsys.readouterr().out

    def fail(**_options):
        raise runtime_probe.RuntimeValidationError("probe failed", exit_code=42)

    monkeypatch.setattr(runtime_probe, "_validate_runtime", fail)
    assert runtime_probe.main([]) == 42
    assert "probe failed" in capsys.readouterr().err


def test_runtime_rejects_backend_and_cpu_calculation_mismatch(monkeypatch) -> None:
    _install_fake_runtime(monkeypatch)
    with pytest.raises(runtime_probe.RuntimeValidationError) as mismatch:
        runtime_probe._validate_runtime(
            validate_torchvision=False,
            expected_backend="cu128",
        )
    assert mismatch.value.exit_code == 30

    torch = cast(Any, sys.modules["torch"])
    torch.arange = lambda *_args, **_kwargs: FakeTensor()
    monkeypatch.setattr(FakeScalar, "item", lambda _self: 1.0)
    with pytest.raises(runtime_probe.RuntimeValidationError) as calculation:
        runtime_probe._validate_runtime(
            validate_torchvision=False,
            expected_backend="cpu",
        )
    assert calculation.value.exit_code == 18


@pytest.mark.parametrize(
    "optional_package, expected_code", [("vision", 23), ("audio", 24)]
)
def test_runtime_rejects_broken_optional_package_ops(
    monkeypatch, optional_package: str, expected_code: int
) -> None:
    _install_fake_runtime(monkeypatch)
    if optional_package == "vision":
        torchvision = cast(Any, sys.modules["torchvision"])
        torchvision.ops.nms = lambda *_args: FakeTensor([1])
    else:
        torchaudio = cast(Any, sys.modules["torchaudio"])
        torchaudio.functional.gain = lambda *_args: FakeTensor([999])

    with pytest.raises(runtime_probe.RuntimeValidationError) as captured:
        runtime_probe._validate_runtime(
            validate_torchvision=optional_package == "vision",
            validate_torchaudio=optional_package == "audio",
            expected_backend="cpu",
        )

    assert captured.value.exit_code == expected_code


def test_runtime_rejects_failed_cuda_tensor_operation(monkeypatch) -> None:
    _install_fake_runtime(monkeypatch, cuda_runtime="12.8")
    torch = cast(Any, sys.modules["torch"])
    original_tensor = torch.tensor

    def tensor(values, **options):
        if options.get("device") == "cuda:0":
            raise RuntimeError("device failed")
        return original_tensor(values, **options)

    torch.tensor = tensor
    with pytest.raises(runtime_probe.RuntimeValidationError) as captured:
        runtime_probe._validate_runtime(
            validate_torchvision=False,
            expected_backend="cu128",
        )

    assert captured.value.exit_code == 21


def test_runtime_compile_profile_executes_compiled_function(monkeypatch) -> None:
    _install_fake_runtime(monkeypatch, cuda_runtime="12.8")

    report = runtime_probe._validate_runtime(
        validate_torchvision=False,
        expected_backend="cu128",
        probe_profile="compile",
    )

    assert report["compile_test"] == "PASS"


def test_runtime_compile_profile_rejects_compiler_failure(monkeypatch) -> None:
    _install_fake_runtime(monkeypatch, cuda_runtime="12.8")
    torch = cast(Any, sys.modules["torch"])

    def fail_compile(*_args, **_kwargs):
        raise RuntimeError("compiler unavailable")

    torch.compile = fail_compile

    with pytest.raises(runtime_probe.RuntimeValidationError) as captured:
        runtime_probe._validate_runtime(
            validate_torchvision=False,
            expected_backend="cu128",
            probe_profile="compile",
        )

    assert captured.value.exit_code == 27


def test_minor_probe_requires_native_architecture(monkeypatch) -> None:
    _install_fake_runtime(monkeypatch, cuda_runtime="12.8")
    torch = cast(Any, sys.modules["torch"])
    torch.cuda.get_arch_list = lambda: ["compute_89"]

    with pytest.raises(runtime_probe.RuntimeValidationError) as captured:
        runtime_probe._validate_runtime(
            validate_torchvision=False,
            expected_backend="cu128",
            require_native_architecture=True,
        )

    assert captured.value.exit_code == 28
