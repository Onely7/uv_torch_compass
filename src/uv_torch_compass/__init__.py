"""Configure a verified PyTorch backend for uv projects."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("uv-torch-compass")
except PackageNotFoundError:  # pragma: no cover - source trees without install metadata
    __version__ = "0.1.1"

__all__ = ["__version__"]
