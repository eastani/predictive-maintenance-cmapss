"""Predictive maintenance reference pipeline (CMAPSS)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pdm")
except PackageNotFoundError:  # local checkout, package not installed
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
