"""
phat-tails: Pareto Hybrids with Asymmetric Tails.

The core distribution classes depend on NumPy and SciPy only and are imported
eagerly. The heavier sub-packages (``phat.bootstrap``, ``phat.learn``,
``phat.tseries``) are imported on first attribute access so that
``import phat`` never pulls in Numba, TensorFlow or ``arch``.
"""
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

from phat.dists import CarBenHybrid, CarBenLeft, CarBenRight, Phat, PhatStack

try:
    __version__ = version("phat-tails")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0+unknown"

_LAZY = {
    "two_tailed_hill_double_bootstrap": "phat.bootstrap",
    "PhatFit": "phat.fit",
    "PhatNet": "phat.learn",
    "PhatLoss": "phat.learn",
    "PhatMetric": "phat.learn",
    "DataSplit": "phat.learn",
    "Garchcaster": "phat.tseries",
}

__all__ = [
    "CarBenHybrid", "CarBenLeft", "CarBenRight", "Phat", "PhatStack",
    *_LAZY,
]


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'phat' has no attribute {name!r}")
    return getattr(import_module(module), name)


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
