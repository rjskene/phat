"""
phat-tails: Pareto Hybrids with Asymmetric Tails.

The core distribution classes depend on NumPy and SciPy only and are imported
eagerly. The heavier sub-packages (``phat.bootstrap``, ``phat.tseries``) are
imported on first attribute access so that ``import phat`` never pulls in
Numba or ``arch``.
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
    "PhatFitResult": "phat.fit",
    "fit_phat": "phat.fit",
    "Garchcaster": "phat.tseries",
}

# names of the TensorFlow module (``phat.learn``) that was removed in 1.0
_REMOVED = ("PhatNet", "PhatLoss", "PhatMetric", "DataSplit", "learn")

__all__ = [
    "CarBenHybrid", "CarBenLeft", "CarBenRight", "Phat", "PhatStack",
    *_LAZY,
]


_SUBMODULES = ("bootstrap", "fit", "tseries", "utils")


def __getattr__(name: str):
    if name in _SUBMODULES:
        return import_module(f"phat.{name}")
    if name in _REMOVED:
        raise AttributeError(
            f"phat.{name} was removed in 1.0 together with the TensorFlow module phat.learn. "
            "Use Phat.fit: Phat.fit(values, xi_left=..., xi_right=...) fits the location and "
            "scale by maximum likelihood with the tail shapes held fixed."
        )
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'phat' has no attribute {name!r}")
    return getattr(import_module(module), name)


def __dir__():
    return sorted(set(globals()) | set(_LAZY) | set(_SUBMODULES))
