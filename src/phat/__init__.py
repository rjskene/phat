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
    "PhatFitResult": "phat.fit",
    "fit_phat": "phat.fit",
    "PhatNet": "phat.learn",
    "PhatLoss": "phat.learn",
    "PhatMetric": "phat.learn",
    "DataSplit": "phat.learn",
    "Garchcaster": "phat.tseries",
}

# extra that provides the optional dependencies of each lazily imported module
_EXTRAS = {"phat.learn": "learn", "phat.tseries": "garch"}

__all__ = [
    "CarBenHybrid", "CarBenLeft", "CarBenRight", "Phat", "PhatStack",
    *_LAZY,
]


_SUBMODULES = ("bootstrap", "fit", "learn", "tseries", "utils")


def __getattr__(name: str):
    if name in _SUBMODULES:
        return import_module(f"phat.{name}")
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'phat' has no attribute {name!r}")
    try:
        return getattr(import_module(module), name)
    except ModuleNotFoundError as exc:
        extra = _EXTRAS.get(module)
        if extra is None or (exc.name or '').split('.')[0] == 'phat':
            raise
        raise ImportError(
            f"phat.{name} needs {exc.name!r}, which is not installed: "
            f'pip install "phat-tails[{extra}]"'
        ) from exc


def __dir__():
    return sorted(set(globals()) | set(_LAZY) | set(_SUBMODULES))
