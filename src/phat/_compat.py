"""
Helpers that keep names from phat-tails 0.0.x working for one release.

Renamed keyword arguments and attributes still work and emit a
``DeprecationWarning`` that points at the caller.
"""
from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

P = ParamSpec('P')
R = TypeVar('R')

_REMOVAL = 'The old name will be removed in a future release.'


def _warn(old: str, new: str, stacklevel: int) -> None:
    message = f'`{old}` is deprecated; use `{new}`. {_REMOVAL}'
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)


def renamed_kwargs(**renames: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator: accept ``old=`` for a keyword argument now called ``new`` (``old='new'``)."""

    def deco(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for old, new in renames.items():
                if old in kwargs:
                    if new in kwargs:
                        raise TypeError(f'{func.__qualname__}() got both `{old}` (deprecated) and `{new}`')
                    _warn(old, new, stacklevel=3)
                    kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)

        return wrapper

    return deco


class renamed_attribute:
    """Descriptor: ``obj.old`` reads and writes ``obj.new`` and warns."""

    def __init__(self, new: str) -> None:
        self.new = new
        self.old = ''

    def __set_name__(self, owner: type, name: str) -> None:
        self.old = name

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        _warn(self.old, self.new, stacklevel=3)
        return getattr(obj, self.new)

    def __set__(self, obj: Any, value: Any) -> None:
        _warn(self.old, self.new, stacklevel=3)
        setattr(obj, self.new, value)


class AliasDict(dict):
    """``dict`` whose old keys still resolve, with a warning, to the renamed ones."""

    def __init__(self, data: dict, aliases: dict[str, str]) -> None:
        super().__init__(data)
        self._aliases = aliases

    def __missing__(self, key: str) -> Any:
        if key in self._aliases:
            _warn(key, self._aliases[key], stacklevel=3)
            return self[self._aliases[key]]
        raise KeyError(key)
