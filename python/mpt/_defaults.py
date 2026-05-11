"""Toolbox-wide default options.

Centralises the few user-tunable toolbox defaults (currently:
``truncation_sigmas``, ``kernel_precision``). The defaults are consulted by
:func:`mpt._kernel.gaussian_kernel_sum` and (via that helper) by every
centres-path consumer in the toolbox. Per-call keyword arguments
always override the defaults set here.

Defaults persist within the Python process but not across processes.

Usage::

    >>> import mpt
    >>> mpt.get_defaults()
    {'truncation_sigmas': inf, 'kernel_precision': 'double'}

    >>> mpt.set_default(truncation_sigmas=6)
    >>> mpt.get_default('truncation_sigmas')
    6.0

    >>> mpt.set_default(truncation_sigmas=6, kernel_precision='single')

    >>> mpt.reset_defaults()
"""

from __future__ import annotations

import math
from typing import Any

# Module-level mutable state. Hidden behind the accessor functions.
_FACTORY_DEFAULTS: dict[str, Any] = {
    "truncation_sigmas": math.inf,
    "kernel_precision": "double",
}

_DEFAULTS: dict[str, Any] = dict(_FACTORY_DEFAULTS)


def _validate_one(name: str, value: Any) -> Any:
    """Validate a single (name, value) pair; return normalised value."""
    name = name.lower()
    if name == "truncation_sigmas":
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"'truncation_sigmas' must be numeric (got {value!r})"
            )
        if not (v > 0):
            raise ValueError(
                f"'truncation_sigmas' must be positive (math.inf to disable); "
                f"got {value!r}"
            )
        return v
    if name == "kernel_precision":
        if not isinstance(value, str):
            raise ValueError(
                f"'kernel_precision' must be 'double' or 'single' (got {value!r})"
            )
        v = value.lower()
        if v not in ("double", "single"):
            raise ValueError(
                f"'kernel_precision' must be 'double' or 'single' (got {value!r})"
            )
        return v
    raise ValueError(
        f"Unknown default {name!r}. "
        f"Valid names: {', '.join(_FACTORY_DEFAULTS)}"
    )


def get_defaults() -> dict[str, Any]:
    """Return a copy of the current toolbox defaults."""
    return dict(_DEFAULTS)


def get_default(name: str) -> Any:
    """Return the current value of a single default.

    Parameters
    ----------
    name : str
        Name of the default (e.g. ``'truncation_sigmas'``).

    Returns
    -------
    The current value.

    Raises
    ------
    KeyError
        If ``name`` is not a recognised default.
    """
    key = name.lower()
    if key not in _DEFAULTS:
        raise KeyError(
            f"Unknown default {name!r}. "
            f"Valid names: {', '.join(_FACTORY_DEFAULTS)}"
        )
    return _DEFAULTS[key]


def set_default(**kwargs: Any) -> dict[str, Any]:
    """Set one or more toolbox defaults.

    Returns the previous values (as a dict) so callers can restore
    them later via ``set_default(**prev)``.

    Examples
    --------
    >>> prev = set_default(truncation_sigmas=6, kernel_precision='single')
    >>> # ... do work ...
    >>> set_default(**prev)   # restore
    """
    if not kwargs:
        return get_defaults()
    old: dict[str, Any] = {}
    new: dict[str, Any] = {}
    for name, value in kwargs.items():
        key = name.lower()
        if key not in _FACTORY_DEFAULTS:
            raise ValueError(
                f"Unknown default {name!r}. "
                f"Valid names: {', '.join(_FACTORY_DEFAULTS)}"
            )
        new[key] = _validate_one(key, value)
        old[key] = _DEFAULTS[key]
    _DEFAULTS.update(new)
    return old


def reset_defaults() -> dict[str, Any]:
    """Reset all defaults to their factory values; return the previous values."""
    old = dict(_DEFAULTS)
    _DEFAULTS.clear()
    _DEFAULTS.update(_FACTORY_DEFAULTS)
    return old
