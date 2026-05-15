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
    "show_hints": True,
}

_DEFAULTS: dict[str, Any] = dict(_FACTORY_DEFAULTS)

# Session-scoped flag: True after the kernel-evaluation hint has fired
# once in this Python process. Reset by reset_defaults().
_HINT_FIRED_KERNEL_EVAL: bool = False

# Session-scoped set of (func_name, chosen, routing_reason) triples
# already printed by _maybe_show_dispatch_msg in this Python process.
# Reset by reset_defaults(). See _maybe_show_dispatch_msg for the
# rationale (once-per-unique-decision throttling, parallel to the
# MATLAB internal.maybeShowDispatchMsg helper).
_DISPATCH_MSG_SEEN: set[tuple[str, str, str]] = set()


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
    if name == "show_hints":
        if not isinstance(value, bool):
            raise ValueError(
                f"'show_hints' must be True or False (got {value!r})"
            )
        return value
    raise ValueError(
        f"Unknown default {name!r}. "
        f"Valid names: {', '.join(_FACTORY_DEFAULTS)}"
    )


def get_defaults() -> dict[str, Any]:
    """Return a copy of the current toolbox defaults."""
    return dict(_DEFAULTS)


def show_defaults() -> None:
    """Print the current defaults with brief descriptions.

    Designed for interactive use at the REPL. Programmatic callers
    should use :func:`get_defaults` (which returns a dict) instead.

    Mirrors MATLAB's ``mptDefaults`` (called with no arguments and
    no requested output).
    """
    hints = _DEFAULTS["show_hints"]
    hints_str = "True" if hints is True else ("False" if hints is False else str(hints))
    trunc = _DEFAULTS["truncation_sigmas"]
    trunc_str = "inf" if trunc == math.inf else repr(trunc)
    prec = _DEFAULTS["kernel_precision"]
    prec_str = f"'{prec}'"

    lines = [
        "",
        "Current MPT defaults:",
        "",
        f"  truncation_sigmas: {trunc_str:<12}  Gaussian kernel truncation in sigmas.",
        "                                   inf = exact (default); 6 keeps",
        "                                   ~8 sig figs and is faster.",
        f"  kernel_precision : {prec_str:<12}  Kernel-matrix arithmetic precision.",
        "                                   'double' (default) or 'single'.",
        f"  show_hints       : {hints_str:<12}  Informational console messages from",
        "                                   the toolbox: kernel-eval tip and",
        "                                   dispatch decisions. True or False.",
        "",
        "Usage:",
        "  mpt.set_default(name=value)     set",
        "  prev = mpt.set_default(...)     save previous values",
        "  mpt.set_default(**prev)         restore",
        "  mpt.reset_defaults()            factory defaults",
        "  help(mpt.set_default)           full help",
        "",
    ]
    print("\n".join(lines))


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
    """Reset all defaults to their factory values; return the previous values.

    Also clears two session-scoped flags:
      - the flag that suppresses repeat firings of informational hints
        (so the next eligible call will see the hint again);
      - the set of (function, chosen, routing_reason) triples that have
        already produced a one-time dispatch-decision message (so each
        previously-seen routing decision will print again on its next
        occurrence).
    """
    global _HINT_FIRED_KERNEL_EVAL
    old = dict(_DEFAULTS)
    _DEFAULTS.clear()
    _DEFAULTS.update(_FACTORY_DEFAULTS)
    _HINT_FIRED_KERNEL_EVAL = False
    _DISPATCH_MSG_SEEN.clear()
    return old


# ---------------------------------------------------------------
# Informational hints (one-shot per Python process)
# ---------------------------------------------------------------
_KERNEL_EVAL_HINT_MESSAGE = (
    "mpt tip: kernel-matrix construction is running with default settings\n"
    "(truncation off, double precision). For typical perceptual-modelling\n"
    "workloads at scale, opting in to k=6 truncation and single-precision\n"
    "kernel arithmetic typically gives ~3-10x speedup with ~7 significant\n"
    "figures preserved:\n"
    "\n"
    "    mpt.set_default(truncation_sigmas=6, kernel_precision='single')\n"
    "\n"
    "Affects functions that build a kernel matrix: eval_exp_tens,\n"
    "entropy_exp_tens (Shannon), spectral_entropy, template_harmonicity,\n"
    "virtual_pitches, and cos_sim_exp_tens when routed to Bulger's method.\n"
    "Does not affect Mobius-method paths (cos_sim_exp_tens at default\n"
    "workloads, tensor_harmonicity, entropy_exp_tens with 'renyi2').\n"
    "\n"
    "To silence: mpt.set_default(show_hints=False).\n"
)


def _maybe_show_kernel_eval_hint(
    *,
    effective_truncation_sigmas: float | None = None,
    effective_kernel_precision: str | None = None,
) -> None:
    """Print the kernel-evaluation hint at most once per session.

    Parameters
    ----------
    effective_truncation_sigmas, effective_kernel_precision : optional
        The effective values used for the call (after defaults
        resolution and per-call kwargs). If supplied, the hint fires
        only when *both* effective values match the v2.1 factory
        defaults — i.e., the user hasn't opted in via either route.
        If not supplied, the global defaults are inspected instead.

    Silently no-ops if any of:
      - ``show_hints`` is False
      - the effective values differ from the v2.1 defaults (the user
        has already opted in to the faster regime, so the hint is
        redundant)
      - the hint has already fired this session
    """
    global _HINT_FIRED_KERNEL_EVAL
    if _HINT_FIRED_KERNEL_EVAL:
        return
    if not _DEFAULTS.get("show_hints", True):
        return
    trunc = (effective_truncation_sigmas if effective_truncation_sigmas is not None
             else _DEFAULTS["truncation_sigmas"])
    prec = (effective_kernel_precision if effective_kernel_precision is not None
            else _DEFAULTS["kernel_precision"])
    if trunc != math.inf:
        return
    if str(prec).lower() != "double":
        return
    print(_KERNEL_EVAL_HINT_MESSAGE)
    _HINT_FIRED_KERNEL_EVAL = True


def _format_dispatch_time(t: float) -> str:
    """Short human-readable duration for dispatch messages."""
    if t < 1.0:
        return f"{t * 1000:.0f} ms"
    if t < 60.0:
        return f"{t:.1f} s"
    if t < 3600.0:
        return f"{t / 60:.1f} min"
    return f"{t / 3600:.1f} hr"


def _maybe_show_dispatch_msg(
    func_name: str,
    chosen: str,
    routing_reason: str,
    est_sec: float,
    is_probed: bool,
) -> None:
    """Print a dispatch-decision message at most once per session.

    Prints if and only if the (func_name, chosen, routing_reason)
    triple has not been printed before in this Python process. The
    seen-set is cleared by :func:`reset_defaults`.

    When ``is_probed`` is True, the message includes the empirical
    extrapolated time estimate:

        ``<func_name>: chose '<chosen>' path (estimated X s);
         Ctrl+C to cancel.``

    When ``is_probed`` is False, the message reports the routing
    reason (e.g. a hard rule or analytical pre-screen):

        ``<func_name>: chose '<chosen>' path (<routing_reason>).``

    Gating (v2.2.x): dispatch messages are NOT gated by per-call
    ``verbose``. They are gated by the toolbox-wide ``show_hints``
    flag (``mpt.set_default(show_hints=...)``), matching the
    kernel-evaluation hint's gating model. Rationale: internal
    toolbox callers (e.g. batched-raw paths, entropy evaluations)
    routinely pass ``verbose=False`` to inner calls to prevent
    flooding. With the once-per-session throttle in place, flooding
    is no longer a concern, and users benefit from seeing the routing
    decision even when internal callers pass ``verbose=False``. To
    fully silence dispatch messages:
    ``mpt.set_default(show_hints=False)``.

    Parallels MATLAB ``internal.maybeShowDispatchMsg``; the design
    rationale and the user-visible behaviour are identical.
    """
    # Master switch: show_hints False → silent for all dispatch messages.
    if not _DEFAULTS.get("show_hints", True):
        return

    key = (func_name, chosen, routing_reason)
    if key in _DISPATCH_MSG_SEEN:
        return
    _DISPATCH_MSG_SEEN.add(key)
    if is_probed:
        print(
            f"{func_name}: chose '{chosen}' path "
            f"(estimated {_format_dispatch_time(est_sec)}); "
            f"Ctrl+C to cancel."
        )
    else:
        print(f"{func_name}: chose '{chosen}' path ({routing_reason}).")
