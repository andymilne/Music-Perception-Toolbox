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
    {'truncation_sigmas': 6.0, 'kernel_precision': 'double'}

    >>> mpt.set_default(truncation_sigmas=math.inf)   # exact (untruncated)
    >>> mpt.get_default('truncation_sigmas')
    inf

    >>> mpt.set_default(truncation_sigmas=math.inf, kernel_precision='single')

    >>> mpt.reset_defaults()
"""

from __future__ import annotations

import contextlib
import math
import threading
import warnings
from typing import Any

# Module-level mutable state. Hidden behind the accessor functions.
_FACTORY_DEFAULTS: dict[str, Any] = {
    "truncation_sigmas": 6.0,
    "kernel_precision": "double",
    "show_hints": True,
    "kernel_chunk_bytes": "auto",
}

_DEFAULTS: dict[str, Any] = dict(_FACTORY_DEFAULTS)

# One-time-per-process flag: True once the truncation-default notice has
# been printed (or suppressed). Not cleared by reset_defaults; a fresh
# interpreter (module re-import) re-arms it, so the notice reappears once
# per session. The test suite calls _suppress_truncation_notice() so it
# never prints during tests.
_TRUNCATION_NOTICE_SHOWN: bool = False

# Set of (func_name, chosen) pairs already printed by
# _maybe_show_dispatch_msg in the current top-level toolbox call. Cleared
# at the start of every top-level call by :func:`_dispatch_scope`, so each
# top-level user call sees each unique (func, chosen) decision once. Also cleared
# by :func:`reset_defaults`. See :func:`_maybe_show_dispatch_msg` for the
# rationale (per-top-level-call throttling, parallel to the MATLAB
# ``internal.maybeShowDispatchMsg`` + ``internal.dispatchScope`` pair).
_DISPATCH_MSG_SEEN: set[tuple[str, str]] = set()

# Thread-local depth counter for the dispatch-scope context manager.
# Depth 0 outside any toolbox call; depth 1 on the outermost entry to a
# public toolbox function; depth >1 for nested toolbox calls within that
# top-level call. The seen-set is cleared only on the 0→1 transition.
_dispatch_scope_state = threading.local()


def _get_dispatch_depth() -> int:
    """Return the current dispatch-scope depth (0 if outside any scope)."""
    return getattr(_dispatch_scope_state, "depth", 0)


@contextlib.contextmanager
def _dispatch_scope():
    """Mark a top-level toolbox entry; reset the dispatch seen-set on entry.

    Depth-counted re-entrant context manager. The seen-set is cleared
    only on the outermost entry (depth 0 → 1), so nested toolbox calls
    within one top-level user call share the same seen-set and won't
    re-announce decisions already announced earlier in that call.

    Each top-level user call (REPL invocation, script-level call) sees
    each unique (func, chosen) dispatch decision exactly once. Two
    different routing reasons that lead to the same chosen path within
    one top-level call collapse to a single announce — the visible
    distinction that matters is which path ran, not why. Repeat
    top-level calls re-announce.

    Parallels MATLAB ``internal.dispatchScope`` (``acquire`` / ``release``
    with a depth-tracked persistent state).
    """
    depth = getattr(_dispatch_scope_state, "depth", 0)
    if depth == 0:
        _DISPATCH_MSG_SEEN.clear()
    _dispatch_scope_state.depth = depth + 1
    try:
        yield
    finally:
        cur = getattr(_dispatch_scope_state, "depth", 1)
        _dispatch_scope_state.depth = max(0, cur - 1)


def _with_dispatch_scope(fn):
    """Decorator: wrap ``fn``'s body in :func:`_dispatch_scope`.

    Applied to every public toolbox entry point whose dispatch decisions
    (or whose inner calls' dispatch decisions) should be announced
    per-top-level-call. Uses :func:`functools.wraps` so the decorated
    function preserves ``__doc__``, ``__name__``, ``__module__``, etc.
    """
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _dispatch_scope():
            return fn(*args, **kwargs)

    return wrapper


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
    if name == "kernel_chunk_bytes":
        if isinstance(value, str):
            v = value.lower()
            if v != "auto":
                raise ValueError(
                    f"'kernel_chunk_bytes' string value must be 'auto' "
                    f"(got {value!r})"
                )
            return v
        try:
            iv = int(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"'kernel_chunk_bytes' must be 'auto' or a positive integer "
                f"(got {value!r})"
            )
        if iv <= 0:
            raise ValueError(
                f"'kernel_chunk_bytes' must be 'auto' or a positive integer "
                f"(got {value!r})"
            )
        return iv
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
        f"  truncation_sigmas: {trunc_str:<12}  Gaussian kernel truncation radius, in sigmas.",
        "                                   inf = exact; larger is more accurate, slower.",
        "                                   Worst-case error vs exact: 4 -> ~1e-3,",
        "                                   5 -> ~1e-5, 6 (default) -> ~2e-8.",
        f"  kernel_precision : {prec_str:<12}  Kernel-matrix arithmetic precision.",
        "                                   'double' (default) or 'single'.",
        f"  show_hints       : {hints_str:<12}  Informational console messages from",
        "                                   the toolbox: dispatch decisions.",
        "                                   True or False.",
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
    if key == "truncation_sigmas":
        _maybe_show_truncation_notice()
    return _DEFAULTS[key]


class TruncationDefaultWarning(UserWarning):
    """Category for the one-time notice that kernel truncation is on by default.

    Emitted once per process by :func:`_maybe_show_truncation_notice`.
    Suppress it with, e.g.::

        import warnings, mpt
        warnings.filterwarnings("ignore", category=mpt.TruncationDefaultWarning)

    (the MATLAB counterpart is ``warning('off', 'mpt:truncationDefault')``).
    """


_TRUNCATION_NOTICE_MESSAGE = (
    "Kernel evaluation truncates the Gaussian kernel at 6 sigma by "
    "default, which runs much faster than the exact sum; the speed-up "
    "grows with tuple size r and multiset size, where the exact sum has "
    "many kernel centres and becomes expensive. Worst-case error vs the "
    "exact result is about 2e-8 at 6 sigma (the default), ~1e-5 at 5, "
    "and ~1e-3 at 4. Set "
    "mpt.set_default(truncation_sigmas=float('inf')) for the exact "
    "result. This warning shows only once per session."
)


def _maybe_show_truncation_notice() -> None:
    """Warn, at most once per process, that kernel truncation is on by default.

    Called from :func:`get_default` whenever the ``truncation_sigmas``
    default is resolved (i.e. a kernel evaluation runs without an
    explicit value), so it fires on first use regardless of which public
    function the script calls. Emitted once, when it has not already
    fired this process and the truncation default is still at its factory
    value of 6. Stays silent once the user sets any other value, and
    throughout the test suite (which pins ``inf`` and calls
    :func:`_suppress_truncation_notice`).

    It is issued as a :class:`TruncationDefaultWarning` (goes to stderr,
    suppressible via the warnings machinery) rather than printed to
    stdout, so it never lands in the middle of a script's own output.
    It is deliberately not gated by ``show_hints``; ``show_hints``
    governs only the dispatch-decision messages.
    """
    global _TRUNCATION_NOTICE_SHOWN
    if _TRUNCATION_NOTICE_SHOWN:
        return
    if _DEFAULTS.get("truncation_sigmas") != 6:
        return
    _TRUNCATION_NOTICE_SHOWN = True
    warnings.warn(_TRUNCATION_NOTICE_MESSAGE, TruncationDefaultWarning, stacklevel=2)


def _suppress_truncation_notice() -> None:
    """Mark the truncation notice as shown without printing it.

    Used by the test suite so the notice never appears during tests.
    """
    global _TRUNCATION_NOTICE_SHOWN
    _TRUNCATION_NOTICE_SHOWN = True


def _rearm_truncation_notice() -> None:
    """Clear the shown flag so the notice can fire again.

    Used at test-session teardown so running the suite in a long-lived
    interpreter does not permanently silence the notice.
    """
    global _TRUNCATION_NOTICE_SHOWN
    _TRUNCATION_NOTICE_SHOWN = False


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

    Also clears the set of (function, chosen, routing_reason) triples
    seen by :func:`_maybe_show_dispatch_msg` in the current top-level
    call (so each previously-seen routing decision will print again on
    its next occurrence). Note: under normal use this set is also
    cleared automatically at the start of each top-level toolbox call
    by :func:`_dispatch_scope`.
    """
    old = dict(_DEFAULTS)
    _DEFAULTS.clear()
    _DEFAULTS.update(_FACTORY_DEFAULTS)
    _DISPATCH_MSG_SEEN.clear()
    # Flush the kernel_chunk_bytes 'auto' resolution cache so a
    # subsequent call re-queries the OS.
    from ._utils import flush_kernel_chunk_bytes_cache
    flush_kernel_chunk_bytes_cache()
    return old


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
    """Print a dispatch-decision message at most once per top-level call.

    Prints if and only if the (func_name, chosen, routing_reason)
    triple has not been printed before in the current top-level
    toolbox call. The seen-set is cleared on every top-level entry
    by :func:`_dispatch_scope`, and also explicitly by
    :func:`reset_defaults`.

    When ``is_probed`` is True, the message includes the empirical
    extrapolated time estimate:

        ``<func_name>: chose '<chosen>' path (estimated X s);
         Ctrl+C to cancel.``

    When ``is_probed`` is False, the message reports only the path:

        ``<func_name>: chose '<chosen>' path.``

    The routing reason (e.g. ``"r1_hard_rule"``, ``"user_method"``) is
    retained in the seen-set key so that different reasons for the same
    chosen path each get one announce, but is not printed in the message
    text itself --- the user-facing distinction that matters is which
    path ran, not why.

    Gating: dispatch messages are NOT gated by per-call
    ``verbose``. They are gated by the toolbox-wide ``show_hints``
    flag (``mpt.set_default(show_hints=...)``). Rationale: internal
    toolbox callers (e.g. batched-raw paths, entropy evaluations)
    routinely pass ``verbose=False`` to inner calls to prevent
    flooding. With the per-top-level-call throttle in place, flooding
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

    key = (func_name, chosen)
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
        print(f"{func_name}: chose '{chosen}' path.")
