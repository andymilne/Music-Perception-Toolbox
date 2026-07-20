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


#: The toolbox's tightest meaningful accuracy floor. ``truncation_sigmas
#: = inf`` is understood as "as accurate as the toolbox targets", namely
#: relative contributions below this are negligible --- NOT literally
#: exhaustive summation into the denormal far tail (where the Möbius
#: decomposition's cancellation error dominates a true value already far
#: below the floor). This single floor is shared by every truncation
#: path: the kernel sums, the single- and multi-attribute point
#: evaluators, and the relative factored read-back
#: (:func:`mpt._mobius._factored_target_eps`). Read it through
#: :func:`accuracy_floor_eps` (which honours a temporary override), not
#: directly, so all paths see the same value.
_ACCURACY_FLOOR_EPS: float = 1e-12

#: Thread-local override for the accuracy floor, set only within the
#: :func:`accuracy_floor_context` context manager. ``None`` means "use
#: the module default :data:`_ACCURACY_FLOOR_EPS`".
_accuracy_floor_state = threading.local()


def accuracy_floor_eps() -> float:
    """Current accuracy-floor epsilon (honouring any active override).

    Returns the temporary value set by :func:`accuracy_floor_context`
    when one is active on this thread, otherwise the module default
    :data:`_ACCURACY_FLOOR_EPS` (1e-12). This is the single read point
    for the floor; every truncation path resolves ``inf`` against it.
    """
    return getattr(_accuracy_floor_state, "eps", None) or _ACCURACY_FLOOR_EPS


def accuracy_floor_sigmas() -> float:
    """Finite truncation width achieving the current accuracy floor.

    A density kernel ``G(d; sigma)`` falls below ``eps`` at
    ``|d| > k sigma`` with ``k = sqrt(-2 ln eps)``; this returns that
    ``k`` for the currently active :func:`accuracy_floor_eps`. It is the
    finite width to which ``truncation_sigmas = inf`` resolves.
    """
    return math.sqrt(-2.0 * math.log(accuracy_floor_eps()))


# --- Truncation scalar identity -------------------------------------
# Truncation at k sigmas discards Gaussian contributions whose value
# has fallen below a single floor, exp(-k^2/2), stated on the
# normalised value scale and therefore identical for every Gaussian
# kernel regardless of width. The three helpers below are the one
# place this floor and its two distance manifestations are written;
# every truncation path derives its cutoff through them so the
# density-vs-inner-product width distinction cannot drift between
# call sites.
#
# The two kernels the toolbox truncates have different widths:
#   - the *density* kernel  G(d; sigma)   = exp(-|d|^2 / (2 sigma^2)),
#   - the *inner-product* kernel G(d; sigma sqrt 2)
#                                = exp(-|d|^2 / (4 sigma^2)),
# the latter arising because an inner product convolves two density
# kernels. At the shared value floor exp(-k^2/2) the density kernel
# sits at distance |d| = k sigma, whereas the (sqrt 2 wider)
# inner-product kernel sits at |d| = sqrt 2 * k sigma, i.e.
# |d|^2 = 2 (k sigma)^2 --- the factor 2 that distinguishes the two.
#
# Each helper resolves its argument through resolve_truncation_sigmas,
# so the truncation policy is enforced in one place: None takes the
# global default and math.inf (the user-facing "exact" sentinel)
# resolves to the finite accuracy-floor width (default ~7.43 sigma, the
# 1e-12 floor). There is therefore no "nothing discarded" state:
# truncation always applies at least at the accuracy floor. Genuinely
# exhaustive summation is reachable only internally, by widening that
# floor via accuracy_floor_context, never by a user value.


def truncation_floor(truncation_sigmas: Any = None) -> float:
    """Kernel-value floor ``exp(-k^2/2)`` at the resolved truncation width.

    The largest normalised Gaussian value truncation discards --- the
    same for the density and inner-product kernels, since it is stated
    on the value scale. The argument is resolved through
    :func:`resolve_truncation_sigmas`, so ``None`` takes the default
    and ``math.inf`` takes the finite accuracy-floor width (giving the
    1e-12 floor). Always returns a finite positive value.
    """
    k = resolve_truncation_sigmas(truncation_sigmas)
    # resolve_truncation_sigmas maps the "exact" sentinel to
    # accuracy_floor_sigmas(), whose floor is by construction exactly
    # accuracy_floor_eps(); return that directly rather than through a
    # log/exp round-trip (which perturbs it in the last ULP).
    if k == accuracy_floor_sigmas():
        return accuracy_floor_eps()
    return math.exp(-0.5 * k * k)


def truncation_radius(truncation_sigmas: Any, sigma: float) -> float:
    """Distance ``k * sigma`` at which the density kernel reaches the floor.

    Beyond ``|d| = k sigma`` the density kernel ``G(d; sigma)`` sits
    below :func:`truncation_floor`; this is the radius the centres
    evaluator uses to size its spatial index. The width is resolved as
    in :func:`truncation_floor`.
    """
    k = resolve_truncation_sigmas(truncation_sigmas)
    return k * sigma


def truncation_ip_sqdist(truncation_sigmas: Any, sigma: float) -> float:
    """Squared distance ``2 (k sigma)^2`` at which the IP kernel reaches the floor.

    Beyond ``|d|^2 = 2 (k sigma)^2`` the inner-product kernel
    ``G(d; sigma sqrt 2)`` --- value ``exp(-|d|^2 / (4 sigma^2))`` ---
    sits below :func:`truncation_floor`. The factor 2 relative to
    :func:`truncation_radius` squared (``(k sigma)^2``) is exactly the
    ``sqrt 2`` extra width of the inner-product kernel. The width is
    resolved as in :func:`truncation_floor`.
    """
    k = resolve_truncation_sigmas(truncation_sigmas)
    return 2.0 * (k * sigma) ** 2


@contextlib.contextmanager
def accuracy_floor_context(eps: float):
    """Temporarily override the accuracy floor on the current thread.

    **Internal / test-only.** This is deliberately *not* a standard
    user-facing default (it is absent from :func:`set_default`'s
    validated keys and is not exported at package level), because it
    changes the meaning of the "exact" (``inf``) sentinel across the
    whole toolbox and is intended for controlled situations --- chiefly
    regenerating golden reference values at maximal accuracy.

    Within the ``with`` block, ``truncation_sigmas = inf`` resolves to
    the width ``sqrt(-2 ln eps)`` instead of the default 1e-12 floor's
    ~7.43 sigma, and the relative factored read-back targets ``eps``.
    Set ``eps`` small (e.g. ``1e-300``) for effectively exhaustive,
    maximal-accuracy evaluation:

        >>> from mpt._defaults import accuracy_floor_context
        >>> with accuracy_floor_context(1e-300):
        ...     golden = eval_exp_tens(dens, x, truncation_sigmas=math.inf)

    The override is thread-local and scoped: it never leaks past the
    ``with`` block, even on exception, and does not affect other
    threads. Nesting restores the enclosing value on exit.

    Parameters
    ----------
    eps : float
        Temporary accuracy floor in ``(0, 1)``. Smaller means more
        accurate (wider effective truncation). ``eps -> 0`` approaches
        exhaustive double-precision summation.
    """
    if not (0.0 < eps < 1.0):
        raise ValueError(
            f"accuracy-floor eps must be in (0, 1); got {eps!r}."
        )
    prev = getattr(_accuracy_floor_state, "eps", None)
    _accuracy_floor_state.eps = float(eps)
    try:
        yield
    finally:
        _accuracy_floor_state.eps = prev


def resolve_truncation_sigmas(truncation_sigmas: Any = None) -> float:
    """Resolve the truncation knob to an effective finite width.

    ``None`` resolves against the current default. A non-finite value
    (``math.inf``, the documented "exact" sentinel) resolves to
    :func:`accuracy_floor_sigmas`, the finite width at which a density
    kernel falls below the toolbox accuracy floor
    (:func:`accuracy_floor_eps`, default 1e-12). This makes ``inf`` mean
    "accuracy-floor accuracy" uniformly across the absolute and
    relative, single- and multi-attribute paths, rather than literally
    exhaustive summation into the denormal far tail. Finite positive
    values pass through unchanged.

    The floor is overridable for special cases (chiefly golden-value
    regeneration) via :func:`accuracy_floor_context`; within such a
    context ``inf`` resolves to that context's (typically wider) width.
    """
    if truncation_sigmas is None:
        truncation_sigmas = _DEFAULTS["truncation_sigmas"]
    k = float(truncation_sigmas)
    if not math.isfinite(k):
        return accuracy_floor_sigmas()
    return k

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

# Once-per-top-level-call latch for the up-front evaluation time warning.
# Reset on the same 0->1 scope transition as the dispatch seen-set, so a
# single user call emits at most one time warning (from the first
# evaluation whose estimate crosses the threshold, direct or wrapped).
_TIME_WARN_EMITTED: list[bool] = [False]

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
        _TIME_WARN_EMITTED[0] = False
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
    "default, which runs much faster than a wider cutoff; the speed-up "
    "grows with tuple size r and multiset size, where a wider cutoff has "
    "many kernel centres and becomes expensive. Worst-case error vs an "
    "essentially exhaustive sum is about 2e-8 at 6 sigma (the default), "
    "~1e-5 at 5, and ~1e-3 at 4. Set "
    "mpt.set_default(truncation_sigmas=float('inf')) for the accuracy "
    "floor (~7.43 sigma, ~1e-12 error). This warning shows only once "
    "per session."
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
    _TIME_WARN_EMITTED[0] = False
    # Flush the kernel_chunk_bytes 'auto' resolution cache so a
    # subsequent call re-queries the OS.
    from ._utils import flush_kernel_chunk_bytes_cache
    flush_kernel_chunk_bytes_cache()
    return old


def _maybe_show_dispatch_msg(
    func_name: str,
    chosen: str,
    routing_reason: str,
) -> None:
    """Print a dispatch-decision message at most once per top-level call.

    Prints if and only if the (func_name, chosen, routing_reason)
    triple has not been printed before in the current top-level
    toolbox call. The seen-set is cleared on every top-level entry
    by :func:`_dispatch_scope`, and also explicitly by
    :func:`reset_defaults`.

    The message reports the routing decision only:

        ``<func_name>: chose '<chosen>' path.``

    The routing reason (e.g. ``"r1_hard_rule"``, ``"user_method"``) is
    retained in the seen-set key so that different reasons for the same
    chosen path each get one announce, but is not printed in the message
    text itself --- the user-facing distinction that matters is which
    path ran, not why.

    This function announces the routing DECISION only. Time estimates
    (``"estimated X s; Ctrl+C to cancel"``) are a separate concern. On
    the evaluation path they come from the self-calibrated
    ``_maybe_warn_eval_time``, emitted at most once per top-level call
    and gated by ``show_hints``; other paths (cosine similarity, batched
    estimates) use :func:`estimate_comp_time`. The decision announce and
    the time estimate never double-report a single dispatch.

    Gating: dispatch decisions are NOT gated by per-call
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
    print(f"{func_name}: chose '{chosen}' path.")
