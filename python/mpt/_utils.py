"""Internal utility functions for the Music Perception Toolbox."""

from __future__ import annotations

import threading
import time
import warnings
from contextlib import contextmanager
from functools import lru_cache

import numpy as np


def validate_weights(
    w: np.ndarray | None, n: int, *, allow_empty: bool = True
) -> np.ndarray:
    """Validate and normalise a weight vector.

    Parameters
    ----------
    w : array-like or None
        Weights. ``None`` or an empty array gives all ones.
        A scalar is broadcast to length *n*.
    n : int
        Required length of the output vector.
    allow_empty : bool
        If True (default), ``None`` / empty → ones.

    Returns
    -------
    np.ndarray
        1-D float64 weight vector of length *n*.
    """
    if w is None or (hasattr(w, "__len__") and len(w) == 0):
        if allow_empty:
            return np.ones(n, dtype=np.float64)
        raise ValueError("w must not be empty.")
    w = np.asarray(w, dtype=np.float64).ravel()
    if w.size == 1:
        if w[0] == 0:
            warnings.warn("All weights in w are zero.")
        w = np.full(n, w[0], dtype=np.float64)
    if w.size != n:
        raise ValueError(
            f"w must have the same number of entries as p ({n}), got {w.size}."
        )
    return w


# ---------------------------------------------------------------------------
#  Computation-time estimation
# ---------------------------------------------------------------------------

_rate_cache: dict[int, float] = {}


def estimate_comp_time(
    n_pairs: int | float,
    dim: int,
    label: str = "",
    verbose: bool = True,
    min_print_sec: float = 10.0,
) -> float:
    """Estimate computation time for kernel evaluation.

    Parameters
    ----------
    n_pairs : int or float
        Total number of (tuple, query) pair evaluations.
    dim : int
        Dimensionality of the difference vectors.
    label : str
        Description for console output. Empty string suppresses output.
    verbose : bool
        If False, suppresses all console output.
    min_print_sec : float
        Minimum estimated time in seconds below which printing is
        suppressed even when ``verbose`` is True. Default 10 — below
        this threshold the wait is short enough to be its own
        diagnostic; above it the estimate is informative enough to
        justify the screen real estate. The print includes a
        ``Ctrl+C to cancel`` reminder, since every printed estimate
        by definition takes long enough to be worth offering
        cancellation. Pass ``0`` to print every estimate regardless
        of size.

    Returns
    -------
    float
        Estimated time in seconds.
    """
    if dim not in _rate_cache:
        n_cal = 1000
        rng = np.random.default_rng(42)
        u = rng.standard_normal((dim, n_cal))
        v = rng.standard_normal((dim, n_cal))
        ww = rng.standard_normal(n_cal)

        # Warm-up
        d = u[:, :, None] - v[:, None, :]
        q = np.sum(d**2, axis=0)
        e = np.exp(-q).reshape(n_cal, n_cal)
        _ = ww @ e

        # Timed run
        t0 = time.perf_counter()
        d = u[:, :, None] - v[:, None, :]
        q = np.sum(d**2, axis=0)
        e = np.exp(-q).reshape(n_cal, n_cal)
        _ = ww @ e
        elapsed = time.perf_counter() - t0

        _rate_cache[dim] = (n_cal * n_cal) / max(elapsed, 1e-12)

    est_sec = float(n_pairs) / _rate_cache[dim]

    if verbose and label and est_sec >= min_print_sec:
        if est_sec < 1:
            ts = f"{est_sec * 1000:.0f} ms"
        elif est_sec < 60:
            ts = f"{est_sec:.1f} s"
        elif est_sec < 3600:
            ts = f"{est_sec / 60:.1f} min"
        else:
            ts = f"{est_sec / 3600:.1f} hr"
        # Print and cancellation thresholds are by construction the same;
        # any printed estimate carries the cancellation reminder.
        print(f"{label}: estimated time ~{ts} (Ctrl+C to cancel).")

    return est_sec


# ---------------------------------------------------------------------------
#  Batched-mode empirical-calibration print helper
# ---------------------------------------------------------------------------


def maybe_print_batched_estimate(
    label: str,
    n_rows: int,
    est_total: float,
    *,
    verbose: bool = True,
    min_print_sec: float = 10.0,
) -> None:
    """Print a batched-mode upfront time estimate, gated on threshold.

    Used by the batched dispatch helpers in ``template_harmonicity``,
    ``virtual_pitches``, ``spectral_entropy``, ``entropy_exp_tens``,
    ``tensor_harmonicity``, and similar functions, which compute their
    estimates empirically (warm-up plus a sample of K rows) rather than
    via :func:`estimate_comp_time`. The threshold and formatting match
    the scalar-mode print path so behaviour is consistent across
    dispatch modes.

    Parameters
    ----------
    label : str
        Function name tag used in the printed line, e.g.
        ``"spectral_entropy"``.
    n_rows : int
        Total number of rows in the batched call (M).
    est_total : float
        Empirical estimate in seconds (calibration time plus
        per-row time × M).
    verbose : bool
        If False, suppresses output.
    min_print_sec : float
        Minimum threshold; estimates below this are silent. Default
        10, matching :func:`estimate_comp_time`.
    """
    if not verbose or est_total < min_print_sec:
        return
    if est_total < 1:
        ts = f"{est_total * 1000:.0f} ms"
    elif est_total < 60:
        ts = f"{est_total:.1f} s"
    elif est_total < 3600:
        ts = f"{est_total / 60:.1f} min"
    else:
        ts = f"{est_total / 3600:.1f} hr"
    print(
        f"{label} (batched, {n_rows} rows): "
        f"estimated time ~{ts} (Ctrl+C to cancel)."
    )


# ---------------------------------------------------------------------------
#  Adaptive progress-print stride
# ---------------------------------------------------------------------------


def progress_stride(t_per_row: float, target_sec: float = 5.0) -> int:
    """Compute progress-print stride for batched loops.

    Returns the smallest "nice" stride from the set
    ``{1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000}``
    such that each print interval takes at least ``target_sec`` seconds
    at the given per-row cost ``t_per_row`` (seconds per row, from an
    empirical calibration). If ``t_per_row`` is so small that even a
    stride of 10000 rows finishes in less than the target, the function
    returns 10000 (the largest available option); progress prints will
    then be more frequent than the target, which is the best achievable
    without wider stride options.

    Used by the batched helpers (``cos_sim_exp_tens``, ``template_harmonicity``,
    ``spectral_entropy``, ``virtual_pitches``, ``entropy_exp_tens``) to
    set the cadence of their "X / Y rows computed" progress prints.
    Callers also gate the prints on ``est_total >= target_sec`` (i.e.
    only show progress at all when the loop is expected to take long
    enough to warrant it); the stride determines cadence within that,
    and adapts down to 1 when individual rows are themselves slow
    enough that per-row prints don't exceed the target interval.

    Parameters
    ----------
    t_per_row : float
        Per-row cost in seconds. Non-positive or non-finite values
        fall back to a stride of 1.
    target_sec : float
        Target print interval in seconds. Defaults to 5.

    Returns
    -------
    int
        A positive integer from the option set.
    """
    options = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
    if not (t_per_row > 0) or not np.isfinite(t_per_row):
        return options[0]
    desired = target_sec / t_per_row
    for opt in options:
        if opt >= desired:
            return opt
    return options[-1]


# ---------------------------------------------------------------------------
#  Position-aware variance
# ---------------------------------------------------------------------------


def position_variance(
    idx: np.ndarray,
    signs: np.ndarray,
    sigma: float,
) -> float:
    """Variance of a signed sum of independently jittered positions.

    Used by the position-aware paths of :func:`sameness`,
    :func:`coherence`, and other measures whose readouts depend on
    differences of intervals derived from a shared position set under
    independent positional jitter ``p_k ~ N(p_k, sigma**2)``.

    Computes::

        V = sigma**2 * sum_a (sum of signs at index a)**2

    Repeated indices add their signs algebraically before squaring,
    so a position that enters once with sign +1 and once with sign -1
    contributes nothing to the variance, while one that enters twice
    with the same sign contributes ``4 * sigma**2``.

    Parameters
    ----------
    idx : array-like of int
        Position indices (with possible repeats).
    signs : array-like
        Signed contributions, same length as *idx*. Typically +1 or -1.
    sigma : float
        Per-position standard deviation.

    Returns
    -------
    float
        Variance of the signed sum.
    """
    idx = np.asarray(idx).ravel()
    signs = np.asarray(signs, dtype=np.float64).ravel()
    u_idx, grp = np.unique(idx, return_inverse=True)
    net = np.zeros(u_idx.size, dtype=np.float64)
    np.add.at(net, grp, signs)
    return float(sigma**2 * np.sum(net**2))


# ---------------------------------------------------------------
# Memory-aware chunk-size resolution for kernel-matrix workloads
# ---------------------------------------------------------------

def _available_memory_bytes_linux() -> int | None:
    """Return ``MemAvailable`` from ``/proc/meminfo`` in bytes, or None."""
    try:
        with open("/proc/meminfo", "r") as f:
            text = f.read()
    except OSError:
        return None
    import re
    m = re.search(r"^MemAvailable:\s+(\d+)\s+kB", text, flags=re.MULTILINE)
    if m is None:
        return None
    return int(m.group(1)) * 1024


def _available_memory_bytes_macos() -> int | None:
    """Return available memory on macOS via ``vm_stat``, or None.

    Uses ``free + inactive + speculative`` pages, matching the
    approximation used by Activity Monitor and standard third-party
    memory tools.
    """
    import re
    import subprocess
    try:
        out = subprocess.check_output(
            ["vm_stat"], stderr=subprocess.DEVNULL, text=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    page_match = re.search(r"page size of (\d+) bytes", out)
    if page_match is None:
        return None
    page_size = int(page_match.group(1))

    def _pages(label: str) -> int | None:
        m = re.search(rf"Pages {label}:\s+(\d+)", out)
        return int(m.group(1)) if m else None

    free = _pages("free")
    inactive = _pages("inactive")
    spec = _pages("speculative")
    if free is None or inactive is None or spec is None:
        return None
    return (free + inactive + spec) * page_size


def _available_memory_bytes_windows() -> int | None:
    """Return available physical memory on Windows via GlobalMemoryStatusEx."""
    import ctypes
    try:
        class _MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        ms = _MemStatus()
        ms.dwLength = ctypes.sizeof(_MemStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            return None
        return int(ms.ullAvailPhys)
    except (OSError, AttributeError, OSError):
        return None


_AVAILABLE_MEMORY_FALLBACK_BYTES: int = 4 * 1024 ** 3  # 4 GiB


def available_memory_bytes() -> int:
    """Return currently available physical memory in bytes.

    Linux uses ``/proc/meminfo``'s ``MemAvailable`` (the kernel's
    estimate of memory available without swapping). macOS uses
    ``vm_stat`` with the ``free + inactive + speculative`` page-count
    approximation. Windows uses ``GlobalMemoryStatusEx``. If all
    platform queries fail, a 4 GiB fallback is returned.
    """
    import sys
    val: int | None = None
    if sys.platform.startswith("linux"):
        val = _available_memory_bytes_linux()
    elif sys.platform == "darwin":
        val = _available_memory_bytes_macos()
    elif sys.platform.startswith("win"):
        val = _available_memory_bytes_windows()
    if val is None:
        return _AVAILABLE_MEMORY_FALLBACK_BYTES
    return val


# Per-call pin + cross-call TTL fallback for the 'auto' resolution.
# The pin is thread-local so concurrent threads don't share state; the
# TTL cache is module-global behind a lock (the underlying OS query is
# cheap and idempotent, so accepting some cross-thread reuse is fine).
_chunk_bytes_state = threading.local()  # .pin: int | None; .pin_raw: str | None
_chunk_bytes_ttl_lock = threading.Lock()
_chunk_bytes_ttl_cache: int | None = None
_chunk_bytes_ttl_raw: str | None = None
_chunk_bytes_ttl_time: float = 0.0
_CHUNK_BYTES_TTL_SECONDS = 10.0


def kernel_chunk_bytes_resolved() -> int:
    """Return the resolved per-chunk byte budget for kernel-matrix work.

    Reads the ``kernel_chunk_bytes`` toolbox default. Positive integers
    are returned as-is. The factory value ``'auto'`` resolves to half
    of currently available physical memory (see
    :func:`available_memory_bytes`).

    To avoid per-call OS queries when a top-level function recurses
    (e.g. :func:`cosine_sim_exp_tens` batched-raw mode dispatching
    inner per-pair calls), top-level entry points wrap their body in
    :func:`pin_kernel_chunk_bytes`. While pinned, this function
    returns the same cached value rather than re-querying the OS.

    Outside any per-call pin, ``'auto'`` resolutions are cached for
    up to 10 s as a fallback for back-to-back small calls in script
    loops. Changing ``kernel_chunk_bytes`` via ``set_default`` flushes
    the cache naturally (cache lookup is keyed on the raw value).

    The peak transient allocation in a single kernel-matrix chunk
    runs roughly ``(2 * dim + 2) × n_j × n_q × bytes_per_scalar`` —
    NumPy briefly holds the broadcast difference tensor, its square,
    and the summed-then-exponentiated intermediate simultaneously.
    The factor of two between the ``'auto'`` budget (half of
    available memory) and the actual peak gives a safety margin
    against the per-query allocation under-estimate.
    """
    from ._defaults import get_default
    global _chunk_bytes_ttl_cache, _chunk_bytes_ttl_raw, _chunk_bytes_ttl_time
    val = get_default("kernel_chunk_bytes")
    if not (isinstance(val, str) and val == "auto"):
        return int(val)

    # Per-call pin (thread-local).
    pin = getattr(_chunk_bytes_state, "pin", None)
    pin_raw = getattr(_chunk_bytes_state, "pin_raw", None)
    if pin is not None and pin_raw == val:
        return pin

    # TTL fallback (10 s, shared across threads).
    with _chunk_bytes_ttl_lock:
        if (
            _chunk_bytes_ttl_cache is not None
            and _chunk_bytes_ttl_raw == val
            and (time.monotonic() - _chunk_bytes_ttl_time)
                <= _CHUNK_BYTES_TTL_SECONDS
        ):
            return _chunk_bytes_ttl_cache
        fresh = max(1, available_memory_bytes() // 2)
        _chunk_bytes_ttl_cache = fresh
        _chunk_bytes_ttl_raw = val
        _chunk_bytes_ttl_time = time.monotonic()
        return fresh


@contextmanager
def pin_kernel_chunk_bytes():
    """Pin the resolved kernel_chunk_bytes budget for the duration of the
    ``with`` block.

    Used at top-level entry points (e.g. ``cosine_sim_exp_tens``,
    ``eval_exp_tens``) so that recursive or nested inner calls share
    one OS query rather than re-resolving ``'auto'`` per call. Nested
    pins are no-ops; only the outermost ``with`` block actually
    resolves and pins. Thread-local: each thread has its own pin.

    For numeric ``kernel_chunk_bytes`` defaults the pin is a no-op
    (no OS query to amortise).

    Sibling calls — e.g. a ``template_harmonicity`` loop invoking
    ``eval_exp_tens`` repeatedly — each get their own pin instance.
    To avoid one OS query per sibling, ``pin_kernel_chunk_bytes``
    consults the module-level TTL cache before resolving fresh; if
    the cache is warm (set within the last
    ``_CHUNK_BYTES_TTL_SECONDS``), the cached value is pinned and no
    new OS query fires.
    """
    global _chunk_bytes_ttl_cache, _chunk_bytes_ttl_raw, _chunk_bytes_ttl_time
    if getattr(_chunk_bytes_state, "pin", None) is not None:
        # Already pinned by an outer call; nested no-op.
        yield
        return
    from ._defaults import get_default
    raw = get_default("kernel_chunk_bytes")
    if not (isinstance(raw, str) and raw == "auto"):
        # Numeric value: nothing to pin.
        yield
        return
    # Use TTL cache if valid; otherwise resolve fresh and update TTL.
    with _chunk_bytes_ttl_lock:
        if (
            _chunk_bytes_ttl_cache is not None
            and _chunk_bytes_ttl_raw == raw
            and (time.monotonic() - _chunk_bytes_ttl_time)
                <= _CHUNK_BYTES_TTL_SECONDS
        ):
            _chunk_bytes_state.pin = _chunk_bytes_ttl_cache
        else:
            fresh = max(1, available_memory_bytes() // 2)
            _chunk_bytes_state.pin = fresh
            _chunk_bytes_ttl_cache = fresh
            _chunk_bytes_ttl_raw = raw
            _chunk_bytes_ttl_time = time.monotonic()
    _chunk_bytes_state.pin_raw = raw
    try:
        yield
    finally:
        _chunk_bytes_state.pin = None
        _chunk_bytes_state.pin_raw = None


def flush_kernel_chunk_bytes_cache() -> None:
    """Clear all kernel_chunk_bytes resolution caches (pin + TTL).

    Called by ``reset_defaults`` so the next resolution re-queries
    the OS.
    """
    global _chunk_bytes_ttl_cache, _chunk_bytes_ttl_raw, _chunk_bytes_ttl_time
    _chunk_bytes_state.pin = None
    _chunk_bytes_state.pin_raw = None
    with _chunk_bytes_ttl_lock:
        _chunk_bytes_ttl_cache = None
        _chunk_bytes_ttl_raw = None
        _chunk_bytes_ttl_time = 0.0


def with_kernel_chunk_bytes_pin(func):
    """Decorator: enter :func:`pin_kernel_chunk_bytes` for the call.

    Applied to top-level user-facing functions whose internal control
    flow may recurse or invoke other top-level toolbox functions —
    e.g. :func:`cosine_sim_exp_tens` in batched-raw mode.
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with pin_kernel_chunk_bytes():
            return func(*args, **kwargs)

    return wrapper
