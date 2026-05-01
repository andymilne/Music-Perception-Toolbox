"""Internal utility functions for the Music Perception Toolbox."""

from __future__ import annotations

import time
import warnings
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

    if verbose and label:
        if est_sec < 1:
            ts = f"{est_sec * 1000:.0f} ms"
        elif est_sec < 60:
            ts = f"{est_sec:.1f} s"
        elif est_sec < 3600:
            ts = f"{est_sec / 60:.1f} min"
        else:
            ts = f"{est_sec / 3600:.1f} hr"
        cancel = " (Ctrl+C to cancel)" if est_sec > 2 else ""
        print(f"{label}: estimated time ~{ts}{cancel}.")

    return est_sec
