"""Per-position pulse-level circular measures (non-Fourier).

Public entry points (each returns a per-position profile around the
periodic cycle, either at supplied query points or at the integer-grid
positions of the cycle):

* :func:`edges` --- circular edge detection via convolution with the
  first derivative of a von Mises kernel.
* :func:`mean_offset` --- per-position net upward-arc balance to all
  events (generalises Huron's average pitch height).
* :func:`circ_apm` --- circular autocorrelation phase matrix and its
  metrical-weight / lag-weight profiles.
* :func:`markov_s` --- optimal S-step Markov predictor.

The Fourier-based per-position measure :func:`proj_centroid` lives in
:mod:`._circular.dft`; see USER_GUIDE §6.5 ("Scale and rhythm
structure") for the user-facing description of this module alongside
:mod:`._circular.scale`.
"""
from __future__ import annotations

import numpy as np
from scipy.special import i0 as _besseli0

from .._utils import validate_weights






# ===================================================================
#  Edges
# ===================================================================


def edges(
    p,
    w=None,
    period: float = 1200.0,
    x=None,
    *,
    kappa: float = 6.7,
) -> tuple[np.ndarray | list[np.ndarray], np.ndarray | list[np.ndarray]]:
    """Edge detection on a weighted circular multiset.

    Accepts two input forms, dispatched on ``p``'s shape:

    - 1-D ``p``: single multiset, returns ``(e, e_signed)``.
    - 2-D ``P`` (shape ``(M, K)``): batched, returns
      ``(e_list, e_signed_list)`` — length-``M`` lists of arrays.
      Per-row dedup over permutation + period symmetries.

    Computes the "edginess" at each query point by evaluating the
    circular convolution of the weighted multiset with the first
    derivative of a von Mises kernel, and taking absolute values.
    By default, query points are ``0, 1, ..., period-1``.

    Positions near a sharp transition between event-dense and
    event-sparse regions receive high edge weights; positions in
    uniformly dense or sparse regions receive low weights.

    Parameters
    ----------
    p : array-like
        Pitch or position values (length *K*; or ``(M, K)`` for
        batched).
    w : array-like or None
        Weights (``None`` for all ones; same shape as ``p`` or
        length-``K`` broadcast in batched mode).
    period : float
        Period of the circular domain.
    x : array-like or None
        Query points (default: ``0:period-1``; shared in batched
        mode).
    kappa : float
        Concentration parameter of the von Mises kernel (default 6.7).
        Larger values detect sharper edges.

    Returns
    -------
    e, e_signed
        Absolute and signed edge weights. Single arrays in scalar
        mode; lists of arrays in batched mode.

    References
    ----------
    Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of
    rhythmic structure on tapping accuracy. *Attention, Perception,
    & Psychophysics*, 85, 2673–2699.
    """
    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.ndim == 2:
        return _edges_batched(p_arr, w, period, x, kappa)

    p_arr = p_arr.ravel()
    K = len(p_arr)
    w_arr = validate_weights(w, K)

    if x is None:
        x = np.arange(period)
    x = np.asarray(x, dtype=np.float64).ravel()

    theta = 2 * np.pi * (x[:, None] - p_arr[None, :]) / period
    norm_const = 2 * np.pi * _besseli0(kappa)
    kernel = -kappa * np.sin(theta) * np.exp(kappa * np.cos(theta)) / norm_const

    e_signed = (kernel @ w_arr)
    e = np.abs(e_signed)
    return e, e_signed




def _edges_batched(P, W, period, x, kappa):
    """Batched dispatch for ``edges``."""
    M, K = P.shape

    use_w = W is not None
    if use_w:
        W_arr = np.asarray(W, dtype=np.float64)
        if W_arr.ndim == 1 and W_arr.size == K:
            W_broadcast = W_arr
            W_full = None
        elif W_arr.shape == P.shape:
            W_broadcast = None
            W_full = W_arr
        else:
            raise ValueError(
                "W must be None, a matrix the same shape as P, or "
                "a length-K vector broadcast across rows."
            )
    else:
        W_broadcast = None
        W_full = None

    e_list = [np.array([], dtype=np.float64) for _ in range(M)]
    es_list = [np.array([], dtype=np.float64) for _ in range(M)]
    cache: dict = {}

    for i in range(M):
        p_row = P[i]
        mask = ~np.isnan(p_row)
        p_valid = p_row[mask]
        if len(p_valid) == 0:
            continue
        if W_full is not None:
            w_valid = W_full[i, mask]
        elif W_broadcast is not None:
            w_valid = W_broadcast[mask]
        else:
            w_valid = np.ones_like(p_valid)

        p_mod = np.mod(p_valid, period)
        sort_idx = np.argsort(p_mod)
        p_sorted = p_mod[sort_idx]
        w_sorted = w_valid[sort_idx]
        key = (
            tuple(np.round(p_sorted, 12).tolist()),
            tuple(np.round(w_sorted, 12).tolist()),
        )

        if key in cache:
            e_list[i], es_list[i] = cache[key]
            continue

        e_i, es_i = edges(p_valid, w_valid, period, x, kappa=kappa)
        e_list[i] = e_i
        es_list[i] = es_i
        cache[key] = (e_i, es_i)

    return e_list, es_list




# ===================================================================
#  Mean offset
# ===================================================================


def mean_offset(
    p,
    w=None,
    period: float = 1200.0,
    x=None,
) -> np.ndarray | list[np.ndarray]:
    """Mean offset (net upward arc) of a weighted circular multiset.

    Accepts two input forms, dispatched on ``p``'s shape:

    - 1-D ``p``: single multiset, returns a length-``len(x)`` array
      (or length-``period`` if ``x`` is None).
    - 2-D ``P`` (shape ``(M, K)``): batched, returns a length-``M``
      list of arrays. Per-row canonical-form dedup over permutation
      + period symmetries (not transposition).

    In a pitch-class context, this formalises and generalises
    Huron's (2008) "average pitch height." The term "mode height"
    for a closely related concept is used by Hearne (2020) and
    Tymoczko (2023).

    Parameters
    ----------
    p : array-like
        Pitch or position values (length *K*; or ``(M, K)`` for
        batched).
    w : array-like or None
        Weights (``None`` for all ones; same shape as ``p`` or a
        length-``K`` vector broadcast across rows in batched mode).
    period : float
        Period of the circular domain.
    x : array-like or None
        Query points (default: ``0:period-1``; shared across all
        rows in batched mode).

    Returns
    -------
    np.ndarray or list of np.ndarray
        Mean offset values. Single array in scalar mode; list of
        per-row arrays in batched mode.

    References
    ----------
    Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of
    rhythmic structure on tapping accuracy. *Attention, Perception,
    & Psychophysics*, 85, 2673–2699.
    """
    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.ndim == 2:
        return _mean_offset_batched(p_arr, w, period, x)

    p_arr = p_arr.ravel()
    K = len(p_arr)
    w_arr = validate_weights(w, K)

    if x is None:
        x = np.arange(period)
    x = np.asarray(x, dtype=np.float64).ravel()

    upward = (p_arr[None, :] - x[:, None]) % period
    downward = (x[:, None] - p_arr[None, :]) % period

    h = ((upward - downward) @ w_arr) / period
    return h




def _mean_offset_batched(P, W, period, x):
    """Batched dispatch for ``mean_offset``."""
    M, K = P.shape

    use_w = W is not None
    if use_w:
        W_arr = np.asarray(W, dtype=np.float64)
        if W_arr.ndim == 1 and W_arr.size == K:
            W_broadcast = W_arr
            W_full = None
        elif W_arr.shape == P.shape:
            W_broadcast = None
            W_full = W_arr
        else:
            raise ValueError(
                "W must be None, a matrix the same shape as P, or "
                "a length-K vector broadcast across rows."
            )
    else:
        W_broadcast = None
        W_full = None

    out = [np.array([], dtype=np.float64) for _ in range(M)]
    cache: dict = {}

    for i in range(M):
        p_row = P[i]
        mask = ~np.isnan(p_row)
        p_valid = p_row[mask]
        if len(p_valid) == 0:
            continue
        if W_full is not None:
            w_valid = W_full[i, mask]
        elif W_broadcast is not None:
            w_valid = W_broadcast[mask]
        else:
            w_valid = np.ones_like(p_valid)

        p_mod = np.mod(p_valid, period)
        sort_idx = np.argsort(p_mod)
        p_sorted = p_mod[sort_idx]
        w_sorted = w_valid[sort_idx]
        key = (
            tuple(np.round(p_sorted, 12).tolist()),
            tuple(np.round(w_sorted, 12).tolist()),
        )

        if key in cache:
            out[i] = cache[key]
            continue

        h_i = mean_offset(p_valid, w_valid, period, x)
        out[i] = h_i
        cache[key] = h_i

    return out




# ===================================================================
#  Circular autocorrelation phase matrix
# ===================================================================


def circ_apm(
    p,
    w=None,
    period: int = 12,
    *,
    decay: float = 0.0,
) -> tuple[np.ndarray | list[np.ndarray], np.ndarray | list[np.ndarray], np.ndarray | list[np.ndarray]]:
    """Circular autocorrelation phase matrix.

    Accepts two input forms, dispatched on ``p``'s shape:

    - 1-D ``p``: single multiset, returns ``(R, r_phase, r_lag)``.
    - 2-D ``P`` (shape ``(M, K)``): batched, returns three length-``M``
      lists. Per-row dedup over permutation + period symmetries.

    Returns the circular autocorrelation phase matrix (APM) of the
    weighted multiset *p* within a cycle of length *period* (*p*
    represents pitches or positions). The APM decomposes the circular
    autocorrelation into contributions at each combination of lag
    and phase.

    Parameters
    ----------
    p : array-like of int
        Pitch or position values. Non-negative integers < *period*
        (or ``(M, K)`` matrix in batched mode).
    w : array-like or None
        Weights (``None`` for all ones; same shape as ``p`` or
        length-``K`` broadcast in batched mode).
    period : int
        Cycle length.
    decay : float
        Exponential decay rate (default 0, no decay).

    Returns
    -------
    R, r_phase, r_lag
        In scalar mode: APM (*period* × *period*) plus column-sum
        (metrical weight) and row-sum (autocorrelation) vectors.
        In batched mode: three length-``M`` lists. Note: each ``R``
        is dense ``period × period``; for large ``M`` and ``period``
        memory grows quickly — process in chunks if needed.

    References
    ----------
    Eck, D. (2006). Beat tracking using an autocorrelation phase
    matrix. *Proc. ICMC*.

    Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of
    rhythmic structure on tapping accuracy. *Attention, Perception,
    & Psychophysics*, 85, 2673–2699.
    """
    p_arr = np.asarray(p)
    if p_arr.ndim == 2:
        return _circ_apm_batched(p_arr, w, period, decay)

    p_arr = np.asarray(p_arr, dtype=np.int64).ravel()
    K = len(p_arr)
    w_arr = validate_weights(w, K)
    period = int(period)

    if np.any(p_arr >= period):
        raise ValueError("All positions in p must be less than period.")

    N = period
    s = np.zeros(N)
    for i in range(K):
        s[p_arr[i]] += w_arr[i]

    steps = np.arange(N)
    R = np.zeros((N, N))

    for lag in range(N):
        for phi in range(N):
            pos1 = (lag * steps + phi) % N
            pos2 = (lag * (steps + 1) + phi) % N
            s1 = s[pos1]
            s2 = s[pos2]

            if decay > 0:
                idx1 = lag * steps + phi
                idx2 = lag * (steps + 1) + phi
                s1 = s1 * np.exp(-decay * idx1)
                s2 = s2 * np.exp(-decay * idx2)

            R[lag, phi] = np.dot(s1, s2)

    r_phase = np.sum(R, axis=0)
    r_lag = np.sum(R, axis=1)
    return R, r_phase, r_lag




def _circ_apm_batched(P, W, period, decay):
    """Batched dispatch for ``circ_apm``."""
    M, K = P.shape

    use_w = W is not None
    if use_w:
        W_arr = np.asarray(W, dtype=np.float64)
        if W_arr.ndim == 1 and W_arr.size == K:
            W_broadcast = W_arr
            W_full = None
        elif W_arr.shape == P.shape:
            W_broadcast = None
            W_full = W_arr
        else:
            raise ValueError(
                "W must be None, a matrix the same shape as P, or "
                "a length-K vector broadcast across rows."
            )
    else:
        W_broadcast = None
        W_full = None

    R_list: list = [None] * M
    rp_list: list = [None] * M
    rl_list: list = [None] * M
    cache: dict = {}

    for i in range(M):
        p_row = P[i]
        # circ_apm is integer-only; floats with fractional parts indicate
        # an upstream bug, but NaN-padding is allowed.
        mask = ~np.isnan(p_row.astype(np.float64))
        p_valid = p_row[mask]
        if len(p_valid) == 0:
            R_list[i] = np.array([])
            rp_list[i] = np.array([])
            rl_list[i] = np.array([])
            continue
        p_valid_int = np.asarray(p_valid, dtype=np.int64)
        if not np.array_equal(p_valid_int, p_valid):
            raise ValueError(
                f"circ_apm requires integer pitches; row {i} contains "
                f"non-integer values."
            )
        if W_full is not None:
            w_valid = W_full[i, mask]
        elif W_broadcast is not None:
            w_valid = W_broadcast[mask]
        else:
            w_valid = np.ones_like(p_valid_int, dtype=np.float64)

        # Reduce mod period for canonical key (and for the call itself).
        p_mod = np.mod(p_valid_int, period)
        sort_idx = np.argsort(p_mod)
        p_sorted = p_mod[sort_idx]
        w_sorted = w_valid[sort_idx]
        key = (
            tuple(p_sorted.tolist()),
            tuple(np.round(w_sorted, 12).tolist()),
        )

        if key in cache:
            R_list[i], rp_list[i], rl_list[i] = cache[key]
            continue

        R_i, rp_i, rl_i = circ_apm(p_sorted, w_sorted, period, decay=decay)
        R_list[i] = R_i
        rp_list[i] = rp_i
        rl_list[i] = rl_i
        cache[key] = (R_i, rp_i, rl_i)

    return R_list, rp_list, rl_list




# ===================================================================
#  Markov predictor
# ===================================================================


def markov_s(
    p: np.ndarray,
    w: np.ndarray | None,
    period: int,
    S: int = 3,
) -> np.ndarray:
    """Optimal S-step Markov predictor for a periodic weighted multiset.

    Returns the predicted weight at each integer position
    ``0, 1, ..., period-1`` of a cycle (*p* represents pitches or
    positions). For each position *j*, the predictor finds all
    positions whose S-step future context (the binary pattern of
    events and non-events) is identical, and averages their weights.

    Parameters
    ----------
    p : array-like of int
        Pitch or position values. Non-negative integers < *period*.
    w : array-like or None
        Weights (``None`` for all ones).
    period : int
        Cycle length.
    S : int
        Lookahead context steps (default 3).

    Returns
    -------
    np.ndarray
        Predicted event weights (length *period*).

    References
    ----------
    Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of
    rhythmic structure on tapping accuracy. *Attention, Perception,
    & Psychophysics*, 85, 2673–2699.
    """
    p = np.asarray(p, dtype=np.int64).ravel()
    K = len(p)
    w = validate_weights(w, K)
    period = int(period)

    if np.any((p < 0) | (p >= period)):
        raise ValueError("Positions must be non-negative integers < period.")

    N = period
    w_cycle = np.zeros(N)
    for i in range(K):
        w_cycle[p[i]] += w[i]

    bin_cycle = (w_cycle != 0)

    # E[i,j] = 1 iff positions i and j have the same binary status
    E = (bin_cycle[:, None] == bin_cycle[None, :])
    T = np.ones((N, N), dtype=bool)
    for k in range(1, S + 1):
        T &= np.roll(np.roll(E, k, axis=1), k, axis=0)

    y = (w_cycle @ T.astype(np.float64)) / np.sum(T, axis=0)
    return y