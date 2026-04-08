"""Measures on circular pitch-class and time-class sets.

Includes the DFT-based family (balance, evenness), scale-theoretic
measures (coherence, sameness), and pulse-level predictors
(edges, projected centroid, mean offset, autocorrelation phase
matrix, Markov predictor).
"""

from __future__ import annotations

import numpy as np
from scipy.special import i0 as _besseli0

from ._utils import validate_weights


# ===================================================================
#  DFT of a circular set
# ===================================================================


def dft_circular(
    p: np.ndarray,
    w: np.ndarray | None,
    period: float,
) -> tuple[np.ndarray, np.ndarray]:
    """DFT of a set of points on a circle.

    Parameters
    ----------
    p : array-like
        Pitch-class (or time-class) values (length *K*).
    w : array-like or None
        Weights (``None`` for uniform).
    period : float
        Period of the circular domain.

    Returns
    -------
    F : np.ndarray
        Complex Fourier coefficients (length *K*). ``F[0]`` is *k* = 0.
    mag : np.ndarray
        Magnitudes |F[k]|.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    K = len(p)
    w = validate_weights(w, K)

    # Sort by pitch class
    idx = np.argsort(p)
    p = p[idx]
    w = w[idx]

    z = w * np.exp(2j * np.pi * p / period)
    F = np.fft.fft(z) / np.sum(w)
    mag = np.abs(F)
    return F, mag


# ===================================================================
#  Balance
# ===================================================================


def balance(p: np.ndarray, w: np.ndarray | None, period: float) -> float:
    """Balance of a circular set.

    ``b = 1 − |F(0)|``, where *F(0)* is the weighted centre of
    gravity on the unit circle. Range [0, 1].

    Parameters
    ----------
    p : array-like
        Positions.
    w : array-like or None
        Weights.
    period : float
        Period.

    Returns
    -------
    float
    """
    _, mag = dft_circular(p, w, period)
    return float(1 - mag[0])


# ===================================================================
#  Evenness
# ===================================================================


def evenness(p: np.ndarray, period: float) -> float:
    """Evenness of a circular set.

    ``e = |F(1)|``. Always uses uniform weights. Range [0, 1].

    Parameters
    ----------
    p : array-like
        Positions.
    period : float
        Period.

    Returns
    -------
    float
    """
    _, mag = dft_circular(p, None, period)
    return float(mag[1])


# ===================================================================
#  Coherence
# ===================================================================


def coherence(
    p: np.ndarray,
    period: float,
    *,
    strict: bool = True,
) -> tuple[float, int]:
    """Coherence quotient (Carey 2002, 2007; Rothenberg 1978).

    Measures how consistently the rank ordering of intervals by generic
    span matches their ordering by specific size.  Because only rank
    ordering matters, positions may be integers (within an equal
    division) or floats (e.g. cents values of a just-intonation scale).

    Parameters
    ----------
    p : array-like
        Positions (length *K*).  Non-negative values less than *period*.
    period : float
        Period of the circular domain.
    strict : bool
        If True (default), strict propriety (Rothenberg 1978).

    Returns
    -------
    c : float
        Coherence quotient in [0, 1].
    nc : int
        Number of coherence failures.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    period = float(period)
    p = np.sort(p % period)
    K = len(p)
    if len(np.unique(p)) != K:
        raise ValueError("p must not contain duplicate pitch classes (mod period).")
    if K < 2:
        raise ValueError(f"At least 2 events required (got {K}).")

    # Interval-size table
    size_span = np.zeros((K - 1, K), dtype=np.float64)
    for g in range(1, K):
        size_span[g - 1] = (p[(np.arange(K) + g) % K] - p) % period

    nc = 0
    for g2 in range(2, K):
        sizes2 = size_span[g2 - 1]
        for g1 in range(1, g2):
            sizes1 = size_span[g1 - 1]
            diffs = sizes2[:, None] - sizes1[None, :]
            if strict:
                nc += int(np.sum(diffs <= 0))
            else:
                nc += int(np.sum(diffs < 0))

    max_nc = K * (K - 1) * (K - 2) * (3 * K - 5) / 24
    c = 1 - nc / max_nc
    return c, nc


# ===================================================================
#  Sameness
# ===================================================================


def sameness(
    p: np.ndarray,
    period: int,
) -> tuple[float, int]:
    """Sameness quotient (Carey 2002, 2007).

    Parameters
    ----------
    p : array-like of int
        Positions.
    period : int
        Equal division size.

    Returns
    -------
    sq : float
        Sameness quotient in [0, 1].
    n_diff : int
        Number of ambiguities.
    """
    p = np.asarray(p, dtype=np.int64).ravel()
    period = int(period)
    p = np.sort(p % period)
    K = len(p)
    if len(np.unique(p)) != K:
        raise ValueError("p must not contain duplicate pitch classes (mod period).")
    if K < 2:
        raise ValueError(f"At least 2 events required (got {K}).")

    size_span = np.zeros((K - 1, K), dtype=np.int64)
    for g in range(1, K):
        size_span[g - 1] = (p[(np.arange(K) + g) % K] - p) % period

    size_counts = np.zeros((K - 1, period), dtype=np.int64)
    for g in range(K - 1):
        for k in range(K):
            s = size_span[g, k]
            size_counts[g, s] += 1

    col_totals = np.sum(size_counts, axis=0)
    n_diff = int((np.sum(col_totals**2) - np.sum(size_counts**2)) // 2)

    max_diff = K * (K - 1) ** 2 / 2
    sq = 1 - n_diff / max_diff
    return sq, n_diff


# ===================================================================
#  Edges
# ===================================================================


def edges(
    p: np.ndarray,
    w: np.ndarray | None,
    period: float,
    x: np.ndarray | None = None,
    *,
    kappa: float = 6.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Edge detection on a circular set.

    Convolves with the derivative of a von Mises kernel.

    Parameters
    ----------
    p : array-like
        Event positions.
    w : array-like or None
        Weights.
    period : float
        Period.
    x : array-like or None
        Query points (default ``0:period-1``).
    kappa : float
        Concentration parameter.

    Returns
    -------
    e : np.ndarray
        Absolute edge weights.
    e_signed : np.ndarray
        Signed edge weights.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    K = len(p)
    w = validate_weights(w, K)

    if x is None:
        x = np.arange(period)
    x = np.asarray(x, dtype=np.float64).ravel()

    theta = 2 * np.pi * (x[:, None] - p[None, :]) / period  # (nQ, K)
    norm_const = 2 * np.pi * _besseli0(kappa)
    kernel = -kappa * np.sin(theta) * np.exp(kappa * np.cos(theta)) / norm_const

    e_signed = (kernel @ w)  # (nQ,)
    e = np.abs(e_signed)
    return e, e_signed


# ===================================================================
#  Projected centroid
# ===================================================================


def proj_centroid(
    p: np.ndarray,
    w: np.ndarray | None,
    period: float,
    x: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """Projected centroid of a circular set.

    Parameters
    ----------
    p, w, period :
        Event set.
    x : array-like or None
        Query points (default ``0:period-1``).

    Returns
    -------
    y : np.ndarray
        Projected centroid values.
    cent_mag : float
        Centroid magnitude |F(0)|.
    cent_phase : float
        Centroid phase in user units.
    """
    if x is None:
        x = np.arange(period)
    x = np.asarray(x, dtype=np.float64).ravel()

    F, _ = dft_circular(p, w, period)
    F0 = F[0]
    cent_mag = float(np.abs(F0))
    cent_phase_rad = float(np.angle(F0) % (2 * np.pi))

    query_angles = 2 * np.pi * x / period
    y = cent_mag * np.cos(cent_phase_rad - query_angles)

    cent_phase = cent_phase_rad * period / (2 * np.pi)
    return y, cent_mag, cent_phase


# ===================================================================
#  Mean offset
# ===================================================================


def mean_offset(
    p: np.ndarray,
    w: np.ndarray | None,
    period: float,
    x: np.ndarray | None = None,
) -> np.ndarray:
    """Mean offset (net upward arc) of a circular set.

    Parameters
    ----------
    p, w, period :
        Event set.
    x : array-like or None
        Query points (default ``0:period-1``).

    Returns
    -------
    np.ndarray
        Mean offset at each query point.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    K = len(p)
    w = validate_weights(w, K)

    if x is None:
        x = np.arange(period)
    x = np.asarray(x, dtype=np.float64).ravel()

    # upward and downward arcs: (nQ, K)
    upward = (p[None, :] - x[:, None]) % period
    downward = (x[:, None] - p[None, :]) % period

    h = ((upward - downward) @ w) / period
    return h


# ===================================================================
#  Circular autocorrelation phase matrix
# ===================================================================


def circ_apm(
    p: np.ndarray,
    w: np.ndarray | None,
    period: int,
    *,
    decay: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Circular autocorrelation phase matrix.

    Parameters
    ----------
    p : array-like of int
        Event positions.
    w : array-like or None
        Weights.
    period : int
        Cycle length.
    decay : float
        Exponential decay rate (default 0, no decay).

    Returns
    -------
    R : (period, period) array
        APM. ``R[l, phi]`` for lag *l*, phase *phi*.
    r_phase : (period,) array
        Column sum (metrical weight model).
    r_lag : (period,) array
        Row sum (circular autocorrelation).
    """
    p = np.asarray(p, dtype=np.int64).ravel()
    K = len(p)
    w = validate_weights(w, K)
    period = int(period)

    if np.any(p >= period):
        raise ValueError("All positions in p must be less than period.")

    N = period
    s = np.zeros(N)
    for i in range(K):
        s[p[i]] += w[i]

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


# ===================================================================
#  Markov predictor
# ===================================================================


def markov_s(
    p: np.ndarray,
    w: np.ndarray | None,
    period: int,
    S: int = 3,
) -> np.ndarray:
    """Optimal S-step Markov predictor for a periodic event sequence.

    Parameters
    ----------
    p : array-like of int
        Event positions.
    w : array-like or None
        Weights.
    period : int
        Cycle length.
    S : int
        Lookahead context steps (default 3).

    Returns
    -------
    np.ndarray
        Predicted event weights (length *period*).
    """
    p = np.asarray(p, dtype=np.int64).ravel()
    K = len(p)
    w = validate_weights(w, K)
    period = int(period)

    if np.any((p < 0) | (p >= period)):
        raise ValueError("Positions must be non-negative integers < period.")

    N = period
    x_w = np.zeros(N)
    for i in range(K):
        x_w[p[i]] += w[i]

    x_bin = (x_w != 0)

    # E[i,j] = 1 iff positions i and j have the same binary status
    E = (x_bin[:, None] == x_bin[None, :])
    T = np.ones((N, N), dtype=bool)
    for k in range(1, S + 1):
        T &= np.roll(np.roll(E, k, axis=1), k, axis=0)

    y = (x_w @ T.astype(np.float64)) / np.sum(T, axis=0)
    return y
