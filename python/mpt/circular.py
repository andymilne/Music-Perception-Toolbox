"""Measures on circular multisets of pitches or positions.

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
    """Balance of a weighted circular multiset.

    Computes the balance of a weighted multiset of points on a circle
    (*p* represents pitches or positions), defined as ``1 - |F[0]|``
    where ``F[0]`` is the k = 0 DFT coefficient (see :func:`dft_circular`).

    Balance ranges from 0 to 1: 1 = perfectly balanced (centre of
    gravity at the origin); 0 = all weight at one point.

    Parameters
    ----------
    p : array-like
        Pitch or position values (length *K*).
    w : array-like or None
        Weights (``None`` for uniform).
    period : float
        Period of the circular domain.

    Returns
    -------
    float
        Balance in [0, 1].

    References
    ----------
    Milne, A. J., Bulger, D., & Herff, S. A. (2017). Exploring the
    space of perfectly balanced rhythms and scales. *Journal of
    Mathematics and Music*, 11(2–3), 101–133.
    """
    _, mag = dft_circular(p, w, period)
    return float(1 - mag[0])


# ===================================================================
#  Evenness
# ===================================================================


def evenness(p: np.ndarray, period: float) -> float:
    """Evenness of a circular multiset.

    Computes the evenness of a multiset of *K* points on a circle
    (*p* represents pitches or positions), defined as ``|F[1]|``
    where ``F[1]`` is the k = 1 DFT coefficient (see :func:`dft_circular`).

    Evenness ranges from 0 to 1: 1 = maximally even (equally spaced);
    0 = maximally uneven. Uses uniform weights regardless of input.

    Parameters
    ----------
    p : array-like
        Pitch or position values (length *K*).
    period : float
        Period of the circular domain.

    Returns
    -------
    float
        Evenness in [0, 1].

    References
    ----------
    Milne, A. J., Bulger, D., & Herff, S. A. (2017). Exploring the
    space of perfectly balanced rhythms and scales. *Journal of
    Mathematics and Music*, 11(2–3), 101–133.
    """
    _, mag = dft_circular(p, None, period)
    return float(mag[1])


# ===================================================================
#  Coherence
# ===================================================================


def coherence(
    p: np.ndarray,
    period: int,
    *,
    strict: bool = True,
) -> tuple[float, int]:
    """Coherence quotient of a circular set.

    Returns the coherence quotient of the set of pitches or positions
    *p* within an equal division of size *period* (Carey, 2002). A
    coherence failure occurs when a pair with a larger generic span
    does not have a strictly greater specific size (strict propriety).

    Parameters
    ----------
    p : array-like of int
        Pitch or position values. Non-negative integers less than
        *period*. Duplicates (modulo *period*) are not allowed.
    period : int
        Size of the equal division.
    strict : bool
        If True (default), strict propriety; if False, propriety.

    Returns
    -------
    c : float
        Coherence quotient in [0, 1] (1 = no failures).
    nc : int
        Number of coherence failures.

    References
    ----------
    Carey, N. (2002). On coherence and sameness. *Journal of Music
    Theory*, 46(1/2), 1–56.

    Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of
    rhythmic structure on tapping accuracy. *Attention, Perception,
    & Psychophysics*, 85, 2673–2699.
    """
    p = np.asarray(p, dtype=np.int64).ravel()
    period = int(period)
    p = np.sort(p % period)
    K = len(p)
    if len(np.unique(p)) != K:
        raise ValueError("p must not contain duplicate pitch classes (mod period).")
    if K < 2:
        raise ValueError(f"At least 2 events required (got {K}).")

    # Interval-size table
    size_span = np.zeros((K - 1, K), dtype=np.int64)
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
    """Sameness quotient of a circular set.

    Returns the sameness quotient of the set of pitches or positions
    *p* within an equal division of size *period* (Carey, 2002). An
    ambiguity occurs when two different generic spans share the same
    specific size.

    Parameters
    ----------
    p : array-like of int
        Pitch or position values. Non-negative integers less than
        *period*. Duplicates (modulo *period*) are not allowed.
    period : int
        Size of the equal division.

    Returns
    -------
    sq : float
        Sameness quotient in [0, 1] (1 = no ambiguities).
    n_diff : int
        Number of ambiguities.

    References
    ----------
    Carey, N. (2002). On coherence and sameness. *Journal of Music
    Theory*, 46(1/2), 1–56.
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
    """Edge detection on a weighted circular multiset.

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
        Pitch or position values (length *K*).
    w : array-like or None
        Weights (``None`` for uniform).
    period : float
        Period of the circular domain.
    x : array-like or None
        Query points (default: ``0:period-1``).
    kappa : float
        Concentration parameter of the von Mises kernel (default 6.7).
        Larger values detect sharper edges.

    Returns
    -------
    e : np.ndarray
        Absolute edge weights (non-negative).
    e_signed : np.ndarray
        Signed edge weights (positive = rising edge).

    References
    ----------
    Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of
    rhythmic structure on tapping accuracy. *Attention, Perception,
    & Psychophysics*, 85, 2673–2699.
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
    """Projected centroid of a weighted circular multiset.

    Computes the projection of the circular centroid (centre of
    gravity) onto each angular position. The centroid is the k = 0
    Fourier coefficient of the multiset (see :func:`dft_circular`).

    Parameters
    ----------
    p : array-like
        Pitch or position values (length *K*).
    w : array-like or None
        Weights (``None`` for uniform).
    period : float
        Period of the circular domain.
    x : array-like or None
        Query points (default: ``0:period-1``).

    Returns
    -------
    y : np.ndarray
        Projected centroid values.
    cent_mag : float
        Centroid magnitude |F[0]| (degree of imbalance).
    cent_phase : float
        Centroid phase in the units of *p*.

    References
    ----------
    Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of
    rhythmic structure on tapping accuracy. *Attention, Perception,
    & Psychophysics*, 85, 2673–2699.
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
    """Mean offset (net upward arc) of a weighted circular multiset.

    Computes, at each query point, the sum of upward arc lengths to
    all elements minus the sum of downward arc lengths, with each
    arc normalised by the period. By default, query points are
    ``0, 1, ..., period-1``.

    In a pitch-class context, this formalises and generalises
    Huron's (2008) "average pitch height." The term "mode height"
    for a closely related concept is used by Hearne (2020) and
    Tymoczko (2023).

    Parameters
    ----------
    p : array-like
        Pitch or position values (length *K*).
    w : array-like or None
        Weights (``None`` for uniform).
    period : float
        Period of the circular domain.
    x : array-like or None
        Query points (default: ``0:period-1``).

    Returns
    -------
    np.ndarray
        Mean offset values.

    References
    ----------
    Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of
    rhythmic structure on tapping accuracy. *Attention, Perception,
    & Psychophysics*, 85, 2673–2699.
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

    Returns the circular autocorrelation phase matrix (APM) of the
    weighted multiset *p* within a cycle of length *period* (*p*
    represents pitches or positions). The APM decomposes the circular
    autocorrelation into contributions at each combination of lag
    and phase.

    Parameters
    ----------
    p : array-like of int
        Pitch or position values. Non-negative integers < *period*.
    w : array-like or None
        Weights (``None`` for uniform).
    period : int
        Cycle length.
    decay : float
        Exponential decay rate (default 0, no decay).

    Returns
    -------
    R : np.ndarray
        APM (*period* × *period*).
    r_phase : np.ndarray
        Column sum (1 × *period*). Model of metrical weight.
    r_lag : np.ndarray
        Row sum (*period* × 1). Circular autocorrelation.

    References
    ----------
    Eck, D. (2006). Beat tracking using an autocorrelation phase
    matrix. *Proc. ICMC*.

    Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of
    rhythmic structure on tapping accuracy. *Attention, Perception,
    & Psychophysics*, 85, 2673–2699.
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
        Weights (``None`` for uniform).
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
