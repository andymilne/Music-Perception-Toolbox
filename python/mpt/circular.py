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
        Weights (``None`` for all ones).
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


def dft_circular_simulate(
    p: np.ndarray,
    w: np.ndarray | None,
    period: float,
    sigma: float,
    *,
    n_draws: int = 10000,
    rng_seed: int | None = None,
    return_samples: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monte Carlo Argand DFT under positional jitter.

    Estimates the distribution of Argand-DFT coefficient magnitudes
    when each event position is independently perturbed by Gaussian
    noise:

        P_k = (p_k + eta_k) mod period,  eta_k ~ N(0, sigma**2)

    For each draw, the perturbed positions are sorted (a perceptual
    re-ordering at the level of the listener: events identified only
    by their sorted position on the cycle, not by which underlying
    event they came from), the resulting Argand vector is formed
    ``z_j = w_j * exp(2j * pi * P_j / period)``, and the DFT is
    computed.

    Returns the per-coefficient mean and standard deviation of the
    magnitudes ``|F[k]|`` for ``k = 0, 1, ..., K-1``, taken across
    draws. Optionally also returns the full ``(n_draws, K)`` sample
    matrix.

    For ``sigma -> 0`` the mean magnitudes converge to the
    deterministic Argand-DFT magnitudes from :func:`dft_circular`,
    and the standard deviations converge to zero. Resort effects are
    negligible while ``sigma`` is small relative to the smallest
    event-to-event gap; beyond that, resort is captured by sorting
    each draw.

    Parameters
    ----------
    p : array-like
        Event positions (length *K*).
    w : array-like or None
        Weights (length *K*, scalar, or ``None`` for all ones).
    period : float
        Period of the circular domain.
    sigma : float
        Positional jitter standard deviation (non-negative; same
        units as *p* and *period*).
    n_draws : int
        Number of Monte Carlo draws. Default 10000.
    rng_seed : int or None
        Seed for the random number generator. ``None`` (default)
        uses fresh entropy; pass an integer for reproducibility.
    return_samples : bool
        If True, also return the ``(n_draws, K)`` sample matrix.

    Returns
    -------
    mag_mean : np.ndarray
        Mean magnitudes ``E[|F[k]|]`` (length *K*).
    mag_std : np.ndarray
        Magnitude standard deviations (length *K*).
    mags : np.ndarray
        ``(n_draws, K)`` matrix of magnitude samples (only when
        ``return_samples`` is True).

    See Also
    --------
    dft_circular, balance, evenness, proj_centroid

    Examples
    --------
    >>> # Son clave under sigma = 1/8 of a pulse
    >>> p = [0, 3, 6, 10, 12]
    >>> m, s = dft_circular_simulate(p, None, 16, 1/8, rng_seed=42)
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    K = len(p)
    w = validate_weights(w, K)
    sigma = float(sigma)

    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    rng = np.random.default_rng(rng_seed)
    eta = sigma * rng.standard_normal((K, n_draws))
    P = (p[:, None] + eta) % period

    if np.allclose(w, w[0]):
        # Uniform weights: just sort P, weights don't change
        P_sorted = np.sort(P, axis=0)
        z = w[0] * np.exp(2j * np.pi * P_sorted / period)
        sum_w = w[0] * K
    else:
        sortIdx = np.argsort(P, axis=0)
        P_sorted = np.take_along_axis(P, sortIdx, axis=0)
        w_matrix = np.take_along_axis(
            w[:, None] * np.ones_like(P), sortIdx, axis=0
        )
        z = w_matrix * np.exp(2j * np.pi * P_sorted / period)
        sum_w = float(w.sum())

    F = np.fft.fft(z, axis=0) / sum_w
    mags = np.abs(F)
    mag_mean = mags.mean(axis=1)
    mag_std = mags.std(axis=1, ddof=1)

    if return_samples:
        return mag_mean, mag_std, mags.T
    return mag_mean, mag_std


# ===================================================================
#  Balance
# ===================================================================


def balance(
    p: np.ndarray,
    w: np.ndarray | None,
    period: float,
    sigma: float = 0.0,
    *,
    return_std: bool = False,
    n_draws: int = 10000,
    rng_seed: int | None = None,
) -> float | tuple[float, float]:
    """Balance of a weighted circular multiset.

    Computes the balance of a weighted multiset of points on a circle
    (*p* represents pitches or positions), defined as ``1 - |F[0]|``
    where ``F[0]`` is the k = 0 DFT coefficient (see
    :func:`dft_circular`).

    Balance ranges from 0 to 1: 1 = perfectly balanced (centre of
    gravity at the origin); 0 = all weight at one point.

    With ``sigma > 0``, returns the *expected* balance under
    independent Gaussian positional jitter on each event, estimated
    by Monte Carlo via :func:`dft_circular_simulate`. For ``sigma = 0``
    the deterministic v2.0 value is recovered exactly.

    Parameters
    ----------
    p : array-like
        Pitch or position values (length *K*).
    w : array-like or None
        Weights (``None`` for all ones).
    period : float
        Period of the circular domain.
    sigma : float
        Positional jitter standard deviation (non-negative; default 0).
    return_std : bool
        If True, also return the standard deviation of
        ``1 - |F[0]|`` under jitter (0 when ``sigma == 0``). Default
        False (scalar return for backward compatibility with v2.0).
    n_draws : int
        Number of Monte Carlo draws when ``sigma > 0``. Default 10000.
    rng_seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    b : float
        Balance (mean under jitter, in [0, 1]).
    b_std : float
        Standard deviation, returned only when ``return_std=True``.

    References
    ----------
    Milne, A. J., Bulger, D., & Herff, S. A. (2017). Exploring the
    space of perfectly balanced rhythms and scales. *Journal of
    Mathematics and Music*, 11(2–3), 101–133.

    Milne, A. J. & Herff, S. A. (2020). The perceptual relevance of
    balance, evenness, and entropy in musical rhythms. *Cognition*,
    203, 104233.
    """
    if sigma == 0:
        _, mag = dft_circular(p, w, period)
        b = float(1 - mag[0])
        return (b, 0.0) if return_std else b
    m, s = dft_circular_simulate(
        p, w, period, sigma, n_draws=n_draws, rng_seed=rng_seed
    )
    b, b_std = float(1 - m[0]), float(s[0])
    return (b, b_std) if return_std else b


# ===================================================================
#  Evenness
# ===================================================================


def evenness(
    p: np.ndarray,
    period: float,
    sigma: float = 0.0,
    *,
    return_std: bool = False,
    n_draws: int = 10000,
    rng_seed: int | None = None,
) -> float | tuple[float, float]:
    """Evenness of a circular multiset.

    Computes the evenness of a multiset of *K* points on a circle
    (*p* represents pitches or positions), defined as ``|F[1]|``
    where ``F[1]`` is the k = 1 DFT coefficient (see
    :func:`dft_circular`).

    Evenness ranges from 0 to 1: 1 = maximally even (equally spaced);
    0 = maximally uneven. Uses uniform weights regardless of input.

    With ``sigma > 0``, returns the *expected* evenness under
    independent Gaussian positional jitter on each event, estimated
    by Monte Carlo via :func:`dft_circular_simulate`. The perturbed
    positions are sorted before computing the DFT, capturing the
    perceptual reordering that occurs when noise is comparable to
    the smallest event-to-event gap. For ``sigma = 0`` the
    deterministic v2.0 value is recovered exactly.

    Parameters
    ----------
    p : array-like
        Pitch or position values (length *K*).
    period : float
        Period of the circular domain.
    sigma : float
        Positional jitter standard deviation (non-negative; default 0).
    return_std : bool
        If True, also return the standard deviation of ``|F[1]|``
        under jitter (0 when ``sigma == 0``). Default False (scalar
        return for backward compatibility with v2.0).
    n_draws : int
        Number of Monte Carlo draws when ``sigma > 0``. Default 10000.
    rng_seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    e : float
        Evenness (mean under jitter, in [0, 1]).
    e_std : float
        Standard deviation, returned only when ``return_std=True``.

    References
    ----------
    Milne, A. J., Bulger, D., & Herff, S. A. (2017). Exploring the
    space of perfectly balanced rhythms and scales. *Journal of
    Mathematics and Music*, 11(2–3), 101–133.

    Milne, A. J. & Herff, S. A. (2020). The perceptual relevance of
    balance, evenness, and entropy in musical rhythms. *Cognition*,
    203, 104233.
    """
    if sigma == 0:
        _, mag = dft_circular(p, None, period)
        e = float(mag[1])
        return (e, 0.0) if return_std else e
    m, s = dft_circular_simulate(
        p, None, period, sigma, n_draws=n_draws, rng_seed=rng_seed
    )
    e, e_std = float(m[1]), float(s[1])
    return (e, e_std) if return_std else e


# ===================================================================
#  Coherence
# ===================================================================


def coherence(
    p: np.ndarray,
    period: float,
    sigma: float = 0.0,
    *,
    strict: bool = True,
    sigma_space: str = "position",
) -> tuple[float, float]:
    """Coherence quotient of a circular set, optionally smoothed.

    Returns the coherence quotient of the set of pitches or positions
    *p* within an equal division of size *period* (Carey, 2002). A
    coherence failure occurs when a pair with a larger generic span
    does not have a strictly greater specific size (strict propriety).

    With ``sigma > 0``, each indicator ``[d2 <= d1]`` is replaced by
    the Gaussian-CDF probability ``P(D2 <= D1)`` under jitter, which
    smoothly interpolates between 0 and 1 as the means cross.
    ``p`` and ``period`` may be float when ``sigma > 0``.

    Parameters
    ----------
    p : array-like
        Pitch or position values. Non-negative; values less than
        *period*. Must be integer when ``sigma == 0``; may be float
        when ``sigma > 0``. Duplicates (modulo *period*) not allowed.
    period : float
        Size of the equal division. Must be integer when
        ``sigma == 0``.
    sigma : float
        Positional or interval uncertainty (default 0). In the same
        units as *p* and *period*.
    strict : bool
        Default True. Controls tie handling at ``sigma == 0`` only.
        If True, ties (a larger generic span with equal specific
        size) count as failures (Rothenberg's strict propriety). If
        False, only strictly smaller specific sizes count
        (Rothenberg's propriety). At ``sigma > 0`` ties have measure
        zero and this flag has no effect; the soft path uses
        ``P(D2 <= D1)``, which assigns 0.5 to ties as a natural
        limiting case.
    sigma_space : {'position', 'interval'}
        How sigma is interpreted (default 'position'). 'position'
        treats sigma as positional uncertainty on each ``p_k``,
        propagated through index sharing among interval pairs.
        'interval' treats sigma as independent uncertainty per
        derived interval (``V = 2 * sigma**2`` uniformly). At
        ``sigma == 0`` the two flags coincide.

    Returns
    -------
    c : float
        Coherence quotient. ``[0, 1]`` when ``sigma == 0``; may go
        below 0 at large ``sigma`` when soft failures exceed maxNC.
    nc : float
        Number of coherence failures (integer when ``sigma == 0``).

    References
    ----------
    Carey, N. (2002). On coherence and sameness. *Journal of Music
    Theory*, 46(1/2), 1–56.

    Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of
    rhythmic structure on tapping accuracy. *Attention, Perception,
    & Psychophysics*, 85, 2673–2699.

    Rothenberg, D. (1978). A model for pattern perception with
    musical applications. Part I. *Mathematical Systems Theory*,
    11, 199–234.
    """
    from math import erfc, sqrt

    from ._utils import position_variance

    if sigma_space not in ("position", "interval"):
        raise ValueError(
            f"sigma_space must be 'position' or 'interval' "
            f"(got {sigma_space!r})."
        )

    p = np.asarray(p, dtype=np.float64).ravel()
    period = float(period)
    sigma = float(sigma)
    p = np.sort(p % period)
    K = len(p)
    if len(np.unique(p)) != K:
        raise ValueError("p must not contain duplicate values (mod period).")
    if K < 2:
        raise ValueError(f"At least 2 events required (got {K}).")

    if sigma == 0.0:
        if not np.all(np.abs(p - np.round(p)) == 0):
            raise ValueError(
                "For sigma == 0, p must contain integers. "
                "Use sigma > 0 for non-integer positions."
            )
        if abs(period - round(period)) != 0:
            raise ValueError(
                f"For sigma == 0, period must be integer (got {period})."
            )

    # Interval-size table with index provenance
    size_span = np.zeros((K - 1, K), dtype=np.float64)
    src_from = np.zeros((K - 1, K), dtype=np.int64)
    src_to = np.zeros((K - 1, K), dtype=np.int64)
    for g in range(1, K):
        to_idx = (np.arange(K) + g) % K
        size_span[g - 1] = (p[to_idx] - p) % period
        src_from[g - 1] = np.arange(K)
        src_to[g - 1] = to_idx

    if sigma == 0.0:
        # Discrete v2 count, preserves strict flag exactly
        nc = 0.0
        for g2 in range(2, K):
            sizes2 = size_span[g2 - 1]
            for g1 in range(1, g2):
                sizes1 = size_span[g1 - 1]
                diffs = sizes2[:, None] - sizes1[None, :]
                if strict:
                    nc += float(np.sum(diffs <= 0))
                else:
                    nc += float(np.sum(diffs < 0))
    else:
        # Soft count under sigma jitter
        use_position = sigma_space == "position"
        nc = 0.0
        for g2 in range(2, K):
            for g1 in range(1, g2):
                for i in range(K):
                    for j in range(K):
                        delta = size_span[g2 - 1, i] - size_span[g1 - 1, j]
                        if use_position:
                            V = position_variance(
                                [src_to[g2 - 1, i], src_from[g2 - 1, i],
                                 src_to[g1 - 1, j], src_from[g1 - 1, j]],
                                [+1, -1, -1, +1], sigma)
                        else:
                            V = 2.0 * sigma**2
                        if V == 0.0:
                            if delta < 0:
                                nc += 1.0
                            elif delta == 0:
                                nc += 0.5
                        else:
                            # Standard normal CDF via erfc:
                            #   Phi(z) = 0.5 * erfc(-z / sqrt(2))
                            nc += 0.5 * erfc(delta / sqrt(2.0 * V))

    max_nc = K * (K - 1) * (K - 2) * (3 * K - 5) / 24
    c = 1.0 - nc / max_nc
    return c, nc


# ===================================================================
#  Sameness
# ===================================================================


def sameness(
    p: np.ndarray,
    period: float,
    sigma: float = 0.0,
    *,
    sigma_space: str = "position",
) -> tuple[float, float]:
    """Sameness quotient of a circular set, optionally smoothed.

    Returns the sameness quotient of the set of pitches or positions
    *p* within an equal division of size *period* (Carey, 2002). An
    ambiguity occurs when two different generic spans share the same
    specific size.

    With ``sigma > 0``, the strict equality test is replaced by a
    Gaussian match kernel ``exp(-(d1 - d2)**2 / (2 V))``, where ``V``
    is the variance of the difference under the chosen jitter model.
    The kernel equals 1 when intervals coincide, decays smoothly as
    the gap grows, and recovers the discrete indicator at
    ``sigma == 0``. ``p`` and ``period`` may be float when
    ``sigma > 0``.

    Parameters
    ----------
    p : array-like
        Pitch or position values. Non-negative; values less than
        *period*. Must be integer when ``sigma == 0``; may be float
        when ``sigma > 0``. Duplicates (modulo *period*) not allowed.
    period : float
        Size of the equal division. Must be integer when
        ``sigma == 0``.
    sigma : float
        Positional or interval uncertainty (default 0). In the same
        units as *p* and *period*.
    sigma_space : {'position', 'interval'}
        How sigma is interpreted (default 'position'). 'position'
        treats sigma as positional uncertainty on each ``p_k``,
        propagated through index sharing among interval pairs.
        'interval' treats sigma as independent uncertainty per
        derived interval (``V = 2 * sigma**2`` uniformly). At
        ``sigma == 0`` the two flags coincide.

    Returns
    -------
    sq : float
        Sameness quotient. ``[0, 1]`` when ``sigma == 0``; may go
        below 0 at large ``sigma`` when soft matches exceed maxDiff.
    n_diff : float
        Number of ambiguities (integer when ``sigma == 0``).

    References
    ----------
    Carey, N. (2002). On coherence and sameness. *Journal of Music
    Theory*, 46(1/2), 1–56.

    Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of
    rhythmic structure on tapping accuracy. *Attention, Perception,
    & Psychophysics*, 85, 2673–2699.
    """
    from ._utils import position_variance

    if sigma_space not in ("position", "interval"):
        raise ValueError(
            f"sigma_space must be 'position' or 'interval' "
            f"(got {sigma_space!r})."
        )

    p = np.asarray(p, dtype=np.float64).ravel()
    period = float(period)
    sigma = float(sigma)
    p = np.sort(p % period)
    K = len(p)
    if len(np.unique(p)) != K:
        raise ValueError("p must not contain duplicate values (mod period).")
    if K < 2:
        raise ValueError(f"At least 2 events required (got {K}).")

    # Interval-size table with index provenance
    size_span = np.zeros((K - 1, K), dtype=np.float64)
    src_from = np.zeros((K - 1, K), dtype=np.int64)
    src_to = np.zeros((K - 1, K), dtype=np.int64)
    for g in range(1, K):
        to_idx = (np.arange(K) + g) % K
        size_span[g - 1] = (p[to_idx] - p) % period
        src_from[g - 1] = np.arange(K)
        src_to[g - 1] = to_idx

    if sigma == 0.0:
        # Discrete v2 path
        if not np.all(np.abs(p - np.round(p)) == 0):
            raise ValueError(
                "For sigma == 0, p must contain integers. "
                "Use sigma > 0 for non-integer positions."
            )
        if abs(period - round(period)) != 0:
            raise ValueError(
                f"For sigma == 0, period must be integer (got {period})."
            )
        period_int = int(round(period))
        size_counts = np.zeros((K - 1, period_int), dtype=np.int64)
        for g in range(K - 1):
            for k in range(K):
                s = int(round(size_span[g, k]))
                size_counts[g, s] += 1
        col_totals = np.sum(size_counts, axis=0)
        n_diff = float(
            (np.sum(col_totals**2) - np.sum(size_counts**2)) / 2
        )
    else:
        # Soft count under sigma jitter
        use_position = sigma_space == "position"
        n_diff = 0.0
        for g1 in range(K - 1):
            for g2 in range(K - 1):
                if g1 == g2:
                    continue
                for k in range(K):
                    for l in range(K):
                        # Straight (unwrapped) size difference: sizes
                        # are absolute magnitudes in [0, period).
                        dx = size_span[g1, k] - size_span[g2, l]
                        if use_position:
                            V = position_variance(
                                [src_to[g1, k], src_from[g1, k],
                                 src_to[g2, l], src_from[g2, l]],
                                [+1, -1, -1, +1], sigma)
                        else:
                            V = 2.0 * sigma**2
                        if V == 0.0:
                            if dx == 0:
                                n_diff += 1.0
                        else:
                            n_diff += np.exp(-dx**2 / (2.0 * V))
        n_diff /= 2.0  # unordered pairs of generic spans

    max_diff = K * (K - 1) ** 2 / 2
    sq = 1.0 - n_diff / max_diff
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
        Weights (``None`` for all ones).
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
    sigma: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """Projected centroid of a weighted circular multiset.

    Computes the projection of the circular centroid (centre of
    gravity) onto each angular position. The centroid is the k = 0
    Fourier coefficient of the multiset (see :func:`dft_circular`).

    With ``sigma > 0``, returns the *expected* projection under
    independent Gaussian positional jitter on each event. Because
    ``y(x)`` is linear in ``F[0]`` and ``F[0]`` is permutation-
    invariant, the result has a clean closed form::

        E[y(x)] = alpha_1 * y_deterministic(x)

    where ``alpha_1 = exp(-2 * pi**2 * sigma**2 / period**2)``. No
    Monte Carlo is needed; the deterministic projection is simply
    damped by the kernel-smoothing factor ``alpha_1``. Phase is
    preserved in expectation, so ``cent_phase`` is unchanged from
    the deterministic case. ``cent_mag`` returns
    ``alpha_1 * |F[0]|`` — the magnitude of the *expected* centroid,
    consistent with the projection. (For ``E[|F[0]|]``, the average
    centroid magnitude under jitter — a different scalar that picks
    up positive bias from the Rayleigh-style geometry — call
    :func:`balance` with ``sigma > 0`` and read off ``1 - b``.)

    At ``sigma = 0`` the v2.0 deterministic value is recovered exactly.

    Parameters
    ----------
    p : array-like
        Pitch or position values (length *K*).
    w : array-like or None
        Weights (``None`` for all ones).
    period : float
        Period of the circular domain.
    x : array-like or None
        Query points (default: ``0:period-1``).
    sigma : float
        Positional jitter standard deviation (non-negative; default 0).

    Returns
    -------
    y : np.ndarray
        Mean projected centroid values.
    cent_mag : float
        Centroid magnitude scaled by ``alpha_1`` when ``sigma > 0``.
    cent_phase : float
        Centroid phase, unchanged from the deterministic case.

    References
    ----------
    Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of
    rhythmic structure on tapping accuracy. *Attention, Perception,
    & Psychophysics*, 85, 2673–2699.
    """
    if x is None:
        x = np.arange(period)
    x = np.asarray(x, dtype=np.float64).ravel()
    sigma = float(sigma)
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    F, _ = dft_circular(p, w, period)
    F0 = F[0]
    if sigma > 0:
        alpha1 = np.exp(-2 * np.pi**2 * sigma**2 / period**2)
        F0 = alpha1 * F0

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
        Weights (``None`` for all ones).
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
        Weights (``None`` for all ones).
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
