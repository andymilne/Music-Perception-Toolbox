"""Integer-position scale-theoretic measures (non-Fourier).

Public entry points (each returns a scalar per multiset of integer
positions on a periodic cycle, using a closed-form soft-equality
kernel rather than a Fourier transform):

* :func:`coherence` --- Carey / Rothenberg propriety quotient.
* :func:`sameness` --- Carey sameness quotient.

Both accept an optional positional-jitter ``sigma`` with a closed-form
Gaussian kernel (Gaussian-CDF for ``coherence``, Gaussian-match for
``sameness``) and the ``sigmaSpace`` flag controlling whether the
variance is per-event-position or per-interval (see USER_GUIDE §6.5).

The Fourier-based scale-structure measures (:func:`balance`,
:func:`evenness`) live in :mod:`._circular.dft`; see USER_GUIDE §6.5
("Scale and rhythm structure") for the user-facing description of
this module and the non-Fourier members of :mod:`._circular.pulse`.
"""
from __future__ import annotations

import numpy as np





# ===================================================================
#  Coherence
# ===================================================================


def coherence(
    p,
    period: float,
    sigma: float = 0.0,
    *,
    strict: bool = True,
    sigma_space: str = "position",
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Coherence quotient of a circular set, optionally smoothed.

    Accepts two input forms, dispatched on ``p``'s shape:

    - 1-D ``p``: single multiset, returns ``(c, nc)``.
    - 2-D ``P`` (shape ``(M, K)``): batched, returns
      ``(c_array, nc_array)`` of length ``M``. Per-row dedup is over
      **permutation, period, and transposition** symmetries via the
      necklace canonical form of the cyclic adjacent intervals.

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
        Pitch or position values (length *K*; or ``(M, K)`` for
        batched). Non-negative; values less than *period*. Must be
        integer when ``sigma == 0``; may be float when ``sigma > 0``.
        Duplicates (modulo *period*) not allowed.
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

    from .._utils import position_variance

    if sigma_space not in ("position", "interval"):
        raise ValueError(
            f"sigma_space must be 'position' or 'interval' "
            f"(got {sigma_space!r})."
        )

    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.ndim == 2:
        return _coherence_batched(
            p_arr, period, sigma, strict=strict, sigma_space=sigma_space,
        )

    p = p_arr.ravel()
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




def _necklace_canonical(p_sorted_mod, period):
    """Necklace canonical form of cyclic adjacent intervals.

    Returns the lexicographically smallest rotation of the cyclic
    sequence of adjacent intervals around the circle. Two pitch
    multisets that are transpositions of each other on the circle
    have identical necklace forms; multisets with different cyclic
    interval structure (including reflections that aren't also
    rotations) have different forms.

    Used by ``coherence`` and ``sameness`` batched dispatch as a
    transposition-invariant cache key.

    Parameters
    ----------
    p_sorted_mod : np.ndarray
        Pitch values, already sorted and reduced modulo *period*.
    period : float
        Period of the circular domain.

    Returns
    -------
    tuple of float
        Canonical interval sequence (length len(p_sorted_mod) for K >= 1,
        empty for K = 0).
    """
    K = len(p_sorted_mod)
    if K == 0:
        return ()
    if K == 1:
        return (float(period),)
    period_f = float(period)
    # Cast to plain floats for stable hashing across numpy dtypes.
    intervals = [
        float(p_sorted_mod[i + 1]) - float(p_sorted_mod[i])
        for i in range(K - 1)
    ]
    intervals.append(period_f - float(p_sorted_mod[-1]) + float(p_sorted_mod[0]))
    intervals = [round(v, 12) for v in intervals]
    best = tuple(intervals)
    for i in range(1, K):
        rot = tuple(intervals[i:] + intervals[:i])
        if rot < best:
            best = rot
    return best




def _coherence_batched(P, period, sigma, *, strict, sigma_space):
    """Batched dispatch for ``coherence``.

    Returns ``(c_array, nc_array)`` of length ``M``. NaN-padded rows
    are dropped per row; rows with no valid pitches give NaN entries.
    Per-row dedup uses the necklace canonical form of the cyclic
    adjacent intervals, which collapses **permutation, period, and
    transposition** symmetries onto a single cached result. (The output
    is fully transposition-invariant on the circle for both
    ``sigma_space='position'`` and ``'interval'``.)
    """
    M, K = P.shape
    c_out = np.full(M, np.nan)
    nc_out = np.full(M, np.nan)
    cache: dict = {}

    for i in range(M):
        p_row = P[i]
        mask = ~np.isnan(p_row)
        p_valid = p_row[mask]
        if len(p_valid) == 0:
            continue
        p_canon = np.sort(np.mod(p_valid, float(period)))
        key = _necklace_canonical(p_canon, period)

        if key in cache:
            c_out[i], nc_out[i] = cache[key]
            continue

        c_i, nc_i = coherence(
            p_valid, period, sigma, strict=strict, sigma_space=sigma_space,
        )
        c_out[i] = c_i
        nc_out[i] = nc_i
        cache[key] = (c_i, nc_i)

    return c_out, nc_out




# ===================================================================
#  Sameness
# ===================================================================


def sameness(
    p,
    period: float,
    sigma: float = 0.0,
    *,
    sigma_space: str = "position",
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Sameness quotient of a circular set, optionally smoothed.

    Accepts two input forms, dispatched on ``p``'s shape:

    - 1-D ``p``: single multiset, returns ``(sq, n_diff)``.
    - 2-D ``P`` (shape ``(M, K)``): batched, returns
      ``(sq_array, n_diff_array)`` of length ``M``. Per-row dedup
      over **permutation, period, and transposition** symmetries
      via the necklace canonical form of the cyclic adjacent
      intervals.

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
    from .._utils import position_variance

    if sigma_space not in ("position", "interval"):
        raise ValueError(
            f"sigma_space must be 'position' or 'interval' "
            f"(got {sigma_space!r})."
        )

    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.ndim == 2:
        return _sameness_batched(
            p_arr, period, sigma, sigma_space=sigma_space,
        )

    p = p_arr.ravel()
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




def _sameness_batched(P, period, sigma, *, sigma_space):
    """Batched dispatch for ``sameness``.

    Per-row dedup uses the necklace canonical form of the cyclic
    adjacent intervals — permutation, period, and transposition
    symmetries are collapsed onto a single cached result.
    """
    M, K = P.shape
    sq_out = np.full(M, np.nan)
    nd_out = np.full(M, np.nan)
    cache: dict = {}

    for i in range(M):
        p_row = P[i]
        mask = ~np.isnan(p_row)
        p_valid = p_row[mask]
        if len(p_valid) == 0:
            continue
        p_canon = np.sort(np.mod(p_valid, float(period)))
        key = _necklace_canonical(p_canon, period)

        if key in cache:
            sq_out[i], nd_out[i] = cache[key]
            continue

        sq_i, nd_i = sameness(
            p_valid, period, sigma, sigma_space=sigma_space,
        )
        sq_out[i] = sq_i
        nd_out[i] = nd_i
        cache[key] = (sq_i, nd_i)

    return sq_out, nd_out