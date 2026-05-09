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
    p,
    w=None,
    period: float = 1200.0,
):
    """DFT of a set of points on a circle.

    Accepts two input forms, dispatched on ``p``'s shape:

    - 1-D ``p``: single multiset, returns ``(F, mag)`` (length-K
      arrays each). The v2.0 case.
    - 2-D ``P`` (shape ``(M, K)``): batched, returns
      ``(F_list, mag_list)`` — length-``M`` lists of 1-D arrays
      (lengths can differ across rows after NaN-padded entries are
      dropped). Per-row canonical-form dedup over permutation + period
      symmetries: structurally-identical canonical inputs share one
      cached pair. Transposition dedup is **not** applied — the DFT
      is transposition-equivariant rather than invariant, so
      transposed inputs would need a phase post-transform.

    Parameters
    ----------
    p : array-like
        Pitch-class (or time-class) values (length *K* for 1-D input;
        ``(M, K)`` for batched).
    w : array-like or None
        Weights (``None`` for all ones; same shape as ``p`` or a
        length-``K`` vector broadcast across rows in batched mode).
    period : float
        Period of the circular domain.

    Returns
    -------
    F, mag
        Complex Fourier coefficients and magnitudes. Single arrays in
        scalar mode; lists of arrays in batched mode.
    """
    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.ndim == 2:
        return _dft_circular_batched(p_arr, w, period)

    p_arr = p_arr.ravel()
    K = len(p_arr)
    w_arr = validate_weights(w, K)

    # Sort by pitch class
    idx = np.argsort(p_arr)
    p_arr = p_arr[idx]
    w_arr = w_arr[idx]

    z = w_arr * np.exp(2j * np.pi * p_arr / period)
    F = np.fft.fft(z) / np.sum(w_arr)
    mag = np.abs(F)
    return F, mag


def _dft_circular_batched(P, W, period):
    """Batched dispatch for ``dft_circular``.

    Returns ``(F_list, mag_list)`` — length-``M`` lists of 1-D arrays.
    Per-row dedup uses sorted modular pitches + matching weights as
    the canonical key (permutation + period symmetries; not
    transposition).
    """
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

    F_list: list = [np.array([], dtype=np.complex128) for _ in range(M)]
    mag_list: list = [np.array([], dtype=np.float64) for _ in range(M)]

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

        # Canonical key: sort(mod(p, period)) plus matching weights.
        p_mod = np.mod(p_valid, period)
        sort_idx = np.argsort(p_mod)
        p_sorted = p_mod[sort_idx]
        w_sorted = w_valid[sort_idx]
        key = (
            tuple(np.round(p_sorted, 12).tolist()),
            tuple(np.round(w_sorted, 12).tolist()),
        )

        if key in cache:
            F_list[i], mag_list[i] = cache[key]
            continue

        F_i, mag_i = dft_circular(p_valid, w_valid, period)
        F_list[i] = F_i
        mag_list[i] = mag_i
        cache[key] = (F_i, mag_i)

    return F_list, mag_list


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
    p,
    w=None,
    period: float = 1200.0,
    sigma: float = 0.0,
    *,
    return_std: bool = False,
    n_draws: int = 10000,
    rng_seed: int | None = None,
    rng_scope: str = "canonical",
):
    """Balance of a weighted circular multiset.

    Accepts two input forms, dispatched on ``p``'s shape:

    - 1-D ``p``: single multiset, returns a scalar (or ``(b, b_std)``
      when ``return_std=True``).
    - 2-D ``P`` (shape ``(M, K)``): batched, returns a length-``M``
      array (or pair of arrays). Per-row dedup over permutation +
      period symmetries via sorted-modular canonical key.

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
        Pitch or position values (length *K*; or ``(M, K)`` for batched).
    w : array-like or None
        Weights (``None`` for all ones; same shape as ``p`` or
        length-``K`` broadcast in batched mode).
    period : float
        Period of the circular domain.
    sigma : float
        Positional jitter standard deviation (non-negative; default 0).
    return_std : bool
        If True, also return the standard deviation of ``1 - |F[0]|``
        under jitter (0 when ``sigma == 0``). Default False.
    n_draws : int
        Number of Monte Carlo draws when ``sigma > 0``. Default 10000.
    rng_seed : int or None
        Base RNG seed for reproducibility. In batched mode this is
        the *base* seed from which per-row seeds are derived (see
        ``rng_scope``). If None, a session-random base is generated
        once per batched call so within-call dedup remains
        reproducible.
    rng_scope : {'canonical', 'row'}
        Batched-mode only; ignored in scalar mode. ``'canonical'``
        (default): derive each row's seed from the canonical-form
        key, so transposition-equivalent rows get the same seed and
        the same Monte-Carlo realisation, enabling full dedup.
        ``'row'``: derive each row's seed from the row index, giving
        independent per-row realisations and disabling dedup.

    Returns
    -------
    b : float or np.ndarray
        Balance value(s) in [0, 1].
    b_std : float or np.ndarray
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
    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.ndim == 2:
        return _balance_batched(
            p_arr, w, period, sigma,
            return_std=return_std, n_draws=n_draws,
            rng_seed=rng_seed, rng_scope=rng_scope,
        )

    if sigma == 0:
        _, mag = dft_circular(p_arr, w, period)
        b = float(1 - mag[0])
        return (b, 0.0) if return_std else b
    m, s = dft_circular_simulate(
        p_arr, w, period, sigma, n_draws=n_draws, rng_seed=rng_seed
    )
    b, b_std = float(1 - m[0]), float(s[0])
    return (b, b_std) if return_std else b


# ===================================================================
#  Evenness
# ===================================================================


def evenness(
    p,
    period: float = 1200.0,
    sigma: float = 0.0,
    *,
    return_std: bool = False,
    n_draws: int = 10000,
    rng_seed: int | None = None,
    rng_scope: str = "canonical",
):
    """Evenness of a circular multiset.

    Accepts two input forms, dispatched on ``p``'s shape:

    - 1-D ``p``: single multiset, returns a scalar (or ``(e, e_std)``
      when ``return_std=True``).
    - 2-D ``P`` (shape ``(M, K)``): batched, returns a length-``M``
      array (or pair of arrays). Per-row dedup over permutation +
      period symmetries.

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
        Pitch or position values (length *K*; or ``(M, K)`` for batched).
    period : float
        Period of the circular domain.
    sigma : float
        Positional jitter standard deviation (non-negative; default 0).
    return_std : bool
        If True, also return the standard deviation of ``|F[1]|``
        under jitter (0 when ``sigma == 0``). Default False.
    n_draws : int
        Number of Monte Carlo draws when ``sigma > 0``. Default 10000.
    rng_seed : int or None
        Base RNG seed; in batched mode used as the base from which
        per-row seeds are derived (see ``rng_scope``).
    rng_scope : {'canonical', 'row'}
        Batched-mode only. See :func:`balance` for details.

    Returns
    -------
    e : float or np.ndarray
        Evenness value(s) in [0, 1].
    e_std : float or np.ndarray
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
    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.ndim == 2:
        return _evenness_batched(
            p_arr, period, sigma,
            return_std=return_std, n_draws=n_draws,
            rng_seed=rng_seed, rng_scope=rng_scope,
        )

    if sigma == 0:
        _, mag = dft_circular(p_arr, None, period)
        e = float(mag[1])
        return (e, 0.0) if return_std else e
    m, s = dft_circular_simulate(
        p_arr, None, period, sigma, n_draws=n_draws, rng_seed=rng_seed
    )
    e, e_std = float(m[1]), float(s[1])
    return (e, e_std) if return_std else e


# ===================================================================
#  Monte-Carlo batched dispatch helpers (Tier 4)
# ===================================================================


def _fnv1a_32(data: bytes) -> int:
    """FNV-1a 32-bit hash. Deterministic, no dependencies, identical
    across MATLAB and Python so the same canonical-form key derives
    the same per-row seed in either language."""
    h = 0x811C9DC5  # 2166136261
    for byte in data:
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF  # 16777619
    return h


def _derive_canonical_seed(base_seed: int, key) -> int:
    """Derive a 32-bit seed from a base seed and a canonical-form key."""
    key_bytes = repr(key).encode("utf-8")
    return (base_seed + _fnv1a_32(key_bytes)) & 0xFFFFFFFF


def _resolve_base_seed(rng_seed):
    """Materialise a base seed for batched MC.

    If ``rng_seed`` is given, use it. Otherwise generate a one-shot
    session-random base so that within-call dedup is reproducible
    while across-call results differ.
    """
    if rng_seed is None:
        return int(np.random.default_rng().integers(0, 2**32))
    return int(rng_seed) & 0xFFFFFFFF


def _balance_batched(
    P, W, period, sigma,
    *,
    return_std, n_draws, rng_seed, rng_scope,
):
    """Batched dispatch for ``balance``.

    Per-row dedup uses sorted-modular canonical key (permutation +
    period). For ``sigma > 0``, dedup also requires the seed to be
    deterministic per canonical key — that's the ``rng_scope='canonical'``
    contract. ``rng_scope='row'`` derives per-row seeds from the row
    index and skips caching entirely.
    """
    if rng_scope not in ("canonical", "row"):
        raise ValueError(
            f"rng_scope must be 'canonical' or 'row' (got {rng_scope!r})."
        )

    M, K = P.shape
    use_w = W is not None
    if use_w:
        W_arr = np.asarray(W, dtype=np.float64)
        if W_arr.ndim == 1 and W_arr.size == K:
            W_broadcast, W_full = W_arr, None
        elif W_arr.shape == P.shape:
            W_broadcast, W_full = None, W_arr
        else:
            raise ValueError(
                "W must be None, a matrix the same shape as P, or "
                "a length-K vector broadcast across rows."
            )
    else:
        W_broadcast, W_full = None, None

    base_seed = _resolve_base_seed(rng_seed)
    use_cache = (sigma == 0) or (rng_scope == "canonical")
    cache: dict = {}

    b_out = np.full(M, np.nan)
    bstd_out = np.full(M, np.nan) if return_std else None

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

        # Canonical key over sorted modular pitches + matching weights.
        p_mod = np.mod(p_valid, float(period))
        sort_idx = np.argsort(p_mod)
        p_sorted = p_mod[sort_idx]
        w_sorted = w_valid[sort_idx]
        key = (
            tuple(np.round(p_sorted, 12).tolist()),
            tuple(np.round(w_sorted, 12).tolist()),
        )

        if use_cache and key in cache:
            cached_b, cached_std = cache[key]
            b_out[i] = cached_b
            if bstd_out is not None:
                bstd_out[i] = cached_std
            continue

        # Determine the per-row seed.
        if sigma == 0:
            row_seed = None  # ignored on deterministic path
        elif rng_scope == "canonical":
            row_seed = _derive_canonical_seed(base_seed, key)
        else:
            row_seed = (base_seed + i) & 0xFFFFFFFF

        b_i, bstd_i = balance(
            p_valid, w_valid, period, sigma,
            return_std=True, n_draws=n_draws, rng_seed=row_seed,
        )
        b_out[i] = b_i
        if bstd_out is not None:
            bstd_out[i] = bstd_i
        if use_cache:
            cache[key] = (b_i, bstd_i)

    return (b_out, bstd_out) if return_std else b_out


def _evenness_batched(
    P, period, sigma,
    *,
    return_std, n_draws, rng_seed, rng_scope,
):
    """Batched dispatch for ``evenness`` (no weights)."""
    if rng_scope not in ("canonical", "row"):
        raise ValueError(
            f"rng_scope must be 'canonical' or 'row' (got {rng_scope!r})."
        )

    M, K = P.shape
    base_seed = _resolve_base_seed(rng_seed)
    use_cache = (sigma == 0) or (rng_scope == "canonical")
    cache: dict = {}

    e_out = np.full(M, np.nan)
    estd_out = np.full(M, np.nan) if return_std else None

    for i in range(M):
        p_row = P[i]
        mask = ~np.isnan(p_row)
        p_valid = p_row[mask]
        if len(p_valid) == 0:
            continue

        p_mod = np.mod(p_valid, float(period))
        p_sorted = np.sort(p_mod)
        key = tuple(np.round(p_sorted, 12).tolist())

        if use_cache and key in cache:
            cached_e, cached_std = cache[key]
            e_out[i] = cached_e
            if estd_out is not None:
                estd_out[i] = cached_std
            continue

        if sigma == 0:
            row_seed = None
        elif rng_scope == "canonical":
            row_seed = _derive_canonical_seed(base_seed, key)
        else:
            row_seed = (base_seed + i) & 0xFFFFFFFF

        e_i, estd_i = evenness(
            p_valid, period, sigma,
            return_std=True, n_draws=n_draws, rng_seed=row_seed,
        )
        e_out[i] = e_i
        if estd_out is not None:
            estd_out[i] = estd_i
        if use_cache:
            cache[key] = (e_i, estd_i)

    return (e_out, estd_out) if return_std else e_out


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
):
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

    from ._utils import position_variance

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
):
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
    from ._utils import position_variance

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
):
    """Edge detection on a weighted circular multiset.

    Accepts two input forms, dispatched on ``p``'s shape:

    - 1-D ``p``: single multiset, returns ``(e, e_signed)``. The v2.0
      case.
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
#  Projected centroid
# ===================================================================


def proj_centroid(
    p,
    w=None,
    period: float = 1200.0,
    x=None,
    sigma: float = 0.0,
):
    """Projected centroid of a weighted circular multiset.

    Accepts two input forms, dispatched on ``p``'s shape:

    - 1-D ``p``: single multiset, returns ``(y, cent_mag, cent_phase)``.
    - 2-D ``P`` (shape ``(M, K)``): batched, returns three length-``M``
      lists. Per-row dedup over permutation + period symmetries.

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
    sigma : float
        Positional jitter standard deviation (non-negative; default 0).

    Returns
    -------
    y, cent_mag, cent_phase
        In scalar mode: a length-``len(x)`` array, plus two floats.
        In batched mode: three length-``M`` lists.

    References
    ----------
    Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of
    rhythmic structure on tapping accuracy. *Attention, Perception,
    & Psychophysics*, 85, 2673–2699.
    """
    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.ndim == 2:
        return _proj_centroid_batched(p_arr, w, period, x, sigma)

    if x is None:
        x = np.arange(period)
    x = np.asarray(x, dtype=np.float64).ravel()
    sigma = float(sigma)
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    F, _ = dft_circular(p_arr.ravel(), w, period)
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


def _proj_centroid_batched(P, W, period, x, sigma):
    """Batched dispatch for ``proj_centroid``."""
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

    y_list = [np.array([], dtype=np.float64) for _ in range(M)]
    cm_list: list = [None] * M
    cp_list: list = [None] * M
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
            y_list[i], cm_list[i], cp_list[i] = cache[key]
            continue

        y_i, cm_i, cp_i = proj_centroid(p_valid, w_valid, period, x, sigma)
        y_list[i] = y_i
        cm_list[i] = cm_i
        cp_list[i] = cp_i
        cache[key] = (y_i, cm_i, cp_i)

    return y_list, cm_list, cp_list


# ===================================================================
#  Mean offset
# ===================================================================


def mean_offset(
    p,
    w=None,
    period: float = 1200.0,
    x=None,
):
    """Mean offset (net upward arc) of a weighted circular multiset.

    Accepts two input forms, dispatched on ``p``'s shape:

    - 1-D ``p``: single multiset, returns a length-``len(x)`` array
      (or length-``period`` if ``x`` is None). The v2.0 case.
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
):
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
