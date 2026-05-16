"""DFT-based circular measures.

This module hosts both the Fourier engine and the measures derived from
it. Public entry points:

* :func:`dft_circular` --- Argand DFT of a (possibly weighted) point
  set on a circle of given period.
* :func:`dft_circular_simulate` --- Monte Carlo estimator of the
  distribution of $|F(k)|$ under independent Gaussian positional
  jitter.
* :func:`balance` --- balance quotient $1 - |F(0)|$ (Monte Carlo
  routed when ``sigma > 0``).
* :func:`evenness` --- evenness quotient $|F(1)|$ (Monte Carlo routed
  when ``sigma > 0``).
* :func:`proj_centroid` --- projection of the $k = 0$ Fourier
  coefficient onto each angular position, with a closed-form mean
  under positional jitter.

The non-public helpers :func:`_fnv1a_32`, :func:`_derive_canonical_seed`,
and :func:`_resolve_base_seed` provide canonical-seed derivation for the
batched Monte Carlo paths.

See USER_GUIDE §6.4 ("Balance and evenness, Fourier-based measures")
for the user-facing description of the scalar measures; the per-position
:func:`proj_centroid` is described in USER_GUIDE §6.5 alongside the
non-Fourier per-position measures in :mod:`._circular.pulse`.
"""
from __future__ import annotations

import numpy as np

from .._utils import validate_weights






# ===================================================================
#  DFT of a circular set
# ===================================================================


def dft_circular(
    p,
    w=None,
    period: float = 1200.0,
) -> tuple[np.ndarray | list[np.ndarray], np.ndarray | list[np.ndarray]]:
    """DFT of a set of points on a circle.

    Accepts two input forms, dispatched on ``p``'s shape:

    - 1-D ``p``: single multiset, returns ``(F, mag)`` (length-K
      arrays each).
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
) -> float | np.ndarray | tuple[float | np.ndarray, float | np.ndarray]:
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
    the deterministic value is recovered exactly.

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
) -> float | np.ndarray | tuple[float | np.ndarray, float | np.ndarray]:
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
    deterministic value is recovered exactly.

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
#  Projected centroid
# ===================================================================


def proj_centroid(
    p,
    w=None,
    period: float = 1200.0,
    x=None,
    sigma: float = 0.0,
) -> tuple[np.ndarray | list[np.ndarray], float | list[float], float | list[float]]:
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

    At ``sigma = 0`` the deterministic value is recovered exactly.

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