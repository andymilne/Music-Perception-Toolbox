"""Expectation tensor construction, evaluation, and similarity.

The central data structure is :class:`ExpTensDensity`, a precomputed
Gaussian-mixture density representing the expected distribution of
r-ads (ordered r-tuples) from a weighted pitch multiset.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import permutations
from math import factorial

import numpy as np
from scipy.special import comb as _comb

from ._utils import estimate_comp_time, validate_weights
from .spectra import add_spectra


def _nchoosek_indices(n: int, r: int) -> np.ndarray:
    """Return all r-combinations of range(n) as an (r, C(n,r)) array."""
    from itertools import combinations

    combos = list(combinations(range(n), r))
    return np.array(combos, dtype=np.intp).T  # r x C(n,r)


# -------------------------------------------------------------------
#  Data class
# -------------------------------------------------------------------


@dataclass
class ExpTensDensity:
    """Precomputed expectation tensor density.

    Attributes match the MATLAB struct fields; see :func:`build_exp_tens`.
    """

    p: np.ndarray
    w: np.ndarray
    sigma: float
    r: int
    is_rel: bool
    is_per: bool
    period: float
    dim: int

    # For eval_exp_tens
    centres: np.ndarray  # dim x nJ
    w_j: np.ndarray  # (nJ,)
    n_j: int

    # For cos_sim_exp_tens
    u_perm: np.ndarray  # r x nJ_perm
    w_perm: np.ndarray  # (nJ_perm,)
    n_j_perm: int
    v_comb: np.ndarray  # r x nK
    wv_comb: np.ndarray  # (nK,)
    n_k: int


# -------------------------------------------------------------------
#  build_exp_tens
# -------------------------------------------------------------------


def build_exp_tens(
    p: np.ndarray,
    w: np.ndarray | None,
    sigma: float,
    r: int,
    is_rel: bool,
    is_per: bool,
    period: float,
    *,
    verbose: bool = True,
) -> ExpTensDensity:
    """Precompute an r-ad expectation tensor density object.

    Precomputes the tuple index sets, pitch or position matrices,
    weight vectors, and (for the relative case) reduced interval
    centres for the weighted multiset (*p*, *w*). The returned
    object can be passed to :func:`eval_exp_tens` and
    :func:`cos_sim_exp_tens` in place of the raw arguments, avoiding
    redundant recomputation across multiple calls.

    Parameters
    ----------
    p : array-like
        Pitch or position values.
    w : array-like or None
        Weights (``None`` or empty for uniform).
    sigma : float
        Gaussian kernel standard deviation.
    r : int
        Tuple size (≥ 2 when *is_rel* is True).
    is_rel : bool
        Transposition-invariant (relative) quadratic form.
    is_per : bool
        Periodic wrapping to ``[-period/2, period/2)``.
    period : float
        Period for wrapping.
    verbose : bool
        Print progress information.

    Returns
    -------
    ExpTensDensity
        Precomputed density object.

    References
    ----------
    Milne, A. J., Sethares, W. A., Laney, R., & Sharp, D. B.
    (2011). Modelling the similarity of pitch collections with
    expectation tensors. *Journal of Mathematics and Music*,
    5(1), 1–20.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    w = validate_weights(w, len(p))

    r = int(r)
    if r < 1 or r != int(r):
        raise ValueError("r must be a positive integer.")
    if r > len(p):
        raise ValueError("r must not exceed the number of pitches.")
    if is_rel and r < 2:
        raise ValueError("For relative densities, r must be at least 2.")

    dim = r - int(is_rel)
    n = len(p)

    n_perms = factorial(r)
    n_combs = int(_comb(n, r, exact=True))
    n_j = n_perms * n_combs
    n_k = n_combs

    if verbose:
        print(
            f"build_exp_tens: building {n_j} ordered {r}-tuples "
            f"from {n} pitches."
        )

    # All r-combinations (r x C(n,r))
    nck = _nchoosek_indices(n, r)

    # All permutations of range(r) — each column is one permutation
    all_perms = np.array(list(permutations(range(r))), dtype=np.intp).T  # r x r!

    # Build ordered r-tuples (perm side)
    j_idx = np.empty((r, n_j), dtype=np.intp)
    offset = 0
    for i in range(n_perms):
        j_idx[:, offset : offset + n_combs] = nck[all_perms[:, i], :]
        offset += n_combs

    u_perm = p[j_idx]  # r x nJ
    w_perm = np.prod(w[j_idx], axis=0)  # (nJ,)

    # r-combinations (comb side, for cos_sim)
    v_comb = p[nck]  # r x nK
    wv_comb = np.prod(w[nck], axis=0)  # (nK,)

    # Reduce to interval centres if relative
    if is_rel:
        centres = u_perm[1:, :] - u_perm[0, :]  # (r-1) x nJ
    else:
        centres = u_perm.copy()  # r x nJ

    return ExpTensDensity(
        p=p,
        w=w,
        sigma=sigma,
        r=r,
        is_rel=is_rel,
        is_per=is_per,
        period=period,
        dim=dim,
        centres=centres,
        w_j=w_perm,
        n_j=n_j,
        u_perm=u_perm,
        w_perm=w_perm,
        n_j_perm=n_j,
        v_comb=v_comb,
        wv_comb=wv_comb,
        n_k=n_k,
    )


# -------------------------------------------------------------------
#  eval_exp_tens
# -------------------------------------------------------------------


def eval_exp_tens(
    dens: ExpTensDensity,
    x: np.ndarray,
    normalize: str = "none",
    *,
    verbose: bool = True,
) -> np.ndarray:
    """Evaluate an expectation tensor density at query points.

    The density at a query point *x* is a weighted sum of Gaussian
    kernels centred at all ordered r-tuples drawn from (*p*, *w*).
    The cosine similarity is computed analytically — no grid
    evaluation is required.

    When *is_rel* is False (absolute), query points are
    r-dimensional pitch or position vectors. When *is_rel* is True
    (relative / transposition-invariant), the effective
    dimensionality is r − 1 and query points are (r−1)-dimensional
    interval vectors.

    Parameters
    ----------
    dens : ExpTensDensity
        Precomputed density from :func:`build_exp_tens`.
    x : (dim, nQ) array
        Query points (dim = r − is_rel). Each column is one point.
    normalize : str
        ``'none'`` (default), ``'gaussian'``, or ``'pdf'``.
    verbose : bool
        Print progress.

    Returns
    -------
    np.ndarray
        (nQ,) density values.

    See Also
    --------
    build_exp_tens, cos_sim_exp_tens
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    centres = dens.centres
    w_j = dens.w_j
    n_j = dens.n_j
    sigma = dens.sigma
    r = dens.r
    dim = dens.dim
    is_rel = dens.is_rel
    is_per = dens.is_per
    period = dens.period

    if x.shape[0] != dim:
        raise ValueError(
            f"x must have {dim} rows (each column is a {dim}-D query point)."
        )
    n_q = x.shape[1]

    n_pairs = int(n_j) * int(n_q)
    estimate_comp_time(n_pairs, dim, "eval_exp_tens", verbose)

    vals = _eval_core(centres, w_j, n_j, x, n_q, dim, sigma, r, is_rel, is_per, period)

    # Normalization
    if normalize != "none":
        det_m = (1.0 / r) if is_rel else 1.0
        gauss_const = (2 * np.pi * sigma**2) ** (-dim / 2) * np.sqrt(det_m)
        vals = vals * gauss_const

        if normalize == "pdf":
            sum_w = np.sum(w_j)
            if sum_w > 0:
                vals = vals / sum_w
            else:
                warnings.warn(
                    "Sum of weight products is zero; cannot normalize to pdf."
                )

    return vals


def _eval_core(
    centres, w_j, n_j, x, n_q, dim, sigma, r, is_rel, is_per, period
):
    """Evaluate with automatic memory-aware chunking."""
    bytes_needed = (dim + 1) * int(n_j) * int(n_q) * 8
    mem_limit = 4_000_000_000  # 4 GB default

    if bytes_needed <= mem_limit:
        return _eval_full(centres, w_j, n_j, x, n_q, dim, sigma, r, is_rel, is_per, period)

    chunk_size = max(1, int(mem_limit / ((dim + 1) * int(n_j) * 8)))
    vals = np.zeros(n_q)
    for c_start in range(0, n_q, chunk_size):
        c_end = min(c_start + chunk_size, n_q)
        idx = slice(c_start, c_end)
        n_qc = c_end - c_start
        vals[idx] = _eval_full(
            centres, w_j, n_j, x[:, idx], n_qc, dim, sigma, r, is_rel, is_per, period
        )
    return vals


def _eval_full(centres, w_j, n_j, x_q, n_qc, dim, sigma, r, is_rel, is_per, period):
    """Fully vectorized density evaluation."""
    # D shape: (dim, nJ, nQc)
    D = centres[:, :, None] - x_q[:, None, :]

    if is_per:
        D = np.mod(D + period / 2, period) - period / 2

    if is_rel:
        Q = np.sum(D**2, axis=0) - np.sum(D, axis=0) ** 2 / r
    else:
        Q = np.sum(D**2, axis=0)

    # E shape: (nJ, nQc)
    E = np.exp(-Q / (2 * sigma**2))

    # Weighted sum: (nJ,) @ (nJ, nQc) → (nQc,)
    return w_j @ E


# -------------------------------------------------------------------
#  eval_exp_tens from raw args (convenience)
# -------------------------------------------------------------------


def eval_exp_tens_raw(
    p: np.ndarray,
    w: np.ndarray | None,
    sigma: float,
    r: int,
    is_rel: bool,
    is_per: bool,
    period: float,
    x: np.ndarray,
    normalize: str = "none",
    *,
    verbose: bool = True,
) -> np.ndarray:
    """Build and evaluate in one call. See :func:`eval_exp_tens`."""
    dens = build_exp_tens(p, w, sigma, r, is_rel, is_per, period, verbose=verbose)
    return eval_exp_tens(dens, x, normalize, verbose=verbose)


# -------------------------------------------------------------------
#  cos_sim_exp_tens
# -------------------------------------------------------------------


def cos_sim_exp_tens(
    dens_x: ExpTensDensity,
    dens_y: ExpTensDensity,
    *,
    verbose: bool = True,
) -> float:
    """Cosine similarity of two r-ad expectation tensor densities.

    Computes the cosine similarity between the r-ad expectation
    tensor densities of two precomputed weighted multisets. The
    similarity is computed analytically — no grid evaluation is
    required.

    Four variants are available depending on *is_per* and *is_rel*.

    Parameters
    ----------
    dens_x, dens_y : ExpTensDensity
        Precomputed density objects from :func:`build_exp_tens`.
        Must share the same *r*, *sigma*, *is_rel*, *is_per*, and
        (if periodic) *period*.
    verbose : bool
        Print progress.

    Returns
    -------
    float
        Cosine similarity in [0, 1] for non-negative weights.
        NaN if *r* exceeds the number of elements in either multiset.

    References
    ----------
    Originally by David Bulger, Macquarie University (2016).
    Adapted for the Music Perception Toolbox v2 by Andrew J. Milne.

    See Also
    --------
    build_exp_tens, eval_exp_tens, batch_cos_sim_exp_tens
    """
    if dens_x.r != dens_y.r:
        raise ValueError("Both densities must have the same r.")
    if dens_x.is_rel != dens_y.is_rel:
        raise ValueError("Both densities must have the same is_rel.")
    if dens_x.is_per != dens_y.is_per:
        raise ValueError("Both densities must have the same is_per.")
    if dens_x.is_per and dens_x.period != dens_y.period:
        raise ValueError("Both densities must have the same period.")
    if dens_x.sigma != dens_y.sigma:
        raise ValueError("Both densities must have the same sigma.")

    r = dens_x.r
    sigma = dens_x.sigma
    is_rel = dens_x.is_rel
    is_per = dens_x.is_per
    period = dens_x.period

    # Early return for degenerate case
    if r > min(len(dens_x.p), len(dens_y.p)):
        return float("nan")

    n_jx, n_kx = dens_x.n_j_perm, dens_x.n_k
    n_jy, n_ky = dens_y.n_j_perm, dens_y.n_k

    total_pairs = n_jx * n_ky + n_jx * n_kx + n_jy * n_ky
    estimate_comp_time(total_pairs, r, "cos_sim_exp_tens", verbose)

    ip_xy = _ip_core(
        dens_x.u_perm, dens_x.w_perm, n_jx,
        dens_y.v_comb, dens_y.wv_comb, n_ky,
        r, sigma, is_rel, is_per, period,
    )
    ip_xx = _ip_core(
        dens_x.u_perm, dens_x.w_perm, n_jx,
        dens_x.v_comb, dens_x.wv_comb, n_kx,
        r, sigma, is_rel, is_per, period,
    )
    ip_yy = _ip_core(
        dens_y.u_perm, dens_y.w_perm, n_jy,
        dens_y.v_comb, dens_y.wv_comb, n_ky,
        r, sigma, is_rel, is_per, period,
    )

    denom = np.sqrt(ip_xx * ip_yy)
    if denom == 0:
        return float("nan")
    return float(ip_xy / denom)


def cos_sim_exp_tens_raw(
    p1, w1, p2, w2,
    sigma, r, is_rel, is_per, period,
    *, verbose=True,
) -> float:
    """Build two densities and compute cosine similarity in one call.

    Convenience wrapper around :func:`build_exp_tens` and
    :func:`cos_sim_exp_tens` for the raw-argument calling convention.

    Parameters
    ----------
    p1, w1 : array-like
        Pitch or position values and weights for the first multiset.
    p2, w2 : array-like
        Pitch or position values and weights for the second multiset.
    sigma, r, is_rel, is_per, period :
        Tensor parameters (see :func:`build_exp_tens`).
    verbose : bool
        Print progress.

    Returns
    -------
    float
        Cosine similarity.
    """
    dx = build_exp_tens(p1, w1, sigma, r, is_rel, is_per, period, verbose=verbose)
    dy = build_exp_tens(p2, w2, sigma, r, is_rel, is_per, period, verbose=verbose)
    return cos_sim_exp_tens(dx, dy, verbose=verbose)


def _compute_Q(D, r, is_rel, is_per, period):
    """Compute the quadratic form from (already-wrapped) differences D.

    When *is_rel* and *is_per* are both True, pairwise differences
    between components of D are wrapped to ``[-period/2, period/2)``
    before squaring. This restores exact transposition invariance on
    the circle, which is otherwise broken by component-wise wrapping.

    The two formulas are algebraically identical in the non-periodic
    case: ``sum_{i<j} (d_i - d_j)^2 == r * (sum(d^2) - sum(d)^2/r)``.
    """
    if is_rel:
        if is_per:
            Q = np.zeros(D.shape[1:])
            for i in range(r):
                for j in range(i + 1, r):
                    delta = D[i] - D[j]
                    delta = np.mod(delta + period / 2, period) - period / 2
                    Q += delta**2
            Q = Q / r
        else:
            Q = np.sum(D**2, axis=0) - np.sum(D, axis=0) ** 2 / r
    else:
        Q = np.sum(D**2, axis=0)
    return Q


def _ip_core(U, wU, nJ, V, wV, nK, r, sigma, is_rel, is_per, period):
    """Core inner product (perm-side × comb-side)."""
    bytes_needed = (r + 2) * int(nJ) * int(nK) * 8
    mem_limit = 4_000_000_000

    if bytes_needed <= mem_limit:
        return _ip_full(U, wU, nJ, V, wV, nK, r, sigma, is_rel, is_per, period)

    chunk_size = max(1, int(mem_limit / ((r + 2) * int(nJ) * 8)))
    acc = np.zeros(nJ)
    for c in range(0, nK, chunk_size):
        c_end = min(c + chunk_size, nK)
        idx = slice(c, c_end)
        n_kc = c_end - c

        Dc = U[:, :, None] - V[:, idx][:, None, :]
        if is_per:
            Dc = np.mod(Dc + period / 2, period) - period / 2
        Qc = _compute_Q(Dc, r, is_rel, is_per, period)
        Ec = np.exp(-Qc / (4 * sigma**2))
        acc += Ec @ wV[idx]

    return float(wU @ acc)


def _ip_full(U, wU, nJ, V, wV, nK, r, sigma, is_rel, is_per, period):
    """Fully vectorized inner product."""
    D = U[:, :, None] - V[:, None, :]  # (r, nJ, nK)

    if is_per:
        D = np.mod(D + period / 2, period) - period / 2

    Q = _compute_Q(D, r, is_rel, is_per, period)

    E = np.exp(-Q / (4 * sigma**2))  # (nJ, nK)
    return float(wU @ (E @ wV))


# -------------------------------------------------------------------
#  Canonicalization helpers (for batch_cos_sim_exp_tens)
# -------------------------------------------------------------------


def _lex_compare(a: tuple, b: tuple) -> int:
    """Lexicographic comparison. Returns -1, 0, or +1."""
    for ai, bi in zip(a, b):
        if ai < bi:
            return -1
        if ai > bi:
            return 1
    return 0


def _cyclic_canonical(
    p_sorted: np.ndarray,
    w_sorted: np.ndarray | None,
    period: float,
) -> tuple[tuple, tuple | None, float]:
    """Lexicographically smallest rotation of a periodic pitch set.

    Tries all n rotations (subtract each sorted pitch, mod period,
    re-sort with weights) and returns the lex-smallest form plus
    the shift that produced it. This captures all
    transposition-modulo-period equivalences.

    Returns
    -------
    best_p : tuple
        Canonical pitch tuple.
    best_w : tuple or None
        Canonical weight tuple (if weights provided).
    best_shift : float
        The pitch value subtracted to produce the canonical form.
    """
    n = len(p_sorted)
    has_w = w_sorted is not None

    best_p = tuple(p_sorted - p_sorted[0])
    best_w = tuple(w_sorted) if has_w else None
    best_shift = p_sorted[0]

    for rot in range(1, n):
        shifted = np.mod(p_sorted - p_sorted[rot], period)
        si = np.argsort(shifted)
        shifted = shifted[si]
        t_p = tuple(shifted)

        cmp = _lex_compare(t_p, best_p)
        if cmp < 0:
            best_p = t_p
            best_w = tuple(w_sorted[si]) if has_w else None
            best_shift = p_sorted[rot]
        elif cmp == 0 and has_w:
            t_w = tuple(w_sorted[si])
            if _lex_compare(t_w, best_w) < 0:
                best_w = t_w
                best_shift = p_sorted[rot]

    return best_p, best_w, best_shift


def _canonicalize_set(
    p: np.ndarray,
    w: np.ndarray | None,
    is_rel: bool,
    is_per: bool,
    period: float,
) -> tuple[tuple, tuple | None]:
    """Canonical form of a pitch/weight set under isPer/isRel.

    Returns hashable tuples suitable for use as dict keys.
    """
    has_w = w is not None

    # Sort pitches, align weights
    si = np.argsort(p)
    p = p[si]
    if has_w:
        w = w[si]

    # Reduce modulo period
    if is_per:
        p = np.mod(p, period)
        si = np.argsort(p)
        p = p[si]
        if has_w:
            w = w[si]

    # Remove transposition
    if is_rel:
        if is_per:
            # Cyclic canonical form: the lex-smallest rotation captures
            # all transposition-modulo-period equivalences.
            ca_p, ca_w, _ = _cyclic_canonical(p, w, period)
            return ca_p, ca_w
        else:
            p = p - p[0]

    return tuple(p), (tuple(w) if has_w else None)


# -------------------------------------------------------------------
#  batch_cos_sim_exp_tens
# -------------------------------------------------------------------


def batch_cos_sim_exp_tens(
    p_mat_a: np.ndarray,
    p_mat_b: np.ndarray,
    sigma: float,
    r: int,
    is_rel: bool,
    is_per: bool,
    period: float,
    *,
    weights_a: np.ndarray | None = None,
    weights_b: np.ndarray | None = None,
    spectrum: list | None = None,
    precision: int | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Batch cosine similarity of expectation tensors.

    Computes cosine similarity for many paired weighted multisets
    (*p* represents pitches or positions). Each row of *p_mat_a*
    and *p_mat_b* defines one pair.

    The function deduplicates equivalent rows before computation.
    Density structs (via ``build_exp_tens``) are cached per unique
    individual set and ``cos_sim_exp_tens`` is called once per
    unique (A, B) pair, with results mapped back to all matching
    rows. The equivalences exploited depend on the mode:

    - **Absolute non-periodic** (*is_rel* = False, *is_per* = False):
      co-transposition — (A+c, B+c) is equivalent to (A, B).
      Mechanism: subtract A's minimum from both A and B.
    - **Absolute periodic** (*is_rel* = False, *is_per* = True):
      co-transposition and octave displacement.
      Mechanism: mod-reduce both sets, then apply the cyclic
      canonical form to A and the same shift to B.
    - **Relative non-periodic** (*is_rel* = True, *is_per* = False):
      independent transposition of each set (co-transposition is a
      special case).
      Mechanism: subtract the minimum from each set independently.
    - **Relative periodic** (*is_rel* = True, *is_per* = True):
      independent transposition and octave displacement of each set
      (co-transposition is a special case).
      Mechanism: independent cyclic canonical form for each set.

    Parameters
    ----------
    p_mat_a : (nRows, nA) array
        Multiset A values. NaN entries are ignored.
    p_mat_b : (nRows, nB) array
        Multiset B values.
    sigma, r, is_rel, is_per, period :
        Tensor parameters.
    weights_a, weights_b : array, optional
        Weight matrices matching *p_mat_a* / *p_mat_b*.
    spectrum : list, optional
        Arguments for :func:`~mpt.spectra.add_spectra`.
    precision : int, optional
        Round pitch and weight values to this many decimal places
        before processing (and again after canonicalization, to
        absorb arithmetic noise from mod-reduction and subtraction).
        Ensures that nominally identical multisets differing only by
        floating-point noise are correctly deduplicated. For pitch
        data in cents on a 12-TET grid, 4 is more than sufficient;
        for fractional-cent values (e.g., from JI ratios), 6
        preserves all meaningful precision. Default: no rounding
        (full floating-point precision).

        **Limitation:** ``precision`` rounds to decimal places, so it
        cannot resolve discrepancies when pitches lie on an irrational
        grid — e.g., N-EDO tunings where the step size 1200/N is a
        repeating decimal (such as 22-EDO: 1200/22 ≈ 54.5454...).
        Different transpositions of the same set will produce
        different last-digit truncations that no decimal precision
        can collapse. In such cases, convert to integer EDO steps
        before calling this function (scaling *sigma* and *period*
        accordingly) to make deduplication exact.
    verbose : bool
        Print progress.

    Returns
    -------
    np.ndarray
        (nRows,) cosine similarities. NaN for invalid rows.
    """
    p_mat_a = np.asarray(p_mat_a, dtype=np.float64)
    p_mat_b = np.asarray(p_mat_b, dtype=np.float64)
    if p_mat_a.ndim == 1:
        p_mat_a = p_mat_a.reshape(1, -1)
    if p_mat_b.ndim == 1:
        p_mat_b = p_mat_b.reshape(1, -1)

    n_rows = p_mat_a.shape[0]
    if p_mat_b.shape[0] != n_rows:
        raise ValueError("p_mat_a and p_mat_b must have the same number of rows.")

    use_wa = weights_a is not None
    use_wb = weights_b is not None
    use_spec = spectrum is not None

    if use_wa:
        weights_a = np.asarray(weights_a, dtype=np.float64)
        if weights_a.shape != p_mat_a.shape:
            raise ValueError("weights_a must be the same shape as p_mat_a.")
    if use_wb:
        weights_b = np.asarray(weights_b, dtype=np.float64)
        if weights_b.shape != p_mat_b.shape:
            raise ValueError("weights_b must be the same shape as p_mat_b.")

    # Apply precision rounding
    if precision is not None:
        p_mat_a = np.round(p_mat_a, precision)
        p_mat_b = np.round(p_mat_b, precision)
        if use_wa:
            weights_a = np.round(weights_a, precision)
        if use_wb:
            weights_b = np.round(weights_b, precision)

    s = np.full(n_rows, np.nan)

    # ── Phase 1: Canonicalize and build individual-set keys ──────────

    # key_a[i] and key_b[i] are hashable canonical forms for valid rows
    key_a: list[tuple | None] = [None] * n_rows
    key_b: list[tuple | None] = [None] * n_rows
    valid = [False] * n_rows

    # Also store the canonical pitches/weights for later extraction
    canon_data_a: dict[tuple, tuple[np.ndarray, np.ndarray | None]] = {}
    canon_data_b: dict[tuple, tuple[np.ndarray, np.ndarray | None]] = {}

    for i in range(n_rows):
        pa_row = p_mat_a[i]
        pb_row = p_mat_b[i]
        mask_a = ~np.isnan(pa_row)
        mask_b = ~np.isnan(pb_row)
        pa_valid = pa_row[mask_a]
        pb_valid = pb_row[mask_b]

        if len(pa_valid) < r or len(pb_valid) < r:
            continue

        wa_valid = weights_a[i, mask_a] if use_wa else None
        wb_valid = weights_b[i, mask_b] if use_wb else None

        # Canonicalize each set and apply joint co-transposition
        # normalization for the absolute case.
        if is_rel:
            # Relative: independent canonicalization (each set normalized
            # for transposition independently)
            ca_p, ca_w = _canonicalize_set(pa_valid, wa_valid, is_rel, is_per, period)
            cb_p, cb_w = _canonicalize_set(pb_valid, wb_valid, is_rel, is_per, period)
        else:
            # Absolute: joint co-transposition normalization.
            # cosSimExpTens(A-c, B-c) = cosSimExpTens(A, B) because the
            # raw tuple differences cancel. We find A's canonical form
            # and apply the same shift to B.

            # Canonicalize A
            si_a = np.argsort(pa_valid)
            pa_s = pa_valid[si_a]
            wa_s = wa_valid[si_a] if wa_valid is not None else None

            if is_per:
                pa_s = np.mod(pa_s, period)
                si = np.argsort(pa_s)
                pa_s = pa_s[si]
                if wa_s is not None:
                    wa_s = wa_s[si]
                # Cyclic canonical form — collapses all rotations
                ca_p, ca_w, shift = _cyclic_canonical(pa_s, wa_s, period)
            else:
                shift = pa_s[0]
                ca_p = tuple(pa_s - shift)
                ca_w = tuple(wa_s) if wa_s is not None else None

            # Apply the same shift to B
            si_b = np.argsort(pb_valid)
            pb_s = pb_valid[si_b]
            wb_s = wb_valid[si_b] if wb_valid is not None else None

            if is_per:
                pb_shifted = np.mod(pb_s - shift, period)
                si = np.argsort(pb_shifted)
                cb_p = tuple(pb_shifted[si])
                cb_w = tuple(wb_s[si]) if wb_s is not None else None
            else:
                cb_p = tuple(pb_s - shift)
                cb_w = tuple(wb_s) if wb_s is not None else None

        # Re-round after canonicalization to collapse floating-point
        # noise introduced by mod-reduction and subtraction.
        if precision is not None:
            ca_p = tuple(round(x, precision) for x in ca_p)
            cb_p = tuple(round(x, precision) for x in cb_p)
            if ca_w is not None:
                ca_w = tuple(round(x, precision) for x in ca_w)
            if cb_w is not None:
                cb_w = tuple(round(x, precision) for x in cb_w)

        # Hashable key = (canonical_pitches, canonical_weights_or_None)
        ka = (ca_p, ca_w)
        kb = (cb_p, cb_w)

        key_a[i] = ka
        key_b[i] = kb
        valid[i] = True

        # Cache canonical arrays (first occurrence wins)
        if ka not in canon_data_a:
            canon_data_a[ka] = (
                np.array(ca_p, dtype=np.float64),
                np.array(ca_w, dtype=np.float64) if ca_w is not None else None,
            )
        if kb not in canon_data_b:
            canon_data_b[kb] = (
                np.array(cb_p, dtype=np.float64),
                np.array(cb_w, dtype=np.float64) if cb_w is not None else None,
            )

    # ── Phase 2: Deduplicate individual sets, then pairs ─────────────

    unique_a_keys = list(canon_data_a.keys())
    unique_b_keys = list(canon_data_b.keys())
    n_unique_a = len(unique_a_keys)
    n_unique_b = len(unique_b_keys)

    # Build pair keys and deduplicate
    pair_key_to_idx: dict[tuple, int] = {}
    row_to_pair: list[int | None] = [None] * n_rows
    unique_pair_list: list[tuple] = []

    for i in range(n_rows):
        if not valid[i]:
            continue
        pk = (key_a[i], key_b[i])
        if pk not in pair_key_to_idx:
            pair_key_to_idx[pk] = len(unique_pair_list)
            unique_pair_list.append(pk)
        row_to_pair[i] = pair_key_to_idx[pk]

    n_unique_pairs = len(unique_pair_list)

    if verbose:
        n_valid = sum(valid)
        print(
            f"batch_cos_sim_exp_tens: {n_rows} rows, {n_valid} valid, "
            f"{n_unique_a} unique A-sets, {n_unique_b} unique B-sets, "
            f"{n_unique_pairs} unique pairs."
        )
        if is_rel:
            print(
                "  Canonicalization: A-sets and B-sets independently "
                "normalized for transposition"
                + (" and octave equivalence." if is_per else ".")
            )
        else:
            print(
                "  Canonicalization: joint co-transposition"
                + (" with octave equivalence" if is_per else "")
                + "; B-set counts reflect position relative to A."
            )

    # ── Phase 3: Build density structs for unique individual sets ─────

    dens_a: dict[tuple, object] = {}
    for ka in unique_a_keys:
        p_arr, w_arr = canon_data_a[ka]
        if use_spec:
            p_arr, w_arr = add_spectra(p_arr, w_arr, *spectrum)
        dens_a[ka] = build_exp_tens(
            p_arr, w_arr, sigma, r, is_rel, is_per, period, verbose=False
        )

    dens_b: dict[tuple, object] = {}
    for kb in unique_b_keys:
        p_arr, w_arr = canon_data_b[kb]
        if use_spec:
            p_arr, w_arr = add_spectra(p_arr, w_arr, *spectrum)
        dens_b[kb] = build_exp_tens(
            p_arr, w_arr, sigma, r, is_rel, is_per, period, verbose=False
        )

    if verbose:
        print(
            f"batch_cos_sim_exp_tens: built {n_unique_a + n_unique_b} "
            f"density structs ({n_unique_a} A + {n_unique_b} B)."
        )

    # ── Phase 4: Compute similarity for each unique pair ─────────────

    unique_s = [None] * n_unique_pairs
    for up, (ka, kb) in enumerate(unique_pair_list):
        unique_s[up] = cos_sim_exp_tens(dens_a[ka], dens_b[kb], verbose=False)

        if verbose and ((up + 1) % 100 == 0 or up + 1 == n_unique_pairs):
            print(f"  {up + 1} / {n_unique_pairs} unique pairs computed.")

    # ── Phase 5: Map results back ────────────────────────────────────

    for i in range(n_rows):
        if valid[i]:
            s[i] = unique_s[row_to_pair[i]]

    if verbose:
        print("batch_cos_sim_exp_tens: done.")

    return s
