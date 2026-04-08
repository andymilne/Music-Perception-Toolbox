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

    Parameters
    ----------
    p : array-like
        Pitch values.
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

    Parameters
    ----------
    dens : ExpTensDensity
        Precomputed density from :func:`build_exp_tens`.
    x : array-like
        Query points. For 1-D densities, a 1-D array of length *nQ*.
        For multi-dimensional densities, a ``(dim, nQ)`` array.
    normalize : str
        ``'none'`` (default), ``'gaussian'``, or ``'pdf'``.
    verbose : bool
        Print time estimates.

    Returns
    -------
    np.ndarray
        1-D array of density values (length *nQ*).
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
    """Cosine similarity of two r-ad expectation tensors.

    Parameters
    ----------
    dens_x, dens_y : ExpTensDensity
        Precomputed densities from :func:`build_exp_tens`.
    verbose : bool
        Print time estimates.

    Returns
    -------
    float
        Cosine similarity in [−1, 1].
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
    x_p, x_w, y_p, y_w,
    sigma, r, is_rel, is_per, period,
    *, verbose=True,
) -> float:
    """Build two densities and compute cosine similarity in one call."""
    dx = build_exp_tens(x_p, x_w, sigma, r, is_rel, is_per, period, verbose=verbose)
    dy = build_exp_tens(y_p, y_w, sigma, r, is_rel, is_per, period, verbose=verbose)
    return cos_sim_exp_tens(dx, dy, verbose=verbose)


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
        if is_rel:
            Qc = np.sum(Dc**2, axis=0) - np.sum(Dc, axis=0) ** 2 / r
        else:
            Qc = np.sum(Dc**2, axis=0)
        Ec = np.exp(-Qc / (4 * sigma**2))
        acc += Ec @ wV[idx]

    return float(wU @ acc)


def _ip_full(U, wU, nJ, V, wV, nK, r, sigma, is_rel, is_per, period):
    """Fully vectorized inner product."""
    D = U[:, :, None] - V[:, None, :]  # (r, nJ, nK)

    if is_per:
        D = np.mod(D + period / 2, period) - period / 2

    if is_rel:
        Q = np.sum(D**2, axis=0) - np.sum(D, axis=0) ** 2 / r
    else:
        Q = np.sum(D**2, axis=0)

    E = np.exp(-Q / (4 * sigma**2))  # (nJ, nK)
    return float(wU @ (E @ wV))


# -------------------------------------------------------------------
#  batch_cos_sim_exp_tens
# -------------------------------------------------------------------


def batch_cos_sim_exp_tens(
    pitches_a: np.ndarray,
    pitches_b: np.ndarray,
    sigma: float,
    r: int,
    is_rel: bool,
    is_per: bool,
    period: float,
    *,
    weights_a: np.ndarray | None = None,
    weights_b: np.ndarray | None = None,
    spectrum: list | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Batch cosine similarity of expectation tensors.

    Parameters
    ----------
    pitches_a : (nRows, nA) array
        Pitch sets A. NaN entries are ignored.
    pitches_b : (nRows, nB) array
        Pitch sets B.
    sigma, r, is_rel, is_per, period :
        Tensor parameters.
    weights_a, weights_b : array, optional
        Weight matrices matching *pitches_a* / *pitches_b*.
    spectrum : list, optional
        Arguments for :func:`add_spectra` (mode and params as a list,
        e.g. ``['harmonic', 12, 'powerlaw', 1]``).
    verbose : bool
        Print progress.

    Returns
    -------
    np.ndarray
        (nRows,) cosine similarities. NaN for invalid rows.
    """
    pitches_a = np.asarray(pitches_a, dtype=np.float64)
    pitches_b = np.asarray(pitches_b, dtype=np.float64)
    if pitches_a.ndim == 1:
        pitches_a = pitches_a.reshape(1, -1)
    if pitches_b.ndim == 1:
        pitches_b = pitches_b.reshape(1, -1)

    n_rows = pitches_a.shape[0]
    if pitches_b.shape[0] != n_rows:
        raise ValueError("pitches_a and pitches_b must have the same number of rows.")

    use_wa = weights_a is not None
    use_wb = weights_b is not None
    use_spec = spectrum is not None

    if use_wa:
        weights_a = np.asarray(weights_a, dtype=np.float64)
        if weights_a.shape != pitches_a.shape:
            raise ValueError("weights_a must be the same shape as pitches_a.")
    if use_wb:
        weights_b = np.asarray(weights_b, dtype=np.float64)
        if weights_b.shape != pitches_b.shape:
            raise ValueError("weights_b must be the same shape as pitches_b.")

    # Build deduplication keys
    s = np.full(n_rows, np.nan)
    key_to_result: dict[tuple, float] = {}
    n_unique = 0

    for i in range(n_rows):
        pa_row = pitches_a[i]
        pb_row = pitches_b[i]
        mask_a = ~np.isnan(pa_row)
        mask_b = ~np.isnan(pb_row)
        pa_valid = np.sort(pa_row[mask_a])
        pb_valid = np.sort(pb_row[mask_b])

        if len(pa_valid) < r or len(pb_valid) < r:
            continue

        # Build key
        key_parts = [tuple(pa_valid), tuple(pb_valid)]
        if use_wa:
            wa_valid = weights_a[i, mask_a]
            sort_idx = np.argsort(pa_row[mask_a])
            key_parts.append(tuple(wa_valid[sort_idx]))
        if use_wb:
            wb_valid = weights_b[i, mask_b]
            sort_idx = np.argsort(pb_row[mask_b])
            key_parts.append(tuple(wb_valid[sort_idx]))
        key = tuple(key_parts)

        if key in key_to_result:
            s[i] = key_to_result[key]
            continue

        pa = pa_valid
        pb = pb_valid
        wa = weights_a[i, mask_a] if use_wa else None
        wb = weights_b[i, mask_b] if use_wb else None

        if use_spec:
            pa, wa = add_spectra(pa, wa, *spectrum)
            pb, wb = add_spectra(pb, wb, *spectrum)

        val = cos_sim_exp_tens_raw(
            pa, wa, pb, wb, sigma, r, is_rel, is_per, period, verbose=False
        )
        key_to_result[key] = val
        s[i] = val
        n_unique += 1

        if verbose and (n_unique % 100 == 0):
            print(f"  {n_unique} unique pairs computed.")

    if verbose:
        n_valid = int(np.sum(~np.isnan(s)))
        print(
            f"batch_cos_sim_exp_tens: {n_rows} rows, {n_valid} valid, "
            f"{len(key_to_result)} unique pairs."
        )

    return s
