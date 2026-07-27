"""Gaussian-kernel sum helper with optional grid-bucket truncation.

The single centres-path numerical kernel used directly by the single-multiset
centres evaluator and the single-multiset centres-IP path in :mod:`mpt.tensor`, and
by the relative-mode orbit evaluator in :mod:`mpt._mobius`. All other
centres-path consumers (``entropy_exp_tens``, the harmony measures,
etc.) reach it indirectly through those entry points. Routing every
centres-path computation through this single helper is what lets the
``truncation_sigmas`` and ``kernel_precision`` options be applied
uniformly across the toolbox.

See :mod:`mpt._defaults` for the global-defaults machinery.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._defaults import get_default
from ._utils import kernel_chunk_bytes_resolved
from ._tensor.dispatch import _compute_Q


def gaussian_kernel_sum(
    C: np.ndarray,
    wJ: np.ndarray,
    X: np.ndarray,
    sigma: float,
    *,
    is_rel: bool = False,
    r: int = 0,
    is_per: bool = False,
    period: float = 0.0,
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    wrap: str = 'full-image',
) -> np.ndarray:
    """Compute the Gaussian kernel sum over centres against queries.

    Returns a 1-D array ``v`` of length ``nQ`` where::

        v[q] = sum_j wJ[j] * exp(-Q(c_j - x_q) / (2 * sigma**2))

    and ``Q`` is the quadratic form determined by ``is_rel`` and ``r``:

    - abs mode (``is_rel=False``):  ``Q(D) = sum(D**2)``
    - rel mode (``is_rel=True``):   ``Q(D) = sum(D**2) - sum(D)**2 / r``

    Parameters
    ----------
    C : (dim, nJ) ndarray
        Centres.
    wJ : (nJ,) ndarray
        Weights.
    X : (dim, nQ) ndarray
        Queries.
    sigma : float
        Gaussian width.
    is_rel : bool, default False
        Use the rel-mode quadratic form (requires ``r >= 2``).
    r : int, default 0
        Tensor order, required for ``is_rel=True``.
    is_per : bool, default False
        Periodic mode: wraps differences to ``[-period/2, period/2)``
        before applying ``Q``. Currently falls through to the exact path
        regardless of ``truncation_sigmas``; periodic-mode truncation is
        a follow-up.
    period : float, default 0.0
        Period; required for ``is_per=True``.
    truncation_sigmas : float, optional
        If finite, centres beyond a Q-ball of squared radius
        ``(truncation_sigmas * sigma)**2`` are skipped via a grid-bucket
        spatial index. Discarded centres' kernel value is bounded by
        ``exp(-truncation_sigmas**2 / 2)``. If ``None``, uses the
        toolbox default (factory: ``6``; set ``math.inf`` for the exact
        untruncated result).
    kernel_precision : {'double', 'single'}, optional
        ``'single'`` casts hot-loop arrays to single precision (~2x
        speedup, ~1e-7 relative accuracy). Output is always cast back
        to double. If ``None``, uses the toolbox default
        (factory: ``'double'``).

    Returns
    -------
    (nQ,) ndarray, double dtype
    """
    if truncation_sigmas is None:
        truncation_sigmas = get_default("truncation_sigmas")
    # Resolve the "exact" sentinel (inf) to the finite accuracy-floor
    # width, uniformly with every other truncation path. After this,
    # truncation_sigmas is always finite and truncation always applies
    # (except in periodic mode, handled below).
    from ._defaults import resolve_truncation_sigmas
    truncation_sigmas = resolve_truncation_sigmas(truncation_sigmas)
    if kernel_precision is None:
        kernel_precision = get_default("kernel_precision")
    kernel_precision = kernel_precision.lower()
    if kernel_precision not in ("double", "single"):
        raise ValueError(
            f"kernel_precision must be 'double' or 'single' (got {kernel_precision!r})"
        )
    if truncation_sigmas <= 0:
        raise ValueError("truncation_sigmas must be positive (math.inf to disable)")

    C = np.asarray(C)
    wJ = np.asarray(wJ).ravel()
    X = np.asarray(X)

    if C.ndim != 2:
        raise ValueError(f"C must be 2-D (dim, nJ); got shape {C.shape}")
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (dim, nQ); got shape {X.shape}")
    dim, nJ = C.shape
    if X.shape[0] != dim:
        raise ValueError(
            f"C and X must have the same number of rows (got {dim} vs {X.shape[0]})"
        )
    if wJ.size != nJ:
        raise ValueError(f"wJ must have length nJ = {nJ}; got {wJ.size}")
    if is_rel and r < 2:
        raise ValueError(f"rel mode requires r >= 2 (got {r})")
    if is_per and period <= 0:
        raise ValueError("periodic mode requires period > 0")

    nQ = X.shape[1]

    dtype = np.float32 if kernel_precision == "single" else np.float64
    C_w = C.astype(dtype, copy=False)
    wJ_w = wJ.astype(dtype, copy=False)
    X_w = X.astype(dtype, copy=False)
    sigma_w = dtype(sigma)
    inv2s2 = dtype(1.0 / (2.0 * float(sigma) ** 2))

    use_truncation = (
        math.isfinite(truncation_sigmas)
        and truncation_sigmas > 0
        and nJ > 0
        and nQ > 0
    )

    if use_truncation and not is_per:
        # Dispatch on dimensionality: the 1-D abs case admits a much
        # tighter path via sorted centres + searchsorted, which yields a
        # contiguous centre window per query and so avoids the 3**dim
        # neighbour-offset expansion and ragged scatter the general path
        # needs in higher dimensions.
        dim_w = C_w.shape[0]
        if dim_w == 1 and not is_rel:
            v = _truncated_kernel_sum_1d_vectorised(
                C_w, wJ_w, X_w, sigma_w,
                float(truncation_sigmas), inv2s2,
            )
        else:
            v = _truncated_kernel_sum(
                C_w, wJ_w, X_w, sigma_w, is_rel, r,
                float(truncation_sigmas), inv2s2,
            )
    elif use_truncation and is_per and C_w.shape[0] == 1 and not is_rel:
        # Circular 1-D truncation, valid only when the window is
        # narrower than the circle; otherwise no savings (and the
        # replication trick would double count), so fall through to the
        # exact periodic path.
        from ._defaults import truncation_radius
        radius = truncation_radius(float(truncation_sigmas), float(sigma))
        if 2.0 * radius < float(period):
            v = _truncated_kernel_sum_1d_circular(
                C_w, wJ_w, X_w, sigma_w, dtype(period),
                float(truncation_sigmas), inv2s2,
            )
        else:
            v = _exact_kernel_sum(
                C_w, wJ_w, X_w, is_rel, r, is_per, dtype(period), inv2s2,
                sigma_w, truncation_sigmas, wrap,
            )
    else:
        v = _exact_kernel_sum(
            C_w, wJ_w, X_w, is_rel, r, is_per, dtype(period), inv2s2,
            sigma_w, truncation_sigmas, wrap,
        )

    return v.astype(np.float64, copy=False)


# ---------------------------------------------------------------------
# Exact path
# ---------------------------------------------------------------------

def _exact_kernel_sum(C, wJ, X, is_rel, r, is_per, period, inv2s2, sigma,
                      truncation_sigmas=None, wrap='full-image'):
    dim, nJ = C.shape
    nQ = X.shape[1]
    if nJ == 0 or nQ == 0:
        return np.zeros(nQ, dtype=C.dtype)

    # Memory-budget chunking on nQ. Peak per-chunk transient ~
    # (2*dim + 2) × nJ × nQ × bytes_per_scalar (broadcast difference,
    # its square, and the summed/exponentiated intermediate co-resident).
    bytes_per_scalar = C.dtype.itemsize
    bytes_needed = (2 * dim + 2) * nJ * nQ * bytes_per_scalar
    BUDGET = kernel_chunk_bytes_resolved()
    if bytes_needed <= BUDGET:
        return _eval_chunk(C, wJ, X, is_rel, r, is_per, period, inv2s2, sigma,
                           truncation_sigmas, wrap)

    chunk = max(1, BUDGET // ((2 * dim + 2) * nJ * bytes_per_scalar))
    out = np.zeros(nQ, dtype=C.dtype)
    for c0 in range(0, nQ, chunk):
        c1 = min(c0 + chunk, nQ)
        out[c0:c1] = _eval_chunk(
            C, wJ, X[:, c0:c1], is_rel, r, is_per, period, inv2s2, sigma,
            truncation_sigmas, wrap
        )
    return out


def _eval_chunk(C, wJ, Xq, is_rel, r, is_per, period, inv2s2, sigma,
                truncation_sigmas=None, wrap='full-image'):
    dim = C.shape[0]
    # Abs-per full-image is the hot path at large K; handle it up
    # front without building the joint (dim, nJ, nQc) diff tensor.
    # Compute each slot's difference and theta on the (nJ, nQc) slice
    # and multiply into a running product. Peak working memory drops
    # from (2*dim + 1) * nJ * nQc to 2 * nJ * nQc, and the exp calls
    # each get a smaller array with better cache behaviour. Output
    # bit-identical to the joint form. ~25% faster at K=50, more at
    # larger K where the 3-D tensor no longer fits in cache.
    if is_per and not is_rel and wrap != 'single-image':
        from ._wrapped_kernel import wrapped_gaussian_1d
        d_k = C[0, :, None] - Xq[0, None, :]
        E = wrapped_gaussian_1d(
            d_k, float(sigma), float(period), truncation_sigmas,
            exponent_denominator=2,
        )
        for k in range(1, dim):
            d_k = C[k, :, None] - Xq[k, None, :]
            theta_k = wrapped_gaussian_1d(
                d_k, float(sigma), float(period), truncation_sigmas,
                exponent_denominator=2,
            )
            E *= theta_k
        return wJ @ E  # (nQc,)

    # Other modes still need the joint (dim, nJ, nQc) diff tensor for
    # the shared Q-form path.
    D = C[:, :, None] - Xq[:, None, :]
    # Outer wrap is only needed for abs+per. For rel+per, _compute_Q
    # applies the pairwise wrap inside (Eq 6 of the preprint) to
    # restore exact transposition invariance on the circle; the
    # outer wrap would be redundant there. Mathematically equivalent
    # to np.mod(D + period/2, period) - period/2 at every input
    # (including exact half-period boundaries); ~2x faster by
    # avoiding np.mod's two-pass implementation. Reduction-order
    # numerical agreement (~1e-13).
    if is_per and not is_rel:
        # Abs-per single-image opt-in: nearest-image reduction, then
        # fall through to the shared Q-form path.
        D = D - period * np.floor(D / period + 0.5)
    Q = _compute_Q(D, r, is_rel, is_per, period, reduced=is_rel)
    # Use the direct division (Q / (2*sigma^2)) rather than Q * inv2s2,
    # to match v2.0/v2.1 ULP-for-ULP at default settings (in all modes
    # except rel+per, where v2.X corrects an inherited v1 single-axis-
    # wrap form to the pairwise-wrap form, in line with cosSimExpTens).
    E = np.exp(-Q / (2 * sigma ** 2))      # (nJ, nQc)
    return wJ @ E                          # (nQc,)


# ---------------------------------------------------------------------
# Truncated path — grid-bucket spatial index
# ---------------------------------------------------------------------

def _truncated_kernel_sum(C, wJ, X, sigma, is_rel, r, k_sigma, inv2s2):
    """Truncated kernel sum, fully vectorised over queries.

    Builds the bucket grid, then for all queries at once: expands the
    3**dim neighbour-offset coordinates, applies vectorised in-bounds and
    bucket-exists masks, ragged-expands to (query, centre) pairs via a
    cumsum trick, computes the kernel for all surviving pairs in one
    pass, and scatter-accumulates into the per-query output via
    ``np.bincount``. Handling every query in one pass this way, rather
    than looping queries and looking buckets up in a dict, is worth
    ~5-40x depending on dim (largest where the per-query work is small);
    measured for this implementation, so not comparable with the MATLAB
    figure quoted in internal/gaussianKernelSum.m.

    The dim=1 abs case is handled by ``_truncated_kernel_sum_1d_vectorised``;
    this function covers dim>=2 (and dim=1 rel as an edge case).
    """
    dim, nJ = C.shape
    nQ = X.shape[1]
    if nJ == 0:
        return np.zeros(nQ, dtype=C.dtype)

    # Coordinate transform to make Q-ball spherical (rel-mode only).
    if is_rel:
        e = np.ones(dim, dtype=np.float64)
        M = np.eye(dim, dtype=np.float64) - np.outer(e, e) / r
        lams, U = np.linalg.eigh(M)
        lams = np.clip(lams, 0.0, None)
        T = (U @ np.diag(np.sqrt(lams))).T
        T = T.astype(C.dtype, copy=False)
    else:
        T = np.eye(dim, dtype=C.dtype)

    Ct = T @ C                                  # (dim, nJ) transformed
    Xt = T @ X                                  # (dim, nQ) transformed

    from ._defaults import truncation_radius
    bucket_size = truncation_radius(k_sigma, float(sigma))
    threshold2 = bucket_size ** 2

    cmin = Ct.min(axis=1)
    cmax = Ct.max(axis=1)
    n_buckets = np.maximum(
        1,
        np.ceil((cmax - cmin) / bucket_size).astype(np.int64) + 1,
    )

    # Centre bucket coords (0-indexed).
    buck_c = np.floor((Ct - cmin[:, None]) / bucket_size).astype(np.int64)
    buck_c = np.clip(buck_c, 0, (n_buckets - 1)[:, None])
    lin_c = _sub_to_ind(n_buckets, buck_c)

    # Sort centres by bucket linear index.
    order = np.argsort(lin_c, kind="stable")
    sorted_lin = lin_c[order]
    if sorted_lin.size == 0:
        return np.zeros(nQ, dtype=C.dtype)

    # Build a sorted run table (replacement for the previous Python
    # dict): bucket_lin -> (start, count) into `order`. Sorted-ascending
    # bucket_lin lets us look up via searchsorted in vectorised form.
    boundaries = np.concatenate(
        ([0], 1 + np.flatnonzero(sorted_lin[1:] != sorted_lin[:-1]))
    )
    run_lin = sorted_lin[boundaries]            # (n_runs,)
    run_starts = boundaries                     # (n_runs,)
    run_ends = np.concatenate((boundaries[1:], [sorted_lin.size]))
    run_counts = run_ends - run_starts          # (n_runs,)

    # Neighbour offsets (3^dim) and query bucket coords.
    offsets = _neighbour_offsets(dim)           # (dim, 3^dim)
    n_offsets = offsets.shape[1]
    buck_x = np.floor((Xt - cmin[:, None]) / bucket_size).astype(np.int64)
    buck_x = np.clip(buck_x, 0, (n_buckets - 1)[:, None])

    out = np.zeros(nQ, dtype=C.dtype)

    # --- All queries at once ---
    # Neighbour coords: (dim, nQ, n_offsets).
    nb_all = buck_x[:, :, None] + offsets[:, None, :]

    # In-bounds: shape (nQ, n_offsets).
    in_bounds = np.all(
        (nb_all >= 0) & (nb_all < n_buckets[:, None, None]),
        axis=0,
    )

    # Linear bucket index for each (query, offset) pair.
    nb_lin = _sub_to_ind(
        n_buckets, nb_all.reshape(dim, nQ * n_offsets)
    ).reshape(nQ, n_offsets)

    # Vectorised bucket lookup via searchsorted on the sorted run table.
    pos = np.searchsorted(run_lin, nb_lin)
    pos = np.minimum(pos, run_lin.size - 1)
    bucket_exists = run_lin[pos] == nb_lin
    valid = in_bounds & bucket_exists

    # For valid pairs, look up the run (start, count); zeros elsewhere.
    starts_grid = run_starts[pos]               # (nQ, n_offsets)
    counts_grid = np.where(valid, run_counts[pos], 0)

    counts_flat = counts_grid.reshape(-1)
    total = int(counts_flat.sum())
    if total == 0:
        return out

    # === Ragged expansion: (query, bucket) pairs -> (query, centre) ===
    # cumsum trick: for each pair i with count c_i, positions in
    # [cumulative_start[i], cumulative_end[i]) get centre offsets
    # 0..c_i-1 within the run.
    pair_q_idx = np.repeat(np.arange(nQ, dtype=np.int64), n_offsets)
    starts_flat = starts_grid.reshape(-1)
    cumulative_end = np.cumsum(counts_flat)
    cumulative_start = cumulative_end - counts_flat

    q_arr = np.repeat(pair_q_idx, counts_flat)
    pair_centre_offset = np.arange(total, dtype=np.int64) - np.repeat(
        cumulative_start, counts_flat
    )
    order_idx = np.repeat(starts_flat, counts_flat) + pair_centre_offset
    c_arr = order[order_idx]

    # === One vectorised kernel computation over all candidate pairs ===
    Dq = C[:, c_arr] - X[:, q_arr]              # (dim, total)
    if is_rel:
        Q = np.sum(Dq * Dq, axis=0) - np.sum(Dq, axis=0) ** 2 / r
    else:
        Q = np.sum(Dq * Dq, axis=0)
    keep = Q <= threshold2

    if not np.any(keep):
        return out

    Qk = Q[keep]
    q_kept = q_arr[keep]
    c_kept = c_arr[keep]
    kernel_vals = np.exp(-Qk * inv2s2)
    contribs = wJ[c_kept] * kernel_vals

    # Scatter-accumulate via bincount (faster than np.add.at).
    out = np.bincount(q_kept, weights=contribs, minlength=nQ).astype(C.dtype)
    return out


# ---------------------------------------------------------------------
# Vectorised 1-D abs-mode truncated path
#
# For dim=1 absolute-mode workloads, the centres can be sorted along
# their only axis and per-query active windows located via vectorised
# searchsorted on the bounds [x - kσ, x + kσ]. All queries then
# process a fixed-width slice of centres (the maximum window size in
# the batch), padded with zero-weight entries where their own window
# is shorter. A fixed-width slab needs neither the 3**dim
# neighbour-offset expansion nor the ragged scatter that
# _truncated_kernel_sum uses to stay vectorised in higher dimensions,
# and is ~10-30× faster at typical orbit-path sizes (N ≈ 50-300
# partials); measured for this implementation.
#
# Used in particular by mobius.eval_orbit_abs for the per-block
# 1-D kernel sum that arises after factoring the block's
# m-dimensional quadratic form Q_B = var(x_B) + m·(x̄_B - p)².
# ---------------------------------------------------------------------

def _truncated_kernel_sum_1d_vectorised(C, wJ, X, sigma, k_sigma, inv2s2):
    """Compute ``sum_i wJ[i] * exp(-(X[q] - C[i])^2/(2σ²))`` for each
    query column ``q``, including only centres within ``k_sigma · σ``
    of the query.

    Inputs:
        C  : (1, nJ) ndarray  — centres (single coordinate axis).
        wJ : (nJ,)  ndarray   — centre weights.
        X  : (1, nQ) ndarray  — queries.
        sigma : float
        k_sigma : float       — truncation radius in units of σ.
        inv2s2 : float        — 1/(2σ²) precomputed.

    Returns ``(nQ,)`` array of kernel sums.
    """
    nJ = C.shape[1]
    nQ = X.shape[1]
    if nJ == 0 or nQ == 0:
        return np.zeros(nQ, dtype=C.dtype)

    from ._defaults import truncation_radius
    threshold = truncation_radius(k_sigma, float(sigma))
    c_axis = C[0]                          # (nJ,)
    x_axis = X[0]                          # (nQ,)

    # Sort centres along the single axis; reuse for all queries.
    order = np.argsort(c_axis, kind="stable")
    c_sorted = c_axis[order]
    w_sorted = wJ[order]

    # For each query, find the inclusive lower / exclusive upper
    # bounds in the sorted centre array (the active window).
    i_low = np.searchsorted(c_sorted, x_axis - threshold, side="left")
    i_high = np.searchsorted(c_sorted, x_axis + threshold, side="right")
    win = i_high - i_low                   # (nQ,)
    max_win = int(win.max(initial=0))

    if max_win == 0:
        # No centre is within the truncation radius for any query.
        return np.zeros(nQ, dtype=C.dtype)
    if max_win >= nJ:
        # Truncation window covers the entire centre array for at
        # least one query — no savings; fall through to dense compute.
        diffs = x_axis[:, None] - c_sorted[None, :]
        kernel = np.exp(-(diffs * diffs) * inv2s2)
        return (kernel * w_sorted[None, :]).sum(axis=1).astype(
            C.dtype, copy=False
        )

    # Build (nQ, max_win) index matrix into the sorted arrays.
    # Each row is i_low[q] + [0, 1, ..., max_win-1]; entries beyond
    # i_high[q] are masked out with zero weight.
    offsets = np.arange(max_win, dtype=np.int64)
    idx = i_low[:, None] + offsets[None, :]           # (nQ, max_win)
    # Clip to valid range; out-of-range entries will be zero-masked.
    mask = idx < i_high[:, None]
    idx_clipped = np.minimum(idx, nJ - 1)             # safe for indexing

    p_slices = c_sorted[idx_clipped]                  # (nQ, max_win)
    w_slices = w_sorted[idx_clipped]
    diffs = x_axis[:, None] - p_slices                # (nQ, max_win)
    kernel = np.exp(-(diffs * diffs) * inv2s2)

    # Apply the in-window mask by zeroing out-of-window contributions.
    kernel = np.where(mask, kernel, 0.0)

    return (kernel * w_slices).sum(axis=1).astype(C.dtype, copy=False)


def _truncated_kernel_sum_1d_circular(C, wJ, X, sigma, period, k_sigma,
                                      inv2s2):
    """Circular twin of :func:`_truncated_kernel_sum_1d_vectorised`.

    Computes ``sum_i wJ[i] * exp(-wrap(X[q]-C[i])^2/(2σ²))`` on the circle
    of circumference ``period``, including only centres within
    ``k_sigma·σ`` (wrapped) of each query. Requires the window to be
    narrower than the circle (``2·radius < period``); the caller guards
    this. Centres are replicated at ``c-P, c, c+P`` so a wrapped window
    maps to a contiguous range of the sorted array, and since the window
    is narrower than ``P`` at most one copy of any centre falls inside,
    so there is no double counting. Distances are then plain (the nearest
    copy already realises the wrapped distance).
    """
    nJ = C.shape[1]
    nQ = X.shape[1]
    if nJ == 0 or nQ == 0:
        return np.zeros(nQ, dtype=C.dtype)

    from ._defaults import truncation_radius
    threshold = truncation_radius(k_sigma, float(sigma))

    c_axis = np.mod(C[0], period)
    x_axis = np.mod(X[0], period)

    # Triple the centres across one period on each side.
    c3 = np.concatenate([c_axis - period, c_axis, c_axis + period])
    w3 = np.tile(wJ, 3)
    order = np.argsort(c3, kind="stable")
    c_sorted = c3[order]
    w_sorted = w3[order]

    i_low = np.searchsorted(c_sorted, x_axis - threshold, side="left")
    i_high = np.searchsorted(c_sorted, x_axis + threshold, side="right")
    win = i_high - i_low
    max_win = int(win.max(initial=0))
    if max_win == 0:
        return np.zeros(nQ, dtype=C.dtype)

    offsets = np.arange(max_win, dtype=np.int64)
    idx = i_low[:, None] + offsets[None, :]
    mask = idx < i_high[:, None]
    idx_clipped = np.minimum(idx, c_sorted.shape[0] - 1)

    p_slices = c_sorted[idx_clipped]
    w_slices = w_sorted[idx_clipped]
    diffs = x_axis[:, None] - p_slices
    kernel = np.exp(-(diffs * diffs) * inv2s2)
    kernel = np.where(mask, kernel, 0.0)
    return (kernel * w_slices).sum(axis=1).astype(C.dtype, copy=False)


def _sub_to_ind(siz: np.ndarray, subs: np.ndarray) -> np.ndarray:
    """Vectorised sub2ind for (nDim, n) subscript columns (0-indexed)."""
    n_dim, n = subs.shape
    if n_dim == 1:
        return subs[0].astype(np.int64, copy=False)
    lin = subs[0].astype(np.int64, copy=False).copy()
    stride = 1
    for d in range(1, n_dim):
        stride *= int(siz[d - 1])
        lin = lin + subs[d].astype(np.int64) * stride
    return lin


def _neighbour_offsets(dim: int) -> np.ndarray:
    """Return (dim, 3**dim) integer offsets covering all 3-cube neighbours."""
    n_off = 3 ** dim
    offsets = np.zeros((dim, n_off), dtype=np.int64)
    for i in range(n_off):
        idx = i
        for d in range(dim):
            offsets[d, i] = (idx % 3) - 1
            idx //= 3
    return offsets
