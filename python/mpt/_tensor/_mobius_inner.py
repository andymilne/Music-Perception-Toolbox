"""Möbius-method inner-product matrices for the multi-attribute path.

The per-attribute inner-product machinery of the Möbius decomposition
(Milne 2026, Sec. 4): given two densities' values and weights for one
attribute, each routine here returns the matrix of that attribute's
contribution to the inner product between every pair of events, which the
cosine path then combines across attributes.

The routines divide by the attribute's mode. Absolute attributes contract
their orbit tables directly. Relative attributes marginalise a translation
over a grid, periodic ones over the shared uniform grid on [0, P) and
non-periodic ones over a truncated line grid; spectra get a dedicated
branch that exploits the shared partial structure. The closed-form
routines cover the cases where the matrix is available analytically
without enumerating tuples.

This module is the twin of the MATLAB +mobius package's inner-product
files: mobius.maPerAttrInnerMatrix, mobius.relInnerBatched,
mobius.spectralRelInnerMatrix, mobius.closedFormAttrMatrixFrom,
mobius.closedFormAttrCentres, and mobius.maRelAttrPrefersCentres, together
with the sparse and direct-enumeration helpers they call. Names follow the
MATLAB side so the two implementations correspond function for function.
"""
from __future__ import annotations

import math
from itertools import permutations

import numpy as np

from .._utils import kernel_chunk_bytes_resolved
from .dispatch import (
    _compute_Q,
    _compute_Q_inner_blocks,
    _inner_r_vec,
    _ORBIT_K_MINUS_R_MIN,
    _ORBIT_SIGMA_OVER_P_THRESHOLD,
)


_ORBIT_GRID_SLAB_ELEMS = 1 << 19


def _trunc_kernel_exp(exp_arg, sigma, truncation_sigmas):
    """Evaluate ``exp(-exp_arg / (4 σ²))`` with optional truncation.

    ``exp_arg`` is the non-negative quantity entering the kernel
    exponent. For a 1-D pairwise Gaussian inner-product kernel,
    ``exp_arg = (p_x - p_y)²``; for an r-D r-tuple kernel,
    ``exp_arg = ||d||² = Σ_a d_a²``.

    When ``truncation_sigmas`` is finite, entries with
    ``exp_arg > 2 · (truncation_sigmas · σ)²`` are zeroed without
    evaluating ``np.exp``, saving work proportional to the pruned
    fraction. The threshold uniformly matches "the inner-product
    kernel value falls below ``exp(-truncation_sigmas² / 2)``": that
    kernel is ``G(d; σ√2)``, so its value at distance ``|d|`` is
    ``exp(-|d|² / (4 σ²))``, and the cutoff condition is
    ``|d|² > 2 (truncation_sigmas · σ)²``.

    For r-tuple kernels the same threshold on ``Σ_a d_a²`` is correct
    because the r-D kernel is ``Π_a G(d_a; σ√2) = exp(-Σ_a d_a² /
    (4 σ²))``.

    When ``truncation_sigmas`` is ``None`` it resolves against the
    global default. The "exact" sentinel ``math.inf`` resolves to the
    finite accuracy-floor width (:func:`mpt._defaults.accuracy_floor_sigmas`,
    default 1e-12), uniformly with every other truncation path, so
    truncation always applies.
    """
    from .._defaults import truncation_ip_sqdist
    cutoff = truncation_ip_sqdist(truncation_sigmas, sigma)
    mask = exp_arg <= cutoff
    out = np.zeros_like(exp_arg)
    out[mask] = np.exp(-exp_arg[mask] / (4 * sigma ** 2))
    return out


# Sparse-orbit cost model. The sparse per-pair orbit undercuts the dense
# batched contraction only when the slot kernel is both large and sparse;
# below these thresholds the dense einsum's constant factors win. Tunable.
_ORBIT_SPARSE_MIN_KERNEL = 200_000     # K_x * K_y floor


_ORBIT_SPARSE_MAX_DENSITY = 0.20       # nnz / (K_x * K_y) ceiling


def _build_sparse_kernel_abs(pX, pY, sigma, truncation_sigmas):
    """Spatially-culled absolute-mode kernel as a scipy.sparse matrix.

    Keeps only slot pairs within the truncation radius via a 1-D sorted
    window (O(n log n + nnz)), matching :func:`_trunc_kernel_exp`'s cutoff
    without the dense O(n^2) distance pass.
    """
    import scipy.sparse as sp
    R = np.sqrt(2.0) * float(truncation_sigmas) * sigma
    order = np.argsort(pY)
    pYs = pY[order]
    rows, cols, vals = [], [], []
    for i in range(pX.size):
        x = pX[i]
        lo = np.searchsorted(pYs, x - R)
        hi = np.searchsorted(pYs, x + R)
        if hi > lo:
            jj = order[lo:hi]
            d = x - pY[jj]
            rows.append(np.full(jj.size, i, dtype=np.intp))
            cols.append(jj.astype(np.intp))
            vals.append(np.exp(-d * d / (4.0 * sigma * sigma)))
    if not rows:
        return sp.csr_matrix((pX.size, pY.size))
    return sp.csr_matrix(
        (np.concatenate(vals),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(pX.size, pY.size))


def _rel_per_sparse_prep(pY, period):
    """One-time sorted-tripled centre arrays for the circular builder.

    Folds the B-side slot values into ``[0, P)``, sorts them, and
    replicates each at ``c - P, c, c + P`` so a wrapped window maps to a
    contiguous range of the sorted array. Returns ``(c3, j3)``: the
    tripled sorted coordinates and, aligned with them, the original
    column index of each copy. Shared across all u-nodes of a pair.
    """
    pYm = np.mod(np.asarray(pY, dtype=np.float64), period)
    order = np.argsort(pYm, kind="stable")
    pYs = pYm[order]
    c3 = np.concatenate([pYs - period, pYs, pYs + period])
    j3 = np.concatenate([order, order, order]).astype(np.intp)
    return c3, j3


def _build_sparse_kernel_rel_per(pX, c3, j3, pY, sigma, cutoff, period, u):
    """Circular twin of :func:`_build_sparse_kernel_abs` at shift ``u``.

    Entries ``exp(-wrap(pX[i] + u - pY[j])^2 / (4 sigma^2))`` for wrapped
    squared distance at most ``cutoff``; the caller guards
    ``2 sqrt(cutoff) < period``, so each row's window covers at most one
    copy of any centre (no double counting). Candidates are located via
    the sorted-tripled window (with a hair of padding), then retained
    and evaluated with the dense path's own arithmetic --- the raw
    difference, its floor-wrap, the ``exp_arg <= cutoff`` retention, and
    ``exp(-exp_arg / (4 sigma^2))`` --- so the sparse kernel densifies
    to :func:`_trunc_kernel_exp`'s output bit-for-bit.
    """
    import scipy.sparse as sp
    n_x = pX.size
    n_y = pY.size
    R = float(np.sqrt(cutoff))
    R_pad = R * (1.0 + 1e-9) + 1e-9 * period
    x = np.mod(pX + u, period)
    lo = np.searchsorted(c3, x - R_pad, side="left")
    hi = np.searchsorted(c3, x + R_pad, side="right")
    counts = hi - lo
    total = int(counts.sum())
    if total == 0:
        return sp.csr_matrix((n_x, n_y))
    row_ptr = np.zeros(n_x + 1, dtype=np.intp)
    np.cumsum(counts, out=row_ptr[1:])
    rows = np.repeat(np.arange(n_x, dtype=np.intp), counts)
    flat = (np.arange(total, dtype=np.intp)
            - np.repeat(row_ptr[:-1], counts)
            + np.repeat(lo, counts))
    cols = j3[flat]
    # Dense-path arithmetic on the candidates: raw difference,
    # floor-wrap, inclusive cutoff, exp.
    d = pX[rows] + u - pY[cols]
    d = d - period * np.floor(d / period + 0.5)
    exp_arg = d ** 2
    keep = exp_arg <= cutoff
    if not np.any(keep):
        return sp.csr_matrix((n_x, n_y))
    vals = np.exp(-exp_arg[keep] / (4 * sigma ** 2))
    K = sp.csr_matrix((vals, (rows[keep], cols[keep])), shape=(n_x, n_y))
    K.sort_indices()
    return K


def _orbit_safe_submatrix_sparse(Px_s, Wx_s, Py_s, Wy_s, sigma, r,
                                 prefactor, truncation_sigmas,
                                 return_cancellation_ratio):
    """Safe-submatrix orbit inner products via the sparse per-pair path.

    Mirrors the dense batched safe-submatrix output: returns the flattened
    ``(N_xs * N_ys,)`` vector in row-major (x, y) order and the worst
    cancellation ratio. Each event uses only its non-zero-weight slots, so
    variable cardinality is handled naturally (a zero-weight slot
    contributes zero to every orbit term).
    """
    from .._mobius import inner_product_orbit_sparse
    N_xs = Px_s.shape[1]
    N_ys = Py_s.shape[1]
    flat = np.empty(N_xs * N_ys, dtype=np.float64)
    worst = 1.0
    y_slots = []
    for j in range(N_ys):
        vy = Wy_s[:, j] != 0.0
        y_slots.append((Py_s[vy, j], Wy_s[vy, j]))
    for i in range(N_xs):
        vx = Wx_s[:, i] != 0.0
        pxi, wxi = Px_s[vx, i], Wx_s[vx, i]
        for j in range(N_ys):
            pyj, wyj = y_slots[j]
            Ks = _build_sparse_kernel_abs(pxi, pyj, sigma, truncation_sigmas)
            if return_cancellation_ratio:
                v, ratio = inner_product_orbit_sparse(
                    Ks, wxi, wyj, r, prefactor=prefactor,
                    return_cancellation_ratio=True)
                if ratio < worst:
                    worst = ratio
            else:
                v = inner_product_orbit_sparse(
                    Ks, wxi, wyj, r, prefactor=prefactor)
            flat[i * N_ys + j] = v
    return flat, worst


def _ma_per_attr_inner_matrix(
    Px, Wx, Py, Wy, sigma, r, is_rel, is_per, period,
    *, return_cancellation_ratio=False, truncation_sigmas=None,
    prune_zero_weight_events=True, wrap='full-image',
):
    """Per-attribute (event_X, event_Y) inner product matrix for the
    MA path under the Möbius method.

    ``Px`` is (K, N_x), ``Wx`` is (K, N_x); same shape for Y. Returns
    an (N_x, N_y) matrix where entry (n_X, n_Y) is the per-attribute
    inner product over the K slot values of event n_X (X-side) against
    those of n_Y (Y-side).

    Strategy (in parity with MATLAB ``mobius.maPerAttrInnerMatrix``):

    - r = 1: direct kernel sum with NaN -> zero-weight padding (no
      Möbius decomposition, cancellation impossible).

    - r >= 2 abs: hybrid safe/unsafe partition. An event is "safe" on
      this attribute iff its non-NaN slot count K_eff satisfies
      ``K_eff - r >= _ORBIT_K_MINUS_R_MIN`` (= 2; the precision margin
      used elsewhere in the Möbius machinery). Safe-vs-safe pairs flow
      through the vectorised batched Möbius method with within-safe-group
      zero-padding. Pairs involving any unsafe event flow through
      :func:`_inner_product_direct_abs`, which is exact for any
      K >= r (no Möbius alternating sum, so no cancellation).

    - r >= 2 rel: per-event-pair loop with zero-pad. Auto dispatch
      routes any rel group globally to Bulger's method; this path runs
      only on explicit ``method='mobius'`` opt-in. Events with K_eff - r
      below the precision margin in this niche regime may lose
      precision in the Möbius relative-mode u-grid integration; users
      wanting exact rel + ragged Möbius-method behaviour should either
      filter events to K_eff >= r + 2 or use ``method='auto'`` (which
      routes to Bulger's method).

    With ``return_cancellation_ratio=True``, additionally returns the
    worst-case (minimum) cancellation ratio across the (N_x, N_y)
    entries — a scalar in (0, 1]. Direct-enum entries always have
    ratio 1.0; the worst ratio comes from the safe-Möbius submatrix.
    If no safe pairs exist, the worst ratio is 1.0.

    ``truncation_sigmas`` is honoured in every kernel-evaluation
    branch (r=1 abs, r>=2 abs safe, r>=2 abs unsafe via
    :func:`_batched_direct_enum_abs`, and rel-per via
    :func:`_rel_inner_batched`): kernel entries whose
    underlying squared distance exceeds the truncation cutoff are
    zeroed without evaluating ``np.exp``. ``None`` resolves to the
    global default ``mpt.get_default('truncation_sigmas')``.

    ``prune_zero_weight_events`` (default ``True``) drops events with
    column-wise weight identically zero (treating NaN as missing)
    before dispatching to any sub-helper. Such events contribute zero
    to every kernel entry, so the result is mathematically
    unchanged; the saving is wall-clock — the einsum / batched
    Möbius contraction operates on smaller matrices. Combines
    naturally with the global ``truncation_sigmas`` since
    :func:`mpt.weight_events` hard-zeros the factor outside the
    cutoff. Result is scattered back into the full (N_x, N_y) output
    shape with zeros in the dropped rows / columns. Set
    ``prune_zero_weight_events=False`` to bypass.
    """
    from .._mobius import inner_product_orbit_pw_batched
    from .._defaults import get_default

    if truncation_sigmas is None:
        truncation_sigmas = get_default('truncation_sigmas')

    K_x_max, N_x = Px.shape
    K_y_max, N_y = Py.shape

    # --- Zero-weight-event pruning (auto, before dispatch) ---
    # An event contributes zero to every output entry iff its weight
    # column is identically zero in this attribute (NaN entries are
    # missing slots — equivalent to zero in the IP). Drop such events,
    # recurse on the smaller matrices, scatter the result back.
    if prune_zero_weight_events and (N_x > 0) and (N_y > 0):
        # nanmax over a column returns NaN only when ALL entries are
        # NaN (which is an invalid event); for any partial-NaN column
        # it returns the max over the non-NaN slots. Compare > 0 to
        # find columns with at least one positive-magnitude slot.
        with np.errstate(invalid='ignore'):
            col_max_x = np.nanmax(np.abs(Wx), axis=0)
            col_max_y = np.nanmax(np.abs(Wy), axis=0)
        keep_x = np.isfinite(col_max_x) & (col_max_x > 0.0)
        keep_y = np.isfinite(col_max_y) & (col_max_y > 0.0)
        if not (keep_x.all() and keep_y.all()):
            if not (keep_x.any() and keep_y.any()):
                # Every event on at least one side has zero weight;
                # the IP is the all-zero matrix.
                result = np.zeros((N_x, N_y), dtype=np.float64)
                if return_cancellation_ratio:
                    return result, 1.0
                return result
            sub = _ma_per_attr_inner_matrix(
                Px[:, keep_x], Wx[:, keep_x],
                Py[:, keep_y], Wy[:, keep_y],
                sigma, r, is_rel, is_per, period,
                return_cancellation_ratio=return_cancellation_ratio,
                truncation_sigmas=truncation_sigmas,
                prune_zero_weight_events=False,   # avoid infinite recursion
                wrap=wrap,
            )
            if return_cancellation_ratio:
                sub_ip, sub_ratio = sub
            else:
                sub_ip = sub
            result = np.zeros((N_x, N_y), dtype=np.float64)
            result[np.ix_(np.where(keep_x)[0], np.where(keep_y)[0])] = sub_ip
            if return_cancellation_ratio:
                return result, sub_ratio
            return result

    # --- r = 1: direct kernel sum, zero-pad fine (no cancellation) ---
    if r == 1:
        Px_, Wx_, Py_, Wy_ = _zero_pad_nan(Px, Wx, Py, Wy)
        prefactor = sigma * np.sqrt(np.pi)
        # Memory: each of diffs, diffs**2, K_tens is
        # (K_x_max, chunk_N_x, K_y_max, N_y) * 8 bytes. Up to ~3 live
        # arrays during evaluation; budget accordingly.
        per_row_bytes = 3 * K_x_max * K_y_max * N_y * 8
        mem_limit = kernel_chunk_bytes_resolved()
        chunk_N_x = max(1, min(N_x, mem_limit // max(per_row_bytes, 1)))

        # Abs-per full-image: build the pairwise kernel via the 1-D
        # wrapped Gaussian (overlap convention, exponent_denominator=4).
        # Single-image opt-in reduces to the nearest image; the pre-v3
        # code did the reduction unconditionally.
        abs_per_full_image = is_per and str(wrap) == 'full-image'
        if abs_per_full_image:
            from .._wrapped_kernel import wrapped_gaussian_1d
            trunc_eff = float(get_default('truncation_sigmas')
                              if truncation_sigmas is None
                              else truncation_sigmas)

        if chunk_N_x >= N_x:
            # Fast path: single shot.
            diffs = Px_[:, :, None, None] - Py_[None, None, :, :]
            if abs_per_full_image:
                K_tens = wrapped_gaussian_1d(
                    diffs, sigma, period, trunc_eff,
                    exponent_denominator=4,
                )
            else:
                if is_per:
                    diffs = diffs - period * np.floor(diffs / period + 0.5)
                K_tens = _trunc_kernel_exp(diffs ** 2, sigma,
                                            truncation_sigmas)
            out = np.einsum(
                'xn,xnym,ym->nm', Wx_, K_tens, Wy_, optimize=True,
            )
            result = out * prefactor
        else:
            # Chunked path: process N_x in chunks.
            result = np.empty((N_x, N_y), dtype=np.float64)
            for n_start in range(0, N_x, chunk_N_x):
                n_end = min(n_start + chunk_N_x, N_x)
                Px_chunk = Px_[:, n_start:n_end]
                Wx_chunk = Wx_[:, n_start:n_end]
                diffs = Px_chunk[:, :, None, None] - Py_[None, None, :, :]
                if abs_per_full_image:
                    K_tens = wrapped_gaussian_1d(
                        diffs, sigma, period, trunc_eff,
                        exponent_denominator=4,
                    )
                else:
                    if is_per:
                        diffs = diffs - period * np.floor(diffs / period + 0.5)
                    K_tens = _trunc_kernel_exp(diffs ** 2, sigma,
                                                truncation_sigmas)
                out_chunk = np.einsum(
                    'xn,xnym,ym->nm', Wx_chunk, K_tens, Wy_, optimize=True,
                )
                result[n_start:n_end, :] = out_chunk * prefactor

        if return_cancellation_ratio:
            return result, 1.0
        return result

    # --- r >= 2 rel: batched translation-grid integration, zero-pad ---
    if is_rel:
        Px_, Wx_, Py_, Wy_ = _zero_pad_nan(Px, Wx, Py, Wy)
        return _rel_inner_batched(
            Px_, Wx_, Py_, Wy_, sigma, r, is_per, period,
            return_cancellation_ratio=return_cancellation_ratio,
            truncation_sigmas=truncation_sigmas,
        )

    # --- r >= 2 abs: hybrid safe/unsafe partition ---

    K_MARGIN_MIN = _ORBIT_K_MINUS_R_MIN

    # Per-event K_eff (count of non-NaN slots), per side.
    K_eff_x = np.sum(~(np.isnan(Px) | np.isnan(Wx)), axis=0)   # (N_x,)
    K_eff_y = np.sum(~(np.isnan(Py) | np.isnan(Wy)), axis=0)   # (N_y,)

    safe_x_mask = (K_eff_x - r) >= K_MARGIN_MIN
    safe_y_mask = (K_eff_y - r) >= K_MARGIN_MIN
    safe_x_idx = np.where(safe_x_mask)[0]
    unsafe_x_idx = np.where(~safe_x_mask)[0]
    safe_y_idx = np.where(safe_y_mask)[0]
    unsafe_y_idx = np.where(~safe_y_mask)[0]

    out = np.zeros((N_x, N_y), dtype=np.float64)
    worst_ratio = 1.0

    # --- Safe x Safe submatrix: vectorised batched Möbius method ---
    if safe_x_idx.size > 0 and safe_y_idx.size > 0:
        Px_s = Px[:, safe_x_idx]
        Wx_s = Wx[:, safe_x_idx]
        Py_s = Py[:, safe_y_idx]
        Wy_s = Wy[:, safe_y_idx]
        # Within-safe zero-pad (K still varies per event in safe group).
        Px_s, Wx_s, Py_s, Wy_s = _zero_pad_nan(Px_s, Wx_s, Py_s, Wy_s)

        N_xs = safe_x_idx.size
        N_ys = safe_y_idx.size
        prefactor = (sigma * np.sqrt(np.pi)) ** r

        # Sparse-orbit fast path: when the slot kernel is large and the
        # (non-periodic) slots are well-separated, a spatially-culled
        # per-pair orbit beats the dense batched contraction. Gate on a
        # cheap density probe from one representative safe pair.
        use_sparse = False
        if (not is_per) and r >= 2 \
                and K_x_max * K_y_max >= _ORBIT_SPARSE_MIN_KERNEL:
            vx0 = Wx_s[:, 0] != 0.0
            vy0 = Wy_s[:, 0] != 0.0
            K0 = _build_sparse_kernel_abs(
                Px_s[vx0, 0], Py_s[vy0, 0], sigma, truncation_sigmas)
            if K0.nnz <= _ORBIT_SPARSE_MAX_DENSITY * K_x_max * K_y_max:
                use_sparse = True

        if use_sparse:
            flat, wr = _orbit_safe_submatrix_sparse(
                Px_s, Wx_s, Py_s, Wy_s, sigma, r, prefactor,
                truncation_sigmas, return_cancellation_ratio)
            if return_cancellation_ratio:
                worst_ratio = min(worst_ratio, float(wr))
            out[np.ix_(safe_x_idx, safe_y_idx)] = flat.reshape(N_xs, N_ys)

        else:
            # Memory: each of diffs, diffs**2, K_tens is
            # (K_x_max, chunk_N_xs, K_y_max, N_ys) * 8 bytes; ~3 live.
            # K_pairs reshape adds N_pairs * K_x_max * K_y_max * 8.
            per_row_bytes = 4 * K_x_max * K_y_max * N_ys * 8
            mem_limit = kernel_chunk_bytes_resolved()
            chunk_N_xs = max(1, min(N_xs, mem_limit // max(per_row_bytes, 1)))

            if chunk_N_xs >= N_xs:
                # Fast path: single shot.
                diffs = Px_s[:, :, None, None] - Py_s[None, None, :, :]
                # Abs-per full-image: 1-D wrapped Gaussian kernel per
                # slot (overlap convention, exponent_denominator=4).
                # Single-image opt-in reduces to the nearest image; the
                # pre-v3 code did the reduction unconditionally.
                if is_per and str(wrap) == 'full-image':
                    from .._wrapped_kernel import wrapped_gaussian_1d
                    trunc_eff = float(get_default('truncation_sigmas')
                                       if truncation_sigmas is None
                                       else truncation_sigmas)
                    K_tens = wrapped_gaussian_1d(
                        diffs, sigma, period, trunc_eff,
                        exponent_denominator=4,
                    )
                else:
                    if is_per:
                        diffs = diffs - period * np.floor(
                            diffs / period + 0.5)
                    K_tens = _trunc_kernel_exp(
                        diffs ** 2, sigma, truncation_sigmas)
                K_pairs = np.transpose(K_tens, (1, 3, 0, 2)).reshape(
                    N_xs * N_ys, K_x_max, K_y_max,
                )
                w_A_pairs = np.broadcast_to(
                    Wx_s.T[:, None, :], (N_xs, N_ys, K_x_max),
                ).reshape(N_xs * N_ys, K_x_max)
                w_B_pairs = np.broadcast_to(
                    Wy_s.T[None, :, :], (N_xs, N_ys, K_y_max),
                ).reshape(N_xs * N_ys, K_y_max)

                if return_cancellation_ratio:
                    flat, ratios = inner_product_orbit_pw_batched(
                        K_pairs, w_A_pairs, w_B_pairs, r,
                        prefactor=prefactor,
                        return_cancellation_ratio=True,
                    )
                    worst_ratio = min(worst_ratio, float(np.min(ratios)))
                else:
                    flat = inner_product_orbit_pw_batched(
                        K_pairs, w_A_pairs, w_B_pairs, r,
                        prefactor=prefactor,
                    )
                out[np.ix_(safe_x_idx, safe_y_idx)] = flat.reshape(N_xs, N_ys)
            else:
                # Chunked path: process safe_x_idx in chunks of chunk_N_xs.
                safe_flat = np.empty((N_xs, N_ys), dtype=np.float64)
                for n_start in range(0, N_xs, chunk_N_xs):
                    n_end = min(n_start + chunk_N_xs, N_xs)
                    n_chunk = n_end - n_start
                    Px_chunk = Px_s[:, n_start:n_end]
                    Wx_chunk = Wx_s[:, n_start:n_end]
                    diffs = Px_chunk[:, :, None, None] - Py_s[None, None, :, :]
                    if is_per and str(wrap) == 'full-image':
                        from .._wrapped_kernel import wrapped_gaussian_1d
                        trunc_eff = float(
                            get_default('truncation_sigmas')
                            if truncation_sigmas is None
                            else truncation_sigmas)
                        K_tens = wrapped_gaussian_1d(
                            diffs, sigma, period, trunc_eff,
                            exponent_denominator=4,
                        )
                    else:
                        if is_per:
                            diffs = diffs - period * np.floor(
                                diffs / period + 0.5)
                        K_tens = _trunc_kernel_exp(
                            diffs ** 2, sigma, truncation_sigmas)
                    K_pairs = np.transpose(K_tens, (1, 3, 0, 2)).reshape(
                        n_chunk * N_ys, K_x_max, K_y_max,
                    )
                    w_A_pairs = np.broadcast_to(
                        Wx_chunk.T[:, None, :], (n_chunk, N_ys, K_x_max),
                    ).reshape(n_chunk * N_ys, K_x_max)
                    w_B_pairs = np.broadcast_to(
                        Wy_s.T[None, :, :], (n_chunk, N_ys, K_y_max),
                    ).reshape(n_chunk * N_ys, K_y_max)

                    if return_cancellation_ratio:
                        flat, ratios = inner_product_orbit_pw_batched(
                            K_pairs, w_A_pairs, w_B_pairs, r,
                            prefactor=prefactor,
                            return_cancellation_ratio=True,
                        )
                        worst_ratio = min(worst_ratio, float(np.min(ratios)))
                    else:
                        flat = inner_product_orbit_pw_batched(
                            K_pairs, w_A_pairs, w_B_pairs, r,
                            prefactor=prefactor,
                        )
                    safe_flat[n_start:n_end, :] = flat.reshape(n_chunk, N_ys)
                out[np.ix_(safe_x_idx, safe_y_idx)] = safe_flat

    # --- Pairs involving any unsafe event: K-grouped batched direct ---
    # All pairs not in (safe_x, safe_y) flow through ordered-r-tuple
    # direct enumeration. Was previously a Python double-loop
    # (one ``_inner_product_direct_abs`` call per pair); for
    # variable-K_a workloads with many unsafe events this dominated
    # the runtime by 10–100× over the actual computation.
    #
    # K_a grouping: partition the unsafe-involved event-index union
    # by K_eff value per side, then batch direct enumeration per
    # (K_eff_x, K_eff_y) sub-block. Within a sub-block, every event
    # shares an ordered-r-tuple shape (nJ = K_eff! / (K_eff - r)!),
    # so the IP matrix can be computed as a single contracted
    # tensor op. This removes the Python per-pair overhead entirely.
    #
    # Coverage: (unsafe_x, all_y) ∪ (safe_x, unsafe_y) covers every
    # pair where at least one side is unsafe, without double counting.
    needed_x_idx = np.concatenate([unsafe_x_idx, safe_x_idx]) \
        if unsafe_x_idx.size > 0 else np.array([], dtype=np.intp)
    needed_y_idx_full = np.arange(N_y)
    # Split needed pairs into two coverage zones to mirror the
    # inline-method structure exactly, preserving fill ordering.
    _ma_fill_direct_enum_groups(
        out, Px, Wx, Py, Wy,
        unsafe_x_idx, np.arange(N_y),
        K_eff_x, K_eff_y, sigma, r, is_per, period,
        truncation_sigmas=truncation_sigmas,
    )
    if unsafe_y_idx.size > 0 and safe_x_idx.size > 0:
        _ma_fill_direct_enum_groups(
            out, Px, Wx, Py, Wy,
            safe_x_idx, unsafe_y_idx,
            K_eff_x, K_eff_y, sigma, r, is_per, period,
            truncation_sigmas=truncation_sigmas,
        )

    if return_cancellation_ratio:
        return out, worst_ratio
    return out


def _ma_fill_direct_enum_groups(
    out, Px, Wx, Py, Wy, x_idx, y_idx,
    K_eff_x, K_eff_y, sigma, r, is_per, period,
    *, truncation_sigmas=None,
):
    """K-grouped batched direct-enum fill into ``out`` for a rectangle
    of (x_idx, y_idx) pairs.

    Partitions ``x_idx`` by K_eff_x value and ``y_idx`` by K_eff_y
    value, then computes each (K_x_val, K_y_val) sub-block as a single
    vectorised tensor contraction. Output entries at (x_idx[i],
    y_idx[j]) are filled in place.

    ``truncation_sigmas`` is forwarded to
    :func:`_batched_direct_enum_abs`; ``None`` resolves to the
    global default.

    No-op if either side is empty.
    """
    if x_idx.size == 0 or y_idx.size == 0:
        return

    # Unique K_eff values present on each side (within the index sets).
    unique_K_x = np.unique(K_eff_x[x_idx])
    unique_K_y = np.unique(K_eff_y[y_idx])

    for K_x_val in unique_K_x:
        x_grp = x_idx[K_eff_x[x_idx] == K_x_val]
        if x_grp.size == 0 or int(K_x_val) < r:
            # K < r: ordered r-tuple set is empty; IP = 0.
            continue
        # Pack non-NaN slots to the top of each group column. The
        # build_exp_tens convention has NaN already at the bottom, so
        # in the common case this is a memory-cheap slice; in the
        # general case _pack_nan_top handles arbitrary NaN positions.
        Px_grp, Wx_grp = _pack_nan_top(Px[:, x_grp], Wx[:, x_grp])
        Px_grp = Px_grp[:int(K_x_val), :]
        Wx_grp = Wx_grp[:int(K_x_val), :]
        for K_y_val in unique_K_y:
            y_grp = y_idx[K_eff_y[y_idx] == K_y_val]
            if y_grp.size == 0 or int(K_y_val) < r:
                continue
            Py_grp, Wy_grp = _pack_nan_top(Py[:, y_grp], Wy[:, y_grp])
            Py_grp = Py_grp[:int(K_y_val), :]
            Wy_grp = Wy_grp[:int(K_y_val), :]
            sub_ip = _batched_direct_enum_abs(
                Px_grp, Wx_grp, Py_grp, Wy_grp,
                sigma, r, is_per, period,
                truncation_sigmas=truncation_sigmas,
            )
            out[np.ix_(x_grp, y_grp)] = sub_ip


def _pack_nan_top(P, W):
    """Pack non-NaN slots to the top of each column.

    Returns ``(P_packed, W_packed)`` of the same shape, where for each
    column ``n`` the first ``K_eff[n]`` rows are the valid slots
    (preserving their original order) and the rest are NaN. The
    ``build_exp_tens`` convention already places NaN at the bottom, in
    which case this is mathematically a no-op (still copies for
    cleanliness). Per-event packing handles user-constructed densities
    with arbitrary NaN positions.
    """
    K, N = P.shape
    P_packed = np.full_like(P, np.nan)
    W_packed = np.full_like(W, np.nan)
    for n in range(N):
        valid = ~(np.isnan(P[:, n]) | np.isnan(W[:, n]))
        k = int(valid.sum())
        if k == 0:
            continue
        P_packed[:k, n] = P[valid, n]
        W_packed[:k, n] = W[valid, n]
    return P_packed, W_packed


def _batched_direct_enum_abs(
    Px_group, Wx_group, Py_group, Wy_group,
    sigma, r, is_per, period,
    *, truncation_sigmas=None,
):
    """Batched direct r-tuple enumeration IP for groups at fixed K_x, K_y.

    Vectorised replacement for repeated calls to
    :func:`_inner_product_direct_abs` when every event in
    ``Px_group`` has the same ``K_x = K_eff_x`` and every event in
    ``Py_group`` has the same ``K_y = K_eff_y`` (no NaN within the
    first K rows of either side).

    Inputs
    ------
    Px_group : (K_x, N_x) ndarray
        Slot positions, no NaN.
    Wx_group : (K_x, N_x) ndarray
        Slot weights, no NaN.
    Py_group, Wy_group : (K_y, N_y) ndarrays
        Same for Y side.
    sigma, r, is_per, period
        Group parameters.
    truncation_sigmas : float, optional
        Kernel-truncation cutoff in σ units. Kernel entries whose
        squared distance exceeds the cutoff are zeroed without
        evaluating ``np.exp``. ``None`` resolves to the global default
        ``mpt.get_default('truncation_sigmas')``.

    Returns
    -------
    ip : (N_x, N_y) ndarray
        Inner-product matrix (no Möbius alternating sum; exact for any
        K_x, K_y >= r).
    """
    from .._defaults import get_default

    if truncation_sigmas is None:
        truncation_sigmas = get_default('truncation_sigmas')

    K_x, N_x = Px_group.shape
    K_y, N_y = Py_group.shape

    if K_x < r or K_y < r:
        return np.zeros((N_x, N_y), dtype=np.float64)

    if r == 1:
        diffs = Px_group[:, :, None, None] - Py_group[None, None, :, :]
        if is_per:
            diffs = diffs - period * np.floor(diffs / period + 0.5)
        K_mat = _trunc_kernel_exp(diffs ** 2, sigma, truncation_sigmas)
        ip = np.einsum(
            'xn,xnym,ym->nm', Wx_group, K_mat, Wy_group, optimize=True,
        )
        return ip * (sigma * np.sqrt(np.pi))

    # --- r >= 2: enumerate ordered r-tuple indices ---
    from itertools import permutations
    idx_x = np.array(list(permutations(range(K_x), r)),
                     dtype=np.intp)        # (nJ_x, r)
    idx_y = np.array(list(permutations(range(K_y), r)),
                     dtype=np.intp)        # (nJ_y, r)
    nJ_x = idx_x.shape[0]                  # K_x! / (K_x - r)!
    nJ_y = idx_y.shape[0]

    # Gather tuple slot positions and weights per event. The fancy
    # index Px_group[idx_x.T, :] has shape (r, nJ_x, N_x); we want
    # U_x of shape (r, N_x, nJ_x) and Wj_x of shape (N_x, nJ_x).
    U_x = Px_group[idx_x.T, :].transpose(0, 2, 1)
    U_y = Py_group[idx_y.T, :].transpose(0, 2, 1)
    Wj_x = np.prod(Wx_group[idx_x.T, :], axis=0).T  # (N_x, nJ_x)
    Wj_y = np.prod(Wy_group[idx_y.T, :], axis=0).T  # (N_y, nJ_y)

    # Memory estimate: the difference tensor is (r, N_x, nJ_x, N_y, nJ_y).
    # For unsafe events (K_eff in {r, r+1}), nJ_x = r! or (r+1)!/(1!),
    # which is small. For r=3, K=4 -> nJ=24; r=4, K=5 -> nJ=120. Even
    # with N_x = N_y = 100 this is <100 MB at worst. No chunking needed
    # in the unsafe regime. Document so future use on safe-K paths
    # adds a chunking guard.
    diffs = U_x[:, :, :, None, None] - U_y[:, None, None, :, :]
    if is_per:
        diffs = diffs - period * np.floor(diffs / period + 0.5)
    Q = np.sum(diffs ** 2, axis=0)                  # (N_x, nJ_x, N_y, nJ_y)
    K_mat = _trunc_kernel_exp(Q, sigma, truncation_sigmas)

    ip = np.einsum(
        'xj,xjyk,yk->xy', Wj_x, K_mat, Wj_y, optimize=True,
    )
    return ip * (sigma * np.sqrt(np.pi)) ** r


def _zero_pad_nan(Px, Wx, Py, Wy):
    """Replace NaN entries in P / W with 0 (zero-weight padding).

    Returns new arrays (does not mutate inputs). The Möbius method's weighted
    contractions read ``w_i^m``, so a zero-weight slot kills any Möbius
    term involving that slot regardless of the corresponding p value
    — mathematically equivalent to per-event truncation.
    """
    nan_x = np.isnan(Px) | np.isnan(Wx)
    if nan_x.any():
        Px = np.where(nan_x, 0.0, Px)
        Wx = np.where(nan_x, 0.0, Wx)
    nan_y = np.isnan(Py) | np.isnan(Wy)
    if nan_y.any():
        Py = np.where(nan_y, 0.0, Py)
        Wy = np.where(nan_y, 0.0, Wy)
    return Px, Wx, Py, Wy


def _rel_per_inner_sparse(Px, Wx, Py, Wy, sigma, r, period, u_grid, du,
                          cutoff, return_cancellation_ratio):
    """Periodic relative inner products via the circular sparse route.

    Per pair: the B-side sorted-tripled arrays are prepared once, then
    each u-node builds its circular sparse kernel and runs the sparse
    orbit collapse. Values match the dense slab route exactly (the
    circular window retains precisely the entries the truncated dense
    kernel keeps, and zero-weight slots contribute zero to every orbit
    term), and the mass-aware pair ratio |sum_u F_u| / sum_u max|term_u|
    matches the dense diagnostic. The normalisation tail is shared with
    the dense route.
    """
    from .._mobius import inner_product_orbit_sparse
    K_x, N_x = Px.shape
    K_y, N_y = Py.shape
    integral = np.zeros((N_x, N_y), dtype=np.float64)
    worst = 1.0
    y_pre = []
    for j in range(N_y):
        vy = Wy[:, j] != 0.0
        c3, j3 = _rel_per_sparse_prep(Py[vy, j], period)
        y_pre.append((c3, j3, Py[vy, j], Wy[vy, j]))
    for i in range(N_x):
        vx = Wx[:, i] != 0.0
        pxi = Px[vx, i]
        wxi = Wx[vx, i]
        for j in range(N_y):
            c3, j3, pyj, wyj = y_pre[j]
            F = 0.0
            M = 0.0
            for u in u_grid:
                Ks = _build_sparse_kernel_rel_per(
                    pxi, c3, j3, pyj, sigma, cutoff, period, float(u))
                if return_cancellation_ratio:
                    v, _, m = inner_product_orbit_sparse(
                        Ks, wxi, wyj, r,
                        return_cancellation_ratio=True,
                        return_term_mass=True)
                    M += m
                else:
                    v = inner_product_orbit_sparse(Ks, wxi, wyj, r)
                F += v
            integral[i, j] = F * du
            if return_cancellation_ratio:
                ratio = abs(F) / M if M > 0 else 1.0
                if ratio < worst:
                    worst = ratio
    c = sigma * np.sqrt(2 * np.pi / r)
    out = (sigma * np.sqrt(np.pi)) ** r * integral / c ** 2
    if return_cancellation_ratio:
        return out, worst
    return out


#: Enables the spectral (Fourier) branch of the relative-mode
#: per-attribute inner matrix. Module-level for tests.
_SPECTRAL_IP_ENABLED = True


#: Mode-cutoff width in sigmas for the spectral inner product. The
#: block spectra are truncated where the Gaussian envelope reaches
#: machine precision rather than the requested accuracy floor: the
#: alternating partition sum amplifies per-term error, and the mode
#: count grows only as sqrt(log(1/eps)).
_SPECTRAL_IP_MODE_SIGMAS = 8.6


#: Largest mode-grid point count admitted, per side. The grid is
#: (r-1)-dimensional, so this is what keeps r = 4 at small sigma/P out
#: of memory trouble; above it the branch stands down.
_SPECTRAL_IP_MAX_POINTS = 4_000_000


#: Cost-gate constant: the branch is taken while the mode grid stays
#: below this multiple of K^2 * n_events_x * n_events_y. It is
#: per-language --- the MATLAB twin uses 282 --- because the branch and
#: the grid sit at different points on the cost curve in each language:
#: the vectorised NumPy/BLAS spectral branch runs about twice as fast as
#: the MATLAB port while the two grid contractions are comparable, so
#: the branch is worth taking on more shapes here and clears a higher
#: bar. Both routes compute the same full-image measure, so which one
#: runs affects only time, not the answer, and the constant is
#: recalibrated per language rather than matched.
#:
#: Calibrated by routing regret --- the wall time actually paid against
#: an oracle that always picks the faster route --- over measured cells
#: spanning both periodic modes, r = 2..4, K = 4..30, event counts 1..16
#: and sigma/P from 0.002 to 0.2, scored by the geometric mean of the
#: per-cell regret so every cell counts equally rather than a few
#: huge-grid cells dominating a wall-time sum. The value below is the
#: optimum and also has the best worst case (2.03x); the plateau from
#: roughly 1000 to 1780 is flat within 0.5% on the geometric mean.
#:
#: An earlier value of 3160 was fitted before the per-event index hoist
#: and the self inner-product shortcut made the branch cheaper, and on a
#: grid whose smallest sigma/P was 0.0125. Re-measured on the faster
#: branch across the full sigma/P range --- including the sharp-kernel
#: cells at 0.002 to 0.005 where the branch loses, which were expected
#: to pull the optimum back up and did not --- 3160 gives 1.016x of
#: oracle against 1.007x here, and declines 13 winners to avoid 4 losers
#: where this value declines far fewer. The low-sigma/P columns are why
#: the constant is not lower still: at sigma/P = 0.002 the branch is
#: several times slower and the gate must stay conservative enough to
#: decline it.
#:
#: The form was selected, not assumed. Every subset of
#: {log gridSize, log K, log N, log N_u, log P(r), log B(r), isPer} was
#: fitted as a log-linear model of log(t_grid / t_spectral) and scored
#: by BIC and by cross-validated regret over 40 random halves. No fitted
#: subset beat this form, whose exponents (1, -2, -2) are pinned by the
#: cost algebra rather than estimated. BIC's own optimum decides worse
#: than most of the family, because likelihood weights cells far from
#: the boundary where the decision is easy, while the decision depends
#: only on the sign near zero. Offset families (a separate constant per
#: isPer, per r, or per (r, isPer)) all fit the full data better and
#: generalise worse.
_SPECTRAL_IP_COST_C = 1100.0


def _rel_per_image_count(sigma, period, truncation_sigmas):
    """Number of periodic images each side needed for the relative-periodic
    kernel to reach the caller's own accuracy floor.

    The relative-periodic inner product marginalises a rigid common shift.
    Taking that average over a kernel that carries every periodic image
    yields the lattice-sum (full-image) measure exactly; taking it over a
    nearest-image kernel yields a different measure, which departs from it
    as ``sigma/period`` grows. Summing images restores the identity, and
    the count needed is set by the accuracy already being asked for
    elsewhere rather than by a fixed constant.

    After the nearest-image reduction the difference satisfies
    ``|d| <= period/2``, so the image at offset ``l`` is bounded by
    ``exp(-((|l| - 1/2) period)^2 / (4 sigma^2))``. Requiring the first
    omitted image to fall below the kernel-value floor gives

        n > 2 (sigma/period) sqrt(ln(1/tol)) - 1/2 .

    Returns 0 whenever the nearest image alone already meets the floor,
    which is the case throughout the range musical work normally occupies
    (0 up to sigma/period ~ 0.05 at the default truncation, and the two
    measures agree there to 2.5e-13). In that regime the full-image
    kernel and the nearest-image kernel are the same object and this
    costs nothing.
    """
    from .._defaults import truncation_floor

    if not (np.isfinite(sigma) and np.isfinite(period)) or period <= 0.0:
        return 0
    tol = truncation_floor(truncation_sigmas)
    if not (0.0 < tol < 1.0):
        return 0
    n = 2.0 * (sigma / period) * np.sqrt(np.log(1.0 / tol)) - 0.5
    if not np.isfinite(n) or n <= 0.0:
        return 0
    return int(np.ceil(n))


def _spectral_rel_inner_matrix(Px, Wx, Py, Wy, sigma, r, is_per, period):
    """Relative-mode per-attribute inner matrix by the spectral form.

    The relative inner product is the integral of the absolute inner
    product under a rigid diagonal shift, and transforming along that
    shift constrains each partition's total mode index to zero. Every
    partition therefore contributes a product of per-block spectra
    evaluated at block-summed frequencies, so each event carries a
    single spectrum

        S_n(xi) = sum_pi mu(pi) prod_B A_{|B|}(eta_B),
        A_m(eta) = sum_i w_{i,n}^m exp(-i eta p_{i,n}),

    and the matrix over event pairs is the Gram matrix of those
    spectra against the envelope exp(-sigma^2 sum_s xi_s^2), scaled by

        C_r = r * (sigma^2 * dxi)^(r-1).

    The spectra are per-event objects, so the K-dependence is paid once
    per event rather than once per pair, and the per-pair cost is
    independent of K. Frequencies are exact on the circle
    (xi_m = 2 pi m / P, valid at any sigma/P, since the wrapped-Gaussian
    coefficients are closed form); on the line they are spaced
    2 pi / L for an embedding period L covering both sides' spans plus a
    truncation margin. Hermitian symmetry cuts the working sets in half
    on two sides: (i) env is even under xi -> -xi and
    S_n(-xi) = conj(S_n(xi)) since the underlying Gaussian mixture is
    real, so only the DC point plus one point per (xi, -xi) pair is
    evaluated on the grid, with the off-DC envelope doubled to account
    for the conjugate partner; (ii) real event weights give
    A_m(-eta) = conj(A_m(eta)), so per-event phases are evaluated only
    on non-negative modes and the full A_m table is assembled by
    reflection and conjugation. The first halving cuts the partition
    loop and final matmul; the second halves the per-event phase
    matmul, which dominates at r = 2 where the partition loop is
    trivial. Returns ``None`` when the mode grid would exceed
    ``_SPECTRAL_IP_MAX_POINTS``, so the caller falls through.
    """
    from .._mobius import get_set_partitions_with_mobius

    Px = np.asarray(Px, dtype=np.float64)
    Py = np.asarray(Py, dtype=np.float64)
    Wx = np.asarray(Wx, dtype=np.float64)
    Wy = np.asarray(Wy, dtype=np.float64)

    def _span(P_, W_):
        live = np.abs(W_) > 0.0
        if not live.any():
            return 0.0, 0.0
        vals = P_[live]
        return float(vals.min()), float(vals.max())

    if is_per:
        L = float(period)
    else:
        lo_x, hi_x = _span(Px, Wx)
        lo_y, hi_y = _span(Py, Wy)
        L = ((hi_x - lo_x) + (hi_y - lo_y)
             + 2.0 * (_SPECTRAL_IP_MODE_SIGMAS + 2.0) * sigma)
        if not np.isfinite(L) or L <= 0.0:
            return None
    dxi = 2.0 * np.pi / L
    M = int(np.ceil(_SPECTRAL_IP_MODE_SIGMAS / np.sqrt(2.0)
                    * L / (2.0 * np.pi * sigma))) + 2
    grid_size = (2 * M + 1) ** (r - 1)
    if grid_size > _SPECTRAL_IP_MAX_POINTS:
        return None
    # Cost gate: the grid path pays K^2 per event pair, the branch pays
    # the mode grid. Decline where the mode grid is not repaid.
    k_slots = float(Px.shape[0])
    n_pairs = float(Px.shape[1]) * float(Py.shape[1])
    if grid_size > _SPECTRAL_IP_COST_C * k_slots ** 2 * n_pairs:
        return None
    axes = [np.arange(-M, M + 1, dtype=np.int64)] * (r - 1)
    grids = np.meshgrid(*axes, indexing='ij')
    xs = [g.ravel() for g in grids]
    del grids
    xs.append(-sum(xs))
    a = (dxi * sigma) ** 2
    quad = np.zeros(xs[0].size, dtype=np.float64)
    for x in xs:
        quad += x.astype(np.float64) ** 2
    env = np.exp(-a * quad)
    keep = env > 1e-18
    xs = [x[keep] for x in xs]
    env = env[keep]
    n_pts = env.size
    if n_pts == 0:
        return np.zeros((Px.shape[1], Py.shape[1]), dtype=np.float64)

    # Hermitian symmetry: env is even under xi -> -xi (it depends only
    # on sum_s xi_s^2), and each event's spectrum satisfies
    # S_n(-xi) = conj(S_n(xi)) because the underlying Gaussian mixture
    # is real-valued. Grid points therefore come in (xi, -xi) pairs
    # whose contributions to Re((SX * env) @ conj(SY).T) are equal, so
    # keeping only the DC point plus one point per pair, and doubling
    # the off-DC envelope, reproduces the full sum. Lex-positivity is
    # marked by a scalar key with key(-m) = -key(m); base = 2M + 1
    # keeps int64 range comfortable across the parameter regime the
    # branch operates in.
    base = np.int64(2 * M + 1)
    key = np.zeros(n_pts, dtype=np.int64)
    for i in range(r - 1):
        key += xs[i] * base ** i
    is_dc = key == 0
    keep_h = (key > 0) | is_dc
    xs = [x[keep_h] for x in xs]
    env = env[keep_h] * np.where(is_dc[keep_h], 1.0, 2.0)
    n_pts = env.size

    W_ax = r * M
    # Phase-side Hermitian: since W_[:, n] is real, A_m(-eta) = conj(A_m(eta)),
    # so phases are only evaluated on non-negative modes and the full
    # A[m] table is assembled by reflecting the positive half and
    # conjugating. Halves the O(K * W_ax) phase matmul, which dominates
    # the per-event cost at r = 2 where the partition loop is trivial.
    ax_modes_pos = dxi * np.arange(0, W_ax + 1)
    partitions = get_set_partitions_with_mobius(r)

    # The gather indices ``eta + W_ax`` and block sizes depend only on the
    # mode grid, not on the event, so they are built once here rather than
    # rebuilt inside the event loop. At r = 4 rebuilding them cost several
    # times a single assembly, so hoisting is the dominant saving on the
    # branch's most expensive shapes. Each entry is (mu, [(block_size,
    # gather_index_vector), ...]).
    part_plan = []
    for blocks, mu in partitions:
        gathers = []
        for B in blocks:
            idx = list(B)
            eta = xs[idx[0]].copy()
            for sl in idx[1:]:
                eta = eta + xs[sl]
            gathers.append((len(B), eta + W_ax))
        part_plan.append((float(mu), gathers))

    def _spectra(P_, W_):
        out = np.empty((P_.shape[1], n_pts), dtype=np.complex128)
        for n in range(P_.shape[1]):
            phase_pos = np.exp(-1j * np.outer(ax_modes_pos, P_[:, n]))
            A = {}
            for m in range(1, r + 1):
                A_pos_m = phase_pos @ (W_[:, n] ** m)
                Am = np.empty(2 * W_ax + 1, dtype=np.complex128)
                # index W_ax..2*W_ax carries modes 0..W_ax (positive half);
                # index 0..W_ax-1 carries modes -W_ax..-1 by conjugation.
                Am[W_ax:] = A_pos_m
                Am[:W_ax] = np.conj(A_pos_m[1:])[::-1]
                A[m] = Am
            tot = np.zeros(n_pts, dtype=np.complex128)
            for mu, gathers in part_plan:
                term = np.full(n_pts, mu, dtype=np.complex128)
                for blk_size, eidx in gathers:
                    term *= A[blk_size][eidx]
                tot += term
            out[n] = tot
        return out

    SX = _spectra(Px, Wx)
    # Self inner product (same event set on both sides): the spectra are
    # identical, so compute them once. The cosine forms three inner
    # matrices per call, two of which --- <X, X> and <Y, Y> --- are self
    # inner products, so this halves their spectra work.
    if Px is Py and Wx is Wy:
        SY = SX
    else:
        SY = _spectra(Py, Wy)
    C_r = r * (sigma ** 2 * dxi) ** (r - 1)
    return C_r * np.real((SX * env) @ np.conj(SY).T)


def _rel_inner_batched(
    Px, Wx, Py, Wy, sigma, r, is_per, period,
    *, return_cancellation_ratio=False, truncation_sigmas=None,
    samples_per_sigma=None,
):
    """Batched relative-mode case of ``_ma_per_attr_inner_matrix``
    (periodic and non-periodic), all (event_X, event_Y) pairs at once.

    Every pair's inner product marginalises a translation u over a
    grid. In periodic mode the grid is the shared uniform grid over
    ``[0, P)`` with ``auto_ntau_default(period, sigma)`` nodes — the
    single node-count source shared with the flat single-multiset and
    nested relative-periodic paths, so the same level returns the same
    value whether reached flat or nested. In non-periodic mode each
    pair uses a grid of the same shape *centred on its own mean
    offset*: by translation invariance the integrand for pair
    (n_X, n_Y) depends on u only through u + (mean_Y - mean_X), so
    shifting each pair's window by that offset lets all pairs share
    one grid whose extent is the maximum within-pair spread plus the
    margin of :func:`_rel_window_margin` per side, rather than the
    global span of the data. That margin places every kernel entry
    strictly outside the truncation radius at the window edges, so the
    endpoint integrand is exactly zero for any finite truncation and
    the plain Riemann sum equals the trapezoidal rule exactly.

    The pair-and-grid batch is processed in slabs of at most
    ``_ORBIT_GRID_SLAB_ELEMS`` kernel entries, built directly in the
    contraction's (batch, K, K) layout — no transposition copies — so
    the working set stays memory-resident and the per-op cost of the
    batched Möbius contraction is flat in N and K (mirroring the
    single-multiset slabbing in ``_orbit_inner_rel``).

    Ragged (NaN-padded) events arrive zero-padded: a zero-weight slot
    contributes a zero factor to every Möbius term in which its axis
    value appears, so the result is exact for the K_eff events.
    Events with K_eff - r below the precision margin in this regime
    may lose precision in the alternating sum; ``method='auto'``
    routing accounts for this via the dispatcher's precision guard.

    With ``return_cancellation_ratio=True``, additionally returns the
    minimum across event pairs of the per-pair mass-aware cancellation
    diagnostic ``|sum_u F_u| / sum_u max_orb(|term_orb_u|)`` — the
    magnitude of each pair's integrated alternating sum relative to
    the integral of its worst-magnitude partition term, which bounds
    the relative error of that pair's integral. A pointwise worst-case
    over individual (pair, u) cells is the wrong aggregate: at sharp
    sigma, translation bands where the true integrand is exactly zero
    arise from exact cancellation of nonzero orbit terms, and such a
    band's pointwise ratio is ~0 while contributing nothing to any
    integral. Pairs with zero accumulated term mass report 1.0.

    ``truncation_sigmas`` is honoured on the per-u kernel tensor;
    ``None`` resolves to ``mpt.get_default('truncation_sigmas')``.
    """

    # ---- Spectral (Fourier) branch, r = 2..4 --------------------
    # Replaces the translation grid with a mode sum: each event's
    # spectrum is built once and the matrix over event pairs is their
    # Gram matrix, so the per-pair cost carries no K and no grid nodes.
    # Cancellation-ratio requests fall through (the spectral form has
    # no per-node terms matching that diagnostic), as do configurations
    # whose mode grid would be too large (the helper returns None).
    if (
        _SPECTRAL_IP_ENABLED
        and 2 <= r <= 4
        and not return_cancellation_ratio
    ):
        _spec = _spectral_rel_inner_matrix(
            Px, Wx, Py, Wy, float(sigma), int(r), bool(is_per),
            float(period),
        )
        if _spec is not None:
            return _spec

    from .._mobius import (
        inner_product_orbit_grid,
        inner_product_orbit_pw_batched,
    )
    from .._defaults import get_default
    from ._nested_contraction import auto_ntau_default

    if truncation_sigmas is None:
        truncation_sigmas = get_default('truncation_sigmas')

    from .._defaults import resolve_samples_per_sigma
    samples_per_sigma = resolve_samples_per_sigma(
        samples_per_sigma, r, truncation_sigmas
    )

    K_x, N_x = Px.shape
    K_y, N_y = Py.shape

    # Shared-weights fast path: when every event carries the same
    # weight vector on this attribute (the common case — uniform
    # weights, no ragged padding), all batch cells share w_A and w_B,
    # so the cheaper shared-weights grid contraction applies (the
    # per-batch-weights variant prepends the batch index to every
    # weight operand of the einsum, which costs ~2-3x per kernel op).
    shared_w = (np.all(Wx == Wx[:, :1]) and np.all(Wy == Wy[:, :1]))

    if is_per:
        N_u = auto_ntau_default(period, sigma)
        u_grid = np.linspace(0.0, period, N_u, endpoint=False)
        du = period / N_u
        centres = np.zeros((N_x, N_y), dtype=np.float64)
    else:
        # Per-pair centred common grid. Weighted-slot means keep the
        # centre finite for zero-padded events (all-zero-weight events
        # contribute nothing regardless of centre).
        def _col_means(P, W):
            wsum = W.sum(axis=0)
            safe = np.where(wsum > 0, wsum, 1.0)
            return (P * W).sum(axis=0) / safe

        mx = _col_means(Px, Wx)
        my = _col_means(Py, Wy)
        centres = my[None, :] - mx[:, None]          # (N_x, N_y)

        def _spread(P, W):
            masked = np.where(W > 0, P, np.nan)
            lo = np.nanmin(masked, axis=0)
            hi = np.nanmax(masked, axis=0)
            s = hi - lo
            return np.where(np.isfinite(s), s, 0.0)

        margin = _rel_window_margin(truncation_sigmas)
        span = (float(np.max(_spread(Px, Wx)) + np.max(_spread(Py, Wy)))
                + 2.0 * margin * sigma)
        N_u = max(
            64,
            int(np.ceil(max(span, 1.0) / sigma * samples_per_sigma)),
        )
        u_grid = np.linspace(-0.5 * span, 0.5 * span, N_u)
        du = span / (N_u - 1)

    # Sparse-orbit fast path (periodic): when the slot kernel is large
    # and the truncation window fits inside the circle, each u-node's
    # kernel is a circular band of width 2R out of the period, so a
    # spatially-culled per-node orbit beats the dense slab contraction;
    # the win repeats across every u-node while the sort is paid once
    # per pair. Same size/density thresholds as the absolute path, with
    # a cheap density probe at u = 0 from the first pair (the band
    # fraction is u-independent, so one node is representative).
    if is_per and r >= 2 and K_x * K_y >= _ORBIT_SPARSE_MIN_KERNEL:
        from .._defaults import truncation_ip_sqdist
        cutoff = float(truncation_ip_sqdist(truncation_sigmas, sigma))
        # Strict margin: the builder's padded candidate window must never
        # admit both period-copies of a centre (both would pass the exact
        # wrapped-distance filter and be double-counted by duplicate
        # summation), so the window must fit inside the circle with room
        # for the padding.
        if 2.0 * np.sqrt(cutoff) < period * (1.0 - 1e-8):
            vx0 = Wx[:, 0] != 0.0
            vy0 = Wy[:, 0] != 0.0
            c3p, j3p = _rel_per_sparse_prep(Py[vy0, 0], period)
            K0 = _build_sparse_kernel_rel_per(
                Px[vx0, 0], c3p, j3p, Py[vy0, 0], sigma, cutoff,
                period, 0.0)
            if K0.nnz <= _ORBIT_SPARSE_MAX_DENSITY * K_x * K_y:
                return _rel_per_inner_sparse(
                    Px, Wx, Py, Wy, sigma, r, period, u_grid, du, cutoff,
                    return_cancellation_ratio)

    PxT = np.ascontiguousarray(Px.T)                 # (N_x, K_x)
    PyT = np.ascontiguousarray(Py.T)                 # (N_y, K_y)
    WxT = np.ascontiguousarray(Wx.T)
    WyT = np.ascontiguousarray(Wy.T)

    integral = np.zeros((N_x, N_y), dtype=np.float64)
    worst_ratio = 1.0   # min over pairs of the per-pair mass-aware ratio

    # Slab sizing: one u-node across a row-chunk of pairs, widened in u
    # while the kernel slab stays under the element budget.
    per_pair = K_x * K_y
    nc_x = max(1, min(N_x, _ORBIT_GRID_SLAB_ELEMS // max(N_y * per_pair, 1)))
    for n_start in range(0, N_x, nc_x):
        n_end = min(n_start + nc_x, N_x)
        nc = n_end - n_start
        n_pairs = nc * N_y
        n_uc = max(1, _ORBIT_GRID_SLAB_ELEMS // max(n_pairs * per_pair, 1))

        # Base differences and per-pair centres for this row chunk,
        # flattened to (pairs, K_x, K_y) so the u broadcast below is 4-D
        # with no singleton axes (cheaper numpy staging than the
        # equivalent 5-D form). The two collections may differ in size,
        # so the A-side carries K_x pitches and the B-side K_y.
        base = (PxT[n_start:n_end, None, :, None]
                - PyT[None, :, None, :]
                + centres[n_start:n_end, :, None, None]
                ).reshape(n_pairs, K_x, K_y)
        if not shared_w:
            w_A = np.broadcast_to(
                WxT[n_start:n_end, None, :], (nc, N_y, K_x),
            ).reshape(n_pairs, K_x)
            w_B = np.broadcast_to(
                WyT[None, :, :], (nc, N_y, K_y),
            ).reshape(n_pairs, K_y)

        F_sum = np.zeros(n_pairs, dtype=np.float64)
        mass_sum = np.zeros(n_pairs, dtype=np.float64)
        for u_start in range(0, N_u, n_uc):
            u_end = min(u_start + n_uc, N_u)
            u_s = u_grid[u_start:u_end]
            nu = u_end - u_start
            diffs = base[None, :, :, :] + u_s[:, None, None, None]
            if is_per:
                diffs = diffs - period * np.floor(diffs / period + 0.5)
                n_img = _rel_per_image_count(sigma, period, truncation_sigmas)
                if n_img > 0:
                    # Full-image kernel. The transposition average of the
                    # wrapped (theta) kernel equals the lattice-sum form
                    # exactly, so summing images here upgrades the whole
                    # contraction to the full-image measure with the orbit
                    # reduction, the grid, and the slabbing untouched.
                    # Accumulated in a loop rather than on a trailing axis
                    # so peak memory stays flat in the image count.
                    K_uc = _trunc_kernel_exp(
                        diffs ** 2, sigma, truncation_sigmas)
                    for _l in range(1, n_img + 1):
                        shift = _l * period
                        K_uc = K_uc + _trunc_kernel_exp(
                            (diffs + shift) ** 2, sigma, truncation_sigmas)
                        K_uc = K_uc + _trunc_kernel_exp(
                            (diffs - shift) ** 2, sigma, truncation_sigmas)
                else:
                    K_uc = _trunc_kernel_exp(
                        diffs ** 2, sigma, truncation_sigmas)
            else:
                K_uc = _trunc_kernel_exp(diffs ** 2, sigma, truncation_sigmas)
            K_uc = K_uc.reshape(nu * n_pairs, K_x, K_y)
            if shared_w:
                if return_cancellation_ratio:
                    flat, _, mass = inner_product_orbit_grid(
                        K_uc, Wx[:, 0], Wy[:, 0], r,
                        return_cancellation_ratio=True,
                        return_term_mass=True,
                    )
                else:
                    flat = inner_product_orbit_grid(K_uc, Wx[:, 0], Wy[:, 0], r)
                    mass = None
            else:
                w_A_uc = np.broadcast_to(
                    w_A[None, :, :], (nu, n_pairs, K_x),
                ).reshape(nu * n_pairs, K_x)
                w_B_uc = np.broadcast_to(
                    w_B[None, :, :], (nu, n_pairs, K_y),
                ).reshape(nu * n_pairs, K_y)
                if return_cancellation_ratio:
                    flat, _, mass = inner_product_orbit_pw_batched(
                        K_uc, w_A_uc, w_B_uc, r, prefactor=1.0,
                        return_cancellation_ratio=True,
                        return_term_mass=True,
                    )
                else:
                    flat = inner_product_orbit_pw_batched(
                        K_uc, w_A_uc, w_B_uc, r, prefactor=1.0,
                    )
                    mass = None
            if mass is not None:
                mass_sum += mass.reshape(nu, n_pairs).sum(axis=0)
            F_sum += flat.reshape(nu, n_pairs).sum(axis=0)

        if return_cancellation_ratio:
            with np.errstate(divide='ignore', invalid='ignore'):
                pair_ratios = np.where(
                    mass_sum > 0, np.abs(F_sum) / mass_sum, 1.0,
                )
            cmin = float(np.min(pair_ratios)) if pair_ratios.size else 1.0
            if cmin < worst_ratio:
                worst_ratio = cmin
        integral[n_start:n_end, :] = F_sum.reshape(nc, N_y) * du

    c = sigma * np.sqrt(2 * np.pi / r)
    out = (sigma * np.sqrt(np.pi)) ** r * integral / c ** 2
    if return_cancellation_ratio:
        return out, worst_ratio
    return out


def _rel_window_margin(truncation_sigmas):
    """Margin, in sigmas per side, of the non-periodic relative
    translation window.

    The inner-product kernel is G(d; sigma*sqrt(2)) (a convolution of
    two sigma-kernels), and the toolbox truncation zeroes entries whose
    kernel value falls below exp(-t^2 / 2), i.e. at distances beyond
    sqrt(2)*t*sigma.

    The margin's contract is subordination to the kernel-truncation
    contract: the toolbox's accuracy guarantee is stated in
    truncation-sigmas terms (t = 6 accepts ~2e-8 worst-case error;
    t = 8 sits below the 1e-12 cross-language parity floor), so the
    window need only keep its own error comfortably below the kernel
    error the caller has already accepted. A margin of 8 achieves that
    for every t: each kernel factor is at least e^-16 down at the
    window edge, an r-tuple term needs r such factors, so the window
    tail is bounded near 1e-14 relative independent of t (and measures
    at ~1e-27 in practice). The rule is therefore
    ``min(sqrt(2)*t + 0.1, 8)``: for t below ~5.59 the integrand's
    compact support (which ends exactly sqrt(2)*t*sigma beyond the
    extreme pair difference) fits inside the cap, so the cheaper exact
    window is taken — the endpoint integrand vanishes identically and
    the plain Riemann sum equals the trapezoidal rule exactly (the
    0.1-sigma clearance is needed because the extreme pair sits
    exactly at margin*sigma at the endpoint and the truncation mask is
    inclusive; it exceeds the floating-point wobble of the span
    arithmetic by ~12 orders of magnitude). For larger or disabled t
    the margin caps at the empirically validated 8, where the window
    error is subdominant to the kernel contract at every t.
    """
    if truncation_sigmas is None or not np.isfinite(truncation_sigmas):
        return 8.0
    return min(np.sqrt(2.0) * float(truncation_sigmas) + 0.1, 8.0)


#: Predicted wall-time constants for the centres-vs-grid gate.
#: Both sides model per-event-pair cost in nanoseconds; the gate
#: compares predictions rather than raw op counts because the actual
#: per-op costs of the two paths differ by two orders of magnitude
#: and their scaling with K differs from the raw counts.
#:
#: Centres side: per-element cost of the pairwise Gaussian overlap
#: over the (r_a!·C(K, r_a))² tuple-centre pairs, dominated by the
#: exp evaluation and augmented by an O(r_a-1) sum for Q and, in the
#: periodic case, an O((r_a-1)(r_a-2)) pairwise-wrap inner loop.
#:
#: Grid side: cost of the grid path, which for r_a ∈ {2, 3, 4} is
#: served by the spectral branch of ``_rel_inner_batched``
#: rather than a u-grid contraction; the spectral cost scales as
#: (K + const) · grid_size and grid_size scales roughly as N_u
#: within a fixed r_a, so a per-r_a coefficient on N_u·K plus a
#: fixed setup floor captures the observed sub-K² wall behaviour
#: (which the raw N_u·K² proxy overpredicts by up to 30x at large K,
#: causing the previous gate to over-select centres at r_a = 2).
#:
#: Calibrated in Python by a 168-cell sweep across r_a ∈ {2, 3, 4},
#: K ∈ {5..80}, σ ∈ {5, 10, 15, 20, 25, 30}, span ∈ {1200, 2400, 3000,
#: 3600, 4800}, and both periodic modes. The values below reproduce
#: the sign of every training cell (102/102) and 88% of held-out
#: cells, with the small remaining misses all in the c/g ∈ [0.7, 1.4]
#: neighbourhood of the crossover where either path is nearly as
#: cheap as the other.
_CENTRES_NS_BASE = 45.0       # per-element base (exp dominates)


_CENTRES_NS_LIN  = 15.0       # per-element linear-in-(r_a - 1) term


_CENTRES_NS_WRAP = 10.0       # per-element (r_a-1)(r_a-2) term, is_per only


_GRID_NS_FLOOR   = 1_000_000.0  # 1 ms fixed per-pair setup


_GRID_NS_PER_OP = {2: 30.0, 3: 700.0, 4: 2000.0}  # ns per (N_u·K) op, per r_a


def _predicted_centres_wall_ns(K, r_a, is_per):
    """Nanosecond wall-time estimate for one event pair on the centres
    (pairwise closed-form) path."""
    import math
    n_e = (math.factorial(r_a) * math.comb(K, r_a)) ** 2
    per_el = _CENTRES_NS_BASE + _CENTRES_NS_LIN * (r_a - 1)
    if is_per:
        per_el += _CENTRES_NS_WRAP * (r_a - 1) * (r_a - 2)
    return per_el * n_e


def _predicted_grid_wall_ns(K, r_a, sigma, span_or_period, is_per):
    """Nanosecond wall-time estimate for one event pair on the grid
    (spectral or u-grid) path. ``span_or_period`` is the period in the
    periodic case, or the sum of the two side spans in the
    non-periodic case (matching the current N_u sizing)."""
    from ._nested_contraction import auto_ntau_default
    if is_per:
        n_u = auto_ntau_default(span_or_period, sigma)
    else:
        from .._defaults import get_default
        margin = _rel_window_margin(get_default('truncation_sigmas'))
        span = span_or_period + 2.0 * margin * sigma
        n_u = max(64, int(np.ceil(max(span, 1.0) / sigma * 10.0)))
    # Extrapolate calibrated r_a in {2, 3, 4} to r_a >= 5 by tripling
    # per r_a increment; centres cost grows faster than that in K, so
    # the extrapolation only affects the tiny-K corner.
    if r_a in _GRID_NS_PER_OP:
        g_op = _GRID_NS_PER_OP[r_a]
    else:
        g_op = _GRID_NS_PER_OP[4] * 3.0 ** (r_a - 4)
    return _GRID_NS_FLOOR + g_op * float(n_u) * float(K)


def _ma_rel_attr_prefers_centres(Px, Py, sigma, r_a, is_rel, is_per, period):
    """True when a relative attribute's inner matrices should use the
    pairwise closed form over tuple-centres rather than the
    translation-grid contraction.

    The gate compares predicted wall time on the two paths rather than
    raw kernel-op counts. Raw counts were misleading here because the
    centres path pays roughly one exp per op (~50 ns) while the grid
    path — served by the spectral branch of
    ``_rel_inner_batched`` at r_a ∈ {2, 3, 4} — pays a
    sub-K² per-op cost after a fixed setup, so the per-op cost ratio
    between them varies with r_a and configuration and is far from
    unity. The old raw-count comparison consequently over-selected
    centres for r_a = 2 across a wide K band (roughly 15..90 at σ/P ~
    1/240), where centres was up to 100x slower than grid.

    Decision-safety layer: below the ``sigma/P`` threshold the centres
    route's minimum-image relative-periodic reading coincides with the
    grid route's all-image reading, so the two agree numerically; above
    it, the grid route is kept regardless of cost so that
    ``method='mobius'`` on that side opts into the all-image measure
    without the gate flipping to a different reading. The non-periodic
    closed form is exact for the non-periodic reading and always
    measure-safe.

    See ``_CENTRES_NS_BASE`` and ``_GRID_NS_PER_OP`` for the cost-model
    calibration notes.
    """
    from .dispatch import _ORBIT_SIGMA_OVER_P_THRESHOLD

    if not is_rel or r_a < 2:
        return False
    if is_per and (sigma / period) > _ORBIT_SIGMA_OVER_P_THRESHOLD:
        return False
    K = int(Px.shape[0])
    if K < r_a:
        return False
    if is_per:
        span_or_period = float(period)
    else:
        span_or_period = (float(np.nanmax(Px) - np.nanmin(Px))
                          + float(np.nanmax(Py) - np.nanmin(Py)))
    c_wall_ns = _predicted_centres_wall_ns(K, r_a, is_per)
    g_wall_ns = _predicted_grid_wall_ns(
        K, r_a, sigma, span_or_period, is_per,
    )
    return c_wall_ns < g_wall_ns


def _closed_form_attr_centres(dens, a):
    """Materialised tuple-centres and metric parameters for attribute ``a``,
    rebuilt in isolation as a single-multiset density.

    The attribute's expectation tensor is a finite Gaussian mixture over its
    materialised perm-side tuple-centres (the same reduction the entropy
    reading uses), so its inner product with another attribute is the pairwise
    Gaussian overlap of those centres. This is exact for the absolute and
    absolute-periodic readings, the analytic relative quadratic (also exact)
    for relative-non-periodic, and the minimum-image pairwise-wrap for
    relative-periodic -- a deliberate measure choice, not the all-image torus
    inner product the tau-grid contraction computes (consistent with the flat
    per-attribute matrix; see the note in
    :func:`_closed_form_attr_matrix_from`).

    Returns ``(centres, w_j, event_of_j, inner_block_size, is_per, period,
    r_a, is_rel, sigma, n_events)``. Variable-K (NaN-padded) slots are carried
    by the rebuild, which drops every tuple touching a padded slot to zero
    weight.
    """
    from .build import build_exp_tens as _bld
    nested = getattr(dens, "nested", None)
    spec = nested[a] if nested is not None else None
    sigma = float(dens.sigma[a])
    is_per = bool(dens.is_per[a])
    period = float(dens.period[a])
    if spec is not None:
        da = _bld([dens.p_attr[a]], [dens.w[a]], specs=[spec],
                  sigma=[sigma], is_per=[is_per], period=[period],
                  verbose=False)
    else:
        is_sym_vec = np.asarray(
            getattr(dens, "is_sym", np.ones(int(dens.n_attrs), dtype=bool))
        ).ravel()
        da = _bld([dens.p_attr[a]], [dens.w[a]], [sigma], [int(dens.r[a])],
                  [bool(dens.is_rel[a])], [is_per], [period],
                  [bool(is_sym_vec[a])], verbose=False)
    return (da.centres[0], da.w_j, da.event_of_j, int(_inner_r_vec(da)[0]),
            is_per, period, int(da.r[0]), bool(da.is_rel[0]), sigma,
            int(da.n))


def _closed_form_attr_matrix_from(cx, cy, truncation_sigmas=None,
                                  wrap_a='full-image'):
    """(N_x, N_y) per-attribute inner matrix from precomputed tuple-centres.

    The full pairwise centre-overlap is formed as one ``(n_jx, n_jy)`` array
    and aggregated to events by two incidence matmuls, vectorising the
    event-pair grid that the per-event-pair contraction loops scalar-wise. The
    constant per-attribute Gaussian prefactor is dropped: it is identical
    across this matrix and the self matrices, so it cancels in the cosine and
    one-sided ratios.

    Absolute-periodic uses the full-image (torus) measure: the r-tuple
    kernel is the product across slots of the 1D wrapped Gaussian
    ``theta(d) = sum_n exp(-(d + n P)^2 / (4 sigma^2))``. Because Q
    factors across slots in absolute mode, the product-of-theta form
    (``r * (2L+1)`` per pair) is the cheaper representation of the
    all-image kernel than the r-dim lattice sum (``(2L+1)^r``); the
    image sum switches on only when the accuracy floor requires it.

    Relative-periodic uses the minimum-image pairwise-wrap metric of
    :func:`_compute_Q` (exactly period-shift invariant, and matching the flat
    per-attribute path). This is the toolbox's defined relative-periodic
    measure. It does not equal the all-image transposition average (the torus
    inner product), which the tau-grid contraction computes; the two coincide
    for sigma << period and diverge as sigma approaches the period. Because the
    minimum-image kernel couples all slots within a tuple (a pairwise quadratic
    form), it does not factor per level, so the per-level orbit (Möbius)
    reduction is unavailable here and a symmetric level is enumerated over its
    full orbit. When that enumeration becomes the dominant cost, the dispatch
    in :func:`_nested_attr_matrix_dispatch` routes the attribute to the
    all-image tau-grid contraction instead -- a deliberate measure change,
    documented there.
    """
    (Cx, Wx, Ex, bs, is_per, period, r_a, is_rel, sigma, Nx) = cx
    (Cy, Wy, Ey, _bs, _ip, _pe, _ra, _ir, _sg, Ny) = cy
    Wx = np.ones(Cx.shape[1]) if Wx is None else np.asarray(Wx, float).ravel()
    Wy = np.ones(Cy.shape[1]) if Wy is None else np.asarray(Wy, float).ravel()
    d = Cx.shape[0]
    njx, njy = Cx.shape[1], Cy.shape[1]
    Ex = np.asarray(Ex)
    Ey = np.asarray(Ey)
    # Y-side incidence (njy, Ny): njy is a contraction axis, summed by event.
    GY = np.zeros((njy, Ny))
    GY[np.arange(njy), Ey] = 1.0
    inv4s2 = 1.0 / (4.0 * sigma ** 2)
    # Abs-per full-image: per-slot image sum before the r-tuple product.
    # L = 0 at sigma/P below the accuracy-floor threshold, so this
    # collapses to the single-Gaussian route unchanged. When the user
    # has opted this attribute into ``wrap='single-image'`` the L is
    # forced to 0 regardless.
    if is_per and not is_rel:
        if wrap_a == 'full-image':
            from .._defaults import get_default
            ts = (get_default("truncation_sigmas")
                  if truncation_sigmas is None else truncation_sigmas)
            L_abs_per = _rel_per_image_count(sigma, period, ts)
        else:
            L_abs_per = 0
    else:
        L_abs_per = 0
    # Chunk the X-tuple axis so the pairwise difference array never exceeds a
    # fixed budget: the full (njx, njy) overlap is materialised only one
    # |chunk| x njy block at a time, which keeps the materialised-centre path
    # within memory for large-K factors while the per-event aggregation stays
    # exact. For the common small-tuple case (one chunk) this is identical to
    # the unchunked form.
    chunk = max(1, min(njx, int(16_000_000 // max(njy * max(d, 1), 1))))
    M = np.zeros((Nx, Ny))
    for s in range(0, njx, chunk):
        e = min(s + chunk, njx)
        D = Cx[:, s:e, None] - Cy[:, None, :]
        if bs >= 2:
            Q = _compute_Q_inner_blocks(D, bs, is_per, period, reduced=True)
            kernel_val = np.exp(-Q * inv4s2)
        elif is_per and not is_rel:
            # Abs-per full-image via the shared wrapped-Gaussian helper,
            # which picks image-sum or Fourier by cost (crossover at
            # sigma/P ~ 0.2). ``wrap_a='single-image'`` forces the
            # single-image path.
            if wrap_a == 'single-image':
                D = D - period * np.floor(D / period + 0.5)
                Q = _compute_Q(D, r_a, is_rel, is_per, period,
                               reduced=is_rel)
                kernel_val = np.exp(-Q * inv4s2)
            else:
                from .._wrapped_kernel import wrapped_gaussian_1d
                from .._defaults import get_default
                ts = (get_default("truncation_sigmas")
                      if truncation_sigmas is None else truncation_sigmas)
                theta_per_slot = wrapped_gaussian_1d(
                    D, sigma, period, ts, exponent_denominator=4
                )
                kernel_val = theta_per_slot.prod(axis=0)
        else:
            Q = _compute_Q(D, r_a, is_rel, is_per, period, reduced=is_rel)
            kernel_val = np.exp(-Q * inv4s2)
        ov = (Wx[s:e, None] * Wy[None, :]) * kernel_val
        # X-side incidence matmul (mirror of the MATLAB implementation):
        # scatter-adds row-by-row are far slower than aggregating the
        # chunk with a second incidence product.
        nc = e - s
        GX = np.zeros((Nx, nc))
        GX[Ex[s:e], np.arange(nc)] = 1.0
        M += GX @ (ov @ GY)
    return M
