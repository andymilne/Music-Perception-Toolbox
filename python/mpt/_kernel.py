"""Gaussian-kernel sum helper with optional grid-bucket truncation.

The single centres-path numerical kernel used by ``eval_exp_tens``,
``cos_sim_exp_tens``, ``entropy_exp_tens``, and (eventually) every
centres-path consumer in the toolbox. Routes through here so the
``truncation_sigmas`` and ``kernel_precision`` options are applied uniformly.

See :mod:`mpt._defaults` for the global-defaults machinery.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._defaults import get_default


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
        toolbox default (factory: ``math.inf`` = off).
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
        and not is_per
        and nJ > 0
        and nQ > 0
    )

    if use_truncation:
        v = _truncated_kernel_sum(
            C_w, wJ_w, X_w, sigma_w, is_rel, r,
            float(truncation_sigmas), inv2s2,
        )
    else:
        v = _exact_kernel_sum(
            C_w, wJ_w, X_w, is_rel, r, is_per, dtype(period), inv2s2,
            sigma_w,
        )

    return v.astype(np.float64, copy=False)


# ---------------------------------------------------------------------
# Exact path
# ---------------------------------------------------------------------

def _exact_kernel_sum(C, wJ, X, is_rel, r, is_per, period, inv2s2, sigma):
    dim, nJ = C.shape
    nQ = X.shape[1]
    if nJ == 0 or nQ == 0:
        return np.zeros(nQ, dtype=C.dtype)

    # Memory-budget chunking on nQ.
    bytes_per_scalar = C.dtype.itemsize
    bytes_needed = (dim + 1) * nJ * nQ * bytes_per_scalar
    BUDGET = 1 * 1024 ** 3
    if bytes_needed <= BUDGET:
        return _eval_chunk(C, wJ, X, is_rel, r, is_per, period, inv2s2, sigma)

    chunk = max(1, BUDGET // ((dim + 1) * nJ * bytes_per_scalar))
    out = np.zeros(nQ, dtype=C.dtype)
    for c0 in range(0, nQ, chunk):
        c1 = min(c0 + chunk, nQ)
        out[c0:c1] = _eval_chunk(
            C, wJ, X[:, c0:c1], is_rel, r, is_per, period, inv2s2, sigma
        )
    return out


def _eval_chunk(C, wJ, Xq, is_rel, r, is_per, period, inv2s2, sigma):
    # D: (dim, nJ, nQc)
    D = C[:, :, None] - Xq[:, None, :]
    if is_per:
        # Match v2.0/v2.1 wrap (np.mod-based) for bit-identical default-path.
        D = np.mod(D + period / 2, period) - period / 2
    if is_rel:
        Q = np.sum(D ** 2, axis=0) - np.sum(D, axis=0) ** 2 / r
    else:
        Q = np.sum(D ** 2, axis=0)
    # Use the direct division (Q / (2*sigma^2)) rather than Q * inv2s2,
    # to match v2.0/v2.1 ULP-for-ULP at default settings.
    E = np.exp(-Q / (2 * sigma ** 2))      # (nJ, nQc)
    return wJ @ E                          # (nQc,)


# ---------------------------------------------------------------------
# Truncated path — grid-bucket spatial index
# ---------------------------------------------------------------------

def _truncated_kernel_sum(C, wJ, X, sigma, is_rel, r, k_sigma, inv2s2):
    dim, nJ = C.shape
    nQ = X.shape[1]
    if nJ == 0:
        return np.zeros(nQ, dtype=C.dtype)

    # Coordinate transform to make Q-ball spherical.
    if is_rel:
        e = np.ones(dim, dtype=np.float64)
        M = np.eye(dim, dtype=np.float64) - np.outer(e, e) / r
        lams, U = np.linalg.eigh(M)
        lams = np.clip(lams, 0.0, None)
        T = (U @ np.diag(np.sqrt(lams))).T  # so that T @ D gives transformed coords
        T = T.astype(C.dtype, copy=False)
    else:
        T = np.eye(dim, dtype=C.dtype)

    Ct = T @ C                      # (dim, nJ) transformed
    Xt = T @ X                      # (dim, nQ) transformed

    threshold2 = (k_sigma * float(sigma)) ** 2
    bucket_size = k_sigma * float(sigma)

    cmin = Ct.min(axis=1)
    cmax = Ct.max(axis=1)
    n_buckets = np.maximum(
        1,
        np.ceil((cmax - cmin) / bucket_size).astype(np.int64) + 1,
    )

    # Centre bucket coords (0-indexed in Python).
    buck_c = np.floor((Ct - cmin[:, None]) / bucket_size).astype(np.int64)
    buck_c = np.clip(buck_c, 0, (n_buckets - 1)[:, None])

    lin_c = _sub_to_ind(n_buckets, buck_c)

    # Group centres by bucket.
    order = np.argsort(lin_c, kind="stable")
    sorted_lin = lin_c[order]
    if sorted_lin.size == 0:
        return np.zeros(nQ, dtype=C.dtype)
    boundaries = np.concatenate(
        ([0], 1 + np.flatnonzero(sorted_lin[1:] != sorted_lin[:-1]))
    )
    starts = boundaries
    ends = np.concatenate((boundaries[1:], [sorted_lin.size]))
    run_lin = sorted_lin[boundaries]

    # Build a dict bucket_linear_index -> (start, end) into `order`.
    bucket_map: dict[int, tuple[int, int]] = {
        int(rl): (int(s), int(e))
        for rl, s, e in zip(run_lin, starts, ends)
    }

    # Neighbour offsets (3^dim).
    offsets = _neighbour_offsets(dim)  # (dim, 3**dim)

    # Query bucket coords.
    buck_x = np.floor((Xt - cmin[:, None]) / bucket_size).astype(np.int64)
    buck_x = np.clip(buck_x, 0, (n_buckets - 1)[:, None])

    out = np.zeros(nQ, dtype=C.dtype)

    for q in range(nQ):
        qb = buck_x[:, q]                         # (dim,)
        # Candidate-collection across 3**dim neighbour buckets.
        nb_all = qb[:, None] + offsets             # (dim, 3**dim)
        # Mask to in-bounds neighbours.
        in_bounds = np.all(
            (nb_all >= 0) & (nb_all < n_buckets[:, None]),
            axis=0,
        )
        nb_valid = nb_all[:, in_bounds]
        if nb_valid.size == 0:
            continue
        lin_nb = _sub_to_ind(n_buckets, nb_valid)
        cand_parts = []
        for lin in lin_nb:
            run = bucket_map.get(int(lin))
            if run is not None:
                s, e = run
                cand_parts.append(order[s:e])
        if not cand_parts:
            continue
        candidates = np.concatenate(cand_parts)

        # Compute Q for candidates.
        Dq = C[:, candidates] - X[:, q : q + 1]   # (dim, ncand)
        if is_rel:
            Q = np.sum(Dq * Dq, axis=0) - np.sum(Dq, axis=0) ** 2 / r
        else:
            Q = np.sum(Dq * Dq, axis=0)
        keep = Q <= threshold2
        if np.any(keep):
            surv = candidates[keep]
            Qkept = Q[keep]
            kernel_vals = np.exp(-Qkept * inv2s2)
            out[q] = float(np.sum(wJ[surv] * kernel_vals))

    return out


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
