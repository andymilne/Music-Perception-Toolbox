"""Factored Möbius point evaluator for a flat multi-attribute MAET.

The MAET density is an outer sum over events of the tensor product over
attributes, the tensor product acting as the pointwise product of the
per-attribute densities, each a function of that attribute's query block
alone::

    f(x_1, ..., x_A) = sum_n prod_a [Sym or Ord]^{r_a}(M_{a,n})(x_a)

Evaluation therefore factorises completely across attributes within an
event: there is no joint cross-attribute tuple sum. Each per-attribute
factor is an ordinary single-multiset expectation-tensor density,
evaluated by the Möbius point evaluators :func:`eval_orbit_abs`
(absolute mode) and :func:`eval_orbit_rel` (relative mode), which compute
the value from the raw values via the set-partition Möbius decomposition
--- polynomial in K rather than the O(K^r) of the materialised tuple
centres.

Scope: flat (non-nested) attributes. A nested attribute is stitched by
contraction elsewhere; its per-attribute density is a recursive
construction, not a single Möbius sum.

Twin of MATLAB ``mobius.evalMaOrbit``.
"""
from __future__ import annotations

import numpy as np


def eval_ma_orbit(
    dens,
    x,
    *,
    truncation_sigmas=None,
    kernel_precision=None,
    return_cancellation_ratio: bool = False,
):
    """Evaluate a flat MA density at joint query points by the Möbius route.

    ``x`` is a ``(D, n_q)`` matrix with ``D = sum_a dim_per_attr[a]``; its
    rows are split into per-attribute blocks of width ``dim_per_attr[a]``
    in attribute order. Returns the raw (un-normalised) values ``(n_q,)``
    --- the caller applies :func:`_ma_eval_normalize`, identically to the
    joint-centres path.

    With ``return_cancellation_ratio=True`` also returns the per-query
    worst-case (minimum across events and attributes) mass-aware
    cancellation ratio of the underlying Möbius alternating sums.
    """
    from .._defaults import resolve_truncation_sigmas
    from .._mobius import eval_orbit_abs, eval_orbit_rel

    A = int(dens.n_attrs)
    N = int(dens.n)
    dims = [int(v) for v in np.atleast_1d(dens.dim_per_attr)]
    sigma = [float(v) for v in np.atleast_1d(dens.sigma)]
    r_vec = [int(v) for v in np.atleast_1d(dens.r)]
    is_rel = [bool(v) for v in np.atleast_1d(dens.is_rel)]
    is_per = [bool(v) for v in np.atleast_1d(dens.is_per)]
    period = [float(v) for v in np.atleast_1d(dens.period)]

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(
            f"Query must be 2-D (D, n_q); got {x.ndim} dimension(s)."
        )
    D = int(sum(dims))
    if x.shape[0] != D:
        raise ValueError(
            f"Query has {x.shape[0]} rows but the joint effective "
            f"dimension is D = {D}."
        )
    n_q = int(x.shape[1])
    if n_q == 0:
        empty = np.zeros(0, dtype=np.float64)
        if return_cancellation_ratio:
            return empty, np.ones(0, dtype=np.float64)
        return empty

    ts = resolve_truncation_sigmas(truncation_sigmas)

    # Split the joint query into per-attribute blocks once.
    x_blocks = []
    off = 0
    for a in range(A):
        x_blocks.append(x[off:off + dims[a], :])
        off += dims[a]

    P = [np.asarray(p, dtype=np.float64) for p in dens.p_attr]
    W = [np.asarray(w, dtype=np.float64) for w in dens.w]

    total = np.zeros(n_q, dtype=np.float64)
    ratio = (np.ones(n_q, dtype=np.float64)
             if return_cancellation_ratio else None)

    for n in range(N):
        prod = np.ones(n_q, dtype=np.float64)
        for a in range(A):
            p_an = P[a][:, n]
            w_an = W[a][:, n]
            # Drop absent slots (NaN in this event) so ragged cardinality
            # needs no special case: the per-attribute factor is built
            # from that event's live slots alone.
            live = ~np.isnan(p_an)
            p_an = p_an[live]
            w_an = w_an[live]

            evaluator = eval_orbit_rel if is_rel[a] else eval_orbit_abs
            kw = dict(
                is_per=is_per[a],
                period=period[a],
                truncation_sigmas=ts,
                kernel_precision=kernel_precision,
            )
            if return_cancellation_ratio:
                f_a, r_a = evaluator(
                    p_an, w_an, sigma[a], r_vec[a], x_blocks[a],
                    return_cancellation_ratio=True, **kw,
                )
                ratio = np.minimum(
                    ratio, np.asarray(r_a, dtype=np.float64).ravel()
                )
            else:
                f_a = evaluator(
                    p_an, w_an, sigma[a], r_vec[a], x_blocks[a], **kw,
                )
            prod *= np.asarray(f_a, dtype=np.float64).ravel()
        total += prod

    if return_cancellation_ratio:
        return total, ratio
    return total
