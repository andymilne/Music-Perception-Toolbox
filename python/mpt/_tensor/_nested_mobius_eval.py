"""Per-level Möbius point evaluator for a nested attribute.

A nested attribute's per-event density is a recursive construction over
the tag tree: at the leaf level an ``r_0``-tuple of distinct values from
one finest group, at each higher level an ``r_l``-tuple of distinct
level-``(l-1)`` sub-tuples. The tuple-centres route materialises every
nested tuple (``M_perm`` per event, a product of per-level factorials
and binomials) and sums a Gaussian per centre. This module evaluates the
same density without materialising any tuple, by applying the Möbius
set-partition decomposition **level by level**:

* at a symmetric level, the sum over ordered ``r``-tuples of *distinct*
  children is written by inclusion–exclusion over the set partitions
  ``π`` of the ``r`` tuple slots,

      Σ_{distinct} Π_t M[c_t, t] = Σ_π μ(π) Π_{B∈π} Σ_c Π_{t∈B} M[c, t],

  where ``M[c, t]`` is child ``c``'s sub-density evaluated at the query
  coordinates of slot ``t`` — the same identity the flat evaluator
  :func:`mpt._mobius.eval_orbit_abs` uses, with children in place of
  values;
* at an ordered level the sum runs over children in listed order, a
  dynamic programme over the children (no factorial);
* at the leaf, ``M[v, t] = w_v θ(x_t - p_v)`` is the weighted
  one-body kernel of value ``v`` at slot ``t``.

Because the query side is a single fixed point rather than a summed
tuple set, the identity needs no orbit table: the set-partition lists
of :func:`mpt._mobius.get_partition_block_structure` (Bell numbers
``B_r``) are all it reads, so every level up to ``r = 10`` is
available, as for the flat evaluator.

Relative levels. A co-transposition unit at level ``u`` integrates each
level-``u`` block over its own translation, exactly as the flat relative
evaluator does over the whole tuple: the reduced block query
``(x_1, …, x_{s-1})`` is lifted to ``(u, u + x_1, …, u + x_{s-1})`` on a
translation grid (the line for a non-periodic attribute, ``[0, P)`` for a
periodic one — the all-image measure), the absolute sub-density is
evaluated at every node, and the quadrature is divided by the
translation-mode normaliser ``σ√(2π/s)``. For the outer unit that is
the root; for an inner or intermediate unit the integral sits inside the
recursion at the unit's level, one independent integral per block, so
the cost is a sum over blocks rather than a product.

Twin of MATLAB ``mobius.evalNestedAttrOrbit``.
"""
from __future__ import annotations

import math

import numpy as np

from .._mobius import get_partition_block_structure, mobius_partition_combine
from .._utils import kernel_chunk_bytes_resolved
from ._nested_contraction import build_recipe


def eval_nested_attr_orbit(p, w, tags, r_levels, sym_levels, rel_unit,
                           sigma, x, *, is_per=False, period=0.0,
                           wrap='full-image', truncation_sigmas=None,
                           samples_per_sigma=None):
    """Evaluate one event's nested-attribute density at query points.

    Parameters
    ----------
    p, w : (K,) arrays
        The event's live (non-NaN) values and their weights.
    tags : (K, L-1) int array
        Grouping tags of the live values, innermost grouping first.
    r_levels, sym_levels : length-L sequences
        Per-level tuple size and symmetry, innermost first.
    rel_unit : None or int
        Co-transposition unit (level index), ``None`` for absolute.
    sigma : float
    x : (dim_a, n_q) array
        Query coordinates in the attribute's reduced layout (the layout
        ``build_exp_tens`` documents for ``centres``).
    is_per, period, wrap, truncation_sigmas, samples_per_sigma :
        Kernel and quadrature controls, as on the flat evaluators.

    Returns
    -------
    (n_q,) ndarray
        Raw (un-normalised) per-event density values, on the same scale
        as the tuple-centres route's raw values.
    """
    from .._defaults import (resolve_samples_per_sigma,
                             resolve_truncation_sigmas)

    p = np.asarray(p, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    K = int(p.size)
    r_levels = [int(v) for v in np.asarray(r_levels).ravel()]
    sym_levels = [bool(v) for v in np.asarray(sym_levels).ravel()]
    L = len(r_levels)
    tags = np.asarray(tags)
    tags2 = tags.reshape(K, -1) if tags.ndim == 2 else tags.reshape(K, 1)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n_q = int(x.shape[1])
    if n_q == 0:
        return np.zeros(0, dtype=np.float64)

    ts = float(resolve_truncation_sigmas(truncation_sigmas))
    spp = None
    if rel_unit is not None:
        s_unit = int(np.prod(r_levels[:rel_unit + 1]))
        spp = int(resolve_samples_per_sigma(samples_per_sigma,
                                            max(2, s_unit), truncation_sigmas))

    ctx = _Ctx(p, w, float(sigma), bool(is_per), float(period), str(wrap),
               ts, r_levels, rel_unit, spp)

    # The tree: build_recipe groups values by the tag columns exactly as
    # the enumeration does; only level / val_idx / children / r / sym are
    # read here.
    root = build_recipe(np.asarray(r_levels), np.asarray(sym_levels), tags2)

    if x.shape[0] != ctx.width(L - 1):
        raise ValueError(
            f"nested query has {x.shape[0]} rows; expected "
            f"{ctx.width(L - 1)} for this attribute's reduced layout.")
    return _contract(root, x, ctx)


class _Ctx:
    """Per-call constants and the slot-width bookkeeping."""
    __slots__ = ("p", "w", "sigma", "is_per", "period", "wrap", "ts",
                 "r_levels", "rel_unit", "spp", "_s")

    def __init__(self, p, w, sigma, is_per, period, wrap, ts, r_levels,
                 rel_unit, spp):
        self.p, self.w = p, w
        self.sigma, self.is_per, self.period = sigma, is_per, period
        self.wrap, self.ts = wrap, ts
        self.r_levels, self.rel_unit, self.spp = r_levels, rel_unit, spp
        # s[l] = number of leaf slots spanned by one level-l sub-tuple.
        self._s = [int(np.prod(r_levels[:l + 1])) for l in range(len(r_levels))]

    def span(self, level):
        """Leaf slots of one level-``level`` sub-tuple (absolute layout)."""
        return self._s[level]

    def width(self, level):
        """Query rows of one level-``level`` sub-tuple in the reduced
        layout: the absolute span until the co-transposition unit, and
        thereafter one coordinate fewer per unit block."""
        s = self._s[level]
        u = self.rel_unit
        if u is None or level < u:
            return s
        s_u = self._s[u]
        return (s // s_u) * (s_u - 1)


# ---------------------------------------------------------------------
#  Contraction
# ---------------------------------------------------------------------

def _contract(node, xq, ctx):
    """Value (n_q,) of ``node``'s sub-density at the (reduced) query rows
    ``xq`` (width(level), n_q)."""
    if ctx.rel_unit is not None and node.level == ctx.rel_unit:
        return _integrate_translation(node, xq, ctx)
    return _contract_abs(node, xq, ctx)


def _contract_abs(node, xq, ctx):
    """Absolute contraction at ``node``: ``xq`` has span(level) rows."""
    n_q = xq.shape[1]
    r = int(node.r)
    if node.level == 0:
        vals = node.val_idx
        # Leaf kernel M[v, t, q] = w_v theta(x_t - p_v), chunked over q.
        out = np.empty(n_q, dtype=np.float64)
        budget = max(kernel_chunk_bytes_resolved(), 1)
        per_q = max(1, int(len(vals)) * r * 8 * 4)
        step = max(1, budget // per_q)
        for c0 in range(0, n_q, step):
            c1 = min(n_q, c0 + step)
            d = xq[None, :, c0:c1] - ctx.p[vals][:, None, None]
            M = ctx.w[vals][:, None, None] * _theta(d, ctx)
            out[c0:c1] = _combine(M, r, node.sym)
        return out
    child_w = ctx.width(node.level - 1)
    M = np.empty((len(node.children), r, n_q), dtype=np.float64)
    for i, child in enumerate(node.children):
        for b in range(r):
            M[i, b, :] = _contract(child, xq[b * child_w:(b + 1) * child_w, :],
                                   ctx)
    return _combine(M, r, node.sym)


def _theta(d, ctx):
    """One-body kernel per coordinate: Gaussian, or its wrapped form on a
    periodic attribute under the declared measure."""
    if ctx.is_per:
        if ctx.wrap == 'single-image':
            d = d - ctx.period * np.round(d / ctx.period)
        else:
            from .._wrapped_kernel import wrapped_gaussian_1d
            return wrapped_gaussian_1d(d, ctx.sigma, ctx.period, ctx.ts,
                                       exponent_denominator=2)
    return np.exp(-(d * d) / (2.0 * ctx.sigma * ctx.sigma))


def _combine(M, r, sym):
    """Σ over r-tuples of distinct children of Π_t M[c_t, t, :].

    ``M`` is (n_children, r, n_q). Symmetric: the Möbius set-partition sum
    over the r slots. Ordered: children in listed order, by a dynamic
    programme over the children."""
    n, _, n_q = M.shape
    if r == 1:
        return M[:, 0, :].sum(axis=0)
    if n < r:
        return np.zeros(n_q, dtype=np.float64)
    if sym:
        unique_blocks, part_block_idx, mus = get_partition_block_structure(r)
        block_contribs = []
        for B in unique_blocks:
            prod = M[:, B[0], :]
            for t in B[1:]:
                prod = prod * M[:, t, :]
            block_contribs.append(prod.sum(axis=0))
        total, _ = mobius_partition_combine(block_contribs, part_block_idx,
                                            mus, track_max=False)
        return total
    # Ordered: dp[t] = Σ over increasing index sequences of length t
    # (slots 0..t-1) of the slot-wise product.
    dp = [np.ones(n_q, dtype=np.float64)] + [
        np.zeros(n_q, dtype=np.float64) for _ in range(r)]
    for i in range(n):
        for t in range(r, 0, -1):
            dp[t] = dp[t] + dp[t - 1] * M[i, t - 1, :]
    return dp[r]


# ---------------------------------------------------------------------
#  Translation quadrature at the co-transposition unit
# ---------------------------------------------------------------------

def _integrate_translation(node, xq, ctx):
    """(1/Z) ∫ value_abs(node, (u, u + x_1, …, u + x_{s-1})) du."""
    s = ctx.span(node.level)
    n_q = xq.shape[1]
    if s < 2:
        # A one-slot unit is translation-degenerate: constant, the total
        # weight of the sub-density (the flat r = 1 convention).
        return np.full(n_q, float(ctx.w[node.val_idx].sum()))
    sigma = ctx.sigma
    if ctx.is_per:
        N_u = max(64, int(math.ceil(ctx.period / sigma * ctx.spp)))
        u_grid = np.linspace(0.0, ctx.period, N_u, endpoint=False)
        du = ctx.period / N_u
    else:
        p_node = ctx.p[node.val_idx]
        x_min = float(min(0.0, xq.min(initial=0.0)))
        x_max = float(max(0.0, xq.max(initial=0.0)))
        u_min = float(p_node.min()) - x_max - 8.0 * sigma
        u_max = float(p_node.max()) - x_min + 8.0 * sigma
        N_u = max(64, int(math.ceil(max(u_max - u_min, 1.0) / sigma
                                    * ctx.spp)))
        u_grid = np.linspace(u_min, u_max, N_u)
        du = None
    # Expanded query set: (s, n_q * N_u), query-major.
    x_full = np.vstack([np.zeros((1, n_q)), xq])          # (s, n_q)
    X = (x_full[:, :, None] + u_grid[None, None, :]).reshape(s, n_q * N_u)
    F = _contract_abs(node, X, ctx).reshape(n_q, N_u)
    if ctx.is_per:
        integral = F.sum(axis=1) * du
    else:
        integral = np.trapezoid(F, u_grid, axis=1)
    return integral / (sigma * math.sqrt(2.0 * math.pi / s))
