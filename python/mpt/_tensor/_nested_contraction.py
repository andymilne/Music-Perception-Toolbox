"""Fast tree-contraction of the nested-attribute analytical inner product.

Replaces the O((K^leaves)^2) full enumeration in the nested cosine path
with a bottom-up contraction over the tag tree. The contraction reproduces
the exact closed-form inner product (TISMIR preprint Sec 2.6, Eq 7) for the
factorisable cases, and a transposition-average surrogate for the one case
that does not factorise (relative + periodic, Eq 6).

A single contraction kernel serves all modes; only the per-quadrature-node
leaf-kernel batch and the node reduction differ:

  - absolute (any periodicity): one node, no quadrature -- exact.
  - relative periodic (outer): transposition-average over tau in [0, P) --
    surrogate, exact below a sigma/period threshold (warned elsewhere).

The recipe (tag tree + per-node permutation/combination index arrays) is
built once and reused across the three inner products (XY, XX, YY) and
across all quadrature nodes.

Each symmetric level independently uses the orbit (Möbius) reduction when
its read-arity makes the orbit cheaper than r! enumeration, and explicit
permutation/combination enumeration otherwise (the per-level choice mirrors
the flat path's calibrated policy). The contraction therefore composes with
both the Bulger and Möbius decompositions rather than replacing them.
"""
from __future__ import annotations

import functools
import math
from functools import lru_cache
from itertools import combinations, permutations

import numpy as np


# ----------------------------------------------------------------------
#  Recipe: tag tree + cached permutation / combination index arrays
# ----------------------------------------------------------------------
@lru_cache(maxsize=None)
def _tuple_indices(n: int, r: int, sym: bool):
    """X-side and Y-side tuple index arrays over ``range(n)``.

    X side uses permutations of r-combinations when ``sym`` (exchangeable
    multiset), else combinations only; Y side is always combinations. This
    is the permutation/combination reduction whose r! cancels in the
    cosine. Returns (Xtup, Ytup) as (T, r) int arrays.
    """
    combs = list(combinations(range(n), r))
    if not combs:
        empty = np.empty((0, r), dtype=np.intp)
        return empty, empty
    ytup = np.asarray(combs, dtype=np.intp)
    if sym:
        xtup = np.asarray([p for c in combs for p in permutations(c)],
                          dtype=np.intp)
    else:
        xtup = ytup
    return xtup, ytup


def _perm_count(g, r):
    # C(g, r) * r!  = number of ordered r-tuples of distinct items from g.
    if r > g or r < 0:
        return 0
    n = 1
    for i in range(r):
        n *= (g - i)
    return n


def _orbit_eligible(g, r, sym, is_rel, is_per):
    if not sym or not (2 <= r <= _ORBIT_MAX_R):
        return False
    if not _flat_orbit_precision_ok([r], [g]):
        return False                       # K (=g) too close to r: cancellation
    if r <= 6:
        return bool(_flat_orbit_beats_enum(r, g, is_rel, is_per))
    return True                            # r in 7..R_MAX: enumeration infeasible


class _Node:
    __slots__ = ("level", "slots", "children", "xtup", "ytup", "r", "sym",
                 "use_orbit")

    def __init__(self, level, slots, children, xtup, ytup, r, sym,
                 use_orbit=False):
        self.level = level          # tree level (0 = leaf / finest group)
        self.slots = slots          # global slot indices spanned by node
        self.children = children    # list[_Node] (empty at leaf)
        self.xtup = xtup            # (T, r) X-side tuple indices
        self.ytup = ytup            # (T, r) Y-side tuple indices
        self.r = r                  # this level's read-arity
        self.sym = sym              # this level's [sym] flag
        self.use_orbit = use_orbit  # True: orbit-reduce this level (skip xtup)


def build_recipe(r_levels, sym_levels, tags, is_rel=False, is_per=False):
    """Build the contraction tree once.

    ``r_levels`` / ``sym_levels`` are per-level (length L, level 0 = finest).
    ``tags`` is (K_total, L-1): column (l-1) groups slots for level l; the
    outermost level L-1 partitions by the last column, level 0 is the
    within-finest-group leaf. Mirrors the enumeration's nesting.
    """
    r_levels = np.asarray(r_levels, dtype=np.intp).ravel()
    sym_levels = np.asarray(sym_levels, dtype=bool).ravel()
    L = int(r_levels.size)
    K_total = int(tags.shape[0]) if tags.ndim else int(tags.size)
    tags2 = tags.reshape(K_total, -1) if tags.ndim == 2 else \
        tags.reshape(K_total, 1)

    def build(level, slots):
        slots = np.asarray(slots, dtype=np.intp)
        if level == 0:
            r0 = int(r_levels[0])
            sy0 = bool(sym_levels[0])
            if _orbit_eligible(len(slots), r0, sy0, is_rel, is_per):
                empty = np.empty((0, r0), dtype=np.intp)
                return _Node(0, slots, [], empty, empty, r0, sy0, True)
            xt, yt = _tuple_indices(len(slots), r0, sy0)
            return _Node(0, slots, [], xt, yt, r0, sy0, False)
        col = level - 1
        keys = tags2[slots, col]
        children = []
        for k in sorted(set(int(v) for v in keys)):
            sub = slots[keys == k]
            children.append(build(level - 1, sub))
        rl = int(r_levels[level])
        syl = bool(sym_levels[level])
        if _orbit_eligible(len(children), rl, syl, is_rel, is_per):
            empty = np.empty((0, rl), dtype=np.intp)
            return _Node(level, slots, children, empty, empty, rl, syl, True)
        xt, yt = _tuple_indices(len(children), rl, syl)
        return _Node(level, slots, children, xt, yt, rl, syl, False)

    return build(L - 1, np.arange(K_total, dtype=np.intp))


# ----------------------------------------------------------------------
#  Contraction (vectorised over the quadrature batch Q)
# ----------------------------------------------------------------------
def _combine(M, xtup, ytup):
    """Sum_{tx,ty} prod_t M[:, xtup[tx,t], ytup[ty,t]]  -> (Q,)."""
    if xtup.shape[0] == 0 or ytup.shape[0] == 0:
        return np.zeros(M.shape[0], dtype=M.dtype)
    r = xtup.shape[1]
    # prod over the r tuple positions; broadcast (Q, Tx, Ty)
    P = M[:, xtup[:, 0][:, None], ytup[:, 0][None, :]]
    for t in range(1, r):
        P = P * M[:, xtup[:, t][:, None], ytup[:, t][None, :]]
    return P.sum(axis=(1, 2))


# Use the orbit (Möbius) reduction at a symmetric level once r is large
# enough that |Omega_r| beats r! (r! crosses the orbit-entry count near r=5).
# The orbit-vs-enumeration choice at each symmetric level reuses the flat
# path's calibrated policy (dispatch._orbit_beats_pairwise_per_attr K-vs-r
# crossover + _orbit_safe_for_precision K>=r+2 guard), applied per level with
# K = g (the level's child/slot count). For r in 7..R_MAX (no flat K-threshold
# entry, and where enumeration's C(g,r)*r! is infeasible anyway) orbit is the
# only viable route, so it is used whenever precision-safe.
from .dispatch import (
    _orbit_beats_pairwise_per_attr as _flat_orbit_beats_enum,
    _orbit_safe_for_precision as _flat_orbit_precision_ok,
    _ORBIT_R_MAX_SHIPPED as _ORBIT_MAX_R,
)
# Below this alternating-sum cancellation ratio the orbit value has lost too
# many digits; fall back to the enumerated combine for that quadrature node.
_ORBIT_CANCEL_FLOOR = 1e-10


def _node_span(node):
    return len(node.slots) if node.level == 0 else len(node.children)


@functools.lru_cache(maxsize=None)
def _tuple_sides(n, r, sym):
    """Cached (perm-side, comb-side) tuple indices for a size-n level."""
    return _tuple_indices(n, r, sym)


def _combine_pair(M, r, sym, use_orbit):
    """Combine a (Q, gx, gy) block at one level: X-side perm tuples over gx,
    Y-side comb tuples over gy (the r!-cancelled perm x comb form, same scale
    as ``_combine``). gx and gy are read from the block, so unequal X/Y spans
    -- ragged siblings *or* two densities whose nested cardinalities differ --
    are handled directly. ``r``/``sym`` are the (shared) per-level parameters;
    ``use_orbit`` requests the Moebius reduction (both sides orbit-eligible),
    matching the per-size tuple sourcing of the enumerated path.

    For a square block with gx == gy this reproduces the old ``_combine_node``
    exactly (``_tuple_sides`` returns the same cached arrays the recipe stored),
    so the X == Y cosine path is unchanged."""
    if use_orbit:
        empty = np.empty((0, r), dtype=np.intp)
        return _combine_orbit(M, r, empty, empty)
    gx, gy = M.shape[1], M.shape[2]
    xtup = _tuple_sides(gx, r, sym)[0]
    ytup = _tuple_sides(gy, r, sym)[1]
    return _combine(M, xtup, ytup)


def _combine_orbit(M, r, xtup, ytup):
    """(B,) = Sum_{cX,cY} perm(M[cX,cY]) via the partition-lattice orbit
    reduction (= inner_product_orbit_grid / r!), vectorised over the leading
    batch, with a cancellation guard that reverts to the enumerated combine
    for any element that loses digits. Supports rectangular M (gx != gy)."""
    from .._mobius import inner_product_orbit_grid
    gx, gy = M.shape[1], M.shape[2]
    wx = np.ones(gx, dtype=M.dtype)
    wy = np.ones(gy, dtype=M.dtype)
    fr = float(math.factorial(r))
    vals, ratios = inner_product_orbit_grid(
        M, wx, wy, r, prefactor=1.0, return_cancellation_ratio=True)
    out = vals / fr
    bad = ratios < _ORBIT_CANCEL_FLOOR
    if np.any(bad):
        if xtup.shape[0] > 0:            # enumerated fallback is feasible
            idx = np.nonzero(bad)[0]
            out[idx] = _combine(M[idx], xtup, ytup)
        else:
            import warnings
            warnings.warn(
                "Nested orbit reduction lost precision to alternating-sum "
                "cancellation at a symmetric level where enumeration is "
                "infeasible; the value may be inaccurate.", stacklevel=2)
    return out


# ----------------------------------------------------------------------
#  Bottom-up, batched contraction (vectorised over the quadrature batch
#  AND over sibling pairs). The X and Y trees are walked in lockstep: the
#  rectangular leaf kernel K has shape (Q, nX, nY), so the X axis is indexed
#  by X-side slots and the Y axis by Y-side slots. When the two densities
#  share an identical nested structure (the XX, YY, and equal-cardinality XY
#  cases) the X and Y node lists coincide and every step reduces to the
#  earlier single-tree code. When their leaf cardinalities differ (e.g. a
#  4-pitch prototype against an 8-pitch window), the spans gx and gy differ
#  per level and the per-size tuple sourcing in ``_combine_pair`` handles it.
#  An r0=1 leaf level collapses to a single weighted block-sum (einsum);
#  every higher level batches its gx*gy combines into one call. Ragged levels
#  (siblings with differing child counts or structure) fall back to a
#  per-pair loop, preserving correctness.
# ----------------------------------------------------------------------
def _siblings_uniform(nodes):
    rep = nodes[0]
    span = (len(rep.slots) if rep.level == 0 else len(rep.children))
    for nd in nodes:
        s = len(nd.slots) if nd.level == 0 else len(nd.children)
        if (s != span or nd.r != rep.r or nd.sym != rep.sym
                or nd.use_orbit != rep.use_orbit):
            return False, span
    return True, span


def _leaf_overlaps(xnodes, ynodes, K):
    """(Q, gx, gy) pairwise overlaps among leaf siblings, X-side vs Y-side."""
    gx, gy = len(xnodes), len(ynodes)
    Q, nX, nY = K.shape
    r, sym = xnodes[0].r, xnodes[0].sym
    if r == 1:
        # r0 = 1: M[a,b] = sum_{i in Sxa, j in Syb} K[:, i, j] (weights folded).
        Gx = np.zeros((gx, nX), dtype=K.dtype)
        for a, nd in enumerate(xnodes):
            Gx[a, nd.slots] = 1.0
        Gy = np.zeros((gy, nY), dtype=K.dtype)
        for b, nd in enumerate(ynodes):
            Gy[b, nd.slots] = 1.0
        return np.einsum('ai,qij,bj->qab', Gx, K, Gy, optimize=True)
    ux, mx = _siblings_uniform(xnodes)
    uy, my = _siblings_uniform(ynodes)
    if ux and uy:
        blocks = np.empty((gx, gy, Q, mx, my), dtype=K.dtype)
        for a in range(gx):
            sa = K[:, xnodes[a].slots]
            for b in range(gy):
                blocks[a, b] = sa[:, :, ynodes[b].slots]
        use_orbit = xnodes[0].use_orbit and ynodes[0].use_orbit
        vals = _combine_pair(blocks.reshape(gx * gy * Q, mx, my),
                             r, sym, use_orbit)
        return vals.reshape(gx, gy, Q).transpose(2, 0, 1)
    M = np.empty((Q, gx, gy), dtype=K.dtype)
    for a in range(gx):
        sa = K[:, xnodes[a].slots]
        for b in range(gy):
            uo = xnodes[a].use_orbit and ynodes[b].use_orbit
            M[:, a, b] = _combine_pair(sa[:, :, ynodes[b].slots], r, sym, uo)
    return M


def _subtree_overlaps(xnodes, ynodes, K):
    """(Q, gx, gy) pairwise overlaps among sibling subtrees, X-side vs Y."""
    if xnodes[0].level == 0:
        return _leaf_overlaps(xnodes, ynodes, K)
    gx, gy = len(xnodes), len(ynodes)
    Q = K.shape[0]
    r, sym = xnodes[0].r, xnodes[0].sym
    xsizes = [len(nd.children) for nd in xnodes]
    xoffs = np.cumsum([0] + xsizes)
    ysizes = [len(nd.children) for nd in ynodes]
    yoffs = np.cumsum([0] + ysizes)
    xflat = [c for nd in xnodes for c in nd.children]
    yflat = [c for nd in ynodes for c in nd.children]
    Mc = _subtree_overlaps(xflat, yflat, K)         # (Q, Gcx, Gcy)
    ux, gcx = _siblings_uniform(xnodes)
    uy, gcy = _siblings_uniform(ynodes)
    if ux and uy:
        blocks = np.empty((gx, gy, Q, gcx, gcy), dtype=K.dtype)
        for a in range(gx):
            ra = slice(xoffs[a], xoffs[a] + gcx)
            for b in range(gy):
                blocks[a, b] = Mc[:, ra, yoffs[b]:yoffs[b] + gcy]
        use_orbit = xnodes[0].use_orbit and ynodes[0].use_orbit
        vals = _combine_pair(blocks.reshape(gx * gy * Q, gcx, gcy),
                             r, sym, use_orbit)
        return vals.reshape(gx, gy, Q).transpose(2, 0, 1)
    M = np.empty((Q, gx, gy), dtype=K.dtype)
    for a in range(gx):
        ra = slice(xoffs[a], xoffs[a + 1])
        for b in range(gy):
            uo = xnodes[a].use_orbit and ynodes[b].use_orbit
            M[:, a, b] = _combine_pair(
                Mc[:, ra, yoffs[b]:yoffs[b + 1]], r, sym, uo)
    return M


def _contract(xn: _Node, yn: _Node, K):
    """Overlap (Q,) of two nested structures under rectangular kernel ``K``.

    ``K`` is (Q, nX, nY); the X axis is indexed by ``xn`` slots, the Y axis by
    ``yn`` slots. For the XX / YY inner products (and equal-cardinality XY)
    ``xn`` and ``yn`` are the same recipe and this is the original single-tree
    walk; when the densities' nested cardinalities differ the two trees share
    topology but have differing leaf spans, handled per level."""
    if xn.level == 0:
        block = K[:, xn.slots][:, :, yn.slots]
        return _combine_pair(block, xn.r, xn.sym,
                             xn.use_orbit and yn.use_orbit)
    Mc = _subtree_overlaps(xn.children, yn.children, K)
    return _combine_pair(Mc, xn.r, xn.sym, xn.use_orbit and yn.use_orbit)


# ----------------------------------------------------------------------
#  Leaf-kernel batches + drivers
# ----------------------------------------------------------------------
def _wrap(d, period):
    return d - period * np.round(d / period)


def _trunc(K, sigma, truncation_sigmas):
    """Apply the exp(-truncation_sigmas^2 / 2) kernel cutoff in place."""
    if truncation_sigmas is None or not math.isfinite(truncation_sigmas):
        return K
    floor = math.exp(-0.5 * truncation_sigmas ** 2)
    K[K < floor] = 0.0
    return K


def _ip_absolute(recipe_x, recipe_y, vX, vY, wX, wY, sigma, is_per, period,
                 truncation_sigmas):
    d = vX[:, None] - vY[None, :]
    if is_per:
        d = _wrap(d, period)
    K = np.exp(-d ** 2 / (4.0 * sigma ** 2))[None, :, :]   # (1, nX, nY)
    K = K * (wX[None, :, None] * wY[None, None, :])
    _trunc(K, sigma, truncation_sigmas)
    return float(_contract(recipe_x, recipe_y, K)[0])


def auto_ntau(period, sigma, tol):
    """Trapezoidal nodes for the transposition average.

    The integrand is periodic-smooth with bandwidth ~ period/sigma, so the
    trapezoidal rule converges spectrally past ~pi*period/sigma nodes. A
    generous multiple keeps the quadrature error well under ``tol``.
    """
    base = 2.0 * math.pi * period / sigma
    margin = 1.0 + 0.5 * max(0.0, -math.log10(max(tol, 1e-16))) / 12.0
    return int(max(64, math.ceil(base * margin)))


def _ip_rel_periodic(recipe_x, recipe_y, vX, vY, wX, wY, sigma, period,
                     truncation_sigmas, ntau):
    taus = np.linspace(0.0, period, ntau, endpoint=False)
    d = vX[:, None, None] - (vY[None, :, None] + taus[None, None, :])  # nX,nY,T
    d = _wrap(d, period)
    K = np.exp(-d ** 2 / (4.0 * sigma ** 2)).transpose(2, 0, 1)        # T,nX,nY
    K = K * (wX[None, :, None] * wY[None, None, :])
    _trunc(K, sigma, truncation_sigmas)
    return float(_contract(recipe_x, recipe_y, K).mean())


def cos_sim_nested(recipe_x, vX, vY, sigma, *, recipe_y=None, wX=None, wY=None,
                   is_rel, is_per, period, r_total=None, truncation_sigmas=None,
                   ntau=None):
    """Cosine similarity via the contraction for one nested attribute.

    ``recipe_x`` describes the X density's nesting; ``recipe_y`` the Y
    density's (defaults to ``recipe_x`` when both sides share the structure).
    Absolute and relative-non-periodic factorise exactly; relative-periodic
    uses the transposition-average surrogate.
    """
    if recipe_y is None:
        recipe_y = recipe_x
    n = vX.shape[0]
    if wX is None:
        wX = np.ones(n)
    if wY is None:
        wY = np.ones(vY.shape[0])
    if is_rel and is_per:
        if ntau is None:
            tol = max(math.exp(-0.5 * (truncation_sigmas or math.inf) ** 2),
                      1e-12)
            ntau = auto_ntau(period, sigma, tol)
        ip = lambda rx, ry, a, b, wa, wb: _ip_rel_periodic(
            rx, ry, a, b, wa, wb, sigma, period, truncation_sigmas, ntau)
    elif (not is_rel):
        ip = lambda rx, ry, a, b, wa, wb: _ip_absolute(
            rx, ry, a, b, wa, wb, sigma, is_per, period, truncation_sigmas)
    else:  # relative, non-periodic
        tol = max(math.exp(-0.5 * (truncation_sigmas or math.inf) ** 2), 1e-12)
        taus = auto_taus_line(np.concatenate([vX, vY]),
                              np.concatenate([vX, vY]), sigma, tol)
        ip = lambda rx, ry, a, b, wa, wb: _ip_rel_nonper(
            rx, ry, a, b, wa, wb, sigma, truncation_sigmas, taus)
    xy = ip(recipe_x, recipe_y, vX, vY, wX, wY)
    xx = ip(recipe_x, recipe_x, vX, vX, wX, wX)
    yy = ip(recipe_y, recipe_y, vY, vY, wY, wY)
    return xy / math.sqrt(xx * yy)


# ----------------------------------------------------------------------
#  Relative non-periodic: transposition integral over the line (exact)
# ----------------------------------------------------------------------
def auto_taus_line(vX, vY, sigma, tol):
    """Trapezoidal tau-grid over the line for the relative quotient.

    The integrand is a sum of Gaussians in tau spanning the alignment range
    +/- a few sigma; a step <~ sigma resolves it (smooth -> fast trapezoid).
    """
    spread = float(max(np.max(vX), np.max(vY)) - min(np.min(vX), np.min(vY)))
    pad = (6.0 + 0.5 * max(0.0, -math.log10(max(tol, 1e-16)))) * sigma
    hi = spread + pad
    step = sigma / 4.0
    n = int(max(64, math.ceil(2.0 * hi / step)))
    return np.linspace(-hi, hi, n)


def _ip_rel_nonper(recipe_x, recipe_y, vX, vY, wX, wY, sigma,
                   truncation_sigmas, taus):
    d = vX[:, None, None] - (vY[None, :, None] + taus[None, None, :])
    K = np.exp(-d ** 2 / (4.0 * sigma ** 2)).transpose(2, 0, 1)   # (T, nX, nY)
    K = K * (wX[None, :, None] * wY[None, None, :])
    _trunc(K, sigma, truncation_sigmas)
    return float(_contract(recipe_x, recipe_y, K).sum())  # common dtau cancels


# ----------------------------------------------------------------------
#  Dispatch support: analytic tuple counts and contraction work
# ----------------------------------------------------------------------
def tuple_counts(r_levels, sym_levels, tags):
    """(M_perm, M_comb): per-event perm/comb tuple counts, fully analytic.

    Mirrors the counts of _nested_enum_indices without enumerating. At each
    level the sum over r-group selections of the product of per-group
    sub-counts is the elementary symmetric polynomial e_r of those
    sub-counts (DP, no subset enumeration); a symmetric level multiplies by
    r!. The leaf contributes C(group_size, r_0) times r_0! when symmetric.
    Used for the speed dispatch (enumeration cost ~ N^2 * M_perm * M_comb).
    """
    r_levels = [int(x) for x in np.asarray(r_levels).ravel()]
    sym_levels = [bool(x) for x in np.asarray(sym_levels).ravel()]
    L = len(r_levels)
    K_total = int(tags.shape[0]) if tags.ndim == 2 else int(tags.size)
    tags2 = (tags.reshape(K_total, -1) if tags.ndim == 2
             else tags.reshape(K_total, 1))

    def fact(m):
        f = 1
        for i in range(2, m + 1):
            f *= i
        return f

    def comb(nn, kk):
        if kk < 0 or kk > nn:
            return 0
        num = 1
        for i in range(kk):
            num = num * (nn - i) // (i + 1)
        return num

    def e_r(xs, k):
        # elementary symmetric polynomial of degree k (e_0 = 1; e_k = 0 if
        # k > len(xs)), Newton-free DP in O(len(xs) * k).
        e = [1] + [0] * k
        for x in xs:
            for j in range(k, 0, -1):
                e[j] += e[j - 1] * x
        return e[k]

    def count(slots, level, use_sym):
        if level == 0:
            r0 = r_levels[0]
            c = comb(len(slots), r0)
            return c * (fact(r0) if (use_sym and sym_levels[0]) else 1)
        col = level - 1
        keys = tags2[slots, col]
        groups = {}
        for sidx, k in zip(slots.tolist(), keys.tolist()):
            groups.setdefault(int(k), []).append(sidx)
        rl = r_levels[level]
        subs = [count(np.asarray(g, dtype=np.intp), level - 1, use_sym)
                for g in groups.values()]
        e = e_r(subs, rl)
        return e * (fact(rl) if (use_sym and sym_levels[level]) else 1)

    all_slots = np.arange(K_total, dtype=np.intp)
    m_perm = count(all_slots, L - 1, True)
    m_comb = count(all_slots, L - 1, False)
    return int(m_perm), int(m_comb)


def recipe_work(recipe: _Node):
    """Approximate combine flop count of one contraction over the tree.

    Orbit-eligible symmetric levels (5 <= r <= 8) are costed at the orbit
    reduction's |Omega_r| * g^2 rather than the enumerated r! * C(g,r)^2,
    so the dispatch reflects the actual route taken at each level.
    """
    from .._mobius import get_orbit_table

    def node_combine_cost(node):
        if node.use_orbit:
            g = len(node.slots) if node.level == 0 else len(node.children)
            return len(get_orbit_table(node.r)) * g * g * max(1, node.r)
        return node.xtup.shape[0] * node.ytup.shape[0] * max(1, node.r)

    def w(node):
        tot = node_combine_cost(node)
        if node.level != 0:
            g = len(node.children)
            tot += g * g
            for c in node.children:
                tot += w(c)
        return tot
    return int(w(recipe))


def quad_nodes(is_rel, is_per, sigma, period, vmin, vmax, truncation_sigmas):
    """Quadrature node count Q for the cost estimate (1 / ntau / ntau_line)."""
    if not is_rel:
        return 1
    tol = max(math.exp(-0.5 * (truncation_sigmas or math.inf) ** 2), 1e-12)
    if is_per:
        return auto_ntau(period, sigma, tol)
    spread = float(vmax - vmin)
    pad = (6.0 + 0.5 * max(0.0, -math.log10(max(tol, 1e-16)))) * sigma
    return int(max(64, math.ceil(2.0 * (spread + pad) / (sigma / 4.0))))


# ----------------------------------------------------------------------
#  Per-event-pair bare inner product (mode dispatch; shared quadrature)
# ----------------------------------------------------------------------
def make_quadrature(is_rel, is_per, sigma, period, vmin, vmax,
                    truncation_sigmas):
    """Shared quadrature grid for all event-pairs and the IP triple.

    A common grid means the constant dtau / 1-over-ntau factor is identical
    across XY, XX, YY and cancels in the cosine. Returns a dict the IP
    helper consumes.
    """
    if not is_rel:
        return {"mode": "abs", "is_per": bool(is_per)}
    tol = max(math.exp(-0.5 * (truncation_sigmas or math.inf) ** 2), 1e-12)
    if is_per:
        ntau = auto_ntau(period, sigma, tol)
        return {"mode": "relper",
                "taus": np.linspace(0.0, period, ntau, endpoint=False)}
    spread = float(vmax - vmin)
    pad = (6.0 + 0.5 * max(0.0, -math.log10(max(tol, 1e-16)))) * sigma
    hi = spread + pad
    n = int(max(64, math.ceil(2.0 * hi / (sigma / 4.0))))
    return {"mode": "relnonper", "taus": np.linspace(-hi, hi, n)}


def nested_ip(recipe_x, recipe_y, vX, vY, wX, wY, sigma, period,
              truncation_sigmas, quad):
    """Bare inner product for one event-pair, on the shared quadrature.

    ``recipe_x`` indexes the X (``vX``) axis of the rectangular kernel,
    ``recipe_y`` the Y (``vY``) axis; pass the same recipe for both when the
    two densities share their nested structure.
    """
    mode = quad["mode"]
    if mode == "abs":
        # Periodicity comes from the density's [per] flag (threaded through the
        # quadrature dict), not from whether ``period`` happens to be finite:
        # an absolute non-periodic attribute may still carry a finite period.
        return _ip_absolute(recipe_x, recipe_y, vX, vY, wX, wY, sigma,
                            bool(quad["is_per"]), period, truncation_sigmas)
    if mode == "relper":
        taus = quad["taus"]
        d = vX[:, None, None] - (vY[None, :, None] + taus[None, None, :])
        d = _wrap(d, period)
        K = np.exp(-d ** 2 / (4.0 * sigma ** 2)).transpose(2, 0, 1)
        K = K * (wX[None, :, None] * wY[None, None, :])
        _trunc(K, sigma, truncation_sigmas)
        return float(_contract(recipe_x, recipe_y, K).sum())
    # relnonper
    return _ip_rel_nonper(recipe_x, recipe_y, vX, vY, wX, wY, sigma,
                          truncation_sigmas, quad["taus"])
