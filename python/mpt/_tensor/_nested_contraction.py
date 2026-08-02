"""Fast tree-contraction of the nested-attribute analytical inner product.

Replaces the O((K^leaves)^2) full enumeration in the nested cosine path
with a bottom-up contraction over the tag tree. The contraction reproduces
the exact closed-form inner product (TISMIR preprint Sec 2.6, Eq 7) for the
factorisable cases, and the all-image transposition average over the period
(Eq 6) for the one case that does not factorise per coordinate, relative-periodic.

A single contraction kernel serves all modes; only the per-quadrature-node
leaf-kernel batch and the node reduction differ:

  - absolute (any periodicity): one node, no quadrature -- exact. The kernel
    is a one-body product across coordinates, so each level reduces independently.
  - relative non-periodic: a translation integral over the line, exact to the
    quadrature; the same measure as the analytic relative quadratic.
  - relative periodic: a transposition average over tau in [0, P). This is the
    all-image torus-quotient measure. It is a *different* measure from the
    minimum-image pairwise-wrap that the flat per-attribute path (and the
    nested centres path) computes; the two coincide for sigma << period and
    diverge as sigma approaches the period. Only in this all-image form does
    the relative-periodic kernel factor per coordinate (each tau node is a one-body
    product), which is what makes the per-level orbit reduction available --
    the minimum-image kernel is an irreducible pairwise (two-body) quadratic,
    so it admits no such per-level reduction. The trapezoidal tau-grid is exact
    to quadrature accuracy below a sigma/period threshold (warned elsewhere).

The recipe (tag tree + per-node permutation/combination index arrays) is
built once and reused across the three inner products (XY, XX, YY) and
across all quadrature nodes.

Each symmetric level independently uses the orbit (Möbius) reduction when
its tuple size makes the orbit cheaper than r! enumeration, and explicit
permutation/combination enumeration otherwise (the per-level choice mirrors
the flat path's calibrated policy). The contraction therefore composes with
both the Bulger and Möbius decompositions rather than replacing them.
"""
from __future__ import annotations

import contextlib
import functools
import math
import threading
import warnings
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


@lru_cache(maxsize=None)
def _admitting_sigmas(bound):
    """Largest ``truncationSigmas`` whose floor would admit an error bound.

    The guard admits the orbit route when ``truncation_floor(ts) >= bound``,
    both being absolute quantities on the value scale, and that floor is
    ``exp(-ts^2 / 2)``, so the condition inverts to
    ``ts <= sqrt(-2 ln bound)``. Returns ``None`` when ``bound`` is at or
    above 1 -- the top of the normalised value scale -- where no positive
    setting satisfies it.
    """
    if not (0.0 < bound < 1.0):
        return None
    return math.sqrt(-2.0 * math.log(bound))


from .dispatch import _ORBIT_R_MAX_SHIPPED as _ORBIT_MAX_R


def _orbit_eligible(K, r, sym, is_rel, is_per):
    """Is the Möbius reduction *structurally* available at this level?

    Structure only: the level must be symmetric and r within the shipped
    orbit order. Whether the Möbius route is also the *faster* one is a
    separate question, settled in :func:`_combine_pair`, because it turns
    on the batch extent and no block exists yet when the recipe is built.
    The crossover moves by up to 11 in K across the batch range, so a
    decision taken here could not express it.

    Precision is not judged here either. A size margin between K and r is
    the wrong variable: at r = 2, K = r + 1 the orbit error is 4e-16.
    The estimate in :func:`_combine_pair` measures the error the
    computation actually incurred and compares it against the accuracy
    the caller asked for.
    """
    return bool(sym) and 2 <= r <= _ORBIT_MAX_R


class _Node:
    __slots__ = ("level", "val_idx", "children", "xtup", "ytup", "r", "sym",
                 "use_orbit")

    def __init__(self, level, val_idx, children, xtup, ytup, r, sym,
                 use_orbit=False):
        self.level = level          # tree level (0 = leaf / finest group)
        self.val_idx = val_idx          # global value indices spanned by node
        self.children = children    # list[_Node] (empty at leaf)
        self.xtup = xtup            # (T, r) X-side tuple indices
        self.ytup = ytup            # (T, r) Y-side tuple indices
        self.r = r                  # this level's tuple size
        self.sym = sym              # this level's [sym] flag
        self.use_orbit = use_orbit  # True: orbit-reduce this level (skip xtup)


def build_recipe(r_levels, sym_levels, tags, is_rel=False, is_per=False):
    """Build the contraction tree once.

    ``r_levels`` / ``sym_levels`` are per-level (length L, level 0 = finest).
    ``tags`` is (K_total, L-1): column (l-1) groups values for level l; the
    outermost level L-1 partitions by the last column, level 0 is the
    within-finest-group leaf. Mirrors the enumeration's nesting.
    """
    r_levels = np.asarray(r_levels, dtype=np.intp).ravel()
    sym_levels = np.asarray(sym_levels, dtype=bool).ravel()
    L = int(r_levels.size)
    K_total = int(tags.shape[0]) if tags.ndim else int(tags.size)
    tags2 = tags.reshape(K_total, -1) if tags.ndim == 2 else \
        tags.reshape(K_total, 1)

    def build(level, val_idx):
        val_idx = np.asarray(val_idx, dtype=np.intp)
        if level == 0:
            r0 = int(r_levels[0])
            sy0 = bool(sym_levels[0])
            if _orbit_eligible(len(val_idx), r0, sy0, is_rel, is_per):
                empty = np.empty((0, r0), dtype=np.intp)
                return _Node(0, val_idx, [], empty, empty, r0, sy0, True)
            xt, yt = _tuple_indices(len(val_idx), r0, sy0)
            return _Node(0, val_idx, [], xt, yt, r0, sy0, False)
        col = level - 1
        keys = tags2[val_idx, col]
        children = []
        for k in sorted(set(int(v) for v in keys)):
            sub = val_idx[keys == k]
            children.append(build(level - 1, sub))
        rl = int(r_levels[level])
        syl = bool(sym_levels[level])
        if _orbit_eligible(len(children), rl, syl, is_rel, is_per):
            empty = np.empty((0, rl), dtype=np.intp)
            return _Node(level, val_idx, children, empty, empty, rl, syl, True)
        xt, yt = _tuple_indices(len(children), rl, syl)
        return _Node(level, val_idx, children, xt, yt, rl, syl, False)

    return build(L - 1, np.arange(K_total, dtype=np.intp))


# ----------------------------------------------------------------------
#  Contraction (vectorised over the quadrature batch Q)
# ----------------------------------------------------------------------
def _combine(M, xtup, ytup):
    """Sum_{tx,ty} prod_t M[:, xtup[tx,t], ytup[ty,t]]  -> (Q,)."""
    if xtup.shape[0] == 0 or ytup.shape[0] == 0:
        return np.zeros(M.shape[0], dtype=M.dtype)
    r = xtup.shape[1]
    # prod over the r coordinates; broadcast (Q, Tx, Ty)
    P = M[:, xtup[:, 0][:, None], ytup[:, 0][None, :]]
    for t in range(1, r):
        P = P * M[:, xtup[:, t][:, None], ytup[:, t][None, :]]
    return P.sum(axis=(1, 2))


# Use the orbit (Möbius) reduction at a symmetric level once the level's
# member count makes it cheaper than enumeration. For r <= 6 the choice reuses
# Routing is settled by internal.orbitCostModel / _orbit_cost, a power-law
# model in the quantities each route works on. It replaces two earlier
# criteria: a measured K threshold table covering r = 2..6, and a flop-count
# comparison above that. Neither predicted time well --- against 444 timed
# cells the flop criterion placed the crossover within one of its true K in
# 6 of 21 (r, B) combinations, missing by up to 15, where the model manages
# 19 of 21. Neither took the batch extent, which moves the crossover by up
# to 11 in K.


@functools.lru_cache(maxsize=None)
def _tuple_sides(n, r, sym):
    """Cached (perm-side, comb-side) tuple indices for a size-n level."""
    return _tuple_indices(n, r, sym)


# ---------------------------------------------------------------------------
# Orbit accuracy guard
#
# The Moebius (orbit) reduction sums signed terms that largely cancel, so its
# answer carries fewer digits than the terms it was built from; enumeration
# sums only non-negative terms and loses nothing. Adding the orbit terms
# carries a forward error bounded by eps times the sum of their magnitudes,
# which the orbit routines report alongside the value, so
#
#     estimated error  =  eps * sum|term| / r!
#
# on the same scale _combine_orbit returns. This is a heuristic, not a
# guaranteed limit: sequential summation of n terms admits
# (n - 1) * eps * sum|term| in the worst case. Measured against enumeration
# over four weight profiles and r = 3..7 it came out 2x to 44x above the true
# error, where the earlier |Omega_r| * eps * max|term| form ran 5x to 1100x
# above --- the difference being that the earlier form assumed every term was
# as large as the largest, when in fact they decay.
#
# The guard compares that estimate against the accuracy the caller asked for
# via truncationSigmas, and prefers enumeration when it exceeds it.
_ORBIT_GUARD = threading.local()

# The enumerated fallback materialises a (Q, Tx, Ty) array, so its cost is
# the batch extent times the tuple counts, not the tuple counts alone.
_ORBIT_ENUM_MAX_WORK = 16_000_000    # total kernel products
_ORBIT_ENUM_MAX_ELEMS = 16_000_000   # peak array, ~128 MB at float64


@contextlib.contextmanager
def orbit_guard_scope(truncation_sigmas):
    """Bind the accuracy budget for one batched call.

    Warnings are raised at most once per scope, so a call that walks many
    levels and event pairs reports a single message rather than one per
    block.
    """
    from .._defaults import truncation_floor, get_default
    prev = getattr(_ORBIT_GUARD, "state", None)
    _ORBIT_GUARD.state = {
        "floor": float(truncation_floor(truncation_sigmas)),
        "sigmas": truncation_sigmas,
        "warned_cost": False,
        "warned_accuracy": False,
        "enabled": bool(get_default("post_hoc_guards")),
    }
    try:
        yield
    finally:
        _ORBIT_GUARD.state = prev


def _orbit_budget():
    return getattr(_ORBIT_GUARD, "state", None)


def _n_orbits(r):
    from .._mobius import get_orbit_table
    return len(get_orbit_table(r))


def _enum_work(Q, gx, gy, r):
    """Kernel products the enumerated route performs for this block.

    Chunking bounds peak memory but not total work, so feasibility is
    judged on the work: batch extent times the two tuple counts.
    """
    try:
        return Q * math.comb(gx, r) * math.factorial(r) * math.comb(gy, r)
    except ValueError:
        return float("inf")


def _combine_chunked(M, xtup, ytup, max_elems):
    """Enumerated combine, chunked over the batch to bound peak memory."""
    Q = M.shape[0]
    per = max(xtup.shape[0] * ytup.shape[0], 1)
    step = max(1, int(max_elems // per))
    if step >= Q:
        return _combine(M, xtup, ytup)
    out = np.empty(Q, dtype=M.dtype)
    for s in range(0, Q, step):
        e = min(s + step, Q)
        out[s:e] = _combine(M[s:e], xtup, ytup)
    return out


def _combine_pair(M, r, sym, use_orbit, *, cost_check=True):
    """Combine a (Q, gx, gy) block at one level: X-side perm tuples over gx,
    Y-side comb tuples over gy (the r!-cancelled perm x comb form, same scale
    as ``_combine``). gx and gy are read from the block, so unequal X/Y spans
    -- ragged siblings *or* two densities whose nested cardinalities differ --
    are handled directly. ``r``/``sym`` are the (shared) per-level parameters;
    ``use_orbit`` requests the Moebius reduction (both sides orbit-eligible),
    matching the per-size tuple sourcing of the enumerated path. Both
    enumerated routes -- the one taken when the level is not orbit-eligible
    and the guard's fallback -- chunk over the batch, so peak memory is
    bounded by ``_ORBIT_ENUM_MAX_ELEMS`` whatever the tuple counts.

    For a square block with gx == gy this reproduces the old ``_combine_node``
    exactly (``_tuple_sides`` returns the same cached arrays the recipe stored),
    so the X == Y cosine path is unchanged."""
    from .._defaults import truncation_floor
    # Imported unconditionally: the warning branch below also consults the
    # model, and binding this inside the cost_check branch left it unbound
    # whenever an explicit request bypassed the cost decision.
    from .._orbit_cost import orbit_cost_model
    gx, gy = M.shape[1], M.shape[2]
    if use_orbit:
        # The recipe said the Möbius route is structurally available here.
        # Whether it is also the faster one depends on the batch extent,
        # which only exists now, so the cost question is settled here.
        # K is taken as max(gx, gy), matching how the warning strings
        # report the shape; the model was fitted on square blocks, so a
        # markedly ragged level is outside what it was validated on.
        # ``cost_check=False`` skips this, so an explicit user request for
        # the Möbius route is honoured rather than overridden on cost.
        if cost_check:
            use_orbit = orbit_cost_model(r, max(gx, gy), M.shape[0])[0]
    if use_orbit:
        vals, bound = _combine_orbit(M, r, return_bound=True)
        budget = _orbit_budget()
        if budget is None:
            return vals
        if not budget["enabled"]:
            # ``post_hoc_guards`` is off. The check below inspects a result
            # that has already been computed and, when it diverts, pays for
            # the enumerated route on top of this one --- so with it active
            # the measured cost of the Möbius route is not the cost of
            # choosing it. Calibration runs switch it off so the two routes
            # can be timed as the alternatives they are.
            return vals
        # The bound and the truncation floor are both absolute quantities on
        # the value scale, which is the single error measure the toolbox
        # judges accuracy by. Comparing them directly is the whole test; a
        # ratio to the returned value would reintroduce a denominator that
        # legitimately approaches zero.
        if bound <= budget["floor"]:
            return vals                       # inside the requested accuracy
        work = _enum_work(M.shape[0], gx, gy, r)
        admit = _admitting_sigmas(bound)
        if work <= _ORBIT_ENUM_MAX_WORK:
            if not budget["warned_cost"]:
                budget["warned_cost"] = True
                head = (f"The Mobius route's error bound ({bound:.1e}) exceeds "
                        f"the accuracy implied by truncationSigmas="
                        f"{budget['sigmas']!r} ({budget['floor']:.1e}), so "
                        f"enumeration was used instead.")
                if admit is None:
                    tail = (" The bound is at or above the value scale "
                            "itself, so no truncationSigmas setting would "
                            "admit the Mobius route here.")
                elif orbit_cost_model(r, max(gx, gy), M.shape[0])[0]:
                    # The Mobius route is the cheaper one at this level's
                    # sizes, so trading accuracy for it does buy speed.
                    tail = (f" Setting truncationSigmas to {admit:.3g} or "
                            f"below would admit the Mobius route, which is "
                            f"the faster of the two at r={r}, K={max(gx, gy)},"
                            f" at the cost of an error that may exceed the "
                            f"tighter figure.")
                else:
                    # Enumeration is also the cheaper route at these sizes,
                    # so there is nothing to be gained by loosening the
                    # budget; saying otherwise would offer a false trade.
                    tail = (f" Loosening truncationSigmas would not help: "
                            f"enumeration is also the faster of the two at "
                            f"r={r}, K={max(gx, gy)}.")
                warnings.warn(head + tail, RuntimeWarning, stacklevel=2)
            xtup = _tuple_sides(gx, r, sym)[0]
            ytup = _tuple_sides(gy, r, sym)[1]
            return _combine_chunked(M, xtup, ytup, _ORBIT_ENUM_MAX_ELEMS)
        if not budget["warned_accuracy"]:
            budget["warned_accuracy"] = True
            head = (f"The Mobius route's error bound ({bound:.1e}) exceeds the "
                    f"accuracy implied by truncationSigmas="
                    f"{budget['sigmas']!r} ({budget['floor']:.1e}), and "
                    f"enumeration is not feasible at r={r}, "
                    f"K={max(gx, gy)}. The returned value may carry an error "
                    f"above {budget['floor']:.1e}.")
            if admit is None:
                # The bound is at or above the value scale itself, so no
                # truncation setting can accommodate it; saying otherwise
                # would offer a lever that cannot help.
                tail = (" The bound is at or above the value scale itself, "
                        "so no truncationSigmas setting would admit this "
                        "route; the weight profile is too steeply peaked for "
                        "the Mobius reduction at this tuple size.")
            else:
                tail = (f" Setting truncationSigmas to {admit:.3g} or below "
                        f"would bring the requested accuracy within the "
                        f"bound this is judged against.")
            warnings.warn(head + tail, RuntimeWarning, stacklevel=2)
        return vals
    xtup = _tuple_sides(gx, r, sym)[0]
    ytup = _tuple_sides(gy, r, sym)[1]
    return _combine_chunked(M, xtup, ytup, _ORBIT_ENUM_MAX_ELEMS)


def _combine_orbit(M, r, return_bound=False):
    """(B,) = Sum_{cX,cY} perm(M[cX,cY]) via the partition-lattice orbit
    reduction (= inner_product_orbit_grid / r!), vectorised over the leading
    batch. Supports rectangular M (gx != gy).

    With ``return_bound``, also returns an estimate of the rounding
    error, ``eps * sum|term| / r!`` on the returned scale, which the
    caller compares against the accuracy the user asked for. It is an
    estimate rather than a guaranteed limit; see the note at the return
    site, and tools/calibrate_orbit_cancellation.py for the measured
    conservatism.
    """
    from .._mobius import inner_product_orbit_grid
    gx, gy = M.shape[1], M.shape[2]
    wx = np.ones(gx, dtype=M.dtype)
    wy = np.ones(gy, dtype=M.dtype)
    fr = float(math.factorial(r))
    if return_bound:
        vals, ratios, mass, mass_sum = inner_product_orbit_grid(
            M, wx, wy, r, prefactor=1.0, return_cancellation_ratio=True,
            return_term_mass=True)
    else:
        vals, ratios = inner_product_orbit_grid(
            M, wx, wy, r, prefactor=1.0, return_cancellation_ratio=True)
    # ``ratios`` is requested only because return_term_mass depends on it;
    # the cancellation ratio informs no decision. Accuracy is judged solely
    # by the absolute bound below, against the truncation floor on the value
    # scale.
    out = vals / fr
    if return_bound:
        # Estimated rounding error of the alternating sum, as eps times
        # the sum of the terms' magnitudes.
        #
        # This is NOT a guaranteed upper limit. Sequential summation of
        # n terms admits (n-1) * eps * sum|term| in the worst case, and
        # both this estimate and the |Omega_r| * eps * max|term| it
        # replaces sit below that. Each is a heuristic; this one was
        # measured against enumeration over four weight profiles and
        # r = 3..7 and came out 2x to 44x above the true error, where
        # its predecessor ran 5x to 1100x above. Preferred because it is
        # the better-validated of the two, not because it is provably
        # safe. Widening to the guaranteed limit would refuse the Mobius
        # route almost everywhere; see tests/precision_audit/
        # 17_nested_bound_looseness.py.
        bound = (np.finfo(float).eps * float(np.max(mass_sum))
                 / fr) if mass_sum.size else 0.0
        return out, bound
    return out


# ----------------------------------------------------------------------
#  Bottom-up, batched contraction (vectorised over the quadrature batch
#  AND over sibling pairs). The X and Y trees are walked in lockstep: the
#  rectangular leaf kernel K has shape (Q, nX, nY), so the X axis is indexed
#  by X-side values and the Y axis by Y-side values. When the two densities
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
    span = (len(rep.val_idx) if rep.level == 0 else len(rep.children))
    for nd in nodes:
        s = len(nd.val_idx) if nd.level == 0 else len(nd.children)
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
            Gx[a, nd.val_idx] = 1.0
        Gy = np.zeros((gy, nY), dtype=K.dtype)
        for b, nd in enumerate(ynodes):
            Gy[b, nd.val_idx] = 1.0
        return np.einsum('ai,qij,bj->qab', Gx, K, Gy, optimize=True)
    ux, mx = _siblings_uniform(xnodes)
    uy, my = _siblings_uniform(ynodes)
    if ux and uy:
        blocks = np.empty((gx, gy, Q, mx, my), dtype=K.dtype)
        for a in range(gx):
            sa = K[:, xnodes[a].val_idx]
            for b in range(gy):
                blocks[a, b] = sa[:, :, ynodes[b].val_idx]
        use_orbit = xnodes[0].use_orbit and ynodes[0].use_orbit
        vals = _combine_pair(blocks.reshape(gx * gy * Q, mx, my),
                             r, sym, use_orbit)
        return vals.reshape(gx, gy, Q).transpose(2, 0, 1)
    M = np.empty((Q, gx, gy), dtype=K.dtype)
    for a in range(gx):
        sa = K[:, xnodes[a].val_idx]
        for b in range(gy):
            uo = xnodes[a].use_orbit and ynodes[b].use_orbit
            M[:, a, b] = _combine_pair(sa[:, :, ynodes[b].val_idx], r, sym, uo)
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

    ``K`` is (Q, nX, nY); the X axis is indexed by ``xn`` values, the Y axis by
    ``yn`` values. For the XX / YY inner products (and equal-cardinality XY)
    ``xn`` and ``yn`` are the same recipe and this is the original single-tree
    walk; when the densities' nested cardinalities differ the two trees share
    topology but have differing leaf spans, handled per level."""
    if xn.level == 0:
        block = K[:, xn.val_idx][:, :, yn.val_idx]
        return _combine_pair(block, xn.r, xn.sym,
                             xn.use_orbit and yn.use_orbit)
    Mc = _subtree_overlaps(xn.children, yn.children, K)
    return _combine_pair(Mc, xn.r, xn.sym, xn.use_orbit and yn.use_orbit)


# ----------------------------------------------------------------------
#  Leaf-kernel batches + drivers
# ----------------------------------------------------------------------
def _wrap(d, period):
    return d - period * np.round(d / period)


def _theta_truncation_L(sigma, period, truncation_sigmas):
    """Number of periodic-image shifts per side to include in the 1D
    wrapped Gaussian ``theta(d) = sum_n exp(-(d + n P)^2 / (4 sigma^2))``
    so the first omitted term is below the truncation floor. Callers
    are responsible for having reduced ``d`` to ``[-P/2, P/2]`` first
    (``_wrap``), so the worst-case first-omitted term is at
    ``|d + n P| >= (L + 1/2) P``.
    """
    from .._defaults import truncation_floor
    if sigma <= 0.0 or period <= 0.0:
        return 0
    floor = truncation_floor(truncation_sigmas)
    if floor <= 0.0 or floor >= 1.0:
        floor = 1e-15
    rhs = 2.0 * sigma / period * math.sqrt(-math.log(floor))
    return max(0, int(math.ceil(rhs - 0.5)))


def _trunc(K, sigma, truncation_sigmas):
    """Zero kernel entries below the truncation floor exp(-k^2/2).

    The floor is resolved through :func:`mpt._defaults.truncation_floor`,
    so None takes the default and math.inf takes the finite
    accuracy-floor width (the 1e-12 floor) --- truncation always
    applies, uniformly with every other path.
    """
    from .._defaults import truncation_floor
    floor = truncation_floor(truncation_sigmas)
    K[K < floor] = 0.0
    return K


def _ip_absolute(recipe_x, recipe_y, vX, vY, wX, wY, sigma, is_per, period,
                 truncation_sigmas, wrap_a='full-image'):
    """Absolute-mode inner product for one nested attribute.

    The absolute-mode r-tuple kernel factors across coordinates (unlike
    relative-mode, whose ``Q`` couples them via the projected form),
    so the per-position 1D kernel here is the object the outer contraction
    multiplies across the r_a coordinates. In periodic mode that per-coordinate 1D
    kernel is the wrapped Gaussian (theta):
    ``theta(d) = sum_n exp(-(d + n P)^2 / (4 sigma^2))``. Reducing
    ``d`` to ``[-P/2, P/2]`` first lets ``L = 0`` — i.e. reduce to the
    single-image Gaussian — cover the small-sigma regime, and the
    image sum switches on only when the accuracy floor requires it.
    When the user has opted this attribute into
    ``wrap_a='single-image'`` the L is forced to 0 regardless.

    The abs-per r-tuple kernel is ``prod_a theta(d_a)``. Its lattice
    representation is the sum over ``Z^r`` of Gaussians in the shifted
    r-tuple; the product-of-theta form is the cheaper one to compute
    (``(2L+1) * r`` vs ``(2L+1)^r`` per pair). The r-dim outer product
    is applied downstream in ``_contract``, so this function returns
    the 1D per-position kernel matrix.
    """
    d = vX[:, None] - vY[None, :]
    if is_per:
        d = _wrap(d, period)
        L = (_theta_truncation_L(sigma, period, truncation_sigmas)
             if wrap_a == 'full-image' else 0)
        if L == 0:
            K = np.exp(-d ** 2 / (4.0 * sigma ** 2))
        else:
            n_shift = np.arange(-L, L + 1, dtype=np.float64) * period
            d_shift = d[..., None] + n_shift               # (nX, nY, 2L+1)
            K = np.exp(-d_shift ** 2 / (4.0 * sigma ** 2)).sum(axis=-1)
    else:
        K = np.exp(-d ** 2 / (4.0 * sigma ** 2))
    K = K[None, :, :]                                       # (1, nX, nY)
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


def auto_ntau_default(period, sigma):
    """Transposition-average node count with ``tol`` taken from the global
    ``truncation_sigmas`` default.

    This is the single source of the all-image (relative-periodic) node count
    for every path that evaluates it -- the flat single-multiset and
    multi-attribute Möbius integrators and the nested contraction -- so their
    transposition grids coincide exactly and the same level returns the same
    value whether reached flat or nested.
    """
    from .._defaults import get_default, truncation_floor
    ts = get_default("truncation_sigmas")
    # ``truncation_floor`` resolves None -> default and inf -> the
    # accuracy-floor width, so ``tol`` is the same kernel-value floor
    # every other truncation path uses --- and honours
    # :func:`accuracy_floor_context` when goldens are being regenerated.
    tol = truncation_floor(ts)
    return auto_ntau(period, sigma, tol)


def _ip_rel_periodic(recipe_x, recipe_y, vX, vY, wX, wY, sigma, period,
                     truncation_sigmas, ntau):
    """One event-pair relative-periodic inner product: the transposition
    average over tau in [0, P) of a per-position wrapped-Gaussian kernel.

    The 1D wrapped-Gaussian kernel
    ``theta(d) = sum_n exp(-(d + n P)^2 / (4 sigma^2))`` is the periodic
    (torus) overlap of two unit-height Gaussians; the r-fold product
    ``prod_a theta(delta_a - tau)`` is the r-tuple absolute-mode kernel
    and averaging over tau in [0, P) projects it onto the relative
    (diagonal-invariant) subspace. This is the (C) full-image measure
    (a.k.a. all-image, torus-quotient), agreeing with the flat spectral
    branch to floating-point precision and matching the reference
    lattice sum on 1^perp. The kernel is now unconditional in the
    dispatch's sense of computing (C) whenever this function is called;
    the toolbox's periodic-relative measure no longer depends on
    dispatch except by the user's ``wrap='full-image' vs 'single-image'``
    choice (v3+).

    Truncation of the image sum: ``d`` is nearest-image reduced to
    ``[-P/2, P/2]`` so ``L = 0`` suffices whenever
    ``exp(-(P/2)^2 / (4 sigma^2))`` is below the floor; ``L`` grows
    as the floor tightens or sigma approaches P/2.
    """
    taus = np.linspace(0.0, period, ntau, endpoint=False)
    d = vX[:, None, None] - (vY[None, :, None] + taus[None, None, :])  # nX,nY,T
    d = _wrap(d, period)
    # 1D wrapped-Gaussian theta at each (nX, nY, T). Nearest-image
    # reduction above lets L be small; the sum here is what turns the
    # inner one-body product into the abs-per r-tuple kernel that the
    # outer tau-average projects to rel-per.
    L = _theta_truncation_L(sigma, period, truncation_sigmas)
    if L == 0:
        K = np.exp(-d ** 2 / (4.0 * sigma ** 2))
    else:
        n_shift = np.arange(-L, L + 1, dtype=np.float64) * period
        # (nX, nY, T, 2L+1) then sum over image axis -> (nX, nY, T)
        d_shift = d[..., None] + n_shift
        K = np.exp(-d_shift ** 2 / (4.0 * sigma ** 2)).sum(axis=-1)
    K = K.transpose(2, 0, 1)                                            # T,nX,nY
    K = K * (wX[None, :, None] * wY[None, None, :])
    _trunc(K, sigma, truncation_sigmas)
    return float(_contract(recipe_x, recipe_y, K).mean())


def cos_sim_nested(recipe_x, vX, vY, sigma, *, recipe_y=None, wX=None, wY=None,
                   is_rel, is_per, period, r_total=None, truncation_sigmas=None,
                   ntau=None, wrap_a='full-image'):
    """Cosine similarity via the contraction for one nested attribute.

    ``recipe_x`` describes the X density's nesting; ``recipe_y`` the Y
    density's (defaults to ``recipe_x`` when both sides share the structure).
    Absolute and relative-non-periodic factorise per coordinate and are computed
    exactly (to the line quadrature for relative-non-periodic); relative-
    periodic uses the all-image transposition average over the period -- the
    torus-quotient measure, which differs from the minimum-image pairwise-wrap
    of the flat and centres paths (they coincide for sigma << period).
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
            ntau = auto_ntau_default(period, sigma)
        ip = lambda rx, ry, a, b, wa, wb: _ip_rel_periodic(
            rx, ry, a, b, wa, wb, sigma, period, truncation_sigmas, ntau)
    elif (not is_rel):
        ip = lambda rx, ry, a, b, wa, wb: _ip_absolute(
            rx, ry, a, b, wa, wb, sigma, is_per, period, truncation_sigmas,
            wrap_a)
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


def _shared_leaf_template(node, v, w):
    """Detect a spectral-augmentation leaf: a two-level node whose children are
    all ``r == 1`` leaves sharing one partial template (a common set of offsets
    and weights, translated per child by a single reference value).

    Returns ``(ref_vals, offsets, weights)`` -- the per-child reference value, the
    shared offset profile, and the shared weight profile -- or ``None`` when the
    node is not of this form. The detection is exact (the offsets and weights of
    every child must coincide), so any departure falls back to the generic path.
    """
    if node.level != 1 or not node.children:
        return None
    rep = node.children[0]
    if rep.level != 0 or int(rep.r) != 1 or rep.children:
        return None
    s0 = np.asarray(rep.val_idx, dtype=np.intp)
    width = s0.size
    if width < 2:                      # Kp == 1 is a plain fundamental: leave it
        return None                    # on the generic path (no numerics change)
    v0 = v[s0]
    w0 = w[s0]
    off = v0 - v0[0]
    ref_vals = np.empty(len(node.children), dtype=np.float64)
    for a, ch in enumerate(node.children):
        if ch.level != 0 or int(ch.r) != 1 or ch.children:
            return None
        sa = np.asarray(ch.val_idx, dtype=np.intp)
        if sa.size != width:
            return None
        va = v[sa]
        if not (np.array_equal(va - va[0], off)
                and np.array_equal(w[sa], w0)):
            return None
        ref_vals[a] = va[0]
    return ref_vals, off, w0


def _ip_rel_nonper_factored(recipe_x, recipe_y, vX, vY, wX, wY, sigma,
                            truncation_sigmas, taus):
    """Closed-form inner-partial reduction of the relative-non-periodic inner
    product for spectrally-augmented ordered cells.

    When each side is an ordered cell (outer ``[sym] = 0`` with tuple size equal
    to the cell length) whose tones carry a shared partial template, the inner
    partial index sums analytically into the template cross-correlation
    ``g(delta) = sum_{p,q} wX_p wY_q exp(-(delta + offX_p - offY_q)^2 / 4 sigma^2)``,
    and the cell overlap reduces to the reference-value differences alone:
    ``sum_tau prod_a g(refX_a - refY_a - tau)``. This evaluates only the
    per-position note overlaps, never the full partial-by-partial kernel, and is
    exact to floating-point summation order. Returns ``None`` when the structure
    is not of this form (then the caller uses the generic contraction).
    """
    if recipe_x.sym or recipe_y.sym:
        return None                    # need ordered cells (outer [sym] = 0)
    if (int(recipe_x.r) != len(recipe_x.children)
            or int(recipe_y.r) != len(recipe_y.children)):
        return None                    # need the whole cell as one ordered tuple
    tx = _shared_leaf_template(recipe_x, vX, wX)
    ty = _shared_leaf_template(recipe_y, vY, wY)
    if tx is None or ty is None:
        return None
    cX, offX, wtX = tx
    cY, offY, wtY = ty
    if cX.size != cY.size:             # diagonal needs equal cell lengths
        return None
    dpq = offX[:, None] - offY[None, :]                     # (Kx, Ky)
    wpq = wtX[:, None] * wtY[None, :]
    delta = (cX - cY)[None, :] - taus[:, None]              # (T, r)
    K = np.exp(-(delta[..., None, None] + dpq) ** 2
               / (4.0 * sigma ** 2)) * wpq                  # (T, r, Kx, Ky)
    from .._defaults import truncation_floor
    floor = truncation_floor(truncation_sigmas)
    K[K < floor] = 0.0             # per-term floor, matching _trunc exactly
    m_diag = K.sum(axis=(-1, -2))                           # (T, r)
    return float(m_diag.prod(axis=1).sum())   # common dtau cancels in the cosine


def _all_shared_templates(recipe, P, W):
    """Per-event shared-leaf-template detection across one whole side.

    Returns ``(ref_vals, offsets, weights)`` -- ``ref_vals`` an
    ``(N, n_children)``
    array of the per-event note reference values, and the single offset and weight
    profile common to every event -- or ``None`` when any event departs from
    one shared template (then the caller uses the generic kernel). This is
    :func:`_shared_leaf_template` applied to every event, with the
    offsets and weights required identical across events.
    """
    N = P.shape[1]
    first = _shared_leaf_template(recipe, P[:, 0], W[:, 0])
    if first is None:
        return None
    c0, off, wt = first
    n_children = c0.size
    ref_vals = np.empty((N, n_children), dtype=np.float64)
    ref_vals[0] = c0
    for i in range(1, N):
        ti = _shared_leaf_template(recipe, P[:, i], W[:, i])
        if ti is None:
            return None
        ci, offi, wti = ti
        if (ci.size != n_children or not np.array_equal(offi, off)
                or not np.array_equal(wti, wt)):
            return None
        ref_vals[i] = ci
    return ref_vals, off, wt


def _shared_template_matrix(recipe_x, recipe_y, PX, PY, WX, WY, sigma,
                            truncation_sigmas, taus, mem_budget):
    """Vectorised ``(N_x, N_y)`` relative-non-periodic inner matrix for
    ordered cells carrying a shared partial template.

    When every event on each side is an ordered cell (outer ``[sym] = 0``
    with tuple size equal to the cell length) whose tones share one partial
    template, the inner partial index sums into the template cross-
    correlation -- the offsets and weights are common to every event -- and
    the matrix forms only the per-position note overlaps over the whole
    event-pair grid, never the full partial-by-partial value kernel. This is
    the matrix-form, whole-grid counterpart of
    :func:`_ip_rel_nonper_factored`, evaluated by the same expression and so
    equal to it up to floating-point summation order. Returns ``None`` when
    the structure is not of this form (then the caller uses the generic
    kernel); a NaN-padded (ragged) cell fails detection and so falls back.
    """
    if recipe_x.sym or recipe_y.sym:
        return None                    # need ordered cells (outer [sym] = 0)
    if (int(recipe_x.r) != len(recipe_x.children)
            or int(recipe_y.r) != len(recipe_y.children)):
        return None                    # need the whole cell as one ordered tuple
    tx = _all_shared_templates(recipe_x, PX, WX)
    ty = _all_shared_templates(recipe_y, PY, WY)
    if tx is None or ty is None:
        return None
    cX, offX, wtX = tx                 # cX (Nx, r)
    cY, offY, wtY = ty                 # cY (Ny, r)
    r = cX.shape[1]
    if cY.shape[1] != r:               # diagonal needs equal cell lengths
        return None
    dpq = offX[:, None] - offY[None, :]                    # (Kx, Ky)
    wpq = wtX[:, None] * wtY[None, :]
    Kx, Ky = dpq.shape
    T = int(len(taus))
    Nx, Ny = cX.shape[0], cY.shape[0]
    from .._defaults import truncation_floor
    floor = truncation_floor(truncation_sigmas)
    m_idx = np.repeat(np.arange(Nx), Ny)
    n_idx = np.tile(np.arange(Ny), Nx)
    B = Nx * Ny
    per = r * max(T, 1) * Kx * Ky
    chunk = max(1, min(B, int(mem_budget // max(per, 1))))
    out = np.empty(B, dtype=np.float64)
    for s0 in range(0, B, chunk):
        e0 = min(s0 + chunk, B)
        cx = cX[m_idx[s0:e0]]                              # (nb, r)
        cy = cY[n_idx[s0:e0]]                              # (nb, r)
        delta = (cx - cy)[:, :, None] - taus[None, None, :]   # (nb, r, T)
        K = np.exp(-(delta[..., None, None] + dpq) ** 2
                   / (4.0 * sigma ** 2)) * wpq             # (nb, r, T, Kx, Ky)
        K[K < floor] = 0.0
        m_diag = K.sum(axis=(-1, -2))                      # (nb, r, T)
        out[s0:e0] = m_diag.prod(axis=1).sum(axis=1)       # prod over r, sum over T
    return out.reshape(Nx, Ny)


def _ip_rel_nonper_generic(recipe_x, recipe_y, vX, vY, wX, wY, sigma,
                           truncation_sigmas, taus):
    d = vX[:, None, None] - (vY[None, :, None] + taus[None, None, :])
    K = np.exp(-d ** 2 / (4.0 * sigma ** 2)).transpose(2, 0, 1)   # (T, nX, nY)
    K = K * (wX[None, :, None] * wY[None, None, :])
    _trunc(K, sigma, truncation_sigmas)
    return float(_contract(recipe_x, recipe_y, K).sum())  # common dtau cancels


def _ip_rel_nonper(recipe_x, recipe_y, vX, vY, wX, wY, sigma,
                   truncation_sigmas, taus):
    fast = _ip_rel_nonper_factored(recipe_x, recipe_y, vX, vY, wX, wY, sigma,
                                   truncation_sigmas, taus)
    if fast is not None:
        return fast
    return _ip_rel_nonper_generic(recipe_x, recipe_y, vX, vY, wX, wY, sigma,
                                  truncation_sigmas, taus)


def _tau_window(vx, vy, taus, sigma, truncation_sigmas):
    """Per-pair slice of a uniform line tau-grid, or ``(None, None)``.

    On the line the grid must span the whole value range, because any two
    events may be that far apart; but a single event pair aligns only over
    the taus near its own offset. Every other node gives kernel entries
    below the truncation floor, which are zeroed and then summed as exact
    zeros -- so restricting each pair to its own window leaves the result
    bit-identical while the grid shrinks from the passage's range to the
    event's own.

    A kernel entry survives truncation when
    ``|v_x - v_y - tau| <= 2 * sigma * sqrt(-log(floor))``, so the window
    runs from ``min(v_x) - max(v_y)`` to ``max(v_x) - min(v_y)``, widened
    by that margin at each end.

    All windows share one width so the pairs stay in a single batch: the
    start index varies per pair, and ``valid`` masks the tail where a
    clamped window overruns its own end (at the grid edges).

    Returns the taus for each pair, shape ``(nb, W)``, with the matching
    mask; or ``(None, None)`` when windowing would not pay, in which case
    the caller uses the whole grid.
    """
    from .._defaults import truncation_floor
    T = int(taus.shape[0])
    if T < 3:
        return None, None
    step = float(taus[1] - taus[0])
    if not np.isfinite(step) or step <= 0.0:
        return None, None          # not a uniform ascending grid
    floor = float(truncation_floor(truncation_sigmas))
    if not (0.0 < floor < 1.0):
        return None, None          # no truncation, so no node is removable
    margin = 2.0 * sigma * math.sqrt(-math.log(floor))

    t0 = float(taus[0])
    lo = vx.min(axis=0) - vy.max(axis=0) - margin      # (nb,)
    hi = vx.max(axis=0) - vy.min(axis=0) + margin
    start = np.clip(np.floor((lo - t0) / step).astype(np.int64), 0, T - 1)
    stop = np.clip(np.ceil((hi - t0) / step).astype(np.int64), 0, T - 1)
    W = int((stop - start).max()) + 1
    if W >= T:
        return None, None          # the window is the grid; nothing saved

    idx = start[:, None] + np.arange(W)[None, :]       # (nb, W)
    valid = (idx <= stop[:, None]).astype(np.float64)
    np.minimum(idx, T - 1, out=idx)
    return taus[idx], valid


def nested_attr_matrix(recipe_x, recipe_y, PX, PY, WX, WY, sigma,
                       is_per, period, truncation_sigmas, *, taus=None,
                       periodic_taus=True, taus_reduce="mean",
                       mem_budget=16_000_000):
    """(N_x, N_y) per-attribute inner matrix via the per-level contraction,
    vectorised over the whole event-pair grid.

    This is the matrix form of :func:`nested_ip`: instead of looping the
    event-pair grid in Python, the grid is folded into the leading batch axis
    of :func:`_contract`, so every (i, j) entry is reduced per level -- the
    orbit (Möbius) reduction at symmetric levels, enumeration at ordered ones,
    selected by :func:`build_recipe` -- exactly as the flat per-attribute
    matrix reduces a single level, and with the same memory profile (only the
    small per-event value kernel and the per-level intermediates are formed,
    never the materialised tuple set).

    ``PX``/``PY`` are ``(K, N)`` value arrays; ``WX``/``WY`` the matching
    weights or ``None``. NaN-padded (variable-K) values are carried as
    zero-weight. The mode is set by ``taus``:

    - ``taus=None`` -- the absolute (``is_per=False``) or absolute-periodic
      (``is_per=True``) inner product, a one-body product per coordinate.
    - ``taus`` given with ``periodic_taus=True``, ``taus_reduce='mean'`` -- the
      relative-periodic transposition average over the period (the all-image
      torus measure, which is what makes the per-level orbit reduction
      available for relative-periodic).
    - ``taus`` given with ``periodic_taus=False``, ``taus_reduce='sum'`` -- the
      relative-non-periodic translation integral over the line (the constant
      step cancels in the cosine, so the raw sum is returned). This equals the
      analytic relative quadratic to the grid accuracy and, unlike the centres
      path, never materialises the tuple set -- the win for spectrally
      augmented cells, where the inner partial level reduces by einsum.

    The transposition / translation nodes share the batch axis with the event
    pairs and are reduced per pair after the contraction. ``mem_budget`` caps
    the per-chunk leaf-kernel size.
    """
    with orbit_guard_scope(truncation_sigmas):
        return _nested_attr_matrix_impl(
            recipe_x, recipe_y, PX, PY, WX, WY, sigma,
            is_per, period, truncation_sigmas, taus=taus,
            periodic_taus=periodic_taus,
            taus_reduce=taus_reduce, mem_budget=mem_budget)


def _nested_attr_matrix_impl(recipe_x, recipe_y, PX, PY, WX, WY, sigma,
                             is_per, period, truncation_sigmas, *,
                             taus=None, periodic_taus=True,
                             taus_reduce="mean",
                             mem_budget=16_000_000):
    """Body of nested_attr_matrix, run inside the accuracy scope."""
    PX = np.asarray(PX, dtype=np.float64)
    PY = np.asarray(PY, dtype=np.float64)
    nX, Nx = PX.shape
    nY, Ny = PY.shape
    WX = np.ones((nX, Nx)) if WX is None else np.asarray(WX, dtype=np.float64)
    WY = np.ones((nY, Ny)) if WY is None else np.asarray(WY, dtype=np.float64)
    # Relative-non-periodic ordered cells carrying a shared partial template
    # (spectral augmentation) reduce the inner partial index analytically; the
    # vectorised template matrix forms only the per-position note overlaps,
    # never the full value kernel below. Falls through when not of that form.
    if taus is not None and not periodic_taus and taus_reduce == "sum":
        M = _shared_template_matrix(recipe_x, recipe_y, PX, PY, WX, WY, sigma,
                                    truncation_sigmas, taus, mem_budget)
        if M is not None:
            return M
    if np.isnan(PX).any() or np.isnan(PY).any():
        fill = float(min(np.nanmin(PX), np.nanmin(PY)))
        mX, mY = np.isnan(PX), np.isnan(PY)
        PX = np.where(mX, fill, PX)
        WX = np.where(mX | np.isnan(WX), 0.0, WX)
        PY = np.where(mY, fill, PY)
        WY = np.where(mY | np.isnan(WY), 0.0, WY)
    inv = 1.0 / (4.0 * sigma ** 2)
    T = 0 if taus is None else int(len(taus))
    m_idx = np.repeat(np.arange(Nx), Ny)
    n_idx = np.tile(np.arange(Ny), Nx)
    B = Nx * Ny
    per = max(T, 1) * nX * nY
    chunk = max(1, min(B, int(mem_budget // max(per, 1))))
    out = np.empty(B, dtype=np.float64)
    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        mi, ni = m_idx[s:e], n_idx[s:e]
        vx, vy = PX[:, mi], PY[:, ni]               # (nX, nb), (nY, nb)
        wx, wy = WX[:, mi], WY[:, ni]
        if taus is None:
            d = vx.T[:, :, None] - vy.T[:, None, :]  # (nb, nX, nY)
            if is_per:
                d = _wrap(d, period)
            K = np.exp(-(d ** 2) * inv)
            K = K * (wx.T[:, :, None] * wy.T[:, None, :])
            _trunc(K, sigma, truncation_sigmas)
            out[s:e] = _contract(recipe_x, recipe_y, K)
        else:
            tw, valid = None, None
            if not periodic_taus:
                tw, valid = _tau_window(vx, vy, taus, sigma,
                                        truncation_sigmas)
            if tw is None:
                d = (vx.T[:, :, None, None]
                     - (vy.T[:, None, :, None] + taus[None, None, None, :]))
                if periodic_taus:
                    d = _wrap(d, period)
                W = T
            else:
                d = (vx.T[:, :, None, None]
                     - (vy.T[:, None, :, None] + tw[:, None, None, :]))
                W = tw.shape[1]
            K = np.exp(-(d ** 2) * inv)
            K = K * (wx.T[:, :, None, None] * wy.T[:, None, :, None])
            nb = e - s
            K = K.transpose(0, 3, 1, 2).reshape(nb * W, nX, nY)
            _trunc(K, sigma, truncation_sigmas)
            vals = _contract(recipe_x, recipe_y, K).reshape(nb, W)
            if valid is not None:
                vals = vals * valid
            if taus_reduce == "mean":
                # The mean is over the whole grid, so a window divides by the
                # full node count, not by the window width.
                out[s:e] = vals.sum(1) / T
            else:
                out[s:e] = vals.sum(1)
    return out.reshape(Nx, Ny)


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

    def count(val_idx, level, use_sym):
        if level == 0:
            r0 = r_levels[0]
            c = comb(len(val_idx), r0)
            return c * (fact(r0) if (use_sym and sym_levels[0]) else 1)
        col = level - 1
        keys = tags2[val_idx, col]
        groups = {}
        for sidx, k in zip(val_idx.tolist(), keys.tolist()):
            groups.setdefault(int(k), []).append(sidx)
        rl = r_levels[level]
        subs = [count(np.asarray(grp, dtype=np.intp), level - 1, use_sym)
                for grp in groups.values()]
        e = e_r(subs, rl)
        return e * (fact(rl) if (use_sym and sym_levels[level]) else 1)

    all_values = np.arange(K_total, dtype=np.intp)
    m_perm = count(all_values, L - 1, True)
    m_comb = count(all_values, L - 1, False)
    return int(m_perm), int(m_comb)


def recipe_work(recipe: _Node):
    """Approximate combine flop count of one contraction over the tree.

    Orbit-eligible symmetric levels (5 <= r <= 8) are costed at the orbit
    reduction's |Omega_r| * K^2 rather than the enumerated r! * C(K,r)^2,
    so the dispatch reflects the actual route taken at each level.
    """
    from .._mobius import get_orbit_table

    def node_combine_cost(node):
        if node.use_orbit:
            K = len(node.val_idx) if node.level == 0 else len(node.children)
            return len(get_orbit_table(node.r)) * K * K * max(1, node.r)
        return node.xtup.shape[0] * node.ytup.shape[0] * max(1, node.r)

    def w(node):
        tot = node_combine_cost(node)
        if node.level != 0:
            K = len(node.children)
            tot += K * K
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
              truncation_sigmas, quad, wrap_a='full-image'):
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
                            bool(quad["is_per"]), period, truncation_sigmas,
                            wrap_a)
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
