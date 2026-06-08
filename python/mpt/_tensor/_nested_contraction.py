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
"""
from __future__ import annotations

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


class _Node:
    __slots__ = ("level", "slots", "children", "xtup", "ytup")

    def __init__(self, level, slots, children, xtup, ytup):
        self.level = level          # tree level (0 = leaf / finest group)
        self.slots = slots          # global slot indices spanned by node
        self.children = children    # list[_Node] (empty at leaf)
        self.xtup = xtup            # (T, r) X-side tuple indices
        self.ytup = ytup            # (T, r) Y-side tuple indices


def build_recipe(r_levels, sym_levels, tags):
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
            xt, yt = _tuple_indices(len(slots), r0, bool(sym_levels[0]))
            return _Node(0, slots, [], xt, yt)
        col = level - 1
        keys = tags2[slots, col]
        children = []
        for k in sorted(set(int(v) for v in keys)):
            sub = slots[keys == k]
            children.append(build(level - 1, sub))
        rl = int(r_levels[level])
        xt, yt = _tuple_indices(len(children), rl, bool(sym_levels[level]))
        return _Node(level, slots, children, xt, yt)

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


def _contract(xn: _Node, yn: _Node, K):
    """Overlap (Q,) between xn's slots (X side) and yn's slots (Y side)."""
    if xn.level == 0:
        sub = K[:, xn.slots][:, :, yn.slots]            # (Q, m, m)
        return _combine(sub, xn.xtup, yn.ytup)
    g = len(xn.children)
    Q = K.shape[0]
    M = np.empty((Q, g, g), dtype=K.dtype)
    for a in range(g):
        xa = xn.children[a]
        for b in range(g):
            M[:, a, b] = _contract(xa, yn.children[b], K)
    return _combine(M, xn.xtup, yn.ytup)


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


def _ip_absolute(recipe, vX, vY, wX, wY, sigma, is_per, period,
                 truncation_sigmas):
    d = vX[:, None] - vY[None, :]
    if is_per:
        d = _wrap(d, period)
    K = np.exp(-d ** 2 / (4.0 * sigma ** 2))[None, :, :]   # (1, n, n)
    K = K * (wX[None, :, None] * wY[None, None, :])
    _trunc(K, sigma, truncation_sigmas)
    return float(_contract(recipe, recipe, K)[0])


def auto_ntau(period, sigma, tol):
    """Trapezoidal nodes for the transposition average.

    The integrand is periodic-smooth with bandwidth ~ period/sigma, so the
    trapezoidal rule converges spectrally past ~pi*period/sigma nodes. A
    generous multiple keeps the quadrature error well under ``tol``.
    """
    base = 2.0 * math.pi * period / sigma
    margin = 1.0 + 0.5 * max(0.0, -math.log10(max(tol, 1e-16))) / 12.0
    return int(max(64, math.ceil(base * margin)))


def _ip_rel_periodic(recipe, vX, vY, wX, wY, sigma, period,
                     truncation_sigmas, ntau):
    taus = np.linspace(0.0, period, ntau, endpoint=False)
    d = vX[:, None, None] - (vY[None, :, None] + taus[None, None, :])  # n,n,T
    d = _wrap(d, period)
    K = np.exp(-d ** 2 / (4.0 * sigma ** 2)).transpose(2, 0, 1)        # T,n,n
    K = K * (wX[None, :, None] * wY[None, None, :])
    _trunc(K, sigma, truncation_sigmas)
    return float(_contract(recipe, recipe, K).mean())


def cos_sim_nested(recipe, vX, vY, sigma, *, wX=None, wY=None, is_rel,
                   is_per, period, r_total=None, truncation_sigmas=None,
                   ntau=None):
    """Cosine similarity via the contraction for one nested attribute.

    Absolute and relative-non-periodic factorise exactly; relative-periodic
    uses the transposition-average surrogate.
    """
    n = vX.shape[0]
    if wX is None:
        wX = np.ones(n)
    if wY is None:
        wY = np.ones(n)
    if is_rel and is_per:
        if ntau is None:
            tol = max(math.exp(-0.5 * (truncation_sigmas or math.inf) ** 2),
                      1e-12)
            ntau = auto_ntau(period, sigma, tol)
        ip = lambda a, b, wa, wb: _ip_rel_periodic(
            recipe, a, b, wa, wb, sigma, period, truncation_sigmas, ntau)
    elif (not is_rel):
        ip = lambda a, b, wa, wb: _ip_absolute(
            recipe, a, b, wa, wb, sigma, is_per, period, truncation_sigmas)
    else:  # relative, non-periodic
        tol = max(math.exp(-0.5 * (truncation_sigmas or math.inf) ** 2), 1e-12)
        taus = auto_taus_line(np.concatenate([vX, vY]),
                              np.concatenate([vX, vY]), sigma, tol)
        ip = lambda a, b, wa, wb: _ip_rel_nonper(
            recipe, a, b, wa, wb, sigma, truncation_sigmas, taus)
    xy = ip(vX, vY, wX, wY)
    xx = ip(vX, vX, wX, wX)
    yy = ip(vY, vY, wY, wY)
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


def _ip_rel_nonper(recipe, vX, vY, wX, wY, sigma, truncation_sigmas, taus):
    d = vX[:, None, None] - (vY[None, :, None] + taus[None, None, :])
    K = np.exp(-d ** 2 / (4.0 * sigma ** 2)).transpose(2, 0, 1)   # (T, n, n)
    K = K * (wX[None, :, None] * wY[None, None, :])
    _trunc(K, sigma, truncation_sigmas)
    return float(_contract(recipe, recipe, K).sum())   # common dtau cancels


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
    """Approximate combine flop count of one contraction over the tree."""
    def w(node):
        if node.level == 0:
            return node.xtup.shape[0] * node.ytup.shape[0] * \
                max(1, node.xtup.shape[1])
        tot = node.xtup.shape[0] * node.ytup.shape[0] * \
            max(1, node.xtup.shape[1])
        g = len(node.children)
        tot += g * g  # child-pair assembly
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
        return {"mode": "abs"}
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


def nested_ip(recipe, vX, vY, wX, wY, sigma, period, truncation_sigmas, quad):
    """Bare inner product for one event-pair, on the shared quadrature."""
    mode = quad["mode"]
    if mode == "abs":
        # is_per folded into the kernel by the caller's value wrapping? No:
        # absolute periodic wrap is applied here via period when finite.
        return _ip_absolute(recipe, vX, vY, wX, wY, sigma,
                            math.isfinite(period) and period > 0, period,
                            truncation_sigmas)
    if mode == "relper":
        taus = quad["taus"]
        d = vX[:, None, None] - (vY[None, :, None] + taus[None, None, :])
        d = _wrap(d, period)
        K = np.exp(-d ** 2 / (4.0 * sigma ** 2)).transpose(2, 0, 1)
        K = K * (wX[None, :, None] * wY[None, None, :])
        _trunc(K, sigma, truncation_sigmas)
        return float(_contract(recipe, recipe, K).sum())
    # relnonper
    return _ip_rel_nonper(recipe, vX, vY, wX, wY, sigma, truncation_sigmas,
                          quad["taus"])
