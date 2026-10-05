"""Culling and cache-sized chunks on the joint-centres path of ``eval_maet``.

The joint-centres path (taken where an attribute is at r = 1 or carries
a kernel covariance) evaluates each query against only the centres
within the truncation width on one culling coordinate, and holds a dense
chunk to a cache-sized working set. Neither may change a value: a
culled pair is one the truncation would have set to zero, so culled and
dense evaluations agree to rounding. What is pinned here is that
agreement on every kind of culling coordinate (absolute, periodic on a
single image, relative, nested inner unit, kernel covariance), on the
attributes that are never culled on (periodic on the full image), at
the boundary of the window, across the wrap of a periodic coordinate,
and across block and chunk boundaries; that the blocks of queries are
the longest their three bounds allow, and that a block on a periodic
coordinate holds no centre twice; that a window reaching round the
cycle onto itself is not culled on; and that the decision to cull
follows the kernel width. The MATLAB twin is
``tests/test_ma_eval_cull.m``.
"""

import math

import numpy as np
import pytest

import mpt
import mpt._tensor.eval as ev


def _build(ps, sigma, r, rel, per, period, exch=None, wrap=None, w=None):
    w = [np.ones(np.shape(p)) for p in ps] if w is None else w
    args = [ps, w, sigma, r, rel, per, period]
    if exch is not None:
        args.append(exch)
    kw = {} if wrap is None else {"wrap": wrap}
    return mpt.build_maet(*args, verbose=False, **kw)


def _queries(dens, n, rng, spread=0.2):
    """Half near actual centres, half spread over their range."""
    c = np.vstack(dens.centres)
    pick = rng.integers(0, c.shape[1], n // 2)
    near = c[:, pick] + rng.normal(0.0, spread, (dens.dim, n // 2))
    lo = c.min(axis=1, keepdims=True) - 1.0
    hi = c.max(axis=1, keepdims=True) + 1.0
    far = lo + (hi - lo) * rng.uniform(0.0, 1.0, (dens.dim, n - n // 2))
    return np.hstack([near, far])


def _eval(dens, X, mode, monkeypatch):
    monkeypatch.setattr(ev, "_MA_CULL_MODE", mode)
    return mpt.eval_maet(dens, X, method="centres", verbose=False)


def _assert_cull_matches_dense(dens, X, monkeypatch, rtol=1e-12):
    dense = _eval(dens, X, "never", monkeypatch)
    culled = _eval(dens, X, "always", monkeypatch)
    scale = max(float(np.max(np.abs(dense))), 1e-300)
    assert np.max(np.abs(culled - dense)) <= rtol * scale
    assert np.any(dense > 0.0)


N = 300
RNG_SEED = 11


def _onset(rng, n=N, span=60.0):
    return np.sort(rng.uniform(0.0, span, n))[None, :]


def _shapes():
    """Every kind of culling coordinate, each beside an onset at r = 1."""
    rng = np.random.default_rng(RNG_SEED)
    on = _onset(rng)
    T2 = np.repeat(np.arange(2), 3)
    A_ = rng.standard_normal((2, 2))
    Sig = A_ @ A_.T * 0.05 + 0.02 * np.eye(2)
    out = {
        "absolute, r = 1 twice": _build(
            [on, rng.uniform(60, 72, (1, N))], [0.1, 0.3], [1, 1],
            [0, 0], [0, 0], [0, 0]),
        "absolute at r = 2": _build(
            [on, rng.uniform(60, 72, (3, N))], [0.1, 0.3], [1, 2],
            [0, 0], [0, 0], [0, 0]),
        "relative at r = 3": _build(
            [on, rng.uniform(60, 72, (4, N))], [0.1, 0.3], [1, 3],
            [0, 1], [0, 0], [0, 0]),
        "periodic absolute, single image": _build(
            [on, rng.uniform(0, 12, (2, N))], [0.1, 0.3], [1, 2],
            [0, 0], [0, 1], [0, 12], wrap=["full-image", "single-image"]),
        "periodic absolute, full image, tabulated": _build(
            [on, np.round(rng.uniform(0, 12, (2, N)))], [0.1, 0.3], [1, 2],
            [0, 0], [0, 1], [0, 12]),
        "periodic relative": _build(
            [on, rng.uniform(0, 12, (3, N))], [0.1, 0.3], [1, 2],
            [0, 1], [0, 1], [0, 12]),
        "kernel covariance": _build(
            [rng.uniform(0, 10, (2, N)), on], [Sig, 0.1], [2, 1],
            [0, 0], [0, 0], [0, 0], exch=[False, True]),
        "nested inner unit": mpt.build_maet(
            [np.sort(rng.uniform(0, 12, (6, N)), axis=0), on], None,
            specs=[{"tags": T2, "r": [1, 2], "exch": [True, True],
                    "rel": [0, 1]},
                   {"r": 1, "exch": True, "rel": False}],
            sigma=[0.5, 0.1], per=[False, False], period=[0.0, 0.0],
            verbose=False),
    }
    pn = rng.uniform(60, 72, (3, N))
    pn[2, ::3] = np.nan
    wts = rng.uniform(0, 1, (1, N))
    wts[0, ::5] = 0.0
    out["absent values and zero weights"] = _build(
        [on, pn], [0.1, 0.3], [1, 2], [0, 0], [0, 0], [0, 0],
        w=[wts, np.ones((3, N))])
    return out


SHAPES = _shapes()


@pytest.mark.parametrize("name", list(SHAPES))
@pytest.mark.parametrize("k", [6.0, math.inf])
def test_culled_equals_dense(name, k, monkeypatch):
    mpt.set_default(truncation_sigmas=k)
    dens = SHAPES[name]
    X = _queries(dens, 400, np.random.default_rng(1))
    _assert_cull_matches_dense(dens, X, monkeypatch)


@pytest.mark.parametrize("name", ["absolute at r = 2", "relative at r = 3",
                                  "periodic relative"])
def test_culled_equals_dense_in_single_precision(name, monkeypatch):
    mpt.set_default(truncation_sigmas=6.0, kernel_precision="single")
    dens = SHAPES[name]
    X = _queries(dens, 400, np.random.default_rng(2))
    _assert_cull_matches_dense(dens, X, monkeypatch, rtol=1e-5)


def test_relative_window_is_wide_enough(monkeypatch):
    # The culling coordinate is the relative attribute's one difference
    # (r = 2, where Q equals D^2 / 2 exactly), so pairs whose difference
    # lies between k sigma and sqrt(2) k sigma carry weight between
    # exp(-k^2 / 2) and exp(-k^2 / 4) and are lost to a window without
    # the sqrt(2). The attribute at r = 1 puts the density on the
    # joint-centres path and, with a wide kernel over one value, adds
    # nothing to the exponent.
    k, s = 3.0, 0.5
    mpt.set_default(truncation_sigmas=k)
    rng = np.random.default_rng(3)
    dens = _build([rng.uniform(0, 100, (2, 400)), np.zeros((1, 400))],
                  [s, 50.0], [2, 1], [1, 0], [0, 0], [0, 0])
    c = dens.centres[0]
    pick = rng.integers(0, c.shape[1], 600)
    X = np.vstack([c[:, pick] + rng.uniform(-1.4, 1.4, 600) * k * s,
                   np.zeros(600)])
    _assert_cull_matches_dense(dens, X, monkeypatch)


def test_window_boundary(monkeypatch):
    # Queries at exactly k sigma from a centre on the culling coordinate,
    # and a hair either side, are kept or dropped as the truncation
    # itself would keep or drop them.
    k, s = 3.0, 0.5
    mpt.set_default(truncation_sigmas=k)
    on = np.arange(0.0, 200.0, 10.0)[None, :]
    dens = _build([on, np.zeros((1, on.shape[1]))], [s, 5.0], [1, 1],
                  [0, 0], [0, 0], [0, 0])
    offs = k * s * np.array([1.0 - 1e-12, 1.0, 1.0 + 1e-12])
    xs = (on[0, 5] + np.concatenate([offs, -offs]))
    X = np.vstack([xs, np.zeros_like(xs)])
    X = np.tile(X, (1, 8))
    _assert_cull_matches_dense(dens, X, monkeypatch, rtol=0.0)


def test_periodic_window_wraps(monkeypatch):
    # Centres and queries crowd both ends of the cycle, so each window
    # crosses the wrap.
    mpt.set_default(truncation_sigmas=6.0)
    rng = np.random.default_rng(5)
    P = 4.0
    ph = np.concatenate([rng.uniform(0.0, 0.2, 100),
                         rng.uniform(P - 0.2, P, 100)])[None, :]
    dens = _build([ph, rng.uniform(60, 72, (1, 200))], [0.02, 0.3],
                  [1, 1], [0, 0], [1, 0], [P, 0],
                  wrap=["single-image", "full-image"])
    X = np.vstack([np.concatenate([rng.uniform(0.0, 0.1, 200),
                                   rng.uniform(P - 0.1, P, 200)]),
                   rng.uniform(60, 72, 400)])
    _assert_cull_matches_dense(dens, X, monkeypatch)


def test_blocks_and_chunks(monkeypatch):
    # A tiny chunk budget splits the culled queries into many blocks and
    # the dense evaluation into many chunks; neither changes a value.
    mpt.set_default(truncation_sigmas=6.0)
    dens = SHAPES["absolute at r = 2"]
    X = _queries(dens, 300, np.random.default_rng(6))
    whole = _eval(dens, X, "never", monkeypatch)
    mpt.set_default(kernel_chunk_bytes=4096)
    dense = _eval(dens, X, "never", monkeypatch)
    culled = _eval(dens, X, "always", monkeypatch)
    assert np.max(np.abs(dense - whole)) <= 1e-12 * np.max(np.abs(whole))
    assert np.max(np.abs(culled - whole)) <= 1e-12 * np.max(np.abs(whole))


def test_queries_beyond_every_centre(monkeypatch):
    mpt.set_default(truncation_sigmas=6.0)
    dens = SHAPES["absolute, r = 1 twice"]
    X = np.vstack([np.full(50, 1e4), np.full(50, 66.0)])
    assert np.all(_eval(dens, X, "always", monkeypatch) == 0.0)


def _plan(dens, X):
    from mpt._tensor.dispatch import _inner_r_vec
    xl = [X[int(sum(dens.dim_per_attr[:a])):int(sum(dens.dim_per_attr[:a + 1]))]
          for a in range(dens.n_attrs)]
    return ev._ma_cull_plan(
        dens.centres, xl, int(dens.n_j), X.shape[1], dens.dim_per_attr,
        dens.sigma, dens.rel, dens.per, dens.period, _inner_r_vec(dens),
        getattr(dens, "wrap", None), 6.0, np.float64)


def test_full_image_periodic_is_never_culled_on():
    rng = np.random.default_rng(7)
    dens = _build([rng.uniform(0, 4, (1, 500)), rng.uniform(0, 12, (1, 500))],
                  [0.02, 0.1], [1, 1], [0, 0], [1, 1], [4.0, 12.0])
    X = rng.uniform(0, 4, (2, 400))
    assert ev._ma_cull_candidates(
        dens.dim_per_attr, dens.sigma, dens.rel, dens.per, dens.period,
        None, getattr(dens, "wrap", None), 6.0) == []
    assert _plan(dens, X) is None


def test_decision_follows_the_kernel_width():
    rng = np.random.default_rng(8)
    on = rng.uniform(0, 100, (1, 3000))
    pitch = rng.uniform(60, 72, (1, 3000))
    X = np.vstack([rng.uniform(0, 100, 500), rng.uniform(60, 72, 500)])
    narrow = _build([on, pitch], [0.05, 0.3], [1, 1], [0, 0], [0, 0], [0, 0])
    wide = _build([on, pitch], [8.0, 3.0], [1, 1], [0, 0], [0, 0], [0, 0])
    plan = _plan(narrow, X)
    assert plan is not None
    # The culling coordinate is the onset, whose spread is widest
    # against its window: each query meets few centres.
    order, lo, hi = plan
    assert np.mean(hi - lo) < 0.05 * narrow.n_j
    assert _plan(wide, X) is None


def _block_ok(lo, hi, q, s, e, n_j, group_pairs, block_cost):
    width = int(hi[q[e - 1]] - lo[q[s]])
    pairs = width * (e - s)
    own = int(np.sum(hi[q[s:e]] - lo[q[s:e]]))
    return (width <= n_j and pairs <= group_pairs
            and pairs - own <= block_cost)


@pytest.mark.parametrize("seed", range(4))
def test_blocks_are_the_longest_their_bounds_allow(seed):
    # Runs as a plan makes them: both ends non-decreasing in a key the
    # queries do not arrive sorted by, some runs empty.
    rng = np.random.default_rng(seed)
    n_q, n_j = 400, 1000
    key = rng.uniform(0.0, 1.0, n_q)
    lo = np.floor(np.clip(key - 0.05, 0.0, 1.0) * n_j).astype(np.int64)
    hi = np.floor(np.clip(key + 0.05, 0.0, 1.0) * n_j).astype(np.int64)
    empty = rng.uniform(size=n_q) < 0.1
    hi[empty] = lo[empty]
    hi = np.maximum(hi, lo)
    group_pairs, block_cost = int(rng.integers(500, 20000)), 600
    q, starts = ev._ma_cull_blocks(lo, hi, n_j, group_pairs, block_cost)
    assert sorted(q.tolist()) == np.flatnonzero(hi > lo).tolist()
    assert np.all(np.diff(lo[q]) >= 0) and np.all(np.diff(hi[q]) >= 0)
    ends = np.append(starts[1:], q.size)
    for s, e in zip(starts.tolist(), ends.tolist()):
        assert e - s == 1 or _block_ok(lo, hi, q, s, e, n_j, group_pairs,
                                       block_cost)
        if e < q.size:
            assert not _block_ok(lo, hi, q, s, e + 1, n_j, group_pairs,
                                 block_cost)


def test_periodic_block_holds_no_centre_twice(monkeypatch):
    # Few centres on a periodic coordinate, a window a third of the
    # cycle wide, and no limit on a block's wasted pairs: a block's range
    # of the plan's order, which repeats the centres near either end of
    # the cycle, would otherwise cover some centres twice.
    mpt.set_default(truncation_sigmas=6.0)
    monkeypatch.setattr(ev, "_MA_CULL_BLOCK_COST", 10 ** 9)
    rng = np.random.default_rng(9)
    P, n = 1.0, 12
    with pytest.warns(UserWarning, match="opted into the single-image"):
        dens = _build([rng.uniform(0, P, (1, n)),
                       rng.uniform(60, 72, (1, n))],
                      [0.06, 2.0], [1, 1], [0, 0], [1, 0], [P, 0],
                      wrap=["single-image", "full-image"])
    X = np.vstack([rng.uniform(0, P, 200), rng.uniform(60, 72, 200)])
    _assert_cull_matches_dense(dens, X, monkeypatch)


def test_window_reaching_round_the_cycle_is_not_culled_on(monkeypatch):
    # 2 k sigma falls short of the period by less than the margin that
    # widens the window, so the widened window would reach round the
    # cycle onto itself: the coordinate is a candidate, but the plan
    # passes it over, even when culling is forced. The other attribute,
    # periodic on the full image, is never a candidate.
    mpt.set_default(truncation_sigmas=6.0)
    monkeypatch.setattr(ev, "_MA_CULL_MODE", "always")
    rng = np.random.default_rng(10)
    P = 1.0
    s = (0.5 * P - 1e-12) / 6.0
    with pytest.warns(UserWarning, match="opted into the single-image"):
        dens = _build([rng.uniform(0, P, (1, 50)),
                       rng.uniform(0, 12, (1, 50))],
                      [s, 0.3], [1, 1], [0, 0], [1, 1], [P, 12.0],
                      wrap=["single-image", "full-image"])
    X = np.vstack([rng.uniform(0, P, 40), rng.uniform(0, 12, 40)])
    assert len(ev._ma_cull_candidates(
        dens.dim_per_attr, dens.sigma, dens.rel, dens.per, dens.period,
        None, dens.wrap, 6.0)) == 1
    assert _plan(dens, X) is None
