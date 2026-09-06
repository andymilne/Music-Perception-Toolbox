"""Per-level Möbius point evaluation of nested densities.

``eval_exp_tens(method='mobius')`` on a nested density runs the per-level
Möbius evaluator (:mod:`mpt._tensor._nested_mobius_eval`), which must
agree with the tuple-centres route on every nested shape: any depth,
symmetric or ordered levels, absolute or any co-transposition unit,
periodic or not, either ``wrap``, ragged (NaN-padded) events, and a
nested attribute tensored with flat ones. Mirror of MATLAB
tests/test_nested_mobius_eval.m; the reference block shares its numbers
with that file.
"""
import math

import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, eval_exp_tens
from mpt._tensor._nested_mobius_eval import eval_nested_attr_orbit

P = 12.0


@pytest.fixture(autouse=True)
def _at_the_accuracy_floor():
    prev = mpt.get_default('truncation_sigmas')
    mpt.set_default(truncation_sigmas=math.inf)
    try:
        yield
    finally:
        mpt.set_default(truncation_sigmas=prev)


def _density(tags, r, sym, rel, per, K, seed, *, sigma=0.7, N=2,
             wrap='full-image'):
    rng = np.random.default_rng(seed)
    p = np.sort(rng.uniform(0.0, P, size=(K, N)), axis=0)
    spec = {"tags": tags, "r": r, "sym": sym, "rel": rel}
    return build_exp_tens([p], None, specs=[spec], sigma=[sigma],
                          is_per=[per], period=[P], wrap=[wrap],
                          verbose=False)


def _queries(d, seed, n_far=3, n_near=3):
    rng = np.random.default_rng(seed + 100)
    X = rng.uniform(-2.0, 14.0, size=(d.dim, n_far + n_near))
    if d.dim and n_near:
        # Near queries sit on actual tuple centres (all attributes stacked,
        # in slot order), so the density carries mass there; a query
        # assembled from independently drawn coordinates lands at ~1e-14
        # of the peak, where both routes are at their noise floor.
        c = np.vstack(d.centres)
        pick = rng.integers(0, c.shape[1], n_near)
        X[:, :n_near] = c[:, pick] + rng.normal(0.0, 0.3, size=(d.dim, n_near))
    return X


def _assert_routes_agree(d, X, rtol):
    # Both routes at a 40-sigma width: at the accuracy-floor width the
    # centres route drops every tuple kernel below 1e-12, and over a few
    # thousand tuples the dropped mass can reach 1e-9 of the maximum
    # (seen on the MATLAB twin), which is truncation, not disagreement.
    vc = eval_exp_tens(d, X, method="centres", truncation_sigmas=40.0,
                       verbose=False)
    vm = eval_exp_tens(d, X, method="mobius", truncation_sigmas=40.0,
                       verbose=False)
    scale = max(float(np.max(np.abs(vc))), 1e-300)
    np.testing.assert_allclose(vm, vc, rtol=0, atol=rtol * scale)


T2 = np.repeat(np.arange(2), 3)          # two groups of three
T3 = np.repeat(np.arange(3), 3)          # three groups of three
T3L = np.array([[0, 0], [0, 0], [1, 0], [1, 0],
                [2, 1], [2, 1], [3, 1], [3, 1]])   # three levels


@pytest.mark.parametrize("sym", [[True, True], [True, False], [False, True]])
@pytest.mark.parametrize("per", [False, True])
@pytest.mark.parametrize("tags,r,rel,K", [
    (T2, [1, 2], [0, 0], 6),
    (T3, [2, 2], [0, 0], 9),
    (T3, [2, 2], [1, 0], 9),      # inner unit
    (T3, [2, 2], [0, 1], 9),      # outer unit
    (T3, [2, 3], [0, 0], 9),
])
def test_two_levels_match_centres(tags, r, rel, K, per, sym):
    d = _density(tags, r, sym, rel, per, K, seed=1)
    X = _queries(d, seed=1)
    # Relative-periodic compares the all-image (Möbius) against the
    # minimum-image (centres) reading; at sigma/P = 0.058 they agree to
    # ~1e-8, elsewhere to the floor.
    rtol = 1e-7 if (per and any(rel)) else 1e-10
    _assert_routes_agree(d, X, rtol)


@pytest.mark.parametrize("rel", [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
@pytest.mark.parametrize("sym,per", [([True, True, True], False),
                                     ([True, False, True], True),
                                     ([False, True, False], False)])
def test_three_levels_match_centres(rel, sym, per):
    d = _density(T3L, [2, 2, 2], sym, rel, per, 8, seed=2, sigma=0.6)
    if d.dim == 0:
        pytest.skip("degenerate zero-dimensional layout")
    X = _queries(d, seed=2)
    rtol = 1e-7 if (per and any(rel)) else 1e-10
    _assert_routes_agree(d, X, rtol)


def test_single_image_wrap_is_honoured():
    d = _density(T3, [2, 2], [True, True], [0, 0], True, 9, seed=3,
                 sigma=0.4, wrap='single-image')
    X = _queries(d, seed=3)
    _assert_routes_agree(d, X, 1e-10)


def test_ragged_events_with_nan_padding():
    rng = np.random.default_rng(4)
    p = np.sort(rng.uniform(0.0, P, size=(6, 3)), axis=0)
    p[5, 0] = np.nan                    # event 0 has one value fewer
    p[4:, 2] = np.nan                   # event 2 has two fewer
    spec = {"tags": T2, "r": [1, 2], "sym": [True, True], "rel": [0, 0]}
    d = build_exp_tens([p], None, specs=[spec], sigma=[0.7], is_per=[False],
                       period=[0.0], verbose=False)
    X = _queries(d, seed=4)
    _assert_routes_agree(d, X, 1e-10)


def test_nested_tensored_with_flat_attributes():
    rng = np.random.default_rng(5)
    p0 = np.sort(rng.uniform(0.0, P, size=(6, 3)), axis=0)
    p1 = np.sort(rng.uniform(0.0, P, size=(3, 3)), axis=0)
    p2 = rng.uniform(0.0, 10.0, size=(1, 3))
    specs = [{"tags": T2, "r": [1, 2], "sym": [True, True], "rel": [0, 1]},
             {"r": 2, "sym": True, "rel": False},
             {"r": 1, "sym": True, "rel": False}]
    d = build_exp_tens([p0, p1, p2], None, specs=specs,
                       sigma=[0.5, 0.8, 1.0], is_per=[True, False, False],
                       period=[P, 0.0, 0.0], verbose=False)
    X = _queries(d, seed=5)
    _assert_routes_agree(d, X, 1e-7)


def test_auto_keeps_centres_and_mobius_is_accepted():
    d = _density(T2, [1, 2], [True, True], [0, 0], False, 6, seed=6)
    X = _queries(d, seed=6)
    va = eval_exp_tens(d, X, method="auto", verbose=False)
    vc = eval_exp_tens(d, X, method="centres", verbose=False)
    vm = eval_exp_tens(d, X, method="mobius", verbose=False)
    np.testing.assert_allclose(va, vc, rtol=1e-12)
    np.testing.assert_allclose(vm, vc, rtol=0, atol=1e-10 * np.max(np.abs(vc)))


def test_ordered_flat_attribute_still_refuses_mobius():
    rng = np.random.default_rng(7)
    p = np.sort(rng.uniform(0.0, P, size=(4, 2)), axis=0)
    d = build_exp_tens([p], None, specs=[{"r": 2, "sym": False, "rel": False}],
                       sigma=[0.5], is_per=[False], period=[0.0], verbose=False)
    with pytest.raises(ValueError, match="ordered"):
        eval_exp_tens(d, np.zeros((d.dim, 2)), method="mobius", verbose=False)


def test_per_level_evaluator_reference_values():
    """Fixed inputs; the MATLAB test asserts the same numbers."""
    p = np.array([1.0, 2.5, 4.0, 7.0, 8.2, 11.0])
    w = np.array([1.0, 0.5, 1.0, 1.0, 0.7, 1.0])
    tags = np.array([0, 0, 0, 1, 1, 1])
    x = np.array([[0.5, 3.0, 7.0, 11.0], [2.0, 2.7, 8.0, 1.0],
                  [1.5, 4.0, 9.0, 3.0], [7.0, 8.0, 6.0, 4.0]])
    v = eval_nested_attr_orbit(p, w, tags, [2, 2], [True, True], None, 0.7,
                               x, truncation_sigmas=math.inf)
    np.testing.assert_allclose(
        v, [2.00328713036504e-15, 2.01827912258021e-05,
            1.38777878078145e-17, 4.71134784883262e-17],
        rtol=1e-9, atol=1e-16)
    v = eval_nested_attr_orbit(p, w, tags, [2, 2], [True, False], 0, 0.7,
                               x[:2], truncation_sigmas=math.inf)
    np.testing.assert_allclose(
        v, [0.884583285039345, 1.76658427991777,
            8.14454249730343e-08, 5.86413326215107e-15], rtol=1e-9)
    v = eval_nested_attr_orbit(p, w, tags, [2, 2], [False, True], 1, 0.7,
                               x[:3], is_per=True, period=12.0,
                               truncation_sigmas=math.inf)
    np.testing.assert_allclose(
        v, [1.44730864634257e-05, 0.00179025660699691,
            0.00635877431113622, 0.000329925782403053], rtol=1e-9)


def test_per_level_evaluator_is_much_cheaper_than_the_centres_route():
    """The point of the route: r = (3, 3) over four groups of three has
    5184 tuple centres per event; the per-level sum touches none."""
    import time
    rng = np.random.default_rng(8)
    tags = np.repeat(np.arange(4), 3)
    p = np.sort(rng.uniform(0.0, P, size=(12, 4)), axis=0)
    spec = {"tags": tags, "r": [3, 3], "sym": [True, True], "rel": [0, 0]}
    d = build_exp_tens([p], None, specs=[spec], sigma=[0.7], is_per=[False],
                       period=[0.0], verbose=False)
    X = rng.uniform(0.0, P, size=(d.dim, 50))
    t0 = time.perf_counter()
    vc = eval_exp_tens(d, X, method="centres", verbose=False)
    tc = time.perf_counter() - t0
    t0 = time.perf_counter()
    vm = eval_exp_tens(d, X, method="mobius", verbose=False)
    tm = time.perf_counter() - t0
    np.testing.assert_allclose(vm, vc, rtol=0, atol=1e-7 * np.max(np.abs(vc)))
    assert tm < tc
