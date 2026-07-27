"""Cost-model calibration for multi-attribute ``eval_exp_tens``.

The MA eval selector :func:`mpt._tensor.dispatch._select_ma_eval` is a
pure cost model with no probe: because the MAET density factorises
across attributes (Milne 2026, Eq. maet-density), the joint-centres
tuple count and the factored-Möbius cost are both closed-form from the
shape, and the crossover is sharp. A small set of per-language
calibration constants (the ``_MA_COST_*`` group in
:mod:`mpt._tensor.dispatch`) absorbs the implementation constant
factors --- materialisation overhead, Möbius setup cost, u-grid node
costs --- which differ between NumPy and MATLAB. These constants are
settled *once per language* by measurement, not rediscovered at
runtime by a probe; this test is what guarantees the calibration stays
current as the code evolves, and is the template for the MATLAB
mirror.

The contract is deliberately *asymmetric*. The two possible wrong picks
are not equally costly:

- Picking **Möbius** when centres would have been marginally faster
  costs at most a fraction of a millisecond (the factored path is flat
  and fast even where it is not optimal).
- Picking **centres** when Möbius is much faster is the harmful
  mistake: the joint-centres path materialises the joint tuple set,
  whose size grows as a *product* across attributes, so a bad centres
  pick can be orders of magnitude slower or exhaust memory outright.

So the test does not demand the model always pick the strictly faster
route (a symmetric "selection accuracy" contract that would fail on
sub-millisecond ties for no practical gain). It demands the model is
never *badly* wrong: whenever it picks centres, centres must be feasible
and within a modest factor of Möbius. Picking Möbius is always allowed
--- it is the failure-safe route.
"""

import time

import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, eval_exp_tens
from mpt._tensor.dispatch import _select_ma_eval


# The largest factor by which a *chosen* centres route may be slower
# than Möbius before it counts as a harmful miscalibration. Generous
# because the harm being guarded against is orders-of-magnitude / OOM,
# not tens of percent; a tight bound here would make the test a flaky
# wall-time assertion rather than a calibration guard.
_MAX_TOLERATED_CENTRES_SLOWDOWN = 3.0


def _build(sig, r_vec, is_rel, is_per, period, K, N, seed=0):
    g = np.random.default_rng(seed)
    A = len(sig)
    pas = [g.uniform(0, 100, (K, N)) for _ in range(A)]
    return build_exp_tens(pas, None, sig, r_vec, is_rel, is_per, period,
                          verbose=False)


def _build_span(sig, r_vec, is_rel, is_per, period, K, span, seed=0):
    """Build a single-multiset-per-attribute density spanning ``span``.

    The relative-mode Möbius cost scales with the u-grid node count, which
    grows with the source spread; only realistic (multi-octave) spans reach
    the regime where relative Möbius is expensive. The [0, 100] grid above
    does not, so relative-crossover cells build over ``span`` instead.
    """
    g = np.random.default_rng(seed)
    A = len(sig)
    pas = [np.sort(g.uniform(0, span, (K, 1)), axis=0) for _ in range(A)]
    return build_exp_tens(pas, None, sig, r_vec, is_rel, is_per, period,
                          verbose=False)


def _bench(fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


# Shape grid spanning the crossover and both sides of it: single- and
# multi-attribute, absolute and relative, small-to-moderate K. Large-K
# cells are omitted from the *timed* grid because the centres route is
# infeasible there (that is exactly why the model must pick Möbius, but
# we cannot time an OOM); feasibility at large K is covered separately
# by the memory-guard test below.
_GRID = [
    # (label, sigma, r_vec, is_rel, is_per, period, K, N)
    ("A1 abs r2 K6",  [30.],      [2],      [False],        [False],        [0.],         6,  1),
    ("A1 abs r2 K10", [30.],      [2],      [False],        [False],        [0.],         10, 1),
    ("A1 abs r2 K20", [30.],      [2],      [False],        [False],        [0.],         20, 1),
    ("A1 abs r3 K6",  [30.],      [3],      [False],        [False],        [0.],         6,  1),
    ("A1 abs r3 K10", [30.],      [3],      [False],        [False],        [0.],         10, 1),
    ("A1 rel r2 K8",  [30.],      [2],      [True],         [False],        [0.],         8,  1),
    ("A1 rel r3 K8",  [30.],      [3],      [True],         [False],        [0.],         8,  1),
    ("A2 abs r2 K5",  [30., 25.], [2, 2],   [False, False], [False, False], [0., 0.],     5,  1),
    ("A2 abs r2 K8",  [30., 25.], [2, 2],   [False, False], [False, False], [0., 0.],     8,  1),
    ("A2 abs r2 K12", [30., 25.], [2, 2],   [False, False], [False, False], [0., 0.],     12, 1),
    ("A2 mix K6",     [30., 25.], [2, 3],   [True, False],  [False, False], [0., 0.],     6,  1),
    ("A2 rel r2 K8",  [30., 25.], [2, 2],   [True, True],   [False, False], [0., 0.],     8,  1),
    ("A2 abs r2 N2",  [30., 25.], [2, 2],   [False, False], [False, False], [0., 0.],     6,  2),
    ("A3 mix K6",     [30., 25., 40.], [2, 3, 2], [True, False, True], [False, False, True], [0., 0., 1200.], 6, 1),
]


def test_ma_cost_model_convention_prefers_all_image_mobius():
    """Above the σ/P threshold in relative-periodic mode, the model
    prefers the (C) all-image Möbius form (cheaper and memory-safe) as
    the default. In v3 the choice is the user's ``wrap=`` field on the
    density (default ``'full-image'``); the previous "warning that it
    deviates" was retired because there is no deviation — full-image is
    the toolbox measure now, and single-image is an explicit opt-in.
    The (A) single-image form remains available via method='centres'."""
    import warnings
    dens = _build([40., 40.], [3, 3], [True, True], [True, True],
                  [1200., 1200.], K=6, N=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chosen, reason = _select_ma_eval(dens, 200, method="auto")
    assert chosen == "mobius", (
        f"chose {chosen} ({reason}); above σ/P the all-image Möbius form "
        f"is the preferred default."
    )
    # Single-image is available on demand.
    assert _select_ma_eval(dens, 200, method="centres")[0] == "centres"


def test_ma_cost_model_convention_holds_at_large_shape_no_oom():
    """The all-image preference is memory-safe at any shape: even where
    the single-image centres route would be infeasible to materialise,
    auto-dispatch takes Möbius (bounded cost), never risking OOM."""
    import warnings
    # K=40, r=3, relative-periodic above threshold: single-image joint
    # set would be ~hundreds of GB, but Möbius is n_j-free.
    dens = _build([40., 40.], [3, 3], [True, True], [True, True],
                  [1200., 1200.], K=40, N=1)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        chosen, reason = _select_ma_eval(dens, 200, method="auto")
    assert chosen == "mobius", (
        f"chose {chosen} ({reason}); the all-image Möbius form must be "
        f"preferred at large shapes to stay memory-safe."
    )


def test_ma_cost_model_raises_when_single_image_forced_and_infeasible():
    """When Möbius is *unavailable* (r beyond the feasibility bound, so
    no all-image substitute exists) and the single-image centres route
    would be infeasible to materialise, auto-dispatch raises a clear
    error rather than crashing with an out-of-memory kill."""
    from mpt._tensor.dispatch import SingleImageInfeasibleError
    # r = 11 > orbit feasibility bound forces centres; K = 20 makes the
    # centre set ~10^12 tuples, far past the memory budget.
    dens = _build([6.], [11], [False], [False], [0.], K=20, N=1)
    with pytest.raises(SingleImageInfeasibleError):
        _select_ma_eval(dens, 200, method="auto")


def test_ma_cost_model_infeasible_single_image_honours_user_override():
    """The infeasibility error is a *dispatch* guard: an explicit
    method='centres' is the user's own choice and is honoured by the
    selector (the attempt may still fail downstream, but the selector
    does not second-guess an explicit request)."""
    dens = _build([6.], [11], [False], [False], [0.], K=20, N=1)
    chosen, reason = _select_ma_eval(dens, 200, method="centres")
    assert chosen == "centres"
    assert reason == "user override"


@pytest.mark.parametrize("case", _GRID, ids=[c[0] for c in _GRID])
def test_ma_cost_model_never_badly_wrong(case):
    """Whenever the cost model picks centres, centres is within a modest
    factor of Möbius. Picking Möbius is always acceptable."""
    label, sig, r_vec, is_rel, is_per, period, K, N = case
    dens = _build(sig, r_vec, is_rel, is_per, period, K, N)
    n_q = 200
    g = np.random.default_rng(1)
    x = g.uniform(0, 100, (dens.dim, n_q))

    chosen, reason = _select_ma_eval(dens, n_q, method="auto")
    if chosen == "mobius":
        # Failure-safe route; never a harmful pick.
        return

    # Chosen centres. Distinguish *why*: a hard-rule pick (convention
    # exactness in relative-periodic mode above the sigma/P threshold,
    # the K-r precision floor, or the orbit feasibility bound) is a
    # correctness choice, not a cost choice --- centres is deliberately
    # preferred there even when slower, because Möbius would be
    # inexact or infeasible on that attribute. Only a *cost-model* pick
    # ("cost model ..." reason) is subject to the speed contract; that
    # is the pick that must never be badly wrong.
    if not reason.startswith("cost model"):
        pytest.skip(
            f"{label}: centres chosen for correctness ({reason}), not "
            f"cost; the speed contract does not apply."
        )

    # Cost-model centres pick: it must not be badly slower than Möbius.
    t_centres = _bench(lambda: eval_exp_tens(
        dens, x, method="centres", verbose=False))
    t_mobius = _bench(lambda: eval_exp_tens(
        dens, x, method="mobius", verbose=False))
    slowdown = t_centres / max(t_mobius, 1e-9)
    assert slowdown <= _MAX_TOLERATED_CENTRES_SLOWDOWN, (
        f"{label}: cost model chose centres ({reason}) but centres is "
        f"{slowdown:.1f}x slower than Möbius "
        f"(centres {t_centres*1e3:.2f} ms, Möbius {t_mobius*1e3:.2f} ms). "
        f"The _MA_COST_* calibration constants need revisiting."
    )


@pytest.mark.parametrize("A,r,K,seed,expect", [
    (3, 2, 5, 0, "centres"),
    (3, 2, 6, 1, "centres"),
    (3, 2, 8, 0, "centres"),
    (3, 3, 12, 1, "centres"),
])
def test_ma_cost_model_rel_ma_crossover_placed_correctly(A, r, K, seed, expect):
    """The relative multi-attribute eval pick must sit where measured
    times sit.

    This guards the direction the asymmetric centres-only contract above
    cannot: an under-priced Möbius estimate that picks Möbius when centres
    is several times faster. With the factored centres route (per-attribute
    culled kernels; cost is the sum, not the product, of per-attribute
    tuple counts), centres wins throughout this non-periodic relative span
    --- measured ~1-2 ms against ~270-990 ms for Möbius across these cells
    --- so every cell pins centres. The Möbius side of the contract is
    carried by the periodic and infeasible-centres diversions tested
    elsewhere in this module and in the routing suite.
    """
    sig = [15.0] * A
    dens = _build_span(sig, [r] * A, [True] * A, [False] * A, [0.] * A,
                       K, span=3600.0, seed=seed)
    chosen, reason = _select_ma_eval(dens, 200, method="auto")
    assert chosen == expect, (
        f"A{A} rel r{r}K{K}: expected {expect} but cost model chose "
        f"{chosen} ({reason}); the relative-Möbius node cost calibration "
        f"has drifted."
    )


def test_ma_cost_model_diverts_infeasible_centres_to_mobius():
    """At a shape where the joint tuple set is far too large to
    materialise, the model must pick Möbius --- the centres route would
    exhaust memory. This is the catastrophic pick the asymmetric
    contract exists to forbid."""
    # Two attributes, r=3, K=40: joint tuples ~ (3! C(40,3))^2 ~ 5.6e12.
    dens = _build([30., 25.], [3, 3], [False, False], [False, False],
                  [0., 0.], K=40, N=1)
    chosen, reason = _select_ma_eval(dens, 200, method="auto")
    assert chosen == "mobius", (
        f"cost model chose {chosen} ({reason}) at a shape whose joint "
        f"tuple set is infeasible to materialise; it must pick Möbius."
    )


def test_ma_cost_model_respects_user_override():
    """Explicit method overrides bypass the cost model in both
    directions."""
    dens = _build([30., 25.], [2, 2], [False, False], [False, False],
                  [0., 0.], K=8, N=1)
    assert _select_ma_eval(dens, 200, method="centres")[0] == "centres"
    assert _select_ma_eval(dens, 200, method="mobius")[0] == "mobius"


def test_ma_cost_model_small_corner_may_choose_centres():
    """The cost model is not degenerate: at the smallest shapes it is
    allowed to (and the constant is tuned so it does not have to) pick
    centres. This test documents that centres remains reachable rather
    than dead code --- it asserts only that the selector returns a valid
    route, not which, so it does not pin the crossover (that is the
    calibration test's job)."""
    dens = _build([30.], [2], [False], [False], [0.], K=4, N=1)
    chosen, _ = _select_ma_eval(dens, 10, method="auto")
    assert chosen in ("centres", "mobius")


# ---------------------------------------------------------------------
# Inner-product side: the same single-image infeasibility guard applies
# to the forced-Bulger paths (Möbius unavailable by precision floor or
# feasibility bound, and the Bulger tuple-pair kernel too large).
# ---------------------------------------------------------------------

def _single_multiset_ip_select(K, r, is_rel=False, is_per=False, period=0.0, method="auto"):
    from mpt._tensor.dispatch import _select_ma_inner_product_method
    chosen = _select_ma_inner_product_method(
        r_vec=np.array([r]), k_vec=np.array([K]), A=1,
        N_x=1, N_y=1,
        any_per=is_per,
        any_rel_nonper=(is_rel and not is_per),
        any_rel_per=(is_rel and is_per),
        sigma_over_P_max=0.0, user_method=method,
        rel_vec=np.array([is_rel]), nu_vec=None,
    )
    return (chosen,)


def test_ip_raises_when_bulger_forced_by_feasibility_and_infeasible():
    """r beyond the shipped orbit order forces Bulger (Möbius
    unavailable); at large K the tuple-pair kernel is infeasible, so
    auto-dispatch raises rather than OOM."""
    from mpt._tensor.dispatch import SingleImageInfeasibleError
    with pytest.raises(SingleImageInfeasibleError):
        _single_multiset_ip_select(K=15, r=9)


def test_ip_raises_when_bulger_forced_by_precision_and_infeasible():
    """The K-r precision floor forces Bulger at high tuple order; the
    tuple-pair kernel is then infeasible, so auto-dispatch raises."""
    from mpt._tensor.dispatch import SingleImageInfeasibleError
    with pytest.raises(SingleImageInfeasibleError):
        _single_multiset_ip_select(K=9, r=8)  # K - r = 1 < 2


def test_ip_infeasible_bulger_honours_user_override():
    """An explicit method='bulger' is the user's own choice; the
    selector honours it without raising (the attempt may fail
    downstream)."""
    chosen = _single_multiset_ip_select(K=15, r=9, method="bulger")[0]
    assert chosen == "bulger"


def test_ip_forced_bulger_feasible_does_not_raise():
    """Where Bulger is forced but feasible (low tuple order or small
    collection), no error is raised."""
    # r = 1 forces Bulger and is always feasible (monad inner product).
    assert _single_multiset_ip_select(K=100, r=1)[0] == "bulger"
