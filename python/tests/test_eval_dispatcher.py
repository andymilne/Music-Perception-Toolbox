"""Tests for the eval_exp_tens dispatcher at the single-multiset corner.

The dispatcher routes between the joint-centres body and the factored
orbit-Möbius point evaluator according to ``method`` and the cost model
in :func:`mpt._tensor.dispatch._select_ma_eval`, of which the single
multiset is the A = 1 case. These tests verify that the explicit methods
produce mutually consistent output and that the precision and convention
guards behave as documented. (Whole-call auto-routing is covered in
test_dispatcher_probe.py.)
"""
import numpy as np

from mpt._defaults import accuracy_floor_context
import pytest

from mpt.tensor import (
    build_exp_tens,
    eval_exp_tens,
)


P = 1200.0
ATOL = 1e-12
RTOL = 1e-10




# -------------------------------------------------------------------
#  Numerical agreement of explicit methods
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "r,K,is_rel,is_per",
    [
        (2, 9, False, True),
        (2, 9, False, False),
        (3, 8, False, True),
        (3, 8, False, False),
        (4, 10, False, True),
        # Rel mode: still tests that explicit method='mobius' produces
        # a value consistent with centres at moderate σ/P.
        (3, 8, True, True),
        (3, 8, True, False),
    ],
)
def test_eval_methods_agree(r, K, is_rel, is_per):
    """At well-conditioned cells, centres / orbit / auto agree to
    floating-point precision (with abs floor for near-zero density)."""
    rng = np.random.default_rng(0)
    sigma = 33.0 if is_per else 30.0
    period = P if is_per else 0.0
    if is_per:
        p = rng.uniform(0, P, K)
    else:
        p = rng.uniform(-300, 300, K)
    w = rng.uniform(0.5, 1.5, K)
    T = build_exp_tens(p, w, sigma, r, is_rel, is_per, period, verbose=False)

    dim = r - 1 if is_rel else r
    if is_per:
        x = rng.uniform(0, P, (dim, 30))
    else:
        x = rng.uniform(-300, 300, (dim, 30))

    # Widen the accuracy floor so the paths are compared exhaustively
    # rather than each dropping marginally different far tails, matching
    # the MATLAB twin (test_ma_eval_dispatch.m), which brackets the same
    # comparison with internal.accuracyFloor('setEps', 1e-300).
    with accuracy_floor_context(1e-300):
        v_auto = eval_exp_tens(T, x, method='auto', verbose=False)
        v_centres = eval_exp_tens(T, x, method='centres', verbose=False)
        v_orbit = eval_exp_tens(T, x, method='mobius', verbose=False)

    # Max-relative agreement, normalised by the largest density value, as
    # the MATLAB twin does: a per-element rtol is dominated by near-zero
    # tail entries, where the two paths legitimately differ by their own
    # rounding while the values themselves are negligible.
    denom = max(float(np.max(np.abs(v_centres))), 1e-12)
    assert float(np.max(np.abs(v_auto - v_centres))) / denom < 1e-9
    assert float(np.max(np.abs(v_centres - v_orbit))) / denom < 1e-6


# -------------------------------------------------------------------
#  K-r precision guard
# -------------------------------------------------------------------


def test_eval_orbit_at_K_minus_r_below_guard_via_explicit():
    """User-forced method='mobius' at K-r < 2 still runs (no guard
    on explicit override). Auto would have picked centres."""
    rng = np.random.default_rng(0)
    K, r = 4, 3  # K - r = 1, below guard
    p = rng.uniform(0, P, K)
    w = rng.uniform(0.5, 1.5, K)
    T = build_exp_tens(p, w, 33.0, r, False, True, P, verbose=False)
    x = rng.uniform(0, P, (r, 5))
    # Auto routes to centres.
    v_auto = eval_exp_tens(T, x, method='auto', verbose=False)
    v_centres = eval_exp_tens(T, x, method='centres', verbose=False)
    assert np.allclose(v_auto, v_centres, atol=ATOL, rtol=RTOL)
    # Explicit orbit runs without error (precision may be reduced).
    v_orbit = eval_exp_tens(T, x, method='mobius', verbose=False)
    assert np.all(np.isfinite(v_orbit))


# -------------------------------------------------------------------
#  Memory-saving regime
# -------------------------------------------------------------------


def test_eval_orbit_memory_efficient_at_high_r():
    """At r=4 K=20 the centres tensor has ~116k distinct 4-tuples
    × dim × n_q. Orbit path materialises only O(K) per partition.
    Both must agree to FP precision on a small query batch."""
    rng = np.random.default_rng(0)
    r, K = 4, 20
    p = rng.uniform(0, P, K)
    w = rng.uniform(0.5, 1.5, K)
    T = build_exp_tens(p, w, 50.0, r, False, True, P, verbose=False)
    x = rng.uniform(0, P, (r, 5))
    v_centres = eval_exp_tens(T, x, method='centres', verbose=False)
    v_orbit = eval_exp_tens(T, x, method='mobius', verbose=False)
    # At low cancellation, orbit should match centres at FP.
    assert np.allclose(v_centres, v_orbit, atol=ATOL, rtol=1e-9)


# -------------------------------------------------------------------
#  Near-tie safety factor is gated on the centres memory risk
# -------------------------------------------------------------------


def _small_dens(r, K, is_rel, is_per, sigma_over_P=0.0083):
    rng = np.random.default_rng(0)
    p = np.sort(rng.uniform(0.0, P, K))
    w = 0.2 + 0.8 * rng.random(K)
    return build_exp_tens(p, w, sigma_over_P * P, r, is_rel, is_per,
                          P if is_per else 0.0, verbose=False)


@pytest.mark.parametrize("r,K,is_rel,is_per", [
    (2, 12, False, True),    # abs-periodic: measured 1.6x in centres' favour
    (3, 12, False, False),   # abs-non-periodic: measured 1.4x
])
def test_small_shape_near_tie_follows_the_cheaper_estimate(
        r, K, is_rel, is_per):
    """Below the centres working-set soft budget the near-tie safety
    factor does not apply, so 'auto' follows the cost model's own
    ranking rather than being pushed to Möbius by the multiplier.

    Both cells sit inside the multiplier's old reach: the model ranks
    centres cheaper, and ``_MA_MOBIUS_SAFETY`` used to flip them.
    """
    from mpt._tensor.dispatch import (
        _ma_eval_costs_ms, _select_ma_eval,
        _estimate_ma_joint_working_set_bytes,
        _CENTRES_WORKING_SET_SOFT_BUDGET, _MA_MOBIUS_SAFETY,
    )
    dens = _small_dens(r, K, is_rel, is_per)
    n_q = 24
    ws = _estimate_ma_joint_working_set_bytes([r], [K], [is_rel])
    assert ws <= _CENTRES_WORKING_SET_SOFT_BUDGET
    centres_ms, mobius_ms = _ma_eval_costs_ms(dens, n_q)
    # The cell is a near tie the old unconditional factor would flip.
    assert centres_ms < mobius_ms < centres_ms * _MA_MOBIUS_SAFETY
    chosen, _ = _select_ma_eval(dens, n_q, method="auto")
    assert chosen == "centres"


def test_large_working_set_keeps_the_mobius_safety_margin():
    """Above the soft budget the safety factor still applies: the
    centres path can exhaust memory there, so a near tie must break
    toward the bounded-cost Möbius route."""
    from mpt._tensor.dispatch import (
        _select_ma_eval, _estimate_ma_joint_working_set_bytes,
        _CENTRES_WORKING_SET_SOFT_BUDGET, _MA_MOBIUS_SAFETY_SMALL,
        _MA_MOBIUS_SAFETY,
    )
    assert _MA_MOBIUS_SAFETY_SMALL < _MA_MOBIUS_SAFETY
    # r = 4, K = 200: 24 * C(200, 4) tuples, working set ~2.5 GB, far
    # above the soft budget; Möbius must be selected.
    ws = _estimate_ma_joint_working_set_bytes([4], [200], [False])
    assert ws > _CENTRES_WORKING_SET_SOFT_BUDGET
    dens = _small_dens(4, 200, False, True)
    chosen, _ = _select_ma_eval(dens, 24, method="auto")
    assert chosen == "mobius"


def test_forced_methods_still_override_the_cost_model():
    """The gate changes only the 'auto' comparison."""
    from mpt._tensor.dispatch import _select_ma_eval
    dens = _small_dens(2, 12, False, True)
    assert _select_ma_eval(dens, 24, method="centres")[0] == "centres"
    assert _select_ma_eval(dens, 24, method="mobius")[0] == "mobius"
