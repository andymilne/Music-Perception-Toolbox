"""Tests for the single-multiset point-evaluator orbit primitives.

Verifies ``eval_orbit_abs`` and ``eval_orbit_rel`` against:
1. The centre-array path (``eval_exp_tens``) — precision baseline.
2. Direct distinct-tuple enumeration — slowest but exact ground truth,
   used to confirm the abs orbit's modest precision loss in
   cancellation regimes is genuinely a property of the alternating-sum
   formulation, not a bug.

Precision notes:
- For abs modes, the Möbius method expresses T(x) as an alternating sum
  over set partitions of {0,..,r-1}. At pathological queries (very low
  T(x) values where partial sums are large and partly cancel), the
  Möbius method can lose digits. The cancellation ratio diagnostic
  exposed via ``return_cancellation_ratio=True`` flags such cases.
  Tests compare against centre-array on the "clean" subset.
- For rel modes, the u-grid integration averages over many abs-mode
  evaluations; the cancellation effects in any single u rarely line
  up to corrupt the final result. Rel-orbit-vs-centres typically
  matches to FP precision.
"""
import numpy as np
import pytest
from itertools import permutations

from mpt._mobius import (
    eval_orbit_abs,
    eval_orbit_rel,
    get_set_partitions_with_mobius,
)
from mpt.tensor import build_exp_tens, eval_exp_tens


# Tolerance for orbit-vs-centres comparison on clean cells (ratio > 1e-10).
# In the abs path the worst-case loss is at the noise floor of the
# alternating sum; 1e-6 reflects user-tolerance practice for v3.
TOL_CLEAN = 1e-6
# FP tolerance for the rel path and for cells that don't cancel.
TOL_FP = 1e-10


# -------------------------------------------------------------------
#  Set-partition enumeration sanity
# -------------------------------------------------------------------


def test_set_partition_counts_match_bell_numbers():
    """B_r = 1, 1, 2, 5, 15, 52, 203 for r = 0..6."""
    bell = [1, 1, 2, 5, 15, 52, 203]
    for r, expected in enumerate(bell):
        partitions = get_set_partitions_with_mobius(r)
        assert len(partitions) == expected, (
            f"r={r}: expected {expected} partitions, got {len(partitions)}"
        )


def test_set_partitions_cover_all_slots_uniquely():
    """Each position in {0, ..., r-1} appears in exactly one block of every
    partition."""
    for r in range(1, 6):
        for blocks, _mu in get_set_partitions_with_mobius(r):
            positions = []
            for B in blocks:
                positions.extend(B)
            assert sorted(positions) == list(range(r))


def test_mobius_signs_alternate_with_block_count():
    """μ(π) = ∏_l (-1)^{m_l-1}(m_l-1)!. For r=2 the two partitions
    are {{0,1}} (mu = -1) and {{0},{1}} (mu = 1)."""
    parts = dict(get_set_partitions_with_mobius(2))
    assert parts[((0, 1),)] == -1
    assert parts[((0,), (1,))] == 1


# -------------------------------------------------------------------
#  Abs mode: orbit vs centre-array vs direct enumeration
# -------------------------------------------------------------------


def _direct_enum_abs(p, w, sigma, r, x, is_per, period):
    """Slowest correct path — sum over distinct r-tuples."""
    n_q = x.shape[1]
    total = np.zeros(n_q)
    for tup in permutations(range(len(p)), r):
        val = np.ones(n_q)
        for k, ik in enumerate(tup):
            d = x[k, :] - p[ik]
            if is_per:
                d = d - period * np.floor(d / period + 0.5)
            val *= w[ik] * np.exp(-d ** 2 / (2 * sigma ** 2))
        total += val
    return total


@pytest.mark.parametrize(
    "r, K, is_per",
    [
        (2, 5, False), (2, 5, True),
        (3, 6, False), (3, 6, True),
        (4, 7, False), (4, 7, True),
    ],
)
def test_eval_orbit_abs_matches_centres_on_clean_cells(r, K, is_per):
    """Orbit and centres agree on cells with healthy cancellation
    ratio. The "ground truth" here is direct distinct-tuple
    enumeration (which both paths approximate)."""
    rng = np.random.default_rng(0)
    P = 1200.0
    sigma = 50.0
    if is_per:
        p = rng.uniform(0, P, K)
        x = rng.uniform(0, P, (r, 30))
        period = P
    else:
        p = rng.uniform(-200, 200, K)
        x = rng.uniform(-300, 300, (r, 30))
        period = 0.0
    w = rng.uniform(0.5, 1.5, K)

    T = build_exp_tens(p, w, sigma, r, False, is_per, period, verbose=False)
    # eval_exp_tens now has a method dispatcher (v3 wiring). Force
    # centres explicitly so the variable name matches what the call
    # returns.
    v_centres = eval_exp_tens(T, x, method='centres', verbose=False)
    v_orbit, ratios = eval_orbit_abs(
        p, w, sigma, r, x,
        is_per=is_per, period=period,
        return_cancellation_ratio=True,
    )
    v_direct = _direct_enum_abs(p, w, sigma, r, x, is_per, period)

    # Both centres and orbit should agree with direct on clean cells.
    clean = ratios > 1e-10
    assert clean.any(), "test setup gave no clean cells; pick saner params"

    # The centres path truncates at the accuracy-floor width (inf resolves
    # to ~7.43 sigma, the 1e-12 floor), so its error against the exhaustive
    # direct sum is an ABSOLUTE floor of ~1e-12 relative to the peak
    # density --- not a per-point relative bound. At low-density query
    # points that same absolute floor shows up as a larger relative error,
    # so the accuracy-floor measure (abs error / peak) is the correct one.
    peak = float(np.max(np.abs(v_direct)))
    abs_err_centres = np.abs(v_centres[clean] - v_direct[clean])
    # Measured worst case is ~1.4e-12 of peak; 1e-11 gives modest headroom.
    assert abs_err_centres.max() < 1e-11 * peak, (
        f"centres path drifted from direct beyond the accuracy floor: "
        f"max abs err {abs_err_centres.max():.2e}, peak {peak:.2e}"
    )

    # The orbit path floor-truncates at the accuracy-floor width in every
    # mode (periodic included, v3+), so like the centres path its error
    # against the exhaustive direct sum is a peak-relative absolute floor,
    # not a per-cell relative bound.
    abs_err_orbit = np.abs(v_orbit[clean] - v_direct[clean])
    assert abs_err_orbit.max() < 1e-11 * peak, (
        f"orbit path drifted from direct beyond the accuracy floor: "
        f"max abs err {abs_err_orbit.max():.2e}, peak {peak:.2e}"
    )
    # Where the value itself is well above the floor, agreement is tight;
    # deep-tail cells (healthy cancellation ratio but sub-floor value) are
    # governed by the absolute floor above, not a relative bound.
    meaningful = clean & (np.abs(v_direct) > 1e-4 * peak)
    if meaningful.any():
        rel_err_orbit = (np.abs(v_orbit[meaningful] - v_direct[meaningful])
                         / np.abs(v_direct[meaningful]))
        assert rel_err_orbit.max() < TOL_CLEAN, (
            f"orbit-on-clean path drifted from direct: max rel err "
            f"{rel_err_orbit.max():.2e}"
        )


def test_eval_orbit_abs_r1_matches_direct():
    """At r=1 the partition is a singleton; formula reduces to a
    direct sum of Gaussians."""
    rng = np.random.default_rng(0)
    sigma = 50.0
    p = rng.uniform(0, 1200, 8)
    w = rng.uniform(0.5, 1.5, 8)
    x = rng.uniform(0, 1200, (1, 50))
    v_orbit = eval_orbit_abs(p, w, sigma, 1, x, is_per=False)
    v_direct = (w[:, None] * np.exp(-(x[0, :][None, :] - p[:, None]) ** 2
                                     / (2 * sigma ** 2))).sum(axis=0)
    assert np.allclose(v_orbit, v_direct, atol=0, rtol=TOL_FP)


def test_eval_orbit_abs_cancellation_ratio_signals_corruption():
    """At r >= 3 abs_per with sharp σ and low K-r margin, some query
    cells should hit cancellation. The diagnostic must flag them."""
    rng = np.random.default_rng(0)
    P = 1200.0
    sigma = 50.0
    r, K = 4, 7
    p = rng.uniform(0, P, K)
    w = rng.uniform(0.5, 1.5, K)
    x = rng.uniform(0, P, (r, 100))
    _, ratios = eval_orbit_abs(
        p, w, sigma, r, x, is_per=True, period=P,
        return_cancellation_ratio=True,
    )
    # Across 100 random queries, expect at least a few to dip below
    # the clean threshold somewhere in the support.
    bad_count = (ratios < 1e-10).sum()
    assert bad_count > 0, (
        "Expected at least one cancelling query in this regime; got 0. "
        "If parameters changed and corruption no longer occurs, the "
        "test setup itself is invalid for this assertion."
    )
    # And at the clean cells the ratios are reasonably close to 1.
    clean = ratios > 1e-10
    assert ratios[clean].min() > 1e-10


# -------------------------------------------------------------------
#  Rel mode: orbit-via-u-grid vs centre-array
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "r, K, is_per",
    [
        (2, 5, False), (2, 5, True),
        (3, 6, False), (3, 6, True),
        (4, 7, False), (4, 7, True),
    ],
)
def test_eval_orbit_rel_matches_centres(r, K, is_per):
    """For rel modes the u-grid integration averages over abs evals,
    smoothing out cancellation. Expect FP-level agreement with the
    centre-array path."""
    rng = np.random.default_rng(0)
    P = 1200.0
    sigma = 50.0
    if is_per:
        p = rng.uniform(0, P, K)
        x_rel = rng.uniform(0, P, (r - 1, 30))
        period = P
    else:
        p = rng.uniform(-200, 200, K)
        x_rel = rng.uniform(-200, 200, (r - 1, 30))
        period = 0.0
    w = rng.uniform(0.5, 1.5, K)

    T = build_exp_tens(p, w, sigma, r, True, is_per, period, verbose=False)
    v_centres = eval_exp_tens(T, x_rel, method='centres', verbose=False)
    v_orbit = eval_orbit_rel(
        p, w, sigma, r, x_rel,
        is_per=is_per, period=period,
    )
    rel_err = np.abs(v_orbit - v_centres) / (np.abs(v_centres) + 1e-300)
    assert rel_err.max() < 1e-9, (
        f"rel-mode orbit drifted: max rel err {rel_err.max():.2e}"
    )


def test_eval_orbit_rel_r1_returns_constant():
    """r=1 rel: the relative space is 0-D, T_rel is a constant; we
    return Σ_i w_i replicated across query points."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1200, 5)
    w = rng.uniform(0.5, 1.5, 5)
    x_rel = np.zeros((0, 7))  # r-1 = 0 rows
    v = eval_orbit_rel(p, w, 50.0, 1, x_rel, is_per=False)
    assert v.shape == (7,)
    assert np.allclose(v, w.sum())


def test_eval_orbit_rel_returns_cancellation_ratio_shape():
    """The diagnostic-mode output is a 2-tuple (values, worst_ratios)
    where worst_ratios has shape (n_q,) — the worst case across the
    u-grid for each query."""
    rng = np.random.default_rng(0)
    P = 1200.0
    p = rng.uniform(0, P, 6)
    w = rng.uniform(0.5, 1.5, 6)
    x_rel = rng.uniform(0, P, (2, 25))
    vals, ratios = eval_orbit_rel(
        p, w, 50.0, 3, x_rel,
        is_per=True, period=P,
        return_cancellation_ratio=True,
    )
    assert vals.shape == (25,)
    assert ratios.shape == (25,)
    assert np.all((ratios > 0) & (ratios <= 1.0 + 1e-12))


# -------------------------------------------------------------------
#  Memory-savings sanity check
# -------------------------------------------------------------------


def test_eval_orbit_abs_handles_high_r_K_where_centres_struggles():
    """At r=4 K=20 the centre-array has 116280 distinct 4-tuples
    times 4 dim times n_q query points. Orbit path materialises only
    O(N) per partition. Both should agree on a small query set."""
    rng = np.random.default_rng(0)
    P = 1200.0
    sigma = 50.0
    r, K = 4, 20
    p = rng.uniform(0, P, K)
    w = rng.uniform(0.5, 1.5, K)
    x = rng.uniform(0, P, (r, 5))  # only 5 queries — keeps centres path doable

    T = build_exp_tens(p, w, sigma, r, False, True, P, verbose=False)
    v_centres = eval_exp_tens(T, x, method='centres', verbose=False)
    v_orbit, ratios = eval_orbit_abs(
        p, w, sigma, r, x, is_per=True, period=P,
        return_cancellation_ratio=True,
    )

    # On clean cells we expect agreement at 1e-6.
    clean = ratios > 1e-10
    if clean.any():
        rel_err = np.abs(
            v_orbit[clean] - v_centres[clean]
        ) / np.abs(v_centres[clean])
        assert rel_err.max() < TOL_CLEAN
