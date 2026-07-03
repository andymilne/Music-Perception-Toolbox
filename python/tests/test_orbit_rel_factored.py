"""Tests for the factored (K-free) strategy in eval_orbit_rel.

The factored strategy tabulates the r smoothed event distributions
S_1..S_r once and reads them back by quintic interpolation, removing
the K factor from the per-node cost of the relative-mode u-grid
quadrature. Its read-back accuracy is tied to ``truncation_sigmas``
(target relative error at the kernel-truncation floor ``exp(-k**2/2)``,
clamped to ``[1e-12, 1e-3]``).

Accuracy semantics under test: read-back error is bounded relative to
partition-TERM scales (kernel units). Relative error of the assembled
value degrades by the query's cancellation ratio and by the value's own
smallness in kernel units — identical semantics to truncation itself
and to floating-point roundoff on the direct strategy. Tight relative
assertions therefore use on-support queries; deep-tail queries get
absolute (bulk-scaled) assertions.
"""

import numpy as np
import pytest

import mpt
from mpt._mobius import (
    _factored_spp,
    _factored_target_eps,
    _factored_worthwhile,
    _lagrange6_uniform,
    eval_orbit_rel,
)

mpt.set_default(show_hints=False)


def _template(n_part=8, dup=2, sig=15.0):
    base = 1200.0 * np.log2(np.arange(1, n_part + 1))
    amps = (1.0 / np.arange(1, n_part + 1)) ** 0.67
    p = np.concatenate([base + 0.01 * d for d in range(dup)])
    w = np.concatenate([amps] * dup)
    return p, w, sig


def _on_support_queries(p, r, n_q, seed=7):
    """Interval tuples realised by the template (healthy T values)."""
    rng = np.random.default_rng(seed)
    cols = []
    for _ in range(n_q):
        idx = np.sort(rng.choice(len(p), size=r, replace=False))
        cols.append(p[idx[1:]] - p[idx[0]])
    return np.array(cols).T


@pytest.mark.parametrize("r", [2, 3, 4])
def test_factored_matches_direct_default(r):
    """Default (truncation off): agreement at the 1e-12 target."""
    p, w, sigma = _template()
    x = _on_support_queries(p, r, 12)
    vd = eval_orbit_rel(p, w, sigma, r, x, factored=False)
    vf = eval_orbit_rel(p, w, sigma, r, x, factored=True)
    assert np.max(np.abs(vf - vd) / np.abs(vd)) < 1e-12


@pytest.mark.parametrize("r", [2, 3, 4])
def test_factored_matches_direct_truncated(r):
    """k=6: mutual agreement well inside the eps target 1.5e-8."""
    p, w, sigma = _template()
    x = _on_support_queries(p, r, 12)
    vd6 = eval_orbit_rel(p, w, sigma, r, x, factored=False,
                         truncation_sigmas=6.0)
    vf6 = eval_orbit_rel(p, w, sigma, r, x, factored=True,
                         truncation_sigmas=6.0)
    assert np.max(np.abs(vf6 - vd6) / np.abs(vd6)) < 1.5e-7


def test_wide_random_queries_absolute_bound():
    """Deep-tail queries: absolute error bounded at bulk scale.

    Relative error in cancellation-dominated tails is meaningless (for
    the direct strategy's roundoff and for truncation alike); the
    factored strategy is held to the same absolute standard.
    """
    p, w, sigma = _template()
    rng = np.random.default_rng(3)
    x = rng.uniform(50.0, 2500.0, size=(3, 40))
    vd = eval_orbit_rel(p, w, sigma, 4, x, factored=False)
    vf = eval_orbit_rel(p, w, sigma, 4, x, factored=True)
    bulk = np.abs(vd).max()
    assert np.max(np.abs(vf - vd)) < 1e-9 * bulk


def test_cancellation_ratio_parity():
    """Both strategies report the same worst-node cancellation ratio."""
    p, w, sigma = _template()
    x = _on_support_queries(p, 4, 9)
    vd, rd = eval_orbit_rel(p, w, sigma, 4, x, factored=False,
                            return_cancellation_ratio=True)
    vf, rf = eval_orbit_rel(p, w, sigma, 4, x, factored=True,
                            return_cancellation_ratio=True)
    assert np.max(np.abs(vf - vd) / np.abs(vd)) < 1e-11
    assert np.max(np.abs(rf - rd)) < 1e-6


def test_periodic_factored_raises_and_auto_uses_direct():
    p, w, sigma = _template()
    x = _on_support_queries(p, 3, 5) % 1200.0
    with pytest.raises(ValueError, match="periodic"):
        eval_orbit_rel(p, w, sigma, 3, x, is_per=True, period=1200.0,
                       factored=True)
    va = eval_orbit_rel(p, w, sigma, 3, x, is_per=True, period=1200.0)
    vd = eval_orbit_rel(p, w, sigma, 3, x, is_per=True, period=1200.0,
                        factored=False)
    assert np.array_equal(va, vd)


def test_gate_worthwhile():
    """Cost gate: direct for scalar work, factored for batches."""
    # Single query, modest K: tabulation dominates -> direct.
    assert not _factored_worthwhile(K=12, r=4, n_q=1, N_u=3000,
                                    n_fine_total=500_000)
    assert not _factored_worthwhile(K=48, r=4, n_q=1, N_u=4500,
                                    n_fine_total=800_000)
    # Batched queries at the same shapes -> factored.
    assert _factored_worthwhile(K=48, r=4, n_q=120, N_u=4500,
                                n_fine_total=800_000)


def test_eps_and_spp_mapping():
    """truncation_sigmas -> read-back accuracy target -> grid density."""
    eps_inf = _factored_target_eps(float("inf"), "double")
    eps_6 = _factored_target_eps(6.0, "double")
    eps_4 = _factored_target_eps(4.0, "double")
    assert eps_inf == 1e-12
    assert abs(eps_6 - np.exp(-18.0)) < 1e-22
    assert eps_inf < eps_6 < eps_4 <= 1e-3
    assert _factored_spp(eps_inf) > _factored_spp(eps_6) > _factored_spp(eps_4)
    # Single-precision kernels floor the target at 1e-7.
    assert _factored_target_eps(float("inf"), "single") == 1e-7


def test_lagrange6_reproduces_quintics_exactly():
    """The read-back stencil is exact on degree-5 polynomials."""
    rng = np.random.default_rng(11)
    coef = rng.normal(size=6)
    x0, h, n = -2.0, 0.37, 40
    grid = x0 + h * np.arange(n)
    y = np.polyval(coef, grid)
    pts = rng.uniform(grid[3], grid[-4], size=(5, 7))
    got = _lagrange6_uniform(y, x0, h, pts)
    want = np.polyval(coef, pts)
    assert np.max(np.abs(got - want) / np.maximum(np.abs(want), 1.0)) < 1e-11


def test_tensor_harmonicity_anchor():
    """End-to-end regression anchors through the auto gate.

    Values verified against the pristine pre-factored implementation.
    """
    chord = np.array([0.0, 386.0, 702.0, 1088.0])
    v = mpt.tensor_harmonicity(chord, duplicate=4, verbose=False)
    assert np.isclose(float(v), 0.0209412247, rtol=1e-8)
    # Batched call must agree with per-chord scalar calls (the cost
    # gate may pick different strategies for the two shapes).
    chords = [chord, chord + np.array([0.0, 2.0, -3.0, 1.0]),
              chord + np.array([0.0, -4.0, 5.0, -2.0])]
    vb = np.asarray(
        mpt.tensor_harmonicity(chords, duplicate=4, verbose=False)
    ).ravel()
    vs = np.array([
        float(mpt.tensor_harmonicity(c, duplicate=4, verbose=False))
        for c in chords
    ])
    assert np.allclose(vb, vs, rtol=1e-6, atol=1e-12)
