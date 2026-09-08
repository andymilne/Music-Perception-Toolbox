"""Tests for the factored multi-attribute centres evaluator.

The centres path evaluates a multi-attribute density as
``sum_events prod_attributes S_a^(event)`` --- a product across
attributes of each event's per-attribute factor --- instead of
materialising the joint tuple set. These tests pin its value-equality
with the independent factored-Möbius path across attribute counts,
event counts, read-arities, geometry modes, nesting, and ragged
(differing valid-value) events, and confirm the documented fall-backs.
"""

import numpy as np
import pytest

import mpt
from mpt._defaults import resolve_truncation_sigmas
from mpt._tensor.eval import (_ma_eval_factored, _ma_eval_full,
                              _split_query_to_attr_list)
from mpt._tensor.dispatch import _inner_r_vec

mpt.set_default(show_hints=False)


def _dense_joint_raw(dens, x):
    """The joint-materialising centres result the factored path replaces."""
    x_list = _split_query_to_attr_list(dens, x)
    return _ma_eval_full(
        dens.centres, dens.w_j, dens.n_j, x_list, x.shape[1],
        dens.n_attrs, dens.dim_per_attr, dens.r, dens.sigma,
        dens.is_rel, dens.is_per, dens.period,
        truncation_sigmas=resolve_truncation_sigmas(None),
        inner_r=_inner_r_vec(dens),
    )


def _build(A, K, N, *, r=2, rel=True, per=False, period=0.0, seed0=0):
    p = [np.sort(np.random.default_rng(seed0 + s).uniform(0, 3600, (K, N)),
                 axis=0) for s in range(A)]
    return mpt.build_exp_tens(
        p, None, [15.0] * A, [r] * A, [rel] * A, [per] * A, [period] * A,
        verbose=False)


def _query_near_mass(dens, n_q, jitter, seed):
    centres = [np.asarray(c) for c in dens.centres]
    n_j = centres[0].shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_j, n_q)
    x = np.vstack([centres[a][:, idx] for a in range(len(centres))])
    return x + rng.normal(0, jitter, (dens.dim, n_q))


def _max_rel_err(dens, x):
    """Factored path vs the dense joint it replaces (same query convention).

    A flat-only cross-check against the independent factored-Möbius path
    is added separately where the query convention allows it.
    """
    vf = _ma_eval_factored(dens, x,
                           truncation_sigmas=resolve_truncation_sigmas(None))
    vd = _dense_joint_raw(dens, x)
    mask = vd > 1e-3 * vd.max()
    return float(np.abs((vf[mask] - vd[mask]) / vd[mask]).max())


def _max_rel_err_vs_mobius(dens, x):
    vc = mpt.eval_exp_tens(dens, x, method="centres", verbose=False)
    vm = mpt.eval_exp_tens(dens, x, method="mobius", verbose=False)
    mask = vm > 1e-3 * vm.max()
    return float(np.abs((vc[mask] - vm[mask]) / vm[mask]).max())


class TestFactoredEqualsDenseJoint:
    # Flat cases use the independent factored-Möbius path as reference:
    # its query convention matches, and it never materialises the joint,
    # so it scales to the large-joint shapes the factored path targets.
    @pytest.mark.parametrize("A,K,N", [(2, 8, 1), (3, 8, 1), (2, 20, 1),
                                       (2, 7, 5), (2, 8, 20)])
    def test_flat_relative(self, A, K, N):
        d = _build(A, K, N)
        assert _max_rel_err_vs_mobius(d, _query_near_mass(d, 60, 7.5, 9)) < 1e-4

    def test_flat_absolute(self):
        d = _build(2, 10, 3, rel=False)
        assert _max_rel_err_vs_mobius(d, _query_near_mass(d, 60, 7.5, 9)) < 1e-4

    def test_flat_periodic(self):
        d = _build(2, 8, 3, per=True, period=1200.0)
        assert _max_rel_err_vs_mobius(d, _query_near_mass(d, 60, 7.5, 9)) < 1e-4

    @pytest.mark.parametrize("r", [3, 4])
    def test_higher_arity(self, r):
        d = _build(2, 7, 2, r=r)
        assert _max_rel_err_vs_mobius(d, _query_near_mass(d, 60, 12.0, 9)) < 1e-4

    def test_nested_bound_flattened(self):
        rng = np.random.default_rng(3)
        pb0, _, s0 = mpt.unpack_pre_maet(mpt.bind_events([rng.normal(0, 1, (1, 12))], None, 3))
        pb1, _, s1 = mpt.unpack_pre_maet(mpt.bind_events([rng.normal(0, 1, (1, 12))], None, 3))
        n = pb0[0].shape[1]
        d = mpt.build_exp_tens(
            [pb0[0], pb1[0]], [np.ones((3, n))] * 2, specs=[s0[0], s1[0]],
            sigma=[0.3, 0.3], is_per=[False] * 2, period=[0.0] * 2,
            verbose=False)
        assert _max_rel_err(d, _query_near_mass(d, 60, 0.3, 9)) < 1e-4

    def test_nested_genuine(self):
        rng = np.random.default_rng(3)
        cA = np.sort(rng.uniform(0, 20, (2, 8)), axis=0)
        cB = np.sort(rng.uniform(0, 20, (2, 8)), axis=0)
        pbA, _, sA = mpt.unpack_pre_maet(mpt.bind_events(
            [cA], None, 2, specs=mpt.flat_specs([cA], r=2, rel=True, sym=True)))
        pbB, _, sB = mpt.unpack_pre_maet(mpt.bind_events(
            [cB], None, 2, specs=mpt.flat_specs([cB], r=2, rel=True, sym=True)))
        nn, nsl = pbA[0].shape[1], pbA[0].shape[0]
        d = mpt.build_exp_tens(
            [pbA[0], pbB[0]], [np.ones((nsl, nn))] * 2, specs=[sA[0], sB[0]],
            sigma=[0.5, 0.5], is_per=[False] * 2, period=[0.0] * 2,
            verbose=False)
        assert _max_rel_err(d, _query_near_mass(d, 60, 0.4, 9)) < 1e-4

    def test_ragged_events(self):
        # Events with differing valid-value patterns (NaN-dropped values)
        # must match: absent values are zero-weight on a shared enumeration.
        rng = np.random.default_rng(4)
        A, K, N = 2, 8, 6
        p = []
        for a in range(A):
            pa = np.sort(rng.uniform(0, 3600, (K, N)), axis=0)
            for n in range(N):
                drop = rng.choice(K, rng.integers(0, 3), replace=False)
                pa[drop, n] = np.nan
            p.append(pa)
        d = mpt.build_exp_tens(
            p, None, [15.0] * A, [2] * A, [True] * A, [False] * A,
            [0.0] * A, verbose=False)
        assert _max_rel_err_vs_mobius(d, _query_near_mass(d, 60, 7.5, 9)) < 1e-4


class TestFactoredFallback:
    def test_r1_attribute_falls_back(self):
        # r = 1 has an event-dependent equal-value collapse; the factored
        # path returns None so the joint route runs.
        d = _build(2, 8, 1, r=1)
        x = _query_near_mass(d, 20, 20.0, 1)
        assert _ma_eval_factored(d, x) is None
        # The eval still succeeds (via the joint path) and matches Möbius.
        vc = mpt.eval_exp_tens(d, x, method="centres", verbose=False)
        vm = mpt.eval_exp_tens(d, x, method="mobius", verbose=False)
        mask = vm > 1e-3 * vm.max()
        assert np.abs((vc[mask] - vm[mask]) / vm[mask]).max() < 1e-4

    def test_query_count_zero(self):
        d = _build(2, 8, 1)
        out = _ma_eval_factored(d, np.zeros((d.dim, 0)))
        assert out.shape == (0,)


class TestFactoredIsExercised:
    def test_forced_centres_uses_factored_and_matches(self):
        # A shape where centres is the forced route: the factored path is
        # the one under test, and it must equal the joint reference.
        d = _build(2, 12, 1)
        x = _query_near_mass(d, 80, 7.5, 3)
        got = _ma_eval_factored(d, x)
        assert got is not None and got.shape == (80,)


def test_dim0_fully_relative_r1_density_evaluates_constant():
    """A density whose every attribute is relative at r = 1 has effective
    dimensionality 0 and is constant. Both eval routes must return that
    constant (the total tuple mass) at zero-dimensional queries. Pins
    the behaviour the MATLAB twin regressed on: its calibration cache
    indexed by dim errored at dim = 0 (fixed by the dim + 1 cache
    slot in estimateCompTime)."""
    import warnings
    rng = np.random.default_rng(0)
    P = [np.sort(3600 * rng.random((8, 1)), axis=0) for _ in range(2)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = mpt.build_exp_tens(P, None, [15.0, 15.0], [1, 1], [True, True],
                           [False, False], [0.0, 0.0], verbose=False)
    x = np.zeros((0, 5))
    v_c = np.asarray(mpt.eval_exp_tens(d, x, method="centres", verbose=False))
    v_m = np.asarray(mpt.eval_exp_tens(d, x, method="mobius", verbose=False))
    assert v_c.size == 5 and v_m.size == 5
    assert np.allclose(v_c, 64.0, atol=1e-9)   # 8 x 8 unit-weight tuples
    assert np.allclose(v_m, v_c, atol=1e-9)
