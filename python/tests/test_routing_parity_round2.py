"""Regressions from the routing-parity audit, round 2 (September 2026).

Each block pins one item of the audit's verified findings, in both the
Python and the MATLAB suite (``tests/test_routing_parity_round2.m``), so
that the two implementations keep taking the same route to the same
number for the same input:

* A-5 / B-7 -- Rényi-2 entropy of a relative attribute at r = 1 is 0 by
  convention (unit overlap, unit mass), owned by the general
  per-attribute loop so the single-multiset corner inherits it;
* A-8 -- the batched-raw form whitens a kernel covariance at its entry
  and continues (the only reachable outcome is the ordered-at-r > 1
  refusal, in both languages);
* A-12 -- the flat selector's working-set guard (both languages);
* A-13 -- ``method='factored'`` no longer exists; the accepted set is
  MATLAB's;
* A-14 -- the raw-MA scalar-vs-list form takes the all-r = 1 broadcast
  fast path under ``'auto'`` / ``'bulger'`` and honours a forced method;
* A-15 -- the single-attribute helper route agrees with the log-kernel
  core within the truncation floor (MATLAB now takes the same leaf);
* A-16 -- the sweep self-IP memo, the nested centres bundle cache, and
  the density-list dedup (whose key now carries the wrap and [sym]);
* B-6 -- Shannon / normalized cell masses on an absolute-periodic axis
  follow the declared wrap and the resolved truncation width.
"""
import math
import warnings

import numpy as np
import pytest

import mpt
from mpt import (build_exp_tens, cos_sim_exp_tens, entropy_exp_tens,
                 sweep_cos_sim_exp_tens)
from mpt._defaults import resolve_truncation_sigmas, truncation_floor
from mpt._tensor import cosine as _cos
from mpt._tensor.cosine import _ip_full_ma, _ip_via_helper
from mpt._tensor.dispatch import _select_ma_inner_product_method
from mpt._tensor._mobius_inner import _closed_form_attr_centres
from mpt.entropy import _cell_masses_ma_absolute


@pytest.fixture(autouse=True)
def _quiet():
    mpt.set_default(show_hints=False)
    warnings.simplefilter("ignore", UserWarning)
    yield
    warnings.resetwarnings()
    mpt.reset_defaults()


P = 12.0


def _flat(seed, sigma, r=2, *, is_rel=False, is_per=False,
          wrap='full-image', K=5, N=2, is_sym=True):
    rng = np.random.default_rng(seed)
    p = np.sort(rng.uniform(0.0, P, size=(K, N)), axis=0)
    return build_exp_tens([p], None, [sigma], [r], [is_rel], [is_per],
                          [P if is_per else 0.0], [is_sym], wrap=[wrap],
                          verbose=False)


def _chord(values, sigma, r=2, *, is_rel=False, is_per=False,
           wrap='full-image'):
    v = np.asarray(values, float).reshape(-1, 1)
    return build_exp_tens([v], None, [sigma], [r], [is_rel], [is_per],
                          [P if is_per else 0.0], wrap=[wrap],
                          verbose=False)


def _nested(values, sigma, *, chord=3):
    """One relative nested attribute (two levels of r = 2, symmetric)."""
    v = np.asarray(values, float).reshape(-1, 1)
    tags = np.repeat(np.arange(v.shape[0] // chord), chord)
    spec = dict(r=[2, 2], sym=[True, True], tags=tags, rel=[0, 1])
    return build_exp_tens([v], None, specs=[spec], sigma=[sigma],
                          is_per=[False], period=[0.0], verbose=False)


# --------------------------------------------------------------- A-5 / B-7

def test_renyi2_relative_r1_is_zero_at_the_single_multiset_corner():
    d = _chord([0.0, 4.0, 7.0], 1.0, r=1, is_rel=True)
    assert entropy_exp_tens(d, method='renyi2', verbose=False) == 0.0


def test_renyi2_relative_r1_is_zero_for_many_events():
    d = _flat(3, 1.0, r=1, is_rel=True, K=4, N=3)
    assert entropy_exp_tens(d, method='renyi2', verbose=False) == 0.0


def test_renyi2_relative_r1_attribute_contributes_no_entropy():
    # Tensoring a relative r = 1 attribute with an absolute r = 2 one
    # must leave H_2 at the absolute attribute's own value: I_a = 1 and
    # Z_a = 1 for every event pair and event.
    rng = np.random.default_rng(5)
    p_abs = np.sort(rng.uniform(0.0, P, size=(5, 2)), axis=0)
    p_rel = np.sort(rng.uniform(0.0, P, size=(4, 2)), axis=0)
    d_abs = build_exp_tens([p_abs], None, [0.7], [2], [False], [False],
                           [0.0], verbose=False)
    d_both = build_exp_tens([p_abs, p_rel], None, [0.7, 1.0], [2, 1],
                            [False, True], [False, False], [0.0, 0.0],
                            verbose=False)
    h_abs = entropy_exp_tens(d_abs, method='renyi2', verbose=False)
    h_both = entropy_exp_tens(d_both, method='renyi2', verbose=False)
    assert np.isfinite(h_abs)
    assert abs(h_both - h_abs) <= 1e-12 * max(abs(h_abs), 1.0)


# --------------------------------------------------------------------- A-8

def test_batched_raw_kernel_cov_whitens_then_refuses_ordered_r2():
    # The covariance is validated and whitened at the batched-raw entry;
    # what stops the computation is the batched form's ordered-at-r > 1
    # refusal, the same outcome MATLAB now reaches (no
    # 'batchedUnsupported' short-circuit).
    S = np.array([[1.0, 0.3], [0.3, 1.0]])
    P1 = np.array([[0.0, 4.0], [0.0, 7.0]])
    P2 = np.array([[0.0, 3.0], [0.0, 5.0]])
    with pytest.raises(NotImplementedError, match="ordered"):
        cos_sim_exp_tens(P1, None, P2, None, S, 2, False, False, 0.0,
                         False, verbose=False)
    # A malformed covariance is caught by the whitening step itself.
    with pytest.raises(ValueError, match="symmetric"):
        cos_sim_exp_tens(P1, None, P2, None,
                         np.array([[1.0, 0.5], [0.0, 1.0]]),
                         2, False, False, 0.0, False, verbose=False)


# -------------------------------------------------------------------- A-12

def test_selector_working_set_guard_routes_to_mobius():
    # n_J = N * r! * C(K, r): K = 60, r = 3 gives 205 320 tuples per
    # event; 100 events, 2 * 3 * 8 bytes each: ~985 MB > 256 MB.
    common = dict(r_vec=np.array([3]), k_vec=np.array([60]), A=1,
                  N_x=100, N_y=100, any_per=False, any_rel_nonper=False,
                  any_rel_per=False, sigma_over_P_max=0.0,
                  rel_vec=np.array([False]), truncation_sigmas=6.0)
    chosen, pw, orb = _select_ma_inner_product_method(
        user_method='auto', return_costs=True, **common)
    assert chosen == 'mobius' and math.isnan(pw) and math.isnan(orb)
    # The same shape at 5 events (49 MB) is priced instead.
    chosen, pw, orb = _select_ma_inner_product_method(
        user_method='auto', return_costs=True,
        **{**common, 'N_x': 5, 'N_y': 5})
    assert np.isfinite(pw) and np.isfinite(orb)
    # A relative-periodic attribute above the sigma/P threshold is left
    # to the wrap rule, whatever the working set.
    chosen, pw, orb = _select_ma_inner_product_method(
        user_method='auto', return_costs=True,
        **{**common, 'any_rel_per': True, 'any_per': True,
           'rel_vec': np.array([True]), 'sigma_over_P_max': 0.2,
           'wrap_vec': ['single-image'], 'per_vec': [True]})
    assert chosen == 'bulger'


# -------------------------------------------------------------------- A-13

def test_method_factored_is_gone():
    dx, dy = _flat(1, 0.6), _flat(2, 0.6)
    with pytest.raises(ValueError) as exc:
        cos_sim_exp_tens(dx, dy, method='factored', verbose=False)
    msg = str(exc.value)
    assert "'auto', 'bulger', 'centres', 'mobius', 'contract'" in msg
    assert 'factored' not in msg.split(';')[0]
    for name in ('_cos_sim_exp_tens_ma_factored', '_ma_factored_ip_supported',
                 '_ma_ip_factored', '_ma_ip_factor_dense',
                 '_ma_ip_per_event_factors', '_nested_factor_cullable',
                 '_ma_ip_factor_nested_culled'):
        assert not hasattr(_cos, name), name


# -------------------------------------------------------------------- A-14

def _raw_ma_r1(seed, n_list=4):
    rng = np.random.default_rng(seed)
    ref = [np.sort(rng.uniform(0.0, P, (5, 2)), axis=0)]
    lst = [[np.sort(rng.uniform(0.0, P, (5, 2)), axis=0)]
           for _ in range(n_list)]
    return ref, lst


def test_raw_ma_list_form_takes_r1_fast_path(monkeypatch):
    ref, lst = _raw_ma_r1(11)
    calls = []
    orig = _cos._r1_broadcast_fast

    def spy(*args, **kwargs):
        out = orig(*args, **kwargs)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(_cos, '_r1_broadcast_fast', spy)
    args = (ref, None, lst, None, [0.5], [1], [False], [False], [0.0])
    fast = cos_sim_exp_tens(*args, verbose=False)
    assert calls == [True]
    # Reversed operand order takes the same path.
    calls.clear()
    fast_rev = cos_sim_exp_tens(lst, None, ref, None, [0.5], [1], [False],
                                [False], [0.0], verbose=False)
    assert calls == [True]
    # A forced 'mobius' names the per-pair route and never asks.
    calls.clear()
    forced = cos_sim_exp_tens(*args, method='mobius', verbose=False)
    assert calls == []
    monkeypatch.setattr(_cos, '_r1_broadcast_fast', lambda *a, **k: None)
    slow = cos_sim_exp_tens(*args, verbose=False)
    assert fast.shape == (4,)
    assert np.allclose(fast, slow, rtol=0, atol=1e-12)
    assert np.allclose(fast_rev, slow, rtol=0, atol=1e-12)
    assert np.allclose(forced, slow, rtol=0, atol=1e-9)


# -------------------------------------------------------------------- A-15

@pytest.mark.parametrize("r,is_rel,is_per,wrap,sigma", [
    (2, False, False, 'full-image', 0.7),
    (3, False, False, 'full-image', 0.9),
    (2, False, True, 'full-image', 0.15 * P),
    (2, False, True, 'single-image', 0.15 * P),
    (2, False, True, 'full-image', 0.03 * P),
    (2, True, False, 'full-image', 0.6),
    (1, False, True, 'full-image', 0.2 * P),
])
@pytest.mark.parametrize("ts", [None, 4.0, math.inf])
def test_single_attribute_helper_agrees_with_log_kernel_core(
        r, is_rel, is_per, wrap, sigma, ts):
    dx = _flat(21, sigma, r, is_rel=is_rel, is_per=is_per, wrap=wrap, K=6)
    dy = _flat(22, sigma, r, is_rel=is_rel, is_per=is_per, wrap=wrap, K=5)
    tsr = resolve_truncation_sigmas(ts)
    period = P if is_per else 0.0
    h = _ip_via_helper(dx.u_perm[0], dx.w_j, dy.v_comb[0], dy.wv_comb,
                       r, sigma, is_rel, is_per, period,
                       truncation_sigmas=tsr, wrap_a=wrap)
    g = _ip_full_ma(dx.u_perm, dx.w_j, dx.n_j, dy.v_comb, dy.wv_comb,
                    dy.n_k, 1, dx.r, dx.sigma, dx.is_rel, dx.is_per,
                    dx.period, truncation_sigmas=tsr,
                    inner_r=np.zeros(1, dtype=int), wrap=[wrap])
    assert abs(h - g) <= max(truncation_floor(ts), 1e-12) * abs(g)


@pytest.mark.parametrize("r,is_rel,is_per,wrap,sigma", [
    (2, False, False, 'full-image', 0.7),
    (2, False, True, 'full-image', 0.15 * P),
    (2, True, False, 'full-image', 0.6),
])
@pytest.mark.parametrize("ts", [4.0, math.inf])
def test_helper_route_agrees_with_log_kernel_core_at_the_cosine(
        r, is_rel, is_per, wrap, sigma, ts):
    # A second attribute holding one shared value per event multiplies
    # every kernel entry by exactly 1 and leaves the tuple counts
    # unchanged, so the two-attribute density is the same inner product
    # computed through the log-kernel form (the MATLAB twin pins the
    # agreement this way, having no direct access to the core).
    rng = np.random.default_rng(23)
    px = np.sort(rng.uniform(0.0, P, (6, 2)), axis=0)
    py = np.sort(rng.uniform(0.0, P, (5, 2)), axis=0)
    period = P if is_per else 0.0
    one = 3.0 * np.ones((1, 2))
    dx1 = build_exp_tens([px], None, [sigma], [r], [is_rel], [is_per],
                         [period], wrap=[wrap], verbose=False)
    dy1 = build_exp_tens([py], None, [sigma], [r], [is_rel], [is_per],
                         [period], wrap=[wrap], verbose=False)
    dx2 = build_exp_tens([px, one], None, [sigma, 1.0], [r, 1],
                         [is_rel, False], [is_per, False], [period, 0.0],
                         wrap=[wrap, 'full-image'], verbose=False)
    dy2 = build_exp_tens([py, one], None, [sigma, 1.0], [r, 1],
                         [is_rel, False], [is_per, False], [period, 0.0],
                         wrap=[wrap, 'full-image'], verbose=False)
    s1 = cos_sim_exp_tens(dx1, dy1, method='bulger', truncation_sigmas=ts,
                          verbose=False)
    s2 = cos_sim_exp_tens(dx2, dy2, method='bulger', truncation_sigmas=ts,
                          verbose=False)
    assert abs(s1 - s2) <= 10 * max(truncation_floor(ts), 1e-12) * abs(s2)


def test_single_attribute_cosine_honours_kernel_precision():
    dx, dy = _flat(31, 0.6, 2, K=8), _flat(32, 0.6, 2, K=8)
    dbl = cos_sim_exp_tens(dx, dy, method='bulger', verbose=False)
    sgl = cos_sim_exp_tens(dx, dy, method='bulger',
                           kernel_precision='single', verbose=False)
    assert 0 < abs(sgl - dbl) <= 1e-4 * abs(dbl)


# -------------------------------------------------------------------- A-16

def test_sweep_self_ip_memo_key():
    rng = np.random.default_rng(41)
    sx = build_exp_tens([rng.normal(size=(4, 5)) * 3], None, [0.9], [2],
                        [False], [False], [0.0], verbose=False)
    sy = build_exp_tens([rng.normal(size=(4, 3)) * 3], None, [0.9], [2],
                        [False], [False], [0.0], verbose=False)
    sweep_cos_sim_exp_tens(sx, sy, np.array([-2.0, 0.0, 1.5]),
                           method='mixture', verbose=False)
    keys = [k for k in sy.pruned()._self_ip_cache if k[0] == 'sweep']
    assert len(keys) == 1
    assert keys[0][1] == resolve_truncation_sigmas(None)


def test_nested_centres_bundle_is_cached_on_the_density():
    rng = np.random.default_rng(7)
    d = _nested(np.sort(rng.uniform(0.0, P, 9)), 0.5).pruned()
    assert d._nested_centres_cache == {}
    b1 = _closed_form_attr_centres(d, 0)
    assert 0 in d._nested_centres_cache
    assert _closed_form_attr_centres(d, 0) is b1


def test_density_list_dedup_key_carries_the_wrap():
    # Two pairs identical up to the wrap of the periodic attribute at a
    # sigma/P where the two measures differ: the dedup must keep them
    # apart, and the list form must return the two scalar values.
    sig = 0.3 * P
    xf = _chord([0.0, 4.0, 7.0], sig, is_per=True, wrap='full-image')
    yf = _chord([0.0, 3.0, 7.0], sig, is_per=True, wrap='full-image')
    xs = _chord([0.0, 4.0, 7.0], sig, is_per=True, wrap='single-image')
    ys = _chord([0.0, 3.0, 7.0], sig, is_per=True, wrap='single-image')
    s_full = cos_sim_exp_tens(xf, yf, verbose=False)
    s_single = cos_sim_exp_tens(xs, ys, verbose=False)
    assert abs(s_full - s_single) > 1e-6
    out = cos_sim_exp_tens([xf, xs], [yf, ys], mode='pairwise',
                           verbose=False)
    assert abs(out[0] - s_full) <= 1e-12
    assert abs(out[1] - s_single) <= 1e-12


# --------------------------------------------------------------------- B-6

def test_full_image_cell_masses_sum_to_total_and_differ_from_single_image():
    sig = 0.3 * P
    axes = [np.linspace(0.0, P, 64, endpoint=False)]
    d_full = _chord([0.0, 4.0, 7.0], sig, r=1, is_per=True,
                    wrap='full-image')
    d_single = _chord([0.0, 4.0, 7.0], sig, r=1, is_per=True,
                      wrap='single-image')
    ts = resolve_truncation_sigmas(math.inf)
    m_full = _cell_masses_ma_absolute(d_full, axes, truncation_sigmas=ts)
    m_single = _cell_masses_ma_absolute(d_single, axes, truncation_sigmas=ts)
    total = float(np.sum(d_full.w_j))
    assert abs(m_full.sum() - total) <= 1e-10
    assert abs(m_single.sum() - total) > 1e-3
    assert np.max(np.abs(m_full - m_single)) > 1e-3


def test_shannon_entropy_follows_the_wrap_and_the_width():
    big, small = 0.3 * P, 0.02 * P
    kw = dict(method='shannon', n_points_per_dim=64, verbose=False)
    h_full = entropy_exp_tens(_chord([0.0, 4.0, 7.0], big, r=1, is_per=True,
                                     wrap='full-image'), **kw)
    h_single = entropy_exp_tens(_chord([0.0, 4.0, 7.0], big, r=1,
                                       is_per=True, wrap='single-image'),
                                **kw)
    assert abs(h_full - h_single) > 1e-6
    # Below the overlap regime the two readings coincide.
    h_full_s = entropy_exp_tens(_chord([0.0, 4.0, 7.0], small, r=1,
                                       is_per=True, wrap='full-image'), **kw)
    h_single_s = entropy_exp_tens(_chord([0.0, 4.0, 7.0], small, r=1,
                                         is_per=True, wrap='single-image'),
                                  **kw)
    assert abs(h_full_s - h_single_s) <= 1e-9
    # The per-call width governs the image count: a width so narrow
    # that no image is admitted (L = 0) reads a single Gaussian.
    d = _chord([0.0, 4.0, 7.0], big, r=1, is_per=True, wrap='full-image')
    h_narrow = entropy_exp_tens(d, truncation_sigmas=1.0, **kw)
    h_floor = entropy_exp_tens(d, truncation_sigmas=math.inf, **kw)
    assert abs(h_narrow - h_floor) > 1e-6
    assert abs(h_full - h_floor) <= 1e-9   # default width already at L >= 1
