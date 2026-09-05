"""Regressions from the routing-parity audit, round 1 (September 2026).

Each block pins one item of the audit's verified findings, in both the
Python and the MATLAB suite (``tests/test_routing_parity_round1.m``), so
that the two implementations keep taking the same route to the same
number for the same input:

* A-1 / B-1 -- the per-call ``truncation_sigmas`` reaches the whole nested
  path (measure rule, kernels, memo key) instead of the global default;
* B-9 -- the flat selector's wrap rule scans relative-periodic attributes
  only;
* B-10 -- a wrap disagreement between the two densities raises instead of
  silently reading ``dens_x``;
* B-11 -- the tau-grid node count, the selector's window margin and the
  centres-vs-grid estimate use the per-call width;
* B-14 -- nested routes skip ``<X,X>`` under ``'oneSidedDenom'``;
* A-7 / B-8 -- the Möbius point evaluator's non-finite guard lives on the
  general path, so every shape inherits it;
* A-9 / B-13 -- ``kernel_precision`` is honoured on the Möbius and centres
  evaluation routes, and forwarded on the entropy list form;
* B-5 -- the factored MA evaluation route passes the attribute's wrap;
* A-17 -- ``eval_exp_tens`` validates ``normalize``; the sweep orbit route
  receives a resolved width;
* B-15 -- ``windowed_tensor_similarity`` honours ``truncation_sigmas`` and
  ``kernel_precision``.
"""
import warnings

import numpy as np
import pytest

import mpt
from mpt import (build_exp_tens, cos_sim_exp_tens, entropy_exp_tens,
                 eval_exp_tens, sweep_cos_sim_exp_tens,
                 windowed_tensor_similarity)
from mpt._tensor import eval as _ev
from mpt._tensor.cosine import (_nested_admissible_routes,
                                _nested_enumeration_admissible)
from mpt._tensor._nested_contraction import auto_ntau_default
from mpt._tensor._mobius_inner import _predicted_grid_wall_ns
from mpt._tensor.dispatch import _select_ma_inner_product_method


@pytest.fixture(autouse=True)
def _quiet():
    mpt.set_default(show_hints=False)
    warnings.simplefilter("ignore", UserWarning)
    yield
    warnings.resetwarnings()
    mpt.reset_defaults()


P = 12.0
_RNG = np.random.default_rng(7)
_VX = np.sort(_RNG.uniform(0.0, P, 9))
_VY = np.sort(_RNG.uniform(0.0, P, 9))


def _nested(values, sigma, *, is_rel=True, is_per=True, wrap='full-image',
            chord=3):
    """One nested attribute: two levels of r = 2, symmetric, chord tags."""
    v = np.asarray(values, float).reshape(-1, 1)
    tags = np.repeat(np.arange(v.shape[0] // chord), chord)
    spec = dict(r=[2, 2], sym=[True, True], tags=tags,
                rel=([0, 1] if is_rel else None))
    return build_exp_tens([v], None, specs=[spec], sigma=[sigma],
                          is_per=[is_per], period=[P if is_per else 0.0],
                          wrap=[wrap], verbose=False)


def _flat(seed, sigma, r=2, *, is_rel=False, is_per=True, wrap='full-image',
          K=5, N=2):
    rng = np.random.default_rng(seed)
    p = np.sort(rng.uniform(0.0, P, size=(K, N)), axis=0)
    return build_exp_tens([p], None, [sigma], [r], [is_rel], [is_per],
                          [P if is_per else 0.0], wrap=[wrap], verbose=False)


# --------------------------------------------------------------- A-1 / B-1


def test_nested_measure_rule_reads_the_per_call_width():
    """At sigma/P = 0.045 the threshold is 0.03 at ts = 6 and 0.05 at
    ts = 4, so the admissible set depends on the width the call passes."""
    dx, dy = _nested(_VX, 0.045 * P), _nested(_VY, 0.045 * P)
    assert _nested_admissible_routes(dx, dy, 0, ts=6.0) == ["taugrid"]
    assert _nested_admissible_routes(dx, dy, 0, ts=4.0) == ["centres",
                                                            "taugrid"]
    assert not _nested_enumeration_admissible(dx, dy, 6.0)
    assert _nested_enumeration_admissible(dx, dy, 4.0)
    # Through the public entry point: a forced centres route is refused
    # at the tighter width and honoured at the looser one.
    with pytest.raises(ValueError, match="cannot be honoured"):
        cos_sim_exp_tens(dx, dy, method="centres", truncation_sigmas=6.0,
                         verbose=False)
    v = cos_sim_exp_tens(dx, dy, method="centres", truncation_sigmas=4.0,
                         verbose=False)
    assert np.isfinite(v)


@pytest.mark.parametrize("is_per", [False, True])
def test_nested_contraction_truncates_at_the_per_call_width(is_per):
    """A coarse per-call width changes the contraction's number and is
    what the self-inner-product memo is keyed on."""
    sigma = 1.0 if is_per else 0.4
    dx = _nested(_VX, sigma, is_rel=False, is_per=is_per)
    dy = _nested(_VY, sigma, is_rel=False, is_per=is_per)
    ref = cos_sim_exp_tens(dx, dy, method="contract", verbose=False)
    dx._self_ip_cache.clear()
    dy._self_ip_cache.clear()
    coarse = cos_sim_exp_tens(dx, dy, method="contract",
                              truncation_sigmas=2.0, verbose=False)
    assert abs(coarse - ref) > 1e-6
    keys = list(dx._self_ip_cache)
    assert keys and all(k[0] == "contract" and k[1] == 2.0 for k in keys)


def test_nested_ma_contraction_truncates_at_the_per_call_width():
    """The multi-attribute nested path resolves the same width."""
    ex = np.array([[3.0]])

    def _d(values):
        v = np.asarray(values, float).reshape(-1, 1)
        tags = np.repeat(np.arange(3), 3)
        spec = dict(r=[2, 2], sym=[True, True], tags=tags, rel=None)
        return build_exp_tens([v, ex], None,
                              specs=[spec, dict(r=1, rel=False, sym=True)],
                              sigma=[0.4, 1.0], is_per=[False, False],
                              period=[0.0, 0.0], verbose=False)
    dx, dy = _d(_VX), _d(_VY)
    ref = cos_sim_exp_tens(dx, dy, method="contract", verbose=False)
    dx._self_ip_cache.clear()
    dy._self_ip_cache.clear()
    coarse = cos_sim_exp_tens(dx, dy, method="contract",
                              truncation_sigmas=2.0, verbose=False)
    assert abs(coarse - ref) > 1e-6
    keys = list(dx._self_ip_cache)
    assert keys and all(k[0] == "contract_ma" and k[1] == 2.0 for k in keys)


# --------------------------------------------------------------------- B-14


def test_nested_route_skips_xx_under_one_sided_denominator():
    dx = _nested(_VX, 0.4, is_rel=False, is_per=False)
    dy = _nested(_VY, 0.4, is_rel=False, is_per=False)
    v = cos_sim_exp_tens(dx, dy, method="contract",
                         normalize="oneSidedDenom", verbose=False)
    assert np.isfinite(v)
    assert not dx._self_ip_cache          # <X,X> neither formed nor memoised
    assert dy._self_ip_cache              # <Y,Y> is the denominator
    b = cos_sim_exp_tens(dx, dy, method="bulger",
                         normalize="oneSidedDenom", verbose=False)
    assert v == pytest.approx(b, rel=1e-6)


# --------------------------------------------------------------------- B-10


def test_wrap_mismatch_raises_on_the_flat_path():
    x = _flat(1, 0.2 * P, wrap='full-image')
    y = _flat(2, 0.2 * P, wrap='single-image')
    with pytest.raises(ValueError, match="wrap mismatch"):
        cos_sim_exp_tens(x, y, verbose=False)


def test_wrap_mismatch_raises_on_the_nested_path():
    dx = _nested(_VX, 0.2 * P, wrap='full-image')
    dy = _nested(_VY, 0.2 * P, wrap='single-image')
    with pytest.raises(ValueError, match="wrap mismatch"):
        cos_sim_exp_tens(dx, dy, method="contract", verbose=False)


# ---------------------------------------------------------------------- B-9


def test_rel_nonper_default_wrap_does_not_mix_with_rel_per_single_image():
    """A relative-non-periodic attribute has no wrap to declare, so it
    cannot make a single-image rel-per declaration 'mixed'."""
    chosen = _select_ma_inner_product_method(
        r_vec=np.array([2, 2]), k_vec=np.array([6, 6]), A=2,
        N_x=2, N_y=2, any_per=True, any_rel_nonper=True, any_rel_per=True,
        sigma_over_P_max=0.2, user_method="auto",
        rel_vec=np.array([True, True]), nu_vec=np.array([200.0, 200.0]),
        wrap_vec=['full-image', 'single-image'], per_vec=[False, True],
        truncation_sigmas=6.0)
    assert chosen == "bulger"


# --------------------------------------------------------------------- B-11


def test_tau_grid_node_count_follows_the_width():
    assert auto_ntau_default(P, 0.5, 3.0) < auto_ntau_default(P, 0.5, 8.0)
    assert auto_ntau_default(P, 0.5) == auto_ntau_default(
        P, 0.5, mpt.get_default("truncation_sigmas"))


def test_grid_estimate_follows_the_width():
    assert (_predicted_grid_wall_ns(12, 2, 0.5, P, True, 3.0)
            < _predicted_grid_wall_ns(12, 2, 0.5, P, True, 8.0))
    assert (_predicted_grid_wall_ns(12, 2, 0.5, 30.0, False, 2.0)
            < _predicted_grid_wall_ns(12, 2, 0.5, 30.0, False, 8.0))


# ---------------------------------------------------------------- A-7 / B-8


def _nan_orbit(*args, **kwargs):
    n_q = int(np.asarray(args[1]).shape[-1])
    out = np.ones(n_q)
    out[0] = np.nan
    return out


@pytest.mark.parametrize("A", [1, 2])
def test_non_finite_mobius_output_falls_back_to_centres(monkeypatch, A):
    """Every shape reaches the guard through the one MA evaluator."""
    rng = np.random.default_rng(11)
    p = [np.sort(rng.uniform(0.0, P, size=(6, 1)), axis=0) for _ in range(A)]
    d = build_exp_tens(p, None, [0.8] * A, [2] * A, [False] * A,
                       [False] * A, [0.0] * A, verbose=False)
    xq = rng.uniform(0.0, P, size=(2 * A, 7))
    ref = eval_exp_tens(d, xq, method="centres", verbose=False)
    import mpt._tensor.dispatch as _disp
    monkeypatch.setattr(_disp, "_select_ma_eval",
                        lambda *a, **k: ("mobius", "forced by test"))
    import mpt._tensor._ma_eval_orbit as _orb
    monkeypatch.setattr(_orb, "eval_ma_orbit", _nan_orbit)
    with pytest.warns(RuntimeWarning, match="non-finite"):
        got = eval_exp_tens(d, xq, method="mobius", verbose=False)
    assert np.all(np.isfinite(got))
    assert got == pytest.approx(ref, rel=1e-12, abs=1e-15)


# --------------------------------------------------------------- A-9 / B-13


@pytest.mark.parametrize("method", ["centres", "mobius"])
def test_eval_single_precision_is_honoured_on_both_routes(method):
    d = _flat(5, 0.6, r=2, is_per=False, K=8, N=3)
    xq = np.random.default_rng(3).uniform(0.0, P, size=(2, 64))
    double = eval_exp_tens(d, xq, method=method, verbose=False)
    single = eval_exp_tens(d, xq, method=method, kernel_precision="single",
                           verbose=False)
    assert single == pytest.approx(double, rel=1e-4)
    assert np.max(np.abs(single - double)) > 0.0


def test_entropy_list_form_forwards_the_width():
    """A relative density reaches the grid evaluator, where the width
    matters; the list form must give the scalar form's number."""
    d = _flat(4, 0.5, r=2, is_rel=True, is_per=False, K=5, N=2)
    kw = dict(method="shannon", n_points_per_dim=64, x_min=-6.0, x_max=6.0,
              verbose=False)
    scalar = entropy_exp_tens(d, truncation_sigmas=1.5, **kw)
    listed = entropy_exp_tens([d], truncation_sigmas=1.5, **kw)
    assert listed.shape == (1,)
    assert listed[0] == pytest.approx(scalar, rel=1e-12)
    assert abs(scalar - entropy_exp_tens(d, **kw)) > 1e-6


# ---------------------------------------------------------------------- B-5


def _two_abs_per(wrap, seed):
    rng = np.random.default_rng(seed)
    p = [np.sort(rng.uniform(0.0, P, size=(4, 2)), axis=0) for _ in range(2)]
    return build_exp_tens(p, None, [0.2 * P] * 2, [2, 2], [False, False],
                          [True, True], [P, P], wrap=[wrap, wrap],
                          verbose=False)


def test_factored_eval_route_honours_the_wrap(monkeypatch):
    xq = np.random.default_rng(9).uniform(0.0, P, size=(4, 12))
    outs = {}
    for wrap in ("full-image", "single-image"):
        d = _two_abs_per(wrap, 21)
        fac = eval_exp_tens(d, xq, method="centres", verbose=False)
        monkeypatch.setattr(_ev, "_ma_eval_factored",
                            lambda *a, **k: None)
        joint = eval_exp_tens(d, xq, method="centres", verbose=False)
        monkeypatch.undo()
        assert fac == pytest.approx(joint, rel=1e-10)
        outs[wrap] = fac
    assert np.max(np.abs(outs["full-image"] - outs["single-image"])) > 1e-6


# --------------------------------------------------------------------- A-17


def test_eval_rejects_an_unknown_normalize():
    d = _flat(1, 0.5, is_per=False)
    with pytest.raises(ValueError, match="normalize must be one of"):
        eval_exp_tens(d, np.zeros((2, 3)), "gaussianish", verbose=False)


def test_sweep_orbit_route_resolves_an_explicit_inf():
    rng = np.random.default_rng(31)
    p_x = [rng.normal(0.0, 3.0, size=(4, 5))]
    p_y = [rng.normal(0.0, 3.0, size=(4, 3))]
    args = ([0.9], [2], [0], [0], [None], [1])
    dx = build_exp_tens(p_x, None, *args, verbose=False)
    dy = build_exp_tens(p_y, None, *args, verbose=False)
    off = np.array([[-2.0, 0.0, 1.5]])
    from mpt._defaults import accuracy_floor_sigmas
    a = sweep_cos_sim_exp_tens(dx, dy, off, method="orbit",
                               truncation_sigmas=np.inf, verbose=False)
    b = sweep_cos_sim_exp_tens(dx, dy, off, method="orbit",
                               truncation_sigmas=accuracy_floor_sigmas(),
                               verbose=False)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------- B-15


def _windowed_case():
    rng = np.random.default_rng(41)
    ctx = build_exp_tens([np.sort(rng.uniform(0.0, 40.0, size=(6, 4)),
                                  axis=0)],
                         None, [1.5], [2], [False], [False], [0.0],
                         verbose=False)
    qry = build_exp_tens([np.sort(rng.uniform(10.0, 20.0, size=(4, 2)),
                                  axis=0)],
                         None, [1.5], [2], [False], [False], [0.0],
                         verbose=False)
    spec = {"size": [4.0], "mix": [0.0]}
    off = np.vstack([np.linspace(-10.0, 10.0, 9)] * 2)
    return ctx, qry, spec, off


def test_windowed_similarity_truncates_at_the_per_call_width():
    ctx, qry, spec, off = _windowed_case()
    ref = windowed_tensor_similarity(ctx, qry, spec, off, verbose=False)
    coarse = windowed_tensor_similarity(ctx, qry, spec, off,
                                        truncation_sigmas=1.0, verbose=False)
    assert coarse.shape == ref.shape
    assert np.max(np.abs(coarse - ref)) > 1e-6


def test_windowed_similarity_honours_single_precision():
    ctx, qry, spec, off = _windowed_case()
    ref = windowed_tensor_similarity(ctx, qry, spec, off, verbose=False)
    single = windowed_tensor_similarity(ctx, qry, spec, off,
                                        kernel_precision="single",
                                        verbose=False)
    assert single == pytest.approx(ref, rel=1e-4)
    assert np.max(np.abs(single - ref)) > 0.0
