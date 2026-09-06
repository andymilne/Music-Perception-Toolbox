"""The bare inner product on the canonical scale.

``cos_sim_exp_tens(..., normalize='none')`` returns :math:`\\langle X, Y
\\rangle` on one scale whatever route ran, and the Rényi-2 entropy is
computed from it. Both are pinned here against an explicit enumeration
of every tuple pair on every shape the routes cover: flat symmetric,
ordered, relative, periodic; nested with every level pattern, absolute
and with either co-transposition unit, periodic or not; mixed densities;
and every forced method. Mirror of MATLAB tests/test_inner_product_scale.m.
"""
import math

import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, cos_sim_exp_tens, entropy_exp_tens

P = 12.0
T2 = np.repeat(np.arange(2), 3)
T3 = np.repeat(np.arange(3), 3)
T3L = np.array([[0, 0], [0, 0], [1, 0], [1, 0], [2, 1], [2, 1], [3, 1], [3, 1]])


@pytest.fixture(autouse=True)
def _floor_and_quiet():
    prev = mpt.get_default("truncation_sigmas")
    prev_h = mpt.get_default("show_hints")
    mpt.set_default(truncation_sigmas=math.inf, show_hints=False)
    try:
        yield
    finally:
        mpt.set_default(truncation_sigmas=prev, show_hints=prev_h)


def reference_attr_matrix_and_mass(dens, a):
    """Reference (event, event) inner matrix and per-event mass of one
    attribute by explicit tuple enumeration on the canonical scale: kernel
    overlap ``(pi sigma^2)^{d/2} / sqrt(det M) exp(-Q/(4 sigma^2))``, mass
    ``(2 pi sigma^2)^{d/2} / sqrt(det M)``, over the attribute's full
    ordered tuple set (every arrangement a symmetric level admits). This
    is the enumeration the Rényi-2 entropy carried for nested and
    ordered attributes until the inner-product machinery served it."""
    from mpt._tensor.build import build_exp_tens
    from mpt._tensor.dispatch import (
        _compute_Q_inner_blocks, _compute_Q, _inner_r_vec,
        _quadratic_form_det, _gaussian_mass_const,
    )

    nested = getattr(dens, "nested", None)
    spec = nested[a] if nested is not None else None
    sigma = float(dens.sigma[a])
    is_per = bool(dens.is_per[a])
    period = float(dens.period[a])
    if spec is not None:
        # Nested attribute: rebuild from its resolved spec.
        da = build_exp_tens(
            [dens.p_attr[a]], [dens.w[a]], specs=[spec],
            sigma=[sigma], is_per=[is_per], period=[period], verbose=False,
        )
    else:
        # Flat ordered attribute: rebuild from its flat parameters with
        # is_sym=False, so the materialised tuples are the C(K, r_a)
        # ordered sub-tuples (one kernel each, no orbit).
        r_a0 = int(dens.r[a])
        is_rel0 = bool(dens.is_rel[a])
        sym0 = bool(np.atleast_1d(getattr(dens, "is_sym",
                                          np.ones(dens.n_attrs, bool)))[a])
        da = build_exp_tens(
            [dens.p_attr[a]], [dens.w[a]],
            [sigma], [r_a0], [is_rel0], [is_per], [period], [sym0],
            verbose=False,
        )
    centres = da.centres[0]            # (d_a, n_j) reduced centres
    w_j = da.w_j                       # (n_j,)
    event_of_j = da.event_of_j         # (n_j,) -> event index 0..N-1
    d_a = centres.shape[0]
    n_j = w_j.shape[0]
    N = int(dens.n)

    block_size = int(_inner_r_vec(da)[0])   # s_u (inner/intermediate) or 0
    is_rel = bool(da.is_rel[0])
    r_a = int(da.r[0])
    det_m = _quadratic_form_det(r_a, block_size, is_rel)
    vol = _gaussian_mass_const(sigma, d_a, det_m)              # mass
    pref = _gaussian_mass_const(sigma, d_a, det_m, half=True)  # overlap

    I_a = np.zeros((N, N), dtype=np.float64)
    Z_a = np.zeros(N, dtype=np.float64)
    if n_j > 0:
        # Pairwise (block-)metric quadratic form on the reduced centres.
        D = centres[:, :, None] - centres[:, None, :]   # (d_a, n_j, n_j)
        if block_size >= 2:
            Q = _compute_Q_inner_blocks(
                D, block_size, is_per, period, reduced=True)
            overlap = pref * np.exp(-Q / (4 * sigma ** 2))
        elif is_per and not is_rel:
            # Abs-per: full-image via shared helper (image-sum or
            # Fourier by cost); single-image opt-in evaluates the
            # nearest image only.
            wrap_a = 'full-image'
            if hasattr(dens, 'wrap') and dens.wrap is not None:
                wrap_a = str(dens.wrap[a])
            if wrap_a == 'single-image':
                D = D - period * np.floor(D / period + 0.5)
                Q = _compute_Q(D, r_a, is_rel, is_per, period,
                               reduced=is_rel)
                overlap = pref * np.exp(-Q / (4 * sigma ** 2))
            else:
                from mpt._wrapped_kernel import wrapped_gaussian_1d
                from mpt._defaults import get_default
                ts = get_default("truncation_sigmas")
                theta_per_position = wrapped_gaussian_1d(
                    D, sigma, period, ts, exponent_denominator=4
                )
                overlap = pref * theta_per_position.prod(axis=0)
        else:
            Q = _compute_Q(D, r_a, is_rel, is_per, period, reduced=is_rel)
            overlap = pref * np.exp(-Q / (4 * sigma ** 2))
        wo = (w_j[:, None] * w_j[None, :]) * overlap
        # Aggregate tuples into their events (G is the N x n_j incidence).
        G = np.zeros((N, n_j), dtype=np.float64)
        G[event_of_j, np.arange(n_j)] = 1.0
        I_a = G @ wo @ G.T
        Z_a = vol * (G @ w_j)
    return I_a, Z_a




def reference_self_ip_and_mass(dens):
    """``(<T,T>, Z)`` by enumeration, composing attributes as the Rényi-2
    factorisation does."""
    dens = dens.pruned()
    A, N = dens.n_attrs, dens.n
    nested = getattr(dens, "nested", [None] * A)
    P_xx = np.ones((N, N))
    Zs = np.ones((N, A))
    for a in range(A):
        r_a = int(dens.r[a])
        if nested[a] is None and bool(dens.is_rel[a]) and r_a == 1:
            continue                       # 0-D point mass: unit overlap and mass
        I, Z = reference_attr_matrix_and_mass(dens, a)
        P_xx *= I
        Zs[:, a] = Z
    return float(P_xx.sum()), float(np.prod(Zs, axis=1).sum())


def _flat(K, N, r, rel, per, sym=True, sigma=0.7, seed=0):
    rng = np.random.default_rng(seed)
    p = np.sort(rng.uniform(0.0, P, size=(K, N)), axis=0)
    w = rng.uniform(0.5, 1.5, size=(K, N))
    return build_exp_tens([p], [w], specs=[{"r": r, "sym": sym, "rel": rel}],
                          sigma=[sigma], is_per=[per], period=[P], verbose=False)


def _nested(tags, r, sym, rel, per, K, N=3, sigma=0.7, seed=0):
    rng = np.random.default_rng(seed)
    p = np.sort(rng.uniform(0.0, P, size=(K, N)), axis=0)
    w = rng.uniform(0.5, 1.5, size=(K, N))
    spec = {"tags": tags, "r": r, "sym": sym, "rel": rel}
    return build_exp_tens([p], [w], specs=[spec], sigma=[sigma], is_per=[per],
                          period=[P], verbose=False)


FLAT = [
    (6, 3, 2, False, False, True), (6, 3, 3, False, False, True),
    (6, 3, 2, True, False, True), (6, 3, 3, True, False, True),
    (6, 3, 2, False, True, True), (6, 3, 2, True, True, True),
    (6, 3, 1, False, False, True), (6, 3, 1, False, True, True),
    (6, 3, 2, False, False, False), (6, 3, 3, True, True, False),
    (6, 3, 3, True, False, False),
]
NESTED = [
    (T2, [1, 2], [1, 1], [0, 0], False, 6), (T2, [1, 2], [1, 0], [0, 0], False, 6),
    (T2, [1, 2], [0, 1], [0, 0], True, 6), (T3, [2, 2], [1, 1], [0, 0], False, 9),
    (T3, [2, 2], [1, 1], [0, 0], True, 9), (T3, [2, 2], [1, 1], [1, 0], False, 9),
    (T3, [2, 2], [1, 1], [0, 1], False, 9), (T3, [2, 2], [1, 1], [0, 1], True, 9),
    (T3, [3, 2], [1, 1], [0, 1], False, 9), (T3, [1, 3], [1, 1], [0, 1], False, 9),
    (T3, [1, 3], [1, 1], [0, 1], True, 9), (T3, [2, 2], [1, 0], [1, 0], True, 9),
    (T2, [2, 2], [0, 0], [0, 0], False, 6), (T3, [2, 3], [1, 1], [0, 1], True, 9),
    (T3L, [2, 2, 2], [1, 1, 1], [0, 0, 0], False, 8),
    (T3L, [2, 2, 2], [1, 0, 1], [0, 0, 1], True, 8),
    (T3L, [2, 2, 2], [0, 1, 0], [0, 0, 1], False, 8),
]


def _tol(dens):
    # Relative-periodic attributes: the full-image routes and the
    # single-image enumeration differ by the measure gap at sigma/P.
    rel_per = any(bool(dens.is_rel[a]) and bool(dens.is_per[a])
                  for a in range(dens.n_attrs))
    return 1e-3 if rel_per else 1e-9


def _check_every_method(dens, methods):
    ref, _ = reference_self_ip_and_mass(dens)
    tol = _tol(dens)
    for m in methods:
        try:
            val = cos_sim_exp_tens(dens, dens, normalize="none", method=m,
                                   verbose=False)
        except ValueError as exc:
            # a forced route the shape does not admit; the message says so
            assert "not available" in str(exc) or "cannot be honoured" in str(exc)
            continue
        assert val == pytest.approx(ref, rel=tol), (m, val, ref)


@pytest.mark.parametrize("K,N,r,rel,per,sym", FLAT)
def test_flat_routes_agree_on_the_bare_inner_product(K, N, r, rel, per, sym):
    d = _flat(K, N, r, rel, per, sym)
    _check_every_method(d, ("auto", "bulger", "mobius", "centres"))


@pytest.mark.parametrize("tags,r,sym,rel,per,K", NESTED)
def test_nested_routes_agree_on_the_bare_inner_product(tags, r, sym, rel, per, K):
    d = _nested(tags, r, sym, rel, per, K)
    _check_every_method(d, ("auto", "bulger", "mobius", "centres", "contract"))


def test_relative_nonperiodic_contraction_scale_is_data_independent():
    # The line grid clamps to 64 nodes on narrow data; the trapezoid
    # weighting keeps the bare value on scale either way.
    from mpt._tensor import cosine as C
    for seed, sigma in [(1, 0.7), (2, 1.5), (0, 0.3), (3, 1.5)]:
        d = _nested(T3, [2, 2], [1, 1], [0, 1], False, 9, sigma=sigma, seed=seed)
        ref, _ = reference_self_ip_and_mass(d)
        t = C._try_nested_contract(d, d, normalize="none", verbose=False,
                                   force=True, method_name="contract",
                                   force_route="contract_relnonper",
                                   truncation_sigmas=None)
        val = t[0] * C._ip_canonical_scale(d, "contract", list(C._LAST_NESTED_ROUTES))
        assert val == pytest.approx(ref, rel=1e-9)


def test_mixed_density_and_cross_terms():
    rng = np.random.default_rng(5)
    p0 = np.sort(rng.uniform(0, P, size=(6, 3)), axis=0)
    p1 = np.sort(rng.uniform(0, P, size=(4, 3)), axis=0)
    p2 = np.sort(rng.uniform(0, P, size=(4, 3)), axis=0)
    specs = [{"tags": T2, "r": [1, 2], "sym": [1, 1], "rel": [0, 0]},
             {"r": 2, "sym": True, "rel": True},
             {"r": 2, "sym": False, "rel": False}]
    d = build_exp_tens([p0, p1, p2], None, specs=specs, sigma=[0.7, 0.5, 0.9],
                       is_per=[False, False, True], period=[P, P, P], verbose=False)
    _check_every_method(d, ("auto", "bulger", "mobius", "centres", "contract"))
    # a cross term between two different densities: the same scale, so the
    # cosine recomposes from three bare values
    e = build_exp_tens([p0[:, ::-1] + 0.3, p1 + 0.1, p2 - 0.2], None, specs=specs,
                       sigma=[0.7, 0.5, 0.9], is_per=[False, False, True],
                       period=[P, P, P], verbose=False)
    xy = cos_sim_exp_tens(d, e, normalize="none", verbose=False)
    xx = cos_sim_exp_tens(d, d, normalize="none", verbose=False)
    yy = cos_sim_exp_tens(e, e, normalize="none", verbose=False)
    cos = cos_sim_exp_tens(d, e, verbose=False)
    assert xy / math.sqrt(xx * yy) == pytest.approx(cos, rel=1e-9)


def test_none_forms_no_self_inner_product():
    d = _flat(6, 3, 2, False, False)
    e = _flat(6, 3, 2, False, False, seed=1)
    cos_sim_exp_tens(d, e, normalize="none", verbose=False)
    assert not d._self_ip_cache and not e._self_ip_cache
    cos_sim_exp_tens(d, e, normalize="oneSidedDenom", verbose=False)
    assert not d._self_ip_cache and e._self_ip_cache


@pytest.mark.parametrize("K,N,r,rel,per,sym", FLAT)
def test_renyi2_flat_matches_the_enumeration(K, N, r, rel, per, sym):
    d = _flat(K, N, r, rel, per, sym)
    ip, Z = reference_self_ip_and_mass(d)
    h = entropy_exp_tens(d, method="renyi2", base=math.e, verbose=False)
    assert h == pytest.approx(-math.log(ip / Z ** 2), abs=_tol(d) * 10)


@pytest.mark.parametrize("tags,r,sym,rel,per,K", NESTED)
def test_renyi2_nested_matches_the_enumeration(tags, r, sym, rel, per, K):
    d = _nested(tags, r, sym, rel, per, K)
    ip, Z = reference_self_ip_and_mass(d)
    h = entropy_exp_tens(d, method="renyi2", base=math.e, verbose=False)
    assert h == pytest.approx(-math.log(ip / Z ** 2), abs=_tol(d) * 10)


def test_renyi2_ragged_events():
    rng = np.random.default_rng(4)
    p = np.sort(rng.uniform(0.0, P, size=(6, 3)), axis=0)
    p[5, 0] = np.nan
    p[4:, 2] = np.nan
    d = build_exp_tens([p], None, specs=[{"tags": T2, "r": [1, 2], "sym": [1, 1],
                                          "rel": [0, 0]}],
                       sigma=[0.7], is_per=[False], period=[0.0], verbose=False)
    ip, Z = reference_self_ip_and_mass(d)
    h = entropy_exp_tens(d, method="renyi2", base=math.e, verbose=False)
    assert h == pytest.approx(-math.log(ip / Z ** 2), rel=1e-9)


def test_renyi2_takes_the_inner_product_route():
    # A shape the nested contraction serves far more cheaply than the
    # enumeration: the entropy now runs whatever the selector picks.
    from mpt._tensor import cosine as C
    d = _nested(np.repeat(np.arange(4), 3), [2, 3], [1, 1], [0, 0], False, 12)
    entropy_exp_tens(d, method="renyi2", base=math.e, verbose=False)
    assert C._LAST_NESTED_ROUTES == ["contract"]
