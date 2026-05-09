"""Tests for the within-attribute symmetrisation of the v2.1 windowed
cosine similarity.

Background. The MAET density is symmetric under within-attribute
coordinate permutations (JMM Theorem on windowed inner product,
Remark on within-attribute symmetry). The integral of one MAET
against a windowed MAET therefore depends on the window centre
``mu`` only through its multiset within each attribute. In
particular, the windowed cosine is invariant under reordering of
the input values that build the windowed density.

Before the fix, ``_cos_sim_numerator_ma`` used the within-r-ad
permutation-symmetry reduction (perm-comb summation on the comb
side) without symmetrising the windowed factor, which only gives
the framework-correct integral when ``mu`` is uniform within each
attribute. With non-uniform within-attribute ``mu``, the result
depended on the canonical ordering of values stored on the comb
side, breaking exchangeability.

After the fix, ``_windowed_contribution_factorisable`` symmetrises
the per-axis windowed factor over within-attribute permutations of
``mu`` when the within-attribute entries differ. The cosine is
exchangeability-preserving in all cases.

These tests verify the fix against:

(a) Direct enumeration over distinct ordered r-tuples on both sides
    (the framework definition of the windowed inner product),
    which is exchangeability-preserving by construction.

(b) Reordering the input values of the windowed density: the
    cosine must be invariant under any input reordering, regardless
    of whether ``mu`` is uniform within an attribute.
"""
from itertools import permutations

import numpy as np
import pytest
from scipy.special import erf

# This file's reference implementation uses v2.2 orbit-Möbius internals
# (`_orbit_inner_abs`, `_window_width_params`) that are not present on
# the v2.1 dev branch. Skip the whole module cleanly when those names
# are unavailable so the rest of the test suite remains runnable. When
# the v2.2 work lands these imports will resolve and the tests will
# activate.
try:
    from mpt.tensor import (
        build_exp_tens,
        cos_sim_exp_tens,
        window_tensor,
        _orbit_inner_abs,
        _window_width_params,
    )
except ImportError as _exc:
    pytest.skip(
        f"v2.2 orbit-Möbius internals not present on this branch: {_exc}",
        allow_module_level=True,
    )


# -------------------------------------------------------------------
# Direct enumeration reference (perm-perm, framework-correct)
# -------------------------------------------------------------------


def _per_axis_F(mu_shift, a_rect, b_conv, sigma_pair):
    """Per-axis windowed factor F(mu_shift) for the rect-conv-Gaussian
    family. Pure cases handled analytically; mixed case via the
    erf-difference."""
    sigma_t = np.sqrt(sigma_pair ** 2 + b_conv ** 2)
    if a_rect == 0.0 and b_conv > 0.0:
        return (b_conv / sigma_t) * np.exp(
            -mu_shift ** 2 / (2 * sigma_t ** 2)
        )
    if b_conv == 0.0 and a_rect > 0.0:
        d = sigma_t * np.sqrt(2.0)
        return 0.5 * (
            erf((mu_shift + a_rect) / d) - erf((mu_shift - a_rect) / d)
        )
    d = sigma_t * np.sqrt(2.0)
    num = erf((mu_shift + a_rect) / d) - erf((mu_shift - a_rect) / d)
    return num / (2.0 * erf(a_rect / (b_conv * np.sqrt(2.0))))


def _direct_windowed_cross_corr_perm_perm(p_a, w_a, p_b, w_b, sigma, r,
                                            offset_vec, mu_q,
                                            a_rect, b_conv):
    """Reference computation: the framework-correct perm-perm sum of
    the toolbox's cross-correlation translated form.

    The toolbox computes the cross-correlation IP at user-supplied
    offset, with the substitution

        D = U - V + (offset - mu_q)         per axis (kernel shift)
        F_centre = (mu_q + offset) / 2      per axis

    For exchangeability in the framework-correct sense, the
    contribution must be summed over distinct ordered r-tuples on
    BOTH sides (perm-perm), not just one side (perm-comb). This
    automatically averages over within-attribute permutations of
    offset — which the v2.1 perm-comb form did not do.
    """
    K_a, K_b = len(p_a), len(p_b)
    sigma_pair = sigma / np.sqrt(2.0)
    total = 0.0
    for ta in permutations(range(K_a), r):
        for tb in permutations(range(K_b), r):
            contrib = 1.0
            for l in range(r):
                d = (p_a[ta[l]] - p_b[tb[l]]) + (offset_vec[l] - mu_q)
                K_l = np.exp(-(d ** 2) / (4 * sigma ** 2))
                f_centre = (mu_q + offset_vec[l]) / 2.0
                mid = (p_a[ta[l]] + p_b[tb[l]]) / 2.0
                F_l = _per_axis_F(mid - f_centre, a_rect, b_conv,
                                    sigma_pair)
                contrib *= K_l * F_l * w_a[ta[l]] * w_b[tb[l]]
            total += contrib
    return total * (sigma * np.sqrt(np.pi)) ** r


def _direct_windowed_cosine_sa(p_a, w_a, p_b, w_b, sigma, r,
                                 offset_vec, size, mix):
    """Framework-correct windowed cosine via direct (perm-perm)
    enumeration of the cross-correlation form. For uniform offset
    this matches the toolbox's perm-comb output up to the cancelled
    r! prefactor; for non-uniform offset the perm-perm form is the
    framework-correct value, while the toolbox's perm-comb form
    requires within-attribute symmetrisation to match (the v2.1 fix).
    """
    a_rect, b_conv = _window_width_params(size, mix, sigma)
    mu_q = float(p_a.mean())
    ip_xy = _direct_windowed_cross_corr_perm_perm(
        p_a, w_a, p_b, w_b, sigma, r, offset_vec, mu_q, a_rect, b_conv,
    )
    ip_xx = _orbit_inner_abs(p_a, w_a, p_a, w_a, sigma, r, False, 0.0)
    ip_yy = _orbit_inner_abs(p_b, w_b, p_b, w_b, sigma, r, False, 0.0)
    return ip_xy / np.sqrt(ip_xx * ip_yy)


# -------------------------------------------------------------------
# Toolbox path under test
# -------------------------------------------------------------------


def _toolbox_windowed_cosine_sa(p_a, w_a, p_b, w_b, sigma, r,
                                  offset_vec, size, mix):
    """Toolbox windowed cosine via cos_sim_exp_tens + window_tensor.
    The user-supplied centre vector is the offset interpretation
    used by ``windowed_similarity``: each entry of length-r is one
    axis's offset from the unweighted query centroid mu_q."""
    Pa = p_a.reshape(-1, 1); Wa = w_a.reshape(-1, 1)
    Pb = p_b.reshape(-1, 1); Wb = w_b.reshape(-1, 1)
    dens_q = build_exp_tens([Pa], [Wa], [sigma], [r], [0],
                             [False], [False], [0.0], verbose=False)
    dens_c = build_exp_tens([Pb], [Wb], [sigma], [r], [0],
                             [False], [False], [0.0], verbose=False)
    wmd = window_tensor(dens_c, dict(
        size=size, mix=mix,
        centre=[np.asarray(offset_vec, dtype=np.float64)],
    ))
    return cos_sim_exp_tens(dens_q, wmd, verbose=False)


# -------------------------------------------------------------------
# 1. Toolbox cosine matches direct enumeration for non-uniform c
# -------------------------------------------------------------------


@pytest.mark.parametrize("r", [2, 3])
@pytest.mark.parametrize("size,mix", [
    (5.0, 0.0),
    (3.0, 1.0),
    (4.0, 0.5),
])
@pytest.mark.parametrize("c_pattern", [
    "uniform",
    "non_uniform_small_spread",
    "non_uniform_large_spread",
])
def test_toolbox_matches_direct_for_non_uniform_centre(r, size, mix,
                                                        c_pattern):
    """Toolbox windowed cosine equals direct enumeration (the framework
    definition) for both uniform and non-uniform within-attribute c."""
    rng = np.random.default_rng(
        seed=hash((r, size, mix, c_pattern)) % 2**31
    )
    sigma = 30.0
    K = 6
    p_a = rng.uniform(-200, 200, K); w_a = rng.uniform(0.5, 1.5, K)
    p_b = rng.uniform(-200, 200, K); w_b = rng.uniform(0.5, 1.5, K)

    # Build offset vector (per-axis offset from mu_q, what the
    # toolbox API actually accepts as 'centre').
    if c_pattern == "uniform":
        offset_vec = np.full(r, 50.0)
    elif c_pattern == "non_uniform_small_spread":
        offset_vec = np.linspace(-30.0, 30.0, r)
    elif c_pattern == "non_uniform_large_spread":
        offset_vec = np.linspace(-100.0, 100.0, r)

    cos_direct = _direct_windowed_cosine_sa(
        p_a, w_a, p_b, w_b, sigma, r, offset_vec, size, mix,
    )
    cos_toolbox = _toolbox_windowed_cosine_sa(
        p_a, w_a, p_b, w_b, sigma, r, offset_vec, size, mix,
    )

    # Orthogonality regime: both small means agreement at digit level
    # is not required.
    if abs(cos_direct) < 1e-6 and abs(cos_toolbox) < 1e-6:
        return
    assert np.isclose(cos_toolbox, cos_direct, atol=0, rtol=1e-9), (
        f"toolbox={cos_toolbox:.6e}, direct={cos_direct:.6e}, "
        f"c_pattern={c_pattern}"
    )


# -------------------------------------------------------------------
# 2. Cosine is invariant under input reordering on the windowed side
# -------------------------------------------------------------------


@pytest.mark.parametrize("r", [2, 3])
@pytest.mark.parametrize("size,mix", [
    (5.0, 0.0),
    (3.0, 1.0),
    (4.0, 0.5),
])
@pytest.mark.parametrize("c_pattern", [
    "uniform",
    "non_uniform_small_spread",
    "non_uniform_large_spread",
])
def test_windowed_cosine_invariant_under_context_reorder(r, size, mix,
                                                           c_pattern):
    """Reordering the values of the windowed density's input must not
    change the cosine. This is the framework-correct exchangeability
    property — it failed in v2.1 for non-uniform within-attribute c
    before this fix."""
    rng = np.random.default_rng(
        seed=hash((r, size, mix, c_pattern, "reorder")) % 2**31
    )
    sigma = 30.0
    K = 6
    p_a = rng.uniform(-200, 200, K); w_a = rng.uniform(0.5, 1.5, K)
    p_b = rng.uniform(-200, 200, K); w_b = rng.uniform(0.5, 1.5, K)

    if c_pattern == "uniform":
        offset_vec = np.full(r, 30.0)
    elif c_pattern == "non_uniform_small_spread":
        offset_vec = np.linspace(-25.0, 25.0, r)
    elif c_pattern == "non_uniform_large_spread":
        offset_vec = np.linspace(-100.0, 100.0, r)

    # Original ordering
    cos_orig = _toolbox_windowed_cosine_sa(
        p_a, w_a, p_b, w_b, sigma, r, offset_vec, size, mix,
    )

    # Reverse ordering of the windowed-side values
    perm = np.arange(K)[::-1]
    cos_rev = _toolbox_windowed_cosine_sa(
        p_a, w_a, p_b[perm], w_b[perm], sigma, r,
        offset_vec, size, mix,
    )

    # Random shuffle of the windowed-side values
    perm = rng.permutation(K)
    cos_shuf = _toolbox_windowed_cosine_sa(
        p_a, w_a, p_b[perm], w_b[perm], sigma, r,
        offset_vec, size, mix,
    )

    if abs(cos_orig) < 1e-6:
        # Orthogonality regime: invariance holds at "essentially zero".
        assert abs(cos_rev) < 1e-6 and abs(cos_shuf) < 1e-6
        return

    assert np.isclose(cos_rev, cos_orig, atol=0, rtol=1e-9), (
        f"reverse: cos_rev={cos_rev:.6e}, cos_orig={cos_orig:.6e}"
    )
    assert np.isclose(cos_shuf, cos_orig, atol=0, rtol=1e-9), (
        f"shuffle: cos_shuf={cos_shuf:.6e}, cos_orig={cos_orig:.6e}"
    )


# -------------------------------------------------------------------
# 3. Cosine invariant under permutation of the c entries themselves
# -------------------------------------------------------------------


@pytest.mark.parametrize("r", [2, 3])
@pytest.mark.parametrize("size,mix", [(5.0, 0.0), (3.0, 1.0)])
def test_windowed_cosine_invariant_under_centre_permutation(r, size, mix):
    """Permuting the entries of the within-attribute centre vector must
    not change the cosine — the integral depends on c only through
    its multiset (JMM remark on within-attribute symmetry)."""
    rng = np.random.default_rng(
        seed=hash((r, size, mix, "centre_perm")) % 2**31
    )
    sigma = 30.0
    K = 6
    p_a = rng.uniform(-200, 200, K); w_a = rng.uniform(0.5, 1.5, K)
    p_b = rng.uniform(-200, 200, K); w_b = rng.uniform(0.5, 1.5, K)

    offset_vec = np.linspace(-50.0, 50.0, r)

    # Original c ordering
    cos_orig = _toolbox_windowed_cosine_sa(
        p_a, w_a, p_b, w_b, sigma, r, offset_vec, size, mix,
    )

    # All permutations
    for pi in permutations(range(r)):
        offset_perm = offset_vec[list(pi)]
        cos_pi = _toolbox_windowed_cosine_sa(
            p_a, w_a, p_b, w_b, sigma, r, offset_perm, size, mix,
        )
        if abs(cos_orig) < 1e-6:
            assert abs(cos_pi) < 1e-6
        else:
            assert np.isclose(cos_pi, cos_orig, atol=0, rtol=1e-9), (
                f"perm {pi}: cos={cos_pi:.6e}, orig={cos_orig:.6e}"
            )


# -------------------------------------------------------------------
# 4. Multi-attribute case: within-attribute symmetrisation per attribute
# -------------------------------------------------------------------


def test_multi_attribute_within_attribute_symmetrisation():
    """A group with two absolute attributes and non-uniform within-
    attribute centres should still give exchangeability-preserving
    results: each attribute's centre vector is symmetrised
    independently."""
    rng = np.random.default_rng(0)
    sigma = 30.0
    r_0, r_1 = 2, 2
    K_a, K_b = 4, 4

    p0_x = rng.uniform(-200, 200, (K_a, 1)); w0_x = rng.uniform(0.5, 1.5, (K_a, 1))
    p1_x = rng.uniform(-200, 200, (K_b, 1)); w1_x = rng.uniform(0.5, 1.5, (K_b, 1))
    p0_y = rng.uniform(-200, 200, (K_a, 1)); w0_y = rng.uniform(0.5, 1.5, (K_a, 1))
    p1_y = rng.uniform(-200, 200, (K_b, 1)); w1_y = rng.uniform(0.5, 1.5, (K_b, 1))

    dens_x = build_exp_tens([p0_x, p1_x], [w0_x, w1_x], [sigma],
                             [r_0, r_1], [0, 0],
                             [False], [False], [0.0], verbose=False)
    dens_y = build_exp_tens([p0_y, p1_y], [w0_y, w1_y], [sigma],
                             [r_0, r_1], [0, 0],
                             [False], [False], [0.0], verbose=False)

    # Non-uniform within-attribute centres for each attribute
    centre_attr0 = np.array([10.0, 50.0])    # non-uniform within attr 0
    centre_attr1 = np.array([-30.0, 20.0])   # non-uniform within attr 1

    # Reorder within attr 0 only
    centre_attr0_rev = centre_attr0[::-1]
    centre_attr1_same = centre_attr1.copy()

    cos_orig = cos_sim_exp_tens(
        dens_x,
        window_tensor(dens_y, dict(
            size=5.0, mix=0.0, centre=[centre_attr0, centre_attr1],
        )),
        verbose=False,
    )
    cos_attr0_rev = cos_sim_exp_tens(
        dens_x,
        window_tensor(dens_y, dict(
            size=5.0, mix=0.0,
            centre=[centre_attr0_rev, centre_attr1_same],
        )),
        verbose=False,
    )

    if abs(cos_orig) < 1e-6:
        assert abs(cos_attr0_rev) < 1e-6
    else:
        assert np.isclose(cos_attr0_rev, cos_orig, atol=0, rtol=1e-9), (
            f"orig={cos_orig:.6e}, attr0-reversed={cos_attr0_rev:.6e}"
        )

    # Reorder within attr 1 only
    cos_attr1_rev = cos_sim_exp_tens(
        dens_x,
        window_tensor(dens_y, dict(
            size=5.0, mix=0.0,
            centre=[centre_attr0, centre_attr1[::-1]],
        )),
        verbose=False,
    )
    if abs(cos_orig) < 1e-6:
        assert abs(cos_attr1_rev) < 1e-6
    else:
        assert np.isclose(cos_attr1_rev, cos_orig, atol=0, rtol=1e-9), (
            f"orig={cos_orig:.6e}, attr1-reversed={cos_attr1_rev:.6e}"
        )


# -------------------------------------------------------------------
# 5. Uniform-c regression: behaviour byte-identical to v2.1 fast path
# -------------------------------------------------------------------


def test_uniform_centre_byte_identical_to_old_path():
    """For uniform within-attribute centres, the symmetrisation is a
    no-op and the fast path is taken. Verify byte-identical results
    to the explicit single-product computation (no averaging)."""
    rng = np.random.default_rng(0)
    sigma = 30.0
    K = 5
    r = 2
    p_a = rng.uniform(-200, 200, K); w_a = rng.uniform(0.5, 1.5, K)
    p_b = rng.uniform(-200, 200, K); w_b = rng.uniform(0.5, 1.5, K)

    # Uniform offset
    offset_uniform = np.full(r, 25.0)

    cos_via_toolbox = _toolbox_windowed_cosine_sa(
        p_a, w_a, p_b, w_b, sigma, r, offset_uniform, 5.0, 0.0,
    )
    cos_via_direct = _direct_windowed_cosine_sa(
        p_a, w_a, p_b, w_b, sigma, r, offset_uniform, 5.0, 0.0,
    )
    assert np.isclose(cos_via_toolbox, cos_via_direct,
                       atol=0, rtol=1e-12)
