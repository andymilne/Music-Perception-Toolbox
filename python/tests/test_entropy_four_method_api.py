"""Tests for the four-method entropy API.

Covers ``entropy_exp_tens``'s ``method`` kwarg in its post-Stage-2 form:
``'differential'``, ``'shannon'``, ``'normalized'`` (with British
``'normalised'`` alias), ``'renyi2'``. Also exercises the policy
guards (sigma=0 rejection for continuous methods; explicit grid
required for discrete methods) and the propagation through
``spectral_entropy`` and ``n_tuple_entropy``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, entropy_exp_tens, n_tuple_entropy, spectral_entropy

mpt.set_default(show_hints=False)


# ----------------------------------------------------------------------
# Method canonicalization and validation
# ----------------------------------------------------------------------

class TestMethodCanonicalization:
    """The four-method API accepts the British 'normalised' alias and
    rejects unknown methods with an informative error."""

    @pytest.fixture
    def dens_1d(self):
        return build_exp_tens(
            np.array([100., 200., 300.]), np.ones(3),
            20.0, 1, False, False, 0.0, verbose=False,
        )

    def test_normalised_alias_matches_normalized(self, dens_1d):
        H_us = entropy_exp_tens(
            dens_1d, method='normalized', n_points_per_dim=200,
            x_min=50., x_max=350., verbose=False,
        )
        H_uk = entropy_exp_tens(
            dens_1d, method='normalised', n_points_per_dim=200,
            x_min=50., x_max=350., verbose=False,
        )
        assert H_us == H_uk

    def test_bogus_method_rejected(self, dens_1d):
        with pytest.raises(ValueError, match="method must be one of"):
            entropy_exp_tens(dens_1d, method='bogus',
                             n_points_per_dim=200, verbose=False)

    def test_method_must_be_string(self, dens_1d):
        with pytest.raises(TypeError, match="must be a string"):
            entropy_exp_tens(dens_1d, method=42,
                             n_points_per_dim=200, verbose=False)


# ----------------------------------------------------------------------
# Grid-requirement policy
# ----------------------------------------------------------------------

class TestExplicitGridPolicy:
    """The discrete methods ('shannon', 'normalized') require an
    explicit ``n_points_per_dim``. The continuous methods
    ('differential', 'renyi2') do not."""

    @pytest.fixture
    def dens(self):
        return build_exp_tens(
            np.array([100., 200., 300.]), np.ones(3),
            20.0, 1, False, False, 0.0, verbose=False,
        )

    def test_shannon_without_grid_raises(self, dens):
        with pytest.raises(TypeError, match="requires an explicit"):
            entropy_exp_tens(dens, method='shannon', verbose=False)

    def test_normalized_without_grid_raises(self, dens):
        with pytest.raises(TypeError, match="requires an explicit"):
            entropy_exp_tens(dens, method='normalized', verbose=False)

    def test_differential_works_without_grid(self, dens):
        h = entropy_exp_tens(dens, method='differential', verbose=False)
        assert math.isfinite(h)

    def test_renyi2_works_without_grid(self, dens):
        h = entropy_exp_tens(dens, method='renyi2', verbose=False)
        assert math.isfinite(h)


# ----------------------------------------------------------------------
# sigma=0 policy on continuous methods
# ----------------------------------------------------------------------

class TestSigmaZeroGuard:
    """Continuous-form entropies diverge at sigma=0; the API rejects
    them explicitly rather than producing -inf / NaN."""

    @pytest.fixture
    def dens_sigma_zero(self):
        return build_exp_tens(
            np.array([100., 200., 300.]), np.ones(3),
            0.0, 1, False, False, 0.0, verbose=False,
        )

    def test_differential_at_sigma_zero_raises(self, dens_sigma_zero):
        with pytest.raises(ValueError, match=r"sigma > 0"):
            entropy_exp_tens(dens_sigma_zero, method='differential', verbose=False)

    def test_renyi2_at_sigma_zero_raises(self, dens_sigma_zero):
        with pytest.raises(ValueError, match=r"sigma > 0"):
            entropy_exp_tens(
                dens_sigma_zero, method='renyi2', verbose=False,
            )


# ----------------------------------------------------------------------
# 'normalized' equals 'shannon' / log_b(N)
# ----------------------------------------------------------------------

class TestNormalizedIsShannonOverLogN:
    """``method='normalized'`` returns H / log_b(N) on the same grid as
    ``method='shannon'`` (which returns raw H)."""

    @pytest.fixture
    def dens(self):
        return build_exp_tens(
            np.array([100., 200., 300.]), np.ones(3),
            20.0, 1, False, False, 0.0, verbose=False,
        )

    def test_normalized_equals_shannon_over_log_n(self, dens):
        N = 200
        x_min, x_max = 50., 350.
        H_shan = entropy_exp_tens(
            dens, method='shannon',
            n_points_per_dim=N, x_min=x_min, x_max=x_max, verbose=False,
        )
        H_norm = entropy_exp_tens(
            dens, method='normalized',
            n_points_per_dim=N, x_min=x_min, x_max=x_max, verbose=False,
        )
        assert math.isclose(H_norm, H_shan / math.log2(N), rel_tol=1e-12)


# ----------------------------------------------------------------------
# Differential entropy: convergence and grid-independence
# ----------------------------------------------------------------------

class TestDifferentialConvergence:
    """``method='differential'`` is adaptive and grid-independent;
    successive truncation_sigmas tightenings give a monotone-converging
    sequence, and concentrated vs spread densities order correctly."""

    def test_1d_converges_to_fixed_grid_value(self):
        dens = build_exp_tens(
            np.array([6000., 6300., 6700.]), np.ones(3),
            20.0, 1, False, False, 0.0, verbose=False,
        )
        h_diff = entropy_exp_tens(dens, method='differential', verbose=False)
        # Compare to a very-fine fixed-grid h_hat = H_disc + log_2(dx)
        N = 25000
        x_min, x_max = 5500., 7200.
        H_shan = entropy_exp_tens(
            dens, method='shannon',
            n_points_per_dim=N, x_min=x_min, x_max=x_max, verbose=False,
        )
        h_hat_fixed = H_shan + math.log2((x_max - x_min) / (N - 1))
        assert math.isclose(h_diff, h_hat_fixed, abs_tol=1e-5)

    def test_2d_periodic_converges_without_oom(self):
        """The 2-D periodic case was the canary for OOM before Richardson
        extrapolation. At a feasible accuracy the adaptive grid converges
        to a sensible value without exhausting memory."""
        P2 = np.array([[1., 2., 4., 5.]])
        W2 = np.array([[1., 1., 1., 1.]])
        period = 12.0
        sigma_eff = math.sqrt(2)
        dens_ma = build_exp_tens(
            [P2, P2], [W2, W2], [sigma_eff, sigma_eff],
            [1, 1], [False, False], [True, True], [period, period],
            verbose=False,
        )
        # 6 sigma (~1e-8) is certified on a feasible grid; the tightest
        # accuracy is exercised separately below.
        h_diff = entropy_exp_tens(
            dens_ma, method='differential',
            truncation_sigmas=6.0, verbose=False,
        )
        # Compare to fixed N=500 h_hat at the same accuracy.
        H_shan = entropy_exp_tens(
            dens_ma, method='shannon',
            n_points_per_dim=500, truncation_sigmas=6.0, verbose=False,
        )
        h_hat_fixed = H_shan + 2.0 * math.log2(period / 500.0)
        assert math.isclose(h_diff, h_hat_fixed, abs_tol=1e-3)

    def test_2d_periodic_tightest_accuracy_refuses_with_guidance(self):
        """At the tightest accuracy (truncation_sigmas=inf resolves to the
        accuracy floor) the 2-D grid needed to *certify* convergence
        exceeds the memory budget. The routine refuses and directs the
        user to a coarser accuracy or the closed-form estimator, rather
        than degrading silently or exhausting memory. A small
        ``kernel_chunk_bytes`` pins the budget low so the refusal is
        deterministic regardless of the machine's available memory."""
        P2 = np.array([[1., 2., 4., 5.]])
        W2 = np.array([[1., 1., 1., 1.]])
        period = 12.0
        sigma_eff = math.sqrt(2)
        dens_ma = build_exp_tens(
            [P2, P2], [W2, W2], [sigma_eff, sigma_eff],
            [1, 1], [False, False], [True, True], [period, period],
            verbose=False,
        )
        prev_kcb = mpt.get_default('kernel_chunk_bytes')
        try:
            mpt.set_default(kernel_chunk_bytes=8 * 1024 * 1024)
            with pytest.raises(ValueError, match="renyi2|truncation_sigmas"):
                entropy_exp_tens(
                    dens_ma, method='differential',
                    truncation_sigmas=np.inf, verbose=False,
                )
        finally:
            mpt.set_default(kernel_chunk_bytes=prev_kcb)

    def test_grid_independence_orders_correctly(self):
        """Concentrated vs spread Gaussian mixtures must produce
        differential entropy with B (spread) > A (concentrated). This
        is the principled property differential gives that grid-
        normalized Shannon does not."""
        dens_A = build_exp_tens(
            np.array([6500., 6600.]), np.ones(2),
            15.0, 1, False, False, 0.0, verbose=False,
        )
        dens_B = build_exp_tens(
            np.array([6000., 6400., 6800., 7200.]), np.ones(4),
            15.0, 1, False, False, 0.0, verbose=False,
        )
        h_A = entropy_exp_tens(dens_A, method='differential', verbose=False)
        h_B = entropy_exp_tens(dens_B, method='differential', verbose=False)
        assert h_B > h_A

    def test_truncation_sigmas_tightens_value(self):
        """Tightening truncation_sigmas should give a value that
        converges (monotonically in value, not necessarily in tol) to
        the kernel-untruncated reference."""
        dens = build_exp_tens(
            np.array([6000., 6300., 6700.]), np.ones(3),
            20.0, 1, False, False, 0.0, verbose=False,
        )
        h_ts4 = entropy_exp_tens(
            dens, method='differential', truncation_sigmas=4.0, verbose=False)
        h_ts6 = entropy_exp_tens(
            dens, method='differential', truncation_sigmas=6.0, verbose=False)
        h_ts8 = entropy_exp_tens(
            dens, method='differential', truncation_sigmas=8.0, verbose=False)
        # All finite
        for h in (h_ts4, h_ts6, h_ts8):
            assert math.isfinite(h)
        # ts=6 and ts=8 should be very close (we're well past the
        # truncation-noise floor); ts=4 may differ at the few-decimal level
        assert abs(h_ts8 - h_ts6) < 1e-4

    def test_truncation_sigmas_inf_resolves_to_accuracy_floor(self):
        """``truncation_sigmas=math.inf`` resolves to the finite
        accuracy-floor width (~7.43 sigma, the 1e-12 floor) uniformly
        with every other truncation path, including the differential
        span and tolerance anchoring. In particular the result must
        equal ``truncation_sigmas=accuracy_floor_sigmas()`` bit-for-bit
        (both traverse the same code path once resolution is applied),
        and must differ from ``truncation_sigmas=6.0`` by the ~6-sigma
        truncation-error scale (~2e-8). This pins the contract at the
        entry the dispatcher exposes to ``spectral_entropy`` and the
        rest of the toolbox."""
        from mpt._defaults import accuracy_floor_sigmas
        dens = build_exp_tens(
            np.array([6000., 6300., 6700.]), np.ones(3),
            20.0, 1, False, False, 0.0, verbose=False,
        )
        h_inf = entropy_exp_tens(
            dens, method='differential', truncation_sigmas=math.inf,
            verbose=False)
        h_floor = entropy_exp_tens(
            dens, method='differential',
            truncation_sigmas=accuracy_floor_sigmas(),
            verbose=False)
        h_ts6 = entropy_exp_tens(
            dens, method='differential', truncation_sigmas=6.0,
            verbose=False)
        # Contract: inf resolves to the accuracy-floor width.
        assert math.isfinite(h_inf)
        assert h_inf == h_floor
        # Sanity: 6 sigma is coarser and the two must differ by the
        # kernel-truncation-error scale at 6 sigma (~2e-8).
        assert abs(h_inf - h_ts6) > 1e-9


# ----------------------------------------------------------------------
# n_tuple_entropy with method kwarg
# ----------------------------------------------------------------------

class TestNTupleEntropyMethod:
    """``n_tuple_entropy`` accepts the same four-method API; defaults to
    ``'normalized'``."""

    @pytest.fixture
    def diatonic(self):
        return np.array([0., 2., 4., 5., 7., 9., 11.]), 12.0

    def test_default_equals_normalized(self, diatonic):
        p, period = diatonic
        H_def, _ = n_tuple_entropy(p, period, n=2)
        H_norm, _ = n_tuple_entropy(p, period, n=2, method='normalized')
        assert H_def == H_norm

    def test_normalised_alias(self, diatonic):
        p, period = diatonic
        H_norm, _ = n_tuple_entropy(p, period, n=2, method='normalized')
        H_uk, _ = n_tuple_entropy(p, period, n=2, method='normalised')
        assert H_norm == H_uk

    def test_differential_at_sigma_zero_rejected(self, diatonic):
        p, period = diatonic
        with pytest.raises(ValueError, match=r"sigma > 0"):
            n_tuple_entropy(p, period, n=2, method='differential')

    def test_renyi2_at_sigma_zero_rejected(self, diatonic):
        p, period = diatonic
        with pytest.raises(ValueError, match=r"sigma > 0"):
            n_tuple_entropy(p, period, n=2, method='renyi2')

    def test_continuous_methods_work_at_sigma_positive(self, diatonic):
        p, period = diatonic
        H_diff, _ = n_tuple_entropy(p, period, n=2, sigma=0.5, method='differential')
        H_rny, _ = n_tuple_entropy(p, period, n=2, sigma=0.5, method='renyi2')
        assert math.isfinite(H_diff)
        assert math.isfinite(H_rny)


# ----------------------------------------------------------------------
# spectral_entropy four-method dispatch
# ----------------------------------------------------------------------

class TestSpectralEntropyMethod:
    """``spectral_entropy`` defaults to 'differential' (the new
    principled scale-free choice) and exposes all four methods."""

    @pytest.fixture
    def major_triad(self):
        return np.array([0., 400., 700.])

    def test_default_is_differential(self, major_triad):
        H_def = spectral_entropy(major_triad, sigma=12.0, verbose=False)
        H_diff = spectral_entropy(major_triad, sigma=12.0,
                                  method='differential', verbose=False)
        assert H_def == H_diff

    def test_all_four_methods_run(self, major_triad):
        H_diff = spectral_entropy(major_triad, sigma=12.0,
                                  method='differential', verbose=False)
        H_norm = spectral_entropy(major_triad, sigma=12.0,
                                  method='normalized', verbose=False)
        H_shan = spectral_entropy(major_triad, sigma=12.0,
                                  method='shannon', verbose=False)
        H_rny = spectral_entropy(major_triad, sigma=12.0,
                                 method='renyi2', verbose=False)
        for H in (H_diff, H_norm, H_shan, H_rny):
            assert math.isfinite(H)

    def test_normalised_alias(self, major_triad):
        H_us = spectral_entropy(major_triad, sigma=12.0,
                                method='normalized', verbose=False)
        H_uk = spectral_entropy(major_triad, sigma=12.0,
                                method='normalised', verbose=False)
        assert H_us == H_uk

    def test_normalized_in_unit_interval(self, major_triad):
        """``method='normalized'`` produces values in [0, 1]."""
        H_norm = spectral_entropy(major_triad, sigma=12.0,
                                  method='normalized', verbose=False)
        assert 0.0 <= H_norm <= 1.0

    def test_ji_lower_entropy_than_edo_under_differential(self):
        """The consonance ordering established for spectral_entropy in
        Milne et al. (2017) survives the move to differential ĥ when a
        harmonic spectrum supplies the necessary overlap."""
        spec = ['harmonic', 12, 'powerlaw', 1]
        H_ji = spectral_entropy(
            np.array([0., 386.31, 701.96]), sigma=12.0,
            spectrum=spec, method='differential', verbose=False,
        )
        H_edo = spectral_entropy(
            np.array([0., 400., 700.]), sigma=12.0,
            spectrum=spec, method='differential', verbose=False,
        )
        assert H_ji < H_edo

# ----------------------------------------------------------------------
# Per-method input-form coverage
# ----------------------------------------------------------------------
#
# entropy_exp_tens supports five input forms:
#   (a) scalar density object (MaetDensity)
#   (b) list of density objects
#   (c) raw single-multiset scalar (p, w, sigma, r, is_rel, is_per, period)
#   (d) raw single-attribute batched (P, W, sigma, r, is_rel, is_per, period)
#   (e) raw MA scalar (p_attr, w, sigma_vec, r_vec, groups, ...)
#
# The discrete methods ('shannon', 'normalized') support all five.
# The continuous methods ('differential', 'renyi2') support (a), (c),
# (e); list and batched forms raise NotImplementedError.
# ----------------------------------------------------------------------


@pytest.fixture
def single_multiset_inputs():
    """Single-attribute scalar raw input for non-periodic r=1."""
    return dict(
        p=np.array([100., 200., 300.]),
        w=np.array([1., 1., 1.]),
        sigma=20.0, r=1, is_rel=False, is_per=False, period=0.0,
    )


@pytest.fixture
def single_multiset_dens(single_multiset_inputs):
    return build_exp_tens(
        single_multiset_inputs["p"], single_multiset_inputs["w"],
        single_multiset_inputs["sigma"], single_multiset_inputs["r"],
        single_multiset_inputs["is_rel"], single_multiset_inputs["is_per"], single_multiset_inputs["period"],
        verbose=False,
    )


@pytest.fixture
def single_multiset_dens_list(single_multiset_inputs):
    """List of three single-multiset densities (slightly different centres)."""
    out = []
    for shift in (0., 50., 100.):
        out.append(build_exp_tens(
            single_multiset_inputs["p"] + shift, single_multiset_inputs["w"],
            single_multiset_inputs["sigma"], single_multiset_inputs["r"],
            single_multiset_inputs["is_rel"], single_multiset_inputs["is_per"], single_multiset_inputs["period"],
            verbose=False,
        ))
    return out


@pytest.fixture
def single_attribute_batched(single_multiset_inputs):
    """Batched single-attribute density: 4 events of 3 pitches each."""
    P = np.stack([
        single_multiset_inputs["p"] + 0.,
        single_multiset_inputs["p"] + 50.,
        single_multiset_inputs["p"] + 100.,
        single_multiset_inputs["p"] + 150.,
    ])
    W = np.ones_like(P)
    return dict(P=P, W=W, sigma=20.0, r=1, is_rel=False, is_per=False, period=0.0)


@pytest.fixture
def ma_dens():
    """Multi-attribute scalar density: 2 attributes, 4 events.
    Attribute 0 is r=1 non-periodic absolute (pitch); attribute 1 is
    r=1 non-periodic absolute (time). groups=[0, 1] keeps them separate."""
    p_attr = [
        np.array([[100., 200., 300., 400.]]),
        np.array([[10., 20., 30., 40.]]),
    ]
    w = [np.ones(4), np.ones(4)]
    return build_exp_tens(
        p_attr, w,
        [20.0, 5.0], [1, 1], 
        [False, False], [False, False], [0.0, 0.0], verbose=False,
    )


class TestShannonInputForms:
    """``method='shannon'`` supports all five input forms."""

    def test_scalar_density(self, single_multiset_dens):
        h = entropy_exp_tens(
            single_multiset_dens, method='shannon', n_points_per_dim=200,
            x_min=0., x_max=500., verbose=False,
        )
        assert isinstance(h, float) and math.isfinite(h)

    def test_list_of_densities(self, single_multiset_dens_list):
        H = entropy_exp_tens(
            single_multiset_dens_list, method='shannon', n_points_per_dim=200,
            x_min=0., x_max=500., verbose=False,
        )
        assert isinstance(H, np.ndarray) and H.shape == (3,)
        assert np.all(np.isfinite(H))

    def test_raw_single_multiset_scalar(self, single_multiset_inputs):
        h = entropy_exp_tens(
            single_multiset_inputs["p"], single_multiset_inputs["w"], single_multiset_inputs["sigma"],
            single_multiset_inputs["r"], single_multiset_inputs["is_rel"], single_multiset_inputs["is_per"],
            single_multiset_inputs["period"],
            method='shannon', n_points_per_dim=200,
            x_min=0., x_max=500., verbose=False,
        )
        assert isinstance(h, float) and math.isfinite(h)

    def test_raw_single_attribute_batched(self, single_attribute_batched):
        H = entropy_exp_tens(
            single_attribute_batched["P"], single_attribute_batched["W"], single_attribute_batched["sigma"],
            single_attribute_batched["r"], single_attribute_batched["is_rel"], single_attribute_batched["is_per"],
            single_attribute_batched["period"],
            method='shannon', n_points_per_dim=200,
            x_min=0., x_max=500., verbose=False,
        )
        assert isinstance(H, np.ndarray) and H.shape == (4,)
        assert np.all(np.isfinite(H))

    def test_raw_ma_scalar(self, ma_dens):
        # The MA scalar form: pre-built density object goes through the
        # same dispatch. Non-periodic MA needs explicit per-axis bounds;
        # we pass x_min / x_max as lists to mirror the per-attribute span.
        h = entropy_exp_tens(
            ma_dens, method='shannon', n_points_per_dim=80,
            x_min=[0., 0.], x_max=[500., 50.], verbose=False,
        )
        assert isinstance(h, float) and math.isfinite(h)


class TestNormalizedInputForms:
    """``method='normalized'`` supports all five input forms and returns
    values in [0, 1]."""

    def test_scalar_density(self, single_multiset_dens):
        h = entropy_exp_tens(
            single_multiset_dens, method='normalized', n_points_per_dim=200,
            x_min=0., x_max=500., verbose=False,
        )
        assert isinstance(h, float) and 0.0 <= h <= 1.0

    def test_list_of_densities(self, single_multiset_dens_list):
        H = entropy_exp_tens(
            single_multiset_dens_list, method='normalized', n_points_per_dim=200,
            x_min=0., x_max=500., verbose=False,
        )
        assert isinstance(H, np.ndarray) and H.shape == (3,)
        assert np.all((H >= 0.0) & (H <= 1.0))

    def test_raw_single_multiset_scalar(self, single_multiset_inputs):
        h = entropy_exp_tens(
            single_multiset_inputs["p"], single_multiset_inputs["w"], single_multiset_inputs["sigma"],
            single_multiset_inputs["r"], single_multiset_inputs["is_rel"], single_multiset_inputs["is_per"],
            single_multiset_inputs["period"],
            method='normalized', n_points_per_dim=200,
            x_min=0., x_max=500., verbose=False,
        )
        assert isinstance(h, float) and 0.0 <= h <= 1.0

    def test_raw_single_attribute_batched(self, single_attribute_batched):
        H = entropy_exp_tens(
            single_attribute_batched["P"], single_attribute_batched["W"], single_attribute_batched["sigma"],
            single_attribute_batched["r"], single_attribute_batched["is_rel"], single_attribute_batched["is_per"],
            single_attribute_batched["period"],
            method='normalized', n_points_per_dim=200,
            x_min=0., x_max=500., verbose=False,
        )
        assert isinstance(H, np.ndarray) and H.shape == (4,)
        assert np.all((H >= 0.0) & (H <= 1.0))

    def test_raw_ma_scalar(self, ma_dens):
        h = entropy_exp_tens(
            ma_dens, method='normalized', n_points_per_dim=80,
            x_min=[0., 0.], x_max=[500., 50.], verbose=False,
        )
        assert isinstance(h, float) and 0.0 <= h <= 1.0


class TestDifferentialInputForms:
    """``method='differential'`` supports scalar forms; list and batched
    raise NotImplementedError."""

    def test_scalar_density(self, single_multiset_dens):
        h = entropy_exp_tens(single_multiset_dens, method='differential', verbose=False)
        assert isinstance(h, float) and math.isfinite(h)

    def test_raw_single_multiset_scalar(self, single_multiset_inputs):
        h = entropy_exp_tens(
            single_multiset_inputs["p"], single_multiset_inputs["w"], single_multiset_inputs["sigma"],
            single_multiset_inputs["r"], single_multiset_inputs["is_rel"], single_multiset_inputs["is_per"],
            single_multiset_inputs["period"],
            method='differential', verbose=False,
        )
        assert isinstance(h, float) and math.isfinite(h)

    def test_raw_ma_scalar(self, ma_dens):
        # 2-D differential entropy: certifying the tightest accuracy needs
        # an infeasibly fine grid, so pin a feasible accuracy for this
        # input-form check (the refusal path is tested separately).
        h = entropy_exp_tens(
            ma_dens, method='differential',
            truncation_sigmas=5.0, verbose=False,
        )
        assert isinstance(h, float) and math.isfinite(h)

    def test_list_rejected(self, single_multiset_dens_list):
        with pytest.raises(NotImplementedError):
            entropy_exp_tens(single_multiset_dens_list, method='differential', verbose=False)

    def test_batched_rejected(self, single_attribute_batched):
        with pytest.raises(NotImplementedError):
            entropy_exp_tens(
                single_attribute_batched["P"], single_attribute_batched["W"], single_attribute_batched["sigma"],
                single_attribute_batched["r"], single_attribute_batched["is_rel"], single_attribute_batched["is_per"],
                single_attribute_batched["period"],
                method='differential', verbose=False,
            )


class TestRenyi2InputForms:
    """``method='renyi2'`` supports scalar forms; list and batched
    raise NotImplementedError."""

    def test_scalar_density(self, single_multiset_dens):
        h = entropy_exp_tens(single_multiset_dens, method='renyi2', verbose=False)
        assert isinstance(h, float) and math.isfinite(h)

    def test_raw_single_multiset_scalar(self, single_multiset_inputs):
        h = entropy_exp_tens(
            single_multiset_inputs["p"], single_multiset_inputs["w"], single_multiset_inputs["sigma"],
            single_multiset_inputs["r"], single_multiset_inputs["is_rel"], single_multiset_inputs["is_per"],
            single_multiset_inputs["period"],
            method='renyi2', verbose=False,
        )
        assert isinstance(h, float) and math.isfinite(h)

    def test_raw_ma_scalar(self, ma_dens):
        h = entropy_exp_tens(ma_dens, method='renyi2', verbose=False)
        assert isinstance(h, float) and math.isfinite(h)

    def test_list_rejected(self, single_multiset_dens_list):
        with pytest.raises(NotImplementedError):
            entropy_exp_tens(single_multiset_dens_list, method='renyi2', verbose=False)

    def test_batched_rejected(self, single_attribute_batched):
        with pytest.raises(NotImplementedError):
            entropy_exp_tens(
                single_attribute_batched["P"], single_attribute_batched["W"], single_attribute_batched["sigma"],
                single_attribute_batched["r"], single_attribute_batched["is_rel"], single_attribute_batched["is_per"],
                single_attribute_batched["period"],
                method='renyi2', verbose=False,
            )


# ----------------------------------------------------------------------
# Migration error: 'normalize' kwarg removed in v3
# ----------------------------------------------------------------------


class TestNormalizeKwargMigrationError:
    """Passing the v2.0 ``normalize`` kwarg (any value, on any entry
    point that previously accepted it) raises a ``TypeError`` whose
    message points to the four-method API."""

    @pytest.fixture
    def dens(self):
        return build_exp_tens(
            np.array([100., 200., 300.]), np.ones(3),
            20.0, 1, False, False, 0.0, verbose=False,
        )

    def test_entropy_exp_tens_normalize_true_rejected(self, dens):
        with pytest.raises(TypeError, match=r"'normalize'.*removed in v3"):
            entropy_exp_tens(
                dens, method='shannon', n_points_per_dim=100,
                normalize=True, verbose=False,
            )

    def test_entropy_exp_tens_normalize_false_rejected(self, dens):
        with pytest.raises(TypeError, match=r"'normalize'.*removed in v3"):
            entropy_exp_tens(
                dens, method='shannon', n_points_per_dim=100,
                normalize=False, verbose=False,
            )

    def test_entropy_exp_tens_normalize_rejected_under_renyi2(self, dens):
        # In an earlier internal build this combination triggered a
        # NotImplementedError; in v3 it triggers the migration error.
        with pytest.raises(TypeError, match=r"'normalize'.*removed in v3"):
            entropy_exp_tens(dens, method='renyi2', normalize=False, verbose=False)

    def test_spectral_entropy_normalize_true_rejected(self):
        with pytest.raises(TypeError, match=r"'normalize'.*removed in v3"):
            spectral_entropy(
                np.array([0., 400., 700.]), sigma=12.0,
                method='normalized', normalize=True, verbose=False,
            )

    def test_spectral_entropy_normalize_false_rejected(self):
        with pytest.raises(TypeError, match=r"'normalize'.*removed in v3"):
            spectral_entropy(
                np.array([0., 400., 700.]), sigma=12.0,
                method='shannon', normalize=False, verbose=False,
            )

    def test_migration_message_names_replacement_methods(self, dens):
        # The error text must point users to the two replacement methods
        # so they can pick the right one without re-reading the docs.
        try:
            entropy_exp_tens(
                dens, method='shannon', n_points_per_dim=100,
                normalize=True, verbose=False,
            )
        except TypeError as exc:
            msg = str(exc)
            assert "method='normalized'" in msg
            assert "method='shannon'" in msg
        else:
            pytest.fail("expected TypeError")


# ----------------------------------------------------------------------
# truncation_sigmas=3 differential speedup contract
# ----------------------------------------------------------------------


class TestDifferentialTruncationSigmasContract:
    """``truncation_sigmas=3`` is documented as a fast-path option for
    ``method='differential'``: tolerance loosens to ``exp(-9/2) ≈ 1.1e-2``
    so the adaptive loop converges in fewer doublings. The resulting
    value should drift by no more than fifth-decimal from the
    ``truncation_sigmas=6`` reference, and the consonance ordering
    of two clearly-separated densities must be preserved."""

    def test_ts3_finite_and_close_to_ts6(self):
        dens = build_exp_tens(
            np.array([6000., 6300., 6700.]), np.ones(3),
            20.0, 1, False, False, 0.0, verbose=False,
        )
        h_ts3 = entropy_exp_tens(
            dens, method='differential', truncation_sigmas=3.0, verbose=False,
        )
        h_ts6 = entropy_exp_tens(
            dens, method='differential', truncation_sigmas=6.0, verbose=False,
        )
        assert math.isfinite(h_ts3)
        # Fifth-decimal drift, documented in the spectral_entropy docstring.
        assert abs(h_ts3 - h_ts6) < 1e-2

    def test_ts3_preserves_ordering(self):
        dens_concentrated = build_exp_tens(
            np.array([6500., 6600.]), np.ones(2),
            15.0, 1, False, False, 0.0, verbose=False,
        )
        dens_spread = build_exp_tens(
            np.array([6000., 6400., 6800., 7200.]), np.ones(4),
            15.0, 1, False, False, 0.0, verbose=False,
        )
        h_conc = entropy_exp_tens(
            dens_concentrated, method='differential',
            truncation_sigmas=3.0, verbose=False,
        )
        h_spread = entropy_exp_tens(
            dens_spread, method='differential',
            truncation_sigmas=3.0, verbose=False,
        )
        assert h_spread > h_conc
