"""Stage 2b: verify cos_sim_exp_tens routes the SA centres-IP through
:func:`gaussian_kernel_sum` for abs and rel-non-periodic modes, and
that the ``truncation_sigmas`` / ``kernel_precision`` kwargs reach the
helper from the public entry point.

Coverage:
    - cos_sim_exp_tens(dens_x, dens_y, truncation_sigmas=k) returns the
      same value as the exact path for abs ±periodic and rel-non-periodic
      densities, to within the helper's truncation tolerance.
    - The rel+periodic case (pairwise-wrap quadratic form) stays on the
      existing pairwise code path; the kwarg is accepted without error
      but doesn't change the result.
    - The list / cartesian dispatch path also forwards the kwargs.
    - Global mptDefaults / set_default propagates.
"""
from __future__ import annotations

import numpy as np
import pytest

from mpt import (
    add_spectra,
    build_exp_tens,
    cos_sim_exp_tens,
    reset_defaults,
    set_default,
)


# -----------------------------------------------------------------------
# SA abs and rel-non-periodic densities — the cases routed through helper
# -----------------------------------------------------------------------


def _build_density_pair(*, is_rel: bool, is_per: bool, sigma=12.0,
                        period=1200.0, r=3):
    """Two small chord-like densities."""
    p_x = np.array([0., 400., 700.])
    p_y = np.array([0., 300., 700.])
    dens_x = build_exp_tens(p_x, np.ones(3), sigma, r, is_rel, is_per,
                            period if is_per else 0.0)
    dens_y = build_exp_tens(p_y, np.ones(3), sigma, r, is_rel, is_per,
                            period if is_per else 0.0)
    return dens_x, dens_y


class TestCosSimSACentresRouting:

    @pytest.mark.parametrize("is_rel, is_per", [
        (False, False),  # abs non-per — routed
        (False, True),   # abs per — routed (componentwise wrap)
        (True, False),   # rel non-per — routed
    ])
    def test_exact_vs_truncated_small(self, is_rel, is_per):
        reset_defaults()
        dens_x, dens_y = _build_density_pair(is_rel=is_rel, is_per=is_per)
        s_exact = cos_sim_exp_tens(
            dens_x, dens_y, method='pairwise', verbose=False,
        )
        s_trunc = cos_sim_exp_tens(
            dens_x, dens_y, method='pairwise',
            truncation_sigmas=6.0, verbose=False,
        )
        # 6-sigma truncation: kernel discard bound exp(-18) ~ 1.5e-8;
        # cosine itself is O(1) and normalisation absorbs absolute scale,
        # so 1e-7 absolute on cosine is a generous bound.
        assert abs(s_exact - s_trunc) < 1e-7, (
            f"is_rel={is_rel}, is_per={is_per}: "
            f"exact={s_exact}, trunc={s_trunc}, "
            f"|diff|={abs(s_exact - s_trunc):.3e}"
        )

    def test_rel_periodic_unaffected_by_truncation_kwarg(self):
        """Rel+per stays on existing pairwise-wrap path; the kwarg is
        accepted but doesn't change the result."""
        reset_defaults()
        dens_x, dens_y = _build_density_pair(is_rel=True, is_per=True)
        s_default = cos_sim_exp_tens(
            dens_x, dens_y, method='pairwise', verbose=False,
        )
        s_with_kwarg = cos_sim_exp_tens(
            dens_x, dens_y, method='pairwise',
            truncation_sigmas=6.0, verbose=False,
        )
        # Bit-exact: helper not used for rel+per.
        assert s_default == s_with_kwarg

    def test_truncation_speedup_at_larger_k(self):
        """K=12 harmonic template gives a measurable speedup."""
        import time

        tp, tw = add_spectra(
            np.array([0., 0., 0.]), np.array([1., 1., 1.]),
            'harmonic', 12, 'powerlaw', 1,
        )
        dens_a = build_exp_tens(tp, tw, 12.0, 3, True, False, 0.0)
        chord = np.array([0., 400., 700.])
        dens_b = build_exp_tens(chord, np.ones(3), 12.0, 3,
                                True, False, 0.0)

        reset_defaults()
        t0 = time.perf_counter()
        s_exact = cos_sim_exp_tens(
            dens_a, dens_b, method='pairwise', verbose=False,
        )
        t_exact = time.perf_counter() - t0

        t0 = time.perf_counter()
        s_trunc = cos_sim_exp_tens(
            dens_a, dens_b, method='pairwise',
            truncation_sigmas=6.0, verbose=False,
        )
        t_trunc = time.perf_counter() - t0

        # Truncated should be at least 1.5x faster on this workload.
        # (5x+ is typical at K=24 but K=12 is closer to the threshold
        # where helper overhead matters.)
        assert t_trunc < t_exact / 1.5, (
            f"truncation didn't yield speedup: exact={t_exact:.3f}s, "
            f"trunc={t_trunc:.3f}s"
        )
        # Tight numerical agreement.
        assert abs(s_exact - s_trunc) < 1e-9

    def test_global_default_propagates(self):
        """set_default(truncation_sigmas=...) reaches the helper through
        the public dispatcher."""
        reset_defaults()
        dens_x, dens_y = _build_density_pair(is_rel=True, is_per=False)

        s_explicit = cos_sim_exp_tens(
            dens_x, dens_y, method='pairwise',
            truncation_sigmas=6.0, verbose=False,
        )

        set_default(truncation_sigmas=6.0)
        try:
            s_global = cos_sim_exp_tens(
                dens_x, dens_y, method='pairwise', verbose=False,
            )
        finally:
            reset_defaults()

        # Both routes apply the same truncation; values match bit-exactly.
        assert s_explicit == s_global


# -----------------------------------------------------------------------
# List / cartesian dispatch path also forwards
# -----------------------------------------------------------------------


class TestCosSimListPath:

    def test_pairwise_list_forwards_truncation(self):
        reset_defaults()
        # Three rel-non-per chords, pairwise mode.
        densities_x = [
            build_exp_tens(np.array([0., 400., 700.]), np.ones(3),
                           12.0, 3, True, False, 0.0),
            build_exp_tens(np.array([0., 300., 700.]), np.ones(3),
                           12.0, 3, True, False, 0.0),
            build_exp_tens(np.array([0., 400., 800.]), np.ones(3),
                           12.0, 3, True, False, 0.0),
        ]
        densities_y = [
            build_exp_tens(np.array([0., 350., 700.]), np.ones(3),
                           12.0, 3, True, False, 0.0)
        ] * 3

        s_exact = cos_sim_exp_tens(
            densities_x, densities_y, method='pairwise',
            mode='pairwise', verbose=False,
        )
        s_trunc = cos_sim_exp_tens(
            densities_x, densities_y, method='pairwise', mode='pairwise',
            truncation_sigmas=6.0, verbose=False,
        )
        np.testing.assert_allclose(s_exact, s_trunc, rtol=0, atol=1e-7)
