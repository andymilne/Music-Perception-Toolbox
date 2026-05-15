"""Stage 2c: verify wrapper routing forwards truncation/precision kwargs
and that the values remain consistent with exact evaluation.

These tests cover the Stage 2c invariant that user-facing wrappers
(``tensor_harmonicity``, ``template_harmonicity``, ``virtual_pitches``,
``spectral_entropy``) do NOT pin down an internal algorithm choice and
DO forward ``truncation_sigmas`` / ``kernel_precision`` to their
underlying :func:`eval_exp_tens` calls.

The numerical contract for any wrapper W and a representative chord
battery is: ``W(...)`` (default) and ``W(..., truncation_sigmas=k)``
agree to within the truncation tolerance.
"""
from __future__ import annotations

import numpy as np
import pytest

from mpt import (
    reset_defaults,
    set_default,
    spectral_entropy,
    template_harmonicity,
    tensor_harmonicity,
    virtual_pitches,
)


CHORD_BATTERY = {
    "unison": np.array([0.0, 1.0]),
    "major_triad": np.array([0.0, 400.0, 700.0]),
    "minor_triad": np.array([0.0, 300.0, 700.0]),
    "diminished_triad": np.array([0.0, 300.0, 600.0]),
    "augmented_triad": np.array([0.0, 400.0, 800.0]),
    "narrow_cluster": np.array([0.0, 50.0, 100.0]),
    "octave_plus_fifth": np.array([0.0, 700.0, 1200.0]),
}


# -----------------------------------------------------------------------
#  tensor_harmonicity  (the wrapper that was rerouted in Stage 2c)
# -----------------------------------------------------------------------

class TestTensorHarmonicityRouting:

    @pytest.mark.parametrize("name", list(CHORD_BATTERY.keys()))
    def test_scalar_exact_vs_truncated_k6(self, name):
        """Default (exact) and ``truncation_sigmas=6`` agree to ~1e-7."""
        reset_defaults()
        chord = CHORD_BATTERY[name]

        h_exact = tensor_harmonicity(
            chord, None, sigma=12.0, verbose=False,
        )
        h_trunc = tensor_harmonicity(
            chord, None, sigma=12.0, truncation_sigmas=6.0, verbose=False,
        )

        # Absolute tolerance: 6-sigma truncation per slot caps weight
        # discarded at ~erfc(6/sqrt(2)) ~ 2e-9; the normalised harmonicity
        # absorbs this. 1e-7 is a comfortable bound.
        assert abs(h_exact - h_trunc) < 1e-7, (
            f"{name}: |H_exact - H_trunc| = {abs(h_exact - h_trunc):.3e} "
            f"(H_exact = {h_exact:.6f})"
        )

    def test_global_default_propagates_into_wrapper(self):
        """Setting truncation_sigmas via set_default affects the wrapper."""
        chord = CHORD_BATTERY["major_triad"]

        reset_defaults()
        h_explicit_k6 = tensor_harmonicity(
            chord, None, sigma=12.0, truncation_sigmas=6.0, verbose=False,
        )

        set_default(truncation_sigmas=6.0)
        try:
            h_global_k6 = tensor_harmonicity(chord, None, sigma=12.0, verbose=False)
        finally:
            reset_defaults()

        assert h_explicit_k6 == h_global_k6, (
            f"global default not propagating: explicit={h_explicit_k6}, "
            f"global={h_global_k6}"
        )

    def test_batched_matches_scalar_with_truncation(self):
        """Batched dispatch returns same values as per-row scalar."""
        reset_defaults()
        P = np.stack([
            CHORD_BATTERY["major_triad"],
            CHORD_BATTERY["minor_triad"],
            CHORD_BATTERY["diminished_triad"],
        ])

        H_batch = tensor_harmonicity(
            P, None, sigma=12.0, truncation_sigmas=6.0, verbose=False,
        )
        H_scalar = np.array([
            tensor_harmonicity(P[i], None, sigma=12.0,
                               truncation_sigmas=6.0, verbose=False)
            for i in range(P.shape[0])
        ])
        np.testing.assert_allclose(H_batch, H_scalar, rtol=0, atol=1e-12)


# -----------------------------------------------------------------------
#  template_harmonicity, virtual_pitches, spectral_entropy
#  (already routed correctly; verify kwargs threading works)
# -----------------------------------------------------------------------

class TestKwargsThreading:

    def test_template_harmonicity_accepts_truncation_kwargs(self):
        reset_defaults()
        chord = CHORD_BATTERY["major_triad"]
        # Should not raise and should produce a result that's close to
        # the exact one.
        h_max_exact, _ = template_harmonicity(
            chord, None, sigma=12.0, verbose=False,
        )
        h_max_trunc, _ = template_harmonicity(
            chord, None, sigma=12.0, truncation_sigmas=6.0, verbose=False,
        )
        # Wider tolerance: template_harmonicity's max-correlation peak
        # picking can drift slightly with sub-1e-7 changes in the
        # underlying vals, but the magnitude should match closely.
        assert abs(h_max_exact - h_max_trunc) < 1e-5

    def test_virtual_pitches_accepts_truncation_kwargs(self):
        reset_defaults()
        chord = CHORD_BATTERY["major_triad"]
        p_exact, w_exact = virtual_pitches(
            chord, None, sigma=12.0, verbose=False,
        )
        p_trunc, w_trunc = virtual_pitches(
            chord, None, sigma=12.0, truncation_sigmas=6.0, verbose=False,
        )
        # Pitch axis is shared; the salience profiles should match closely.
        np.testing.assert_allclose(p_exact, p_trunc)
        np.testing.assert_allclose(w_exact, w_trunc, rtol=0, atol=1e-5)

    def test_spectral_entropy_accepts_truncation_kwargs(self):
        reset_defaults()
        chord = CHORD_BATTERY["major_triad"]
        spec = ["harmonic", 12, "powerlaw", 1]
        h_exact = spectral_entropy(
            chord, None, sigma=12.0, spectrum=spec, verbose=False,
        )
        h_trunc = spectral_entropy(
            chord, None, sigma=12.0, spectrum=spec,
            truncation_sigmas=6.0, verbose=False,
        )
        # Entropy values are O(1); 1e-7 absolute is well above truncation.
        assert abs(h_exact - h_trunc) < 1e-5
