"""Regression tests for periodic operations on unfolded coordinates.

A family of periodic routines build their result by summing Gaussian
images across the period: the differential/Shannon grid cell mass
(:func:`mpt.entropy._phi_diff_axis_periodic`), the wrapped-window factor
(:func:`mpt._tensor.windowing._wrapped_window_factor_1d`), and the
windowed inner-product image sum
(:func:`mpt._tensor.windowing._periodic_image_sum_contribution`). Each
must reduce its input coordinate modulo the period first. Otherwise a
coordinate many periods from the canonical ``[0, period)`` window --- for
example an absolute spectral partial thousands of cents above the grid
--- never contributes, and the routine returns a degenerate,
input-independent value. These tests lock in the modulo reduction.
"""
import numpy as np

from mpt import entropy_exp_tens, add_spectra
from mpt._tensor.windowing import (
    _wrapped_window_factor_1d,
    _periodic_image_sum_contribution,
)

PERIOD = 1200.0  # one octave in cents


def _spectral_partials(midi_chord):
    """Absolute harmonic partials (cents) for a MIDI chord."""
    cents = np.asarray(midi_chord, dtype=float) * 100.0
    return add_spectra(cents, None, "harmonic", 12, "powerlaw", 1.0)


def test_differential_periodic_unfolded_matches_folded():
    # Periodic differential entropy must not depend on whether the caller
    # pre-folds the centres into [0, period).
    p, w = _spectral_partials([64, 59, 56, 40])  # E major; partials past 3P
    assert p.max() > 3 * PERIOD  # the regime that triggered the bug
    unfolded = entropy_exp_tens(p, w, 10.0, 1, False, True, PERIOD,
                                method="differential", base=np.e, verbose=False)
    folded = entropy_exp_tens(p % PERIOD, w, 10.0, 1, False, True, PERIOD,
                              method="differential", base=np.e, verbose=False)
    assert np.isclose(unfolded, folded, atol=1e-9)
    # A sane positive differential entropy, not the degenerate constant.
    assert 4.0 < unfolded < 8.0


def test_differential_periodic_varies_with_harmony():
    # The degenerate bug returned the same constant for every chord;
    # distinct harmonies must give distinct periodic differential entropy.
    pe, we = _spectral_partials([64, 59, 56, 40])  # E major
    pb, wb = _spectral_partials([71, 66, 62, 47])  # B minor
    he = entropy_exp_tens(pe, we, 10.0, 1, False, True, PERIOD,
                          method="differential", base=np.e, verbose=False)
    hb = entropy_exp_tens(pb, wb, 10.0, 1, False, True, PERIOD,
                          method="differential", base=np.e, verbose=False)
    assert not np.isclose(he, hb, atol=1e-3)


def test_wrapped_window_factor_period_invariant():
    # The wrapped window is periodic in its offset; shifting the offset by
    # whole periods must not change the value, even when the near images
    # underflow to zero.
    a_, b_, tol = 0.0, 100.0, 1e-12  # pure Gaussian window, width 100
    base = float(_wrapped_window_factor_1d(
        np.array([0.3 * PERIOD]), a_, b_, PERIOD, tol)[0])
    assert base > 0.0
    for k in (3.3, 10.3, -7.7, 41.3):
        val = float(_wrapped_window_factor_1d(
            np.array([k * PERIOD]), a_, b_, PERIOD, tol)[0])
        assert np.isclose(val, base, rtol=0, atol=1e-15)


def test_windowed_inner_product_period_invariant():
    # The windowed inner-product image sum depends on (midpoint - centre)
    # modulo the period; shifting the window centre by whole periods must
    # leave the per-pair contribution unchanged.
    cx = np.array([[100.0, 500.0]])
    cy = np.array([[300.0]])
    tol = 1e-12

    def contrib(centre):
        return _periodic_image_sum_contribution(
            cx, cy, np.array([centre]), 1.0, 0.0, 100.0,
            False, 1, 1, PERIOD, tol)[0]

    base = contrib(200.0)
    for k in (1, 7, 33):
        assert np.allclose(base, contrib(200.0 + k * PERIOD), atol=1e-12)
