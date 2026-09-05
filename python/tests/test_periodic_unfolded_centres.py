"""Regression tests for periodic operations on unfolded coordinates.

Several periodic routines must remain correct when their input
coordinates lie far from the canonical ``[0, period)`` window --- for
example absolute spectral partials thousands of cents above the grid.
The differential/Shannon grid cell mass
(:func:`mpt.entropy._phi_diff_axis_periodic`) reduces the per-edge
offsets to their nearest image first, so it is invariant to whole-period
shifts of a centre, and then integrates the density the attribute's
``wrap`` declares: the wrapped normal under the default
``'full-image'``, the minimum-image Gaussian of Eq. (1) under
``'single-image'``. The wrapped-window factor
(:func:`mpt._tensor.windowing._wrapped_window_factor_1d`) and the windowed
inner-product image sum
(:func:`mpt._tensor.windowing._periodic_image_sum_contribution`) sum
Gaussian images across the period, reducing the offset to its minimum
image first so the dominant image is reached. These tests lock in both
the period-invariance and the wrap semantics of the entropy cell mass.
"""
import warnings

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


def test_differential_periodic_follows_the_wrap():
    # At non-negligible sigma/period the periodic differential entropy
    # must track the density the wrap declares: the wrapped normal under
    # the default 'full-image', the minimum-image density of Eq. (1)
    # under 'single-image'.
    from mpt import build_exp_tens
    c = np.array([0.0, 400.0, 700.0, 1100.0])
    w = np.ones(4)

    def fine(mode, s, G=200_000):
        xs = np.linspace(0.0, PERIOD, G, endpoint=False)
        if mode == "min":
            d = xs[None, :] - c[:, None]
            d = d - PERIOD * np.round(d / PERIOD)
            f = (w[:, None] * np.exp(-d**2 / (2 * s**2))).sum(0)
        else:  # wrapped normal: sum periodic images
            f = sum((w[:, None] * np.exp(
                -(xs[None, :] - c[:, None] - n * PERIOD)**2 / (2 * s**2))).sum(0)
                for n in range(-6, 7))
        f /= f.sum() * (PERIOD / G)
        return -(f * np.log(f + 1e-300)).sum() * (PERIOD / G)

    s = 300.0  # sigma/period = 0.25, where the two forms visibly differ
    htb_full = entropy_exp_tens(c, w, s, 1, False, True, PERIOD,
                                method="differential", base=np.e,
                                verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        d_single = build_exp_tens([c.reshape(-1, 1)], [w.reshape(-1, 1)],
                                  [s], [1], [False], [True], [PERIOD],
                                  wrap=['single-image'], verbose=False)
    htb_single = entropy_exp_tens(d_single, method="differential",
                                  base=np.e, verbose=False)
    h_min, h_wrap = fine("min", s), fine("wrap", s)
    assert abs(h_min - h_wrap) > 1e-4          # the two forms really differ here
    assert abs(htb_full - h_wrap) < abs(htb_full - h_min)   # default: wrapped
    assert abs(htb_full - h_wrap) < 1e-3       # and matches it to grid precision
    assert abs(htb_single - h_min) < abs(htb_single - h_wrap)  # opt-in: min-image
    assert abs(htb_single - h_min) < 1e-3


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
