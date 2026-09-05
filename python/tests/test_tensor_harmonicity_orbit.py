"""Tests for ``tensor_harmonicity`` after the v3 orbit-eval refactor.

Pre-v3, ``tensor_harmonicity`` materialised a centres array of size
``(r-1, K!/(K-r)!)`` inside ``build_exp_tens``. With the default
64-partial harmonic template, a 4-pitch chord blew the build to
~9·10⁸ four-tuples (~25 GB just for the index tensor), so the function
emitted a "computation time grows rapidly" warning at K > 3 and
typically OOM'd on a 4-pitch chord with the default spectrum.

The v3 refactor routes ``tensor_harmonicity`` directly to
:func:`mpt._mobius.eval_orbit_rel`, which evaluates the rel-mode
template tensor at the chord's interval vector via Möbius point
evaluation. Memory is independent of ``K!/(K-r)!``; runtime grows
as ``B_r · r · K · N_u`` per query.

These tests verify:
1. The previously-failing 4-pitch case now runs and returns a finite,
   positive value within a sane time budget.
2. Numerical agreement with the centres path on small chords where
   the centres path is still feasible (regression check).
3. The musical-intuition rankings the earlier tests asserted at K ≤ 3
   continue to hold at K = 4 (smell-test the new path's plausibility).
"""
import time

import numpy as np

import mpt
from mpt.harmony import tensor_harmonicity
from mpt._mobius import eval_orbit_rel
from mpt.tensor import build_exp_tens, eval_exp_tens
from mpt.spectra import add_spectra


def test_tensor_harmonicity_4pitch_default_spectrum_runs():
    """4-pitch chord with default 64-partial template: previously
    OOM'd on the centres path; now runs via orbit eval."""
    chord = np.array([0.0, 400.0, 700.0, 1100.0])  # major-7
    t0 = time.perf_counter()
    h = tensor_harmonicity(chord, sigma=12.0)
    elapsed = time.perf_counter() - t0
    assert np.isfinite(h)
    assert h > 0
    # Loose timing budget; the orbit path takes ~13 s on the dev
    # container and should never approach a minute on any reasonable
    # workstation.
    assert elapsed < 60.0, f"orbit eval took {elapsed:.1f} s (> 60 s budget)"


def test_tensor_harmonicity_matches_centres_at_K_eq_2():
    """At a 2-pitch chord with a small spectrum (12 partials), the
    centres path is feasible. Verify the v3 orbit path returns the
    same value to FP precision."""
    chord = np.array([0.0, 700.0])  # perfect fifth
    sigma = 12.0
    spectrum = ["harmonic", 12, "powerlaw", 1]

    h_orbit = tensor_harmonicity(chord, sigma=sigma, spectrum=spectrum)

    # Compute the same quantity via the centres path directly.
    n_pitches = len(chord)
    tmpl_p, tmpl_w = add_spectra(np.zeros(n_pitches), np.ones(n_pitches),
                                 *spectrum)
    T = build_exp_tens(
        tmpl_p, tmpl_w, sigma, n_pitches, True, False, 1200, verbose=False,
    )
    intervals = (np.sort(chord)[1:] - np.sort(chord)[0]).reshape(-1, 1)
    h_centres = float(eval_exp_tens(
        T, intervals, normalize="none", method="centres", verbose=False,
    )[0])

    assert np.isclose(h_orbit, h_centres, atol=1e-12, rtol=1e-10)


def test_tensor_harmonicity_matches_centres_at_K_eq_3_with_small_spectrum():
    """At a 3-pitch chord with a 12-partial template, K = 36 and the
    centres path is still feasible. Verify FP-precision agreement."""
    chord = np.array([0.0, 400.0, 700.0])  # major triad
    sigma = 12.0
    spectrum = ["harmonic", 12, "powerlaw", 1]

    h_orbit = tensor_harmonicity(chord, sigma=sigma, spectrum=spectrum)

    n_pitches = len(chord)
    tmpl_p, tmpl_w = add_spectra(np.zeros(n_pitches), np.ones(n_pitches),
                                 *spectrum)
    T = build_exp_tens(
        tmpl_p, tmpl_w, sigma, n_pitches, True, False, 1200, verbose=False,
    )
    intervals = (np.sort(chord)[1:] - np.sort(chord)[0]).reshape(-1, 1)
    h_centres = float(eval_exp_tens(
        T, intervals, normalize="none", method="centres", verbose=False,
    )[0])

    assert np.isclose(h_orbit, h_centres, atol=1e-12, rtol=1e-10)


def test_tensor_harmonicity_4pitch_ranking_matches_intuition():
    """At a 4-pitch chord cardinality, common-practice harmonicity
    expectations: an almost-unison '0,0,0,0' chord (everything aligned
    with a unison fundamental) beats a stack of unrelated tritones
    [0, 600, 100, 700]. This is a smell-test for the orbit path
    rather than a strong precision check."""
    sigma = 12.0
    spectrum = ["harmonic", 12, "powerlaw", 1]
    h_aligned = tensor_harmonicity(
        [0.0, 0.0, 0.0, 0.0], sigma=sigma, spectrum=spectrum,
    )
    h_dissonant = tensor_harmonicity(
        [0.0, 600.0, 100.0, 700.0], sigma=sigma, spectrum=spectrum,
    )
    assert h_aligned > h_dissonant


def test_tensor_harmonicity_normalize_options_unchanged():
    """Each ``normalize`` option ('none', 'gaussian', 'pdf') must
    produce a finite, non-negative scalar. The relative ordering of
    the three for a fixed chord is fixed by the constants the path
    multiplies, so a coarse sanity check is sufficient."""
    chord = np.array([0.0, 700.0])
    sigma = 12.0
    spectrum = ["harmonic", 12, "powerlaw", 1]

    h_none = tensor_harmonicity(
        chord, sigma=sigma, spectrum=spectrum, normalize="none",
    )
    h_gauss = tensor_harmonicity(
        chord, sigma=sigma, spectrum=spectrum, normalize="gaussian",
    )
    h_pdf = tensor_harmonicity(
        chord, sigma=sigma, spectrum=spectrum, normalize="pdf",
    )

    for h in (h_none, h_gauss, h_pdf):
        assert np.isfinite(h)
        assert h >= 0


def test_tensor_harmonicity_rejects_invalid_normalize():
    """Mismatched 'normalize' value raises ValueError, matching the
    centres path's behaviour."""
    chord = np.array([0.0, 700.0])
    try:
        tensor_harmonicity(chord, sigma=12.0, normalize="not-a-mode")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown normalize mode")
