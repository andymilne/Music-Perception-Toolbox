"""Regression tests for the auto-prune of zero-weight tuples in
``_cell_masses_ma_absolute`` / ``_cell_masses_sa_absolute``, plus the
``sd`` / ``width`` keyword-only API on ``weight_events``.

The auto-prune mirrors the eval-path prune in ``_tensor/eval.py``.
Without it, when ``weight_events`` truncates most events to zero weight
under a Gaussian or rectangle window, the cell-mass einsum builds
``(n_j, n_cells)`` erf-difference matrices over every upstream tuple
including the zero-weighted ones. For long sequences this can OOM,
even though zero-weight tuples contribute exactly zero to the result.

The tests cover:
1. The ``sd`` / ``width`` keyword-only API on :func:`weight_events`:
   exactly one must be supplied; ``width`` is internally converted to
   an SD as ``sd = width / (2 * sqrt(3))`` so the two parameterizations
   produce the same density when their values are paired by that ratio.
2. Differential entropy on a long sequence (1000 events) with a
   Gaussian window that zeros most events: must (a) return a finite
   value in bounded time, and (b) agree with the manually-pruned
   computation to high precision.
3. ``method='shannon'`` on the same long-sequence input: same numerical
   parity check.
"""
import math
import time

import numpy as np
import pytest

import mpt
from mpt import (
    add_spectra,
    entropy_exp_tens,
    weight_events,
)


# ---------------------------------------------------------------------
# sd / width keyword-only API
# ---------------------------------------------------------------------

def _trivial_two_attr_inputs(n=4):
    """Minimal 2-attribute inputs (pitch K=1, time K=1) for API tests."""
    pitches = np.arange(n, dtype=float) + 60.0      # (1, n)
    times = np.arange(n, dtype=float) * 0.25        # (1, n)
    p_attr = [pitches.reshape(1, n), times.reshape(1, n)]
    w = [np.ones((1, n)), np.ones((1, n))]
    return p_attr, w


def test_weight_events_requires_sd_or_width():
    """Calling with neither sd nor width must raise TypeError."""
    p_attr, w = _trivial_two_attr_inputs()
    with pytest.raises(TypeError, match="exactly one of `sd` or `width`"):
        weight_events(
            p_attr, w,
            input_attr=1, target_attr=0,
            centre=0.5, shape=0.0,
            is_per=False, period=0.0,
            delete_input=True,
        )


def test_weight_events_rejects_both_sd_and_width():
    """Calling with both sd and width must raise TypeError."""
    p_attr, w = _trivial_two_attr_inputs()
    with pytest.raises(TypeError, match="exactly one of `sd` or `width`"):
        weight_events(
            p_attr, w,
            input_attr=1, target_attr=0,
            centre=0.5, shape=0.0,
            is_per=False, period=0.0,
            sd=1.0, width=1.0,
            delete_input=True,
        )


def test_weight_events_sd_and_width_yield_same_density_under_conversion():
    """``sd = s`` and ``width = s * 2 * sqrt(3)`` must produce identical
    output (same SD specified two ways).
    """
    p_attr, w = _trivial_two_attr_inputs(n=20)
    s = 1.0
    L = s * 2.0 * math.sqrt(3.0)

    # Rectangle (shape = 1)
    _, w_sd, _ = weight_events(
        p_attr, w,
        input_attr=1, target_attr=0,
        centre=2.0, shape=1.0,
        is_per=False, period=0.0,
        sd=s, delete_input=True,
    )
    _, w_width, _ = weight_events(
        p_attr, w,
        input_attr=1, target_attr=0,
        centre=2.0, shape=1.0,
        is_per=False, period=0.0,
        width=L, delete_input=True,
    )
    np.testing.assert_allclose(w_sd[0], w_width[0], rtol=0, atol=1e-12)

    # Gaussian (shape = 0)
    _, w_sd_g, _ = weight_events(
        p_attr, w,
        input_attr=1, target_attr=0,
        centre=2.0, shape=0.0,
        is_per=False, period=0.0,
        sd=s, delete_input=True,
    )
    _, w_width_g, _ = weight_events(
        p_attr, w,
        input_attr=1, target_attr=0,
        centre=2.0, shape=0.0,
        is_per=False, period=0.0,
        width=L, delete_input=True,
    )
    np.testing.assert_allclose(w_sd_g[0], w_width_g[0], rtol=0, atol=1e-12)


def test_weight_events_rect_width_is_full_support():
    """At shape=1, ``width=L`` gives a half-open rectangle ``[-L/2, +L/2)``
    around the centre: the lower edge ``-L/2`` is kept, the upper edge
    ``+L/2`` is excluded, and events just beyond are zeroed. The half-open
    rule gives exactly ``N`` pulses for full support ``N*IOI`` on a regular
    grid (a closed interval over-counts).
    """
    # Place events at -L/2, -L/4, 0, +L/4, +L/2, and just past +L/2.
    L = 1.0
    eps = 1e-6
    times = np.array([-L/2, -L/4, 0.0, +L/4, +L/2, +L/2 + eps])
    n = len(times)
    pitches = np.full((1, n), 60.0)
    p_attr = [pitches, times.reshape(1, n)]
    w = [np.ones((1, n)), np.ones((1, n))]

    _, w_out, _ = weight_events(
        p_attr, w,
        input_attr=1, target_attr=0,
        centre=0.0, shape=1.0,
        is_per=False, period=0.0,
        width=L, delete_input=True,
    )
    # Lower edge and interior (indices 0-3) kept; upper edge +L/2 (index 4)
    # and just-past (index 5) excluded.
    np.testing.assert_allclose(w_out[0][0, :4], 1.0)
    assert w_out[0][0, 4] == 0.0
    assert w_out[0][0, 5] == 0.0


def test_weight_events_sd_rejects_nonpositive():
    p_attr, w = _trivial_two_attr_inputs()
    for bad in [0.0, -1.0]:
        with pytest.raises(ValueError, match="sd must be finite and > 0"):
            weight_events(
                p_attr, w,
                input_attr=1, target_attr=0,
                centre=0.5, shape=0.0,
                is_per=False, period=0.0,
                sd=bad, delete_input=True,
            )


def test_weight_events_width_rejects_nonpositive():
    p_attr, w = _trivial_two_attr_inputs()
    for bad in [0.0, -1.0]:
        with pytest.raises(ValueError, match="width must be finite and > 0"):
            weight_events(
                p_attr, w,
                input_attr=1, target_attr=0,
                centre=0.5, shape=0.0,
                is_per=False, period=0.0,
                width=bad, delete_input=True,
            )


# ---------------------------------------------------------------------
# Auto-prune in _cell_masses_ma_absolute (multi-attribute path)
# ---------------------------------------------------------------------

def _build_long_sequence_density_inputs(n_events: int = 1000, seed: int = 0):
    """Construct a long pitch-and-time sequence as MA input.

    Pitch attribute: 4 voices per event, spread across 4000-7000 cents,
    spectrally expanded with 12 harmonic partials at 1/n rolloff
    (48 partials per event). Time attribute: K=1 evenly spaced.
    """
    rng = np.random.default_rng(seed)
    pitches = rng.uniform(4000.0, 7000.0, size=(n_events, 4))
    times = np.arange(n_events, dtype=float) * 0.25  # 16th-note grid
    K = 4 * 12
    p_partials = np.zeros((n_events, K))
    w_partials = np.zeros((n_events, K))
    for i in range(n_events):
        p_aug, w_aug = add_spectra(pitches[i], None,
                                   'harmonic', 12, 'powerlaw', 1.0)
        p_partials[i] = p_aug
        w_partials[i] = w_aug
    p_attr = [p_partials.T, times.reshape(1, n_events)]
    w = [w_partials.T, np.ones((1, n_events))]
    return p_attr, w, times


@pytest.fixture(autouse=True)
def _restore_truncation_default():
    prev = mpt.get_default('truncation_sigmas')
    yield
    mpt.set_default(truncation_sigmas=prev)


def test_differential_entropy_bounded_after_weight_events_truncation():
    """A long sequence with a narrow Gaussian window must complete in
    bounded time and return a finite value via ``method='differential'``.

    Pre-patch this OOM'd because zero-weight events were carried through
    the cell-mass erf-difference einsum.
    """
    n_events = 1000
    p_attr, w, times = _build_long_sequence_density_inputs(n_events)

    # Gaussian window of sd=1 (in 0.25-QN units of the time axis) at
    # mid-sequence; truncation_sigmas=3 hard-zeros far-away events.
    mpt.set_default(truncation_sigmas=3.0)
    c = float(times[n_events // 2])

    p_w, w_w, g_w = weight_events(
        p_attr, w,
        input_attr=1, target_attr=0,
        centre=c, shape=0.0,
        is_per=False, period=0.0,
        sd=1.0, delete_input=True,
    )
    # Sanity: weight_events should zero most events, but not prune them.
    assert p_w[0].shape[1] == n_events
    n_nonzero_events = int((w_w[0].sum(axis=0) > 0).sum())
    assert n_nonzero_events < 30  # narrow window keeps only a handful

    t0 = time.time()
    H = entropy_exp_tens(
        p_w, w_w, [10.0], [1], [False], [False], [0.0],
        method='differential', base=2.0,
    )
    elapsed = time.time() - t0
    assert np.isfinite(H)
    # Generous ceiling: pre-patch this OOM'd in seconds; post-patch
    # should complete in a few seconds at most.
    assert elapsed < 60.0, (
        f'differential entropy on long truncated sequence took '
        f'{elapsed:.1f}s; auto-prune regression suspected.'
    )


def test_differential_entropy_matches_manual_prune():
    """Auto-prune must be numerically equivalent to manual upstream
    pruning: the cell-mass contraction drops zero-weight tuples
    exactly, so the entropy is unchanged.
    """
    n_events = 500
    p_attr, w, times = _build_long_sequence_density_inputs(n_events, seed=1)
    mpt.set_default(truncation_sigmas=3.0)
    c = float(times[n_events // 2])

    p_w, w_w, g_w = weight_events(
        p_attr, w,
        input_attr=1, target_attr=0,
        centre=c, shape=0.0,
        is_per=False, period=0.0,
        sd=1.0, delete_input=True,
    )
    H_auto = entropy_exp_tens(
        p_w, w_w, [10.0], [1], [False], [False], [0.0],
        method='differential', base=2.0,
    )

    # Manually prune zero-weight events upstream and recompute.
    keep = w_w[0].sum(axis=0) > 0
    p_w_pruned = [p[:, keep] for p in p_w]
    w_w_pruned = [ww[:, keep] for ww in w_w]
    H_manual = entropy_exp_tens(
        p_w_pruned, w_w_pruned, [10.0], [1],
        [False], [False], [0.0],
        method='differential', base=2.0,
    )

    assert H_auto == pytest.approx(H_manual, rel=0, abs=1e-8), (
        f'auto-prune differs from manual prune: {H_auto} vs {H_manual}'
    )


def test_shannon_grid_matches_manual_prune():
    """Same parity check for ``method='shannon'`` on a fixed fine grid.

    Shannon on a 1-cent grid (Δ=1) and differential entropy in nats
    agree mathematically (h = H_disc + log Δ; log(1) = 0), so the same
    prune-invariance check applies.
    """
    n_events = 500
    p_attr, w, times = _build_long_sequence_density_inputs(n_events, seed=2)
    mpt.set_default(truncation_sigmas=3.0)
    c = float(times[n_events // 2])

    p_w, w_w, g_w = weight_events(
        p_attr, w,
        input_attr=1, target_attr=0,
        centre=c, shape=0.0,
        is_per=False, period=0.0,
        sd=1.0, delete_input=True,
    )
    H_auto = entropy_exp_tens(
        p_w, w_w, [10.0], [1], [False], [False], [0.0],
        method='shannon', base=2.0,
        n_points_per_dim=2001, x_min=3000.0, x_max=13000.0,
    )

    keep = w_w[0].sum(axis=0) > 0
    p_w_pruned = [p[:, keep] for p in p_w]
    w_w_pruned = [ww[:, keep] for ww in w_w]
    H_manual = entropy_exp_tens(
        p_w_pruned, w_w_pruned, [10.0], [1],
        [False], [False], [0.0],
        method='shannon', base=2.0,
        n_points_per_dim=2001, x_min=3000.0, x_max=13000.0,
    )
    assert H_auto == pytest.approx(H_manual, rel=0, abs=1e-10)
