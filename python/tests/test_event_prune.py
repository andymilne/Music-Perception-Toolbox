"""Regression tests for the density-level live-event prune consumed by
the inner-product / total-mass paths (``method='renyi2'`` on
:func:`entropy_exp_tens` and :func:`cos_sim_exp_tens`).

A dead event --- one whose weight is zero or NaN on an attribute, so it
contributes nothing to any inner product or total mass --- is dropped at
the density level before the per-attribute IP / mass work, via
``ExpTensDensity.pruned()`` / ``MaetDensity.pruned()`` (the MATLAB
counterpart is ``internal.prunedExpTens``). This is the sibling, one
level up, of the cell-mass tuple prune in
``test_cell_mass_zero_weight_prune``: that one drops zero-weight tuples
inside the grid path; this one drops whole events before the IP / mass
reduction.

The prune is needed because ``maPerAttrInnerMatrix`` /
``_ma_per_attr_inner_matrix`` already prune *per attribute*, but
``weight_events`` writes its window factor to only the target attribute,
so an event hard-zeroed through one attribute under ``truncation_sigmas``
remains present (all-ones) on the others --- the event-level kill is
invisible to a per-attribute prune.

The liveness rule (single predicate: a weight contributes iff finite and
nonzero):
  * SA: an element is live iff its weight is finite and nonzero.
  * MA: an event is live iff *every* attribute has at least one finite,
    nonzero slot in that event's column (the per-attribute factors
    multiply, so an all-zero or all-NaN column kills the event; a
    partly-zero column does not).

The tests cover:
1. The liveness rule for SA and MA, including the partly-zero-column
   case (must stay live) and the all-zero / all-NaN column case (kills).
2. ``pruned()`` returns ``self`` when nothing is dead (the common
   un-windowed path pays only a mask scan).
3. ``pruned()`` drops exactly the dead events and keeps the rest.
4. Rényi-2 entropy (SA and MA) and cosine similarity (SA and MA) are
   numerically unchanged with the dead events present versus removed ---
   the whole point, since dead events contribute exactly zero.
5. A realistic ``weight_events`` truncation case (MA): auto-prune
   matches manual upstream pruning, and the windowed density completes
   in bounded time.
6. Edge cases of a fully dead density.
"""
import time

import numpy as np
import pytest

import mpt
from mpt import (
    build_exp_tens,
    cos_sim_exp_tens,
    entropy_exp_tens,
    weight_events,
)


@pytest.fixture(autouse=True)
def _quiet_and_restore():
    prev_hints = mpt.get_default('show_hints')
    prev_trunc = mpt.get_default('truncation_sigmas')
    mpt.set_default(show_hints=False)
    yield
    mpt.set_default(show_hints=prev_hints, truncation_sigmas=prev_trunc)


# ---------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------

def _sa_density(p, w):
    return build_exp_tens(np.asarray(p, float), np.asarray(w, float),
                          0.5, 1, False, False, 0.0, verbose=False)


def _ma_density(p_attr, w, r=(1, 1)):
    return build_exp_tens(
        [np.asarray(P, float) for P in p_attr],
        None if w is None else [np.asarray(W, float) for W in w],
        [0.5, 0.15], list(r), [False, False], [False, False],
        [0.0, 0.0], verbose=False,
    )


# ---------------------------------------------------------------------
# Liveness rule
# ---------------------------------------------------------------------

def test_sa_live_events_rule():
    """SA: live iff finite and nonzero; 0 and NaN both fail."""
    dens = _sa_density([60., 62., 64., 66., 68.],
                       [1.0, 0.0, np.nan, 2.0, -3.0])
    np.testing.assert_array_equal(
        dens.live_events, np.array([True, False, False, True, True])
    )


def test_ma_live_events_rule_partly_zero_column_stays_live():
    """MA: an all-zero (or all-NaN) column on any attribute kills the
    event; a partly-zero column does not.
    """
    # Pitch attribute, K=2 slots per event. Event 1 has both pitch slots
    # zero (all-zero column -> dead). Events 0 and 2 have one zero slot
    # each (partly-zero -> still live). Event 3 has an all-NaN pitch
    # column (-> dead). Time attribute is fully live throughout.
    p_pitch = np.array([[60., 62., 64., 66.],
                        [67., 69., 71., 73.]])
    w_pitch = np.array([[1.0, 0.0, 1.0, np.nan],
                        [0.0, 0.0, 1.0, np.nan]])
    p_time = np.array([[0., 1., 2., 3.]])
    w_time = np.array([[1., 1., 1., 1.]])
    dens = _ma_density([p_pitch, p_time], [w_pitch, w_time], r=(2, 1))
    np.testing.assert_array_equal(
        dens.live_events, np.array([True, False, True, False])
    )


def test_ma_live_events_all_zero_on_second_attribute_kills():
    """A live pitch column cannot rescue an event whose time column is
    zero --- the per-attribute factors multiply.
    """
    p_pitch = np.array([[60., 62., 64.]])
    w_pitch = np.array([[1., 1., 1.]])
    p_time = np.array([[0., 1., 2.]])
    w_time = np.array([[1., 0., 1.]])         # event 1 dead via time
    dens = _ma_density([p_pitch, p_time], [w_pitch, w_time], r=(1, 1))
    np.testing.assert_array_equal(
        dens.live_events, np.array([True, False, True])
    )


# ---------------------------------------------------------------------
# pruned(): self when clean, subset when not
# ---------------------------------------------------------------------

def test_pruned_returns_self_when_all_live_sa():
    dens = _sa_density([60., 62., 64.], [1., 1., 2.])
    assert dens.pruned() is dens


def test_pruned_returns_self_when_all_live_ma():
    dens = _ma_density([[[60., 62., 64.]], [[0., 1., 2.]]], None)
    assert dens.pruned() is dens


def test_pruned_drops_dead_events_sa():
    dens = _sa_density([60., 62., 64., 66.], [1., 0., np.nan, 2.])
    pr = dens.pruned()
    assert pr is not dens
    np.testing.assert_array_equal(np.asarray(pr.p), [60., 66.])
    np.testing.assert_array_equal(np.asarray(pr.w), [1., 2.])


def test_pruned_drops_dead_events_ma():
    p_pitch = np.array([[60., 62., 64.]])
    w_pitch = np.array([[1., 0., 1.]])         # event 1 dead
    p_time = np.array([[0., 1., 2.]])
    dens = _ma_density([p_pitch, p_time], [w_pitch, np.ones((1, 3))])
    pr = dens.pruned()
    assert pr is not dens
    assert pr.n == 2
    np.testing.assert_array_equal(pr.p_attr[0], [[60., 64.]])
    np.testing.assert_array_equal(pr.p_attr[1], [[0., 2.]])


# ---------------------------------------------------------------------
# Numerical invariance: dead events change nothing
# ---------------------------------------------------------------------

def test_renyi2_sa_invariant_to_dead_events():
    """SA Rényi-2 is identical with dead events present vs removed."""
    p = np.array([60., 62., 64., 66., 68., 70.])
    w = np.array([1.0, 0.7, 1.3, 0.9, 1.1, 0.5])
    dead = np.array([1, 4])                    # zero these out
    w_dead = w.copy()
    w_dead[dead] = 0.0
    keep = np.ones(len(p), bool)
    keep[dead] = False

    h_with = entropy_exp_tens(_sa_density(p, w_dead),
                              method='renyi2', verbose=False)
    h_without = entropy_exp_tens(_sa_density(p[keep], w[keep]),
                                 method='renyi2', verbose=False)
    assert h_with == pytest.approx(h_without, rel=0, abs=0.0)


def test_renyi2_ma_invariant_to_dead_events():
    """MA Rényi-2 is identical with dead events present vs removed."""
    rng = np.random.default_rng(0)
    n = 14
    p_pitch = rng.uniform(60, 72, (1, n))
    p_time = rng.uniform(0, 3, (1, n))
    w_pitch = np.ones((1, n))
    dead = [3, 7, 9]
    w_pitch[0, dead] = 0.0                     # kill via pitch column
    keep = [i for i in range(n) if i not in dead]

    d_with = _ma_density([p_pitch, p_time], [w_pitch, np.ones((1, n))])
    d_without = _ma_density(
        [p_pitch[:, keep], p_time[:, keep]],
        [w_pitch[:, keep], np.ones((1, len(keep)))],
    )
    h_with = entropy_exp_tens(d_with, method='renyi2', verbose=False)
    h_without = entropy_exp_tens(d_without, method='renyi2', verbose=False)
    assert h_with == pytest.approx(h_without, rel=0, abs=0.0)


def test_cos_sim_sa_invariant_to_dead_events():
    px = np.array([60., 62., 64., 66., 68.])
    wx = np.array([1.0, 0.0, 1.0, 0.0, 1.0])   # events 1, 3 dead
    keep = wx != 0
    dy = _sa_density([61., 63., 65.], [1., 1., 1.])

    s_with = cos_sim_exp_tens(_sa_density(px, wx), dy, verbose=False)
    s_without = cos_sim_exp_tens(_sa_density(px[keep], wx[keep]), dy,
                                 verbose=False)
    assert s_with == pytest.approx(s_without, rel=0, abs=0.0)


def test_cos_sim_ma_invariant_to_dead_events():
    rng = np.random.default_rng(1)
    n = 12
    px_pitch = rng.uniform(60, 72, (1, n))
    px_time = rng.uniform(0, 3, (1, n))
    wx_pitch = np.ones((1, n))
    dead = [2, 5, 8]
    wx_pitch[0, dead] = 0.0
    keep = [i for i in range(n) if i not in dead]

    py_pitch = rng.uniform(60, 72, (1, 7))
    py_time = rng.uniform(0, 3, (1, 7))
    dy = _ma_density([py_pitch, py_time], None)

    dx_with = _ma_density([px_pitch, px_time], [wx_pitch, np.ones((1, n))])
    dx_without = _ma_density(
        [px_pitch[:, keep], px_time[:, keep]],
        [wx_pitch[:, keep], np.ones((1, len(keep)))],
    )
    s_with = cos_sim_exp_tens(dx_with, dy, verbose=False)
    s_without = cos_sim_exp_tens(dx_without, dy, verbose=False)
    assert s_with == pytest.approx(s_without, rel=0, abs=0.0)


# ---------------------------------------------------------------------
# Realistic weight_events truncation case (MA)
# ---------------------------------------------------------------------

def _windowed_ma_inputs(n_events: int, seed: int):
    rng = np.random.default_rng(seed)
    pitches = rng.uniform(60.0, 84.0, size=(1, n_events))
    times = np.arange(n_events, dtype=float) * 0.25
    p_attr = [pitches, times.reshape(1, n_events)]
    w = [np.ones((1, n_events)), np.ones((1, n_events))]
    c = float(times[n_events // 2])
    p_w, w_w, g_w = weight_events(
        p_attr, w, [0, 1],
        input_attr=1, target_attr=0,
        centre=c, shape=0.0,
        is_per=False, period=0.0,
        sd=1.0, delete_input=True,
    )
    return p_w, w_w, g_w


def test_renyi2_ma_windowed_matches_manual_prune():
    """The truncated-window scenario the prune exists for: most events
    hard-zeroed via the target attribute, auto-prune must match manual
    upstream pruning to the bit.
    """
    mpt.set_default(truncation_sigmas=3.0)
    n_events = 600
    p_w, w_w, g_w = _windowed_ma_inputs(n_events, seed=2)

    # weight_events(delete_input=True) removes the time attribute, so the
    # windowed density carries a single attribute (pitch) whose column was
    # zeroed for far events --- the realistic truncation scenario.
    n_live = int((w_w[0].sum(axis=0) > 0).sum())
    assert 0 < n_live < 40              # narrow window keeps a handful

    d_auto = build_exp_tens(p_w, w_w, [0.5], [1], 
                            [False], [False], [0.0], verbose=False)
    keep = w_w[0].sum(axis=0) > 0
    d_manual = build_exp_tens(
        [p[:, keep] for p in p_w], [ww[:, keep] for ww in w_w],
        [0.5], [1], 
        [False], [False], [0.0], verbose=False,
    )
    h_auto = entropy_exp_tens(d_auto, method='renyi2', verbose=False)
    h_manual = entropy_exp_tens(d_manual, method='renyi2', verbose=False)
    assert h_auto == pytest.approx(h_manual, rel=0, abs=0.0)


def test_renyi2_ma_windowed_bounded_time():
    """Soft guard: a large windowed density must not walk the full event
    set. Correctness is anchored by the parity test above; this catches
    a prune that silently stops firing.
    """
    mpt.set_default(truncation_sigmas=3.0)
    n_events = 3000
    p_w, w_w, g_w = _windowed_ma_inputs(n_events, seed=3)
    d = build_exp_tens(p_w, w_w, [0.5], [1], 
                       [False], [False], [0.0], verbose=False)
    t0 = time.time()
    h = entropy_exp_tens(d, method='renyi2', verbose=False)
    elapsed = time.time() - t0
    assert np.isfinite(h)
    assert elapsed < 20.0, (
        f'renyi2 on a {n_events}-event windowed density took '
        f'{elapsed:.1f}s; live-event prune regression suspected.'
    )


# ---------------------------------------------------------------------
# Fully dead density
# ---------------------------------------------------------------------

def test_all_dead_ma_renyi2_is_zero():
    p_pitch = np.array([[60., 62., 64.]])
    p_time = np.array([[0., 1., 2.]])
    dens = _ma_density([p_pitch, p_time],
                       [np.zeros((1, 3)), np.ones((1, 3))])
    assert int(dens.live_events.sum()) == 0
    assert dens.pruned().n == 0
    assert entropy_exp_tens(dens, method='renyi2', verbose=False) == 0.0


def test_all_dead_sa_renyi2_raises():
    """An entirely zero-mass SA density is genuinely degenerate; the
    prune surfaces it as the existing non-positive-mass error rather
    than masking it.
    """
    dens = _sa_density([60., 62., 64.], [0., 0., 0.])
    assert int(dens.live_events.sum()) == 0
    with pytest.raises(FloatingPointError):
        entropy_exp_tens(dens, method='renyi2', verbose=False)
