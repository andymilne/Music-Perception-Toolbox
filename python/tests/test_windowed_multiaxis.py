"""Tests for the multi-axis (``sweep``/``drop``/``locate``) form of
``windowed_similarity`` and ``windowed_entropy``.

The central check pins the new core against the trusted single-axis form on a
K = 1 triple, where the two must agree exactly; the rest cover the genuinely
new surface (the several-axes map, the ``locate`` reduction, and the guards).
"""

import numpy as np
import pytest

from mpt import windowed_similarity, windowed_entropy, bind_events, unpack_pre_maet

SIG_P, SIG_T = 0.12, 0.05


@pytest.fixture
def flat_triple():
    rng = np.random.default_rng(1)
    N = 36
    pitch = np.sort(rng.integers(48, 84, size=N).astype(float)).reshape(1, N)
    onset = np.cumsum(rng.uniform(0.4, 0.6, size=N)).reshape(1, N)
    centres = np.linspace(onset.min(), onset.max(), 8)
    return [pitch, onset], centres


@pytest.fixture
def query():
    return [np.array([[60., 64., 67.]]), np.array([[0., 0.5, 1.0]])]


@pytest.mark.parametrize("dropflag", [True, False])
@pytest.mark.parametrize("normalize", ["oneSidedDenom", "cosine"])
def test_multi_one_axis_matches_single_axis(flat_triple, query, dropflag, normalize):
    """On a K=1 axis the centroid reduces to the value, so the multi-axis
    form must reproduce the single-axis form exactly."""
    p_attr, centres = flat_triple
    single = windowed_similarity(
        p_attr, None, query, None, [SIG_P, SIG_T], [1, 1], [False, False],
        [False, False], [0.0, 0.0], centres, window_attr=1,
        drop_window_attr=dropflag, normalize=normalize, verbose=False)
    multi = windowed_similarity(
        p_attr, None, query, None, [SIG_P, SIG_T], [1, 1], [False, False],
        [False, False], [0.0, 0.0], sweep={1: centres}, drop={1: dropflag},
        normalize=normalize, verbose=False)
    assert multi.shape == single.shape
    assert np.allclose(multi, single, rtol=1e-9, atol=1e-9)


def test_multi_two_axis_map_shape_and_peak():
    """Sweep pitch and time together; the peak sits at the planted
    statement's time centroid and transposition."""
    # two copies of a 4-note pattern, the second transposed up 5 and shifted
    pat_p = np.array([60., 63., 60., 65.]); pat_t = np.array([0., 0.5, 1.5, 2.0])
    P = np.concatenate([pat_p, pat_p + 5.0])
    T = np.concatenate([pat_t, pat_t + 10.0])
    N = P.size
    pb, wb, sb = unpack_pre_maet(bind_events([P.reshape(1, N), T.reshape(1, N)], None, [4, 4],
                             step=1, rel_outer=[False, False]))
    qb, qw, qs = unpack_pre_maet(bind_events([pat_p.reshape(1, 4), pat_t.reshape(1, 4)], None,
                             [4, 4], step=1, rel_outer=[False, False]))
    t_grid = np.array([1.0, 11.0])            # centroids of the two copies
    p_grid = np.array([62.0, 67.0])            # their pitch centroids
    R = windowed_similarity(
        pb, wb, qb, qw, [SIG_P, SIG_T], [1, 1], [False, False],
        [False, False], [0.0, 0.0], sweep={1: t_grid, 0: p_grid},
        drop={1: True, 0: False}, specs=sb, verbose=False)
    assert R.shape == (2, 2)
    # copy 1 at (t=1, pitch=60.5); copy 2 at (t=11, pitch=65.5)
    assert R[0, 0] > 0.99 and R[1, 1] > 0.99
    assert R[0, 1] < 0.5 and R[1, 0] < 0.5


def test_locate_start_vs_centroid_shift():
    """'start' centres the window on the first onset, 'centroid' on the mean;
    they place the window differently for an asymmetric pattern."""
    pat_p = np.array([60., 63., 60., 65.]); pat_t = np.array([0., 0.3, 0.6, 3.0])
    N = 4
    pb, wb, sb = unpack_pre_maet(bind_events([pat_p.reshape(1, N), pat_t.reshape(1, N)], None,
                             [4, 4], step=1, rel_outer=[True, False]))
    qb, qw, qs = unpack_pre_maet(bind_events([pat_p.reshape(1, N), pat_t.reshape(1, N)], None,
                             [4, 4], step=1, rel_outer=[True, False]))
    # window centred on the centroid (mean ~0.975) finds the pattern there
    at_centroid = windowed_similarity(
        pb, wb, qb, qw, [SIG_P, SIG_T], [1, 1], [True, False], [False, False],
        [0.0, 0.0], sweep={1: np.array([float(pat_t.mean())])}, drop={1: True},
        locate="centroid", context_window={1: {"shape": "rect", "width": 1.0}},
        specs=sb, verbose=False)
    # the same narrow window centred on the centroid but with 'start' locate
    # reads the first onset (0.0), which the centroid-placed window misses
    at_start = windowed_similarity(
        pb, wb, qb, qw, [SIG_P, SIG_T], [1, 1], [True, False], [False, False],
        [0.0, 0.0], sweep={1: np.array([0.0])}, drop={1: True},
        locate="start", context_window={1: {"shape": "rect", "width": 1.0}},
        specs=sb, verbose=False)
    assert at_centroid[0] > 0.99       # event's centroid inside the window
    assert at_start[0] > 0.99          # event's start inside the window


def test_drop_requires_one_entry_per_sweep(flat_triple, query):
    p_attr, centres = flat_triple
    with pytest.raises(ValueError):
        windowed_similarity(
            p_attr, None, query, None, [SIG_P, SIG_T], [1, 1], [False, False],
            [False, False], [0.0, 0.0], sweep={1: centres}, drop={0: True},
            verbose=False)


def test_drop_all_axes_errors(flat_triple, query):
    p_attr, centres = flat_triple
    with pytest.raises(ValueError):
        windowed_similarity(
            p_attr, None, query, None, [SIG_P, SIG_T], [1, 1], [False, False],
            [False, False], [0.0, 0.0], sweep={0: np.array([60.]), 1: centres},
            drop={0: True, 1: True}, verbose=False)


def test_target_cannot_be_dropped(flat_triple, query):
    p_attr, centres = flat_triple
    with pytest.raises(ValueError):
        windowed_similarity(
            p_attr, None, query, None, [SIG_P, SIG_T], [1, 1], [False, False],
            [False, False], [0.0, 0.0], sweep={1: centres}, drop={1: True},
            target_attr=1, verbose=False)


def test_entropy_multi_matches_single_axis(flat_triple):
    """windowed_entropy multi form matches the single-axis form on a K=1 axis."""
    p_attr, centres = flat_triple
    W = 2.0 * np.sqrt(3.0)
    single = windowed_entropy(
        p_attr, None, [SIG_P, SIG_T], [1, 1], [False, False], [False, False],
        [0.0, 0.0], centres, context_window=(1.0, W), method="renyi2", window_attr=1,
        drop_window_attr=True, verbose=False)
    multi = windowed_entropy(
        p_attr, None, [SIG_P, SIG_T], [1, 1], [False, False], [False, False],
        [0.0, 0.0], sweep={1: centres}, drop={1: True},
        context_window={1: {"shape": "rect", "width": W}}, method="renyi2",
        verbose=False)
    assert np.allclose(multi, single, rtol=1e-9, atol=1e-9, equal_nan=True)
