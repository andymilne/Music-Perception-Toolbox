"""Tests for several swept attributes in ``swept_similarity`` and
``swept_entropy``: the output's dimensions, the per-attribute ``locate``
map, the window specification forms, and the guards on ``drop`` and
``target_attr``.
"""

import numpy as np
import pytest

from mpt import swept_similarity, swept_entropy, bind_events, unpack_pre_maet

SIG_P, SIG_T = 0.12, 0.05


@pytest.fixture
def flat_triple():
    rng = np.random.default_rng(1)
    N = 36
    pitch = np.sort(rng.integers(48, 84, size=N).astype(float)).reshape(1, N)
    onset = np.cumsum(rng.uniform(0.4, 0.6, size=N)).reshape(1, N)
    sweep_values = np.linspace(onset.min(), onset.max(), 8)
    return [pitch, onset], sweep_values


@pytest.fixture
def query():
    return [np.array([[60., 64., 67.]]), np.array([[0., 0.5, 1.0]])]


@pytest.mark.parametrize("normalize", ["oneSidedDenom", "cosine"])
def test_output_dimensions_follow_attribute_order(flat_triple, query, normalize):
    """One dimension per swept attribute, in attribute order, whatever the
    order the maps name them in; each slice is the one-attribute sweep."""
    p_attr, sv = flat_triple
    geom = ([SIG_P, SIG_T], [1, 1], [False, False], [False, False], [0.0, 0.0])
    pv = np.array([62.0, 64.0, 66.0])
    kw = dict(align={1: "window", 0: "both"}, drop=[1],
              window={1: {"shape": "rect", "width": 1.0},
                      0: {"shape": "rect", "width": 8.0}},
              normalize=normalize, verbose=False)
    a = swept_similarity(p_attr, None, query, None, *geom,
                         sweep={1: sv, 0: pv}, **kw)
    b = swept_similarity(p_attr, None, query, None, *geom,
                         sweep={0: pv, 1: sv}, **kw)
    assert a.shape == (pv.size, sv.size)
    assert np.array_equal(a, b)
    row = swept_similarity(p_attr, None, query, None, *geom,
                           sweep={1: sv, 0: pv[1:2]}, **kw)
    assert np.allclose(row[0], a[1], rtol=0, atol=1e-12)


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
    R = swept_similarity(
        pb, wb, qb, qw, [SIG_P, SIG_T], [1, 1], [False, False],
        [False, False], [0.0, 0.0], sweep={1: t_grid, 0: p_grid},
        align={1: "window", 0: "both"}, drop=[1],
        window={0: {"shape": "rect", "width": 5.0},
                1: {"shape": "rect", "width": 2.0}},
        specs=sb, verbose=False)
    assert R.shape == (2, 2)                  # (pitch, time)
    # copy 1 at (t=1, pitch=60.5); copy 2 at (t=11, pitch=65.5)
    assert R[0, 0] > 0.99 and R[1, 1] > 0.99
    assert R[0, 1] < 0.5 and R[1, 0] < 0.5


def test_locate_start_vs_centroid_shift():
    """'start' evaluates the window at the first onset, 'centroid' at the
    mean; they place an asymmetric pattern differently in the window."""
    pat_p = np.array([60., 63., 60., 65.]); pat_t = np.array([0., 0.3, 0.6, 3.0])
    N = 4
    pb, wb, sb = unpack_pre_maet(bind_events([pat_p.reshape(1, N), pat_t.reshape(1, N)], None,
                             [4, 4], step=1, rel_outer=[True, False]))
    qb, qw, qs = unpack_pre_maet(bind_events([pat_p.reshape(1, N), pat_t.reshape(1, N)], None,
                             [4, 4], step=1, rel_outer=[True, False]))
    # a window aligned at the centroid (mean ~0.975) finds the pattern there
    at_centroid = swept_similarity(
        pb, wb, qb, qw, [SIG_P, SIG_T], [1, 1], [True, False], [False, False],
        [0.0, 0.0], sweep={1: np.array([float(pat_t.mean())])},
        align={1: "window"}, drop=[1],
        locate="centroid", window={1: {"shape": "rect", "width": 1.0}},
        specs=sb, verbose=False)
    # with 'start', the event is located at its first onset (0.0), so a
    # window aligned there finds it
    at_start = swept_similarity(
        pb, wb, qb, qw, [SIG_P, SIG_T], [1, 1], [True, False], [False, False],
        [0.0, 0.0], sweep={1: np.array([0.0])}, align={1: "window"},
        drop=[1], locate="start", window={1: {"shape": "rect", "width": 1.0}},
        specs=sb, verbose=False)
    assert at_centroid[0] > 0.99       # event's centroid inside the window
    assert at_start[0] > 0.99          # event's start inside the window


def test_locate_map_names_the_axes_separately():
    """A per-attribute locate map picks each window attribute's rule, an attribute the map
    does not name taking 'centroid' (MATLAB: the {a, rule; ...} cell)."""
    pat_p = np.array([60., 63., 60., 65.])
    pat_t = np.array([0., 0.5, 1.5, 2.0])
    pp = np.concatenate([pat_p, pat_p + 5.0]).reshape(1, 8)
    tt = np.concatenate([pat_t, pat_t + 10.0]).reshape(1, 8)
    pb, wb, sb = unpack_pre_maet(bind_events([pp, tt], None, [4, 4], step=1))
    qb, qw, _ = unpack_pre_maet(bind_events(
        [pat_p.reshape(1, 4), pat_t.reshape(1, 4)], None, [4, 4], step=1))

    def at(locate):
        return swept_similarity(
            pb, wb, qb, qw, [SIG_P, SIG_T], [1, 1], [False, False],
            [False, False], [0.0, 0.0],
            sweep={0: np.array([62.0, 67.0]), 1: np.array([1.0, 11.0])},
            align={0: "both", 1: "window"}, drop=[1],
            window={0: {"shape": "rect", "width": 5.0},
                    1: {"shape": "rect", "width": 2.0}},
            locate=locate,
            specs=sb, verbose=False)

    map_default = at({1: "centroid"})
    map_start = at({1: "start"})
    both_start = at("start")
    assert np.allclose(map_default, at("centroid"), rtol=1e-9, atol=1e-9)
    assert np.allclose(map_start, at({0: "centroid", 1: "start"}),
                       rtol=1e-9, atol=1e-9)
    assert np.max(np.abs(map_start - both_start)) > 0.1


def test_drop_names_only_window_attributes(flat_triple, query):
    p_attr, sv = flat_triple
    with pytest.raises(ValueError, match="not a window attribute"):
        swept_similarity(
            p_attr, None, query, None, [SIG_P, SIG_T], [1, 1], [False, False],
            [False, False], [0.0, 0.0], sweep={1: sv}, align={1: "window"},
            window={1: {"shape": "rect", "width": 1.0}}, drop=[0, 1], verbose=False)


def test_drop_all_axes_errors(flat_triple, query):
    p_attr, sv = flat_triple
    with pytest.raises(ValueError):
        swept_similarity(
            p_attr, None, query, None, [SIG_P, SIG_T], [1, 1], [False, False],
            [False, False], [0.0, 0.0], sweep={0: np.array([60.]), 1: sv},
            align={0: "window", 1: "window"}, drop=[0, 1],
            window={0: {"shape": "rect", "width": 5.0},
                    1: {"shape": "rect", "width": 1.0}},
            verbose=False)


def test_target_cannot_be_dropped(flat_triple, query):
    p_attr, sv = flat_triple
    with pytest.raises(ValueError):
        swept_similarity(
            p_attr, None, query, None, [SIG_P, SIG_T], [1, 1], [False, False],
            [False, False], [0.0, 0.0], sweep={1: sv}, align={1: "window"},
            window={1: {"shape": "rect", "width": 1.0}}, drop=[1],
            target_attr=1, verbose=False)


def test_window_specification_forms_agree(flat_triple):
    """{'shape', 'width'} and {'shape', 'sd'} name the same window when the
    width is the standard deviation times 2 sqrt 3; a positional
    (shape, width) and an unknown key are refused."""
    p_attr, sv = flat_triple
    W = 2.0 * np.sqrt(3.0)
    run = lambda spec: swept_entropy(
        p_attr, None, [SIG_P, SIG_T], [1, 1], [False, False], [False, False],
        [0.0, 0.0], sweep={1: sv}, window={1: spec}, drop=[1],
        method="renyi2", verbose=False)
    a = run({"shape": "rect", "width": W})
    assert np.array_equal(a, run({"shape": 1.0, "width": W}))
    assert np.allclose(a, run({"shape": 1.0, "sd": 1.0}), rtol=0, atol=1e-12)
    for bad in [(1.0, W), ("rect", W, "closed"), ["gaussian", 1.0]]:
        with pytest.raises(ValueError, match="name the window's scale"):
            run(bad)
    with pytest.raises(ValueError, match="unknown key"):
        run({"shape": "gaussian", "decayRate": 1.0})
