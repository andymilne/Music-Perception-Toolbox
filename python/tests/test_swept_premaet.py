"""Tests for the pre-MAET ``swept_similarity`` and ``swept_entropy``.

Each new-function result is checked against the equivalent inline pipeline
(``weight_events`` / ``translate_attributes`` / ``build_maet`` /
``entropy_maet`` / ``sim_maet``) it replaces, so the functions
are pinned to the hand-written composition rather than to remembered
numbers. Each role of ``align`` is exercised, alone and together.
"""

import numpy as np
import pytest

from mpt import (
    swept_similarity, swept_entropy,
    weight_events, translate_attributes, bind_events,
    unpack_pre_maet,
    build_maet, entropy_maet, sim_maet,
)

SD = 1.0
W_WIDTH = 2.0 * np.sqrt(3.0) * SD          # variance-matched rectangular support
SIG_P, SIG_T = 0.12, 0.05


@pytest.fixture
def triple():
    rng = np.random.default_rng(0)
    N = 40
    pitch = np.sort(rng.integers(48, 84, size=N).astype(float)).reshape(1, N)
    onset = np.cumsum(rng.uniform(0.4, 0.6, size=N)).reshape(1, N)
    return [pitch, onset], np.linspace(onset.min(), onset.max(), 9)


@pytest.fixture
def query():
    return [np.array([[60., 64., 67.]]), np.array([[0., 0.5, 1.0]])]


# --------------------------------------------------------------------------
# swept_entropy
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method,shape", [
    ("differential", 0.0), ("renyi2", 0.0), ("renyi2", 1.0),
])
def test_entropy_drop_window_axis(triple, method, shape):
    p_attr, centres = triple
    kw = {"width": W_WIDTH} if shape == 1.0 else {"sd": SD}
    ref = np.empty(len(centres))
    for i, c in enumerate(centres):
        pw, ww, _ = unpack_pre_maet(weight_events(p_attr, None, 1, 0, float(c), shape,
                                  is_per=False, period=0.0, drop_input_attr=True, **kw))
        dens = build_maet(pw, ww, [SIG_P], [1], [False], [False], [0.0], verbose=False)
        ref[i] = entropy_maet(dens, method=method, verbose=False)
    got = swept_entropy(p_attr, None, [SIG_P, SIG_T], [1, 1], [False, False],
                        [False, False], [0.0, 0.0], sweep={1: centres},
                        window={1: (shape, W_WIDTH)}, drop=[1],
                        method=method, verbose=False)
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("method", ["shannon", "normalized"])
def test_entropy_grid_methods_take_the_grid(triple, method):
    """The discrete methods take their grid through swept_entropy, the same
    grid at every sweep value, and match the composition."""
    p_attr, centres = triple
    grid = dict(n_points_per_dim=400, x_min=40.0, x_max=90.0)
    ref = np.empty(len(centres))
    for i, c in enumerate(centres):
        pw, ww, _ = unpack_pre_maet(weight_events(
            p_attr, None, 1, 0, float(c), 0.0, is_per=False, period=0.0,
            sd=SD, drop_input_attr=True))
        dens = build_maet(pw, ww, [SIG_P], [1], [False], [False], [0.0],
                          verbose=False)
        ref[i] = entropy_maet(dens, method=method, verbose=False, **grid)
    got = swept_entropy(p_attr, None, [SIG_P, SIG_T], [1, 1], [False, False],
                        [False, False], [0.0, 0.0], sweep={1: centres},
                        window={1: (0.0, W_WIDTH)}, drop=[1],
                        method=method, verbose=False, **grid)
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-9)
    with pytest.raises(TypeError, match="n_points_per_dim"):
        swept_entropy(p_attr, None, [SIG_P, SIG_T], [1, 1], [False, False],
                      [False, False], [0.0, 0.0], sweep={1: centres},
                      window={1: (0.0, W_WIDTH)}, drop=[1], method=method,
                      verbose=False)


def test_entropy_retain_axis(triple):
    p_attr, centres = triple
    ref = np.empty(len(centres))
    for i, c in enumerate(centres):
        pw, ww, _ = unpack_pre_maet(weight_events(p_attr, None, 1, 0, float(c), 1.0,
                                  is_per=False, period=0.0, width=W_WIDTH, drop_input_attr=False))
        dens = build_maet(pw, ww, [SIG_P, SIG_T], [1, 1], [False, False],
                              [False, False], [0.0, 0.0], verbose=False)
        ref[i] = entropy_maet(dens, method="renyi2", verbose=False)
    got = swept_entropy(p_attr, None, [SIG_P, SIG_T], [1, 1], [False, False],
                        [False, False], [0.0, 0.0], sweep={1: centres},
                        window={1: (1.0, W_WIDTH)}, method="renyi2",
                        verbose=False)
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-9)


def test_entropy_drop_r_ge_2_now_allowed(triple):
    """Dropping a bundled attribute is allowed: the centroid `locate` reduces it
    for the window, then it is removed from the density (deletion is no
    longer restricted to r = 1)."""
    p_attr, centres = triple
    got = swept_entropy(p_attr, None, [SIG_P, SIG_T], [1, 2], [False, False],
                        [False, False], [0.0, 0.0], sweep={1: centres},
                        window={1: (1.0, W_WIDTH)}, drop=[1], verbose=False)
    assert got.shape == (len(centres),)
    assert np.all(np.isfinite(got))


def test_entropy_requires_width(triple):
    """Every window states its width, and every swept attribute its window."""
    p_attr, centres = triple
    with pytest.raises(ValueError, match="width is required"):
        swept_entropy(p_attr, None, [SIG_P, SIG_T], [1, 1], [False, False],
                      [False, False], [0.0, 0.0], sweep={1: centres},
                      window={1: (1.0, None)}, verbose=False)
    with pytest.raises(ValueError, match="window"):
        swept_entropy(p_attr, None, [SIG_P, SIG_T], [1, 1], [False, False],
                      [False, False], [0.0, 0.0], sweep={1: centres},
                      verbose=False)


# --------------------------------------------------------------------------
# swept_similarity: one swept attribute
# --------------------------------------------------------------------------
# A half-open rectangle exactly as wide as the query, pinned to the
# composition it stands for; the warning that it leaves out the query's
# last event is expected.
@pytest.mark.filterwarnings("ignore:attribute 1. the window")
@pytest.mark.parametrize("normalize", ["oneSidedDenom", "cosine"])
def test_similarity_locked(triple, query, normalize):
    p_attr, centres = triple
    qv = query[1].ravel(); q_ext = float(qv.max() - qv.min()); mu_q = float(qv.mean())
    ref = np.empty(len(centres))
    for i, c in enumerate(centres):
        pc, wc, _ = unpack_pre_maet(weight_events(p_attr, None, 1, 0, float(c), 1.0,
                                  width=q_ext, is_per=False, period=0.0, drop_input_attr=False))
        pq, wq, _ = unpack_pre_maet(translate_attributes(query, None, [None, np.array([[c - mu_q]])]))
        ref[i] = sim_maet(pc, wc, pq, wq, [SIG_P, SIG_T], [1, 1],
                                  [False, False], [False, False], [0.0, 0.0],
                                  normalize=normalize, verbose=False)
    got = swept_similarity(p_attr, None, query, None, [SIG_P, SIG_T], [1, 1],
                           [False, False], [False, False], [0.0, 0.0],
                           sweep={1: centres}, align={1: "both"},
                           window={1: ("rect", q_ext)},
                           normalize=normalize, verbose=False)
    assert got.shape == (len(centres),)
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-9)


def test_similarity_decoupled_correlogram(triple, query):
    """``align='independent'`` fixes the window at each anchor while the
    query slides across the lags, its placements measured from each anchor
    (one row of query values per window value): the anchor x lag
    correlogram surface."""
    p_attr, centres = triple
    pitch, onset = p_attr
    mu_q = float(query[1].mean())
    anchors = centres[1:8]
    tau = np.linspace(-1.0, 1.0, 9)
    half = 1.5
    ref = np.full((len(anchors), len(tau)), np.nan)
    for ia, a in enumerate(anchors):
        keep = np.abs(onset.ravel() - a) <= half
        pc = [pitch[:, keep], onset[:, keep]]
        for it, t in enumerate(tau):
            pq, wq, _ = unpack_pre_maet(translate_attributes(query, None, [None, np.array([[(a - t) - mu_q]])]))
            ref[ia, it] = sim_maet(pc, None, pq, wq, [SIG_P, SIG_T], [1, 1],
                                           [False, False], [False, False], [0.0, 0.0],
                                           normalize="oneSidedDenom", verbose=False)
    qc2d = anchors[:, None] - tau[None, :]
    got = swept_similarity(p_attr, None, query, None, [SIG_P, SIG_T], [1, 1],
                           [False, False], [False, False], [0.0, 0.0],
                           sweep={1: (anchors, qc2d)},
                           align={1: "independent"},
                           window={1: (1.0, 2 * half)},
                           normalize="oneSidedDenom", verbose=False)
    assert got.shape == (len(anchors), len(tau))
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-9, equal_nan=True)


@pytest.mark.filterwarnings("ignore:attribute 1. the window")
def test_similarity_generative_sweep_matches_explicit(triple, query):
    p_attr, _ = triple
    onset = p_attr[1]
    # A window-only role: its generated sweep values default to the
    # context's extent, stepped at half the window's sd.
    common = dict(align={1: "window"}, drop=[1], window={1: ("rect", 1.0)},
                  normalize="oneSidedDenom", verbose=False)
    geom = ([SIG_P, SIG_T], [1, 1], [False, False], [False, False], [0.0, 0.0])
    explicit = np.arange(onset.min(), onset.max() + 1e-9, 1.0)
    g_exp = swept_similarity(p_attr, None, query, None, *geom,
                             sweep={1: explicit}, **common)
    g_gen = swept_similarity(p_attr, None, query, None, *geom,
                             start={1: float(onset.min())},
                             stop={1: float(onset.max())}, step={1: 1.0},
                             **common)
    assert g_gen.shape == g_exp.shape
    assert np.allclose(g_gen, g_exp, rtol=1e-9, atol=1e-9)
    # start and stop default to the context's extent on the attribute
    g_def = swept_similarity(p_attr, None, query, None, *geom,
                             step={1: 1.0}, **common)
    assert np.array_equal(g_def, g_gen)
    # step defaults to half the window's sd (a Gaussian window of the width
    # of a rectangle of width 1 has sd 1 / (2 sqrt 3)); a pure rectangle
    # takes its pieces instead (test_swept_sweep_values)
    gauss = {**common, "window": {1: (0.0, 1.0)}}
    st = 1.0 / (2.0 * np.sqrt(3.0)) / 2.0
    n = int(np.floor((onset.max() - onset.min()) / st + 1e-9)) + 1
    g_half = swept_similarity(p_attr, None, query, None, *geom,
                              sweep={1: onset.min() + st * np.arange(n)},
                              **gauss)
    g_dflt = swept_similarity(p_attr, None, query, None, *geom, **{
        **gauss, "start": {1: float(onset.min())}})
    assert np.array_equal(g_dflt, g_half)


@pytest.mark.filterwarnings("ignore:attribute 1. the window")
def test_sweep_and_generator_mutually_exclusive(triple, query):
    p_attr, centres = triple
    with pytest.raises(ValueError):
        swept_similarity(p_attr, None, query, None, [SIG_P, SIG_T], [1, 1],
                         [False, False], [False, False], [0.0, 0.0],
                         sweep={1: centres}, step={1: 1.0},
                         align={1: "both"}, window={1: ("rect", 1.0)},
                         verbose=False)


# --------------------------------------------------------------------------
# swept_similarity: several swept attributes, and locate
# --------------------------------------------------------------------------


def test_multi_axis_two_dim_map(triple, query):
    """Sweeping pitch (compared) and time (dropped) yields a 2-D map."""
    p_attr, centres = triple
    pitch = p_attr[0]
    p_centroid = float(query[0].mean())
    pgrid = p_centroid + np.arange(-4.0, 5.0, 2.0)
    R = swept_similarity(p_attr, None, query, None, [SIG_P, SIG_T], [1, 1],
                         [False, False], [False, False], [0.0, 0.0],
                         sweep={1: centres, 0: pgrid},
                         align={1: "window", 0: "both"}, drop=[1],
                         window={0: {"shape": "rect", "width": 16.0},
                                    1: ("rect", 1.0)},
                         normalize="oneSidedDenom", verbose=False)
    # one dimension per swept attribute, in attribute order
    assert R.shape == (len(pgrid), len(centres))
    assert np.all(np.isfinite(R))


def test_locate_is_wired():
    """On an asymmetric bundle, 'centroid' and 'start' place the window (and
    the query translation) differently, so the self-similarity sweep peaks at
    different centres and the two arrays must not coincide."""
    qp = np.array([[60., 64.]]); qo = np.array([[0.0, 0.6]])   # centroid 0.3, start 0.0
    qb, qw, qs = unpack_pre_maet(bind_events([qp, qo], None, [1, 2], step=1, rel_outer=[False, False]))
    centres = np.linspace(-0.4, 0.7, 12)        # keeps both window centres non-empty
    common = dict(sweep={1: centres}, align={1: "both"},
                  window={1: (1.0, 1.5)}, normalize="cosine", specs=qs,
                  verbose=False)
    a = swept_similarity(qb, qw, qb, qw, [SIG_P, SIG_T], [1, 2], [False, False],
                         [False, False], [0.0, 0.0], locate="centroid", **common)
    b = swept_similarity(qb, qw, qb, qw, [SIG_P, SIG_T], [1, 2], [False, False],
                         [False, False], [0.0, 0.0], locate="start", **common)
    assert a.shape == b.shape == (len(centres),)
    # both attain a clear self-similarity peak, but at different centres
    assert np.nanmax(a) > 0.9 and np.nanmax(b) > 0.9
    assert abs(centres[np.nanargmax(a)] - centres[np.nanargmax(b)]) > 0.2
    assert not np.allclose(a, b, equal_nan=True)


def test_query_windowed_with_weight_events():
    """The documented route for a window on the query: weight_events on the
    query before the call. The query is asymmetric (two close events plus a
    far one), so a narrow window around the close pair reshapes it rather
    than rescaling it, and the reshaped query matches the context better."""
    pc = [np.array([[0., 100., 200., 300., 400., 500.]]),
          np.array([[0., 1., 2., 3., 4., 5.]])]
    wc = [np.ones((1, 6)), np.ones((1, 6))]
    pq = [np.array([[0., 100., 500.]]), np.array([[0., 1., 4.]])]
    wq = [np.ones((1, 3)), np.ones((1, 3))]
    run = lambda q, w: np.asarray(swept_similarity(
        pc, wc, q, w, [30.0, 0.25], [1, 1], [False, False], [False, False],
        [0.0, 0.0], sweep={1: np.arange(6.0)}, align={1: "both"},
        window={1: ("gauss", 2.0)}, normalize="cosine", verbose=False))
    pq_w, wq_w, _ = unpack_pre_maet(weight_events(
        pq, wq, 1, 0, 0.5, 0.0, sd=0.8, drop_input_attr=False))
    plain, narrow = run(pq, wq), run(pq_w, wq_w)
    assert not np.allclose(plain, narrow)
    assert np.nanmax(narrow) > np.nanmax(plain)
    assert narrow.min() >= 0.0 and narrow.max() <= 1.0 + 1e-12
