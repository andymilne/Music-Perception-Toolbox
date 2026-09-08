"""Tests for the pre-MAET ``windowed_similarity`` and ``windowed_entropy``.

Each new-function result is checked against the equivalent inline pipeline
(``weight_events`` / ``translate_attributes`` / ``build_exp_tens`` /
``entropy_exp_tens`` / ``cos_sim_exp_tens``) it replaces, so the functions
are pinned to the hand-written composition rather than to remembered
numbers. Both argument surfaces are exercised: the single-axis form
(``window_attr`` / ``centres`` / ``drop_window_attr``, plus the decoupled
``query_centres`` correlogram) and the multi-axis form (``sweep`` / ``drop``).
"""

import numpy as np
import pytest

from mpt import (
    windowed_similarity, windowed_entropy,
    weight_events, translate_attributes, bind_events,
    unpack_pre_maet,
    build_exp_tens, entropy_exp_tens, cos_sim_exp_tens,
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
# windowed_entropy
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
        dens = build_exp_tens(pw, ww, [SIG_P], [1], [False], [False], [0.0], verbose=False)
        ref[i] = entropy_exp_tens(dens, method=method, verbose=False)
    got = windowed_entropy(p_attr, None, [SIG_P, SIG_T], [1, 1], [False, False],
                           [False, False], [0.0, 0.0], centres,
                           context_window=(shape, W_WIDTH), method=method,
                           window_attr=1, drop_window_attr=True, verbose=False)
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_entropy_retain_axis(triple):
    p_attr, centres = triple
    ref = np.empty(len(centres))
    for i, c in enumerate(centres):
        pw, ww, _ = unpack_pre_maet(weight_events(p_attr, None, 1, 0, float(c), 1.0,
                                  is_per=False, period=0.0, width=W_WIDTH, drop_input_attr=False))
        dens = build_exp_tens(pw, ww, [SIG_P, SIG_T], [1, 1], [False, False],
                              [False, False], [0.0, 0.0], verbose=False)
        ref[i] = entropy_exp_tens(dens, method="renyi2", verbose=False)
    got = windowed_entropy(p_attr, None, [SIG_P, SIG_T], [1, 1], [False, False],
                           [False, False], [0.0, 0.0], centres,
                           context_window=(1.0, W_WIDTH), method="renyi2",
                           window_attr=1, drop_window_attr=False, verbose=False)
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-9)


def test_entropy_drop_r_ge_2_now_allowed(triple):
    """Dropping a bundled axis is allowed: the centroid `locate` reduces it
    for the window, then it is removed from the density (deletion is no
    longer restricted to r = 1)."""
    p_attr, centres = triple
    got = windowed_entropy(p_attr, None, [SIG_P, SIG_T], [1, 2], [False, False],
                           [False, False], [0.0, 0.0], centres,
                           context_window=(1.0, W_WIDTH), window_attr=1,
                           drop_window_attr=True, verbose=False)
    assert got.shape == (len(centres),)
    assert np.all(np.isfinite(got))


def test_entropy_marginalise_not_implemented(triple):
    """marginalise (integrate out a retained axis) is reserved but unimplemented."""
    p_attr, centres = triple
    with pytest.raises(NotImplementedError):
        windowed_entropy(p_attr, None, [SIG_P, SIG_T], [1, 1], [False, False],
                         [False, False], [0.0, 0.0], centres,
                         context_window=(1.0, W_WIDTH), window_attr=1,
                         drop_window_attr=False, marginalise=0, verbose=False)


def test_entropy_requires_width(triple):
    """No query means no extent to size a default window from, so an explicit
    width is required."""
    p_attr, centres = triple
    with pytest.raises(ValueError):
        windowed_entropy(p_attr, None, [SIG_P, SIG_T], [1, 1], [False, False],
                         [False, False], [0.0, 0.0], centres,
                         context_window=(1.0, None), window_attr=1,
                         drop_window_attr=False, verbose=False)


# --------------------------------------------------------------------------
# windowed_similarity: single-axis surface
# --------------------------------------------------------------------------
@pytest.mark.parametrize("normalize", ["oneSidedDenom", "cosine"])
def test_similarity_locked(triple, query, normalize):
    p_attr, centres = triple
    qv = query[1].ravel(); q_ext = float(qv.max() - qv.min()); mu_q = float(qv.mean())
    ref = np.empty(len(centres))
    for i, c in enumerate(centres):
        pc, wc, _ = unpack_pre_maet(weight_events(p_attr, None, 1, 0, float(c), 1.0,
                                  width=q_ext, is_per=False, period=0.0, drop_input_attr=False))
        pq, wq, _ = unpack_pre_maet(translate_attributes(query, None, [None, np.array([[c - mu_q]])]))
        ref[i] = cos_sim_exp_tens(pc, wc, pq, wq, [SIG_P, SIG_T], [1, 1],
                                  [False, False], [False, False], [0.0, 0.0],
                                  normalize=normalize, verbose=False)
    got = windowed_similarity(p_attr, None, query, None, [SIG_P, SIG_T], [1, 1],
                              [False, False], [False, False], [0.0, 0.0], centres,
                              normalize=normalize, window_attr=1, drop_window_attr=False, verbose=False)
    assert got.shape == (len(centres),)
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-9)


def test_similarity_decoupled_correlogram(triple, query):
    """2-D ``query_centres`` fixes the window at each anchor while the query
    slides across the lags: the anchor x lag correlogram surface."""
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
            ref[ia, it] = cos_sim_exp_tens(pc, None, pq, wq, [SIG_P, SIG_T], [1, 1],
                                           [False, False], [False, False], [0.0, 0.0],
                                           normalize="oneSidedDenom", verbose=False)
    qc2d = anchors[:, None] - tau[None, :]
    got = windowed_similarity(p_attr, None, query, None, [SIG_P, SIG_T], [1, 1],
                              [False, False], [False, False], [0.0, 0.0], anchors,
                              query_centres=qc2d, context_window=(1.0, 2 * half),
                              normalize="oneSidedDenom", window_attr=1, drop_window_attr=False, verbose=False)
    assert got.shape == (len(anchors), len(tau))
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_similarity_generative_sweep_matches_explicit(triple, query):
    p_attr, _ = triple
    onset = p_attr[1]
    explicit = np.arange(onset.min(), onset.max() + 1e-9, 1.0)
    g_exp = windowed_similarity(p_attr, None, query, None, [SIG_P, SIG_T], [1, 1],
                                [False, False], [False, False], [0.0, 0.0], explicit,
                                normalize="oneSidedDenom", window_attr=1, drop_window_attr=False, verbose=False)
    g_gen = windowed_similarity(p_attr, None, query, None, [SIG_P, SIG_T], [1, 1],
                                [False, False], [False, False], [0.0, 0.0],
                                start=float(onset.min()), stop=float(onset.max()), step=1.0,
                                normalize="oneSidedDenom", window_attr=1, drop_window_attr=False, verbose=False)
    assert g_gen.shape == g_exp.shape
    assert np.allclose(g_gen, g_exp, rtol=1e-9, atol=1e-9)


def test_centres_and_generative_mutually_exclusive(triple, query):
    p_attr, centres = triple
    with pytest.raises(ValueError):
        windowed_similarity(p_attr, None, query, None, [SIG_P, SIG_T], [1, 1],
                            [False, False], [False, False], [0.0, 0.0], centres,
                            step=1.0, window_attr=1, drop_window_attr=False, verbose=False)


# --------------------------------------------------------------------------
# windowed_similarity: multi-axis surface and locate
# --------------------------------------------------------------------------
def test_single_axis_equals_one_entry_sweep(triple, query):
    """The single-axis surface is exactly the one-entry multi-axis form."""
    p_attr, centres = triple
    single = windowed_similarity(p_attr, None, query, None, [SIG_P, SIG_T], [1, 1],
                                 [False, False], [False, False], [0.0, 0.0], centres,
                                 normalize="oneSidedDenom", window_attr=1,
                                 drop_window_attr=False, verbose=False)
    multi = windowed_similarity(p_attr, None, query, None, [SIG_P, SIG_T], [1, 1],
                                [False, False], [False, False], [0.0, 0.0],
                                sweep={1: centres}, drop={1: False},
                                normalize="oneSidedDenom", verbose=False)
    assert np.allclose(single, multi, rtol=1e-12, atol=1e-12)


def test_multi_axis_two_dim_map(triple, query):
    """Sweeping pitch (compared) and time (dropped) yields a 2-D map."""
    p_attr, centres = triple
    pitch = p_attr[0]
    p_centroid = float(query[0].mean())
    pgrid = p_centroid + np.arange(-4.0, 5.0, 2.0)
    R = windowed_similarity(p_attr, None, query, None, [SIG_P, SIG_T], [1, 1],
                            [False, False], [False, False], [0.0, 0.0],
                            sweep={1: centres, 0: pgrid}, drop={1: True, 0: False},
                            context_window={0: {"shape": "rect", "width": 6.0}},
                            normalize="oneSidedDenom", verbose=False)
    assert R.shape == (len(centres), len(pgrid))
    assert np.all(np.isfinite(R))


def test_locate_is_wired():
    """On an asymmetric bundle, 'centroid' and 'start' place the window (and
    the query translation) differently, so the self-similarity sweep peaks at
    different centres and the two arrays must not coincide."""
    qp = np.array([[60., 64.]]); qo = np.array([[0.0, 0.6]])   # centroid 0.3, start 0.0
    qb, qw, qs = unpack_pre_maet(bind_events([qp, qo], None, [1, 2], step=1, rel_outer=[False, False]))
    centres = np.linspace(-0.4, 0.7, 12)        # keeps both window centres non-empty
    common = dict(window_attr=1, drop_window_attr=False, context_window=(1.0, 1.5),
                  normalize="cosine", specs=qs, verbose=False)
    a = windowed_similarity(qb, qw, qb, qw, [SIG_P, SIG_T], [1, 2], [False, False],
                            [False, False], [0.0, 0.0], centres, locate="centroid", **common)
    b = windowed_similarity(qb, qw, qb, qw, [SIG_P, SIG_T], [1, 2], [False, False],
                            [False, False], [0.0, 0.0], centres, locate="start", **common)
    assert a.shape == b.shape == (len(centres),)
    # both attain a clear self-similarity peak, but at different centres
    assert np.nanmax(a) > 0.9 and np.nanmax(b) > 0.9
    assert abs(centres[np.nanargmax(a)] - centres[np.nanargmax(b)]) > 0.2
    assert not np.allclose(a, b, equal_nan=True)


def _qw_setup():
    """Asymmetric query: two close events plus one far one, so a narrow query
    window genuinely reshapes it rather than rescaling it uniformly."""
    pc = [np.array([[0., 100., 200., 300., 400., 500.]]),
          np.array([[0., 1., 2., 3., 4., 5.]])]
    wc = [np.ones((1, 6)), np.ones((1, 6))]
    pq = [np.array([[0., 100., 500.]]), np.array([[0., 1., 4.]])]
    wq = [np.ones((1, 3)), np.ones((1, 3))]
    common = dict(centres=np.arange(6.0), window_attr=1, drop_window_attr=False,
                  normalize="cosine", verbose=False)
    run = lambda **kw: np.asarray(windowed_similarity(
        pc, wc, pq, wq, [30.0, 0.25], [1, 1], [False, False], [False, False],
        [0.0, 0.0], **common, **kw))
    return run


def test_query_window_is_applied():
    """A query window must reshape the query. Regression: the argument was
    accepted, validated, and threaded to the worker, but never read, so a
    caller asking for a windowed query silently got an unwindowed one."""
    run = _qw_setup()
    plain = run(context_window=("gauss", 2.0))
    narrow = run(context_window=("gauss", 2.0), query_window=("gauss", 0.8))
    assert not np.allclose(plain, narrow)
    # Suppressing the far query event leaves the near pair, which matches the
    # context far better at its best offset.
    assert np.nanmax(narrow) > np.nanmax(plain)
    assert narrow.min() >= 0.0 and narrow.max() <= 1.0 + 1e-12


def test_wide_rect_query_window_is_an_exact_no_op():
    """A rectangle wider than the query's own extent includes every event at
    unit weight, so it must reproduce the unwindowed profile exactly."""
    run = _qw_setup()
    plain = run(context_window=("gauss", 2.0))
    wide = run(context_window=("gauss", 2.0), query_window=("rect", 1000.0))
    assert np.array_equal(plain, wide)


def test_gaussian_query_window_converges_to_no_op():
    """Widening a Gaussian query window must approach the unwindowed profile,
    and at the O(1/sd^2) rate set by the window's curvature across the query."""
    run = _qw_setup()
    plain = run(context_window=("gauss", 2.0))
    d = [float(np.max(np.abs(run(context_window=("gauss", 2.0),
                                 query_window=("gauss", sd)) - plain)))
         for sd in (100.0, 1000.0, 10000.0)]
    assert d[0] > d[1] > d[2]
    for lo, hi in zip(d[1:], d[:-1]):
        assert 50.0 < hi / lo < 200.0
