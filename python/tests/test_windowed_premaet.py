"""Tests for the pre-MAET ``windowed_similarity`` and ``windowed_entropy``.

Each new-function result is checked against the equivalent inline pipeline
(``weight_events`` / ``translate_attributes`` / ``build_exp_tens`` /
``entropy_exp_tens`` / ``cos_sim_exp_tens``) it replaces, so the functions
are pinned to the hand-written composition rather than to remembered
numbers.
"""

import numpy as np
import pytest

from mpt import (
    windowed_similarity, windowed_entropy,
    weight_events, translate_attributes,
    build_exp_tens, entropy_exp_tens, cos_sim_exp_tens,
)

SD = 1.0
W_WIDTH = 2.0 * np.sqrt(3.0) * SD          # variance-matched rectangular support
SIG_P, SIG_T = 0.12, 0.05


@pytest.fixture
def carrier():
    rng = np.random.default_rng(0)
    N = 40
    pitch = np.sort(rng.integers(48, 84, size=N).astype(float)).reshape(1, N)
    onset = np.cumsum(rng.uniform(0.4, 0.6, size=N)).reshape(1, N)
    return [pitch, onset], np.linspace(onset.min(), onset.max(), 9)


@pytest.fixture
def query():
    return [np.array([[60., 64., 67.]]), np.array([[0., 0.5, 1.0]])]


@pytest.mark.parametrize("method,shape", [
    ("differential", 0.0), ("renyi2", 0.0), ("renyi2", 1.0),
])
def test_entropy_drop_window_axis(carrier, method, shape):
    p_attr, centres = carrier
    kw = {"width": W_WIDTH} if shape == 1.0 else {"sd": SD}
    ref = np.empty(len(centres))
    for i, c in enumerate(centres):
        pw, ww, _ = weight_events(p_attr, None, 1, 0, float(c), shape,
                                  is_per=False, period=0.0, drop_input_attr=True, **kw)
        dens = build_exp_tens(pw, ww, [SIG_P], [1], [False], [False], [0.0], verbose=False)
        ref[i] = entropy_exp_tens(dens, method=method, verbose=False)
    got = windowed_entropy(p_attr, None, [SIG_P, SIG_T], [1, 1], [False, False],
                           [False, False], [0.0, 0.0], centres,
                           window=(shape, W_WIDTH), method=method,
                           window_attr=1, drop_window_attr=True, verbose=False)
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_entropy_retain_axis(carrier):
    p_attr, centres = carrier
    ref = np.empty(len(centres))
    for i, c in enumerate(centres):
        pw, ww, _ = weight_events(p_attr, None, 1, 0, float(c), 1.0,
                                  is_per=False, period=0.0, width=W_WIDTH, drop_input_attr=False)
        dens = build_exp_tens(pw, ww, [SIG_P, SIG_T], [1, 1], [False, False],
                              [False, False], [0.0, 0.0], verbose=False)
        ref[i] = entropy_exp_tens(dens, method="renyi2", verbose=False)
    got = windowed_entropy(p_attr, None, [SIG_P, SIG_T], [1, 1], [False, False],
                           [False, False], [0.0, 0.0], centres,
                           window=(1.0, W_WIDTH), method="renyi2",
                           window_attr=1, drop_window_attr=False, verbose=False)
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-9)


def test_entropy_drop_r_ge_2_rejected(carrier):
    p_attr, centres = carrier
    with pytest.raises(ValueError):
        windowed_entropy(p_attr, None, [SIG_P, SIG_T], [1, 2], [False, False],
                         [False, False], [0.0, 0.0], centres,
                         window=(1.0, W_WIDTH), window_attr=1, drop_window_attr=True, verbose=False)


def test_entropy_marginalise_not_implemented(carrier):
    """marginalise (integrate out a retained axis) is reserved but unimplemented."""
    p_attr, centres = carrier
    with pytest.raises(NotImplementedError):
        windowed_entropy(p_attr, None, [SIG_P, SIG_T], [1, 1], [False, False],
                         [False, False], [0.0, 0.0], centres,
                         window=(1.0, W_WIDTH), window_attr=1,
                         drop_window_attr=False, marginalise=0, verbose=False)


def test_entropy_drop_and_marginalise_same_axis_rejected(carrier):
    """A dropped axis is already gone, so it cannot also be marginalised."""
    p_attr, centres = carrier
    with pytest.raises(ValueError):
        windowed_entropy(p_attr, None, [SIG_P, SIG_T], [1, 1], [False, False],
                         [False, False], [0.0, 0.0], centres,
                         window=(1.0, W_WIDTH), window_attr=1,
                         drop_window_attr=True, marginalise=1, verbose=False)


def test_entropy_requires_width(carrier):
    p_attr, centres = carrier
    with pytest.raises(ValueError):
        windowed_entropy(p_attr, None, [SIG_P, SIG_T], [1, 1], [False, False],
                         [False, False], [0.0, 0.0], centres,
                         window=(1.0, None), window_attr=1, drop_window_attr=False, verbose=False)


@pytest.mark.parametrize("normalize", ["oneSidedDenom", "cosine"])
def test_similarity_locked(carrier, query, normalize):
    p_attr, centres = carrier
    qv = query[1].ravel(); q_ext = float(qv.max() - qv.min()); mu_q = float(qv.mean())
    ref = np.empty(len(centres))
    for i, c in enumerate(centres):
        pc, wc, _ = weight_events(p_attr, None, 1, 0, float(c), 1.0,
                                  width=q_ext, is_per=False, period=0.0, drop_input_attr=False)
        pq, wq, _ = translate_attributes(query, None, [None, np.array([[c - mu_q]])])
        ref[i] = cos_sim_exp_tens(pc, wc, pq, wq, [SIG_P, SIG_T], [1, 1],
                                  [False, False], [False, False], [0.0, 0.0],
                                  normalize=normalize, verbose=False)
    got = windowed_similarity(p_attr, None, query, None, [SIG_P, SIG_T], [1, 1],
                              [False, False], [False, False], [0.0, 0.0], centres,
                              normalize=normalize, window_attr=1, drop_window_attr=False, verbose=False)
    assert got.shape == (len(centres),)
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-9)


def test_similarity_decoupled_correlogram(carrier, query):
    p_attr, centres = carrier
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
            pq, wq, _ = translate_attributes(query, None, [None, np.array([[(a - t) - mu_q]])])
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


def test_similarity_generative_sweep_matches_explicit(carrier, query):
    p_attr, _ = carrier
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


def test_centres_and_generative_mutually_exclusive(carrier, query):
    p_attr, centres = carrier
    with pytest.raises(ValueError):
        windowed_similarity(p_attr, None, query, None, [SIG_P, SIG_T], [1, 1],
                            [False, False], [False, False], [0.0, 0.0], centres,
                            step=1.0, window_attr=1, drop_window_attr=False, verbose=False)


def test_similarity_translate_context(carrier, query):
    """context_window=(None, ...) translates the context whole (no shape
    resolution); regression for the translate-context path."""
    p_attr, centres = carrier
    mu_c = float(p_attr[1].mean()); mu_q = float(query[1].mean())
    # inline: translate context to each centre, query to same centre, score
    ref = np.empty(len(centres))
    for i, c in enumerate(centres):
        pc, wc, _ = translate_attributes(p_attr, None, [None, np.array([[c - mu_c]])])
        pq, wq, _ = translate_attributes(query, None, [None, np.array([[c - mu_q]])])
        ref[i] = cos_sim_exp_tens(pc, wc, pq, wq, [SIG_P, SIG_T], [1, 1],
                                  [False, False], [False, False], [0.0, 0.0],
                                  normalize="oneSidedDenom", verbose=False)
    got = windowed_similarity(p_attr, None, query, None, [SIG_P, SIG_T], [1, 1],
                              [False, False], [False, False], [0.0, 0.0], centres,
                              context_window=(None, None), normalize="oneSidedDenom",
                              window_attr=1, drop_window_attr=False, verbose=False)
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-9)
