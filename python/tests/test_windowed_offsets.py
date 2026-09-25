"""Tests for ``offsets`` in ``windowed_similarity``: translation of the query
indexed by its offset (article, Sec. 3, attribute translation).

The travelling-window form is pinned to window centres written out by
hand as the offset plus the query's position, and the one-pass sweep route
is pinned to the per-offset comparison.
"""

import numpy as np
import pytest

import mpt
from mpt import windowed_similarity, sweep_sim_maet, build_maet

GEOM = ([20.0, 0.5], [1, 1], [False, False], [True, False], [1200.0, 0.0])


@pytest.fixture
def melody():
    clave = np.array([0.0, 3.0, 6.0, 10.0, 12.0])
    t = np.concatenate([clave, clave + 16.0])[None, :]
    p = mpt.transform_attributes(
        np.array([60, 62, 64, 67, 64, 67, 69, 71, 74, 71]), None,
        ('midi', 'cents'))[None, :]
    return [p, t], [p[:, :3], t[:, :3]]           # context, query (C D E)


def test_offsets_equal_centres_at_offset_plus_position(melody):
    ctx, qry = melody
    mu = np.arange(-4.0, 24.01, 0.5)
    kw = dict(window_attr=1, context_window=('gaussian', 16.0), verbose=False)
    new = windowed_similarity(ctx, None, qry, None, *GEOM, offsets=mu, **kw)
    old = windowed_similarity(ctx, None, qry, None, *GEOM, mu + 3.0,
                              drop_window_attr=False, **kw)
    np.testing.assert_array_equal(new, old)
    assert mu[np.argmax(new)] == 0.0               # the query where it stands


def test_correlogram_route_matches_per_offset_comparison(melody, monkeypatch):
    """The correlogram, routed through sweep_sim_maet, agrees with the
    per-offset comparison it falls back to where no sweep route applies
    (forced here by declining the route), and with the window left at
    each centre: each column is the window at the centres with the query
    translated by that offset."""
    from mpt._tensor import windowed as W
    ctx, qry = melody
    mu = np.arange(-4.0, 24.01, 2.0)
    centres = np.array([0.0, 8.0, 16.0, 24.0])
    kw = dict(window_attr=1, context_window=('gaussian', 16.0), verbose=False)
    new = windowed_similarity(ctx, None, qry, None, *GEOM, centres,
                              offsets=mu, **kw)
    assert new.shape == (centres.size, mu.size)
    monkeypatch.setattr(W, "_sweep_row", lambda *a, **k: None)
    old = windowed_similarity(ctx, None, qry, None, *GEOM, centres,
                              offsets=mu, **kw)
    np.testing.assert_allclose(new, old, rtol=0, atol=1e-10)
    col = windowed_similarity(ctx, None, qry, None, *GEOM, centres,
                              offsets=np.full((centres.size, 1), mu[3]), **kw)
    np.testing.assert_allclose(col.ravel(), old[:, 3], rtol=0, atol=1e-12)


def test_translated_attribute_without_window_is_pure_translation(melody):
    """An attribute named in ``offsets`` with no window is swept by pure
    translation, with no window anywhere: the result is sweep_sim_maet at
    the same offsets."""
    ctx, qry = melody
    mu = np.arange(-2.0, 22.01, 1.0)
    got = windowed_similarity(ctx, None, qry, None, *GEOM, offsets={1: mu},
                              verbose=False)
    dx = build_maet(ctx, None, *GEOM, verbose=False)
    dy = build_maet(qry, None, *GEOM, verbose=False)
    want = sweep_sim_maet(dx, dy, np.vstack([np.zeros_like(mu), mu]),
                          normalize='oneSidedDenom', verbose=False)
    np.testing.assert_allclose(np.asarray(got).ravel(), want, rtol=0,
                               atol=1e-10)


def test_multi_axis_offsets_match_per_offset_comparison(melody):
    """Pitch translated with no window, time windowed and dropped: the one
    pass at each window position agrees with comparing offset by offset
    (a pitch window wide enough to be inert, on the old sweep form)."""
    ctx, qry = melody
    mu = np.arange(-600.0, 601.0, 100.0)
    centres = np.arange(0.0, 28.1, 2.0)
    new = windowed_similarity(
        ctx, None, qry, None, *GEOM, offsets={0: mu}, sweep={1: centres},
        drop={1: True}, context_window={1: {'shape': 'gaussian',
                                            'width': 12.0}}, verbose=False)
    q_pos = float(np.mean(qry[0]))
    old = windowed_similarity(
        ctx, None, qry, None, *GEOM, sweep={0: mu + q_pos, 1: centres},
        drop={0: False, 1: True},
        context_window={0: {'shape': 'rect', 'width': 1e7},
                        1: {'shape': 'gaussian', 'width': 12.0}},
        verbose=False)
    assert new.shape == (mu.size, centres.size)
    np.testing.assert_allclose(new, old, rtol=0, atol=1e-10)


def test_offsets_refusals(melody):
    ctx, qry = melody
    mu = np.arange(0.0, 4.0)
    with pytest.raises(ValueError, match="compared"):
        windowed_similarity(ctx, None, qry, None, *GEOM, offsets=mu,
                            window_attr=1, drop_window_attr=True)
    rel = ([20.0, 0.5], [1, 1], [False, True], [True, False], [1200.0, 0.0])
    with pytest.raises(ValueError, match="relative"):
        windowed_similarity(ctx, None, qry, None, *rel, offsets=mu,
                            window_attr=1)
