"""Tests for what the sweep values of ``swept_similarity`` apply to
(``align``): the window, the query, both together, or each independently.

Each role is pinned to the composition it stands for, written out with
``weight_events`` / ``translate_attributes`` / ``sim_maet``; the one-pass
route through ``sweep_sim_maet`` is pinned to the per-value comparison it
replaces; and every refusal names the role the caller probably meant.
"""

import numpy as np
import pytest

import mpt
from mpt import (swept_similarity, sweep_sim_maet, build_maet, sim_maet,
                 weight_events, translate_attributes, unpack_pre_maet)

GEOM = ([20.0, 0.5], [1, 1], [False, False], [True, False], [1200.0, 0.0])


def _h(sigma, r=1):
    """Half the standard deviation of a translation profile's peaks."""
    return sigma * np.sqrt(2.0 / r) / 2.0


def _lat(g, h):
    """The default step on a lattice of spacing g: g / k, k the smallest
    whole number bringing it to h or below."""
    return g / np.ceil(g / h - 1e-9)


@pytest.fixture
def melody():
    clave = np.array([0.0, 3.0, 6.0, 10.0, 12.0])
    t = np.concatenate([clave, clave + 16.0])[None, :]
    p = mpt.transform_attributes(
        np.array([60, 62, 64, 67, 64, 67, 69, 71, 74, 71]), None,
        ('midi', 'cents'))[None, :]
    return [p, t], [p[:, :3], t[:, :3]]           # context, query (C D E)


def _by_hand(ctx, qry, window_at, query_to, q_ref, sd):
    """The context windowed on time at ``window_at`` (Gaussian, standard
    deviation ``sd``; ``None`` for no window) against the query translated
    in time so that ``q_ref`` lands at ``query_to``."""
    pc, wc = ctx, None
    if window_at is not None:
        pc, wc, _ = unpack_pre_maet(weight_events(
            ctx, None, 1, 0, float(window_at), 0.0, sd=sd, is_per=False,
            period=0.0, drop_input_attr=False))
    pq, wq, _ = unpack_pre_maet(translate_attributes(
        qry, None, [None, np.array([[query_to - q_ref]])]))
    return float(sim_maet(pc, wc, pq, wq, *GEOM, normalize='oneSidedDenom',
                          verbose=False))


def test_both_is_window_and_query_at_each_value(melody):
    ctx, qry = melody
    s = np.arange(-2.0, 26.01, 2.0)
    width = 16.0
    got = swept_similarity(ctx, None, qry, None, *GEOM, sweep={1: s},
                           align={1: 'both'},
                           window={1: {'shape': 'gaussian', 'width': width}},
                           verbose=False)
    q_ref = float(np.mean(qry[1]))
    want = [_by_hand(ctx, qry, v, v, q_ref, width / (2 * np.sqrt(3)))
            for v in s]
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    # One rule: the window is aligned at s and the query's reference lands
    # at s, whatever that reference is; queryRef 0 puts the query's start,
    # not its middle, under the window's centre.
    got0, mu0 = swept_similarity(ctx, None, qry, None, *GEOM,
                                 sweep={1: s}, align={1: 'both'},
                                 query_ref={1: 0.0},
                                 window={1: {'shape': 'gaussian', 'width': width}},
                                 return_offsets=True, verbose=False)
    want0 = [_by_hand(ctx, qry, v, v, 0.0, width / (2 * np.sqrt(3)))
             for v in s]
    np.testing.assert_allclose(got0, want0, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(mu0[1], s)
    # The translation applied, mu = s - queryRef, is returned on request.
    _, mu = swept_similarity(ctx, None, qry, None, *GEOM, sweep={1: s},
                             align={1: 'both'},
                             window={1: {'shape': 'gaussian', 'width': width}},
                             return_offsets=True, verbose=False)
    assert list(mu) == [1]
    np.testing.assert_allclose(mu[1], s - q_ref, rtol=0, atol=1e-12)


def test_independent_route_matches_per_value_comparison(melody, monkeypatch):
    """The correlogram, routed through sweep_sim_maet, agrees with the
    per-value comparison it falls back to where no sweep route applies
    (forced here by declining the route), and with the composition written
    out by hand; a 2-D query list holds one row per window value."""
    from mpt._tensor import swept as W
    ctx, qry = melody
    wv = np.array([0.0, 8.0, 16.0, 24.0])
    qv = np.arange(-2.0, 26.01, 2.0)
    kw = dict(align={1: 'independent'},
              window={1: {'shape': 'gaussian', 'width': 16.0}},
              verbose=False)
    new = swept_similarity(ctx, None, qry, None, *GEOM,
                           sweep={1: (wv, qv)}, **kw)
    assert new.shape == (wv.size, qv.size)
    q_ref = float(np.mean(qry[1]))
    sd = 16.0 / (2 * np.sqrt(3))
    want = [[_by_hand(ctx, qry, a, b, q_ref, sd) for b in qv] for a in wv]
    np.testing.assert_allclose(new, want, rtol=0, atol=1e-10)
    monkeypatch.setattr(W, "_sweep_row", lambda *a, **k: None)
    old = swept_similarity(ctx, None, qry, None, *GEOM,
                           sweep={1: (wv, qv)}, **kw)
    np.testing.assert_allclose(new, old, rtol=0, atol=1e-10)
    lag = wv[:, None] - np.array([[2.0, 0.0, -2.0]])
    rows = swept_similarity(ctx, None, qry, None, *GEOM,
                            sweep={1: (wv, lag)}, **kw)
    want_rows = [[_by_hand(ctx, qry, a, b, q_ref, sd) for b in lag[i]]
                 for i, a in enumerate(wv)]
    np.testing.assert_allclose(rows, want_rows, rtol=0, atol=1e-10)


def test_query_is_pure_translation(melody):
    """With ``align='query'`` there is no window anywhere: the result is
    sweep_sim_maet at the same offsets, and ``query_ref=0`` makes the sweep
    values those offsets."""
    ctx, qry = melody
    mu = np.arange(-2.0, 22.01, 1.0)
    got = swept_similarity(ctx, None, qry, None, *GEOM, sweep={1: mu},
                           align={1: 'query'}, query_ref={1: 0.0},
                           verbose=False)
    dx = build_maet(ctx, None, *GEOM, verbose=False)
    dy = build_maet(qry, None, *GEOM, verbose=False)
    want = sweep_sim_maet(dx, dy, np.vstack([np.zeros_like(mu), mu]),
                          normalize='oneSidedDenom', verbose=False)
    np.testing.assert_allclose(np.asarray(got).ravel(), want, rtol=0,
                               atol=1e-10)
    # The canonical call: a bare sweep is attribute translation over the
    # whole context, the sweep values being the offsets (align 'query' and
    # query_ref 0 are the defaults).
    plain, offs = swept_similarity(ctx, None, qry, None, *GEOM,
                                   sweep={1: mu}, return_offsets=True,
                                   verbose=False)
    np.testing.assert_array_equal(plain, got)
    np.testing.assert_array_equal(offs[1], mu)


def test_query_with_window_elsewhere_matches_per_value(melody, monkeypatch):
    """Pitch translated with no window, time windowed and dropped: the one
    pass at each window value agrees with comparing value by value; and
    pitch translated with no window alongside time translated with the
    window ('both'): the pitch translations are computed in one pass on the
    query already translated in time."""
    from mpt._tensor import swept as W
    ctx, qry = melody
    mu = np.arange(-600.0, 601.0, 100.0)
    tv = np.arange(0.0, 28.1, 2.0)
    cases = [
        dict(sweep={0: mu, 1: tv}, align={0: 'query', 1: 'window'},
             drop=[1], window={1: {'shape': 'gaussian', 'width': 12.0}}),
        dict(sweep={0: mu, 1: tv}, align={0: 'query', 1: 'both'},
             window={1: {'shape': 'gaussian', 'width': 12.0}}),
    ]
    for kw in cases:
        new = swept_similarity(ctx, None, qry, None, *GEOM,
                               query_ref={0: 0.0}, verbose=False, **kw)
        assert new.shape == (mu.size, tv.size)
        with monkeypatch.context() as m:
            m.setattr(W, "_sweep_row", lambda *a, **k: None)
            old = swept_similarity(ctx, None, qry, None, *GEOM,
                                   query_ref={0: 0.0}, verbose=False, **kw)
        np.testing.assert_allclose(new, old, rtol=0, atol=1e-10)


def test_window_on_a_compared_attribute_compares_in_place(melody):
    """align='window' on an absolute attribute kept in the comparison: the
    query is compared as written with the windowed context, and windows
    that tile the context give contributions that sum to the whole-context
    similarity (the inner product is linear in the weights)."""
    ctx, qry = melody
    s = np.arange(0.0, 29.0, 2.0)
    got = swept_similarity(ctx, None, qry, None, *GEOM, sweep={1: s},
                           align={1: 'window'},
                           window={1: {'shape': 'gaussian', 'width': 8.0}},
                           verbose=False)
    want = [_by_hand(ctx, qry, v, 0.0, 0.0, 8.0 / (2 * np.sqrt(3)))
            for v in s]
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    tiles = np.arange(-2.0, 30.0, 4.0) + 2.0      # half-open [c-2, c+2)
    parts = swept_similarity(ctx, None, qry, None, *GEOM,
                             sweep={1: tiles}, align={1: 'window'},
                             window={1: {'shape': 'rect', 'width': 4.0}}, verbose=False)
    whole = float(sim_maet(ctx, None, qry, None, *GEOM,
                           normalize='oneSidedDenom', verbose=False))
    assert float(np.sum(parts)) == pytest.approx(whole, rel=1e-12)


def test_default_step_lands_on_every_exact_match(melody):
    """The default translation step is a whole fraction of the lattice the
    values lie on, at most h, half the peaks' standard deviation
    (sigma * sqrt(2 / r) / 2), so every exact match is on the grid: onsets
    on a grid of 1 with h = 0.15 step at 1/7 (not 0.15, which misses the
    whole-number offsets), and a context transposed off the query's lattice
    is still found exactly, in pitch and time. Values on no lattice keep
    h."""
    ctx, qry = melody
    geom = ([20.0, 0.15 * np.sqrt(2.0)],) + GEOM[1:]      # h = 0.15
    S, mu = swept_similarity(ctx, None, qry, None, *geom, sweep=1,
                             return_offsets=True, verbose=False)
    np.testing.assert_allclose(np.diff(mu[1]), 1.0 / 7.0, atol=1e-9)
    assert np.min(np.abs(mu[1] - 16.0)) < 1e-9
    k = int(np.argmin(np.abs(mu[1])))
    assert abs(mu[1][k]) < 1e-9 and S[k] > 1 - 1e-9
    shifted = [ctx[0] + 13.7, ctx[1]]
    S2, mu2 = swept_similarity(shifted, None, qry, None, *GEOM,
                               sweep=[0, 1], return_offsets=True,
                               verbose=False)
    i, j = np.unravel_index(int(np.argmax(S2)), S2.shape)
    assert abs(mu2[0][i] - 13.7) < 1e-9 and abs(mu2[1][j]) < 1e-9
    assert S2[i, j] > 1 - 1e-9
    loose = [ctx[0], ctx[1] + 0.05 * np.sin(np.arange(1, ctx[1].size + 1))]
    _, mu3 = swept_similarity(loose, None, qry, None, *GEOM, sweep=1,
                              return_offsets=True, verbose=False)
    np.testing.assert_allclose(np.diff(mu3[1]), _h(GEOM[0][1]), atol=1e-9)


def test_bare_attribute_takes_default_sweep_values(melody):
    """sweep=a asks for default sweep values. For a translation, every offset
    at which query and context overlap, stepped at h, half the peaks' sd, on
    the values' lattice; on a periodic attribute, one period; start
    / stop / step override one default at a time. 'both' translates the query
    too and takes the same defaults; 'window' keeps the context's extent at
    half the window's sd."""
    ctx, qry = melody
    S, mu = swept_similarity(ctx, None, qry, None, *GEOM, sweep=1,
                             return_offsets=True, verbose=False)
    lo = ctx[1].min() - qry[1].max()
    hi = ctx[1].max() - qry[1].min()
    np.testing.assert_allclose(mu[1][[0, -1]], [lo, hi], atol=1e-9)
    np.testing.assert_allclose(np.diff(mu[1]), _lat(1.0, _h(GEOM[0][1])),
                               atol=1e-9)
    want = swept_similarity(ctx, None, qry, None, *GEOM, sweep={1: mu[1]},
                            verbose=False)
    np.testing.assert_array_equal(S, want)
    _, mp = swept_similarity(ctx, None, qry, None, *GEOM, sweep=0,
                             return_offsets=True, verbose=False)
    assert mp[0][0] == 0.0 and mp[0][-1] < 1200.0
    st_p = _lat(100.0, _h(GEOM[0][0]))
    np.testing.assert_allclose(np.diff(mp[0]), st_p, atol=1e-9)
    assert mp[0].size == int(round(1200.0 / st_p))
    S2 = swept_similarity(ctx, None, qry, None, *GEOM, sweep=[0, 1],
                          verbose=False)
    assert S2.shape == (mp[0].size, mu[1].size)
    _, ms = swept_similarity(ctx, None, qry, None, *GEOM, sweep=1,
                             step={1: 1.0}, return_offsets=True,
                             verbose=False)
    np.testing.assert_allclose(ms[1], np.arange(lo, hi + 1e-9, 1.0))
    # a bare number applies to the one swept attribute
    _, mb_ = swept_similarity(ctx, None, qry, None, *GEOM, sweep=1,
                              start=0.0, step=1.0, return_offsets=True,
                              verbose=False)
    np.testing.assert_allclose(mb_[1], np.arange(0.0, hi + 1e-9, 1.0))
    with pytest.raises(ValueError, match="bare `step`"):
        swept_similarity(ctx, None, qry, None, *GEOM, sweep=[0, 1],
                         step=1.0, verbose=False)
    # 'both' translates the query too, so it takes the same defaults:
    # the offsets mu are those of 'query', whatever the window.
    _, mb = swept_similarity(ctx, None, qry, None, *GEOM, sweep=1,
                             align={1: 'both'},
                             window={1: {'shape': 'rect', 'width': 8.0}},
                             return_offsets=True, verbose=False)
    np.testing.assert_allclose(mb[1], mu[1], atol=1e-9)
    # a windowed role that does not translate the query steps the
    # context's extent at half the window's sd (a Gaussian of the width of
    # a rectangle of width 8 has sd 8 / (2 sqrt 3))
    s_w, sv_w = swept_similarity(ctx, None, qry, None, *GEOM, sweep=1,
                                 align={1: 'window'}, drop=[1],
                                 window={1: {"shape": 0.0, "width": 8.0}},
                                 return_sweep_values=True, verbose=False)
    step_w = 8.0 / (2.0 * np.sqrt(3.0)) / 2.0
    np.testing.assert_allclose(np.diff(sv_w[1]), step_w, atol=1e-9)
    assert sv_w[1][0] == ctx[1].min() and sv_w[1][-1] <= ctx[1].max()
    assert np.ravel(s_w).size == int(
        np.floor((ctx[1].max() - ctx[1].min()) / step_w + 1e-9)) + 1


def test_refusals(melody):
    ctx, qry = melody
    v = np.arange(0.0, 4.0)
    run = lambda **kw: swept_similarity(ctx, None, qry, None, *GEOM,
                                        verbose=False, **kw)
    win = {1: {'shape': 'rect', 'width': 8.0}}
    with pytest.raises(ValueError, match="compared and cannot be dropped"):
        run(sweep={1: v}, align={1: 'both'}, window=win, drop=[1])
    rel = ([20.0, 0.5], [1, 1], [False, True], [True, False], [1200.0, 0.0])
    with pytest.raises(ValueError, match="relative"):
        swept_similarity(ctx, None, qry, None, *rel, sweep={1: v},
                         align={1: 'both'}, window=win)
    with pytest.raises(ValueError, match="not a window attribute"):
        run(sweep={1: v}, align={1: 'both'}, window=win, drop=[0])
    with pytest.raises(ValueError, match="its `align` is 'query'"):
        run(sweep={1: v}, align={1: 'query'}, window=win)
    with pytest.raises(ValueError, match="its `align` is 'query'"):
        run(sweep={1: v}, window=win)
    with pytest.raises(ValueError, match="give its shape and scale"):
        run(sweep={1: (v, v)}, align={1: 'independent'})
    with pytest.raises(ValueError, match="give its shape and scale"):
        run(sweep={1: v}, align={1: 'window'}, drop=[1])
    with pytest.raises(ValueError, match="no sweep values"):
        run(sweep={1: v}, align={0: 'query', 1: 'query'})
    with pytest.raises(ValueError, match="one of"):
        run(sweep={1: v}, align={1: 'follow'}, window=win)
    with pytest.raises(ValueError, match="two lists"):
        run(sweep={1: v}, align={1: 'independent'}, window=win)
    with pytest.raises(ValueError, match="one row per window value"):
        run(sweep={1: (v, np.zeros((2, 3)))}, align={1: 'independent'},
            window=win)
    no_sigma = ([20.0, float('nan')], [1, 1], [False, False], [True, False],
                [1200.0, 0.0])
    with pytest.raises(ValueError, match="give `step`"):
        swept_similarity(ctx, None, qry, None, *no_sigma, sweep=1,
                         verbose=False)
    with pytest.raises(ValueError, match="query's sweep values given"):
        run(sweep=1, align={1: 'independent'}, window=win)
    with pytest.raises(ValueError, match="not translated along"):
        run(sweep={1: v}, align={1: 'both'}, window=win, query_ref={0: 0.0})
    with pytest.raises(ValueError, match="at least one attribute"):
        run()
    with pytest.raises(ValueError, match="not both"):
        run(sweep={1: v}, step={1: 1.0}, align={1: 'both'}, window=win)


def test_independent_window_values_may_be_generated(melody):
    """For 'independent', `start` / `stop` / `step` generate the window's
    list; the query's is given in the pair."""
    ctx, qry = melody
    qv = np.arange(0.0, 24.01, 4.0)
    kw = dict(align={1: 'independent'}, window={1: {'shape': 'rect', 'width': 8.0}},
              verbose=False)
    gen = swept_similarity(ctx, None, qry, None, *GEOM,
                           sweep={1: (None, qv)}, step={1: 4.0}, **kw)
    lst = swept_similarity(ctx, None, qry, None, *GEOM,
                           sweep={1: (np.arange(0.0, 28.01, 4.0), qv)},
                           **kw)
    assert np.array_equal(gen, lst)


def test_default_window_holds_the_query(melody):
    """A window that travels with the query may be left out: it is then the
    smallest closed rectangle, aligned at the query's middle, that holds the query,
    so the query's own statement scores 1."""
    ctx, qry = melody
    s = np.arange(0.0, 26.01, 1.0)
    got = swept_similarity(ctx, None, qry, None, *GEOM, sweep={1: s},
                           align={1: 'both'}, verbose=False)
    q_ref = float(np.mean(qry[1]))                      # 3: onsets 0, 3, 6
    same = swept_similarity(ctx, None, qry, None, *GEOM, sweep={1: s},
                            align={1: 'both'},
                            window={1: {'shape': 'rect', 'width': 6.0,
                                            'edges': 'closed'}},
                            verbose=False)
    assert np.array_equal(got, same)
    assert got[int(q_ref)] == pytest.approx(1.0, abs=1e-12)
    # width None asks for the default width with the shape given
    gauss = swept_similarity(ctx, None, qry, None, *GEOM, sweep={1: s},
                             align={1: 'both'},
                             window={1: {'shape': 'gaussian'}}, verbose=False)
    wide = swept_similarity(ctx, None, qry, None, *GEOM, sweep={1: s},
                            align={1: 'both'},
                            window={1: {'shape': 'gaussian', 'width': 6.0}},
                            verbose=False)
    assert np.array_equal(gauss, wide)


def test_window_that_cuts_the_query_warns(melody):
    """A travelling window that leaves out some of the query's own events
    warns; a closed rectangle as wide as the query, or a wider half-open
    one, does not."""
    import warnings
    ctx, qry = melody
    s = np.arange(0.0, 26.01, 1.0)
    run = lambda spec: swept_similarity(
        ctx, None, qry, None, *GEOM, sweep={1: s}, align={1: 'both'},
        window={1: spec}, verbose=False)
    with pytest.warns(UserWarning, match="leaves out 1 of the query's 3"):
        half = run({'shape': 'rect', 'width': 6.0})     # upper edge excluded
    with pytest.warns(UserWarning, match="leaves out 2 of the query's 3"):
        run({'shape': 'rect', 'width': 2.0})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        closed = run({'shape': 'rect', 'width': 6.0, 'edges': 'closed'})
        run({'shape': 'rect', 'width': 6.5})
        run({'shape': 'gaussian', 'width': 2.0})
    # at s = 3 the query lands on its own statement (queryRef 3: onsets
    # 0, 3, 6): the closed rectangle holds all of it, the half-open one not
    assert closed[3] == pytest.approx(1.0, abs=1e-12)
    assert half[3] < closed[3]


def test_offsets_read_through_differencing(melody):
    """With query and context written from time 0, the returned offsets mu
    are the time at which the query starts in the context, and keep that
    reading through differencing: the motif and its differenced form, whose
    middles differ (and so their windows), peak at the same offsets."""
    ctx, qry = melody
    kw = dict(step={1: 0.5}, align={1: 'both'},
              window={1: {'shape': 'gaussian', 'width': 16.0}}, return_offsets=True,
              verbose=False)
    geom_d = ([20.0 * np.sqrt(2), 0.5], [1, 1], [False, False],
              [False, False], [0.0, 0.0])
    plain, mu_p = swept_similarity(ctx, None, qry, None, *GEOM, **kw)
    ctx_d = mpt.unpack_pre_maet(mpt.difference_events(ctx, None, [1, 0]))[0]
    qry_d = mpt.unpack_pre_maet(mpt.difference_events(qry, None, [1, 0]))[0]
    diff, mu_d = swept_similarity(ctx_d, None, qry_d, None, *geom_d, **kw)
    # the middles differ (3 and 4.5), so the offsets are measured from
    # different reference values but mean the same thing
    assert mu_p[1][np.argmax(plain)] == 0.0
    i0 = int(np.flatnonzero(mu_d[1] == 0.0)[0])
    i16 = int(np.flatnonzero(mu_d[1] == 16.0)[0])
    # the differenced form finds the transposed statement at 16 as well,
    # equally strongly; the original statement is still at 0
    assert diff[i0] == pytest.approx(diff.max(), rel=1e-4)
    assert diff[i16] == pytest.approx(diff.max(), rel=1e-4)
    assert diff[i0] > diff[i0 - 1] and diff[i0] > diff[i0 + 1]
