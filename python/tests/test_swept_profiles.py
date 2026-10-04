"""The swept functions evaluate their windows through the implementation
``weight_events`` uses, so every profile of ``weight_events`` that is
aligned at a reference value is available as a window, with the same
factors; and ``weight_events`` gains ``locate`` and ``edges`` from them.
"""

import numpy as np
import pytest

import mpt
from mpt import swept_similarity, swept_entropy, sim_maet, weight_events

GEOM = ([20.0, 0.25], [1, 1], [False, False], [True, False], [1200.0, 0.0])


@pytest.fixture
def melody():
    t = np.array([[0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 5.5, 6.0, 7.0]])
    p = 100.0 * np.array([[60, 62, 64, 67, 65, 60, 62, 64, 67, 65.0]])
    return [p, t], [p[:, :4], t[:, :4]]


def _by_weight_events(ctx, qry, at, **profile):
    """The 'window' alignment written out: the context weighted on time by
    weight_events at ``at``, time dropped, compared with the query."""
    pm = weight_events(ctx, None, 1, 0, at, profile.pop("shape"),
                       per=False, period=0.0, drop_input_attr=True,
                       **profile)
    pc, wc, _ = mpt.unpack_pre_maet(pm)
    g = [GEOM[i][:1] for i in range(5)]
    return float(sim_maet(pc, wc, qry[:1], None, *g,
                          normalize="oneSidedDenom", verbose=False))


@pytest.mark.parametrize("spec, profile", [
    ({"shape": "rect", "width": 2.0}, dict(shape=1.0, width=2.0)),
    ({"shape": "gaussian", "width": 2.0}, dict(shape=0.0, width=2.0)),
    ({"shape": "exponential", "sd": 1.0},
     dict(shape="exponential", sd=1.0)),
    ({"shape": "exponentialBefore", "decay_rate": 0.5},
     dict(shape="exponentialBefore", decay_rate=0.5)),
    ({"shape": "exponentialAfter", "sd": 1.5},
     dict(shape="exponentialAfter", sd=1.5)),
])
def test_window_is_weight_events(melody, spec, profile):
    """Each window profile gives, at every sweep value, the similarity of
    the context weighted by weight_events with the same profile."""
    ctx, qry = melody
    at = np.array([1.0, 2.5, 4.0, 6.0])
    S = swept_similarity(ctx, None, qry, None, *GEOM, sweep={1: at},
                         align={1: "window"}, window={1: spec}, drop=[1],
                         verbose=False)
    want = [_by_weight_events(ctx, qry, s, **dict(profile)) for s in at]
    np.testing.assert_allclose(np.ravel(S), want, rtol=1e-12, atol=1e-15)


def test_window_profile_function(melody):
    """A callable window receives the displacement p_a(n) - s."""
    ctx, qry = melody
    f = lambda d: np.exp(-np.abs(d))          # noqa: E731
    at = np.array([1.0, 4.0])
    S = swept_similarity(ctx, None, qry, None, *GEOM, sweep={1: at},
                         align={1: "window"}, window={1: f}, drop=[1],
                         verbose=False)
    want = [_by_weight_events(ctx, qry, s, shape=f) for s in at]
    np.testing.assert_allclose(np.ravel(S), want, rtol=1e-12)
    with pytest.raises(ValueError, match="give `step`"):
        swept_similarity(ctx, None, qry, None, *GEOM, sweep=1,
                         align={1: "window"}, window={1: f}, drop=[1],
                         verbose=False)


def test_window_to_one_side_under_both(melody):
    """A window extending to one side only, aligned with the query
    ('both'), keeps only the context on one side of the query's
    reference: exponentialAfter with query_ref at the query's first onset
    scores the exact statement below 1 only through the decay over the
    query's own span."""
    ctx, qry = melody
    S, mu = swept_similarity(
        ctx, None, qry, None, *GEOM, sweep={1: np.array([0.0])},
        align={1: "both"}, query_ref={1: 0.0},
        window={1: {"shape": "exponentialAfter", "sd": 100.0}},
        return_offsets=True, verbose=False)
    assert np.ravel(S)[0] == pytest.approx(1.0, abs=0.05)
    assert mu[1][0] == 0.0


def test_anchored_profiles_refused(melody):
    ctx, qry = melody
    with pytest.raises(ValueError, match="anchored"):
        swept_similarity(ctx, None, qry, None, *GEOM,
                         sweep={1: np.array([1.0])}, align={1: "window"},
                         window={1: {"shape": "uShape", "sd": 1.0}},
                         drop=[1], verbose=False)


def test_exponential_takes_no_width(melody):
    ctx, qry = melody
    with pytest.raises(ValueError, match="does not apply"):
        swept_similarity(ctx, None, qry, None, *GEOM,
                         sweep={1: np.array([1.0])}, align={1: "window"},
                         window={1: {"shape": "exponential", "width": 2.0}},
                         drop=[1],
                         verbose=False)


def test_closed_edges_only_for_rectangles(melody):
    ctx, qry = melody
    with pytest.raises(ValueError, match="rectangle"):
        swept_similarity(ctx, None, qry, None, *GEOM,
                         sweep={1: np.array([1.0])}, align={1: "window"},
                         window={1: {"shape": "gaussian", "width": 2.0,
                                     "edges": "closed"}},
                         drop=[1], verbose=False)


def test_periodic_window_wraps():
    """On a periodic attribute the window's displacement wraps, as
    weight_events wraps it: a window aligned near one end of the cycle
    reaches events near the other."""
    t = np.array([[0.1, 3.9, 2.0]])                 # cycle of 4
    p = np.array([[60.0, 64.0, 67.0]])
    geom = ([0.5, 0.1], [1, 1], [False, False], [False, True], [0.0, 4.0])
    H = swept_entropy([p, t], None, *geom, sweep={1: np.array([0.0])},
                      window={1: {"shape": "rect", "width": 1.0}}, drop=[1],
                      method="renyi2", verbose=False)
    pm = weight_events([p, t], None, 1, 0, 0.0, 1.0, width=1.0, per=True,
                       period=4.0, drop_input_attr=True)
    pc, wc, _ = mpt.unpack_pre_maet(pm)
    np.testing.assert_allclose(np.ravel(wc[0]), [1.0, 1.0, 0.0])
    want = mpt.entropy_maet(mpt.build_maet(pc, wc, [0.5], [1], [False],
                                           [False], [0.0], verbose=False),
                            method="renyi2")
    assert float(np.ravel(H)[0]) == pytest.approx(float(want), rel=1e-12)


def test_weight_events_locate_and_edges():
    """weight_events reduces a multi-valued input by `locate`, and closes
    a rectangle's upper edge with edges='closed'."""
    t = np.array([[0.0, 1.0, 2.0], [0.5, 1.5, 2.5]])      # K = 2 onsets
    p = np.array([[60.0, 62.0, 64.0]])
    kw = dict(per=False, period=0.0, drop_input_attr=True)
    w_start = mpt.unpack_pre_maet(weight_events(
        [p, t], None, 1, 0, 1.0, 1.0, width=2.0, locate="start",
        **kw))[1][0]
    np.testing.assert_array_equal(np.ravel(w_start), [1.0, 1.0, 0.0])
    w_open = mpt.unpack_pre_maet(weight_events(
        [p, t[:1]], None, 1, 0, 1.0, 1.0, width=2.0, **kw))[1][0]
    w_closed = mpt.unpack_pre_maet(weight_events(
        [p, t[:1]], None, 1, 0, 1.0, 1.0, width=2.0, edges="closed",
        **kw))[1][0]
    np.testing.assert_array_equal(np.ravel(w_open), [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(np.ravel(w_closed), [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="rectangle"):
        weight_events([p, t[:1]], None, 1, 0, 1.0, 0.0, width=2.0,
                      edges="closed", **kw)
