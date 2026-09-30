"""A rectangle's reference value: 'ref' in a window specification.

A window's reference value is the point placed at each sweep value. For a
rectangle it is its centre by default; ``'ref': 'start'`` places its
start there, so it covers [s, s + width), and ``'ref': 'end'`` its end,
covering [s - width, s). Either is the centred window at a shifted sweep
value, and a rectangle's default sweep values (its pieces) shift with it.

Mirror of MATLAB tests/test_swept_window_ref.m.
"""

import numpy as np
import pytest

import mpt

CLAVE = np.array([0.0, 3, 6, 10, 12])
T = np.concatenate([CLAVE, CLAVE + 16])
P = 100.0 * np.array([60, 62, 64, 67, 64, 67, 69, 71, 74, 71])
CTX = [P[None, :], T[None, :]]
QRY = [P[None, :3], T[None, :3]]
GEOM = ([20.0, 0.5], [1, 1], [False, False], [True, False], [1200.0, 0.0])
W = 10.0
S = np.arange(0.0, 20.5, 0.5)


def _rect(ref=None, width=W, **kw):
    spec = {"shape": "rect", "width": width, **kw}
    if ref is not None:
        spec["ref"] = ref
    return spec


G_MAJOR = [100.0 * np.array([[67, 71, 74]]), np.zeros((1, 3))]


def _mass(win, **kw):
    """The G-major triad's presence in the window: the one-sided
    similarity of the notes in it with the triad, a count of the notes on
    its tones in units of the triad's three."""
    return mpt.swept_similarity(CTX, None, G_MAJOR, None, *GEOM,
                                align={1: "window"}, window={1: win},
                                drop=[1], normalize="oneSidedDenom",
                                verbose=False, **kw)


def _total(win, **kw):
    return mpt.swept_mass(CTX, None, *GEOM, window={1: win}, drop=[1],
                          verbose=False, **kw)


def test_start_and_end_are_the_centred_window_shifted():
    centre = _mass(_rect(), sweep={1: S})
    np.testing.assert_allclose(_mass(_rect("start"), sweep={1: S}),
                               _mass(_rect(), sweep={1: S + W / 2}),
                               rtol=0, atol=1e-14)
    np.testing.assert_allclose(_mass(_rect("end"), sweep={1: S}),
                               _mass(_rect(), sweep={1: S - W / 2}),
                               rtol=0, atol=1e-14)
    np.testing.assert_allclose(_mass(_rect("centre"), sweep={1: S}), centre,
                               rtol=0, atol=0)
    sim = lambda win, s: mpt.swept_similarity(
        CTX, None, QRY, None, *GEOM, sweep={1: s}, align={1: "window"},
        window={1: win}, drop=[1], verbose=False)
    np.testing.assert_allclose(sim(_rect("start"), S),
                               sim(_rect(), S + W / 2), rtol=0, atol=1e-14)
    np.testing.assert_allclose(_total(_rect("end"), sweep={1: S}),
                               _total(_rect(), sweep={1: S - W / 2}),
                               rtol=0, atol=1e-14)


def test_half_open_edges_follow_the_rectangle():
    """Half-open: the lower edge is included and the upper not, wherever
    the reference lies. With width 3, onsets 3 and 6 are one width apart,
    so a window starting at 3 or ending at 6 holds the note at 3 only;
    closed, it holds both."""
    count = lambda win, s: float(_total(win, sweep={1: [s]})[0])
    assert count(_rect("start", 3.0), 3.0) == pytest.approx(1.0)
    assert count(_rect("end", 3.0), 6.0) == pytest.approx(1.0)
    assert count(_rect("start", 3.0, edges="closed"), 3.0) == \
        pytest.approx(2.0)
    assert count(_rect("end", 3.0, edges="closed"), 6.0) == \
        pytest.approx(2.0)


def test_default_pieces_shift_with_the_reference():
    m, sv = _mass(_rect("start"), sweep=1, return_sweep_values=True)
    s = sv[1]
    assert s[0] == T.min() and s[-1] == T.max()
    # both ends are breakpoints here (the first and last onsets), so each
    # keeps its own value and the piece beside it is sampled just inside
    assert 0 < s[1] - s[0] < 1e-4 and 0 < s[-1] - s[-2] < 1e-4
    brk = np.mean(s[2:-2].reshape(-1, 2), axis=1)
    want = np.unique(np.concatenate([T - W, T]))
    want = want[(want > T.min()) & (want < T.max())]
    np.testing.assert_allclose(brk, want, atol=1e-9)
    np.testing.assert_allclose(m, _mass(_rect(), sweep={1: s + W / 2}),
                               rtol=0, atol=1e-14)


def test_an_end_on_a_breakpoint_keeps_its_own_value():
    """With the window starting at s, the window at s = 0 holds C, D, and
    E, and just above it D, E, and G: the C leaves and the G at onset 10
    enters, so the G-major count rises from one note to two. The first
    piece is sampled just above 0 as well as at 0, so no line joins the
    two values."""
    m, sv = _mass(_rect("start"), sweep=1, return_sweep_values=True)
    s = sv[1]
    assert m[0] == pytest.approx(float(_mass(_rect("start"),
                                             sweep={1: [0.0]})[0]))
    assert m[1] == pytest.approx(float(_mass(_rect("start"),
                                             sweep={1: [1.0]})[0]))
    assert m[0] != pytest.approx(m[1])
    assert m[1] == pytest.approx(m[2])


def test_refusals():
    with pytest.raises(ValueError, match="needs a rectangle"):
        _mass({"shape": "gaussian", "sd": 2.0, "ref": "start"}, sweep={1: S})
    with pytest.raises(ValueError, match="'centre', 'start', or 'end'"):
        _mass(_rect("middle"), sweep={1: S})
    with pytest.raises(ValueError, match="needs a given width"):
        mpt.swept_similarity(CTX, None, QRY, None, *GEOM, sweep={1: S},
                             align={1: "both"},
                             window={1: {"shape": "rect", "ref": "start"}},
                             verbose=False)
