"""The sweep values returned by the swept functions.

``return_sweep_values=True`` makes ``swept_similarity``, ``swept_entropy``
and ``swept_mass`` also return the sweep values, listed or generated, as
a dict ``{a: values}``: the axes of the profile.

Mirror of MATLAB tests/test_swept_sweep_values.m.
"""

import numpy as np

import mpt

CLAVE = np.array([0.0, 3, 6, 10, 12])
T = np.concatenate([CLAVE, CLAVE + 16])
P = 100.0 * np.array([60, 62, 64, 67, 64, 67, 69, 71, 74, 71])
CTX = [P[None, :], T[None, :]]
QRY = [P[None, :3], T[None, :3]]
GEOM = ([20.0, 0.5], [1, 1], [False, False], [True, False], [1200.0, 0.0])
WIN = {"shape": "gaussian", "sd": 4.0}
S_LIST = np.arange(0, 28.5, 0.5)


def _mass(**kw):
    return mpt.swept_mass(CTX, None, *GEOM, window={1: WIN}, drop=[1],
                          region={0: (-50, 50)}, normalize="total",
                          return_sweep_values=True, **kw)


def test_listed_values_returned_as_given():
    m, sv = _mass(sweep={1: S_LIST})
    assert set(sv) == {1}
    np.testing.assert_array_equal(sv[1], S_LIST)
    assert m.shape == S_LIST.shape


def test_generated_values_match_explicit_list():
    m_l, _ = _mass(sweep={1: S_LIST})
    m_g, sv = _mass(sweep=1, step=0.5)
    np.testing.assert_allclose(sv[1], S_LIST, atol=1e-12)
    np.testing.assert_allclose(m_g, m_l, atol=1e-12)


def test_defaulted_values_are_axes_of_profile():
    h, sv = mpt.swept_entropy(CTX, None, *GEOM, sweep=1, window={1: WIN},
                              drop=[1], method="renyi2",
                              return_sweep_values=True)
    assert sv[1].size == h.size and sv[1][0] == 0 and sv[1][-1] <= 28


def test_default_return_unchanged():
    out = mpt.swept_mass(CTX, None, *GEOM, sweep=1, window={1: WIN},
                         drop=[1], region={0: (-50, 50)})
    assert isinstance(out, np.ndarray)


def test_similarity_query_equals_offsets_both_differs():
    _, mu_q, sv_q = mpt.swept_similarity(
        CTX, None, QRY, None, *GEOM, sweep=1, return_offsets=True,
        return_sweep_values=True)
    np.testing.assert_allclose(sv_q[1], mu_q[1], atol=1e-12)
    _, mu_b, sv_b = mpt.swept_similarity(
        CTX, None, QRY, None, *GEOM, sweep=1, align={1: "both"},
        return_offsets=True, return_sweep_values=True)
    np.testing.assert_allclose(sv_b[1] - mu_b[1], T[:3].mean(), atol=1e-12)


def test_similarity_sweep_values_alone():
    out = mpt.swept_similarity(CTX, None, QRY, None, *GEOM, sweep=1,
                               return_sweep_values=True)
    assert len(out) == 2 and isinstance(out[1], dict)


def test_independent_gives_window_and_query_lists():
    s, sv = mpt.swept_similarity(
        CTX, None, QRY, None, *GEOM, sweep={1: ([0.0, 16.0], [0.0, 8, 16])},
        align={1: "independent"}, window={1: ("rect", 8.0)},
        return_sweep_values=True)
    w_vals, q_vals = sv[1]
    np.testing.assert_array_equal(w_vals, [0, 16])
    np.testing.assert_array_equal(q_vals, [0, 8, 16])
    assert s.shape == (2, 3)


def test_rectangle_takes_its_pieces():
    """A pure rectangle placed alone makes the profile piecewise constant:
    its default sweep values sample each piece just inside both ends, the
    breakpoints being each event's value plus or minus half the width, and
    each value is the profile's value at its sweep value and throughout
    its piece."""
    width = 4.0
    m, sv = mpt.swept_mass(CTX, None, *GEOM, sweep=1,
                           window={1: ("rect", width)}, drop=[1],
                           region={0: (-50, 50)}, normalize="total",
                           return_sweep_values=True)
    s = sv[1]
    assert s[0] == T.min() and s[-1] == T.max()
    breaks = np.unique(np.concatenate([T - width / 2, T + width / 2]))
    breaks = breaks[(breaks > T.min()) & (breaks < T.max())]
    inner = s[1:-1].reshape(-1, 2)
    np.testing.assert_allclose(inner.mean(axis=1), breaks, atol=1e-9)
    assert np.all(np.diff(inner, axis=1) < 1e-4)
    # each piece is constant: its midpoint has the value of its two ends
    edges = np.concatenate([[T.min()], breaks, [T.max()]])
    mids = (edges[:-1] + edges[1:]) / 2
    m_mid = mpt.swept_mass(CTX, None, *GEOM, sweep={1: mids},
                           window={1: ("rect", width)}, drop=[1],
                           region={0: (-50, 50)}, normalize="total")
    left, right = m[0::2], m[1::2]
    np.testing.assert_allclose(left, m_mid, atol=1e-12)
    np.testing.assert_allclose(right, m_mid, atol=1e-12)
    # a given step keeps the uniform grid
    _, su = mpt.swept_mass(CTX, None, *GEOM, sweep=1, step=1.0,
                           window={1: ("rect", width)}, drop=[1],
                           return_sweep_values=True)
    np.testing.assert_allclose(np.diff(su[1]), 1.0)

