"""``mass_maet`` and ``swept_mass``: the total mass of a density.

The total mass is the sum of the density's tuples' weight products, each
kernel taken with unit mass. It is pinned to the density's own tuple
weights (``w_j``), to hand-computed weight sums, and, for ``swept_mass``,
to the explicit composition it stands in for (``weight_events`` ->
``build_maet`` -> ``mass_maet``). Its use as the normalizer of a
one-sided similarity is pinned on the diatonic scale's fifths. The values
pinned here are shared with ``test_massMaet.m``.
"""
import numpy as np
import pytest

import mpt
from mpt import (bind_events, build_maet, mass_maet, pack_pre_maet,
                 swept_mass, unpack_pre_maet, weight_events)

P1 = [np.array([[60.0, 64.0, 67.0]]).T]
W1 = [np.array([[1.0, 0.5, 2.0]]).T]


@pytest.fixture(autouse=True)
def _quiet():
    prev = mpt.set_default(show_hints=False)
    yield
    mpt.set_default(**prev)


def _flat(sigma, r, rel, per, period, **kw):
    return build_maet(P1, W1, [sigma], [r], [rel], [per], [period],
                      verbose=False, **kw)


def test_total_is_sum_of_weight_products():
    for r, rel, per in [(1, False, False), (2, False, False),
                        (2, True, False), (3, True, False),
                        (2, True, True), (3, False, True)]:
        d = _flat(1.0, r, rel, per, 1200.0 if per else 0.0)
        assert mass_maet(d) == pytest.approx(float(np.sum(d.w_j)),
                                             rel=1e-14)


def test_values_by_hand():
    """Weights 1, 0.5, and 2: r = 1 sums them (3.5); an exchangeable pair
    takes both orders of each pair of values, 2 (0.5 + 2 + 1) = 7; an
    ordered pair one order only, 3.5; an exchangeable triple all six
    orders of the one triple, 6."""
    assert mass_maet(_flat(1.0, 1, False, False, 0.0)) == pytest.approx(3.5)
    assert mass_maet(_flat(1.0, 2, True, False, 0.0)) == pytest.approx(7.0)
    d = build_maet(P1, W1, [1.0], [2], [False], [False], [0.0], [False],
                   verbose=False)
    assert mass_maet(d) == pytest.approx(3.5)
    assert mass_maet(_flat(1.0, 3, True, False, 0.0)) == pytest.approx(6.0)


def test_multi_attribute_product():
    pa = [np.array([[60.0, 62.0, 64.0, 65.0]]),
          np.array([[0.0, 1.0, 2.0, 3.0]])]
    wa = [np.array([[1.0, 2.0, 1.0, 0.5]]), np.array([[1.0, 1.0, 3.0, 2.0]])]
    d = build_maet(pa, wa, [1.0, 0.2], [1, 1], [False, False],
                   [False, False], [0.0, 0.0], verbose=False)
    assert mass_maet(d) == pytest.approx(1 + 2 + 3 + 1)


def test_nested_is_sum_of_weight_products():
    pit = np.array([[60.0, 62.0, 64.0, 65.0, 67.0]])
    on = np.arange(5.0)[None, :]
    pb, wb, sb = unpack_pre_maet(bind_events([pit, on], None, [2, 1],
                                             step=1))
    d = build_maet(pb, wb, sigma=[0.5, 0.2], per=[False, False],
                   period=[0.0, 0.0], specs=sb, verbose=False)
    assert mass_maet(d) == pytest.approx(float(np.sum(d.w_j)), rel=1e-14)


def test_kernel_covariance_attribute():
    p = [np.array([[0.0, 1.0, 2.0]]).T]
    dc = build_maet(p, None, [0.25 * np.eye(3)], [3], [False], [False],
                    [0.0], [False], verbose=False)
    assert mass_maet(dc) == pytest.approx(float(np.sum(dc.w_j)))


def test_pre_maet_and_list_inputs():
    pm = pack_pre_maet(P1, W1, mpt.flat_specs(P1, sigma=[1.0],
                                               per=[False],
                                               period=[0.0]))
    d = build_maet(pm, verbose=False)
    assert mass_maet(pm) == mass_maet(d)
    out = mass_maet([d, pm])
    assert isinstance(out, np.ndarray) and out.shape == (2,)


def test_region_is_gone():
    with pytest.raises(TypeError):
        mass_maet(_flat(1.0, 1, False, False, 0.0), {0: (61, 66)})


def test_normalizes_a_one_sided_similarity():
    """A relative, periodic, exchangeable dyad density: the one-sided
    similarity with a lone fifth counts the scale's pairs a fifth or a
    fourth apart, and the ratio of masses makes it their share of all
    pairs (6 of the diatonic scale's 21), up to the kernels' overlap with
    neighbouring intervals (about 2e-5 at sigma = 10 cents)."""
    diat = np.array([0.0, 200, 400, 500, 700, 900, 1100])
    fifth = np.array([0.0, 700.0])
    geom = (10, 2, True, True, 1200)
    count = float(mpt.sim_maet(diat, None, fifth, None, *geom,
                               normalize="oneSidedDenom", verbose=False))
    m_d = mass_maet(build_maet(diat, None, *geom, verbose=False))
    m_f = mass_maet(build_maet(fifth, None, *geom, verbose=False))
    assert count == pytest.approx(6.0, abs=1e-4)
    assert (m_d, m_f) == (pytest.approx(42.0), pytest.approx(2.0))
    assert count * m_f / m_d == pytest.approx(6 / 21, abs=1e-5)


# -----------------------------------------------------------------------
# swept_mass
# -----------------------------------------------------------------------


def _melody():
    pit = np.array([[60, 62, 64, 65, 67, 65, 64, 62, 60, 67, 72, 67]],
                   dtype=float)
    on = np.arange(12.0)[None, :]
    return pack_pre_maet([pit, on], None,
                         mpt.flat_specs([pit, on], sigma=[0.5, 0.1],
                                        per=[False, False],
                                        period=[0.0, 0.0]))


def test_swept_mass_matches_composition():
    pm = _melody()
    vals = np.arange(0.0, 12.0, 2.0)
    win = {1: {"shape": "gaussian", "sd": 1.5}}
    got = swept_mass(pm, sweep={1: vals}, window=win, drop=1)
    p, w, s = unpack_pre_maet(pm)
    for i, c in enumerate(vals):
        pw, ww, _ = unpack_pre_maet(weight_events(
            p, w, 1, 0, float(c), 0.0, sd=1.5, drop_input_attr=True,
            specs=s))
        d = build_maet(pw, ww, [0.5], [1], [False], [False], [0.0],
                       verbose=False)
        assert got[i] == pytest.approx(mass_maet(d), rel=1e-12)


def test_swept_mass_counts_the_window():
    pm = _melody()
    got = swept_mass(pm, sweep={1: [2.0, 5.0]},
                     window={1: {"shape": "rect", "width": 4.0}},
                     drop=1)
    assert np.allclose(got, [4.0, 4.0])


def test_swept_mass_takes_no_region():
    with pytest.raises(TypeError):
        swept_mass(_melody(), sweep={1: [2.0]},
                   window={1: {"shape": "rect", "width": 4.0}},
                   drop=1, region={0: (60, 62)})
