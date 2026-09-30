"""``mass_maet`` and ``swept_mass``: the mass of a density in a region.

Each closed form is pinned to a numerical integral of the density
``eval_maet`` returns under ``normalize='gaussian'`` (kernels of unit
mass, untruncated), and ``swept_mass`` to the explicit composition it
stands in for (``weight_events`` -> ``build_maet`` -> ``mass_maet``).
The values pinned at the end are shared with ``test_massMaet.m``.
"""
import numpy as np
import pytest
from scipy.integrate import dblquad, quad
from scipy.stats import norm

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


def _ev(dens, *x):
    X = np.array([[v] for v in x], dtype=float)
    return float(np.asarray(mpt.eval_maet(
        dens, X, "gaussian", truncation_sigmas=np.inf,
        verbose=False)).ravel()[0])


def _flat(sigma, r, rel, per, period, **kw):
    return build_maet(P1, W1, [sigma], [r], [rel], [per], [period],
                      verbose=False, **kw)


# -----------------------------------------------------------------------
# Closed forms against numerical integration
# -----------------------------------------------------------------------


def test_total_is_sum_of_weight_products():
    for r, rel in [(1, False), (2, False), (2, True), (3, True)]:
        d = _flat(1.0, r, rel, False, 0.0)
        assert mass_maet(d) == pytest.approx(float(np.sum(d.w_j)), rel=1e-14)
        assert mass_maet(d, normalize="total") == pytest.approx(1.0)


def test_box_absolute_r1():
    d = _flat(1.0, 1, False, False, 0.0)
    num = quad(lambda x: _ev(d, x), 61, 66)[0]
    assert mass_maet(d, {0: (61, 66)}) == pytest.approx(num, abs=1e-10)


def test_box_absolute_r2_per_coordinate_rows():
    d = _flat(1.0, 2, False, False, 0.0)
    B = np.array([[59.0, 62.0], [63.0, 70.0]])
    num = dblquad(lambda y, x: _ev(d, x, y), 59, 62, 63, 70)[0]
    assert mass_maet(d, {0: B}) == pytest.approx(num, abs=1e-9)


def test_box_relative_dyads_is_an_interval_band():
    d = _flat(1.0, 2, True, False, 0.0)
    num = quad(lambda x: _ev(d, x), 2, 5)[0]
    assert mass_maet(d, {0: (2, 5)}) == pytest.approx(num, abs=1e-10)
    # An exchangeable density holds both orderings: the band and its
    # negative carry the same mass.
    assert mass_maet(d, {0: (-5, -2)}) == pytest.approx(
        mass_maet(d, {0: (2, 5)}), rel=1e-12)


def test_box_periodic_full_image():
    d = _flat(2.5, 1, False, True, 12.0)
    num = quad(lambda x: _ev(d, x), 10, 17, limit=200)[0]
    assert mass_maet(d, {0: (10, 17)}) == pytest.approx(num, abs=1e-10)
    assert mass_maet(d, {0: (0, 12)}) == pytest.approx(3.5, rel=1e-12)
    # Bounds may wrap: (10, 17) is (10, 12) and (0, 5) on the circle.
    assert mass_maet(d, {0: (10, 17)}) == pytest.approx(
        mass_maet(d, {0: (-2, 5)}), rel=1e-12)


def test_box_periodic_single_image_is_renormalized():
    d = _flat(2.5, 1, False, True, 12.0, wrap=["single-image"])
    cut = 1.0 - 2.0 * norm.cdf(-6.0 / 2.5)
    # The single-image kernel has a kink at each centre's antipode (13
    # here), so the quadrature is split there.
    num = quad(lambda x: _ev(d, x), 10, 17, limit=200, points=[13],
               epsabs=1e-13, epsrel=1e-12)[0] / cut
    assert mass_maet(d, {0: (10, 17)}) == pytest.approx(num, abs=1e-10)
    assert mass_maet(d, {0: (3, 15)}) == pytest.approx(3.5, rel=1e-12)


def test_box_relative_periodic_dyads():
    d = _flat(0.8, 2, True, True, 12.0)
    assert mass_maet(d, {0: (0, 12)}) == pytest.approx(mass_maet(d),
                                                        rel=1e-12)
    sd = 0.8 * np.sqrt(2.0)
    cut = 1.0 - 2.0 * norm.cdf(-6.0 / sd)
    num = quad(lambda x: _ev(d, x), 2, 6, limit=200)[0] / cut
    assert mass_maet(d, {0: (2, 6)}) == pytest.approx(num, rel=1e-6)


def test_gaussian_region_absolute():
    d = _flat(1.0, 1, False, False, 0.0)
    num = quad(lambda x: _ev(d, x) * np.exp(-(x - 63) ** 2 / 4.5), 40, 90)[0]
    assert mass_maet(d, {0: ("gaussian", 63, 1.5)}) == pytest.approx(
        num, abs=1e-10)


def test_gaussian_region_periodic():
    d = _flat(2.5, 1, False, True, 12.0)

    def g(x):
        dd = (x - 1 + 6) % 12 - 6
        return np.exp(-dd ** 2 / 8.0)

    num = quad(lambda x: _ev(d, x) * g(x), -5, 7, limit=200)[0]
    assert mass_maet(d, {0: ("gaussian", 1, 2.0)}) == pytest.approx(
        num, abs=1e-10)


def test_gaussian_region_on_correlated_relative_coordinates():
    p = [np.array([[0.0, 3.0, 7.0, 10.0]]).T]
    d = build_maet(p, None, [1.0], [3], [True], [False], [0.0],
                   verbose=False)
    num = dblquad(lambda y, x: _ev(d, x, y) * np.exp(
        -((x - 3) ** 2 + (y - 7) ** 2) / (2 * 1.2 ** 2)),
        -5, 12, -5, 15)[0]
    assert mass_maet(d, {0: ("gaussian", [3, 7], 1.2)}) == pytest.approx(
        num, abs=1e-8)


def test_multi_attribute_box_and_share():
    pa = [np.array([[60.0, 62.0, 64.0, 65.0]]),
          np.array([[0.0, 1.0, 2.0, 3.0]])]
    d = build_maet(pa, None, [1.0, 0.2], [1, 1], [False, False],
                   [False, False], [0.0, 0.0], verbose=False)
    num = dblquad(lambda y, x: _ev(d, x, y), 59, 63.5, -1, 1.5)[0]
    assert mass_maet(d, {0: (59, 63.5), 1: (-1, 1.5)}) == pytest.approx(
        num, abs=1e-9)
    share = mass_maet(d, {0: (59, 63.5)}, normalize="total")
    assert share == pytest.approx(mass_maet(d, {0: (59, 63.5)}) / 4.0)


def test_nested_absolute_box():
    pit = np.array([[60.0, 62.0, 64.0, 65.0, 67.0]])
    on = np.arange(5.0)[None, :]
    pb, wb, sb = unpack_pre_maet(bind_events([pit, on], None, [2, 1],
                                             step=1))
    d = build_maet(pb, wb, sigma=[0.5, 0.2], is_per=[False, False],
                   period=[0.0, 0.0], specs=sb, verbose=False)
    assert mass_maet(d) == pytest.approx(
        mass_maet(d, {0: np.array([[-np.inf, np.inf]])}), rel=1e-14)
    got = mass_maet(d, {0: np.array([[59, 63], [61, 66]])})
    ref = 0.0
    for n in range(pb[0].shape[1]):
        a, b = pb[0][:, n]
        ref += ((norm.cdf((63 - a) / 0.5) - norm.cdf((59 - a) / 0.5))
                * (norm.cdf((66 - b) / 0.5) - norm.cdf((61 - b) / 0.5)))
    assert got == pytest.approx(ref, rel=1e-12)


def test_pre_maet_and_list_inputs():
    pm = pack_pre_maet(P1, W1, mpt.flat_specs(P1, sigma=[1.0], is_per=[False],
                                               period=[0.0]))
    d = build_maet(pm, verbose=False)
    assert mass_maet(pm, {0: (61, 66)}) == mass_maet(d, {0: (61, 66)})
    out = mass_maet([d, pm], {0: (61, 66)})
    assert isinstance(out, np.ndarray) and out.shape == (2,)


# -----------------------------------------------------------------------
# Refusals
# -----------------------------------------------------------------------


def test_refusals():
    d3 = _flat(1.0, 3, True, False, 0.0)
    with pytest.raises(ValueError, match="no closed form"):
        mass_maet(d3, {0: (0, 5)})
    with pytest.warns(UserWarning, match="degenerate"):
        d1 = _flat(1.0, 1, True, False, 0.0)
    with pytest.raises(ValueError, match="no coordinates"):
        mass_maet(d1, {0: (0, 5)})
    dp = _flat(1.0, 1, False, True, 12.0)
    with pytest.raises(ValueError, match="one period"):
        mass_maet(dp, {0: (0, 13)})
    with pytest.raises(ValueError, match="lo <= hi"):
        mass_maet(dp, {0: (5, 1)})
    with pytest.raises(ValueError, match="normalize"):
        mass_maet(dp, normalize="pdf")
    with pytest.raises(ValueError, match="rows"):
        mass_maet(_flat(1.0, 2, False, False, 0.0),
                  {0: np.zeros((3, 2))})
    p = [np.array([[0.0, 1.0, 2.0]]).T]
    dc = build_maet(p, None, [0.25 * np.eye(3)], [3], [False], [False],
                    [0.0], [False], verbose=False)
    with pytest.raises(ValueError, match="kernel covariance"):
        mass_maet(dc, {0: (0, 1)})
    assert mass_maet(dc) == pytest.approx(float(np.sum(dc.w_j)))


# -----------------------------------------------------------------------
# swept_mass
# -----------------------------------------------------------------------


def _melody():
    pit = np.array([[60, 62, 64, 65, 67, 65, 64, 62, 60, 67, 72, 67]],
                   dtype=float)
    on = np.arange(12.0)[None, :]
    return pack_pre_maet([pit, on], None,
                         mpt.flat_specs([pit, on], sigma=[0.5, 0.1],
                                        is_per=[False, False],
                                        period=[0.0, 0.0]))


def test_swept_mass_matches_composition():
    pm = _melody()
    vals = np.arange(0.0, 12.0, 2.0)
    got = swept_mass(pm, sweep={1: vals}, window={1: ("rect", 4.0)},
                     drop=1, region={0: (63.5, 67.5)})
    got_share = swept_mass(pm, sweep={1: vals}, window={1: ("rect", 4.0)},
                           drop=1, region={0: (63.5, 67.5)},
                           normalize="total")
    p, w, s = unpack_pre_maet(pm)
    for i, c in enumerate(vals):
        pw, ww, sw = unpack_pre_maet(weight_events(
            p, w, 1, 0, float(c), 1.0, width=4.0, drop_input_attr=True,
            specs=s))
        d = build_maet(pw, ww, [0.5], [1], [False], [False], [0.0],
                       verbose=False)
        assert got[i] == pytest.approx(mass_maet(d, {0: (63.5, 67.5)}),
                                       rel=1e-12)
        assert got_share[i] == pytest.approx(
            mass_maet(d, {0: (63.5, 67.5)}, normalize="total"), rel=1e-12)


def test_swept_mass_without_region_counts_the_window():
    pm = _melody()
    got = swept_mass(pm, sweep={1: [2.0, 5.0]}, window={1: ("rect", 4.0)},
                     drop=1)
    assert np.allclose(got, [4.0, 4.0])


def test_swept_mass_refuses_region_on_dropped_attribute():
    with pytest.raises(ValueError, match="dropped"):
        swept_mass(_melody(), sweep={1: [2.0]}, window={1: ("rect", 4.0)},
                   drop=1, region={1: (0, 3)})


# -----------------------------------------------------------------------
# Values shared with test_massMaet.m
# -----------------------------------------------------------------------


def test_cross_language_values():
    assert mass_maet(_flat(1.0, 2, True, False, 0.0), {0: (2, 5)}) == \
        pytest.approx(1.1795017491767164, rel=1e-12)
    assert mass_maet(_flat(2.5, 1, False, True, 12.0), {0: (10, 17)}) == \
        pytest.approx(1.7491633889274478, rel=1e-12)
    assert mass_maet(_flat(2.5, 1, False, True, 12.0,
                           wrap=["single-image"]), {0: (10, 17)}) == \
        pytest.approx(1.7385705553936834, rel=1e-12)
    assert mass_maet(_flat(2.5, 1, False, True, 12.0),
                     {0: ("gaussian", 1, 2.0)}) == \
        pytest.approx(1.2307207984885715, rel=1e-12)
