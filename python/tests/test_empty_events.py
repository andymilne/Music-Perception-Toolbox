"""Events empty on some attributes and populated on others.

A grid point with nothing sounding is an event that carries a time and no
pitch. It has to be keepable: the populated attributes are doing work even
where one is empty, and dropping it would make the event index no longer a
uniform time index. Such an event admits no tuple and so contributes
nothing, to the density, the inner product, or any entropy taken from
them. A partly filled event -- fewer than r values but more than none --
likewise contributes nothing, a NaN being a position with no value, read as
a value of weight 0; it is warned of, since it may be a mistake. Only an
attribute whose positions, values and NaNs together, are fewer than r is an
error.

Mirror of MATLAB tests/test_empty_events.m.
"""

import numpy as np
import pytest

from mpt import build_maet, eval_maet, sim_maet, entropy_maet

TOL = 1e-12


def _two_attr(p_attr, r=(1, 1)):
    return build_maet(p_attr, None, [1.0, 0.1], list(r), [False, False],
                      [False, False], [0, 0], verbose=False)


def _one_attr(p, r):
    return build_maet([p], None, [1.0], [r], [False], [False], [0],
                      verbose=False)


@pytest.fixture
def with_and_without():
    """One live event and one empty on pitch, against the live one alone."""
    with_empty = _two_attr([np.array([[60.0, np.nan]]),
                            np.array([[0.0, 1.0]])])
    alone = _two_attr([np.array([[60.0]]), np.array([[0.0]])])
    return with_empty, alone


def test_an_event_empty_on_one_attribute_builds(with_and_without):
    with_empty, _ = with_and_without
    assert with_empty.n == 2


def test_it_contributes_nothing_to_the_density(with_and_without):
    with_empty, alone = with_and_without
    query = [[60.0], [0.0]]
    assert eval_maet(with_empty, query, verbose=False) == pytest.approx(
        eval_maet(alone, query, verbose=False), abs=TOL)


def test_it_contributes_nothing_to_the_inner_product(with_and_without):
    with_empty, alone = with_and_without
    assert sim_maet(with_empty, alone, verbose=False) == pytest.approx(1.0,
                                                                      abs=TOL)
    assert sim_maet(with_empty, with_empty, verbose=False) == pytest.approx(
        1.0, abs=TOL)


def test_it_contributes_nothing_to_the_entropy(with_and_without):
    """Renyi-2 is closed form, so the two agree exactly rather than to a
    quadrature tolerance."""
    with_empty, alone = with_and_without
    assert entropy_maet(with_empty, method="renyi2",
                        verbose=False) == pytest.approx(
        entropy_maet(alone, method="renyi2", verbose=False), abs=TOL)


def test_a_silent_slice_beside_a_chord_at_r_two():
    d = _one_attr(np.array([[60.0, np.nan], [64.0, np.nan]]), 2)
    alone = _one_attr(np.array([[60.0], [64.0]]), 2)
    assert sim_maet(d, alone, verbose=False) == pytest.approx(1.0, abs=TOL)


def test_a_nested_attribute_tolerates_an_empty_event():
    values = np.array([[60.0, np.nan], [64.0, np.nan],
                       [62.0, np.nan], [65.0, np.nan]])
    spec = {"r": [2, 2], "exch": [0, 1],
            "tags": np.array([[0], [0], [1], [1]])}
    d = build_maet([values], None, [1.0], [1], [False], [False], [0],
                   nested=[spec], verbose=False)
    assert sim_maet(d, d, verbose=False) == pytest.approx(1.0, abs=TOL)


def test_every_event_empty_gives_a_zero_mass_density():
    d = _one_attr(np.array([[np.nan, np.nan]]), 1)
    assert eval_maet(d, [[60.0]], verbose=False) == pytest.approx(0.0,
                                                                  abs=TOL)


@pytest.mark.parametrize("exch", [True, False])
def test_a_partly_filled_event_contributes_nothing_with_a_warning(exch):
    """Two values asked of an event that has one: its NaN is a value of
    weight 0, so the event gives no pair, as if it were not there."""
    P = np.array([[60.0, 62.0, 65.0], [64.0, np.nan, 69.0]])
    with pytest.warns(UserWarning, match="1 event\\(s\\) on attribute 0 hold too few"):
        d = build_maet([P], None, [1.0], [2], [False], [False], [0], [exch],
                       verbose=False)
    alone = build_maet([P[:, [0, 2]]], None, [1.0], [2], [False], [False],
                       [0], [exch], verbose=False)
    assert sim_maet(d, alone, verbose=False) == pytest.approx(1.0, abs=TOL)
    assert entropy_maet(d, method="renyi2", verbose=False) == pytest.approx(
        entropy_maet(alone, method="renyi2", verbose=False), abs=1e-9)


def test_an_ordered_whole_tuple_with_a_gap_contributes_nothing():
    """At r = K on an ordered attribute the one tuple runs through every
    position, so a single missing value removes the event."""
    P = np.array([[60.0, 62.0, 65.0], [64.0, np.nan, 69.0],
                  [67.0, 71.0, 72.0]])
    with pytest.warns(UserWarning, match="too few non-NaN"):
        d = build_maet([P], None, [1.0], [3], [False], [False], [0], [False],
                       verbose=False)
    alone = build_maet([P[:, [0, 2]]], None, [1.0], [3], [False], [False],
                       [0], [False], verbose=False)
    assert sim_maet(d, alone, verbose=False) == pytest.approx(1.0, abs=TOL)


@pytest.mark.parametrize("exch", [True, False])
def test_fewer_positions_than_r_is_an_error(exch):
    """No event of a two-position attribute can supply a triple."""
    with pytest.raises(ValueError, match="2 position\\(s\\) per event"):
        build_maet([np.array([[60.0, 62.0], [64.0, 65.0]])], None, [1.0], [3],
                   [False], [False], [0], [exch], verbose=False)


def test_a_nested_event_without_a_full_tuple_contributes_nothing():
    V = np.array([[60, 60, 61.], [64, 64, 65], [67, np.nan, 68],
                  [72, np.nan, 73]])
    spec = {"tags": np.array([[0], [0], [1], [1]]), "r": [2, 2],
            "exch": [True, False], "rel": [False, False]}
    with pytest.warns(UserWarning, match="too few non-NaN"):
        d = build_maet([V], None, [1.0], [4], [False], [False], [0],
                       nested=[spec], verbose=False)
    alone = build_maet([V[:, [0, 2]]], None, [1.0], [4], [False], [False],
                       [0], nested=[spec], verbose=False)
    assert sim_maet(d, alone, verbose=False) == pytest.approx(1.0, abs=TOL)


def test_nested_positions_without_a_full_tuple_are_an_error():
    spec = {"tags": np.array([[0], [0]]), "r": [2, 2],
            "exch": [True, False], "rel": [False, False]}
    with pytest.raises(ValueError, match="no event can supply one"):
        build_maet([np.array([[60.0], [64.0]])], None, [1.0], [4], [False],
                   [False], [0], nested=[spec], verbose=False)
