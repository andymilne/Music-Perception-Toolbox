"""Events empty on some attributes and populated on others.

A grid point with nothing sounding is an event that carries a time and no
pitch. It has to be keepable: the populated attributes are doing work even
where one is empty, and dropping it would make the event index no longer a
uniform time index. Such an event admits no tuple and so contributes
nothing, to the density, the inner product, or any entropy taken from
them. A partly filled event -- fewer than r values but more than none --
is still an error.

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


def test_a_partly_filled_event_is_still_an_error():
    """Two values asked of an event that has one is a mistake, not a rest."""
    with pytest.raises(ValueError, match="non-NaN value"):
        _one_attr(np.array([[60.0, 62.0], [64.0, np.nan]]), 2)
