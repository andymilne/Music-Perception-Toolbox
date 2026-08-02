"""The spectral IP cost gate, and the regime that pins its constant.

``_SPECTRAL_IP_COST_C`` decides whether the mode grid is repaid against
the translation grid's K^2 per event pair. Its value is a measurement,
not a convention, so these tests pin the behaviour the measurement
established rather than the number alone --- a future change should have
to break a stated property, not just edit a constant.

The regime that matters most is small sigma/P. There the mode grid is
large while K and the event count are small, and taking the branch is
several times slower. That corner is what the constant exists for: an
earlier attempt to remove the gate entirely looked correct on a grid
that started at sigma/P = 0.0125 and was wrong by 7.8x at 0.0025.
"""
import numpy as np
import pytest

from mpt._tensor._mobius_inner import (_SPECTRAL_IP_COST_C, _SPECTRAL_IP_MAX_POINTS)
from mpt._tensor._mobius_inner import (_spectral_rel_inner_matrix)

PERIOD = 1200.0


def _cols(K, N=1, seed=0):
    rng = np.random.default_rng(seed)
    P = np.sort(rng.uniform(0, PERIOD, (K, N)), axis=0)
    return P, np.ones((K, N))


def _mode_grid(sigma, r, is_per, span=0.0):
    """The branch's own sizing, reproduced so the tests can state which
    side of the gate a cell sits on rather than assuming it."""
    L = PERIOD if is_per else span + 2.0 * (8.6 + 2.0) * sigma
    M = int(np.ceil(8.6 / np.sqrt(2.0) * L / (2.0 * np.pi * sigma))) + 2
    return (2 * M + 1) ** (r - 1)


# ---------------------------------------------------------------------
# The small-sigma/P corner the constant exists for
# ---------------------------------------------------------------------

def test_declines_where_the_branch_is_measurably_slower():
    # sigma/P = 0.0025, r = 3, K = 4, one event: measured 7.8x slower
    # on the branch (56.0 ms against 7.1 ms). The mode grid is ~610k
    # against a gate allowance of C * 16.
    Px, Wx = _cols(4, seed=29)
    assert _mode_grid(3.0, 3, True) > _SPECTRAL_IP_COST_C * 4 ** 2
    assert _spectral_rel_inner_matrix(
        Px, Wx, Px, Wx, 3.0, 3, True, PERIOD) is None


def test_takes_the_branch_where_slots_repay_the_grid():
    # Same sigma, many more values: the grid route's K^2 per pair now
    # dominates and the branch must be taken.
    Px, Wx = _cols(40, seed=31)
    assert _spectral_rel_inner_matrix(
        Px, Wx, Px, Wx, 3.0, 3, True, PERIOD) is not None


def test_more_events_make_the_branch_worth_taking():
    # The gate scales its allowance with the event-pair count, because
    # the grid route pays K^2 per pair while the mode grid is built once.
    Px1, Wx1 = _cols(6, N=1, seed=5)
    Px8, Wx8 = _cols(6, N=8, seed=5)
    sigma = 0.0025 * PERIOD
    declined_at_1 = _spectral_rel_inner_matrix(
        Px1, Wx1, Px1, Wx1, sigma, 3, True, PERIOD) is None
    taken_at_8 = _spectral_rel_inner_matrix(
        Px8, Wx8, Px8, Wx8, sigma, 3, True, PERIOD) is not None
    assert declined_at_1 and taken_at_8


# ---------------------------------------------------------------------
# The gate is a threshold, and behaves like one
# ---------------------------------------------------------------------

@pytest.mark.parametrize("r", [2, 3, 4])
def test_decision_is_monotone_in_event_count(r):
    # Raising the event count only ever makes the branch more
    # attractive: the allowance grows while the mode grid does not.
    sigma = 0.005 * PERIOD
    seen_taken = False
    for N in (1, 2, 4, 8, 16):
        Px, Wx = _cols(8, N=N, seed=3)
        taken = _spectral_rel_inner_matrix(
            Px, Wx, Px, Wx, sigma, r, True, PERIOD) is not None
        if taken:
            seen_taken = True
        elif seen_taken:
            pytest.fail(f"r={r}: branch declined at N={N} after being "
                        f"taken at a smaller N")


@pytest.mark.parametrize("r", [2, 3, 4])
def test_decision_is_monotone_in_slot_count(r):
    sigma = 0.005 * PERIOD
    seen_taken = False
    for K in (4, 6, 8, 12, 20, 30):
        if K < r:
            continue
        Px, Wx = _cols(K, seed=4)
        taken = _spectral_rel_inner_matrix(
            Px, Wx, Px, Wx, sigma, r, True, PERIOD) is not None
        if taken:
            seen_taken = True
        elif seen_taken:
            pytest.fail(f"r={r}: branch declined at K={K} after being "
                        f"taken at a smaller K")


# ---------------------------------------------------------------------
# The memory guard is separate from the cost gate
# ---------------------------------------------------------------------

def test_max_points_declines_independently_of_the_cost_gate():
    # Very small sigma/P at r = 4 puts the mode grid past the memory
    # guard, which must decline whatever the event count allows.
    Px, Wx = _cols(30, N=16, seed=7)
    sigma = 0.0005 * PERIOD
    assert _mode_grid(sigma, 4, True) > _SPECTRAL_IP_MAX_POINTS
    assert _spectral_rel_inner_matrix(
        Px, Wx, Px, Wx, sigma, 4, True, PERIOD) is None
