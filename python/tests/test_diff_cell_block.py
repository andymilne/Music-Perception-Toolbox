"""Regression test — adaptive differential entropy cell-block streaming.

The adaptive ``method='differential'`` evaluator refines the integration
grid by doubling the cell count until convergence. For a narrow-kernel
density spread over a wide span (e.g. a spectral pitch density windowed to
a few events, with partials covering several thousand cents) the grid
reaches hundreds of thousands of cells. Building the per-axis
``(n_tuples, n_cells)`` erf-difference matrix whole then peaks at
``n_tuples * n_cells`` elements -- the regression that OOM-killed the
windowed-chorale differential-entropy sweep.

``_contract_cell_axes`` streams the leading axis in cell blocks of
``_DIFF_CELL_BLOCK`` elements (for ``D <= 2``), which bounds the peak.
Blocking the leading axis leaves each cell's full sum over tuples intact:
every block sums the same tuple values into the same cell, so the streamed
result agrees with the whole-matrix path to within the toolbox's accuracy
floor. The agreement is numerical, not bit-for-bit. ``numpy.einsum``
selects its accumulation order from the operand layout, and a block narrow
enough to yield a contiguous single-column ``(n_tuples, 1)`` matrix
reduces over tuples in a different order from a wide one, which moves the
last bit of each cell mass. The cells sum to one, so the departure sits at
the level of a few units in the last place and is some four orders of
magnitude inside the 1e-12 floor.

This test pins the invariant that matters --- that no tuple contribution
is dropped or double-counted as the block size varies --- by comparing the
default block against a single huge block, and a deliberately tiny block
(many chunks) against the same reference, for both ``D == 1`` and
``D == 2`` absolute-mode densities. A contraction that split a cell's sum
across blocks would miss mass outright and fail by orders of magnitude,
far above the tolerance used here.
"""

import numpy as np
import pytest

import mpt
import mpt.entropy as E
from mpt import build_exp_tens, entropy_exp_tens


def _h(dens, block, ts=5.0):
    """Differential entropy at a given ``_DIFF_CELL_BLOCK`` setting.

    ``ts`` pins a feasible accuracy: for D >= 2 the tightest accuracy
    would need an infeasibly fine grid, and the streaming-vs-whole
    equality this module checks is independent of the absolute accuracy
    (both sides use the same grid), so any feasible ts exercises it. The
    value is kept modest so the grid stays small on any machine."""
    saved = E._DIFF_CELL_BLOCK
    try:
        E._DIFF_CELL_BLOCK = int(block)
        return float(
            entropy_exp_tens(
                dens, method="differential", base=2.0,
                truncation_sigmas=ts, verbose=False,
            )
        )
    finally:
        E._DIFF_CELL_BLOCK = saved


@pytest.fixture
def d1():
    """D == 1 absolute density: a single non-periodic attribute, r = 1."""
    rng = np.random.default_rng(1)
    n = 48
    p = rng.uniform(0.0, 4000.0, (1, n))      # wide span vs sigma -> fine grid
    w = rng.uniform(0.2, 1.0, (1, n))
    return build_exp_tens(
        [p], [w], [20.0], [1], [False], [False], [0.0], verbose=False
    )


@pytest.fixture
def d2():
    """D == 2 absolute density: two non-periodic attributes, r = 1 each."""
    rng = np.random.default_rng(2)
    n = 40
    p0 = rng.uniform(0.0, 600.0, (1, n))
    p1 = rng.uniform(0.0, 300.0, (1, n))
    w0 = rng.uniform(0.2, 1.0, (1, n))
    w1 = np.ones((1, n))
    return build_exp_tens(
        [p0, p1], [w0, w1], [20.0, 15.0], [1, 1],
        [False, False], [False, False], [0.0, 0.0], verbose=False,
    )


# A block large enough that the leading axis is never split (single block).
_WHOLE = int(1e12)

# The toolbox's stated accuracy floor. The streamed and whole-matrix paths
# sum the same values per cell but not necessarily in the same order, so
# they agree to a few units in the last place rather than exactly.
_REL_TOL = 1e-12


def _agree(got, ref):
    """Streamed entropy agrees with the whole-matrix reference."""
    return abs(got - ref) <= _REL_TOL * abs(ref)


def test_d1_streaming_matches_whole(d1):
    ref = _h(d1, _WHOLE)
    assert _agree(_h(d1, E._DIFF_CELL_BLOCK), ref)   # default block
    assert _agree(_h(d1, 64), ref)                   # tiny block -> many chunks


def test_d2_streaming_matches_whole(d2):
    ref = _h(d2, _WHOLE)
    assert _agree(_h(d2, E._DIFF_CELL_BLOCK), ref)   # default block
    assert _agree(_h(d2, 64), ref)                   # tiny block -> many chunks


def test_block_size_is_bounded_but_positive():
    # A pathological block budget must still yield at least one cell per
    # block (the contraction divides the budget by the tuple/trailing
    # count and floors at 1), so streaming never stalls.
    assert E._DIFF_CELL_BLOCK > 0
