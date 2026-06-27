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
Blocking the leading axis leaves each cell's full sum over tuples intact,
so the streamed result must be bit-identical to the whole-matrix path
regardless of block size. This test pins that invariant by comparing the
default block against a single huge block, and a deliberately tiny block
(many chunks) against the same reference, for both ``D == 1`` and
``D == 2`` absolute-mode densities.

Any change to the cell-mass contraction must keep these byte-exact.
"""

import numpy as np
import pytest

import mpt
import mpt.entropy as E
from mpt import build_exp_tens, entropy_exp_tens


def _h(dens, block):
    """Differential entropy at a given ``_DIFF_CELL_BLOCK`` setting."""
    saved = E._DIFF_CELL_BLOCK
    try:
        E._DIFF_CELL_BLOCK = int(block)
        return float(
            entropy_exp_tens(dens, method="differential", base=2.0, verbose=False)
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


def test_d1_streaming_matches_whole(d1):
    ref = _h(d1, _WHOLE)
    assert _h(d1, E._DIFF_CELL_BLOCK) == ref      # default block
    assert _h(d1, 64) == ref                      # tiny block -> many chunks


def test_d2_streaming_matches_whole(d2):
    ref = _h(d2, _WHOLE)
    assert _h(d2, E._DIFF_CELL_BLOCK) == ref      # default block
    assert _h(d2, 64) == ref                      # tiny block -> many chunks


def test_block_size_is_bounded_but_positive():
    # A pathological block budget must still yield at least one cell per
    # block (the contraction divides the budget by the tuple/trailing
    # count and floors at 1), so streaming never stalls.
    assert E._DIFF_CELL_BLOCK > 0
