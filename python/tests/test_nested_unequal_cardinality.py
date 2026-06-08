"""Nested contraction with unequal inner cardinalities at read-arity r > 1.

These guard two defects fixed together:

1. The tree contraction built a single recipe from the X-side tags and walked
   it for *both* axes of the rectangular leaf kernel ``K`` of shape
   ``(Q, nX, nY)``. When the two densities' nested cardinalities differed
   (e.g. a 4-pitch prototype against an 8-pitch merged window), the Y axis was
   indexed with X-side slots and the inner product collapsed to ~0 for every
   cross-comparison while self-similarity stayed 1. The contraction now threads
   a separate recipe per side; the X axis is indexed by X slots, the Y axis by
   Y slots, and the per-size tuple sourcing handles differing leaf spans. For
   matching structures the two trees coincide and the result is unchanged.

2. The absolute-mode inner product inferred periodicity from
   ``isfinite(period) and period > 0`` rather than the density's ``[per]``
   flag, so an absolute *non-periodic* attribute carrying a finite period
   wrongly wrapped. Periodicity now follows ``[per]``.

Ground truth is an independent brute-force enumeration of the nested tuples
(below) and, for the relative-periodic surrogate, the exact ``bulger``
enumeration in the small-sigma regime where the surrogate is exact.
"""
from itertools import combinations, permutations
import math

import numpy as np
import pytest

from mpt import build_exp_tens, cos_sim_exp_tens


# ----------------------------------------------------------------------
#  Independent brute-force oracle for the absolute nested inner product
# ----------------------------------------------------------------------
def _xtuples(n, r, sym):
    combs = list(combinations(range(n), r))
    return ([p for c in combs for p in permutations(c)] if sym else combs)


def _ytuples(n, r, sym):
    return list(combinations(range(n), r))


def _bf_ip(x_chords, y_chords, r_in, r_out, sym_in, sym_out, sigma):
    """Absolute nested inner product by direct enumeration of the tuples.

    X side enumerates permutations of r-combinations at each symmetric level,
    Y side combinations only (the r!-cancelled perm x comb convention). Outer
    positions multiply; inner pitch positions multiply; everything sums.
    """
    def g(a, b):
        return math.exp(-(a - b) ** 2 / (4.0 * sigma ** 2))

    total = 0.0
    for tx in _xtuples(len(x_chords), r_out, sym_out):
        for ty in _ytuples(len(y_chords), r_out, sym_out):
            prod = 1.0
            for k in range(r_out):
                xch, ych = x_chords[tx[k]], y_chords[ty[k]]
                inner = 0.0
                for a in _xtuples(len(xch), r_in, sym_in):
                    for b in _ytuples(len(ych), r_in, sym_in):
                        term = 1.0
                        for m in range(r_in):
                            term *= g(xch[a[m]], ych[b[m]])
                        inner += term
                prod *= inner
            total += prod
    return total


def _bf_cos(X, Y, r_in, r_out, sym_in, sym_out, sigma):
    xy = _bf_ip(X, Y, r_in, r_out, sym_in, sym_out, sigma)
    xx = _bf_ip(X, X, r_in, r_out, sym_in, sym_out, sigma)
    yy = _bf_ip(Y, Y, r_in, r_out, sym_in, sym_out, sigma)
    return xy / math.sqrt(xx * yy)


# ----------------------------------------------------------------------
#  Density builders
# ----------------------------------------------------------------------
def _nested_density(chords, r_in, r_out, sym_in, sym_out, *, rel_out,
                    is_per, period, sigma):
    counts = [len(c) for c in chords]
    tags = np.concatenate([np.full(counts[k], k) for k in range(len(chords))])
    pitches = np.array([v for c in chords for v in c], float).reshape(-1, 1)
    spec = {"tags": tags, "r": [r_in, r_out],
            "sym": [sym_in, sym_out], "rel": [0, rel_out]}
    return build_exp_tens([pitches], None, specs=[spec], sigma=[sigma],
                          is_per=[is_per], period=[period], verbose=False)


def _cos(chords_x, chords_y, *, r_in, r_out, sym_in, sym_out, rel_out,
         is_per, period, sigma, method="contract"):
    dx = _nested_density(chords_x, r_in, r_out, sym_in, sym_out,
                         rel_out=rel_out, is_per=is_per, period=period,
                         sigma=sigma)
    dy = _nested_density(chords_y, r_in, r_out, sym_in, sym_out,
                         rel_out=rel_out, is_per=is_per, period=period,
                         sigma=sigma)
    return cos_sim_exp_tens(dx, dy, method=method, verbose=False)


# ----------------------------------------------------------------------
#  1. Absolute mode: contraction == independent brute force
# ----------------------------------------------------------------------
_ABS_CASES = [
    # (label, X, Y, r_in, r_out, sym_in, sym_out)
    ("equal 2v2 ri2", [[1., 3.], [7., 9.]],
     [[1.2, 2.8], [6.7, 9.3]], 2, 2, True, False),
    ("equal 3v3 ri2", [[1., 3., 5.], [7., 9., 11.]],
     [[1.2, 2.8, 5.1], [6.7, 9.3, 10.8]], 2, 2, True, False),
    ("uneq 2v3 ri2", [[1., 3.], [7., 9.]],
     [[1.2, 2.8, 5.1], [6.7, 9.3, 10.8]], 2, 2, True, False),
    ("uneq 2v4 ri2", [[1., 3.], [7., 9.]],
     [[1., 3., 5., 2.], [7., 9., 11., 8.]], 2, 2, True, False),
    ("uneq 3v4 ri3", [[1., 3., 5.], [7., 9., 11.]],
     [[1., 3., 5., 2.], [7., 9., 11., 8.]], 3, 2, True, False),
    ("uneq 2v3 outer-sym", [[1., 3.], [7., 9.]],
     [[1.2, 2.8, 5.1], [6.7, 9.3, 10.8]], 2, 2, True, True),
    ("uneq 2v3 inner-ordered", [[1., 3.], [7., 9.]],
     [[1.2, 2.8, 5.1], [6.7, 9.3, 10.8]], 2, 2, False, False),
]


@pytest.mark.parametrize("label,X,Y,r_in,r_out,sym_in,sym_out", _ABS_CASES,
                         ids=[c[0] for c in _ABS_CASES])
def test_absolute_contraction_matches_bruteforce(
        label, X, Y, r_in, r_out, sym_in, sym_out):
    sigma = 0.5
    got = _cos(X, Y, r_in=r_in, r_out=r_out, sym_in=sym_in, sym_out=sym_out,
               rel_out=0, is_per=False, period=1e9, sigma=sigma)
    want = _bf_cos(X, Y, r_in, r_out, sym_in, sym_out, sigma)
    assert got == pytest.approx(want, abs=1e-10)


# ----------------------------------------------------------------------
#  2. The reported regression: r>1, unequal cardinality, relative-periodic
# ----------------------------------------------------------------------
def test_unequal_cardinality_r2_relperiodic_nonzero_symmetric():
    """4-pitch prototype vs 8-pitch (doubled) window at inner r=2 was exactly
    0 for every cross-comparison; it must now be a sensible nonzero value,
    self-similarity 1, and symmetric."""
    P, sigma = 12.0, 0.15
    proto = [[0., 4., 7., 0.], [7., 11., 2., 7.], [0., 4., 7., 0.]]
    window = [c + c for c in proto]           # each chord doubled -> 8 pitches
    kw = dict(r_in=2, r_out=3, sym_in=True, sym_out=False, rel_out=1,
              is_per=True, period=P, sigma=sigma)
    xy = _cos(proto, window, **kw)
    yx = _cos(window, proto, **kw)
    assert xy > 1e-3                          # was ~0
    assert xy == pytest.approx(yx, abs=1e-12)  # symmetric
    assert _cos(proto, proto, **kw) == pytest.approx(1.0, abs=1e-12)
    assert _cos(window, window, **kw) == pytest.approx(1.0, abs=1e-12)


def test_equal_cardinality_r2_unchanged():
    """Matching structures must be untouched by the two-recipe threading."""
    P, sigma = 12.0, 0.15
    a = [[0., 4., 7., 0.], [7., 11., 2., 7.], [0., 4., 7., 0.]]
    b = [[0., 3., 7., 0.], [7., 11., 2., 7.], [0., 3., 7., 0.]]
    kw = dict(r_in=2, r_out=3, sym_in=True, sym_out=False, rel_out=1,
              is_per=True, period=P, sigma=sigma)
    assert _cos(a, a, **kw) == pytest.approx(1.0, abs=1e-12)
    assert _cos(a, b, **kw) == pytest.approx(_cos(b, a, **kw), abs=1e-12)


# ----------------------------------------------------------------------
#  3. Relative-periodic: contraction == exact enumeration (surrogate regime)
# ----------------------------------------------------------------------
_REL_CASES = [
    ("uneq 2v3", [[0., 4.], [7., 11.]],
     [[0., 4., 7.], [2., 5., 9.]], 2, 2),
    ("uneq 2v4", [[0., 4.], [7., 11.]],
     [[0., 4., 0., 4.], [7., 11., 7., 11.]], 2, 2),
    ("uneq 3v4", [[0., 4., 7.], [2., 5., 9.], [4., 7., 11.]],
     [[0., 4., 7., 0.], [2., 5., 9., 2.], [4., 7., 11., 4.]], 2, 3),
]


@pytest.mark.parametrize("label,X,Y,r_in,r_out", _REL_CASES,
                         ids=[c[0] for c in _REL_CASES])
def test_relperiodic_unequal_contract_matches_enumeration(
        label, X, Y, r_in, r_out):
    # Small sigma/period keeps the transposition-average surrogate exact, so
    # it agrees with the bulger enumeration to numerical precision.
    kw = dict(r_in=r_in, r_out=r_out, sym_in=True, sym_out=False, rel_out=1,
              is_per=True, period=12.0, sigma=0.2)
    contract = _cos(X, Y, method="contract", **kw)
    bulger = _cos(X, Y, method="bulger", **kw)
    assert contract == pytest.approx(bulger, abs=1e-9)


# ----------------------------------------------------------------------
#  4. Absolute non-periodic with a finite stored period must not wrap
# ----------------------------------------------------------------------
def test_absolute_nonperiodic_finite_period_does_not_wrap():
    X = [[1., 3.], [7., 9.]]
    Y = [[1.2, 2.8, 5.1], [6.7, 9.3, 10.8]]
    sigma = 0.5
    want = _bf_cos(X, Y, 2, 2, True, False, sigma)   # genuinely non-periodic
    # A small finite period that *would* alias if periodicity were (wrongly)
    # inferred from period rather than the [per] flag.
    got_small = _cos(X, Y, r_in=2, r_out=2, sym_in=True, sym_out=False,
                     rel_out=0, is_per=False, period=2.0, sigma=sigma)
    got_huge = _cos(X, Y, r_in=2, r_out=2, sym_in=True, sym_out=False,
                    rel_out=0, is_per=False, period=1e9, sigma=sigma)
    assert got_small == pytest.approx(want, abs=1e-10)
    assert got_small == pytest.approx(got_huge, abs=1e-10)
