"""Tests for nested binding (representation B) in the MAET build core.

Covers the predictions of the toolbox specification §6/§7.1/§10 for the
two-level nested attribute, ``[rel] = absolute`` only (the ``[rel]``
co-transposition-unit selector is a later step):

* Outer ``r = K`` reproduces the old separate-attribute binding (the
  tensor join): a single nested attribute read at the whole-tuple level
  evaluates identically to the equivalent flat tensor-joined density
  (§6.5, §10).
* The within-event/across-event partial symmetry ``S_{r_inner} x ...``
  with ``sym_outer = 0``: inner slots orbit within each source event,
  but the bound events keep sequence order (no cross-tag interleaving).
* Pooled within-source reading (outer ``r < K``) sums the per-event
  sub-tuples into one shared lower-dimensional space -- a genuinely
  different object from old binding (§6.5).
* The §6.3 inner/outer ``[rel]`` projection-rank dimensions (4/2/3/2 for
  two dyads) -- recorded here as the geometric target the ``[rel]``
  selector step must reproduce.
* The flat path is untouched: ``nested = None`` and an explicit all-None
  list behave identically, and the existing suite (flat) is unchanged.
"""

import numpy as np
import pytest
from numpy.linalg import matrix_rank

from mpt import build_exp_tens, eval_exp_tens


def _ev(d, x):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    return float(np.ravel(eval_exp_tens(d, x, verbose=False))[0])


def test_outer_rK_reproduces_old_binding():
    """Outer r=K nested attribute == old separate-attribute tensor join."""
    # Old binding: two single-note events -> two attributes, each one slot.
    d_old = build_exp_tens(
        [np.array([[0.0]]), np.array([[7.0]])], None, [50.0, 50.0],
        [1, 1], [False, False], [False, False], [0.0, 0.0], verbose=False,
    )
    # Nested: one attribute, K_total=2 slots [0, 7], one per source event,
    # inner r=1, outer r=K=2.
    nspec = dict(tags=[0, 1], r_inner=1, r_outer=2,
                 sym_inner=True, sym_outer=False)
    d_nest = build_exp_tens(
        [np.array([[0.0], [7.0]])], None, [50.0], [1],
        [False], [False], [0.0], nested=[nspec], verbose=False,
    )
    assert d_old.dim == d_nest.dim == 2
    for x in ([0.0, 7.0], [0.0, 0.0], [3.0, 7.0], [-5.0, 12.0]):
        assert _ev(d_old, x) == pytest.approx(_ev(d_nest, x), abs=1e-12)


def test_inner_orbit_outer_order_preserved():
    """sym_inner=1, sym_outer=0: inner orbits, no cross-tag interleaving."""
    nspec = dict(tags=[0, 0, 1, 1], r_inner=2, r_outer=2,
                 sym_inner=True, sym_outer=False)
    d = build_exp_tens(
        [np.array([[0.0], [4.0], [7.0], [11.0]])], None, [50.0], [1],
        [False], [False], [0.0], nested=[nspec], verbose=False,
    )
    assert d.dim == 4
    centres = {tuple(c) for c in np.asarray(d.centres[0]).T.tolist()}
    # Inner S_2 orbit within each dyad (0<->4, 7<->11); dyad-A slots always
    # precede dyad-B slots (outer order kept). (2!)^2 = 4 centres.
    expected = {(0, 4, 7, 11), (0, 4, 11, 7), (4, 0, 7, 11), (4, 0, 11, 7)}
    assert centres == expected
    # No centre interleaves the two source events (e.g. (0,7,4,11)).
    for c in centres:
        assert {c[0], c[1]} == {0, 4} and {c[2], c[3]} == {7, 11}


def test_pooled_outer_r_less_than_K():
    """Outer r<K pools within-source sub-tuples into one shared space."""
    nspec = dict(tags=[0, 0, 1, 1], r_inner=2, r_outer=1,
                 sym_inner=False, sym_outer=False)
    d = build_exp_tens(
        [np.array([[0.0], [4.0], [7.0], [11.0]])], None, [50.0], [1],
        [False], [False], [0.0], nested=[nspec], verbose=False,
    )
    assert d.dim == 2
    centres = {tuple(c) for c in np.asarray(d.centres[0]).T.tolist()}
    assert centres == {(0, 4), (7, 11)}


def test_inner_outer_rel_projection_dims():
    """§6.3 prediction: inner/outer [rel] give dims 4/2/3/2 (two dyads).

    Geometric target for the [rel] selector step; verified here by
    projection rank, independent of the build path.
    """
    D = 4
    perA, perB, glob = [1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]

    def comp(removed):
        if not removed:
            return D
        return D - matrix_rank(np.array(removed, dtype=float))

    assert comp([]) == 4                       # (inner 0, outer 0) absolute
    assert comp([perA, perB]) == 2             # (inner 1, outer 0)
    assert comp([glob]) == 3                   # (inner 0, outer 1)
    assert comp([perA, perB, glob]) == 2       # (inner 1, outer 1)


def test_nested_rel_not_yet_supported():
    """is_rel on a nested attribute raises until the [rel] selector lands."""
    nspec = dict(tags=[0, 1], r_inner=1, r_outer=2,
                 sym_inner=True, sym_outer=False)
    with pytest.raises(NotImplementedError):
        build_exp_tens(
            [np.array([[0.0], [7.0]])], None, [50.0], [1],
            [True], [False], [0.0], nested=[nspec], verbose=False,
        )


def test_nested_precondition_insufficient_tags():
    """r_outer exceeding the available distinct source events errors."""
    nspec = dict(tags=[0, 0], r_inner=1, r_outer=2,
                 sym_inner=True, sym_outer=False)  # only one tag present
    with pytest.raises(ValueError):
        build_exp_tens(
            [np.array([[0.0], [4.0]])], None, [50.0], [1],
            [False], [False], [0.0], nested=[nspec], verbose=False,
        )


def test_explicit_all_none_nested_equals_default():
    """nested=[None]*A is identical to nested=None (flat path untouched)."""
    pAttr = [np.array([[0.0, 4.0], [7.0, 11.0]])]
    args = (None, [50.0], [2], [False], [False], [0.0])
    d_default = build_exp_tens(pAttr, *args, verbose=False)
    d_none = build_exp_tens(pAttr, *args, nested=[None], verbose=False)
    assert d_default.dim == d_none.dim
    for x in ([0.0, 7.0], [4.0, 11.0], [2.0, 9.0]):
        assert _ev(d_default, x) == pytest.approx(_ev(d_none, x), abs=1e-14)
