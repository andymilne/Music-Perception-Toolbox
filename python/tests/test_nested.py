"""Tests for nested binding (representation B) in the MAET build core.

Covers the toolbox specification §6/§7.1/§8/§10 for the two-level nested
attribute. The nested spec uses per-level vectors (innermost-outward):
``{tags, r, sym, rel}``, where ``rel`` is the co-transposition-unit
selector (a per-level vector or the depth-proof strings ``'innermost'`` /
``'outermost'``; a bare scalar/bool is rejected for a nested attribute).

* Outer ``r = K`` reproduces the old separate-attribute binding (tensor
  join): a single nested attribute read at the whole-tuple level
  evaluates identically to the equivalent flat tensor-joined density.
* The within-event/across-event partial symmetry with ``sym`` outer = 0:
  inner slots orbit within each source event, but the bound events keep
  sequence order (no cross-tag interleaving).
* Pooled within-source reading (outer ``r < K``) sums per-event
  sub-tuples into one shared lower-dimensional space.
* The §6.3 inner/outer ``[rel]`` projection-rank dims (4/2/3/2 for two
  dyads). The **outer / whole-tuple** unit is implemented here: it is the
  flat ``is_rel`` reduction on the whole tuple (dim 4 -> 3) and is exactly
  global-transposition-invariant. (The inner unit is a later step.)
* ``[rel]`` validation: scalar/bool rejected for nested; a user
  ``is_rel_vec`` entry on a nested attribute is rejected (set via spec);
  subsumption warns on multiple set levels; insufficient source events
  errors.
"""

import numpy as np
import pytest
from numpy.linalg import matrix_rank

from mpt import build_exp_tens, eval_exp_tens, cos_sim_exp_tens


def _ev(d, x):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    return float(np.ravel(eval_exp_tens(d, x, verbose=False))[0])


def _nest(p, spec):
    return build_exp_tens([np.asarray(p, dtype=float).reshape(-1, 1)], None,
                          [50.0], [1], [False], [False], [0.0],
                          nested=[spec], verbose=False)


def test_outer_rK_reproduces_old_binding():
    """Outer r=K nested attribute == old separate-attribute tensor join."""
    d_old = build_exp_tens(
        [np.array([[0.0]]), np.array([[7.0]])], None, [50.0, 50.0],
        [1, 1], [False, False], [False, False], [0.0, 0.0], verbose=False,
    )
    d_nest = _nest([0.0, 7.0],
                   dict(tags=[0, 1], r=[1, 2], sym=[True, False]))
    assert d_old.dim == d_nest.dim == 2
    for x in ([0.0, 7.0], [0.0, 0.0], [3.0, 7.0], [-5.0, 12.0]):
        assert _ev(d_old, x) == pytest.approx(_ev(d_nest, x), abs=1e-12)


def test_inner_orbit_outer_order_preserved():
    """sym inner=1, outer=0: inner orbits, no cross-tag interleaving."""
    d = _nest([0.0, 4.0, 7.0, 11.0],
              dict(tags=[0, 0, 1, 1], r=[2, 2], sym=[True, False]))
    assert d.dim == 4
    centres = {tuple(c) for c in np.asarray(d.centres[0]).T.tolist()}
    expected = {(0, 4, 7, 11), (0, 4, 11, 7), (4, 0, 7, 11), (4, 0, 11, 7)}
    assert centres == expected
    for c in centres:
        assert {c[0], c[1]} == {0, 4} and {c[2], c[3]} == {7, 11}


def test_pooled_outer_r_less_than_K():
    """Outer r<K pools within-source sub-tuples into one shared space."""
    d = _nest([0.0, 4.0, 7.0, 11.0],
              dict(tags=[0, 0, 1, 1], r=[2, 1], sym=[False, False]))
    assert d.dim == 2
    centres = {tuple(c) for c in np.asarray(d.centres[0]).T.tolist()}
    assert centres == {(0, 4), (7, 11)}


def test_outer_rel_dim_and_transposition_invariance():
    """Outer/whole [rel] unit: dim 4->3 and exact global-transposition invariance."""
    spec = dict(tags=[0, 0, 1, 1], r=[2, 2], sym=[True, False], rel="outermost")
    d = _nest([0.0, 4.0, 7.0, 11.0], spec)
    d_shift = _nest([5.0, 9.0, 12.0, 16.0], dict(spec))
    assert d.dim == 3
    assert np.allclose(np.sort(np.asarray(d.centres[0]), axis=1),
                       np.sort(np.asarray(d_shift.centres[0]), axis=1))
    assert float(cos_sim_exp_tens(d, d_shift, verbose=False)) == pytest.approx(1.0, abs=1e-9)


def test_absolute_not_transposition_invariant():
    """Absolute nested density is not global-transposition invariant."""
    spec = dict(tags=[0, 0, 1, 1], r=[2, 2], sym=[True, False])  # rel absent
    d = _nest([0.0, 4.0, 7.0, 11.0], spec)
    d_shift = _nest([5.0, 9.0, 12.0, 16.0], dict(spec))
    assert d.dim == 4
    assert float(cos_sim_exp_tens(d, d_shift, verbose=False)) < 0.999


def test_rel_outermost_vector_matches_string():
    """rel=[0,1] (outermost level set) equals rel='outermost'."""
    base = dict(tags=[0, 0, 1, 1], r=[2, 2], sym=[True, False])
    d_str = _nest([0.0, 4.0, 7.0, 11.0], dict(base, rel="outermost"))
    d_vec = _nest([0.0, 4.0, 7.0, 11.0], dict(base, rel=[0, 1]))
    assert d_str.dim == d_vec.dim == 3
    assert np.allclose(np.sort(np.asarray(d_str.centres[0]), axis=1),
                       np.sort(np.asarray(d_vec.centres[0]), axis=1))


def test_rel_subsumption_warns_on_multiple():
    """rel=[1,1] warns (finer subsumes coarser) and resolves to innermost."""
    spec = dict(tags=[0, 0, 1, 1], r=[2, 2], sym=[True, False], rel=[1, 1])
    with pytest.warns(UserWarning):
        with pytest.raises(NotImplementedError):
            _nest([0.0, 4.0, 7.0, 11.0], spec)


def test_inner_outer_rel_projection_dims():
    """§6.3 prediction: inner/outer [rel] give dims 4/2/3/2 (two dyads)."""
    D = 4
    perA, perB, glob = [1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]

    def comp(removed):
        if not removed:
            return D
        return D - matrix_rank(np.array(removed, dtype=float))

    assert comp([]) == 4
    assert comp([perA, perB]) == 2
    assert comp([glob]) == 3
    assert comp([perA, perB, glob]) == 2


def test_scalar_rel_rejected_for_nested():
    """A bare scalar/bool [rel] is rejected for a nested attribute."""
    with pytest.raises(ValueError):
        _nest([0.0, 7.0], dict(tags=[0, 1], r=[1, 2], sym=[True, False], rel=True))


def test_inner_rel_not_yet_supported():
    """The inner co-transposition unit raises until its block path lands."""
    with pytest.raises(NotImplementedError):
        _nest([0.0, 4.0, 7.0, 11.0],
              dict(tags=[0, 0, 1, 1], r=[2, 2], sym=[True, False], rel="innermost"))


def test_user_is_rel_on_nested_rejected():
    """Setting is_rel_vec on a nested attribute errors (use the spec's rel)."""
    spec = dict(tags=[0, 1], r=[1, 2], sym=[True, False])
    with pytest.raises(ValueError):
        build_exp_tens([np.array([[0.0], [7.0]])], None, [50.0], [1],
                       [True], [False], [0.0], nested=[spec], verbose=False)


def test_nested_precondition_insufficient_tags():
    """r_outer exceeding the available distinct source events errors."""
    with pytest.raises(ValueError):
        _nest([0.0, 4.0],
              dict(tags=[0, 0], r=[1, 2], sym=[True, False]))


def test_explicit_all_none_nested_equals_default():
    """nested=[None]*A is identical to nested=None (flat path untouched)."""
    pAttr = [np.array([[0.0, 4.0], [7.0, 11.0]])]
    args = (None, [50.0], [2], [False], [False], [0.0])
    d_default = build_exp_tens(pAttr, *args, verbose=False)
    d_none = build_exp_tens(pAttr, *args, nested=[None], verbose=False)
    assert d_default.dim == d_none.dim
    for x in ([0.0, 7.0], [4.0, 11.0], [2.0, 9.0]):
        assert _ev(d_default, x) == pytest.approx(_ev(d_none, x), abs=1e-14)


def test_outer_rebuild_roundtrip_no_false_guard():
    """A reconstruction forwards the normalised spec (carrying 'proj') with
    the derived is_rel=True; this must round-trip without tripping the
    user-is_rel guard. Regression for the rebuild path."""
    fresh = _nest([0.0, 4.0, 7.0, 11.0],
                  dict(tags=[0, 0, 1, 1], r=[2, 2], sym=[True, False],
                       rel="outermost"))
    reb_spec = dict(tags=np.array([0, 0, 1, 1]), r=np.array([2, 2]),
                    sym=np.array([True, False]), rel="outermost",
                    proj="outer", rel_unit=1)
    d = build_exp_tens([np.array([[0.0], [4.0], [7.0], [11.0]])], None,
                       [50.0], [1], [True], [False], [0.0],
                       nested=[reb_spec], verbose=False)
    assert d.dim == fresh.dim == 3
    for x in ([1.0, 3.0, 5.0], [0.0, 0.0, 0.0], [-2.0, 4.0, 1.0]):
        assert _ev(d, x) == pytest.approx(_ev(fresh, x), abs=1e-12)
