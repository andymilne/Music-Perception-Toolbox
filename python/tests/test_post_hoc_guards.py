"""The ``post_hoc_guards`` default and what switching it off does.

Two checks in the toolbox inspect a route's output after computing it
and may then recompute by another route: the nested accuracy guard in
``_combine_pair`` and the corruption check in the flat cosine path. With
either active the measured cost of the Möbius route is not the cost of
choosing it, because a diverting check pays for both routes. The default
switches them off so the routes can be timed as the alternatives they
are.

Mirror of MATLAB ``tests/test_post_hoc_guards.m``.
"""
import numpy as np
import pytest

import mpt
from mpt._defaults import truncation_floor
from mpt._tensor._nested_contraction import (
    _combine_pair, _tuple_sides, _combine_chunked, _combine_orbit,
    _ORBIT_ENUM_MAX_ELEMS, orbit_guard_scope,
)


def test_default_is_on_and_round_trips():
    assert mpt.get_defaults()["post_hoc_guards"] is True
    try:
        mpt.set_default(post_hoc_guards=False)
        assert mpt.get_defaults()["post_hoc_guards"] is False
    finally:
        mpt.reset_defaults()
    assert mpt.get_defaults()["post_hoc_guards"] is True


def test_rejects_non_boolean():
    with pytest.raises(ValueError, match="post_hoc_guards"):
        mpt.set_default(post_hoc_guards="yes")
    mpt.reset_defaults()


def _block(r, K, seed=0):
    return np.random.default_rng(seed).uniform(0.0, 1.0, (8, K, K))


def test_guard_on_diverts_to_enumeration_at_r6():
    """At r = 6 the bound exceeds the floor, so the guard should divert:
    the returned value is enumeration's, not the Möbius route's."""
    r, K = 6, 6
    M = _block(r, K)
    xt, yt = _tuple_sides(K, r, True)
    enum = np.asarray(_combine_chunked(M, xt, yt, _ORBIT_ENUM_MAX_ELEMS))
    orbit, bound = _combine_orbit(M, r, return_bound=True)
    assert float(np.max(bound)) > truncation_floor(6), (
        "premise: the guard must fire at this shape"
    )
    assert not np.array_equal(np.asarray(orbit), enum), (
        "premise: the two routes must differ for this test to bite"
    )
    try:
        mpt.set_default(post_hoc_guards=True)
        with orbit_guard_scope(6.0):
            got = np.asarray(_combine_pair(M, r, True, True))
    finally:
        mpt.reset_defaults()
    np.testing.assert_array_equal(got, enum)


def test_guard_off_keeps_the_mobius_result_at_r6():
    """With the guard off the same block returns the Möbius value."""
    r, K = 6, 6
    M = _block(r, K)
    orbit = np.asarray(_combine_orbit(M, r, return_bound=False))
    try:
        mpt.set_default(post_hoc_guards=False)
        with orbit_guard_scope(6.0):
            got = np.asarray(_combine_pair(M, r, True, True))
    finally:
        mpt.reset_defaults()
    np.testing.assert_array_equal(got, orbit)


def test_guard_off_still_within_the_stated_accuracy_here():
    """Switching the guard off is not a licence to exceed the accuracy
    truncationSigmas states: at this shape the bound is loose and the
    Möbius route's actual error sits well inside the floor."""
    r, K = 6, 6
    M = _block(r, K)
    xt, yt = _tuple_sides(K, r, True)
    enum = np.asarray(_combine_chunked(M, xt, yt, _ORBIT_ENUM_MAX_ELEMS))
    orbit = np.asarray(_combine_orbit(M, r, return_bound=False))
    assert np.max(np.abs(orbit - enum)) < truncation_floor(6)
