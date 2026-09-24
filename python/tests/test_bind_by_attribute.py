"""Run-length (bind-by-attribute) binding, and the ragged-orbit regression.

bind_events(..., group_by=a) gathers consecutive events sharing a constant
value on attribute ``a`` into one super-event; group sizes vary, so the outer
level is ragged (NaN-padded to the maximum group size, padded positions at zero
weight). The ragged-orbit test pins that the orbit (Möbius) reduction and the
permutation/combination enumeration agree on such a density.
"""

import numpy as np
import pytest

import mpt
import mpt._tensor._nested_contraction as nc
from mpt import (bind_events, build_maet, flat_specs, sim_maet,
                 unpack_pre_maet)


def _density(seed, sizes, r_outer=None, exch_outer=True):
    r = np.random.default_rng(seed)
    onset = np.concatenate([[k] * s for k, s in enumerate(sizes)])
    onset = onset.reshape(1, -1).astype(float)
    pitch = r.uniform(55, 79, size=onset.shape[1]).reshape(1, -1)
    pb, wb, sp = unpack_pre_maet(bind_events([pitch, onset], None, None, group_by=1,
                             r_outer=r_outer, exch_outer=exch_outer))
    return pb, wb, sp


def test_run_length_structure():
    pitch = np.array([[60, 64, 67, 62, 65, 60, 59, 62, 67, 71]], float)
    onset = np.array([[0, 0, 0, 1, 1, 2, 3, 3, 3, 3]], float)  # 3,2,1,4
    pb, wb, sp = unpack_pre_maet(bind_events([pitch, onset], None, None, group_by=1))
    assert pb[0].shape == (4, 4)               # L_max=4, n_prime=4
    assert sp[0]["r"] == [1, 1]                # inner r preserved, outer = min = 1
    assert np.array_equal(sp[0]["tags"], np.repeat(np.arange(4), 1))
    # NaN padding per group: 4-3, 4-2, 4-1, 4-4
    assert list(np.isnan(pb[0]).sum(0)) == [1, 2, 3, 0]
    # padded positions carry zero weight
    assert np.all(wb[0][np.isnan(pb[0])] == 0.0)


def test_r_outer_defaults_to_min_group_size():
    _, _, sp = _density(0, [6, 4, 5], r_outer=None)
    assert sp[0]["r"][1] == 4                   # min group size


def test_self_similarity_is_one():
    pb, wb, sp = _density(1, [3, 2, 1, 4])      # r_outer = min = 1
    d = build_maet(pb, wb, sigma=[40.0, 0.01], is_per=[False, False],
                       period=[0.0, 0.0], specs=sp, verbose=False)
    assert abs(sim_maet(d, d, method="auto", verbose=False) - 1.0) < 1e-12


def test_ragged_orbit_matches_enumeration():
    """The orbit reduction and enumeration agree on a ragged outer level."""
    def build(seed):
        pb, wb, sp = _density(seed, [4, 3, 5], r_outer=3, exch_outer=True)
        return build_maet(pb, wb, sigma=[30.0, 0.02], is_per=[False, False],
                              period=[0.0, 0.0], specs=sp, verbose=False)
    X, Y = build(10), build(20)
    orig = nc._orbit_eligible
    try:
        nc._orbit_eligible = lambda *a, **k: True       # force orbit
        v_orbit = sim_maet(X, Y, method="contract", verbose=False)
        nc._orbit_eligible = lambda *a, **k: False      # force enumeration
        v_enum = sim_maet(X, Y, method="contract", verbose=False)
    finally:
        nc._orbit_eligible = orig
    assert abs(v_orbit - v_enum) < 1e-9


def test_high_arity_ragged_runs_via_orbit():
    """At an outer tuple size where enumeration is infeasible, auto still completes
    (the orbit path carries the ragged density)."""
    pb, wb, sp = _density(3, [7, 6, 8], r_outer=6, exch_outer=True)
    d = build_maet(pb, wb, sigma=[30.0, 0.01], is_per=[False, False],
                       period=[0.0, 0.0], specs=sp, verbose=False)
    assert abs(sim_maet(d, d, method="auto", verbose=False) - 1.0) < 1e-9


def test_consecutive_runs_not_global():
    # non-adjacent equal values do not merge
    onset = np.array([[0, 0, 1, 0]], float)
    pitch = np.array([[60, 61, 62, 63]], float)
    pb, _, sp = unpack_pre_maet(bind_events([pitch, onset], None, None, group_by=1))
    assert pb[0].shape[1] == 3                  # [0,0], [1], [0]


def test_validation():
    pitch = np.array([[60, 61, 62]], float)
    onset = np.array([[0, 0, 1]], float)
    with pytest.raises(ValueError):             # bind_orders + group_by exclusive
        bind_events([pitch, onset], None, 2, group_by=1)
    with pytest.raises(ValueError):             # group_by must be K=1
        chord = np.array([[60, 62], [64, 66], [67, 69]]).T.astype(float)
        bind_events([chord, onset], None, None, group_by=0)
    with pytest.raises(NotImplementedError):    # circular not yet supported
        bind_events([pitch, onset], None, None, group_by=1, circular=True)


def test_r_outer_exceeding_smallest_group_errors():
    """A group smaller than r_outer admits no r_outer-tuple -> build errors."""
    pb, wb, sp = _density(5, [4, 3, 5], r_outer=4)   # size-3 group < 4
    with pytest.raises(Exception):
        build_maet(pb, wb, sigma=[30.0, 0.02], is_per=[False, False],
                       period=[0.0, 0.0], specs=sp, verbose=False)


def test_group_by_name_and_carried_spec_fields():
    """group_by takes a name, and the kernel geometry crosses to the spec.

    Run-length binding regroups values without touching them, so an
    attribute's sigma (and periodicity) belong to the nested spec just as
    they did to the flat one; naming the grouping attribute is the same
    selection bind_attributes and select_pre_maet accept.
    """
    pitch = np.array([[60, 64, 67, 62, 65]], float)
    chord = np.array([[0, 0, 0, 1, 1]], float)
    specs = flat_specs([pitch, chord], sigma=[0.5, 0.25], is_per=[True, False],
                       period=[12.0, 0.0], name=['pitch', 'chord'])
    by_name = unpack_pre_maet(bind_events([pitch, chord], None, None,
                                          group_by='chord', specs=specs,
                                          r_outer=2))
    by_index = unpack_pre_maet(bind_events([pitch, chord], None, None,
                                           group_by=1, specs=specs,
                                           r_outer=2))
    assert np.allclose(by_name[0][0], by_index[0][0], equal_nan=True)
    assert by_name[2][0]['sigma'] == 0.5
    assert by_name[2][0]['is_per'] is True
    assert by_name[2][0]['period'] == 12.0
    assert by_name[2][1]['sigma'] == 0.25
    # The spec carries its own sigma, so the density builds without one.
    build_maet(by_name[0], by_name[1], specs=by_name[2], verbose=False)


def test_group_by_weights_have_one_row_per_value():
    """An inner attribute of K > 1 gets its weight on every value row.

    A weight given per event applies to each of that event's values, so the
    bound weight matrix has L_max * K_a rows, as the bound value matrix does.
    """
    coords = np.array([[60, 64, 67, 62], [0.0, 1.0, 2.0, 3.0]], float)  # K = 2
    chord = np.array([[0, 0, 1, 1]], float)
    w_event = np.array([[1.0, 0.5, 2.0, 0.25]])
    pb, wb, sp = unpack_pre_maet(
        bind_events([coords, chord], [w_event, np.ones_like(chord)], None,
                    group_by=1, r_outer=2))
    assert wb[0].shape == pb[0].shape           # (L_max * K_a, n')
    # Row block ell = 0 holds the first position of each group, both of its
    # values carrying that event's weight.
    assert np.allclose(wb[0][:2, :], np.array([[1.0, 2.0], [1.0, 2.0]]))
    assert np.allclose(wb[0][2:4, :], np.array([[0.5, 0.25], [0.5, 0.25]]))
    build_maet(pb, wb, sigma=[1.0, 1.0], is_per=[False, False],
               period=[0.0, 0.0], specs=sp, verbose=False)
