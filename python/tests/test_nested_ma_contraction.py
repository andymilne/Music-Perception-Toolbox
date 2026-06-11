"""Multi-attribute nested contraction: a nested attribute tensored with one
or more plain attributes.

Previously the fast tree-contraction handled only a single-attribute density;
tensoring any further attribute (e.g. an inversion flag) made it decline, and
the dispatcher fell back to the joint-tuple enumeration --- which at inner
``r = 2`` with wide chords is the combinatorial blow-up the contraction was
built to avoid (minutes per cosine). The MA path now routes each nested factor
through the contraction and each plain factor through the per-attribute MA
matrix, combining them per event-pair (JMM Eq 3.4): ``<X,Y> = Σ_{i,j} Π_a
I_a(i,j)``. Per-attribute prefactors are constant and cancel in the cosine, so
mixing the two matrix conventions is exact.

These check the MA contraction against the exact ``bulger`` enumeration across
single- and multi-event densities, matching and mismatching flags, equal and
unequal nested cardinalities, and both inner arities.
"""
import numpy as np
import pytest

from mpt import build_exp_tens, cos_sim_exp_tens

SIG = 0.15
SF = 0.1          # flag kernel width (small -> strict categorical match)
P = 12.0


def _dens(events, flags, r_in, rel_out=1):
    """Two-attribute density: a nested harmonic attribute (outer relative,
    periodic) tensored with a 1-D non-periodic flag attribute.

    ``events`` is a list over events, each a list of chords (lists of pitch
    classes); ``flags`` a matching list of scalar flag coordinates.
    """
    n_chords = len(events[0])
    n_slots = len(events[0][0])
    tags = np.concatenate([np.full(n_slots, k) for k in range(n_chords)])
    cols = [np.array([v for ch in ev for v in ch], float) for ev in events]
    p_harm = np.stack(cols, axis=1)                       # (slots, N_events)
    p_flag = np.array(flags, float).reshape(1, -1)        # (1, N_events)
    specs = [{"tags": tags, "r": [r_in, n_chords], "sym": [True, False],
              "rel": [0, rel_out]},
             {"r": 1, "sym": False, "rel": False}]
    return build_exp_tens([p_harm, p_flag], None, specs=specs,
                          sigma=[SIG, SF], is_per=[True, False],
                          period=[P, 1.0], verbose=False)


def _both(X, Y):
    c = cos_sim_exp_tens(X, Y, method="contract", verbose=False)
    b = cos_sim_exp_tens(X, Y, method="bulger", verbose=False)
    return c, b


C, E, G, Eb = 0.0, 4.0, 7.0, 3.0
_IVI = [[C, E, G], [G, 11.0, 2.0], [C, E, G]]          # I - V - I (major)
_ivi = [[C, Eb, G], [G, 11.0, 2.0], [C, Eb, G]]        # i - V - i (minor)
_IVI2 = [[C, E, G, C, E, G], [G, 11.0, 2.0, G, 11.0, 2.0],
         [C, E, G, C, E, G]]                            # each chord doubled


@pytest.mark.parametrize("r_in", [1, 2])
def test_ma_nested_flag_match_and_differ(r_in):
    # Flags agree -> joint equals the harmonic match; flags differ -> the
    # strict flag kernel drives the joint to ~0. Both must equal bulger.
    X = _dens([_IVI], [0.5], r_in)
    c, b = _both(X, _dens([_IVI], [0.5], r_in))
    assert c == pytest.approx(b, abs=1e-9) and c == pytest.approx(1.0, abs=1e-9)
    c, b = _both(X, _dens([_ivi], [-0.5], r_in))
    assert c == pytest.approx(b, abs=1e-9) and abs(c) < 1e-6


@pytest.mark.parametrize("r_in", [1, 2])
def test_ma_nested_unequal_cardinality_with_flag(r_in):
    # 3-pitch chords vs the same chords doubled to 6 pitches, plus the flag.
    X = _dens([_IVI], [0.5], r_in)
    Y = _dens([_IVI2], [0.5], r_in)
    c, b = _both(X, Y)
    assert c == pytest.approx(b, abs=1e-9)
    assert c > 1e-3                       # nonzero (the bug returned 0 at r=2)
    assert _both(Y, X)[0] == pytest.approx(c, abs=1e-9)   # symmetric


@pytest.mark.parametrize("r_in", [1, 2])
def test_ma_nested_multi_event(r_in):
    # Two events per side: exercises the per-event-pair matrix combination,
    # not just the single-event factorisation.
    X = _dens([_IVI, _ivi], [0.5, -0.5], r_in)
    Y = _dens([_IVI, [[2., 5., 9.], [9., 1., 4.], [2., 5., 9.]]],
              [0.5, 0.5], r_in)
    c, b = _both(X, Y)
    assert c == pytest.approx(b, abs=1e-9)
    assert _both(X, X)[0] == pytest.approx(1.0, abs=1e-9)


def test_ma_nested_factorises_into_harmonic_times_flag():
    # For single-event densities the cosine factorises exactly into the
    # harmonic cosine (1-attribute contraction) times the flag cosine.
    import math
    r_in = 2
    Xj = _dens([_IVI], [0.5], r_in)
    Yj = _dens([_ivi], [0.5], r_in)         # flags agree -> flag cosine = 1
    joint = cos_sim_exp_tens(Xj, Yj, method="contract", verbose=False)

    def _harm(ev):
        n_slots = len(ev[0])
        tags = np.concatenate([np.full(n_slots, k) for k in range(len(ev))])
        p = np.array([v for ch in ev for v in ch], float).reshape(-1, 1)
        return build_exp_tens([p], None,
                              specs=[{"tags": tags, "r": [r_in, len(ev)],
                                      "sym": [True, False], "rel": [0, 1]}],
                              sigma=[SIG], is_per=[True], period=[P],
                              verbose=False)
    s_harm = cos_sim_exp_tens(_harm(_IVI), _harm(_ivi), method="contract",
                              verbose=False)
    s_flag = math.exp(-(0.5 - 0.5) ** 2 / (4 * SF ** 2))   # = 1
    assert joint == pytest.approx(s_harm * s_flag, abs=1e-9)


# ---------------------------------------------------------------------
# Regression: the nested contraction must also serve one-sided
# normalisation, not cosine alone. Before the fix both gates declined any
# non-cosine normalisation, so method='contract' + 'oneSidedDenom' raised and
# method='auto' fell back to the joint-tuple enumeration (the combinatorial
# blow-up the contraction exists to avoid). The contraction returns the bare
# (xy, xx, yy) triple and the denominator is chosen downstream, so it is
# correct for either normalisation.
# ---------------------------------------------------------------------


def _dens_sa(events, r_in, rel_out=1):
    """Single nested harmonic attribute (no flag): the single-attribute gate."""
    n_chords = len(events[0])
    n_slots = len(events[0][0])
    tags = np.concatenate([np.full(n_slots, k) for k in range(n_chords)])
    cols = [np.array([v for ch in ev for v in ch], float) for ev in events]
    p_harm = np.stack(cols, axis=1)
    specs = [{"tags": tags, "r": [r_in, n_chords], "sym": [True, False],
              "rel": [0, rel_out]}]
    return build_exp_tens([p_harm], None, specs=specs, sigma=[SIG],
                          is_per=[True], period=[P], verbose=False)


@pytest.mark.parametrize("r_in", [1, 2])
def test_one_sided_contraction_matches_bulger_sa(r_in):
    # Single nested attribute. method='contract' previously raised for
    # 'oneSidedDenom'; it must now run and equal the exact enumeration, and
    # the auto dispatch must pick the contraction and agree.
    X = _dens_sa([_IVI2], r_in)        # doubled chords: the costly enumeration case
    Y = _dens_sa([_IVI], r_in)
    contract = cos_sim_exp_tens(X, Y, method="contract",
                                normalize="oneSidedDenom", verbose=False)
    bulger = cos_sim_exp_tens(X, Y, method="bulger",
                              normalize="oneSidedDenom", verbose=False)
    auto = cos_sim_exp_tens(X, Y, normalize="oneSidedDenom", verbose=False)
    assert contract == pytest.approx(bulger, abs=1e-9)
    assert auto == pytest.approx(bulger, abs=1e-9)
    assert np.isfinite(bulger) and bulger > 0.0


@pytest.mark.parametrize("r_in", [1, 2])
def test_one_sided_contraction_matches_bulger_ma(r_in):
    # Multi-attribute (nested (x) flag): the second gate.
    X = _dens([_IVI2], [0.5], r_in)
    Y = _dens([_IVI], [0.5], r_in)
    contract = cos_sim_exp_tens(X, Y, method="contract",
                                normalize="oneSidedDenom", verbose=False)
    bulger = cos_sim_exp_tens(X, Y, method="bulger",
                              normalize="oneSidedDenom", verbose=False)
    auto = cos_sim_exp_tens(X, Y, normalize="oneSidedDenom", verbose=False)
    assert contract == pytest.approx(bulger, abs=1e-9)
    assert auto == pytest.approx(bulger, abs=1e-9)
    assert np.isfinite(bulger) and bulger > 0.0
