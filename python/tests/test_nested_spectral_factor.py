"""Diagonal-exact factorisation of the relative-non-periodic nested inner
product for spectrally-augmented ordered cells.

The factorisation reduces the inner partial index analytically into the partial
template cross-correlation, evaluating only the per-position reference-value overlaps.
It must reproduce the generic partial-by-partial contraction to floating-point
summation order, must engage only for genuine spectral cells (>= 2 partials per
tone), and must leave plain fundamentals (one value per tone) on the generic
path so their numerics are unchanged.
"""
import numpy as np
import pytest

import mpt
from mpt import add_spectra, bind_events, build_exp_tens, cos_sim_exp_tens, unpack_pre_maet
from mpt._tensor._nested_contraction import build_recipe, auto_taus_line
from tests.references.nested_ip_reference import (
    _ip_rel_nonper, _ip_rel_nonper_generic, _ip_rel_nonper_factored,
)

SIG = 0.15
TS = 4.0


def _flat(pitches, n_partials, rho=1.0):
    p = np.asarray(pitches, float)
    n = p.size
    pp, wp = add_spectra(p, None, 'harmonic', n_partials, 'powerlaw', rho,
                         units=12.0)
    v = np.concatenate([pp.reshape(n, n_partials)[i] for i in range(n)])
    w = np.concatenate([wp.reshape(n, n_partials)[i] for i in range(n)])
    return v, w


def _recipe(cell_len, n_partials):
    tags = np.repeat(np.arange(cell_len), n_partials).reshape(-1, 1)
    return build_recipe(np.array([1, cell_len]), np.array([1, 0]), tags,
                        is_rel=True, is_per=False)


# (X, Y, KpX, KpY, rhoX, rhoY): same/different templates, lengths, self/cross
_CASES = [
    ([60, 63, 60, 65], [70, 73, 70, 63], 12, 12, 1.0, 1.0),
    ([60, 63, 60, 65], [60, 70, 60, 65], 12, 12, 1.0, 1.0),
    ([60, 63, 60, 65], [60, 63, 60, 65], 12, 12, 1.0, 1.0),  # self
    ([60, 63, 60, 65], [71, 59, 62, 64], 8, 8, 1.0, 1.0),
    ([60, 63, 60, 65], [70, 73, 70, 63], 12, 7, 1.0, 0.6),   # diff templates
    ([55, 60, 67], [60, 65, 72], 10, 10, 1.0, 1.0),          # 3-note cells
    ([48, 51, 48, 53, 55], [60, 63, 60, 65, 67], 6, 6, 1.0, 1.0),  # 5-note
]


@pytest.mark.parametrize("X,Y,kpx,kpy,rx_rho,ry_rho", _CASES)
def test_factored_matches_generic(X, Y, kpx, kpy, rx_rho, ry_rho):
    vX, wX = _flat(X, kpx, rx_rho)
    vY, wY = _flat(Y, kpy, ry_rho)
    rx, ry = _recipe(len(X), kpx), _recipe(len(Y), kpy)
    taus = auto_taus_line(np.concatenate([vX, vY]), np.concatenate([vX, vY]),
                          SIG, max(np.exp(-0.5 * TS ** 2), 1e-12))
    assert _ip_rel_nonper_factored(rx, ry, vX, vY, wX, wY, SIG, TS,
                                   taus) is not None
    fast = _ip_rel_nonper(rx, ry, vX, vY, wX, wY, SIG, TS, taus)
    gen = _ip_rel_nonper_generic(rx, ry, vX, vY, wX, wY, SIG, TS, taus)
    assert fast == pytest.approx(gen, rel=1e-12, abs=1e-12)


def test_fundamental_falls_through():
    # Kp == 1 (no inner multiset) must not engage the factorisation, so plain
    # relative fundamentals keep their existing numerics bit-for-bit.
    vX, wX = _flat([60, 63, 60, 65], 1)
    vY, wY = _flat([70, 73, 70, 63], 1)
    rx = _recipe(4, 1)
    taus = auto_taus_line(np.concatenate([vX, vY]), np.concatenate([vX, vY]),
                          SIG, 1e-7)
    assert _ip_rel_nonper_factored(rx, rx, vX, vY, wX, wY, SIG, TS,
                                   taus) is None
    assert (_ip_rel_nonper(rx, rx, vX, vY, wX, wY, SIG, TS, taus)
            == _ip_rel_nonper_generic(rx, rx, vX, vY, wX, wY, SIG, TS, taus))


def _cell(pitches, n_partials=12):
    p = np.asarray(pitches, float)
    n = p.size
    pp, wp = add_spectra(p, None, 'harmonic', n_partials, 'powerlaw', 1.0,
                         units=12.0)
    p_attr = [pp.reshape(n, n_partials).T,
              np.arange(n, dtype=float).reshape(1, n)]
    w_attr = [wp.reshape(n, n_partials).T, None]
    pb, wb, sb = unpack_pre_maet(bind_events(p_attr, w_attr, [n, 1], step=1, rel_outer=True))
    return build_exp_tens(pb, wb, sigma=[SIG, 0.125], is_per=[False, False],
                          period=[0.0, 0.0], specs=sb, verbose=False)


def test_cosine_values_stable():
    mpt.set_default(show_hints=False, truncation_sigmas=4.0,
                    kernel_precision='double')
    als = _cell([60, 63, 60, 65])
    # exact restatement -> 1; whole-cell transposition -> 1 (relative quotient,
    # to the trapezoidal tau-quadrature accuracy ~1e-7, not the factorisation);
    # one note an octave away -> partial credit via shared partials.
    assert float(cos_sim_exp_tens(als, _cell([60, 63, 60, 65]),
                                  verbose=False)) == pytest.approx(1.0, abs=1e-9)
    assert float(cos_sim_exp_tens(als, _cell([67, 70, 67, 72]),
                                  verbose=False)) == pytest.approx(1.0, abs=1e-6)
    octave_fold = float(cos_sim_exp_tens(als, _cell([70, 73, 70, 63]),
                                         verbose=False))
    assert 0.50 < octave_fold < 0.60
