"""Variable-K (NaN-padded) nested attributes on the contraction path.

A padded value is exactly equivalent to a zero-weight value at any finite
value, so the nested contraction must accept NaN-padded events and agree
with the exact enumeration (method='bulger') -- including under one-sided
normalisation, an accompanying plain attribute, and all four
[rel]/[per] crossings of the outer level.
"""
import numpy as np
import pytest

from mpt import bind_events, flat_specs, build_exp_tens, cos_sim_exp_tens, unpack_pre_maet


def _bound(chords, *, r_inner, rel_outer, per, flag=None):
    L = len(chords)
    k_max = max(len(c) for c in chords)
    P = np.full((k_max, L), np.nan)
    W = np.full((k_max, L), np.nan)
    for j, c in enumerate(chords):
        P[:len(c), j] = c
        W[:len(c), j] = 1.0
    specs = flat_specs([P], r=r_inner, rel=False, sym=True)
    pb, wb, sb = unpack_pre_maet(bind_events([P], [W], L, rel_outer=rel_outer, specs=specs))
    attrs, ws, sp = [pb[0]], [wb[0]], [sb[0]]
    sigma, is_per, period = [0.15], [per], [12.0]
    if flag is not None:
        attrs.append(np.array([[float(flag)]]))
        ws.append(np.array([[1.0]]))
        sp.extend(flat_specs([attrs[-1]], r=1, rel=False, sym=False))
        sigma.append(0.1)
        is_per.append(False)
        period.append(0.0)
    return build_exp_tens(attrs, ws, specs=sp, sigma=sigma, is_per=is_per,
                          period=period, verbose=False)


_X = ([60.0, 64.0, 67.0, 72.0], [55.0, 59.0, 62.0])          # ragged: K = 4, 3
_Y = ([60.0, 64.0, 67.0], [55.0, 59.0, 62.0, 65.0, 67.0])    # ragged: K = 3, 5


@pytest.mark.parametrize("r_inner", [1, 2, 3])
@pytest.mark.parametrize("rel_outer", [False, True])
@pytest.mark.parametrize("per", [False, True])
def test_nan_padded_contract_equals_enumeration(r_inner, rel_outer, per):
    dx = _bound(_X, r_inner=r_inner, rel_outer=rel_outer, per=per)
    dy = _bound(_Y, r_inner=r_inner, rel_outer=rel_outer, per=per)
    v_auto = float(cos_sim_exp_tens(dx, dy, normalize='oneSidedDenom',
                                    verbose=False))
    v_enum = float(cos_sim_exp_tens(dx, dy, normalize='oneSidedDenom',
                                    method='bulger', verbose=False))
    assert v_auto == pytest.approx(v_enum, abs=1e-9)


@pytest.mark.parametrize("r_inner", [1, 2])
def test_nan_padded_contract_ma_with_plain_attribute(r_inner):
    dx = _bound(_X, r_inner=r_inner, rel_outer=True, per=True, flag=+0.5)
    dy = _bound(_Y, r_inner=r_inner, rel_outer=True, per=True, flag=-0.5)
    v_auto = float(cos_sim_exp_tens(dx, dy, verbose=False))
    v_enum = float(cos_sim_exp_tens(dx, dy, method='bulger', verbose=False))
    assert v_auto == pytest.approx(v_enum, abs=1e-9)
