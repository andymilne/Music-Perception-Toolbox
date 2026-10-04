"""The nested routes' scales, and a sweep that combines two routes.

A nested attribute's inner product can be computed by up to two routes --
the materialised tuple centres and a per-level contraction (absolute; a
translation-line or period grid when relative) -- and the routes' bare
matrices differ by constants known in closed form
(:func:`~mpt._tensor.cosine._nested_route_scale`). Every combiner puts its
nested matrices on one common scale (``_nested_attr_matrix_common``), so a
ratio may draw its terms from different routes.

This file pins two things:

* each route, taken to the common scale, equals a first-principles
  reference -- every ordered tuple pair, the Gaussian of the coordinate
  differences (wrapped Gaussian when periodic; the transposition integral
  or its period average when relative) -- across two- and three-level
  nestings, ordered and unordered levels, NaN padding, absolute and
  relative, periodic and not;
* the contraction sweep, whose cross term is priced on its own and may
  take a route its self inner products did not, matches the per-offset
  similarity when the two routes differ. Before the common scale, the
  routes' constants entered that ratio: the one-sided similarity of a
  three-chord query at inner r = 1 came out 39 times too small.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import mpt
from mpt import build_maet, sim_maet
from mpt._tensor import cosine as C
from mpt._tensor.build import _nested_enum_indices
from mpt._tensor.sweep import sweep_sim_maet
from mpt._tensor._mobius_inner import _nested_orbit_mult
from mpt._tensor._nested_contraction import auto_ntau_default, auto_taus_line
from mpt._defaults import resolve_truncation_sigmas, truncation_floor

TS = resolve_truncation_sigmas(None)
BASE = np.array([0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0, 12.0, 14.0, 16.0])


def _tree(r_levels, group_sizes):
    """Tags for a regular tag tree: group_sizes[0] leaves per finest group,
    group_sizes[l] groups of level l under each level-(l+1) group."""
    L = len(r_levels)
    paths = []

    def rec(level, path):
        if level == 0:
            paths.extend([path] * group_sizes[0])
            return
        for g in range(group_sizes[level]):
            rec(level - 1, path + (g,))

    rec(L - 1, ())
    tags = np.zeros((len(paths), L - 1), dtype=int)
    for j in range(L - 1):
        keys = [p[:L - 1 - j] for p in paths]
        uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
        tags[:, j] = [uniq[k] for k in keys]
    return tags[:, 0] if L == 2 else tags


def _pair(r_levels, exch, rel, per, sigma, group_sizes, seed, P=12.0):
    rng = np.random.default_rng(seed)
    tags = _tree(r_levels, group_sizes)
    K = len(tags)
    VX = BASE[rng.integers(0, BASE.size, size=(K, 3))] \
        + rng.normal(0, 0.3 * sigma, size=(K, 3))
    WX = rng.uniform(0.5, 1.5, size=(K, 3))
    # Y: X's first two events, transposed when relative and jittered, so the
    # cross terms are substantial.
    VY = VX[:, :2] + (3.0 if any(rel) else 0.0) \
        + rng.normal(0, 0.5 * sigma, size=(K, 2))
    WY = WX[:, :2] * rng.uniform(0.7, 1.3, size=(K, 2))
    if group_sizes[0] > r_levels[0]:
        VX[0, 2] = np.nan      # a NaN-padded value
        WX[0, 2] = 0.0
    spec = dict(tags=tags, r=list(r_levels), exch=list(exch), rel=list(rel),
                sigma=sigma, per=per, period=P if per else 0.0, name="x")
    return VX, WX, VY, WY, spec


def _theta(d, sigma, P, nimg=2):
    n = np.arange(-nimg, nimg + 1)
    return np.exp(-((d[..., None] + n * P) ** 2) / (4 * sigma ** 2)).sum(-1)


def _reference(VX, WX, VY, WY, spec):
    """Common-scale event-pair matrix from every ordered tuple pair."""
    r, ex, sigma = spec["r"], spec["exch"], spec["sigma"]
    P, per, rel = spec["period"], spec["per"], any(spec["rel"])
    tags = np.asarray(spec["tags"]).reshape(len(spec["tags"]), -1)
    s = int(np.prod(r))
    taus = np.linspace(0.0, P, 512, endpoint=False) if (per and rel) else None

    def tuples(V, W, n):
        vals = np.where(np.isfinite(V[:, n]))[0]
        perm, _ = _nested_enum_indices(vals, tags[vals], r, ex)
        return V[perm, n].T, np.prod(W[perm, n], axis=0)

    M = np.zeros((VX.shape[1], VY.shape[1]))
    for i in range(VX.shape[1]):
        tx, wx = tuples(VX, WX, i)
        for j in range(VY.shape[1]):
            ty, wy = tuples(VY, WY, j)
            D = tx[:, None, :] - ty[None, :, :]
            if not rel:
                k = (np.prod(_theta(D, sigma, P), -1) if per
                     else np.exp(-(D ** 2).sum(-1) / (4 * sigma ** 2)))
            elif not per:
                U = D - D.mean(-1, keepdims=True)
                k = np.exp(-(U ** 2).sum(-1) / (4 * sigma ** 2))
            else:
                # The transposition average of the all-image kernel, on the
                # centres scale: P sqrt(s) / (2 sigma sqrt(pi)) times it.
                acc = np.ones(D.shape[:2] + (taus.size,))
                for c in range(s):
                    acc *= _theta(D[..., c, None] + taus, sigma, P)
                k = acc.mean(-1) * P * math.sqrt(s) / (2 * sigma
                                                      * math.sqrt(math.pi))
            M[i, j] = (wx[:, None] * wy[None, :] * k).sum()
    return M


CASES = [
    # r_levels, exch, rel, per, sigma, group sizes
    ([1, 3], [1, 0], [0, 1], True, 0.3, [3, 3]),      # Analysis 1.3, inner r = 1
    ([2, 2], [1, 1], [0, 1], True, 0.3, [3, 2]),
    ([2, 2], [0, 1], [0, 1], True, 0.3, [3, 3]),
    ([2, 3], [1, 0], [0, 1], False, 0.3, [3, 3]),
    ([2, 3], [1, 0], [0, 0], True, 0.3, [3, 3]),
    ([2, 2], [1, 1], [0, 0], False, 0.3, [3, 3]),
    ([1, 2, 2], [1, 0, 1], [0, 0, 1], True, 0.3, [2, 2, 2]),
    ([2, 2, 2], [1, 1, 0], [0, 0, 1], False, 0.4, [2, 2, 2]),
]


@pytest.mark.parametrize("r_levels,exch,rel,per,sigma,gs", CASES)
def test_every_route_on_the_common_scale_matches_the_reference(
        r_levels, exch, rel, per, sigma, gs):
    VX, WX, VY, WY, spec = _pair(r_levels, exch, rel, per, sigma, gs, seed=0)
    dx = build_maet([VX], [WX], specs=[spec], verbose=False)
    dy = build_maet([VY], [WY], specs=[spec], verbose=False)
    ref = _reference(VX, WX, VY, WY, spec)
    rxx = np.diag(_reference(VX, WX, VX, WX, spec))
    ryy = np.diag(_reference(VY, WY, VY, WY, spec))
    norm = np.sqrt(rxx[:, None] * ryy[None, :])      # errors on the cosine scale
    if not any(rel):
        routes = ["contract", "centres"]
    elif per:
        routes = ["taugrid", "centres"]
    else:
        routes = ["contract_relnonper", "centres"]
    P = spec["period"]
    for route in routes:
        if route == "taugrid":
            taus = np.linspace(0.0, P, auto_ntau_default(P, sigma, TS),
                               endpoint=False)
        elif route == "contract_relnonper":
            allv = np.concatenate([VX[np.isfinite(VX)], VY[np.isfinite(VY)]])
            taus = auto_taus_line(allv, allv, sigma, truncation_floor(TS))
        else:
            taus = None
        M = C._nested_attr_matrix_common(dx, dy, 0, route, taus,
                                         truncation_sigmas=TS)
        assert np.max(np.abs(M - ref) / norm) < 1e-8, route


def test_common_factor_is_the_closed_form():
    """centres / taugrid = (P / sigma) sqrt(s / 4 pi) |G| -- the value that
    the cadence sweep was missing (39.09 at s = 3, sigma = 0.15, P = 12)."""
    spec = dict(tags=np.repeat(np.arange(3), 3), r=[1, 3], exch=[1, 0],
                rel=[0, 1], sigma=0.15, per=True, period=12.0, name="x")
    V = np.tile(np.array([[60.0], [64.0], [67.0]]), (3, 1))
    d = build_maet([V], [np.ones_like(V)], specs=[spec], verbose=False)
    f = C._nested_common_factor(d, 0, "taugrid")
    assert f == pytest.approx(12.0 / 0.15 * math.sqrt(3 / (4 * math.pi)),
                              rel=1e-14)
    assert f == pytest.approx(39.0882, abs=1e-4)
    spec2 = dict(spec, r=[2, 3])
    d2 = build_maet([V], [np.ones_like(V)], specs=[spec2], verbose=False)
    f2 = C._nested_common_factor(d2, 0, "taugrid")
    G = _nested_orbit_mult([2, 3], [1, 0])
    assert G == 8
    assert f2 == pytest.approx(
        12.0 / 0.15 * math.sqrt(6 / (4 * math.pi)) * G, rel=1e-14)


def _build_cadence():
    """Query and context through bind_events, as Analysis 1.3 builds them."""
    import pandas as pd
    ctx_chords = [[60, 64, 67, 72], [65, 69, 72], [62, 67, 71, 65],
                  [60, 64, 67], [57, 60, 64], [62, 65, 69], [55, 59, 62, 65],
                  [60, 64, 67, 72]]
    q_chords = ([60, 64, 67], [62, 67, 71], [60, 64, 67])

    def bound(chords, L):
        onset = [float(j) for j, c in enumerate(chords) for _ in c]
        pitch = [float(v) for c in chords for v in c]
        table = pd.DataFrame({"onset_beats": onset, "pitch": pitch,
                              "weight": np.ones(len(pitch))})
        pm = mpt.pre_maet_from_attr_table(
            table,
            specs=(dict(column="pitch", sigma=0.15, r=1, exch=True,
                        per=True, period=12.0),
                   dict(column="onset", sigma=0.1)),
            time="beats", weights="weight")
        p, w, specs = mpt.unpack_pre_maet(pm)
        p, w, specs = list(p), list(w or []), list(specs)
        specs[0] = dict(specs[0], r=1, rel=False, exch=True)
        return mpt.bind_events(p, w or None, [L, 1], rel_outer=True,
                               specs=specs)

    return bound(ctx_chords, 3), bound(list(q_chords), 3)


def _dens(pm):
    p, w, specs = mpt.unpack_pre_maet(pm)
    return build_maet(list(p), list(w), specs=list(specs), verbose=False)


@pytest.mark.parametrize("normalize", ["oneSidedDenom", "cosine"])
def test_sweep_matches_per_offset_when_routes_differ(monkeypatch, normalize):
    """Force the cross term onto the tau grid and the self inner products
    onto the centres (both admissible at sigma/P = 0.0125): the sweep must
    still equal the per-offset similarity."""
    ctx, qry = _build_cadence()
    dc, dq = _dens(ctx), _dens(qry)
    real_plan = C._nested_attr_plan

    def split_plan(dens_x, dens_y, a, force_route=None, skip_xx=False,
                   skip_yy=False, ts=None):
        # The sweep prices its cross term with both skip flags set; the
        # ordinary contraction (which supplies the self inner products)
        # does not. The plan honours only a forced 'centres', so the tau
        # grid is returned here as the plan would return it.
        if skip_xx and skip_yy:
            period, sigma = float(dens_x.period[a]), float(dens_x.sigma[a])
            return "taugrid", np.linspace(
                0.0, period, auto_ntau_default(period, sigma, ts),
                endpoint=False)
        return real_plan(dens_x, dens_y, a, force_route="centres",
                         skip_xx=skip_xx, skip_yy=skip_yy, ts=ts)

    monkeypatch.setattr(C, "_nested_attr_plan", split_plan)
    t_ctx = np.asarray(dc.p_attr[1], float)[0]
    offs = np.vstack([np.zeros(t_ctx.size), t_ctx - 2.0])
    swept = np.asarray(sweep_sim_maet(dc, dq, offs, normalize=normalize,
                                      verbose=False), float).ravel()
    monkeypatch.setattr(C, "_nested_attr_plan", real_plan)
    p, w, specs = mpt.unpack_pre_maet(qry)
    for m, mu in enumerate(t_ctx - 2.0):
        pq = [p[0], np.asarray(p[1], float) + mu]
        dq_m = build_maet(pq, list(w), specs=list(specs), verbose=False)
        ref = float(sim_maet(dc, dq_m, normalize=normalize, verbose=False))
        assert swept[m] == pytest.approx(ref, rel=1e-7, abs=1e-12)


@pytest.mark.parametrize("normalize", ["oneSidedDenom", "cosine"])
@pytest.mark.parametrize("which", ["single", "multi"])
def test_self_only_forms_the_same_self_inner_products(normalize, which):
    """The sweep asks the ordinary contraction for its self inner products
    only; they must be the ones the full triple carries."""
    ctx, qry = _build_cadence()
    if which == "single":
        p, w, specs = mpt.unpack_pre_maet(ctx)
        q, v, qspecs = mpt.unpack_pre_maet(qry)
        mk = lambda P, W, S: build_maet([P[0]], [W[0]], specs=[S[0]],
                                        verbose=False)
        pair = lambda: (mk(p, w, specs), mk(q, v, qspecs))
    else:
        pair = lambda: (_dens(ctx), _dens(qry))
    dx, dy = pair()
    full = C._try_nested_contract(dx, dy, normalize=normalize, verbose=False,
                                  force=True)
    dx, dy = pair()            # fresh densities: no memo carried over
    part = C._try_nested_contract(dx, dy, normalize=normalize, verbose=False,
                                  force=True, self_only=True)
    assert part[0] is None
    assert part[2] == pytest.approx(full[2], rel=1e-14)
    if normalize == "cosine":
        assert part[1] == pytest.approx(full[1], rel=1e-14)
