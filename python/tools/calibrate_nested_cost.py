"""Timing grid for calibrating the nested-attribute inner-product cost model.

Companion to ``tools/fit_nested_cost.py``, which FITS the constants in
:mod:`mpt._tensor._nested_cost` to what this script MEASURES. The constants
absorb per-language and per-machine constant factors --- interpreter overhead,
array layout, BLAS --- so each language and each machine must be calibrated on
its own measurements.

HOW TO RUN
----------
From the ``python`` directory, with the package importable::

    PYTHONPATH=. python3 tools/calibrate_nested_cost.py > nested_cost.txt

The whole grid runs in roughly ten minutes on a quiet two-core machine. Where
a run must be split into short pieces --- a laptop, a session that cannot be
left alone for ten minutes --- the cell list is a plain enumerable sequence
and a driver can walk it a piece at a time::

    PYTHONPATH=. python3 tools/calibrate_nested_cost.py --start 0 --seconds 150
    PYTHONPATH=. python3 tools/calibrate_nested_cost.py --start 48 --seconds 150
    ...

Each run prints the index to resume from. Every cell is seeded from its own
index, so a cell measures the same shape wherever it is run and the pieces
concatenate into one grid. ``--list`` prints the cell list without timing
anything.

``--out PATH`` appends each row to a CSV file as the cell completes, writing
the header once when the file is new. Successive chunks append into one file
the fitter reads directly, a run that is interrupted keeps every cell it
finished, and a reader can watch the file while the run proceeds. Standard
output is unchanged, so the ``BEGIN_CSV`` block is still printed at the end of
each chunk::

    PYTHONPATH=. python3 tools/calibrate_nested_cost.py --out nested.csv \\
        --start 0 --seconds 150
    PYTHONPATH=. python3 tools/calibrate_nested_cost.py --out nested.csv \\
        --start 48 --seconds 150
    PYTHONPATH=. python3 tools/fit_nested_cost.py nested.csv

WHAT TO SEND BACK
-----------------
Everything between the ``BEGIN_CSV`` and ``END_CSV`` markers, inclusive of the
header row, from every piece. That block alone is sufficient to fit the
constants: its columns fully determine the features (mode, level arities and
symmetries, chord size, chord count, event count, sigma, span or period, wrap)
as well as the analytic terms the laws are fitted against.

WHAT IS TIMED
-------------
Per cell, each route that carries the cell's measure is forced and timed on
the three inner matrices the cosine needs (xy, xx, yy), with the densities'
memos cold at the start of every repeat --- the quantity the per-attribute
laws predict. The joint-tuple enumeration is timed on the same triple through
:func:`~mpt._tensor.cosine._cos_sim_exp_tens_ma_pairwise`, so the two sides
are measured over the same work.

A route whose analytic term exceeds its skip bound is not timed and is
recorded as ``-1``; the bounds keep the run finite where a single call would
otherwise run for minutes, and a cell over the bound is one whose routing is
decided by orders of magnitude rather than by the fitted constants.

SECTIONS
--------
A  Shape sweep: every level shape against every legal chord size and chord
   count, at two events, in all four modes. This is what determines the
   exponents.
B  Event sweep: one and four events on a reduced shape set, in all four
   modes. Both sides price per event pair but do not scale with it alike ---
   the enumeration builds one joint kernel over all events, the contraction
   repeats a per-pair cost --- so a grid at one event count leaves that
   unconstrained.
C  Sigma sweep on relative-periodic shapes: the transposition-average node
   count is set by period over sigma, and this is what separates the node
   count's contribution from everything else that moves with sigma.
D  Symmetry patterns: an ordered level has no orbit to collapse, so it
   changes the perm-to-comb ratio and with it the centres route's restriction.
E  Multi-attribute cells: a nested attribute tensored with a flat ``r = 1``
   attribute and with a flat symmetric ``r = 2`` attribute. These constrain
   nothing in the per-attribute laws --- they are the same attribute --- but
   they are where the plan-versus-enumeration comparison actually bites, so
   they are measured and reported.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

import mpt
from mpt import build_exp_tens
from mpt._tensor import cosine as _cos
from mpt._tensor._nested_cost import nested_attr_terms
from mpt._tensor.cosine import (
    _cos_sim_exp_tens_ma_pairwise,
    _nested_attr_matrix,
    _nested_attr_plan,
)

PERIOD = 12.0
SPAN = 10.0
SIGMA_NONPER = 0.24

#: Analytic-term bounds above which a route is not timed (recorded as -1).
#: Set from the sandbox measurements: the centres route runs about 0.5 us per
#: term unit at total order 6, so three million units is a few seconds, and
#: the contraction routes run about four orders of magnitude cheaper per unit.
CENTRES_TERM_SKIP = 3.0e6
CONTRACT_TERM_SKIP = 4.0e8
BULGER_TERM_SKIP = 6.0e6

BUDGET_SEC = 1.0
LONG_CALL_SEC = 0.3
N_TIMED = 5


def _time_ms(setup, run):
    """Median of several timed repeats, each on freshly set-up state.

    ``setup()`` returns the arguments ``run()`` consumes and is not timed; it
    is what makes each repeat cold, which is the state a real call finds. A
    first call slower than ``LONG_CALL_SEC`` is reported from that call alone:
    repeating a call that already ran for a third of a second removes scatter
    whose share of the total is negligible and costs the rest of the budget.
    """
    args = setup()
    t0 = time.perf_counter()
    run(args)
    first = time.perf_counter() - t0
    if first > LONG_CALL_SEC:
        return first * 1000.0
    samples = []
    spent = 0.0
    for _ in range(N_TIMED):
        args = setup()
        t0 = time.perf_counter()
        run(args)
        dt = time.perf_counter() - t0
        samples.append(dt * 1000.0)
        spent += dt
        if spent > BUDGET_SEC and len(samples) >= 3:
            break
    return sorted(samples)[len(samples) // 2]


# --------------------------------------------------------------- densities


def _nested_spec(r_levels, sym, chord, ngroup, is_rel):
    tags = np.repeat(np.arange(ngroup), chord)
    return dict(r=list(r_levels), sym=[bool(s) for s in sym], tags=tags,
                rel=([0] * (len(r_levels) - 1) + [1] if is_rel else None))


def _build(cell, rng, jitter):
    """One density for ``cell``; ``jitter`` separates the X and Y sides."""
    r_levels, sym = cell["r_levels"], cell["sym"]
    chord, ngroup, N = cell["chord"], cell["ngroup"], cell["N"]
    is_rel, is_per = cell["rel"], cell["per"]
    K = chord * ngroup
    hi = PERIOD if is_per else SPAN
    v = np.sort(rng.uniform(0.0, hi, (K, N)), axis=0) + jitter
    if is_per:
        v = np.mod(v, PERIOD)
        v = np.sort(v, axis=0)
    p = [v]
    specs = [_nested_spec(r_levels, sym, chord, ngroup, is_rel)]
    sigma = [cell["sigma"]]
    per = [is_per]
    period = [PERIOD if is_per else 0.0]
    wraps = [cell["wrap"]]
    flat_r = cell["flat_r"]
    if flat_r:
        flat_K = cell["flat_K"]
        p.append(rng.uniform(0.0, 5.0, (flat_K, N)) + jitter)
        specs.append(dict(r=flat_r, rel=False, sym=True))
        sigma.append(0.5)
        per.append(False)
        period.append(0.0)
        wraps.append('full-image')
    return build_exp_tens(p, None, specs=specs, sigma=sigma, is_per=per,
                          period=period, wrap=wraps, verbose=False)


def _pair(cell, seed):
    rng = np.random.default_rng(seed)
    return _build(cell, rng, 0.0), _build(cell, rng, 0.13)


def _forced_plan(dens_x, dens_y, route):
    """The shared quadrature grid the plan would build for ``route``."""
    orig = _cos._nested_attr_route
    _cos._nested_attr_route = lambda dx, dy, a, **kw: route
    try:
        return _nested_attr_plan(dens_x, dens_y, 0)[1]
    finally:
        _cos._nested_attr_route = orig


# ------------------------------------------------------------------- cells


LEVEL_SHAPES = [
    ([1, 2], 2), ([2, 2], 4), ([1, 3], 3), ([2, 3], 6), ([3, 2], 6),
]
SYM_PATTERNS = [[1, 1], [0, 1], [1, 0]]
MODES = [(False, False), (False, True), (True, False), (True, True)]


def _legal(r_levels, chord, ngroup):
    return chord >= r_levels[0] and ngroup >= r_levels[1]


def _cell(section, r_levels, sym, chord, ngroup, N, rel, per, sop,
          flat_r=0, flat_K=0, wrap='full-image'):
    sigma = sop * PERIOD if per else SIGMA_NONPER
    return dict(section=section, r_levels=list(r_levels),
                sym=list(sym), chord=chord, ngroup=ngroup, N=N,
                rel=bool(rel), per=bool(per), sop=(sop if per else 0.0),
                sigma=sigma, flat_r=flat_r, flat_K=flat_K, wrap=wrap)


def cells():
    """The cell list, in a fixed order, as plain dictionaries.

    A driver may enumerate this and call :func:`run_cell` one cell at a time;
    nothing carries over between cells.
    """
    out = []
    # A: shape sweep.
    for r_levels, _R in LEVEL_SHAPES:
        for chord in (2, 3, 4):
            for ngroup in (2, 3, 4):
                if not _legal(r_levels, chord, ngroup):
                    continue
                for rel, per in MODES:
                    out.append(_cell("A", r_levels, [1, 1], chord, ngroup,
                                     2, rel, per, 0.02))
    # B: event sweep on a reduced shape set.
    for r_levels, _R in LEVEL_SHAPES:
        chord = max(2, r_levels[0])
        ngroup = max(2, r_levels[1])
        for N in (1, 4):
            for rel, per in MODES:
                out.append(_cell("B", r_levels, [1, 1], chord, ngroup, N,
                                 rel, per, 0.02))
    # C: sigma sweep, relative-periodic only (the tau-grid node count).
    for r_levels, _R in LEVEL_SHAPES:
        chord = max(2, r_levels[0])
        ngroup = max(2, r_levels[1])
        for sop in (0.005, 0.05):
            out.append(_cell("C", r_levels, [1, 1], chord, ngroup, 2,
                             True, True, sop))
    # D: symmetry patterns.
    for r_levels, _R in LEVEL_SHAPES:
        chord = max(2, r_levels[0]) + 1
        ngroup = max(2, r_levels[1])
        for sym in SYM_PATTERNS[1:]:
            for rel, per in MODES:
                out.append(_cell("D", r_levels, sym, chord, ngroup, 2,
                                 rel, per, 0.02))
    # E: multi-attribute cells.
    for r_levels, _R in ([1, 2], 2), ([2, 2], 4), ([2, 3], 6):
        for flat_r, flat_K in ((1, 3), (2, 5)):
            for rel, per in MODES:
                out.append(_cell("E", r_levels, [1, 1], 3, 3, 2, rel, per,
                                 0.02, flat_r=flat_r, flat_K=flat_K))
    return out


# ------------------------------------------------------------------ timing


def _routes_for(cell):
    """The routes to time on this cell: the ones that carry its measure."""
    if not cell["rel"]:
        return ["centres", "contract"]
    if cell["per"]:
        return ["centres", "taugrid"]
    return ["centres", "contract_relnonper"]


def run_cell(cell, seed):
    """Time one cell and return ``(row, human)``.

    ``seed`` is the cell's own seed, so a cell measures the same shape
    wherever in a split run it is reached.
    """
    dx, dy = _pair(cell, seed)
    terms, info = nested_attr_terms(dx, dy, 0)
    from mpt._tensor._nested_cost import predict_nested_pairwise_kernel_size
    term_bulger = predict_nested_pairwise_kernel_size(dx, dy)

    ms = {}
    for route in _routes_for(cell):
        bound = (CENTRES_TERM_SKIP if route == "centres"
                 else CONTRACT_TERM_SKIP)
        if terms[route] > bound:
            ms[route] = -1.0
            continue
        taus = _forced_plan(dx, dy, route)

        def setup(route=route):
            a, b = _pair(cell, seed)
            return a, b, taus

        def run(args, route=route):
            a, b, t = args
            for u, v in ((a, b), (a, a), (b, b)):
                _nested_attr_matrix(u, v, 0, route, t).sum()

        ms[route] = _time_ms(setup, run)

    if term_bulger <= BULGER_TERM_SKIP:
        def setup_b():
            return _pair(cell, seed)

        def run_b(args):
            a, b = args
            _cos_sim_exp_tens_ma_pairwise(a, b, verbose=False)

        ms["bulger"] = _time_ms(setup_b, run_b)
    else:
        ms["bulger"] = -1.0

    row = ",".join(str(v) for v in (
        cell["section"],
        "|".join(str(v) for v in cell["r_levels"]),
        "|".join(str(v) for v in cell["sym"]),
        cell["chord"], cell["ngroup"], cell["chord"] * cell["ngroup"],
        cell["N"], int(cell["rel"]), int(cell["per"]),
        f"{cell['sigma']:g}", f"{PERIOD if cell['per'] else SPAN:g}",
        cell["wrap"], cell["flat_r"], cell["flat_K"],
        int(info["total_order"]),
        f"{info['m_perm_x']:g}", f"{info['m_comb_x']:g}",
        f"{info['m_perm_y']:g}", int(info["restricted_x"]),
        f"{info['work_x']:g}", f"{info['n_tau']:g}", f"{info['n_line']:g}",
        f"{terms['centres']:.6g}", f"{terms['taugrid']:.6g}",
        f"{terms['contract_relnonper']:.6g}", f"{terms['contract']:.6g}",
        f"{term_bulger:.6g}",
        f"{ms.get('centres', -1.0):.4f}", f"{ms.get('taugrid', -1.0):.4f}",
        f"{ms.get('contract_relnonper', -1.0):.4f}",
        f"{ms.get('contract', -1.0):.4f}", f"{ms['bulger']:.4f}",
    ))
    mode = f"{'rel' if cell['rel'] else 'abs'}-{'per' if cell['per'] else 'np'}"
    human = (f"{cell['section']} r={cell['r_levels']} sym={cell['sym']} "
             f"{cell['chord']}x{cell['ngroup']} N={cell['N']} {mode} "
             f"sigma={cell['sigma']:g} flat={cell['flat_r']} | "
             + " ".join(f"{k}={v:.3f}" for k, v in ms.items()))
    return row, human


CSV_HEADER = (
    "section,r_levels,sym,chord,n_chords,K,N,rel,per,sigma,span,wrap,"
    "flat_r,flat_K,total_order,m_perm_x,m_comb_x,m_perm_y,restricted,"
    "work,n_tau,n_line,term_centres,term_taugrid,term_relnonper,"
    "term_contract,term_bulger,ms_centres,ms_taugrid,ms_relnonper,"
    "ms_contract,ms_bulger"
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many cells (0 = no limit)")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="stop once this much wall time has been spent "
                         "(0 = no limit); the resume index is printed")
    ap.add_argument("--section", default="",
                    help="restrict to these section letters, e.g. ABC")
    ap.add_argument("--out", default="",
                    help="append each row to this CSV file as the cell "
                         "completes (header written once, when the file is "
                         "new); stdout is unchanged")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    all_cells = cells()
    if args.section:
        keep = set(args.section.upper())
        all_cells = [c for c in all_cells if c["section"] in keep]
    if args.list:
        for i, c in enumerate(all_cells):
            print(i, c)
        print(f"{len(all_cells)} cells.")
        return 0

    # Rows are appended as they are measured, and flushed, so a run that is
    # stopped --- by the budget, by the user, or by the machine --- leaves
    # every cell it finished on disk, and a reader can consume the file while
    # the run proceeds. The header is written only when the file is new, so
    # successive chunks append into one grid that the fitter reads directly.
    out = None
    if args.out:
        fresh = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
        out = open(args.out, "a", encoding="utf-8")
        if fresh:
            out.write(CSV_HEADER + "\n")
            out.flush()

    prev_hints = mpt.get_default("show_hints")
    mpt.set_default(show_hints=False)
    rows = []
    t_start = time.perf_counter()
    i = args.start
    try:
        while i < len(all_cells):
            if args.limit and (i - args.start) >= args.limit:
                break
            if args.seconds and (time.perf_counter() - t_start) > args.seconds:
                break
            row, human = run_cell(all_cells[i], seed=1000 + i)
            rows.append(row)
            if out is not None:
                out.write(row + "\n")
                out.flush()
            print(f"[{i}] {human}", file=sys.stderr, flush=True)
            i += 1
    finally:
        mpt.set_default(show_hints=prev_hints)
        if out is not None:
            out.close()

    print("\nBEGIN_CSV")
    print(CSV_HEADER)
    for row in rows:
        print(row)
    print("END_CSV")
    print(f"# {len(rows)} cells; next index {i} of {len(all_cells)}; "
          f"{time.perf_counter() - t_start:.1f} s")
    if i < len(all_cells):
        print(f"# resume with --start {i}")
    if args.out:
        print(f"# rows appended to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
