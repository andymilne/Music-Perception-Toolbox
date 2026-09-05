"""Fit the nested inner-product cost-model constants to measured timings.

Companion to ``tools/calibrate_nested_cost.py``, which MEASURES the grid. This
script FITS ``_NESTED_COST_LAW`` and ``_NESTED_FLOOR_MS`` in
:mod:`mpt._tensor._nested_cost` to those measurements and prints
ready-to-paste constant blocks for both languages.

HOW TO RUN
----------
From the ``python`` directory, with the package importable::

    PYTHONPATH=. python3 tools/fit_nested_cost.py CSV [CSV ...]

A CSV is either a bare file with the header the harness prints, or a whole
harness transcript, in which case only the block between the ``BEGIN_CSV`` and
``END_CSV`` markers is read. Several files may be given and are pooled --- use
this to concatenate the pieces of a split run --- but they must all come from
the same machine and language, since the constants absorb per-language,
per-machine constant factors.

Useful options::

    --lang {python,matlab,both}   which paste block(s) to print
    --min-cells N                 rows a structure key needs before it gets a
                                  row of its own (default 6); a key with
                                  fewer falls back to the route's pooled fit
    --floor-cells N               cells per key that set the floor (default 3)
    --list-worst N                worst-fitted cells to list per route
    --pooling {key,route,shared-exp}   pooling level of the fitted laws
    --cv [N]                      random-half cross-validation, N repeats
    --compare-pooling             cross-validate every pooling level
    --score-on FILE               fit here, score there (cross-machine)

CROSS-VALIDATION
----------------
The constants are per-machine and per-language, so an in-sample fit says
nothing about the machine the model will run on. ``--cv`` splits the cells in
half forty times, fits on one half and scores on the other, and reports the
mean and spread of the *held-out routing regret* --- wall time the model's
picks would have cost over an oracle's --- beside the held-out prediction
log-ratio error. Regret is the number to read: the routing decision is a
comparison of two estimates, so a model wrong by one common factor routes
perfectly, and a model can win on prediction error and route worse.

``--compare-pooling`` scores the three pooling levels of :func:`fit_model`
against each other, which is the question of how much per-key structure the
data support. ``--score-on`` fits on the CSVs given and scores on another
machine's, which is the question of whether the *form* transfers when the
constants cannot.

WHAT IT FITS
------------
One power law per route and structure key,

    t_ms = exp(a) * term ** b,

with a per-key floor applied by ``max``. The objective is the *log ratio* of
predicted to measured, for the reason the multi-attribute eval fitter gives:
the routing decision depends only on the ratio of two estimates, so a cell
costing 1 ms deserves the same weight as one costing 4 s, and a cell
over-priced three-fold is exactly as wrong as one under-priced three-fold.

For a pure power law that objective is *linear* in ``(a, b)`` after taking
logarithms, so ordinary least squares on ``log t = a + b log term`` is its
exact minimiser --- there is nothing for an iteration to improve. The floor
is what makes the model non-linear, and it is handled by the outer loop:
estimate the floor, fit ``(a, b)`` on the cells the floor does not bind,
recompute which cells the fitted law now puts under the floor, refit. That is
the same reweight-and-refit structure ``tools/fit_ma_eval_cost.py`` uses,
reduced to what this model actually needs.

``b`` is constrained non-negative --- no route gets cheaper as its term grows
--- which is the one place a non-negative solve is called for; where the
unconstrained slope comes out negative the exponent is pinned at zero and the
intercept refitted, which says the route is flat in its term over the measured
range.

The floor is read from the smallest-term cells of each key, as
``_ORBIT_REL_FLOOR_MS`` was: the smallest measured time there is what the
route cannot go under. The harness always computes all three inner matrices,
so the split between the fixed and the per-matrix half of the floor is not
identified by these measurements; the whole floor is assigned to the
per-matrix half, which is the conservative reading (it discounts a
memoised-self call rather than over-charging it).

WHAT IT REPORTS
---------------
Per route, the geometric mean, geometric spread and worst ratio of predicted
over measured, for the shipped constants and the fitted ones. Then two regret
tables against an oracle that always picks the measured-faster arm: the
per-attribute route choice, over cells where both of an attribute's routes
were timed, and the plan-versus-enumeration choice, over cells where the
plan's chosen route and the enumeration were both timed. Regret is the total
time of the model's picks over the total time of the oracle's.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from mpt._tensor._nested_cost import (
    _NESTED_COST_LAW,
    _NESTED_ENUM_SAFETY,
    _NESTED_FLOOR_MS,
    _NESTED_LAW_KEYS,
    _nested_cost_key,
)

ROUTES = ("centres", "taugrid", "contract_relnonper", "contract", "bulger")
TERM_COL = {r: f"term_{'relnonper' if r == 'contract_relnonper' else r}"
            for r in ROUTES}
MS_COL = {r: f"ms_{'relnonper' if r == 'contract_relnonper' else r}"
          for r in ROUTES}

#: Pooling levels the fit can be run at, coarsest last. See :func:`fit_model`.
POOLINGS = ("key", "route", "shared-exp")


# ----------------------------------------------------------------- input


def read_rows(paths):
    rows = []
    header = None
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh]
        if any(ln == "BEGIN_CSV" for ln in lines):
            lo = lines.index("BEGIN_CSV") + 1
            hi = lines.index("END_CSV")
            block = lines[lo:hi]
        else:
            block = [ln for ln in lines if ln and not ln.startswith("#")]
        for ln in block:
            if not ln:
                continue
            if ln.startswith("section,"):
                header = ln.split(",")
                continue
            if header is None:
                raise SystemExit(f"{path}: no header row found")
            f = ln.split(",")
            if len(f) != len(header):
                raise ValueError(f"{path}: expected {len(header)} fields, "
                                 f"got {len(f)} in {ln!r}")
            row = dict(zip(header, f))
            for k, v in list(row.items()):
                if k in ("section", "r_levels", "sym", "wrap"):
                    continue
                row[k] = float(v)
            row["src"] = path
            rows.append(row)
    if not rows:
        raise SystemExit("no data rows found")
    return rows


def route_key(row):
    """Structure key for the fitted laws.

    The per-attribute routes key on the nested attribute's total tuple order;
    the enumeration keys on the density's largest ``r``, which for these
    single-nested cells is that same order (a flat companion at ``r = 1`` or
    ``r = 2`` never exceeds it in this grid).
    """
    return _nested_cost_key(int(row["total_order"]))


# ------------------------------------------------------------------- fit


def _lstsq_log(term, meas):
    """Least squares for ``log t = a + b log term``, with ``b >= 0``.

    Ordinary least squares here *is* the log-ratio objective's minimiser, so
    no iteration is wanted. The non-negativity is imposed by pinning: an
    unconstrained slope below zero would say the route gets faster as its
    work grows, which is never the model we want to ship, so the exponent is
    set to zero and the intercept refitted at that exponent.
    """
    x = np.log(np.maximum(term, 1.0))
    y = np.log(meas)
    A = np.column_stack([np.ones_like(x), x])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    if b < 0.0:
        b = 0.0
        a = float(np.mean(y))
    return a, b


def _fit_shared_exp(rows, floor_cells, passes=4):
    """One exponent shared by every route, with a per-route intercept.

    The coarsest pooling the comparison considers: ``log t = a_route +
    b log term`` solved jointly over all routes, so the five routes differ
    only by a constant factor. Five intercepts and one slope replace the
    twenty intercepts and twenty slopes of the per-key form. The floors stay
    per route --- they are measurements, not fitted parameters, and pooling
    them would price a route's setup at another route's.
    """
    data = {}
    for route in ROUTES:
        usable = [r for r in rows
                  if r[MS_COL[route]] > 0.0 and r[TERM_COL[route]] > 0.0]
        if not usable:
            continue
        term = np.array([r[TERM_COL[route]] for r in usable])
        meas = np.array([r[MS_COL[route]] for r in usable])
        order = np.argsort(term)
        n_floor = min(floor_cells, len(term))
        floor = float(np.min(meas[order[:n_floor]]))
        data[route] = [term, meas, floor, np.ones(len(term), dtype=bool)]
    if not data:
        return {}, {}
    routes = list(data)
    b = 0.0
    inter = {rt: 0.0 for rt in routes}
    for _ in range(passes):
        cols, ys = [], []
        for j, rt in enumerate(routes):
            term, meas, _floor, free = data[rt]
            if free.sum() < 1:
                continue
            x = np.log(np.maximum(term[free], 1.0))
            block = np.zeros((len(x), len(routes) + 1))
            block[:, j] = 1.0
            block[:, -1] = x
            cols.append(block)
            ys.append(np.log(meas[free]))
        if not cols:
            break
        A = np.vstack(cols)
        y = np.concatenate(ys)
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        b = float(sol[-1])
        if b < 0.0:
            b = 0.0
            for j, rt in enumerate(routes):
                term, meas, _floor, free = data[rt]
                inter[rt] = (float(np.mean(np.log(meas[free])))
                             if free.any() else 0.0)
        else:
            for j, rt in enumerate(routes):
                inter[rt] = float(sol[j])
        changed = False
        for rt in routes:
            term, meas, floor, free = data[rt]
            pred = np.exp(inter[rt]) * np.maximum(term, 1.0) ** b
            new_free = pred > floor
            if new_free.sum() >= 2 and not np.array_equal(new_free, free):
                data[rt][3] = new_free
                changed = True
        if not changed:
            break
    laws = {rt: (inter[rt], b) for rt in routes}
    floors = {rt: (0.0, data[rt][2] / 3.0) for rt in routes}
    return laws, floors


def fit_model(rows, pooling, min_cells, floor_cells):
    """``(laws, floors)`` as ``{route: {key: ...}}`` at one pooling level.

    ``pooling`` is one of

    ``key``
        the shipped form: one law per route and structure key, a key with
        fewer than ``min_cells`` measured cells falling back to the route's
        pooled law;
    ``route``
        one law per route, every structure key sharing it --- the per-key
        dimension dropped;
    ``shared-exp``
        one exponent for all routes, one intercept per route.

    Every level returns the same nested dict, so nothing downstream of here
    knows which was used. Routes and keys the given rows cannot fit fall back
    to the shipped constants, which is what makes a random-half fit scorable
    even when a half happens to miss a route.
    """
    laws, floors = {}, {}
    if pooling == "shared-exp":
        L, F = _fit_shared_exp(rows, floor_cells)
        for route in ROUTES:
            if route in L:
                laws[route] = {k: L[route] for k in _NESTED_LAW_KEYS}
                floors[route] = {k: F[route] for k in _NESTED_LAW_KEYS}
            else:
                laws[route], floors[route] = {}, {}
    else:
        big = 10 ** 9 if pooling == "route" else min_cells
        for route in ROUTES:
            laws[route], floors[route] = fit_route(rows, route, big,
                                                   floor_cells)
    for route in ROUTES:
        for k in _NESTED_LAW_KEYS:
            laws[route].setdefault(k, _NESTED_COST_LAW[route][k])
            floors[route].setdefault(k, _NESTED_FLOOR_MS[route][k])
    return laws, floors


def fit_route(rows, route, min_cells, floor_cells, passes=4):
    """``({key: (a, b)}, {key: (fixed, per_matrix)})`` for one route."""
    usable = [r for r in rows
              if r[MS_COL[route]] > 0.0 and r[TERM_COL[route]] > 0.0]
    if not usable:
        return {}, {}
    by_key = {}
    for r in usable:
        by_key.setdefault(route_key(r), []).append(r)

    # Pooled fit, used both as the starting point and as the fallback for a
    # key too thinly measured to carry its own row.
    pooled = _fit_group(usable, route, floor_cells, passes)

    laws, floors = {}, {}
    for key in _NESTED_LAW_KEYS:
        group = by_key.get(key, [])
        if len(group) >= min_cells:
            laws[key], floors[key] = _fit_group(group, route, floor_cells,
                                                passes)
        else:
            laws[key], floors[key] = pooled
    return laws, floors


def _fit_group(group, route, floor_cells, passes):
    term = np.array([r[TERM_COL[route]] for r in group])
    meas = np.array([r[MS_COL[route]] for r in group])
    # Floor: the smallest measured time among the smallest-term cells. Those
    # are the cells where the route is flat in its term, so what they measure
    # is the setup the multiplicative law cannot express.
    order = np.argsort(term)
    n_floor = min(floor_cells, len(group))
    floor = float(np.min(meas[order[:n_floor]]))
    free = np.ones(len(group), dtype=bool)
    a = b = 0.0
    for _ in range(passes):
        if free.sum() >= 2:
            a, b = _lstsq_log(term[free], meas[free])
        elif free.sum() == 1:
            a, b = float(np.log(meas[free][0])), 0.0
        else:
            break
        pred = np.exp(a) * np.maximum(term, 1.0) ** b
        new_free = pred > floor
        if new_free.sum() < 2 or np.array_equal(new_free, free):
            break
        free = new_free
    # The harness computes all three matrices, so only the product
    # ``fixed + 3 * per_matrix`` is identified; the whole floor goes to the
    # per-matrix half (see the module docstring).
    return (a, b), (0.0, floor / 3.0)


# --------------------------------------------------------------- scoring


def predict(row, route, laws, floors):
    key = route_key(row)
    a, b = laws[route][key]
    f, pm = floors[route][key]
    t = math.exp(a) * max(row[TERM_COL[route]], 1.0) ** b
    return max(t, f + pm * 3.0)


def quality(rows, route, laws, floors):
    ok = [r for r in rows if r[MS_COL[route]] > 0.0]
    if not ok:
        return None
    ratio = np.array([predict(r, route, laws, floors) / r[MS_COL[route]]
                      for r in ok])
    lr = np.log(ratio)
    return (float(np.exp(np.mean(lr))), float(np.exp(np.std(lr))),
            float(np.max(np.maximum(ratio, 1.0 / ratio))), len(ok))


def _admissible(row):
    if not row["rel"]:
        return ("centres", "contract")
    if row["per"]:
        return ("centres", "taugrid")
    return ("centres", "contract_relnonper")


def attr_regret(rows, laws, floors):
    """Regret of the per-attribute route choice against the oracle."""
    tot_m = tot_o = 0.0
    miss = 0
    worst = 1.0
    worst_row = None
    n = 0
    for row in rows:
        adm = [rt for rt in _admissible(row) if row[MS_COL[rt]] > 0.0]
        if len(adm) < 2:
            continue
        n += 1
        pick = min(adm, key=lambda rt: predict(row, rt, laws, floors))
        t_m = row[MS_COL[pick]]
        t_o = min(row[MS_COL[rt]] for rt in adm)
        tot_m += t_m
        tot_o += t_o
        if t_m > t_o * 1.0000001:
            miss += 1
            if t_m / t_o > worst:
                worst, worst_row = t_m / t_o, row
    return (tot_m / max(tot_o, 1e-12), miss, worst, worst_row, n)


def plan_regret(rows, laws, floors):
    """Regret of the plan-versus-enumeration choice against the oracle."""
    tot_m = tot_o = 0.0
    miss = 0
    worst = 1.0
    worst_row = None
    n = 0
    for row in rows:
        adm = [rt for rt in _admissible(row) if row[MS_COL[rt]] > 0.0]
        if not adm or row["ms_bulger"] <= 0.0:
            continue
        n += 1
        pick = min(adm, key=lambda rt: predict(row, rt, laws, floors))
        plan_ms = predict(row, pick, laws, floors)
        enum_ms = predict(row, "bulger", laws, floors)
        take_enum = enum_ms * _NESTED_ENUM_SAFETY < plan_ms
        t_m = row["ms_bulger"] if take_enum else row[MS_COL[pick]]
        t_o = min(row["ms_bulger"], min(row[MS_COL[rt]] for rt in adm))
        tot_m += t_m
        tot_o += t_o
        if t_m > t_o * 1.0000001:
            miss += 1
            if t_m / t_o > worst:
                worst, worst_row = t_m / t_o, row
    return (tot_m / max(tot_o, 1e-12), miss, worst, worst_row, n)


def log_ratio_rms(rows, laws, floors):
    """RMS of ``log(predicted / measured)`` over every timed route and cell."""
    lr = []
    for route in ROUTES:
        for r in rows:
            if r[MS_COL[route]] > 0.0:
                lr.append(math.log(predict(r, route, laws, floors)
                                   / r[MS_COL[route]]))
    return float(np.sqrt(np.mean(np.square(lr)))) if lr else float("nan")


# ---------------------------------------------------- cross-validation


def _score(rows, laws, floors):
    return (attr_regret(rows, laws, floors)[0],
            plan_regret(rows, laws, floors)[0],
            log_ratio_rms(rows, laws, floors))


def cv_scores(rows, pooling, min_cells, floor_cells, repeats, seed=20260904):
    """Random-half cross-validation of one pooling level.

    Half the cells fit the laws, the other half score them: routing regret
    against the measured oracle on cells the fit never saw, and the same
    prediction log-ratio error the fit minimises. Repeated ``repeats`` times
    with independent halves; the spread across repeats is what says whether a
    difference between two forms is a difference at all.

    Regret is the quantity to read. It is what routing consumes --- the ratio
    of two predictions decides the pick, and a form that mis-prices both arms
    by the same factor routes perfectly --- so a form can lose on log-ratio
    error and still be the one to ship.
    """
    rng = np.random.default_rng(seed)
    n = len(rows)
    out = []
    for _ in range(repeats):
        idx = rng.permutation(n)
        half = n // 2
        train = [rows[i] for i in idx[:half]]
        test = [rows[i] for i in idx[half:]]
        laws, floors = fit_model(train, pooling, min_cells, floor_cells)
        out.append(_score(test, laws, floors))
    a = np.array(out)
    return a.mean(axis=0), a.std(axis=0), a.shape[0]


def n_params(pooling, rows, min_cells, floor_cells=3):
    """Free law parameters at one pooling level, counted on these rows.

    Distinct fitted ``(a, b)`` pairs, so a key that falls back to its route's
    pooled law is not counted twice. Floors are excluded: they are read off
    the smallest-term cells rather than fitted. Counted on the rows given ---
    pass the cross-validation's own half, since a key that clears
    ``min_cells`` on the whole grid may not clear it on half of it, and it is
    the half's parameter count the held-out score is paying for.
    """
    laws, _ = fit_model(rows, pooling, min_cells, floor_cells)
    if pooling == "shared-exp":
        return 1 + len({laws[r][k][0]
                        for r in ROUTES for k in _NESTED_LAW_KEYS})
    return 2 * sum(len(set(laws[r].values())) for r in ROUTES)


def label(row):
    mode = f"{'rel' if row['rel'] else 'abs'}-{'per' if row['per'] else 'np'}"
    return (f"{row['section']} r=[{row['r_levels']}] sym=[{row['sym']}] "
            f"{int(row['chord'])}x{int(row['n_chords'])} N={int(row['N'])} "
            f"{mode} sigma={row['sigma']:g} R={int(row['total_order'])}")


# ---------------------------------------------------------------- output


def emit_python(laws, floors):
    print("# ---- Python: paste into mpt/_tensor/_nested_cost.py ----")
    print("_NESTED_COST_LAW = {")
    for route in ROUTES:
        body = ", ".join(
            f"{k}: ({laws[route][k][0]:.4f}, {laws[route][k][1]:.4f})"
            for k in _NESTED_LAW_KEYS)
        print(f'    "{route}": {{{body}}},')
    print("}")
    print("_NESTED_FLOOR_MS = {")
    for route in ROUTES:
        body = ", ".join(
            f"{k}: ({floors[route][k][0]:.4g}, {floors[route][k][1]:.4g})"
            for k in _NESTED_LAW_KEYS)
        print(f'    "{route}": {{{body}}},')
    print("}")


def emit_matlab(laws, floors):
    print("% ---- MATLAB: paste into the nested cost model ----")
    print(f"    NESTED_LAW_KEYS = "
          f"[{', '.join(str(k) for k in _NESTED_LAW_KEYS)}];")
    for route in ROUTES:
        name = route.upper()
        a = ", ".join(f"{laws[route][k][0]:.4f}" for k in _NESTED_LAW_KEYS)
        b = ", ".join(f"{laws[route][k][1]:.4f}" for k in _NESTED_LAW_KEYS)
        print(f"    NESTED_COST_LAW_A_{name} = [{a}];")
        print(f"    NESTED_COST_LAW_B_{name} = [{b}];")
    for route in ROUTES:
        name = route.upper()
        f0 = ", ".join(f"{floors[route][k][0]:.4g}" for k in _NESTED_LAW_KEYS)
        pm = ", ".join(f"{floors[route][k][1]:.4g}" for k in _NESTED_LAW_KEYS)
        print(f"    NESTED_FLOOR_FIXED_{name} = [{f0}];")
        print(f"    NESTED_FLOOR_PER_MATRIX_{name} = [{pm}];")


# ------------------------------------------------------------------ main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--lang", choices=("python", "matlab", "both"),
                    default="both")
    ap.add_argument("--min-cells", type=int, default=6)
    ap.add_argument("--floor-cells", type=int, default=3)
    ap.add_argument("--list-worst", type=int, default=5)
    ap.add_argument("--pooling", choices=POOLINGS, default="key",
                    help="pooling level of the fitted laws (default key)")
    ap.add_argument("--cv", type=int, nargs="?", const=40, default=0,
                    metavar="N",
                    help="random-half cross-validation, N repeats "
                         "(default 40 when given without a value)")
    ap.add_argument("--compare-pooling", action="store_true",
                    help="cross-validate every pooling level and stop")
    ap.add_argument("--score-on", metavar="FILE", action="append", default=[],
                    help="fit on the CSVs above, score on this one "
                         "(repeatable; cross-machine transfer)")
    args = ap.parse_args(argv)

    rows = read_rows(args.csv)

    if args.compare_pooling or args.cv:
        reps = args.cv or 40
        print(f"\n--- random-half cross-validation, {reps} repeats, "
              f"{len(rows)} cells ---")
        print(f"{'pooling':<12s} {'params':>7s} {'attr regret':>19s} "
              f"{'plan regret':>19s} {'log-ratio rms':>19s}")
        levels = POOLINGS if args.compare_pooling else (args.pooling,)
        for pooling in levels:
            mean, sd, _ = cv_scores(rows, pooling, args.min_cells,
                                    args.floor_cells, reps)
            print(f"{pooling:<12s} "
                  f"{n_params(pooling, rows[:len(rows) // 2], args.min_cells):>7d} "
                  f"{mean[0]:>10.4f} +- {sd[0]:<6.4f} "
                  f"{mean[1]:>10.4f} +- {sd[1]:<6.4f} "
                  f"{mean[2]:>10.4f} +- {sd[2]:<6.4f}")
        if args.compare_pooling:
            return 0

    if args.score_on:
        test = read_rows(args.score_on)
        print(f"\n--- transfer: fit on {len(rows)} cells, score on "
              f"{len(test)} cells of {', '.join(args.score_on)} ---")
        print(f"{'pooling':<12s} {'attr regret':>12s} {'plan regret':>12s} "
              f"{'log-ratio rms':>14s}")
        for pooling in POOLINGS:
            laws, floors = fit_model(rows, pooling, args.min_cells,
                                     args.floor_cells)
            a, p, lr = _score(test, laws, floors)
            print(f"{pooling:<12s} {a:>12.4f} {p:>12.4f} {lr:>14.4f}")
        a, p, lr = _score(test, {r: _NESTED_COST_LAW[r] for r in ROUTES},
                          {r: _NESTED_FLOOR_MS[r] for r in ROUTES})
        print(f"{'shipped':<12s} {a:>12.4f} {p:>12.4f} {lr:>14.4f}")
        # An oracle fitted ON the test set: the regret no law of this form
        # can beat on these cells, and so the bar the transfer is against.
        laws, floors = fit_model(test, args.pooling, args.min_cells,
                                 args.floor_cells)
        a, p, lr = _score(test, laws, floors)
        print(f"{'in-sample':<12s} {a:>12.4f} {p:>12.4f} {lr:>14.4f}")
        return 0

    print(f"{len(rows)} cells.")
    for route in ROUTES:
        n = sum(1 for r in rows if r[MS_COL[route]] > 0.0)
        keys = sorted({route_key(r) for r in rows if r[MS_COL[route]] > 0.0})
        print(f"  {route:<20s} timed on {n:4d} cells; keys {keys}")

    shipped_laws = {r: _NESTED_COST_LAW[r] for r in ROUTES}
    shipped_floors = {r: _NESTED_FLOOR_MS[r] for r in ROUTES}

    laws, floors = fit_model(rows, args.pooling, args.min_cells,
                             args.floor_cells)

    print("\n--- fitted laws: t_ms = exp(a) * term ** b, floor by max ---")
    print(f"{'route':<20s} {'key':>4s} {'a':>10s} {'b':>8s} {'floor_ms':>10s}")
    for route in ROUTES:
        for k in _NESTED_LAW_KEYS:
            a, b = laws[route][k]
            f, pm = floors[route][k]
            print(f"{route:<20s} {k:>4d} {a:>10.4f} {b:>8.4f} "
                  f"{f + 3 * pm:>10.4f}")

    print("\n--- fit quality: predicted / measured ---")
    print(f"{'route':<20s} {'constants':<9s} {'geo mean':>9s} {'geo sd':>8s} "
          f"{'worst':>8s} {'cells':>6s}")
    for route in ROUTES:
        for tag, L, F in (("shipped", shipped_laws, shipped_floors),
                          ("fitted", laws, floors)):
            q = quality(rows, route, L, F)
            if q is None:
                continue
            print(f"{route:<20s} {tag:<9s} {q[0]:>9.3f} {q[1]:>8.3f} "
                  f"{q[2]:>8.1f} {q[3]:>6d}")

    print("\n--- routing regret vs oracle ---")
    print(f"{'decision':<24s} {'constants':<9s} {'total/oracle':>13s} "
          f"{'misroutes':>10s} {'worst':>8s} {'cells':>6s}  worst cell")
    for name, fn in (("per-attribute route", attr_regret),
                     ("plan vs enumeration", plan_regret)):
        for tag, L, F in (("shipped", shipped_laws, shipped_floors),
                          ("fitted", laws, floors)):
            reg, miss, worst, wrow, n = fn(rows, L, F)
            detail = label(wrow) if wrow is not None else "-"
            print(f"{name:<24s} {tag:<9s} {reg:>13.3f} {miss:>10d} "
                  f"{worst:>8.2f} {n:>6d}  {detail}")

    if args.list_worst:
        for route in ROUTES:
            ok = [r for r in rows if r[MS_COL[route]] > 0.0]
            if not ok:
                continue
            ratio = [(predict(r, route, laws, floors) / r[MS_COL[route]], r)
                     for r in ok]
            ratio.sort(key=lambda t: -max(t[0], 1.0 / t[0]))
            print(f"\n--- worst {args.list_worst} {route} cells "
                  f"(fitted pred/meas) ---")
            for ra, r in ratio[:args.list_worst]:
                print(f"  {ra:6.3f}  pred {predict(r, route, laws, floors):9.3f}"
                      f"  meas {r[MS_COL[route]]:9.3f}  {label(r)}")

    print()
    if args.lang in ("python", "both"):
        emit_python(laws, floors)
        print()
    if args.lang in ("matlab", "both"):
        emit_matlab(laws, floors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
