"""Fit the multi-attribute eval cost-model constants to measured timings.

Companion to ``tools/calibrate_ma_eval_cost.py`` (and its MATLAB twin
``matlab/tests/bench_ma_eval_calibration.m``), which MEASURE the grid.
This script FITS the ``_MA_COST_*`` constants in
:mod:`mpt._tensor.dispatch` to those measurements and prints
ready-to-paste constant blocks for both languages.

HOW TO RUN
----------
From the ``python`` directory, with the package importable::

    PYTHONPATH=. python3 tools/fit_ma_eval_cost.py CSV [CSV ...]

A CSV is either a bare file with the header

    rel,per,r,K,nQ,sigma,span,joint,cen_ms,mob_ms

or a whole harness transcript, in which case only the block between the
``BEGIN_CSV`` and ``END_CSV`` markers is read. Several files may be
given and are pooled (use this to add extra cells to the standard
grid); they must all come from the *same* machine and language, since
the constants absorb per-language, per-machine constant factors.

Useful options::

    --lang {python,matlab,both}   which paste block(s) to print
    --exp-grid 1.0,1.05,...       candidate periodic per-query exponents
    --cull-grid 12,14,...         candidate values of CENTRES_CULL_C
    --fix NAME=VALUE              pin a constant instead of fitting it
    --no-exp                      shorthand for --exp-grid 1.0
    --cv [N]                      random-half cross-validation, N repeats
    --compare-forms               cross-validate every form in CV_FORMS
    --score-on FILE               fit here, score there (cross-machine)

CROSS-VALIDATION
----------------
The constants are per-machine, so a fit that scores well on the machine it
was measured on says nothing about the machine it will run on. ``--cv``
answers the first half of that: half the cells fit the constants, the other
half score them, forty independent halves, and what is reported is the mean
and spread of the *held-out routing regret* --- wall time the model's picks
would have cost over the oracle's --- beside the held-out prediction
log-ratio error of each route.

Regret is the number to read. Routing compares two estimates, so a form that
mis-prices both arms by one common factor routes perfectly; a form can lose
on log-ratio error and still be the one to ship, and a form can win on it
and route worse.

``--compare-forms`` runs that over ``CV_FORMS``, which is where the model's
only shape freedom lives: the two per-query exponents and the culling
constant, fitted or pinned at round values. Everything else in the model is
linear and is solved rather than chosen, so pinning one of those three is the
only simplification available that does not delete a term.

``--score-on`` answers the second half: fit on one machine's cells, score on
another's. Expect the log-ratio error to be large across machines --- the
constants really are per-machine --- and the regret to survive, since the
ratio is what routing consumes.

WHAT IT FITS
------------
The model is *linear* in every constant except ``CENTRES_CULL_C`` (which
sits inside a power) and ``CENTRES_QUERY_JOINT_EXP_PER`` (which is a
power). The linear part is solved by non-negative least squares --- no
constant in this model can be negative and an unconstrained solve
happily returns negative setup terms that then misprice small shapes ---
and the two non-linear parameters are profiled over a grid, the linear
solve being redone at each grid point.

The features are NOT re-derived here. ``dispatch._ma_eval_cost_features``
runs the shipped model with each constant replaced by a tracked unit so
that the returned coefficients are, by construction, the model's own;
the fitter only builds the density each row describes and calls it. That
is what keeps the fit and the shipped model from drifting apart.

The objective is *relative* error, not absolute::

    minimise  sum_i ( (pred_i - meas_i) / meas_i )^2

because the routing decision depends only on the ratio of the two
estimates, so a cell costing 0.5 ms deserves the same weight as one
costing 5 s. In absolute-error space the handful of multi-second cells
would set every constant and the crossover region would be fitted by
whatever was left over.

The two routes are fitted separately: centres constants against
``cen_ms`` over the rows where centres was timed, Moebius constants
against ``mob_ms`` over all rows.

WHAT IT REPORTS
---------------
Per route, the geometric mean and geometric standard deviation of
predicted over measured and the worst cell in either direction; and, over
the cells where both arms were timed, the *routing regret* of the cost
model against an oracle that always picks the measured-faster arm ---
total time of the model's picks over total time of the oracle's picks,
plus the count and the worst single-cell slowdown. Both are reported for
the currently shipped constants and for the fitted ones, which is the
comparison that says whether a refit was worth anything.

The density each row describes is rebuilt with an evenly spaced value
set whose span is the *expected range* of the harness's ``K`` uniform
draws over the cell's span, ``span * (K - 1) / (K + 1)``. The cost model
reads the value range (it sets the culling factor and the non-periodic
u-grid window) but the CSV records only the interval the values were
drawn from, so the expectation is the best available reconstruction;
``--spread full`` uses the interval itself instead.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

import mpt
from mpt._tensor.dispatch import (
    _CENTRES_WORKING_SET_SOFT_BUDGET,
    _MA_COST_LINEAR_NAMES,
    _MA_COST_NONLINEAR_NAMES,
    _MA_MOBIUS_SAFETY,
    _MA_MOBIUS_SAFETY_SMALL,
    _estimate_ma_joint_working_set_bytes,
    _ma_cost_constants,
    _ma_eval_cost_features,
)

CSV_HEADER = "rel,per,r,K,nQ,sigma,span,joint,cen_ms,mob_ms"

CENTRES_NAMES = tuple(n for n in _MA_COST_LINEAR_NAMES
                      if n.startswith("CENTRES_"))
MOBIUS_NAMES = tuple(n for n in _MA_COST_LINEAR_NAMES
                     if n.startswith("MOBIUS_"))


# ----------------------------------------------------------------- input


def read_rows(paths, spread_mode):
    """Parse the CSV block(s) and return a list of row dicts."""
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh]
        if any(ln == "BEGIN_CSV" for ln in lines):
            lo = lines.index("BEGIN_CSV") + 1
            hi = lines.index("END_CSV")
            block = lines[lo:hi]
        else:
            block = [ln for ln in lines if ln]
        for ln in block:
            if not ln or ln.startswith("rel,"):
                continue
            f = ln.split(",")
            if len(f) != 10:
                raise ValueError(f"{path}: expected 10 fields, got {ln!r}")
            rows.append(dict(
                rel=bool(int(f[0])), per=bool(int(f[1])),
                r=int(f[2]), K=int(f[3]), nq=int(f[4]),
                sigma=float(f[5]), span=float(f[6]), joint=int(f[7]),
                cen=float(f[8]), mob=float(f[9]), src=path,
            ))
    for row in rows:
        K, span = row["K"], row["span"]
        row["spread"] = (span if spread_mode == "full"
                         else span * (K - 1) / (K + 1))
    if not rows:
        raise SystemExit("no data rows found")
    return rows


def build_density(row):
    """Rebuild the single-attribute density a CSV row describes."""
    p = np.linspace(0.0, row["spread"], row["K"])
    w = np.ones(row["K"])
    return mpt.build_exp_tens(
        p, w, row["sigma"], row["r"], row["rel"], row["per"],
        row["span"] if row["per"] else 0.0, verbose=False,
    )


# ------------------------------------------------------------------- fit


def design(rows, consts):
    """Feature matrices for both routes at the given constant vector."""
    n = len(rows)
    Xc = np.zeros((n, len(CENTRES_NAMES)))
    Xm = np.zeros((n, len(MOBIUS_NAMES)))
    for i, row in enumerate(rows):
        _, cf, _, mf = _ma_eval_cost_features(row["dens"], row["nq"],
                                              consts=consts)
        for j, name in enumerate(CENTRES_NAMES):
            Xc[i, j] = cf.get(name, 0.0)
        for j, name in enumerate(MOBIUS_NAMES):
            Xm[i, j] = mf.get(name, 0.0)
    return Xc, Xm


_CENTRES_DESIGN_CACHE = {}


def centres_design(all_rows, consts):
    """Centres features over ``all_rows``, memoised on the non-linear triple.

    The centres branch is exactly linear in the centres constants once
    ``CENTRES_CULL_C`` and the two per-query exponents are fixed --- that is
    why the profile in :func:`fit_route` does one linear solve per grid point
    rather than iterating --- so its feature matrix depends on the constant
    vector *only* through those three. Cross-validation refits the same grid
    of triples on forty different halves of one row set, so the matrix is
    computed once per triple and the halves index into it. Without this the
    profile is rebuilt forty times over and the run costs hours rather than
    minutes.

    ``all_rows`` must therefore be the *same* list on every call --- the whole
    row set --- with the training subset selected by the mask, not by passing
    a shorter list.
    """
    key = tuple(float(consts[n]) for n in _MA_COST_NONLINEAR_NAMES)
    X = _CENTRES_DESIGN_CACHE.get(key)
    if X is None:
        X = np.zeros((len(all_rows), len(CENTRES_NAMES)))
        for i, row in enumerate(all_rows):
            _, cf, _, _ = _ma_eval_cost_features(row["dens"], row["nq"],
                                                 consts=consts)
            for j, name in enumerate(CENTRES_NAMES):
                X[i, j] = cf.get(name, 0.0)
        _CENTRES_DESIGN_CACHE[key] = X
    return X


def nnls_relative(X, y, names, held, w=None):
    """Non-negative least squares in relative-error space.

    ``held`` maps a constant name to a value it is pinned at; its
    contribution moves to the right-hand side. A column that is
    identically zero over these rows is pinned too: no measurement
    constrains it, and letting the solve "choose" zero for it would
    silently delete a term the grid simply does not exercise (the
    multi-attribute factored-centres overhead, for instance, has no
    single-attribute cell at all).
    """
    from scipy.optimize import nnls

    held = dict(held)
    for j, n in enumerate(names):
        if n not in held and not np.any(X[:, j]):
            held[n] = None            # value filled in by the caller
    free = [j for j, n in enumerate(names) if n not in held]
    offs = np.zeros_like(y)
    for j, n in enumerate(names):
        if n in held and held[n]:
            offs += held[n] * X[:, j]
    if w is None:
        w = 1.0 / y
    A = X[:, free] * w[:, None]
    b = (y - offs) * w
    sol = nnls(A, b)[0] if free else np.zeros(0)
    out = {n: v for n, v in held.items() if v is not None}
    for j, val in zip(free, sol):
        out[names[j]] = float(val)
    return out


def log_rms(pred, meas, mask):
    """Root-mean-square of ``log(predicted / measured)``.

    The objective the fit actually minimises, and the one the routing
    decision cares about: selection compares the two estimates, so what
    matters is the ratio, and a cell over-priced 3x is exactly as wrong
    as one under-priced 3x. Plain relative error is not that measure ---
    it is bounded by 1 below and unbounded above, so it punishes
    over-pricing and shrugs at under-pricing, and a least-squares fit in
    that metric lands systematically low. Which is the failure mode this
    model already had.
    """
    lr = np.log(pred[mask] / meas[mask])
    return float(np.sqrt(np.mean(lr ** 2)))


def score_consts(rows, mask, consts, route):
    """Log-ratio error of the *model* at ``consts``.

    Scoring candidates through the model rather than through the stale
    feature matrix matters: the Moebius node cost is a ``min()`` over
    two strategies, so a solve that drives one strategy's constants to
    zero makes that branch win at zero cost. Its residual on the frozen
    features looks fine; its actual predictions are nonsense. Scoring
    the real model rejects such a solution instead of converging to it.
    """
    cen, mob = predict(rows, consts)
    pred = cen if route == "centres" else mob
    meas = np.array([r["cen" if route == "centres" else "mob"]
                     for r in rows])
    return log_rms(pred, meas, mask)


def fit_route(all_rows, mask, names, consts0, fixed, nonlin_grid,
              nonlin_names, route, passes=4, irls=4, model_score=True):
    """Profile over the non-linear parameters, NNLS the rest at each.

    Non-negative least squares cannot be run in log space directly, so
    the log-ratio objective is approached by iteratively reweighted
    least squares: residuals are scaled by the geometric mean of the
    measured and currently predicted times rather than by the measured
    time alone, which makes the weighting symmetric in the ratio. Two
    loops are wrapped around that: the IRLS reweighting and, because the
    Moebius feature matrix depends on the constants through the ``min()``
    over node strategies, a refresh of the features themselves. Every
    iterate is scored by :func:`score_consts` on the real model and the
    best is kept, so neither loop can wander somewhere worse.
    """
    y_key = "cen" if route == "centres" else "mob"
    rows = [r for r, m in zip(all_rows, mask) if m]
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    y = np.array([r[y_key] for r in rows])
    keep = np.ones(len(rows), dtype=bool)
    best = None
    for combo in nonlin_grid:
        consts = dict(consts0)
        consts.update(dict(zip(nonlin_names, combo)))
        w = None
        for _ in range(passes):
            if route == "centres":
                X = centres_design(all_rows, consts)[idx]
            else:
                X = design(rows, consts)[1]
            for _ in range(irls):
                beta = nnls_relative(
                    X, y, names,
                    {n: consts[n] for n in fixed if n in names}, w=w)
                for n in names:
                    beta.setdefault(n, consts[n])
                cand = dict(consts)
                cand.update(beta)
                pred = np.maximum(X @ np.array([cand[n] for n in names]),
                                  1e-12)
                # The centres model is exactly linear once the non-linear
                # parameters are fixed, so its features are the model and
                # X @ beta is its prediction. The Moebius one is not: its
                # min() over node strategies can move under the solve, so
                # score it by re-running the model.
                score = (score_consts(rows, keep, cand, route) if model_score
                         else log_rms(pred, y, keep))
                if best is None or score < best[0]:
                    best = (score, {n: cand[n] for n in names},
                            dict(zip(nonlin_names, combo)))
                w = 1.0 / np.sqrt(y * pred)
            if cand == consts:
                break
            consts = cand
    return best


# --------------------------------------------------------------- reports


def predict(rows, consts):
    cen = np.zeros(len(rows))
    mob = np.zeros(len(rows))
    for i, row in enumerate(rows):
        c, _, m, _ = _ma_eval_cost_features(row["dens"], row["nq"],
                                            consts=consts)
        cen[i], mob[i] = c, m
    return cen, mob


def quality(pred, meas, mask):
    """Geometric mean / spread / worst of predicted over measured."""
    ratio = pred[mask] / meas[mask]
    lr = np.log(ratio)
    worst = float(np.max(np.maximum(ratio, 1.0 / ratio)))
    return (float(np.exp(np.mean(lr))), float(np.exp(np.std(lr))), worst,
            int(mask.sum()))


def routing(rows, cen_p, mob_p, both):
    """Regret of the cost model's picks against the measured oracle."""
    tot_model = tot_oracle = 0.0
    misses = 0
    worst = 1.0
    worst_row = None
    for i, row in enumerate(rows):
        if not both[i]:
            continue
        ws = _estimate_ma_joint_working_set_bytes(
            [row["r"]], [row["K"]], [row["rel"]])
        safety = (_MA_MOBIUS_SAFETY if ws > _CENTRES_WORKING_SET_SOFT_BUDGET
                  else _MA_MOBIUS_SAFETY_SMALL)
        pick = "mobius" if mob_p[i] < cen_p[i] * safety else "centres"
        t_model = row["mob"] if pick == "mobius" else row["cen"]
        t_oracle = min(row["cen"], row["mob"])
        tot_model += t_model
        tot_oracle += t_oracle
        if t_model > t_oracle:
            misses += 1
            if t_model / t_oracle > worst:
                worst = t_model / t_oracle
                worst_row = row
    return tot_model / tot_oracle, misses, worst, worst_row


NONLIN = ("CENTRES_CULL_C", "CENTRES_QUERY_JOINT_EXP_PER",
          "CENTRES_QUERY_JOINT_EXP_REL_PER")

#: The forms compared under cross-validation, as
#: ``(name, cull_grid, abs_exp_grid, rel_exp_grid)`` with ``None`` meaning
#: "profile over this run's default grid". The three non-linear parameters
#: are the only shape freedom the model has; everything else is linear and is
#: solved, not chosen. Pinning one of them at a round value is therefore the
#: only simplification available short of deleting a term, and the pinned
#: candidates are the values the two languages might share: exponent 1, which
#: is what the per-query cost would be if the query loop were linear in the
#: joint tuple count and is what MATLAB ships, and the fitted Python values
#: 1.15 and 1.3 bracketing its 1.2.
CV_FORMS = (
    ("exps fitted, cull profiled",   None,    None,   None),
    ("abs exp 1.0, rel fitted",      None,    [1.0],  None),
    ("abs fitted, rel exp 1.0",      None,    None,   [1.0]),
    ("abs fitted, rel exp 1.15",     None,    None,   [1.15]),
    ("abs fitted, rel exp 1.3",      None,    None,   [1.3]),
    ("both exps 1.0, cull profiled", None,    [1.0],  [1.0]),
    ("exps fitted, cull 14.32",      [14.32], None,   None),
    ("both exps 1.0, cull 14.32",    [14.32], [1.0],  [1.0]),
    ("abs 1.0, rel 1.15, cull 14.32", [14.32], [1.0], [1.15]),
    ("abs 1.0, rel 1.2, cull 14.32",  [14.32], [1.0], [1.2]),
    ("abs 1.0, rel 1.3, cull 14.32",  [14.32], [1.0], [1.3]),
)


def fit_centres(all_rows, mask, base, fixed, cull_grid, exp_grid,
                rel_exp_grid):
    grid = [(c, e, er) for c in cull_grid for e in exp_grid
            for er in rel_exp_grid]
    rms, beta, nl = fit_route(all_rows, mask, CENTRES_NAMES, base, fixed,
                              grid, NONLIN, "centres", passes=1,
                              model_score=False)
    return rms, beta, nl


def fit_mobius(all_rows, mask, base, fixed):
    return fit_route(all_rows, mask, MOBIUS_NAMES, base, fixed, [()], (),
                     "mobius", passes=5)


def score_held_out(rows, sel, consts):
    """``(regret, centres log-ratio rms, Moebius log-ratio rms)`` on ``sel``.

    Regret is the number to read: routing compares the two estimates, so a
    form that mis-prices both arms by one factor routes as well as a form
    that prices them exactly. The two log-ratio errors are reported beside it
    because a form can buy regret with a wild fit that happens to keep the
    ratio, and that is worth seeing.
    """
    sub = [r for r, m in zip(rows, sel) if m]
    if not sub:
        return float("nan"), float("nan"), float("nan")
    cen, mob = predict(sub, consts)
    cen_ok = np.array([r["cen"] > 0 for r in sub])
    mob_ok = np.array([r["mob"] > 0 for r in sub])
    both = cen_ok & mob_ok
    cen_m = np.array([r["cen"] for r in sub])
    mob_m = np.array([r["mob"] for r in sub])
    reg = routing(sub, cen, mob, both)[0] if both.any() else float("nan")
    return (reg,
            log_rms(cen, cen_m, cen_ok) if cen_ok.any() else float("nan"),
            log_rms(mob, mob_m, mob_ok) if mob_ok.any() else float("nan"))


def cv_forms(rows, cen_ok, mob_ok, base, fixed, forms, cull_grid, exp_grid,
             rel_exp_grid, repeats, seed=20260904):
    """Random-half cross-validation of several forms on one row set.

    Each repeat splits the cells in half, fits on one half and scores on the
    other. The Moebius constants are fitted once per repeat and shared by
    every form: no form here touches a Moebius parameter, and the Moebius
    features do not depend on the centres constants, so refitting them per
    form would spend the time to reach the same numbers.
    """
    rng = np.random.default_rng(seed)
    n = len(rows)
    acc = {name: [] for name, *_ in forms}
    for _ in range(repeats):
        perm = rng.permutation(n)
        train = np.zeros(n, dtype=bool)
        train[perm[:n // 2]] = True
        test = ~train
        _, beta_m, _ = fit_mobius(rows, mob_ok & train, base, fixed)
        for name, cull, eg, reg_ in forms:
            _, beta_c, nl_c = fit_centres(
                rows, cen_ok & train, base, fixed,
                cull if cull is not None else cull_grid,
                eg if eg is not None else exp_grid,
                reg_ if reg_ is not None else rel_exp_grid)
            C = dict(base)
            C.update(nl_c)
            C.update(beta_c)
            C.update(beta_m)
            acc[name].append(score_held_out(rows, test, C))
    return {k: (np.array(v).mean(axis=0), np.array(v).std(axis=0))
            for k, v in acc.items()}


def label(row):
    return (f"{'rel' if row['rel'] else 'abs'}-"
            f"{'per' if row['per'] else 'nonper'} r={row['r']} "
            f"K={row['K']} nQ={row['nq']} sigma={row['sigma']:g} "
            f"span={row['span']:g}")


# ---------------------------------------------------------------- output


PY_ORDER = [
    ("CENTRES_SETUP_MS", "{:.4g}"),
    ("CENTRES_CALL_PER_JOINT_MS", "{:.4g}"),
    ("CENTRES_QUERY_BASE_MS", "{:.4g}"),
    ("CENTRES_QUERY_PER_JOINT_MS", "{:.4g}"),
    ("CENTRES_QUERY_BASE_PER_MS", "{:.4g}"),
    ("CENTRES_QUERY_PER_JOINT_PER_MS", "{:.4g}"),
    ("CENTRES_QUERY_JOINT_EXP_PER", "{:.4g}"),
    ("CENTRES_QUERY_PER_JOINT_REL_PER_MS", "{:.4g}"),
    ("CENTRES_QUERY_JOINT_EXP_REL_PER", "{:.4g}"),
    ("CENTRES_FACTORED_QUERY_BASE_MS", "{:.4g}"),
    ("CENTRES_CULL_C", "{:.4g}"),
    ("MOBIUS_SETUP_MS", "{:.4g}"),
    ("MOBIUS_SETUP_PER_BELL_MS", "{:.4g}"),
    ("MOBIUS_QUERY_PER_OP_MS", "{:.4g}"),
    ("MOBIUS_REL_NODE_DIRECT_BASE_MS", "{:.4g}"),
    ("MOBIUS_REL_NODE_DIRECT_PER_OP_MS", "{:.4g}"),
    ("MOBIUS_REL_NODE_DIRECT_PER_OP_PER_MS", "{:.4g}"),
    ("MOBIUS_REL_NODE_FACTORED_PER_BELL_MS", "{:.4g}"),
    ("MOBIUS_REL_TABULATION_PER_NODE_MS", "{:.4g}"),
    ("MOBIUS_SPECTRAL_PER_MODE_MS_R2", "{:.4g}"),
    ("MOBIUS_SPECTRAL_PER_MODE_MS_R3", "{:.4g}"),
    ("MOBIUS_SPECTRAL_PER_MODE_MS_R4", "{:.4g}"),
    ("MOBIUS_SPECTRAL_PERIODIC_K_MS_R2", "{:.4g}"),
    ("MOBIUS_SPECTRAL_PERIODIC_K_MS_R3", "{:.4g}"),
    ("MOBIUS_SPECTRAL_PERIODIC_K_MS_R4", "{:.4g}"),
]


def emit_python(C):
    print("# ---- Python: paste into mpt/_tensor/dispatch.py ----")
    for name, fmt in PY_ORDER:
        print(f"_MA_COST_{name} = " + fmt.format(C[name]))


def emit_matlab(C):
    print("% ---- MATLAB: paste into +internal/maEvalCostsMs.m ----")
    for name, fmt in PY_ORDER:
        if name.startswith("MOBIUS_SPECTRAL_"):
            continue
        print(f"    MA_COST_{name} = " + fmt.format(C[name]) + ";")
    print("    % inside the relative branch:")
    print("    FOUR_PER_MODE = [{:.4g}, {:.4g}, {:.4g}];".format(
        C["MOBIUS_SPECTRAL_PER_MODE_MS_R2"],
        C["MOBIUS_SPECTRAL_PER_MODE_MS_R3"],
        C["MOBIUS_SPECTRAL_PER_MODE_MS_R4"]))
    print("    FOUR_PERIODIC_K_MS = [{:.4g}, {:.4g}, {:.4g}];".format(
        C["MOBIUS_SPECTRAL_PERIODIC_K_MS_R2"],
        C["MOBIUS_SPECTRAL_PERIODIC_K_MS_R3"],
        C["MOBIUS_SPECTRAL_PERIODIC_K_MS_R4"]))


# ------------------------------------------------------------------ main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--lang", choices=("python", "matlab", "both"),
                    default="both")
    ap.add_argument("--spread", choices=("expected", "full"),
                    default="expected")
    ap.add_argument("--cull-grid", default="")
    ap.add_argument("--exp-grid", default="",
                    help="candidate absolute-periodic per-query exponents")
    ap.add_argument("--exp-rel-grid", default="",
                    help="candidate relative-periodic per-query exponents")
    ap.add_argument("--no-exp", action="store_true")
    ap.add_argument("--fix", action="append", default=[])
    ap.add_argument("--list-worst", type=int, default=8)
    ap.add_argument("--cv", type=int, nargs="?", const=40, default=0,
                    metavar="N",
                    help="random-half cross-validation, N repeats "
                         "(default 40 when given without a value)")
    ap.add_argument("--compare-forms", action="store_true",
                    help="cross-validate every form in CV_FORMS and stop")
    ap.add_argument("--score-on", metavar="FILE", action="append", default=[],
                    help="fit on the CSVs above, score on this one "
                         "(repeatable; cross-machine transfer)")
    args = ap.parse_args(argv)

    fixed = {}
    for item in args.fix:
        name, _, val = item.partition("=")
        fixed[name.strip()] = float(val)

    rows = read_rows(args.csv, args.spread)
    for row in rows:
        row["dens"] = build_density(row)
    cen_ok = np.array([r["cen"] > 0 for r in rows])
    mob_ok = np.array([r["mob"] > 0 for r in rows])
    both = cen_ok & mob_ok
    print(f"{len(rows)} cells; centres timed on {int(cen_ok.sum())}, "
          f"Moebius on {int(mob_ok.sum())}, both on {int(both.sum())}.")

    cull_grid = ([float(v) for v in args.cull_grid.split(",")]
                 if args.cull_grid else
                 list(np.round(np.exp(np.linspace(np.log(4.0),
                                                  np.log(120.0), 25)), 3)))
    if args.no_exp:
        exp_grid = rel_exp_grid = [1.0]
    else:
        exp_grid = ([float(v) for v in args.exp_grid.split(",")]
                    if args.exp_grid
                    else list(np.round(np.arange(0.85, 1.16, 0.05), 4)))
        rel_exp_grid = ([float(v) for v in args.exp_rel_grid.split(",")]
                        if args.exp_rel_grid
                        else list(np.round(np.arange(1.0, 1.46, 0.05), 4)))

    shipped = _ma_cost_constants()
    cen0, mob0 = predict(rows, shipped)
    base = dict(shipped)
    base.update(fixed)

    if args.compare_forms or args.cv:
        reps = args.cv or 40
        forms = (CV_FORMS if args.compare_forms
                 else (("as invoked", cull_grid, exp_grid, rel_exp_grid),))
        print(f"\n--- random-half cross-validation, {reps} repeats, "
              f"{len(rows)} cells ---")
        print(f"{'form':<30s} {'held-out regret':>21s} "
              f"{'centres log rms':>21s} {'Moebius log rms':>21s}")
        out = cv_forms(rows, cen_ok, mob_ok, base, fixed, forms, cull_grid,
                       exp_grid, rel_exp_grid, reps)
        for name, *_ in forms:
            mean, sd = out[name]
            print(f"{name:<30s} "
                  f"{mean[0]:>12.4f} +- {sd[0]:<6.4f} "
                  f"{mean[1]:>12.4f} +- {sd[1]:<6.4f} "
                  f"{mean[2]:>12.4f} +- {sd[2]:<6.4f}")
        if args.compare_forms:
            return 0

    if args.score_on:
        test = read_rows(args.score_on, args.spread)
        for row in test:
            row["dens"] = build_density(row)
        print(f"\n--- transfer: fit on {len(rows)} cells, score on "
              f"{len(test)} cells of {', '.join(args.score_on)} ---")
        print(f"{'form':<30s} {'regret':>9s} {'centres log rms':>17s} "
              f"{'Moebius log rms':>17s}")
        all_test = np.ones(len(test), dtype=bool)
        _, beta_m, _ = fit_mobius(rows, mob_ok, base, fixed)
        for name, cull, eg, reg_ in CV_FORMS:
            _, beta_c, nl_c = fit_centres(
                rows, cen_ok, base, fixed,
                cull if cull is not None else cull_grid,
                eg if eg is not None else exp_grid,
                reg_ if reg_ is not None else rel_exp_grid)
            C = dict(base)
            C.update(nl_c)
            C.update(beta_c)
            C.update(beta_m)
            r, lc, lm = score_held_out(test, all_test, C)
            print(f"{name:<30s} {r:>9.4f} {lc:>17.4f} {lm:>17.4f}")
        r, lc, lm = score_held_out(test, all_test, shipped)
        print(f"{'shipped constants':<30s} {r:>9.4f} {lc:>17.4f} {lm:>17.4f}")
        return 0

    # Centres: profile (cull C, periodic exponent). The centres branch
    # structure does not depend on the centres constants, so one linear
    # solve per grid point is exact.
    grid = [(c, e, er) for c in cull_grid for e in exp_grid
            for er in rel_exp_grid]
    rms_c, beta_c, nl_c = fit_route(
        rows, cen_ok, CENTRES_NAMES, base, fixed, grid,
        ("CENTRES_CULL_C", "CENTRES_QUERY_JOINT_EXP_PER",
         "CENTRES_QUERY_JOINT_EXP_REL_PER"), "centres", passes=1,
        model_score=False)
    # Moebius: no non-linear parameter of its own, but the min() over
    # node strategies makes its features constant-dependent, so iterate.
    rms_m, beta_m, _ = fit_route(
        rows, mob_ok, MOBIUS_NAMES, base, fixed, [()], (), "mobius",
        passes=5)

    fitted = dict(base)
    fitted.update(nl_c)
    fitted.update(beta_c)
    fitted.update(beta_m)
    cen1, mob1 = predict(rows, fitted)

    print(f"\nprofiled: CENTRES_CULL_C = {nl_c['CENTRES_CULL_C']:g}, "
          f"EXP_PER = {nl_c['CENTRES_QUERY_JOINT_EXP_PER']:g}, "
          f"EXP_REL_PER = {nl_c['CENTRES_QUERY_JOINT_EXP_REL_PER']:g} "
          f"(rms log-ratio: centres {rms_c:.4f}, Moebius {rms_m:.4f})")

    print("\n--- fit quality: predicted / measured ---")
    hdr = f"{'route':<10s} {'constants':<10s} {'geo mean':>9s} " \
          f"{'geo sd':>8s} {'worst':>8s} {'cells':>6s}"
    print(hdr)
    for route, pred0, pred1, meas_key, mask in (
            ("centres", cen0, cen1, "cen", cen_ok),
            ("mobius", mob0, mob1, "mob", mob_ok)):
        meas = np.array([r[meas_key] for r in rows])
        for tag, pred in (("shipped", pred0), ("fitted", pred1)):
            g, s, w, n = quality(pred, meas, mask)
            print(f"{route:<10s} {tag:<10s} {g:>9.3f} {s:>8.3f} "
                  f"{w:>8.1f} {n:>6d}")

    print("\n--- routing regret vs oracle (cells with both arms timed) ---")
    print(f"{'constants':<10s} {'total/oracle':>13s} {'misroutes':>10s} "
          f"{'worst cell':>11s}  worst cell detail")
    for tag, cp, mp in (("shipped", cen0, mob0), ("fitted", cen1, mob1)):
        reg, miss, worst, wrow = routing(rows, cp, mp, both)
        detail = label(wrow) if wrow is not None else "-"
        print(f"{tag:<10s} {reg:>13.3f} {miss:>10d} {worst:>11.2f}  {detail}")

    if args.list_worst:
        print(f"\n--- worst {args.list_worst} centres cells "
              f"(fitted pred/actual) ---")
        meas = np.array([r["cen"] for r in rows])
        idx = np.where(cen_ok)[0]
        ratio = cen1[idx] / meas[idx]
        order = idx[np.argsort(-np.maximum(ratio, 1.0 / ratio))]
        for i in order[:args.list_worst]:
            print(f"  {cen1[i] / rows[i]['cen']:6.3f}  "
                  f"pred {cen1[i]:9.3f} meas {rows[i]['cen']:9.3f}  "
                  f"{label(rows[i])}")
        print(f"\n--- worst {args.list_worst} Moebius cells "
              f"(fitted pred/actual) ---")
        meas = np.array([r["mob"] for r in rows])
        idx = np.where(mob_ok)[0]
        ratio = mob1[idx] / meas[idx]
        order = idx[np.argsort(-np.maximum(ratio, 1.0 / ratio))]
        for i in order[:args.list_worst]:
            print(f"  {mob1[i] / rows[i]['mob']:6.3f}  "
                  f"pred {mob1[i]:9.3f} meas {rows[i]['mob']:9.3f}  "
                  f"{label(rows[i])}")

    print()
    if args.lang in ("python", "both"):
        emit_python(fitted)
        print()
    if args.lang in ("matlab", "both"):
        emit_matlab(fitted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
