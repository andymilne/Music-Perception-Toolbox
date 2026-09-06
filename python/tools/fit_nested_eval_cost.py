"""Fit the nested-evaluation cost row from bench_nested_eval output.

    PYTHONPATH=. python3 tools/fit_nested_eval_cost.py python.csv [matlab.csv ...]

Each file is the console output of ``tools/bench_nested_eval.py`` or
``matlab/tools/benchNestedEval.m`` (the ``BEGIN_CSV`` / ``END_CSV``
lines and any preamble are skipped). For every file the script fits the
two closed-form laws that ``mpt._tensor.dispatch._nested_eval_costs_ms``
and MATLAB ``internal.nestedEvalCostsMs`` price a nested attribute
with, in log space (so every cell counts by its ratio, not its size):

    centres_ms = C0 + C1 * T + n_q * C2 * T**gamma * d**delta
    mobius_ms  = M0 + N * (M1 + n_q * M2 * (K * s)**alpha * n_u)

with ``T = m_perm * N`` the attribute's tuple-centre count, ``d`` its
reduced dimension, ``s = prod(r_levels)`` its leaf slots, ``K`` its
values per event, and ``n_u`` the translation-grid node count (1 for an
absolute attribute). It then reports the fit residuals, the routing
regret against the measured oracle in-sample and on ten random
half-splits by shape, and the constants in the form the two modules
carry them. Cells whose centres route hit the 4e7 guard (NaN) are used
for the Möbius fit only.

Why these forms: over the September 2026 grid (3096 cells per language)
the Möbius route's per-event per-query cost is a near power law in
``K * s`` (log residual 0.18--0.19 in both languages), independent of the
tuple count; the centres route is the familiar setup + per-tuple +
per-query structure, with the dimension entering the per-query slope
(a per-query constant independent of the tuple count fitted to zero in
both languages and is not carried).
Nothing finer is warranted: the routing regret is what the row is for,
and it sits at 1.03--1.05 of the oracle with these forms.
"""
from __future__ import annotations

import sys

import numpy as np

try:
    import pandas as pd
    from scipy.optimize import least_squares
except ImportError as exc:  # pragma: no cover
    sys.exit(f"fit_nested_eval_cost needs pandas and scipy: {exc}")


def load(path):
    rows = [ln for ln in open(path)
            if ln[:1].isdigit() or ln.startswith("groups")]
    from io import StringIO
    d = pd.read_csv(StringIO("".join(rows)))
    rl = d.r_levels.astype(str).str.split("x")
    d["s"] = rl.str[0].astype(int) * rl.str[1].astype(int)
    d["T"] = d.m_perm * d.n
    d["nu"] = np.where(d.rel_unit >= 0, d.n_u, 1.0)
    return d


def mobius_ms(p, d):
    M0, M1, M2 = np.exp(p[:3])
    return M0 + d.n * (M1 + d.n_q * M2 * (d.k * d.s) ** p[3] * d.nu)


def centres_ms(p, c):
    C0, C1, C2 = np.exp(p[:3])
    return C0 + C1 * c["T"] + c.n_q * C2 * c["T"] ** p[3] * c.d ** p[4]


def timed_both(d, exclude_pathological=False):
    c = d[d.centres_ms.notna()]
    if exclude_pathological:
        # Python before the bucket-cull guard of September 2026 paid
        # 3**d neighbour lookups per query however few tuples there were;
        # those cells no longer describe the route (the guard makes them
        # dense truncated sums, priced by the tuple term like the rest).
        c = c[~((c.per == 0) & (3.0 ** c.d >= c["T"]))]
    return c


def fit(d, exclude_pathological=False):
    y = np.log(d.mobius_ms.values)
    rm = least_squares(lambda p: np.log(mobius_ms(p, d)) - y,
                       [0.0, -2.0, -10.0, 1.0])
    c = timed_both(d, exclude_pathological)
    yc = np.log(c.centres_ms.values)
    rc = least_squares(lambda p: np.log(centres_ms(p, c)) - yc,
                       [-1.0, -5.0, -8.0, 1.0, 0.5])
    return rm.x, rc.x, float(np.std(rm.fun)), float(np.std(rc.fun))


def regret(pm, pc, c):
    choose_m = mobius_ms(pm, c) < centres_ms(pc, c)
    cost = np.where(choose_m, c.mobius_ms, c.centres_ms)
    return cost / np.minimum(c.mobius_ms, c.centres_ms)


def report(path, exclude_pathological):
    d = load(path)
    c = timed_both(d, exclude_pathological)
    pm, pc, sm, sc = fit(d, exclude_pathological)
    print(f"== {path}: {len(d)} cells, {len(c)} with both routes timed")
    print(f"   Möbius fit log-residual sd {sm:.3f}; centres {sc:.3f}")
    r = regret(pm, pc, c)
    print(f"   routing regret in-sample: geomean {np.exp(np.log(r).mean()):.4f}, "
          f"max {r.max():.2f}, {int((r > 1.3).sum())} of {len(c)} beyond 1.3x")
    base = c.centres_ms / np.minimum(c.mobius_ms, c.centres_ms)
    print(f"   always-centres (the previous 'auto'): geomean "
          f"{np.exp(np.log(base).mean()):.3f}, max {base.max():.1f}")
    rng = np.random.default_rng(0)
    keys = d.groupby(["groups", "group_size", "r_levels", "sym",
                      "rel_unit", "per"]).ngroup().values
    ckeys = keys[c.index.map(lambda i: d.index.get_loc(i)).values]
    regs = []
    for _ in range(10):
        u = np.unique(keys)
        rng.shuffle(u)
        half = set(u[: len(u) // 2])
        tr = d[[k in half for k in keys]]
        te = c[[k not in half for k in ckeys]]
        pm2, pc2, _, _ = fit(tr, exclude_pathological)
        rr = regret(pm2, pc2, te)
        regs.append((np.exp(np.log(rr).mean()), rr.max(), (rr > 1.3).mean()))
    g, mx, fr = np.array(regs).mean(0)
    print(f"   held-out (10 half-splits by shape): geomean {g:.4f}, "
          f"max {mx:.2f}, fraction beyond 1.3x {fr:.3f}")
    nn = d[d.centres_ms.isna()]
    if len(nn):
        frac = float((mobius_ms(pm, nn) < centres_ms(pc, nn)).mean())
        print(f"   cells whose centres route hit the guard: "
              f"{frac:.3f} routed to Möbius")
    M0, M1, M2 = np.exp(pm[:3])
    C0, C1, C2 = np.exp(pc[:3])
    print("   constants:")
    print(f"     MOBIUS_SETUP_MS      = {M0:.6g}")
    print(f"     MOBIUS_PER_EVENT_MS  = {M1:.6g}")
    print(f"     MOBIUS_PER_OP_MS     = {M2:.6g}")
    print(f"     MOBIUS_OP_EXP        = {pm[3]:.4f}")
    print(f"     CENTRES_SETUP_MS     = {C0:.6g}")
    print(f"     CENTRES_PER_TUPLE_MS = {C1:.6g}")
    print(f"     CENTRES_PER_QUERY_MS = {C2:.6g}")
    print(f"     CENTRES_TUPLE_EXP    = {pc[3]:.4f}")
    print(f"     CENTRES_DIM_EXP      = {pc[4]:.4f}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    excl = "--exclude-pathological" in sys.argv
    if not args:
        sys.exit(__doc__)
    for path in args:
        report(path, excl)
