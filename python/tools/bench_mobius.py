"""bench_mobius.py

Wall-time comparison of the three inner-product routes in the Music
Perception Toolbox -- direct enumeration, Bulger's within-r-ad
decomposition, and the Moebius-orbit decomposition -- across tuple size
r, multiset size K, and the four modes.

The point of the benchmark is the RELATIVE timings and their scaling,
not the absolute ones: absolute times establish only that the regimes
reported are feasible. Accordingly the script reports, per cell, the
ratio of each method to the fastest, and fits the exponent of K for
each method so that the measured scaling can be checked against the
predicted degrees (2r for direct and Bulger, 2 for orbit).

Protocol notes (these matter for comparability):

* Correctness is checked BEFORE any timing, with truncation disabled.
  Bulger's method is the reference: it is a restriction of a sum of
  non-negative terms, so it does not suffer the cancellation the
  alternating Moebius sums can. Each method's relative deviation from it
  is recorded per cell (column `rel_dev`) and reported, so the
  cancellation regimes -- K close to r, and relative-periodic at large
  sigma/P -- show up as loss of agreement rather than being hidden by a
  pass/fail threshold. Deviations beyond `TOL` are flagged but the cell
  is still timed.
* Timing uses truncation_sigmas=inf so that the three methods do the
  same arithmetic. A separate pass (--truncation) re-times the routed
  default (6 sigma) to quantify what truncation buys.
* Densities are built once per cell, outside the timed region: the timed
  unit is the similarity call alone, so construction is charged to no
  method.
* Repetitions are auto-scaled so each timed unit takes ~TARGET_MS,
  after a warm-up call that absorbs first-call costs (orbit-table load,
  einsum path setup, BLAS thread spin-up).
* Each method is forced explicitly. The aim is the real-world cost and
  feasibility of the three routes in each regime, so the toolbox's
  automatic dispatch is deliberately bypassed: timing 'auto' would
  measure the routing policy, not the methods it routes between.

Usage:
    python3 bench_mobius.py                    # standard sweep
    python3 bench_mobius.py --quick            # small sweep, for a smoke test
    python3 bench_mobius.py --truncation       # add the 6-sigma pass
    python3 bench_mobius.py --out results.csv

Writes a tidy CSV (one row per cell x method) and prints a summary.
"""
from __future__ import annotations

import argparse
import csv
import platform
import sys
import time

import numpy as np

import mpt

# Silence one-time hints, and switch off the post-hoc guards: with guards
# on, a route that diverts pays for both routes, so the measured cost
# would not be the cost of the route being forced.
try:
    mpt.set_default(show_hints=False)
except Exception:
    pass
try:
    mpt.set_default(post_hoc_guards=False)
except Exception:
    pass                      # older builds lack the knob; forcing still holds

# --- sweep definition -------------------------------------------------------

# Defaults chosen so a full run finishes in a few minutes. The orbit
# method's advantage is already unambiguous by r = 4, K = 34; larger
# cells cost minutes each in the relative modes and add nothing but
# confirmation. Widen with --rs and --ks if a fuller grid is wanted.
RS = (2, 3, 4)
KS = (6, 12, 20, 34)
MODES = (                      # (is_rel, is_per, label)
    (False, False, 'absolute non-periodic'),
    (False, True,  'absolute periodic'),
    (True,  False, 'relative non-periodic'),
    (True,  True,  'relative periodic'),
)
METHODS = ('direct', 'bulger', 'mobius')   # forced; the dispatcher is
                                           # deliberately bypassed (see below)

SIGMA = 30.0          # cents; well inside the regime where truncation matters
PERIOD = 1200.0
TOL = 1e-9            # agreement tolerance between methods (untruncated)
TARGET_MS = 150.0     # per timed unit
SEED = 20260822

# --- work limits ------------------------------------------------------------
# Both enumerating methods cost roughly one unit per tuple PAIR, and the pair
# count explodes: at r = 5, K = 48 it is 3.5e14, which no budget accommodates.
# Cells are therefore skipped when their predicted cost exceeds CELL_BUDGET_S.
# The estimate uses a measured rate (about 1e-8 s per pair on the machine this
# was calibrated on, re-measured at startup) times the pair count:
#
#     direct:  (r! C(K,r))^2          both sides enumerated
#     bulger:  C(K,r) * r! C(K,r)     one side restricted (Eq. S8)
#     mobius:  |Omega_r| contractions, effectively independent of K
#
# Raise CELL_BUDGET_S to reach further into the enumerating methods' range;
# the orbit method is unaffected either way.
CELL_BUDGET_S = 2.0       # per timed cell, per method, predicted
PAIR_RATE_S = 1.0e-8      # seconds per tuple pair; re-calibrated at startup
OMEGA = {2: 4, 3: 10, 4: 33, 5: 92, 6: 306, 7: 948, 8: 3210}


class LCG:
    """Numerical Recipes ranqd1. Used in place of NumPy's generator so the
    MATLAB twin can draw byte-identical inputs and the two languages
    benchmark the same problem, not merely the same problem sizes."""

    def __init__(self, seed):
        self.state = int(seed)

    def uniform01(self, n):
        out = np.empty(n)
        s = self.state
        for i in range(n):
            s = (1664525 * s + 1013904223) % (1 << 32)
            out[i] = s / float(1 << 32)
        self.state = s
        return out


def draw(K, rng):
    u = rng.uniform01(2 * K)
    p = np.sort(u[:K] * 3 * PERIOD)
    w = 0.4 + 0.6 * u[K:]
    return p, w


def make_pair(K, rng):
    """Two weighted multisets of K atoms, in cents over three octaves."""
    p, wp = draw(K, rng)
    q, wq = draw(K, rng)
    return (p, wp), (q, wq)


def build(pair, r, is_rel, is_per):
    (p, wp), (q, wq) = pair
    A = mpt.build_exp_tens(p, wp, SIGMA, r, is_rel, is_per, PERIOD, verbose=False)
    B = mpt.build_exp_tens(q, wq, SIGMA, r, is_rel, is_per, PERIOD, verbose=False)
    return A, B


def call(A, B, method, trunc):
    kw = dict(method=method, verbose=False)
    if trunc is not None:
        kw['truncation_sigmas'] = trunc
    return float(mpt.cos_sim_exp_tens(A, B, **kw))


def timed(fn, target_ms=TARGET_MS):
    """Median of repeated batches, each batch auto-sized to ~target_ms.

    Returns (seconds_per_call, n_reps). One warm-up call first, so that
    orbit-table loading and einsum path setup are not charged to the
    measurement.
    """
    fn()                                    # warm-up
    n, elapsed = 1, 0.0
    while True:                             # size the batch
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        elapsed = time.perf_counter() - t0
        if elapsed * 1e3 >= target_ms or n >= 1 << 20:
            break
        n = max(2 * n, int(n * target_ms / max(elapsed * 1e3, 1e-6)))
    per_call = []
    for _ in range(5):                      # five batches, take the median
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        per_call.append((time.perf_counter() - t0) / n)
    return float(np.median(per_call)), n


def pair_count(method, r, K):
    """Tuple pairs a method must visit; None when K-independent."""
    from math import comb, factorial
    if K < r:
        return 0
    ordered = factorial(r) * comb(K, r)
    if method == 'direct':
        return ordered * ordered
    if method == 'bulger':
        return comb(K, r) * ordered
    return None                       # mobius: |Omega_r| contractions


#: Measured per-call cost of the orbit method, in seconds, by mode and r.
#: The relative modes carry a one-dimensional quadrature over the common
#: shift, which the absolute modes do not, and it dominates: at r = 5 the
#: relative non-periodic call is ~30x the absolute one. These are order-of-
#: magnitude figures used only for budgeting.
ORBIT_S = {
    (False, False): {2: 6e-4, 3: 1.3e-3, 4: 3.8e-3, 5: 1.2e-2, 6: 4e-2},
    (False, True):  {2: 6e-4, 3: 1.2e-3, 4: 3.6e-3, 5: 1.2e-2, 6: 4e-2},
    (True,  False): {2: 1.3e-3, 3: 1.6e-2, 4: 8.5e-2, 5: 3.4e-1, 6: 1.4e0},
    (True,  True):  {2: 4e-4, 3: 9e-4, 4: 2.5e-2, 5: 9.1e-2, 6: 3.6e-1},
}


def predict_s(method, r, K, rate, is_rel=False, is_per=False):
    """Predicted seconds for ONE call."""
    n = pair_count(method, r, K)
    if n is None:
        return ORBIT_S.get((is_rel, is_per), ORBIT_S[(False, False)]).get(r, 1.0)
    # Measured multipliers on the per-pair rate. Periodic wrapping is the
    # large one: at r = 3, K = 20 the periodic Bulger call is ~18x the
    # non-periodic, and at K = 34 nearer 40x, the wrap being applied per
    # coordinate of every tuple pair. The relative modes add a quadrature
    # over the common shift. Both are deliberate over-estimates, so the
    # budget errs toward skipping a cell rather than running for minutes.
    scale = 1.0
    if is_per:
        scale *= 40.0
    if is_rel:
        scale *= 4.0
    return n * rate * scale


def calibrate_rate():
    """Measure seconds per tuple pair on this machine, on a small safe cell."""
    rng = LCG(SEED)
    p, w = draw(12, rng)
    q, wq = draw(12, rng)
    A = mpt.build_exp_tens(p, w, SIGMA, 3, False, False, PERIOD, verbose=False)
    B = mpt.build_exp_tens(q, wq, SIGMA, 3, False, False, PERIOD, verbose=False)
    fn = lambda: mpt.cos_sim_exp_tens(A, B, method='bulger',
                                      truncation_sigmas=float('inf'), verbose=False)
    t, _ = timed(fn, 60.0)
    return t / pair_count('bulger', 3, 12)


def fit_exponent(Ks, ts, tail=3):
    """Slope of log t against log K over the largest `tail` values of K.

    Fitting the whole sweep understates the exponent badly: at small K
    the per-call overhead (argument handling, object access, BLAS entry)
    dominates the arithmetic, so the curve is flat there for every
    method. The asymptotic regime is the large-K tail, which is what the
    predicted degrees describe.
    """
    Ks, ts = np.asarray(Ks, float), np.asarray(ts, float)
    ok = (Ks > 0) & (ts > 0)
    Ks, ts = Ks[ok], ts[ok]
    if Ks.size < 3:
        return float('nan')
    order = np.argsort(Ks)
    Ks, ts = Ks[order][-tail:], ts[order][-tail:]
    if Ks.size < 3:
        return float('nan')
    return float(np.polyfit(np.log(Ks), np.log(ts), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='small sweep')
    ap.add_argument('--truncation', action='store_true',
                    help='add a pass at the 6-sigma default')
    ap.add_argument('--rs', type=str, default=None,
                    help='comma-separated tuple sizes, e.g. 2,3,4,5')
    ap.add_argument('--ks', type=str, default=None,
                    help='comma-separated multiset sizes, e.g. 6,12,20,34,48')
    ap.add_argument('--budget', type=float, default=CELL_BUDGET_S,
                    help='predicted seconds per cell per method; cells above are skipped')
    ap.add_argument('--out', default='bench_mobius_python.csv')
    args = ap.parse_args()

    rs = tuple(int(x) for x in args.rs.split(',')) if args.rs else \
        ((2, 3) if args.quick else RS)
    ks = tuple(int(x) for x in args.ks.split(',')) if args.ks else \
        ((6, 12, 20) if args.quick else KS)
    modes = MODES[:1] if args.quick else MODES

    rng = LCG(SEED)
    rows = []
    failures = []
    skipped = []

    print(f'python {platform.python_version()} | numpy {np.__version__} | '
          f'mpt {getattr(mpt, "__version__", "?")}')
    print(f'{platform.platform()}')
    rate = calibrate_rate()
    est = 0.0
    for is_rel, is_per, _ in modes:
        for r in rs:
            for K in ks:
                if K < r + 1:
                    continue
                for m in METHODS:
                    p_s = predict_s(m, r, K, rate, is_rel, is_per)
                    if p_s > args.budget:
                        continue
                    # Six batches (warm-up plus five timed), each sized to
                    # TARGET_MS or to one call, whichever is larger. Cheap
                    # cells therefore cost ~0.9 s regardless of how fast the
                    # call is: this floor, not the arithmetic, sets the run
                    # time over most of the grid.
                    est += 6 * max(TARGET_MS / 1e3, p_s)
                    if args.truncation:
                        est *= 1.0      # counted below instead
    if args.truncation:
        est *= 2.0
    print(f'calibrated at {rate * 1e9:.1f} ns per tuple pair; '
          f'cell budget {args.budget:g} s; estimated run {est / 60:.1f} min\n')
    header = f"{'mode':<22}{'r':>3}{'K':>5}  " + ''.join(f'{m:>12}' for m in METHODS) + '   winner'
    print(header)
    print('-' * len(header))

    for is_rel, is_per, mlabel in modes:
        for r in rs:
            for K in ks:
                if K < r + 1:
                    continue
                pair = make_pair(K, rng)
                A, B = build(pair, r, is_rel, is_per)

                # --- correctness first, untruncated ------------------------
                vals = {}
                for m in METHODS:
                    if predict_s(m, r, K, rate, is_rel, is_per) > args.budget:
                        skipped.append((mlabel, r, K, m,
                                        predict_s(m, r, K, rate, is_rel, is_per)))
                        continue
                    try:
                        vals[m] = call(A, B, m, float('inf'))
                    except Exception as exc:
                        failures.append((mlabel, r, K, m, f'{type(exc).__name__}: {exc}'))
                # Bulger is the accuracy reference. Where it was skipped as
                # over budget there is no reference, and the deviation is
                # recorded as NaN rather than as a spurious zero.
                ref = vals.get('bulger', float('nan'))
                devs = {}
                for m, v in vals.items():
                    if m == 'bulger' or not np.isfinite(ref) or not np.isfinite(v):
                        devs[m] = float('nan')
                        continue
                    devs[m] = abs(v - ref) / max(1.0, abs(ref))
                    if devs[m] > TOL:
                        failures.append((mlabel, r, K, m,
                                         f'deviates from bulger by {devs[m]:.2e}'))

                # --- timing -------------------------------------------------
                times = {}
                for m in vals:
                    A2, B2 = build(pair, r, is_rel, is_per)   # fresh objects
                    try:
                        t, n = timed(lambda: call(A2, B2, m, float('inf')))
                        times[m] = t
                        rows.append(dict(language='python', mode=mlabel, is_rel=is_rel,
                                         is_per=is_per, r=r, K=K, method=m,
                                         truncation='inf', seconds=t, reps=n,
                                         value=vals[m], rel_dev=devs.get(m, float('nan'))))
                    except Exception as exc:
                        failures.append((mlabel, r, K, m, f'timing: {exc}'))

                if args.truncation:
                    for m in vals:
                        A2, B2 = build(pair, r, is_rel, is_per)
                        try:
                            t, n = timed(lambda: call(A2, B2, m, 6.0))
                            rows.append(dict(language='python', mode=mlabel, is_rel=is_rel,
                                             is_per=is_per, r=r, K=K, method=m,
                                             truncation='6', seconds=t, reps=n,
                                             value=call(A2, B2, m, 6.0),
                                             rel_dev=float('nan')))
                        except Exception:
                            pass

                comp = [m for m in METHODS if m in times]
                best = min(comp, key=lambda m: times[m]) if comp else '-'
                cells = ''.join(
                    (f'{times[m] * 1e3:>11.3f}m' if m in times else f'{"-":>12}')
                    for m in METHODS)
                print(f'{mlabel:<22}{r:>3}{K:>5}  {cells}   {best}')

    # --- scaling fits -------------------------------------------------------
    print('\nfitted exponent of K over the three largest K (log t vs log K), untruncated:')
    print(f"  {'mode':<22}{'r':>3}   " + ''.join(f'{m:>10}' for m in METHODS)
          + '     predicted')
    for is_rel, is_per, mlabel in modes:
        for r in rs:
            line = f'  {mlabel:<22}{r:>3}   '
            for m in METHODS:
                sel = [(row['K'], row['seconds']) for row in rows
                       if row['mode'] == mlabel and row['r'] == r
                       and row['method'] == m and row['truncation'] == 'inf']
                e = fit_exponent([k for k, _ in sel], [t for _, t in sel])
                line += f'{e:>10.2f}' if np.isfinite(e) else f'{"-":>10}'
            print(line + f'     {2 * r} / {2 * r} / 2')

    print('\nagreement with Bulger (relative deviation, untruncated); '
          'large values indicate cancellation, not error:')
    worst = sorted((rw for rw in rows
                    if rw['method'] == 'mobius' and rw['truncation'] == 'inf'
                    and np.isfinite(rw['rel_dev'])),
                   key=lambda rw: -rw['rel_dev'])[:8]
    for rw in worst:
        print(f"   {rw['mode']:<22} r={rw['r']}  K={rw['K']:>3}   {rw['rel_dev']:.2e}")

    if skipped:
        print(f'\nskipped as over budget ({len(skipped)}); raise --budget to include:')
        for sk in sorted(skipped, key=lambda s: s[4])[:6]:
            print(f'   {sk[0]:<22} r={sk[1]}  K={sk[2]:>3}  {sk[3]:<7} '
                  f'predicted {sk[4]:.1f} s')
        if len(skipped) > 6:
            print(f'   ... and {len(skipped) - 6} more')

    if failures:
        print(f'\nFAILURES ({len(failures)}):')
        for f in failures[:25]:
            print('  ', f)
    else:
        print('\nall available methods agreed to within '
              f'{TOL:g} on every cell')

    with open(args.out, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f'\nwrote {args.out}  ({len(rows)} rows)')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
