"""Cross-language benchmark runner (Python side).

Generates deterministic inputs (identical to the MATLAB runner),
times eval_exp_tens and cos_sim_exp_tens across a small grid varying
one axis at a time from a base configuration, and writes CSV.

Usage:
    python3 bench_xlang.py                             # writes bench_python.csv, auto method
    python3 bench_xlang.py --out X.csv                 # custom output path
    python3 bench_xlang.py --method bulger             # force cossim to bulger route
    python3 bench_xlang.py --method mobius             # force cossim to mobius route

Forcing a cossim method isolates route-vs-routing discrepancies:
if the two languages disagree at method='auto' but agree at
method='bulger', the disagreement is which route the two auto
dispatchers picked, not the routes themselves. If they still
disagree at method='bulger', the bulger path itself has a
cross-language mismatch.

See BENCH_SPEC.md for the grid and input formulae.
"""
import argparse
import csv
import inspect
import math
import sys
import time
from pathlib import Path

# Prepend the repo's python/ directory to sys.path so the bench runs
# against the local repo's mpt (v3+ with wrap= support), not any
# older mpt that may be installed on the machine.
_REPO_PYTHON = Path(__file__).resolve().parent.parent / 'python'
if _REPO_PYTHON.is_dir():
    sys.path.insert(0, str(_REPO_PYTHON))

import numpy as np

import mpt

# Sanity check: the bench uses the v3+ ``wrap=`` keyword throughout,
# so make sure ``build_exp_tens`` is the version that supports it.
# If not, print where mpt was loaded from and bail out with a clear
# actionable message rather than 27 identical FAIL lines.
_bet_params = inspect.signature(mpt.build_exp_tens).parameters
if 'wrap' not in _bet_params:
    _mpt_file = getattr(mpt, '__file__', '<unknown>')
    print(f"ERROR: the mpt package loaded here does not support the "
          f"wrap= keyword on build_exp_tens.")
    print(f"  loaded from: {_mpt_file}")
    print(f"  expected:    {_REPO_PYTHON / 'mpt' / '__init__.py'}")
    print()
    print("Fix by one of:")
    print("  1. Pull the latest Python changes into the repo at "
          f"{_REPO_PYTHON.parent}")
    print("  2. `pip uninstall mpt` to remove any older installed copy")
    print("  3. Run the bench from the repo's python/ directory instead")
    sys.exit(2)


PERIOD = 1200.0


# ---------------------------------------------------------------------
# Deterministic input generation (matches bench_xlang.m)
# ---------------------------------------------------------------------

def make_inputs(A, N, K, period):
    """Return per-attribute cell of positions (K, N) and weights (K, N)."""
    S = K + N + A
    p_all = []
    w_all = []
    for a in range(A):
        pa = np.zeros((K, N))
        wa = np.zeros((K, N))
        for n in range(N):
            for j in range(K):
                idx = (j + 3 * n + 7 * a) % S
                pa[j, n] = period * idx / S
                wa[j, n] = 0.7 + 0.3 * math.cos(
                    (j + 2 * n + 5 * a) / S * math.pi
                )
        p_all.append(pa)
        w_all.append(wa)
    return p_all, w_all


def make_queries(dim, n_q, period):
    """Return X of shape (dim, n_q)."""
    S = dim + n_q + 1
    X = np.zeros((dim, n_q))
    for d in range(dim):
        for q in range(n_q):
            X[d, q] = period * (0.5 + 0.4 * math.sin(
                (d + 2 * q + 1) / S * math.pi
            ))
    return X


# ---------------------------------------------------------------------
# Case iteration
# ---------------------------------------------------------------------

BASE = dict(
    sigma_over_P=0.10, r=2, isRel=False, isPer=True,
    wrap='full-image', A=1, N=1, K=8, nQ=10,
)


def iter_cases():
    """Yield (label, config-dict). Duplicates deduped by fingerprint."""
    seen = set()

    def emit(label, cfg):
        fp = tuple(sorted(cfg.items()))
        if fp in seen:
            return None
        seen.add(fp)
        return label, cfg

    yield emit('base', dict(BASE))

    for v in [0.002, 0.005, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50]:
        cfg = dict(BASE); cfg['sigma_over_P'] = v
        y = emit(f'sigma_over_P={v}', cfg)
        if y: yield y

    for v in [1, 2, 3, 4]:
        cfg = dict(BASE); cfg['r'] = v
        y = emit(f'r={v}', cfg)
        if y: yield y

    for (rel, per) in [(False, False), (False, True), (True, False), (True, True)]:
        cfg = dict(BASE); cfg['isRel'] = rel; cfg['isPer'] = per
        y = emit(f'isRel={rel}&isPer={per}', cfg)
        if y: yield y

    for v in ['full-image', 'single-image']:
        cfg = dict(BASE); cfg['wrap'] = v
        y = emit(f'wrap={v}', cfg)
        if y: yield y

    for v in [1, 2, 3]:
        cfg = dict(BASE); cfg['A'] = v
        y = emit(f'A={v}', cfg)
        if y: yield y

    for v in [1, 5, 20, 50, 100]:
        cfg = dict(BASE); cfg['N'] = v
        y = emit(f'N={v}', cfg)
        if y: yield y

    for v in [4, 8, 16, 50, 100]:
        cfg = dict(BASE); cfg['K'] = v
        y = emit(f'K={v}', cfg)
        if y: yield y

    for v in [1, 10, 100]:
        cfg = dict(BASE); cfg['nQ'] = v
        y = emit(f'nQ={v}', cfg)
        if y: yield y


# ---------------------------------------------------------------------
# Timing and measurement
# ---------------------------------------------------------------------

def adaptive_time(fn, target_inner_ms=30.0, n_outer=7, n_warmup=2,
                  max_inner=1_000_000):
    """Adaptive-inner-loop timing with outer-minimum aggregation.

    A single call to fn() is measured; if it is sub-millisecond, best-of-N
    on raw single-call measurements is dominated by timer-resolution
    noise (macOS perf_counter is ~1 us, but background OS activity and
    thread scheduling add several us of jitter per measurement). Fix:
    loop N times inside a single tic-toc so the measured window is
    ~30 ms, drowning out per-measurement jitter, then divide by N.
    n_outer independent measurements are collected and the minimum is
    returned as the estimate (minimum is robust to sporadic background
    load and reflects the intrinsic per-call cost when the CPU is not
    contested).

    Returns (best_seconds_per_call, first_result, n_inner).
    """
    # First call warms caches, does JIT compilation, and returns the
    # result we hand back to the caller.
    result = fn()

    # Extra warm-ups so the sizing measurement isn't polluted by cold
    # cache/JIT effects.
    for _ in range(n_warmup):
        fn()

    # Quick size: one measurement to estimate single-call time, then
    # compute n_inner so a single outer window is ~target_inner_ms.
    t0 = time.perf_counter()
    fn()
    single_s = max(time.perf_counter() - t0, 1e-9)
    n_inner = max(1, int(target_inner_ms / 1000.0 / single_s))
    n_inner = min(n_inner, max_inner)

    # Timed measurements
    ts = []
    for _ in range(n_outer):
        t0 = time.perf_counter()
        for _ in range(n_inner):
            fn()
        t1 = time.perf_counter()
        ts.append((t1 - t0) / n_inner)
    return min(ts), result, n_inner


def best_of_3(fn):
    """Legacy: single-call best-of-3. Retained for reference; new callers
    should use adaptive_time. Returned tuple matches (t, result)."""
    result = fn()
    ts = [None, None, None]
    for i in range(3):
        t0 = time.perf_counter()
        r = fn()
        t1 = time.perf_counter()
        ts[i] = t1 - t0
    return min(ts), result


def run_eval(cfg):
    sigma = cfg['sigma_over_P'] * PERIOD
    A, N, K, r = cfg['A'], cfg['N'], cfg['K'], cfg['r']
    isRel, isPer, nQ = cfg['isRel'], cfg['isPer'], cfg['nQ']
    wrap = cfg['wrap']
    period_val = PERIOD if isPer else 0.0

    p_all, w_all = make_inputs(A, N, K, PERIOD)
    dim_per = [r - int(isRel) for _ in range(A)]
    dim = sum(dim_per)
    X = make_queries(dim, nQ, PERIOD)

    d = mpt.build_exp_tens(
        p_all, w_all,
        [sigma] * A, [r] * A,
        [isRel] * A, [isPer] * A, [period_val] * A,
        wrap=wrap, verbose=False,
    )
    n_j = int(getattr(d, 'n_j', -1))

    def call():
        return mpt.eval_exp_tens(d, X, verbose=False)

    t, result, n_inner = adaptive_time(call)
    return t, result, n_j, n_inner


def run_cossim(cfg, method='auto'):
    sigma = cfg['sigma_over_P'] * PERIOD
    A, N, K, r = cfg['A'], cfg['N'], cfg['K'], cfg['r']
    isRel, isPer = cfg['isRel'], cfg['isPer']
    wrap = cfg['wrap']
    period_val = PERIOD if isPer else 0.0

    p_all, w_all = make_inputs(A, N, K, PERIOD)
    # Second density: shift positions by a fixed offset.
    p_all_2 = [pa + 37.5 for pa in p_all]

    dx = mpt.build_exp_tens(
        p_all, w_all,
        [sigma] * A, [r] * A,
        [isRel] * A, [isPer] * A, [period_val] * A,
        wrap=wrap, verbose=False,
    )
    dy = mpt.build_exp_tens(
        p_all_2, w_all,
        [sigma] * A, [r] * A,
        [isRel] * A, [isPer] * A, [period_val] * A,
        wrap=wrap, verbose=False,
    )
    n_j = int(getattr(dx, 'n_j', -1))

    def call():
        return mpt.cos_sim_exp_tens(dx, dy, method=method, verbose=False)

    t, result, n_inner = adaptive_time(call)
    return t, result, n_j, n_inner


def checksum_eval(v):
    return float(np.sum(np.abs(np.asarray(v).ravel())))


def checksum_cossim(v):
    return float(v)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='bench_python.csv')
    ap.add_argument('--method', default='auto',
                    choices=['auto', 'bulger', 'mobius'],
                    help="Force cos_sim_exp_tens method (default: auto). "
                         "'auto' lets the cost model choose; 'bulger' and "
                         "'mobius' force each specific route to isolate "
                         "route-vs-routing discrepancies.")
    args = ap.parse_args()

    # Silence dispatch messages: with adaptive-inner-loop timing the
    # bench runs each call thousands of times, and 'chose X path' fires
    # on every top-level entry (the throttle resets per top-level
    # call). show_hints=False disables the whole dispatch-message
    # facility globally for this run; correctness is unaffected.
    try:
        mpt.set_default(show_hints=False)
    except Exception:
        pass

    rows = []
    n_cases = 0
    for item in iter_cases():
        if item is None:
            continue
        label, cfg = item
        n_cases += 1

    print(f"Running {n_cases} unique configurations with cossim "
          f"method='{args.method}', each measured for eval and cossim "
          f"(adaptive inner loop, ~30 ms window, min of 7)...")

    idx = 0
    for item in iter_cases():
        if item is None:
            continue
        label, cfg = item
        idx += 1
        print(f"  [{idx}/{n_cases}] {label}: ", end='', flush=True)

        # eval
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                t_eval, v_eval, n_j_eval, n_in_eval = run_eval(cfg)
                cs_eval = checksum_eval(v_eval)
                rows.append(dict(
                    label=label, operation='eval',
                    elapsed_s=t_eval, checksum=cs_eval,
                    n_j=n_j_eval, n_inner=n_in_eval,
                    **cfg,
                ))
                print(f"eval {t_eval*1e6:.1f}us(x{n_in_eval})  ",
                      end='', flush=True)
            except Exception as e:
                print(f"eval FAIL ({e})  ", end='', flush=True)

            # cossim
            # Graceful skip: forcing bulger at A>=3 materialises the joint
            # tuple set (n_j^A pair matrix), which balloons quadratically
            # and OOMs the process well before the dispatcher's memory
            # guard could catch it (250+ GB at A=3, K=8, r=2). auto and
            # mobius handle A=3 fine (mobius factorises across attributes),
            # so we only skip the forced-bulger pass.
            if args.method == 'bulger' and int(cfg['A']) >= 3:
                print("cossim SKIP (bulger joint tuple set too large "
                      "at A>=3)")
            else:
                try:
                    t_cos, v_cos, n_j_cos, n_in_cos = run_cossim(cfg, method=args.method)
                    cs_cos = checksum_cossim(v_cos)
                    rows.append(dict(
                        label=label, operation='cossim',
                        elapsed_s=t_cos, checksum=cs_cos,
                        n_j=n_j_cos, n_inner=n_in_cos,
                        **cfg,
                    ))
                    print(f"cossim {t_cos*1e6:.1f}us(x{n_in_cos})")
                except Exception as e:
                    print(f"cossim FAIL ({e})")

    # Write CSV
    fieldnames = [
        'label', 'operation', 'elapsed_s', 'checksum', 'n_j', 'n_inner',
        'sigma_over_P', 'r', 'isRel', 'isPer', 'wrap',
        'A', 'N', 'K', 'nQ',
    ]
    out_path = Path(args.out)
    with out_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == '__main__':
    main()
