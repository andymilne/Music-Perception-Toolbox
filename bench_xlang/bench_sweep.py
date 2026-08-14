"""Cross-language sweep benchmark: the point-set query shape.

Times the three ways of scoring one context density against M query
densities (all r = 1, one value per event per attribute), under both
normalisations:

* ``broadcast``   -- one ``cos_sim_exp_tens(dX, [d1, ..., dM])`` call
  (the batched kernel pass; self terms memoised across the sweep).
* ``loop_memo``   -- M scalar calls against the same context object
  (per-pair path; the context's self term is memoised after the first
  call).
* ``loop_fresh``  -- M scalar calls with every memo cleared before each
  pair (each pair pays its own self terms; the steady per-pair cost
  with no cross-pair reuse).

Inputs are deterministic formulae (BENCH_SPEC conventions), identical
in ``bench_sweep.m``, so the ``checksum`` column doubles as a
cross-language value-parity check.

Writes ``bench_sweep_python.csv``; join against the MATLAB CSV with
``compare_sweep.py``.
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import mpt                                            # noqa: E402
mpt.set_default(show_hints=False)   # dispatch prints would sit inside the timed closures
import warnings                                       # noqa: E402
from mpt._defaults import TruncationDefaultWarning    # noqa: E402
warnings.filterwarnings("ignore", category=TruncationDefaultWarning)
from bench_xlang import adaptive_time                 # noqa: E402

SIG_P, SIG_T = 0.25, 0.08
M_QUERIES = 100
GRID_N = (300, 1200, 5000)
NORMS = ("cosine", "oneSidedDenom")


def context_arrays(n):
    """Deterministic context: n pitches in [40, 90), increasing times."""
    j = np.arange(1, n + 1, dtype=float)
    pitches = 40.0 + 50.0 * np.mod(7.0 * j * j + 3.0 * j, 997.0) / 997.0
    times = np.cumsum(0.05 + 0.2 * np.mod(3.0 * j, 11.0) / 11.0)
    return pitches, times


def query_arrays(k):
    """Deterministic 3-event query, offset by the sweep index k."""
    q_p = np.array([60.0, 63.5, 68.25]) + 0.7 * k
    q_t = np.array([0.5, 1.25, 2.0]) + 1.3 * k
    return q_p, q_t


def build(pitches, times):
    return mpt.build_exp_tens(
        [np.asarray(pitches).reshape(1, -1),
         np.asarray(times).reshape(1, -1)],
        None, [SIG_P, SIG_T], [1, 1], [False, False], [False, False],
        [0.0, 0.0], verbose=False)


def clear_memos(densities):
    for d in densities:
        d._self_ip_cache.clear()
        if d._pruned_cached is not None and d._pruned_cached is not d:
            d._pruned_cached._self_ip_cache.clear()


def main():
    out_path = os.path.join(os.path.dirname(__file__),
                            "bench_sweep_python.csv")
    rows = []
    for n in GRID_N:
        p_ctx, t_ctx = context_arrays(n)
        d_ctx = build(p_ctx, t_ctx)
        d_qs = [build(*query_arrays(k)) for k in range(M_QUERIES)]
        for norm in NORMS:

            # Every variant's timed closure is one complete operation
            # from a cold memo, matching the MATLAB bench's value
            # semantics (its memo cannot cross calls): broadcast and
            # loop_memo pay the shared self term once per timed call,
            # loop_fresh once per pair. Python's object-attached memo
            # would otherwise persist across the timer's repetitions
            # and the cosine rows would compare Python-warm against
            # MATLAB-cold.
            def run_broadcast():
                clear_memos([d_ctx] + d_qs)
                return mpt.cos_sim_exp_tens(
                    d_ctx, d_qs, normalize=norm, verbose=False)

            def run_loop_memo():
                clear_memos([d_ctx] + d_qs)
                return np.array([
                    mpt.cos_sim_exp_tens(d_ctx, d, normalize=norm,
                                         verbose=False)
                    for d in d_qs])

            def run_loop_fresh():
                out = np.empty(len(d_qs))
                for i, d in enumerate(d_qs):
                    clear_memos([d_ctx, d])
                    out[i] = mpt.cos_sim_exp_tens(
                        d_ctx, d, normalize=norm, verbose=False)
                return out

            variants = (("broadcast", run_broadcast),
                        ("loop_memo", run_loop_memo),
                        ("loop_fresh", run_loop_fresh))
            for name, fn in variants:
                clear_memos([d_ctx] + d_qs)
                t, result, n_inner = adaptive_time(fn)
                checksum = float(np.sum(np.asarray(result, dtype=float)))
                per_offset_ms = t / M_QUERIES * 1000.0
                rows.append(dict(
                    language="python", N=n, M=M_QUERIES,
                    normalize=norm, variant=name,
                    ms_per_offset=f"{per_offset_ms:.6f}",
                    n_inner=n_inner,
                    checksum=f"{checksum:.12e}",
                ))
                print(f"N={n:5d} {norm:14s} {name:11s}: "
                      f"{per_offset_ms:8.4f} ms/offset  "
                      f"checksum {checksum:.12e}")

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
