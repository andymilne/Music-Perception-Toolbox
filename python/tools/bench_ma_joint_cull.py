"""Timing of the joint-centres path of ``eval_maet``: dense chunks and culling.

Twin of matlab/tests/bench_ma_joint_cull.m. The joint-centres path (taken
where an attribute is at r = 1 or carries a kernel covariance) holds a
dense chunk to a cache-sized working set and culls on one coordinate
where that pays (see ``_ma_cull_plan`` in ``mpt/_tensor/eval.py``). This
script measures, on densities that take the path, the time per query of
the dense evaluation at several chunk caps and of the culled evaluation
at several block costs, to check three constants on a given machine:

* ``_MA_CACHE_CHUNK_BYTES`` (8 MB): the cap whose column is fastest;
* ``_MA_CULL_BLOCK_COST`` (4096): the fixed cost of a block of culled
  queries, in pairs of the dense broadcast, the most pairs a block may
  evaluate beyond its queries' own runs; the culled columns run at 1024,
  4096 and 16384, and the middle one should be about the fastest;
* ``_MA_CULL_PAIR_COST`` (1.25): the cost of a pair in a culled query's
  run relative to a dense pair, the blocks' wasted pairs and fixed cost
  spread over the runs, printed as ``ratio`` (culled time at the default
  block cost per pair in the runs, over dense time per pair at the 8 MB
  cap). Culling is chosen when the pairs in the runs, at that cost,
  undercut the dense ones.

HOW TO RUN
----------
From the ``python`` directory, with the package importable::

    PYTHONPATH=. python3 tools/bench_ma_joint_cull.py

Each cell is the minimum of five runs after a warm-up.
"""
import time

import numpy as np

import mpt
import mpt._tensor.eval as ev
from mpt._tensor.dispatch import _inner_r_vec

N_Q = 500
CAPS_MB = [1, 2, 4, 8, 16, 32]
BLOCK_COSTS = [1024, 4096, 16384]


def _time_ms(fn, repeats=5):
    fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def _cells(rng):
    """(label, density, queries): an attribute at r = 1 on each."""
    out = []
    for A, s, N in [(2, 0.1, 10000), (2, 0.3, 10000), (3, 0.1, 10000)]:
        ps = [rng.uniform(0, 10, (1, N)) for _ in range(A)]
        d = mpt.build_maet(ps, None, [s] * A, [1] * A, [False] * A,
                           [False] * A, [0.0] * A, verbose=False)
        out.append((f"A={A} r=1 N={N} sigma={s}", d,
                    rng.uniform(0, 10, (A, N_Q))))
    for N in (300, 3000):
        ps = [np.sort(rng.uniform(0, 100, N))[None, :],
              rng.uniform(60, 72, (3, N))]
        d = mpt.build_maet(ps, None, [0.3, 0.3], [1, 2], [False, False],
                           [False, False], [0.0, 0.0], verbose=False)
        X = np.vstack([rng.uniform(0, 100, N_Q),
                       rng.uniform(60, 72, (2, N_Q))])
        out.append((f"onset r=1 + 3 pitches r=2, N={N}", d, X))
    return out


def _kept_pairs(d, X):
    rows = np.cumsum([0] + [int(v) for v in d.dim_per_attr])
    xl = [X[rows[a]:rows[a + 1]] for a in range(d.n_attrs)]
    plan = ev._ma_cull_plan(
        d.centres, xl, int(d.n_j), X.shape[1], d.dim_per_attr, d.sigma,
        d.rel, d.per, d.period, _inner_r_vec(d), getattr(d, "wrap", None),
        mpt.get_default("truncation_sigmas"), np.float64)
    return float(np.sum(plan[2] - plan[1]))


def main():
    prev = (mpt.get_default("show_hints"), ev._MA_CULL_MODE,
            ev._MA_CACHE_CHUNK_BYTES, ev._MA_CULL_BLOCK_COST)
    mpt.set_default(show_hints=False, truncation_sigmas=6.0)
    try:
        rng = np.random.default_rng(0)
        print("microseconds per query; dense at each chunk cap, culled at "
              "each block cost, auto, and the culled-to-dense pair-cost "
              "ratio")
        print(f"{'cell':36s} " + " ".join(f"{c:>6d}MB" for c in CAPS_MB)
              + " " + " ".join(f"{'b' + str(b):>8s}" for b in BLOCK_COSTS)
              + "     auto  ratio")
        for label, d, X in _cells(rng):
            def run(mode, cap, block=prev[3]):
                ev._MA_CULL_MODE = mode
                ev._MA_CACHE_CHUNK_BYTES = cap
                ev._MA_CULL_BLOCK_COST = block
                return mpt.eval_maet(d, X, method="centres", verbose=False)
            dense = [_time_ms(lambda c=c: run("never", c * 2 ** 20))
                     for c in CAPS_MB]
            culled = [_time_ms(lambda b=b: run("always", 8 * 2 ** 20, b))
                      for b in BLOCK_COSTS]
            auto = _time_ms(lambda: run("auto", 8 * 2 ** 20))
            ev._MA_CULL_MODE = "always"
            kept = _kept_pairs(d, X)
            dense8 = dense[CAPS_MB.index(8)]
            culled0 = _time_ms(lambda: run("always", 8 * 2 ** 20))
            ratio = (culled0 / kept) / (dense8 / (float(d.n_j) * X.shape[1]))
            per_q = 1e3 / X.shape[1]
            print(f"{label:36s} "
                  + " ".join(f"{t * per_q:8.1f}" for t in dense)
                  + " " + " ".join(f"{t * per_q:8.1f}" for t in culled)
                  + f" {auto * per_q:8.1f} {ratio:6.2f}",
                  flush=True)
    finally:
        mpt.set_default(show_hints=prev[0])
        ev._MA_CULL_MODE = prev[1]
        ev._MA_CACHE_CHUNK_BYTES = prev[2]
        ev._MA_CULL_BLOCK_COST = prev[3]


if __name__ == "__main__":
    main()
