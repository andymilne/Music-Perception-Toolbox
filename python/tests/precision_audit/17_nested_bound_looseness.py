"""How loose is the nested accuracy guard's bound?

The guard diverts the Möbius route when

    bound = |Omega_r| * eps * max|term|   (on the scale combineOrbit returns)

exceeds ``truncation_floor(truncationSigmas)``. That bound assumes every
rounding error across the orbit sum aligns in sign. This script measures
it against the error it is bounding --- the Möbius route's actual
deviation from enumeration on the same block --- so the guard can be
judged on whether it diverts where the accuracy the caller asked for is
genuinely at risk.

Blocks are built from Gaussian kernel values between point sets rather
than from uniform noise: an entry of the block M is a kernel value, so
its distribution is what the guard actually meets. Four weight profiles
are swept, matching the ones the guard's own docstring cites.
"""
import numpy as np

from mpt._defaults import truncation_floor
from mpt._tensor._nested_contraction import (
    _combine_orbit, _combine_chunked, _tuple_sides, _ORBIT_ENUM_MAX_ELEMS,
)

FLOOR = truncation_floor(6)
Q = 8
SEEDS = 3


def weights(kind, K, rng):
    if kind == "uniform":
        return np.ones(K)
    if kind == "power":
        return (np.arange(1, K + 1, dtype=float)) ** -1.0
    if kind == "geometric":
        return 10.0 ** -np.arange(K, dtype=float)
    if kind == "dominant":
        w = np.ones(K)
        w[0] = 1000.0
        return w
    raise ValueError(kind)


def kernel_block(K, sigma, kind, rng):
    """A (Q, K, K) block of Gaussian kernel values, weighted per side."""
    wx = weights(kind, K, rng)
    wy = weights(kind, K, rng)
    M = np.empty((Q, K, K))
    for q in range(Q):
        px = np.sort(rng.uniform(0.0, 1200.0, K))
        py = np.sort(rng.uniform(0.0, 1200.0, K))
        d = px[:, None] - py[None, :]
        M[q] = np.exp(-(d ** 2) / (4.0 * sigma ** 2)) * wx[:, None] * wy[None, :]
    return M


def main():
    print(f"floor(6) = {FLOOR:.3e}. Blocks are Gaussian kernel values, "
          f"Q={Q}, sigma=60, P=1200, {SEEDS} seeds.\n")
    header = (f"{'weights':<10} {'r':>2} {'K':>2} | {'bound':>10} "
              f"{'actual err':>11} {'loose by':>9} | {'fires':>5} "
              f"{'needed':>6} {'verdict':>12}")
    print(header)
    print("-" * len(header))
    for kind in ("uniform", "power", "geometric", "dominant"):
        for r in (3, 4, 5, 6, 7):
            for K in (r, r + 2):
                bs, es = [], []
                for s in range(SEEDS):
                    rng = np.random.default_rng(1000 * r + 10 * K + s)
                    M = kernel_block(K, 60.0, kind, rng)
                    vals, bound = _combine_orbit(M, r, return_bound=True)
                    xt, yt = _tuple_sides(K, r, True)
                    ref = _combine_chunked(M, xt, yt, _ORBIT_ENUM_MAX_ELEMS)
                    bs.append(float(np.max(bound)))
                    es.append(float(np.max(np.abs(
                        np.asarray(vals) - np.asarray(ref)))))
                b, e = max(bs), max(es)
                fires, needed = b > FLOOR, e > FLOOR
                if fires and not needed:
                    verdict = "OVER-REJECTS"
                elif needed and not fires:
                    verdict = "MISSES"
                else:
                    verdict = ""
                loose = b / e if e > 0 else float("inf")
                print(f"{kind:<10} {r:>2} {K:>2} | {b:>10.2e} {e:>11.2e} "
                      f"{loose:>9.1e} | {str(fires):>5} {str(needed):>6} "
                      f"{verdict:>12}")
        print()


if __name__ == "__main__":
    main()
