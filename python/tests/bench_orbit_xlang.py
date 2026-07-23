"""Cross-language speed benchmark for the orbit IP/eval machinery.

Targeted spot-checks (not a sweep) at three representative (r, K)
configurations. The companion MATLAB script
``matlab/tests/bench_orbit_xlang.m`` runs the same configurations.
Compare median wall times to gauge whether MATLAB is within ~2x of
Python (the gating threshold for shipping the v2.2 MATLAB port as-is
vs investing in precomputed contraction paths or a sharper greedy
heuristic in ``mobius.contract``).

What we measure:
  - ``inner_product_orbit`` on a single (K_x, K_y) kernel.
  - ``inner_product_orbit_pw_batched`` on a batch of P pairs.
  - End-to-end ``cos_sim_exp_tens`` single-multiset orbit (a sanity check that
    includes dispatcher overhead).

Each configuration runs a small warm-up to amortise import / cache
costs, then reports the median over N_REPS runs.

Output format is fixed (one CSV-style line per measurement) so the
MATLAB script can produce a directly comparable file. Concatenate
both files and read into a comparison table.
"""

import time
import numpy as np

import mpt
from mpt._mobius import inner_product_orbit, inner_product_orbit_pw_batched
from mpt.tensor import cos_sim_exp_tens


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# (r, K) configurations to spot-check. Span the typical range:
# r=2..5, K small enough to keep runtimes < 10 s/each per MATLAB run.
CONFIGS = [
    {'r': 2, 'K': 8,  'P': 50},
    {'r': 3, 'K': 8,  'P': 50},
    {'r': 3, 'K': 12, 'P': 50},
    {'r': 4, 'K': 8,  'P': 20},
    {'r': 5, 'K': 6,  'P': 10},
]

N_REPS = 7         # median of N_REPS timings per measurement
N_WARMUP = 2       # untimed warmup runs


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _make_kernel(K_x, K_y, seed=0):
    """Reproducible (K_x, K_y) Gaussian-style kernel + weight vectors."""
    rng = np.random.default_rng(seed)
    p_x = np.sort(rng.uniform(0, 2000, K_x))
    p_y = np.sort(rng.uniform(0, 2000, K_y))
    sigma = 30.0
    K = np.exp(-(p_x[:, None] - p_y[None, :]) ** 2 / (4 * sigma ** 2))
    w_a = rng.uniform(0.5, 1.5, K_x)
    w_b = rng.uniform(0.5, 1.5, K_y)
    return K, w_a, w_b


def _timed(fn, n_warmup=N_WARMUP, n_reps=N_REPS):
    """Median wall-clock seconds over n_reps runs (after n_warmup)."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ----------------------------------------------------------------------
# Benchmarks
# ----------------------------------------------------------------------

def bench_orbit_single(r, K, P_unused):
    """Time inner_product_orbit on a single (K, K) kernel."""
    K_mat, w_a, w_b = _make_kernel(K, K, seed=r * 7)
    return _timed(
        lambda: inner_product_orbit(K_mat, w_a, w_b, r, prefactor=1.0),
    )


def bench_orbit_batched(r, K, P):
    """Time inner_product_orbit_pw_batched on P pairs of (K, K) kernels."""
    rng = np.random.default_rng(seed=r * 11 + K)
    K_pairs = np.empty((P, K, K))
    w_A = np.empty((P, K))
    w_B = np.empty((P, K))
    for i in range(P):
        K_mat, w_a, w_b = _make_kernel(K, K, seed=r * 11 + i)
        K_pairs[i] = K_mat
        w_A[i] = w_a
        w_B[i] = w_b
    return _timed(
        lambda: inner_product_orbit_pw_batched(
            K_pairs, w_A, w_B, r, prefactor=1.0,
        ),
    )


def bench_cossim_orbit_sm(r, K, P_unused):
    """End-to-end cos_sim_exp_tens single-multiset orbit (with dispatcher overhead)."""
    rng = np.random.default_rng(seed=r * 13 + K)
    p1 = np.sort(rng.uniform(0, 2000, K))
    p2 = np.sort(rng.uniform(0, 2000, K))
    w = np.ones(K)
    return _timed(
        lambda: cos_sim_exp_tens(
            p1, w, p2, w, 30.0, r, False, False, 0.0,
            method='mobius', verbose=False,
        ),
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == '__main__':
    print("language,bench,r,K,P,t_median_ms")
    for cfg in CONFIGS:
        r, K, P = cfg['r'], cfg['K'], cfg['P']
        # Skip r=2,K=8: orbit at r=2 has only 4 orbit terms; both
        # languages are sub-millisecond and the comparison is noisy.
        # Keep it as a sanity floor anyway.

        t_single  = bench_orbit_single(r, K, P) * 1000
        t_batched = bench_orbit_batched(r, K, P) * 1000
        t_cossim  = bench_cossim_orbit_sm(r, K, P) * 1000

        print(f"python,inner_product_orbit,{r},{K},1,{t_single:.4f}")
        print(f"python,inner_product_orbit_pw_batched,{r},{K},{P},{t_batched:.4f}")
        print(f"python,cos_sim_exp_tens_sa_orbit,{r},{K},1,{t_cossim:.4f}")
