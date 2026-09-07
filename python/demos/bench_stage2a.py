"""Stage 2a benchmark: measure actual eval_exp_tens speedup on the
demo-style tensor-harmonicity workload, near the regime of
demo_triad_consonance.py (harmonic-24 template, dup=3, r=3 rel;
sigma = 12 here against the demo's 10, and a coarser grid, below).
"""

import time

import numpy as np

import mpt
from mpt.spectra import add_spectra
from mpt.tensor import build_exp_tens, eval_exp_tens


def main():
    # --- Demo workload setup ---
    spec = ("harmonic", 24, "powerlaw", 1)
    dup = 3
    sigma = 12.0
    r = 3

    # Build the harmonic template - 24 partials, duplicated 3x = 72 partials
    p_root = np.zeros(dup)
    w_root = np.ones(dup)
    tp, tw = add_spectra(p_root, w_root, *spec)
    print(f"Template: {tp.size} partials in [{tp.min():.0f}, {tp.max():.0f}] cents")

    # Build the density (r=3 rel, non-periodic)
    dens = build_exp_tens(
        tp, tw, sigma, r, True, False, 0.0, verbose=False,
    )
    print(f"Density: n_j = {dens.n_j}")

    # Query grid: upper-triangle of (0..1200) x (0..1200).
    # The demo uses step=10, but the exact path is slow (~1 hour on this workload);
    # benchmark at step=20 for a tractable comparison. Truncation speedup
    # scales identically.
    step = 20
    ints = np.arange(0, 1201, step)
    I1, I2 = np.meshgrid(ints, ints, indexing='ij')
    mask = I1 <= I2
    queries = np.stack([I1[mask], I2[mask]], axis=0).astype(np.float64)
    print(f"Queries: {queries.shape[1]} upper-triangle points at {step}-cent step\n")

    # --- Benchmark: force centres path; vary truncation ---
    print(f"{'Mode':<40s} {'wall (s)':>10s} {'speedup':>10s}")
    print("-" * 65)

    # Reference: the exact, untruncated computation. This is not the
    # default (truncation_sigmas defaults to 6), so it is asked for.
    t0 = time.perf_counter()
    v_ref = eval_exp_tens(dens, queries, method='centres',
                          truncation_sigmas=float('inf'), verbose=False)
    t_ref = time.perf_counter() - t0
    print(f"{'exact (truncationSigmas=Inf)':<40s} {t_ref:>10.3f} {1.0:>10.2f}x")

    # Truncated at k=5, 6, 7
    for k in [5, 6, 7]:
        t0 = time.perf_counter()
        v_trunc = eval_exp_tens(
            dens, queries, method='centres',
            truncation_sigmas=k, verbose=False,
        )
        t_trunc = time.perf_counter() - t0
        speedup = t_ref / t_trunc if t_trunc > 0 else float('inf')
        # Verify accuracy
        weight_mass = float(np.sum(np.abs(dens.w_j)))
        err = float(np.max(np.abs(v_trunc - v_ref)))
        print(
            f"{'truncationSigmas=' + str(k):<40s} "
            f"{t_trunc:>10.3f} {speedup:>9.1f}x  "
            f"max abs err {err:.2e}"
        )

    # Single precision (no truncation)
    t0 = time.perf_counter()
    v_single = eval_exp_tens(
        dens, queries, method='centres', truncation_sigmas=float('inf'),
        kernel_precision='single', verbose=False,
    )
    t_single = time.perf_counter() - t0
    speedup = t_ref / t_single if t_single > 0 else float('inf')
    err = float(np.max(np.abs(v_single - v_ref)))
    print(
        f"{'kernelPrecision=single (no trunc)':<40s} "
        f"{t_single:>10.3f} {speedup:>9.1f}x  "
        f"max abs err {err:.2e}"
    )

    # Combined: truncation=6 + single
    t0 = time.perf_counter()
    v_both = eval_exp_tens(
        dens, queries, method='centres',
        truncation_sigmas=6, kernel_precision='single', verbose=False,
    )
    t_both = time.perf_counter() - t0
    speedup = t_ref / t_both if t_both > 0 else float('inf')
    err = float(np.max(np.abs(v_both - v_ref)))
    print(
        f"{'k=6 + single':<40s} "
        f"{t_both:>10.3f} {speedup:>9.1f}x  "
        f"max abs err {err:.2e}"
    )

    print("\nNote: weight_mass =", float(np.sum(np.abs(dens.w_j))))
    print("      v_ref peak/min:", v_ref.max(), v_ref.min())


if __name__ == "__main__":
    main()
