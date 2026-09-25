"""Benchmark of the kernel-evaluation controls, truncation_sigmas and
kernel_precision, on the centres path of eval_maet, for the
tensor-harmonicity workload of demo_triad_consonance.py (harmonic-24
template, dup=3, r=3 relative, at the demo's 10-cent grid; sigma = 12
here against the demo's 10).

Not a demo: a development benchmark. Run from the python/ folder:
    python tools/bench_stage2a.py
"""

import time

import numpy as np

import mpt

LABEL_SINGLE = "kernel_precision='single', truncation_sigmas=inf"
LABEL_BOTH = "kernel_precision='single', truncation_sigmas=6"


def main():
    # Routing announcements would interleave with the table.
    mpt.set_default(show_hints=False)
    # --- Demo workload setup ---
    spec = ("harmonic", 24, "powerlaw", 1)
    dup = 3
    sigma = 12.0
    r = 3

    # Build the harmonic template - 24 partials, duplicated 3x = 72 partials
    p_root = np.zeros(dup)
    w_root = np.ones(dup)
    tp, tw = mpt.add_spectra(p_root, w_root, *spec)
    print(f"Template: {tp.size} partials in [{tp.min():.0f}, {tp.max():.0f}] cents")

    # Build the density (r=3 rel, non-periodic)
    dens = mpt.build_maet(
        tp, tw, sigma, r, True, False, 0.0, verbose=False,
    )
    print(f"Density: n_j = {dens.n_j}")

    # Query grid: upper triangle of (0..1200) x (0..1200), at the demo's step.
    step = 10
    ints = np.arange(0, 1201, step)
    I1, I2 = np.meshgrid(ints, ints, indexing='ij')
    mask = I1 <= I2
    queries = np.stack([I1[mask], I2[mask]], axis=0).astype(np.float64)
    print(f"Queries: {queries.shape[1]} upper-triangle points at {step}-cent step\n")

    # --- Benchmark: force centres path; vary truncation ---
    print(f"{'Mode':<52s} {'wall (s)':>10s} {'speedup':>10s}")
    print("-" * 77)

    # Reference: the accuracy floor. truncation_sigmas=inf resolves to the
    # finite width (~7.43 sigma) at which the kernel falls to 1e-12 of its
    # peak; it is not the default (6), so it is asked for.
    t0 = time.perf_counter()
    v_ref = mpt.eval_maet(dens, queries, method='centres',
                          truncation_sigmas=float('inf'), verbose=False)
    t_ref = time.perf_counter() - t0
    print(f"{'truncation_sigmas=inf (accuracy floor)':<52s} {t_ref:>10.3f} {1.0:>10.2f}x")

    # Truncated at k=5, 6, 7
    for k in [5, 6, 7]:
        t0 = time.perf_counter()
        v_trunc = mpt.eval_maet(
            dens, queries, method='centres',
            truncation_sigmas=k, verbose=False,
        )
        t_trunc = time.perf_counter() - t0
        speedup = t_ref / t_trunc if t_trunc > 0 else float('inf')
        # Verify accuracy
        err = float(np.max(np.abs(v_trunc - v_ref)))
        print(
            f"{'truncation_sigmas=' + str(k):<52s} "
            f"{t_trunc:>10.3f} {speedup:>9.1f}x  "
            f"max abs err {err:.2e}"
        )

    # Single precision, at the accuracy floor
    t0 = time.perf_counter()
    v_single = mpt.eval_maet(
        dens, queries, method='centres', truncation_sigmas=float('inf'),
        kernel_precision='single', verbose=False,
    )
    t_single = time.perf_counter() - t0
    speedup = t_ref / t_single if t_single > 0 else float('inf')
    err = float(np.max(np.abs(v_single - v_ref)))
    print(
        f"{LABEL_SINGLE:<52s} "
        f"{t_single:>10.3f} {speedup:>9.1f}x  "
        f"max abs err {err:.2e}"
    )

    # Combined: truncation_sigmas=6 and single precision
    t0 = time.perf_counter()
    v_both = mpt.eval_maet(
        dens, queries, method='centres',
        truncation_sigmas=6, kernel_precision='single', verbose=False,
    )
    t_both = time.perf_counter() - t0
    speedup = t_ref / t_both if t_both > 0 else float('inf')
    err = float(np.max(np.abs(v_both - v_ref)))
    print(
        f"{LABEL_BOTH:<52s} "
        f"{t_both:>10.3f} {speedup:>9.1f}x  "
        f"max abs err {err:.2e}"
    )

    print("\nNote: weight_mass =", float(np.sum(np.abs(dens.w_j))))
    print("      v_ref peak/min:", v_ref.max(), v_ref.min())


if __name__ == "__main__":
    main()
