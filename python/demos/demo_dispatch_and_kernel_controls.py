"""demo_dispatch_and_kernel_controls.py

A tour of the toolbox's performance controls:

  1. Per-call method dispatch — Bulger's method vs the Möbius method
     are picked automatically by the cost-model dispatcher. Forcing
     either by hand demonstrates the speed-up that 'auto' gets you
     transparently.
  2. Kernel truncation — `truncation_sigmas` skips Gaussian
     contributions beyond k standard deviations from a centre.
  3. Single-precision kernel — `kernel_precision='single'` casts the
     kernel matrix to float32 for a ~2x speedup at ~7 sig fig
     precision.
  4. Toolbox-wide defaults — `mpt.set_default` lets all of the above
     be flipped globally so user code doesn't need per-call kwargs.
  5. Renyi-2 entropy — `method='renyi2'` on `entropy_exp_tens` for a
     closed-form alternative to the numerical Shannon path.

The dispatcher and a kernel truncation at 6 sigma are on by default
(`truncation_sigmas=6` drops contributions below exp(-18), about 1.5e-8
of a kernel's peak); `kernel_precision` defaults to double. Every
control can be set per call or toolbox-wide.
"""

# ---- user-adjustable parameters ----
N_EVENTS = 20         # number of source events per density
R = 3                 # tensor order
SIGMA = 30.0          # Gaussian uncertainty (cents)
N_REPEATS = 3         # repetitions per timing measurement
RNG_SEED = 0
# ------------------------------------

import time
import numpy as np

import mpt

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(RNG_SEED)


def time_call(fn, repeats=N_REPEATS):
    """Median wall time of `repeats` calls to `fn()`."""
    ts = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), result


# Build two random pitch collections.
p_x = np.sort(rng.uniform(0, 1200, N_EVENTS))
p_y = np.sort(rng.uniform(0, 1200, N_EVENTS))
w_x = rng.uniform(0.5, 1.5, N_EVENTS)
w_y = rng.uniform(0.5, 1.5, N_EVENTS)

dens_x = mpt.build_exp_tens(p_x, w_x, SIGMA, R, False, False, 1200.0, verbose=False)
dens_y = mpt.build_exp_tens(p_y, w_y, SIGMA, R, False, False, 1200.0, verbose=False)


# A larger source set for the centres-path sections (2-4). MATLAB's
# BLAS is so fast at modest scales that the kernel matmul is in the
# tens of ms range, where the fixed-cost overhead of truncation's
# spatial index and the float32 cast can be comparable to the
# variable-cost savings they buy. N=50 (with 1000 query points
# below) pushes the kernel matmul into the hundreds-of-ms range, so
# the savings dominate and the features show clearly. Section 1
# stays at N=20 because that's already enough to make Bulger's
# method look pathological against the Möbius method.
N_BIG = 50
p_big = np.sort(rng.uniform(0, 1200, N_BIG))
w_big = rng.uniform(0.5, 1.5, N_BIG)
dens_big = mpt.build_exp_tens(p_big, w_big, SIGMA, R, False, False, 1200.0, verbose=False)


# ===================================================================
#  1. Method dispatch — Bulger's method vs the Möbius method
# ===================================================================

# Warm-up: flush first-call costs (orbit-table .pkl load, np.einsum_path
# cache priming, function resolution) out of the timed section.
mpt.cos_sim_exp_tens(dens_x, dens_y, method='mobius', verbose=False)
mpt.cos_sim_exp_tens(dens_x, dens_y, method='bulger', verbose=False)

print(f"=== 1. Method dispatch (N={N_EVENTS}, r={R}, sigma={SIGMA}, abs nonper) ===\n")

t_auto, c_auto = time_call(
    lambda: mpt.cos_sim_exp_tens(dens_x, dens_y, verbose=False),
)
t_bulger, c_bulger = time_call(
    lambda: mpt.cos_sim_exp_tens(dens_x, dens_y, method='bulger', verbose=False),
)
t_mobius, c_mobius = time_call(
    lambda: mpt.cos_sim_exp_tens(dens_x, dens_y, method='mobius', verbose=False),
)

print(f"  method='auto'    : {t_auto*1000:6.1f} ms   cosine = {c_auto:.10f}")
print(f"  method='bulger'  : {t_bulger*1000:6.1f} ms   cosine = {c_bulger:.10f}")
print(f"  method='mobius'  : {t_mobius*1000:6.1f} ms   cosine = {c_mobius:.10f}")
print(f"  (bulger and mobius agree to {abs(c_bulger - c_mobius):.2e})")

# explain_dispatch reports the choice 'auto' makes, and why, without
# computing anything.
print("\n  explain_dispatch(dens_x, dens_y):")
print(mpt.explain_dispatch(dens_x, dens_y))


# ===================================================================
#  2. Kernel truncation
# ===================================================================
#
# truncation_sigmas affects the centres path of eval_exp_tens and
# Bulger's method in cos_sim_exp_tens, both of which form a kernel over
# tuple centres. The Möbius method evaluates the same density
# analytically without a centres matrix, so the truncation control
# does not apply to it. The default is 6 sigma; inf gives the exact,
# untruncated computation.

print(f"\n=== 2. Kernel truncation (N={N_BIG}, eval at 1000 query points) ===\n")

queries = np.sort(rng.uniform(0, 1200, (R, 1000)), axis=0)


def max_abs_err(v, ref):
    """Max abs deviation, normalised by ref's peak magnitude.

    Robust against near-zero values where relative error explodes.
    """
    scale = float(np.max(np.abs(ref))) + 1e-30
    return float(np.max(np.abs(v - ref))) / scale

t_no_trunc, v_no_trunc = time_call(
    lambda: mpt.eval_exp_tens(dens_big, queries, method='centres',
                              truncation_sigmas=float('inf'), verbose=False),
)
t_k6, v_k6 = time_call(
    lambda: mpt.eval_exp_tens(dens_big, queries, method='centres',
                              truncation_sigmas=6, verbose=False),
)
t_k4, v_k4 = time_call(
    lambda: mpt.eval_exp_tens(dens_big, queries, method='centres',
                              truncation_sigmas=4, verbose=False),
)

rel_err_k6 = max_abs_err(v_k6, v_no_trunc)
rel_err_k4 = max_abs_err(v_k4, v_no_trunc)

print(f"  truncation_sigmas=inf  : {t_no_trunc*1000:6.1f} ms  (exact reference)")
print(f"  truncation_sigmas=6    : {t_k6*1000:6.1f} ms  peak-normalised err = {rel_err_k6:.2e}  (the default)")
print(f"  truncation_sigmas=4    : {t_k4*1000:6.1f} ms  peak-normalised err = {rel_err_k4:.2e}")
print(f"  (Truncating at k sigmas drops kernel contributions below exp(-k^2/2).")
print(f"   k=6 ~ exp(-18) ~ 1.5e-8; k=4 ~ exp(-8) ~ 3e-4.)")


# ===================================================================
#  3. Single-precision kernel
# ===================================================================

print(f"\n=== 3. kernel_precision ===\n")

t_double, v_double = time_call(
    lambda: mpt.eval_exp_tens(dens_big, queries, method='centres',
                              kernel_precision='double', verbose=False),
)
t_single, v_single = time_call(
    lambda: mpt.eval_exp_tens(dens_big, queries, method='centres',
                              kernel_precision='single', verbose=False),
)

rel_err_single = max_abs_err(v_single, v_double)

print(f"  kernel_precision='double' : {t_double*1000:6.1f} ms  (reference)")
print(f"  kernel_precision='single' : {t_single*1000:6.1f} ms  peak-normalised err = {rel_err_single:.2e}")
print(f"  (Speedup is workload- and platform-dependent: on memory-bandwidth-")
print(f"   bound problems the gain is small even at scale. Python typically")
print(f"   sees ~2x at this size; MATLAB with MKL sees less. Precision")
print(f"   retained: ~7 sig figs vs ~15.)")


# ===================================================================
#  4. Toolbox-wide defaults
# ===================================================================

print(f"\n=== 4. Toolbox-wide defaults ===\n")

print(f"  Current defaults: {mpt.get_defaults()}")
print("  Setting global: truncation_sigmas=4, kernel_precision='single'")
prev = mpt.set_default(truncation_sigmas=4, kernel_precision='single')
print(f"  New defaults:     {mpt.get_defaults()}")

t_global, _ = time_call(
    lambda: mpt.eval_exp_tens(dens_big, queries, method='centres', verbose=False),
)
print(f"  eval with global defaults active : {t_global*1000:6.1f} ms")

# Per-call kwargs always override the global defaults: here to the
# exact, untruncated, double-precision computation.
t_override, _ = time_call(
    lambda: mpt.eval_exp_tens(dens_big, queries, method='centres',
                              truncation_sigmas=float('inf'),
                              kernel_precision='double', verbose=False),
)
print(f"  per-call override to exact/double : {t_override*1000:6.1f} ms")

mpt.set_default(**prev)  # restore
print(f"  Restored; defaults now: {mpt.get_defaults()}")


# ===================================================================
#  5. Renyi-2 differential entropy
# ===================================================================

print(f"\n=== 5. Renyi-2 differential entropy ===\n")

t_shannon, h_shannon = time_call(
    lambda: mpt.entropy_exp_tens(dens_x, method='shannon',
                                 x_min=0.0, x_max=1200.0,
                                 n_points_per_dim=100, verbose=False),
)
t_renyi2, h_renyi2 = time_call(
    lambda: mpt.entropy_exp_tens(dens_x, method='renyi2', verbose=False),
)

print(f"  method='shannon' (numerical grid)  : {t_shannon*1000:7.1f} ms   H  = {h_shannon:.4f}")
print(f"  method='renyi2'  (closed-form)     : {t_renyi2*1000:7.1f} ms   H2 = {h_renyi2:.4f}")
print(f"  (Shannon here is a grid-discretised entropy at 100 cells/dim;")
print(f"   Renyi-2 is a continuous differential entropy. Numerical values")
print(f"   are not directly comparable, but ranking behaviour is similar")
print(f"   and Renyi-2 is far cheaper at high r where the grid would OOM.)")


print("\n=== Done. See USER_GUIDE.md sec.4 for the full method-selection API. ===")
