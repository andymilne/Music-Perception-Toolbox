"""demo_dispatch_and_kernel_controls.py

A tour of the toolbox's performance controls:

  1. Method dispatch -- for flat attributes, Bulger's method and the
     Möbius method (and, for an ordered attribute, the materialized
     tuple centres) are chosen automatically by a cost model. Forcing
     each by hand shows the speed-up that 'auto' gets you, and
     `explain_dispatch` shows why it chose.
  2. Nested attributes -- a bound, spectrally enriched attribute is
     contracted level by level instead of enumerating its nested
     tuples.
  3. Sweeps -- the similarity at many translations of a query is
     computed in one pass by `sweep_sim_maet` (mixture, orbit, or
     contraction route), not one comparison per offset;
     `swept_similarity` takes the same routes automatically.
  4. Kernel truncation -- `truncation_sigmas` skips Gaussian
     contributions beyond k standard deviations from a centre.
  5. Single-precision kernel -- `kernel_precision='single'` casts the
     kernel matrix to float32, at ~7 significant figures.
  6. Toolbox-wide defaults -- `mpt.set_default` sets any of the above,
     and the other defaults, for every later call.
  7. Entropy estimators -- 'shannon' (a grid), 'differential' (adaptive,
     continuous), and 'renyi2' (closed form), and where each applies.

The dispatcher and a kernel truncation at 6 sigma are on by default
(`truncation_sigmas=6` drops contributions below exp(-18), about 1.5e-8
of a kernel's peak); `kernel_precision` defaults to double. Every
control can be set per call or toolbox-wide.

The MATLAB mirror is demo_dispatchAndKernelControls.m.
"""

# ---- user-adjustable parameters ----
N_EVENTS = 20         # number of source events per density
R = 3                 # tuple size
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

dens_x = mpt.build_maet(p_x, w_x, SIGMA, R, False, False, 1200.0, verbose=False)
dens_y = mpt.build_maet(p_y, w_y, SIGMA, R, False, False, 1200.0, verbose=False)


# A larger source set for the centres-path sections (4-6). MATLAB's
# BLAS is so fast at modest scales that the kernel matmul is in the
# tens of ms range, where the fixed-cost overhead of truncation's
# spatial index and the float32 cast can be comparable to the
# variable-cost savings they produce. N=50 (with 1000 query points
# below) pushes the kernel matmul into the hundreds-of-ms range, so
# the savings dominate and the features show clearly. Section 1
# stays at N=20 because that is already enough to make Bulger's
# method look pathological against the Möbius method.
N_BIG = 50
p_big = np.sort(rng.uniform(0, 1200, N_BIG))
w_big = rng.uniform(0.5, 1.5, N_BIG)
dens_big = mpt.build_maet(p_big, w_big, SIGMA, R, False, False, 1200.0, verbose=False)


def nested_melody(n_notes, n_partials, n_bound):
    """A random melody (whole semitones between MIDI 54 and 72, in
    cents) whose pitches are enriched with harmonic partials and bound
    in groups of consecutive notes: one nested pitch attribute, each
    super-event an ordered group of notes, each note a multiset of
    partials."""
    p = np.round(rng.uniform(5400, 7200, n_notes) / 100) * 100
    specs = mpt.flat_specs([p[None, :]], sigma=15.0, is_per=False, period=0.0)
    pm = mpt.pack_pre_maet([p[None, :]], None, specs)
    pm = mpt.add_spectra(pm, 'harmonic', n_partials, 'powerlaw', 1,
                         attribute=0, units=1200.0)
    return mpt.bind_events(pm, [n_bound])


# Each top-level call announces the route it chose (show_hints, Section
# 6). The timings below repeat every call, so the announcements are
# switched off here and explain_dispatch reports the choices instead;
# they are restored at the end.
_prev_hints = mpt.set_default(show_hints=False)


# ===================================================================
#  1. Method dispatch -- flat attributes
# ===================================================================

# Warm-up: flush first-call costs (orbit-table .pkl load, np.einsum_path
# cache priming, function resolution) out of the timed section.
mpt.sim_maet(dens_x, dens_y, method='mobius', verbose=False)
mpt.sim_maet(dens_x, dens_y, method='bulger', verbose=False)

print(f"=== 1. Method dispatch (N={N_EVENTS}, r={R}, sigma={SIGMA}, absolute, non-periodic) ===\n")

t_auto, c_auto = time_call(
    lambda: mpt.sim_maet(dens_x, dens_y, verbose=False),
)
t_bulger, c_bulger = time_call(
    lambda: mpt.sim_maet(dens_x, dens_y, method='bulger', verbose=False),
)
t_mobius, c_mobius = time_call(
    lambda: mpt.sim_maet(dens_x, dens_y, method='mobius', verbose=False),
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
#  2. Nested attributes
# ===================================================================
#
# A nested attribute (here, 3 consecutive notes bound into one
# super-event, each note enriched with 8 partials) has 8^3 = 512 tuples
# per super-event. Bulger's method enumerates every pair of them; the
# contraction reduces the nesting level by level, never forming the
# tuples. 'auto' prices both and takes the contraction.

print(f"\n=== 2. Nested attributes (3 notes bound, 8 partials each) ===\n")

pm_nx = nested_melody(16, 8, 3)
pm_ny = nested_melody(16, 8, 3)
dens_nx = mpt.build_maet(pm_nx, verbose=False)
dens_ny = mpt.build_maet(pm_ny, verbose=False)
mpt.sim_maet(dens_nx, dens_ny, verbose=False)            # warm-up

t_n_auto, c_n_auto = time_call(
    lambda: mpt.sim_maet(dens_nx, dens_ny, verbose=False))
t_n_bulger, c_n_bulger = time_call(
    lambda: mpt.sim_maet(dens_nx, dens_ny, method='bulger', verbose=False))
print(f"  method='auto'   : {t_n_auto*1000:6.1f} ms   cosine = {c_n_auto:.10f}")
print(f"  method='bulger' : {t_n_bulger*1000:6.1f} ms   cosine = {c_n_bulger:.10f}")
print("\n  explain_dispatch(dens_nx, dens_ny):")
print(mpt.explain_dispatch(dens_nx, dens_ny))


# ===================================================================
#  3. Sweeps: many translations in one pass
# ===================================================================
#
# Comparing a query with a context at M translations one offset at a
# time costs M builds and M comparisons. sweep_sim_maet computes all M
# at once: the 'mixture' route makes one pass over the tuple pairs and
# evaluates a Gaussian mixture in the offset; the 'orbit' route
# evaluates the Möbius inner product at the shifted values; the
# 'contract' route (densities with a nested attribute) carries the
# offsets through the level-by-level contraction. 'auto' picks. The
# same routes serve swept_similarity wherever the context is the
# same at every translation (see demo_swept_similarity).

print(f"\n=== 3. Sweeps (41 translations of the query) ===\n")

mus = np.arange(-200.0, 200.01, 10.0)


def per_offset_flat():
    """One build and one comparison per offset."""
    return np.array([
        mpt.sim_maet(dens_x, mpt.build_maet(p_y + mu, w_y, SIGMA, R, False,
                                            False, 1200.0, verbose=False),
                     verbose=False)
        for mu in mus])


t_loop, s_loop = time_call(per_offset_flat)
print(f"  flat, r = {R}:")
print(f"    one offset at a time  : {t_loop*1000:7.1f} ms")
for method in ('auto', 'mixture', 'orbit'):
    t_sw, s_sw = time_call(lambda: mpt.sweep_sim_maet(
        dens_x, dens_y, mus[None, :], method=method, verbose=False))
    print(f"    sweep, {method!r:<13s} : {t_sw*1000:7.1f} ms   "
          f"max |diff| = {np.max(np.abs(s_sw - s_loop)):.1e}")
print("    (The mixture route enumerates and stores every pair of tuples,")
print(f"     [C({N_EVENTS}, {R}) {R}!]^2 of them here, so it is the slowest at")
print("     this size; 'auto' prices the routes and avoids it.)")


def per_offset_nested():
    out = []
    for mu in mus:
        pm_t = mpt.translate_attributes(pm_ny, [mu])
        out.append(mpt.sim_maet(dens_nx, mpt.build_maet(pm_t, verbose=False),
                                verbose=False))
    return np.array(out)


t_nloop, s_nloop = time_call(per_offset_nested)
t_nsw, s_nsw = time_call(lambda: mpt.sweep_sim_maet(
    dens_nx, dens_ny, mus[None, :], verbose=False))
print(f"  nested (Section 2):")
print(f"    one offset at a time  : {t_nloop*1000:7.1f} ms")
print(f"    sweep, 'auto'         : {t_nsw*1000:7.1f} ms   "
      f"max |diff| = {np.max(np.abs(s_nsw - s_nloop)):.1e}   (the contraction route)")


# ===================================================================
#  4. Kernel truncation
# ===================================================================
#
# truncation_sigmas applies on every route (Bulger's method, the
# Möbius method, and the centres path alike); it is timed here on the
# centres path of eval_maet, where the kernel over tuple centres is the
# dominant cost. The default is 6 sigma; inf gives the accuracy floor,
# the finite width (~7.43 sigma) at which the kernel falls to 1e-12 of
# its peak, rather than an unbounded sum.

print(f"\n=== 4. Kernel truncation (N={N_BIG}, eval at 1000 query points) ===\n")

queries = np.sort(rng.uniform(0, 1200, (R, 1000)), axis=0)


def max_abs_err(v, ref):
    """Max abs deviation, normalized by ref's peak magnitude.

    Robust against near-zero values where relative error explodes.
    """
    scale = float(np.max(np.abs(ref))) + 1e-30
    return float(np.max(np.abs(v - ref))) / scale

t_no_trunc, v_no_trunc = time_call(
    lambda: mpt.eval_maet(dens_big, queries, method='centres',
                              truncation_sigmas=float('inf'), verbose=False),
)
t_k6, v_k6 = time_call(
    lambda: mpt.eval_maet(dens_big, queries, method='centres',
                              truncation_sigmas=6, verbose=False),
)
t_k4, v_k4 = time_call(
    lambda: mpt.eval_maet(dens_big, queries, method='centres',
                              truncation_sigmas=4, verbose=False),
)

rel_err_k6 = max_abs_err(v_k6, v_no_trunc)
rel_err_k4 = max_abs_err(v_k4, v_no_trunc)

print(f"  truncation_sigmas=inf  : {t_no_trunc*1000:6.1f} ms  (accuracy floor, the reference)")
print(f"  truncation_sigmas=6    : {t_k6*1000:6.1f} ms  peak-normalized err = {rel_err_k6:.2e}  (the default)")
print(f"  truncation_sigmas=4    : {t_k4*1000:6.1f} ms  peak-normalized err = {rel_err_k4:.2e}")
print(f"  (Truncating at k sigmas drops kernel contributions below exp(-k^2/2).")
print(f"   k=6 ~ exp(-18) ~ 1.5e-8; k=4 ~ exp(-8) ~ 3e-4.)")


# ===================================================================
#  5. Single-precision kernel
# ===================================================================

print(f"\n=== 5. kernel_precision ===\n")

t_double, v_double = time_call(
    lambda: mpt.eval_maet(dens_big, queries, method='centres',
                              kernel_precision='double', verbose=False),
)
t_single, v_single = time_call(
    lambda: mpt.eval_maet(dens_big, queries, method='centres',
                              kernel_precision='single', verbose=False),
)

rel_err_single = max_abs_err(v_single, v_double)

print(f"  kernel_precision='double' : {t_double*1000:6.1f} ms  (reference)")
print(f"  kernel_precision='single' : {t_single*1000:6.1f} ms  peak-normalized err = {rel_err_single:.2e}")
print("  (The speed-up depends on the workload and the platform, and is")
print("   negligible where the computation is bound by memory bandwidth.")
print("   Precision retained: about 7 significant figures, against about 15.)")


# ===================================================================
#  6. Toolbox-wide defaults
# ===================================================================

print(f"\n=== 6. Toolbox-wide defaults ===\n")

# What each default controls:
#   truncation_sigmas     kernel truncation radius (Section 4).
#   kernel_precision      kernel-matrix arithmetic (Section 5).
#   show_hints            console messages: each call's routing decision
#                         and one-time tips. Off in this demo (switched
#                         off at the top of the script).
#   kernel_chunk_bytes    the memory budget per chunk of a kernel
#                         computation; 'auto' takes half the available
#                         physical memory, or give a byte count.
#   kernel_threads        (Python) threads for the kernel sums; 'auto'
#                         uses the machine's cores.
#   post_hoc_guards       checks that inspect a route's result and may
#                         recompute it by another route; switch off only
#                         for calibration runs.
#   orbit_cost_intercept, rel_attr_route
#                         calibration levers for the cost model, not
#                         part of the public interface.
print(f"  Current defaults: {mpt.get_defaults()}")
print("  Setting global: truncation_sigmas=4, kernel_precision='single'")
prev = mpt.set_default(truncation_sigmas=4, kernel_precision='single')
print(f"  New defaults:     {mpt.get_defaults()}")

t_global, _ = time_call(
    lambda: mpt.eval_maet(dens_big, queries, method='centres', verbose=False),
)
print(f"  eval with global defaults active : {t_global*1000:6.1f} ms")

# Per-call kwargs always override the global defaults: here to the
# accuracy-floor, double-precision computation.
t_override, _ = time_call(
    lambda: mpt.eval_maet(dens_big, queries, method='centres',
                              truncation_sigmas=float('inf'),
                              kernel_precision='double', verbose=False),
)
print(f"  per-call override to inf/double : {t_override*1000:6.1f} ms")

mpt.set_default(**prev)  # restore
print(f"  Restored; defaults now: {mpt.get_defaults()}")


# ===================================================================
#  7. Entropy estimators
# ===================================================================
#
# 'shannon' is the discrete entropy of the density's masses on a grid
# of cells; 'differential' is the continuous differential entropy,
# refined on an adaptive nested grid until it converges to the accuracy
# truncation_sigmas sets; 'renyi2' is the continuous Rényi-2 entropy in
# closed form, with no grid. The comparison is across the density's
# dimension, r = 1, 2, 3 (absolute, so the dimension is r): both grids
# grow as their resolution to the power of the dimension, while the
# closed form's cost barely changes. At one dimension all three are
# cheap and Rényi-2 has no speed advantage; it is at two and three
# that the difference shows.

print(f"\n=== 7. Entropy estimators, by dimension ===\n")

mpt.entropy_maet(dens_x, method='renyi2', verbose=False)   # warm-up
refused = False
print("  r   'shannon' (100 cells/dim)   'differential' (adaptive)"
      "          'renyi2' (closed form)")
for r_e in (1, 2, 3):
    dens_e = mpt.build_maet(p_x, w_x, SIGMA, r_e, False, False, 1200.0,
                            verbose=False)
    # Shannon: 100^r cells; at r = 3 a million, some tens of seconds.
    if r_e < 3:
        t_sh, h_sh = time_call(lambda: mpt.entropy_maet(
            dens_e, method='shannon', x_min=0.0, x_max=1200.0,
            n_points_per_dim=100, verbose=False))
        sh = f"{t_sh*1000:8.1f} ms  H = {h_sh:6.2f}"
    else:
        sh = "  (10^6 cells: not run)"
    # Differential: from two dimensions the adaptive grid needed at the
    # default accuracy (truncation_sigmas = 6) can exceed the feasible
    # size, depending on the data; the call then refuses with guidance,
    # and a coarser accuracy converges. At three it is not attempted.
    if r_e < 3:
        try:
            t_di, h_di = time_call(lambda: mpt.entropy_maet(
                dens_e, method='differential', verbose=False))
            di = f"{t_di*1000:8.1f} ms  h = {h_di:6.2f}"
        except ValueError:
            refused = True
            t_di, h_di = time_call(lambda: mpt.entropy_maet(
                dens_e, method='differential', truncation_sigmas=4,
                verbose=False))
            di = f"{t_di*1000:8.1f} ms  h = {h_di:6.2f} (sigmas = 4)*"
    else:
        di = "  (not attempted)"
    t_r2, h_r2 = time_call(lambda: mpt.entropy_maet(
        dens_e, method='renyi2', verbose=False))
    print(f"  {r_e}   {sh:<28s}{di:<34s}{t_r2*1000:8.1f} ms  h_2 = {h_r2:6.2f}")
print("  (Entropies in bits. Shannon is discrete, so its value depends on")
print("   the cell size; the two differential entropies are continuous,")
print("   with h_2 <= h.)")
if refused:
    print("  (* The default accuracy was refused, and the value shown is at")
    print("   truncation_sigmas = 4.)")


mpt.set_default(**_prev_hints)
print("\n=== Done. See USER_GUIDE.md sec. 5 (Method selection; Kernel-evaluation")
print("    controls; Toolbox defaults API; Four-method entropy API and adaptive")
print("    differential entropy). ===")
