"""demo_preprocessing -- Pre-MAET preprocessing operations and compositions.

Demonstrates the four per-event preprocessing helpers in MPT,
applied to a small fragment of J. S. Bach, BWV 347 ("Ich dank dir,
lieber Herre"). The fragment is cadence 1's three-chord approach
(antepenult i, penult V, tonic I, at quarter-note positions
t = 5, 6, 7) reduced to the soprano line for clarity. Two
attributes are kept: the soprano pitch (group 0, treated as periodic
mod 12 so it lives on the pitch-class circle) and the event time in
quarter-notes (group 1, non-periodic). Each subsequent section
illustrates one operation or one composition; the operations leave
the source ``p_attr`` untouched.

Operations
    difference_events    (D): per-attribute difference orders.
    bind_events          (B): per-attribute bind orders (n-gram
                              expansion).
    translate_attributes (T): per-group translation of values.
    weight_events        (W): per-event window via Design P (one
                              factor per input attribute, peak-
                              normalised fixed-variance family).

Compositions
    D o B == B o D       (n-tuple entropy pipeline commutation,
                          value-wise after attribute permutation).
    D o T == D           (differencing absorbs absolute translation;
                          T o D adds mu to every difference).
    T o W centre shift   (W with centre c after T(mu) equals W with
                          centre c - mu before T; W leaves T
                          invariant on values).

The MATLAB mirror is demos/demo_preprocessing.m.

See also: difference_events, bind_events, translate_attributes,
weight_events.
"""
import numpy as np

import mpt


# ===================================================================
#  1. BWV 347 cadence 1 input (soprano + time)
# ===================================================================

print("=== 1. Inputs (BWV 347 cadence 1 soprano, three-chord approach) ===")

# Two attributes, both K_a = 1, three events.
p_attr = [
    np.array([[67, 66, 64]], dtype=float),   # a_0: soprano (G4, F#4, E4)
    np.array([[ 5,  6,  7]], dtype=float),   # a_1: event time (quarter-notes)
]
w       = None                # weights: default (uniform)
groups  = [0, 1]              # attribute a -> group g(a)
is_rel  = [False, False]      # both groups are absolute
is_per  = [True, False]       # group 0 is periodic (PC), group 1 isn't
periods = [12.0, 0.0]         # period 12 (semitones) for PC

print(f"  pitch (a_0):  {p_attr[0].ravel().tolist()}")
print(f"  time  (a_1):  {p_attr[1].ravel().tolist()}")
print(f"  group 0 (PC):   periodic, P = 12")
print(f"  group 1 (time): non-periodic\n")


# ===================================================================
#  2. difference_events (D): turn pitches into intervals
# ===================================================================

print("=== 2. difference_events (D) ===")

# Take the first difference of pitch and leave time alone. The first
# event is dropped (leading-drop alignment): N' = N - max(k) = 2.
diff_orders = [1, 0]
pD, wD, gD = mpt.difference_events(p_attr, w, groups, diff_orders)

print(f"  diff_orders = {diff_orders}")
print(f"  D(pitch)    = {pD[0].ravel().tolist()}   "
      f"(interval sequence: F#-G, E-F#)")
print(f"  D(time)     = {pD[1].ravel().tolist()}   "
      f"(unchanged in value, leading event dropped)\n")


# ===================================================================
#  3. bind_events (B): expand attributes into 2-grams
# ===================================================================

print("=== 3. bind_events (B) ===")

# Bind 2 consecutive events into 2-grams on both attributes. Each
# source attribute yields L_a = 2 super-attributes (the two slots of
# the 2-gram), so A' = sum_a L_a = 4. Trailing-drop alignment gives
# N' = N - max(L) + 1 = 2.
bind_orders = [2, 2]
pB, wB, gB = mpt.bind_events(p_attr, w, groups, bind_orders)

print(f"  bind_orders  = {bind_orders}")
print(f"  A' = {len(pB)} (each source attribute expands to L_a = 2 super-attributes)")
print(f"  pitch 2-grams: a_0 slot 0 = {pB[0].ravel().tolist()}; "
      f"slot 1 = {pB[1].ravel().tolist()}")
print(f"  time  2-grams: a_1 slot 0 = {pB[2].ravel().tolist()}; "
      f"slot 1 = {pB[3].ravel().tolist()}\n")


# ===================================================================
#  4. translate_attributes (T): transpose pitch up a perfect fourth
# ===================================================================

print("=== 4. translate_attributes (T) ===")

# Translate pitch (group 0) by +5 semitones; leave time alone.
# Dict form keeps it a single translation: {group: scalar}.
# A length-G list would instead be read as a 2-position sweep
# (broadcast across attributes) under the orientation grammar.
mu_pitch = 5.0
mu = {0: mu_pitch, 1: 0.0}
pT = mpt.translate_attributes(p_attr, groups, mu, is_rel, is_per, periods)

print(f"  mu (per group) = {{0: {mu_pitch}, 1: 0.0}}   (group 0: pitch; group 1: time)")
print(f"  T(pitch)       = {pT[0].ravel().tolist()}   (G->C, F#->B, E->A)")
print(f"  T(time)        = {pT[1].ravel().tolist()}   (unchanged)\n")


# ===================================================================
#  5. weight_events (W): window the time attribute at the penult
# ===================================================================

print("=== 5. weight_events (W) ===")

# Apply a window on the time axis (input attribute 1) centred at the
# penult event (t = 6) with standard deviation 1 quarter-note and
# gamma = 0 (pure Gaussian).
w_out = mpt.weight_events(
    p_attr, w, groups,
    input_attrs=[1],
    centre=[6.0],
    width=[1.0],
    shape=[0.0],         # gamma = 0 -> pure Gaussian
    is_per=[False],
    periods=[0.0],
)

print(f"  input_attrs = [1] (time); centre = [6]; width = [1]; shape = [0] (Gaussian)")
print(f"  w_out[0] (pitch, untouched): {w_out[0]}")
print(f"  w_out[1] (time, windowed):   "
      f"{[round(v, 4) for v in w_out[1].ravel().tolist()]}")
print("  (peak at t = 6; falls off symmetrically by exp(-(t-6)^2 / 2).)\n")


# ===================================================================
#  6. D o B == B o D (n-tuple entropy pipeline commutation)
# ===================================================================

print("=== 6. D o B == B o D ===")

# Path 1: D then B.
pD1, wD1, gD1 = mpt.difference_events(p_attr, w, groups, [1, 0])
pDB, wDB, gDB = mpt.bind_events(pD1, wD1, gD1, [2, 2])

# Path 2: B then D (difference group 0 (pitch slots), leave group 1
# (time slots) alone). After B, A' = 4 but G is still 2 (super-
# attributes inherit their source group), so the per-group length-G
# form is the natural spelling.
pB1, wB1, gB1 = mpt.bind_events(p_attr, w, groups, [2, 2])
pBD, wBD, gBD = mpt.difference_events(pB1, wB1, gB1, [1, 0])

max_diff_pitch = max(
    float(np.max(np.abs(pDB[0] - pBD[0]))),
    float(np.max(np.abs(pDB[1] - pBD[1]))),
)
max_diff_time = max(
    float(np.max(np.abs(pDB[2] - pBD[2]))),
    float(np.max(np.abs(pDB[3] - pBD[3]))),
)
print(f"  pitch slots agree to max |.| = {max_diff_pitch}")
print(f"  time  slots agree to max |.| = {max_diff_time}")
print("  (Both routes yield the same value-wise output --- this is the")
print("   commutation property used by the n-tuple entropy pipeline.)\n")


# ===================================================================
#  7. D o T == D (differencing absorbs absolute translation)
# ===================================================================

print("=== 7. D o T == D ===")

# Difference applied to a transposed copy returns the same intervals
# as differencing the original: translation is wiped out by the
# difference operator (T o D, by contrast, adds mu to every
# difference).
pT_for_D = mpt.translate_attributes(p_attr, groups, mu, is_rel, is_per, periods)
pDT, wDT, gDT = mpt.difference_events(pT_for_D, w, groups, [1, 0])

print(f"  D(T(pitch)) = {pDT[0].ravel().tolist()}")
print(f"  D(pitch)    = {pD[0].ravel().tolist()}")
print(f"  difference max = {float(np.max(np.abs(pDT[0] - pD[0])))}  "
      "(zero --- translation absorbed)\n")


# ===================================================================
#  8. T o W centre-shift rule
# ===================================================================

print("=== 8. T o W centre shift ===")

# Path 1: T(mu) first (transposing pitch by +5), then W centred at
# the original pitch c = 67 (G4).
c_pitch  = 67.0
width_w  = 2.0
gamma_w  = 0.3
pT_path  = mpt.translate_attributes(
    p_attr, groups, {0: mu_pitch, 1: 0.0}, is_rel, is_per, periods,
)
w_path1 = mpt.weight_events(
    pT_path, w, groups,
    input_attrs=[0], centre=[c_pitch], width=[width_w], shape=[gamma_w],
    is_per=[False], periods=[0.0],
)

# Path 2: W centred at c - mu = 62 BEFORE T (T leaves weights
# untouched).
w_path2 = mpt.weight_events(
    p_attr, w, groups,
    input_attrs=[0], centre=[c_pitch - mu_pitch], width=[width_w], shape=[gamma_w],
    is_per=[False], periods=[0.0],
)

print(f"  T then W (centre c = {c_pitch}):")
print(f"    w_path1[0] = {[round(v, 4) for v in w_path1[0].ravel().tolist()]}")
print(f"  W (centre c - mu = {c_pitch - mu_pitch}) before T:")
print(f"    w_path2[0] = {[round(v, 4) for v in w_path2[0].ravel().tolist()]}")
print(f"  difference max = {float(np.max(np.abs(w_path1[0] - w_path2[0])))}  "
      "(zero --- centre-shift rule holds)")

# ===================================================================
#  9. Pre-MAET into the raw multi-attribute form (route (ii))
# ===================================================================

print("\n=== 9. Raw form: pre-MAET feeds directly into tensor functions ===")

# Two routes lead from a pre-MAET triple to a density value, an
# entropy, or a similarity:
#
#   (i)  build a MaetDensity once via build_exp_tens, then pass the
#        struct to entropy_exp_tens / eval_exp_tens / cos_sim_exp_tens.
#        Preferred when the same density is re-evaluated many times,
#        because the structural work (group canonicalisation, tuple
#        index pre-computation, weight products) is paid once.
#
#   (ii) call the raw multi-attribute form of each function directly,
#        passing (p_attr, w, sigma, r, groups, is_rel, is_per, periods)
#        as positional arguments. The function builds the density
#        internally and returns the answer; no struct is exposed.
#        Convenient for single-shot uses and keeps the call shape
#        symmetric with build_exp_tens itself.
#
# Section 9 below exercises route (ii) on the original p_attr and on
# the differenced / translated pre-MAETs. Section 10 then exercises
# route (i) on the same set, building each density once via
# build_exp_tens and reusing it across entropy_exp_tens, eval_exp_tens,
# cos_sim_exp_tens, and the LIST form of cos_sim_exp_tens, with parity
# assertions confirming the two routes return identical values.
sigma = [0.5, 0.25]   # kernel std: 0.5 semitones (PC), 0.25 quarter-notes (time)
r     = [1, 1]        # single-slot attributes (K_a = 1) in both groups

# --- 9a. entropy_exp_tens (raw MA form) ---
# Signature:
#   H = entropy_exp_tens(p_attr, w, sigma, r, groups, is_rel, is_per, periods, ...)
H_orig = mpt.entropy_exp_tens(
    p_attr, w, sigma, r, groups, is_rel, is_per, periods,
    method="renyi2", normalize=False, verbose=False,
)
print(f"  entropy_exp_tens(p_attr, w, sigma, r, groups, is_rel, is_per, periods)")
print(f"    = {H_orig:.4f}  (Renyi-2)")

# --- 9b. eval_exp_tens at the penult event (pitch = 66, t = 6) ---
# Query points are (A, M_q) with one column per query and row a
# giving attribute a's value(s). Single query here, so a (2, 1)
# column.
Xq = np.array([[66.0], [6.0]])
val_at_penult = mpt.eval_exp_tens(
    p_attr, w, sigma, r, groups, is_rel, is_per, periods, Xq,
    verbose=False,
)
print(f"  eval_exp_tens(p_attr, w, sigma, r, groups, is_rel, is_per, periods, Xq)")
print(f"    = {float(val_at_penult[0]):.4f}")
print("  (Density peak near an actual event; the value reflects the")
print("   contribution from event 2 at (66, 6) plus tails from its neighbours.)")

# --- 9c. cos_sim_exp_tens on two pre-MAETs (raw MA form) ---
# Signature:
#   s = cos_sim_exp_tens(p_X, w_X, p_Y, w_Y, sigma, r, groups,
#                        is_rel, is_per, periods, ...)
# Compare the original chorale fragment against the transposed copy
# (Section 4). Group 0's PC kernel is narrow (sigma = 0.5 semitones),
# so the 5-semitone shift puts every event out of kernel reach of its
# original PC, and the similarity collapses to 0. Pre-MAET D in step
# 9d below recovers it.
sim_T = mpt.cos_sim_exp_tens(
    p_attr, w, pT, w, sigma, r, groups, is_rel, is_per, periods,
    verbose=False,
)
print(f"  cos_sim_exp_tens(p_attr, w, pT, w, sigma, r, groups, is_rel, is_per, periods)")
print(f"    = {float(sim_T):.4f}")

# --- 9d. cos_sim of the differenced pair: D(T) == D identity in action ---
# Section 7's identity D o T == D guarantees that the differenced
# original and the differenced transposed copy are value-wise
# identical, so their cosine similarity must be exactly 1. The
# algebraic identity from Section 7 surfacing as a downstream
# observable; no build_exp_tens required.
pDT_again, wDT_again, gDT_again = mpt.difference_events(pT, w, groups, [1, 0])
sim_diffed = mpt.cos_sim_exp_tens(
    pD, wD, pDT_again, wDT_again,
    sigma, r, gD, is_rel, is_per, periods,
    verbose=False,
)
print(f"  cos_sim_exp_tens(pD, wD, pD(T), wD(T), ...)")
print(f"    = {float(sim_diffed):.4f}  (exactly 1: D absorbs T)")

# ===================================================================
#  10. Pre-MAET via build_exp_tens dens structs (route (i))
# ===================================================================

print("\n=== 10. Dens form: build once, query many; parity with route (ii) ===")

# Build each pre-MAET into a MaetDensity once. After this the
# structural work --- group canonicalisation, tuple-index
# pre-computation, weight products --- is paid; subsequent
# entropy/eval/cos_sim calls just consume the struct.
dens_orig = mpt.build_exp_tens(
    p_attr, w, sigma, r, groups, is_rel, is_per, periods, verbose=False,
)
dens_T = mpt.build_exp_tens(
    pT, w, sigma, r, groups, is_rel, is_per, periods, verbose=False,
)
dens_D = mpt.build_exp_tens(
    pD, wD, sigma, r, gD, is_rel, is_per, periods, verbose=False,
)
pDT_4, wDT_4, gDT_4 = mpt.difference_events(pT, w, groups, [1, 0])
dens_DT = mpt.build_exp_tens(
    pDT_4, wDT_4, sigma, r, gDT_4, is_rel, is_per, periods, verbose=False,
)

# --- 10a. entropy_exp_tens on the struct; same answer as 9a. ---
H_orig_dens = mpt.entropy_exp_tens(
    dens_orig, method="renyi2", normalize=False, verbose=False,
)
delta_a = abs(float(H_orig_dens) - float(H_orig))
print("  entropy_exp_tens(dens_orig)")
print(f"    = {float(H_orig_dens):.4f}  (Renyi-2; parity vs 9a: |delta| = {delta_a:.2e})")
assert delta_a < 1e-12, "Section 10a: entropy raw and dens forms disagree."

# --- 10b. eval_exp_tens at the same query; same answer as 9b. ---
val_at_penult_dens = mpt.eval_exp_tens(dens_orig, Xq, verbose=False)
delta_b = abs(float(val_at_penult_dens[0]) - float(val_at_penult[0]))
print("  eval_exp_tens(dens_orig, Xq)")
print(f"    = {float(val_at_penult_dens[0]):.4f}  (parity vs 9b: |delta| = {delta_b:.2e})")
assert delta_b < 1e-12, "Section 10b: eval raw and dens forms disagree."

# --- 10c. cos_sim_exp_tens(dens_orig, dens_T); same answer as 9c. ---
sim_T_dens = mpt.cos_sim_exp_tens(dens_orig, dens_T, verbose=False)
delta_c = abs(float(sim_T_dens) - float(sim_T))
print("  cos_sim_exp_tens(dens_orig, dens_T)")
print(f"    = {float(sim_T_dens):.4f}  (parity vs 9c: |delta| = {delta_c:.2e})")
assert delta_c < 1e-12, "Section 10c: cos_sim raw and dens forms disagree."

# --- 10d. cos_sim_exp_tens(dens_D, dens_DT) on the differenced pair; ---
#       same answer as 9d. (Section 7 identity: should be exactly 1.)
sim_diffed_dens = mpt.cos_sim_exp_tens(dens_D, dens_DT, verbose=False)
delta_d = abs(float(sim_diffed_dens) - float(sim_diffed))
print("  cos_sim_exp_tens(dens_D, dens_DT)")
print(f"    = {float(sim_diffed_dens):.4f}  (parity vs 9d: |delta| = {delta_d:.2e})")
assert delta_d < 1e-12, "Section 10d: cos_sim raw and dens forms disagree."

# --- 10e. LIST form: one reference against many candidates. ---
# Scalar-vs-list cos_sim_exp_tens broadcasts dens_orig against each
# candidate in the list, returning a length-n list of similarity
# scalars. Useful for "compare one reference density against many"
# workflows.
#
# Four entries are returned for [dens_orig, dens_T, dens_D, dens_DT]:
#   entry 0:  sim(orig, orig) = 1 by definition.
#   entry 1:  sim(orig, T) -- matches 9c's sim_T.
#   entry 2:  sim(orig, D(orig)) -- new value; how similar the original
#             p_attr is to its first-difference.
#   entry 3:  sim(orig, D(T))   -- Section 7's identity D o T == D
#             forces this to equal entry 2.
sim_list = mpt.cos_sim_exp_tens(
    [dens_orig, dens_T, dens_D, dens_DT], dens_orig, verbose=False,
)
sim_list_vals = [float(v) for v in sim_list]
print("  cos_sim_exp_tens([dens_orig, dens_T, dens_D, dens_DT], dens_orig)")
print("    = [{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*sim_list_vals))
print("    (entry 0: self = 1; entry 1: vs T (= 9c);")
print("     entry 2: vs D(orig); entry 3: vs D(T) -- equals entry 2 by D o T == D.)")
assert abs(sim_list_vals[0] - 1.0)         < 1e-12, "10e: self-similarity not 1."
assert abs(sim_list_vals[1] - float(sim_T)) < 1e-12, "10e: LIST entry 1 != 9c value."
assert abs(sim_list_vals[2] - sim_list_vals[3]) < 1e-12, \
    "10e: D o T == D identity violated (entries 2 and 3 should match)."
