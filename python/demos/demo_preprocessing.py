"""demo_preprocessing -- Pre-MAET preprocessing operations and compositions.

Demonstrates the pre-MAET preprocessing helpers in MPT, applied to a
fragment of J. S. Bach, BWV 347 ("Ich dank dir, lieber Herre"): the
soprano over quarter-notes t = 1 to 7, which is bar 1 entire followed by
cadence 1's three-chord approach (antepenult i, penult V, tonic I at
t = 5, 6, 7), so the fragment ends on its cadential goal. Two attributes
are kept: the soprano pitch (attribute 0, treated as periodic mod 12 so
it lives on the pitch-class circle) and the event time in quarter-notes
(attribute 1, non-periodic).

The events carry metrical weights rather than uniform ones, so that each
operation's weight rule is visible in its output rather than described:
the chorale is in 4/4 with a one-quarter pickup, so t = 1 and t = 5 fall
on the downbeat (weight 1), t = 3 and t = 7 on the third beat (0.75),
and t = 2, 4, 6 on the weak beats (0.5).

Every operation's result is displayed with ``show_pre_maet``, which
prints a pre-MAET in the layout of the article's tables: brace-delimited
cells where the attribute is unordered, parentheses where it is ordered,
brackets within brackets where it is nested, and weights as
parenthesized superscripts.

Operations
    difference_events    (D): per-attribute difference orders.
    bind_events          (B): per-attribute bind orders (n-gram
                              expansion).
    translate_attributes (T): per-attribute translation of values.
    weight_events        (W): per-event window (one factor per
                              input attribute, peak-normalised
                              fixed-variance family).
    transform_attributes (F): per-attribute elementwise maps
                              (log, scale conversion, user function)
                              and the sign attribute.

Compositions
    D o B == B o D       (n-tuple entropy pipeline commutation,
                          value-wise and weight-wise after attribute
                          permutation).
    D o T == D           (differencing absorbs absolute translation;
                          T o D adds mu to every difference).
    T o W centre shift   (W with centre c after T(mu) equals W with
                          centre c - mu before T; W leaves T
                          invariant on values).

The MATLAB mirror is demos/demo_preprocessing.m.

See also: show_pre_maet, difference_events, bind_events,
translate_attributes.
"""
import numpy as np

import mpt

# Shown on every table, so that the parameters that would build the
# density travel with the values they would be built from.
KERNEL = dict(sigma=[0.5, 0.25], is_per=[True, False], period=[12.0, 0.0],
              names=['pitch', 'time'])


# ===================================================================
#  1. BWV 347 input (soprano + time, metrically weighted)
# ===================================================================

print("=== 1. Inputs (BWV 347, soprano, t = 1..7) ===")

# Two attributes, both K_a = 1, seven events.
p_attr = [
    np.array([[69, 69, 69, 71, 67, 66, 64]], dtype=float),  # a_0: soprano
    np.array([[ 1,  2,  3,  4,  5,  6,  7]], dtype=float),  # a_1: time (QN)
]
# Metrical weight: downbeat 1, third beat 0.75, weak beats 0.5. The same
# weighting is carried on both attributes, since a metrically weak event
# is weak in every attribute it carries.
metre = np.array([[1.0, 0.5, 0.75, 0.5, 1.0, 0.5, 0.75]])
w = [metre.copy(), metre.copy()]

# The three parts travel together as one pre-MAET, which every operator
# below takes whole and returns whole.
pm = mpt.pre_maet(p_attr, w)

is_rel  = [False, False]      # both attributes are absolute
is_per  = [True, False]       # attribute 0 is periodic (PC), attribute 1 isn't
periods = [12.0, 0.0]         # period 12 (semitones) for PC

mpt.show_pre_maet(pm, **KERNEL)
print()


# ===================================================================
#  2. difference_events (D): turn pitches into intervals
# ===================================================================

print("=== 2. difference_events (D) ===")

# Take the first difference of pitch and leave time alone. The first
# event is dropped (leading-drop alignment): N' = N - max(k) = 6.
# Weights propagate as the rolling product w'(n) = prod_{j=0..k} w(n-j),
# the probability that the k+1 contributing events are jointly
# perceived, so a difference is only as strong as its weaker endpoint
# allows: the superscripts below are the products of consecutive metre
# weights on the differenced attribute, and the surviving metre weights
# on the undifferenced one.
diff_orders = [1, 0]
pmD = mpt.difference_events(pm, diff_orders)

print(f"  diff_orders = {diff_orders}   (pitch differenced, time left alone)")
mpt.show_pre_maet(pmD, **KERNEL)
print()


# ===================================================================
#  3. bind_events (B): expand attributes into 2-grams
# ===================================================================

print("=== 3. bind_events (B) ===")

# Bind 2 consecutive events into 2-grams on both attributes. Each source
# attribute becomes ONE nested attribute: the two bound events form the
# ordered outer level, each event's own value the inner level (inheriting
# the source r/is_rel/is_sym). A' = A = 2. Trailing-drop alignment gives
# N' = N - max(L) + 1 = 6. B gathers each super-event's constituent
# weights alongside its values rather than combining them, so both of a
# 2-gram's metre weights survive, in order, inside the cell.
bind_orders = [2, 2]
pmB = mpt.bind_events(pm, bind_orders)
specB = pmB["specs"]

print(f"  bind_orders = {bind_orders}   "
      f"(A' = {len(pmB['p_attr'])}: each source attribute -> one nested "
      f"attribute)")
mpt.show_pre_maet(pmB, **KERNEL)
s0 = specB[0]
print(f"  spec[0]: r = {s0['r']}, sym = {s0['sym']}, rel = {s0['rel']}, "
      f"tags = {np.asarray(s0['tags']).ravel().tolist()}")
print()


# ===================================================================
#  3b. bind_events again (B o B): deepen a nested attribute to L = 3
# ===================================================================

print("=== 3b. bind_events again (B o B): deepen to L = 3 ===")

# bind_events accepts the (p_attr, w, specs) triple it produces, so a
# second bind deepens the *already-nested* attribute rather than starting
# over. The existing tag matrix is tiled and a fresh outermost grouping
# column is appended; r/sym/rel each gain one outer level. The hierarchy
# grows note -> 2-event group (first bind) -> 2-group window (second
# bind). Trailing-drop again: N'' = N' - max(L) + 1 = 5.
pmBB = mpt.bind_events(pmB, [2, 2])
specBB = pmBB["specs"]

sBB = specBB[0]
print(f"  N'' = {pmBB['p_attr'][0].shape[1]}  (three-level super-events)")
mpt.show_pre_maet(pmBB, max_events=4, **KERNEL)
print(f"  spec[0]: r = {sBB['r']}, sym = {sBB['sym']}, rel = {sBB['rel']}")
print("  (inner tag column tiled; a new outermost column appended.)")

# Build the absolute L=3 nest and confirm a clean self-similarity.
# The specs here carry no kernel geometry, so these arguments supply it;
# where a spec does carry a value, an argument overrides it instead, which
# is what makes a sweep one call per value (demo_pre_maet_io, section 4).
kwBB = dict(sigma=[0.5, 0.25], is_per=[True, False], period=[12.0, 0.0],
            verbose=False)
dBB_abs = mpt.build_exp_tens(pmBB, **kwBB)
sm_abs = float(mpt.cos_sim_exp_tens(dBB_abs, dBB_abs, verbose=False))
print(f"  absolute build: dim = {dBB_abs.dim}, "
      f"cosine self-match = {sm_abs:.4f}")

# Outermost [rel] on pitch quotients the whole 3-level tuple by a common
# shift: the doubly-bound pitch structure is then invariant to transposing
# every note together. Absolute (no [rel]) is not --- the narrow PC kernel
# (sigma = 0.5) puts a 5-semitone shift out of reach.
# Replacing one part of a pre-MAET leaves the rest in place: here the
# specs, and below the values.
specBB_out = [dict(specBB[0]), dict(specBB[1])]
specBB_out[0]["rel"] = [0, 0, 1]                 # outermost unit on pitch
pmBB_out = mpt.pre_maet(pmBB, specs=specBB_out)
dBB_out = mpt.build_exp_tens(pmBB_out, **kwBB)
pmBB_T = mpt.pre_maet([pmBB["p_attr"][0] + 5.0, pmBB["p_attr"][1]],
                      pmBB["w_attr"], specBB)   # transpose all pitches +5
pmBB_out_T = mpt.pre_maet(pmBB_T, specs=specBB_out)
dBB_out_T = mpt.build_exp_tens(pmBB_out_T, **kwBB)
dBB_abs_T = mpt.build_exp_tens(pmBB_T, **kwBB)
sim_out = float(mpt.cos_sim_exp_tens(dBB_out, dBB_out_T, verbose=False))
sim_abs = float(mpt.cos_sim_exp_tens(dBB_abs, dBB_abs_T, verbose=False))
print(f"  outer pitch (rel=[0,0,1]): dim = {dBB_out.dim}, "
      f"vs +5 transpose = {sim_out:.4f}  (global-transposition invariant)")
print(f"  absolute pitch:            vs +5 transpose = {sim_abs:.4f}  "
      "(not invariant)\n")


# ===================================================================
#  4. translate_attributes (T): transpose pitch up a perfect fourth
# ===================================================================

print("=== 4. translate_attributes (T) ===")

# Translate pitch (attribute 0) by +5 semitones; leave time alone.
# Offsets are a per-attribute list: a scalar broadcasts across the
# attribute's values (here K=1 each). is_rel is read from specs
# (synthesised flat: both absolute), so neither is a no-op. T moves
# values only: the weights below are the metre weights unchanged.
mu_pitch = 5.0
mu = [mu_pitch, 0.0]
pmT = mpt.translate_attributes(pm, mu)

print(f"  mu (per attribute) = {mu}   (G->C, F#->B, E->A; time untouched)")
mpt.show_pre_maet(pmT, **KERNEL)
print()


# ===================================================================
#  5. weight_events (W): window the time attribute at the cadence
# ===================================================================

print("=== 5. weight_events (W) ===")

# Apply a window on the time axis (input attribute 1) centred at the
# penult event (t = 6) with standard deviation 2 quarter-notes and
# gamma = 0 (pure Gaussian). The factor lands back on the time attribute
# (target attribute 1), the in-place weighting case, and the input is
# kept (drop_input_attr=False). The window multiplies the metre weights
# it finds rather than replacing them, so the time row below carries
# metre times envelope, and the pitch row is untouched.
pmW = mpt.weight_events(
    pm,
    input_attr=1, target_attr=1,
    centre=6.0, sd=2.0, shape=0.0,    # gamma = 0 -> pure Gaussian
    drop_input_attr=False,
)

print("  input_attr = 1 (time); target_attr = 1; centre = 6; sd = 2; "
      "shape = 0 (Gaussian)")
mpt.show_pre_maet(pmW, decimals=3, **KERNEL)
print()


# ===================================================================
#  6. D o B == B o D (n-tuple entropy pipeline commutation)
# ===================================================================

print("=== 6. B o D == D o B (pipeline commutation) ===")

# Both pre-MAET operators speak the (p_attr, w, specs) triple, so the
# two routes coincide. Differencing pairs values position by position
# across (super-)events and the sliding bind window commutes with it, on
# the ordered/K=1 domain where difference is defined. The two operators
# propagate weights by different rules --- D takes the rolling product,
# B gathers --- and the composition agrees on the weights as well.
#   D then B: difference each attribute (order 1), then bind 2-grams.
pmDB = mpt.bind_events(mpt.difference_events(pm, [1, 1]), [2, 2])
#   B then D: bind 2-grams, then difference each nested attribute
#   position by position.
pmBD = mpt.difference_events(mpt.bind_events(pm, [2, 2]), [1, 1])

mpt.show_pre_maet(pmDB, title="  D then B:", **KERNEL)
mpt.show_pre_maet(pmBD, title="  B then D:", **KERNEL)

vals_agree = all(np.allclose(pmDB["p_attr"][a], pmBD["p_attr"][a],
                             equal_nan=True) for a in range(2))
wts_agree = all(np.allclose(np.asarray(pmDB["w_attr"][a]),
                            np.asarray(pmBD["w_attr"][a]),
                            equal_nan=True) for a in range(2))
specs_agree = all(
    list(np.ravel(pmDB["specs"][a][k]))
    == list(np.ravel(pmBD["specs"][a][k]))
    for a in range(2) for k in ("tags", "r", "sym", "rel")
)
print(f"  values agree: {vals_agree};  weights agree: {wts_agree};  "
      f"specs agree: {specs_agree}")
assert vals_agree and wts_agree and specs_agree, \
    "Section 6: the two routes disagree."
print()


# ===================================================================
#  7. D o T == D (differencing absorbs absolute translation)
# ===================================================================

print("=== 7. D o T == D ===")

# Difference applied to a transposed copy returns the same intervals
# as differencing the original: translation is wiped out by the
# difference operator (T o D, by contrast, adds mu to every
# difference).
pmDT = mpt.difference_events(pmT, [1, 0])

mpt.show_pre_maet(pmDT, title="  D(T(p)):", **KERNEL)
print(f"  vs D(p) above: max |difference| = "
      f"{float(np.max(np.abs(pmDT['p_attr'][0] - pmD['p_attr'][0])))}"
      "  (zero: translation absorbed)")
print()


# ===================================================================
#  8. T o W centre-shift rule
# ===================================================================

print("=== 8. T o W centre shift ===")

# Path 1: T(mu) first (transposing pitch by +5), then W centred at
# the original pitch c = 69 (A4).
c_pitch  = 69.0
width_w  = 2.0
gamma_w  = 0.3
w_path1 = mpt.weight_events(
    mpt.translate_attributes(pm, [mu_pitch, 0.0]),
    input_attr=0, target_attr=0, centre=c_pitch, sd=width_w, shape=gamma_w,
    drop_input_attr=False,
)["w_attr"]

# Path 2: W centred at c - mu = 64 BEFORE T (T leaves weights
# untouched).
w_path2 = mpt.weight_events(
    pm,
    input_attr=0, target_attr=0, centre=c_pitch - mu_pitch, sd=width_w,
    shape=gamma_w, drop_input_attr=False,
)["w_attr"]

print(f"  T then W (centre c = {c_pitch}):")
print(f"    w_path1[0] = "
      f"{[round(v, 4) for v in np.asarray(w_path1[0]).ravel().tolist()]}")
print(f"  W (centre c - mu = {c_pitch - mu_pitch}) before T:")
print(f"    w_path2[0] = "
      f"{[round(v, 4) for v in np.asarray(w_path2[0]).ravel().tolist()]}")
print("  difference max = "
      f"{float(np.max(np.abs(np.asarray(w_path1[0]) - np.asarray(w_path2[0]))))}  "
      "(zero --- centre-shift rule holds)")

# ===================================================================
#  8b. transform_attributes: the measurement scale, and its order with D
# ===================================================================

print("\n=== 8b. transform_attributes (F): scale choice and order with D ===")

# The kernel of build_exp_tens has a fixed width in whatever units the
# values carry, so the choice of scale is made before the tensor. The
# bare-array form converts a vector in one call (this replaces the
# former convert_pitch):
f_hz = np.array([392.00, 369.99, 329.63])          # G4, F#4, E4 in Hz
p_cents = mpt.transform_attributes(f_hz, None, ('hz', 'cents'))
print(f"  Hz -> cents: [{p_cents[0]:.1f} {p_cents[1]:.1f} {p_cents[2]:.1f}]")

# Order with differencing carries meaning. (i) F then D on inter-onset
# intervals in log2 gives log ratios: a doubling is +1, a halving -1.
ioi = np.array([[0.25, 0.5, 0.5, 1.0]])            # seconds
pmLD = mpt.difference_events(
    mpt.transform_attributes([ioi], None, [('log', {'base': 2})]), 1)
p_ld = pmLD["p_attr"]
print(f"  log2(IOI) then D: {p_ld[0].ravel()}  (log ratios)")

# (ii) D then a compressive transform on the signed pitch intervals.
# log(x + 1) admits the zero of a repeated note with the constant written
# down. Negative values are refused unless a sign attribute is requested:
# with sign=True the transform is applied to |x| and a sign attribute
# at the 2-point simplex's vertices, {-1/2, 0, +1/2}, is inserted right
# after its source, so the pre-MAET
# grows from one attribute to two (note the two sigmas below).
pmDp = mpt.difference_events([p_attr[0]], [w[0]], 1)
pmF = mpt.transform_attributes(pmDp, [('log', {'offset': 1})], sign=True)
p_dp = pmDp["p_attr"]
p_f, s_f = pmF["p_attr"], pmF["specs"]
print(f"  D(pitch)        = {p_dp[0].ravel()}")
print(f"  log(|D(pitch)|+1) = {np.round(p_f[0].ravel(), 4)}, "
      f"sign = {p_f[1].ravel()} (spec name '{s_f[1]['name']}')")
dens_f = mpt.build_exp_tens(pmF, sigma=[0.2, 0.3],
                            is_per=[False, False], period=[0, 0],
                            verbose=False)
print(f"  build_exp_tens on the two-attribute pre-MAET: dim = {dens_f.dim}")

# (iii) A zero under 'log' is an error with remedies, never -inf.
try:
    mpt.transform_attributes([np.array([[0.5, 0.0, 0.25]])], None, ['log'])
except ValueError as err:
    print(f"  zero IOI under 'log' -> {str(err).split(';')[0]}")

# (iv) A callable is accepted alongside the named transforms.
p_user = mpt.transform_attributes([np.array([[1.0, 4.0, 9.0]])], None,
                                  [lambda x: np.sqrt(x) + 1])["p_attr"]
print(f"  user function sqrt(x) + 1: {p_user[0].ravel()}\n")

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
#        passing (p_attr, w, sigma, r, is_rel, is_per, periods)
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
r     = [1, 1]        # single-value attributes (K_a = 1)

# The raw form takes the parts positionally, so the pre-MAETs above are
# read out into the lists it expects.
pT = pmT["p_attr"]
pD, wD = pmD["p_attr"], pmD["w_attr"]

# --- 9a. entropy_exp_tens (raw MA form) ---
# Signature:
#   H = entropy_exp_tens(p_attr, w, sigma, r, is_rel, is_per, periods, ...)
H_orig = mpt.entropy_exp_tens(
    p_attr, w, sigma, r, is_rel, is_per, periods,
    method="renyi2", verbose=False,
)
print(f"  entropy_exp_tens(p_attr, w, sigma, r, is_rel, is_per, periods)")
print(f"    = {H_orig:.4f}  (Renyi-2)")

# --- 9b. eval_exp_tens at the penult event (pitch = 66, t = 6) ---
# Query points are supplied as a length-A list, one (K_a, M_q) matrix
# per attribute. Single query here, so a (1, 1) column for each of the
# two attributes: pitch = 66, t = 6.
Xq = [np.array([[66.0]]), np.array([[6.0]])]
val_at_penult = mpt.eval_exp_tens(
    p_attr, w, sigma, r, is_rel, is_per, periods, Xq,
    verbose=False,
)
print(f"  eval_exp_tens(p_attr, w, sigma, r, is_rel, is_per, periods, Xq)")
print(f"    = {float(val_at_penult[0]):.4f}")
print("  (Density peak near an actual event; the value reflects the")
print("   contribution from event 6 at (66, 6) plus tails from its neighbours.)")

# --- 9c. cos_sim_exp_tens on two pre-MAETs (raw MA form) ---
# Signature:
#   s = cos_sim_exp_tens(p_X, w_X, p_Y, w_Y, sigma, r,
#                        is_rel, is_per, periods, ...)
# Compare the original chorale fragment against the transposed copy
# (Section 4). Group 0's PC kernel is narrow (sigma = 0.5 semitones),
# so the 5-semitone shift puts every event out of kernel reach of its
# original PC, and the similarity collapses to 0. Pre-MAET D in step
# 9d below recovers it.
sim_T = mpt.cos_sim_exp_tens(
    p_attr, w, pT, w, sigma, r, is_rel, is_per, periods,
    verbose=False,
)
print(f"  cos_sim_exp_tens(p_attr, w, pT, w, sigma, r, is_rel, is_per, periods)")
print(f"    = {float(sim_T):.4f}")

# --- 9d. cos_sim of the differenced pair: D(T) == D identity in action ---
# Section 7's identity D o T == D guarantees that the differenced
# original and the differenced transposed copy are value-wise
# identical, so their cosine similarity must be exactly 1. The
# algebraic identity from Section 7 surfacing as a downstream
# observable; no build_exp_tens required.

sim_diffed = mpt.cos_sim_exp_tens(
    pD, wD, pmDT["p_attr"], pmDT["w_attr"],
    sigma, r, is_rel, is_per, periods,
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
    p_attr, w, sigma, r, is_rel, is_per, periods, verbose=False,
)
dens_T = mpt.build_exp_tens(
    pT, w, sigma, r, is_rel, is_per, periods, verbose=False,
)
dens_D = mpt.build_exp_tens(
    pD, wD, sigma, r, is_rel, is_per, periods, verbose=False,
)

dens_DT = mpt.build_exp_tens(
    pmDT["p_attr"], pmDT["w_attr"], sigma, r, is_rel, is_per, periods,
    verbose=False,
)

# --- 10a. entropy_exp_tens on the struct; same answer as 9a. ---
H_orig_dens = mpt.entropy_exp_tens(
    dens_orig, method="renyi2", verbose=False,
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
