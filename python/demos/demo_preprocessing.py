"""demo_preprocessing -- Pre-MAET preprocessing operations and compositions.

Demonstrates the pre-MAET preprocessing operations, applied to a
fragment of J. S. Bach, BWV 347 ("Ich dank dir, lieber Herre"): the
soprano over quarter-notes t = 1 to 7, which is bar 1 entire followed by
cadence 1's three-chord approach (antepenult i, penult V, tonic I at
t = 5, 6, 7), so the fragment ends on its cadential goal. Two attributes
are kept: the soprano pitch (attribute 0, treated as periodic mod 12 so
it lives on the pitch-class circle) and the event time in quarter-notes
(attribute 1, non-periodic). The pre-MAET carries each attribute's
kernel geometry (sigma, periodicity, and period) in its specs, so every
operation below takes it whole and returns it whole, and the tensor
functions read the geometry from it.

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
    difference_events    (D): event differencing, with per-attribute
                              difference orders.
    bind_events          (B): event binding, with per-attribute bind
                              orders (n-grams of consecutive events).
    translate_attributes (T): attribute translation.
    weight_events        (W): event weighting, a per-event window (one
                              factor per input attribute, from a
                              peak-normalized family of fixed variance).
    select_pre_maet      (S): a selection of attributes and events.
    bind_attributes,          one attribute from several, and several
    separate_attributes:      from one.
    transform_attributes (F): attribute rescaling, by per-attribute
                              elementwise maps (log, scale conversion,
                              user function), with an optional sign
                              attribute.

Compositions
    D o B == B o D       (event differencing and event binding commute,
                          in values, weights, and specs).
    D o T == D           (differencing absorbs attribute translation;
                          T o D adds mu to every difference).
    T o W centre shift   (W with centre c after T(mu) equals W with
                          centre c - mu before T; T leaves weights
                          unchanged, and W leaves values unchanged).
    D o F vs F o D       (log then difference gives log ratios;
                          difference then log(x + 1) with a sign
                          attribute gives signed compressed magnitudes).

Spectral enrichment (add_spectra), the sixth preprocessing operation of
the article, is demonstrated in jmm/demo_jmm_2_3_spectral.py.

Where the operations are taken further
    swept_similarity,    translation and event weighting swept along a
    swept_entropy        piece, a similarity or an entropy at each sweep
                         value (demo_swept_similarity.py;
                         jmm/demo_jmm_1_1_entropy.py).
    sweep_sim_maet       a translation sweep on built densities, in one
                         pass.
    n_tuple_entropy      the D o B pipeline of Section 6, packaged
                         (demo_rhythm_tensors.py, demo_sigma_space.py).
    demo_tempo_invariance.py, demo_repetition_handling.py
                         D, B, and F composed for tempo and
                         interval-scale invariance.
    demo_score_workflow.py, demo_score_categoricals.py
                         a pre-MAET built from a score, with
                         select_pre_maet and separate_attributes.
    demo_pre_maet_io.py  showing, writing, and reading a pre-MAET.

See also: show_pre_maet, difference_events, bind_events,
translate_attributes, weight_events, select_pre_maet, bind_attributes,
separate_attributes, transform_attributes, swept_similarity,
swept_entropy, sweep_sim_maet.

The MATLAB mirror is demo_preprocessing.m.
"""
import numpy as np

import mpt

# The toolbox's one-time informational hints (which route a call took,
# and the like) are switched off for a tidy printout, and restored at
# the end.
prev_defaults = mpt.set_default(show_hints=False)


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

# The kernel geometry of each attribute: pitch periodic at the octave
# (12 semitones) with sigma = 0.5 semitones, and time non-periodic with
# sigma = 0.25 quarter-notes. Both attributes are absolute.
specs = mpt.flat_specs(p_attr, names=["pitch", "time"], sigma=[0.5, 0.25],
                       per=[True, False], period=[12.0, 0.0])

# The three parts travel together as one pre-MAET, which every operation
# below takes whole and returns whole.
pm = mpt.pack_pre_maet(p_attr, w, specs)

mpt.show_pre_maet(pm)
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
# on the undifferenced one. The differenced attribute's sigma grows by
# sqrt(2), since a difference of two uncertain values is less certain
# than either; difference_events says so as it runs. Differencing is how
# interval (transposition-invariant) and inter-onset (time-shift-
# invariant) content is obtained: demo_overview.py, section 2c, uses it
# to find a motif at any transposition, and demo_tempo_invariance.py and
# demo_repetition_handling.py build on it.
diff_orders = [1, 0]
pmD = mpt.difference_events(pm, diff_orders)

print(f"  diff_orders = {diff_orders}   (pitch differenced, time left alone)")
mpt.show_pre_maet(pmD)
print()


# ===================================================================
#  3. bind_events (B): expand attributes into 2-grams
# ===================================================================

print("=== 3. bind_events (B) ===")

# Bind 2 consecutive events into 2-grams on both attributes. Each source
# attribute becomes ONE nested attribute: the two bound events form the
# ordered outer level, each event's own value the inner level (inheriting
# the source r/rel/exch). A' = A = 2. Trailing-drop alignment gives
# N' = N - max(L) + 1 = 6. B gathers each super-event's constituent
# weights alongside its values rather than combining them, so both of a
# 2-gram's metre weights survive, in order, inside the cell. Binding is
# how n-grams and nested multisets are compared: n_tuple_entropy is
# differencing then binding (Section 6), and
# jmm/demo_jmm_1_3_cadence_nesting.py finds cadences with nested bound
# events.
bind_orders = [2, 2]
pmB = mpt.bind_events(pm, bind_orders)

print(f"  bind_orders = {bind_orders}   "
      f"(A' = {len(pmB['p_attr'])}: each source attribute -> one nested "
      f"attribute)")
mpt.show_pre_maet(pmB)
s0 = pmB["specs"][0]
print(f"  spec[0]: r = {s0['r']}, exch = {s0['exch']}, rel = {s0['rel']}, "
      f"tags = {np.asarray(s0['tags']).ravel().tolist()}")
print()


# ===================================================================
#  3b. bind_events again (B o B): deepen a nested attribute to L = 3
# ===================================================================

print("=== 3b. bind_events again (B o B): deepen to L = 3 ===")

# bind_events accepts the pre-MAET it produces, so a second bind deepens
# the *already-nested* attribute rather than starting over. The specs
# travel with the pre-MAET, so nothing has to be threaded by hand. The
# existing tag matrix is tiled and a fresh outermost grouping column is
# appended; r/exch/rel each gain one outer level. The hierarchy grows
# note -> 2-event group (first bind) -> 2-group window (second bind).
# Trailing-drop again: N'' = N' - max(L) + 1 = 5.
pmBB = mpt.bind_events(pmB, [2, 2])
specBB = pmBB["specs"]

sBB = specBB[0]
print(f"  N'' = {pmBB['p_attr'][0].shape[1]}  (three-level super-events)")
mpt.show_pre_maet(pmBB, max_events=4)
print(f"  spec[0]: r = {sBB['r']}, exch = {sBB['exch']}, rel = {sBB['rel']}")
print("  (inner tag column tiled; a new outermost column appended.)")

# Build the absolute L = 3 nest and confirm a clean self-similarity. The
# kernel geometry travelled through both binds in the specs, so nothing
# further is supplied here; an argument given at the call would override
# the spec, which is what makes a sweep one call per value
# (demo_pre_maet_io, section 4).
dBB_abs = mpt.build_maet(pmBB, verbose=False)
sm_abs = float(mpt.sim_maet(dBB_abs, dBB_abs, verbose=False))
print(f"  absolute build: dim = {dBB_abs.dim}, "
      f"cosine self-match = {sm_abs:.4f}")

# Making the outermost level of pitch relative ([rel] = 1) reads the
# whole three-level tuple up to a common shift, so the doubly-bound
# pitch structure is invariant to transposing every note together.
# Absolute pitch is not: the narrow pitch kernel (sigma = 0.5) puts a
# 5-semitone shift out of reach. Replacing one part of a pre-MAET leaves
# the rest in place: here the specs.
specBB_out = [dict(specBB[0]), dict(specBB[1])]
specBB_out[0]["rel"] = [0, 0, 1]                 # outermost unit on pitch
pmBB_out = mpt.pack_pre_maet(pmBB, specs=specBB_out)
dBB_out = mpt.build_maet(pmBB_out, verbose=False)
pmBB_T = mpt.translate_attributes(pmBB, [5.0, 0.0])   # every pitch +5
pmBB_out_T = mpt.pack_pre_maet(pmBB_T, specs=specBB_out)
dBB_out_T = mpt.build_maet(pmBB_out_T, verbose=False)
dBB_abs_T = mpt.build_maet(pmBB_T, verbose=False)
sim_out = float(mpt.sim_maet(dBB_out, dBB_out_T, verbose=False))
sim_abs = float(mpt.sim_maet(dBB_abs, dBB_abs_T, verbose=False))
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
# attribute's values (here K = 1 each). rel is read from the specs
# (both attributes absolute), so neither translation is a no-op. T moves
# values only: the weights below are the metre weights unchanged.
#
# One call makes one translation. To compare a query with a context at
# each of many translations -- a sliding comparison, the canonical use of
# translation -- use swept_similarity (pre-MAETs;
# demo_swept_similarity.py) or sweep_sim_maet (densities), which compute
# every offset in one pass rather than building a copy per offset.
mu_pitch = 5.0
mu = [mu_pitch, 0.0]
pmT = mpt.translate_attributes(pm, mu)

print(f"  mu (per attribute) = {mu}   (G->C, F#->B, E->A; time untouched)")
mpt.show_pre_maet(pmT)
print()


# ===================================================================
#  5. weight_events (W): window the time attribute at the cadence
# ===================================================================

print("=== 5. weight_events (W) ===")

# Apply a window on the time attribute (input attribute 1) centred at the
# penult event (t = 6) with standard deviation 2 quarter-notes and
# gamma = 0 (pure Gaussian). The factor lands back on the time attribute
# (target attribute 1), the in-place weighting case, and the input is
# kept (drop_input_attr=False). The window multiplies the metre weights
# it finds rather than replacing them, so the time row below carries
# metre times envelope, and the pitch row is untouched.
#
# One call weights the events at one position. Sweeping a window along
# a piece, with an entropy or a similarity at each position, is
# swept_entropy, or swept_similarity with align='window'
# (demo_swept_similarity.py, sections 5 to 8;
# jmm/demo_jmm_1_1_entropy.py).
pmW = mpt.weight_events(
    pm,
    input_attr=1, target_attr=1,
    centre=6.0, sd=2.0, shape=0.0,    # gamma = 0 -> pure Gaussian
    drop_input_attr=False,
)

print("  input_attr = 1 (time); target_attr = 1; centre = 6; sd = 2; "
      "shape = 0 (Gaussian)")
mpt.show_pre_maet(pmW, decimals=3)
print()


# ===================================================================
#  5b. select_pre_maet (S): keep some attributes and some events
# ===================================================================

print("=== 5b. select_pre_maet (S) ===")

# A filter on the pre-MAET itself, as against selecting rows of the
# attribute table it may have been built from, which is pandas' own job.
# It reads only the two levels every pre-MAET has -- its attributes and
# its events -- so it knows nothing of where the pre-MAET came from.
# Here the cadence's three chords (events 4 to 6) on the pitch attribute
# alone; the kept items come back in the order given, and each keeps its
# tuple size and flags, so a selection cannot change what an attribute
# means. demo_score_workflow.py and demo_score_categoricals.py use it on
# pre-MAETs read from a score.
pmS = mpt.select_pre_maet(pm, attributes=[0], events=[4, 5, 6])

print("  attributes = [0] (pitch); events = [4, 5, 6] (the cadence)")
mpt.show_pre_maet(pmS, decimals=3)
print()


# ===================================================================
#  5c. bind_attributes and separate_attributes: one attribute from
#      several, and several from one
# ===================================================================

print("=== 5c. bind_attributes and separate_attributes ===")

# Binding across attributes, as bind_events binds across events. Pitch
# and time describe the same events, so binding them gives one
# attribute whose value at an event is the ordered pair, read whole
# (r = 2, exch = False) rather than as the product of two attributes.
# The two disagree on periodicity, so the bound attribute is given its
# own: non-periodic, with one sigma for both positions.
pmBA = mpt.bind_attributes(pm, [0, 1], name="pitchTime", r=2, exch=False,
                           sigma=0.5, per=False, period=0.0)
mpt.show_pre_maet(pmBA, decimals=3)
print()

# separate_attributes goes the other way, splitting the bound attribute
# into one attribute per position, each named for the bound attribute and
# its position.
p_back, _, s_back = mpt.unpack_pre_maet(
    mpt.separate_attributes(pmBA, "pitchTime"))
print(f"  separated into {len(p_back)} attributes: "
      f"{', '.join(spec['name'] for spec in s_back)}\n")


# ===================================================================
#  6. D o B == B o D (n-tuple entropy pipeline commutation)
# ===================================================================

print("=== 6. D o B == B o D (event differencing and event binding commute) ===")

# Both operations take a pre-MAET and return one, so they compose
# directly and the two routes coincide. Differencing pairs values
# position by position across (super-)events and the sliding bind window
# commutes with it, on the ordered, K = 1 domain where a difference is
# defined. The two operations propagate weights by different rules ---
# D takes the rolling product, B gathers --- and the composition agrees
# on the weights as well. This pipeline is the n-tuple entropy of Milne
# and Dean (2016), which n_tuple_entropy packages
# (demo_rhythm_tensors.py, demo_sigma_space.py).
#   D then B: difference each attribute (order 1), then bind 2-grams.
pmDB = mpt.bind_events(mpt.difference_events(pm, [1, 1]), [2, 2])
#   B then D: bind 2-grams, then difference each nested attribute
#   position by position.
pmBD = mpt.difference_events(mpt.bind_events(pm, [2, 2]), [1, 1])

mpt.show_pre_maet(pmDB, title="  D then B:")
mpt.show_pre_maet(pmBD, title="  B then D:")

vals_agree = all(np.allclose(pmDB["p_attr"][a], pmBD["p_attr"][a],
                             equal_nan=True) for a in range(2))
wts_agree = all(np.allclose(np.asarray(pmDB["w_attr"][a]),
                            np.asarray(pmBD["w_attr"][a]),
                            equal_nan=True) for a in range(2))
specs_agree = all(
    list(np.ravel(pmDB["specs"][a][k]))
    == list(np.ravel(pmBD["specs"][a][k]))
    for a in range(2) for k in ("tags", "r", "exch", "rel", "sigma")
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

mpt.show_pre_maet(pmDT, title="  D(T(p)):")
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

# The kernel of build_maet has a fixed width in whatever units the
# values carry, so the choice of scale is made before the tensor. The
# bare-array form converts a vector in one call:
f_hz = np.array([392.00, 369.99, 329.63])          # G4, F#4, E4 in Hz
p_cents = mpt.transform_attributes(f_hz, None, ('hz', 'cents'))
print(f"  Hz -> cents: [{p_cents[0]:.1f} {p_cents[1]:.1f} {p_cents[2]:.1f}]")

# A log scale turns uniform scaling -- a tempo change, or an
# augmentation of a melody's intervals -- into a translation, which the
# relative flag or a translation sweep can then absorb:
# demo_repetition_handling.py and demo_tempo_invariance.py build on
# this.

# Order with differencing carries meaning. (i) F then D on inter-onset
# intervals in log2 gives log ratios: a doubling is +1, a halving -1.
ioi = np.array([[0.25, 0.5, 0.5, 1.0]])            # seconds
pmLD = mpt.difference_events(
    mpt.transform_attributes([ioi], None, [('log', {'base': 2})]), 1)
print(f"  log2(IOI) then D: {pmLD['p_attr'][0].ravel()}  (log ratios)")

# (ii) D then a compressive transform on the signed pitch intervals.
# log(x + 1) admits the zero of a repeated note with the constant written
# down. Negative values are refused unless a sign attribute is requested:
# with sign=True the transform is applied to |x| and a sign attribute
# at the 2-point simplex's vertices, {-1/2, 0, +1/2}, is inserted right
# after its source, so the pre-MAET grows from one attribute to two.
pmDp = mpt.difference_events(mpt.select_pre_maet(pm, attributes=[0]), 1)
pmF = mpt.transform_attributes(pmDp, [('log', {'offset': 1})], sign=True)
p_f, s_f = pmF["p_attr"], pmF["specs"]
print(f"  D(pitch)        = {pmDp['p_attr'][0].ravel()}")
print(f"  log(|D(pitch)|+1) = {np.round(p_f[0].ravel(), 4)}, "
      f"sign = {p_f[1].ravel()} (spec name '{s_f[1]['name']}')")
# A log has no single image of the old width, so the rescaled attributes
# carry sigma as NA (demo_pre_maet_io, section 6), and the widths the
# new units call for are supplied here.
dens_f = mpt.build_maet(pmF, sigma=[0.2, 0.3],
                        per=[False, False], period=[0, 0],
                        verbose=False)
print(f"  build_maet on the two-attribute pre-MAET: dim = {dens_f.dim}")

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
#  9. The tensor functions take the pre-MAET whole
# ===================================================================

print("\n=== 9. Pre-MAET form: the tensor functions take the pre-MAET whole ===")

# entropy_maet, eval_maet, and sim_maet each take a pre-MAET whole and
# read the kernel geometry from its specs; the density is built inside
# the call. This is the form to reach for first.

# --- 9a. entropy_maet ---
H_orig = mpt.entropy_maet(pm, method="renyi2", verbose=False)
print("  entropy_maet(pm)")
print(f"    = {H_orig:.4f}  (Renyi-2)")

# The raw positional form, with the parts and the five geometry vectors
# (sigma, r, rel, per, period) written out, reaches the same value;
# it serves data that was never packed as a pre-MAET.
p0, w0, _ = mpt.unpack_pre_maet(pm)
H_raw = mpt.entropy_maet(p0, w0, [0.5, 0.25], [1, 1], [False, False],
                         [True, False], [12.0, 0.0],
                         method="renyi2", verbose=False)
print(f"  raw positional form: {H_raw:.4f}  "
      f"(|delta| = {abs(H_raw - H_orig):.2e})")
assert abs(H_raw - H_orig) < 1e-12, \
    "Section 9a: pre-MAET and raw forms disagree."

# --- 9b. eval_maet at the penult event (pitch = 66, t = 6) ---
# Query points are supplied as a length-A list, one (K_a, M_q) matrix per
# attribute. Single query here, so a (1, 1) matrix for each of the two
# attributes: pitch = 66, t = 6.
Xq = [np.array([[66.0]]), np.array([[6.0]])]
val_at_penult = mpt.eval_maet(pm, Xq, verbose=False)
print("  eval_maet(pm, Xq)")
print(f"    = {float(val_at_penult[0]):.4f}")
print("  (The density at an actual event: nearly all of it is event 6's")
print("   own kernel, its neighbours' tails adding little.)")

# --- 9c. sim_maet of the fragment and its transposed copy (Section 4) ---
# The pitch kernel is narrow (sigma = 0.5 semitones), so the 5-semitone
# shift puts every event out of kernel reach of its original pitch
# class, and the similarity collapses to nearly 0. Differencing, in 9d,
# recovers it.
sim_T = mpt.sim_maet(pm, pmT, verbose=False)
print("  sim_maet(pm, pmT)")
print(f"    = {float(sim_T):.4f}")

# --- 9d. sim_maet of the differenced pair: D o T == D in action ---
# Section 7's identity guarantees that the differenced original and the
# differenced transposed copy are identical in value, so their cosine
# similarity is exactly 1: the algebraic identity surfacing as a
# downstream observable.
sim_diffed = mpt.sim_maet(pmD, pmDT, verbose=False)
print("  sim_maet(pmD, pmDT)")
print(f"    = {float(sim_diffed):.4f}  (exactly 1: D absorbs T)")

# ===================================================================
#  10. Build once, query many
# ===================================================================

print("\n=== 10. Density form: build once, query many; parity with Section 9 ===")

# Where one density is evaluated or compared many times, build it once
# with build_maet and pass the density instead: the structural work
# (canonical forms, tuple indices, weight products) is then paid once.
dens_orig = mpt.build_maet(pm, verbose=False)
dens_T = mpt.build_maet(pmT, verbose=False)
dens_D = mpt.build_maet(pmD, verbose=False)
dens_DT = mpt.build_maet(pmDT, verbose=False)
dens_W = mpt.build_maet(pmW, verbose=False)

# --- 10a. entropy_maet on the density; same answer as 9a. ---
H_orig_dens = mpt.entropy_maet(dens_orig, method="renyi2", verbose=False)
delta_a = abs(float(H_orig_dens) - float(H_orig))
print("  entropy_maet(dens_orig)")
print(f"    = {float(H_orig_dens):.4f}  (Renyi-2; parity vs 9a: |delta| = {delta_a:.2e})")
assert delta_a < 1e-12, "Section 10a: entropy pre-MAET and density forms disagree."

# --- 10b. eval_maet at the same query; same answer as 9b. ---
val_at_penult_dens = mpt.eval_maet(dens_orig, Xq, verbose=False)
delta_b = abs(float(val_at_penult_dens[0]) - float(val_at_penult[0]))
print("  eval_maet(dens_orig, Xq)")
print(f"    = {float(val_at_penult_dens[0]):.4f}  (parity vs 9b: |delta| = {delta_b:.2e})")
assert delta_b < 1e-12, "Section 10b: eval pre-MAET and density forms disagree."

# --- 10c. sim_maet(dens_orig, dens_T); same answer as 9c. ---
sim_T_dens = mpt.sim_maet(dens_orig, dens_T, verbose=False)
delta_c = abs(float(sim_T_dens) - float(sim_T))
print("  sim_maet(dens_orig, dens_T)")
print(f"    = {float(sim_T_dens):.4f}  (parity vs 9c: |delta| = {delta_c:.2e})")
assert delta_c < 1e-12, "Section 10c: sim_maet pre-MAET and density forms disagree."

# --- 10d. sim_maet(dens_D, dens_DT) on the differenced pair; ---
#       same answer as 9d. (Section 7 identity: exactly 1.)
sim_diffed_dens = mpt.sim_maet(dens_D, dens_DT, verbose=False)
delta_d = abs(float(sim_diffed_dens) - float(sim_diffed))
print("  sim_maet(dens_D, dens_DT)")
print(f"    = {float(sim_diffed_dens):.4f}  (parity vs 9d: |delta| = {delta_d:.2e})")
assert delta_d < 1e-12, "Section 10d: sim_maet pre-MAET and density forms disagree."

# --- 10e. List form: one reference against many candidates. ---
# A list of densities against one density returns one similarity per
# entry, for "compare one reference against many" workflows. Three
# entries, [dens_orig, dens_T, dens_W]:
#   entry 0:  sim(orig, orig) = 1 by definition.
#   entry 1:  sim(orig, T), which matches 9c.
#   entry 2:  sim(orig, W), the same values under the cadence window of
#             Section 5, so only the weights differ.
# demo_batch_processing.py takes the list forms further, on a table of
# trials.
sim_list = mpt.sim_maet([dens_orig, dens_T, dens_W], dens_orig,
                        verbose=False)
sim_list_vals = [float(v) for v in sim_list]
print("  sim_maet([dens_orig, dens_T, dens_W], dens_orig)")
print("    = [{:.4f}, {:.4f}, {:.4f}]".format(*sim_list_vals))
print("    (entry 0: self = 1; entry 1: vs T (= 9c); entry 2: vs W, the")
print("     same values under the cadence window, so only the weights differ.)")
assert abs(sim_list_vals[0] - 1.0) < 1e-12, "10e: self-similarity not 1."
assert abs(sim_list_vals[1] - float(sim_T)) < 1e-12, "10e: list entry 1 != 9c value."

mpt.set_default(**prev_defaults)
