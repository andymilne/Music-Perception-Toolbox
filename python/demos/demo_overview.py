"""demo_overview.py

Quick tour of the Music Perception Toolbox (mpt) for Python.

The sections run from the simplest material to the richest: expectation
tensors of a single multiset, beginning with a chord enriched with
spectra and smoothed into a density, multi-attribute expectation tensors
(MAETs) of a sequence of events, and then the families of measures that
stand beside the tensors -- consonance, balance and evenness, and scale
and rhythm structure. Each section ends with a "See also" list of the demos that
take its topic further.

Two running examples recur throughout: the diatonic scale
[0, 2, 4, 5, 7, 9, 11] in 12-EDO, and the son clave rhythm
[0, 3, 6, 10, 12] in a 16-step cycle. Section 2 combines them, as a
diatonic melody set in the clave rhythm.

Uses: transform_attributes, add_spectra, sim_maet, entropy_maet,
      mass_maet, build_maet, plot_maet, flat_specs, pack_pre_maet,
      show_pre_maet, swept_similarity, swept_mass, difference_events,
      template_harmonicity, tensor_harmonicity, spectral_entropy, roughness, balance, evenness,
      dft_circular, coherence, sameness, n_tuple_entropy, mean_offset,
      edges, markov_s, circ_apm, set_default.

The MATLAB mirror is demo_overview.m.
"""

import numpy as np
import matplotlib.pyplot as plt

import mpt

np.set_printoptions(precision=3, suppress=True, linewidth=100)

# The toolbox's one-time informational hints (which route a call took,
# and the like) are switched off for a tidy printout, and restored at
# the end.
prev_defaults = mpt.set_default(show_hints=False)

# The running examples, in the units each section needs.
diat = [0, 2, 4, 5, 7, 9, 11]                    # 12-EDO steps
diat_cents = np.array([0, 200, 400, 500, 700, 900, 1100], dtype=float)
clave = [0, 3, 6, 10, 12]                        # 16-step cycle

# ===================================================================
#  1. Expectation tensors of a single multiset  (User Guide §3.1, §8, §13.1)
# ===================================================================

# An expectation tensor replaces each element of a multiset with a
# Gaussian of width sigma and sums them, over r-tuples of elements. 1a
# builds one from a chord. Three things are computed from one: the
# similarity of two of them (1b), the entropy of one (1c), and the mass
# it holds in a region (1d). Four parameters decide what it represents
# (1e).

# --- 1a. A chord as a density ---

# A sounded pitch is a spectrum of partials, so spectral enrichment
# replaces each chord tone by its harmonics, weighted here by 1/n
# (powerlaw 1). The enriched chord is then an ordinary multiset: its
# expectation tensor at r = 1, absolute and not periodic, is a density
# over pitch, the chord's smoothed spectrum.
print("\n=== 1a. A C major triad as a spectral density ===")
chord = np.array([0.0, 400.0, 700.0])
p, w = mpt.add_spectra(chord, None, "harmonic", 8, "powerlaw", 1)
print(f"  3 pitches × 8 harmonics = {len(p)} partials")
print(f"  C's partials, in cents above C: {np.round(p[:8], 1)}")
dens_chord = mpt.build_maet(p, w, 10, 1, False, False, 0, verbose=False)

# Above, each partial's kernel (its weight times a Gaussian of sigma =
# 10 cents); below, the density they sum to. Partials of different tones that land
# close together merge into one peak: C's fifth harmonic (2786 cents
# above C) and E's fourth (2800), for example. Those coincidences are
# what spectral similarity and harmonicity pick up.
fig, (ax_k, ax_d) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True,
                                 sharey=True)
mpt.plot_maet(dens_chord, method='kernels', ax=ax_k)
mpt.plot_maet(dens_chord, method='density', ax=ax_d)
ax_k.set_ylabel("kernels")
ax_d.set_ylabel("density")
ax_d.set_xlabel("pitch (cents above C)")
ax_d.set_xticks(np.arange(0, p.max() + 1, 400))  # a divisor of the octave
ax_k.set_title("A C major triad, 8 harmonics per tone, sigma = 10 cents")
fig.tight_layout()

# See also:
#   demo_virtual_pitches     harmonic templates matched to enriched chords
#   demo_audio_analysis      measured spectra (audio_peaks) in place of
#                            synthetic ones
#   demo_jmm_2_3_spectral    spectral enrichment inside a multi-attribute
#                            analysis

# --- 1b. Similarity: spectral pitch class similarity (SPCS) ---

print("\n=== 1b. Spectral pitch class similarity ===")
chord_mat = np.array([
    [0, 400, 700],     # Major
    [0, 300, 700],     # Minor
    [0, 300, 600],     # Dim
], dtype=float)
chord_names = ["Major", "Minor", "Dim"]

# SPCS compares such densities with pitch periodic at the octave
# (is_per = True, period 1200), so each spectrum is folded onto one
# octave of pitch classes before the comparison.
#
# Batched call: the scale (a 1-D vector) is broadcast against every
# row of chord_mat. The 'spectrum' kwarg does what 1a did by hand,
# enriching both sides identically before the density is built.
s = mpt.sim_maet(
    diat_cents, None, chord_mat, None,
    10, 1, False, True, 1200,
    spectrum=['harmonic', 24, 'powerlaw', 1],
    verbose=False,
)
for name, val in zip(chord_names, s):
    print(f"  Diatonic vs {name:5s} triad: {val:.3f}")

# --- 1c. Entropy: how evenly a scale's interval content is spread ---

# A relative dyad tensor (r = 2, relative) is a density over the
# intervals between pairs of pitches, so its entropy is high when a
# scale holds many different intervals in similar numbers and low when
# it holds few. Renyi-2 entropy is computed in closed form, so no grid
# enters.
print("\n=== 1c. Interval-content entropy (Renyi-2, bits) ===")
scales = {
    "Whole-tone": [0, 200, 400, 600, 800, 1000],
    "Diatonic":   list(diat_cents),
    "Chromatic":  list(range(0, 1200, 100)),
}
for name, sc in scales.items():
    H = mpt.entropy_maet(np.asarray(sc, dtype=float), None,
                         10, 2, True, True, 1200,
                         method='renyi2', verbose=False)
    print(f"  {name:10s}: {H:.3f}")
# The whole-tone scale holds only even intervals, so its entropy is
# lowest; the chromatic scale holds every interval equally often, so its
# entropy is highest.

# --- 1d. Mass: how much of a density lies in a region ---

# The mass of a density within a region, as a proportion of its whole
# mass (normalize='total'), is in effect the weighted proportion of
# tuples whose values fall in the region. On a relative tensor at r = 2
# the tuples are pairs of notes and their values intervals, so here it
# is the share of each scale's pairs of notes that lie a fifth or a
# fourth apart. Because the density is relative, periodic, and
# exchangeable, every pair of notes a fifth or fourth apart contributes
# the same two coincident kernels, at 700 and 500 cents, whatever its
# transposition, octave, or order; the two regions together therefore
# count each such pair once.
print("\n=== 1d. Share of pairs a fifth or fourth apart ===")
for name, sc in scales.items():
    dens = mpt.build_maet(np.asarray(sc, dtype=float), None,
                          10, 2, True, True, 1200, verbose=False)
    m = (mpt.mass_maet(dens, {0: (650, 750)}, normalize='total')
         + mpt.mass_maet(dens, {0: (450, 550)}, normalize='total'))
    print(f"  {name:10s}: {m:.3f}")
# Six of the diatonic scale's 21 pairs are a fifth apart (0.286), twelve
# of the chromatic scale's 66 (0.182), and none of the whole-tone
# scale's.

# --- 1e. The parameters that define a tensor ---

# The same diatonic scale drawn four ways. The tuple size r sets how many
# elements each point of the density describes, and relative mode
# (is_rel) reads a tuple's intervals rather than its pitches, which
# makes the density transposition-invariant and removes one dimension:
# dim = r - is_rel. All four are periodic at the octave.
#
#   r = 1, absolute   the pitch classes themselves
#   r = 2, absolute   pairs of pitch classes
#   r = 2, relative   the intervals between pairs: the interval vector
#                     <2, 5, 4, 3, 6, 1>, smoothed and mirrored about
#                     the tritone
#   r = 3, relative   trichords, each drawn as the two intervals above
#                     one of its notes
print("\n=== 1e. Tensor parameters (figure) ===")
configs = [(1, False), (2, False), (2, True), (3, True)]
fig, axes = plt.subplots(2, 2, figsize=(9, 8))
for ax, (r, is_rel) in zip(axes.flat, configs):
    dens = mpt.build_maet(diat_cents, None, 15, r, is_rel, True, 1200,
                          verbose=False)
    mpt.plot_maet(dens, method='density', ax=ax)
    dim = r - int(is_rel)
    label = 'interval' if is_rel else 'pitch class'
    ax.set_title(f"r = {r}, {'relative' if is_rel else 'absolute'} "
                 f"(dim = {dim})")
    if dim == 1:
        ax.set_xlabel(f'{label} (cents)')
        ax.set_ylabel('density')
    else:
        ax.set_xlabel(f'{label} 1 (cents)')
        ax.set_ylabel(f'{label} 2 (cents)')
fig.suptitle('The diatonic scale as four expectation tensors')
fig.tight_layout()
print("  Drawn: r = 1 and 2 absolute, r = 2 and 3 relative.")

# See also:
#   demo_maet_plots          every combination of r, is_rel, is_per, and
#                            is_exch, drawn by each of plot_maet's methods
#   demo_triad_spcs_grid     SPCS of every triad containing a fifth
#   demo_edo_approx          how well each n-EDO approximates a JI chord
#   demo_gen_chain_pcs       the same, over generator-chain tunings
#   demo_batch_processing    a feature for every trial of an experiment
#   demo_dispatch_and_kernel_controls
#                            speed controls, and Renyi-2 entropy

# ===================================================================
#  2. Multi-attribute expectation tensors  (User Guide §3.3, §6-§8, §13.2)
# ===================================================================

# A MAET takes a sequence of events, each carrying several attributes
# -- here pitch and onset -- and builds one density over all of them
# jointly. The melody below is two cycles of the son clave, each note a
# diatonic pitch; the second cycle is the first transposed up a fifth.
#
#   cycle 1   C  D  E  G  E   at onsets  0  3  6 10 12
#   cycle 2   G  A  B  D  B   at onsets 16 19 22 26 28

# --- 2a. Events and the pre-MAET ---

# A pre-MAET is everything a MAET is built from: the values of each
# attribute for each event, their weights, and each attribute's
# parameters. Pitch is periodic at the octave, with sigma = 20 cents;
# onset is not periodic, with sigma = 0.5 steps. Pitch is given in MIDI
# note numbers and converted to cents, the units its sigma is read in:
# transform_attributes rescales an attribute (MIDI, Hz, cents, ERB-rate,
# or a logarithm), and a logarithmic rescaling turns a proportional
# change -- a tempo change, say -- into a common shift.
print("\n=== 2a. A melody as a pre-MAET ===")
onsets = np.concatenate([clave, np.asarray(clave) + 16]).astype(float)
midi = np.array([60, 62, 64, 67, 64, 67, 69, 71, 74, 71])
pitch = mpt.transform_attributes(midi, None, ('midi', 'cents'))
# See also:
#   demo_preprocessing        transform_attributes (attribute
#                             rescaling) among the other pre-MAET
#                             preprocessing operations, and how a
#                             rescaling carries sigma with it
#   demo_repetition_handling  a non-linear rescaling (log step size)

p_attr = [pitch[None, :], onsets[None, :]]      # one row per attribute
specs = mpt.flat_specs(p_attr, name=['pitch', 'onset'],
                       sigma=[20, 0.5], is_per=[True, False],
                       period=[1200, 0])
melody = mpt.pack_pre_maet(p_attr, None, specs)
mpt.show_pre_maet(melody, max_events=None)

# The query: the melody's opening three notes, C D E.
query = mpt.pack_pre_maet([pitch[None, :3], onsets[None, :3]], None, specs)

# --- 2b. Where does the motif occur? ---

# swept_similarity translates the query along the onset attribute and
# compares it with the whole melody, in pitch and onset jointly, at each
# offset: attribute translation, the canonical way to find a query.
# Naming the attribute alone (sweep=1) asks for the default offsets,
# every placement at which query and melody overlap, stepped at no more
# than half the width of the profile's peaks and at a whole fraction of
# the grid of steps the notes lie on, so that every exact match is on
# the grid. return_offsets=True also returns those offsets, mu_abs,
# keyed by attribute (here mu_abs[1]): the query and the melody are both
# written from onset 0, so an offset is the onset at which the query
# starts in the melody, 0 being the query where it was taken from.
print("\n=== 2b. Motif search, absolute pitch ===")
S_abs, mu_abs = mpt.swept_similarity(melody, query, sweep=1,
                                     return_offsets=True)


def report_peaks(S, mu, thresh=0.75):
    """Print the local maxima of a similarity profile above thresh."""
    for i in range(1, len(S) - 1):
        if S[i] > thresh and S[i] >= S[i - 1] and S[i] > S[i + 1]:
            print(f"  peak at offset {mu[i]:4.1f} steps: {S[i]:.3f}")


report_peaks(S_abs, mu_abs[1])
# One peak, at offset 0 -- the query's own position. The
# transposed statement in cycle 2 is not found: in absolute mode G A B
# is not C D E.

# --- 2c. Invariance by preprocessing ---

# Differencing the pitch attribute replaces each pitch with the
# interval from the previous one, so a transposed statement has the
# same values as the original. Onset is passed through (order 0),
# keeping each interval at the onset of its second note. Differencing
# drops the first event but leaves every surviving onset where it was,
# so mu still says where the original query starts, and the two profiles
# share one horizontal axis.
# The pitch sigma grows by sqrt(2), since a difference of two uncertain
# values is less certain than either; difference_events announces this.
print("\n=== 2c. Motif search, pitch intervals ===")
melody_d = mpt.difference_events(melody, [1, 0])
query_d = mpt.difference_events(query, [1, 0])
mpt.show_pre_maet(melody_d, max_events=None)
S_diff, mu_diff = mpt.swept_similarity(melody_d, query_d, sweep=1,
                                       return_offsets=True)
report_peaks(S_diff, mu_diff[1])
# Two full matches, at offsets 0 and 16 -- one per clave cycle: the
# rising pair of whole tones is found in both. The profile also has
# partial matches, near 0.5, at offsets 3, 13, and 19, where one of the
# query's two intervals lines up with the melody's. Which preprocessing
# and which mode are chosen is what decides what counts as "the same".

# Top: the melody as a piano roll, the query's notes filled. Bottom: the
# two profiles against the onset of the query's first note, on the same
# horizontal axis, so each peak sits under the statement it found.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                               gridspec_kw={'height_ratios': [1, 1.3]})
note_names = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
ax1.scatter(onsets, midi, s=60, facecolors='none', edgecolors='C0',
            linewidths=1.5, label='melody')
ax1.scatter(onsets[:3], midi[:3], s=60, color='C0', label='query (C D E)')
for t, m in zip(onsets, midi):
    ax1.annotate(note_names[[0, 2, 4, 5, 7, 9, 11].index(m % 12)],
                 (t, m), xytext=(0, 7), textcoords='offset points',
                 ha='center', fontsize=8)
ax1.axvline(16, color='0.6', linestyle='--', linewidth=1)
ax1.set_ylabel('MIDI pitch')
ax1.set_ylim(58, 77)
ax1.set_title('Two clave cycles, the second a fifth higher')
ax1.legend(loc='upper left')
ax2.plot(mu_abs[1], S_abs, linewidth=2, label='absolute pitch')
ax2.plot(mu_diff[1], S_diff, linewidth=2, label='pitch intervals (differenced)')
ax2.axvline(16, color='0.6', linestyle='--', linewidth=1)
ax2.set_xlabel("onset of the query's first note (steps)")
ax2.set_ylabel('similarity to C D E')
ax2.set_title('Where does the opening motif recur?')
ax2.legend(loc='upper right')
fig.tight_layout()

# --- 2d. Mass in a moving window: each triad's share of the notes ---

# The second cycle is the first a fifth higher, so the melody's pitches
# move from the tones of the C major triad (C, E, G) to those of the G
# major triad (G, B, D). swept_mass shows where. Time is the onset
# attribute, measured in steps of the 16-step clave cycle. At each of a
# list of times s, a Gaussian window centred on s, with a standard
# deviation of 4 steps, weights each note according to its distance in
# time from s, so the notes near s count most. The onset attribute is
# then dropped, leaving a density over pitch class alone, and its mass
# is taken inside a region: the range of values to be counted, here the
# pitch classes within 50 cents of one triad tone (pc - 50 to pc + 50
# cents), each note counting by the part of its pitch kernel that falls
# in that range. With normalize='total', the mass is divided by the
# density's whole mass, so the result is the share of the weighted notes
# near s that lie in the range. A region takes a single range per
# attribute, so a triad's share is the sum of three calls, one per tone.
# The times s are left to the defaults, from the first onset to the last
# in steps of half the window's sd (2 steps), and come back with
# return_sweep_values=True, for the plot.
print("\n=== 2d. Share of each triad's tones, in a window over onset ===")
win = {'shape': 'gaussian', 'sd': 4.0}
triads = {'C major': [0, 400, 700], 'G major': [700, 1100, 200]}
share = {}
for name, pcs in triads.items():
    share[name] = 0.0
    for pc in pcs:
        m, sv = mpt.swept_mass(melody, sweep=1, window={1: win},
                               drop=[1], region={0: (pc - 50, pc + 50)},
                               normalize='total', return_sweep_values=True)
        share[name] = share[name] + m
s = sv[1]                                    # the sweep values, 0 to 28
for x in (4, 12, 20, 28):
    k = int(np.flatnonzero(s == x)[0])
    print(f"  onset {x:2d}: C major {share['C major'][k]:.2f}, "
          f"G major {share['G major'][k]:.2f}")
# The C major triad holds most of the weighted notes through the first
# cycle and the G major triad through the second, the two crossing just
# after step 16, where the transposition begins. They overlap on G, the
# tone they share, so the two shares do not sum to 1.

fig, ax = plt.subplots(figsize=(9, 3))
for name, v in share.items():
    ax.plot(s, v, linewidth=2, label=name)
ax.axvline(16, color='0.6', linestyle='--', linewidth=1)
ax.set_xlabel('window centre s (steps)')
ax.set_ylabel('share of the weighted notes')
ax.set_title("Each triad's share of the weighted notes")
ax.set_ylim(0, 1)
ax.legend(loc='center right')
fig.tight_layout()

# See also:
#   demo_pre_maet_io          showing, exporting, and importing a
#                             pre-MAET
#   demo_preprocessing        the pre-MAET preprocessing operations,
#                             and their compositions
#   demo_score_workflow       a pre-MAET read from MusicXML or MIDI
#                             (then demo_score_grid,
#                             demo_score_categoricals)
#   demo_swept_similarity  swept_similarity in depth: translation,
#                             windows, align, and relative or dropped
#                             attributes
#   demo_tempo_invariance     motif search tolerant of tempo change
#   demo_repetition_handling  interval-scale invariance, and what to do
#                             with repeated notes
#   demo_helix_blend          pitch read as pitch class and height at
#                             once
#   jmm/                      the analyses of the JMM article: entropy
#                             across a chorale (1.1), voice-aware
#                             similarity (1.2), cadence finding (1.3),
#                             tuple size in chord matching (1.4), motif
#                             discovery (2.1, 2.2), a motif found in any
#                             key under spectral enrichment (2.3),
#                             phase in Piano Phase (3.1-3.3), and an
#                             expert analysis carried as a nested
#                             attribute (4.1); see jmm/README.md

# ===================================================================
#  3. Consonance and harmonicity  (User Guide §9.1)
# ===================================================================

print("\n=== 3. Harmonicity and entropy (JI major triad) ===")
ji_triad = [0, 386.31, 701.96]
spec = ["harmonic", 24, "powerlaw", 1]

hMax, hEnt = mpt.template_harmonicity(ji_triad, None, 12, chord_spectrum=spec)
print(f"  Template harmonicity (hMax):     {hMax:.4f}")
print(f"  Template harmonicity (hEntropy): {hEnt:.4f}")

h = mpt.tensor_harmonicity(ji_triad, None, 12, spectrum=spec)
print(f"  Tensor harmonicity:              {h:.4f}")

H = mpt.spectral_entropy(ji_triad, None, 12, spectrum=spec)
print(f"  Spectral entropy:                {H:.4f}")

print("\n=== JI vs 12-EDO comparison ===")
# Both features take a 2-D matrix, one chord per row, and return one
# value per row (and deduplicate repeated rows internally).
edo_triad = [0, 400, 700]
triads = np.array([ji_triad, edo_triad])
hMaxBoth, _ = mpt.template_harmonicity(triads, None, 12, chord_spectrum=spec)
HBoth = mpt.spectral_entropy(triads, None, 12, spectrum=spec)
for name, hMax, H in zip(("JI", "12-EDO"), hMaxBoth, HBoth):
    print(f"  {name:6s}  hMax={hMax:.4f}  specEntropy={H:.4f}")

print("\n=== Roughness ===")
p_cents = mpt.transform_attributes([60, 64, 67], None, ('midi', 'cents'))
p_r, w_r = mpt.add_spectra(p_cents, None, "harmonic", 8, "powerlaw", 1)
f_hz = mpt.transform_attributes(p_r, None, ('cents', 'hz'))
r = mpt.roughness(f_hz, w_r)
print(f"  C major triad (8 harmonics): roughness = {r:.4f}")

# See also:
#   demo_triad_consonance    every measure over a grid of triads
#   demo_virtual_pitches     the salience profiles behind template
#                            harmonicity
#   demo_audio_analysis      the same measures from recorded sounds
#   demo_batch_processing    the same measures for a table of trials

# ===================================================================
#  4. Balance and evenness  (User Guide §9.2)
# ===================================================================

print("\n=== 4. Balance and evenness ===")

print("  Diatonic scale [0,2,4,5,7,9,11] in 12-EDO:")
b = mpt.balance(diat, None, 12)
e = mpt.evenness(diat, 12)
print(f"    Balance:  {b:.3f}")
print(f"    Evenness: {e:.3f}")

print("  Son clave [0,3,6,10,12] in 16:")
b = mpt.balance(clave, None, 16)
e = mpt.evenness(clave, 16)
print(f"    Balance:  {b:.3f}")
print(f"    Evenness: {e:.3f}")

# Balance places each element on the unit circle, at angle 2*pi*p/period,
# and is 1 minus the length of their mean: a collection whose elements
# pull equally in every direction balances at the centre. The arrow is
# that mean, the first coefficient dft_circular returns. It is barely visible for the diatonic scale and the clave,
# which are both close to perfectly balanced; the third panel, five
# onsets packed into the first half of the cycle, shows what an
# unbalanced rhythm looks like.
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
for ax, (name, pts, period) in zip(
        axes, [("Diatonic scale", diat, 12), ("Son clave", clave, 16),
               ("First half only", [0, 2, 4, 6, 8], 16)]):
    ring = np.linspace(0, 2 * np.pi, 361)
    ax.plot(np.sin(ring), np.cos(ring), color='0.8', linewidth=1)
    ticks = 2 * np.pi * np.arange(period) / period
    ax.scatter(np.sin(ticks), np.cos(ticks), s=10, color='0.7')
    ang = 2 * np.pi * np.asarray(pts) / period
    ax.scatter(np.sin(ang), np.cos(ang), s=70, color='C0', zorder=3)
    for a, v in zip(ang, pts):
        ax.annotate(str(v), (1.18 * np.sin(a), 1.18 * np.cos(a)),
                    ha='center', va='center', fontsize=9)
    F, _ = mpt.dft_circular(pts, None, period)
    m = F[0]                             # the mean, as cos + i sin
    ax.annotate('', xy=(m.imag, m.real), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='C3', lw=2))
    ax.plot(0, 0, 'k+')
    ax.set_title(f"{name}\nbalance = {mpt.balance(pts, None, period):.3f}")
    ax.set_aspect('equal')
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.axis('off')
fig.suptitle('Balance: the mean of the elements on the circle (red)')
fig.tight_layout(rect=(0, 0, 1, 0.92))

# See also:
#   demo_dft_circular_simulate  balance and evenness under positional
#                               uncertainty (sigma > 0), analytically
#                               and by Monte Carlo

# ===================================================================
#  5. Scale and rhythm structure  (User Guide §9.2)
# ===================================================================

print("\n=== 5. Scale structure (diatonic) ===")
c, nc = mpt.coherence(diat, 12)
sq, nd = mpt.sameness(diat, 12)
print(f"  Coherence: {c:.3f} ({int(nc)} failure)")
print(f"  Sameness:  {sq:.3f} ({int(nd)} ambiguity)")

H1, _ = mpt.n_tuple_entropy(diat, 12, 1)
H2, _ = mpt.n_tuple_entropy(diat, 12, 2)
print(f"  1-tuple entropy: {H1:.3f}")
print(f"  2-tuple entropy: {H2:.3f}")

# Mode brightness. mean_offset, read at a query point, sums the arcs
# from that point up to each pitch class of the scale and subtracts the
# arcs down to them, so at a mode's tonic it says how high the mode's
# pitches sit above its tonic. The seven modes of the diatonic scale,
# their tonics taken down the chain of fifths from F:
modes = ["Lydian", "Ionian", "Mixolydian", "Dorian", "Aeolian",
         "Phrygian", "Locrian"]
tonics = [5, 0, 7, 2, 9, 4, 11]
bright = mpt.mean_offset(diat, None, 12, tonics)
print("  Mode brightness (mean offset at the tonic):")
for name, v in zip(modes, bright):
    print(f"    {name:10s} {v:+.3f}")
# Each step down the chain lowers one degree of the mode by a semitone,
# so brightness falls by 1/6 at each step, from Lydian to Locrian.

print("\n=== Rhythm structure (son clave) ===")
c, nc = mpt.coherence(clave, 16)
sq, nd = mpt.sameness(clave, 16)
print(f"  Coherence: {c:.3f} ({int(nc)} failures)")
print(f"  Sameness:  {sq:.3f} ({int(nd)} ambiguities)")

H1, _ = mpt.n_tuple_entropy(clave, 16, 1)
H2, _ = mpt.n_tuple_entropy(clave, 16, 2)
print(f"  1-tuple entropy: {H1:.3f}")
print(f"  2-tuple entropy: {H2:.3f}")

h = mpt.mean_offset(clave, None, 16)
print(f"  Mean offset: {np.round(h, 3)}")

edg, _ = mpt.edges(clave, None, 16)
print(f"  Edges: {np.round(edg, 3)}")

y = mpt.markov_s(clave, None, 16)
print(f"  Markov(3): {np.round(y, 3)}")

# The circular autocorrelation phase matrix (APM) holds, for each lag
# and phase, how many pairs of onsets fall on successive beats of the
# pulse with that lag, started at that phase. Summed over lags, it
# gives each position a metrical weight: the phase sum.
_, apm_phase, _ = mpt.circ_apm(clave, None, 16)
print(f"  APM phase sum: {np.round(apm_phase, 3)}")
# Onsets 0, 6, 10, and 12 carry the most (48); onset 3, off every
# strong pulse, carries half as much (24).

# The four position-wise measures, one value per pulse of the clave's
# cycle. Pulses holding an onset are drawn dark.
fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
pulses = np.arange(16)
is_onset = np.isin(pulses, clave)
for ax, (vals, lab) in zip(axes, [(h, 'mean offset'), (edg, 'edges'),
                                  (y, 'Markov(3)'),
                                  (apm_phase, 'APM phase sum')]):
    ax.bar(pulses, vals, width=0.7,
           color=np.where(is_onset, 'C0', '#a6c8e6'))
    ax.axhline(0, color='0.5', linewidth=0.8)
    ax.set_ylabel(lab)
axes[0].set_title('Son clave: position-wise measures (onsets dark)')
axes[-1].set_xlabel('pulse')
axes[-1].set_xticks(pulses)
fig.tight_layout()

# See also:
#   demo_sigma_space          soft (sigma > 0) coherence, sameness, and
#                             n-tuple entropy, and what sigma stands for

mpt.set_default(**prev_defaults)
print("\nDone.")
plt.show()
