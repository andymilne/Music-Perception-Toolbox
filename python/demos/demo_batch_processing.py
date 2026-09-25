"""demo_batch_processing.py

Analysing experimental data: perceptual features for a table of trials.

A typical experiment presents a stimulus per trial and the analyst
wants one or more perceptual predictors for every trial, aligned with
the responses. This demo builds a synthetic trial table — 3 scales x 4
chord types x 12 root transpositions = 144 trials — and computes, for
every trial, a paired measure (the spectral pitch-class similarity of
the chord to its scale, SPCS) and several single-set measures of the
chord (spectral entropy, template harmonicity, tensor harmonicity, and
roughness), then tabulates and plots them.

The point of method is that the trial table goes straight in. Every
toolbox feature that accepts a 2-D pitch matrix (one row per trial,
NaN-padded when the chords differ in size) — sim_maet in its
batched-raw mode, spectral_entropy, template_harmonicity,
tensor_harmonicity, virtual_pitches — deduplicates its rows internally
by a canonical key, so the 144 chord rows here cost 4 chord-type
computations, and the 144 (scale, chord) pairs only as many distinct
pairs as there are. No manual unique() step is needed.

The deduplication follows the analysis: each feature keys its rows by
what its value depends on. The single-set features (spectral entropy,
and template and tensor harmonicity) are transposition-invariant
measures of a chord and take no mode flags, so their key ignores
transposition and pitch order: the twelve transpositions of a chord type
collapse to one computation, and the 144 chord rows to 4. SPCS depends
on where the chord lies against its scale, so its key is the (scale,
chord) pair, up to transposing both together, under the analysis
parameters (sigma, r, is_rel, is_per, period). Here, with is_per = True,
pitch is read as pitch class, so the augmented triad's transpositions by
a major third, which give one pitch-class set, collapse, and the 144
pairs cost 120 computations; under is_rel = True (which needs r >= 2)
every transposition of a chord would collapse, a relative density being
transposition-invariant. The analyst changes the flags and the saving
follows, with no change to the calling code. The one feature without a
batched form, roughness (which depends on absolute frequency and so
cannot share work across transpositions), is looped over the distinct
rows.

Workflow 3 shows a second, quite different sense of "batch". Rows of a
2-D pitch matrix are single multisets over one attribute, and what
Workflows 1 and 2 exploit is deduplication *within* such a matrix. A
multi-attribute analysis has no rows to deduplicate: each item is a
whole pre-MAET. Batching there means passing a *list* of them where a
list of densities would go, which loops rather than collapses — the
saving is in the calling code, not in the arithmetic.

Uses: sim_maet, spectral_entropy, template_harmonicity,
      tensor_harmonicity, add_spectra, roughness, transform_attributes,
      pack_pre_maet, flat_specs, translate_attributes, sweep_sim_maet,
      build_maet, set_default.

Requires: matplotlib (pip install matplotlib)

The MATLAB mirror is demo_batchProcessing.m.
"""

import numpy as np
import matplotlib.pyplot as plt

import mpt

# The toolbox's one-time informational hints (which route a call took,
# and the like) are switched off for a tidy printout, and restored at
# the end.
prev_defaults = mpt.set_default(show_hints=False)

# ===================================================================
#  User-adjustable parameters
# ===================================================================

# Spectral parameters
n_harm = 24
rho = 1
spec = ['harmonic', n_harm, 'powerlaw', rho]

# Expectation tensor parameters
sigma = 10
r = 1
is_rel = False
is_per = True
period = 1200

# Reference pitch for roughness (Hz)
f0 = 261.63  # middle C

# ===================================================================
#  Create synthetic dataset
# ===================================================================

# Three 7-note scales (cents)
scales = np.array([
    [0, 200, 400, 500, 700, 900, 1100],   # diatonic
    [0, 200, 300, 500, 700, 800, 1100],   # harmonic minor
    [0, 200, 300, 500, 700, 900, 1100],   # melodic minor
])
scale_names = ['Diatonic', 'Harmonic minor', 'Melodic minor']

# Four chord types (cents relative to root)
chord_types = np.array([
    [0, 400, 700],   # major
    [0, 300, 700],   # minor
    [0, 300, 600],   # diminished
    [0, 400, 800],   # augmented
])
chord_type_names = ['Major', 'Minor', 'Dim', 'Aug']

# 12 root pitch classes
roots = np.arange(0, 1200, 100)
n_roots = len(roots)

n_scales = len(scales)
n_chords = len(chord_types)
n_pairs = n_scales * n_chords * n_roots

# Build all (scale, transposed chord) pairs
p_mat_a = np.zeros((n_pairs, 7))
p_mat_b = np.zeros((n_pairs, 3))
scale_idx = np.zeros(n_pairs, dtype=int)
chord_idx = np.zeros(n_pairs, dtype=int)
root_vals = np.zeros(n_pairs)

idx = 0
for si in range(n_scales):
    for ci in range(n_chords):
        for ri in range(n_roots):
            p_mat_a[idx] = scales[si]
            p_mat_b[idx] = chord_types[ci] + roots[ri]
            scale_idx[idx] = si
            chord_idx[idx] = ci
            root_vals[idx] = roots[ri]
            idx += 1

print(f"Dataset: {n_pairs} trials "
      f"({n_scales} scales × {n_chords} chord types × {n_roots} roots).\n")

# ===================================================================
#  WORKFLOW 1: Paired measure (SPCS) via batched sim_maet
#  Two 2-D matrices, one row per trial, dispatch to batched-raw mode;
#  repeated rows and repeated (scale, chord) pairs are deduplicated
#  internally, and the spectrum is applied inside the call.
# ===================================================================

print("=== Workflow 1: SPCS via batched sim_maet ===\n")

spcs = mpt.sim_maet(
    p_mat_a, None, p_mat_b, None,
    sigma, r, is_rel, is_per, period,
    spectrum=spec,
)
spcs = np.round(spcs, 3)

# The rows were laid out scale by chord type by root, so the profile
# reshapes straight into a scale x chord x root array.
spcs_grid = spcs.reshape(n_scales, n_chords, n_roots)

# Display as scale × chord × root tables
for si in range(n_scales):
    print(f"\n  {scale_names[si]}:")
    header = '  ' + f"{'':8s}" + ''.join(f'{roots[ri]:6d}' for ri in range(n_roots))
    print(header)

    for ci in range(n_chords):
        row = f'  {chord_type_names[ci]:8s}' + ''.join(
            f'{v:6.3f}' for v in spcs_grid[si, ci])
        print(row)

# ===================================================================
#  WORKFLOW 2: Single-set measures on the trial table
#
#  The batched features take the 144-row chord matrix as it is: each
#  deduplicates its rows internally (a canonical key invariant to
#  transposition and pitch order, so the 12 roots x 4 types collapse to
#  4 computations) and returns one value per trial. Each applies the
#  spectrum through its own argument; pre-enriching all pitches would
#  be prohibitively expensive for tensor harmonicity with many partials.
# ===================================================================

print(f"\n=== Workflow 2: Single-set measures (chord features) ===\n")

spec_ent = mpt.spectral_entropy(p_mat_b, None, sigma, spectrum=spec)
h_max, h_ent = mpt.template_harmonicity(
    p_mat_b, None, sigma, chord_spectrum=spec)
tens_harm = mpt.tensor_harmonicity(p_mat_b, None, sigma, spectrum=spec)

# --- Roughness: the one feature without a batched form ---
# roughness takes one multiset of partials in Hz and depends on their
# absolute frequencies, so transpositions do not share work. Loop over
# the distinct chord rows (transposition included) and map back.
sorted_b = np.sort(p_mat_b, axis=1)
unique_chords, inverse_map = np.unique(sorted_b, axis=0, return_inverse=True)
n_unique = len(unique_chords)
print(f"  {n_pairs} trials → {n_unique} distinct chords for the roughness loop.\n")

u_rough = np.full(n_unique, np.nan)
ref_cents = mpt.transform_attributes(f0, None, ('hz', 'cents'))
for ui in range(n_unique):
    p = unique_chords[ui]
    p = p[~np.isnan(p)]  # strip NaN padding (if any)
    p_spec, w_spec = mpt.add_spectra(p, None, *spec)
    f_hz = mpt.transform_attributes(p_spec + ref_cents, None, ('cents', 'hz'))
    u_rough[ui] = mpt.roughness(f_hz, w_spec)
rough = u_rough[inverse_map]

# --- Display: one line per distinct chord (its first trial) ---
print(f"  {'Chord':<14s}  {'specEnt':>8s}  {'hMax':>8s}  {'hEnt':>8s}  "
      f"{'tensHarm':>8s}  {'Rough':>8s}")
print('  ' + '-' * 56)

for ui in range(n_unique):
    first_idx = np.where(inverse_map == ui)[0][0]
    ci = chord_idx[first_idx]
    ri = int(root_vals[first_idx])
    label = f'{chord_type_names[ci]} @ {ri}'

    print(f"  {label:<14s}  {spec_ent[first_idx]:8.4f}  {h_max[first_idx]:8.4f}  "
          f"{h_ent[first_idx]:8.4f}  {tens_harm[first_idx]:8.4f}  "
          f"{rough[first_idx]:8.4f}")

print(f"\n  (The batched features received all {n_pairs} rows and computed "
      f"{n_chords} chord types; roughness ran {n_unique} times.)")

# ===================================================================
#  Plot: SPCS heatmaps
# ===================================================================

fig, axes = plt.subplots(1, n_scales, figsize=(5 * n_scales, 4))
if n_scales == 1:
    axes = [axes]

for si in range(n_scales):
    ax = axes[si]
    im = ax.imshow(spcs_grid[si], aspect='auto', origin='upper',
                   extent=[-50, 1150, n_chords - 0.5, -0.5],
                   cmap='viridis')
    ax.set_yticks(range(n_chords))
    ax.set_yticklabels(chord_type_names)
    ax.set_xlabel('Root (cents)')
    ax.set_title(scale_names[si])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle('SPCS: chord fit at each root', fontweight='bold')
plt.tight_layout()

# ===================================================================
#  WORKFLOW 3: A list of pre-MAETs — batching of a different kind
#
#  Everything above batches ROWS: a 2-D pitch matrix whose rows are
#  single multisets over one attribute, deduplicated internally by a
#  canonical key so that 144 rows cost 4 computations. That collapse is
#  possible because the rows are commensurable — same attribute, same
#  geometry, differing only in their values.
#
#  A multi-attribute item has no row to collapse: it is a whole
#  pre-MAET, with its own event count and its own per-attribute
#  geometry. So the multi-attribute analogue of a batch is a LIST, and
#  a list of pre-MAETs goes wherever a list of densities goes. The
#  functions build each entry and iterate; nothing is deduplicated,
#  because in general nothing is repeated. What the list form saves is
#  the calling code — no per-item build, no loop, one call that returns
#  one value per item — not arithmetic.
#
#  The exception that proves the rule is a translation sweep. Its
#  entries DO share one geometry and differ only by an offset, so
#  sweep_sim_maet computes the whole sweep in one pass rather than one
#  inner product per offset — a genuine collapse, and the one place
#  where a multi-attribute batch is cheaper than the loop it replaces.
#  translate_attributes attaches its offsets to the list it returns, so
#  sim_maet takes that pass at the call site itself.
# ===================================================================

print("\n=== Workflow 3: A list of pre-MAETs (batching, other sense) ===\n")

# Four two-attribute items: a pitch-class attribute and an onset-time
# attribute. They are NOT commensurable rows -- the second has four
# events where the others have three -- so no canonical key could
# collapse them.
def _item(pcs, onsets):
    p = [np.asarray(pcs, dtype=float)[None, :],
         np.asarray(onsets, dtype=float)[None, :]]
    return mpt.pack_pre_maet(p, specs=mpt.flat_specs(
        p, name=["pitch class", "onset"], sigma=[35.0, 0.25],
        is_per=[True, False], period=[1200.0, 0.0]))

items = [_item([0, 400, 700], [0, 1, 2]),
         _item([0, 300, 700, 1000], [0, 1, 2, 3]),
         _item([200, 500, 900], [0, 1, 2]),
         _item([0, 400, 700], [0, 1, 2])]
reference = items[0]

# One call, one value per item. The same call with pre-built densities
# would be identical; the pre-MAETs simply save building them.
sims = mpt.sim_maet(reference, items, verbose=False)
print("  sim_maet(reference, [pm_1, ..., pm_4])")
for i, s in enumerate(sims):
    print(f"    item {i + 1}: {float(s):.4f}")
print("  (item 1 is the reference; item 4 repeats it.)")
print("  Each entry was built and compared in turn -- four densities,")
print("  four inner products. Nothing collapsed: the items differ in")
print("  event count and content, so there is no repeated work to find.")

# The sweep is the exception: one geometry, M offsets. sim_maet reads
# the offsets translate_attributes attached to its list, and
# sweep_sim_maet, given them directly, computes the same sweep. It
# chooses its route by cost and coverage: here, the swept attribute
# (pitch class) being periodic, the orbit route.
offsets = np.array([[0.0, 100.0, 200.0, 300.0]])
pm_sweep = mpt.translate_attributes(reference, [offsets, None])
tagged_sims = mpt.sim_maet(reference, pm_sweep, verbose=False)

dens_ref = mpt.build_maet(reference, verbose=False)
sweep_sims = mpt.sweep_sim_maet(
    dens_ref, dens_ref, np.vstack([offsets, np.zeros_like(offsets)]),
    verbose=False)

print("\n  translate_attributes(reference, [[0, 100, 200, 300], None])")
print("    sim_maet on the tagged list ->",
      " ".join(f"{float(s):.4f}" for s in tagged_sims))
print("    sweep_sim_maet, one pass    ->",
      " ".join(f"{float(s):.4f}" for s in sweep_sims))
print(f"  The two agree to "
      f"{np.max(np.abs(np.asarray(tagged_sims, dtype=float) - sweep_sims)):.1e}. "
      "Here the entries DO share a geometry")
print("  and differ by a known offset, so the sweep is a genuine")
print("  collapse — the one place where a multi-attribute batch is")
print("  cheaper than the loop it replaces.")

mpt.set_default(**prev_defaults)
print("\nDone.")
plt.show()
