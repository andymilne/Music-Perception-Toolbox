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
NaN-padded when the chords differ in size) — cos_sim_exp_tens in its
batched-raw mode, spectral_entropy, template_harmonicity,
tensor_harmonicity, virtual_pitches — deduplicates its rows internally
by a canonical key, so the 144 chord rows here cost 4 chord-type
computations, and the 144 (scale, chord) pairs only as many distinct
pairs as there are. No manual unique() step is needed.

The deduplication is fully automatic in the sense that matters: the key
is built from the density the call would form, so it follows the
analysis parameters (sigma, r, is_rel, is_per, period) rather than
guessing. Two rows collapse only when their densities are structurally
identical under those settings. Here, with is_per = True and
is_rel = False, the twelve transpositions of a chord type share a
pitch-class multiset and collapse to one computation; under
is_per = False they would be twelve distinct chords and none would
collapse, and under is_rel = True every transposition would collapse
whether periodic or not. The analyst changes the mode flags and the
saving follows, with no change to the calling code. The one feature
without a batched form, roughness (which depends on absolute frequency
and so cannot share work across transpositions), is looped over the
distinct rows.

Requires: matplotlib (pip install matplotlib)
"""

import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import mpt

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
#  WORKFLOW 1: Paired measure (SPCS) via batched cos_sim_exp_tens
#  Two 2-D matrices, one row per trial, dispatch to batched-raw mode;
#  repeated rows and repeated (scale, chord) pairs are deduplicated
#  internally, and the spectrum is applied inside the call.
# ===================================================================

print("=== Workflow 1: SPCS via batched cos_sim_exp_tens ===\n")

spcs = mpt.cos_sim_exp_tens(
    p_mat_a, None, p_mat_b, None,
    sigma, r, is_rel, is_per, period,
    spectrum=spec,
)
spcs = np.round(spcs, 3)

# Display as scale × chord × root tables
for si in range(n_scales):
    print(f"\n  {scale_names[si]}:")
    header = '  ' + f"{'':8s}" + ''.join(f'{roots[ri]:6d}' for ri in range(n_roots))
    print(header)

    for ci in range(n_chords):
        row = f'  {chord_type_names[ci]:8s}'
        for ri in range(n_roots):
            mask = (scale_idx == si) & (chord_idx == ci) & (root_vals == roots[ri])
            row += f'{spcs[mask][0]:6.3f}'
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
    S = np.full((n_chords, n_roots), np.nan)
    for ci in range(n_chords):
        for ri in range(n_roots):
            mask = (scale_idx == si) & (chord_idx == ci) & (root_vals == roots[ri])
            S[ci, ri] = spcs[mask][0]

    im = ax.imshow(S, aspect='auto', origin='upper',
                   extent=[-50, 1150, n_chords - 0.5, -0.5],
                   cmap='viridis')
    ax.set_yticks(range(n_chords))
    ax.set_yticklabels(chord_type_names)
    ax.set_xlabel('Root (cents)')
    ax.set_title(scale_names[si])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle('SPCS: chord fit at each scale degree', fontweight='bold')
plt.tight_layout()

print("\nDone.")
plt.show()
