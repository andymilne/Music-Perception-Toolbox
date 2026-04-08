"""demo_batch_processing.py

Batch computation of perceptual features on experimental data with
automatic deduplication of repeated pitch sets.

Two deduplication workflows are shown:

  1. Paired measures (SPCS) — use batch_cos_sim_exp_tens, which handles
     deduplication internally.

  2. Single-set measures (spectral entropy, template harmonicity,
     tensor harmonicity, roughness) — use the unique/map pattern
     shown below, which works for any function.

The dataset is synthetic: 3 scales × 4 chord types × 12 root
transpositions = 144 trials. Many trials share the same scale or
chord pitch-class content, so deduplication avoids redundant
computation.

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
pitches_a = np.zeros((n_pairs, 7))
pitches_b = np.zeros((n_pairs, 3))
scale_idx = np.zeros(n_pairs, dtype=int)
chord_idx = np.zeros(n_pairs, dtype=int)
root_vals = np.zeros(n_pairs)

idx = 0
for si in range(n_scales):
    for ci in range(n_chords):
        for ri in range(n_roots):
            pitches_a[idx] = scales[si]
            pitches_b[idx] = chord_types[ci] + roots[ri]
            scale_idx[idx] = si
            chord_idx[idx] = ci
            root_vals[idx] = roots[ri]
            idx += 1

print(f"Dataset: {n_pairs} trials "
      f"({n_scales} scales × {n_chords} chord types × {n_roots} roots).\n")

# ===================================================================
#  WORKFLOW 1: Paired measure (SPCS) via batch_cos_sim_exp_tens
#  Deduplication is handled internally.
# ===================================================================

print("=== Workflow 1: SPCS via batch_cos_sim_exp_tens ===\n")

spcs = mpt.batch_cos_sim_exp_tens(
    pitches_a, pitches_b, sigma, r, is_rel, is_per, period,
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
#  WORKFLOW 2: Single-set measures via unique/map deduplication
#
#  The pattern is:
#    1. Sort each row and call np.unique to find unique sets
#    2. Compute the measure once per unique set
#    3. Map results back to all rows via the inverse index
#
#  Demonstrated here for spectral entropy, template harmonicity,
#  tensor harmonicity, and roughness applied to chords (pitches_b).
# ===================================================================

print(f"\n=== Workflow 2: Single-set measures (chord features) ===")

# --- Step 1: Deduplicate ---
sorted_b = np.sort(pitches_b, axis=1)
unique_chords, inverse_map = np.unique(sorted_b, axis=0, return_inverse=True)
n_unique = len(unique_chords)

print(f"\n  {n_pairs} trials → {n_unique} unique chord pitch-class sets.\n")

# --- Step 2: Compute once per unique set ---
# Each function handles spectral enrichment via its own parameters,
# rather than pre-enriching all pitches (which would be prohibitively
# expensive for tensor harmonicity with many partials).
u_spec_ent = np.full(n_unique, np.nan)
u_h_max = np.full(n_unique, np.nan)
u_h_ent = np.full(n_unique, np.nan)
u_tens_harm = np.full(n_unique, np.nan)
u_rough = np.full(n_unique, np.nan)

ref_cents = mpt.convert_pitch(f0, 'hz', 'cents')

for ui in range(n_unique):
    p = unique_chords[ui]
    p = p[~np.isnan(p)]  # strip NaN padding (if any)

    # Spectral entropy (uses spectrum parameter internally)
    u_spec_ent[ui] = mpt.spectral_entropy(p, None, sigma, spectrum=spec)

    # Template harmonicity (uses chord_spectrum parameter)
    u_h_max[ui], u_h_ent[ui] = mpt.template_harmonicity(
        p, None, sigma, chord_spectrum=spec)

    # Tensor harmonicity (uses spectrum parameter for the template)
    u_tens_harm[ui] = mpt.tensor_harmonicity(p, None, sigma, spectrum=spec)

    # Roughness (needs Hz and enriched spectra)
    p_spec, w_spec = mpt.add_spectra(p, None, *spec)
    f_hz = mpt.convert_pitch(p_spec + ref_cents, 'cents', 'hz')
    u_rough[ui] = mpt.roughness(f_hz, w_spec)

# --- Step 3: Map back to all rows ---
spec_ent = u_spec_ent[inverse_map]
h_max = u_h_max[inverse_map]
h_ent = u_h_ent[inverse_map]
tens_harm = u_tens_harm[inverse_map]
rough = u_rough[inverse_map]

# --- Display ---
print(f"  {'Chord':<14s}  {'specEnt':>8s}  {'hMax':>8s}  {'hEnt':>8s}  "
      f"{'tensHarm':>8s}  {'Rough':>8s}")
print('  ' + '-' * 56)

for ui in range(n_unique):
    first_idx = np.where(inverse_map == ui)[0][0]
    ci = chord_idx[first_idx]
    ri = int(root_vals[first_idx])
    label = f'{chord_type_names[ci]} @ {ri}'

    print(f"  {label:<14s}  {u_spec_ent[ui]:8.4f}  {u_h_max[ui]:8.4f}  "
          f"{u_h_ent[ui]:8.4f}  {u_tens_harm[ui]:8.4f}  {u_rough[ui]:8.4f}")

print(f"\n  (Only {n_unique} unique computations needed instead of {n_pairs}.)")

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
