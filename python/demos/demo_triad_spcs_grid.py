"""demo_triad_spcs_grid.py

Spectral pitch class similarity (SPCS) of all possible 12-EDO triads
containing a perfect fifth, relative to a user-specified reference triad.

An example of this type of plot appears as Figure 3 in:
  Milne, A. J., Sethares, W. A., Laney, R., & Sharp, D. B. (2011).
  Modelling the similarity of pitch collections with expectation tensors.
  Journal of Mathematics and Music, 5(1), 1-20.

Port of demo_triadSpcsGrid.m from the MATLAB Music Perception Toolbox v2.

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

# Reference triad (in cents). The plot is centred on this chord.
ref_pitches = [0, 400, 700]   # C major
ref_name = 'C major'

# Spectral parameters
n_harm = 24
rho = 1

# Expectation tensor parameters
sigma = 10
r = 1
is_rel = False
is_per = True
period = 1200

# Display options
show_labels = True

# ===================================================================
#  Fixed definitions
# ===================================================================

pcs = np.arange(0, 1200, 100)
n_pcs = len(pcs)

note_names = ['C', 'C#', 'D', 'Eb', 'E', 'F',
              'F#', 'G', 'Ab', 'A', 'Bb', 'B']

ref_root = ref_pitches[0]
ref_third = ref_pitches[1]

# ===================================================================
#  Compute similarities
# ===================================================================

# Build all 144 comparison triads: {root, third, root + 700}
root_grid, third_grid = np.meshgrid(pcs, pcs)
pitches_b = np.column_stack([
    root_grid.ravel(),
    third_grid.ravel(),
    root_grid.ravel() + 700,
])

# Reference triad repeated
pitches_a = np.tile(ref_pitches, (n_pcs * n_pcs, 1))

print(f"Computing SPCS for {ref_name} reference "
      f"(N={n_harm}, rho={rho})...")

sim_vector = mpt.batch_cos_sim_exp_tens(
    pitches_a, pitches_b, sigma, r, is_rel, is_per, period,
    spectrum=['harmonic', n_harm, 'powerlaw', rho],
    verbose=False,
)

sim_matrix = sim_vector.reshape(n_pcs, n_pcs)
print("  Done.")

# ===================================================================
#  Centre the axes on the reference triad
# ===================================================================

root_offsets = (pcs - ref_root + 600) % 1200 - 600
third_offsets = (pcs - ref_third + 600) % 1200 - 600

root_sort_idx = np.argsort(root_offsets)
third_sort_idx = np.argsort(third_offsets)

root_offsets_sorted = root_offsets[root_sort_idx]
third_offsets_sorted = third_offsets[third_sort_idx]

sim_sorted = sim_matrix[np.ix_(third_sort_idx, root_sort_idx)]

root_labels = [note_names[i] for i in root_sort_idx]
third_labels = [note_names[i] for i in third_sort_idx]

# ===================================================================
#  Plot
# ===================================================================

fig, ax = plt.subplots(figsize=(8, 7))

im = ax.imshow(
    sim_sorted,
    extent=[
        root_offsets_sorted[0] - 50, root_offsets_sorted[-1] + 50,
        third_offsets_sorted[0] - 50, third_offsets_sorted[-1] + 50,
    ],
    origin='lower',
    aspect='equal',
    cmap='gray_r',  # darker = higher similarity
    interpolation='nearest',
)

ax.set_xticks(root_offsets_sorted)
ax.set_xticklabels(root_labels)
ax.set_yticks(third_offsets_sorted)
ax.set_yticklabels(third_labels)
ax.set_xlabel('Root of fifth')
ax.set_ylabel('Third')

cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label('Spectral pitch class similarity')

ax.set_title(f'Spectral pitch class similarity: {ref_name} reference\n'
             f'(N = {n_harm} harmonics, ρ = {rho}, σ = {sigma} cents)')

# ===================================================================
#  Chord labels
# ===================================================================

if show_labels:
    for ni in range(n_pcs):
        root = pcs[ni]
        rx = (root - ref_root + 600) % 1200 - 600

        # Major triad: third at root + 400
        maj_third_pc = (root + 400) % 1200
        maj_ty = (maj_third_pc - ref_third + 600) % 1200 - 600
        ax.text(rx, maj_ty, note_names[ni],
                fontsize=8, fontweight='bold', color=(0.8, 0, 0),
                ha='center', va='center')

        # Minor triad: third at root + 300
        min_third_pc = (root + 300) % 1200
        min_ty = (min_third_pc - ref_third + 600) % 1200 - 600
        min_label = note_names[ni][0].lower() + note_names[ni][1:]
        ax.text(rx, min_ty, min_label,
                fontsize=8, fontweight='bold', color=(0, 0, 0.8),
                ha='center', va='center')

    # Mark reference triad at origin
    ax.plot(0, 0, 'ws', markersize=18, markeredgewidth=2,
            markerfacecolor='none', markeredgecolor='white')

plt.tight_layout()
print("Done.")
plt.show()
