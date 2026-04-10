"""demo_edo_approx.py

Spectral pitch class similarity (SPCS) of equal divisions of the octave
(n-EDOs) to a just intonation reference chord, using relative dyad
expectation tensors (r = 2, is_rel = True, dim = 1).

An example of this type of plot appears as Example 6.3 / Figure 4 in:
  Milne, A. J., Sethares, W. A., Laney, R., & Sharp, D. B. (2011).
  Modelling the similarity of pitch collections with expectation tensors.
  Journal of Mathematics and Music, 5(1), 1-20.

Port of demo_edoApprox.m from the MATLAB Music Perception Toolbox v2.

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

# Reference chord in just intonation (cents)
#   4:5:6 major triad: [0, 386.31, 701.96]
#   5:6:7 subminor triad: [0, 315.64, 582.51]
#   4:5:6:7 dominant seventh: [0, 386.31, 701.96, 968.83]
ref_pitches = [0, 1200 * np.log2(5 / 4), 1200 * np.log2(6 / 4)]
ref_name = '4:5:6 JI major triad'

# Range of EDOs to test
n_min = 2
n_max = 102

# Expectation tensor parameters
sigma = 10
r = 2
is_rel = True
is_per = True
period = 1200

# ===================================================================
#  Build pitch matrices
# ===================================================================

edo_range = np.arange(n_min, n_max + 1)
n_edos = len(edo_range)
max_n = n_max

# Reference chord: same for every row
p_mat_a = np.tile(ref_pitches, (n_edos, 1))

# EDO multisets: NaN-padded
p_mat_b = np.full((n_edos, max_n), np.nan)
for i, n in enumerate(edo_range):
    edo = np.arange(n) * (1200 / n)
    p_mat_b[i, :n] = edo

# ===================================================================
#  Compute similarities
# ===================================================================

print(f"Computing SPCS of {n_edos} EDOs against {ref_name}...")
s = mpt.batch_cos_sim_exp_tens(
    p_mat_a, p_mat_b, sigma, r, is_rel, is_per, period,
    verbose=False,
)
s = np.round(s, 3)
print("Done.")

# ===================================================================
#  Plot
# ===================================================================

fig, ax = plt.subplots(figsize=(12, 5))

# Stem plot
markerline, stemlines, baseline = ax.stem(
    edo_range, s, linefmt='-', markerfmt='o', basefmt=' '
)
plt.setp(stemlines, linewidth=0.8, color=(0.2, 0.2, 0.6))
plt.setp(markerline, markersize=4, color=(0.2, 0.2, 0.6))

ax.set_xlabel('n-EDO')
ax.set_ylabel('Spectral pitch class similarity')
ax.set_title(f'SPCS of n-EDOs with {ref_name}\n'
             f'(r = {r}, is_rel = {is_rel}, σ = {sigma} cents)')
ax.set_xlim(n_min - 1, n_max + 1)
ax.grid(True, alpha=0.3)

# Label the top peaks
n_labels = 10
sort_idx = np.argsort(s)[::-1]
for li in range(min(n_labels, n_edos)):
    idx = sort_idx[li]
    n = edo_range[idx]
    ax.annotate(f'  {n}', (n, s[idx]),
                fontsize=8, fontweight='bold',
                ha='left', va='bottom')

plt.tight_layout()

# ===================================================================
#  Console output: top EDOs
# ===================================================================

print(f"\nTop {n_labels} EDOs by SPCS with {ref_name}:")
print(f"{'n-EDO':<8s}  {'SPCS'}")
print('-' * 20)
for li in range(min(n_labels, n_edos)):
    idx = sort_idx[li]
    print(f"{edo_range[idx]:<8d}  {s[idx]:.3f}")

plt.show()
