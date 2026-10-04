"""demo_edo_approx.py

Pitch class similarity (PCS) of equal divisions of the octave
(n-EDOs) to a just intonation reference chord, using relative dyad
expectation tensors (r = 2, rel = True, dim = 1).

An example of this type of plot appears as Example 6.3 / Figure 4 in:
  Milne, A. J., Sethares, W. A., Laney, R., & Sharp, D. B. (2011).
  Modelling the similarity of pitch collections with expectation tensors.
  Journal of Mathematics and Music, 5(1), 1-20.

For each value of n, the n-EDO multiset {0, 1200/n, 2*1200/n, ...,
(n-1)*1200/n} is compared to the reference chord. EDOs whose pitch
classes include good approximations to the chord's intervals will have
higher similarity.

An n-EDO is a one-dimensional tuning: every interval is a multiple of a
single generator (1200/n cents). This is why the paper calls these
"one-dimensional approximations". Separately, because r = 2 and
rel = True, the expectation tensor itself is a one-dimensional
density over intervals (dim = 1).

Uses: sim_maet (batched-raw, broadcast form).

Requires: matplotlib (pip install matplotlib)

The MATLAB mirror is demo_edoApprox.m.
"""

import numpy as np
import matplotlib.pyplot as plt

import mpt

# ===================================================================
#  User-adjustable parameters
# ===================================================================

# Reference chord in just intonation (cents)
#   4:5:6 major triad: [0, 386.31, 701.96]
#   5:6:7 subminor triad: [0, 315.64, 582.51]
#   4:5:6:7 dominant seventh: [0, 386.31, 701.96, 968.83]
# The pitches below are 1:3:5, which has the same pitch classes as
# 4:5:6; the tensor is periodic at the octave, so the two give identical
# results.
ref_pitches = np.array([0, np.log2(3), np.log2(5)]) * 1200
ref_weights = None   # weights for reference pitches (None = all ones;
                     # if specified, must be same length as ref_pitches)
ref_name = '4:5:6 JI major triad'

# Range of EDOs to test
n_min = 2
n_max = 102

# Expectation tensor parameters
sigma = 6         # Gaussian smoothing width (cents)
r = 2             # dyad expectation tensor
rel = True     # relative (transposition-invariant)
per = True     # periodic (pitch-class equivalence)
period = 1200     # one octave in cents

# ===================================================================
#  Build pitch matrices
# ===================================================================

edo_range = np.arange(n_min, n_max + 1)
n_edos = len(edo_range)

# Reference: a single 1-D vector — broadcast across all EDO rows of
# p_mat_b by sim_maet.
# EDO multisets: NaN-padded to n_max columns (the most pitches in any EDO)
p_mat_b = np.full((n_edos, n_max), np.nan)
for i, n in enumerate(edo_range):
    edo = np.arange(n) * (1200 / n)
    p_mat_b[i, :n] = edo

# ===================================================================
#  Compute similarities
# ===================================================================

print(f"Computing PCS of {n_edos} EDOs against {ref_name}...")
s = mpt.sim_maet(
    ref_pitches, ref_weights, p_mat_b, None,
    sigma, r, rel, per, period,
    verbose=True,
)
print("Done.")

# ===================================================================
#  Plot
# ===================================================================

fig, ax = plt.subplots(figsize=(12, 5))

# Stem plot: emphasizes the discrete nature of EDOs
markerline, stemlines, baseline = ax.stem(
    edo_range, s, linefmt='-', markerfmt='o', basefmt=' '
)
plt.setp(stemlines, linewidth=0.8, color=(0.2, 0.2, 0.6))
plt.setp(markerline, markersize=4, color=(0.2, 0.2, 0.6))

ax.set_xlabel('n-EDO')
ax.set_ylabel('Pitch class similarity')
ax.set_title(f'PCS of n-EDOs with {ref_name}\n'
             f'(r = {r}, rel = {rel}, σ = {sigma} cents)')
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

print(f"\nTop {n_labels} EDOs by PCS with {ref_name}:")
print(f"{'n-EDO':<8s}  {'PCS'}")
print('-' * 20)
for li in range(min(n_labels, n_edos)):
    idx = sort_idx[li]
    print(f"{edo_range[idx]:<8d}  {s[idx]:.3f}")

plt.show()
