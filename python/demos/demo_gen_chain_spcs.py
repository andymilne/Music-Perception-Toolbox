"""demo_gen_chain_spcs.py

Spectral pitch class similarity (SPCS) of generator-chain tunings
against a reference chord, as the generator interval is swept
continuously.

A generator-chain of cardinality n is a scale formed by stacking n
copies of a generating interval, reduced modulo the period:
  pitches = mod(arange(n) * gen, period)

Produces both a linear and a circular (polar) plot.

Examples of this type of plot appear as Examples 6.4–6.5 / Figures 5–7
in:
  Milne, A. J., Sethares, W. A., Laney, R., & Sharp, D. B. (2011).
  Modelling the similarity of pitch collections with expectation tensors.
  Journal of Mathematics and Music, 5(1), 1-20.

Port of demo_genChainSpcs.m from the MATLAB Music Perception Toolbox v2.

Requires: matplotlib (pip install matplotlib)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import mpt

# ===================================================================
#  User-adjustable parameters
# ===================================================================

# Reference chord (in cents)
ref_pitches = [0, 1200 * np.log2(5 / 4), 1200 * np.log2(6 / 4)]
ref_name = '4:5:6 JI major triad'

# Generator-chain parameters
n_tones = 19
gen_step = 0.5   # step size for generator sweep (cents)

# Expectation tensor parameters
sigma = 10
r = 2
is_rel = True
is_per = True
period = 1200

# Display options
invert_circular = True   # True: higher similarity towards centre
n_labels = 15
min_label_spacing = 9    # minimum spacing between labels (cents)

# ===================================================================
#  Build pitch matrices (0 to period/2 only)
# ===================================================================

half_range = np.arange(0, period / 2 + gen_step / 2, gen_step)
n_half = len(half_range)

# Reference chord: same for every row
pitches_a = np.tile(ref_pitches, (n_half, 1))

# Generator-chain pitch sets
pitches_b = np.full((n_half, n_tones), np.nan)
for i, gen in enumerate(half_range):
    pitches_b[i, :] = np.mod(np.arange(n_tones) * gen, period)

# ===================================================================
#  Compute similarities
# ===================================================================

print(f"Computing SPCS of {n_tones}-tone generator-chains "
      f"(gen = 0 to {period / 2:.1f}, step {gen_step}) "
      f"against {ref_name}...")

s_half = mpt.batch_cos_sim_exp_tens(
    pitches_a, pitches_b, sigma, r, is_rel, is_per, period,
    verbose=False,
)
s_half = np.round(s_half, 3)
print("Done.")

# Reflect to full range [0, period)
gen_range = np.arange(0, period, gen_step)
s = np.concatenate([s_half, s_half[-2:0:-1]])

# Trim or pad to match gen_range
if len(s) < len(gen_range):
    s = np.concatenate([s, np.full(len(gen_range) - len(s), s[-1])])
elif len(s) > len(gen_range):
    s = s[:len(gen_range)]

# ===================================================================
#  Find peaks
# ===================================================================

pk_idx, pk_props = find_peaks(s, prominence=0.01)
pk_vals = s[pk_idx]

# Sort by height (descending)
sort_order = np.argsort(pk_vals)[::-1]
pk_idx = pk_idx[sort_order]
pk_vals = pk_vals[sort_order]

# ===================================================================
#  Linear plot
# ===================================================================

fig1, ax1 = plt.subplots(figsize=(12, 5))

ax1.plot(gen_range, s, color=(0.2, 0.2, 0.6), linewidth=0.8)
ax1.set_xlabel('Generator (cents)')
ax1.set_ylabel('Spectral pitch class similarity')
ax1.set_title(f'SPCS of {n_tones}-tone generator-chains with {ref_name}\n'
              f'(r = {r}, is_rel = {is_rel}, σ = {sigma} cents, '
              f'period = {period} cents)')
ax1.set_xlim(0, period)
ax1.grid(True, alpha=0.3)

labelled_gens = []
labelled = 0
for li in range(len(pk_idx)):
    gen_val = gen_range[pk_idx[li]]
    if any(abs(gen_val - g) < min_label_spacing for g in labelled_gens):
        continue
    ax1.plot(gen_val, pk_vals[li], 'r.', markersize=3)
    ax1.annotate(f'  {gen_val:.1f}', (gen_val, pk_vals[li]),
                 fontsize=7, color=(0.8, 0, 0), ha='left', va='bottom')
    labelled_gens.append(gen_val)
    labelled += 1
    if labelled >= n_labels:
        break

plt.tight_layout()

# ===================================================================
#  Circular plot
# ===================================================================

theta = 2 * np.pi * gen_range / period
theta_c = np.append(theta, theta[0])
s_c = np.append(s, s[0])

if invert_circular:
    r_c = 1 - s_c
else:
    r_c = s_c

fig2, ax2 = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

ax2.plot(theta_c, r_c, color=(0.2, 0.2, 0.6), linewidth=1)
ax2.set_theta_zero_location('N')
ax2.set_theta_direction(-1)  # clockwise
ax2.set_rlim(0, 1)

# Tick labels in cents
n_ticks = 12
tick_angles = np.linspace(0, 2 * np.pi * (1 - 1 / n_ticks), n_ticks)
tick_labels = [f'{a * period / (2 * np.pi):.0f}' for a in tick_angles]
ax2.set_xticks(tick_angles)
ax2.set_xticklabels(tick_labels)
ax2.set_yticklabels([])

title_extra = 'centre = most similar' if invert_circular else 'perimeter = most similar'
ax2.set_title(f'SPCS of {n_tones}-tone generator-chains with {ref_name}\n'
              f'(r = {r}, is_rel = {is_rel}, σ = {sigma}, '
              f'period = {period}; {title_extra})',
              pad=20)

# Label top peaks (ranked by height, minimum spacing in cents)
labelled_gens_c = []
labelled_c = 0
for li in range(len(pk_idx)):
    gen_val = gen_range[pk_idx[li]]
    if any(abs(gen_val - g) < min_label_spacing for g in labelled_gens_c):
        continue
    peak_theta = 2 * np.pi * gen_val / period
    if invert_circular:
        peak_r = 1 - pk_vals[li]
        label_r = max(0, peak_r - 0.05)
    else:
        peak_r = pk_vals[li]
        label_r = min(1, peak_r + 0.05)
    ax2.plot(peak_theta, peak_r, 'r.', markersize=3)
    ax2.annotate(f'{gen_val:.1f}', (peak_theta, label_r),
                 fontsize=7, color=(0.8, 0, 0), ha='center', va='center')
    labelled_gens_c.append(gen_val)
    labelled_c += 1
    if labelled_c >= n_labels:
        break

plt.tight_layout()

# ===================================================================
#  Console output: top peaks
# ===================================================================

half_period = period / 2
print(f"\nTop peaks (generator values with highest SPCS):")
print(f"{'gen (cents)':<14s}  {'(complement)':<14s}  {'SPCS'}")
print('-' * 42)
labelled_t = []
printed = 0
for li in range(len(pk_idx)):
    gen_val = gen_range[pk_idx[li]]
    if gen_val > half_period:
        gen_val = period - gen_val
    complement = period - gen_val

    if any(abs(gen_val - g) < min_label_spacing for g in labelled_t):
        continue

    idx_gen = max(0, min(int(round(gen_val / gen_step)), len(s) - 1))
    idx_comp = max(0, min(int(round(complement / gen_step)), len(s) - 1))
    best_spcs = max(s[idx_gen], s[idx_comp])

    print(f"{gen_val:<14.1f}  ({complement:<10.1f})  {best_spcs:.3f}")
    labelled_t.append(gen_val)
    printed += 1
    if printed >= n_labels:
        break

plt.show()
