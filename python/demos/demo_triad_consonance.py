"""demo_triad_consonance.py

Computes and plots consonance-related features for triads
[0, interval1, interval2] over a grid of intervals.

Five measures are available (select which to plot below):

  'tmpl_max'   — Template harmonicity hMax (Milne 2013)
  'tmpl_ent'   — Template harmonicity -hEntropy (Harrison 2020)
  'tensor'     — Tensor harmonicity (Smit et al. 2019)
  'spec_ent'   — -Spectral entropy (Milne et al. 2017)
  'rough'      — -Roughness (Sethares 1993)

Each plot has interval1 on the x-axis and interval2 on the y-axis.
Peaks correspond to consonance for all measures (negative measures
are plotted so that peaks = consonance). The plots are symmetric
about the diagonal.

Interactive gamma (power compression) and colormap shift sliders
are provided for exploration.

Port of demo_triadConsonance.m from the MATLAB Music Perception Toolbox v2.

Requires: matplotlib (pip install matplotlib)
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import mpt

# ===================================================================
#  User-editable parameters
# ===================================================================

# Select which measures to plot (comment out to skip)
plot_measures = [
    'tmpl_max',     # Template harmonicity: hMax (Milne 2013)
    'tmpl_ent',     # Template harmonicity: -hEntropy (Harrison 2020)
    'tensor',       # Tensor harmonicity (Smit et al. 2019)
    'spec_ent',     # -Spectral entropy (Milne et al. 2017)
    'rough',        # -Roughness (Sethares 1993)
]

# Grid
step = 20         # grid spacing in cents (smaller = finer but slower;
                  #   step=10 takes ~6 min for all 5 measures,
                  #   step=20 takes ~90 s, step=50 takes ~15 s)
max_int = 1200    # maximum interval in cents

# Reference pitch for roughness calculation (Hz)
f0 = 261.63       # middle C (C4)

# Smoothing widths
sigma_tmpl = 10   # template_harmonicity
sigma_tens = 10   # tensor_harmonicity
sigma_ent = 10    # spectral_entropy

# Spectral parameters for each measure
spec_tmpl = ['harmonic', 24, 'powerlaw', 1]
spec_tens = ['harmonic', 24, 'powerlaw', 1]
spec_ent = ['harmonic', 24, 'powerlaw', 1]
spec_rough = ['harmonic', 24, 'powerlaw', 1]

# Tensor harmonicity: template duplication (0 = auto = chord cardinality)
dup_tens = 0

# Initial gamma for visualization
gamma_init = 1.0


# ===================================================================
#  Helper
# ===================================================================


def apply_gamma(vals, gamma):
    """Power compression normalised to [0, 1] then rescaled."""
    mn, mx = np.nanmin(vals), np.nanmax(vals)
    if mx > mn:
        vn = (vals - mn) / (mx - mn)
        return vn ** gamma * (mx - mn) + mn
    return vals.copy()


# ===================================================================
#  Determine which measures are selected
# ===================================================================

do_tmpl_max = 'tmpl_max' in plot_measures
do_tmpl_ent = 'tmpl_ent' in plot_measures
do_tmpl = do_tmpl_max or do_tmpl_ent
do_tensor = 'tensor' in plot_measures
do_spec_ent = 'spec_ent' in plot_measures
do_rough = 'rough' in plot_measures

# ===================================================================
#  Build grid
# ===================================================================

ints = np.arange(0, max_int + step, step)
n_ints = len(ints)
Ga, Gb = np.meshgrid(ints, ints)

if do_tmpl_max:
    tmpl_harm_max = np.full((n_ints, n_ints), np.nan)
if do_tmpl_ent:
    tmpl_harm_ent = np.full((n_ints, n_ints), np.nan)
if do_tensor:
    tens_harm = np.full((n_ints, n_ints), np.nan)
if do_spec_ent:
    spec_ent_grid = np.full((n_ints, n_ints), np.nan)
if do_rough:
    rough_grid = np.full((n_ints, n_ints), np.nan)

ref_cents = mpt.convert_pitch(f0, 'hz', 'cents')

# ===================================================================
#  Precompute tensor harmonicity template (if selected)
# ===================================================================

if do_tensor:
    dup = dup_tens if dup_tens > 0 else 3  # triads
    print(f"Precomputing tensor harmonicity template (r=3, dup={dup})...")
    tp, tw = mpt.add_spectra(np.zeros(dup), np.ones(dup), *spec_tens)
    T = mpt.build_exp_tens(tp, tw, sigma_tens, 3, True, False, 1200, verbose=False)
    print(f"  Done ({T.n_j} ordered triples).")

# ===================================================================
#  Compute features (upper triangle, mirror for symmetry)
# ===================================================================

n_total = n_ints * (n_ints + 1) // 2
n_done = 0
t0 = time.time()

print(f"Computing features for {n_total} triads (step = {step} cents)...")

for i in range(n_ints):
    for j in range(i, n_ints):
        int1 = ints[i]
        int2 = ints[j]

        # Tensor harmonicity
        if do_tensor:
            h = mpt.eval_exp_tens(T, np.array([[int1], [int2]]), verbose=False)
            tens_harm[j, i] = h[0]
            tens_harm[i, j] = h[0]

        # Template harmonicity
        if do_tmpl:
            h_max, h_ent = mpt.template_harmonicity(
                [0, int1, int2], None, sigma_tmpl,
                spectrum=spec_tmpl, chord_spectrum=spec_tmpl
            )
            if do_tmpl_max:
                tmpl_harm_max[j, i] = h_max
                tmpl_harm_max[i, j] = h_max
            if do_tmpl_ent:
                tmpl_harm_ent[j, i] = h_ent
                tmpl_harm_ent[i, j] = h_ent

        # Spectral entropy
        if do_spec_ent:
            H = mpt.spectral_entropy([0, int1, int2], None, sigma_ent,
                                      spectrum=spec_ent)
            spec_ent_grid[j, i] = H
            spec_ent_grid[i, j] = H

        # Roughness
        if do_rough:
            chord_cents = np.array([ref_cents, ref_cents + int1, ref_cents + int2])
            ep, ew = mpt.add_spectra(chord_cents, None, *spec_rough)
            f_hz = mpt.convert_pitch(ep, 'cents', 'hz')
            rough_grid[j, i] = mpt.roughness(f_hz, ew)
            rough_grid[i, j] = rough_grid[j, i]

        # Progress
        n_done += 1
        if n_done % 500 == 0 or n_done == n_total:
            elapsed = time.time() - t0
            rate = n_done / elapsed
            remain = (n_total - n_done) / rate if rate > 0 else 0
            print(f"  {n_done} / {n_total} triads "
                  f"({elapsed:.1f} s elapsed, ~{remain:.0f} s remaining)")

print(f"All features computed in {time.time() - t0:.1f} s.")

# ===================================================================
#  Assemble measures for plotting
# ===================================================================

spec_str = ', '.join(str(x) for x in spec_tmpl)

all_data = []
all_titles = []

if do_tmpl_max:
    all_data.append(tmpl_harm_max)
    all_titles.append(f'Template harmonicity: hMax (Milne 2013)\n{spec_str}, σ={sigma_tmpl}')
if do_tmpl_ent:
    all_data.append(-tmpl_harm_ent)
    all_titles.append(f'Template harmonicity: −hEntropy (Harrison 2020)\n{spec_str}, σ={sigma_tmpl}')
if do_tensor:
    all_data.append(tens_harm)
    all_titles.append(f'Tensor harmonicity (Smit et al. 2019)\n{spec_str}, σ={sigma_tens}, dup={dup}')
if do_spec_ent:
    all_data.append(-spec_ent_grid)
    all_titles.append(f'−Spectral entropy (Milne et al. 2017)\n{spec_str}, σ={sigma_ent}')
if do_rough:
    all_data.append(-rough_grid)
    all_titles.append(f'−Roughness (Sethares 1993)\n{spec_str}, f₀={f0:.1f} Hz')

n_plots = len(all_data)

if n_plots == 0:
    print("No measures selected — nothing to plot.")
    exit()

# ===================================================================
#  Plot
# ===================================================================

n_cols = min(n_plots, 3)
n_rows = int(np.ceil(n_plots / n_cols))

fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(5.5 * n_cols, 4.5 * n_rows + 1.2))
plt.subplots_adjust(bottom=0.15, hspace=0.4, wspace=0.35)

if n_plots == 1:
    axes = np.array([axes])
axes = np.atleast_1d(axes).ravel()

images = []
raw_data_list = []

for mi in range(n_plots):
    ax = axes[mi]
    data = all_data[mi]
    raw_data_list.append(data)

    V = apply_gamma(data, gamma_init)
    im = ax.imshow(
        V, extent=[0, max_int, 0, max_int],
        origin='lower', aspect='equal', cmap='viridis'
    )
    images.append(im)
    ax.set_xlabel('Interval 1 (cents)')
    ax.set_ylabel('Interval 2 (cents)')
    ax.set_title(all_titles[mi], fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Hide unused subplots
for mi in range(n_plots, len(axes)):
    axes[mi].set_visible(False)

fig.suptitle(f'Triad consonance (step = {step} cents)', fontweight='bold')

# Gamma slider
ax_gamma = fig.add_axes([0.15, 0.05, 0.35, 0.025])
s_gamma = Slider(ax_gamma, 'Gamma', 0.01, 1.0, valinit=gamma_init)

# Colormap shift slider
ax_cmap = fig.add_axes([0.15, 0.02, 0.35, 0.025])
s_cmap = Slider(ax_cmap, 'Cmap shift', 0.0, 0.95, valinit=0.0)


def update(val):
    for im, data in zip(images, raw_data_list):
        V = apply_gamma(data, s_gamma.val)
        im.set_data(V)
        v_min, v_max = np.nanmin(V), np.nanmax(V)
        if v_max > v_min:
            new_low = v_min + s_cmap.val * (v_max - v_min)
            im.set_clim(new_low, v_max)
        else:
            im.set_clim(v_min, v_max)
    fig.canvas.draw_idle()


s_gamma.on_changed(update)
s_cmap.on_changed(update)

print(f"\nDone. Adjust sliders to explore. Close window to exit.")
plt.show()
