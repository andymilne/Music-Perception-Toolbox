"""demo_exp_tensor_plots.py

Visualise expectation tensor densities in 1 to 4 dimensions for
user-specified combinations of r, is_rel, and is_per.

Each figure includes an interactive gamma (power compression) slider.
Surface plots (dim = 2) additionally include a colormap shift slider.

Port of demo_expTensorPlots.m from the MATLAB Music Perception Toolbox v2.

Requires: matplotlib (pip install matplotlib)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import mpt

# ===================================================================
#  User-editable parameters
# ===================================================================

# Pitch set and weights
p = [0, 200, 400, 500, 700, 900, 1100]
w = None  # uniform weights

# Gaussian smoothing width
sigma = 10

# Normalization: 'none', 'gaussian', or 'pdf'
normalize = 'none'

# Initial gamma (power compression): displayed = data^gamma
gamma_init = 1.0

# Period for periodic configurations
period = 1200

# Plot configurations: (r, is_rel, is_per)
configs = [
    (1, False, False),
    (1, False, True),
    (2, False, False),
    (2, False, True),
    (2, True, False),
    (2, True, True),
    (3, False, False),
    (3, False, True),
    (3, True, False),
    (3, True, True),
    (4, False, False),
    (4, False, True),
    (4, True, False),
    (4, True, True),
]

# Grid step sizes per effective dimensionality
step_1d = 1
step_2d = 5
step_3d = 20
step_4d = 50

# Axis range for non-periodic configurations
ax_min_nonper = 0
ax_max_nonper = 2400

# 3D scatter: minimum density threshold (fraction of max)
scatter_thresh_frac = 0.05


# ===================================================================
#  Helper functions
# ===================================================================


def apply_gamma(vals, gamma):
    """Power compression: normalise to [0,1], apply gamma, restore scale."""
    max_v = np.max(vals)
    if max_v > 0:
        return (vals / max_v) ** gamma * max_v
    return vals


# ===================================================================
#  Main loop
# ===================================================================

p_arr = np.asarray(p, dtype=np.float64)

for ci, (r, is_rel, is_per) in enumerate(configs):
    # Validation
    if r > len(p_arr):
        print(f"Config {ci}: r = {r} skipped (pitch set has only {len(p_arr)} elements).")
        continue
    if is_rel and r < 2:
        print(f"Config {ci}: r = {r} with is_rel = True skipped (requires r >= 2).")
        continue

    dim = r - int(is_rel)

    # Axis range
    if is_per:
        ax_min, ax_max = 0, period
    else:
        ax_min, ax_max = ax_min_nonper, ax_max_nonper

    # Grid resolution
    step_size = {1: step_1d, 2: step_2d, 3: step_3d}.get(dim, step_4d)
    res = max(2, round((ax_max - ax_min) / step_size) + 1)

    # Labels
    mode_str = 'relative' if is_rel else 'absolute'
    per_str = 'periodic' if is_per else 'non-periodic'
    ax_label = 'Interval' if is_rel else 'Pitch'
    title_str = f'r = {r}, {mode_str}, {per_str}, σ = {sigma}'

    print(f"Config {ci}: r = {r} ({mode_str}, {per_str}, dim = {dim}, res = {res}): ", end="")

    # Precompute density
    dens = mpt.build_exp_tens(p_arr, w, sigma, r, is_rel, is_per, period, verbose=False)
    print("evaluating... ", end="", flush=True)

    # -----------------------------------------------------------------
    #  dim = 1: line plot with gamma slider
    # -----------------------------------------------------------------
    if dim == 1:
        x = np.linspace(ax_min, ax_max, res)
        vals = mpt.eval_exp_tens(dens, x, normalize, verbose=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.subplots_adjust(bottom=0.18)

        line, = ax.plot(x, apply_gamma(vals, gamma_init), linewidth=1.5)
        ax.set_xlabel(f'{ax_label} 1')
        ax.set_ylabel('Density')
        ax.set_title(title_str)
        if is_per:
            ax.set_xlim(ax_min, ax_max)
        ax.grid(True, alpha=0.3)

        # Gamma slider
        ax_slider = fig.add_axes([0.15, 0.04, 0.55, 0.03])
        slider = Slider(ax_slider, 'Gamma', 0.01, 1.0, valinit=gamma_init)

        def update_1d(val, line=line, vals=vals, ax=ax):
            line.set_ydata(apply_gamma(vals, val))
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()

        slider.on_changed(update_1d)

    # -----------------------------------------------------------------
    #  dim = 2: surface / heatmap with gamma and colormap shift sliders
    # -----------------------------------------------------------------
    elif dim == 2:
        x = np.linspace(ax_min, ax_max, res)
        Ga, Gb = np.meshgrid(x, x)

        # Exploit symmetry: evaluate upper triangle, mirror
        upper_mask = np.triu(np.ones((res, res), dtype=bool))
        Xu = np.vstack([Ga[upper_mask], Gb[upper_mask]])
        vals_u = mpt.eval_exp_tens(dens, Xu, normalize, verbose=False)

        V_raw = np.zeros((res, res))
        V_raw[upper_mask] = vals_u
        V_raw = V_raw + V_raw.T - np.diag(np.diag(V_raw))
        vals = V_raw.ravel()

        fig, ax = plt.subplots(figsize=(9, 7))
        plt.subplots_adjust(bottom=0.22, right=0.85)

        V_display = apply_gamma(vals, gamma_init).reshape(res, res)
        im = ax.imshow(
            V_display, extent=[ax_min, ax_max, ax_min, ax_max],
            origin='lower', aspect='equal', cmap='viridis'
        )
        ax.set_xlabel(f'{ax_label} 1')
        ax.set_ylabel(f'{ax_label} 2')
        ax.set_title(title_str)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Gamma slider
        ax_gamma = fig.add_axes([0.15, 0.08, 0.55, 0.03])
        s_gamma = Slider(ax_gamma, 'Gamma', 0.01, 1.0, valinit=gamma_init)

        # Colormap shift slider
        ax_cmap = fig.add_axes([0.15, 0.03, 0.55, 0.03])
        s_cmap = Slider(ax_cmap, 'Cmap shift', 0.0, 0.95, valinit=0.0)

        def update_2d(val, im=im, vals=vals, s_gamma=s_gamma, s_cmap=s_cmap):
            V = apply_gamma(vals, s_gamma.val).reshape(res, res)
            im.set_data(V)
            v_min, v_max = np.min(V), np.max(V)
            if v_max > v_min:
                new_low = v_min + s_cmap.val * (v_max - v_min)
                im.set_clim(new_low, v_max)
            else:
                im.set_clim(v_min, v_max)
            fig.canvas.draw_idle()

        s_gamma.on_changed(update_2d)
        s_cmap.on_changed(update_2d)

    # -----------------------------------------------------------------
    #  dim = 3: scatter plot with gamma slider
    # -----------------------------------------------------------------
    elif dim == 3:
        x = np.linspace(ax_min, ax_max, res)
        Ga, Gb, Gc = np.meshgrid(x, x, x, indexing='ij')
        X = np.vstack([Ga.ravel(), Gb.ravel(), Gc.ravel()])

        vals = mpt.eval_exp_tens(dens, X, normalize, verbose=False)
        max_val = np.max(vals)

        # Threshold to reduce clutter
        thresh = scatter_thresh_frac * max_val
        mask = vals > thresh
        gx, gy, gz = Ga.ravel()[mask], Gb.ravel()[mask], Gc.ravel()[mask]
        v_mask = vals[mask]

        v_gamma = apply_gamma(v_mask, gamma_init)
        v_norm = v_gamma / np.max(v_gamma) if np.max(v_gamma) > 0 else v_gamma

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.15)

        sc = ax.scatter(gx, gy, gz, c=v_norm, s=10, cmap='viridis',
                        alpha=0.5, edgecolors='none')
        ax.set_xlabel(f'{ax_label} 1')
        ax.set_ylabel(f'{ax_label} 2')
        ax.set_zlabel(f'{ax_label} 3')
        ax.set_title(title_str)
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
        ax.set_zlim(ax_min, ax_max)
        fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.1)

        # Gamma slider
        ax_slider = fig.add_axes([0.15, 0.04, 0.55, 0.03])
        slider = Slider(ax_slider, 'Gamma', 0.01, 1.0, valinit=gamma_init)

        def update_3d(val, sc=sc, v_mask=v_mask):
            vg = apply_gamma(v_mask, val)
            vn = vg / np.max(vg) if np.max(vg) > 0 else vg
            sc.set_array(vn)
            fig.canvas.draw_idle()

        slider.on_changed(update_3d)

    # -----------------------------------------------------------------
    #  dim >= 4: grid of 2D slices
    # -----------------------------------------------------------------
    else:
        x = np.linspace(ax_min, ax_max, res)
        Ga, Gb = np.meshgrid(x, x)

        # Choose fixed values for extra dimensions
        if is_rel:
            all_intervals = np.sort(np.unique(np.diff(np.sort(p_arr))))
            fixed_vals = all_intervals[:min(3, len(all_intervals))]
        else:
            fixed_vals = p_arr[:min(3, len(p_arr))]

        if len(fixed_vals) < 3:
            fixed_vals = np.linspace(ax_min, ax_max, 3)

        n_extra = dim - 2

        # Build all combinations of fixed values
        from itertools import product as iterproduct
        fixed_combos = list(iterproduct(fixed_vals, repeat=n_extra))
        n_slices = len(fixed_combos)

        n_cols = int(np.ceil(np.sqrt(n_slices)))
        n_rows = int(np.ceil(n_slices / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        fig.suptitle(f'{title_str} — 2D slices', fontweight='bold')
        plt.subplots_adjust(bottom=0.12, hspace=0.35, wspace=0.3)

        if n_slices == 1:
            axes = np.array([axes])
        axes = np.atleast_1d(axes).ravel()

        images = []
        slice_vals_list = []

        for si, fixed in enumerate(fixed_combos):
            Xq = np.vstack([Ga.ravel(), Gb.ravel()])
            for d in range(n_extra):
                Xq = np.vstack([Xq, np.full(Ga.size, fixed[d])])

            sv = mpt.eval_exp_tens(dens, Xq, normalize, verbose=False)
            slice_vals_list.append(sv)

            V = apply_gamma(sv, gamma_init).reshape(res, res)
            im = axes[si].imshow(
                V, extent=[ax_min, ax_max, ax_min, ax_max],
                origin='lower', aspect='equal', cmap='viridis'
            )
            images.append(im)

            fix_str = ', '.join(
                f'{ax_label.lower()} {d + 3}={fixed[d]:.1f}'
                for d in range(n_extra)
            )
            axes[si].set_title(fix_str, fontsize=8)
            axes[si].set_xlabel(f'{ax_label} 1')
            axes[si].set_ylabel(f'{ax_label} 2')

        # Hide unused subplots
        for si in range(n_slices, len(axes)):
            axes[si].set_visible(False)

        # Gamma slider
        ax_slider = fig.add_axes([0.15, 0.02, 0.55, 0.025])
        slider = Slider(ax_slider, 'Gamma', 0.01, 1.0, valinit=gamma_init)

        def update_slices(val, images=images, slice_vals_list=slice_vals_list):
            for im, sv in zip(images, slice_vals_list):
                V = apply_gamma(sv, val).reshape(res, res)
                im.set_data(V)
                im.set_clim(np.min(V), np.max(V))
            fig.canvas.draw_idle()

        slider.on_changed(update_slices)

    print("done.")

print("\nAll plots complete. Close windows to exit.")
plt.show()
