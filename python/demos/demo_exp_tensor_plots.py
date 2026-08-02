"""demo_exp_tensor_plots.py

Visualise expectation tensor densities in 1 to 4 dimensions for
user-specified combinations of r, is_rel, and is_per.

Each figure includes an interactive transform-mode selector
(Off / Gamma / Saturation) and an adaptive slider:

  - 'gamma' applies v -> v.^gamma in [0.01, 1] (linear scale)
  - 'sat'   applies v -> 1 - exp(-v / eta) with eta in [0.001, 5]
            (log10 scale; data normalised to [0, 1] then rescaled)

Both gamma and eta have per-mode memory. Surface plots (dim = 2)
additionally include a colormap shift slider, also with per-mode
memory.

Port of demo_expTensorPlots.m from the MATLAB Music Perception
Toolbox v2.

Requires: matplotlib (pip install matplotlib)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import mpt

# ===================================================================
#  User-editable parameters
# ===================================================================

p = [0, 200, 400, 500, 700, 900, 1100]
w = None

sigma = 10
normalize = 'none'

mode_init = 'off'
gamma_init = 1.0
eta_init = 5.0
period = 1200

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

step_1d = 1
step_2d = 5
step_3d = 20
step_4d = 50

ax_min_nonper = 0
ax_max_nonper = 2400

scatter_thresh_frac = 0.05


# ===================================================================
#  Transform helper
# ===================================================================


def apply_transform(vals, mode, gamma, eta):
    """Dispatch on mode.

    'off'   identity; output range = input range.
    'gamma' power compression: data normalised by the empirical
            (min, max), then raised to gamma. Output is in [0, 1].
            Gamma is a display-cosmetic knob, so anchoring at the
            empirical min keeps the slider responsive.
    'sat'   saturation: anchored at 0 (a meaningful baseline of "no
            density"). For tensor density, which is always non-
            negative, vn = vals / max. The saturation curve
            (1 - exp(-vn/eta)) / (1 - exp(-1/eta)) is then applied.
            Output is in [0, 1].
    """
    if mode == 'off':
        return vals

    mn = np.nanmin(vals)
    mx = np.nanmax(vals)

    if mode == 'gamma':
        if mx > mn:
            vn = (vals - mn) / (mx - mn)
            return vn ** gamma
        return vals.copy() if hasattr(vals, 'copy') else vals

    if mode == 'sat':
        if mn >= 0 and mx > 0:
            vn = vals / mx
        elif mx <= 0 and mn < 0:
            vn = (vals - mn) / (-mn)
        elif mx > mn:
            vn = (vals - mn) / (mx - mn)
        else:
            return vals.copy() if hasattr(vals, 'copy') else vals
        num = 1.0 - np.exp(-vn / eta)
        den = 1.0 - np.exp(-1.0 / eta)
        if den > 0:
            return num / den
        return vn

    return vals


# ===================================================================
#  Generic UI helper: add Off/Gamma/Sat toggle + slider to a figure
# ===================================================================


def add_transform_controls(fig, redraw_callback,
                           include_cmap=False,
                           cmap_callback=None,
                           mode_init='off',
                           gamma_init=1.0,
                           eta_init=5.0):
    """Add the radio + adaptive slider UI to a figure.

    `redraw_callback(mode, gamma, eta)` is called whenever mode or
    transform-slider value changes, with the resolved (mode, gamma,
    eta) values.

    If `include_cmap` is True, also adds a cmap-shift slider whose
    changes are reported via `cmap_callback(shift_frac)`. The cmap
    shift value is itself remembered per mode.

    Returns a `state` dict containing widgets and per-mode memory.
    """
    state = {
        'mode': mode_init,
        'gamma': gamma_init,
        'eta': eta_init,
        'cmap_shift_off': 0.0,
        'cmap_shift_gamma': 0.0,
        'cmap_shift_sat': 0.0,
    }

    # Make room at the bottom for the controls
    if include_cmap:
        plt.subplots_adjust(bottom=0.22)
        radio_y = 0.04
        radio_h = 0.13
        xform_y = 0.13
        cmap_y = 0.08
    else:
        plt.subplots_adjust(bottom=0.16)
        radio_y = 0.02
        radio_h = 0.11
        xform_y = 0.075
        cmap_y = None

    ax_radio = fig.add_axes([0.04, radio_y, 0.10, radio_h], frameon=False)
    radios = RadioButtons(ax_radio, ('Off', 'Gamma', 'Sat'),
                           active={'off': 0, 'gamma': 1, 'sat': 2}[mode_init])
    state['radios'] = radios

    ax_xform = fig.add_axes([0.22, xform_y, 0.55, 0.025])
    # Slider is internally 0..1; the meaning depends on mode:
    #   gamma: 0 -> gamma=0.01, 1 -> gamma=1
    #   sat:   0 -> eta=0.001, 1 -> eta=5  (log10 mapped)
    s_xform = Slider(ax_xform, '', 0.0, 1.0, valinit=1.0)
    s_xform.valtext.set_text('')
    state['s_xform'] = s_xform
    state['ax_xform'] = ax_xform

    GAMMA_LO, GAMMA_HI = 0.01, 1.0
    SAT_LOG_LO, SAT_LOG_HI = float(np.log10(0.002)), float(np.log10(5))

    def gamma_to_pos(g):
        return (g - GAMMA_LO) / (GAMMA_HI - GAMMA_LO)

    def pos_to_gamma(p_):
        return GAMMA_LO + p_ * (GAMMA_HI - GAMMA_LO)

    def eta_to_pos(e):
        return (np.log10(e) - SAT_LOG_LO) / (SAT_LOG_HI - SAT_LOG_LO)

    def pos_to_eta(p_):
        return 10 ** (SAT_LOG_LO + p_ * (SAT_LOG_HI - SAT_LOG_LO))

    if include_cmap:
        ax_cmap = fig.add_axes([0.22, cmap_y, 0.55, 0.025])
        s_cmap = Slider(ax_cmap, 'Cmap', 0.0, 0.95, valinit=0.0)
        state['s_cmap'] = s_cmap
    else:
        s_cmap = None

    if mode_init == 'off':
        ax_xform.set_visible(False)

    def do_redraw():
        m = state['mode']
        redraw_callback(m, state['gamma'], state['eta'])
        if include_cmap and cmap_callback is not None:
            cmap_callback(state[f'cmap_shift_{m}'])
        fig.canvas.draw_idle()

    def on_mode(label):
        new_mode = {'Off': 'off', 'Gamma': 'gamma', 'Sat': 'sat'}[label]
        old_mode = state['mode']
        if old_mode == 'gamma':
            state['gamma'] = pos_to_gamma(s_xform.val)
        elif old_mode == 'sat':
            state['eta'] = pos_to_eta(s_xform.val)
        if include_cmap:
            state[f'cmap_shift_{old_mode}'] = s_cmap.val

        state['mode'] = new_mode
        if new_mode == 'off':
            ax_xform.set_visible(False)
            s_xform.valtext.set_text('')
        else:
            ax_xform.set_visible(True)
            if new_mode == 'gamma':
                ax_xform.set_xlabel('Gamma')
                s_xform.set_val(gamma_to_pos(state['gamma']))
                s_xform.valtext.set_text(f'{state["gamma"]:.2f}')
            elif new_mode == 'sat':
                ax_xform.set_xlabel('Saturation (η)')
                s_xform.set_val(eta_to_pos(state['eta']))
                s_xform.valtext.set_text(f'{state["eta"]:.3f}')

        if include_cmap:
            s_cmap.set_val(state[f'cmap_shift_{new_mode}'])
        do_redraw()

    def on_xform(val):
        m = state['mode']
        if m == 'gamma':
            state['gamma'] = pos_to_gamma(val)
            s_xform.valtext.set_text(f'{state["gamma"]:.2f}')
        elif m == 'sat':
            state['eta'] = pos_to_eta(val)
            s_xform.valtext.set_text(f'{state["eta"]:.3f}')
        else:
            return
        do_redraw()

    def on_cmap(val):
        m = state['mode']
        state[f'cmap_shift_{m}'] = val
        do_redraw()

    radios.on_clicked(on_mode)
    s_xform.on_changed(on_xform)
    if include_cmap:
        s_cmap.on_changed(on_cmap)

    state['on_mode'] = on_mode
    state['on_xform'] = on_xform
    if include_cmap:
        state['on_cmap'] = on_cmap

    return state


# ===================================================================
#  Per-config plotting
# ===================================================================


def make_figure_for_config(p_arr, w, sigma, r, is_rel, is_per, period,
                            normalize, ax_min, ax_max, dim, res, ci,
                            ax_label, mode_str, per_str, title_str,
                            mode_init='off', gamma_init=1.0, eta_init=5.0,
                            scatter_thresh_frac=0.05):
    """Build the figure for one config and return (fig, state)."""
    dens = mpt.build_exp_tens(p_arr, w, sigma, r, is_rel, is_per, period,
                              verbose=False)
    figs_state = {}

    if dim == 1:
        x = np.linspace(ax_min, ax_max, res)
        vals = mpt.eval_exp_tens(dens, x, normalize, verbose=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        line, = ax.plot(x, apply_transform(vals, mode_init, gamma_init, eta_init),
                         linewidth=1.5)
        ax.set_xlabel(f'{ax_label} 1')
        ax.set_ylabel('Density')
        ax.set_title(title_str)
        if is_per:
            ax.set_xlim(ax_min, ax_max)
        ax.grid(True, alpha=0.3)

        def redraw(m, g, e, _line=line, _vals=vals, _ax=ax):
            _line.set_ydata(apply_transform(_vals, m, g, e))
            _ax.relim()
            _ax.autoscale_view()

        state = add_transform_controls(fig, redraw, include_cmap=False,
                                        mode_init=mode_init,
                                        gamma_init=gamma_init,
                                        eta_init=eta_init)
        figs_state[fig] = state

    elif dim == 2:
        x = np.linspace(ax_min, ax_max, res)
        Ga, Gb = np.meshgrid(x, x)

        upper_mask = np.triu(np.ones((res, res), dtype=bool))
        Xu = np.vstack([Ga[upper_mask], Gb[upper_mask]])
        vals_u = mpt.eval_exp_tens(dens, Xu, normalize, verbose=False)

        V_raw = np.zeros((res, res))
        V_raw[upper_mask] = vals_u
        V_raw = V_raw + V_raw.T - np.diag(np.diag(V_raw))
        vals = V_raw

        fig, ax = plt.subplots(figsize=(9, 7))
        plt.subplots_adjust(right=0.82)

        V0 = apply_transform(vals, mode_init, gamma_init, eta_init)
        im = ax.imshow(
            V0, extent=[ax_min, ax_max, ax_min, ax_max],
            origin='lower', aspect='equal', cmap='viridis'
        )
        ax.set_xlabel(f'{ax_label} 1')
        ax.set_ylabel(f'{ax_label} 2')
        ax.set_title(title_str)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        def redraw(m, g, e, _im=im, _vals=vals):
            V = apply_transform(_vals, m, g, e)
            _im.set_data(V)

        def cmap_redraw(shift, _im=im):
            cdata = _im.get_array()
            v_min = float(np.nanmin(cdata))
            v_max = float(np.nanmax(cdata))
            if v_max > v_min:
                _im.set_clim(v_min + shift * (v_max - v_min), v_max)
            else:
                _im.set_clim(v_min, v_max)

        state = add_transform_controls(fig, redraw, include_cmap=True,
                                        cmap_callback=cmap_redraw,
                                        mode_init=mode_init,
                                        gamma_init=gamma_init,
                                        eta_init=eta_init)
        figs_state[fig] = state

    elif dim == 3:
        x = np.linspace(ax_min, ax_max, res)
        Ga, Gb, Gc = np.meshgrid(x, x, x, indexing='ij')
        X = np.vstack([Ga.ravel(), Gb.ravel(), Gc.ravel()])
        vals = mpt.eval_exp_tens(dens, X, normalize, verbose=False)
        max_val = np.max(vals)

        thresh = scatter_thresh_frac * max_val
        mask = vals > thresh
        gx, gy, gz = Ga.ravel()[mask], Gb.ravel()[mask], Gc.ravel()[mask]
        v_mask = vals[mask]

        # Initial render uses max-only normalisation (matching the
        # redraw path in add_transform_controls).
        M_init = float(np.max(v_mask))
        if M_init > 0:
            v_norm = v_mask / M_init
        else:
            v_norm = v_mask

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

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

        def redraw(m, g, e, _sc=sc, _v_mask=v_mask):
            v_t = apply_transform(_v_mask, m, g, e)
            # In 'off' mode, v_t is in input range; normalise to [0, 1]
            # for color/alpha (max-only, since data is non-negative).
            # In 'gamma'/'sat' modes v_t is already in [0, 1].
            if m == 'off':
                M = float(np.max(_v_mask))
                if M > 0:
                    v_t = _v_mask / M
            _sc.set_array(v_t)

        state = add_transform_controls(fig, redraw, include_cmap=False,
                                        mode_init=mode_init,
                                        gamma_init=gamma_init,
                                        eta_init=eta_init)
        figs_state[fig] = state

    else:
        x = np.linspace(ax_min, ax_max, res)
        Ga, Gb = np.meshgrid(x, x)

        if is_rel:
            all_intervals = np.sort(np.unique(np.diff(np.sort(p_arr))))
            fixed_vals = all_intervals[:min(3, len(all_intervals))]
        else:
            fixed_vals = p_arr[:min(3, len(p_arr))]
        if len(fixed_vals) < 3:
            fixed_vals = np.linspace(ax_min, ax_max, 3)

        n_extra = dim - 2
        from itertools import product as iterproduct
        fixed_combos = list(iterproduct(fixed_vals, repeat=n_extra))
        n_slices = len(fixed_combos)

        n_cols = int(np.ceil(np.sqrt(n_slices)))
        n_rows = int(np.ceil(n_slices / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(5 * n_cols, 4 * n_rows))
        fig.suptitle(f'{title_str} — 2D slices', fontweight='bold')
        # Tight, explicit layout: small margins, modest gaps, top
        # reserved for the suptitle and bottom for the controls (the
        # latter is also reserved by add_transform_controls).
        plt.subplots_adjust(left=0.05, right=0.97,
                            bottom=0.20, top=0.93,
                            hspace=0.25, wspace=0.20)

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

            V = apply_transform(sv, mode_init, gamma_init, eta_init).reshape(res, res)
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

        for si in range(n_slices, len(axes)):
            axes[si].set_visible(False)

        def redraw(m, g, e, _images=images, _slice_vals_list=slice_vals_list,
                   _res=res):
            for im, sv in zip(_images, _slice_vals_list):
                V = apply_transform(sv, m, g, e).reshape(_res, _res)
                im.set_data(V)
                # Don't clobber clim here; cmap_redraw handles it

        def cmap_redraw(shift, _images=images):
            # Apply the same shift fraction to every panel's own range.
            for im in _images:
                cdata = im.get_array()
                v_min = float(np.nanmin(cdata))
                v_max = float(np.nanmax(cdata))
                if v_max > v_min:
                    im.set_clim(v_min + shift * (v_max - v_min), v_max)
                else:
                    im.set_clim(v_min, v_max)

        state = add_transform_controls(fig, redraw, include_cmap=True,
                                        cmap_callback=cmap_redraw,
                                        mode_init=mode_init,
                                        gamma_init=gamma_init,
                                        eta_init=eta_init)
        figs_state[fig] = state

    return figs_state


# ===================================================================
#  Main loop (executed when run as script)
# ===================================================================

if __name__ == '__main__':
    p_arr = np.array(p, dtype=float)

    all_states = {}

    for ci, (r, is_rel, is_per) in enumerate(configs, start=1):
        if r > len(p_arr):
            continue
        if is_rel and r < 2:
            continue

        dim = r - int(is_rel)

        if is_per:
            ax_min, ax_max = 0, period
        else:
            ax_min, ax_max = ax_min_nonper, ax_max_nonper

        if dim == 1:
            step_size = step_1d
        elif dim == 2:
            step_size = step_2d
        elif dim == 3:
            step_size = step_3d
        else:
            step_size = step_4d

        res = max(2, int(round((ax_max - ax_min) / step_size)) + 1)

        mode_str = 'rel' if is_rel else 'abs'
        per_str = 'per' if is_per else 'non-per'
        ax_label = 'Interval' if is_rel else 'Pitch'
        title_str = (f'Config {ci}: r={r}, {mode_str}, {per_str}, '
                     f'dim={dim}, res={res}')

        print(f"Config {ci}: r = {r} ({mode_str}, {per_str}, "
              f"dim = {dim}, res = {res}): ", end='', flush=True)

        figs_state = make_figure_for_config(
            p_arr, w, sigma, r, is_rel, is_per, period,
            normalize, ax_min, ax_max, dim, res, ci,
            ax_label, mode_str, per_str, title_str,
            mode_init=mode_init,
            gamma_init=gamma_init,
            eta_init=eta_init,
            scatter_thresh_frac=scatter_thresh_frac,
        )
        all_states.update(figs_state)
        print("done.")

    print("\nAll plots complete. Close windows to exit.")
    plt.show()
