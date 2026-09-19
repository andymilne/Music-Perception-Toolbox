"""demo_maet_plots.py

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

Port of demo_maetPlots.m from the MATLAB Music Perception
Toolbox v3.

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

# Each entry is (r, is_rel, is_per, is_exch), and every configuration
# appears twice, unordered and then ordered. An ordered density counts
# each arrangement of a tuple separately, so it is unsymmetric in its
# arguments; the unordered one is its symmetrization. r = 1 has one
# slot, so ordering means nothing there and it appears once only.
#
# Only one to three dimensions are drawn, dim = r - is_rel: a
# four-dimensional density has no honest picture, and the grid of
# two-dimensional slices this demo used to draw for it showed three
# arbitrary cuts rather than the density.
configs = [
    (1, False, False, True),
    (1, False, True, True),
    (2, False, False, True),
    (2, False, False, False),
    (2, False, True, True),
    (2, False, True, False),
    (2, True, False, True),
    (2, True, False, False),
    (2, True, True, True),
    (2, True, True, False),
    (3, False, False, True),
    (3, False, False, False),
    (3, False, True, True),
    (3, False, True, False),
    (3, True, False, True),
    (3, True, False, False),
    (3, True, True, True),
    (3, True, True, False),
    (4, True, False, True),
    (4, True, False, False),
    (4, True, True, True),
    (4, True, True, False),
]

# How the three-dimensional plots are drawn, passed to the toolbox's
# own plotting function as its method:
#   'ellipsoids' - one ellipsoid per tuple centre, shaped by the
#                  kernel's covariance. No grid is evaluated, so it
#                  ignores step_3d and shows every centre whatever the
#                  sampling; it draws the kernels rather than the sum
#                  they make.
#   'points'     - one translucent mark per grid node above a
#                  threshold. Shows what lies between the peaks, and is
#                  the only mode the transform controls can drive.
# MATLAB's plotMaet3d offers a third, 'slices', a true volume
# rendering by texture-mapped surfaces; matplotlib has no counterpart.
plot_3d_mode = 'points'

# The step that matters is the step measured against sigma, not against
# the axis range: a blob is a few sigma across, so a grid coarser than
# sigma steps straight over it and the density appears to have peaks
# missing rather than blurred. At sigma = 10 a ten-cent step puts about
# one sample per sigma, which is the least that shows the shape.
step_1d = 1
step_2d = 5
step_3d = 10

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
                            scatter_thresh_frac=0.05, is_exch=True,
                            plot_3d_mode='points'):
    """Build the figure for one config and return (fig, state)."""
    dens = mpt.build_maet(p_arr, w, sigma, r, is_rel, is_per, period,
                              is_exch, verbose=False)
    figs_state = {}

    if dim == 1:
        x = np.linspace(ax_min, ax_max, res)
        vals = mpt.eval_maet(dens, x, normalize, verbose=False)

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

        # An exchangeable density is symmetric in its arguments, so
        # half the grid can be evaluated and mirrored. An ordered one is
        # not -- being unsymmetric is the whole of what distinguishes it
        # -- so it is evaluated whole.
        if is_exch:
            upper_mask = np.triu(np.ones((res, res), dtype=bool))
            Xu = np.vstack([Ga[upper_mask], Gb[upper_mask]])
            vals_u = mpt.eval_maet(dens, Xu, normalize, verbose=False)
            V_raw = np.zeros((res, res))
            V_raw[upper_mask] = vals_u
            V_raw = V_raw + V_raw.T - np.diag(np.diag(V_raw))
            vals = V_raw
        else:
            X2 = np.vstack([Ga.ravel(), Gb.ravel()])
            vals = np.asarray(mpt.eval_maet(dens, X2, normalize,
                                            verbose=False)
                              ).reshape(res, res)

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
        # Drawn by the toolbox's own plotting function, so that the
        # demo and the toolbox agree on what a density looks like.
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        if plot_3d_mode == 'ellipsoids':
            mpt.plot_maet_3d(dens, ax=ax, colour_gamma=1.0)
        else:
            mpt.plot_maet_3d_points(
                dens, ax=ax, step=(ax_max - ax_min) / (res - 1),
                thresh_frac=scatter_thresh_frac, colour_gamma=1.0)
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
        ax.set_zlim(ax_min, ax_max)
        ax.set_xlabel(f'{ax_label} 1')
        ax.set_ylabel(f'{ax_label} 2')
        ax.set_zlabel(f'{ax_label} 3')
        ax.set_title(f'{title_str} \u2014 {plot_3d_mode}')
        # The transform controls drive the colour and the opacity of a
        # mark, so they are not offered here: the ellipsoids carry their
        # value in geometry that would have to be rebuilt, and the
        # points are drawn by the toolbox rather than by this demo.
        figs_state[ci] = {'fig': fig, 'ax': ax}

    else:
        raise ValueError(
            f'Config {ci} has dim = {dim}. Only one to three '
            'dimensions are drawn: a four-dimensional density has no '
            'honest picture.')

    return figs_state


# ===================================================================
#  Main loop (executed when run as script)
# ===================================================================

if __name__ == '__main__':
    p_arr = np.array(p, dtype=float)

    all_states = {}

    for ci, (r, is_rel, is_per, is_exch) in enumerate(configs, start=1):
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
        else:
            step_size = step_3d

        res = max(2, int(round((ax_max - ax_min) / step_size)) + 1)

        mode_str = 'rel' if is_rel else 'abs'
        per_str = 'per' if is_per else 'non-per'
        ord_str = 'unordered' if is_exch else 'ordered'
        ax_label = 'Interval' if is_rel else 'Pitch'
        title_str = (f'Config {ci}: r={r}, {mode_str}, {per_str}, '
                     f'{ord_str}, dim={dim}, res={res}')

        print(f"Config {ci}: r = {r} ({mode_str}, {per_str}, {ord_str}, "
              f"dim = {dim}, res = {res}): ", end='', flush=True)

        figs_state = make_figure_for_config(
            p_arr, w, sigma, r, is_rel, is_per, period,
            normalize, ax_min, ax_max, dim, res, ci,
            ax_label, mode_str, per_str, title_str,
            mode_init=mode_init,
            gamma_init=gamma_init,
            eta_init=eta_init,
            scatter_thresh_frac=scatter_thresh_frac,
            is_exch=is_exch,
            plot_3d_mode=plot_3d_mode,
        )
        all_states.update(figs_state)
        print("done.")

    print("\nAll plots complete. Close windows to exit.")
    plt.show()
