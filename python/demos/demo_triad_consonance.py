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

Interactive transform-mode selector (Off / Gamma / Saturation) is
provided alongside an adaptive slider whose meaning depends on the
chosen mode:
  - 'gamma' applies v -> v.^gamma in [0.01, 1] (linear scale)
  - 'sat'   applies v -> 1 - exp(-v / eta) with eta in [0.001, 5]
            (log10 scale; data normalised to [0, 1] then rescaled).
Both gamma and eta have per-mode memory. A separate cmap-shift
slider, also with per-mode memory, adjusts the colour scale.

Port of demo_triadConsonance.m from the MATLAB Music Perception
Toolbox v2.

Requires: matplotlib (pip install matplotlib)
"""

import math
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import mpt

# ===================================================================
#  User-editable parameters
# ===================================================================

plot_measures = [
    'tmpl_max',     # Template harmonicity: hMax (Milne 2013)
    'tmpl_ent',     # Template harmonicity: -hEntropy (Harrison 2020)
    'tensor',       # Tensor harmonicity (Smit et al. 2019)
    'spec_ent',     # -Spectral entropy (Milne et al. 2017)
    'rough',        # -Roughness (Sethares 1993)
]

step = 20
max_int = 1200

f0 = 261.63

sigma_tmpl = 10
sigma_tens = 10
sigma_ent = 10

spec_tmpl = ['harmonic', 24, 'powerlaw', 1]
spec_tens = ['harmonic', 24, 'powerlaw', 1]
spec_ent = ['harmonic', 24, 'powerlaw', 1]
spec_rough = ['harmonic', 24, 'powerlaw', 1]

dup_tens = 0

mode_init = 'off'   # 'off', 'gamma', or 'sat'
gamma_init = 1.0
eta_init = 5.0
# ===================================================================
#  Transform helper
# ===================================================================


def apply_transform(vals, mode, gamma, eta):
    """Dispatch on mode.

    'off'   identity; output range = input range.
    'gamma' power compression: data normalised by the empirical
            (min, max), then raised to gamma. Output is in [0, 1].
            Gamma is a display-cosmetic knob with no perceptual
            interpretation tied to absolute scale, so anchoring at
            the empirical min keeps the slider responsive across
            measures with very different ranges.
    'sat'   saturation: anchored at 0 (a meaningful baseline of "no
            density / no roughness / no entropy"). For non-negative
            data vn = vals / max. For non-positive data (e.g.,
            -roughness, -spec_entropy) vn = (vals - min) / (-min).
            Mixed-sign data falls back to min/max. The saturation
            curve (1 - exp(-vn/eta)) / (1 - exp(-1/eta)) is then
            applied. Output is in [0, 1].
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
#  Determine which measures are selected
# ===================================================================

do_tmpl_max = 'tmpl_max' in plot_measures
do_tmpl_ent = 'tmpl_ent' in plot_measures
do_tmpl = do_tmpl_max or do_tmpl_ent
do_tensor = 'tensor' in plot_measures
do_spec_ent = 'spec_ent' in plot_measures
do_rough = 'rough' in plot_measures

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

if do_tensor:
    dup = dup_tens if dup_tens > 0 else 3
    print(f"Tensor harmonicity template setup (r=3, dup={dup})...")
    tp, tw = mpt.add_spectra(np.zeros(dup), np.ones(dup), *spec_tens)
    nJ_template = math.factorial(3) * math.comb(len(tp), 3)
    print(f"  Template: {len(tp)} partials, {nJ_template} ordered triples.")

# ===================================================================
#  Compute features
# ===================================================================
# Exploit symmetry: features are invariant to swapping interval1 and
# interval2, so we build a linear list of unordered (int1, int2) pairs
# (one per upper-triangle entry, j >= i) and compute each feature once
# per unique pair, then mirror into the symmetric output matrix.
# Loop structure: each unique triad {0, ints[i], ints[j]} (j >= i) is
# computed once and mirrored into the symmetric (n_ints, n_ints) result
# grids. The upper-triangle pattern is recommended for *roughness*,
# which has no batched-input dispatch and no internal dedup — every
# iteration of its loop does the full computation from scratch, so
# halving the iteration count halves the actual work. For the three
# batched features (tensor harmonicity via eval_exp_tens, template
# harmonicity, and spectral entropy), the upper triangle is a
# code-organization choice only: passing the full (n_ints**2) grid
# would do the same amount of internal ET work, because the
# canonical-form dedup in the batched dispatch collapses permutation-
# equivalent inputs (i, j) and (j, i) onto a single cached density.
# Keeping the upper-triangle pattern across all four features makes
# the unique-triad structure explicit in the demo code.

n_upper = n_ints * (n_ints + 1) // 2

# Build the linear list of (i, j) pairs with j >= i.
i_lin    = np.zeros(n_upper, dtype=int)
j_lin    = np.zeros(n_upper, dtype=int)
int1_lin = np.zeros(n_upper)
int2_lin = np.zeros(n_upper)
k = 0
for i in range(n_ints):
    for j in range(i, n_ints):
        i_lin[k]    = i
        j_lin[k]    = j
        int1_lin[k] = ints[i]
        int2_lin[k] = ints[j]
        k += 1

print(f"Computing features for {n_upper} unique triads "
      f"(step = {step} cents)...")
t0_total = time.time()

# --- Tensor harmonicity ---
# One eval_exp_tens call: the harmonic-template arrays (tp, tw) are
# queried at all upper-triangle interval pairs in a single
# (2, n_upper) query matrix. eval_exp_tens builds the template tensor
# internally and prints its own time estimate via estimate_comp_time
# when called with verbose=True.
if do_tensor:
    int_mat = np.vstack([int1_lin, int2_lin])    # (2, n_upper)
    t0 = time.time()
    tens_lin = mpt.eval_exp_tens(
        tp, tw, sigma_tens, 3, True, False, 1200, int_mat, verbose=True,
    )
    print(f"  Tensor harmonicity:   {time.time() - t0:.2f} s actual "
          f"({n_upper} triads, batched)")
    tens_harm[j_lin, i_lin] = tens_lin
    tens_harm[i_lin, j_lin] = tens_lin

# --- Template harmonicity ---
# One template_harmonicity call: stack chords as rows of an
# (n_upper, 3) matrix; the function returns h_max and h_entropy as
# 1-D arrays of length n_upper (v2.1+). template_harmonicity prints
# its own time estimate via estimate_comp_time when called with
# verbose=True.
if do_tmpl:
    chord_mat = np.column_stack([
        np.zeros(n_upper), int1_lin, int2_lin
    ])

    t0 = time.time()
    h_max_lin, h_ent_lin = mpt.template_harmonicity(
        chord_mat, None, sigma_tmpl,
        spectrum=spec_tmpl, chord_spectrum=spec_tmpl,
        verbose=True,
    )
    print(f"  Template harmonicity: {time.time() - t0:.2f} s actual "
          f"({n_upper} triads, batched)")
    if do_tmpl_max:
        tmpl_harm_max[j_lin, i_lin] = h_max_lin
        tmpl_harm_max[i_lin, j_lin] = h_max_lin
    if do_tmpl_ent:
        tmpl_harm_ent[j_lin, i_lin] = h_ent_lin
        tmpl_harm_ent[i_lin, j_lin] = h_ent_lin

# --- Spectral entropy ---
# One spectral_entropy call on a stacked chord matrix (v2.1+).
#
# Method choice: we use the default method='shannon'. spectral_entropy
# also supports method='renyi2' (analytical Rényi-2 / collision entropy
# via the inner-product / Möbius form used by entropy_exp_tens; requires
# normalize=False). For consonance ordering both methods give the same
# monotonic ranking of chords, but Shannon is the established choice in
# the consonance literature (e.g. Smit et al. 2019, Milne et al. 2017)
# and is used for this triad comparison.
if do_spec_ent:
    chord_mat_se = np.column_stack([
        np.zeros(n_upper), int1_lin, int2_lin
    ])
    t0 = time.time()
    spec_ent_lin = mpt.spectral_entropy(
        chord_mat_se, None, sigma_ent,
        spectrum=spec_ent, verbose=True,
    )
    print(f"  Spectral entropy:     {time.time() - t0:.2f} s actual "
          f"({n_upper} triads, batched)")
    spec_ent_grid[j_lin, i_lin] = spec_ent_lin
    spec_ent_grid[i_lin, j_lin] = spec_ent_lin

# --- Roughness (no batched mode; explicit loop) ---
if do_rough:
    rough_lin = np.full(n_upper, np.nan)

    print(f"  Roughness: looping over {n_upper} triads...")
    t0 = time.time()
    n_done = 0
    for k in range(n_upper):
        int1k = int1_lin[k]
        int2k = int2_lin[k]

        chord_cents = np.array(
            [ref_cents, ref_cents + int1k, ref_cents + int2k]
        )
        ep, ew = mpt.add_spectra(chord_cents, None, *spec_rough)
        f_hz = mpt.convert_pitch(ep, 'cents', 'hz')
        rough_lin[k] = mpt.roughness(f_hz, ew)

        n_done += 1
        if n_done % 500 == 0 or n_done == n_upper:
            elapsed = time.time() - t0
            rate    = n_done / elapsed if elapsed > 0 else 0
            remain  = (n_upper - n_done) / rate if rate > 0 else 0
            print(f"    {n_done} / {n_upper} triads "
                  f"({elapsed:.1f} s elapsed, ~{remain:.0f} s remaining)")
    print(f"  Roughness:            {time.time() - t0:.2f} s")

    rough_grid[j_lin, i_lin] = rough_lin
    rough_grid[i_lin, j_lin] = rough_lin

print(f"All features computed in {time.time() - t0_total:.1f} s.")

# ===================================================================
#  Assemble measures for plotting
# ===================================================================

spec_str = ', '.join(str(x) for x in spec_tmpl)

all_data = []
all_titles = []

if do_tmpl_max:
    all_data.append(tmpl_harm_max)
    all_titles.append(f'Template harmonicity: hMax\n{spec_str}, σ={sigma_tmpl}')
if do_tmpl_ent:
    all_data.append(-tmpl_harm_ent)
    all_titles.append(f'Template −hEntropy\n{spec_str}, σ={sigma_tmpl}')
if do_tensor:
    all_data.append(tens_harm)
    all_titles.append(f'Tensor harmonicity\n{spec_str}, σ={sigma_tens}, dup={dup}')
if do_spec_ent:
    all_data.append(-spec_ent_grid)
    all_titles.append(f'−Spectral entropy\n{spec_str}, σ={sigma_ent}')
if do_rough:
    all_data.append(-rough_grid)
    all_titles.append(f'−Roughness\n{spec_str}, f₀={f0:.1f} Hz')

n_plots = len(all_data)

if n_plots == 0:
    print("No measures selected — nothing to plot.")
    sys.exit(0)


# ===================================================================
#  Plot with adaptive transform UI
# ===================================================================


def make_triad_figure(all_data, all_titles, max_int, step,
                      mode_init='off', gamma_init=1.0, eta_init=5.0):
    """Build the figure with imshow plots and the transform-mode UI.

    Returns (fig, state). `state` is a dict holding the widgets and
    per-mode memory; useful for non-interactive screenshot generation
    that drives the radios programmatically.
    """
    n_plots = len(all_data)
    n_cols = min(n_plots, 3)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5.5 * n_cols, 4.5 * n_rows + 1.4))
    plt.subplots_adjust(left=0.07, right=0.97,
                        bottom=0.20, top=0.86,
                        hspace=0.45, wspace=0.30)

    if n_plots == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).ravel()

    state = {
        'mode': mode_init,
        'gamma': gamma_init,
        'eta': eta_init,
        'cmap_shift_off': 0.0,
        'cmap_shift_gamma': 0.0,
        'cmap_shift_sat': 0.0,
        'images': [],
        'raw_data': all_data,
    }

    for mi in range(n_plots):
        ax = axes[mi]
        data = all_data[mi]
        V = apply_transform(data, mode_init, gamma_init, eta_init)
        im = ax.imshow(
            V, extent=[0, max_int, 0, max_int],
            origin='lower', aspect='equal', cmap='viridis'
        )
        state['images'].append(im)
        ax.set_xlabel('Interval 1 (cents)')
        ax.set_ylabel('Interval 2 (cents)')
        ax.set_title(all_titles[mi], fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for mi in range(n_plots, len(axes)):
        axes[mi].set_visible(False)

    fig.suptitle(f'Triad consonance (step = {step} cents)',
                  fontweight='bold')

    # ---- Bottom UI: radios on the left, two sliders to the right ----
    ax_radio = fig.add_axes([0.04, 0.025, 0.10, 0.11], frameon=False)
    radios = RadioButtons(ax_radio, ('Off', 'Gamma', 'Sat'),
                           active={'off': 0, 'gamma': 1, 'sat': 2}[mode_init])
    state['radios'] = radios

    ax_xform = fig.add_axes([0.20, 0.085, 0.50, 0.025])
    # Slider is internally 0..1; the meaning depends on mode:
    #   gamma mode: 0 -> gamma=0.01, 1 -> gamma=1
    #   sat mode:   0 -> log10(eta)=-3 (eta=0.001),
    #               1 -> log10(eta)=log10(5) (eta=5)
    s_xform = Slider(ax_xform, '', 0.0, 1.0, valinit=1.0)
    s_xform.valtext.set_text('')
    state['s_xform'] = s_xform
    state['ax_xform'] = ax_xform

    ax_cmap = fig.add_axes([0.20, 0.040, 0.50, 0.025])
    s_cmap = Slider(ax_cmap, 'Cmap', 0.0, 0.95, valinit=0.0)
    state['s_cmap'] = s_cmap

    if mode_init == 'off':
        ax_xform.set_visible(False)

    # Mapping helpers: convert between slider position [0,1] and
    # gamma/eta values for the active mode.
    GAMMA_LO, GAMMA_HI = 0.01, 1.0
    SAT_LOG_LO, SAT_LOG_HI = float(np.log10(0.002)), float(np.log10(5))

    def gamma_to_pos(g):
        return (g - GAMMA_LO) / (GAMMA_HI - GAMMA_LO)

    def pos_to_gamma(p):
        return GAMMA_LO + p * (GAMMA_HI - GAMMA_LO)

    def eta_to_pos(e):
        log_e = np.log10(e)
        return (log_e - SAT_LOG_LO) / (SAT_LOG_HI - SAT_LOG_LO)

    def pos_to_eta(p):
        log_e = SAT_LOG_LO + p * (SAT_LOG_HI - SAT_LOG_LO)
        return 10 ** log_e

    def apply_to_all():
        m = state['mode']
        g = state['gamma']
        e = state['eta']
        cmap_shift = state.get(f'cmap_shift_{m}', 0.0)
        for im, data in zip(state['images'], state['raw_data']):
            V = apply_transform(data, m, g, e)
            im.set_data(V)
            v_min = np.nanmin(V)
            v_max = np.nanmax(V)
            if v_max > v_min:
                new_low = v_min + cmap_shift * (v_max - v_min)
                im.set_clim(new_low, v_max)
            else:
                im.set_clim(v_min, v_max)
        fig.canvas.draw_idle()

    def on_mode(label):
        new_mode = {'Off': 'off', 'Gamma': 'gamma', 'Sat': 'sat'}[label]
        old_mode = state['mode']
        if old_mode == 'gamma':
            state['gamma'] = pos_to_gamma(s_xform.val)
        elif old_mode == 'sat':
            state['eta'] = pos_to_eta(s_xform.val)
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

        s_cmap.set_val(state[f'cmap_shift_{new_mode}'])
        apply_to_all()

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
        apply_to_all()

    def on_cmap(val):
        m = state['mode']
        state[f'cmap_shift_{m}'] = val
        apply_to_all()

    radios.on_clicked(on_mode)
    s_xform.on_changed(on_xform)
    s_cmap.on_changed(on_cmap)

    state['on_mode'] = on_mode
    state['on_xform'] = on_xform
    state['on_cmap'] = on_cmap
    state['apply_to_all'] = apply_to_all

    return fig, state


fig, state = make_triad_figure(all_data, all_titles, max_int, step,
                                mode_init=mode_init,
                                gamma_init=gamma_init,
                                eta_init=eta_init)

if __name__ == '__main__':
    print("\nDone. Pick a transform (Off / Gamma / Sat) and adjust "
          "sliders to explore. Close window to exit.")
    plt.show()
