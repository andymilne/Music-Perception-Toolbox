"""generate_article_figures.py

Generates all figures for the TISMIR article:
  "The Music Perception Toolbox: Analytical Methods for
   Pitch and Rhythm Similarity, Consonance, and Structure"

Requirements:
  - music-perception-toolbox (pip install -e .)
  - matplotlib, scipy, numpy

Output: PDF figures in ./figures/

Andrew J. Milne, 2026.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
# The toolbox is imported as `mpt`. If it is not installed, point MPT_PYTHON
# at the `python` directory of a version 3.0 checkout.
_MPT = os.environ.get("MPT_PYTHON")
if _MPT:
    sys.path.insert(0, _MPT)
import mpt

# The published values were computed with an untruncated kernel;
# version 3 truncates at six sigma by default, so the untruncated
# setting is restored here.
mpt.set_default(truncation_sigmas=float("inf"), show_hints=False)

# ===================================================================
#  Global settings
# ===================================================================

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAVEKW = dict(dpi=300, bbox_inches='tight', pad_inches=0.02)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})


# ===================================================================
#  SPCS probe-tone profiles
#  (a) C major with K&K data, (b) Porcupine[7] in 22-EDO
# ===================================================================

def spcs_probe_profiles():
    print("=== SPCS probe tone profiles ===")

    # Krumhansl & Kessler (1982) C major key profile
    kk_major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                         2.52, 5.19, 2.39, 3.66, 2.29, 2.88])

    def compute_profile(scale_cents, period, sigma=10.0, n_harm=16, rho=1.0):
        n = int(period)
        # pitches_a: scale repeated n times (n x len(scale))
        # pitches_b: one probe tone per row (n x 1)
        pitches_a = np.tile(scale_cents, (n, 1))
        pitches_b = np.arange(n, dtype=float).reshape(n, 1)
        return mpt.sim_maet(
            pitches_a, None, pitches_b, None, sigma, 1, False, True, period,
            spectrum=['harmonic', n_harm, 'powerlaw', rho],
            verbose=False)

    # (a) C major
    c_major = [0, 200, 400, 500, 700, 900, 1100]
    print("  Computing C major profile...")
    spcs = compute_profile(c_major, 1200)

    # Optimal affine rescaling of K&K to SPCS at semitone positions
    spcs_at = spcs[np.arange(0, 1200, 100)]
    A = np.column_stack([kk_major, np.ones(12)])
    ab, *_ = np.linalg.lstsq(A, spcs_at, rcond=None)
    kk_rescaled = kk_major * ab[0] + ab[1]
    r2 = 1 - np.sum((spcs_at - kk_rescaled)**2) / \
             np.sum((spcs_at - spcs_at.mean())**2)
    print(f"  K&K rescaling: a={ab[0]:.4f}, b={ab[1]:.4f}, R²={r2:.4f}")

    # (b) Porcupine[7] in 22-EDO, steps 4333333
    step22 = 1200 / 22
    porcupine = [i * step22 for i in [0, 4, 7, 10, 13, 16, 19]]
    print("  Computing porcupine profile...")
    spcs_porc = compute_profile(porcupine, 1200)

    # --- Plot (font set globally; full-width figure) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # Panel (a)
    ax1.plot(np.arange(1200), spcs, color='#2c3e50', linewidth=0.7,
             label='SPCS')
    kk_cents = np.arange(0, 1200, 100)
    ax1.scatter(kk_cents, kk_rescaled, color='#e74c3c', s=35, zorder=5,
                label='K\\&K (rescaled)', edgecolors='white', linewidths=0.5)
    names = ['C','C♯','D','E♭','E','F','F♯','G','A♭','A','B♭','B']
    for nm, c, v in zip(names, kk_cents, kk_rescaled):
        if nm in ['C','D','E','F','G','A','B']:
            ax1.annotate(nm, xy=(c, v), xytext=(0, 8),
                         textcoords='offset points', ha='center',
                         fontsize=8, fontweight='bold', color='#c0392b')
    ax1.set_xlabel('Probe tone (cents)')
    ax1.set_ylabel('SPCS / K\\&K (rescaled)')
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes,
             fontsize=11, fontweight='bold', va='top')
    ax1.set_xlim(0, 1199)
    ax1.legend(fontsize=8, loc='lower right')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel (b)
    ax2.plot(np.arange(1200), spcs_porc, color='#2c3e50', linewidth=0.7)
    for i, cent in enumerate([int(round(p)) for p in porcupine]):
        c = min(cent, 1199)
        ax2.annotate(f'{cent:.0f}¢', xy=(c, spcs_porc[c]),
                     xytext=(0, 8), textcoords='offset points',
                     ha='center', fontsize=7, color='#c0392b')
    ax2.set_xlabel('Probe tone (cents)')
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes,
             fontsize=11, fontweight='bold', va='top')
    ax2.set_xlim(0, 1199)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/spcs_probe.pdf', **SAVEKW)
    plt.close(fig)
    print("  Saved.")


# ===================================================================
#  FIGURE 2: 2D consonance grids (5 measures, 3×2 layout)
# ===================================================================


# ===================================================================
#  Position-level rhythmic features (son clave)
# ===================================================================

def rhythm_position_features():
    print("=== Circular rhythmic features ===")

    clave = np.array([0, 3, 6, 10, 12], dtype=float)
    period = 16

    # Compute features
    edges_signed = np.asarray(
        mpt.edges(clave, None, period, kappa=2.0)[1]).ravel()
    proj_cent = np.asarray(
        mpt.proj_centroid(clave, None, period)[0]).ravel()
    mo = np.asarray(mpt.mean_offset(clave, None, period)).ravel()
    markov = np.asarray(
        mpt.markov_s(np.array([0,3,6,10,12]), None, period, S=3)).ravel()
    apm_matrix = mpt.circ_apm(clave, None, period)[0]
    apm_by_pos = np.sum(apm_matrix, axis=0)

    is_event = np.zeros(period)
    for c in [0, 3, 6, 10, 12]:
        is_event[int(c)] = 1

    def pos_to_theta(i):
        return np.pi/2 - 2*np.pi*i/period

    def circ_plot(ax, values, label, color, is_discrete=False):
        theta = np.array([pos_to_theta(i) for i in range(period)])
        circle_t = np.linspace(0, 2*np.pi, 200)
        r_base = 1.0
        ax.plot(r_base*np.cos(circle_t), r_base*np.sin(circle_t),
                color='#d5d8dc', linewidth=0.8)

        v = np.array(values, dtype=float)
        vmax = max(abs(v.max()), abs(v.min()), 0.001)
        v_norm = v / vmax * 0.45

        if is_discrete:
            for i in range(period):
                r_end = r_base + v_norm[i]
                x0, y0 = r_base*np.cos(theta[i]), r_base*np.sin(theta[i])
                x1, y1 = r_end*np.cos(theta[i]), r_end*np.sin(theta[i])
                c = '#27ae60' if v_norm[i]>=0 else '#e74c3c' \
                    if color=='signed' else color
                ax.plot([x0,x1],[y0,y1], color=c, linewidth=3.5,
                        solid_capstyle='round')
        else:
            t_data = np.arange(period+1, dtype=float)
            v_wrap = np.concatenate([v_norm, [v_norm[0]]])
            f = interp1d(t_data, v_wrap, kind='cubic')
            t_fine = np.linspace(0, period, 400, endpoint=False)
            v_fine = f(t_fine)
            theta_fine = np.pi/2 - 2*np.pi*t_fine/period
            r_fine = r_base + v_fine
            xf = r_fine*np.cos(theta_fine)
            yf = r_fine*np.sin(theta_fine)
            xb = r_base*np.cos(theta_fine)
            yb = r_base*np.sin(theta_fine)
            ax.fill(np.concatenate([xf, xb[::-1]]),
                    np.concatenate([yf, yb[::-1]]),
                    alpha=0.15, color=color)
            ax.plot(xf, yf, color=color, linewidth=1.5)

        for i in range(period):
            lr = 0.78
            ax.text(lr*np.cos(theta[i]), lr*np.sin(theta[i]),
                    str(i), ha='center', va='center',
                    fontsize=7, color='#95a5a6')

        for i in range(period):
            if is_event[i] > 0:
                ax.plot(r_base*np.cos(theta[i]), r_base*np.sin(theta[i]),
                        'o', color='#2c3e50', markersize=7, zorder=5)
            else:
                ax.plot(r_base*np.cos(theta[i]), r_base*np.sin(theta[i]),
                        'o', color='#d5d8dc', markersize=4, zorder=4)

        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.7, 1.7)
        ax.set_aspect('equal')
        ax.text(0, -1.6, label, ha='center', va='top', fontsize=11,
                fontweight='bold')
        ax.axis('off')

    fig, axes = plt.subplots(3, 2, figsize=(6.0, 9.0))
    circ_plot(axes[0,0], edges_signed, '(a) Edges (signed)',
              '#2980b9', is_discrete=False)
    circ_plot(axes[0,1], proj_cent, '(b) Projected centroid',
              '#3498db', is_discrete=False)
    circ_plot(axes[1,0], mo, '(c) Mean offset',
              '#9b59b6', is_discrete=False)
    circ_plot(axes[1,1], apm_by_pos, '(d) Circ. APM (summed)',
              '#16a085', is_discrete=True)
    circ_plot(axes[2,0], markov, '(e) Markov ($S=3$)',
              '#e67e22', is_discrete=True)
    axes[2,1].set_visible(False)

    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/rhythm.pdf', **SAVEKW)
    plt.close(fig)
    print("  Saved.")


# ===================================================================
#  Balance on the chromatic circle
# ===================================================================

def balance_on_the_circle():
    """Balance for three pitch-class sets on the chromatic circle.

    Both the balance value and the centre of gravity come from the toolbox:
    the value from balance, and the centre of gravity from the zeroth
    coefficient returned by dftCircular, of which balance is one minus the
    modulus. The angular convention matches the rhythmic figure: pitch class
    0 at twelve o'clock, increasing clockwise.
    """
    print("=== Balance visualization ===")

    period = 12.0

    def to_xy(theta):
        """Angle measured anticlockwise from due east -> screen coordinates
        with zero at twelve o'clock and increasing clockwise."""
        return np.sin(theta), np.cos(theta)

    fig, axes = plt.subplots(3, 1, figsize=(2.191, 6.068))
    sets_info = [
        ("(a) Augmented triad {0, 4, 8}", [0, 4, 8]),
        ("(b) Major triad {0, 4, 7}",     [0, 4, 7]),
        ("(c) Cluster {0, 1, 2}",         [0, 1, 2]),
    ]
    for ax, (name, pitches) in zip(axes, sets_info):
        p = np.asarray(pitches, dtype=float)

        bal = float(mpt.balance(p, None, period))
        # dftCircular returns the coefficients with the zeroth already
        # divided by the summed weights, so its modulus is 1 - balance and
        # its argument is the angular position of the centre of gravity.
        coeffs, _ = mpt.dft_circular(p, None, period)
        cog = coeffs[0]

        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color='#bdc3c7', linewidth=1)
        for k in range(int(period)):
            x, y = to_xy(2*np.pi*k/period)
            ax.plot(x, y, 'o', color='#ecf0f1',
                    markersize=6, markeredgecolor='#bdc3c7')
        for pk in p:
            x, y = to_xy(2*np.pi*pk/period)
            ax.plot(x, y, 'o', color='#2c3e50', markersize=8, zorder=5)

        zx, zy = to_xy(np.angle(cog))
        zx, zy = np.abs(cog)*zx, np.abs(cog)*zy
        ax.plot(zx, zy, 'o', color='#e74c3c', markersize=5, zorder=6)
        ax.plot([0, zx], [0, zy], '-', color='#e74c3c',
                linewidth=1.2, alpha=0.7)
        ax.plot(0, 0, '+', color='#7f8c8d', markersize=6)

        ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.55, 1.2)
        ax.set_aspect('equal')
        ax.text(0, -1.22, f'{name}\nBalance = {bal:.3f}',
                ha='center', va='top', fontsize=7.5)
        ax.axis('off')

    fig.subplots_adjust(hspace=0.10, left=0.02, right=0.98,
                        top=0.99, bottom=0.01)
    fig.savefig(f'{OUTPUT_DIR}/balance.pdf', **SAVEKW)
    plt.close(fig)
    print("  Saved.")


# ===================================================================
#  Triad SPCS grids (major and minor references)
# ===================================================================

def triad_similarity_grids():
    print("=== Triad SPCS grids ===")

    note_names = ['C','C♯','D','E♭','E','F','F♯','G','A♭','A','B♭','B']

    def compute_grid(ref_pitches):
        pcs = np.arange(0, 1200, 100)
        n = len(pcs)
        root_g, third_g = np.meshgrid(pcs, pcs)
        pb = np.column_stack([root_g.ravel(), third_g.ravel(),
                              root_g.ravel()+700])
        pa = np.tile(ref_pitches, (n*n, 1))
        sv = mpt.sim_maet(
            pa, None, pb, None, 10, 1, False, True, 1200,
            spectrum=['harmonic', 16, 'powerlaw', 1], verbose=False)
        sim = sv.reshape(n, n)
        ref_root, ref_third = ref_pitches[0], ref_pitches[1]
        ro = (pcs - ref_root + 600) % 1200 - 600
        to = (pcs - ref_third + 600) % 1200 - 600
        ri = np.argsort(ro); ti = np.argsort(to)
        return {
            'sim': sim[np.ix_(ti, ri)],
            'ro': ro[ri], 'to': to[ti],
            'rl': [note_names[i] for i in ri],
            'tl': [note_names[i] for i in ti],
            'pcs': pcs, 'ref_root': ref_root, 'ref_third': ref_third,
        }

    print("  Computing major grid...")
    maj = compute_grid([0, 400, 700])
    print("  Computing minor grid...")
    mino = compute_grid([0, 300, 700])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.5))
    for ax, d, lab in [(ax1, maj, '(a)'), (ax2, mino, '(b)')]:
        ax.imshow(d['sim'], extent=[d['ro'][0]-50, d['ro'][-1]+50,
                  d['to'][0]-50, d['to'][-1]+50],
                  origin='lower', aspect='equal', cmap='gray_r',
                  interpolation='nearest')
        ax.set_xticks(d['ro']); ax.set_xticklabels(d['rl'], fontsize=8)
        ax.set_yticks(d['to']); ax.set_yticklabels(d['tl'], fontsize=8)
        ax.set_xlabel('Root of fifth', fontsize=10)
        ax.set_ylabel('Third', fontsize=10)
        ax.text(0.02, 0.98, lab, transform=ax.transAxes, fontsize=11,
                fontweight='bold', va='top', color='white')
        for ni in range(12):
            root = d['pcs'][ni]
            rx = (root - d['ref_root']+600)%1200-600
            maj_ty = ((root+400)%1200 - d['ref_third']+600)%1200-600
            ax.text(rx, maj_ty, note_names[ni], fontsize=7,
                    fontweight='bold', color='#c0392b',
                    ha='center', va='center')
            min_ty = ((root+300)%1200 - d['ref_third']+600)%1200-600
            mn = note_names[ni][0].lower() + note_names[ni][1:]
            ax.text(rx, min_ty, mn, fontsize=7, fontweight='bold',
                    color='#2980b9', ha='center', va='center')
        ax.plot(0, 0, 'ws', markersize=14, markerfacecolor='none',
                markeredgecolor='white', markeredgewidth=2)

    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/triad_grids.pdf', **SAVEKW)
    plt.close(fig)
    print("  Saved.")


# ===================================================================
#  Generator-chain SPCS (split: triad / tetrad)
# ===================================================================

def generator_chain_tunings(gen_step=0.1):
    print(f"=== Generator-chain tunings (step={gen_step}) ===")

    ref_triad  = np.array([0, 1200*np.log2(5/4), 1200*np.log2(6/4)])
    ref_tetrad = np.array([0, 1200*np.log2(5/4), 1200*np.log2(6/4),
                           1200*np.log2(7/4)])
    n_tones = 24; period = 1200; sigma = 10; r = 2
    chain_weights = (n_tones - np.arange(n_tones)) / n_tones

    half_range = np.arange(0, period/2 + gen_step/2, gen_step)
    n_half = len(half_range)

    # Build chain pitch matrix: each row is one generator's pitch set
    pitches_b = np.array([np.mod(np.arange(n_tones) * gen, period)
                          for gen in half_range])
    weights_b = np.tile(chain_weights, (n_half, 1))

    print("  Computing triad half...")
    pitches_a_tri = np.tile(ref_triad, (n_half, 1))
    s_triad = mpt.sim_maet(
        pitches_a_tri, None, pitches_b, weights_b,
        sigma, r, True, True, period, verbose=False)

    print("  Computing tetrad half...")
    pitches_a_tet = np.tile(ref_tetrad, (n_half, 1))
    s_tetrad = mpt.sim_maet(
        pitches_a_tet, None, pitches_b, weights_b,
        sigma, r, True, True, period, verbose=False)

    gen_range = np.arange(0, period, gen_step)
    s_full = np.concatenate([s_triad, s_tetrad[-2:0:-1]])
    if len(s_full) < len(gen_range):
        s_full = np.concatenate(
            [s_full, np.full(len(gen_range)-len(s_full), s_full[-1])])
    elif len(s_full) > len(gen_range):
        s_full = s_full[:len(gen_range)]

    smax = max(s_full)
    theta = 2*np.pi*gen_range/period
    theta_c = np.append(theta, theta[0])
    s_c = np.append(s_full, s_full[0])

    fig, ax = plt.subplots(figsize=(3.6, 3.6),
                           subplot_kw={'projection': 'polar'})
    ax.plot(theta_c, s_c, color='#2c3e50', linewidth=0.5)
    ax.fill(theta_c, s_c, alpha=0.08, color='#2c3e50')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlim(0, smax*1.25)
    tick_angles = np.linspace(0, 2*np.pi*11/12, 12)
    ax.set_xticks(tick_angles)
    ax.set_xticklabels(
        [f'{a*period/(2*np.pi):.0f}' for a in tick_angles], fontsize=7)
    ax.set_yticklabels([])

    def select_and_label(region_slice, color,
                         n_labels=15, min_cents_sep=9):
        """Rank peaks by height; skip if within min_cents_sep of a
        previously labelled peak."""
        region = s_full[region_slice]
        pk_idx, _ = find_peaks(region, prominence=0.001)
        pk_vals = region[pk_idx]
        order = np.argsort(pk_vals)[::-1]
        pk_idx = pk_idx[order]; pk_vals = pk_vals[order]
        offset = region_slice.start or 0
        labelled_cents = []
        count = 0
        for li in range(len(pk_idx)):
            gv = gen_range[pk_idx[li] + offset]
            if any(abs(gv-g) < min_cents_sep for g in labelled_cents):
                continue
            pt = 2*np.pi*gv/period
            pr = pk_vals[li]
            label_r = pr + smax*0.07
            ax.plot(pt, pr, '.', color=color, markersize=3)
            ax.annotate(f'{gv:.1f}', (pt, label_r), fontsize=5,
                        color=color, ha='center', va='center')
            labelled_cents.append(gv)
            count += 1
            if count >= n_labels: break
        print(f"    Placed {count} labels ({color})")

    select_and_label(slice(0, n_half), '#c0392b')
    select_and_label(slice(n_half, len(s_full)), '#2980b9')

    ax.plot([np.pi,np.pi], [0,smax*1.25], '--',
            color='#95a5a6', linewidth=0.8)
    ax.plot([0,0], [0,smax*1.25], '--',
            color='#95a5a6', linewidth=0.8)
    ax.text(np.pi/2, smax*0.35, '4:5:6', ha='center', va='center',
            fontsize=8, color='#c0392b', fontweight='bold')
    ax.text(3*np.pi/2, smax*0.35, '4:5:6:7', ha='center', va='center',
            fontsize=8, color='#2980b9', fontweight='bold')

    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/gen_chain.pdf', **SAVEKW)
    plt.close(fig)
    print("  Saved.")


# ===================================================================
#  Expectation tensor densities (full page, 4×2)
# ===================================================================

def expectation_tensor_modes():
    print("=== Expectation tensor densities ===")
    from matplotlib.gridspec import GridSpec

    p = np.array([0, 200, 400, 500, 700, 900, 1100], dtype=float)
    sigma = 10; period = 1200; normalize = 'none'

    # Rows ordered by r; within r, relative before absolute.
    # Row 0: r=1 abs (dim=1), Row 1: r=2 rel (dim=1),
    # Row 2: r=2 abs (dim=2), Row 3: r=3 rel (dim=2).
    configs = [
        (1, False, False), (1, False, True),
        (2, True,  False), (2, True,  True),
        (2, False, False), (2, False, True),
        (3, True,  False), (3, True,  True),
    ]

    step_1d = 1; step_2d = 5
    ax_min_nonper = 0; ax_max_nonper = 1200

    fig = plt.figure(figsize=(7.0, 9.0))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.08,
                 height_ratios=[1, 1, 1.6, 1.6])

    # Line-plot axes: row 0 shares x+y within itself; row 1 shares x with
    # row 0 but has independent y (different density scale).
    ax_l00 = fig.add_subplot(gs[0, 0])
    ax_l01 = fig.add_subplot(gs[0, 1], sharex=ax_l00, sharey=ax_l00)
    ax_l10 = fig.add_subplot(gs[1, 0], sharex=ax_l00)
    ax_l11 = fig.add_subplot(gs[1, 1], sharex=ax_l00, sharey=ax_l10)

    # Heatmap axes (rows 2-3): shared x and y
    ax_h00 = fig.add_subplot(gs[2, 0])
    ax_h01 = fig.add_subplot(gs[2, 1], sharex=ax_h00, sharey=ax_h00)
    ax_h10 = fig.add_subplot(gs[3, 0], sharex=ax_h00, sharey=ax_h00)
    ax_h11 = fig.add_subplot(gs[3, 1], sharex=ax_h00, sharey=ax_h00)

    axes_grid = [
        [ax_l00, ax_l01], [ax_l10, ax_l11],
        [ax_h00, ax_h01], [ax_h10, ax_h11],
    ]

    panel_labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']

    for ci, (r, is_rel, is_per) in enumerate(configs):
        row, col = divmod(ci, 2)
        ax = axes_grid[row][col]
        dim = r - int(is_rel)
        ax_min = 0 if is_per else ax_min_nonper
        ax_max = period if is_per else ax_max_nonper
        mode_str = 'relative' if is_rel else 'absolute'
        per_str = 'periodic' if is_per else 'non-periodic'
        ax_label = 'Interval' if is_rel else 'Pitch'
        subtitle = f'$r = {r}$, {mode_str}, {per_str}'

        print(f"  {panel_labels[ci]} {subtitle} (dim={dim})...",
              end=" ", flush=True)
        dens = mpt.build_maet(p, None, sigma, r, is_rel, is_per,
                              period, verbose=False)

        if dim == 1:
            res = max(2, round((ax_max - ax_min) / step_1d) + 1)
            x = np.linspace(ax_min, ax_max, res)
            vals = mpt.eval_maet(dens, x, normalize, verbose=False)
            ax.plot(x, vals, color='#2c3e50', linewidth=0.8)
            ax.set_xlim(ax_min, ax_max)
        elif dim == 2:
            res = max(2, round((ax_max - ax_min) / step_2d) + 1)
            x = np.linspace(ax_min, ax_max, res)
            Ga, Gb = np.meshgrid(x, x)
            upper = np.triu(np.ones((res, res), dtype=bool))
            Xu = np.vstack([Ga[upper], Gb[upper]])
            vu = mpt.eval_maet(dens, Xu, normalize, verbose=False)
            V = np.zeros((res, res))
            V[upper] = vu
            V = V + V.T - np.diag(np.diag(V))
            ax.imshow(V, extent=[ax_min, ax_max, ax_min, ax_max],
                      origin='lower', aspect='equal', cmap='inferno')

        ax.text(0.03, 0.95, f'{panel_labels[ci]} {subtitle}',
                transform=ax.transAxes, fontsize=6.5, va='top',
                color='white' if dim == 2 else '#2c3e50')
        print("done.")

    # Row 0 only: y-axis 0-1 (row 1 auto-scales independently)
    ax_l00.set_ylim(0, 1)

    # Suppress redundant tick labels; add axis labels
    ax_l00.tick_params(labelbottom=False)
    ax_l01.tick_params(labelbottom=False, labelleft=False)
    ax_l10.set_xlabel('Pitch / Interval (cents)', fontsize=7)
    ax_l11.tick_params(labelleft=False)
    ax_l11.set_xlabel('Pitch / Interval (cents)', fontsize=7)
    ax_l00.set_ylabel('Density', fontsize=7)
    ax_l10.set_ylabel('Density', fontsize=7)

    ax_h00.tick_params(labelbottom=False)
    ax_h01.tick_params(labelbottom=False, labelleft=False)
    ax_h10.set_xlabel('Pitch / Interval 1 (cents)', fontsize=7)
    ax_h11.tick_params(labelleft=False)
    ax_h11.set_xlabel('Pitch / Interval 1 (cents)', fontsize=7)
    ax_h00.set_ylabel('Pitch / Interval 2 (cents)', fontsize=7)
    ax_h10.set_ylabel('Pitch / Interval 2 (cents)', fontsize=7)

    fig.savefig(f'{OUTPUT_DIR}/exp_tensors.pdf', **SAVEKW)
    plt.close(fig)
    print("  Saved.")


# ===================================================================
#  Main
# ===================================================================

# The consonance comparison and the tensor harmonicity zoom have their own
# scripts, make_consonance.py and make_th_zoom.py, being far slower than these.

if __name__ == '__main__':
    expectation_tensor_modes()
    generator_chain_tunings(gen_step=0.1)
    spcs_probe_profiles()
    triad_similarity_grids()
    balance_on_the_circle()
    rhythm_position_features()
    print("\nSaved to", OUTPUT_DIR)
