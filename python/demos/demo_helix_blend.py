"""demo_helix_blend.py

Helix blend: routing pitch through two groups of a MAET.

Demonstrates a multi-attribute expectation tensor pattern in which the
same pitch values are routed simultaneously through a periodic
pitch-class group and a linear register group. Sweeping the
register-group sigma while holding the pitch-class-group sigma fixed
morphs the similarity profile of a motif against a longer stream from

  * "matches every octave-displaced recurrence equally"   (large sigma_reg)
  * through graded octave tolerance                       (medium)
  * to "matches only the same-register recurrence"        (small sigma_reg).

Equivalence with Shepard's model. The factored Gaussian

    exp(- d_pc(p1,p2)^2 / (2 sigma_pc^2))
  * exp(-  (p1 - p2)^2  / (2 sigma_reg^2))

is equivalent to a Gaussian kernel of width sigma = sigma_pc on the
pitch-class-cum-register cylinder with stretch h = sigma_pc/sigma_reg.
Shepard's helix itself has no built-in smoothing; this MAET pattern
adds it, parametrised naturally in two pitch-domain sigma values.

Two technical points. First, the sigmas are density widths (MPT's
convention throughout the toolbox); the pairwise inner-product kernel
between two smeared events has effective standard deviation
sqrt(2)*sigma. Second, the MAET pitch-class group uses shortest-arc
distance, so the geometry is the pc cylinder rather than the literal
3-D Shepard helix (a Euclidean embedding that uses chord distance).
At sigma_pc values typical of tonal perception the two are
indistinguishable.

Two parts:

  Part 1. Synthetic. A three-note C-major motif stated at four
          registers, with non-pitch-class-overlapping filler between
          instances.

  Part 2. Fugal texture in C minor (BWV 847-inspired, stylised; not
          transcribed from the score). A six-note subject stated in
          bass, alto, and soprano, with short counter-material
          between entries.

Each part produces three stacked panels:
  (a) the event stream,
  (b) a similarity heatmap over (time offset, sigma_reg),
  (c) three overlaid profile curves at representative sigma_reg values.

Uses: build_exp_tens, windowed_similarity, convert_pitch.
"""

import numpy as np
import matplotlib.pyplot as plt
import mpt


# ==================================================================
#  User parameters
# ==================================================================

# -- Common --
SIG_PC           = 30.0                                    # pc group sigma (cents)
SIG_REG_SWEEP    = np.logspace(np.log10(100), np.log10(8000), 25)
SIG_REG_PROFILES = [200.0, 600.0, 3000.0]                  # three overlaid profiles
WIN_MIX          = 0.5                                     # rectangular x Gaussian

# -- Part 1 (synthetic) --
SIG_TIME_1       = 0.10                                    # sec
WIN_SIZE_TIME_1  = 6.0                                     # effective sd in units of sigma_time
OFFSETS_1        = np.arange(-0.5, 12.5 + 1e-9, 0.02)

# -- Part 2 (fugal texture) --
SIG_TIME_2       = 0.08
WIN_SIZE_TIME_2  = 12.0
OFFSETS_2        = np.arange(-0.5, 8.5 + 1e-9, 0.02)


# ==================================================================
#  Core: two-group (pc, reg) + time MAET, and sigma_reg sweep
# ==================================================================

def build_2group(pitch_cents, time_sec, sigma_pc, sigma_reg, sigma_time):
    """MAET density with pitch routed through two groups plus a time group.

    Attributes: (pitch, pitch, time). Groups: (pc, reg, time), with pc
    periodic at 1200 cents and the other two linear. All r=1.
    """
    p = np.asarray(pitch_cents, dtype=float).reshape(1, -1)
    t = np.asarray(time_sec,    dtype=float).reshape(1, -1)
    return mpt.build_exp_tens(
        [p, p, t], None,
        [sigma_pc, sigma_reg, sigma_time],
        [1, 1, 1],
        None,
        [False, False, False],
        [True,  False, False],
        [1200.0, 0.0, 0.0],
        verbose=False,
    )


def sweep_profiles(query_cents, query_time, context_cents, context_time,
                   sigma_pc, sigma_reg_values, sigma_time,
                   win_size_time, win_mix, offsets):
    """Return a (n_sigma_reg, n_offsets) array of cross-correlation profiles."""
    window_spec = {
        "size": [np.inf, np.inf, win_size_time],
        "mix":  [0.0,     0.0,     win_mix],
    }
    offsets_3d = np.vstack([np.zeros_like(offsets),
                            np.zeros_like(offsets),
                            offsets])
    out = np.empty((len(sigma_reg_values), len(offsets)))
    for i, sig_reg in enumerate(sigma_reg_values):
        q = build_2group(query_cents,   query_time,
                         sigma_pc, sig_reg, sigma_time)
        c = build_2group(context_cents, context_time,
                         sigma_pc, sig_reg, sigma_time)
        out[i, :] = mpt.windowed_similarity(q, c, window_spec, offsets_3d,
                                         verbose=False)
    return out


# ==================================================================
#  Part 1 -- Synthetic motif at four registers
# ==================================================================

def build_part1_stream():
    """Three-note motif at four registers with non-overlapping filler."""
    motif_midi  = np.array([60, 64, 67])                    # C4 E4 G4
    filler_midi = np.array([62, 65, 69])                    # D4 F4 A4
    regs_st     = [0, 12, -12, 24]
    dt          = 0.5

    ctx_midi, ctx_t, motif_event_idx = [], [], []
    t = 0.0
    for shift_st in regs_st:
        for fp in filler_midi:
            ctx_midi.append(fp + shift_st); ctx_t.append(t); t += dt
        for mp in motif_midi:
            motif_event_idx.append(len(ctx_midi))
            ctx_midi.append(mp + shift_st); ctx_t.append(t); t += dt
    ctx_midi = np.array(ctx_midi, dtype=float)
    ctx_t    = np.array(ctx_t,    dtype=float)

    query_midi = motif_midi.astype(float)
    query_t    = np.arange(len(motif_midi)) * dt

    per_entry = len(motif_midi)
    motif_centroids = np.array([
        np.mean(ctx_t[motif_event_idx[k*per_entry:(k+1)*per_entry]])
        for k in range(len(regs_st))
    ])
    return ctx_midi, ctx_t, query_midi, query_t, motif_event_idx, motif_centroids


# ==================================================================
#  Part 2 -- Fugal texture (BWV 847-inspired, stylised)
# ==================================================================

def build_part2_stream():
    """Six-note subject stated in bass, alto, and soprano.

    Between entries, three counter-material notes in a voice other than
    the entering one; this produces partial pc-overlap (one shared pc
    per group) that gives a visible but clearly subordinate background.
    """
    subj_ref = np.array([60, 63, 65, 63, 62, 60])           # C Eb F Eb D C
    cnt1     = np.array([57, 55, 53])                       # A3 G3 F3
    cnt2     = np.array([74, 72, 70])                       # D5 C5 Bb4
    dt       = 0.30
    gap      = 0.30

    entries_st = [-12, 0, 12]

    ctx_midi, ctx_t, subj_event_idx = [], [], []
    t = 0.0
    for k, shift_st in enumerate(entries_st):
        idx0 = len(ctx_midi)
        for mp in subj_ref:
            ctx_midi.append(mp + shift_st); ctx_t.append(t); t += dt
        subj_event_idx.extend(range(idx0, idx0 + len(subj_ref)))
        t -= dt  # undo trailing increment, we'll add gap instead
        if k < len(entries_st) - 1:
            t += gap
            cnt = cnt1 if k == 0 else cnt2
            for mp in cnt:
                ctx_midi.append(mp); ctx_t.append(t); t += dt
            t += gap
    ctx_midi = np.array(ctx_midi, dtype=float)
    ctx_t    = np.array(ctx_t,    dtype=float)

    query_midi = subj_ref.astype(float)
    query_t    = np.arange(len(subj_ref)) * dt

    per_entry = len(subj_ref)
    subj_centroids = np.array([
        np.mean(ctx_t[subj_event_idx[k*per_entry:(k+1)*per_entry]])
        for k in range(len(entries_st))
    ])
    return ctx_midi, ctx_t, query_midi, query_t, subj_event_idx, subj_centroids


# ==================================================================
#  Plotting
# ==================================================================

def plot_part(fig, suptitle,
              ctx_midi, ctx_t, marker_idx, peak_offsets,
              query_centroid_t,
              heat, prof, offsets, sigma_reg_sweep, sigma_reg_profiles,
              label_marked, label_unmarked):
    """All three panels share the "query offset" x-axis:

        offset = context_time - query_centroid_time.

    Under this convention, a marker (motif/subject) occurrence in panel
    (a) sits at the same x-coordinate as its corresponding peak in
    panels (b) and (c).
    """
    is_marked = np.zeros(len(ctx_t), dtype=bool)
    is_marked[list(marker_idx)] = True
    ctx_x = ctx_t - query_centroid_t       # shift to "offset" coordinates

    # Explicit GridSpec with a dedicated colorbar column. This keeps
    # all three main panels at the same x-axis width; only the middle
    # panel's row extends into the colorbar column.
    gs = fig.add_gridspec(3, 2,
                          height_ratios=[1, 2, 1.3],
                          width_ratios=[40, 1],
                          hspace=0.35, wspace=0.04)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    cax = fig.add_subplot(gs[1, 1])
    axes = [ax1, ax2, ax3]

    # (a) Event stream, y-axis in MIDI
    ax = axes[0]
    ax.scatter(ctx_x[~is_marked], ctx_midi[~is_marked], s=30,
               c="#d8d8d8", edgecolors="#666", linewidths=0.6,
               label=label_unmarked, zorder=2)
    ax.scatter(ctx_x[is_marked], ctx_midi[is_marked], s=58,
               c="#b83030", edgecolors="#552020", linewidths=0.6,
               label=label_marked, zorder=3)
    for po in peak_offsets:
        ax.axvline(po, color="#888", linestyle=":", linewidth=0.6, zorder=1)
    ax.set_ylabel("MIDI pitch")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_title("(a) Event stream")

    # (b) Heatmap over (offset, sigma_reg)
    ax = axes[1]
    im = ax.pcolormesh(offsets, sigma_reg_sweep, heat,
                       shading="nearest", cmap="viridis",
                       vmin=0, vmax=max(heat.max(), 1e-3))
    ax.set_yscale("log")
    ax.set_ylabel(r"$\sigma_{\mathrm{reg}}$ (cents)")
    for po in peak_offsets:
        ax.axvline(po, color="white", linestyle="--",
                   linewidth=0.7, alpha=0.6)
    for spv in sigma_reg_profiles:
        ax.axhline(spv, color="white", linestyle=":",
                   linewidth=0.6, alpha=0.6)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("windowed similarity", fontsize=9)
    ax.set_title(r"(b) Similarity heatmap over (offset, $\sigma_{\mathrm{reg}}$)")

    # (c) Three overlaid profiles
    ax = axes[2]
    colours = ["#1f4eb8", "#2b8a3e", "#c25008"]
    for i, sr in enumerate(sigma_reg_profiles):
        ax.plot(offsets, prof[i], linewidth=1.8, color=colours[i],
                label=fr"$\sigma_{{\mathrm{{reg}}}} = {sr:g}$ cents")
    for po in peak_offsets:
        ax.axvline(po, color="#888", linestyle="--",
                   linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Query time offset (s)")
    ax.set_ylabel("Cosine similarity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    ax.set_title(r"(c) Profiles at three representative $\sigma_{\mathrm{reg}}$")

    fig.suptitle(suptitle, fontsize=12, y=0.995)
    # Hide x-tick labels on top two panels (sharex shares limits only)
    ax1.tick_params(axis="x", labelbottom=False)
    ax2.tick_params(axis="x", labelbottom=False)
    # Leave room on the right for the outside-anchored legends.
    fig.subplots_adjust(left=0.08, right=0.80, top=0.94, bottom=0.07)


# ==================================================================
#  Driver
# ==================================================================

def main():
    # -- Part 1 --
    print("Part 1: synthetic motif at four registers.")
    (ctx1_midi, ctx1_t, q1_midi, q1_t,
     motif_idx1, motif_cent1) = build_part1_stream()
    ctx1_cents = ctx1_midi * 100.0
    q1_cents   = q1_midi   * 100.0

    heat1 = sweep_profiles(q1_cents, q1_t, ctx1_cents, ctx1_t,
                           SIG_PC, SIG_REG_SWEEP, SIG_TIME_1,
                           WIN_SIZE_TIME_1, WIN_MIX, OFFSETS_1)
    prof1 = sweep_profiles(q1_cents, q1_t, ctx1_cents, ctx1_t,
                           SIG_PC, SIG_REG_PROFILES, SIG_TIME_1,
                           WIN_SIZE_TIME_1, WIN_MIX, OFFSETS_1)
    peak1 = motif_cent1 - np.mean(q1_t)

    fig1 = plt.figure("Helix blend: synthetic", figsize=(9.8, 8.0))
    plot_part(fig1, "Helix blend (synthetic): C-E-G at four registers",
              ctx1_midi, ctx1_t, motif_idx1, peak1, float(np.mean(q1_t)),
              heat1, prof1, OFFSETS_1, SIG_REG_SWEEP, SIG_REG_PROFILES,
              label_marked="motif events (C-E-G)",
              label_unmarked="filler events")

    # -- Part 2 --
    print("Part 2: fugal texture (BWV 847-inspired, stylised).")
    (ctx2_midi, ctx2_t, q2_midi, q2_t,
     subj_idx2, subj_cent2) = build_part2_stream()
    ctx2_cents = ctx2_midi * 100.0
    q2_cents   = q2_midi   * 100.0

    heat2 = sweep_profiles(q2_cents, q2_t, ctx2_cents, ctx2_t,
                           SIG_PC, SIG_REG_SWEEP, SIG_TIME_2,
                           WIN_SIZE_TIME_2, WIN_MIX, OFFSETS_2)
    prof2 = sweep_profiles(q2_cents, q2_t, ctx2_cents, ctx2_t,
                           SIG_PC, SIG_REG_PROFILES, SIG_TIME_2,
                           WIN_SIZE_TIME_2, WIN_MIX, OFFSETS_2)
    peak2 = subj_cent2 - np.mean(q2_t)

    fig2 = plt.figure("Helix blend: fugal texture", figsize=(9.8, 8.0))
    plot_part(fig2, "Helix blend (BWV 847-inspired, stylised): subject in bass, alto, soprano",
              ctx2_midi, ctx2_t, subj_idx2, peak2, float(np.mean(q2_t)),
              heat2, prof2, OFFSETS_2, SIG_REG_SWEEP, SIG_REG_PROFILES,
              label_marked="subject events",
              label_unmarked="counter-material")

    plt.show()


if __name__ == "__main__":
    main()
