"""demo_helix_blend.py

Helix blend: routing pitch through two groups of a MAET.

Demonstrates a multi-attribute expectation tensor pattern in which the
same pitch values are routed simultaneously through a periodic
pitch-class group and a linear pitch-height group. Sweeping the
pitch-height-group sigma while holding the pitch-class-group sigma fixed
morphs the similarity profile of a motif against a longer stream from

  * "matches every octave-displaced recurrence equally"   (large sigma_ph)
  * through graded octave tolerance                       (medium)
  * to "matches only the same-height recurrence"          (small sigma_ph).

Equivalence with Shepard's model. The factored Gaussian

    exp(- d_pc(p1,p2)^2 / (2 sigma_pc^2))
  * exp(-  (p1 - p2)^2  / (2 sigma_ph^2))

is equivalent to a Gaussian kernel of width sigma = sigma_pc on the
pitch-class-cum-height cylinder with stretch h = sigma_pc/sigma_ph.
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
          heights, with non-pitch-class-overlapping filler between
          instances.

  Part 2. Fugal texture in C minor (BWV 847-inspired, stylised; not
          transcribed from the score). A six-note subject stated in
          bass, alto, and soprano, with short counter-material
          between entries.

Each part produces three stacked panels:
  (a) the event stream,
  (b) a similarity heatmap over (time offset, sigma_ph),
  (c) three overlaid profile curves at representative sigma_ph values.

Uses: windowed_similarity (event weighting over a raw pre-MAET),
transform_attributes.
"""

import numpy as np
import matplotlib.pyplot as plt
import mpt


# ==================================================================
#  User parameters
# ==================================================================

# -- Common --
SIG_PC           = 30.0                                    # pc group sigma (cents)
SIG_PH_SWEEP    = np.logspace(np.log10(100), np.log10(8000), 25)
SIG_PH_PROFILES = [200.0, 600.0, 3000.0]                  # three overlaid profiles
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
#  Core: two-group (pc, ph) + time MAET, and sigma_ph sweep
# ==================================================================

def helix_pre_maet(pitch_cents, time_sec, sigma_pc, sigma_time):
    """The same pitch values routed through two attributes, plus time.

    Attributes: (pitch, pitch, time), read as (pc, ph, time): the first
    pitch copy is periodic at 1200 cents, the second and the time axis
    are linear. All r = 1.

    The pre-MAET carries its own geometry, so nothing has to be threaded
    alongside it. The pitch-height width is left at NaN --- NA, the
    value the sweep supplies --- since it is the one parameter that
    varies and no baseline for it would be honest.
    """
    p = np.asarray(pitch_cents, dtype=float).reshape(1, -1)
    t = np.asarray(time_sec,    dtype=float).reshape(1, -1)
    n = p.shape[1]
    specs = [
        {"name": "pitch class",  "r": 1, "rel": False, "sym": True,
         "sigma": sigma_pc,   "is_per": True,  "period": 1200.0},
        {"name": "pitch height", "r": 1, "rel": False, "sym": True,
         "sigma": float("nan"), "is_per": False, "period": 0.0},
        {"name": "time",         "r": 1, "rel": False, "sym": True,
         "sigma": sigma_time, "is_per": False, "period": 0.0},
    ]
    return mpt.pre_maet([p, p, t], [np.ones((1, n))] * 3, specs)


def sweep_profiles(q_cents, q_t, c_cents, c_t,
                   sigma_pc, sigma_ph_values, sigma_time,
                   win_size_time, win_mix, offsets, show_input=False):
    """Return a (n_sigma_ph, n_offsets) array of windowed-similarity
    profiles (a cross-correlation of the query against the time-windowed
    context).

    Windowing is event weighting: at each sweep position the window,
    centred on that position along the time axis, multiplies the
    per-event weights of the context before its density is built, and
    the query is translated so that its time centroid lands on the same
    position. The window has standard deviation win_size_time * sigma_time
    and shape win_mix (0 Gaussian, 1 rectangular).
    """
    TIME = 2                                     # the swept (window) axis

    # The window family has fixed variance sd^2 for every shape; the
    # width argument is the rectangle-equivalent full width 2*sqrt(3)*sd.
    sd_time = win_size_time * sigma_time
    context_window = (win_mix, 2.0 * np.sqrt(3.0) * sd_time)

    # Sweep positions are absolute times on the context axis; the plotted
    # offset is the position relative to the query's time centroid.
    centres = np.asarray(offsets, dtype=float) + float(np.mean(q_t))

    # Only the pitch-height width varies across the sweep, so the two
    # pre-MAETs are built once and each call names that one parameter.
    # A selective override --- the entries left None keep what the spec
    # carries --- says exactly that, and the pre-MAETs are unchanged by
    # it. The table below shows sigma = NA on the swept attribute, the
    # value each call supplies.
    pm_q = helix_pre_maet(q_cents, q_t, sigma_pc, sigma_time)
    pm_c = helix_pre_maet(c_cents, c_t, sigma_pc, sigma_time)

    if show_input:
        mpt.show_pre_maet(pm_c, max_events=4)
        print()

    out = np.empty((len(sigma_ph_values), len(offsets)))
    for i, sig_ph in enumerate(sigma_ph_values):
        out[i, :] = mpt.windowed_similarity(
            pm_c, pm_q, centres, sigma=[None, sig_ph, None],
            window_attr=TIME, drop_window_attr=False,
            context_window=context_window, locate="centroid",
            normalize="oneSidedDenom", verbose=False)
    return out


# ==================================================================
#  Part 1 -- Synthetic motif at four heights
# ==================================================================

def build_part1_stream():
    """Three-note motif at four heights with non-overlapping filler."""
    motif_midi  = np.array([60, 64, 67])                    # C4 E4 G4
    filler_midi = np.array([62, 65, 69])                    # D4 F4 A4
    heights_st     = [0, 12, -12, 24]
    dt          = 0.5

    ctx_midi, ctx_t, motif_event_idx = [], [], []
    t = 0.0
    for shift_st in heights_st:
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
        for k in range(len(heights_st))
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
              heat, prof, offsets, sigma_ph_sweep, sigma_ph_profiles,
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

    # (b) Heatmap over (offset, sigma_ph)
    ax = axes[1]
    im = ax.pcolormesh(offsets, sigma_ph_sweep, heat,
                       shading="nearest", cmap="viridis",
                       vmin=0, vmax=max(heat.max(), 1e-3))
    ax.set_yscale("log")
    ax.set_ylabel(r"$\sigma_{\mathrm{ph}}$ (cents)")
    for po in peak_offsets:
        ax.axvline(po, color="white", linestyle="--",
                   linewidth=0.7, alpha=0.6)
    for spv in sigma_ph_profiles:
        ax.axhline(spv, color="white", linestyle=":",
                   linewidth=0.6, alpha=0.6)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("windowed similarity", fontsize=9)
    ax.set_title(r"(b) Similarity heatmap over (offset, $\sigma_{\mathrm{ph}}$)")

    # (c) Three overlaid profiles
    ax = axes[2]
    colours = ["#1f4eb8", "#2b8a3e", "#c25008"]
    for i, sr in enumerate(sigma_ph_profiles):
        ax.plot(offsets, prof[i], linewidth=1.8, color=colours[i],
                label=fr"$\sigma_{{\mathrm{{ph}}}} = {sr:g}$ cents")
    for po in peak_offsets:
        ax.axvline(po, color="#888", linestyle="--",
                   linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Query time offset (s)")
    ax.set_ylabel("Cosine similarity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    ax.set_title(r"(c) Profiles at three representative $\sigma_{\mathrm{ph}}$")

    fig.suptitle(suptitle, fontsize=12, y=0.995)
    # Hide x-tick labels on top two panels (sharex shares limits only)
    ax1.tick_params(axis="x", labelbottom=False)
    ax2.tick_params(axis="x", labelbottom=False)
    # Leave room on the right for the outside-anchored legends.
    fig.subplots_adjust(left=0.08, right=0.80, top=0.94, bottom=0.07)


# ==================================================================
#  Driver
# ==================================================================

def report_peaks(prof, offsets, sigma_phs, true_peaks):
    """Print, per profile sigma_ph, the similarity at each true statement
    offset: as sigma_ph widens, octave-displaced statements rise from
    near zero towards the same-height value of 1."""
    for row, sr in zip(prof, sigma_phs):
        vals = [row[int(np.argmin(np.abs(offsets - pk)))] for pk in np.atleast_1d(true_peaks)]
        print(f"  sigma_ph = {sr:6.0f} cents: similarity at the statements = "
              + ", ".join(f"{v:.4f}" for v in vals))


def main():
    # -- Part 1 --
    print("Part 1: synthetic motif at four heights.")
    (ctx1_midi, ctx1_t, q1_midi, q1_t,
     motif_idx1, motif_cent1) = build_part1_stream()
    ctx1_cents = mpt.transform_attributes(ctx1_midi, None, ('midi', 'cents'))
    q1_cents   = mpt.transform_attributes(q1_midi,   None, ('midi', 'cents'))

    heat1 = sweep_profiles(q1_cents, q1_t, ctx1_cents, ctx1_t,
                           SIG_PC, SIG_PH_SWEEP, SIG_TIME_1,
                           WIN_SIZE_TIME_1, WIN_MIX, OFFSETS_1,
                           show_input=True)
    prof1 = sweep_profiles(q1_cents, q1_t, ctx1_cents, ctx1_t,
                           SIG_PC, SIG_PH_PROFILES, SIG_TIME_1,
                           WIN_SIZE_TIME_1, WIN_MIX, OFFSETS_1)
    peak1 = motif_cent1 - np.mean(q1_t)
    report_peaks(prof1, OFFSETS_1, SIG_PH_PROFILES, peak1)

    fig1 = plt.figure("Helix blend: synthetic", figsize=(9.8, 8.0))
    plot_part(fig1, "Helix blend (synthetic): C-E-G at four heights",
              ctx1_midi, ctx1_t, motif_idx1, peak1, float(np.mean(q1_t)),
              heat1, prof1, OFFSETS_1, SIG_PH_SWEEP, SIG_PH_PROFILES,
              label_marked="motif events (C-E-G)",
              label_unmarked="filler events")

    # -- Part 2 --
    print("Part 2: fugal texture (BWV 847-inspired, stylised).")
    (ctx2_midi, ctx2_t, q2_midi, q2_t,
     subj_idx2, subj_cent2) = build_part2_stream()
    ctx2_cents = mpt.transform_attributes(ctx2_midi, None, ('midi', 'cents'))
    q2_cents   = mpt.transform_attributes(q2_midi,   None, ('midi', 'cents'))

    heat2 = sweep_profiles(q2_cents, q2_t, ctx2_cents, ctx2_t,
                           SIG_PC, SIG_PH_SWEEP, SIG_TIME_2,
                           WIN_SIZE_TIME_2, WIN_MIX, OFFSETS_2)
    prof2 = sweep_profiles(q2_cents, q2_t, ctx2_cents, ctx2_t,
                           SIG_PC, SIG_PH_PROFILES, SIG_TIME_2,
                           WIN_SIZE_TIME_2, WIN_MIX, OFFSETS_2)
    peak2 = subj_cent2 - np.mean(q2_t)
    report_peaks(prof2, OFFSETS_2, SIG_PH_PROFILES, peak2)

    fig2 = plt.figure("Helix blend: fugal texture", figsize=(9.8, 8.0))
    plot_part(fig2, "Helix blend (BWV 847-inspired, stylised): subject in bass, alto, soprano",
              ctx2_midi, ctx2_t, subj_idx2, peak2, float(np.mean(q2_t)),
              heat2, prof2, OFFSETS_2, SIG_PH_SWEEP, SIG_PH_PROFILES,
              label_marked="subject events",
              label_unmarked="counter-material")

    plt.show()


if __name__ == "__main__":
    main()
