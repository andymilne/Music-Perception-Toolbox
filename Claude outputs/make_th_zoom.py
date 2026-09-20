"""
Zoomed view of the tensor harmonicity landscape around the triad peaks.
=======================================================================

Tensor harmonicity evaluates the relative K-ad expectation tensor density of a
single harmonic series at a chord's ordered interval vector. For a triad that
vector is two-dimensional, so the whole measure can be drawn as a surface over
the plane of (first interval, second interval) and a chord is simply a point on
it.

This figure zooms in on two regions of that surface: the neighbourhood of the
just major triad 4:5:6 and of the just minor triad 10:12:15. The purpose is to
show where the equal-tempered chords sit relative to the just peaks — near
them, rather than at them, and comfortably within the width that the smoothing
parameter sigma allows.

Rather than calling tensorHarmonicity once per grid point, the harmonic
template density is built once with buildExpTens and then evaluated at every
grid point with evalExpTens. This is exactly what tensorHarmonicity does
internally for a single chord, and returns identical values.

Run:  python3 make_th_zoom.py

This is slow: about an hour at the article's 1-cent resolution. Raise
RESOLUTION for a quicker, coarser run -- see the note beside it below.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# The toolbox is imported as `mpt`. If it is not installed, point MPT_PYTHON
# at the `python` directory of a version 3.0 checkout.
_MPT = os.environ.get("MPT_PYTHON")
if _MPT:
    sys.path.insert(0, _MPT)
import mpt  # noqa: E402
from mpt import add_spectra, build_maet, eval_maet  # noqa: E402

# The published values were computed with an untruncated kernel;
# version 3 truncates at six sigma by default, so the untruncated
# setting is restored here.
mpt.set_default(truncation_sigmas=float("inf"), show_hints=False)

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")

# Parameters as in the consonance comparison, so that this figure
# is a zoom of the same landscape: 36 harmonics per tone with power-law rolloff
# rho = 1, and sigma = 12 cents. Values are identical under versions 2.0 and
# 2.2 of the toolbox; 2.2 evaluates the grid considerably faster.
N_HARMONICS, RHO = 16, 1.0
VMIN_PERCENTILE = 2.0     # low-end clip for the colour scale; see draw()
# Tensor harmonicity evaluates the template density at a single point, so it
# must carry the uncertainty of the query as well as of the template. That is
# exactly equivalent to widening the kernel to sqrt(sigma^2 + tau^2); with
# tau = sigma this is sigma * sqrt(2), matching the consonance comparison.
SIGMA = 12.0 * np.sqrt(2.0)
K = 3                      # triads, so a two-dimensional interval vector
# SLOW: evaluating the two panels takes on the order of an hour at
# RESOLUTION = 1.0, which is the value used in the article. Cost scales as the
# square of the resolution, so 2.0 is about four times faster and 4.0 sixteen
# times. Since sigma * sqrt(2) is about 17 cents here, even 4-cent steps
# sample each peak many times over, and the figure changes little. The grids
# are cached in th_zoom_grids.npz, so re-runs only redraw; delete it to
# recompute.
RESOLUTION = 1.0           # cents per grid step
CHUNK = 400                # query points per call, to bound peak memory

plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})


def build_template():
    """The relative K-ad density of one harmonic series.

    The template pitch is duplicated K times before partials are added, so that
    a single partial may fill more than one slot of a K-tuple; without this,
    unisons and octave doublings would be penalised rather than reinforced.
    """
    p, w = add_spectra(np.zeros(K), np.ones(K),
                       "harmonic", N_HARMONICS, "powerlaw", RHO)
    return build_maet(p, w, SIGMA, K, True, False, 1200, verbose=False)


def evaluate_grid(T, x_range, y_range):
    """Evaluate the density over a grid of ordered interval vectors."""
    xs = np.arange(x_range[0], x_range[1] + RESOLUTION, RESOLUTION)
    ys = np.arange(y_range[0], y_range[1] + RESOLUTION, RESOLUTION)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    q = np.vstack([X.ravel(), Y.ravel()])          # 2 x N, one column per point
    out = np.empty(q.shape[1])
    for i in range(0, q.shape[1], CHUNK):
        out[i:i + CHUNK] = eval_maet(T, q[:, i:i + CHUNK],
                                     normalize="none", verbose=False)
    return xs, ys, out.reshape(X.shape)


# Each region is labelled with the just chord that peaks there and the
# equal-tempered chord nearest to it.
REGIONS = [
    dict(title="(a) Close position",
         x=(250, 550), y=(650, 950),
         chords=[(400, 700, "M"), (300, 800, "M"), (500, 900, "M"),
                 (300, 700, "m"), (400, 900, "m"), (500, 800, "m")],
         # Peaks located from the evaluated grid and identified with the
         # simplest integer triple matching to within a cent.
         just=[(498, 884, "3:4:5"), (386, 702, "4:5:6"), (316, 814, "5:6:8"),
               (267, 702, "6:7:9"), (498, 702, "6:8:9"), (267, 884, "6:7:10")]),
    dict(title="(b) Bass an octave lower",
         x=(1450, 1750), y=(1850, 2150),
         chords=[(1600, 1900, "M"), (1500, 2000, "M"), (1700, 2100, "M"),
                 (1500, 1900, "m"), (1600, 2100, "m"), (1700, 2000, "m")],
         just=[(1586, 1902, "2:5:6"), (1467, 1902, "3:7:9"),
               (1467, 2084, "3:7:10"), (1698, 1902, "3:8:9"),
               (1698, 2084, "3:8:10")]),
]


def main():
    # Evaluating the grid is the expensive step, so cache it: replotting then
    # costs nothing. Delete the .npz to force recomputation.
    cache = os.path.join(os.path.dirname(__file__), "th_zoom_grids.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        grids = [(z[f"x{i}"], z[f"y{i}"], z[f"Z{i}"]) for i in range(len(REGIONS))]
    else:
        T = build_template()
        grids = [evaluate_grid(T, r["x"], r["y"]) for r in REGIONS]
        np.savez_compressed(cache, **{f"{k}{i}": g[j]
                                      for i, g in enumerate(grids)
                                      for j, k in enumerate("xyZ")})

    # A shared logarithmic colour scale: the density spans several orders of
    # magnitude, and the scaling that best matches perception is unsettled, so
    # the log is used here only to make both regions legible at once.
    # The values here span only about 1.5 decades, so the colour scale is set
    # from the data rather than stretched over a fixed range.
    # The colour scale is logarithmic and shared by both panels. Its lower
    # limit is clipped at a low percentile rather than the true minimum: a
    # small very-dark region in panel (b) reaches about 1e-4, which would
    # stretch the scale over nearly four decades and flatten the contrast
    # everywhere else. Clipping costs only the darkest few per cent of the
    # area, which carries no structure of interest.
    allv = np.concatenate([g[2].ravel() for g in grids])
    vmax = allv.max()
    vmin = np.percentile(allv, VMIN_PERCENTILE)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0))

    for ax, region, (xs, ys, Z) in zip(axes, REGIONS, grids):
        # imshow rather than pcolormesh: the grid is regular, and pcolormesh
        # produces visible banding when this many cells are downsampled.
        im = ax.imshow(Z.T, origin="lower", cmap="magma", aspect="auto",
                       interpolation="bilinear", rasterized=True,
                       extent=[xs[0], xs[-1], ys[0], ys[-1]],
                       norm=LogNorm(vmin=vmin, vmax=vmax))
        # Just chords, marked with a cross.
        for jx, jy, jlab in region["just"]:
            ax.plot(jx, jy, marker="+", color="white", markersize=8,
                    markeredgewidth=1.2, linestyle="none")
            ax.annotate(jlab, (jx, jy), textcoords="offset points",
                        xytext=(0, -13), color="white", fontsize=5.8,
                        ha="center")
        # The six 12-EDO voicings: M for major, m for minor, one per inversion.
        for cx, cy, lab in region["chords"]:
            ax.plot(cx, cy, marker="o", markerfacecolor="none",
                    markeredgecolor="white", markersize=6,
                    markeredgewidth=1.2, linestyle="none")
            ax.annotate(lab, (cx, cy), textcoords="offset points",
                        xytext=(6, 3), color="white", fontsize=7.0,
                        fontweight="bold")
        ax.set_xlabel("Interval 1 (cents)")
        ax.set_title(region["title"], fontsize=8.5, loc="left")
    axes[0].set_ylabel("Interval 2 (cents)")

    cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cb.set_label("Tensor harmonicity (log scale)", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    fig.savefig(os.path.join(OUT, "th_zoom.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Report the density at each voicing, for reference.
    for region, (xs, ys, Z) in zip(REGIONS, grids):
        print(region["title"])
        for cx, cy, lab in region["chords"]:
            v = Z[np.argmin(abs(xs - cx)), np.argmin(abs(ys - cy))]
            print(f"   {lab} ({cx:.0f}, {cy:.0f}): {v:.4g}")
    print("wrote th_zoom.pdf")


if __name__ == "__main__":
    main()
