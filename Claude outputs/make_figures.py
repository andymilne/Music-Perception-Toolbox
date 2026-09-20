"""
The worked application: scatter and correlation matrix.
======================================================

Produces worked_example.pdf into ../figures/ : the fit-against-consonance
scatter for the 36 probes, and the correlation structure across all nine
predictors.

Run worked_example.py first; this script reads the JSON it writes.

Run:  python3 make_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")

# A restrained palette: one hue per feature group, plus a neutral grey.
COL_TENSOR = "#3B6EA5"
COL_CONS = "#B4553F"
COL_STRUCT = "#4E8367"
COL_SHARED = "#6E6E6E"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})


# ----------------------------------------------------------------------
# Section banner
# ----------------------------------------------------------------------






# ----------------------------------------------------------------------
# The scatter and the correlation matrix
# ----------------------------------------------------------------------
COLS = ["SPCS", "SPS", "PCS", "PS",
        "tensorHarm", "hMax", "hEntropy", "specEntropy", "roughness"]
LABELS = ["SPCS", "SPS", "PCS", "PS",
          "Tensor harm.", "$h_\\mathrm{Max}$", "$-h_\\mathrm{Entropy}$",
          "$-$Spec. entropy", "$-$Roughness"]
# Entropies and roughness fall as consonance rises, so they are negated here as
# they are elsewhere in the article; every predictor then points in the same
# direction and the sign of each correlation is directly interpretable.
SIGN = {"hEntropy": -1.0, "specEntropy": -1.0, "roughness": -1.0}
N_FIT = 4  # the first four predictors form the fit block


def make_results():
    with open(os.path.join(os.path.dirname(__file__),
                           "worked_example_results.json")) as f:
        rows = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.05),
                             gridspec_kw={"width_ratios": [1.0, 1.18]})

    # --- panel (a): the two families plotted against each other ---
    # Horizontal position is contextual fit (SPCS with the I-IV-V7
    # context); vertical position is intrinsic consonance, taken here as
    # negated spectral entropy so that up means more consonant on both
    # counts. If the two questions were the same question, the probes
    # would lie on a line.
    ax = axes[0]
    # Three visual channels, one per factor of the design: marker shape gives
    # chord quality, fill gives inversion, colour gives scale degree. Fill is
    # used for inversion because the nine horizontal bands (quality x
    # inversion) include pairs separated by only about 1e-4, too close to
    # label on the axis.
    marker = {"maj": "o", "min": "s", "dim": "^"}
    fill = {"root": "full", "1st": "left", "2nd": "none"}
    # Four widely separated hues: blue, green, red, and orange. The earlier
    # blue/purple pair was hard to tell apart at the open-marker sizes used for
    # second inversion.
    colour = {"I": "#1F5FA8", "VI": "#1B7F4B", "#IV": "#C0392B", "bII": "#E08A00"}

    # Join the three inversions of each chord as a closed triangle, so that the
    # effect of inversion can be traced. Consonance is transposition-invariant, so these lines have
    # the same vertical shape at every scale degree; only their horizontal
    # extent -- the spread of fit across inversions -- differs.
    order = ["root", "1st", "2nd"]
    for degree in colour:
        for quality in marker:
            pts = [r for inv in order for r in rows
                   if r["degree"] == degree and r["quality"] == quality
                   and r["inversion"] == inv]
            if len(pts) == 3:
                loop = pts + pts[:1]        # close the triangle
                ax.plot([p["SPCS"] for p in loop],
                        [-p["specEntropy"] for p in loop],
                        color=colour[degree], linewidth=0.7, alpha=0.35,
                        zorder=1)

    for r in rows:
        ax.plot(r["SPCS"], -r["specEntropy"], marker=marker[r["quality"]],
                fillstyle=fill[r["inversion"]], markersize=5.2,
                markerfacecolor=colour[r["degree"]],
                markeredgecolor=colour[r["degree"]], markeredgewidth=0.9,
                linestyle="none", zorder=3)
    ax.set_xlabel("Contextual fit (SPCS with the I\u2013IV\u2013V$^7$ context)")
    ax.set_ylabel("Intrinsic consonance ($-$spectral entropy)")
    ax.set_title("(a)", fontsize=8.5, loc="left")
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.set_axisbelow(True)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.62 * (hi - lo))

    q_handles = [plt.Line2D([], [], marker=m, linestyle="none", color="#555555",
                            markersize=4.8, label=q) for q, m in marker.items()]
    i_handles = [plt.Line2D([], [], marker="o", linestyle="none",
                            fillstyle=fl, markerfacecolor="#555555",
                            markeredgecolor="#555555", markersize=4.8,
                            label=inv) for inv, fl in fill.items()]
    d_handles = [plt.Line2D([], [], marker="o", linestyle="none", color=c,
                            markersize=4.8, label=d) for d, c in colour.items()]
    leg1 = ax.legend(handles=q_handles, loc="upper left", fontsize=6.2, ncol=1,
                     frameon=False, handletextpad=0.3, borderpad=0.2)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=i_handles, loc="upper center", fontsize=6.2, ncol=1,
                     frameon=False, handletextpad=0.3, borderpad=0.2)
    ax.add_artist(leg2)
    ax.legend(handles=d_handles, loc="upper right", fontsize=6.2, ncol=1,
              frameon=False, handletextpad=0.3, borderpad=0.2)

    # --- panel (b): the correlation structure ---
    # Spearman rather than Pearson: the predictors are on different scales and
    # the transform of a raw density that best reflects perception is unsettled,
    # so rank correlations keep the result invariant to any monotone rescaling.
    from scipy.stats import spearmanr
    M = np.array([[SIGN.get(c, 1.0) * r[c] for c in COLS] for r in rows])
    C, _ = spearmanr(M)
    ax = axes[1]
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(COLS)))
    ax.set_yticks(range(len(COLS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=6.4)
    ax.set_yticklabels(LABELS, fontsize=6.4)
    for i in range(len(COLS)):
        for j in range(len(COLS)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                    fontsize=5.1,
                    color="white" if abs(C[i, j]) > 0.6 else "#333333")
    # Rule off the two blocks so the weak off-diagonal region is visible.
    for pos in (N_FIT - 0.5,):
        ax.axhline(pos, color="black", linewidth=1.0)
        ax.axvline(pos, color="black", linewidth=1.0)
    ax.set_title("(b)", fontsize=8.5, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.043, pad=0.03).ax.tick_params(labelsize=6)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "worked_example.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote worked_example.pdf")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    make_results()
