"""
Five consonance-related measures over a two-octave space of triads.
===================================================================

Draws each of the five measures over the same two-octave space of triads
{0, x1, x2}, so that their topographies can be compared directly. Every
measure is given the same spectrum and the same sigma, since the defaults
built into the individual functions differ from one another.

Parameters:

  * 16 harmonics per tone with power-law rolloff rho = 1, for every measure.
  * sigma = 12 cents, as used for spectral entropy and template harmonicity in
    Milne et al. (2023).
  * sigma * sqrt(2) for tensor harmonicity alone. Tensor harmonicity evaluates
    the template density at a single query point, so the uncertainty of the
    query itself is not otherwise represented. Convolving the density with a
    query kernel of width tau is exactly equivalent to evaluating a density of
    width sqrt(sigma^2 + tau^2) at the point, up to a constant factor; taking
    tau = sigma gives sigma * sqrt(2). Template harmonicity needs no such
    adjustment because cross-correlating two densities already integrates over
    both sources of uncertainty.

Roughness and spectral entropy are negated so that, in every panel, brighter
means more consonant.

Run:  python3 make_consonance.py [step]

This is slow: about an hour at the article's 10-cent step. Pass a larger step
for a quicker, coarser run -- see the note at STEP below.

The grid is computed on first run and cached beside the script; delete
consonance_grids.npz to force recomputation. The optional step argument
overrides STEP below.
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

# Use the toolbox on MPT_PYTHON if set, otherwise whatever `import mpt` finds
# (e.g. an installed copy). The two computation passes need different versions;
# see compute_part.
mpt = None


def load_mpt():
    """Import the toolbox, from MPT_PYTHON if set, otherwise from the path.

    Imported lazily: only the two computation passes need it, so the merge and
    draw step runs anywhere, with or without the toolbox installed.
    """
    global mpt
    if mpt is None:
        p = os.environ.get("MPT_PYTHON")
        if p:
            sys.path.insert(0, p)
        import mpt as _m
        mpt = _m
        # The published values were computed with an untruncated kernel;
        # version 3 truncates at six sigma by default, so the untruncated
        # setting is restored here.
        mpt.set_default(truncation_sigmas=float("inf"), show_hints=False)
    return mpt

HERE = os.path.dirname(os.path.abspath(__file__))
# Write the figure to MPT_FIG_OUT if set, else to a sibling "figures"
# directory if one exists, else beside the script.
OUT = os.environ.get("MPT_FIG_OUT") or (
    os.path.join(HERE, "..", "figures")
    if os.path.isdir(os.path.join(HERE, "..", "figures")) else HERE)


def cache_path():
    """The grid cache lives beside the script, not in the current directory."""
    return os.path.join(HERE, "consonance_grids.npz")

SPECTRUM = ["harmonic", 16, "powerlaw", 1.0]
SIGMA = 12.0
SIGMA_TENSOR = SIGMA * np.sqrt(2.0)
# SLOW: computing the grid takes on the order of an hour at STEP = 10, which
# is the resolution used in the article. Cost scales as the square of the
# grid, so STEP = 20 is about four times faster and STEP = 40 sixteen times,
# at correspondingly coarser resolution -- ample for checking that the script
# runs, or for exploring other parameters. The grid is cached in
# consonance_grids.npz, so re-runs only redraw; delete it to recompute. The
# step may also be given on the command line: python3 make_consonance.py 20
STEP = 10                  # chord-space grid resolution in cents
GRID_STEP = 1.0            # spectral-entropy evaluation grid, in cents
MAX_INT = 2400
BASE_HZ_OFFSET = 6000      # put the chord in a normal register for roughness

plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})

KEYS = ["tmpl_max", "tmpl_ent", "tensor", "spec_ent", "rough"]
# Each measure has its own dynamic range, so each panel gets its own gamma
# compression; without it, tensor harmonicity's concentration on a few narrow
# peaks leaves its panel almost entirely dark.
GAMMA = {"tmpl_max": 0.6, "tmpl_ent": 0.6, "tensor": 0.22,
         "spec_ent": 0.6, "rough": 0.6}
LABELS = ["(a) $h_{\\mathrm{Max}}$",
          "(b) $-h_{\\mathrm{Entropy}}$",
          "(c) Tensor harm.",
          "(d) $-$Spec. entropy",
          "(e) $-$Roughness"]


def normalised_entropy(chord):
    """Pielou-normalised Shannon spectral entropy of a chord.

    A grid-based entropy depends on the grid it is computed over, so the
    1-cent spacing is stated here rather than left to the default.
    """
    m = load_mpt()
    return float(m.spectral_entropy(chord, None, SIGMA, spectrum=SPECTRUM,
                                    method="normalized", base=2.0,
                                    resolution=GRID_STEP))


def tensor_grid(ints):
    """Tensor harmonicity over the whole grid in one pass.

    tensorHarmonicity builds the same harmonic-template density on every call,
    so evaluating a grid one chord at a time rebuilds it tens of thousands of
    times. Building it once and evaluating all query points together gives
    identical values far faster. The query point for the triad {0, i, j} is the
    ordered interval vector (i, j) when i <= j, so the grid is evaluated on the
    sorted coordinates and mirrored.
    """
    load_mpt()
    from mpt import add_spectra, build_maet, eval_maet
    p, w = add_spectra(np.zeros(3), np.ones(3), *SPECTRUM)
    T = build_maet(p, w, SIGMA_TENSOR, 3, True, False, 1200, verbose=False)
    n = len(ints)
    I, J = np.meshgrid(ints, ints, indexing="ij")
    lo, hi = np.minimum(I, J).ravel(), np.maximum(I, J).ravel()
    q = np.vstack([lo, hi]).astype(float)
    out = np.empty(q.shape[1])
    chunk = 400
    for k in range(0, q.shape[1], chunk):
        out[k:k + chunk] = eval_maet(T, q[:, k:k + chunk],
                                     normalize="none", verbose=False)
    return out.reshape(n, n)



def draw(ints, res):
    """Full page-width figure, two rows of three panels."""
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.7),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.12, "wspace": 0.14})
    axlist = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]

    for ax, key, lab in zip(axlist, KEYS, LABELS):
        # Each measure is on its own scale, so each panel is normalised to its
        # own range; only the topography is being compared, not magnitudes.
        Z = res[key]
        ax.imshow(Z, origin="lower", aspect="equal",
                  extent=[0, MAX_INT, 0, MAX_INT], cmap="inferno",
                  interpolation="bilinear",
                  norm=PowerNorm(GAMMA[key], vmin=np.nanmin(Z),
                                 vmax=np.nanmax(Z)))
        ax.text(0.05, 0.94, lab, transform=ax.transAxes, fontsize=7.5,
                fontweight="bold", va="top", color="white")
        # Ticks every 400 cents: musically meaningful spacings (major third,
        # fifth at 800, octave at 1200) rather than round decimal numbers.
        ax.set_xticks(np.arange(0, MAX_INT + 1, 400))
        ax.set_yticks(np.arange(0, MAX_INT + 1, 400))
        ax.tick_params(labelsize=6.2)
        # Ticks every 400 cents: musically meaningful landmarks (major third,
        # fifth at 700 is not on the grid but the octave at 1200 and double
        # octave at 2400 are).
        ax.set_xticks(np.arange(0, MAX_INT + 1, 400))
        ax.set_yticks(np.arange(0, MAX_INT + 1, 400))
        ax.tick_params(labelsize=6.2)

    for ax in axlist[:3]:
        ax.tick_params(labelbottom=False)
    axes[0, 2].tick_params(labelbottom=True)
    axes[0, 2].set_xlabel("Interval 1 (cents)", fontsize=8)
    for ax in (axes[1, 0], axes[1, 1]):
        ax.set_xlabel("Interval 1 (cents)", fontsize=8)
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_ylabel("Interval 2 (cents)", fontsize=8)
    axes[1, 2].set_visible(False)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "consonance.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def compute(step):
    """Evaluate all five measures over the upper triangle and mirror it.

    The measures are symmetric in the two intervals, since {0, a, b} and
    {0, b, a} are the same pitch multiset, so only i <= j is computed.
    """
    load_mpt()
    ints = np.arange(0, MAX_INT + step, step)
    n = len(ints)
    t0 = time.time()
    res = {"tensor": tensor_grid(ints)}
    for k in ("tmpl_max", "tmpl_ent", "rough", "spec_ent"):
        res[k] = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            chord = np.array([0.0, ints[i], ints[j]])
            # chord_spectrum adds partials to the chord. Without it the chord
            # is treated as a set of bare pitches, and the cross-correlation
            # compares a harmonic template against pure tones.
            h_max, h_ent = mpt.template_harmonicity(
                chord, None, SIGMA, spectrum=SPECTRUM,
                chord_spectrum=SPECTRUM)
            res["tmpl_max"][i, j] = res["tmpl_max"][j, i] = h_max
            res["tmpl_ent"][i, j] = res["tmpl_ent"][j, i] = -h_ent
            p_s, w_s = mpt.add_spectra(chord, None, *SPECTRUM)
            p_hz = mpt.transform_attributes(p_s + BASE_HZ_OFFSET, None,
                                            ("cents", "hz"))
            res["rough"][i, j] = res["rough"][j, i] = -mpt.roughness(p_hz, w_s)
            res["spec_ent"][i, j] = res["spec_ent"][j, i] = -normalised_entropy(chord)
        if i % 40 == 0:
            print(f"  row {i}/{n} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(cache_path(), ints=ints, **res)
    print(f"wrote {cache_path()}")


def main():
    step = int(sys.argv[1]) if len(sys.argv) > 1 else STEP
    if not os.path.exists(cache_path()):
        compute(step)
    z = np.load(cache_path())
    draw(z["ints"], {k: z[k] for k in KEYS})


if __name__ == "__main__":
    main()
