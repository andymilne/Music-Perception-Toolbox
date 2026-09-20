"""
Worked example for the Music Perception Toolbox article.
=========================================================

Constructs a four-part I-IV-V7 context in C major and a set of chordal
probes, then computes two families of predictor for every probe:

  * EXTRINSIC FIT       - how well the probe matches the context.
                          (S)P(C)S: the four cells of the pitch-similarity
                          family, i.e. cosine similarity of expectation
                          tensor densities, with and without spectral
                          enrichment, on periodic and non-periodic domains.

  * INTRINSIC CONSONANCE - how consonant the probe is in itself, without
                          reference to any context: tensor harmonicity,
                          template harmonicity (hMax and hEntropy),
                          spectral entropy, and sensory roughness.

The point of the example is that these two families are close to
independent: a probe can fit the context well and be dissonant, or fit
badly and be consonant. Demonstrating that requires measures from more
than one of the toolbox's feature groups.

All pitches are held in cents with middle C (C4, MIDI 60) as the origin,
so a MIDI note number m corresponds to (m - 60) * 100 cents.

Run:  python3 worked_example.py
"""

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("MPT_PYTHON", "../../tb/Music-Perception-Toolbox/python"))
import mpt  # noqa: E402

# The published values were computed with an untruncated kernel;
# version 3 truncates at six sigma by default, so the untruncated
# setting is restored here.
mpt.set_default(truncation_sigmas=float("inf"), show_hints=False)


# ----------------------------------------------------------------------
# Model parameters
# ----------------------------------------------------------------------
# Suitable values where nothing is
# known about the timbre of the stimuli: a harmonic spectrum with
# power-law rolloff rho = 1, and a smoothing width of roughly a tenth of
# a semitone.
N_HARMONICS = 16
RHO = 1.0
SIGMA = 10.0          # cents; the (S)P(C)S family
SIGMA_CONS = 12.0     # cents; the consonance family, after Milne et al. (2023)
SIGMA_TENSOR = SIGMA_CONS * np.sqrt(2.0)   # tensor harmonicity: see below
SPECTRUM = ["harmonic", N_HARMONICS, "powerlaw", RHO]


def midi_to_cents(m):
    """MIDI note number -> cents, with middle C (MIDI 60) at 0 cents."""
    return (np.asarray(m, dtype=float) - 60.0) * 100.0


# Cents here are referred to middle C, whereas the cents scale refers them to
# MIDI 0, so 6000 cents (sixty semitones) is added before conversion.
C4_OFFSET_CENTS = 6000.0


def cents_to_hz(c):
    """Cents (middle C at 0) -> frequency in Hz."""
    return mpt.transform_attributes(
        np.asarray(c, dtype=float) + C4_OFFSET_CENTS, None, ("cents", "hz"))


# ----------------------------------------------------------------------
# The context: a four-part I-IV-V7 progression in C major
# ----------------------------------------------------------------------
# Written as MIDI note numbers, bass first, in SATB order B-T-A-S.
# The voice leading observes the usual restrictions: no parallel fifths
# or octaves between any pair of voices, no voice crossing, and the
# seventh of the V7 (F4) is prepared as a common tone from the IV chord.
CONTEXT_CHORDS = {
    "I":  [48, 55, 64, 72],   # C3  G3  E4  C5
    "IV": [41, 57, 65, 72],   # F2  A3  F4  C5
    "V7": [43, 62, 65, 71],   # G2  D4  F4  B4
}


def check_voice_leading(chord_a, chord_b):
    """Report parallel fifths/octaves and voice crossings between two chords.

    Both chords are given as ascending lists of MIDI numbers, one entry
    per voice, so voice v moves from chord_a[v] to chord_b[v]. A pair of
    voices moves in parallel fifths or octaves when the interval class
    between them is 7 or 0 semitones before and after the move, and both
    voices actually move.
    """
    problems = []
    n = len(chord_a)
    for v1, v2 in itertools.combinations(range(n), 2):
        before = abs(chord_a[v1] - chord_a[v2]) % 12
        after = abs(chord_b[v1] - chord_b[v2]) % 12
        moved = (chord_a[v1] != chord_b[v1]) and (chord_a[v2] != chord_b[v2])
        same_direction = np.sign(chord_b[v1] - chord_a[v1]) == np.sign(chord_b[v2] - chord_a[v2])
        if moved and same_direction and before == after and before in (0, 7):
            name = "octaves" if before == 0 else "fifths"
            problems.append(f"parallel {name} between voices {v1} and {v2}")
    for v in range(n - 1):
        if chord_b[v] > chord_b[v + 1]:
            problems.append(f"voice crossing between voices {v} and {v + 1}")
    return problems


# ----------------------------------------------------------------------
# The probes
# ----------------------------------------------------------------------
# Four roots, crossing degrees that are diatonic to C major with degrees
# that are not, and three qualities at each root. Every probe is voiced
# the same way: a close-position triad whose root lies in the octave
# above middle C, over a bass note more than an octave below it. Which
# chord tone is placed in the bass determines the inversion.
#
# Holding the upper voicing fixed and varying only the bass is what makes
# the inversion comparison controlled: any difference between the three
# inversions of a probe is attributable to the bass alone.
PROBE_ROOTS = {          # name -> pitch class relative to C
    "I": 0,              # diatonic: the tonic
    "VI": 9,            # diatonic: the submediant
    "#IV": 6,            # chromatic
    "bII": 1,            # chromatic
}
QUALITIES = {            # name -> intervals above the root, in semitones
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
}
INVERSIONS = {"root": 0, "1st": 1, "2nd": 2}


def build_probe(root_pc, quality, inversion):
    """Return a probe chord as MIDI note numbers, bass first.

    Each probe is voiced to lead as smoothly as possible from the V7 of the
    context, rather than by a fixed formula. The voicings below were found by
    exhaustive search, minimising total squared voice motion from the V7
    subject to conventional constraints: all four voices within standard SATB
    ranges (E2-C4, C3-G4, F3-D5, C4-G5); every chord tone present, with one
    doubled; no voice crossing or overlap; no parallel fifths or octaves; and
    upper-voice motion of a fifth or less, the bass being free to leap as
    basses conventionally are.

    Voicing for musical plausibility rather than by formula means that spacing
    varies from probe to probe, so spacing is not held constant across the set.
    The trade-off is deliberate: a fixed formula gives a cleaner comparison but
    produces transitions with large parallel leaps and voice overlaps.
    """
    name = f"{_DEGREE_NAME[root_pc]} {quality} ({inversion})"
    return list(VOICINGS[name])


# Precomputed by the search described in build_probe.
VOICINGS = {
    "I maj (root)": [
        48,
        64,
        67,
        72
    ],
    "I maj (1st)": [
        40,
        60,
        67,
        72
    ],
    "I maj (2nd)": [
        43,
        60,
        64,
        72
    ],
    "I min (root)": [
        48,
        63,
        67,
        72
    ],
    "I min (1st)": [
        51,
        63,
        67,
        72
    ],
    "I min (2nd)": [
        43,
        63,
        63,
        72
    ],
    "I dim (root)": [
        48,
        63,
        66,
        72
    ],
    "I dim (1st)": [
        51,
        63,
        66,
        72
    ],
    "I dim (2nd)": [
        42,
        63,
        66,
        72
    ],
    "VI maj (root)": [
        45,
        61,
        64,
        69
    ],
    "VI maj (1st)": [
        49,
        61,
        64,
        69
    ],
    "VI maj (2nd)": [
        40,
        61,
        64,
        69
    ],
    "VI min (root)": [
        45,
        60,
        64,
        72
    ],
    "VI min (1st)": [
        48,
        60,
        64,
        69
    ],
    "VI min (2nd)": [
        40,
        60,
        64,
        69
    ],
    "VI dim (root)": [
        45,
        63,
        63,
        72
    ],
    "VI dim (1st)": [
        48,
        63,
        63,
        69
    ],
    "VI dim (2nd)": [
        51,
        60,
        63,
        69
    ],
    "#IV maj (root)": [
        42,
        58,
        66,
        73
    ],
    "#IV maj (1st)": [
        46,
        61,
        66,
        70
    ],
    "#IV maj (2nd)": [
        49,
        61,
        66,
        70
    ],
    "#IV min (root)": [
        42,
        57,
        66,
        73
    ],
    "#IV min (1st)": [
        45,
        61,
        66,
        69
    ],
    "#IV min (2nd)": [
        49,
        61,
        66,
        69
    ],
    "#IV dim (root)": [
        42,
        60,
        66,
        69
    ],
    "#IV dim (1st)": [
        45,
        60,
        66,
        72
    ],
    "#IV dim (2nd)": [
        48,
        60,
        66,
        69
    ],
    "bII maj (root)": [
        49,
        61,
        65,
        68
    ],
    "bII maj (1st)": [
        41,
        61,
        65,
        68
    ],
    "bII maj (2nd)": [
        44,
        61,
        65,
        73
    ],
    "bII min (root)": [
        49,
        61,
        64,
        68
    ],
    "bII min (1st)": [
        40,
        61,
        64,
        68
    ],
    "bII min (2nd)": [
        44,
        61,
        64,
        73
    ],
    "bII dim (root)": [
        49,
        64,
        67,
        73
    ],
    "bII dim (1st)": [
        40,
        61,
        67,
        73
    ],
    "bII dim (2nd)": [
        43,
        61,
        64,
        73
    ]
}

_DEGREE_NAME = {v: k for k, v in PROBE_ROOTS.items()}


def build_all_probes():
    probes = {}
    for rname, rpc in PROBE_ROOTS.items():
        for qname in QUALITIES:
            for iname in INVERSIONS:
                label = f"{rname} {qname} ({iname})"
                probes[label] = {
                    "midi": build_probe(rpc, qname, iname),
                    "degree": rname,
                    "quality": qname,
                    "inversion": iname,
                    "diatonic": rname in ("I", "VI"),
                }
    return probes


# ----------------------------------------------------------------------
# Predictors
# ----------------------------------------------------------------------
def fit_predictors(probe_cents, context_cents):
    """The four cells of the (S)P(C)S family.

    Each is a cosine similarity between absolute monad (r = 1) expectation
    tensor densities. The two binary choices are whether synthetic partials
    are added (spectral or not) and whether the domain wraps at the octave
    (class or not).
    """
    out = {}
    for spectral in (True, False):
        # Spectral enrichment replaces each notated pitch with its partials.
        # Without it, the multiset holds fundamentals only.
        if spectral:
            p_probe, w_probe = mpt.add_spectra(
                probe_cents, None, "harmonic", N_HARMONICS, "powerlaw", RHO)
            p_ctx, w_ctx = mpt.add_spectra(
                context_cents, None, "harmonic", N_HARMONICS, "powerlaw", RHO)
        else:
            p_probe, w_probe = probe_cents, None
            p_ctx, w_ctx = context_cents, None

        for periodic in (True, False):
            name = ("SPCS" if periodic else "SPS") if spectral else ("PCS" if periodic else "PS")
            # r = 1 and is_rel = False give the absolute monad density:
            # a smoothed distribution over individual pitches (or pitch
            # classes, when the domain is periodic).
            d_probe = mpt.build_maet(p_probe, w_probe, SIGMA, 1, False,
                                     periodic, 1200.0, verbose=False)
            d_ctx = mpt.build_maet(p_ctx, w_ctx, SIGMA, 1, False,
                                   periodic, 1200.0, verbose=False)
            out[name] = mpt.sim_maet(d_probe, d_ctx, verbose=False)
    return out


def consonance_predictors(probe_cents):
    """Measures of the probe's consonance considered on its own.

    Roughness is register-dependent and so is computed on frequencies in
    Hz rather than on the pitch representation used by the other measures;
    `average=True` divides by the number of partial pairs so that values
    remain comparable across chords with different numbers of partials.
    """
    # Tensor harmonicity is a raw density value and spans many orders of
    # magnitude across voicings, so it is passed through a saturating
    # transform S = 1 - exp(-T / eta) that maps it into [0, 1). The
    # transform is monotone, so orderings — and every conclusion drawn
    # from them — are unaffected by the choice of eta.
    # Tensor harmonicity evaluates the template density at a single query
    # point, so it must carry the uncertainty of the query as well as of the
    # template. Folding the query in is equivalent to widening the template
    # kernel to sqrt(sigma^2 + tau^2); with tau = sigma that is sigma*sqrt(2).
    # Template harmonicity needs no such adjustment, since cross-correlating
    # two densities already integrates over both.
    th = mpt.tensor_harmonicity(probe_cents, None, SIGMA_TENSOR,
                                spectrum=SPECTRUM, verbose=False)
    h_max, h_entropy = mpt.template_harmonicity(probe_cents, None, SIGMA_CONS,
                                              spectrum=SPECTRUM, chord_spectrum=SPECTRUM)
    s_ent = mpt.spectral_entropy(probe_cents, None, SIGMA_CONS,
                                 spectrum=SPECTRUM, method="normalized")

    partials, weights = mpt.add_spectra(
        probe_cents, None, "harmonic", N_HARMONICS, "powerlaw", RHO
    )
    rough = mpt.roughness(cents_to_hz(partials), weights, average=True)

    return {
        "tensorHarm": th,
        "hMax": h_max,
        "hEntropy": h_entropy,
        "specEntropy": s_ent,
        "roughness": rough,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    # --- context ---
    names = list(CONTEXT_CHORDS)
    print("Context: four-part I-IV-V7 in C major (MIDI numbers, bass first)")
    for name in names:
        print(f"  {name:3s} {CONTEXT_CHORDS[name]}")
    print("\nVoice-leading check within the context:")
    for a, b in zip(names, names[1:]):
        problems = check_voice_leading(CONTEXT_CHORDS[a], CONTEXT_CHORDS[b])
        print(f"  {a} -> {b}: {'clean' if not problems else '; '.join(problems)}")

    context_midi = [m for chord in CONTEXT_CHORDS.values() for m in chord]
    context_cents = midi_to_cents(context_midi)

    # --- probes ---
    probes = build_all_probes()
    print(f"\n{len(probes)} probes. Voice-leading check for the V7 -> probe transition:")
    n_clean = 0
    for label, info in probes.items():
        problems = check_voice_leading(CONTEXT_CHORDS["V7"], info["midi"])
        info["vl_problems"] = problems
        n_clean += not problems
        if problems:
            print(f"  {label:22s} {'; '.join(problems)}")
    print(f"  {n_clean}/{len(probes)} transitions clean")

    # --- predictors ---
    rows = []
    for label, info in probes.items():
        cents = midi_to_cents(info["midi"])
        row = {"label": label, "degree": info["degree"], "quality": info["quality"],
               "inversion": info["inversion"], "diatonic": info["diatonic"]}
        row.update(fit_predictors(cents, context_cents))
        row.update(consonance_predictors(cents))
        rows.append(row)
        print(".", end="", flush=True)
    print()

    with open("worked_example_results.json", "w") as f:
        json.dump(rows, f, indent=1, default=float)

    # --- report ---
    cols = ["SPCS", "SPS", "PCS", "PS", "tensorHarm", "hMax", "hEntropy",
            "specEntropy", "roughness"]
    header = f"{'probe':22s}" + "".join(f"{c:>12s}" for c in cols)
    print("\n" + header)
    for r in rows:
        print(f"{r['label']:22s}" + "".join(f"{r[c]:12.4g}" for c in cols))

    # Correlation structure across probes. If the two families are
    # measuring different things, the fit block and the consonance block
    # should correlate strongly within themselves and weakly across.
    M = np.array([[r[c] for c in cols] for r in rows])
    C = np.corrcoef(M.T)
    print("\nPearson correlations across the probe set:")
    print(f"{'':13s}" + "".join(f"{c:>12s}" for c in cols))
    for i, c in enumerate(cols):
        print(f"{c:13s}" + "".join(f"{C[i, j]:12.2f}" for j in range(len(cols))))

    from scipy.stats import spearmanr
    R, _ = spearmanr(M)
    print("\nSpearman correlations (rank-based, so invariant to any monotone")
    print("transform of any predictor, including the one applied above):")
    print(f"{'':13s}" + "".join(f"{c:>12s}" for c in cols))
    for i, c in enumerate(cols):
        print(f"{c:13s}" + "".join(f"{R[i, j]:12.2f}" for j in range(len(cols))))

    fit, cons = slice(0, 4), slice(4, 9)
    print(f"\nWithin the fit block:        Pearson |r| in "
          f"[{np.abs(C[fit, fit][np.triu_indices(4, 1)]).min():.2f}, "
          f"{np.abs(C[fit, fit][np.triu_indices(4, 1)]).max():.2f}]")
    print(f"Within the consonance block: Pearson |r| in "
          f"[{np.abs(C[cons, cons][np.triu_indices(5, 1)]).min():.2f}, "
          f"{np.abs(C[cons, cons][np.triu_indices(5, 1)]).max():.2f}]")
    print(f"Across the two blocks:       Pearson |r| <= "
          f"{np.abs(C[fit, cons]).max():.2f}, Spearman |rho| <= "
          f"{np.abs(R[fit, cons]).max():.2f}")

    np.save("worked_example_corr.npy", C)
    return rows, cols, C


if __name__ == "__main__":
    main()
