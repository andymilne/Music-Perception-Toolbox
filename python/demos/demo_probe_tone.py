"""demo_probe_tone.py

Probe-tone fit to a context: spectral pitch class similarity (SPCS)
profiles, with and without event weighting by recency and with harmonic
and inharmonic spectra.

In the probe-tone paradigm a listener hears a context and then a single
tone, the probe, and rates how well the probe fits. SPCS models the fit
as the cosine similarity of two spectrally enriched, absolute, periodic
monad (r = 1) expectation tensors: one of the context, one of the probe.
Sweeping the probe across the octave gives a probe-tone profile.

Sections
  1. The C-major scale as a context, with Krumhansl and Kessler's (1982)
     major-key probe-tone ratings (TISMIR article, Figure 4a).
  2. Porcupine[7] in 22-EDO (EDO: equal division of the octave), probed
     at the 22 pitch classes of 22-EDO (TISMIR article, Figure 4b).
  3. A time-ordered context, a melody that moves from C major to G
     major, carried as a pre-MAET with a pitch and a time attribute.
     Event weighting by an exponential recency profile over elapsed
     time models the fading of earlier events from memory; three
     spectra (harmonic, stretched, and stiff-string) show what
     inharmonicity does to the profile.
  4. Continuity: whether each probe continues the melody's final
     melodic direction.

Uses: transform_attributes, sim_maet (batched-raw, broadcast form, with
spectral enrichment via its spectrum keyword), pack_pre_maet,
weight_events, unpack_pre_maet, continuity, set_default.

Requires: matplotlib (pip install matplotlib)

The MATLAB mirror is demo_probeTone.m.
"""

import numpy as np
import matplotlib.pyplot as plt

import mpt

prev_defaults = mpt.set_default(show_hints=False)

# ===================================================================
#  User-adjustable parameters
# ===================================================================

# The TISMIR article's settings (Section 2.4): sigma = 10 cents, 16
# harmonics, power-law rolloff rho = 1. SPCS: r = 1, absolute, periodic
# at the octave.
sigma, r, is_rel, is_per, period = 10.0, 1, False, True, 1200.0
spec_harm = ['harmonic', 16, 'powerlaw', 1.0]

# Krumhansl and Kessler (1982) major-key probe-tone ratings, C to B.
kk_major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
names = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

# Section 3: two inharmonic spectra (User Guide, Section 7.3.6).
# Stretched: partial n at ratio n**beta, so beta = 1.02 puts the
# second partial at 1224 cents. Stiff string: ratio n*sqrt(1 + B n^2),
# with B = 5e-4 inside the range quoted for piano strings (about 1e-5
# to 1e-3).
spectra = {'harmonic':  spec_harm,
           'stretched': ['stretched', 16, 1.02, 'powerlaw', 1.0],
           'stiff':     ['stiff', 16, 5e-4, 'powerlaw', 1.0]}
recency_sd = 3.0            # recency profile's standard deviation (beats)

# One-cent probe grid for the profile curves. In Python a (K, 1) array
# is K one-element multisets, one probe per row; the context, a 1-D
# array, is broadcast against every row.
fine = np.arange(1200.0)[:, None]
chrom = np.arange(0.0, 1200.0, 100.0)[:, None]

# ===================================================================
#  1. C major against Krumhansl and Kessler
# ===================================================================

print("=== 1. C-major scale: SPCS against K&K major-key ratings ===")
c_major = np.array([0, 200, 400, 500, 700, 900, 1100], dtype=float)
spcs_c = mpt.sim_maet(c_major, None, fine, None, sigma, r, is_rel,
                      is_per, period, spectrum=spec_harm,
                      verbose=False)
spcs_c12 = spcs_c[::100]
# The squared correlation is the R^2 of the best affine map from the
# ratings to SPCS, the fit reported in the article. (The article's
# figure script turns kernel truncation off; at the default truncation
# the values agree to about 1e-11.)
r2_c = np.corrcoef(spcs_c12, kk_major)[0, 1] ** 2
slope, icpt = np.polyfit(kk_major, spcs_c12, 1)
for nm, v, k in zip(names, spcs_c12, kk_major):
    print(f"  {nm:3s} SPCS = {v:.3f}   K&K = {k:.2f}")
print(f"  R^2 = {r2_c:.3f}   (article: 0.63)")

# ===================================================================
#  2. Porcupine[7] in 22-EDO
# ===================================================================

print("\n=== 2. Porcupine[7] in 22-EDO, 22 probes ===")
step = 1200.0 / 22
porcupine = np.array([0, 4, 7, 10, 13, 16, 19]) * step   # steps 4333333
spcs_p = mpt.sim_maet(porcupine, None, fine, None, sigma, r, is_rel,
                      is_per, period, spectrum=spec_harm,
                      verbose=False)
probes22 = np.arange(22) * step
spcs_p22 = mpt.sim_maet(porcupine, None, probes22[:, None], None, sigma,
                        r, is_rel, is_per, period, spectrum=spec_harm,
                        verbose=False)
for k in np.argsort(spcs_p22)[::-1][:7]:
    print(f"  step {k:2d} ({probes22[k]:6.1f} cents): "
          f"SPCS = {spcs_p22[k]:.4f}")
# The seven best-fitting probes are the seven scale degrees, step 13
# (degree 5) first. The article reads degree 1 as a major tonic and
# degree 5 as a minor tonic.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(fine, spcs_c, lw=0.8, label='SPCS')
ax1.plot(chrom, kk_major * slope + icpt, 'o', color='C3',
         label='K&K (affinely rescaled)')
ax1.set_title(f'(a) C major, 12-EDO: R$^2$ = {r2_c:.2f}')
ax1.legend(loc='lower right')
ax2.plot(fine, spcs_p, lw=0.8)
ax2.stem(probes22, spcs_p22, linefmt='C1-', markerfmt='C1o', basefmt=' ')
ax2.set_title('(b) Porcupine[7], 22-EDO: the 22 probes')
for ax in (ax1, ax2):
    ax.set_xlabel('probe (cents)')
    ax.set_ylabel('SPCS')
fig.tight_layout()

# ===================================================================
#  3. A time-ordered context: recency and inharmonicity
# ===================================================================

# A melody of 13 events: a C-major phrase, then a phrase in G major
# that ends by rising E-F#-G. Onsets in beats; the final C of the
# first phrase is held for two beats.
midi = np.array([60, 64, 67, 65, 64, 62, 60, 71, 69, 67, 64, 66, 67])
onsets = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13], dtype=float)
pitch = mpt.transform_attributes(midi, None, ('midi', 'cents'))
pm = mpt.pack_pre_maet([pitch[None, :], onsets[None, :]])

# Event weighting: a factor from the time attribute (index 1) into the
# pitch attribute's weights (index 0), decaying exponentially before
# the last onset, so that an event's weight falls with its lag behind
# the end of the context. The time attribute is then dropped.
pm_rec = mpt.weight_events(pm, None, 1, 0, onsets[-1],
                           'exponentialBefore', sd=recency_sd,
                           drop_input_attr=True)
p_rec, w_rec, _ = mpt.unpack_pre_maet(pm_rec)
print("\n=== 3. Melody context: recency weights over elapsed time ===")
print("  " + " ".join(f"{w:.2f}" for w in w_rec[0].ravel()))

# Probes: the twelve chromatic pitch classes. Unweighted context
# (every event at weight 1) versus recency-weighted, under each
# spectrum. Each profile is also compared with the K&K major profile
# in C and rotated to G.
contexts = {'uniform': (pitch, None),
            'recency': (p_rec[0].ravel(), w_rec[0].ravel())}
prof = {}
print(f"\n  {'spectrum':10s} {'weights':8s}  R^2 K&K C  R^2 K&K G  SD")
for sname, spec in spectra.items():
    for cname, (cp, cw) in contexts.items():
        s = mpt.sim_maet(cp, cw, chrom, None, sigma, r, is_rel, is_per,
                         period, spectrum=spec, verbose=False)
        prof[sname, cname] = s
        r2c = np.corrcoef(s, kk_major)[0, 1] ** 2
        r2g = np.corrcoef(s, np.roll(kk_major, 7))[0, 1] ** 2
        print(f"  {sname:10s} {cname:8s}  {r2c:9.2f}  {r2g:9.2f}  "
              f"{s.std():.3f}")
for nm in ('F', 'F#'):
    i = names.index(nm)
    print(f"  harmonic, {nm}: uniform {prof['harmonic', 'uniform'][i]:.3f}"
          f", recency {prof['harmonic', 'recency'][i]:.3f}")
# Recency moves the harmonic profile from C major towards G major: F
# loses fit and F# gains it. The inharmonic spectra flatten the profile
# (smaller SD across probes) and lower its fit to K&K: their upper
# partials are mistuned from the 12-EDO pitch classes, so partials of
# the context and of in-key probes coincide less, and the spectral
# kinship that separates in-key from out-of-key probes weakens.

# ===================================================================
#  4. Continuity: does the probe continue the melody's direction?
# ===================================================================

# continuity returns, for each query (here each probe, from middle C
# upwards), the expected length of the backward run of same-direction
# intervals that the step from the melody's last note to the probe
# extends, and the run's signed size in cents, under the same 10-cent
# uncertainty. The melody ends by rising (E-F#-G), so only probes above
# G extend it.
count, mag = mpt.continuity(midi * 100.0, 6000.0 + chrom.ravel(), sigma)
print("\n=== 4. Continuity of each probe after the melody ===")
print("  (SPCS: harmonic spectrum, recency-weighted context)")
print(f"  {'probe':5s}  {'SPCS':>6s}  {'run':>4s}  {'cents':>6s}")
for nm, s, c, m in zip(names, prof['harmonic', 'recency'], count, mag):
    print(f"  {nm:5s}  {s:6.3f}  {c:4.1f}  {m:6.0f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
x = np.arange(12)
ax1.bar(x - 0.2, prof['harmonic', 'uniform'], 0.4, label='uniform')
ax1.bar(x + 0.2, prof['harmonic', 'recency'], 0.4, label='recency')
ax1.set_title('(a) Harmonic spectrum: event weighting')
for sname in spectra:
    ax2.plot(x, prof[sname, 'recency'], 'o-', label=sname)
ax2.set_title('(b) Recency-weighted: three spectra')
for ax in (ax1, ax2):
    ax.set_xticks(x, names)
    ax.set_xlabel('probe')
    ax.set_ylabel('SPCS')
    ax.legend()
fig.tight_layout()

mpt.set_default(**prev_defaults)
print("\nDone.")
plt.show()
