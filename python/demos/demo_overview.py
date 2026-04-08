"""demo_overview.py

Quick tour of the Music Perception Toolbox (mpt) for Python.
Sections follow the same order as the User Guide.

Uses the son clave rhythm [0, 3, 6, 10, 12] in a 16-step cycle and
the diatonic scale [0, 2, 4, 5, 7, 9, 11] in 12-EDO as running
examples.
"""

import numpy as np
import mpt

np.set_printoptions(precision=3, suppress=True)

# ===================================================================
#  1. Pitch / frequency conversion  (User Guide §6.6)
# ===================================================================

print("=== Pitch conversion ===")
print(f"  MIDI 60 = {mpt.convert_pitch(60, 'midi', 'hz'):.2f} Hz")
print(f"  440 Hz  = {mpt.convert_pitch(440, 'hz', 'cents'):.0f} cents")
print(f"  440 Hz  = {mpt.convert_pitch(440, 'hz', 'erb'):.2f} ERB-rate")

# ===================================================================
#  2. Spectral enrichment  (User Guide §6.2)
# ===================================================================

print("\n=== Spectral enrichment ===")
chord = np.array([0.0, 400.0, 700.0])
p, w = mpt.add_spectra(chord, None, "harmonic", 8, "powerlaw", 1)
print(f"  3 pitches × 8 harmonics = {len(p)} partials")
print(f"  First 8 offsets: {np.round(p[:8], 1)}")

# ===================================================================
#  3. Expectation tensors and SPCS  (User Guide §6.1, §3.3)
# ===================================================================

print("\n=== Spectral pitch class similarity ===")
scale_cents = [0, 200, 400, 500, 700, 900, 1100]
major = [0, 400, 700]
minor = [0, 300, 700]
dim_t = [0, 300, 600]

# Add harmonic spectra (24 partials, 1/n rolloff)
scale_p, scale_w = mpt.add_spectra(scale_cents, None, 'harmonic', 24, 'powerlaw', 1)

for name, chord_cents in [("Major", major), ("Minor", minor), ("Dim", dim_t)]:
    chord_p, chord_w = mpt.add_spectra(chord_cents, None, 'harmonic', 24, 'powerlaw', 1)
    s = mpt.cos_sim_exp_tens_raw(
        scale_p, scale_w, chord_p, chord_w,
        10, 1, False, True, 1200, verbose=False
    )
    print(f"  Diatonic vs {name:5s} triad: {s:.3f}")

# ===================================================================
#  4. Consonance and harmonicity  (User Guide §6.3)
# ===================================================================

print("\n=== Harmonicity and entropy (JI major triad) ===")
ji_triad = [0, 386.31, 701.96]
spec = ["harmonic", 24, "powerlaw", 1]

hMax, hEnt = mpt.template_harmonicity(ji_triad, None, 12, chord_spectrum=spec)
print(f"  Template harmonicity (hMax):     {hMax:.4f}")
print(f"  Template harmonicity (hEntropy): {hEnt:.4f}")

h = mpt.tensor_harmonicity(ji_triad, None, 12, spectrum=spec)
print(f"  Tensor harmonicity:              {h:.4f}")

H = mpt.spectral_entropy(ji_triad, None, 12, spectrum=spec)
print(f"  Spectral entropy:                {H:.4f}")

print("\n=== JI vs 12-EDO comparison ===")
edo_triad = [0, 400, 700]
for name, triad in [("JI", ji_triad), ("12-EDO", edo_triad)]:
    hMax, _ = mpt.template_harmonicity(triad, None, 12, chord_spectrum=spec)
    H = mpt.spectral_entropy(triad, None, 12, spectrum=spec)
    print(f"  {name:6s}  hMax={hMax:.4f}  specEntropy={H:.4f}")

print("\n=== Roughness ===")
p_cents = mpt.convert_pitch([60, 64, 67], "midi", "cents")
p_r, w_r = mpt.add_spectra(p_cents, None, "harmonic", 8, "powerlaw", 1)
f_hz = mpt.convert_pitch(p_r, "cents", "hz")
r = mpt.roughness(f_hz, w_r)
print(f"  C major triad (8 harmonics): roughness = {r:.4f}")

# ===================================================================
#  5. Balance and evenness  (User Guide §6.4)
# ===================================================================

print("\n=== Balance and evenness ===")

print("  Diatonic scale [0,2,4,5,7,9,11] in 12-EDO:")
diat = [0, 2, 4, 5, 7, 9, 11]
b = mpt.balance(diat, None, 12)
e = mpt.evenness(diat, 12)
print(f"    Balance:  {b:.3f}")
print(f"    Evenness: {e:.3f}")

print("  Son clave [0,3,6,10,12] in 16:")
clave = [0, 3, 6, 10, 12]
b = mpt.balance(clave, None, 16)
e = mpt.evenness(clave, 16)
print(f"    Balance:  {b:.3f}")
print(f"    Evenness: {e:.3f}")

# ===================================================================
#  6. Scale and rhythm structure  (User Guide §6.5)
# ===================================================================

print("\n=== Scale structure (diatonic) ===")
c, nc = mpt.coherence(diat, 12)
sq, nd = mpt.sameness(diat, 12)
print(f"  Coherence: {c:.3f} ({nc} failure)")
print(f"  Sameness:  {sq:.3f} ({nd} ambiguity)")

H1, _ = mpt.n_tuple_entropy(diat, 12, 1)
H2, _ = mpt.n_tuple_entropy(diat, 12, 2)
print(f"  1-tuple entropy: {H1:.3f}")
print(f"  2-tuple entropy: {H2:.3f}")

print("\n=== Rhythm structure (son clave) ===")
c, nc = mpt.coherence(clave, 16)
sq, nd = mpt.sameness(clave, 16)
print(f"  Coherence: {c:.3f} ({nc} failures)")
print(f"  Sameness:  {sq:.3f} ({nd} ambiguities)")

H1, _ = mpt.n_tuple_entropy(clave, 16, 1)
H2, _ = mpt.n_tuple_entropy(clave, 16, 2)
print(f"  1-tuple entropy: {H1:.3f}")
print(f"  2-tuple entropy: {H2:.3f}")

h = mpt.mean_offset(clave, None, 16)
print(f"  Mean offset: {np.round(h, 3)}")

edg, _ = mpt.edges(clave, None, 16)
print(f"  Edges: {np.round(edg, 3)}")

y = mpt.markov_s(clave, None, 16)
print(f"  Markov(3): {np.round(y, 3)}")

print("\nDone.")
