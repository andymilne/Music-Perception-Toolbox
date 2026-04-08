"""demo_audio_analysis.py

Extract spectral peaks from audio files and compute perceptual features.

Demonstrates the workflow for analysing real audio: peak extraction
via audio_peaks, then spectral similarity (SPCS), harmonicity,
spectral entropy, roughness, and virtual pitches — all without
spectral enrichment, since the extracted peaks already represent
the full spectrum.

Two peak-extraction passes are shown:
  1. No smoothing — appropriate for steady-state sounds (piano, oboe)
     but produces many spurious peaks for sounds with vibrato or
     frequency jitter (violin, complex music).
  2. With Gaussian smoothing (sigma_peaks cents) — collapses
     vibrato-spread energy into single peaks, giving a cleaner
     representation for all sources.

Perceptual features are computed from the smoothed peaks.

The audio files are in the audio/ subfolder (one level up from
demos/).

Requires: soundfile (pip install soundfile)
"""

import numpy as np
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import mpt

# ===================================================================
#  User-adjustable parameters
# ===================================================================

sigma = 12           # Gaussian smoothing width for perceptual measures
sigma_peaks = 12     # Gaussian smoothing width for peak extraction

# Audio files to analyse (relative to this script's parent directory)
audio_dir = Path(__file__).resolve().parent.parent / 'audio'

audio_files = [
    ('piano_C4.wav',                  'Piano C4'),
    ('violin_A4.wav',                 'Violin A4'),
    ('oboe_A4.wav',                   'Oboe A4'),
    ('Piano_Emin.wav',                'Piano E minor'),
    ('Piano_G7_3rd_inversion.wav',    'Piano G7 (3rd inv.)'),
    ('Piano_Cmin_open.wav',           'Piano C minor (open)'),
    ('music_sample.wav',              'Music sample'),
]

# ===================================================================
#  Pass 1: Extract peaks without smoothing
# ===================================================================

print("=== Pass 1: Peak extraction (no smoothing) ===\n")

raw_results = []
for filename, label in audio_files:
    filepath = audio_dir / filename
    if not filepath.is_file():
        print(f"  {label}: file not found ({filepath}), skipping.")
        continue

    f, w, _ = mpt.audio_peaks(str(filepath))

    raw_results.append({
        'label': label,
        'filename': filename,
        'n_peaks': len(f),
    })
    print()

# ===================================================================
#  Pass 2: Extract peaks with smoothing
# ===================================================================

print(f"=== Pass 2: Peak extraction (sigma = {sigma_peaks} cents) ===\n")

smooth_results = []
for filename, label in audio_files:
    filepath = audio_dir / filename
    if not filepath.is_file():
        continue

    f, w, _ = mpt.audio_peaks(str(filepath), sigma=sigma_peaks)
    p = mpt.convert_pitch(f, 'hz', 'cents')

    smooth_results.append({
        'label': label,
        'f': f,
        'w': w,
        'p': p,
    })
    print()

if len(smooth_results) == 0:
    print("No audio files found. Check the audio/ folder.")
    exit()

# ===================================================================
#  Peak count comparison
# ===================================================================

print("=== Peak count comparison ===\n")
print(f"{'Sound':<25s}  {'Unsmoothed':>10s}  {'Smoothed':>10s}")
print('-' * 50)

for raw, smooth in zip(raw_results, smooth_results):
    print(f"{raw['label']:<25s}  {raw['n_peaks']:10d}  {len(smooth['f']):10d}")

# ===================================================================
#  Single-file features (smoothed peaks, no spectral enrichment)
# ===================================================================

print(f"\n=== Single-file features (smoothed peaks, sigma = {sigma}) ===\n")
print(f"{'Sound':<25s}  {'Peaks':>5s}  {'specEnt':>8s}  {'hMax':>8s}  "
      f"{'hEnt':>8s}  {'Rough':>8s}")
print('-' * 72)

for res in smooth_results:
    f, w, p = res['f'], res['w'], res['p']

    H = mpt.spectral_entropy(p, w, sigma)
    hMax, hEnt = mpt.template_harmonicity(p, w, sigma)
    r = mpt.roughness(f, w)

    res['spec_ent'] = H
    res['h_max'] = hMax
    res['h_ent'] = hEnt
    res['rough'] = r

    print(f"{res['label']:<25s}  {len(f):5d}  {H:8.4f}  {hMax:8.4f}  "
          f"{hEnt:8.4f}  {r:8.4f}")

# ===================================================================
#  Pairwise spectral pitch class similarity
# ===================================================================

print(f"\n=== Pairwise SPCS (smoothed peaks, sigma = {sigma}) ===\n")

labels = [r['label'] for r in smooth_results]
col_w = 12
print(' ' * 26 + ''.join(f"{l[:col_w]:<{col_w}s}" for l in labels))

for i, res_i in enumerate(smooth_results):
    row_str = f"{res_i['label']:<25s} "
    for j, res_j in enumerate(smooth_results):
        if j < i:
            row_str += ' ' * col_w
        elif j == i:
            row_str += f"{'1.000':<{col_w}s}"
        else:
            s = mpt.cos_sim_exp_tens_raw(
                res_i['p'], res_i['w'],
                res_j['p'], res_j['w'],
                sigma, 1, False, True, 1200,
                verbose=False,
            )
            row_str += f"{s:<{col_w}.3f}"
    print(row_str)

# ===================================================================
#  Virtual pitches
# ===================================================================

print(f"\n=== Strongest virtual pitch per sound (smoothed peaks) ===\n")
print(f"{'Sound':<25s}  {'VP (cents)':>10s}  {'VP (MIDI)':>10s}  {'Salience':>10s}")
print('-' * 60)

for res in smooth_results:
    vp_p, vp_w = mpt.virtual_pitches(res['p'], res['w'], sigma)
    max_idx = np.argmax(vp_w)
    best_cents = vp_p[max_idx]
    best_midi = mpt.convert_pitch(best_cents, 'cents', 'midi')

    print(f"{res['label']:<25s}  {best_cents:10.1f}  {best_midi:10.2f}  "
          f"{vp_w[max_idx]:10.3f}")

print("\nDone.")
