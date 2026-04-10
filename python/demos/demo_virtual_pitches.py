"""demo_virtual_pitches.py

Computes and plots virtual pitch (fundamental) salience profiles
for example chords.

Each subplot shows the normalised cross-correlation between the
chord's composite spectrum and a harmonic template, plotted against
pitch. Peaks indicate strong virtual pitches — candidate
fundamentals that are well-supported by the chord's spectral content.

The example chords include 12-TET triads and their just-intonation
counterparts, illustrating how mistuning broadens and reduces virtual
pitch peaks.

Port of demo_virtualPitches.m from the MATLAB Music Perception Toolbox v2.

Requires: matplotlib (pip install matplotlib)
"""

import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import mpt

# ===================================================================
#  User-adjustable parameters
# ===================================================================

# Smoothing width in cents
sigma = 12

# Spectral parameters for the harmonic template
spec = ['harmonic', 36, 'powerlaw', 1]

# Spectral parameters for chord tones (set to None for raw peaks)
chord_spec = ['harmonic', 36, 'powerlaw', 1]

# Grid resolution in cents
resolution = 1

# Chords as MIDI pitches: (pitches, label)
chord_data = [
    ([57, 64, 73],            'Open A major'),
    ([60, 67, 75],            'Open C minor'),
    ([61, 64, 69],            'A major first inversion'),
    ([63, 67, 72],            'C minor first inversion'),
    ([64, 69, 73],            'A major second inversion'),
    ([67, 72, 75],            'C minor second inversion'),
    ([57, 64.02, 72.8631],    'Open just A major'),
    ([60, 67.02, 75.1564],    'Open just C minor'),
    ([60, 63, 66],            'Close C dim.'),
    ([60, 61, 62],            'Cluster (C-Db-D)'),
    ([60, 63.1564, 65.8251],  '5:6:7 dim (just)'),
    ([57, 61, 65],            'A augmented'),
]

# Display range in MIDI note numbers (None for automatic)
display_range = (24, 84)

# ===================================================================
#  Compute virtual pitch profiles
# ===================================================================

n_chords = len(chord_data)
results = []

for midi_pitches, label in chord_data:
    midi_sorted = sorted(midi_pitches)
    p = mpt.convert_pitch(midi_sorted, 'midi', 'cents')

    vp_p, vp_w = mpt.virtual_pitches(
        p, None, sigma,
        spectrum=spec,
        chord_spectrum=chord_spec,
        resolution=resolution,
    )

    results.append({
        'midi': midi_sorted,
        'label': label,
        'vp_p': vp_p,
        'vp_w': vp_w,
    })

# ===================================================================
#  Plot
# ===================================================================

n_cols = 2
n_rows = int(np.ceil(n_chords / n_cols))

fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(12, 2.5 * n_rows + 1))
plt.subplots_adjust(hspace=0.5, wspace=0.3)
axes = axes.ravel()

for i, res in enumerate(results):
    ax = axes[i]
    vp_midi = mpt.convert_pitch(res['vp_p'], 'cents', 'midi')
    vp_w = res['vp_w']

    ax.plot(vp_midi, vp_w, linewidth=0.8, color=(0.1, 0.3, 0.7))

    # Display range
    if display_range is not None:
        ax.set_xlim(*display_range)
    else:
        ax.set_xlim(res['midi'][0] - 24, res['midi'][-1] + 12)
    ax.set_ylim(0, 1.05 * np.max(vp_w))

    # Mark chord tones with dashed vertical lines
    for m in res['midi']:
        ax.axvline(m, linestyle='--', color=(0.7, 0.2, 0.2),
                   alpha=0.4, linewidth=0.6)

    ax.set_xlabel('Pitch (MIDI note number)')
    ax.set_ylabel('Salience')

    midi_str = ', '.join(f'{m:.4g}' for m in res['midi'])
    ax.set_title(f"{res['label']}  [{midi_str}]", fontsize=9)

    ax.set_xticks(np.arange(0, 128, 12))
    ax.grid(True, alpha=0.15)

# Hide unused subplots
for i in range(n_chords, len(axes)):
    axes[i].set_visible(False)

spec_str = ', '.join(str(x) for x in spec)
fig.suptitle(f'Virtual pitch profiles  (σ = {sigma},  {spec_str})',
             fontweight='bold')

plt.tight_layout()

# ===================================================================
#  Console summary
# ===================================================================

print("\n--- Strongest virtual pitch per chord ---")
print(f"{'Chord':<25s}  {'VP (cents)':>10s}  {'VP (MIDI)':>10s}  {'Salience':>10s}")
print('-' * 60)

for res in results:
    max_idx = np.argmax(res['vp_w'])
    best_cents = res['vp_p'][max_idx]
    best_midi = mpt.convert_pitch(best_cents, 'cents', 'midi')
    max_w = res['vp_w'][max_idx]

    print(f"{res['label']:<25s}  {best_cents:10.1f}  {best_midi:10.2f}  {max_w:10.3f}")

print()
plt.show()
