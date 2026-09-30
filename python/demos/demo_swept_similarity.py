"""demo_swept_similarity.py

swept_similarity in depth: what each setting finds, and why.

One melody and one four-note query are used throughout. The melody holds
four statements related to the query in different ways, separated by
bars of filler:

    bar 1   (1) the query exactly               C D E G
    bar 3   (2) the query transposed a fifth    G A B D
    bar 5   (3) the query's pitches, reordered  E G C D
    bar 7   (4) the query's rhythm, other notes F A Bb F

All four share the query's rhythm (a crotchet then two quavers); the
filler bars are plain crotchets. Each section below asks the same
question -- where does the query occur? -- with a different setting, and
each setting finds a different subset of the four statements.

 1. The inputs, as pre-MAETs.
 2. Translation in time (the canonical call, sweep=a): finds (1).
 3. Translation in pitch and time: finds (1) and (2), each in its key.
 4. align='both': a local comparison; finds (1), scored against only
    the music around it.
 5. align='window', time dropped: pitch content without position; finds
    (1) and (3).
 6. A window on time, time dropped, and translation in pitch: which
    transposition each bar holds; finds (1), (2), and (3), and shows why
    cosine suits it.
 7. align='window', time relative (bound onsets): rhythm without
    position; finds (1) to (4).
 8. align='window', time absolute: two voices compared in place.
 9. align='independent': a correlogram of a drifting lag.
10. What the calls compute: sweep_sim_maet in one pass, against
    translated copies compared one by one.

See also
--------
mpt.swept_similarity
mpt.sweep_sim_maet
mpt.translate_attributes
mpt.bind_events
mpt.sim_maet

The MATLAB mirror is demo_sweptSimilarity.m.
"""

import time

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

import mpt

# Keep the dispatcher's per-call announcements out of the printed
# output (show_hints gates only those); restored at the end.
_prev_defaults = mpt.set_default(show_hints=False)


# =====================================================================
# 1. The inputs
# =====================================================================

print("=== 1. Inputs ===")

# The query: C D E G in a crotchet-quaver-quaver rhythm, written from
# beat 0.
QUERY_MIDI = [60, 62, 64, 67]
RHYTHM = [0.0, 1.0, 1.5, 2.0]


def statement(midi, t0):
    """Four notes in the query's rhythm from beat t0, and a closing F."""
    return [(m, t0 + r) for m, r in zip(midi, RHYTHM)] + [(65, t0 + 3.0)]


def filler(t0):
    """A bar of plain crotchets, F A F A."""
    return [(65, t0), (69, t0 + 1), (65, t0 + 2), (69, t0 + 3)]


notes = (filler(0) + statement(QUERY_MIDI, 4)              # (1) bar 1
         + filler(8) + statement([67, 69, 71, 74], 12)     # (2) bar 3
         + filler(16) + statement([64, 67, 60, 62], 20)    # (3) bar 5
         + filler(24) + statement([65, 69, 70, 65], 28))   # (4) bar 7
midi = np.array([m for m, _ in notes], dtype=float)
onset = np.array([t for _, t in notes], dtype=float)

# The melody is in 12-TET, so pitch is in semitones, as MIDI note
# numbers: periodic at the octave with sigma = 0.2 semitones. Onset (in
# beats) is absolute with sigma = 0.1 beats.
SIGMA = [0.2, 0.1]
p_ctx = [midi[None, :], onset[None, :]]
specs = mpt.flat_specs(p_ctx, name=['pitch', 'onset'], sigma=SIGMA,
                       is_per=[True, False], period=[12.0, 0.0])
melody = mpt.pack_pre_maet(p_ctx, None, specs)
query = mpt.pack_pre_maet([np.array(QUERY_MIDI, dtype=float)[None, :],
                           np.array(RHYTHM)[None, :]], None, specs)
mpt.show_pre_maet(query)
print(f"  melody: {midi.size} notes in 8 bars; the statements start at "
      f"beats 4, 12, 20, 28")
print()

BARS = np.arange(8)
STATEMENTS = {1: 4.0, 3: 12.0, 5: 20.0, 7: 28.0}   # bar -> start beat


# =====================================================================
# 2. Translation in time: the canonical call
# =====================================================================

print("=== 2. Translation in time (sweep=1) ===")

# Naming the attribute alone asks for the default sweep values: every
# offset at which query and melody overlap, from -2 to 31 beats. The
# default step is the largest whole fraction of the spacing the notes
# lie on (half a beat) that is no more than half the width of the
# profile's peaks, sigma * sqrt(2) / 2 at tuple size 1: 0.0625 beats, an
# eighth of that spacing, so every exact match falls on the grid. The
# query is translated by each offset and compared with the whole melody,
# in pitch and onset jointly, in one pass through sweep_sim_maet. The
# offsets come back as mu2, a dict keyed by attribute, here mu2[1]; both
# sequences are written from beat 0, so an offset is the beat at which
# the query starts.
S2, mu2 = mpt.swept_similarity(melody, query, sweep=1,
                               return_offsets=True)
print(f"  {mu2[1].size} offsets, {mu2[1][0]:.2f} to {mu2[1][-1]:.2f} beats")
# The matches are the profile's peaks (find_peaks) of at least 0.5.
for i in find_peaks(S2, height=0.5)[0]:
    print(f"  match at beat {mu2[1][i]:5.2f}: {S2[i]:.3f}")
print("  Only (1): pitch is compared as it stands, so (2) is at the wrong")
print("  pitch and (3) in the wrong order; (4) shares only the rhythm.")
print()


# =====================================================================
# 3. Translation in pitch and time
# =====================================================================

print("=== 3. Translation in pitch and time (sweep=[0, 1]) ===")

# Sweeping pitch as well translates the query to every transposition at
# every time offset: a (pitch, time) surface. Pitch being periodic, the
# transpositions cover one octave; the notes lie on a semitone grid, so
# the step is an eighth of a semitone, the largest whole fraction no
# more than half the width of the peaks in pitch.
#
# The query is written from C4 (MIDI 60), so adding its root to each
# pitch offset gives the pitch the root lands on, naming each match's key.
S3, mu3 = mpt.swept_similarity(melody, query, sweep=[0, 1],
                               return_offsets=True)
print(f"  surface {S3.shape[0]} transpositions x {S3.shape[1]} offsets")
best = S3.max(axis=0)
root = float(QUERY_MIDI[0])                        # C4, MIDI 60
names = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
for j in find_peaks(best, height=0.9)[0]:
    shift = mu3[0][np.argmax(S3[:, j])]
    key = names[int(round(root + shift)) % 12]
    print(f"  match at beat {mu3[1][j]:5.2f}, transposed {shift:2.0f} "
          f"semitones (root {key}): {best[j]:.3f}")
print("  (1) untransposed, in C, and (2) a fifth (7 semitones) up, in G.")
print()


# =====================================================================
# 4. align='both': a local comparison
# =====================================================================

print("=== 4. align='both': a local comparison ===")

# Under 'both' the query is translated as in section 2, and a window on
# the melody is aligned with it, so each placement compares the query
# with only the music around it. What this changes is the normalization.
# Cosine divides by the norms of both sides. Under 'query' the melody's
# norm is that of the whole melody, so even the exact statement scores
# only 1/3 (its four notes against the melody's 36, sqrt(4/36)), a value
# that falls as the piece grows. Under 'both' it is the norm of what the
# window keeps: a score of 1 means the window holds the query and nothing
# else, and unmatched notes nearby count against the match. The window's
# width sets how near is nearby.
#
# The window is aligned at the query's middle, the default of query_ref
# under 'both'. The offsets, mu4[1], are still the beats at which the query
# starts, so the peaks read as in section 2. query_ref, the point of the
# query at which the window is aligned, matters mainly for a window that
# extends to one side only: 'exponentialBefore' aligned at the query's
# last onset (query_ref={1: 2.0}) weights the music leading up to where
# the query ends, the most recent most heavily. Each value then depends
# only on music heard by the time the query ends, so it is easily read:
# against the sweep values, the moments at which the query ends (mu + 2),
# the profile says how closely what has just been heard matches the
# query, as a listener could judge it at that moment.
S4, mu4 = mpt.swept_similarity(melody, query, sweep=1,
                               normalize='cosine', return_offsets=True)
i = np.argmax(S4)
print(f"  'query', whole melody  : {S4[i]:.3f} at beat {mu4[1][i]:.2f}")
for width in (3, 4, 6, 8):
    S4, mu4 = mpt.swept_similarity(
        melody, query, sweep=1, align={1: 'both'},
        window={1: ('rect', float(width))}, normalize='cosine',
        return_offsets=True)
    i = np.argmax(S4)
    print(f"  'both', window {width} beats : {S4[i]:.3f} at beat "
          f"{mu4[1][i]:.2f}")
print("  A 3-beat window keeps the statement alone; 4 beats take in its")
print("  closing F, and 6 and 8 the filler either side.")
print()


# =====================================================================
# 5. align='window', time dropped: content without position
# =====================================================================

print("=== 5. align='window', onset dropped ===")

# The query is not translated. A window one bar wide (a half-open
# rectangle, so bars tile without overlap) steps through the melody, and
# onset is marginalized after the window has weighted the events: the
# query is compared with the pitches each bar contains, whatever their
# order or rhythm.
bar_centres = 4.0 * BARS + 2.0
S5 = mpt.swept_similarity(melody, query, sweep={1: bar_centres},
                          align={1: 'window'}, window={1: ('rect', 4.0)},
                          drop=[1])
print("  bar: " + "  ".join(f"{b:5d}" for b in BARS))
print("  sim: " + "  ".join(f"{s:5.2f}" for s in np.ravel(S5)))
print("  (1) and (3) hold the query's pitches; (2) shares two of them,")
print("  G and D, so it scores about half.")
print()


# =====================================================================
# 6. A window on time, translation in pitch: which transposition each
#    bar holds
# =====================================================================

print("=== 6. Window on onset (dropped), translation in pitch ===")

# The window can do what translation cannot: localize on an attribute
# that is not compared. Each bar is windowed on onset, onset is dropped,
# and the query is translated in pitch: every transposition in every bar,
# a (transposition, bar) surface. Within a bar only the pitches count, so
# this finds (1) and (3) untransposed and (2) at 7 semitones together.
# Nothing on the query side could do it: with onset dropped, the query
# has no position for a window to select.
#
# The normalization matters here. The default divides by the query's own
# self-overlap, so a bar that repeats two of a transposition's pitches
# scores as highly as one holding all four: every filler bar, F A F A,
# scores 1 at 5 semitones (F G A C), matching F and A twice each. Cosine
# divides by the windowed bar's own norm as well, so repetition no longer
# stands in for the missing pitches. That local denominator, the norm of
# what the window keeps, is also a reason to window the context on a
# translated attribute, where a window on the query would otherwise do
# much the same (sections 4 and 9).
transp = np.arange(12.0)
S6 = {}
for norm in ('oneSidedDenom', 'cosine'):
    S6[norm] = mpt.swept_similarity(
        melody, query, sweep={0: transp, 1: bar_centres},
        align={0: 'query', 1: 'window'}, window={1: ('rect', 4.0)},
        drop=[1], normalize=norm)
print(f"  surface {S6['cosine'].shape[0]} transpositions x "
      f"{S6['cosine'].shape[1]} bars; best transposition in each bar:")
print("  bar          : " + "  ".join(f"{b:5d}" for b in BARS))
for norm, name in (('oneSidedDenom', 'default     '),
                   ('cosine', 'cosine      ')):
    best = transp[np.argmax(S6[norm], axis=0)]
    print(f"  {name} : " + "  ".join(f"{c:5.0f}" for c in best))
    print("  " + " " * 13 + "  " + "  ".join(
        f"{v:5.2f}" for v in S6[norm].max(axis=0)))
print("  Under the default every bar scores 1 somewhere; under cosine the")
print("  statements (about 0.89; their closing F, absent from the query,")
print("  costs the rest) stand clear of the fillers (0.71).")
print()


# =====================================================================
# 7. align='window', time relative: rhythm without position
# =====================================================================

print("=== 7. align='window', onset relative (bound onsets) ===")

# Relative mode compares each tuple only up to a common translation:
# through its values relative to the lowest. For onsets, a tuple needs
# several of them, so consecutive onsets are bound into 4-note
# super-events (bind_events) with the outer level relative (rel_outer):
# each super-event is then compared by its onsets measured from its first,
# its rhythm. Pitch is left out here, to ask about rhythm alone.
t_specs = mpt.flat_specs([onset[None, :]], name=['onset'], sigma=[0.1],
                         is_per=[False], period=[0.0])
bound_mel = mpt.bind_events(mpt.pack_pre_maet([onset[None, :]], None,
                                              t_specs), 4, rel_outer=True)
bound_qry = mpt.bind_events(mpt.pack_pre_maet([np.array(RHYTHM)[None, :]],
                                              None, t_specs), 4,
                            rel_outer=True)
mpt.show_pre_maet(bound_qry)

# The window still weights each super-event by onset time, the value the
# pre-MAET holds (its four onsets reduced to one, their mean, the default
# of `locate`), before the density is built; the comparison then uses
# only the spacing. Translation would change nothing on a relative
# attribute, so 'window' is the role.
S7 = mpt.swept_similarity(bound_mel, bound_qry, sweep={0: bar_centres},
                          align={0: 'window'}, window={0: ('rect', 4.0)})
print("  bar: " + "  ".join(f"{b:5d}" for b in BARS))
print("  sim: " + "  ".join(f"{s:5.2f}" for s in np.ravel(S7)))
print("  Every statement has the query's rhythm, so all four are found;")
print("  the filler bars, in plain crotchets, are not.")
print()


# =====================================================================
# 8. align='window', time absolute: in place
# =====================================================================

print("=== 8. align='window', onset absolute: two voices in place ===")

# A second voice doubles the melody for four bars, then moves a major
# third above it. Comparing the two in place -- the second voice as the
# query, not translated, onset compared as it stands -- with windows that
# tile the piece shows where their similarity comes from. Under the
# default normalization the profile is linear in the window, so the
# bars' contributions sum to the similarity of the whole voices.
midi_b = midi.copy()
midi_b[onset >= 16] += 4
voice_b = mpt.pack_pre_maet([midi_b[None, :], onset[None, :]], None,
                            specs)
S8 = mpt.swept_similarity(melody, voice_b, sweep={1: bar_centres},
                          align={1: 'window'}, window={1: ('rect', 4.0)})
whole = float(mpt.sim_maet(melody, voice_b, normalize='oneSidedDenom',
                           verbose=False))
print("  bar: " + "  ".join(f"{b:5d}" for b in BARS))
print("  sim: " + "  ".join(f"{s:5.3f}" for s in np.ravel(S8)))
print(f"  sum over bars {np.sum(S8):.6f}; whole voices {whole:.6f}")
assert abs(np.sum(S8) - whole) < 1e-10
print()


# =====================================================================
# 9. align='independent': a correlogram
# =====================================================================

print("=== 9. align='independent': a drifting lag ===")

# Two parts play the same four-note ostinato, the second slightly faster,
# so it runs ever further ahead of the first (phasing). A window steps
# through the first part while the second is translated through a range
# of lags at each window position: every combination, a correlogram.
# query_ref 0 makes the query's sweep values the lags themselves. The
# window here is far narrower than the query, the whole second part, so
# each window position resolves the lag locally. Windowing the second
# part instead, around the same region, and translating it would give
# nearly the same correlogram: on a translated attribute, a window on the
# context and one on the query differ only in whether a near miss is
# weighted where the context's event lies or where the query's does.
OST_MIDI = [60, 64, 67, 71]
n_rep = 16
t_a = 0.5 * np.arange(4 * n_rep)
m_a = np.tile(np.array(OST_MIDI, dtype=float), n_rep)
t_b = t_a * 0.985                      # 1.5% faster
part_a = mpt.pack_pre_maet([m_a[None, :], t_a[None, :]], None, specs)
part_b = mpt.pack_pre_maet([m_a[None, :], t_b[None, :]], None, specs)
win_pos = np.arange(2.0, 30.01, 2.0)
lags = np.arange(-0.2, 0.8001, 0.01)
S9 = mpt.swept_similarity(part_a, part_b, sweep={1: (win_pos, lags)},
                          align={1: 'independent'},
                          window={1: ('rect', 4.0)}, query_ref={1: 0.0})
best_lag = lags[np.argmax(S9, axis=1)]
print(f"  correlogram {S9.shape[0]} window positions x {S9.shape[1]} lags")
print("  best lag rises with position, as the second part runs ahead:")
print("  window: " + " ".join(f"{w:5.1f}" for w in win_pos[::3]))
print("  lag   : " + " ".join(f"{b:5.2f}" for b in best_lag[::3]))
print(f"  (the drift is 1.5% of the time: {0.015 * win_pos[0]:.2f} to "
      f"{0.015 * win_pos[-1]:.2f} beats)")
print()


# =====================================================================
# 10. What the calls compute
# =====================================================================

print("=== 10. sweep_sim_maet in one pass, against copies one by one ===")

# Section 3 translated the query to every (pitch, time) pair. Translating
# every value of an attribute by the same amount changes the inner
# product only through that amount, so sweep_sim_maet computes all the
# offsets from the two densities in one pass, with no translated copy of
# the query built. The same profile, on a coarser grid, three ways; the
# second and third are given swept_similarity's default normalization,
# 'oneSidedDenom', explicitly, since theirs is 'cosine'.
p_grid = np.arange(12.0)
t_grid = np.arange(-2.0, 31.01, 0.5)
t0 = time.perf_counter()
S_ws = mpt.swept_similarity(melody, query, sweep={0: p_grid, 1: t_grid})
t_ws = time.perf_counter() - t0

PP, TT = np.meshgrid(p_grid, t_grid, indexing='ij')
t0 = time.perf_counter()
dens_mel = mpt.build_maet(melody, verbose=False)
dens_qry = mpt.build_maet(query, verbose=False)
S_sw = np.asarray(mpt.sweep_sim_maet(
    dens_mel, dens_qry, np.vstack([PP.ravel(), TT.ravel()]),
    normalize='oneSidedDenom', verbose=False)).reshape(PP.shape)
t_sw = time.perf_counter() - t0

t0 = time.perf_counter()
S_loop = np.empty(PP.size)
for m, (dp, dt) in enumerate(zip(PP.ravel(), TT.ravel())):
    q_m = mpt.translate_attributes(query, [dp, dt])
    S_loop[m] = mpt.sim_maet(dens_mel, mpt.build_maet(q_m, verbose=False),
                             normalize='oneSidedDenom', verbose=False)
S_loop = S_loop.reshape(PP.shape)
t_loop = time.perf_counter() - t0

print(f"  {PP.size} offsets")
print(f"  swept_similarity : {t_ws:7.3f} s")
print(f"  sweep_sim_maet      : {t_sw:7.3f} s   max diff "
      f"{np.max(np.abs(S_sw - S_ws)):.1e}")
print(f"  copy by copy        : {t_loop:7.3f} s   max diff "
      f"{np.max(np.abs(S_loop - S_ws)):.1e}")
assert np.max(np.abs(S_sw - S_ws)) < 1e-10
assert np.max(np.abs(S_loop - S_ws)) < 1e-8
print("  Where a window changes with each sweep value ('both', 'window'),")
print("  there is no shared context to reuse, and each sweep value is one")
print("  sim_maet call.")
print()


# =====================================================================
# Figures
# =====================================================================

print("=== Figures ===")
note_names = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
colours = {1: 'C1', 3: 'C2', 5: 'C3', 7: 'C4'}

# Figure 1: the melody, the statements coloured, and what sections 2, 3,
# 5, and 7 each find, on one time axis.
fig, axs = plt.subplots(5, 1, figsize=(10, 10), sharex=True,
                        gridspec_kw={'height_ratios': [1.4, 1, 1, 1, 1]})
ax = axs[0]
ax.scatter(onset, midi, s=40, facecolors='none', edgecolors='0.5')
for bar, t0_ in STATEMENTS.items():
    sel = (onset >= t0_) & (onset < t0_ + 2.5)
    ax.scatter(onset[sel], midi[sel], s=40, color=colours[bar])
    label = {1: '(1) exact', 3: '(2) transposed', 5: '(3) reordered',
             7: '(4) rhythm only'}[bar]
    ax.text(t0_, 77, label, color=colours[bar], fontsize=8)
for t_, m_ in zip(onset, midi):
    ax.annotate(note_names[int(m_) % 12], (t_, m_), xytext=(0, 6),
                textcoords='offset points', ha='center', fontsize=6)
ax.set_ylim(57, 80)
ax.set_ylabel('MIDI pitch')
ax.set_title('The melody: four statements related to the query C D E G')
axs[1].plot(mu2[1], np.ravel(S2))
axs[1].set_ylabel('2. time')
axs[2].plot(mu3[1], S3.max(axis=0))
axs[2].set_ylabel('3. pitch\n+ time')
axs[3].bar(bar_centres, np.ravel(S5), width=3.6, color='0.6')
axs[3].set_ylabel('5. content')
axs[4].bar(bar_centres, np.ravel(S7), width=3.6, color='0.6')
axs[4].set_ylabel('7. rhythm')
axs[4].set_xlabel('beat (offset of the query, or bar)')
for a in axs[1:]:
    for t0_ in STATEMENTS.values():
        a.axvline(t0_, color='0.85', linewidth=1, zorder=0)
fig.tight_layout()

# Figure 2: the (pitch, time) surface of section 3.
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(S3, origin='lower', aspect='auto',
               extent=[mu3[1][0], mu3[1][-1], mu3[0][0], mu3[0][-1]])
ax.set_xlabel('time offset (beats)')
ax.set_ylabel('transposition (semitones)')
ax.set_title('3. Translation in pitch and time: (1) at (4, 0), (2) at '
             '(12, 7)')
fig.colorbar(im, ax=ax)
fig.tight_layout()

# Figure 3: the (transposition, bar) surfaces of section 6, under the
# default normalization and under cosine.
fig, axs3 = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for a, (norm, name) in zip(axs3, (('oneSidedDenom', 'default'),
                                  ('cosine', 'cosine'))):
    im = a.imshow(S6[norm], origin='lower', aspect='auto', vmin=0, vmax=1,
                  extent=[-0.5, BARS[-1] + 0.5, -50, 1150])
    a.set_xlabel('bar')
    a.set_title(f'6. Transposition in each bar: {name}')
axs3[0].set_ylabel('transposition (semitones)')
fig.colorbar(im, ax=axs3)

# Figure 4: the two voices' similarity, bar by bar (section 8), and the
# correlogram of section 9.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.bar(BARS, np.ravel(S8), color='0.6')
ax1.set_xlabel('bar')
ax1.set_ylabel('contribution to the similarity')
ax1.set_title(f'8. In place: bars sum to {whole:.3f}')
im = ax2.imshow(S9.T, origin='lower', aspect='auto',
                extent=[win_pos[0], win_pos[-1], lags[0], lags[-1]])
ax2.plot(win_pos, best_lag, 'w.-')
ax2.set_xlabel('window position (beats)')
ax2.set_ylabel('lag of the second part (beats)')
ax2.set_title("9. align='independent': the best lag drifts")
fig.colorbar(im, ax=ax2)
fig.tight_layout()

mpt.set_default(**_prev_defaults)
print("  four figures drawn")
print("\n=== Demo complete ===")
plt.show()
