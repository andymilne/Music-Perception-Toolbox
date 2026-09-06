# The JMM article's analyses as demos

Each script here reproduces one worked analysis of *Music Perception
Toolbox* (Journal of Mathematics and Music) or its Online Supplement,
lightly edited from the article's own analysis scripts so that its
docstring says what question the analysis asks, how the toolbox answers
it, and which functions do the work. Run a script from this folder with
the package importable, for example

    PYTHONPATH=../.. python demo_jmm_1_1_entropy.py

Figures are written to `figures/` when matplotlib is available; the
numbers print either way.

## Data

`jmm_data.py` supplies the three works as the article encoded them.

* **Bach, BWV 347** (`data/bwv347.musicxml`): the chorale from the
  music21 corpus (Riemenschneider 2) with the bars 1–4 repeat expanded,
  read with `mpt.read_score` and sampled on the sixteenth-note grid
  (`bwv347_grid`). The encoding is note-for-note the one the article
  used from music21.
* **Reich, *Piano Phase*** (`piano_phase.py`): both voices rendered from
  the article's constants (the twelve-note cell, base inter-onset
  interval, peak tempo deviation, smoothstep accelerandi).
* **Coltrane, *Acknowledgement*** (Theme 2): the melody is read from
  `data/theme_2.mid` with `mpt.read_score`. The transcription is not
  distributed; place your own monophonic MIDI transcription at that path.

## Scripts

| Demo | Article | Question | Toolbox functions |
|:--|:--|:--|:--|
| `demo_jmm_1_1_entropy.py` | Analysis 1.1 | Where is the chorale's spectral pitch content most and least concentrated, per event and under a smooth window? | `add_spectra`, `windowed_entropy` (`method='differential'`) |
| `demo_jmm_1_3_tonic_tuple_size.py` | Analysis 1.3 | How does raising the tuple size sharpen chord matching, across absolute/relative and periodic/non-periodic readings? | `build_exp_tens`, `cos_sim_exp_tens` |
| `demo_jmm_3_1_diff.py` | Analysis 3.1 | Does joint differencing of pitch and time expose the accelerandi of the phasing voice at the timing JND? | `difference_events`, `build_exp_tens`, `eval_exp_tens`, windowed Rényi-2 entropy |
| `demo_jmm_3_2_texture.py` | Analysis 3.2 | How does the pooled texture's local entropy track the phase, at a fusing and a resolving time kernel? | `windowed_entropy` (`method='renyi2'`) |
| `demo_jmm_3_3_xcorr.py` | Analysis 3.3 | Can the running phase between the pianos be read as the ridge of a lag cross-correlogram? | `windowed_similarity` (one-sided matched filter) |

Still to be added from the article's scripts: Analysis 1.2 (voice-aware
versus voice-agnostic similarity under the helix blend, with the
event-pair heat maps), Analysis 1.4 (cadence localization with nested
three-chord queries and the metre-and-fermata weighting), and the
*Acknowledgement* analyses 2.1–2.3 (motif recovery, joint pitch-and-time
windowing, spectral augmentation). The corpus study of the Online
Supplement (all 4/4 four-part chorales, mixed-effects models) depends on
the music21 corpus and statsmodels and is not a demo.
