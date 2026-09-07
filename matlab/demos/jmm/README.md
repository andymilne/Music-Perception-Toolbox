# The JMM article's analyses as demos

Each script here reproduces one worked analysis of *Music Perception
Toolbox* (Journal of Mathematics and Music) or its Online Supplement,
lightly edited from the article's own analysis scripts so that its
header comment says what question the analysis asks, how the toolbox
answers it, and which functions do the work. Each script adds its own
folder and the toolbox root to the path, so it can be run from anywhere,
for example

    run('demos/jmm/demo_jmm_1_1_entropy.m')

Figures are written to `figures/` (created if absent) with `print`, so
the scripts also run headless; the numbers print to the console. The
scripts are the twins of the Python demos in `python/demos/jmm/`, with
the same computations, the same printed numbers, and the same figures.

## Data

The `+jmm` package supplies the three works as the article encoded them.

* **Bach, BWV 347** (`data/bwv347.musicxml`): the chorale from the
  music21 corpus (Riemenschneider 2) with the bars 1–4 repeat expanded,
  read with `readScore` (`jmm.bwv347Notes`) and sampled on the
  sixteenth-note grid (`jmm.bwv347Grid`; `jmm.bwv347Bar` gives the
  played-through bar of a grid time). The encoding is note-for-note the
  one the article used from music21. `jmm.bwv347FermataSpans` gives the
  spans the cadence analysis weights, from the note table's `fermata`
  column.
* **Reich, *Piano Phase*** (`jmm.pianoPhase`): both voices rendered from
  the article's constants (the twelve-note cell, base inter-onset
  interval, peak tempo deviation, smoothstep accelerandi), returned once
  as a struct with the cell, the two rendered voices, the pooled piece,
  the shift centres, and the phase function `lagAt`.
* **Coltrane, *Acknowledgement*** (Theme 2): the melody is read from a
  MIDI transcription with `readScore`. The transcription is not
  distributed; the analyses that need it are not yet ported (see below).

## Scripts

| Demo | Article | Question | Toolbox functions |
|:--|:--|:--|:--|
| `demo_jmm_1_1_entropy.m` | Analysis 1.1 | Where is the chorale's spectral pitch content most and least concentrated, per event and under a smooth window? | `addSpectra`, `windowedEntropy` (`'method', 'differential'`) |
| `demo_jmm_1_2_similarity.m` | Analysis 1.2 | When do two chords count as alike? Six chord pairs under voice-aware, simplex-voice, and voice-agnostic encodings across the pitch–pitch-class blend (`HEATMAPS = true` adds the N × N event-pair matrices). Helpers: `jmm.buildVoiceAware`, `jmm.buildSimplexVoice`, `jmm.buildVoiceAgnostic`. | `buildExpTens`, `cosSimExpTens` (density lists, elementwise and broadcast modes), `simplexVertices` |
| `demo_jmm_1_3_tonic_tuple_size.m` | Analysis 1.3 | How does raising the tuple size sharpen chord matching, across absolute/relative and periodic/non-periodic readings? | `buildExpTens`, `cosSimExpTens` |
| `demo_jmm_1_4_cadence_nesting.m` | Analysis 1.4 | Where do cadences of each type occur, in any key? Nested two- and three-chord prototypes (chords unordered within an ordered, outer-relative succession) swept across the beat aggregates of the chorale, with pitch-derived inversion flags. Helpers: `jmm.bwvWindowState`, `jmm.winEvents`, `jmm.aggregate`, `jmm.boundDensity`, `jmm.buildPair`, `jmm.dyadQuery`, `jmm.queryDensity`, `jmm.sonAt`, `jmm.isRootPosition`, `jmm.isSixFour`, `jmm.b2bar`, `jmm.prototypeSweep`, `jmm.dyadSweep` (the twins of `bwv_window.py`). | `bindEvents`, `flatSpecs`, `buildExpTens`, `cosSimExpTens` (`'normalize', 'oneSidedDenom'`) |
| `demo_jmm_3_1_diff.m` | Analysis 3.1 | Does joint differencing of pitch and time expose the accelerandi of the phasing voice at the timing JND? | `differenceEvents`, `buildExpTens`, `evalExpTens`, windowed Rényi-2 entropy |
| `demo_jmm_3_2_texture.m` | Analysis 3.2 | How does the pooled texture's local entropy track the phase, at a fusing and a resolving time kernel? | `windowedEntropy` (`'method', 'renyi2'`) |
| `demo_jmm_3_3_xcorr.m` | Analysis 3.3 | Can the running phase between the pianos be read as the ridge of a lag cross-correlogram? | `windowedSimilarity` (one-sided matched filter) |

The Python demo's `cos_sim_exp_tens(..., mode='cartesian')` (the full
N × N matrix in one call) has no MATLAB counterpart; the MATLAB scripts
build the same matrices one row at a time with the struct-versus-cell
broadcast form of `cosSimExpTens`, which gives the same numbers.

Still to be added from the article's scripts: the *Acknowledgement*
analyses 2.1–2.3 (motif recovery, joint pitch-and-time windowing,
spectral augmentation), which need the MIDI transcription described
above. The corpus study of the Online Supplement (all 4/4 four-part
chorales, mixed-effects models) depends on the music21 corpus and
statsmodels and is not a demo.
