# The JMM article's analyses as demos

Each script here reproduces one worked analysis of *Music Perception
Toolbox* (Journal of Mathematics and Music) or its Online Supplement,
lightly edited from the article's own analysis scripts so that its
docstring says what question the analysis asks, how the toolbox answers
it, and which functions do the work. Run a script from this folder with
the package importable, for example

    PYTHONPATH=../.. python demo_jmm_1_1_entropy.py

Figures are shown when matplotlib is available; the numbers print
either way. Set `SAVE_FIGURES = True` at the top of a script to write
them to `figures/` beside it instead, which is what makes the scripts
useful headless.

## Data

`jmm_data.py` supplies the works and the derivations as the article
encoded them.

* **Bach, BWV 347** (`data/bwv347.musicxml`): the chorale from the
  music21 corpus (Riemenschneider 2) with the bars 1–4 repeat expanded,
  read with `mpt.read_score` and sampled on the sixteenth-note grid
  (`grid_attr_table`). The encoding is note-for-note the one the article
  used from music21. `bwv347_fermata_spans` gives the spans the cadence
  analysis weights, from the attribute table's `fermata` column.
* **Reich, *Piano Phase*** (`piano_phase.py`): both voices rendered from
  the article's constants (the twelve-note cell, base inter-onset
  interval, peak tempo deviation, smoothstep accelerandi).
* **Coltrane, *Acknowledgement*** (the solo, `acknowledgement`): the
  melody is read from `data/AwakeningSolo.mid` with `mpt.read_score`. The
  transcription is not distributed; place your own monophonic MIDI
  transcription of the solo at that path, or pass the path to
  `acknowledgement`.

* **Ren, Rammos, and Rohrmeier (2024): derivations** (`derivations`): the
  rule-labelled parses of the Jazz Harmony Treebank, read from
  `data/ParseTrees.json` as a table of path positions. The file is not
  distributed; download `ParseTrees.json` from the authors' repository
  and place it at that path, or pass the path to `derivations`.

## Scripts

Analyses are numbered by piece and then by order within it — 1.x Bach,
2.x Coltrane, 3.x Reich, 4.x the supplied parses — in one series
running across the article and the Online Supplement, so an analysis keeps its number wherever it is
printed. Each demo is named after its analysis; the table says which
document carries it.

| Demo | Analysis | Question | Toolbox functions |
|:--|:--|:--|:--|
| `demo_jmm_1_1_entropy.py` | Analysis 1.1 — article §4.1.1 | Where is the chorale's spectral pitch content most and least concentrated, per event and under a smooth window? | `add_spectra`, `windowed_entropy` (`method='differential'`) |
| `demo_jmm_1_2_similarity.py` | Analysis 1.2 — article §4.1.2 | When do two chords count as alike? Six chord pairs under voice-aware, simplex-voice, and voice-agnostic encodings across the pitch–pitch-class blend (`--heatmaps` adds the N × N event-pair matrices). | `build_maet`, `sim_maet` (density lists, `mode='pairwise'` / `'cartesian'`), `simplex_vertices` |
| `demo_jmm_1_3_cadence_nesting.py` | Analysis 1.3 — article §4.1.3, supplement §6 | Where do cadences of each type occur, in any key? Nested two- and three-chord prototypes (chords unordered within an ordered, outer-relative succession) swept across the beat aggregates of the chorale, with pitch-derived inversion flags. Helper: `bwv_window.py`. | `bind_events`, `flat_specs`, `build_maet`, `sim_maet` (`normalize='oneSidedDenom'`) |
| `demo_jmm_1_4_tonic_tuple_size.py` | Analysis 1.4 — supplement §7 | How does raising the tuple size sharpen chord matching, across absolute/relative and periodic/non-periodic readings? | `build_maet`, `sim_maet` |
| `demo_jmm_2_1_joint.py` | Analysis 2.1 — article §4.2.1 | Which four-note cell recurs most in *Acknowledgement*, as a joint object: which interval pattern and rhythm recur together? Rhythm marginalized and conditioned on the motif's intervals. | `difference_events`, `bind_events` (both attributes in step), `select_pre_maet`, `build_maet`, `eval_maet` |
| `demo_jmm_2_2_motif.py` | Analysis 2.2 — supplement §8.1 | The same cell from pitch alone: interval triples against relative pitch quadruples, ranked by the density at each cell. | `difference_events`, `bind_events`, `build_maet`, `eval_maet` |
| `demo_jmm_2_3_spectral.py` | Analysis 2.3 — supplement §8.2 | Where is the motif stated, and in which key? The motif as a query slid across the passage, under bare fundamentals and twelve harmonic partials. Runs for a few minutes. | `add_spectra` (pre-MAET form), `bind_events`, `windowed_similarity` (single-axis and multi-axis sweeps) |
| `demo_jmm_3_1_texture.py` | Analysis 3.1 — article §4.3.1 | How does the pooled texture's local entropy track the phase, at a fusing and a resolving time kernel? | `windowed_entropy` (`method='renyi2'`) |
| `demo_jmm_3_2_diff.py` | Analysis 3.2 — supplement §9 | Does joint differencing of pitch and time expose the accelerandi of the phasing voice at the timing JND? | `difference_events`, `build_maet`, `eval_maet`, windowed Rényi-2 entropy |
| `demo_jmm_3_3_xcorr.py` | Analysis 3.3 — supplement §10 | Can the running phase between the pianos be read as the ridge of a lag cross-correlogram? | `windowed_similarity` (one-sided matched filter) |
| `demo_jmm_4_1_parse.py` | Analysis 4.1 — supplement §11 | What can the framework do with an expert analysis it is given? A rule-labelled derivation carried as a nested attribute: retrieval of a configuration, partial match on the label simplex, reduction by graded weights, and depth as a further coordinate. | `simplex_vertices`, `pack_pre_maet`, `flat_specs`, `bind_attributes`, `bind_events` (`group_by`), `select_pre_maet`, `build_maet`, `sim_maet` (`normalize='oneSidedDenom'`) |

The corpus study of the Online Supplement (all 4/4 four-part
chorales, mixed-effects models) depends on the music21 corpus and
statsmodels and is not a demo.
