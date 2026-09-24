# The JMM article's analyses as demos

Each script here reproduces one worked analysis of *Music Perception
Toolbox* (Journal of Mathematics and Music) or its Online Supplement,
lightly edited from the article's own analysis scripts so that its
header comment says what question the analysis asks, how the toolbox
answers it, and which functions do the work. Each script locates this
folder from the toolbox root and adds it to the path, so it can be run
from anywhere once the toolbox's `matlab` folder is on the path, for
example

    run('demos/jmm/demo_jmm_1_1_entropy.m')

The figures stay on screen and the numbers print to the console. Set
`SAVE_FIGURES = true` at the top of a script to write them to `figures/`
instead (created if absent), which is also what makes the scripts useful
headless. The
scripts are the twins of the Python demos in `python/demos/jmm/`, with
the same computations, the same printed numbers, and the same figures.

## Data

The `+jmm` package supplies the works and the derivations as the
article encoded them.

* **Bach, BWV 347** (`data/bwv347.musicxml`): the chorale from the
  music21 corpus (Riemenschneider 2) with the bars 1–4 repeat expanded,
  read with `readScore` (`jmm.bwv347Notes`) and sampled on the
  sixteenth-note grid (`gridAttrTable`; `jmm.bwv347Bar` gives the
  played-through bar of a grid time). The encoding is note-for-note the
  one the article used from music21. `jmm.bwv347FermataSpans` gives the
  spans the cadence analysis weights, from the attribute table's `fermata`
  column.
* **Reich, *Piano Phase*** (`jmm.pianoPhase`): both voices rendered from
  the article's constants (the twelve-note cell, base inter-onset
  interval, peak tempo deviation, smoothstep accelerandi), returned once
  as a struct with the cell, the two rendered voices, the pooled piece,
  the shift centres, and the phase function `lagAt`.
* **Coltrane, *Acknowledgement*** (the solo, `jmm.acknowledgement`): the
  melody is read from `data/AwakeningSolo.mid` with `readScore`. The
  transcription is not distributed; place your own monophonic MIDI
  transcription of the solo at that path, or pass the path to
  `jmm.acknowledgement`.

* **Ren, Rammos, and Rohrmeier (2024): derivations** (`jmm.derivations`): the
  rule-labelled parses of the Jazz Harmony Treebank, read from
  `data/ParseTrees.json` as a table of path positions. The file is not
  distributed; download `ParseTrees.json` from the authors' repository
  and place it at that path, or pass the path to `jmm.derivations`.

## Scripts

Analyses are numbered by piece and then by order within it — 1.x Bach,
2.x Coltrane, 3.x Reich, 4.x the supplied parses — in one series
running across the article and the Online Supplement, so an analysis keeps its number wherever it is
printed. Each demo is named after its analysis; the table says which
document carries it.

| Demo | Analysis | Question | Toolbox functions |
|:--|:--|:--|:--|
| `demo_jmm_1_1_entropy.m` | Analysis 1.1 — article §4.1.1 | Where is the chorale's spectral pitch content most and least concentrated, per event and under a smooth window? | `addSpectra`, `windowedEntropy` (`'method', 'differential'`) |
| `demo_jmm_1_2_similarity.m` | Analysis 1.2 — article §4.1.2 | When do two chords count as alike? Six chord pairs under voice-aware, simplex-voice, and voice-agnostic encodings across the pitch–pitch-class blend (`HEATMAPS = true` adds the N × N event-pair matrices). Each encoding is one conversion, differing only in `'roles'` and `'chords'`. | `gridAttrTable`, `preMaetFromAttrTable`, `selectPreMaet`, `buildMaet`, `simMaet` (density lists, elementwise and broadcast modes) |
| `demo_jmm_1_3_cadence_nesting.m` | Analysis 1.3 — article §4.1.3, supplement §6 | Where do cadences of each type occur, in any key? Nested two- and three-chord prototypes (chords unordered within an ordered, outer-relative succession) swept across the beat aggregates of the chorale, with pitch-derived inversion flags. Helpers: `jmm.bwvWindowState`, `jmm.boundContext`, `jmm.query`, `jmm.windowStarts`, `jmm.asCompared`, `jmm.dyadQuery`, `jmm.prototypeQuery`, `jmm.sonAt`, `jmm.isRootPosition`, `jmm.isSixFour`, `jmm.b2bar`; the two sweeps are local functions of the demo. | `gridAttrTable` (twice, the second regridding the first), `preMaetFromAttrTable`, `bindEvents` (per-attribute orders), `windowedSimilarity` (`'normalize', 'oneSidedDenom'`), `flatSpecs`, `selectPreMaet` |
| `demo_jmm_1_4_tonic_tuple_size.m` | Analysis 1.4 — supplement §7 | How does raising the tuple size sharpen chord matching, across absolute/relative and periodic/non-periodic readings? | `buildMaet`, `simMaet` |
| `demo_jmm_2_1_joint.m` | Analysis 2.1 — article §4.2.1 | Which four-note cell recurs most in *Acknowledgement*, as a joint object: which interval pattern and rhythm recur together? Rhythm marginalized and conditioned on the motif's intervals. | `preMaetFromAttrTable`, `differenceEvents`, `bindEvents` (both attributes in step), `selectPreMaet`, `buildMaet`, `evalMaet` |
| `demo_jmm_2_2_motif.m` | Analysis 2.2 — supplement §8.1 | The same cell from pitch alone: interval triples against relative pitch quadruples, ranked by the density at each cell. | `preMaetFromAttrTable`, `differenceEvents`, `bindEvents`, `buildMaet`, `evalMaet` |
| `demo_jmm_2_3_spectral.m` | Analysis 2.3 — supplement §8.2 | Where is the motif stated, and in which key? The motif as a query slid across the passage, under bare fundamentals and twelve harmonic partials. Runs for a few minutes. | `preMaetFromAttrTable`, `addSpectra` (pre-MAET form), `bindEvents`, `windowedSimilarity` (single-axis and multi-axis sweeps) |
| `demo_jmm_3_1_texture.m` | Analysis 3.1 — article §4.3.1 | How does the pooled texture's local entropy track the phase, at a fusing and a resolving time kernel? | `windowedEntropy` (`'method', 'renyi2'`) |
| `demo_jmm_3_2_diff.m` | Analysis 3.2 — supplement §9 | Does joint differencing of pitch and time expose the accelerandi of the phasing voice at the timing JND? | `differenceEvents`, `buildMaet`, `evalMaet`, windowed Rényi-2 entropy |
| `demo_jmm_3_3_xcorr.m` | Analysis 3.3 — supplement §10 | Can the running phase between the pianos be read as the ridge of a lag cross-correlogram? | `windowedSimilarity` (one-sided matched filter) |
| `demo_jmm_4_1_parse.m` | Analysis 4.1 — supplement §11 | What can the framework do with an expert analysis it is given? A rule-labelled derivation carried as a nested attribute: retrieval of a configuration, partial match on the label simplex, reduction by graded weights, and depth as a further coordinate. | `simplexVertices`, `packPreMaet`, `flatSpecs`, `bindAttributes`, `bindEvents` (`'groupBy'`), `selectPreMaet`, `buildMaet`, `simMaet` (`'normalize', 'oneSidedDenom'`) |

The Python demo's `sim_maet(..., mode='cartesian')` (the full
N × N matrix in one call) has no MATLAB counterpart; the MATLAB scripts
build the same matrices one row at a time with the struct-versus-cell
broadcast form of `simMaet`, which gives the same numbers.

The corpus study of the Online Supplement (all 4/4 four-part
chorales, mixed-effects models) depends on the music21 corpus and
statsmodels and is not a demo.
