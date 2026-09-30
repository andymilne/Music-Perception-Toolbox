# Music Perception Toolbox — User Guide

Andrew J. Milne, Western Sydney University

---

## Contents

**Part I — Getting started**

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [The toolbox at a glance](#3-the-toolbox-at-a-glance)
4. [Quick start](#4-quick-start)

**Part II — Stage by stage**

5. [Bringing material in](#5-bringing-material-in)
6. [The pre-MAET](#6-the-pre-maet)
7. [Preprocessing](#7-preprocessing)
8. [Densities and measures](#8-densities-and-measures)
9. [Measures on pitch and rhythm sets](#9-measures-on-pitch-and-rhythm-sets)

**Part III — Reference**

10. [API conventions](#10-api-conventions)
11. [Performance and numerical controls](#11-performance-and-numerical-controls)
12. [Function reference](#12-function-reference)
13. [MAETs: mathematical details](#13-maets-mathematical-details)
14. [Worked examples](#14-worked-examples)
15. [Demo scripts](#15-demo-scripts)
16. [Known simplifications and future directions](#16-known-simplifications-and-future-directions)
17. [References](#17-references)
18. [Citation](#18-citation)

[Acknowledgments](#acknowledgments)

---

# Part I — Getting started

## 1. Introduction

The Music Perception Toolbox is an open-source toolbox — available in MATLAB and Python — for computing perceptually and cognitively motivated measures of music. It models how similar musical materials are, from single chords and scales to whole passages; where material recurs, and how any of its measures change through a piece; how that material is distributed — how concentrated or dispersed, and so how predictable, it is, and how much of it lies in a given range; how consonant or harmonic a sound is; and how the pitches of a scale or the onsets of a rhythm are arranged around their cycle. Its input may be a score, an audio recording, material constructed to test a theoretical question, or the stimuli of an experiment.

Most of these measures belong to one family, built on the expectation tensor (Milne, Sethares, Laney, & Sharp, 2011) and its multi-attribute generalization, the MAET (Milne, 2026). An expectation tensor represents a weighted collection of values — the pitches of a chord, the onsets of a rhythm — as a density: a Gaussian mixture, with a kernel centred on each value, or on each pair, triple, or larger tuple of values where intervals and patterns matter. The width of each kernel models uncertainty. On the sounding surface this is perceptual uncertainty, a pitch or an onset never being heard exactly; but it applies equally to cognitive abstractions, such as a voice, an instrument, a metrical position, or a supplied analysis of the music's structure, and it allows an equivalence such as octave or transposition to be imposed fully or held to any degree. A MAET gives each event several such attributes at once — pitch and time, say, or pitch, voice, and instrument — each with its own kernel width and its own equivalences. The similarity of two densities, the entropy of one, and the mass it holds in a region are the measures read from it; the similarity and the mass have closed forms, and so, among the entropies, does the Rényi-2.

A second, separate family measures the arrangement of points on a cycle — the pitch classes of a scale, the onsets of a rhythmic cycle — directly, without building a density. The balance and evenness measures (Milne, Bulger, & Herff, 2017) draw on the discrete Fourier transform of that arrangement, and identify a novel class of perfectly balanced patterns. Further measures in the family describe how consistently a scale's step sizes follow their order, and where a rhythm's onsets cluster or thin out, and were developed and validated for modelling rhythmic perception and performance (Milne & Herff, 2020; Milne, Dean, & Bulger, 2023). A third group measures the consonance and harmonicity of a sound from its spectrum, and several of its measures are built on expectation tensors.

These measures have proven effective predictors across a range of music cognition contexts, including tonal fit and stability in conventional and microtonal tuning systems (Milne, Laney, & Sharp, 2015, 2016; Homer, Harley, & Wiggins, 2024; Hearne, Dean, & Milne, 2025), perceived consonance and affect (Smit et al., 2019; Harrison & Pearce, 2020; Eerola & Lahdelma, 2021), individual differences in harmony perception (Eitel, Ruth, Harrison, Frieler, & Müllensiefen, 2024), and rhythmic complexity and tapping accuracy (Milne & Herff, 2020; Milne, Dean, & Bulger, 2023). They have also guided the design of music-computing interfaces (Sethares, Milne, Tiedje, Prechtl, & Plamondon, 2009; Milne & Dean, 2016; Milne, 2019). Published empirical validation is to date on the single-attribute case; MAETs enable a class of analyses for which existing single-attribute measures would be forced either to pool events into unordered multisets or to abandon the framework entirely.

The MATLAB and Python implementations are functionally identical: the same inputs give the same outputs, to floating-point precision. Every function has full help text with examples (`help functionName` in MATLAB, `help(mpt.function_name)` in Python), and demo scripts in both languages cover the major use cases, beginning with `demo_0_startHere` (§15). `CHANGELOG.md` lists the changes since the last release, `MIGRATION.md` maps code written for earlier versions onto the current names, and `ARCHITECTURE.md` describes the implementation for developers.

**How this guide is organized.** Part I gets an analysis running: installation (§2), the toolbox at a glance (§3), and a quick start (§4). Part II takes an analysis stage by stage: bringing material in (§5), the pre-MAET (§6), preprocessing (§7), densities and the measures read from them (§8), and the measures that take a pitch or rhythm set directly (§9). Part III is reference: the conventions the two languages share (§10), performance and numerical controls (§11), every function (§12), and the mathematics of the MAET (§13), followed by worked examples (§14), the demos (§15), known simplifications (§16), references, and citation.

---

## 2. Installation

### MATLAB

1. Download or clone the repository from [GitHub](https://github.com/andymilne/Music-Perception-Toolbox).
2. Add the MATLAB folder to the MATLAB path:
   ```matlab
   addpath('/path/to/Music-Perception-Toolbox/matlab');
   ```
3. Optionally, save the path for future sessions:
   ```matlab
   savepath;
   ```

The MATLAB implementation requires MATLAB R2019b or later (for the `arguments` block syntax used by most functions). It has no external dependencies. Core functions are in the `matlab/` folder; demo scripts are in `matlab/demos/` and example audio files are in `matlab/audio/`. Only the `matlab/` folder needs to be added to the path.

### Python

The package is on PyPI:

```bash
pip install music-perception-toolbox
```

For audio file support (spectral peak extraction via `audio_peaks`):

```bash
pip install music-perception-toolbox[audio]
```

To work from a clone instead — to run the demos or the test suite, or to track the development version — install the local `python/` directory:

```bash
git clone https://github.com/andymilne/Music-Perception-Toolbox.git
pip install ./Music-Perception-Toolbox/python        # or ./python[audio]
```

Each release is archived with a DOI on Zenodo: <https://doi.org/10.5281/zenodo.19412254>.

The Python implementation requires Python 3.10 or later. Core dependencies (NumPy, SciPy) are installed automatically. The optional `soundfile` library is required only for `audio_peaks`.

All functions are accessible from the top-level `mpt` namespace:

```python
import mpt
s = mpt.sim_maet(...)
```

The Python implementation uses snake_case naming and a few other syntactic differences from the MATLAB version; see [§10](#10-api-conventions) for the full mapping.

---

## 3. The toolbox at a glance

This section is a map: what the toolbox computes, where material comes from, and which function does each job, with one line for each and a pointer to where it is described. Part II (§5–§9) takes each stage in prose; §12 has the full entry for every function. Names are given in MATLAB form; the Python names follow the snake_case rule of §10.1 (`buildMaet` → `build_maet`).

For a guided route through the toolbox by example, run `demo_0_startHere` (Python: `demo_0_start_here.py`) first. It runs no analysis but prints a guide to the demos: where a beginner should start, how the demos fit together, and which demos cover each topic (§15). Throughout this guide, a subsection ends with a **Demos.** line naming the demos that take its topic further, MATLAB name first and Python file second; a part number refers to the demo's own numbered sections, and *JMM* marks one of the article's analyses in `demos/jmm`, which has the same name in both languages.

### 3.1 What a MAET is

A listener does not register a musical event exactly. A tone is heard at a pitch that could plausibly have been any of a range of nearby pitches, an onset at a time that could have been any of a range of nearby times. Replacing each element of a collection by a Gaussian centred on its value, and adding these together, gives a continuous density whose value at any point is the expected number of elements perceived there. A single *kernel width*, $\sigma$, carries that perceptual uncertainty.

Two generalizations make the construction musically useful. First, the density can be over $r$-tuples of elements rather than single elements — pairs, triples, and larger groups — so that interval content, not just pitch content, is represented; $r$ is the attribute's *tuple size*. Second, an event may carry several *attributes* at once — pitch, onset time, duration, a categorical voice label — each with its own tuple size, kernel width, and flags (below), and the density is over all of them jointly. That is the MAET.

Each attribute is independently *absolute* or *relative* (invariant to transposition of the whole tuple), *periodic* or not (pitch classes and metrical positions wrap; pitches and absolute times do not), and *exchangeable* or *ordered* (whether the order of an event's values matters, as it does for a chord's voicing but not for its pitches). These three *flags* — `[rel]`, `[per]`, and `[exch]` — together with the tuple size, are what turn one construction into the different measures the toolbox provides. Collections of unequal size compare directly, since each is embedded as a density before any comparison is made, and no correspondence between their elements is required.

Three things are computed from a density: the *cosine similarity* of two of them, which measures how alike two collections are; the *entropy* of one, which measures how evenly its mass is spread; and the *mass* it holds in a region. The similarity, the mass, and the Rényi-2 entropy have closed forms, so no grid or resolution parameter enters; the other entropies need one (§16).

**Demos.** `demo_overview` / `demo_overview.py` (part 1: a chord as a density and the four parameters drawn); `demo_maetPlots` / `demo_maet_plots.py` (every combination of the parameters, drawn).

### 3.2 Where material comes from

Material enters the toolbox in one of four ways.

- **A score.** `readScore` reads a MIDI or MusicXML file into an *attribute table*, one row per note and one column per attribute (onset, duration, pitch, velocity, part, and so on). `gridAttrTable` samples the table on a regular time grid, and `preMaetFromAttrTable` turns it into a pre-MAET (§5.1–§5.3).
- **Audio.** `audioPeaks` extracts the spectral peaks of a recording, frequencies and amplitudes, which the spectral and consonance measures take directly (§5.4).
- **Material made by hand to answer a theoretical question.** A chord, scale, tuning, or rhythm is a vector of values — pitches in cents or semitones, positions in a cycle — and a set of them is the rows of a matrix. Most functions take these directly, and `packPreMaet` gathers several attributes into a pre-MAET (§5.5).
- **An experiment.** The stimuli of a study are a table of trials, one row per trial, and the batched forms of the measures compute a feature for every row, computing equivalent rows once (§5.5).

`transformAttributes` converts between pitch and frequency scales (Hz, MIDI, cents, octaves, mel, Bark, ERB-rate, Greenwood) and applies logarithmic, power, affine, and user-supplied transforms, at any of these entry points.

**Demos.** `demo_scoreWorkflow` / `demo_score_workflow.py` (a score); `demo_audioAnalysis` / `demo_audio_analysis.py` (audio); `demo_probeTone` / `demo_probe_tone.py` (chords and scales made by hand); `demo_rhythmTensors` / `demo_rhythm_tensors.py` (rhythms); `demo_batchProcessing` / `demo_batch_processing.py` (a table of experimental trials).

### 3.3 The MAET path

Analyses built on MAETs follow one path:

```
attribute table or vectors  →  pre-MAET  →  preprocessing  →  density (MAET)  →  measures
```

A **pre-MAET** is the material chosen for analysis: the events, the values each attribute holds at each event, their weights, and each attribute's parameters — its kernel width σ, its tuple size r, and whether it is relative, periodic, or ordered. Preprocessing reshapes the pre-MAET, each operation taking one and returning another; `buildMaet` turns it into a density; and the measures read the density or compare two. The measures also take a pre-MAET, or for a single multiset the bare values, and build the density themselves.

Here is a small pre-MAET: four events, the first a C major triad at beat 0 and then the melody notes D, E, and F, with two attributes, pitch (in MIDI numbers) and onset (in beats). The pitches here are integers, but any value may be used, so a microtonal pitch is simply a non-integer such as 60.5. The pitch attribute is read as pitch class because its specs make it periodic (`[per] = 1`) with period 12 (`P = 12`), so values an octave apart are the same point.

```matlab
pAttr = {{[60 64 67], 62, 64, 65}, [0 1 1.5 2]};
specs = flatSpecs(pAttr, 'name', {'pitch', 'onset'}, 'sigma', [0.15 0.25], ...
                  'isPer', [true false], 'period', [12 0]);
pm = packPreMaet(pAttr, [], specs);
showPreMaet(pm)
```

```python
p_attr = [[[60, 64, 67], 62, 64, 65], [0, 1, 1.5, 2]]
specs = mpt.flat_specs(p_attr, name=['pitch', 'onset'], sigma=[0.15, 0.25],
                       is_per=[True, False], period=[12.0, 0.0])
pm = mpt.pack_pre_maet(p_attr, None, specs)
mpt.show_pre_maet(pm)
```

```
| attribute                                               |    n = 1     | n = 2 | n = 3 | n = 4 |
|:--------------------------------------------------------|:------------:|:-----:|:-----:|:-----:|
| pitch: sigma = 0.5, r = 1, [rel] = 0, [per] = 1, P = 12 | {60, 64, 67} |  62   |  64   |  65   |
| onset: sigma = 0.25, r = 1, [rel], [per] = 0            |      0       |   1   |  1.5  |   2   |
```

Each attribute is a row, headed by its parameters (`[rel], [per] = 0` is short for both flags being 0), and each event is a column. The braces mark an unordered multiset: the chord's three pitches, which a single-note event does not need. Each attribute's values are given event by event, as a cell (MATLAB) or list (Python) with one entry per event holding that event's values; here the pitch attribute's first entry is the chord and the rest are single notes, and the onset attribute, one value per event, is a plain vector. The toolbox stores each attribute as a matrix with one row per value and one column per event, padded with `NaN` where an event holds fewer values than the widest, and that matrix, given directly, is accepted too (§6.1). The same pre-MAET can also be entered as a CSV file, laid out like the table above and edited in any spreadsheet, and read with `readPreMaet` (§6.4); a score gives the same kind of object through `preMaetFromAttrTable` (§5.3), and §6 describes pre-MAETs in full.

**The pre-MAET** (§6)

| Function | What it does |
|:---|:---|
| `preMaetFromAttrTable` | Build a pre-MAET from an attribute table (§5.3) |
| `packPreMaet` / `unpackPreMaet` | Hold a pre-MAET's values, weights, and specs in one object, and split them again |
| `flatSpecs` | Make the per-attribute specifications for attributes without level structure |
| `showPreMaet` | Print a pre-MAET as a table (markdown, LaTeX, or CSV) |
| `readPreMaet` / `writePreMaet` | Read and write a pre-MAET as CSV |
| `kernelCov` | Build a matrix-valued kernel covariance for an ordered attribute (§13.3) |
| `simplexVertices` | Coordinates for a categorical attribute whose levels are all equally different |

**Preprocessing** (§7)

| Function | What it does |
|:---|:---|
| `bindEvents` | Gather consecutive events into one nested event: n-grams |
| `differenceEvents` | Replace values with the change between successive events: intervals, inter-onset intervals |
| `transformAttributes` | Map values through a transform or a change of scale |
| `translateAttributes` | Shift an attribute's values by an offset |
| `weightEvents` | Multiply a window or profile into the per-event weights |
| `addSpectra` | Replace each pitch with the partials of its spectrum |
| `selectPreMaet` | Keep a selection of the attributes and the events |
| `bindAttributes` / `separateAttributes` | Read several attributes as one tuple, and split one again |

**Densities and measures** (§8)

| Function | What it does |
|:---|:---|
| `buildMaet` | Build the density, the MAET |
| `evalMaet` | The density at query points |
| `plotMaet` | Draw a density of one, two, or three dimensions |
| `simMaet` | Cosine similarity of two densities; with spectral enrichment, spectral pitch (class) similarity |
| `entropyMaet` | Entropy of a density, by four estimators |
| `massMaet` | Mass of a density in a region, or its share of the whole |
| `sweptSimilarity`, `sweptEntropy`, `sweptMass` | Similarity, entropy, or mass at each of a list of values on an attribute: a query translated along a context, a window stepped through it, or both |
| `sweepSimMaet` | Similarity of one density against translated copies of another, in one pass |
| `maetCentres` | The points at which a density places its kernels |

**Demos.** `demo_overview` / `demo_overview.py` (part 2: the path once through, with a motif search); `demo_scoreWorkflow` / `demo_score_workflow.py` (the path from a score file to a result); `demo_preMaetIo` / `demo_pre_maet_io.py` (a pre-MAET shown, written, and read back).

### 3.4 Measures outside the MAET framework

Not every function belongs to the MAET framework. The measures below take a pitch or rhythm set, or a spectrum, directly. Some of them use expectation tensors internally; several of the circular and structural measures do not use them at all. They are described in §9.

| Family | Functions | What they measure |
|:---|:---|:---|
| Consonance and harmonicity (§9.1) | `spectralEntropy`, `templateHarmonicity`, `tensorHarmonicity`, `roughness`, `virtualPitches` | How consonant or harmonic a chord or spectrum is, and the virtual pitches it evokes |
| Balance and evenness (§9.2) | `balanceCircular`, `evennessCircular`, `dftCircular`, `dftCircularSimulate` | How the points of a scale or rhythm are distributed around the cycle, from its discrete Fourier transform |
| Scale and rhythm structure (§9.2) | `coherence`, `sameness`, `nTupleEntropy`, `circApm`, `edges`, `projCentroid`, `meanOffset`, `markovS` | Structural features of points on a cycle, per collection or per position |
| Sequences (§9.3) | `continuity` | The recent trend of a sequence leading up to a point |

Three further functions serve the rest: `explainDispatch` reports how a call will be computed and why, `mptDefaults` inspects and sets the toolbox-wide defaults, and `estimateCompTime` estimates how long a call will take (§11).

**Demos.** `demo_overview` / `demo_overview.py` (parts 3–5: consonance, balance and evenness, and scale structure); `demo_triadConsonance` / `demo_triad_consonance.py`; `demo_dftCircularSimulate` / `demo_dft_circular_simulate.py`; `demo_sigmaSpace` / `demo_sigma_space.py`.

---

## 4. Quick start

Short examples in both languages, each a complete call. Each ends with a pointer to where its topic is described in full and a line naming the demos that take it further.

### Computing SPCS between two chords

**MATLAB:**
```matlab
% Define two weighted pitch multisets (in cents)
major = [0, 400, 700];       % 12-EDO major triad
minor = [0, 300, 700];       % 12-EDO minor triad

% Add harmonic spectra (12 partials, 1/n rolloff)
[maj_p, maj_w] = addSpectra(major, [], 'harmonic', 12, 'powerlaw', 1);
[min_p, min_w] = addSpectra(minor, [], 'harmonic', 12, 'powerlaw', 1);

% Compute SPCS (absolute periodic monad tensor, sigma = 10)
s = simMaet(maj_p, maj_w, min_p, min_w, 10, 1, false, true, 1200);
fprintf('SPCS(major, minor) = %.3f\n', s);
```

**Python:**
```python
import mpt

major = [0, 400, 700]
minor = [0, 300, 700]

maj_p, maj_w = mpt.add_spectra(major, None, 'harmonic', 12, 'powerlaw', 1)
min_p, min_w = mpt.add_spectra(minor, None, 'harmonic', 12, 'powerlaw', 1)

s = mpt.sim_maet(maj_p, maj_w, min_p, min_w, 10, 1, False, True, 1200)
print(f'SPCS(major, minor) = {s:.3f}')
```

**Demos.** `demo_overview` / `demo_overview.py` (part 1b); `demo_triadSpcsGrid` / `demo_triad_spcs_grid.py` (the SPCS of every 12-EDO triad with a fifth).

### Comparing one chord with many

Stack the chords as the rows of a matrix and make one call. The reference is built once, the partials are added to both sides of every row, and rows that are equivalent under the density's symmetries are computed once, so a long list costs little more than its distinct chords.

**MATLAB:**
```matlab
chords = [0 300 700; 0 400 700; 0 500 700];   % one chord per row
s = simMaet(major, [], chords, [], 10, 1, false, true, 1200, ...
            'spectrum', {'harmonic', 12, 'powerlaw', 1});
```

**Python:**
```python
import numpy as np

chords = np.array([[0, 300, 700], [0, 400, 700], [0, 500, 700]])
s = mpt.sim_maet(major, None, chords, None, 10, 1, False, True, 1200,
                 spectrum=('harmonic', 12, 'powerlaw', 1))
```

A density can also be built once with `buildMaet` and passed in place of the raw values. Within one call this saves nothing more, but the density outlives the call: it is the form to use when the same reference is compared again in later calls, as comparisons arrive one at a time, or when the density itself is wanted, to inspect or plot.

**MATLAB:**
```matlab
dens_ref = buildMaet(maj_p, maj_w, 10, 1, false, true, 1200);   % built once

[cp, cw] = addSpectra([0 500 700], [], 'harmonic', 12, 'powerlaw', 1);
s = simMaet(dens_ref, buildMaet(cp, cw, 10, 1, false, true, 1200));
```

**Python:**
```python
dens_ref = mpt.build_maet(maj_p, maj_w, 10, 1, False, True, 1200)

cp, cw = mpt.add_spectra([0, 500, 700], None, 'harmonic', 12, 'powerlaw', 1)
s = mpt.sim_maet(dens_ref, mpt.build_maet(cp, cw, 10, 1, False, True, 1200))
```

The batched form and its broadcasting are described in §10.6, the deduplication in §10.7, and the choice between densities and raw values in §10.5.

**Demos.** `demo_batchProcessing` / `demo_batch_processing.py` (a table of trials in one batched call); `demo_edoApprox` / `demo_edo_approx.py` (every EDO against a JI chord); `demo_genChainPcs` / `demo_gen_chain_pcs.py` (a generator swept through its range).

### Drawing a density

`plotMaet` draws a density directly, choosing a line, a surface, or a volume from its number of dimensions ($r$, less one when relative). The spectral pitch-class density of a major triad is one-dimensional, a line over the octave:

**MATLAB:**
```matlab
[p_spec, w_spec] = addSpectra([0 400 700], [], 'harmonic', 12, 'powerlaw', 1);
dens = buildMaet(p_spec, w_spec, 10, 1, false, true, 1200);
plotMaet(dens, 'method', 'density');
```

**Python:**
```python
p_spec, w_spec = mpt.add_spectra([0, 400, 700], None, 'harmonic', 12, 'powerlaw', 1)
dens = mpt.build_maet(p_spec, w_spec, 10, 1, False, True, 1200)
mpt.plot_maet(dens, method='density')
```

A major pentatonic scale, taken three pitches at a time and made relative ($r = 3$, `[rel] = 1`), becomes a two-dimensional density of the interval content of its three-note subsets, the same wherever it is transposed. By default `plotMaet` draws the kernels, one per tuple, which shows how the density is assembled; `'density'` draws their sum:

**MATLAB:**
```matlab
dens3 = buildMaet([0 200 400 700 900], [], 20, 3, true, true, 1200);
plotMaet(dens3);                                  % the kernels
figure; plotMaet(dens3, 'method', 'density');     % the density
```

**Python:**
```python
dens3 = mpt.build_maet([0, 200, 400, 700, 900], None, 20, 3, True, True, 1200)
mpt.plot_maet(dens3)                              # the kernels
mpt.plot_maet(dens3, method='density')            # the density
```

At $r = 4$ the density has three dimensions and is drawn as a volume. Beside the kernels, `'points'` samples it on a grid, one translucent mark per point where the density is appreciable; the continuous volume, `'density'`, is available in MATLAB only. In MATLAB, overlapping marks are drawn one over another rather than blended, so `plotMaet` warns when they overlap at the view drawn and names a `markScale` that would keep them clear; the value depends on the size of the figure, and 0.5 keeps them clear at the default size (§8.6):

**MATLAB:**
```matlab
dens4 = buildMaet([0 200 400 700 900], [], 20, 4, true, true, 1200);
plotMaet(dens4);                                               % the kernels
figure; plotMaet(dens4, 'method', 'points', 'markScale', 0.5); % sampled
figure; plotMaet(dens4, 'method', 'density');                  % whole
```

**Python:**
```python
dens4 = mpt.build_maet([0, 200, 400, 700, 900], None, 20, 4, True, True, 1200)
mpt.plot_maet(dens4)                              # the kernels
mpt.plot_maet(dens4, method='points')             # the density, sampled
```

The density's values at chosen points come from `evalMaet` (§8.2), for plotting by other means. §8.6 has the methods, the grid, and the three-dimensional views.

**Demos.** `demo_maetPlots` / `demo_maet_plots.py` (every combination of r, [rel], [per], and [exch], by each method); `demo_overview` / `demo_overview.py` (part 1e).

### Computing balance and evenness of a rhythm

**MATLAB:**
```matlab
% Son clave pattern: 5 onsets in a 16-step cycle
pattern = [0, 3, 6, 10, 12];
b = balanceCircular(pattern, [], 16);
e = evennessCircular(pattern, 16);
fprintf('Balance = %.3f, Evenness = %.3f\n', b, e);
```

**Python:**
```python
pattern = [0, 3, 6, 10, 12]
b = mpt.balance(pattern, None, 16)
e = mpt.evenness(pattern, 16)
print(f'Balance = {b:.3f}, Evenness = {e:.3f}')
```

**Demos.** `demo_overview` / `demo_overview.py` (part 4); `demo_dftCircularSimulate` / `demo_dft_circular_simulate.py` (balance and evenness under positional jitter).

### Extracting spectral peaks from audio

**MATLAB:**
```matlab
wav = fullfile(fileparts(which('audioPeaks')), 'audio', 'piano_Cmin_open.wav');
[f, w] = audioPeaks(wav);
p = transformAttributes(f, [], {'hz', 'cents'});
H = spectralEntropy(p, w, 12);
fprintf('Spectral entropy = %.3f\n', H);
```

**Python:**
```python
f, w, detail = mpt.audio_peaks('audio/piano_Cmin_open.wav')   # from the python folder
p = mpt.transform_attributes(f, None, ('hz', 'cents'))
H = mpt.spectral_entropy(p, w, 12)
print(f'Spectral entropy = {H:.3f}')
```

**Demos.** `demo_audioAnalysis` / `demo_audio_analysis.py` (peaks, then similarity, harmonicity, roughness, and virtual pitches); `demo_virtualPitches` / `demo_virtual_pitches.py`.

### From a score to a result

A score is read into an attribute table, which becomes a pre-MAET, whose density is measured. Here the pitch-class entropy of a Bach chorale is traced through the piece in a four-beat window stepped through it: `readScore` reads the score; `preMaetFromAttrTable` makes pitch class (a periodic attribute, one value per chord note, `r = 1`) and onset the two attributes; and `sweptEntropy` weights the events by a window at each step, drops the onset attribute, and takes the Rényi-2 entropy of what remains.

**MATLAB:**
```matlab
score = fullfile(fileparts(which('buildMaet')), 'demos', 'jmm', 'data', 'bwv347.musicxml');
t  = readScore(score);                          % one row per note
pm = preMaetFromAttrTable(t, 'attributes', { ...
        struct('column', 'pitch', 'name', 'pitchClass', 'sigma', 0.5, ...
               'r', 1, 'exch', true, 'isPer', true, 'period', 12), ...
        struct('column', 'onset', 'sigma', 0.25)}, 'time', 'beats');
showPreMaet(pm, 'maxEvents', 6);
H = sweptEntropy(pm, 'sweep', 2, 'window', {2, {'rect', 4}}, 'drop', 2, ...
                 'method', 'renyi2');
```

**Python:**
```python
t = mpt.read_score('demos/jmm/data/bwv347.musicxml')     # from the python folder
pm = mpt.pre_maet_from_attr_table(t, attributes=[
    {'column': 'pitch', 'name': 'pitchClass', 'sigma': 0.5, 'r': 1,
     'exch': True, 'is_per': True, 'period': 12},
    {'column': 'onset', 'sigma': 0.25}], time='beats')
mpt.show_pre_maet(pm, max_events=6)
H = mpt.swept_entropy(pm, sweep=1, window={1: ('rect', 4.0)}, drop=1,
                      method='renyi2')
```

Chords are bound into one event by default, so the pitch-class attribute holds the notes sounding together, and σ is in semitones because the pitch column is in MIDI numbers. §5 covers reading and sampling scores, `demo_scoreWorkflow` follows this path with more choices along the way, and §8.7 describes the swept functions.

**Demos.** `demo_scoreWorkflow` / `demo_score_workflow.py` (the path end to end); `demo_jmm_1_1_entropy` (JMM: windowed spectral entropy of the same chorale); `demo_jmm_3_1_texture` (JMM: local entropy tracking a phase process).

### Probe-tone fit to a context

In the classic probe-tone paradigm a listener hears a context sequence followed by a probe, and rates how well the probe fits. The simplest model of this fit is a single similarity value (SPCS) between the probe and the pooled spectrum of the context. The example below computes this fit directly with `simMaet` on two single-attribute tensors. Context events are recency-weighted, reflecting the intuition that later events are more salient in working memory; at this bare-array level the profile is one line of arithmetic, and §14.4 shows the same idea inside a pre-MAET with `weightEvents`. `demo_probeTone` / `demo_probe_tone.py` reproduces the TISMIR article's probe-tone profiles (C major against Krumhansl and Kessler's ratings, and Porcupine[7] in 22-EDO) and extends them to a time-ordered context with recency weighting and inharmonic spectra.

The pitches are converted from MIDI numbers to cents because `addSpectra` places each partial at its interval above the fundamental in cents by default, $1200 \log_2 n$ for the $n$-th harmonic, and cents are also the unit of the published spectral models ($\sigma = 10$ cents, period 1200). Working in MIDI numbers is equally possible: pass `'units', 12` to `addSpectra` and give $\sigma$ and the period in semitones (0.1 and 12).

**MATLAB:**
```matlab
context = transformAttributes([60 64 67 72], [], {'midi', 'cents'});   % C E G C
probeE  = transformAttributes(64, [], {'midi', 'cents'});              % probe: E

% Recency decay: weights later events more heavily
w = exp(-0.5 * (4 - (1:4)));

% Apply spectral enrichment, folding the decay weights into the pitch weights
spec = {'harmonic', 12, 'powerlaw', 1};
[ctx_p, ctx_w] = addSpectra(context, w, spec{:});
[probe_p, probe_w] = addSpectra(probeE, [], spec{:});

s = simMaet(ctx_p, ctx_w, probe_p, probe_w, 10, 1, false, true, 1200);
fprintf('Probe-E fit (recency-weighted): %.3f\n', s);
```

**Python:**
```python
import numpy as np
context = mpt.transform_attributes(np.array([60, 64, 67, 72]), None, ('midi', 'cents'))
probe_E = mpt.transform_attributes(np.array([64]), None, ('midi', 'cents'))

w = np.exp(-0.5 * (4 - np.arange(1, 5)))

spec = ('harmonic', 12, 'powerlaw', 1.0)
ctx_p, ctx_w = mpt.add_spectra(context, w, *spec)
probe_p, probe_w = mpt.add_spectra(probe_E, None, *spec)

s = mpt.sim_maet(ctx_p, ctx_w, probe_p, probe_w,
                          10.0, 1, False, True, 1200.0)
print(f'Probe-E fit (recency-weighted): {s:.3f}')
```

This approach treats the context as an unordered pitch multiset with per-event salience weights. For questions that depend on *when* events occurred — probe fit as a function of time, for example, rather than a single fit value — time can be carried as a second attribute on a multi-attribute tensor; see [§14.4](#144-probe-tone-scanning-with-irregular-timing-and-event-weights) for a worked example.

**Demos.** `demo_probeTone` / `demo_probe_tone.py` (the published profiles, recency weighting, inharmonic spectra, and continuity).

### Finding where a pattern occurs within a melody

The previous subsection returned a single similarity value — the fit of a probe against the whole context. For longer sequences, it is often useful to identify *where* a query pattern most closely matches the content of the sequence, returning one similarity per position — a similarity profile. With time as a second attribute, `sweptSimilarity` translates the query along the time attribute by each of a list of *sweep values* (attribute translation, §7.4) and compares it with the whole melody at each. By default the sweep values are the offsets added to the query as written; the query and the melody are both written from time 0, so an offset is the time at which the query starts, and a peak at $s$ means the pattern is present in the melody starting at $s$ (§8.7).

**MATLAB:**
```matlab
% Melody: C D E G C E G (MIDI numbers), with event times 0..6
melody_p = [60 62 64 67 60 64 67];
melody_t = 0:6;

% Query: the 2-event pattern E G, written from time 0
query_p = [64 67];
query_t = [0 1];

% Pitch attribute: absolute periodic (sigma = 0.1 semitones, period 12).
% Time attribute: absolute non-periodic (sigma = 0.2 time units).
sigma = [0.1 0.2]; r = [1 1]; isRel = [false false];
isPer = [true false]; period = [12 0];

% Translate the query along time (attribute 2) and compare it with the
% whole melody at each offset. Time is compared, so the query's internal
% timing must match. mu{2} holds the offsets.
[S, mu] = sweptSimilarity({melody_p, melody_t}, [], {query_p, query_t}, [], ...
                          sigma, r, isRel, isPer, period, 'sweep', 2);
plot(mu{2}, S);
```

**Python:**
```python
import numpy as np
import matplotlib.pyplot as plt

melody_p = np.array([60., 62, 64, 67, 60, 64, 67])        # MIDI numbers
melody_t = np.arange(7, dtype=float)

query_p = np.array([64., 67])
query_t = np.array([0., 1.])

S, mu = mpt.swept_similarity(
    [melody_p[None, :], melody_t[None, :]], None,
    [query_p[None, :], query_t[None, :]], None,
    [0.1, 0.2], [1, 1], [False, False], [True, False], [12., 0.],
    sweep=1, return_offsets=True, verbose=False)
plt.plot(mu[1], S)
```

Naming only the swept attribute lets the toolbox choose the sweep values: every placement at which the query overlaps the melody (here offsets from −1 to 6), in steps no wider than half the width of the profile's peaks and aligned so that every exact match falls on the grid. A particular list can be given instead, `'sweep', {2, s}` / `sweep={1: s}`, or the defaults adjusted with `'start'`, `'stop'`, and `'step'` (§8.7). The query E G, written at times 0 and 1, lies at times $(s, s + 1)$ after translation by $s$. The melody contains the E–G pair at events $(t=2, t=3)$ and again at $(t=5, t=6)$, so the profile peaks at $s = 2$ and $s = 5$, the start times of the two pairs. Positions between and outside these produce graded similarity determined by how closely the translated query aligns with nearby melody events; only the kernel width $\sigma$ on time limits which melody events count.

To make each comparison local, a window on the melody can be aligned together with the query (`'align'`, `'both'`): at each sweep value the window is aligned there and the query's middle lands there too, so the window and the query are aligned at the same point. Left unspecified, the window is the smallest closed rectangle that holds the query. Read against the offsets, this gives the same peaks, at 2 and 5:

```matlab
[S, mu] = sweptSimilarity({melody_p, melody_t}, [], {query_p, query_t}, [], ...
                          sigma, r, isRel, isPer, period, ...
                          'sweep', 2, 'align', {2, 'both'});
plot(mu{2}, S);
```

Under the default normalization the window changes little here, the kernel already making distant events count for little; under `'cosine'` it matters (§8.7). See [§14.5](#145-recurrence-of-interval-content-across-a-melody) for a fuller worked example including transposition-invariant comparison of interval content via event differencing.

**Demos.** `demo_sweptSimilarity` / `demo_swept_similarity.py` (every setting, on one melody); `demo_helixBlend` / `demo_helix_blend.py` (a time-windowed sweep of a motif across registers); `demo_jmm_2_3_spectral` (JMM: a motif located in time and key); `demo_jmm_1_3_cadence_nesting` (JMM: cadences located with nested prototypes).

---

# Part II — Stage by stage

## 5. Bringing material in

This section covers the first stage of an analysis: reading a score into an attribute table and sampling it, turning the table into a pre-MAET, extracting peaks from audio, and working with material made by hand or taken from an experiment. The function entries are in §12.1.

### 5.1 Reading a score

`readScore` / `read_score` parses a Standard MIDI File (format 0 or 1) or a MusicXML score (`.musicxml`, `.xml`, or compressed `.mxl`) into an *attribute table* — a MATLAB `table` or pandas `DataFrame` with one row per sounding note, carrying its onset and duration in quarter-note beats and in seconds (following the file's tempo map), its MIDI pitch, its velocity, its part as a categorical whose categories are the part names, its bar, and then whatever its source carries: a MIDI channel, or a MusicXML voice and fermata flag — and `preMaetFromAttrTable` / `pre_maet_from_attr_table` turns that table into a pre-MAET. The attributes are chosen by name from pitch, onset, duration, velocity, part, measure, and fermata; the pitch scale is any pitch scale of `transformAttributes` (`'midi'` by default, `'cents'` for the toolbox's spectral functions); time is in seconds or beats; the weights are velocity / 127, ones, or the duration. Notes that start together are bound into one event by default (`chords = 'bind'`, with an onset tolerance), so that the pitch attribute carries one value per chord note (K = the largest chord, NaN-padded) and an `r = 2` reading of it gives the chords' dyads; `chords = 'separate'` makes every note its own event. Both parsers are self-contained — no toolbox, package, or Java dependency in either language — and read the same file to the same table. A MusicXML tie is merged into one note, a grace note is skipped, and a rest or unpitched note is not a note.

**Demos.** `demo_scoreWorkflow` / `demo_score_workflow.py` (parts 1–2: reading and looking at the table); `demo_jmm_1_1_entropy` (JMM: a chorale read and sampled).

### 5.2 Sampling a score on a grid

`gridAttrTable` samples an attribute table on a regular grid of time points, one event per point, a held note occupying every slice it sounds in and a note shorter than the step still occupying one. A grid gives the event index a uniform meaning in time, which a window measured in beats or a comparison of what sounds at each moment needs. The weighting says what a slice takes from a note overlapping it: `'coverage'` (the default) the fraction of the slice the note fills, `'presence'` the note's full weight in every slice it appears in, and `'item'` the note's weight spread over its slices so that it counts once in all. A slice with nothing sounding is kept as an empty event, so the grid stays uniform. An already-gridded table regrids at a coarser step, and `ungridAttrTable` returns the table it was made from. The full entries are in §12.1, and `demo_scoreGrid` works through the choices.

**Demos.** `demo_scoreGrid` / `demo_score_grid.py` (the step, the weighting, and empty slices); `demo_scoreWorkflow` / `demo_score_workflow.py` (part 3).

### 5.3 From an attribute table to a pre-MAET

`preMaetFromAttrTable` turns an attribute table into a pre-MAET. Each attribute names the column it reads together with its own parameters, so one pitch column listed twice gives two attributes read under different parameters — pitch class and pitch height, say. The conversion fills in what follows from the data and asks for the rest: σ always, and `r` and `exch` wherever an attribute holds several values at an event, since a score fixes what the values are, not how tolerant a match should be nor how many of an event's values a tuple takes. Notes that start together are bound into one event by default (`'chords', 'bind'`), and `'chords', 'separate'` makes every note its own event. The table is an ordinary MATLAB `table` or pandas `DataFrame`, so rows are selected with the host language's own indexing before converting; `selectPreMaet` selects attributes and events of the pre-MAET afterwards. The full list of arguments is in §12.1, and `demo_scoreWorkflow` follows the path from a score file to a result.

**Encoding a categorical column.** `'roles'` maps a categorical column to how it reaches the pre-MAET, as one of `'separateAttributes'`, `'orderedMultiset'`, `'simplex'`, or `'drop'`; a column with no entry is not encoded. A value may instead be a struct/mapping carrying the role under `role` together with the parameters of the attribute the role creates, which is how a simplex-coded category is given its own width. Where a role fixes `r` or `exch` and a value is supplied too, `'orderedMultiset'` takes the supplied value and warns — the role having only arranged existing values into slots, so how many are drawn from them and whether their order counts remain the analyst's questions — while `'simplex'` refuses, the role having replaced the level with coordinates that denote a vertex only read whole and in order. The first two are *structural*: the level is realized as which attribute you are in, or as which position, so the binding of value to level is carried by the layout. They gather an event's rows into one event holding one slot per level, which needs `'chords', 'bind'` and an event holding exactly one row per level — events that do not are dropped, with a warning naming the count, since an analyst may well accept losing a few events to use the encoding. Only one category may be structural, because a structural category individuates the values sounding together and two of them give an attribute set that can never be fully populated; the refusal names the two ways out. A structural category splits every listed attribute except the event-level ones (`onset`), whose value belongs to the event rather than to the note. `'simplex'` is a *value*: the level becomes the coordinates of a vertex of a unit-edge regular simplex (`simplexVertices`), carried as its own attribute read whole, and the binding is the tensor product of the two attributes at the event. Each concurrently-sounding note is then its own event, so it needs `'chords', 'separate'` — unless a structural category is also given, in which case the simplex is tagged within each of its slots. These are the three encodings the JMM article contrasts on BWV 347's voicing: voice-aware is `'orderedMultiset'` on the part with `'chords', 'bind'`; simplex-voice is `'simplex'` with `'chords', 'separate'`; and voice-agnostic is the same one-event-per-note grain with no role at all, which is simplex-voice without its voice attribute.

A caution about the no-role reading at `'chords', 'bind'`. There an event holds the chord as an unordered multiset on every attribute, so where two attributes describe the *same* notes — a pitch-class attribute and a pitch-height one, say — their product pairs every value of one with every value of the other, including the soprano's pitch class with the bass's height, and matching rewards combinations the chord does not contain. Binding a note's attributes to each other needs one event per note.

`'groupBy'` names the column whose equal values in consecutive rows make one event, generalizing the onset binding. The default is the grid position where the table has been gridded, and otherwise the onset within `'chordTolerance'`. An event is a contiguous run, not every row sharing a value, so a bar number that comes round again after a repeat gives two events rather than one. On a gridded table the `onset` attribute reads the grid's onset, the event there being the grid point rather than any one note.

**Demos.** `demo_scoreWorkflow` / `demo_score_workflow.py` (parts 4–5); `demo_scoreCategoricals` / `demo_score_categoricals.py` (a categorical column encoded three ways); `demo_preMaetIo` / `demo_pre_maet_io.py` (part 8: a pre-MAET from a score).

### 5.4 Audio

`audioPeaks` reads an audio file, computes its magnitude spectrum, and returns the frequencies (in Hz) and normalized amplitudes of its peaks. The peaks are the sounded spectrum, so they go to the spectral and consonance measures without `addSpectra`: convert them to cents with `transformAttributes` for the pitch-based measures, and keep them in Hz for `roughness`.

A typical workflow:

```matlab
audioDir = fullfile(fileparts(which('audioPeaks')), 'audio');   % the toolbox's samples
[f, w] = audioPeaks(fullfile(audioDir, 'piano_C4.wav'));
p = transformAttributes(f, [], {'hz', 'cents'});
H = spectralEntropy(p, w, 12);           % no addSpectra needed
r = roughness(f, w);                      % roughness needs Hz
[hMax, hEnt] = templateHarmonicity(p, w, 12);
```

**Demos.** `demo_audioAnalysis` / `demo_audio_analysis.py` (two passes of peak extraction, then the features); `demo_virtualPitches` / `demo_virtual_pitches.py`.

### 5.5 Material made by hand, and experimental stimuli

A theoretical question usually starts from a vector: a chord or scale in cents or semitones, the steps of a tuning, a rhythm as positions in a cycle. The single-multiset forms take it directly, with its weights (`[]` / `None` for all ones) and the parameters given positionally — `simMaet(p1, w1, p2, w2, sigma, r, isRel, isPer, period)` compares two chords, and `balanceCircular(p, [], period)` measures a rhythm — so one attribute needs no pre-MAET. For several attributes per event, pitch and time say, `packPreMaet` gathers the per-attribute values (given event by event, or as matrices with one row per value and one column per event), their weights, and their specs into a pre-MAET (§6.1); `flatSpecs` makes the specs of attributes without level structure.

A set of chords, scales, or rhythms is the rows of a matrix, NaN-padded where they differ in size, and the batched forms return one value per row (§10.6). The stimuli of an experiment, one row per trial, go straight in, and rows that are equivalent under the measure's symmetries — transpositions, reorderings — are computed once (§10.7). `demo_batchProcessing` computes several features for a table of trials; `demo_edoApprox`, `demo_genChainPcs`, `demo_probeTone`, and `demo_rhythmTensors` start from material made by hand.

**Demos.** `demo_probeTone` / `demo_probe_tone.py`; `demo_edoApprox` / `demo_edo_approx.py`; `demo_genChainPcs` / `demo_gen_chain_pcs.py`; `demo_rhythmTensors` / `demo_rhythm_tensors.py`; `demo_batchProcessing` / `demo_batch_processing.py` (a table of trials, and a cell of pre-MAETs).

---

## 6. The pre-MAET

A pre-MAET is everything a density is built from: the per-attribute values at each event, their weights, and the per-attribute specifications (Milne, 2026, Def. 2.6). The three parts travel together, and the toolbox provides one object to hold them, one table to view them, and a CSV form to write and read them. Every preprocessing operation (§7) takes a pre-MAET and returns one, and every measure (§8) takes one wherever it takes a density.

### 6.1 Holding a pre-MAET in one object

The three parts — `pAttr`, `wAttr`, and `specs` — always travel together and always describe the same pre-MAET, so `packPreMaet` / `pack_pre_maet` holds them in one object: a MATLAB struct with the fields `pAttr`, `wAttr`, and `specs`, and a Python dict with the keys `p_attr`, `w_attr`, and `specs`. It is a plain struct or dict rather than a class, so its parts stay ordinary cells, arrays, and structs, and any of them may be read or replaced directly.

`pAttr` holds one entry per attribute, and each attribute's values may be given in either of two forms. *Per event*, the attribute is a cell (MATLAB) or list (Python) with one entry per event, each entry that event's values — a scalar, a vector, or empty for an event with no value on this attribute — so `{[60 64 67], 62, [], 65}` / `[[60, 64, 67], 62, [], 65]` is a triad, a single note, a silence, and another single note. *As a matrix*, it is $K_a \times N$, one column per event, with each event's values at the top of its column and `NaN` below them where it holds fewer than the widest. On an ordered attribute the position of a value within its event is its level — the voice of an SATB voicing, say — so a `NaN` may also stand before or between values to mark an empty slot: `{[60 64 67], [62 NaN 67], [NaN 65]}` gives the second event no value in slot 2 and the third none in slot 1. Positions are kept throughout: by the conversion, by every preprocessing operation, by `showPreMaet`, and through a CSV file (§6.4). The per-event form is the one to write by hand, since it says what each event holds and needs no padding; the toolbox converts it to the matrix, which is what a pre-MAET stores, what every operation returns, and what `unpackPreMaet` gives back. An attribute with one value per event is simply a vector in either form. In Python a matrix is always a NumPy array, and a list is always read per event.

Weights may be given per event in the same way: an attribute's entry in `wAttr` is then a cell or list with one entry per event, each a scalar that weights all of that event's values or a vector with one weight per value. `{[1 0.5 0.5], 1, [], 1}` gives the triad's root twice the weight of its other two notes. The matrix forms of §10.4 are accepted too.

Every function that takes a pre-MAET takes it either whole or in its parts; the two forms are the same call.

```matlab
pm  = packPreMaet(pAttr, wAttr, specs);
pmD = differenceEvents(pm, [1 0]);          % pre-MAET in, pre-MAET out
pmD = differenceEvents(pAttr, wAttr, [1 0], 'specs', specs);   % the same
```

```python
pm  = mpt.pack_pre_maet(p_attr, w_attr, specs)
pmD = mpt.difference_events(pm, [1, 0])
pmD = mpt.difference_events(p_attr, w_attr, [1, 0], specs=specs)
```

The preprocessing operators return the whole pre-MAET, as do `readPreMaet` / `read_pre_maet`; `unpackPreMaet` / `unpack_pre_maet` splits one back into the three parts for the places that want them separately. Because the specs travel with it, a composition needs no threading:

```matlab
pm2 = differenceEvents(bindEvents(pm, [2 2]), [1 1]);
```

`buildMaet`, `evalMaet`, `simMaet`, `entropyMaet`, and `massMaet` take a whole pre-MAET wherever they take a density, since it holds everything `buildMaet` needs, so `simMaet(pmX, pmY)` builds each side and compares them. `showPreMaet` and `writePreMaet` take one in place of the three positional arguments.

A *cell* (MATLAB) or *list* (Python) of pre-MAETs likewise stands wherever a cell or list of densities does, so `simMaet(pmRef, {pm1, pm2, pm3})`, `evalMaet`, `entropyMaet`, and `massMaet` take one without the caller building each entry first. This is a different kind of batching from the row-wise deduplication of a 2-D matrix of single multisets (§10.7). Rows there are commensurable — one attribute, one geometry, differing only in their values — so they collapse to one computation per equivalence class. Each pre-MAET here is a whole item with its own event count and its own per-attribute geometry, so the entries are built and compared in turn and nothing collapses: what the list form saves is the calling code, not the arithmetic.

The exception is a translation sweep, whose entries share one geometry and differ only by a known offset: `sweptSimilarity` and `sweepSimMaet` compute it in one pass (§8.7), where translated copies passed to `simMaet` are built and compared one by one. `demo_batchProcessing` contrasts the two senses of batching side by side.

Replacing one part leaves the rest in place, which is how a variant of a pre-MAET is made:

```matlab
specs2 = pm.specs;  specs2{1}.rel = true;
pm2 = packPreMaet(pm, [], specs2);
```

For a sweep, the parameter goes on the `buildMaet` call rather than into the pre-MAET. All six per-attribute parameters — `sigma`, `isPer`, `period`, `r`, `rel`, and `exch` — may be given there, and a supplied value takes precedence over the specs for every attribute, so a sweep stays one call per value and the pre-MAET it sweeps is left untouched:

```matlab
for s = [0.25 0.5 1]
    dens = buildMaet(pm, 'sigma', [s 0.25]);
end
```

An override may name every attribute, as above, or be **selective**: a length-$A$ cell (MATLAB) or list (Python) whose empty entries keep what the spec carries. This is the form to reach for in a sweep, since it names only the parameter that varies and leaves the rest to the pre-MAET, so the vector cannot drift out of step with the attributes:

```matlab
dens = buildMaet(pm, 'sigma', {[], s, []});     % sweep attribute 2 only
```

```python
dens = mpt.build_maet(pm, sigma=[None, s, None])
```

`NaN` is not available for this: it already means NA, that a preprocessing step could not carry a parameter forward (§6.3), so the empty entry carries the "keep" sense instead. `r`, `rel`, and `exch` are per-level vectors on a nested attribute, where a scalar override has no unambiguous reading, so on such an attribute they are refused and the spec is the place to change them.

`sweptSimilarity`, `sweptEntropy`, and `sweptMass` take pre-MAETs on the same terms — two for the similarity, one for the entropy and the mass — in place of their operands and their five geometry vectors, with the same six overrides:

```matlab
prof = sweptSimilarity(pmContext, pmQuery, 'sweep', {3, sweepValues}, ...
    'align', {3, 'both'}, 'sigma', {[], s, []});
```

The two pre-MAETs of a comparison describe one comparison, so they must agree on the structural geometry — `r`, `[rel]`, `[exch]`, and the nesting — and the context supplies the geometry; `sigma`, `isPer`, and `period` may differ and are taken from the context. What they need not share is their inner cardinality: each side keeps its own grouping, so a nested context whose events hold seven values compares against a nested query whose events hold two, exactly as `simMaet` compares them. A pre-MAET passed here must carry its specs, since that is where the shared geometry is read from.

**Where the pre-MAET sits among the input forms.** Every entry point that takes more than one shape of input documents them in the same order: a single multiset first, where that applies; then the pre-MAET, which is the canonical entry for everything else; then a density built by `buildMaet`; then the raw positional multi-attribute form; then the list and batched forms. The positional form is kept and supported, but the pre-MAET is the one to reach for, since it carries its own geometry and cannot fall out of step with its attributes.

For full API details on each primitive, see §12.

**Demos.** `demo_preMaetIo` / `demo_pre_maet_io.py`; `demo_batchProcessing` / `demo_batch_processing.py` (Workflow 3, a cell of pre-MAETs).

### 6.2 Viewing a pre-MAET

`showPreMaet` / `show_pre_maet` prints a pre-MAET as a table, in the layout of the pre-MAET tables of Milne (2026): one row per attribute and one column per event, the attribute's row headed by the parameters that determine its density ($\sigma$, the tuple size $r$, the `[rel]` and `[per]` flags, and the period where it is periodic), and its cells holding the elements from which the admitted tuples are formed. Because the pre-MAET is the framework's interface — the MAET follows from it mechanically — the table is a complete statement of what a subsequent `simMaet`, `entropyMaet`, or `evalMaet` call will compute.

A cell is brace-delimited where the attribute is unordered (`[exch] = 1`) and parenthesis-delimited where it is ordered; a nested attribute is bracketed level by level, the outermost level outermost, so an ordered run of unordered chords reads `({62, 65, 69, 72}, {55, 59, 62, 65}, {60, 64, 67})`. A single element is written bare at the top level but keeps its brackets inside a nest, so that the level stays visible. On an ordered attribute each slot is a level — a voice, say, or a coordinate — so an empty slot (`NaN`) before the event's last value is written as a blank, `(62, _, 67)` or `(_, 65)`, and every value stays in its slot; on an unordered attribute a slot has no identity, and empty slots are simply omitted. Where the weights are not uniform they appear as parenthesized superscripts on their values, `60^(0.6)`. An attribute carrying a kernel covariance (§13.3) is named by its shape, `sigma = 3x3 covariance`, rather than having a matrix printed in the row.

Either input form is accepted, as elsewhere in the toolbox: a density built by `buildMaet`, from which every field is recovered, or the pre-MAET itself — whole (§6.1) or in its parts — with the kernel parameters supplied alongside.

```matlab
p = {[69 69 69 71 67 66 64], [1 2 3 4 5 6 7]};
w = {[1 0.5 0.75 0.5 1 0.5 0.75], [1 0.5 0.75 0.5 1 0.5 0.75]};
showPreMaet(p, w, [], 'names', {'pitch', 'time'}, 'sigma', [0.5 0.25], ...
            'isPer', [true false], 'period', [12 0]);
```

```
| attribute                                               | n = 1  |  n = 2   |   n = 3   |  n = 4   | n = 5  |  n = 6   |   n = 7   |
|:--------------------------------------------------------|:------:|:--------:|:---------:|:--------:|:------:|:--------:|:---------:|
| pitch: sigma = 0.5, r = 1, [rel] = 0, [per] = 1, P = 12 | 69^(1) | 69^(0.5) | 69^(0.75) | 71^(0.5) | 67^(1) | 66^(0.5) | 64^(0.75) |
| time: sigma = 0.25, r = 1, [rel], [per] = 0             | 1^(1)  | 2^(0.5)  | 3^(0.75)  | 4^(0.5)  | 5^(1)  | 6^(0.5)  | 7^(0.75)  |
```

A spec may carry the attribute's kernel geometry --- `sigma`, `isPer`, and `period` --- alongside the level-structured `r`, `rel`, and `exch`, which is what makes it a complete pre-MAET in the sense of Milne (2026, Def. 2.6). `showPreMaet` then needs no further arguments, as above, and `buildMaet` reads what it is not given: see §6.3.

`maxEvents` elides the middle columns of a long passage and `maxElements` the tail of a large multiset, both after the manner of the article's own tables; `[]` (MATLAB) or `None` (Python) shows everything. The default rendering is markdown, deliberately plain ASCII so that its column widths are identical in the two languages and it survives any terminal encoding; `'format', 'latex'` emits a `booktabs` tabular in the article's own markup, with optional `caption` and `label`, so that a manuscript's pre-MAET table can be generated from the code that runs the analysis.

The demos use it throughout: every multi-attribute demo prints its pre-MAET immediately before the call that consumes it, and `demo_preprocessing` prints one after each preprocessing operation, so that each operation's effect on values, weights, and level structure is read off the table rather than described.

**Demos.** `demo_preMaetIo` / `demo_pre_maet_io.py` (part 1: markdown and LaTeX); `demo_preprocessing` / `demo_preprocessing.py` (a pre-MAET shown after each operation).

### 6.3 Where the kernel parameters live

A pre-MAET is the event sequence, its per-event–attribute element multisets, *and* the per-attribute parameters (Milne 2026, Def. 2.6). The `specs` hold all of it: the level-structured `r`, `rel`, and `exch` (per-level vectors on a nested attribute, alongside its `tags`), and the scalar `sigma`, `isPer`, and `period`.

The three scalars are **optional in the pre-MAET and compulsory at the tensor**. `buildMaet` resolves each per attribute:

- an explicit `sigma`, `isPer`, or `period` argument takes precedence outright and silently, so sweeping a width over a grid while the specs hold a baseline is the ordinary idiom and a disagreement is intent rather than error;
- otherwise the spec supplies it;
- a value missing from both places is an error naming the attribute. `period` alone defaults to 0, being inert on a non-periodic attribute.

`NA` (`NaN`) is a third state, distinct from absent. A preprocessing step that cannot carry a parameter forward writes NA rather than a stale or invented value, `showPreMaet` prints `sigma = NA`, and `buildMaet` refuses it with a message that says a step could not carry it forward — so the gap is visible in the table before it is fatal at the build. Supplying the parameter at the call resolves it.

The five preprocessing operators carry the geometry as follows.

| operation | `sigma` | `period` |
|:--|:--|:--|
| `bindEvents`, `translateAttributes`, `weightEvents` | unchanged | unchanged |
| `differenceEvents`, order $k$ | $\times \sqrt{\binom{2k}{k}}$ | unchanged |
| `transformAttributes`, affine or within `midi`/`cents`/`octave` | $\times$ the gain | $\times$ the gain |
| `transformAttributes`, anything non-linear | NA | NA |

A $k$-th finite difference is the alternating binomial sum, so a value of width $\sigma$ whose per-event errors are independent yields a difference of width $\sigma\sqrt{\binom{2k}{k}}$ — the $\sqrt{2}$ of a first difference, the $\sqrt{6}$ of a second. Independence is a modelling assumption, so `differenceEvents` announces the scaling rather than applying it silently. A kernel covariance is in squared units and takes the factor itself where a width takes its root.

Under a non-linear map, NA does not mean a width would be meaningless in the new coordinate: a $\sigma$ on a log axis is perfectly meaningful, expressing a ratio rather than a difference. It means there is no *canonical* value to carry over, the local scaling varying across the attribute's range, so the toolbox declines to choose and the analyst supplies the width the new units call for.

`preMaetFromAttrTable` sets `isPer = 0` and `period = 0` — a score states its attributes' periodicity, octave equivalence being an equivalence the analyst imposes — and leaves `sigma` unset, since no width is implied by a score. `buildMaet` then names the attribute that still needs one.

**Demos.** `demo_preMaetIo` / `demo_pre_maet_io.py` (parts 3, 4, and 6: nothing supplied, overrides, and NA).

### 6.4 Reading and writing a pre-MAET as CSV

A pre-MAET is a table, and a table is a spreadsheet: `readPreMaet` / `read_pre_maet` and `writePreMaet` / `write_pre_maet` move one between the toolbox and a CSV that Excel or Numbers can edit. The cells use the notation of the article and of `showPreMaet`, so the writer and the reader are inverse and a pre-MAET survives a round trip byte for byte. `showPreMaet(..., 'format', 'csv')` produces the same text without writing a file.

The header is fixed — `name, sigma, r, rel, per, P, exch` — followed by one column per event, whose own headings are free text:

```
name,sigma,r,rel,per,P,exch,n = 1,n = 2,n = 3
pitch,0.5,2,0,1,12,1,"{60, 64, 67}","{62, 65, 69}","{60, 64, 67}"
onset,0.25,1,0,0,,1,0,1,2
```

which reads straight into a density: `buildMaet(p, w, 'specs', specs)`, nothing further supplied.

Conventions worth knowing when writing one by hand:

- `r`, `rel`, and `exch` take a parenthesized tuple on a nested attribute, innermost level first, as the article writes them.
- **Tags are never written.** A nested attribute's level structure is recovered from the bracket structure of its cells, so `"({60, 64, 67}, {62, 67, 71})"` at `r = "(1, 2)"` is enough.
- Events of different sizes are NaN-padded to the widest on reading and written back ragged. On an ordered attribute a blank `_` marks an empty slot before an event's last value, so `(_, 65)` puts 65 in the second slot; trailing empty slots need no blank.
- A weight is written `60^(0.6)`. An attribute whose weights are all exactly 1 is written bare, since a bare value already means unit weight.
- An empty parameter cell is absent; `NA` is NA.
- A **kernel covariance** is carried as the flag and three scalars that generate it, `cov(differenced=1, sd_value=0.2, sd_interval=0.3, sd_shift=0.5)`, the row's own `r` giving the matrix's size. Both languages write that spelling and read either it or `sdValue`. A covariance outside that family has no such scalars and is refused with that reason.
- CSV elides nothing, whatever `maxEvents` says: a file records the pre-MAET rather than displaying it.

`demo_preMaetIo` / `demo_pre_maet_io` works through all four renderings — markdown, LaTeX, CSV out, CSV back — on one pre-MAET, and covers overriding a parameter the pre-MAET carries, nesting, NA, a covariance, and a pre-MAET taken from a score.

**Demos.** `demo_preMaetIo` / `demo_pre_maet_io.py` (part 2: the round trip).

---

## 7. Preprocessing

The toolbox provides eight preprocessing functions that take the pre-MAET feeding `buildMaet` and return a transformed pre-MAET: each takes it whole, as `packPreMaet` builds it (§6.1), or in its parts as `pAttr` and `wAttr` with the `specs` alongside. They are `differenceEvents` and `bindEvents` (across events); `translateAttributes`, `transformAttributes`, and `weightEvents` (per event); `bindAttributes` and `separateAttributes` (across attributes); and `selectPreMaet` (a selection of the attributes and the events). `addSpectra` in its pre-MAET form is a ninth. The `specs` hold the per-attribute level geometry; they are optional and assumed flat when omitted.

Before a pre-MAET is mapped to a density, it can be processed to pose a particular musical question: comparing interonset intervals rather than absolute times, isolating a local passage, matching material up to transposition, or representing each notated pitch as its sounded spectrum. The toolbox provides six such operations, following Milne (2026): *event binding*, *event differencing*, *attribute rescaling*, *attribute translation*, *event weighting*, and *spectral enrichment*. They form a closed family — each takes one pre-MAET and returns another, chaining directly into further preprocessing or into MAET construction — and divide by their scope of action:

**How the operations are named.** The suffix names what the operation is a map *of*, not which attributes a given call happens to touch. An `…Attributes` operation is a pointwise map of an attribute's values, writable without reference to the event index — $p'_a = g_a(p_a)$ for rescaling, $p'_a = p_a + \mu_a$ for translation — and it changes the attribute's scale or origin while leaving the events alone. An `…Events` operation is defined across events and cannot be written without their order: binding gathers $L$ consecutive events, differencing takes the difference between successive ones, and weighting sets a per-event weight $w'(n)$. Every operation is attribute-selective, taking per-attribute orders, offsets, or targets, so that shared property distinguishes nothing and is not what the suffix records. `addSpectra` stands outside the pair, named after itself rather than after the pair, as the sixth operation is in Milne (2026).

A second distinction, which cuts across the naming and concerns how each operation computes rather than what it maps:

- **Cross-event** (each output entry depends on several input events, so the event sequence is reshaped): `bindEvents` gathers $L$ consecutive events into a single nested super-event; `differenceEvents` replaces the sequence with inter-event differences.
- **Per-event** (each event is updated independently, and event count, ordering, and per-event correspondence are preserved): `transformAttributes` maps selected attributes' values through a named transform, a scale conversion, or a user function; `translateAttributes` shifts selected attributes' values by an offset; `weightEvents` multiplies a window factor into the per-event weights; `addSpectra` replaces each pitch with its partials.

All act per attribute (every attribute carries its own self-contained geometry in `specs`; there is no separate group argument), with a scalar order/offset/transform broadcasting across attributes as a common case. Per-event weights propagate consistently through all of them under the toolbox's standard probability-of-perception reading. Multi-value attributes ($K_a > 1$) are accepted by `bindEvents`, `translateAttributes`, `transformAttributes`, and `weightEvents`'s target attribute; `differenceEvents` requires $K_a = 1$ for the reasons given in §7.2, and `weightEvents`'s input attribute is similarly restricted to $K = 1$ (a multi-value driver would not yield a well-defined per-event scalar factor).

Each operation is useful on its own, and they compose freely. Operations targeting disjoint attribute sets commute by construction (each acts only on the slices of the per-attribute value and weight matrices corresponding to its targets), while operations on overlapping targets carry order-dependence that reflects different analytic readings of the same data. Two canonical compositions — *windowed entropy* (`weightEvents` followed by the entropy of the resulting density) and the *translation sweep* (a query translated over a range of offsets and compared with a context at each) — are described in §7.7, and the swept functions that package them, `sweptSimilarity`, `sweptEntropy`, and `sweptMass`, in §8.7.

### 7.1 Event binding

`bindEvents` gathers $L$ consecutive events into a single nested super-event, sliding a window of width $L$ across the input. Each window becomes one *nested* output attribute: the bound events form an **ordered outer level** (carrying event order, so the binding is lossless), with the original per-event values at the inner level. The output is the transformed pre-MAET — its `specs` now encoding the two-level (inner/outer) geometry — feeding straight into `buildMaet`. The outer level defaults to reading the whole window ($r = L$), ordered (`[exch] = 0`, so ordered tuples are not collapsed to unordered ones), and absolute; making the outer level relative (`relOuter = 1`) gives a within-window transposition-invariant n-gram. Per-event weights propagate so that a bound super-event's effective weight is the product of its $L$ constituent events' weights — matching the differencing rule under the toolbox's standard probability-of-perception reading. The default output has $N - L + 1$ super-events; a `circular` option produces $N$ by wrapping around the end of the input, the natural choice for cyclic event sequences. Each super-event is end-aligned to its last event, as in the article (Sec. 3) and as `differenceEvents` is: when attributes are bound over different numbers of events, $L_a$, each super-event spans $\max_a L_a$ consecutive events and an attribute bound over fewer contributes the last $L_a$ of them, so an attribute bound over a single event carries the element multiset of the span's last event. Under this alignment binding and differencing commute: differencing then binding gives the same pre-MAET as binding then differencing, circular or not. Unlike `differenceEvents`, `bindEvents` accepts any $K_a \ge 1$: each element of the underlying events is carried through the nesting, preserving within-attribute element exchangeability.

Where the material does not come in uniform blocks, `'groupBy'` binds by a run of equal values instead: one attribute names the grouping, consecutive events sharing its value become one super-event, and a new group begins wherever that value changes. The groups may then differ in length, so the outer level is ragged — padded to the largest group's size with NaN values at zero weight, which the nested inner product consumes — and `'rOuter'` says how many positions a tuple takes, defaulting to the smallest group's size so that every group contributes tuples of one size into one density. A bar of notes, a chord's derivation path, a gesture's frames: what the analysis treats as one object is read from the data rather than fixed by a window width.

Differencing and binding compose. Differencing-then-binding gives joint distributions of $n$-tuples of consecutive inter-event differences: $n$-grams of melodic intervals when the attribute is pitch, of inter-onset intervals when it is time. With both operations in their circular variants on a cyclic input, the composition recovers the n-tuple entropy of Milne & Dean (2016) as a special case (uniform weights, integer-valued steps, periodic domain, $\sigma \to 0$, integer-step grid) while extending it to the smoothed continuous case, non-integer values, weighted events, and non-periodic domains; and for $n \ge 2$ the bound MAET is itself an $n$-dimensional density, supporting cosine-similarity comparison of $n$-tuple distributions across pieces and the rest of the toolbox pipeline. The convenience wrapper `nTupleEntropy` calls this pipeline with default arguments matching the original Milne & Dean formulation.

Binding alone, applied to the raw event values without a preceding differencing step, gives n-grams in absolute pitch or time register, useful when register or absolute timing carries musical information that the differenced view discards: a leitmotif identified by its specific octave, an onset pattern keyed to a metric position, a chord progression as a sequence of identified harmonic functions. To localize the n-grams in time, an explicit time or event-index attribute can be carried alongside the bound categorical or pitch attributes, with the time stamp travelling with each window: bound over a single event, it is the last constituent's onset (the end-alignment above), and bound over all $L$ events, it is the ordered tuple of their onsets, from which `locate` in `sweptSimilarity` and `sweptEntropy` picks the position.

**Demos.** `demo_preprocessing` / `demo_preprocessing.py` (parts 3 and 3b); `demo_rhythmTensors` / `demo_rhythm_tensors.py` (part 4: circular binding); `demo_jmm_1_3_cadence_nesting` (JMM); `demo_jmm_2_1_joint` (JMM).

### 7.2 Event differencing

`differenceEvents` replaces a sequence of events with a sequence of inter-event differences. Some analyses model events as relative to their predecessors — a sequence of melodic intervals rather than absolute pitches, a sequence of inter-onset intervals (IOIs) rather than absolute time points, a sequence of interval changes rather than intervals — so the tensor represents the distribution of these relative quantities. `differenceEvents` performs this transformation: for each attribute assigned a differencing order $k$ it replaces the input values with their $k$-th finite differences across events (reducing the event count by $k$). Raw signed differences are emitted regardless of periodicity; periodic attributes' mod-period wrap is handled by the kernel at MAET-construction time. Weights propagate as rolling products under the toolbox's standard broadcast convention (§10.4), so the weight of a $k$-th-order difference is the product of its $k + 1$ constituent events' weights — interpretable as the probability that all constituents are jointly perceived under the standard weights-as-salience reading. Scalar-$c$ and vector-of-$c$ inputs therefore produce equivalent downstream densities. The default output drops the leading $k$ events of each attribute; a `circular` option (paralleling the same flag on `bindEvents`) wraps the difference operator at the sequence boundary instead, retaining $N$ events at every order. The circular variant is the mathematically sensible choice for cyclic event sequences (looped rhythms, ostinati) in which the boundary difference is a genuine inter-event interval rather than an artefact of truncation. The transformed pre-MAET feeds directly into `buildMaet`.

Event differencing is distinct from `isRel = true`, and the two can be used together or independently. Event differencing is a preprocessing step that replaces absolute per-event values with differences between adjacent events, so the tensor is constructed from inter-event quantities. `isRel = true` is a property of the tensor construction itself — it makes the within-r-ad density translation-invariant, so that (for example) a dyad tensor at $r = 2$ represents the distribution of intervals between any two elements of the multiset (not only adjacent events). Using event differencing gives a density over sequential differences; using `isRel = true` gives a density over all r-ad interval patterns within the (possibly differenced) input. The two give different interval-based characterizations, and both are analytically supported.

Event differencing requires each attribute to have $K_a = 1$ element per event. The operation is column-wise subtraction across adjacent events, which imposes a cross-event element correspondence (row $i$ at event $n-1$ is paired with row $i$ at event $n$). Within-event element exchangeability — the MAET's treatment of elements within an attribute as interchangeable — does not guarantee such a correspondence, so for multi-element attributes the output would depend on an arbitrary row-listing choice. `differenceEvents` therefore raises an error on any attribute with $K_a \ne 1$.

For analyses that might seem to require multi-element differencing — for example, the voice-agnostic distribution of melodic step-sizes across a polyphonic texture — the principled route is to encode each voice as a separate $K_a = 1$ attribute carrying shared pitch geometry, call `differenceEvents` (each attribute differences canonically), and then stack the differenced attributes into a single multi-element attribute before `buildMaet`. This makes the analytical choice explicit: step 1 preserves voice labels, step 2 produces well-defined per-voice step-sizes, and step 3 deliberately discards voice labels to yield a voice-exchangeable representation. A short MATLAB example:

```matlab
pAttr      = {pS, pA, pT, pB};       % four voices, one K=1 attribute each
pmDiff     = differenceEvents(pAttr, [], 1);   % order-1 difference, all attributes (returns a pre-MAET)
pBundled   = { vertcat(pmDiff.pAttr{:}) };  % stack the four differenced voices into one K=4 attribute
dens = buildMaet(pBundled, [], sigma, r, true, false, 0);
```

**Demos.** `demo_preprocessing` / `demo_preprocessing.py` (part 2); `demo_rhythmTensors` / `demo_rhythm_tensors.py` (part 4: circular differencing); `demo_repetitionHandling` / `demo_repetition_handling.py`; `demo_jmm_3_2_diff` (JMM: joint differencing of pitch and time).

### 7.3 Attribute rescaling

`transformAttributes` maps every value of selected attributes through a transform, leaving non-selected attributes unchanged. The operation is *attribute rescaling* in Milne (2026): it changes the scale on which the kernel's width $\sigma_a$ is measured, so that a constant $\sigma_a$ can express a constant frequency difference, a constant musical interval, a constant auditory-filter distance, or a constant ratio, according to the scale chosen. The map is elementwise, so weights pass through unchanged (a monotone change of scale does not alter any event's probability of perception) and the operation is well-defined for any $K_a \ge 1$. Three kinds of transform are accepted, one per attribute or a single entry broadcast across attributes: a **named transform** — `'log'` (with a `base`, default $e$, and an `offset`, default 0: $\log(x + \text{offset})$), `'power'` (with an `exponent`), and `'affine'` (`scale`, `offset`); a **pitch-scale conversion** given as a pair from `'hz'`, `'midi'`, `'cents'`, `'octave'`, `'mel'`, `'bark'`, `'erb'`, `'greenwood'`, every pair routing through Hz; or a **user function** applied to the attribute's $K_\text{total} \times N$ value matrix. A bare numeric array in place of the pre-MAET is treated as a single attribute and returned as an array, which is the one-line form for converting a vector of frequencies to cents before any MAET call.

The function is the toolbox's home for the choice of measurement scale, and that choice interacts with the rest of the pipeline in three ways worth stating. First, the scale is chosen *before* the kernel: the Gaussian of `buildMaet` has a fixed width in whatever units the values carry, so a logarithmic transform of inter-onset intervals or a conversion of frequencies to cents is what makes a single $\sigma$ mean the same thing at every point of the range (the flat-metric point of §16). Second, the order relative to differencing carries meaning: `'log'` *then* `differenceEvents` gives log ratios (the natural representation of IOI ratios, and of intervals from frequencies in Hz), whereas `differenceEvents` *then* a compressive transform gives signed compressed magnitudes. Third, only `'affine'` is compatible with a periodic attribute (`isPer = true` at build); the other transforms change the metric and cannot be wrapped.

Values outside a transform's domain are refused rather than silently mapped to $\pm\infty$ or `NaN`, with a message that names the attribute and events and lists the remedies. The commonest case is a zero under `'log'`, typically the inter-onset interval of a chord or grace note: bind simultaneous events first (`bindEvents`), drop those events deliberately, or admit them with an explicit `offset` — `{'log', 'offset', c}` computes $\log(x + c)$, which is defined at zero but not unit-free, so the unit of the input is then part of the model, and the constant is written down rather than hidden. Negative values under a magnitude transform (`'log'` or `'power'`) — melodic intervals after differencing, say — are handled by the `sign` option: the transform is applied to $|x|$ and a **sign attribute** with values in $\{-\tfrac12, 0, +\tfrac12\}$ is inserted immediately after the source attribute — the two signs are the vertices of the 2-point simplex at the toolbox's default unit edge length (`simplexVertices(2)` is $[+\tfrac12, -\tfrac12]$), with a zero step at the centroid, so the attribute is on the same scale as any other categorical — carrying the source's spec with `rel` cleared and the name suffixed `'_sign'`; a small $\sigma$ on that attribute makes it effectively categorical. The insertion is explicit rather than automatic because it changes the attribute count, and every downstream per-attribute argument (`sigma`, `r`, `rel`, `exch`, `wrap`, `diffOrders`, …) must then include the new column. The set of named transforms is deliberately small: a change of unit or origin is `'affine'`, a compression is `'log'` or `'power'`, and anything else (a reciprocal, a modular reduction, a standardization against a fixed reference, which is an `'affine'` in disguise) is a one-line user function whose intent is then visible at the call site.

**Demos.** `demo_preprocessing` / `demo_preprocessing.py`; `demo_tempoInvariance` / `demo_tempo_invariance.py` (logarithmic inter-onset intervals); `demo_repetitionHandling` / `demo_repetition_handling.py` (part 4).

### 7.4 Attribute translation

`translateAttributes` shifts selected attributes' values by a chosen offset, leaving non-selected attributes unchanged. The transformed pre-MAET feeds directly into `buildMaet`; weights and `specs` pass through unchanged, since uniform translation does not affect any event's probability of perception. The shift is per-row — one position per row of an attribute, held constant across events — so the operation is well-defined for any $K_a \ge 1$; unlike windowing, where a per-event scalar weight would have to summarize multiple elements' distances from the value at which the window is aligned, translation has no analogous obstruction at $K_a > 1$. Relative geometry is read per-attribute from `specs`. On an absolute attribute the operation is `value + mu`, regardless of periodicity: the wrapped periodic Gaussian kernel of `buildMaet` is invariant under an additive shift by a multiple of $P$, so periodic attributes need no canonical wrap of the translated values — the kernel handles periodicity downstream. On an attribute whose outermost level is relative, a *uniform* shift is a structural no-op — it cancels in every within-tuple difference, so the relative MAET is unchanged — and the function emits a `translateAttributes:noOp` (MATLAB) or `TranslateAttributesNoOpWarning` (Python) and leaves that attribute unchanged; a *non-uniform* (per-row) offset is not a no-op even on a relative attribute and is applied.

`offsets` is a **length-$A$ list** (Python) / $1 \times A$ cell (MATLAB), one entry per attribute. Each entry is `None` / empty `[]` (skip this attribute), a scalar or length-1 vector (broadcast to every row — a global transposition), or a length-$K_\text{total}$ vector (per-row). `NaN` entries skip the corresponding row; $\pm\infty$ is rejected.

One call makes one translation. **Swept translation similarity has two routes: `sweptSimilarity`**, which takes pre-MAETs and sweep values and handles windows and dropped attributes as well (§8.7), **and `sweepSimMaet`**, which takes built densities and an offset matrix. Both compute every offset in one pass (§7.7).

**What an offset means.** An offset is measured from the query's values as given: at offset $\mu$ every element of the translated attribute sits at its given value $+ \mu$, so $\mu = 0$ is the query where it is written, and a sweep's profile is indexed by $\mu$ (the article, §3: "sweeping the offset across a range … traces a cross-correlation profile"). For a sweep over time the natural reading follows from one convention: **before any preprocessing, align the query's first time value with the context's first time value**, by writing it that way or with one translation,

```matlab
pmQuery = translateAttributes(pmQuery, {[], c1 - q1});   % c1, q1: first time values
```

An offset is then the time from the start of the context to the start of the query, which is the query's start time in the context when the context starts at 0. The alignment belongs before preprocessing, not after it. Differencing and binding drop leading events but leave every surviving value where it was, so an offset set up this way keeps its meaning through them, and profiles of one query with and without differencing share one axis; aligning the first values that survive differencing would instead build each sequence's first inter-onset interval into the origin, and move every peak by the difference between the query's and the context's. The convention applies to time values that are positions: where the translated attribute is itself differenced (inter-onset intervals) or rescaled (a logarithmic scale), an offset is in those units, and reads accordingly. `sweepSimMaet` takes offsets in this sense; `sweptSimilarity` takes sweep values, the points where the query's reference value lands; with no window these are these offsets by default (`queryRef` 0), and with a window the offsets are returned alongside the profile (§8.7).

**Demos.** `demo_preprocessing` / `demo_preprocessing.py` (part 4); `demo_sweptSimilarity` / `demo_swept_similarity.py`.

### 7.5 Event weighting

`weightEvents` computes a per-event window factor from the values of one attribute (the *input attribute*) and multiplies it into the weights of another (the *target attribute*). Values and `specs` pass through unchanged, except for the optional deletion of the input attribute discussed below; only the target attribute's weights are updated.

**Input and target.** The signature names two attribute indices: `inputAttr`, the attribute whose values drive the window function, and `targetAttr`, the attribute whose weights receive the resulting per-event factor. They may be equal (the input attribute weights itself — e.g., selecting pitch events by a range of pitches) or distinct (the cross-attribute case — e.g., a time-Gaussian factor applied to pitch events for windowed-entropy analyses). Where an event holds several values on the input attribute (the onsets of a bound super-event, say), `locate` picks the one the window is evaluated at: `'centroid'` (their mean, the default), `'start'`, `'end'`, `'mid'`, or a function of the $K \times N$ value matrix. The per-event factor is therefore always a $(1, N)$ row, broadcast across the target's $K_{\text{target}}$ rows in `buildMaet`'s downstream kernel product.

**Window family.** Three numeric parameters specify the window: a centre $c$, a width $w$ (the standard deviation, in absolute units of the input attribute), and a shape parameter $\gamma \in [0, 1]$. The factor is the peak-normalized convolution of a rectangle and a Gaussian, parameterized so that the total variance is $w^2$ for every choice of $\gamma$:

- The Gaussian component has standard deviation $w \sqrt{1 - \gamma}$.
- The rectangle component has half-width $w \sqrt{3 \gamma}$ (so its variance is $w^2 \gamma$).

The two extremes are the pure Gaussian ($\gamma = 0$: $h(\delta) = \exp(-\delta^2 / (2 w^2))$) and the pure rectangle ($\gamma = 1$: $h(\delta) = \mathbf{1}[\,-w\sqrt{3} \le \delta < w \sqrt{3}\,]$, half-open so that a regular pulse grid yields exactly $N$ pulses for full support $N \cdot \mathrm{IOI}$); intermediate $\gamma$ interpolates continuously between them. The parameter $w$ always means the standard deviation, not the half-extent.

For periodic input attributes, only the centred difference $\delta = v - c$ used inside $h$ is wrapped to $[-P/2, P/2]$; the stored values stay raw. The kernel in `buildMaet` handles the periodicity of the values downstream via the attribute's `[per]` flag.

**Profiles beyond the window family.** The shape argument also accepts a named exponential or a function of the centred difference, so the factor need not be a symmetric window. `'exponentialBefore'` decays for $\delta \le 0$ and is zero above the centre, `'exponentialAfter'` mirrors it, and `'exponential'` decays in both directions; each takes `sd` alone, since an exponential has no finite support for a `width` to describe, and each is scaled so its standard deviation is `sd`, as the convolution family is. A centre at the last event's value with `'exponentialBefore'` is a recency profile, and a centre at the first with `'exponentialAfter'` a primacy profile. A function handle (Python: any callable) receives $\delta$ and returns one non-negative factor per event; it carries its own scale, so neither `sd` nor `width` is accepted with it, and the kernel truncation is not applied to it, since an arbitrary profile need not decay.

**Serial-position profiles.** Four further names anchor themselves at the first and last events' values rather than at a centre, which must then be `NaN` (Python: `None`): `'exponentialFromStart'` and `'exponentialFromEnd'` decay away from one anchor, and `'uShape'` and `'uAsym'` mix both, `alpha` weighting the primacy component against the recency one — `alpha = 1` is pure primacy, `alpha = 0` pure recency. `'uAsym'` takes `decayRateStart` and `decayRateEnd` separately, each falling back to `decayRate`; the others take one rate. Every named profile is scaled by either `sd` or `decayRate`, its reciprocal, and defaults to a rate of 1.

Applied to an event-number attribute — add $1, \dots, N$ as an attribute, drive the profile from it, and drop it with `dropInputAttr` in the same call — these are profiles over position. Applied to a time attribute they are profiles over elapsed time, which needs no extra attribute where a time attribute is already present and is usually the better model: a recency profile over onsets weights a held final chord more heavily than a rapid passing note, which position indexing cannot express.

**`dropInputAttr` (mandatory keyword).** A boolean flag — keyword-only, no default — selects whether the input attribute is retained in the returned pre-MAET or removed. When `dropInputAttr = true` and `inputAttr != targetAttr`, the input attribute is dropped from all three output components and the higher attribute indices are decremented to keep numbering contiguous. When `dropInputAttr = false`, the input attribute is preserved unchanged in the output. `dropInputAttr = true` paired with `inputAttr == targetAttr` raises an error: deleting the input would discard the factor just written to it. The auto-delete pattern is the standard idiom for windowed-entropy and windowed-mass analyses (`sweptEntropy`, `sweptMass`, §8.7) where the input attribute (typically time) provides the scaffolding for the window and is no longer needed downstream once its positions have been transferred to the target's weights.

**Windowing on several attributes.** A single `weightEvents` call windows on one input attribute. To window on several attributes simultaneously — say, time *and* register, both targeting pitch — call `weightEvents` twice in sequence with the same `targetAttr`. Each call's factor multiplies into the target's existing weights, so the composition naturally gives the product window. Set `dropInputAttr = false` on intermediate calls (so the next call's input is still present), and `dropInputAttr = true` on the last (to drop both inputs at once is then a matter of a final lookup or a separate composition step).

**Centre-shift identity.** When event weighting and attribute translation target overlapping attributes, the operations commute up to a shift of the window centre: applying `translateAttributes` by an offset $\mu$ after `weightEvents` centred at $c$ gives the same pre-MAET as applying `weightEvents` centred at $c + \mu$ after the translation. The factor evaluates on the same residual $v - c$ either way. This identity underlies the windowed-entropy and pre-tensor sliding-comparison constructions of §7.7.

**Demos.** `demo_preprocessing` / `demo_preprocessing.py` (part 5); `demo_probeTone` / `demo_probe_tone.py` (part 3: recency weighting); `demo_jmm_1_1_entropy` (JMM: a window swept through a chorale).

### 7.6 Spectral enrichment

A single musical pitch is not a pure tone — it has a spectrum of partials. The `addSpectra` function models this by adding partials to each pitch in a set, returning expanded pitch and weight vectors. These expanded vectors can then be passed to the expectation tensor functions. Spectral enrichment is the one component of the toolbox that is specific to pitched sounds; the expectation tensor framework itself applies to any domain.

The five spectral modes model different physical and psychoacoustic scenarios:

- **Harmonic** — Standard harmonic series: partial n has frequency ratio n relative to the fundamental.
- **Stretched** — Uniform log-frequency stretch: ratio(n) = n^β. With β > 1, partials are wider apart than harmonic; β < 1 compresses them. Useful for matching spectra to non-standard tuning systems.
- **Freqlinear** — Frequency-domain stretch: ratio(n) = (α + n) / (α + 1). A single parameter α controls the departure from harmonicity in the frequency domain. The stretching is non-uniform in log-frequency (unlike `'stretched'`), because it arises from a linear perturbation in the frequency domain. Common in psychoacoustic experiments.
- **Stiff** — Stiff-string inharmonicity: ratio(n) = n√(1 + Bn²). Models the progressive sharpening of partials in stiff vibrating strings (e.g., piano). B is the inharmonicity coefficient (typically 10⁻⁵ for bass strings to 10⁻³ for treble).
- **Custom** — Arbitrary user-specified partial offsets and weights. Can represent any spectrum: harmonic, inharmonic, empirical, or theoretical.

Two weight decay options are available for all non-custom modes:
- **Powerlaw**: weight(n) = 1/n^ρ (ρ = 0: flat; ρ = 1: sawtooth; ρ = 2: approximates many acoustic instruments)
- **Geometric**: weight(n) = τ^(n−1) (τ = 1: flat; τ = 0.5: 6 dB per partial rolloff)

**Demos.** `demo_overview` / `demo_overview.py` (part 1a); `demo_triadConsonance` / `demo_triad_consonance.py`; `demo_jmm_2_3_spectral` (JMM).

### 7.7 Compositions and canonical uses

The preprocessing operations chain freely. Operations targeting disjoint attribute sets commute by construction, since each operation acts only on the slices of the per-attribute value and weight matrices corresponding to its targets. On overlapping targets, order dependence reflects different analytic questions about the data. Two compositions are canonical and named.

**Windowed entropy / local density characterization.** `weightEvents` followed by `buildMaet` produces a pre-MAET in which each tuple's contribution is scaled by the product of the per-event window factors at its constituent events. `entropyMaet` on this windowed density gives the entropy of the local slice of the density visible through the window, useful for characterizing how a passage's local pitch or time content compresses or spreads at different positions or registers. The window scale $w$ and shape $\gamma$ are chosen per call, and a window on several attributes is composed by chaining `weightEvents` calls with the same `targetAttr` — a tight Gaussian in time chained with a broad rectangular in register, both targeting pitch, picks out the rhythmic content of a notes-of-any-pitch window. Sweeping $c$, the value at which the window is aligned, over a grid of positions gives the corresponding local-entropy profile along the window's input attribute.

**Translation sweep.** A query is translated over a grid of offsets $\mu$ and compared with a fixed context at each; the resulting similarity profile shows how its similarity to the context varies along the swept attribute. It uses the query's intrinsic support as its locality, with no externally chosen window, and under `'cosine'` it is symmetric and bounded in $[0, 1]$. **The routes for a swept translation similarity are `sweptSimilarity`** (pre-MAETs and sweep values; its default placement, §8.7) **and `sweepSimMaet`** (built densities and an offset for each swept attribute at each step). Both compute the whole profile in one pass, settling the tuple pairs once and then evaluating each offset (as a Gaussian mixture in the offset, or on the orbit route for a periodic swept attribute; §8.7, "Translation sweeps in one pass"). `translateAttributes` makes a single translation, for use in a pipeline; translating copies one by one and comparing each with `simMaet` gives the same values at many times the cost.

Translation and windowing answer different questions. Pre-tensor translation is the natural choice when two whole densities are being aligned under an unknown transposition or time shift (e.g., cadence-progression comparison up to musical transposition): the query's own support is the matching scale, and the result is the symmetric cosine. A windowed sweep (§8.7) is the natural choice when a fixed query is being matched against a longer context with a controllable inspection scale (e.g., motif-finding within a chorale): an externally chosen window on the context decouples the comparison region's scale from the query. On periodic attributes both routes are well-defined and interchangeable for whole-density alignment; on relative attributes a uniform translation is a no-op, and a window on such an attribute would select in interval-size space, an unusual analytic move that is rarely the question one is asking.

Windowing in the toolbox is always event weighting of this kind: the window multiplies the per-event weights before the density is built, and the attribute the window is defined over is then marginalized (dropped) or compared as the analysis requires. It is not a multiplication of a finished density by a window function. Because the window acts on the events rather than on the kernel-smoothed density, no smoothed mass leaks across the window boundary, and the closed-form cosine similarity, every entropy estimator, and matrix-valued kernel covariances all apply to the windowed density unchanged.

**Demos.** `demo_preprocessing` / `demo_preprocessing.py` (the operations composed, with selectPreMaet, bindAttributes, and separateAttributes); `demo_rhythmTensors` / `demo_rhythm_tensors.py` (part 4: circular differencing and binding against nTupleEntropy); `demo_jmm_3_2_diff` (JMM).

---

## 8. Densities and measures

This section covers building a density from a pre-MAET, reading it — its value at points, its similarity to another, its entropy, its mass in a region — drawing it, and taking any of these measures at each of a list of values along an attribute. The function entries are in §12.4, and the mathematics of the density in §13.

### 8.1 Building a density

`buildMaet` turns a pre-MAET into a density, the MAET. It enumerates the tuples each attribute admits at each event and stores their weight products, so the density can then be evaluated, compared, and measured without repeating that work. It takes the pre-MAET whole (`buildMaet(pm)`), with any of the six per-attribute parameters (`sigma`, `isPer`, `period`, `r`, `rel`, `exch`) given alongside to override what the specs carry (§6.1); or, for a single multiset, the values, weights, and five parameters positionally (`buildMaet(p, w, sigma, r, isRel, isPer, period)`).

Each attribute's parameters decide what it represents: its kernel width σ, the perceptual uncertainty of its values; its tuple size `r`, whether the density is over single values, pairs, triples, and so on; whether it is relative (invariant to transposing the whole tuple) or absolute, periodic (pitch class, metrical position) or not, and ordered or exchangeable. §13 gives the density and its modes in full, and §13.3 the matrix-valued kernel covariances an ordered attribute may carry.

Every measure below also takes a pre-MAET, or the raw values, wherever it takes a density, and builds the density on the spot. Building it once yourself pays when the same density enters several calls (§10.5).

**Demos.** `demo_overview` / `demo_overview.py` (parts 1 and 2); `demo_maetPlots` / `demo_maet_plots.py`; `demo_dispatchAndKernelControls` / `demo_dispatch_and_kernel_controls.py` (parts 1–2: how a density is evaluated).

### 8.2 Evaluating a density

`evalMaet` gives the density's value at query points, the columns of a matrix with one row per coordinate: `r` rows for an absolute attribute and `r − 1` for a relative one, whose coordinates are the tuple's values above its first. Three normalizations are available: `'none'` (the default), the raw sum of kernels, whose scale depends on σ; `'gaussian'`, each kernel integrating to 1, so that the density integrates to the sum of its tuples' weight products; and `'pdf'`, a probability density integrating to 1. `maetCentres` returns the points at which the kernels sit rather than what the density is worth there.

**Demos.** `demo_preprocessing` / `demo_preprocessing.py`; `demo_jmm_2_1_joint` (JMM).

### 8.3 Comparing densities: similarity

The cosine similarity of two densities is the toolbox's central measure of resemblance. `simMaet` computes it in closed form, so no grid or resolution enters. The family of measures it generates is named by two independent choices: whether the pitches are spectrally enriched (S), and whether the domain is periodic, giving pitch *classes* (C). Spectral pitch class similarity (SPCS) is the best validated of these, predicting probe-tone ratings, tonal affinity in microtonal and inharmonic settings, and perceived triadic distance. The same measure applies unchanged to time points, where periodicity gives metrical position rather than pitch class, and to any combination of attributes.

`'normalize'` sets the denominator. `'cosine'` (the default) scores the match of shape alone, bounded in $[-1, 1]$ and unchanged by rescaling either density's weights. `'oneSidedDenom'` divides by the second density's self inner product only, so it scores how much of the second is present in the first: 1 on a self-match, and more where the first carries more matching mass. `'none'` returns the bare inner product. `simMaet` takes two densities, two pre-MAETs, or the raw values; one against a list (one value per entry); and batched matrices, one pair per row (§10.6). Two densities compared must share their attribute structure and per-attribute parameters (§13.2).

**Demos.** `demo_overview` / `demo_overview.py` (part 1b); `demo_triadSpcsGrid` / `demo_triad_spcs_grid.py`; `demo_scoreCategoricals` / `demo_score_categoricals.py` (what each encoding asks of a re-voiced chord); `demo_softeningEquivalences` / `demo_softening_equivalences.py`; `demo_jmm_1_2_similarity` (JMM); `demo_jmm_1_4_tonic_tuple_size` (JMM).

### 8.4 Entropy

`entropyMaet` measures how evenly a density's mass is spread, by one of four estimators (§11.2): `'shannon'` and `'normalized'` on a grid, the latter the ratio in $[0, 1]$ that reproduces published values; `'differential'`, adaptive and grid-free from the caller's point of view; and `'renyi2'`, the collision entropy in closed form. The last two are scale-free, and so comparable across densities of different size, spread, or support; `'renyi2'` is also the fastest, and works at tuple sizes where any grid would exhaust memory.

**Demos.** `demo_overview` / `demo_overview.py` (part 1c); `demo_rhythmTensors` / `demo_rhythm_tensors.py` (part 3: entropy as rhythmic complexity); `demo_dispatchAndKernelControls` / `demo_dispatch_and_kernel_controls.py` (part 7: the estimators compared); `demo_jmm_1_1_entropy` (JMM).

### 8.5 Mass in a region

`massMaet` gives the mass of a density inside a region: how much of the material lies in a range of pitches, say, or near a configuration, and with `'normalize', 'total'` what share of the whole it is. Each tuple's kernel carries unit mass, so a tuple counts by the share of its kernel inside the region, and one just outside a box still contributes the part of its kernel that crosses the edge. A region is a box on some of the attributes, the rest integrated over entirely, or a soft Gaussian region. It can select tuples as well as events — the fifths among all pairs of notes, as a region around 700 cents on a relative attribute at r = 2 — which no weighting of events can. The rules for regions are in the entry in §12.4.

**Demos.** `demo_overview` / `demo_overview.py` (part 1d: the share of a scale's pairs of notes a fifth or fourth apart, as a region of a relative density at r = 2; part 2d: each of two triads' share of a melody's notes, in a window moved along the melody with `sweptMass`).

### 8.6 Drawing a density

`plotMaet` / `mpt.plot_maet` draws a density of one, two, or three
drawn dimensions, dispatching on the dimensionality the density carries
— `dim = r - isRel`. Four or more cannot be drawn and is refused.

It offers three methods, each named for what it shows rather than for
the geometry it uses, since the geometry is what changes with the
dimensionality.

**`kernels`** draws the model rather than the density: one object per
tuple centre, an ellipsoid at three dimensions, an ellipse at two, a
curve at one, coloured by the density at that centre. No grid is
evaluated, so the cost is the number of centres rather than the volume,
and every centre is drawn whatever the sampling. It is where the shape
of a relative kernel becomes visible — the covariance is `I + J`, so
the kernel is elongated by `sqrt(r)` along the all-ones diagonal and
circular across it, which the sum hides. Where kernels overlap it shows
the kernels and not the sum they make.

At one dimension it shows more than that. Each curve is one tuple's own
term — its width the kernel's, its height that tuple's weight — so the
curves *sum* to the line `density` draws, and a peak can be seen to be
one kernel or several. Colour there is the density at the curve's
centre, the total with every neighbour counted, so two curves of equal
height differ in colour exactly where kernels crowd.

**`points`** draws the density sampled: one translucent mark per grid
node above a threshold. Three drawn dimensions only — below that it
samples what `density` already draws whole, a surface carrying every
node at once and a line likewise. At a fine grid it is much the same
picture as `density`; what differs is that the samples stay discrete,
so the grid is visible rather than interpolated away.

It is not the cheap option its simplicity suggests. It evaluates the
same grid as `density` — the evaluation is around 85 per cent of the
cost of either — and its marks go as the cube of the nodes per axis
where a stack's planes go as the first power. Measured on a 121-node
grid it gave 20 frames a second against 67, and on a 241-node grid 4
against 51.

**`density`** draws the density itself: a line at one dimension, a
textured translucent surface at two, and at three a stack of textured
planes square to whichever axis is most nearly square to the view. A
stack is built for each of the three axes and shown one at a time, so
the picture changes as a rotation crosses the diagonal rather than when
it ends. At three dimensions this is a true volume rendering: the value
is read as an extinction per unit of path, so what a ray accumulates
follows the distance it travels through the material rather than the
number of planes that distance is cut into.

The three-dimensional `density` is **MATLAB only** — it rests on
texture-mapped surfaces, which matplotlib has no counterpart for, and
Python raises an error there naming `points`. Every other combination
is in both languages.

```matlab
pAttr = {{[0 200 400 500 700 900 1100]}};     % one event, seven pitches
specs = flatSpecs(pAttr, 'r', 4, 'rel', true, 'exch', true);
dens  = buildMaet(pAttr, [], 'specs', specs, 'sigma', 15, ...
                  'isPer', true, 'period', 1200);
plotMaet(dens);                                 % the kernels
figure; plotMaet(dens, 'method', 'density');    % the density
```

```python
p_attr = [[[0, 200, 400, 500, 700, 900, 1100]]]  # one event, seven pitches
specs = mpt.flat_specs(p_attr, r=4, rel=True, exch=True)
dens = mpt.build_maet(p_attr, None, specs=specs, sigma=[15.0],
                      is_per=[True], period=[1200.0])
mpt.plot_maet(dens)                              # the kernels
mpt.plot_maet(dens, method='points')             # the density sampled
```

#### The grid

`points` and `density` evaluate a grid, which can be asked for either
way: `'step'` is a spacing in the density's own units, `'nodes'` a
count of steps across whatever range is drawn — the same request said
two ways, so give one or the other. The grid has one more point than
`nodes` along each axis.

What matters is the grid measured against sigma, not against the axis
range: a blob is a few sigma across, so a grid coarser than sigma steps
over it and the density appears to have peaks missing rather than
blurred. Roughly one sample per sigma is the least that shows the
shape. Cost goes as the count to the power of the dimensionality, which
is why the default is 1200 steps at one and two dimensions and 120 at
three. `kernels` evaluates no grid and is the method to check a blob
count against.

Cutting more finely than the volume was evaluated does not help the
three-dimensional `density`: opacity reaches the renderer as eight
bits, so a plane fainter than 1/255 rounds away, and material too faint
to clear that in one plane is lost rather than accumulated over many.
There is one plane per plane of the volume for that reason, and the
grid is what cuts it more finely.

#### `points` has a resolution budget

A mark carries a single depth across its whole face, so where two marks
overlap the nearer hides the farther outright rather than blending with
it. Every such contest reverses when the camera passes to the other
side of the cloud, so the same density draws differently from opposite
directions: blobs acquire haloes and hard edges from one of them and
fade smoothly from the other. Marks that only meet cannot do it, and
the automatic marker size keeps them apart — it measures the grid's
spacing on screen and sizes the marks to clear one another on it,
refitting as the figure is resized or zoomed. The spacing it measures
is for an assumed camera, not the one in front of you, so the size
holds at every angle: were it fitted to the current view it would
change as the axes turned, and with it the ink each mark lays down, so
the cloud would brighten and dim as it rotated.

The assumption scales with the step. Ink on screen goes as the assumed
foreshortening squared over the step — the marks number `step`⁻³ and
each covers `(fore × step)²` — so the assumption rises as the square
root of the step, and a cloud drawn at a coarse step is about as bright
as the same cloud drawn at a fine one. `markScale` scales the whole
thing: larger marks and a brighter cloud, at the price of the marks
overlapping over more of the sphere. The drawing warns once, as
`mpt:markOverlap`, when they overlap at the view being drawn, and names
the `markScale` that would have them clear.

Overlap is what haloes need. Once the marks overlap they show at any
elevation, and they are worse below the horizontal than above it;
whether they also worsen as the view flattens towards an axis is
unclear, and if so the effect is slight — a view looking nearly along
an axis, where the marks overlap most, draws perfectly well. Why the
sign of the elevation should matter at all is not established; depth
sorting and the depth test are both symmetric under reversing the
camera. The rule is empirical and is the one to go by: if a drawing
looks haloed, lower `markScale` or turn above the horizontal, and if it
must be trusted from any angle, use `density`. The margin is generous,
a mark rendering appreciably wider than its nominal size, so the marks
read as dots with space between them rather than as a continuous cloud.

The budget is about three points of screen per grid node, and it is a
relation between the step and the size the axes is drawn at, not a
property of the step alone. Left to choose its own step, `points` stays
inside it. A step set by hand can ask for more than the figure affords,
and the marks then overlap however they are sized; the drawing warns
once — `mpt:markOverlap`, suppressible in the usual way — naming the
smallest step that would draw cleanly at that size. Enlarging the
figure, zooming in, or coarsening the step all buy the same thing.
`density` carries its depth per pixel rather than per mark and has no
such budget, so a density too fine for `points` at a usable figure size
is a density to draw with `density`.

The budget is a MATLAB matter. matplotlib depth-sorts and blends the
marks, so `mpt.plot_maet` has no automatic sizing and no warning:
`marker_size` is whatever is asked for.

#### Appearance

`'alphaPeak'` and `'alphaFloor'` set the opacity at the density's own
peak and where there is no density, with `'alphaGamma'` the curve
between them; a floor of 1 turns the fading off and draws the picture
opaque. `'colourGamma'` is the same curve on the colour. Low material
fading to dark panes is what keeps the colour map's low colour from
becoming the ground the density is read against — which is why
`'dark'` defaults to true in MATLAB, at every dimensionality: the map
runs dark to bright, so anything coloured for a low density is nearly
black and would be invisible on white.

Python follows matplotlib's defaults instead: no colour-map
brightening, light panes, and matplotlib's own camera until a view is
given. `'brighten'` has no Python counterpart, and `view` takes
matplotlib's `(elev, azim)` where MATLAB's takes `[azimuth elevation]`
— each language following its own plotting library.

Python's plotting needs `matplotlib`, which the toolbox does not
require; it is imported when a plot is drawn, not when `mpt` is
imported.

**Demos.** `demo_maetPlots` / `demo_maet_plots.py` (every combination of r, [rel], [per], and [exch], by each method); `demo_overview` / `demo_overview.py` (part 1e); `demo_rhythmTensors` / `demo_rhythm_tensors.py` (part 2).

### 8.7 Swept similarity, entropy, and mass

`sweptSimilarity` / `swept_similarity` compares a query with a context at each of a list of values on an attribute, the *sweep values*, and returns one similarity per value. Its canonical use is attribute translation (§7.4): each sweep value translates the query, which is compared with the whole context, giving the cross-correlation of the query against the context. This is computed in one pass by `sweepSimMaet`, and it is what a bare call does:

```matlab
S = sweptSimilarity(pmContext, pmQuery, 'sweep', {a, offsets});
[S, mu] = sweptSimilarity(pmContext, pmQuery, 'sweep', a);   % default offsets
```

Naming the attribute alone asks for default sweep values: every offset at which query and context overlap (one period, on a periodic attribute), stepped at no more than $h$, half the standard deviation of the profile's peaks, so that they are resolved. Translating the query moves all $D$ coordinates of the attribute's tuple alike, so each pair of tuples contributes a Gaussian in the offset of standard deviation $\sigma\sqrt{2/D}$ (Milne 2026, Eq. 10): narrower the larger the tuple, with $D = r$, or the product of a nested attribute's per-level tuple sizes, and $\sqrt{2/(\mathbf{1}^\top \Sigma^{-1} \mathbf{1})}$ for a kernel covariance $\Sigma$. So $h = \sigma\sqrt{2/r}/2$. The step is also chosen so that every exact match lies on the grid. Where the context's values are whole multiples of a spacing $g$ apart, and so are the query's (and $g$ divides the period, on a periodic attribute), every exact match is at the lowest offset plus a whole multiple of $g$, so the step is $g/k$, with $k$ the smallest whole number that brings it to $h$ or below: onsets on whole beats with $h = 0.15$ step at 1/7, not at 0.15, which would miss every whole-beat offset but those it happens to hit. Where $g$ is below $h$ it is the step itself, if at least $h/4$; otherwise (values on no such lattice, as in a performance, or on one too fine) the step is $h$, which leaves every peak within a quarter of its standard deviation of a grid point, at about 97% of its height or more. Where only a window is placed (`'window'`, and `sweptEntropy` and `sweptMass`), the defaults are the context's range on the attribute, stepped at half the window's sd: σ plays no part there, since the window weights the events before any density is built, and the profile changes on the scale of the window, each event's weight following it. For largely separate windows, as when the profile's values are to be used as data rather than plotted, give `'step'` as half the window's width ($\sqrt{3}$ sd), so that neighbouring windows overlap by half. A pure rectangle (`'rect'`) is the exception: the windowed context changes only where an event enters or leaves the window, at each event's value plus or minus half the width, so the profile is constant between these breakpoints. Without a given `'step'`, its default sweep values are these pieces, each sampled just inside both its ends: every value is the profile's value at its sweep value, and a line plot through them draws the steps exactly. The offsets are returned as `mu`. Any of the defaults can be replaced, with a bare number where one attribute is swept: `'sweep', 2, 'step', 0.5` (Python `sweep=1, step=0.5`).

A window on the context (event weighting, §7.5) can be added, to make each comparison local.

**Two attributes.** The *swept attribute* is the one the sweep values lie on: the query is translated along it, and a window is a function of displacement along it. The *target attribute* (`targetAttr`, by default the first attribute not dropped) is the one whose per-event weights a window multiplies. They are usually different: a window over time weights the pitch events.

**The rule.** At each sweep value $s$ on the swept attribute:

- the query, where it is translated, has its *reference value* `queryRef` at $s$: it is translated by $\mu = s - \mathrm{queryRef}$;
- a window, where there is one, has its reference value, $\delta = 0$ of the window function $h(\delta)$ (the midpoint of the rectangle, the Gaussian, or their blend, each symmetric about it), at $s$.

Nothing else places anything. For each swept attribute, `align` says which of the two are placed:

| `align` | At each sweep value $s$ | `queryRef` by default |
| --- | --- | --- |
| `'query'` (default) | the query only: translation over the whole context | 0: sweep values are the offsets added to the query as written |
| `'both'` | the query and a window, both at $s$ | the query's middle, so the window is aligned at the query's middle |
| `'window'` | a window only; the query is left as written | — |
| `'independent'` | a window at each value of one list, the query at each value of another, in every combination (`'sweep'`, `{a, {windowValues, queryValues}}`) | the query's middle |

The query's middle is the mean of its events' values on the swept attribute. A window aligned at $s$ weights each context event $n$ on the target attribute:
$$w'(n) = w(n)\,h\bigl(p_a(n) - s\bigr),$$
where $p_a(n)$ is event $n$'s value on the swept attribute $a$ (the notation of §7), so events far from $s$ are attenuated. (Where an event holds several values on the attribute — the onsets of a bound super-event, say — `locate` says which one stands for it: their mean by default, or the first, the last, or the midpoint of the two. It has no effect where each event holds one value.)

**The window acts on the pre-MAET, before any density is built.** $p_a(n)$ is the value the pre-MAET holds for event $n$ (its onset time, say), and only the weights change. Whether the swept attribute is then built in absolute or relative mode (§13.1), or dropped, matters only afterwards, when the density is built from the weighted events. A relative time attribute of bound events, for example, is still windowed by onset time (each event's onsets reduced to one by `locate`), and then compared through its onsets measured from the first within each event.

**Offsets.** The translation applied to the query at each sweep value, $\mu = s - \mathrm{queryRef}$, is returned on request (MATLAB `[S, mu] = sweptSimilarity(...)`, `mu{a}`; Python `return_offsets=True`, `offsets[a]`). It is the query's shift from where it was written, whatever `queryRef` is, so a windowed sweep can be stepped across the context (`'step'`, the start and stop defaulting to the context's extent) and read against $\mu$. With query and context written from a common origin (the query at the time it was taken from, say), $\mu$ keeps its meaning through preprocessing that keeps the values, such as differencing, which drops the query's first event and so shifts its middle: profiles with and without it share one axis (§14.5).

**Sweep values.** The sweep values themselves, listed or generated, are also returned on request, by all three swept functions: MATLAB `[S, mu, sv] = sweptSimilarity(...)`, `[H, sv] = sweptEntropy(...)`, and `[M, sv] = sweptMass(...)`, `sv` a $1 \times A$ cell with `sv{a}` holding attribute $a$'s values; Python `return_sweep_values=True`, a dict `{a: values}`. They are the axes of the profile, so `plot(sv{a}, M)` draws a sweep whose values were generated from the defaults or from `'start'`, `'stop'`, and `'step'`. Under `'query'` they equal the offsets; under `'both'` they are where the query's middle and the window lie; under `'independent'` the entry is the pair of the window's list and the query's. A profile sampled at the default step can be interpolated to any finer grid without computing it again, since the samples already capture it: `interp1(sv{a}, S, x, 'spline')` in MATLAB, `scipy.interpolate.CubicSpline(sv[a], S)(x)` in Python. This holds for the smooth profiles only; across the jumps of a rectangular window, a spline overshoots.

**Choosing `queryRef`.** Under `'query'` it only relabels the output, a sweep value $s$ under reference $r$ giving the same comparison as $s - r + r'$ under $r'$; under `'both'` it also decides which point of the query lies at the window's centre.

- **0**: sweep values are the offsets $\mu$ added to the query as written — transpositions in cents, or time shifts.
- **The query's middle**: sweep values are where the middle lands, and under `'both'` the window is aligned at the query's middle.
- **A particular point of the query**, such as its first onset or its root: sweep values are where that point lands — the time at which a match starts, or the key of a transposition. Under `'both'` the window's centre then sits at that point.

The query itself is not windowed by `sweptSimilarity`. To weight the query's own events by a window — to taper it, say, or to compare a local region of the query with a local region of the context under `'cosine'` — apply `weightEvents` to the query before the call.

**Choosing a role.**

- **`'query'`** — the canonical sweep: where in the context, or at which transposition, the query best matches the context as a whole. Only the kernel width $\sigma$ on the attribute limits which context events count. The natural choice for transposition, any periodic attribute, and any whole-context cross-correlation; every sweep value is computed in one pass by `sweepSimMaet`.
- **`'both'`** — a local `'query'`: the window fixes the region of the context that counts around the query. Under the default normalization this changes little from `'query'`, the kernel already making distant events count for little; under `'cosine'` the denominator is the norm of the windowed context, so unmatched material inside the window lowers the score and material outside it is ignored. As the window widens, `'both'` becomes `'query'`.
- **`'window'`** — the query is not translated: the window steps through the context and the query, as written, is compared with each region in turn. What this measures depends on whether the swept attribute is absolute, relative, or dropped (below).
- **`'independent'`** — each window position gives a whole profile (the query's list may hold one row per window value, when its placements depend on where the window is): a correlogram, for a best placement that changes across the context, such as a lag between two parts that drifts over time.

**Absolute, relative, or dropped: the swept attribute.** Whether the swept attribute is absolute or relative (its `[rel]` flag, §13.1), and whether it is dropped (`drop`), decides what it contributes to each comparison, whatever the role:

- **Absolute** — compared by position, so translating the query along it changes where the query matches, and all four roles apply. Under `'window'` the query is compared in place: the profile shows where in the context its match with the query as written comes from — two parts of a piece on a shared time axis, say, whose similarity the window resolves in time. Under the default normalization the profile is $\langle h_s \cdot f_X, f_Y\rangle / \langle f_Y, f_Y\rangle$, linear in the window, so windows that tile the context (half-open rectangles a width apart) give contributions that sum to the whole-piece similarity. To find where the query occurs, translate it (`'query'` or `'both'`).
- **Relative** — compared only up to a common translation of each tuple, that is, through its values relative to the lowest (a chord's intervals above its bass, or a bound event's onsets measured from its first; §13.1), so the query's internal spacing must match but its position does not matter. Translation leaves these relative values unchanged (§7.4), so only `'window'` applies (it gives what `'both'` would). The events are still windowed by the values the pre-MAET holds, as above.
- **Dropped** — marginalized after the window has weighted the events (`drop`), so not compared at all: the query is compared with *what* the region contains, not *where* in it. With time as the attribute, C–E–G matches a region containing those pitches whatever their order or rhythm, and the window width sets the time scale of the analysis. Use it when the query has no meaningful arrangement on the attribute (a key profile, a pitch-class set, a chord). Translation has nothing to act on, so only `'window'` applies.

Event differencing (`differenceEvents`, §7.2) is not a fourth option but a change of values: the attribute then holds first differences between successive events (inter-onset intervals, pitch steps), and is absolute or relative like any other. As the swept attribute, its windows therefore select by interval size, and translation adds the same amount to every interval — on logarithmically rescaled inter-onset intervals, a tempo change (§7.3). Usually the attribute differenced (pitch, say) is not the one swept (time), which differencing passes through unchanged at order 0, keeping each event's onset (§14.5).

Several attributes can be swept at once, each in its own role, and each absolute, relative, or dropped. Where the windowed context is the same across the query's translations — `'query'`, the query list of `'independent'` — those translations are computed together by `sweepSimMaet` (a nested attribute on its contraction route), falling back to one comparison per translation where no sweep route applies.

**When a window on the context is needed.** A window localizes; so, on a compared absolute attribute, does translation, the kernel letting the query match only material near where it is translated. A window on the context is therefore indispensable exactly where translation cannot localize: on a dropped or relative swept attribute, alone or alongside translation on another attribute, and in `sweptEntropy`, which has no query. Each bar windowed on time, time dropped, and the query translated in pitch finds at once the bar and the transposition of each statement; the query has no position on the dropped attribute for a window of its own to select. On a translated attribute (`'both'`, `'independent'`), a window on the context does nearly what a window on the query would (`weightEvents` aligned at the matching point of the query, then translated): the two differ only in whether a near miss is weighted where the context's event lies or where the query's does, and they agree exactly for a rectangle wherever no matched pair of events straddles its edges. What a window on the context adds there is the `'cosine'` denominator, the norm of what the window keeps, so that unmatched material nearby counts against the match. Under that denominator repetition within the window also no longer stands in for missing content, as it does under the default, where a bar repeating two of the query's pitches twice each scores as highly as one holding all four (`demo_sweptSimilarity`, §12). To ask which part of the query matches, weight the query.

**Reading the output.** One dimension per sweep list, in attribute order; `'independent'` contributes two, the window's first. A sweep value is where the query's reference lands (under `'window'`, where the window is aligned).

**Window family.** A window is `{shape, width}` (MATLAB; Python `(shape, width)`), `{shape, width, edges}`, or a struct / dict with `'shape'`, `'width'` or `'sd'`, and `'edges'`: the shape parameter $\gamma \in [0, 1]$ (or `'gaussian'` / `'rect'`) and the full width $W$ of the equivalent rectangle, with standard deviation $W / (2\sqrt{3})$ held fixed across the whole shape family, exactly as in `weightEvents`. The swept functions evaluate their windows through `weightEvents`' own implementation, so a window may be any of its profiles that is aligned at a reference value: the exponentials, symmetric (`'exponential'`) or extending to one side only (`'exponentialBefore'`, `'exponentialAfter'`), given as a struct / dict with `'sd'` or `'decayRate'`, and a function of the displacement $p_a(n) - s$. The serial-position profiles, anchored at the first and last events rather than at the sweep value, are refused. On a periodic attribute the displacement wraps, as in `weightEvents`. A rectangle is *half-open* by default, including its lower edge and not its upper, so windows placed a width apart share no event — the right choice for tiling a context; a *closed* rectangle includes both edges, the right choice for a window that must hold a query. The width is the scale of the comparison and nothing in the data can supply it for `'window'` and `'independent'`, where it is required; for `'both'` it may be omitted, and the window is then the smallest closed rectangle that, placed by the rule, holds the query, so an exact match scores 1. A window given for `'both'` that leaves out some of the query's own events (narrower than the query, or a half-open rectangle exactly as wide as it) draws a warning, since the query can then never be matched in full. Generated sweep values (a bare attribute in `'sweep'`, or `start` / `stop` / `step`, each of which overrides one default) depend on the role. Where the sweep values translate the query (`'query'`, `'both'`) they cover every placement at which the query overlaps the context (from its highest value on the context's lowest to its lowest on the context's highest), or one period on a periodic attribute, stepped at half the standard deviation of the profile's peaks, $\sigma\sqrt{2/r}/2$, window or no window. Where they place a window only (`'window'`, and the window's list of `'independent'`) they cover the context's full range, stepped at half the window width (neighbouring windows overlapping by half). The query's list of `'independent'` is always given explicitly.

**Normalization.** The `normalize` keyword (also accepted as `normalise`) selects the denominator. The default, `'oneSidedDenom'`, divides the inner product of the windowed context $h \cdot f_X$ and the query $f_Y$ by the query's own self inner product:

$$ s_{\text{one-sided}} = \frac{\langle h \cdot f_X, f_Y \rangle}{\langle f_Y, f_Y \rangle} $$

This is *magnitude-aware*: self-similarity at full window coverage equals 1, the profile drops below 1 when the window does not cover all of the query's mass, and it rises above 1 when the windowed context carries more matching mass than the query does in total, which is what a sliding-motif or probe-tone analysis wants — a region with a large amount of matching content scores higher than a region with a little. The output is not bounded in $[-1, 1]$, and multiplying the query's weights by a constant rescales the profile. The alternative, `'cosine'`, is the strict shape-only cosine, $\langle h \cdot f_X, f_Y \rangle / \sqrt{\langle h \cdot f_X, h \cdot f_X \rangle \langle f_Y, f_Y \rangle}$, bounded in $[-1, 1]$ and invariant to a positive scalar on either operand: the same shape match scores the same regardless of how much mass falls inside the window. The one-sided default scores *how much of the query is present here*; the cosine option scores *how well the local shape matches, ignoring magnitude*.

`demo_sweptSimilarity` / `demo_swept_similarity.py` works through each role, with the swept attribute absolute, relative, and dropped, on one melody that holds four statements related to a query, each setting finding a different subset of them.

`sweptEntropy` / `swept_entropy` shares the sweep values, window, `locate`, and `drop` arguments, but has no query, so its sweep values always align the window: at each sweep value the windowed density is built, with any dropped attribute first marginalized, and its entropy taken with the chosen `method` (§11.2). Every swept attribute needs a window with a width, the scale of the local region. This is the windowed-entropy construction of §7.7 packaged as a sweep; a sweep value whose window catches no event returns `NaN`, under every method.

`sweptMass` / `swept_mass` takes the same arguments and, at each sweep value, the mass of the windowed density in a region (`massMaet`): how much of the local material lies in a range of pitches, or near a configuration, and with `'normalize', 'total'` what share of it does. The window and the region do different jobs. The window weights events before the density is built, as in the other swept functions; the region is read from the built density, so it counts each tuple by the share of its kernel inside it, and it can select tuples – the fifths among all pairs of notes in a window, say, as a region around 700 cents on a relative attribute at r = 2 – which no weighting of events can. Where the region concerns single events and σ is small, weighting the events with a rectangle on the region's attribute and taking the total mass gives nearly the same answer.

#### Translation sweeps in one pass

A translation sweep (`'align'`, `'query'`, the default) never builds a translated copy of the query, nor compares one offset at a time. Because every value of an attribute shifts by the same amount, the comparison at all offsets can be computed in one pass rather than one inner product per offset; `sweptSimilarity` does this through `sweepSimMaet` wherever the context is not reweighted from one offset to the next.

The identity behind this is a separation. On an absolute attribute the exponent of each tuple pair's contribution splits into two terms:

$$Q_a(d - \mu_a \mathbf{1}) = \underbrace{r_a (\mu_a - \bar{d})^2}_{\text{placement}} + \underbrace{\sum_i (d_i - \bar{d})^2}_{\text{shape}},$$

where $d$ is the difference between the two tuples and $\bar{d}$ its mean. Only the placement term involves the offset $\mu_a$. So each pair of tuples contributes, as a function of the offset, a single Gaussian centred at the pair's mean difference with effective width $\sigma_a / \sqrt{r_a}$, carrying a fixed weight set by the pair's within-tuple shape. The whole sweep is a weighted Gaussian mixture in the offset, built once and then evaluated at each offset in turn. Reading the same separation in the other direction: integrating the profile over all offsets recovers the relative-mode inner product, since the shape term *is* the relative quadratic form.

`sweepSimMaet` / `sweep_sim_maet` computes a sweep this way from two built densities — here those of the melody and the query of the quick start's pattern search (§4), built with `buildMaet`:

**MATLAB:**
```matlab
offsets = [zeros(1, 29); linspace(-1.0, 6.0, 29)];   % translate in time only
S = sweepSimMaet(melody, query, offsets);
```

**Python:**
```python
offsets = np.vstack([np.zeros(29), np.linspace(-1.0, 6.0, 29)])
S = mpt.sweep_sim_maet(melody, query, offsets)
```

`sweptSimilarity` and `sweepSimMaet` are the two routes for a swept translation similarity: the first takes pre-MAETs and sweep values, and adds windows and dropped attributes where they are wanted (above); the second takes built densities and an offset matrix, and so can reuse densities across many sweeps and follow any path of offsets, not only a grid. `translateAttributes` makes one translation at a time, for use in a preprocessing pipeline.

**Which attributes can be swept.** An attribute may be *translated* only if its quadratic form keeps the tuple's own mean, which is to say only in absolute mode. In relative mode a uniform translation cancels in every within-tuple difference, so there is nothing to sweep — the attribute still contributes, through its shape term alone. The same holds per block for a nested attribute at an inner or intermediate co-transposition unit, whose form is the sum over blocks of each block's relative form. A periodic attribute contributes an offset-independent factor when it is not translated; when it *is* translated, the wrapped kernel admits no such separation and the sweep is carried instead by the orbit route below.

**Two routes.** The `method` argument selects the decomposition. `'mixture'` is the separation above: one pass over the tuple pairs, then a mixture evaluation per offset. `'orbit'` evaluates the Möbius/orbit inner product at the shifted values, at a cost that scales with the orbit count rather than with the tuple-pair count $[C(K_{a,n}, r_a) \, r_a!]^2$, which the mixture must both enumerate and store; it is the route for large tuple sizes, and the only one that covers a swept periodic attribute. It is symmetric-only, so it declines an ordered attribute. `'contract'` serves densities with a nested attribute, such as a bound, spectrally enriched pitch attribute: it keeps the level-by-level contraction of the nested inner product and carries the offsets as a batch dimension, so no nested tuple is enumerated, where the mixture would enumerate every combination of the elements at every bound position (12⁴ for twelve partials at each of four positions) and the orbit route declines. It covers any swept attribute that is absolute, flat or nested, periodic or not, alongside attributes that are not swept. `'auto'`, the default, takes `'contract'` wherever a nested attribute is present, and otherwise compares the costs of the other two and picks; `matlab/tools/calibrateSweepRoute.m` measures the crossover on a given machine, though correctness does not depend on it since both routes compute the same quantity.

**Relative-and-periodic attributes** raise a question of measure rather than of speed. The single-wrap and transposition-average kernels agree only below a $\sigma / P$ limit, and above it the attribute's `wrap` selects between them — `'single-image'` for the former, `'full-image'` for the latter — exactly as it does for a comparison made offset by offset.

**Demos.** `demo_sweptSimilarity` / `demo_swept_similarity.py`; `demo_overview` / `demo_overview.py` (part 2d: `sweptMass`); `demo_helixBlend` / `demo_helix_blend.py`; `demo_tempoInvariance` / `demo_tempo_invariance.py` (part 3: a sweep under a tempo-tolerant kernel); `demo_batchProcessing` / `demo_batch_processing.py` (sweepSimMaet against translated copies); `demo_dispatchAndKernelControls` / `demo_dispatch_and_kernel_controls.py` (part 3: sweeps in one pass); `demo_jmm_1_1_entropy` (JMM); `demo_jmm_1_3_cadence_nesting` (JMM); `demo_jmm_2_3_spectral` (JMM); `demo_jmm_3_1_texture` (JMM); `demo_jmm_3_2_diff` (JMM); `demo_jmm_3_3_xcorr` (JMM).

---

## 9. Measures on pitch and rhythm sets

The measures of this section take a pitch or rhythm set, or a spectrum, directly rather than through a pre-MAET. They were developed in separate literatures and are implemented here so that predictors from all of them can be applied to the same material within one analysis. The function entries are in §12.5–§12.8.

### 9.1 Consonance and harmonicity

Four complementary measures address how consonant or harmonic a collection sounds: spectral entropy (`spectralEntropy`), template harmonicity (`templateHarmonicity`), tensor harmonicity (`tensorHarmonicity`), and sensory roughness (`roughness`). They have been used singly and in combination to predict consonance ratings, perceived affect, and tonal stability. Only tensor harmonicity builds a density; the others work on the spectrum directly. Functions in §12.5.

#### templateHarmonicity and tensorHarmonicity compared

These two functions both measure "harmonicity" — the degree to which a chord's intervals resemble those of a harmonic series — but they do so in fundamentally different ways. The distinction is subtle and important.

**templateHarmonicity (cross-correlation).** This function cross-correlates the chord's composite spectrum with a harmonic template:

1. Build the chord's 1-D absolute spectrum (either by enriching each chord pitch with harmonics via `'chordSpectrum'`, or using the pitches/weights as given — e.g., empirical peaks from `audioPeaks`).
2. Build a single harmonic template (one complex tone at 0 cents with harmonics defined by `'spectrum'`).
3. Evaluate both as 1-D expectation tensors on a fine grid.
4. Cross-correlate and normalize.

The maximum of the normalized cross-correlation (hMax) is the cosine similarity between the chord's spectrum and the template at the best-matching transposition. A weighted multiset whose partials align closely with *some* harmonic series will score high. The template is a *single* complex tone. It is not duplicated. There are no chord pitches "placed" into the template — the chord enters only through its composite spectrum, and the template slides across it looking for the best match.

**tensorHarmonicity (tensor lookup).** This function evaluates the relative r-ad expectation tensor of a harmonic series at the chord's intervals (measured from the lowest pitch to each of the remaining pitches):

1. Build a harmonic template spectrum. By default, the template pitch (0 cents) is duplicated K times (where K is the chord cardinality) before adding harmonics. All K copies are rooted at 0 — they are *not* placed at the chord's pitch positions.
2. Build the relative r-ad expectation tensor (r = K, isRel = true) from this duplicated template. This tensor represents the density of all ordered r-tuples of intervals that arise within the harmonic series.
3. Sort the chord's pitches and compute the K − 1 intervals from the lowest pitch to each of the remaining pitches.
4. Evaluate the tensor at that single interval point.

A high density value means the chord's intervals are likely to co-occur in a harmonic series, given the perceptual uncertainty modelled by σ (the Gaussian smoothing applied to the template's expectation tensor).

**Why duplicate the template?** Without duplication, every position in an r-tuple can only be filled by a *different* partial. This means that a unison (two chord tones sharing the same partial, such as two notes an octave apart both activating the 2nd harmonic of the lower note) cannot contribute to the density. With K-fold duplication, each partial can appear in up to K positions in the r-tuple, correctly allowing unisons and other interval repetitions to register as consonant. A critical misreading to avoid: the K copies of the template do *not* represent chord tones placed at their respective pitches. All K copies are rooted at 0 cents. The chord enters only as the query point — the K − 1 intervals at which the density is evaluated.

**Worked example: unison vs octave.** Consider chord A = (0, 0) (a unison) and chord B = (0, 1200) (an octave). With `templateHarmonicity`, both produce very similar composite spectra and both score high. With `tensorHarmonicity`, chord A has intervals [0] and chord B has intervals [1200]; both score high but for different reasons — the unison's density comes from duplicated partials at the same frequency, while the octave's comes from partial pairs an octave apart. Without duplication (duplicate = 1), the unison cannot contribute at all, and the density at [0] would be near zero.

**When to use which:**
- Use `templateHarmonicity` for a *spectral* measure: how well does the chord's overall spectrum match a harmonic series? This may be closer to what the auditory system does when parsing a complex sound into virtual pitches.
- Use `tensorHarmonicity` for an *interval* measure: how probable are the chord's specific intervals within a harmonic series? This may better capture interval-based aspects of consonance that are not reducible to spectral overlap.
- Use `virtualPitches` for the full pitch-indexed cross-correlation profile from which `templateHarmonicity` extracts its summary statistics.

**Demos.** `demo_triadConsonance` / `demo_triad_consonance.py` (five measures over a grid of triads); `demo_virtualPitches` / `demo_virtual_pitches.py`; `demo_audioAnalysis` / `demo_audio_analysis.py`; `demo_overview` / `demo_overview.py` (part 3).

### 9.2 Scale and rhythm structure

Structural and perceptual features of points distributed around a cycle — pitch classes at the octave, rhythmic positions in a metrical cycle. They fall into three groups by output granularity: *period-level* measures returning one value per collection (balance, evenness, the discrete Fourier transform itself, coherence, sameness, and $n$-tuple entropy); *integer-position* measures over an equal division of the period (the circular autocorrelation phase matrix, Markov prediction); and *continuous-position* measures evaluable anywhere on the cycle (edge detection, the projected centroid, mean offset). Six of them also accept a positional uncertainty $\sigma$, which softens their discrete tallies in the same spirit as the expectation tensor's kernel. The measures, their arguments, and the semantics of $\sigma$ under each are documented in §12.7.

**Demos.** `demo_overview` / `demo_overview.py` (parts 4–5); `demo_dftCircularSimulate` / `demo_dft_circular_simulate.py`; `demo_sigmaSpace` / `demo_sigma_space.py` (soft sameness, coherence, and n-tuple entropy); `demo_rhythmTensors` / `demo_rhythm_tensors.py` (part 4).

### 9.3 Sequences

One utility for ordered sequences: `continuity` summarizes the recent trend in a sequence of pitches, interonset intervals, or their differences, leading up to a query point. Serial-position weight profiles are not a separate function: `weightEvents` applies them over any attribute (§7.5). Functions in §12.8.

**Demos.** `demo_probeTone` / `demo_probe_tone.py` (part 4: continuity).

---

# Part III — Reference

## 10. API conventions

The MATLAB and Python implementations are functionally identical: the same inputs produce the same outputs, to floating-point precision. This section covers the translation between the two languages and the calling conventions they share: names, weights, the struct and raw forms, batching, and return values. The Quick Start (§4) shows both languages side by side; the function reference (§12) and the worked examples (§14) use MATLAB syntax, from which the Python equivalents follow by the rules below.

### 10.1 Naming

MATLAB uses camelCase; Python uses snake_case. A few names were shortened where the MATLAB name included a suffix that is redundant in a package namespace:

| MATLAB | Python |
|:---|:---|
| `balanceCircular` | `balance` |
| `evennessCircular` | `evenness` |
| `dftCircular` | `dft_circular` |
| `circApm` | `circ_apm` |
| `markovS` | `markov_s` |

All other names follow the mechanical camelCase → snake_case rule (e.g., `simMaet` → `sim_maet`, `addSpectra` → `add_spectra`).

### 10.2 Terminology

The framework's three structural terms are used consistently across this guide, the API documentation, and the accompanying article:

- An **element** is one weighted entry in an attribute's multiset at a single event: the indivisible unit from which every density is built.
- Its **position** is a scalar location in the attribute's space (a pitch in cents, a time in beats, a category label), carried in the `p` / `pAttr` arrays.
- Its **weight** is the non-negative mass at that position (perceptual salience, amplitude, or probability of perception), carried in the `w` arrays.

An attribute's elements at one event form a $K_a \times N$ matrix, one column per event. A **row** of that matrix is an index across events, not an element: a per-row weight or offset applies to one element in each event. Where the toolbox refers to a **tuple index**, it means a coordinate of an $r_a$-tuple, the ground set of the Möbius partition lattice.

### 10.3 Calling conventions

| Concept | MATLAB | Python |
|:---|:---|:---|
| Default (all ones) weights | `[]` | `None` |
| Boolean flags | `true` / `false` | `True` / `False` |
| Spectrum arguments | Cell array: `{'harmonic', 12, 'powerlaw', 1}` | List: `['harmonic', 12, 'powerlaw', 1]` |
| Name-value pairs | `'name', value` | `name=value` |
| Precomputed density | Struct with `.tag = 'MaetDensity'` | `MaetDensity` dataclass |

### 10.4 Weight arguments

Functions that accept a weight argument `w` treat it as a *broadcast specification* of a weight function, not as a fixed-shape array. The accepted forms, for a multiset of $N$ events with $K$ elements per event (with $K = 1$ for single-element attributes), are:

| Input form | MATLAB | Python |
|:---|:---|:---|
| Default (uniform 1) | `[]` | `None` |
| Uniform scalar `c` | `c` | `c` |
| Per-event (broadcast across rows) | length-$N$ row | 1-D length-$N$, or `(1, N)` |
| Per-row (broadcast across events) | $K \times 1$ column | `(K, 1)`, or 1-D length-$K$ |
| Full per-row-per-event | $K \times N$ matrix | `(K, N)` |
| Per event, value by value | $1 \times N$ cell | list of length $N$ |

In the last form, each entry is a scalar that weights all of that event's values or a vector with one weight per value, matching an attribute given per event (§6.1); slots holding no value take weight 0. A Python list of $K \neq N$ scalars is read per row.

The per-row and full forms are meaningful only when $K > 1$ — pitch attributes with multiple pitches per event (chords with exchangeable voices), or spectrally-enriched pitches where each fundamental is represented by $K$ partials. Functions operating on a 1-D event sequence (`continuity`, `differenceEvents`) accept only the first three forms.

Different inputs that specify the same underlying weight function are semantically equivalent: a scalar $c$ and a length-$N$ vector of $c$s produce identical downstream output, as do a $K \times 1$ column and its $K \times N$ broadcast. Functions preserve the compact representation where they can, for efficiency, but the output of one function fed into another is interpreted under this same convention, so the compact and broadcast-out forms are interchangeable in chained calls. Weights are non-negative; the standard reading is $w_i$ = probability that event (or element) $i$ is perceived, with `differenceEvents` and `continuity` propagating weights consistently with this interpretation.

For the MAET calling form, `w` is a per-attribute cell (MATLAB) / list (Python) whose entries follow the single-attribute rules above. A top-level `[]` / `None` or scalar is accepted as a shortcut that applies the same default — all ones or the given scalar — to every attribute.

### 10.5 Struct vs raw-argument calling

Two ways to pass density information to the core functions, identical in result, useful in different settings.

**Struct calling.** First call `buildMaet` to construct a `MaetDensity` struct (one density type serves the single-multiset and the multi-attribute case alike) from the underlying $(p, w, \sigma, r, \mathrm{isRel}, \mathrm{isPer}, \mathrm{period})$ specification, then pass that struct to `evalMaet`, `simMaet`, or `entropyMaet`. The expensive tuple enumeration and per-event index work is performed once at construction time and shared across all subsequent uses of the same density.

**Raw calling.** Skip the explicit struct and pass $(p, w, \sigma, r, \mathrm{isRel}, \mathrm{isPer}, \mathrm{period})$ directly to the core function. The struct is built internally — once per unique canonical form, when batching or in list mode — and discarded after use. "Raw" here means "untyped numeric inputs" — pitch arrays, weight arrays, and the structural parameters — as opposed to the pre-typed density object.

Within any single call, the two forms are equally efficient: the raw path's internal canonical-form deduplication (see "Canonical-form deduplication" below) builds each unique density just once, so a batched call against a fixed reference (e.g., one row broadcast against many candidates) builds the reference's density only once internally. The struct path's advantage is **across separate calls** — if you compare the same reference against candidates that arrive in multiple successive function calls, building the density once with `buildMaet` and passing the struct keeps that work alive between calls, whereas the raw path rebuilds it every time. The struct form is also useful when you want the density object itself for inspection, plotting, or further processing.

`simMaet`, `evalMaet`, and `entropyMaet` accept the struct, raw, list, and batched-raw forms in both languages, detecting the form from the first argument.

| MATLAB | Python |
|:---|:---|
| `simMaet(dens_x, dens_y)` | `sim_maet(dens_x, dens_y)` |
| `simMaet(p1, w1, p2, w2, ...)` | `sim_maet(p1, w1, p2, w2, ...)` |
| `evalMaet(dens, X)` | `eval_maet(dens, x)` |
| `evalMaet(p, w, sigma, r, ...)` | `eval_maet(p, w, sigma, r, ...)` |

Name-value arguments are uniformly available across both forms where they are meaningful. `'spectrum'` (applies `addSpectra` partials internally before density construction) is accepted in raw scalar and raw batched calls of `simMaet`, `entropyMaet`, `spectralEntropy`, `templateHarmonicity`, `virtualPitches`, and `tensorHarmonicity`; it is rejected in struct calls (where the spectrum is already part of the precomputed density). `'precision'` (decimal-place rounding for FP-noise-tolerant deduplication) and `'dedup'` (toggle internal deduplication) apply to batched-raw modes. `'verbose'` is universal.

### 10.6 Consumer-level batching: rows as multisets

Most consumer-facing functions accept a 2-D pitch matrix in addition to the original 1-D form. The convention throughout the toolbox is:

- **Each row of a 2-D pitch matrix is one multiset.** Row $i$ holds the pitches of the $i$-th multiset.
- **The number of rows $M$ is the batch dimension.** The function returns one result per row.
- **Each column is an element.** The number of columns $K$ is the multiset's element count; for chords this is voice count, for spectra this is partial count.
- **NaN-padding** allows variable-cardinality batches: when row $i$ has fewer than $K$ valid pitches, pad the trailing positions with NaN; the function drops NaN entries before processing that row.
- **Weight matrices match.** When weights are supplied, they have the same shape $(M, K)$ as the pitch matrix; uniform weights can be passed as `[]` (MATLAB) or `None` (Python).

Functions with batched-input dispatch include `simMaet` (for paired-multiset comparison), `evalMaet`, `entropyMaet`, `tensorHarmonicity`, `templateHarmonicity`, `virtualPitches`, `spectralEntropy`, `dftCircular`, `meanOffset`, `edges`, `projCentroid`, `circApm`, `coherence`, `sameness`, `nTupleEntropy`, `balanceCircular`, and `evennessCircular`. Each function's reference entry in §12 documents its specific return shape and any function-specific batching options.

The deduplication described next applies to this row-wise sense of batching. A multi-attribute item has no rows to collapse, so a cell / list of pre-MAETs (or of densities) loops instead; see §6.1.

**Demos.** `demo_batchProcessing` / `demo_batch_processing.py`; `demo_edoApprox` / `demo_edo_approx.py`.

### 10.7 Canonical-form deduplication

Every batched path uses canonical-form deduplication: before computing per-row results, the toolbox identifies rows that map to the same equivalence class under the relevant musical symmetry (permutation of elements, transposition modulo a period, etc.) and computes the result once per equivalence class, mapping it back to all matching rows. For inputs with much repeated structure — generator-chain sweeps, EDO scans, voice-leading enumerations, transposition orbits — this can cut wall time by several factors with no change in the returned values.

Which symmetries are exploited depends on what the function's output is actually invariant to:

- **Transposition-invariant scalar outputs** (`coherence`, `sameness`, `nTupleEntropy`'s `H` value, `simMaet` in `isRel = true`) collapse all transpositions of a multiset onto a single canonical key.
- **Transposition-equivariant outputs** (the DFT-equivariant family — `dftCircular`, `edges`, `projCentroid`, `circApm`) collapse only permutation and period-equivalence onto one canonical key; transposition is left alone because the per-output post-transform required to undo it varies by output and depends on user-supplied query points.
- **Reordering-invariant outputs** (every function above) collapse element permutations and, for periodic attributes, mod-reduction.

The `'precision'` name-value pair (where supported) sets the decimal-place tolerance used in canonical-key construction so that nominally identical multisets differing only by floating-point noise — typically from upstream arithmetic, not from input precision — are correctly identified as equivalent. Default is full floating-point precision; for pitch data on a 12-TET grid, `'precision', 4` is more than sufficient; for fractional-cent values from JI ratios, `'precision', 6` preserves all meaningful precision. For pitches on irrational grids (e.g., $N$-EDO tunings where the step size $1200/N$ is a repeating decimal), decimal-place rounding cannot collapse all transpositions; convert to integer EDO steps first (scaling $\sigma$ and $\mathrm{period}$ accordingly) for exact dedup in such cases.

**Demos.** `demo_batchProcessing` / `demo_batch_processing.py`.

### 10.8 Return values

Most functions return identical outputs in both languages. A few exceptions:

- `coherence` and `sameness` always return both the quotient and the count as a tuple in Python (e.g., `c, nc = mpt.coherence(...)`), whereas in MATLAB the count is a second optional output (`[c, nc] = coherence(...)`).
- `audio_peaks` always returns three values in Python (`f, w, detail = mpt.audio_peaks(...)`) where the MATLAB version returns the detail struct only when a third output is requested.
- `balance` and `evenness` (Python) accept a `return_std=True` flag to return `(mean, std)` as a tuple at `sigma > 0`; the default is the scalar alone. The MATLAB versions use the standard `nargout` idiom (`b = balanceCircular(...)` vs `[b, bStd] = balanceCircular(...)`), so no flag is needed.

### 10.9 Query points

In MATLAB, query-point arguments (`X`, `x`) are row vectors or matrices whose columns are query points. In Python, 1-D query-point arrays are passed as 1-D arrays; for multi-dimensional densities (r − isRel > 1), query points are a `(dim, n_queries)` array. For the most common case — 1-D densities — the usage is identical in both languages: pass a 1-D array.

### 10.10 Full name mapping

| MATLAB | Python | Category |
|:---|:---|:---|
| `buildMaet` | `build_maet` | Tensor core |
| `evalMaet` | `eval_maet` | Tensor core |
| `simMaet` | `sim_maet` | Tensor core |
| `sweepSimMaet` | `sweep_sim_maet` | Tensor core |
| `entropyMaet` | `entropy_maet` | Tensor core |
| `massMaet` | `mass_maet` | Tensor core |
| `maetCentres` | `maet_centres` | Tensor core |
| `plotMaet` | `plot_maet` | Tensor core (plotting) |
| `packPreMaet` | `pack_pre_maet` | Tensor core (constructor) |
| `unpackPreMaet` | `unpack_pre_maet` | Tensor core (constructor) |
| `flatSpecs` | `flat_specs` | Tensor core (constructor) |
| `kernelCov` | `kernel_cov` | Tensor core (constructor) |
| `simplexVertices` | `simplex_vertices` | Tensor core (constructor) |
| `differenceEvents` | `difference_events` | Tensor core (preprocessing) |
| `bindEvents` | `bind_events` | Tensor core (preprocessing) |
| `translateAttributes` | `translate_attributes` | Tensor core (preprocessing) |
| `transformAttributes` | `transform_attributes` | Tensor core (preprocessing) |
| `weightEvents` | `weight_events` | Tensor core (preprocessing) |
| `selectPreMaet` | `select_pre_maet` | Tensor core (preprocessing) |
| `bindAttributes` | `bind_attributes` | Tensor core (preprocessing) |
| `separateAttributes` | `separate_attributes` | Tensor core (preprocessing) |
| `sweptSimilarity` | `swept_similarity` | Tensor core (preprocessing composition) |
| `sweptEntropy` | `swept_entropy` | Tensor core (preprocessing composition) |
| `sweptMass` | `swept_mass` | Tensor core (preprocessing composition) |
| `nTupleEntropy` | `n_tuple_entropy` | Tensor core (preprocessing composition) |
| `showPreMaet` | `show_pre_maet` | Tensor core (utility) |
| `readPreMaet` | `read_pre_maet` | Tensor core (utility) |
| `writePreMaet` | `write_pre_maet` | Tensor core (utility) |
| `estimateCompTime` | `estimate_comp_time` | Tensor core (utility) |
| `addSpectra` | `add_spectra` | Spectra |
| `spectralEntropy` | `spectral_entropy` | Harmony |
| `templateHarmonicity` | `template_harmonicity` | Harmony |
| `tensorHarmonicity` | `tensor_harmonicity` | Harmony |
| `virtualPitches` | `virtual_pitches` | Harmony |
| `roughness` | `roughness` | Harmony |
| `dftCircular` | `dft_circular` | Circular |
| `dftCircularSimulate` | `dft_circular_simulate` | Circular |
| `balanceCircular` | `balance` | Circular |
| `evennessCircular` | `evenness` | Circular |
| `coherence` | `coherence` | Circular |
| `sameness` | `sameness` | Circular |
| `edges` | `edges` | Circular |
| `projCentroid` | `proj_centroid` | Circular |
| `meanOffset` | `mean_offset` | Circular |
| `circApm` | `circ_apm` | Circular |
| `markovS` | `markov_s` | Circular |
| `continuity` | `continuity` | Serial |
| `audioPeaks` | `audio_peaks` | Audio |
| `readScore` | `read_score` | Score input |
| `gridAttrTable` | `grid_attr_table` | Score input |
| `ungridAttrTable` | `ungrid_attr_table` | Score input |
| `preMaetFromAttrTable` | `pre_maet_from_attr_table` | Score input |
| `explainDispatch` | `explain_dispatch` | Diagnostics (§11) |
| `mptDefaults` | `set_default`, `get_default`, `get_defaults`, `reset_defaults`, `show_defaults` | Defaults (§11.4) |

### 10.11 Documentation

In MATLAB, use `help functionName`. In Python, use `help(mpt.function_name)` or access docstrings in your IDE.

---

## 11. Performance and numerical controls

Most quantities the toolbox computes can be reached by more than one route, and some can be computed to more than one accuracy. This section describes how a route is chosen, the four entropy estimators, the controls on kernel evaluation, and the toolbox-wide defaults. Most analyses need none of it: the defaults choose a route by cost, and every route computes the same value to within the accuracy floor.

### 11.1 Method selection

The toolbox has method-selection dispatchers for two of the toolbox's analytical computations: the **inner product** (used by `simMaet`, and internally by `entropyMaet` under `method='renyi2'` for the $\langle T, T \rangle$ term) and **point evaluation** (used by `evalMaet`, and internally by `entropyMaet` under `method='renyi2'` for the total-mass $\int T$ term). Each user-facing function accepts a `'method'` keyword whose `'auto'` setting (default) routes via feasibility, the declared measure, and a per-call cost model; the explicit alternatives let users force a specific decomposition. A dispatcher only chooses *how* a value is reached, never *which* value: every route computes the same quantity under the same declared measure, and two routes that are both admissible agree to within the accuracy floor set by `truncationSigmas`.

Four decompositions of the relevant tuple sums are available:

- **Bulger's method** (`method='bulger'`) — the inner-product decomposition due to David Bulger, in the toolbox since v1. Organizes the tuple sum by combinations on one side and permutations on the other, using the within-tuple multinomial symmetry to avoid enumerating equivalent orderings. **Inner product only.** Has essentially no per-call fixed overhead and tends to win at small $r$ and small $K$. On relative-periodic attributes it computes the pairwise-wrapped (minimum-image) quadratic form.

- **The Möbius method** (`method='mobius'`). Rewrites the tuple sum via Möbius inversion on the partition lattice: the distinct-index constraint becomes an alternating sum over set partitions, with each partition's term factorizing across its blocks. For the inner product this combines with **orbit collapse** under the joint element-permutation symmetry, reducing the partition-pair count from $B_r^2$ to $|\Omega_r|$ orbit equivalence classes (4, 10, 33, 92, 306, 948, 3210 for $r = 2, \ldots, 8$). **Applies uniformly to inner product, point evaluation, and total mass.** Carries fixed per-call overhead (orbit-table lookup, $|\Omega_r|$ tensor contractions) and tends to win at large $r$ or large $K$, where the tuple loop in Bulger's method or the materialized centres array blows up. The decomposition is grid-free in **absolute** mode ($O(B_r\cdot r\cdot K)$ per query for point evaluation, each partition block collapsing to an $O(K)$ event sum at the query). In **relative** mode it is *not* grid-free: the relative tensor is the translation marginal of the absolute tensor, and the alternating partition sum only factorizes across elements at fixed translation $u$, so the Möbius method integrates over a $u$-grid of $N_u$ points (or, for $2 \le r \le 4$ at sufficient $K$ and query count, over a Fourier mode grid). The evaluator serves that grid either by evaluating the absolute-mode sum at every node, at $O(B_r\cdot r\cdot K\cdot N_u)$ per query, or (where its validity conditions hold) by tabulating $r$ smoothed event distributions once and reading them back, at $O(B_r\cdot r\cdot N_u)$ per query; the choice between the two is internal and follows a cost gate, and the read-back's accuracy is tied to `truncationSigmas`. On a relative-periodic attribute the grid and spectral forms compute the all-image (transposition-average) reading, which is the density's definition. The underlying inclusion–exclusion technique was first published in Milne et al. 2011 (framed there as inclusion–exclusion rather than Möbius inversion on the partition lattice; the two are equivalent formulations) and has been used in MPT since v1 for numerical computation of discrete tensors; here it is applied as a closed-form analytical decomposition.

- **The centres method** (`method='centres'`) — the path that evaluates the closed-form expression as written, with no combinatorial decomposition. On **point evaluation** (`evalMaet`) it is the direct strategy: every $r$-tuple in the source Cartesian product is materialized as a Gaussian centre, and each query point sums kernel contributions from all centres. Numerically robust, and it tends to win at small $r$, where materializing the centres array is cheap. On the **inner product** (`simMaet`) it enumerates every ordered $r$-tuple on each side and sums kernel contributions over every pair, with no combinations-vs-permutations factoring and no alternating sum. Cost $O\!\left(\frac{K_x!}{(K_x-r)!} \cdot \frac{K_y!}{(K_y-r)!}\right)$ — far more expensive than either of the other two at any appreciable $K$, but exact at any $K \ge r$ and immune to cancellation, all summands being non-negative. It is the reference route: it shares no reduction with the other two, so agreement with it is a stronger check than agreement between them. Force `method='centres'` on the inner product for benchmarking or verification; under `'auto'` the cosine dispatcher never selects it. One algorithm, one name, across both functions. (`'direct'` was accepted in earlier versions and is retired; `'centres'` names the unrestricted enumeration in both functions.)

- **The nested contraction** (`method='contract'`) — the hierarchical contraction plan for a **nested** density (one built with `specs`), described in the paragraph on nested densities below. Inner product only, and rejected outright on a non-nested density.

**What `method` accepts.** `simMaet` accepts `'auto'`, `'bulger'`, `'centres'`, `'mobius'`, and `'contract'`; `evalMaet` accepts `'auto'`, `'centres'`, and `'mobius'`. Any other value — including the retired `'direct'` and `'factored'` — is an error. The list-mode and batched-raw forms forward `method` (and `truncationSigmas`, `kernelPrecision`) to every entry in both languages. There is no companion keyword: the former `'cancellationThreshold'` has been removed, and passing it raises the usual unknown-keyword error.

**The `'auto'` rules on `simMaet`.** For a pair without nested attributes the selector applies its rules in order, the first that fires deciding the route:

1. *Trivial order.* If $r_a = 1$ on every attribute, Bulger's method (the Möbius decomposition has nothing to decompose).
2. *Shipped orbit tables.* If any $r_a > 8$ (the largest tuple size whose orbit table ships), Bulger's method — after a feasibility guard that raises `SingleImageInfeasibleError` (Python) / `mpt:dispatch:singleImageInfeasible` (MATLAB) when the joint tuple-pair kernel would exceed the memory budget, rather than starting a call that cannot finish.
3. *Working set.* If either side's tuple-centres working set would exceed 256 MiB, the Möbius method, whose work is per event and per attribute rather than over the joint tuple set. This rule is skipped above the σ/period threshold of the next rule, which then decides.
4. *The measure rule.* On relative-periodic attributes the two readings of the periodic Gaussian — the minimum-image (single-image) reading, computed by Bulger's method and the tuple-centres closed form, and the all-image (full-image) reading, computed by the Möbius grid and spectral forms — agree to within the accuracy floor only up to a σ/period threshold that follows from `truncationSigmas` (0.03 at the default width of 6; tighter as more accuracy is asked for, and capped at 0.05 by positive-definiteness for widths of 4 or less). Below the threshold either route is admissible and the cost model decides. Above it the attribute's declared `wrap` decides: `wrap='full-image'` (the default) forces the Möbius method and `wrap='single-image'` forces Bulger's. Declaring both wraps across the relative attributes of a density, or different wraps on the same attribute of the two operands, is an error.
5. *The cost model.* Otherwise the predicted wall time of Bulger's method (from a fitted law in the tuple-pair count, with a memoized self inner product costing nothing) is compared with the predicted wall time of the Möbius method (per relative attribute, the cheaper of its tuple-centres and grid realizations, floored by a set-up cost per tuple size), and the cheaper route is taken; ties go to Bulger's method.

There is no timing probe: routing is decided from structure and the cost models alone, which is why the same call takes the same route on any machine (the hardware factor scales every route alike and cancels in the comparison). There is likewise no cancellation fallback: the Möbius method's agreement with enumeration is governed by `truncationSigmas`, and a small cosine is a legitimate value, not a symptom. Two rules sit outside the selector and apply whatever `method` says: an *ordered* attribute (`isExch = false` with $r_a > 1$) on either side routes the pair to Bulger's method, the only flat inner-product route that carries ordered tuples; and, after the Möbius arm has run, a **post-hoc guard** checks its output for an impossible value — a non-finite inner product, a negative self inner product, or $|\langle X, Y\rangle| > 1.000001\sqrt{\langle X, X\rangle\langle Y, Y\rangle}$ — and on finding one warns, discards the Möbius memo entries on both densities, and re-runs the call through Bulger's method. The guard has no threshold to tune; it is disabled by the `postHocGuards` default.

**The `'auto'` rules on `evalMaet`.** The eval selector chooses between the centres branch and the Möbius evaluator, again in order: an ordered attribute, or $r_a = 1$ on every attribute, goes to centres; a density whose attributes are all nested is decided on its own fitted cost row, the tag-tree centres enumeration against the per-level Möbius evaluator, and a density that mixes nested and flat attributes adds that row to the flat law in the cost model below; any flat $r_a > 10$ goes to centres after the same feasibility guard on the joint working set, that bound being where the set-partition sum itself becomes infeasible rather than anything to do with the orbit tables, which this evaluator does not use; then the measure rule above, on the first relative-periodic attribute over the threshold (`'single-image'` forces centres, `'full-image'` forces Möbius); otherwise the cost model estimates both for the given number of query points and the Möbius evaluator is taken when it is predicted cheaper — by a safety factor of 1.5 when the centres working set exceeds 256 MiB, or outright below that. A forced `'mobius'` is rejected on a density with an ordered attribute; on a nested density it runs the per-level Möbius evaluator, which applies the Möbius set-partition sum at each symmetric level of the tag tree (and a dynamic programme at each ordered one) and so touches no tuple centre — on a density with thousands of nested tuple centres per event it is orders of magnitude faster than the centres branch, and since the cost row was fitted `'auto'` routes to it whenever the row says it is cheaper. After the Möbius evaluator has run, a post-hoc guard re-runs any call whose output is non-finite through the centres branch, with a warning; it is disabled by the same `postHocGuards` default. Inside the centres branch the toolbox picks the shape of the sum for itself (the single-multiset kernel sum, the factored per-attribute form, or the joint materialization); none of these is user-selectable, and all compute the same value.

**Seeing the route before running.** `explain_dispatch(dens_x, dens_y)` / `explainDispatch(densX, densY)` (or, with a single density and a query count, the evaluation form) reports which route the call would take and why — the predicted time of each candidate, the accuracy floor in force, the σ/period limit that follows from it and which test set it, and where the call sits relative to them — without running the call. It invokes the very same selectors, so what it reports is what would happen. Pass `method` and `truncationSigmas` to see the effect of an override.

**Overrides.** Most callers should leave `method` at the default `'auto'`. Override only when measuring the agreement between two routes, or forcing a specific method for benchmarking. A forced `method` bypasses the cost race and the measure rule, but never the structural guards: the feasibility guards still raise, `'contract'` still requires a nested density, and on the cosine path the ordered-attribute rule and the post-hoc guard still apply. On a relative-periodic attribute above the σ/period threshold a forced route computes that route's reading of the kernel, so forcing `'bulger'` on a `wrap='full-image'` attribute (or `'mobius'` on a `wrap='single-image'` one) is a different number, not a faster one; the nested plan, by contrast, refuses a forced route that its declared measure does not admit.

**`method` on a nested density.** On a nested density (one built with `specs`) the `method` names select among the nested path's own routes rather than the flat entry points, because a flat route cannot represent a nested attribute's block structure. `'auto'` (the default) leaves the choice to the dispatcher, which estimates the cost of the contraction plan against joint-tuple enumeration and takes the enumeration only when it is predicted cheaper by a factor of two and is itself admissible. `'contract'` forces the hierarchical contraction plan: it fixes one route per attribute – the level-by-level contraction, materialized tuple centres, the line grid, or the τ-grid – choosing among the routes that carry the attribute's declared measure by the same kind of cost model the flat dispatcher uses, and it raises an error rather than falling back on any case it does not cover (it is rejected outright on a non-nested density). `'mobius'` names that same plan, because the per-level orbit reduction is exactly what the contraction applies at every symmetric level, so on a nested density `'mobius'` and `'contract'` coincide. `'centres'` forces the materialized-centres route for every attribute; on a relative-periodic attribute whose declared measure is the default full-image one, that route is admissible only up to the σ/period threshold, above which `'centres'` raises an error naming the `wrap='single-image'` opt-in (in MATLAB, identifier `simMaet:centresUnavailable`). `'bulger'` remains the joint-tuple enumeration. Whichever plan runs, the measure is declared by the attribute's `wrap` and never by the dispatch: `wrap='full-image'` (the default) declares the all-image reading over the torus and `wrap='single-image'` the minimum-image one, and a minimum-image route is admitted for a full-image attribute only where the two agree inside the truncation floor – a cheaper route to a different number is not a cheaper route. The per-call `truncationSigmas` governs the nested contraction as it governs every other route.

The Python and MATLAB implementations share every routing rule and the form of every cost model (the fitted constants are per-language), so the same density and the same `method` take the same route in either language. The same rules are drawn as decision trees in ARCHITECTURE §4 ("The routing figures"), which also defines the vocabulary they use — the routes, the measures, the guards, and the cost models.

**Demos.** `demo_dispatchAndKernelControls` / `demo_dispatch_and_kernel_controls.py` (parts 1–3).

### 11.2 The four entropy estimators

`entropyMaet` accepts a `'method'` keyword with four choices: `'shannon'`, `'normalized'` (alias `'normalised'`), `'differential'`, and `'renyi2'`. The four-method API distinguishes the discrete-grid and continuous-form entropies, and the kwarg `n_points_per_dim` is required only for the discrete methods.

- **`method='shannon'`.** Raw discrete Shannon entropy $H = -\sum_q q \log_b q$ on a Cartesian-product grid of resolution `n_points_per_dim` per effective dimension. Bin-mass integration via per-axis $\Phi$-difference contractions for `is_rel=False` densities; point-evaluation for `is_rel=True`. On a periodic attribute the cell masses (shared with `method='normalized'`) honour the attribute's `wrap`: under the default `'full-image'` the erf is summed over every periodic image the truncation admits, under `'single-image'` over the nearest image only, so the two readings part company as σ/period grows (from about 0.06 the difference is visible in the value). Supports the full polymorphic-input dispatch (single density, list of densities, raw scalar or batched single-multiset form, raw multi-attribute form).

- **`method='normalized'` (alias `'normalised'`).** Explicit name for the Pielou-style ratio $H / \log_b N$ in $[0, 1]$. It reproduces the values published in Milne et al. (2017) and Smit et al. (2019). Useful for evenness comparisons within a fixed grid alphabet, but **grid-dependent**: the normalizer $\log_b N$ depends on the discretization, so cross-density comparisons between systems of different cardinality or spread require additional care.

- **`method='differential'`.** Adaptive nested-grid evaluation of the differential entropy $\hat h = H_\text{disc} + \log_b(\Delta\text{-volume})$. The span auto-derives per group from `centres ± truncation_sigmas · sigma` (non-periodic) or $[0, \text{period}]$ (periodic); the grid doubles from a sample-per-sigma initial resolution until successive Richardson-extrapolated estimates fall below the truncation-anchored tolerance $\max(\exp(-\text{truncation\_sigmas}^2 / 2), 10^{-12})$. Grid-independent (no caller choice of grid), and the principled scale-free quantity for comparisons across densities of different cardinality, spread, or support. Currently restricted to single-density input. Errors at `sigma=0` (the continuous form diverges).

- **`method='renyi2'`.** Analytical Rényi-2 (collision) entropy $H_2 = -\log_b(\langle T, T\rangle / Z^2)$ computed in closed form via the Möbius method's inner product for $\langle T, T\rangle$ and total mass for $Z$. No grid, no caller choice of resolution; the result is exact (analytical) and works at high $r$ where any grid path would exhaust memory. Currently restricted to single-density input; errors at `sigma=0`. The Rényi-2 entropy of a relative $r = 1$ attribute is 0 by convention: a relative monad is a zero-dimensional point mass and contributes no entropy.

The two continuous methods (`'differential'`, `'renyi2'`) ignore `n_points_per_dim` and the grid-bounds kwargs (`x_min`, `x_max`); the two discrete methods require an explicit `n_points_per_dim` (there is no toolbox-wide default). `spectralEntropy` defaults to `method='differential'`; `nTupleEntropy` defaults to `method='normalized'` (the discrete formulation of Milne & Dean, 2016, at $\sigma = 0$). `MIGRATION.md` covers code written for the earlier implicit grid.

**Demos.** `demo_dispatchAndKernelControls` / `demo_dispatch_and_kernel_controls.py` (part 7).

### 11.3 Kernel-evaluation controls

Four user-controllable defaults govern how the toolbox handles the dense Gaussian kernel matrix in functions that build one. Two of them — `truncationSigmas` and `kernelPrecision` — trade accuracy for speed in the matrix arithmetic; the third — `kernelChunkBytes` — sets the per-chunk byte budget for the memory-aware chunkers that allow large workloads to run without exhausting RAM; the fourth — `kernel_threads`, Python only — sets how many threads the elementwise kernel arithmetic may be spread over. The first two apply to the centres path (`evalMaet`, `entropyMaet` with `method='shannon'`, `spectralEntropy`, `templateHarmonicity`, `virtualPitches`) and to Bulger's method on `simMaet`. The Möbius method builds no kernel matrix, but `truncationSigmas` governs its accuracy too, in the inner product and in point evaluation, and `kernelPrecision` applies to its point evaluation, whose reduced kernel sums it casts (under `'single'` its read-back target is floored at about $10^{-7}$), though not to its inner product. The Möbius inner product is a signed (inclusion–exclusion) sum over partitions, whose value can be a small difference of much larger terms, so single-precision rounding of those terms would be magnified by the cancellation; point evaluation takes `kernelPrecision` only once its sum has reduced to Gaussian kernel sums of positive terms, where no such cancellation arises. The third applies more broadly — to every chunker in the toolbox, including the Möbius relative-mode evaluator's orbit-matrix chunker.

- **`truncation_sigmas` (Python) / `truncationSigmas` (MATLAB)** — type `float`, default `6`. Gaussian kernel contributions are skipped when the centre-to-query distance exceeds $k\sigma$, equivalently when the kernel value drops below $\exp(-k^2/2)$. Truncation perturbs the result by a small absolute amount bounded by that $k\sigma$ tail: the worst-case error versus the exact result is about 1e-3 at $k = 4$, 1e-5 at $k = 5$, 2e-8 at the default $k = 6$, and 1e-10 at $k = 7$, reaching the toolbox's 1e-12 cross-language parity floor by $k = 8$. This is an absolute error — on a cosine similarity directly, and on an evaluated density as a fraction of its scale — and is somewhat larger at large tuple size $r$, where more pairwise coordinates can sit near the truncation boundary at once; the relative error at individual query points deep in the tails is unbounded, but those points carry negligible mass and are never summed, integrated, or compared. Each added sigma shrinks the error by $\exp(-(2k+1)/2)$ at a cost that grows as $k^{r}$ with tuple size, so accuracy is inexpensive. Set `math.inf` / `Inf` for accuracy-floor evaluation: it resolves to the finite width (~7.43σ) at which the kernel falls below the 1e-12 floor, the reference against which cross-language parity is verified. The rule applies on every route, including matrix-valued kernel covariances (`kernel_cov` / `kernelCov`), where it holds in the whitened (Mahalanobis) metric. It is implemented as a grid-bucket spatial index where that pays and as a mask on the exhaustive path otherwise; the two give identical results.

- **`kernel_precision` (Python) / `kernelPrecision` (MATLAB)** — accepts `'double'` (default) or `'single'`. With `'single'`, the kernel-matrix arithmetic casts to `float32` for a workload-dependent speedup (typically ~2× on compute-bound problems, less on memory-bandwidth-bound problems) at the cost of approximately 7 significant figures of precision (vs approximately 15 for double). The cast applies only to the kernel matrix; density coordinates and the final accumulation are preserved at full double.

- **`kernel_chunk_bytes` (Python) / `kernelChunkBytes` (MATLAB)** — accepts `'auto'` (default) or a positive integer (bytes). Sets the per-chunk byte budget used by the memory-aware chunkers when a kernel-matrix or orbit-matrix workload would otherwise exceed available RAM in a single allocation. The `'auto'` value resolves at call time to half of currently available physical memory, queried from the operating system: `/proc/meminfo`'s `MemAvailable` on Linux, `vm_stat`'s `free + inactive + speculative` pages on macOS, `memory().PhysicalMemory.Available` on Windows. The half-of-available factor leaves a safety margin against the broadcast difference tensor, its square, and the summed-then-exponentiated intermediate being briefly co-resident during a chunk's evaluation. A 4 GiB fallback is used if all platform queries fail. An explicit positive-integer override is taken as-is. Set a lower value to reduce peak memory pressure (smaller chunks, more loop iterations, no change in result up to floating-point reduction order); set a higher value on a machine where the toolbox should claim more memory than the half-available default.

- **`kernel_threads` (Python only)** — accepts `'auto'` (default) or a positive integer. Sets how many threads the elementwise Gaussian arithmetic — the bulk of every evaluation route — may be spread over. That arithmetic is independent from one evaluation point to the next, and NumPy releases the interpreter lock inside its array loops, so threading it scales nearly linearly until memory bandwidth saturates: on a sixteen-core machine, about 7× on the kernel portion of a call. The work is split into contiguous spans of evaluation points and each point keeps the arithmetic it has serially, so **the results are bit-identical at any thread count**; only the elapsed time changes. Threads are used only above an internal work threshold, so small calls are unaffected either way, and the per-chunk byte budget is divided among the threads, so peak memory is what it would have been serially. `'auto'` takes `OMP_NUM_THREADS` when the environment sets it — the convention by which a batch scheduler communicates its allocation — and otherwise the core count capped at eight. **Set it to 1 when parallelising at a higher level**, such as a `multiprocessing` pool, `joblib`, or a cluster job array: each worker fanning out again oversubscribes the machine, which is usually slower than running serially. MATLAB needs no counterpart, since its runtime threads elementwise arithmetic itself and reduces each `parfor` worker to a single computational thread; this is therefore one of the toolbox's deliberate language-idiomatic divergences.

`truncationSigmas` and `kernelPrecision` can be set per call (as keyword arguments) or globally via the toolbox-wide defaults API described in the next section. Per-call kwargs always override globals; globals always override factory defaults. `kernelChunkBytes` is global-only: the chunkers consult it on every call, so a per-call kwarg would be a layering inversion. `kernel_threads` is likewise global-only, and for the same reason. The factory defaults are `truncationSigmas = 6` (fast, worst-case error ~2e-8 versus the accuracy floor; set `Inf` for accuracy-floor (1e-12) evaluation), `kernelPrecision = 'double'`, `kernelChunkBytes = 'auto'`, and (Python only) `kernel_threads = 'auto'`.

**Demos.** `demo_dispatchAndKernelControls` / `demo_dispatch_and_kernel_controls.py` (parts 4–5).

### 11.4 Toolbox defaults

`mptDefaults` (MATLAB) and `mpt.set_default` / `mpt.get_defaults` / `mpt.reset_defaults` / `mpt.show_defaults` (Python) provide the canonical entry points for inspecting and changing toolbox-wide settings. The settings currently controlled are `truncationSigmas`, `kernelPrecision`, `kernelChunkBytes`, `showHints` (a boolean that gates the toolbox's informational dispatch-decision messages described below), and, in Python alone, `kernel_threads`.

**Call forms (MATLAB):**

| Form                                  | Effect                                                                            |
| :------------------------------------ | :-------------------------------------------------------------------------------- |
| `mptDefaults`                         | Print current values plus a brief summary of each field.                          |
| `S = mptDefaults`                     | Return the current values as a struct, silently.                                  |
| `val = mptDefaults('name')`           | Return one value.                                                                 |
| `mptDefaults('name', val, ...)`       | Set one or more values. Returns the previous values as a struct.                  |
| `mptDefaults(prevStruct)`             | Restore from a previously-returned struct. Inverse of the setter form.            |
| `mptDefaults('reset')`                | Reset all to factory defaults.                                                    |

**Python equivalents:**

| Form                                  | Effect                                                                            |
| :------------------------------------ | :-------------------------------------------------------------------------------- |
| `mpt.show_defaults()`                 | Print current values plus a brief summary of each field.                          |
| `mpt.get_defaults()`                  | Return the current values as a dict.                                              |
| `mpt.get_default('name')`             | Return one value.                                                                 |
| `mpt.set_default(name=val, ...)`      | Set one or more values. Returns the previous values as a dict.                    |
| `mpt.set_default(**prev)`             | Restore from a previously-returned dict.                                          |
| `mpt.reset_defaults()`                | Reset all to factory defaults.                                                    |

The setter form returns the previous values precisely so they can be passed back in to restore the prior state. The idiomatic pattern is "save, change, work, restore":

```matlab
% MATLAB
prev = mptDefaults('truncationSigmas', Inf, 'kernelPrecision', 'single');
% ... do work with the new defaults ...
mptDefaults(prev);                                 % restore
```

```python
# Python
prev = mpt.set_default(truncation_sigmas=math.inf, kernel_precision='single')
# ... do work with the new defaults ...
mpt.set_default(**prev)                            # restore
```

This is safer than `'reset'` after a block of work, because `'reset'` overwrites any *other* defaults the caller might have set deliberately before the block. The save-and-restore pattern preserves anything the caller didn't explicitly change.

Defaults persist within a single Python process / MATLAB session (not across `clear all` or `import`-reload cycles).

**Informational messages.** The toolbox prints two kinds of informational message that are not gated by per-call `verbose`:

- A **first-use truncation warning**, issued once per session the first time a kernel evaluation resolves the truncation default at its factory value of 6 (i.e. the user has not set their own `truncationSigmas`). It gives the worst-case error versus the exact result at 4, 5, and 6 sigma (about ~1e-3, ~1e-5, and ~2e-8 respectively) and points to `mptDefaults('truncationSigmas', Inf)` for exact output. It fires from the defaults getter (`mptDefaults('truncationSigmas')` in MATLAB, `get_default('truncation_sigmas')` in Python), through which every kernel-evaluating path resolves the default, so it appears on first use regardless of which function the script calls. It is issued as a warning — identifier `mpt:truncationDefault` in MATLAB, category `mpt.TruncationDefaultWarning` in Python — so it goes to stderr and is suppressible through the usual warning controls (`warning('off', 'mpt:truncationDefault')`; `warnings.filterwarnings('ignore', category=mpt.TruncationDefaultWarning)`), and it never lands inside a script's own stdout output. It is **not** gated by `showHints` — it always gets its single showing, so a script that sets `showHints = false` to quiet the dispatch messages still sees the warning once. It reappears in a fresh session (`clear all` / restart), and stays silent only after that first showing or once any explicit truncation value is set.
- **Dispatch decisions** from `evalMaet` and `simMaet`: short messages naming the path the dispatcher selected. The message reads `simMaet: chose 'bulger' path.`; no timing is taken, the route following from structure and the cost model alone (see "Method selection" above). All are throttled to once per top-level user call per unique `(function, chosen)` pair: nested toolbox calls within one user call share a seen-set so a batched-raw loop or a list-mode `sweptSimilarity` sweep produces one announce per unique chosen path (not one per item, and not one per routing reason — two different reasons leading to the same chosen path collapse to a single announce). Each new top-level call re-announces. The seen-set is cleared automatically on every top-level toolbox entry, and also explicitly by `mptDefaults('reset')` / `mpt.reset_defaults()`.

The **dispatch messages** are gated by the `showHints` flag (factory default `true`); the first-use truncation notice is not (it always shows once). Set `showHints` to `false` to silence the dispatch messages:

```python
mpt.set_default(show_hints=False)        # Python
```

```matlab
mptDefaults('showHints', false)          % MATLAB
```

Note: dispatch messages are deliberately *not* gated by per-call `verbose`, because internal toolbox callers (e.g. the batched-raw path inside `simMaet`, the per-pair calls inside `entropyMaet`) pass `verbose=False` to inner calls to prevent flooding. With the per-top-level-call throttle in place this is no longer a concern, and users benefit from seeing the routing decision regardless of any internal `verbose=False`.

**Progress feedback for long batched calls.** Distinct from the `showHints`-gated dispatch announces above, the batched helpers (`simMaet`, `templateHarmonicity`, `spectralEntropy`, `virtualPitches`, `entropyMaet`) also emit two kinds of per-call progress output, both gated on `verbose=True` (the default) and silenced by `verbose=False`. Both are driven by a small up-front empirical calibration (a few sample rows are warmed up and timed to estimate the per-row cost). The first kind is an **estimated total time**, printed only if the extrapolated total exceeds ~10 s: e.g. `simMaet: estimated 1.5 min; Ctrl+C to cancel.`. The second kind is a **periodic countdown** of rows completed, printed only if the extrapolated total exceeds ~5 s, with the print cadence picked from `{1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000}` so each interval takes at least 5 s at the calibrated per-row cost — slow per-row work prints every few rows, fast per-row work prints every several thousand. Together this gives a steady "still working" indicator for long runs without flooding the terminal; calls expected to complete in under 5 s see neither output. The unit reported in the countdown is the natural one for each function — `unique pairs computed` for `simMaet` (after canonical-form deduplication) and `rows computed` for the others.

**Demos.** `demo_dispatchAndKernelControls` / `demo_dispatch_and_kernel_controls.py` (part 6).

---

## 12. Function reference

Functions are grouped by the stage of an analysis they serve, in the order of Part II. For full details use `help functionName` in MATLAB or `help(mpt.function_name)` in Python. Code in this section uses MATLAB syntax; §10 gives the systematic Python equivalents.

### 12.1 Score and audio input

A score is read into an attribute table with `readScore`, optionally sampled on a grid with `gridAttrTable`, and converted into a pre-MAET with `preMaetFromAttrTable`, which reads a table and nothing else. The table is an ordinary MATLAB `table` or pandas `DataFrame`, so it is inspected and filtered with the host language's own tools. §5 describes the path.

| MATLAB | Python | Description |
|:---|:---|:---|
| `readScore` | `read_score` | Parse a MIDI or MusicXML file into an attribute table |
| `gridAttrTable` | `grid_attr_table` | Sample an attribute table on a regular grid, a held note replicating across the points it covers |
| `ungridAttrTable` | `ungrid_attr_table` | Return a gridded attribute table to the table it was made from |
| `preMaetFromAttrTable` | `pre_maet_from_attr_table` | Build a pre-MAET from an attribute table |
| `audioPeaks` | `audio_peaks` | Extract spectral peaks from audio |
| `transformAttributes` | `transform_attributes` | Scale conversions and elementwise transforms (entry in §12.3) |

Pitch and frequency scale conversion is the bare-array form of `transformAttributes` (§12.3): `transformAttributes(values, [], {fromScale, toScale})` converts between `'hz'`, `'midi'`, `'cents'`, `'octave'`, `'mel'`, `'bark'`, `'erb'`, and `'greenwood'`, routing through Hz. Vectorized: accepts scalars, vectors, or matrices. Note that the `'cents'` scale is absolute MIDI cents (A4 = 6900, middle C = 6000), not relative interval cents.

**readScore(path)** — Parses a `.mid` / `.midi` (format 0 or 1), `.musicxml` / `.xml` (partwise or timewise), or `.mxl` file into an attribute table: a MATLAB `table` (Python: pandas `DataFrame`) with one row per sounding note, sorted by onset, part, pitch. Columns carried by both sources: `onsetBeats`, `onsetSeconds`, `durationBeats`, `durationSeconds` (Python `onset_beats` and so on), `pitch` (MIDI number, floating point), `velocity` (0–127; MusicXML `dynamics` × 0.9, 90 where absent), `part` (categorical, the part names as its categories), and `measure`. A MIDI file adds `channel` (1–16), `noteNumber` (the number as recorded, which is what note identity rests on), `program` (the program change in force on that channel at the note's onset, 0 where none was sent, which selects the instrument sound), `weight`, and `soundingDurationBeats` / `soundingDurationSeconds`. A MusicXML score adds `voice`, `staff` (1-based; a part written on more than one staff, as a keyboard part is, says which each note is on), `fermata`, and the articulations `staccato`, `accent`, and `tenuto`. The boolean marks are not mutually exclusive — a note may be both staccato and accented — so each is its own column, and a merged tied note carries a mark any of its segments carries. Three MIDI controller streams are resolved at read, since each changes a note's own columns: sustain (CC64) and sostenuto (CC66) into the sounding durations, with a re-strike on the same channel damping the tail; pitch bend into `pitch`, which is therefore floating point, at 2 semitones unless RPN 0 or an MPE zone says otherwise (a file that bends without declaring either warns); and channel volume (CC7) and expression (CC11) into `weight`, as `(velocity/127) · (cc7/127)² · (cc11/127)²` — the squares being MIDI's specified default response for both controllers (an attenuation of `40·log10(cc/127)` dB, cascaded and so multiplied), and the velocity factor linear because MIDI specifies no velocity curve and because that makes `weight` equal to the `'velocity'` weighting on any file that sends no controller. `weight` is therefore that weighting corrected by the channel's gain, not an estimate of sounding amplitude. No other controller is read, a value sampled at a note's onset being no account of a ramp inside a held note. A column is present only where the source carries the information, so `channel` and `voice` are never the same column and never stand in for one another. The source is in the table's description (`Properties.Description`; Python `df.attrs['source']`). A beat is a quarter note whatever the time signature; seconds follow every tempo change (120 bpm where a file gives none).

**gridAttrTable(T, step, ...)** — Samples an attribute table on a regular grid, one event per grid point. The grid is a series of time points spaced `step` apart. Each point opens a *slice*, reaching from it up to the next, and a note occupies every slice it sounds in, so a held note occupies several and a note shorter than the step still occupies one. A note that ends exactly where a slice begins does not occupy it. The points are equally spaced in the unit the grid steps in — in beats for a beat grid; under a changing tempo that is not equal spacing in seconds, so the slices last different amounts of clock time. A slice with nothing sounding becomes a row whose note columns are all missing; those rows are kept, since they hold the place that makes the event index a uniform index of time, and dropping them is one selection away. Name-value pairs: `'time'` (`'beats'` or `'seconds'` — a metrical grid presupposes a beat map, which a score has and a bare performance may not), `'duration'` (`'duration'` or `'soundingDuration'`, which defines occupancy), `'weights'`, and `'limits'`. The weighting says what a slice takes from a note overlapping it. `'coverage'` (the default) takes the fraction of the slice the note fills — how the slice is filled — which is Analysis 1.3's weighting, "the fraction of the eighth each note sounds". `'presence'` takes the note's full weight in every slice it appears in at all, however briefly: which notes are here, rather than how much of the slice each occupies. It is a membership reading: each slice records the set of what occurs in it, whatever the step. It separates from coverage as the step grows relative to the notes — at a bar-length step, say, a slice holds the set of what occurs in that bar, where coverage would hold a duration-weighted profile of it. `'item'` takes the fraction of the note in the slice, so that its weight is distributed over the slices it spans and it counts once in total; for an attribute constant over the note this gives a density identical to the ungridded one at `r = 1`, the kernel being linear in weight. Coverage and presence differ only where a note does not fill a slice, so on a grid at or finer than the shortest note they agree. For what is sounding at a given moment, the instrument is coverage on a fine grid: an instant has no duration, and coverage approaches the momentary reading as the step shrinks. Adds `gridIndex`, `gridOnsetBeats` and `gridOnsetSeconds`, `noteId` (missing on an empty point, and naming the row of the source each grid row came from), and `weight`. The grid steps in the chosen unit but its points have a time in both, so a metrical grid can be read on a clock — slices of a sixteenth with a σ in milliseconds; the unit the grid did not step in is interpolated from the table's note samples (one pair per onset, one per note end) and continued at the nearest rate beyond the first and last, which is exact wherever the tempo is constant across the bracketing samples and approximate only across a tempo change inside a gap. Only the stepped unit is present where the source carries no second time base. `duration` still means the note's own duration; the slice length is a property of the grid. An already-gridded table regrids at a coarser step, and gridding composes: `gridAttrTable(gridAttrTable(T, fine), coarse)` equals `gridAttrTable(T, coarse)`, each policy combining in the way that makes this hold — coverage sums and rescales by fine / coarse, item sums, presence takes the maximum. That is the route to a weighting the toolbox does not itself offer: weight the fine slices as the analysis requires, then coarsen, and the weights carry. The coarse step must be a whole multiple of the fine one, the time base must be the one the table was gridded over, `'weights'` must name the policy the table's weights already carry, and `'limits'` is refused, the span being the one the table covers. `preMaetFromAttrTable` binds a gridded table on its grid position rather than on onsets.

**ungridAttrTable(G)** — The inverse of `gridAttrTable`: `noteId` names the row of the source each grid row came from, so keeping the first row of each and removing the columns the grid wrote returns the source table. Which columns those are is the toolbox's business rather than the caller's, and it is not a fixed list — the grid *adds* `weight` to a table that had none and *folds* the weight of one that did into a per-slice one, so a caller comparing the two tables' columns would keep a `weight` whose values are no longer the note's but the slice's. Three things do not come back: a note that `'limits'` cut out of the grid's span, which is a truncation the caller asked for; the source's own `weight` column where it had one, the grid having folded it into a per-slice weight; and, in MATLAB only, a logical column, which returns as double, MATLAB having no missing logical for the grid to have given an empty point — Python's nullable `boolean` round-trips. Rows and columns may go before ungridding: the empty points, a run of slices, a column. A note whose first slice was dropped comes back from its next, since the source's columns repeat unchanged across its slices; a note whose slices were all dropped does not come back, having been selected away. Where a kept column takes more than one value within a note — which only a column the caller added can do — the first slice's value is taken, with a warning, a per-slice quantity having no single value to collapse to.

**preMaetFromAttrTable(T, ...)** — `T` is an attribute table, as `readScore` returns and `gridAttrTable` passes on; a score file is read first, with `readScore`, so converting reads a table and nothing else. Name-value pairs: `'attributes'` (required: one entry per attribute, each a struct/mapping naming its `column` together with that attribute's parameters: `name`, `sigma`, `r`, `exch`, `rel`, `isPer`, `period`. The ten names `'pitch'`, `'onset'`, `'duration'`, `'soundingDuration'`, `'velocity'`, `'weight'`, `'noteNumber'`, `'part'`, `'measure'`, and `'fermata'` get the score-specific treatment — the pitch scale, beats against seconds, the grid's onset — and all but `'pitch'` and `'onset'` raise where the source does not carry the column, each being read only where something names it, so a table of two columns converts; any other column of the table is read as it stands, so a table that never saw a score converts too, while a categorical column is refused here and belongs to `'roles'`. Listing one column twice gives two attributes of the same values read under different parameters, which is how pitch class and pitch height are taken from one pitch column), `'pitch'` (scale; default `'midi'`), `'time'` (`'seconds'` or `'beats'`), `'weights'` (`'velocity'`, `'ones'`, `'duration'`, `'weight'`, each asking for its own column), `'roles'` and `'groupBy'` (§5.3), `'parts'` (parts to keep, as 1-based positions or as part names), `'chords'` (`'bind'` or `'separate'`), `'chordTolerance'`, `'names'`. Returns the pre-MAET (§6.1): its `pAttr` is K_a × N per attribute, its `wAttr` the matching weight matrices with 0 in NaN-padded slots (or empty under `'ones'`), and its `specs` named flat specs. The pre-MAET is complete — ready for `buildMaet` with nothing set on the specs afterwards — because the conversion fills in only what follows from the data or from another argument (the values, `r` and `exch` under a structural role, and reading a value as written for `rel` and `isPer`) and asks for the rest: `sigma` always, and `r` and `exch` where an attribute holds more than one value at an event and no role has fixed them. A score fixes what the values are, not how tolerant a match is nor how many of an event's values a tuple takes, and none of those has an identity to default to. It feeds `buildMaet` or any pre-MAET preprocessor (`transformAttributes` for a log-time or cents view, `differenceEvents` for intervals and IOIs, `bindEvents` for n-grams). §5.1 has the conventions.

**audioPeaks(audioFile, ...)** — Reads an audio file, computes the magnitude spectrum, and extracts peaks. Returns frequencies in Hz and normalized amplitudes in [0, 1]. Optional name-value pairs: `'sigma'` (smoothing in cents; default: 0), `'resolution'` (cents grid spacing; default: 1), `'rampDuration'` (onset/offset ramp in seconds; default: 0), `'fMin'`, `'fMax'`, `'minProminence'`, `'noiseFactor'`, `'plot'`.

When sigma > 0, the spectrum is resampled onto a uniform log-frequency (cents) grid before smoothing. This ensures the Gaussian kernel has a fixed perceptual width at all frequencies (a fixed-Hz kernel would over-smooth high partials and under-smooth low ones). Peak positions are converted back to Hz. This smoothing is useful for audio with vibrato or frequency jitter: partials separated by more than approximately 2σ cents are individually resolved, while closer partials are merged into a single peak. The sigma value should therefore be chosen to match the expected extent of frequency variation in the audio. Smoothing is unnecessary for steady-state tones, since the downstream toolbox functions already apply their own Gaussian smoothing via the expectation tensor framework.

### 12.2 Pre-MAET objects

| MATLAB | Python | Description |
|:---|:---|:---|
| `packPreMaet` | `pack_pre_maet` | hold a pre-MAET's three parts in one object, converting any attribute given per event to its matrix (§6.1) |
| `unpackPreMaet` | `unpack_pre_maet` | split a pre-MAET back into `pAttr`, `wAttr`, and `specs` (§6.1) |
| `flatSpecs` | `flat_specs` | synthesize the canonical flat `specs` for bare attributes |
| `kernelCov` | `kernel_cov` | build a matrix-valued kernel covariance for an ordered tuple, of values or of consecutive differences, from three sources of variance |
| `simplexVertices` | `simplex_vertices` | regular-simplex vertices for level-symmetric categorical attributes |
| `showPreMaet` | `show_pre_maet` | print a pre-MAET as a table, in markdown, as a LaTeX tabular, or as CSV (full entry in [§6.2](#62-viewing-a-pre-maet)) |
| `readPreMaet` | `read_pre_maet` | read a pre-MAET from a CSV file a spreadsheet can edit (§6.4) |
| `writePreMaet` | `write_pre_maet` | write a pre-MAET as CSV, the inverse of `readPreMaet` (§6.4) |

The functions below build the objects the core consumes: the pre-MAET itself, its level geometry, the coordinates a categorical attribute is encoded with, and the matrix-valued kernel covariance of an ordered difference attribute. None of them is a pre-MAET operation.

**packPreMaet(pAttr [, wAttr] [, specs])**

Builds the pre-MAET: a MATLAB struct with the fields `pAttr`, `wAttr`, and `specs`, and a Python dict with the keys `p_attr`, `w_attr`, and `specs` (§6.1). The three parts always travel together and always describe the same pre-MAET, so one variable holds the whole of it, and every function that takes a pre-MAET takes it whole or in its parts. It is a plain struct or dict, not a class: the parts stay ordinary cells, arrays, and structs. `wAttr` is empty for unweighted, a scalar applied to every attribute, or a length-$A$ list/cell; `specs` is empty or a length-$A$ list/cell. An existing pre-MAET in place of `pAttr` is validated and returned afresh, with any part given here replacing the one it holds — which is how a variant is made, `packPreMaet(pm, [], specs2)`. Cross-part inconsistencies (a `wAttr` or `specs` whose length is not $A$, a single attribute or spec passed unwrapped) are errors here rather than further downstream.

**unpackPreMaet(pm)**

Splits a pre-MAET into `pAttr`, `wAttr`, and `specs`, in the order the loose-triple signatures take them (§6.1); the unset parts come back empty. The inverse of `packPreMaet`, for the places that want the parts separately.

**flatSpecs(pAttr, 'r', r, 'rel', rel, 'exch', isExch [, 'name', name])**

Convenience constructor for the canonical `specs`. Given a length-$A$ list/cell of per-attribute value matrices (used only for its length $A$ — the values are not inspected), returns a length-$A$ list of flat spec structs `struct('r', ., 'rel', ., 'exch', .)`, broadcasting scalar geometry across attributes. This is the trivial flat-specs synthesis at the entry of a pre-MAET chain (raw attributes carry no level structure yet) and an ergonomic alternative to hand-writing the structs for `buildMaet(..., 'specs', specs)`. Name-value pairs: `'r'` (scalar or length-$A$ read-arity, default 1), `'rel'` (scalar or length-$A$ `[rel]`, default false), `'exch'` (scalar or length-$A$ `[exch]`, default true), and `'name'` (optional per-attribute names). Most pre-MAET chains never call it directly — the preprocessing operators synthesize flat specs internally when `specs` is omitted — but it is the explicit hook when you want to set non-default `r`, `[rel]`, or `[exch]` before differencing, binding, translating, or weighting.

**kernelCov(r, 'differenced', tf [, 'sdValue', sp] [, 'sdInterval', si] [, 'sdShift', ss])** — Kernel covariance for an ordered tuple of $r$ values from three sources of variance: independent noise on the values themselves ($sp$), independent noise on the intervals between consecutive values ($si$), and a common shift of the whole tuple ($ss$). The mandatory `differenced` flag says whether the tuple holds values (`false`: $\Sigma = sp^2 I + si^2 \nabla^{+}(\nabla^{+})^{\top} + ss^2 \mathbf{1}\mathbf{1}^{\top}$, the interval noise accumulating as a random walk centred on the tuple's mean) or their first differences (`true`: $\Sigma = sp^2 \nabla\nabla^{\top} + si^2 I + ss^2 \mathbf{1}\mathbf{1}^{\top}$, the value noise reaching each interval through its shared endpoints), $\nabla$ being the first-differencing map at the size its operand requires and $\nabla^{+}$ its pseudoinverse. All three widths are standard deviations in the attribute's own coordinates (defaults 0), squared internally; the result is an $r \times r$ symmetric positive-definite matrix that drops directly into the `sigma` argument of the tensor functions (§13.3). Raises if `differenced` is omitted, $r < 2$ (at $r = 1$ the covariance reduces to a scalar variance, indistinguishable from a scalar `sigma`; pass the equivalent standard deviation instead), any width is negative or non-finite, or the result is not positive-definite (which needs `sdValue` $> 0$, or `sdInterval` and `sdShift` both non-zero, on undifferenced values; `sdValue` $> 0$ or `sdInterval` $> 0$ on differenced ones).

**simplexVertices(N [, edgeLength])**

Returns the $N$ vertices of a regular $(N-1)$-simplex centred at the origin in $\mathbb{R}^{N-1}$, as an $N \times (N-1)$ matrix whose row $k$ is the coordinate vector for level $k$. All pairwise vertex distances equal `edgeLength` (default 1). Supports the simplex-coded encoding of an $N$-level categorical attribute (voice identity, instrument, etc.) for MAET input: each level becomes one row of the returned matrix, fed as values for the $N - 1$ numerical sub-attributes of a single categorical group (with `isPer = false`, `isRel = false`). Because all vertices are pairwise equidistant, no level is privileged over any other --- in contrast to the dummy or treatment codings familiar from regression. The categorical group's $\sigma$ then controls how sharply the levels are distinguished: at $\sigma = 0$ they are perfectly separated, and as $\sigma \to \infty$ they collapse to a level-blind representation. Construction uses the centred standard basis of $\mathbb{R}^N$ projected onto an orthonormal basis of $\mathbf{1}^\perp$; the result is rotation-equivalent across choices of basis, which is irrelevant for downstream MAET computations. Concrete shapes: $N = 2$ collapses to $\pm \tfrac{1}{2}$ on a line; $N = 3$ to an equilateral triangle in $\mathbb{R}^2$; $N = 4$ to a regular tetrahedron in $\mathbb{R}^3$. The complementary encoding --- one attribute per categorical level --- needs no helper, since it is constructed by simply assigning each level's events to a separate attribute.

**showPreMaet(pm | pAttr [, wAttr] [, specs] [, 'sigma', s] [, 'isRel', rel] [, 'isPer', per] [, 'period', P] [, 'names', nm] [, 'format', f] [, 'maxEvents', m] [, 'maxElements', k] [, 'decimals', d] [, 'weights', wf] [, 'title', t] [, 'caption', c] [, 'label', l] [, 'verbose', v])**

Prints a pre-MAET as a table, in the layout of the pre-MAET tables of Milne (2026), and returns the rendered string (§6.2). Takes either a density built by `buildMaet`, from which every field is recovered, or the pre-MAET itself with the kernel parameters supplied alongside; `isRel` overrides the `specs` `rel` field. One row per attribute, headed by its name and the parameters that determine its density, and one column per event. Cells take braces where the attribute is unordered and parentheses where it is ordered, bracketed level by level when nested; NaN-padded slots are absent rather than shown; non-uniform weights appear as parenthesized superscripts (`weights` forces or suppresses them, default `'auto'`); an attribute carrying a kernel covariance is named by its shape. `maxEvents` and `maxElements` elide a long passage's middle columns and a large multiset's tail (`[]` / `None` shows everything); `decimals` sets the precision, a value rounding to all zeros being shown in scientific notation unless it is floating-point residue. `format` is `'markdown'` (default; plain ASCII, so the column widths are identical in the two languages) or `'latex'` (a `booktabs` tabular in the article's markup, with optional `caption` and `label`). `verbose` (default true) prints; the string is returned regardless in Python, and when an output is requested in MATLAB.

**readPreMaet(source [, 'delimiter', d])**

Reads a pre-MAET from a CSV file, a path or the CSV text itself (§6.4). Returns the pre-MAET (§6.1): its `wAttr` is empty where no cell carried a weight, and each spec holds `r`, `rel`, `exch`, `name` and, where the file gives them, `sigma`, `isPer` and `period`. A nested attribute also carries its `tags`, reconstructed from the bracket structure of its cells, so a file never writes them down. Events of different sizes are NaN-padded to the widest. A kernel covariance is read from `cov(differenced=..., sd_value=..., sd_interval=..., sd_shift=...)` — either spelling of the parameter names — with the row's own `r` giving the matrix's size. The inverse of `writePreMaet`, byte for byte.

**writePreMaet(destination, pm | pAttr [, wAttr] [, specs] [, showPreMaet arguments])**

Writes a pre-MAET as the CSV `readPreMaet` reads (§6.4). `destination` is a path, or empty to return the text without writing it; the text is returned either way. Takes the same inputs and arguments as `showPreMaet` — a built density, or the pre-MAET itself with its kernel parameters — and renders through it under `'format', 'csv'`, so one renderer owns the cell grammar. Weights are decided per attribute and omitted where they are all exactly 1; nothing is elided, whatever `maxEvents` says. A kernel covariance outside the `kernelCov` family cannot be written and is refused with that reason.

### 12.3 Preprocessing

| MATLAB | Python | Description |
|:---|:---|:---|
| `bindEvents` | `bind_events` | gather L consecutive events into one nested super-event attribute |
| `differenceEvents` | `difference_events` | replace event sequences with inter-event differences |
| `transformAttributes` | `transform_attributes` | Scale conversions and elementwise transforms (see §12) |
| `translateAttributes` | `translate_attributes` | shift selected attributes' values by an offset (one translation per call; sweeps go through `sweptSimilarity` or `sweepSimMaet`) |
| `weightEvents` | `weight_events` | multiply a per-event profile factor read from one attribute into another attribute's weights |
| `addSpectra` | `add_spectra` | Add spectral partials to a weighted pitch multiset, or to an attribute of a pre-MAET |
| `selectPreMaet` | `select_pre_maet` | keep a selection of the attributes and the events |
| `bindAttributes` | `bind_attributes` | Gather several attributes into one read as a tuple |
| `separateAttributes` | `separate_attributes` | Split one attribute into one attribute per slot |

The preprocessing functions share a common shape: they take the pre-MAET — the same triple `buildMaet` consumes — apply a transform, and return a pre-MAET of the same form. `specs` is keyword-only and optional; when omitted, a flat (one-level, `r = 1`, `[rel] = false`, `[exch] = true`) geometry is assumed. Because the output is again a pre-MAET, the operators chain in any order and the result feeds directly into `buildMaet`.

**differenceEvents(pm | pAttr, wAttr, diffOrders [, 'circular', false] [, 'specs', specs])**

Cross-event preprocessing for MAET input. Takes a pre-MAET — whole (§6.1) or in its parts — and replaces the event sequences of selected attributes with their $k$-th finite differences across events, returning the transformed pre-MAET. `diffOrders` is per-attribute: `diffOrders[a] = k` produces the $k$-th difference of attribute $a$; order 0 leaves an attribute unchanged (a scalar broadcasts to every attribute). Raw signed differences are emitted regardless of periodicity; periodic attributes' mod-period wrap is handled by the kernel at MAET-construction time. Weights follow the toolbox's standard broadcast convention ([§10.4](#104-weight-arguments)) and propagate as rolling products of width $k + 1$, so the weight of a differenced event is the product of the weights of the $k + 1$ constituent events it depends on (interpretable as the probability that all constituents are jointly perceived). When `'circular'` is `false` (default), the output event count is $N' = N - \max_a k_a$; when different attributes use different orders, lower-order attributes have leading events dropped to keep columns aligned. When `'circular'` is `true`, the difference operator wraps at the sequence boundary ($\Delta p_a(n) = p_a(n) - p_a(\mathrm{prev}(n))$ with $\mathrm{prev}(1) = N$) and every attribute retains $N$ events at every order; no leading-event drop is needed, and weights propagate via a cyclic rolling product. The circular variant parallels the same flag on `bindEvents` and is the natural choice for cyclic event sequences (looped rhythms, ostinati) in which the boundary difference is a genuine inter-event interval. The returned pre-MAET feeds directly into `buildMaet`. Inputs are restricted to $K_a = 1$ per attribute; see [§13.1](#131-the-density-and-its-four-modes) (Preprocessing) for the rationale and the voices-as-attributes pipeline for polyphonic analyses. Event differencing is distinct from `isRel = true`: the former replaces absolute per-event values with differences between adjacent events (a preprocessing step); the latter is a property of tensor construction that makes the within-r-ad density translation-invariant (no preprocessing involved). Both can be used together or independently. Composes with `bindEvents` (below) to produce n-grams of consecutive inter-event differences.

**bindEvents(pm | pAttr, wAttr, bindOrders [, 'circular', false] [, 'specs', specs] [, 'rOuter', …] [, 'exchOuter', false] [, 'relOuter', false] [, 'groupBy', …] [, 'groupAtol', 0])**

Cross-event preprocessing for MAET input. For each attribute, a sliding window of width $L_a$ (`bindOrders`, per-attribute; a scalar broadcasts) is laid across the events and the $L_a$ consecutive events are nested into a single output attribute: the bound events form an **ordered outer level** (event order — `exchOuter = 0` by default, so the binding is lossless and order-preserving), and the original per-event values sit at the inner level. The outer level defaults to reading the whole window (`rOuter = L_a`), ordered (`exchOuter = 0`), and absolute (`relOuter = 0`). An attribute with `L_a = 1` is not nested: each super-event carries its element multiset from one event, the last of the span, so `bindEvents(pm, [4 1])` gives four-note pitch patterns each with the onset of its last note, while `[4 4]` (or the scalar `4`, which applies to every attribute) gives the four onsets as an ordered tuple. Setting `relOuter = 1` makes the outer level relative, giving a within-window transposition-invariant n-gram. The result is a transformed pre-MAET whose `specs` now encode the two-level (inner/outer) geometry, suitable straight as input to `buildMaet`. Per-event weights propagate per row: each bound super-event's effective weight is the product of its $L_a$ constituent events' weights (the kernel product over the nested tuple recovers it). Inputs may have any $K_a \ge 1$; each element of the underlying events is carried through the nesting, preserving within-attribute element exchangeability. The output event count is $N' = N - \max_a L_a + 1$ (non-circular) or $N$ (`'circular'` true, wrapping at the sequence boundary as on `differenceEvents`); each super-event spans $\max_a L_a$ consecutive events and is end-aligned to the last of them, so an attribute with $L_a < \max_a L_a$ contributes the last $L_a$ events of the span (for $L_a = 1$, the element multiset of the span's last event) — the alignment `differenceEvents` uses. `'specs'` lets you bind an attribute that already carries level structure; binding an already-nested ($L \ge 3$ deep) attribute is not yet supported. See [§13.1](#131-the-density-and-its-four-modes) (Preprocessing) and the toolbox specification for the nesting model.

`'groupBy'` binds by a run of equal values instead of by a sliding window. It names one attribute, by index or by name, whose consecutive equal values gather into one super-event: a new group begins wherever that value changes, so the groups are read from the data and may differ in length. This is the form a supplied structure takes when it does not come in uniform blocks — the positions of one chord's derivation path, the notes of one bar, the frames of one gesture. The outer level is then ragged: each super-event is padded to the largest group's size with NaN values at zero weight, which the nested inner product already consumes, and `'rOuter'` says how many positions a tuple takes, defaulting to the smallest group's size — the largest tuple size at which every group is feasible, so every group contributes tuples of one size into one density. The grouping attribute must carry one value per event, constancy across several being ambiguous; `'groupAtol'` sets the tolerance within which two values count as equal (0 by default, exact equality); and `'groupBy'` and `bindOrders` are mutually exclusive, the group sizes coming from the data rather than from an argument. `circular` and `step` have no meaning here and are refused.

Composes with `differenceEvents`: the standard pipeline `differenceEvents` $\to$ `bindEvents` $\to$ `buildMaet` $\to$ `entropyMaet` produces n-tuple entropy on the integer-step grid, recovering Milne & Dean (2016) at $\sigma \to 0$ and uniform weights, and extending it to the smoothed continuous case, weighted events, and non-periodic domains. The convenience function `nTupleEntropy` (full entry in [§12.7](#127-scale-and-rhythm-structure)) wraps this pipeline. Used without a preceding differencing step, `bindEvents` produces n-grams in absolute pitch or time register; pairing the bound block with an explicit time or event-index attribute (carried alongside, with the time stamp travelling with each window) localizes the n-grams in time and supports time-windowed similarity for n-gram pattern matching.

**translateAttributes(pm | pAttr, wAttr, offsets [, 'specs', specs])**

Per-attribute preprocessing for MAET input. Shifts selected attributes' values by a chosen offset, returning the transformed pre-MAET, which feeds directly into `buildMaet`. Weights and `specs` pass through unchanged — only the values move. The transform is per-row: an offset is one position per row of the $K_\text{total} \times N$ position matrix, held constant across events (the columns), which is exactly what makes a uniform shift cancel inside relative attributes.

`offsets` is a **length-$A$ list** (Python) / $1 \times A$ cell (MATLAB), one entry per attribute, each of: `None` / empty `[]` (do not translate this attribute); a scalar or length-1 vector (broadcast to all rows — a global transposition); or a length-$K_\text{total}$ vector (per-row). `NaN` entries skip the corresponding row (left untranslated); $\pm\infty$ is rejected.

Relative geometry is read per-attribute from `specs` (there is no separate `isRel` argument): a *uniform* finite offset on an attribute whose outermost level is relative is a structural no-op (it cancels in every within-tuple difference), so that attribute is left unchanged and a single `translateAttributes:noOp` (MATLAB) / `TranslateAttributesNoOpWarning` (Python) is emitted per call; a *non-uniform* (per-row) offset is **not** a no-op even on a relative attribute and is applied. `isPer` / `period` are not consulted here — translation emits unwrapped values and the periodic kernel in `buildMaet` wraps downstream.

One call makes one translation. **A translation sweep** — a query compared with a context at each of many offsets — **is computed by `sweptSimilarity` (pre-MAETs, §8.7) or `sweepSimMaet` (densities)**, both in one pass over the tuple pairs rather than one comparison per offset; building translated copies and comparing them one by one gives the same values at many times the cost. See [§7.7](#77-compositions-and-canonical-uses) for the conceptual framing of translation versus windowed sliding.

**transformAttributes(pm | pAttr, wAttr, transforms [, 'specs', specs] [, 'sign', false])**

Per-attribute preprocessing for MAET input. Maps every value of selected attributes through a transform, returning the transformed pre-MAET, which feeds directly into `buildMaet`; weights pass through. `transforms` is a length-$A$ list (Python) / $1 \times A$ cell (MATLAB), one entry per attribute, or a single entry broadcast to all. Each entry is `None` / `[]` (leave unchanged); a name, optionally with parameters — Python `('log', {'base': 2})` or `{'name': 'log', 'base': 2}`, MATLAB `{'log', 'base', 2}` or `struct('name', 'log', 'base', 2)` — from `'log'` (`base`, default $e$; `offset`, default 0: $\log(x + \text{offset})$), `'power'` (`exponent`), `'affine'` (`scale`, `offset`); a pitch-scale pair — Python `('hz', 'cents')`, MATLAB `{'hz', 'cents'}` (with $A = 2$ wrap the pair in its own cell) — from `'hz'`, `'midi'`, `'cents'` = 100 × MIDI, `'octave'` = MIDI / 12, `'mel'`, `'bark'`, `'erb'`, `'greenwood'`; or a callable / function handle applied to the $K_\text{total} \times N$ value matrix, which must return the same shape and finite values. A bare numeric array in place of `pAttr` is treated as one attribute and the transformed array is returned alone: `p = transformAttributes(f, [], {'hz', 'cents'})` / `p = mpt.transform_attributes(f, None, ('hz', 'cents'))`.

Values outside a transform's domain raise with the attribute, the offending events, and the remedies (a zero under `'log'` is never mapped to $-\infty$). `sign` (bool or per-attribute) applies a magnitude transform to $|x|$ and inserts a sign attribute in $\{-\tfrac12, 0, +\tfrac12\}$ — the 2-point simplex's vertices at unit edge length — immediately after the source, extending `w` (when a per-attribute list) and `specs` (the source's spec with `rel` cleared and name suffixed `'_sign'`); the attribute count grows accordingly. See [§7.3](#73-attribute-rescaling) for the choice of scale, its interaction with differencing, and periodicity.

**weightEvents(pm | pAttr, wAttr, inputAttr, targetAttr, centre, shape, {'sd', s | 'width', L}, 'dropInputAttr', tf [, 'specs', specs] [, 'isPer', false] [, 'period', 0] [, 'locate', 'centroid'] [, 'edges', 'halfOpen'])**

Per-event preprocessing for MAET input. Reads one value per event from `inputAttr` (where an event holds several, the one `locate` picks), evaluates a window factor centred at `centre` with shape parameter $\gamma$ = `shape`, and multiplies the resulting $(1, N)$ per-event factor into the weights of `targetAttr` (on top of any weight already there), returning the transformed pre-MAET, which feeds directly into `buildMaet`.

The window scale is given by **exactly one** of two keyword-only arguments, naming the same underlying scale on different terms: `sd` is the window's standard deviation, while `width` is the *full support* of the rectangle at `shape = 1` (the conversion is `sd = width / (2 sqrt(3))`). Gaussian users typically think in standard deviations (`sd`); rectangle users typically think in full supports (`width`). Across the whole `shape` family the SD is held constant regardless of which parameter was supplied, so the only effect of the choice is the number the user types. Supplying neither or both raises an error. In terms of the standard deviation $s$, the factor is the peak-normalized convolution of a rectangle of half-width $s\sqrt{3\gamma}$ and a Gaussian of standard deviation $s\sqrt{1 - \gamma}$, so the total variance is $s^2$ for every $\gamma \in [0, 1]$: $\gamma = 0$ is a pure Gaussian, $\gamma = 1$ a pure rectangle, and intermediate values interpolate. **The pure rectangle uses a half-open support $[c - W/2,\, c + W/2)$** (lower edge included, upper excluded, with a floating-point-robust edge test), so a regular pulse grid yields exactly $N$ pulses for full support $N \cdot \mathrm{IOI}$ at every $N$ and a window aligned between pulses captures its intended span; `'edges'`, `'closed'` includes both edges instead.

`inputAttr` and `targetAttr` are scalar attribute indices (1-based MATLAB / 0-based Python); they may be equal or distinct. `inputAttr` must reference an attribute with $K_{\text{input}} = 1$ (single value per event); the factor broadcasts across the target attribute's $K_{\text{target}}$ rows downstream. `isPer` is a scalar boolean that must mirror the `[per]` flag of `inputAttr` in the downstream `buildMaet` call, with `period` the corresponding scalar (ignored when `isPer = false`); when `isPer = true` the difference $\delta = v - c$ is wrapped to $[-\text{period}/2, \text{period}/2]$ before the shape function is evaluated, and stored values are not wrapped.

`dropInputAttr` (Python `drop_input_attr`) is a mandatory keyword-only boolean (no default). When `true` and `inputAttr != targetAttr`, the input attribute is dropped from the returned pre-MAET (its `specs` entry removed and higher indices renumbered). When `false`, the input attribute is preserved. `dropInputAttr = true` paired with `inputAttr == targetAttr` raises an error (the just-written factor would be discarded).

Windowing on several attributes is expressed as a sequence of `weightEvents` calls with the same `targetAttr`; each call's factor multiplies into the target's existing weights. The canonical composition `weightEvents` (with `dropInputAttr = true`, `inputAttr` typically time, `targetAttr` typically pitch) $\to$ `buildMaet` $\to$ `entropyMaet` is the windowed-entropy construction of [§7.7](#77-compositions-and-canonical-uses). `weightEvents` chains cleanly with `translateAttributes` under a centre-shift identity ($\mathcal{T}_\mu \circ \mathcal{W}_c = \mathcal{W}_{c+\mu} \circ \mathcal{T}_\mu$ on overlapping targets) and passes through `differenceEvents` / `bindEvents`.

**addSpectra(p, w, mode, ...)**

Adds partials to each pitch. Mode is one of `'harmonic'`, `'stretched'`, `'freqlinear'`, `'stiff'`, or `'custom'`. All non-custom modes take N (number of partials including the fundamental), followed by a weight-type specification (`'powerlaw', rho` or `'geometric', tau`). The `'stretched'`, `'freqlinear'`, and `'stiff'` modes each have one additional parameter (β, α, or B respectively) between N and the weight type.

The optional name-value pair `'units', U` specifies pitch units per octave (default: 1200, i.e., cents). When using semitones, set `'units', 12`.

Output weights are the product of each pitch's original weight and the spectral weight of each partial.

Two forms. Given a single weighted multiset, as above, the function returns the expanded `(p, w)` pair, and that is the primitive the harmony, entropy, and consonance functions call. Given a pre-MAET, `addSpectra(pm, mode, ..., 'attribute', a)` expands attribute `a` of every event at once and returns a pre-MAET, as the other pre-MAET preprocessors do; `'attribute'` takes a position or a name and is required there, a pre-MAET carrying several attributes and a spectrum belonging to one. The expansion multiplies the attribute's K by the number of partials and leaves N and the spec alone; a padded slot expands to padded partials at weight zero, a missing value having no spectrum, and since partials of one value differ in weight the result always carries weights, even where the input carried none. An attribute read in order at `r > 1` is refused, expansion scrambling positions that carry meaning: add the partials before the attributes are bound, or read that attribute as a multiset. The bare `pAttr` / `wAttr` form the other preprocessors offer is not available here, its positional layout being indistinguishable from the single-multiset form.

**selectPreMaet(pm, ...)** — Keeps a selection of a pre-MAET's attributes and events, and knows nothing of where the pre-MAET came from: it reads the two levels every pre-MAET has and nothing else. Name-value pairs: `'attributes'` (indices, names as the specs carry them, or a logical mask) and `'events'` (indices or a logical mask); `[]` keeps all, and the kept items come back in the order given. A predicate is applied by the caller, which reads the values it wants from `pAttr` and passes the mask. Attributes keep their tuple sizes and flags, so a selection cannot change what an attribute means, and an attribute holding several coordinates of one value — a simplex-coded level, say — moves whole, because it is one attribute and not several. Selecting events may leave an attribute with no value at some kept event; that is allowed and means what it says, the event contributing nothing on that attribute while keeping its place in the sequence. Selecting *rows of a table* is the host language's job, not this function's.

**bindAttributes(pm, attributes, ...)** — Gathers several attributes into one whose value at an event is the tuple of all of them: the counterpart, across attributes, of `bindEvents`, which binds across events. Where three columns carry the three coordinates of one position, or the coordinates of a simplex-coded level, they are three attributes whose product pairs each with every other; binding makes them one attribute read together. `attributes` are the attributes to bind, as indices or names, in the order their values are to be read — the order `'exch', false` makes significant. The bound attribute takes the place of the first of its inputs, and the attributes not listed keep their order around it. Name-value pairs: `'name'` (required, since no input's name describes the result), `'r'` and `'exch'` (required, neither following from the inputs and neither having an identity; `r` = the sum of the inputs' K reads the whole tuple as one object, which is what a coordinate vector wants), `'sigma'`, `'rel'`, `'isPer'`, `'period'` (each inherited where every input agrees on it and required where they differ, there being no reading of a periodic value bound to a non-periodic one that the call has not chosen), and `'specs'`.

**separateAttributes(pm, attribute, ...)** — Splits one attribute into one attribute per slot: the inverse of `bindAttributes`, and the operation by which the conversion's two structural roles differ, since under `'orderedMultiset'` slot *k* is level *k* and splitting that attribute slot by slot gives what the `'separateAttributes'` role builds from the table directly. Each part holds one row of the input and carries its kernel parameters; `r` is 1 and `exch` says nothing, both being determined by there being one value per event. Name-value pairs: `'names'` (one per slot; the default suffixes the source's name with the 1-based slot position) and `'specs'`.

### 12.4 Densities and measures

| MATLAB | Python | Description |
|:---|:---|:---|
| `buildMaet` | `build_maet` | precompute an r-ad expectation tensor density object |
| `evalMaet` | `eval_maet` | evaluate the density at query points |
| `plotMaet` | `plot_maet` | a density of one, two, or three drawn dimensions, as its kernels (`kernels`), as its sampled values (`points`, three dimensions only), or as the density itself (`density`; at three dimensions a volume rendering, MATLAB only) |
| `simMaet` | `sim_maet` | cosine similarity of two densities (single, list, or batched-raw) |
| `sweepSimMaet` | `sweep_sim_maet` | similarity against uniformly translated copies of a density, as one reduced sweep |
| `entropyMaet` | `entropy_maet` | Shannon, normalized, differential, or Rényi-2 entropy of an expectation tensor |
| `massMaet` | `mass_maet` | the mass of a density in a region (a box or a Gaussian), or its share of the whole |
| `maetCentres` | `maet_centres` | the points at which a density places its Gaussians, materialised on demand |
| `sweptSimilarity` | `swept_similarity` | sliding-window similarity profile by event weighting (magnitude-aware or cosine) |
| `sweptEntropy` | `swept_entropy` | sliding-window entropy profile by event weighting |
| `sweptMass` | `swept_mass` | sliding-window mass (or share) in a region, by event weighting |

**buildMaet(p, w, sigma, r, isRel, isPer, period)**

Precomputes tuple indices, pitch matrices, and weight vectors for a weighted pitch multiset. Returns a struct (`tag = 'MaetDensity'`) that can be passed to `evalMaet` and `simMaet`.

Key parameters:
- `p` — Pitch values (vector), or a cell array of per-attribute matrices for a multi-attribute tensor
- `w` — Weights (vector, or empty for all ones)
- `sigma` — Standard deviation of the Gaussian kernel (cents), or a per-attribute vector for MAETs
- `r` — Tuple size (positive integer; r ≥ 2 if isRel = true), or a per-attribute vector for MAETs
- `isRel` — Transposition-invariant if true (effective dim $= r - 1$)
- `isPer` — Periodic wrapping if true
- `period` — Period for wrapping (e.g., 1200 for one octave in cents)

A whole pre-MAET (§6.1) stands in place of `p` and `w`, bringing its `specs` with it: `buildMaet(pm)`, `buildMaet(pm, 'sigma', sigmaVec)`. In that form all six per-attribute parameters — `sigma`, `isPer`, `period`, `r`, `rel`, and `exch` — may be given as name-value arguments, and a supplied value takes precedence over the specs for every attribute, so a sweep over any of them stays one call per value and leaves the pre-MAET untouched. On a nested attribute `r`, `rel`, and `exch` are per-level vectors, where a scalar override has no unambiguous reading, so there they are refused and the spec is the place to change them. An unrecognized name-value argument is an error rather than being silently taken as positional. `evalMaet`, `simMaet`, and `entropyMaet` take one wherever they take a density, building it on the spot. In these four the weights argument stays `w` rather than `wAttr`, since each also accepts a bare vector — the A = N = 1 single-multiset corner — where a per-attribute container would be the wrong shape; `wAttr` names it in the multi-attribute call forms, where it is one.

Multi-attribute call form: `buildMaet({p_1, …, p_A}, w, sigmaVec, rVec, isRelVec, isPerVec, periodVec)`. Each `p_a` is a `K_a × N` matrix of attribute values; `sigmaVec`, `rVec`, `isRelVec`, `isPerVec`, `periodVec` are per-attribute (length-$A$) geometry vectors. Every attribute is self-contained and carries its own geometry, so there is no separate `groups` argument — shared geometry is expressed by repeating a value across the attributes that should share it. Level-structured geometry (nested binding, ordered `[exch] = 0` attributes, per-attribute `[rel]`) is instead supplied through the keyword-only `specs`: `buildMaet({p_1, …, p_A}, w, 'specs', specs, 'sigma', sigmaVec, 'isPer', isPerVec, 'period', periodVec)` (MATLAB) / `build_maet([p_1, …, p_A], w, specs=specs, sigma=sigma_vec, is_per=is_per_vec, period=period_vec)` (Python), where the per-attribute `r`, `[rel]`, and `[exch]` live inside `specs` (see `flatSpecs`, §12.2) and only the scalar/​per-attribute `sigma`, `isPer`, and `period` are passed alongside. Either form returns a struct (`tag = 'MaetDensity'`) compatible with all downstream tensor functions.

**evalMaet(dens, X [, normalize])**

Evaluates the density at the query points given by the columns of X. Three normalization modes:
- `'none'` (default) — Raw weighted sum of Gaussian kernels. The absolute value depends on σ, the number of tuples, and the weight magnitudes. Only the relative values across query points are meaningful. Sufficient for visualization and cosine similarity, where any normalization cancels.
- `'gaussian'` — Each Gaussian component is normalized to integrate to 1. After this normalization, the total density integrates to the sum of all tuple weight products. Useful for comparing densities computed with different σ values: increasing σ spreads the same mass over a wider area rather than inflating the total integral.
- `'pdf'` — Full probability density normalization. Applies the Gaussian normalization above, then divides by the sum of all tuple weight products so the density integrates to 1 over the domain. Useful for comparing densities across pitch multisets of different sizes, or for computing entropy.

Query points X should have `dim` rows, where $\mathrm{dim} = r - \mathrm{isRel}$. For the relative case ($\mathrm{dim} = r - 1$), each column of X specifies the $r - 1$ intervals that define an r-ad (e.g., for a triad with $r = 3$ and isRel = true, each column is a 2-element vector of intervals from the lowest pitch to the middle and highest pitches). For a MAET, X is either a cell array of per-attribute query matrices or a single matrix with per-attribute row blocks concatenated.

**simMaet(dens_x, dens_y)** or **simMaet(p1, w1, p2, w2, sigma, r, isRel, isPer, period)**

Computes the cosine similarity between two expectation tensor densities analytically. The precomputed-struct calling convention avoids recomputing tuple indices on each call. Both conventions support `'verbose', false`. Accepts a `MaetDensity` of any shape, from a single multiset to the general multi-attribute case. Sliding-window comparisons are the province of `sweptSimilarity` (§8.7), which windows the raw events and calls `simMaet` at each position. For multi-attribute comparisons, the two densities must share the same attribute structure and per-attribute parameters; see the compatibility note at the end of §13.2. In the raw scalar form, `'spectrum'` is accepted as a name-value pair (a cell / list of `addSpectra` arguments) and applies the same partials to both `(p1, w1)` and `(p2, w2)` internally before density construction; the struct form rejects it because spectral enrichment must be applied before the density is constructed.

The `normalize` keyword (also accepted as `normalise`) selects the denominator. The default, `'cosine'`, gives the strict shape-only cosine similarity $\langle X, Y\rangle / \sqrt{\langle X, X\rangle \langle Y, Y\rangle}$, bounded in $[-1, 1]$ and invariant to a positive scalar on either operand. The alternative, `'oneSidedDenom'`, divides only by the second operand's self inner product, $\langle X, Y\rangle / \langle Y, Y\rangle$; the result is magnitude-aware, taking the value 1 on a self-match ($X = Y$) but scaling linearly with positive rescalings of the first operand. Either spelling of the keyword is accepted; matching on the value is case-insensitive.

*Batched-raw mode.* When called with two 2-D pitch matrices `P1` and `P2` of size `M`-by-`K` (and likewise-shaped or empty weights `W1`, `W2`), `simMaet(P1, W1, P2, W2, sigma, r, isRel, isPer, period)` returns an `M`-by-1 vector of similarities computed pair-by-pair across rows. Each row is a multiset; rows may use NaN-padding for variable cardinality. Symmetry-equivalent rows (permutations, transpositions in `isRel = true` modes, period-equivalents in `isPer = true` modes) are automatically deduplicated for speed — see §10.7 for the equivalence classes that apply. Three additional name-value pairs are accepted in this mode: `'spectrum'` (applies the same partials to both sides per row), `'precision'` (decimal places of pitch / weight rounding for dedup tolerance), and `'dedup'` (toggle internal deduplication; on by default and currently a no-op when off, emitting a warning).

*Broadcasting in batched-raw mode.* When one of `P1`, `P2` is `M`-by-`K` (with `M > 1`) and the other is a vector of length `K` (1-D, 1-by-`K`, or `K`-by-1 in MATLAB; 1-D or `(1, K)` 2-D in Python), the vector is broadcast across the matrix's `M` rows in NumPy / MATLAB implicit-expansion style. The corresponding weights argument is broadcast in lockstep when non-empty. Eliminates the explicit `repmat(refPitches, M, 1)` / `np.tile(ref_pitches, (M, 1))` idiom for the common "compare one reference multiset against many candidates" use case.

*Shape rule, and a deliberate difference between the languages.* MATLAB enters batched-raw mode on an operand that is a matrix with both dimensions greater than one; a vector is a vector whichever way it is oriented, so a `K`-by-1 column is broadcast as one multiset shared by every row rather than read as `K` rows of one element. Python enters it on `ndim == 2`, so a `(K, 1)` array is `K` single-element rows. Each rule is the idiomatic one in its own language, and the divergence is intentional (ARCHITECTURE §7). A batch of one-element multisets – a probe-tone sweep, say – is written the same way in both: pad to two columns with NaN, `[p(:), nan(K, 1)]` / `np.column_stack([p, np.full(K, np.nan)])`, the padding being stripped per row before each density is built.

*List mode.* When called with two cell arrays (MATLAB) or lists (Python) of density structs, `simMaet` returns a 1-by-`n` cell / 1-D `ndarray` of pairwise similarities. The Option II shape rule is preserved: a length-1 list returns a length-1 cell / array, never a scalar. Density structs of mixed shapes (single-multiset, multi-attribute) are dispatched independently, with downstream compatibility checks per pair.

*Pre-MAETs in list mode.* Wherever a density struct is accepted, a pre-MAET is accepted in its place, including inside a cell / list: `simMaet(pmRef, {pm1, pm2, pm3})` builds each entry and compares it. The entries are looped, not deduplicated — each is a whole item with its own geometry — so the saving is in the calling code (§6.1). `evalMaet` and `entropyMaet` take pre-MAET cells / lists on the same terms.

*List-mode broadcasting.* Either operand may be a single density struct paired with a cell / list of density structs; the single struct is broadcast against every entry of the cell / list. Returns a 1-by-`n` cell / array, mirroring the broadcast available in batched-raw mode. In Python only, list × list calls additionally accept `mode='cartesian'` to compute the full `m`-by-`n` cross-product (returns a 2-D `ndarray`); MATLAB list mode is currently pairwise-only.

*Translation sweeps.* A list of raw `pAttr` blocks is not a form of `simMaet`. To compare one density with translated copies of another, use `sweepSimMaet` (densities and an offset for each swept attribute at each step) or `sweptSimilarity` (pre-MAETs and sweep values), which compute every offset in one pass (§7.7); `simMaet` over a list of densities or pre-MAETs compares its entries one by one.

**sweepSimMaet(densX, densY, offsets [, 'method', m] [, 'normalize', n] [, 'truncationSigmas', t])**

Similarity of `densX` against `densY` with every value of attribute $a$ shifted by `offsets(a, m)`, for each column $m$ of the $A \times M$ offset matrix (a row vector is accepted when $A = 1$). Returns a $1 \times M$ vector indexed as those columns. The whole sweep costs one pass over the tuple pairs plus $M$ evaluations of a mixture in the offset, rather than $M$ separate inner products, by the placement/shape separation of [§8.7](#translation-sweeps-in-one-pass). `'method'` selects the decomposition that carries it — `'mixture'` (the placement/shape split), `'orbit'` (the Möbius/orbit inner product evaluated at the shifted values, whose cost scales with the orbit count rather than the tuple-pair count, and which also covers a swept periodic attribute), `'contract'` (densities with a nested attribute: the level-by-level contraction with the offsets as a batch dimension), or `'auto'` (the default: `'contract'` where a nested attribute is present, otherwise compares the costs of the other two and picks). `'normalize'` and `'truncationSigmas'` are as in `simMaet`; both self inner products are invariant under a uniform translation, so each is computed once for the whole sweep and returned memoised in the optional second and third outputs. Raises when the sweep admits no reduction — a swept relative attribute, a swept nested inner or intermediate one, a swept periodic attribute under the mixture route, or an anisotropic kernel covariance on a swept attribute — in which case translate the query with `translateAttributes` and compare offset by offset.

**entropyMaet(p, w, sigma, r, isRel, isPer, period, ...)**

Entropy of the expectation tensor. Also accepts a precomputed struct as the first argument (`MaetDensity`). The estimator is selected by the `'method'` name-value pair (see [§11.2](#112-the-four-entropy-estimators), "Four-method entropy API"): `'shannon'` (default) is the raw discrete Shannon entropy on a fine grid; `'normalized'` is the same divided by $\log_b N$ into $[0, 1]$ (the ratio of the published measures); `'differential'` is the adaptive continuous differential entropy; and `'renyi2'` is the closed-form Rényi-2 (collision) entropy $H_2 = -\log_b\!\big(\langle T, T\rangle / Z^2\big)$, which needs no grid. The earlier `'normalize'` keyword has been removed — use `method='normalized'` for the $[0, 1]$ ratio. A zero-mass density (every weight zero, or a window aligned at a sweep value with no event in its support) returns `NaN` under every method, its entropy being undefined. Other optional name-value pairs: `'spectrum'`, `'base'` (default: 2), `'nPointsPerDim'` (grid resolution for the grid-based methods), `'xMin'`, `'xMax'` (required when `isPer = false` for grid methods), `'gridLimit'` (default: 10⁸ total grid points), and the kernel-evaluation controls `'truncationSigmas'` and `'kernelPrecision'` (see [§11.3](#113-kernel-evaluation-controls)).

**massMaet(dens [, 'region', {a, spec; ...}] [, 'normalize', n])**

The mass of the density in a region. Each tuple's kernel is taken with unit mass, so the whole density's mass is the sum of its tuples' weight products, and the mass in a region $R$ counts each tuple by the share of its kernel inside it, $m(R) = \sum_j w_j P_j(R)$: the integral over $R$ of what `evalMaet` returns under `'gaussian'`. `'normalize', 'total'` divides by the whole density's mass, giving the share of the density in the region. An attribute the region does not name is integrated over entirely, so without a region the result is the total mass. Each `spec` is a box, `[lo hi]` on every coordinate of the attribute or a row `[lo hi]` per coordinate, or a soft Gaussian region `{'gaussian', centre, sd}` of unit height at its centre; the coordinates are those of the attribute's `evalMaet` query (the $r$ values of an absolute tuple, the $r - 1$ above the first of a relative one). The density factors across attributes within each event, so the joint tuple set is never built, and the integrals are closed forms in the error function. A box needs the kernel's coordinates to be independent, as they are on an absolute attribute and on a relative one of two values; on a relative attribute of three or more, whose coordinates are correlated, a box is refused and a Gaussian region can be used (non-periodic). On a periodic attribute a box spans at most one period and its bounds may wrap. Because a kernel has width, a tuple just outside a box contributes the part of its kernel that crosses the edge, and a region can select tuples, such as the intervals of a relative attribute, where event weighting can only weight events. An exchangeable density holds every ordering of a tuple, so a region on a relative attribute at r = 2 that should ignore order takes the region and its negative. Accepts a pre-MAET, or a cell / list of densities or pre-MAETs (one value per entry). Python: `mass_maet(dens, region={a: spec}, normalize=...)`, with a Gaussian region written `('gaussian', centre, sd)`.

**maetCentres(dens)**

The points at which the density places its Gaussians: a cell / list of one `(r_a - isRel_a)`-by-`nJ` matrix per attribute, in that attribute's own coordinates and unwrapped on a periodic attribute. `buildMaet` builds the per-tuple fields lazily, so a density that took the Möbius route carries no centres array; this call materialises them when they are absent and passes them through when they are not. Useful for plotting a density's support and for reading off which tuples a density is built from; it is not a substitute for `evalMaet`, since the centres say where the kernels sit, not what the density is worth there.

**plotMaet(dens [, 'method', m] [, ...])** — Draws a density of one, two, or three drawn dimensions (`dim = r - isRel`): `'kernels'` (the default) one object per tuple centre, `'points'` the density sampled on a grid (three drawn dimensions only), and `'density'` the density itself (at three dimensions a volume rendering, MATLAB only). Name-value options set the range (`'limits'`), the grid (`'step'` or `'nodes'`), the opacity and colour curves (`'alphaPeak'`, `'alphaFloor'`, `'alphaGamma'`, `'colourGamma'`), and the marks of `'points'` (`'markerSize'`, `'markScale'`). §8.6 describes each method and option.

*Swept functions.*

Each windows or translates at each of a list of sweep values and applies a core measure there, exposing a common analysis as one function (§8.7). `nTupleEntropy`, which chains `differenceEvents`, `bindEvents`, and `entropyMaet`, is another such composition; its entry is in §12.7.

**[S, mu] = sweptSimilarity(pmContext, pmQuery [, six-parameter overrides] [, ...])** or **[S, mu] = sweptSimilarity(pContext, wContext, pQuery, wQuery, sigma, r, isRel, isPer, period [, 'sweep', {a, values; ...} | a] [, 'start' / 'stop' / 'step', {a, value; ...} | value] [, 'align', {a, m; ...}] [, 'window', {a, {shape, width[, edges]}; ...}] [, 'drop', a] [, 'queryRef', {a, value; ...}] [, 'locate', l] [, 'targetAttr', t] [, 'normalize', n] [, 'specs', specs] [, 'isExch', exch] [, 'verbose', v])**

Similarity profile over a list of sweep values (a pre-MAET cross-correlation; §8.7). Takes two whole pre-MAETs (§6.1), the context first — the shared geometry is read from the context's specs and any of the six per-attribute parameters may be given alongside to override it, in full or selective form — or the raw `pAttr` and `wAttr` of a context and a query together with the shared per-attribute geometry of `buildMaet`'s raw form. For each swept attribute `a`, `'sweep'` gives its *sweep values* (or a bare attribute index, `'sweep', a`, or `'start'` / `'stop'` / `'step'` generate them: for `'query'` and `'both'`, every offset at which query and context overlap, or one period on a periodic attribute, at no more than half the peaks' standard deviation $\sigma\sqrt{2/r}$, on the values' lattice where they lie on one; for `'window'`, the context's range at half the window's sd). One rule places everything: at each sweep value $s$, a translated query has its reference value `queryRef` at $s$ (it is translated by $\mu = s - \mathrm{queryRef}$), and a window has its reference value ($\delta = 0$ of the window function $h$) at $s$, weighting each context event on `targetAttr` (default: the first attribute not dropped) by $h(p_a(n) - s)$, $p_a(n)$ being the event's value on the swept attribute $a$. `'align'` says which are placed: `'query'` (the default: translation over the whole context, one pass over all sweep values; `queryRef` defaults to 0, so the sweep values are offsets), `'both'` (query and window at each sweep value; `queryRef` defaults to the query's middle, so the window is aligned at the query's middle), `'window'` (a window only; the attribute may be dropped, `'drop'`, relative, or compared as it stands, the query then compared in place), or `'independent'` (`'sweep'`, `{a, {windowValues, queryValues}}`: every combination, the correlogram; the query's list may be a matrix with one row per window value; `queryRef` defaults to the query's middle). `locate` (`'centroid'` (default), `'start'`, `'end'`, `'mid'`, a function handle, or a map `{a, rule; ...}`) says which value stands for an event holding several. `'window'` is `{shape, width}`, `{shape, width, edges}`, or `struct('shape', ., 'width' | 'sd', ., 'edges', .)` — shape $\gamma \in [0, 1]$ or `'gaussian'` / `'rect'`, full rectangle-equivalent width $W$ with standard deviation $W / (2\sqrt{3})$, and edges `'halfOpen'` (the default for a given width: lower edge included, upper not, for tiling) or `'closed'`. It is required for `'window'` and `'independent'` and refused for `'query'`; for `'both'` it may be left out (or its width given as `[]`), and is then the smallest closed rectangle that, so placed, holds the query; a given window there that leaves out some of the query's own events draws a warning. The query is not windowed here; apply `weightEvents` to it before the call to window it. `normalize` is `'oneSidedDenom'` (default, magnitude-aware), `'cosine'` (bounded shape-only cosine), or `'none'`. `specs` carries nested geometry from `bindEvents`; `isExch` is the per-attribute exchangeability vector of the raw positional form (required for ordered attributes carrying a matrix-valued kernel covariance) and is mutually exclusive with `specs`. Returns `S`, one dimension per sweep list in attribute order (`'independent'` contributing two, the window's first; a single list gives a $1 \times n$ row), and optionally `mu`, a $1 \times A$ cell of the translations applied, $\mu = s - \mathrm{queryRef}$, per attribute the query is translated along (Python: `return_offsets=True`, a dict), and `sv`, a $1 \times A$ cell of the sweep values of each swept attribute (Python: `return_sweep_values=True`, a dict; with both flags the order is `S, offsets, sweep_values`).

**sweptEntropy(pm [, six-parameter overrides] [, ...])** or **sweptEntropy(pAttr, wAttr, sigma, r, isRel, isPer, period [, 'sweep', {a, values; ...} | a] [, 'start' / 'stop' / 'step', ...] [, 'window', {a, {shape, width[, edges]}; ...}] [, 'drop', a] [, 'locate', l] [, 'targetAttr', t] [, 'method', m] [, 'base', b] [, 'nPointsPerDim', n] [, 'xMin', lo] [, 'xMax', hi] [, 'gridLimit', g] [, 'specs', specs] [, 'isExch', exch] [, ...])**

Entropy profile over a list of sweep values. Takes the sweep values, window, `locate`, and `drop` arguments of `sweptSimilarity`, but has no query, so the sweep values always align the window: at each sweep value (or combination of them) the windowed density is built, with any dropped attribute first marginalized, and its entropy taken with `entropyMaet` under the given `method` (default `'differential'`) and `base`; the grid of the discrete methods (`'nPointsPerDim'`, required for `'shannon'` and `'normalized'`; `'xMin'` and `'xMax'` for a kept non-periodic attribute; `'gridLimit'`) is passed to `entropyMaet` at every sweep value, so every window's entropy is taken on the same grid. Every swept attribute needs a `'window'` with a width, the scale of the local region. A window that catches no event gives a zero-mass density, which returns `NaN` under every method. `[H, sv] = sweptEntropy(...)` also returns the sweep values, a $1 \times A$ cell (Python: `return_sweep_values=True`, a dict).

**sweptMass(pm [, six-parameter overrides] [, ...])** or **sweptMass(pAttr, wAttr, sigma, r, isRel, isPer, period [, 'sweep', ...] [, 'start' / 'stop' / 'step', ...] [, 'window', ...] [, 'drop', a] [, 'locate', l] [, 'targetAttr', t] [, 'region', {a, spec; ...}] [, 'normalize', n] [, 'specs', specs] [, 'isExch', exch] [, ...])**

Mass profile over a list of sweep values. Takes the arguments of `sweptEntropy`, and at each sweep value takes the mass of the windowed density in `'region'` with `massMaet`, or with `'normalize', 'total'` its share of the windowed density. The region is keyed by the context's attributes and cannot name a dropped one. Without a region, the result is the window's weighted tuple count. `[M, sv] = sweptMass(...)` also returns the sweep values, a $1 \times A$ cell (Python: `return_sweep_values=True`, a dict). See §8.7.

### 12.5 Consonance and harmonicity

| MATLAB | Python | Description |
|:---|:---|:---|
| `spectralEntropy` | `spectral_entropy` | Entropy of the smoothed composite spectrum |
| `templateHarmonicity` | `template_harmonicity` | Harmonicity via template cross-correlation |
| `tensorHarmonicity` | `tensor_harmonicity` | Harmonicity via expectation tensor lookup |
| `roughness` | `roughness` | Sensory roughness (Plomp–Levelt / Sethares) |
| `virtualPitches` | `virtual_pitches` | Virtual pitch salience profile |

These functions take pitches in cents as absolute pitches (not pitch classes). The functions transpose internally so the lowest pitch is 0.

**spectralEntropy(p, w, sigma, ...)** — Entropy of the composite spectrum (lower entropy = greater consonance). Pitches must be in cents; when using empirical spectral peaks from `audioPeaks` (which returns Hz), convert via `transformAttributes(f, [], {'hz', 'cents'})` first. Can apply spectral enrichment via `'spectrum'`, but this is unnecessary when using empirical peaks since they already represent the full spectrum. The discrete methods (`'shannon'`, `'normalized'`) are computed on the grid `0 : resolution : max + 4σ` with `resolution` in cents (default 1, the grid of Milne et al. 2017 and Smit et al. 2019); a discrete entropy depends on its grid, so change `resolution` only to change the measure.

**templateHarmonicity(p, w, sigma, ...)** — Cross-correlates the chord's spectrum with a harmonic template. Returns hMax (maximum cosine similarity; Milne, 2013) and hEntropy (entropy of the cross-correlation; Harrison, 2020). Separate `'spectrum'` (for the template) and `'chordSpectrum'` (for the chord) parameters. See the comparison with `tensorHarmonicity` below.

**tensorHarmonicity(p, w, sigma, ...)** — Evaluates the relative r-ad expectation tensor of a harmonic series at the chord's intervals (measured from the lowest pitch to each of the remaining pitches). See the comparison with `templateHarmonicity` below.

**roughness(f, w, ...)** — Frequencies must be in Hz (use `transformAttributes` if needed). Optional name-value pairs: `'pNorm'` (default: 1), `'average'` (default: false).

**virtualPitches(p, w, sigma, ...)** — Returns the full cross-correlation profile (pitch-indexed weights) from which `templateHarmonicity` extracts summary statistics. Peaks in vp_w indicate strong virtual pitches (candidate fundamentals).

§9.1 compares `templateHarmonicity` and `tensorHarmonicity` in detail.

### 12.6 Balance and evenness

| MATLAB | Python | Description |
|:---|:---|:---|
| `dftCircular` | `dft_circular` | DFT of points on a circle |
| `dftCircularSimulate` | `dft_circular_simulate` | Monte Carlo Argand DFT under positional jitter |
| `balanceCircular` | `balance` | Balance (1 − \|F(0)\|) |
| `evennessCircular` | `evenness` | Evenness (\|F(1)\|) |

These functions apply equally to pitches (where the circle is one octave or other period) and to positions (where the circle is one rhythmic cycle or other periodic domain).

**dftCircular(p, w, period)** — Returns complex Fourier coefficients F and their magnitudes. F(1) (MATLAB 1-based) is the k = 0 coefficient; F(2) is k = 1; etc. Deterministic.

**dftCircularSimulate(p, w, period, sigma, ...)** — Monte Carlo estimator of the distribution of $|F(k)|$ under independent Gaussian positional jitter. For each draw, the perturbed positions are sorted and the Argand DFT is computed; the function returns the per-coefficient mean and standard deviation across draws. A third optional output (MATLAB `samples`; Python `return_samples=True`) yields the full $n_{\text{draws}} \times K$ magnitude matrix for histogram or quantile analysis. Name-value `nDraws` (default 10000) and `rngSeed` for reproducibility.

**balanceCircular(p, w, period [, sigma], ...)** — Returns a value in [0, 1]. Balance = 1 means the centre of gravity is at the circle's centre (e.g., augmented triad, whole-tone scale, isochronous rhythm). Supports weighted events. With `sigma > 0`, returns the Monte Carlo expectation $E[|\widetilde{F}(0)|]$ — the mean centroid magnitude across noise realizations — under positional jitter $\widetilde{p}_k = (p_k + \eta_k) \bmod P$, $\eta_k \sim \mathcal{N}(0, \sigma^2)$ (via `dftCircularSimulate`); the optional second output is the standard deviation. Note that perfectly balanced sets (where $F(0) = 0$) exhibit a positive Rayleigh-style bias at `sigma > 0` — $E[|\widetilde{F}(0)|]$ is strictly positive because magnitudes of complex Gaussians with zero mean are Rayleigh-distributed. The closed-form mean in this limit is $\sqrt{(1 - \alpha_1^2) \pi / (4K)}$ for uniform weights, $\alpha_1 = \exp(-2\pi^2 \sigma^2 / P^2)$. The `projCentroid` entry below sets out how this differs from its `centMag`.

**evennessCircular(p, period [, sigma], ...)** — Returns a value in [0, 1]. Evenness = 1 means the events are equally spaced (e.g., whole-tone scale, chromatic scale). Always uses uniform (binary) weights, following Milne et al. (2017). With `sigma > 0`, returns the Monte Carlo expectation; the optional second output is the standard deviation. Resort effects are non-trivial here once $\sigma$ is comparable to the smallest event-to-event gap, and Monte Carlo handles them automatically.

**Output convention.** In MATLAB, the standard-deviation output is requested via `nargout`: `b = balanceCircular(...)` returns scalar, `[b, bStd] = balanceCircular(...)` returns mean and SD. In Python, the analogous opt-in is the `return_std=True` flag: `b = mpt.balance(...)` returns scalar, `b, b_std = mpt.balance(..., return_std=True)` returns the tuple. Backward-compatible scalar return is preserved at `sigma = 0` and at `sigma > 0` without the SD opt-in.

### 12.7 Scale and rhythm structure

| MATLAB | Python | Description |
|:---|:---|:---|
| `coherence` | `coherence` | Coherence quotient (Carey / Rothenberg propriety) |
| `sameness` | `sameness` | Sameness quotient (Carey) |
| `nTupleEntropy` | `n_tuple_entropy` | Entropy of n-tuples of consecutive step sizes |
| `circApm` | `circ_apm` | Circular autocorrelation phase matrix |
| `edges` | `edges` | Edge detection via von Mises derivative |
| `projCentroid` | `proj_centroid` | Projected centroid |
| `meanOffset` | `mean_offset` | Mean offset (net upward arc) |
| `markovS` | `markov_s` | Optimal S-step Markov predictor |

All functions in this group operate on multisets of pitches or positions distributed around a periodic cycle and are applicable to both scales and rhythms. The first three take integer positions and an integer period; the remaining five also accept optional query points for evaluation at non-integer positions.

**coherence(p, period [, sigma], ...)** — Coherence quotient in [0, 1] at `sigma = 0`; may go below 0 at large `sigma`. The deterministic (`sigma = 0`) form returns 1 when there are no coherence failures (strict propriety: larger generic spans always have strictly larger specific sizes). The optional `'strict', false` flag uses non-strict propriety (Rothenberg's original definition: failures only when a larger span has a strictly *smaller* size). With `sigma > 0`, each `[d2 ≤ d1]` indicator is replaced by the Gaussian-CDF probability $\Phi((d_1 - d_2)/\sqrt{V})$, where $V$ depends on the `sigmaSpace` flag (`'position'`, default, propagating per-pair variance through endpoint sharing; or `'interval'`, $V = 2\sigma^2$ uniformly). At `sigma > 0` the strict / non-strict tie behaviour is replaced by the soft path's natural averaging (a tied pair contributes 0.5). Float positions and float `period` are accepted when `sigma > 0`.

**sameness(p, period [, sigma], ...)** — Sameness quotient in [0, 1] at `sigma = 0`; may go below 0 at large `sigma`. The deterministic (`sigma = 0`) form returns 1 when each specific interval size belongs to exactly one generic span. With `sigma > 0`, the strict equality test is replaced by a Gaussian match kernel $\exp(-(d_1 - d_2)^2 / (2V))$, with $V$ determined by `sigmaSpace` as for `coherence`. Float positions and float `period` are accepted when `sigma > 0`.

**nTupleEntropy(p, period [, n], ...)** — Shannon entropy of the distribution of n-tuples of consecutive step sizes. When n = 1 (the default), this is IOI / step-size entropy. Optional name-value pairs: `'sigma'` (Gaussian smoothing, default: 0), `'sigmaSpace'` (`'position'` or `'interval'`; default `'position'`, as at `coherence` above), `'method'` (`'shannon'`, `'normalized'`, `'differential'`, or `'renyi2'`; default `'normalized'`, the $[0, 1]$ ratio matching the historical behaviour — see [§11.2](#112-the-four-entropy-estimators)), `'base'` (default: 2), `'nPointsPerDim'` (grid resolution per effective dimension, default: the integer-step grid). With default arguments this exactly replicates Milne & Dean's (2016) discrete formulation; `sigma > 0` gives the smoothed extension of Milne (2024). At `sigma = 0` both `sigmaSpace` flags reduce to the integer step histogram. For `sigma > 0`, `'position'` is the exact position-uncertainty model at every `n` (the steps of a tuple carrying covariance $\sigma^2\,\mathrm{tridiag}(2, -1)$, obtained by binding the $n + 1$ events and taking the window relative), while `'interval'` treats each step's uncertainty independently. This function is a thin wrapper around the `differenceEvents` $\to$ `bindEvents` $\to$ `entropyMaet` pipeline (see [§12.3](#123-preprocessing) and [§13.1](#131-the-density-and-its-four-modes)); call those primitives directly for non-integer values, weighted events, non-periodic domains, or to obtain the n-tuple density itself for similarity comparison and other MAET operations.

**circApm(p, w, period, ...)** — Returns the period × period autocorrelation phase matrix R, the metrical weight profile rPhase (column sum), and the circular autocorrelation rLag (row sum). Optional `'decay'` parameter for exponential decay weighting.

**edges(p, w, period [, x], ...)** — Circular edge detection via convolution with the first derivative of a von Mises kernel. Returns absolute and signed edge weights. Optional `'kappa'` parameter controls kernel width.

**projCentroid(p, w, period [, x] [, sigma])** — Projection of the circular centroid (k = 0 Fourier coefficient) onto each angular position. Returns the projection vector, centroid magnitude, and centroid phase. With `sigma > 0`, returns the *expected* projection under positional jitter $\widetilde{p}_k = (p_k + \eta_k) \bmod P$, $\eta_k \sim \mathcal{N}(0, \sigma^2)$; this has a clean analytical form (no Monte Carlo) because the projection is linear in $F(0)$ and $F(0)$ is permutation-invariant: $E[y(x)] = \alpha_1 \cdot y_{\text{deterministic}}(x)$ with $\alpha_1 = \exp(-2\pi^2 \sigma^2 / P^2)$. `centMag` returns $\alpha_1 \cdot |F(0)| = |E[\widetilde{F}(0)]|$ — the magnitude of the *complex mean centroid* — consistent with the projection. Phase is preserved exactly in expectation. The different scalar $E[|\widetilde{F}(0)|]$ — the *mean centroid magnitude*, which picks up a positive Rayleigh-style bias when the perturbation cloud straddles the origin — is what `balanceCircular(p, w, period, sigma)` returns.

**meanOffset(p, w, period [, x])** — For each query point, the weighted sum of (upward arc − downward arc) to all events, normalized by the period. In a pitch-class context, this formalizes and generalizes Huron's (2008) "average pitch height," making the position-dependence explicit: it returns a value for every position around the circle. The term "mode height" for a closely related concept is used by Hearne (2020) and Tymoczko (2023).

**markovS(p, w, period [, S])** — Optimal S-step Markov predictor (default S = 3). For each position in the cycle, finds all positions with an identical S-step future context and returns their average weight. Originally by David Bulger.

### 12.8 Direction continuity

One utility that reads an ordered event sequence directly, rather than aggregating it into a tensor: a smoothed direction-continuity measure. Position-sensitive similarity of two sequences is instead computed via the core tensor functions on pitch-and-time-attributed multi-attribute tensors; see §8 for `buildMaet` and `simMaet`, and §8.7 for `sweptSimilarity`.

| MATLAB | Python | Description |
|:---|:---|:---|
| `continuity` | `continuity` | Backward same-direction run |

**continuity(seq, x, sigma [, 'w', w] [, 'mode', mode] [, 'theta', theta])** — Expected length and signed magnitude of the backward same-direction run leading up to each query, under Gaussian pitch uncertainty. Returns `[count, magnitude]`: `count` is non-negative, `magnitude` is signed (positive for ascending trends, negative for descending). The ratio `magnitude / count` gives a trend-slope measure. Modes `'strict'` (θ = 0) and `'lenient'` (θ = −1) set the break threshold; an explicit `'theta'` in [−1, +1] overrides. Optional per-event salience weights `w` (`[]` / `None` for all ones, a non-negative scalar, or a length-$N$ non-negative vector) scale each interval's contribution to `count` and `magnitude` by the difference-event salience $w_k \cdot w_{k+1}$ — the same rolling-product rule as `differenceEvents` at order 1. The break threshold acts on the unweighted sign-product, so weights modulate contribution size without shifting the halt condition. Defined only on linearly ordered domains.

### 12.9 Diagnostics, defaults, and estimates

| MATLAB | Python | Description |
|:---|:---|:---|
| `explainDispatch` | `explain_dispatch` | Report the route a call would take, and why, without running it |
| `mptDefaults` | `set_default`, `get_default`, `get_defaults`, `reset_defaults`, `show_defaults` | Inspect, set, and restore the toolbox-wide defaults |
| `estimateCompTime` | `estimate_comp_time` | micro-benchmark-based computation time estimate |

**explainDispatch(densX, densY [, 'method', m] [, 'truncationSigmas', t])**, or **explainDispatch(dens, nQueries, ...)** for an evaluation — Reports which route a similarity or evaluation call would take and why: the predicted time of each candidate, the accuracy floor in force, and the σ/period limit that follows from it. It calls the same selectors the call itself would, so what it reports is what would happen (§11.1).

**mptDefaults(...)** — Inspects and sets the toolbox-wide defaults: `truncationSigmas`, `kernelPrecision`, `kernelChunkBytes`, `showHints`, and, in Python, `kernel_threads`. The call forms, and the Python functions that correspond to them, are in §11.4.

**estimateCompTime(...)** — Estimates how long a call will take from a short micro-benchmark, before it is run.

---

## 13. MAETs: mathematical details

This section gives the density in full: the single-attribute expectation tensor and its four modes, what changes when an event carries several attributes, and the matrix-valued kernel covariances an ordered attribute may carry. §3.1 is the orientation, and §7 describes the preprocessing operations that shape a pre-MAET before its density is built.

### 13.1 The density and its four modes

This section gives the expectation tensor in full. The exposition proceeds in two stages: the single-attribute expectation tensor (ET) is introduced first, as the cleanest teaching unit and the form in which the framework has been empirically validated; the multi-attribute generalization (MAET) is then introduced as a natural extension that accommodates events characterized by several perceptually relevant properties simultaneously. The single-attribute case is a special case of the multi-attribute form; all downstream toolbox functions are designed to handle either.

An expectation tensor represents the distribution of $r$-tuples — or *$r$-ads* (dyads when $r = 2$, triads when $r = 3$, and so on) — of pitches (or time points) that a listener expects to perceive, given a weighted multiset and a model of perceptual uncertainty. Formally, it is an unnormalized Gaussian mixture density: a weighted sum of Gaussian kernels centred at all ordered $r$-tuples drawn from the multiset, with standard deviation $\sigma$ modelling perceptual uncertainty.

Given a multiset $\mathbf{p} = (p_1, p_2, \ldots, p_N)$ with weights $\mathbf{w} = (w_1, w_2, \ldots, w_N)$, the $r$-ad expectation tensor density at a query point $\mathbf{x}$ is:

$$f(\mathbf{x}) = \sum_j w_j \, \exp\!\left(-\frac{(\mathbf{x} - \mathbf{c}_j)^{\mathsf{T}} \, \mathbf{M} \, (\mathbf{x} - \mathbf{c}_j)}{2\sigma^2}\right)$$

where the sum is over all ordered $r$-tuples, $\mathbf{c}_j$ is the $j$-th tuple, and $\mathbf{M}$ is a quadratic form matrix determined by the mode (see below).

The toolbox implements this in two steps: `buildMaet` precomputes the tuple indices and weight products into a density struct, and `evalMaet` evaluates the density at query points. The cosine similarity between two such densities — which quantifies how similar the two weighted multisets are — is computed by `simMaet`. The inner product underlying this cosine similarity has an analytical solution (a finite double sum of Gaussian kernel evaluations), so `simMaet` computes it exactly without discretization, and `buildMaet` and `evalMaet` construct and evaluate the density analytically too. Two analytical decompositions of the inner product are available — Bulger's method and the Möbius method — which give the same values to floating-point precision but scale differently, so a cost model picks the cheaper per call (§11.1); the Möbius method also covers point evaluation and total mass. (The *Shannon* entropy of a Gaussian mixture has no closed form, so `entropyMaet` with the default `method='shannon'` and `spectralEntropy` use a grid — see §16; the closed-form `method='renyi2'` is an alternative, §11.2.)

The term "expectation tensor" reflects two ideas: (a) the density represents the *expected* perceptual distribution of r-ads given a weighted multiset of pitches or time points, smoothed by perceptual uncertainty σ; and (b) "tensor" refers to the fact that the discretized density is a rank-r array (an r-dimensional grid of values), with its domain being the r-fold product of the pitch (or time) space with itself. This is "tensor" in the numerical/data sense (a multidimensional array) rather than the strict algebraic sense (a multilinear map with specific transformation properties).

**The four modes.**

The expectation tensor has two logical flags that control its geometry:

**Periodic (isPer).** When true, the pitch (or time) line is wrapped into a circular domain of circumference `period`, so that values differing by the period are identified. For pitch, this implements pitch-class equivalence (e.g., with period = 1200, the pitch line is wrapped into a circle where 0 cents and 1200 cents are the same point). For rhythmic patterns, the period is the cycle length. When false, values are treated as points on an unbounded line.

**Relative (isRel).** When true, the density is invariant under transposition: shifting all values by the same amount does not change the density. Mathematically, this is achieved by projecting out the mean direction in $\mathbb{R}^r$, reducing the *effective space* (the space in which the density is supported) from $\mathbb{R}^r$ to the $(r-1)$-dimensional quotient $\mathbb{R}^r / \mathbb{R} \cdot \mathbf{1}$. In the formula above, the quadratic form matrix $\mathbf{M}$ becomes the projection matrix $\mathbf{I} - \mathbf{1}\mathbf{1}^{\mathsf{T}}/r$ (where $\mathbf{1}$ is the all-ones vector); the resulting quadratic form $Q(\mathbf{d}) = \sum_i d_i^2 - (\sum_i d_i)^2 / r$ is the squared distance in the quotient space under the Euclidean metric it inherits from $\mathbb{R}^r$. When false (absolute), $\mathbf{M} = \mathbf{I}$ (the identity matrix), the effective space is $\mathbb{R}^r$ itself, and the density depends on the actual values, not just intervals.

When both isPer and isRel are true, the cosine similarity computation (`simMaet`) uses an algebraically equivalent pairwise form of $Q$ with wrapped pairwise differences — $Q(\mathbf{d}) = \sum_{i<j} \mathrm{wrap}(d_i - d_j)^2 / r$ — to maintain exact transposition invariance on the circle. This is necessary because the component-wise periodic wrapping is nonlinear and can otherwise introduce $\pm\mathrm{period}$ artifacts in the pairwise differences between components of the wrapped difference vector.

The four combinations (absolute non-periodic, absolute periodic, relative non-periodic, relative periodic) cover a range of use cases. The choice depends on the question being asked: do we care about absolute positions or only intervals? Do we treat values a period apart as equivalent?

**Demos.** `demo_overview` / `demo_overview.py` (part 1e); `demo_maetPlots` / `demo_maet_plots.py`.

### 13.2 Multiple attributes

The description so far applies to a single attribute — pitch, or time, or another quantity treated on its own. Many music-cognition questions, though, are poorly served by reducing an event to a single attribute: a sequence of notes carries both *what* (pitch) and *when* (time); a polyphonic passage carries multiple pitches per event across voices; a piece may additionally carry register, timbre, metrical position, and spatial location for each event. Reducing all of this to one attribute is often a sensible modelling choice — the empirical literature referenced in §1 is largely in that regime — but there are questions it cannot ask. It cannot distinguish the same chord at beat 1 from the same chord at beat 3. It cannot locate recurrences of a motif within a longer passage. It cannot model contributions of timbral distance or register separation to perceived similarity. The multi-attribute expectation tensor (MAET) generalizes the single-attribute framework to accommodate several attributes per event simultaneously, while preserving the analytical properties that make the single-attribute case tractable. The user retains full control over which attributes to include in any given model — MAETs do not obligate the user to include them, they enable their inclusion when useful.

**Generalized equation.** A MAET represents events via $A$ *attributes* (each a kind of property — pitches in a chord, pitch of a voice, time, register, timbre, …), each carrying its own perceptual parameters (*attributes* and *elements* are defined in detail below). At each event, each attribute $a$ carries $K_a$ *elements* (for example, a pitch attribute representing a three-pitch chord has $K_a = 3$); the tensor is built at tuple size $r_a$ per attribute. The density at a query point $\mathbf{x} = (\mathbf{x}_1, \ldots, \mathbf{x}_A)$ is

$$f(\mathbf{x}_1, \ldots, \mathbf{x}_A) = \sum_j w_j \, \prod_a \exp\!\left(-\frac{(\mathbf{x}_a - \mathbf{c}_{a,j})^{\mathsf{T}} \, \mathbf{M}_a \, (\mathbf{x}_a - \mathbf{c}_{a,j})}{2\sigma_{g(a)}^2}\right)$$

where the sum is over multi-attribute tuples $j$ (formed by taking, per event, the Cartesian product across attributes of each attribute's $r_a$-combinations of its $K_a$ elements), $w_j$ is the tuple's combined weight (the product of its constituent elements' weights), $\mathbf{x}_a$ is the query in attribute $a$'s effective space of dimension $r_a - \mathrm{isRel}_a$, $\mathbf{c}_{a,j}$ is the attribute-$a$ centre for tuple $j$, $\mathbf{M}_a$ is the attribute-level quadratic form matrix ($\mathbf{I}$ for absolute, $\mathbf{I} - \mathbf{1}\mathbf{1}^{\mathsf{T}}/r_a$ for relative — determined by the attribute's $r_a$ and its isRel), and $\sigma_a$ is attribute $a$'s own σ. The density factorizes as a product across attributes, each attribute carrying its own σ, isRel, isPer, and period. The single-attribute tensor is the special case $A = 1$. The cosine-similarity inner product factors similarly and remains closed-form analytical in the multi-attribute case.

**Elements and attributes.** Two concepts carry the structure:

- An **element** is one weighted entry in an attribute's multiset at a single event: a scalar *position* in the attribute's space carrying a non-negative *weight*. An attribute with $K_a$ elements per event carries $K_a$ positions at each event. Elements within an attribute are treated as *exchangeable* — the tensor enumerates unordered *combinations* of $r_a$ elements taken from the $K_a$ available, so permuting the elements of one attribute at one event does not change the resulting tensor.
- An **attribute** is a kind of per-event property: several pitches in a chord, the pitch of voice 1, time, register, spectral centroid. Different attributes are *not* exchangeable — they are joined by Cartesian product across per-attribute combinations, so the tensor distinguishes "attribute-1 value paired with attribute-2 value" from the reverse pairing.

Every attribute is self-contained: it carries its own σ, isRel, isPer, period, and r. There is no separate "group" construct — attributes that should behave alike (e.g. several pitch voices) simply take equal parameter values. The word *group* survives in this guide only as informal English shorthand ("the pitch attributes"), never as an API argument or a structural unit.

The distinction between "several elements within one attribute" and "several attributes" is the modelling choice between exchangeable and distinguishable components. Consider a three-voice chord:

- Represented as **one pitch attribute with $K_a = 3$ elements** (and $r_a \in \{1, 2, 3\}$ depending on how we want to count intervals), the tensor treats the three pitches of each event as an unordered multiset — which pitches are sounding matters, which voice carries which does not.
- Represented as **three separate pitch attributes each with $K_a = 1$ element** (and $r_a = 1$), the tensor distinguishes (soprano, alto, tenor) as an ordered triple — voice identity matters. If we still want all three voices to behave alike (they are all pitches), we give all three attributes the same σ, isRel, isPer, and period — parameter-sharing by equal values, with voice identity left intact.

In the common case of a sequence of events over time with one or more pitches per event, the pitches form a pitch attribute (or several, depending on exchangeability), and time is a separate attribute with its own σ, isRel, isPer, and period — the pitches may be exchangeable (a single pitch attribute with $K_a > 1$) or not (several voice attributes sharing the same pitch parameters), but time always carries its own parameters because events at different times carry different time values.

Within this structure, every attribute carries its own self-contained geometry — there is no separate groups argument to `buildMaet`. The configurable parameters are:

- **Per attribute**: σ (perceptual uncertainty), `isRel` (transposition invariance), `isPer` (periodicity), `period` (when periodic), and r (tuple size). Attributes that should behave as one kind simply take equal values (e.g. several voice attributes given the same σ, `isRel`, `isPer`, and `period`).
- **Implicit**: the number of elements per event (K) is read from the shape of the input.

**Use cases.** A common MAET configuration is pitch combined with time — a pitch attribute (with a σ, isPer, and isRel each chosen to suit the analysis desired) and a time attribute carrying its own time σ and isPer=false. This unifies analyses of sequences of events over time with the rest of the toolbox: a sliding probe-tone analysis, a motif-recurrence search, and a pairwise similarity between two sequences all become ordinary `simMaet` operations on time-attributed tensors. Harmonic analyses occur when the pitch attribute has $K_a > 1$ pitches per event (exchangeable voices); polyphonic (voice-aware) analyses occur when voices are provided as several pitch attributes sharing the same parameters (non-exchangeable voices). But MAETs are not limited to pitch × time. Register can be added as an additional pitch-like attribute with a broader σ. Timbre can be attached as a numerical descriptor (e.g., spectral centroid) with its own σ, allowing two passages identical in pitch but different in instrumentation to be distinguished as such. Metrical position can be a periodic attribute with period equal to one bar. Spatial location in a stereo or surround context can be an angular (periodic) attribute with its own σ. Any quantity that can be represented as one or more values per event, and for which perceptual uncertainty is meaningful, can in principle become an attribute. Sliding-window analyses window the events before the density is built (§7.5 and §8.7), and §14 has worked examples, among them §14.4.

**Compatibility between multi-attribute densities.** Cosine similarity between two MAETs (via `simMaet`) requires the two densities to share the same attribute structure: the same number of attributes, the same per-attribute tuple size $r_a$, and the same per-attribute σ, isRel, isPer, and period (every attribute carries its own self-contained geometry). A mismatch on any of these raises an informative error (e.g., `simMaet:rMismatch`). What can differ between query and context are the per-event element counts $K_a$ (so a monophonic query can be compared against a polyphonic context on the same pitch attribute) and of course the number of events itself. This is a stronger compatibility requirement than the single-attribute case, where the parameters are passed as scalar arguments at the call site and therefore trivially match; the MA form requires densities to be built with `buildMaet` first, and the structural parameters are then read off the density objects.

**Demos.** `demo_overview` / `demo_overview.py` (part 2); `demo_softeningEquivalences` / `demo_softening_equivalences.py` (pairing attributes); `demo_helixBlend` / `demo_helix_blend.py`.

### 13.3 Matrix-valued kernel covariances

On a restricted but musically important class of attributes, the per-attribute `sigma` may be a matrix rather than a scalar. A symmetric positive-definite $r \times r$ covariance matrix $\Sigma$ replaces the isotropic $\sigma^2 I$: the Gaussian kernel becomes $\exp(-\delta^{\top} \Sigma^{-1} \delta / 2)$, so different directions in the attribute's tuple space carry different perceptual uncertainty, and correlations between tuple coordinates — such as the anti-correlations consecutive intervals inherit from their shared endpoints — enter the kernel directly. A scalar `sigma` is a standard deviation, as everywhere in the toolbox; a matrix `sigma` is the covariance, in squared units of the attribute's values.

The restrictions, all validated with an immediate error: the attribute must be ordered (`isExch = false`), absolute (`isRel = false`), non-periodic (`isPer = false`), and non-nested, with its tuple the whole multiset ($r = K$) and no NaN pads; `'spectrum'` input is not supported. Exact common-shift invariance (transposition, tempo) remains the province of `isRel = true`: a matrix covariance expresses graded shift tolerance via a large-but-finite variance along the all-ones direction, whose precision tends to the relative-mode projector in the limit, and an infinite entry is rejected rather than approximated.

Matrix `sigma` is accepted by `buildMaet`, `evalMaet`, `simMaet`, `entropyMaet` (all four methods), `sweptSimilarity`, and `sweptEntropy` (the last two via their `isExch` keyword to reach ordered attributes). Inner products require the same covariance on both operands, per attribute. Under `normalize = 'none'` the bare inner product is on the same scale as with a scalar `sigma`: each attribute carrying a covariance contributes $\det(\Sigma)^{1/2}$ where a scalar contributes $\sigma^{r}$, so $\Sigma = \sigma^2 I$ reproduces the scalar-`sigma` value exactly. Sliding analyses go through `sweptSimilarity` / `sweptEntropy`, which window the raw events before each density is built, so a matrix covariance carries through a windowed sweep unchanged.

The common covariances are built by `kernelCov(r, 'differenced', tf, 'sdValue', sp, 'sdInterval', si, 'sdShift', ss)` (Python `kernel_cov(r, sd_value, sd_interval, sd_shift, differenced=...)`) from three sources of variance, each included only when its width is set: independent noise on the values themselves (onsets, or pitches), `sdValue`; independent noise on the intervals between consecutive values, `sdInterval`; and a common shift of the whole tuple, `sdShift`. How each reaches the tuple depends on whether the tuple holds the values or their first differences, which the mandatory `differenced` flag declares. With $\nabla$ the first-differencing map, $(\nabla \mathbf{p})_i = p_{i+1} - p_i$, taken at the size its operand requires — $(r - 1) \times r$ on a tuple of $r$ values, $r \times (r + 1)$ on a tuple of the $r$ differences of $r + 1$ values — and $\nabla^{+}$ its pseudoinverse,

$$\Sigma = \mathrm{sd\_value}^2 \, I + \mathrm{sd\_interval}^2 \, \nabla^{+}(\nabla^{+})^{\top} + \mathrm{sd\_shift}^2 \, \mathbf{1}\mathbf{1}^{\top} \quad (\texttt{differenced = false}),$$

$$\Sigma = \mathrm{sd\_value}^2 \, \nabla\nabla^{\top} + \mathrm{sd\_interval}^2 \, I + \mathrm{sd\_shift}^2 \, \mathbf{1}\mathbf{1}^{\top} \quad (\texttt{differenced = true}).$$

The two are one model: differencing a tuple of $r$ values carries the first to the second at tuple size $r - 1$, since $\nabla\mathbf{1} = \mathbf{0}$ annihilates the ridge and $\nabla\nabla^{+} = I$. Not every combination is admissible: the undifferenced form is positive-definite when `sdValue` $> 0$, or when `sdInterval` and `sdShift` are both non-zero (the centred walk annihilates $\mathbf{1}$ and the ridge is rank one, so neither serves alone); the differenced form when `sdValue` $> 0$ or `sdInterval` $> 0$, only the ridge alone failing. On undifferenced values, interval noise accumulates from one value to the next, a random walk that $\nabla^{+}$ centres on the tuple's mean so that no value is privileged, and the ridge along $\mathbf{1}$ tolerates a common shift of every value (a transposition of pitches, a displacement of onsets), widening with `sdShift`, `isRel = true` being the exact quotient it approaches. On differenced values, value noise reaches each interval through its two endpoints ($D D^{\top}$ is tridiagonal, $2$ on the diagonal and $-1$ beside it, adjacent intervals sharing an endpoint; the counterpart of `sigmaSpace = 'position'` in `nTupleEntropy`), interval noise is independent per interval (`sigmaSpace = 'interval'`), and on times the two are the two levels of the Wing & Kristofferson (1973) timing model. The ridge then adds a constant to every interval, seldom the equivalence wanted for uneven rhythms, so `sdShift` is usually omitted there; on log-differenced values it becomes a common factor on the intervals (a tempo change, or intervallic augmentation) and is wanted again. The covariance applies in whatever coordinates the attribute carries (log inter-onset intervals for multiplicative tempo tolerance; semitones or cents for pitch); all three widths are standard deviations in those coordinates. The returned matrix drops directly into the `sigma` argument of the functions above.

```python
# Sliding rhythm-shape comparison, tempo-tolerant: ordered log-IOI
# triples, position noise from the shared onsets, graded tempo ridge.
Sigma = mpt.kernel_cov(3, sd_value=0.03, sd_shift=0.4, differenced=True)
prof = mpt.swept_similarity(
    p_context, w_context, p_query, w_query,
    [Sigma, sigma_t], [3, 1], [False, False], [False, False], [0.0, 0.0],
    is_exch=[False, True], sweep={1: onsets}, align={1: 'window'},
    drop=[1], window={1: ("rect", 2.0)})
```

**Demos.** `demo_tempoInvariance` / `demo_tempo_invariance.py`; `demo_softeningEquivalences` / `demo_softening_equivalences.py` (the kernelCov identity); `demo_repetitionHandling` / `demo_repetition_handling.py`.

---

## 14. Worked examples

The worked examples below use MATLAB syntax. Python equivalents are in `python/demos/` — start with `demo_overview.py` for a quick tour of all function families, or see the individual demo scripts listed in [§15](#15-demo-scripts) for the Python version of each example below.

### 14.1 EDO approximation via relative dyad tensors

Which equal divisions of the octave best approximate a just-intonation major triad? This is Example 6.3 / Figure 4 of Milne et al. (2011), which uses a relative dyad tensor (r = 2, isRel = true, isPer = true) — a one-dimensional density over intervals, distinct from the standard monad SPCS parameters (r = 1, isRel = false) used in later work.

```matlab
% JI major triad
ref = [0, 386.31, 701.96];

% Parameters
sigma = 10;
r = 2;
isRel = true;
isPer = true;
period = 1200;
nHarm = 12;
rho = 1;

% Add spectra to reference
[ref_p, ref_w] = addSpectra(ref, [], 'harmonic', nHarm, 'powerlaw', rho);
dens_ref = buildMaet(ref_p, ref_w, sigma, r, isRel, isPer, period);

% Sweep n-EDOs
nRange = 3:53;
s = zeros(size(nRange));
for i = 1:numel(nRange)
    n = nRange(i);
    edo = (0:n-1) * period / n;
    [edo_p, edo_w] = addSpectra(edo, [], 'harmonic', nHarm, 'powerlaw', rho);
    dens_edo = buildMaet(edo_p, edo_w, sigma, r, isRel, isPer, period);
    s(i) = simMaet(dens_ref, dens_edo, 'verbose', false);
end

bar(nRange, s);
xlabel('n-EDO');
ylabel('SPCS');
title('SPCS of n-EDOs to JI major triad');
```

**Demos.** `demo_edoApprox` / `demo_edo_approx.py`.

### 14.2 Consonance landscape of triads

Plot five consonance measures for triads [0, x, y] over a grid of intervals.

```matlab
% Grid of intervals (cents)
step = 10;
ints = 0:step:1200;
[X, Y] = meshgrid(ints, ints);

% Compute roughness for each triad
sigma = 12;
spec = {'harmonic', 12, 'powerlaw', 1};
R = NaN(size(X));
for i = 1:numel(X)
    p = [0, X(i), Y(i)];
    [fp, fw] = addSpectra(p, [], spec{:});
    f_hz = transformAttributes(fp, [], {'cents', 'hz'});
    R(i) = roughness(f_hz, fw);
end

imagesc(ints, ints, -R);
axis xy;
xlabel('Interval 1 (cents)');
ylabel('Interval 2 (cents)');
title('Smoothness (negative roughness)');
colorbar;
```

**Demos.** `demo_triadConsonance` / `demo_triad_consonance.py`; `demo_triadSpcsGrid` / `demo_triad_spcs_grid.py`.

### 14.3 Rhythmic structure features

Compute and compare structural features of different rhythmic patterns.

```matlab
patterns = {
    [0, 2, 4, 6, 8, 10, 12, 14],   % isochronous 8 in 16
    [0, 3, 6, 10, 12],               % son clave
    [0, 2, 5, 7, 9, 12, 14],         % bossa nova
};
period = 16;

for i = 1:numel(patterns)
    p = patterns{i};
    fprintf('Pattern %d: %s\n', i, mat2str(p));
    fprintf('  Balance   = %.3f\n', balanceCircular(p, [], period));
    fprintf('  Evenness  = %.3f\n', evennessCircular(p, period));
    fprintf('  Coherence = %.3f\n', coherence(p, period));
    fprintf('  Sameness  = %.3f\n', sameness(p, period));
    fprintf('  IOI entropy (n=1) = %.3f\n', nTupleEntropy(p, period, 1));
    fprintf('  2-tuple entropy   = %.3f\n', nTupleEntropy(p, period, 2));
    fprintf('\n');
end
```

**Demos.** `demo_rhythmTensors` / `demo_rhythm_tensors.py`; `demo_dftCircularSimulate` / `demo_dft_circular_simulate.py`.

### 14.4 Probe-tone scanning with irregular timing and event weights

The Quick Start example of probe-tone fitting treated the context as an unordered pitch multiset with per-event salience weights, computing a single cosine similarity via `simMaet`. In realistic listening situations, context events are not uniformly spaced in time (ritardandi, held tones, metrical anacruses all break uniform spacing), and they are not equally salient (metrical position, loudness, duration, and phenomenal accent all modulate how strongly each event registers). A thorough probe-tone model accounts for both factors.

Two complementary approaches are available. The first is the pooled-context approach of the Quick Start: treat the context as a weighted multiset and compute a single similarity. The second is a *time-resolved* approach: represent the context as a multi-attribute tensor carrying both pitch and time, and sweep a window along the time attribute to obtain a similarity *profile* showing how probe fit evolves moment by moment. The two approaches answer different questions — "what is the aggregate fit?" versus "where and when does the probe fit best?" — and both are naturally expressed with MAETs.

The example below scans twelve chromatic probes against a short melodic line `C F G C` representing a I–IV–V–I cadence, using the pooled-context approach. The final tonic is held for two beats (irregular time grid); the first and last events are metrically accented (irregular salience). Both effects are carried in a two-attribute pre-MAET: the salience sits on the pitch weights, and `weightEvents` multiplies in an exponential recency profile over elapsed time, dropping the time attribute once it has done its work. Each probe is compared against the same weighted context via `simMaet` in list mode — the context density (a single struct) is broadcast against the cell / list of twelve probe densities in one call, avoiding an explicit loop.

**MATLAB:**
```matlab
% Context: melodic line of a cadential progression
context = transformAttributes([60 65 67 60], [], {'midi', 'cents'});   % C F G C

% Irregular event times: final tonic held twice as long
t_events = [0 1 2 4];

% Per-event salience: accented first and last events
salience = [1.0; 0.6; 0.6; 1.3];

% Salience on the pitch weights, time as a second attribute
pm = packPreMaet({context, t_events}, {salience(:).', []});

% Exponential decay before the final onset, over elapsed time
% (sd = 1 / 0.4 is the decay's time constant); the time attribute is
% dropped once it has weighted the pitches
pm = weightEvents(pm, 2, 1, t_events(end), 'exponentialBefore', ...
                  'sd', 2.5, 'dropInputAttr', true);
[ctxVals, w] = unpackPreMaet(pm);

% Apply spectral enrichment and build the context density once
spec = {'harmonic', 12, 'powerlaw', 1};
[ctx_p, ctx_w] = addSpectra(ctxVals{1}, w{1}, spec{:});
ctx_dens = buildMaet(ctx_p, ctx_w, 10, 1, false, true, 1200);

% Build all 12 chromatic probe densities up front
probes     = transformAttributes(60:71, [], {'midi', 'cents'});
probe_dens = cell(1, 12);
for i = 1:12
    [probe_p, probe_w] = addSpectra(probes(i), [], spec{:});
    probe_dens{i} = buildMaet(probe_p, probe_w, 10, 1, false, true, 1200);
end

% List-mode broadcast: single context vs cell of probes
fitCell = simMaet(ctx_dens, probe_dens, 'verbose', false);
fit     = cell2mat(fitCell);

bar(0:11, fit);
xlabel('Probe pitch class (semitones from C)');
ylabel('Spectral pitch-class fit');
title('Probe fit to C–F–G–C with time-aware, salience-weighted context');
```

**Python:**
```python
import numpy as np
context  = mpt.transform_attributes(np.array([60, 65, 67, 60]), None, ('midi', 'cents'))
t_events = np.array([0.0, 1.0, 2.0, 4.0])
salience = np.array([1.0, 0.6, 0.6, 1.3])

pm = mpt.pack_pre_maet([context[None, :], t_events[None, :]],
                        [salience[None, :], None])
pm = mpt.weight_events(pm, 1, 0, t_events[-1], 'exponentialBefore',
                       sd=2.5, drop_input_attr=True)
ctx_vals, w, _ = mpt.unpack_pre_maet(pm)

spec = ('harmonic', 12, 'powerlaw', 1.0)
ctx_p, ctx_w = mpt.add_spectra(ctx_vals[0], w[0], *spec)
ctx_dens = mpt.build_maet(ctx_p, ctx_w, 10., 1, False, True, 1200.)

probes = mpt.transform_attributes(np.arange(60, 72), None, ('midi', 'cents'))
probe_dens = []
for i in range(12):
    pp, pw = mpt.add_spectra(np.array([probes[i]]), None, *spec)
    probe_dens.append(
        mpt.build_maet(pp, pw, 10., 1, False, True, 1200.)
    )

# List-mode broadcast: single context vs list of probes
fit = mpt.sim_maet(ctx_dens, probe_dens, verbose=False)
```

The resulting profile peaks at C (the tonic, present at both endpoints and carried by the most heavily weighted final event), with a secondary peak at G. The emphasis on the final tonic and the decay of the intermediate events fall out of the recency profile composed with the metrical salience already on the pitch weights — the decay measured in elapsed time rather than in event count, because the profile reads the time attribute's values.

For a time-resolved view — how probe fit varies moment by moment along the context, rather than aggregated to a single number — the context can be carried with both pitch and time as attributes and scanned with `sweptSimilarity`. The following uses a single probe (C) to show the pattern. The probe is a single event with no arrangement in time, so the sweep values align the window only (`'align'`, `'window'`) and the time attribute is dropped: at each sweep value the context's events are reweighted by a window in time, the time attribute is marginalized, and the probe's pitch is compared with what remains.

**MATLAB:**
```matlab
% Same context as above, now with an explicit time attribute; w{1} is the
% per-event salience-and-decay weight vector computed above.
probeC = transformAttributes(60, [], {'midi', 'cents'});

% Align a Gaussian window of standard deviation 1 time unit (full width
% 2 * sqrt(3)) at each sweep value along the context.
t_sweep = linspace(-0.5, 4.5, 51);
profile = sweptSimilarity({context, t_events}, {w{1}, ones(1, 4)}, ...
                          {probeC, 0}, [], ...
                          [10 0.3], [1 1], [false false], ...
                          [true false], [1200 0], ...
                          'sweep', {2, t_sweep}, 'align', {2, 'window'}, ...
                          'drop', 2, 'window', {2, {'gaussian', 2 * sqrt(3)}});
plot(t_sweep, profile);
xlabel('Time'); ylabel('Fit of probe C');
title('Time-resolved probe fit');
```

**Python:**
```python
probe_C = mpt.transform_attributes(np.array([60]), None, ('midi', 'cents'))

t_sweep = np.linspace(-0.5, 4.5, 51)
profile = mpt.swept_similarity(
    [context[None, :], t_events[None, :]], [w[0].reshape(1, -1), np.ones((1, 4))],
    [probe_C[None, :], np.array([[0.]])], None,
    [10., 0.3], [1, 1], [False, False], [True, False], [1200., 0.],
    sweep={1: t_sweep}, align={1: 'window'}, drop=[1],
    window={1: ("gaussian", 2 * np.sqrt(3))}, verbose=False)
```

The resulting profile peaks at time 4, the final C, which carries the most weight, with a smaller local peak near time 0, the first C, and is lower between them. The pooled-context scan and the time-resolved scan are complementary: the former asks "how well does the probe fit the context as a whole?", the latter "where within the context does the probe fit best?".

**Demos.** `demo_probeTone` / `demo_probe_tone.py`.

### 14.5 Recurrence of interval content across a melody

For locating recurrent interval content within a monophonic melody — *where does this motif reappear, regardless of its starting pitch?* — the analysis combines two steps: convert the pitch sequence into a sequence of inter-event intervals with `differenceEvents`, then translate the query along the differenced sequence with `sweptSimilarity`, here with a time window travelling with it. The first step handles transposition invariance by construction (intervals are the same regardless of absolute pitch), and the second returns a similarity profile showing where the motif recurs. `differenceEvents` returns a pre-MAET and `sweptSimilarity` takes one, so the two chain directly; the example below reads the differenced values out of the pre-MAET and passes them in the positional form, so that every geometry parameter is visible at the call (`demo_overview` §2c shows the pre-MAET form).

The example below uses an eight-note melody, C D E♭ F A C D E♭, containing the interval pattern $(+2, +1)$ — a whole step followed by a semitone — at two places: positions 0–2 (C D E♭) and positions 5–7 (C D E♭ again, an octave higher). The query G A B♭ carries the same $(+2, +1)$ pattern but at a different absolute pitch. After `differenceEvents`, both melody and query live in interval space, and `sweptSimilarity` locates the motif by translating the query along the melody's time attribute to each sweep value, with a time window travelling with it (`'align'`, `'both'`).

**MATLAB:**
```matlab
% Melody: C D Eb F A C D Eb, one event per time unit.
melody_p = transformAttributes([60 62 63 65 69 72 74 75], [], {'midi', 'cents'});
melody_t = 0:7;

% Query: G A Bb — same (+2, +1) pattern transposed.
query_p = transformAttributes([67 69 70], [], {'midi', 'cents'});
query_t = [0 1 2];

% Convert each event sequence into inter-event differences: pitch gets
% order-1 differencing (intervals); time gets order-0 (pass through, but
% the leading event is dropped so the event counts align with the pitch
% differences).
% differenceEvents returns a pre-MAET; its pAttr field holds the values.
pmMel  = differenceEvents({melody_p, melody_t}, [], [1 0]);
pmQry  = differenceEvents({query_p,  query_t},  [], [1 0]);
mel_pd = pmMel.pAttr;
qry_pd = pmQry.pAttr;
% mel_pd{1} = [200 100 200 400 300 200 100]  (cents, signed)
% mel_pd{2} = [1 2 3 4 5 6 7]                (time of each interval event)

% Per-attribute geometry of the differenced sequences: r = 1, absolute,
% non-periodic. Pitch attribute: sigma = 50 cents gives clean separation
% between 100-cent interval categories; non-periodic so that, e.g., +1300
% is distinct from +100. Time attribute: sigma = 0.3 time units.
sigma = [50 0.3]; r = [1 1]; isRel = [false false];
isPer = [false false]; period = [0 0];

% At each sweep value a Gaussian time window of standard deviation 0.6
% (full width 0.6 * 2 * sqrt(3)) is aligned there and the query's middle
% lands there too ('align', 'both'), so the window is centred on the
% query. The sweep values are stepped across the melody (start and stop
% default to its extent), and the second output is the translation
% applied at each, mu. Melody and query were both written from time 0
% before differencing, so mu is the time at which the original query
% starts in the melody. Differencing shifted the query's middle (to 1.5,
% the mean time of its two intervals), and the window with it, but mu is
% measured from the query as written.
[S, mu] = sweptSimilarity(mel_pd, [], qry_pd, [], ...
                          sigma, r, isRel, isPer, period, ...
                          'step', {2, 0.25}, 'align', {2, 'both'}, ...
                          'window', {2, {'gaussian', 0.6 * 2 * sqrt(3)}});
offsets = mu{2};
bar(offsets, S);
xlabel('Offset of the query (time at which it starts)');
ylabel('Interval-content similarity to query (+2, +1)');
title('Transposition-invariant interval recurrence profile');
```

**Python:**
```python
import numpy as np
melody_p = mpt.transform_attributes(np.array([60, 62, 63, 65, 69, 72, 74, 75]), None, ('midi', 'cents'))
melody_t = np.arange(8, dtype=float)

query_p = mpt.transform_attributes(np.array([67, 69, 70]), None, ('midi', 'cents'))
query_t = np.arange(3, dtype=float)

# difference_events returns a pre-MAET; its "p_attr" key holds the values.
mel_pd = mpt.difference_events([melody_p[None, :], melody_t[None, :]],
                               None, [1, 0])["p_attr"]
qry_pd = mpt.difference_events([query_p[None, :],  query_t[None, :]],
                               None, [1, 0])["p_attr"]

S, mu = mpt.swept_similarity(
    mel_pd, None, qry_pd, None,
    [50., 0.3], [1, 1], [False, False], [False, False], [0., 0.],
    step={1: 0.25}, align={1: 'both'},
    window={1: ("gaussian", 0.6 * 2 * np.sqrt(3))},
    return_offsets=True, verbose=False)
offsets = mu[1]
```

Differencing drops the leading event of both sequences but leaves the surviving time values where they were (the query's two intervals sit at t = 1 and t = 2), so an offset $\mu$ still says where the original query starts. The profile peaks sharply at offsets $\mu = 0$ and $\mu = 5$ — the starts of the two $(+2, +1)$ occurrences in the melody — with identical heights, reflecting that the two occurrences are identical in interval content. Between the peaks, the profile is depressed where the melody's local interval content diverges from the query (most strongly around $\mu \approx 3$, where the intervals $(+4, +3)$ span the leap from F to A to C), and has a lower maximum at $\mu = 1.75$, where the query's $+200$ falls close to the melody's $+200$ at t = 3.

Downstream aggregations on the returned profile are one-line computations:

```matlab
% Best match offset and strength
[sim, idx] = max(S);
best_offset = offsets(idx);

% Recency-weighted typicality over the profile
w_prof = exp(-0.5 * (numel(S) - (1:numel(S))));
typicality = (w_prof(:).' * S(:)) / sum(w_prof);
```

Window width is a modelling choice: narrower windows are more selective (picking out single motif occurrences), wider windows smooth over multiple adjacent events and give a coarser view. The shape parameter of the window controls the softness of the window edge (`0` or `'gaussian'` is a pure Gaussian, `1` or `'rect'` is a rectangular cutoff); intermediate values interpolate between the two extremes.

**Demos.** `demo_sweptSimilarity` / `demo_swept_similarity.py`; `demo_preprocessing` / `demo_preprocessing.py` (part 2); `demo_jmm_2_2_motif` (JMM).

### 14.6 Virtual pitch analysis

Identify the strongest virtual pitches of a chord.

```matlab
% C major triad in absolute cents (MIDI 60, 64, 67)
p = transformAttributes([60 64 67], [], {'midi', 'cents'});
spec = {'harmonic', 36, 'powerlaw', 1};

[vp_p, vp_w] = virtualPitches(p, [], 12, 'chordSpectrum', spec);

% Plot with MIDI pitch axis
vp_midi = transformAttributes(vp_p, [], {'cents', 'midi'});
plot(vp_midi, vp_w);
xlabel('Virtual pitch (MIDI)');
ylabel('Salience');
title('Virtual pitches of C major triad');

% Mark chord tones
hold on;
chord_midi = [60 64 67];
for m = chord_midi
    xline(m, '--r');
end
hold off;
```

**Demos.** `demo_virtualPitches` / `demo_virtual_pitches.py`.

### 14.7 Working with audio files

Extract peaks from audio and compute multiple features.

```matlab
% Extract peaks
audioDir = fullfile(fileparts(which('audioPeaks')), 'audio');
[f, w] = audioPeaks(fullfile(audioDir, 'music_sample.wav'), 'sigma', 12, 'plot', true);

% Convert to cents
p = transformAttributes(f, [], {'hz', 'cents'});

% Spectral entropy (no addSpectra — peaks are already the spectrum)
H = spectralEntropy(p, w, 12);

% Template harmonicity
[hMax, hEnt] = templateHarmonicity(p, w, 12);

% Roughness (needs Hz)
r = roughness(f, w);

fprintf('Spectral entropy  = %.3f\n', H);
fprintf('Harmonicity hMax  = %.3f\n', hMax);
fprintf('Harmonicity hEnt  = %.3f\n', hEnt);
fprintf('Roughness         = %.3f\n', r);
```

**Demos.** `demo_audioAnalysis` / `demo_audio_analysis.py`.

---

## 15. Demo scripts

Demo scripts are included in each language, with user-adjustable parameters at the top. The first to open is `demo_0_startHere` (MATLAB) or `demo_0_start_here.py` (Python), which runs no analysis but prints a guide to the others: a route through them for a beginner, how they fit together, and the demos grouped by topic. The first demo on that route is `demo_overview` (MATLAB) or `demo_overview.py` (Python), which exercises every major function family in a single script, from single multisets through multi-attribute tensors to the structural measures, and ends each section with pointers to the demos that go further.

### MATLAB demos

MATLAB demo scripts are in `matlab/demos/`. To run a demo, open it in the MATLAB editor or navigate to the `demos/` folder and run it from there.

| Script | Description | Based on |
|:---|:---|:---|
| `demo_0_startHere` | A guide to the demos, printed when run: where to begin, how the demos fit together, and the demos by topic (a demo appears under every topic it covers) | — |
| `demo_overview` | Quick tour of all major function families: a spectrally enriched chord as a density over pitch, SPCS, entropy, and the mass in a region of a single multiset's density, the four tensor parameters drawn, a multi-attribute motif search in absolute and differenced form, each of two triads' share of a melody's notes in a moving window, harmonicity, roughness, balance, evenness, coherence, sameness, n-tuple entropy, mean offset (including the brightness of the diatonic modes), edges, the APM phase sum, and Markov; each section points to the demos that go further | — |
| `demo_audioAnalysis` | Two-pass peak extraction (unsmoothed then smoothed) from audio files, with spectral similarity, harmonicity, roughness, and virtual pitch analysis | — |
| `demo_batchProcessing` | Analysing experimental data: a trial table of scale x chord x root, with a paired measure (SPCS via batched-raw `simMaet`) and single-set measures (spectral entropy, template and tensor harmonicity, roughness) per trial. The trial matrix goes straight in: every batched feature deduplicates its rows internally. Workflow 3 sets that row-wise sense of batching against the other one — a cell of whole pre-MAETs, which loops rather than collapses, and the translation sweep that is the exception | — |
| `demo_edoApprox` | PCS of n-EDOs against a JI chord | Milne et al. (2011), Ex. 6.3 / Fig. 4 |
| `demo_maetPlots` | Every combination of r, isRel, isPer, and isExch of the diatonic scale, drawn by each of `plotMaet`'s methods (kernels, points, density), in one to three dimensions | — |
| `demo_genChainPcs` | PCS of generator-chain tunings as the generator is swept (linear and circular plots) | Milne et al. (2011), Ex. 6.4–6.5 / Figs. 5–7 |
| `demo_triadConsonance` | Five consonance measures over a grid of triad intervals | — |
| `demo_triadSpcsGrid` | SPCS heatmap of 12-EDO triads with a fifth | Milne et al. (2011), Fig. 3 |
| `demo_virtualPitches` | Virtual pitch salience profiles for example chords | — |
| `demo_helixBlend` | Pitch-class and register routed simultaneously through periodic pitch-class and linear register attributes; a time-windowed `sweptSimilarity` sweep of a motif against a stream, repeated across the register σ, produces a continuum between pitch-class and pitch-height similarity | — |
| `demo_preprocessing` | The pre-MAET preprocessing operations (`differenceEvents`, `bindEvents`, `translateAttributes`, `weightEvents`, `transformAttributes`, `selectPreMaet`, `bindAttributes`, `separateAttributes`) on a chorale fragment, and the compositions that commute or absorb one another | — |
| `demo_scoreWorkflow` | From a score file to a result: read, look at the table, select, grid, encode a categorical column, build, and compare two chords across the pitch-class/pitch-height blend. The spine of the `demo_score*` family, pointing at the others where a step has more to it | — |
| `demo_scoreGrid` | Sampling a score on a grid: choosing the step, the three weight policies (`'coverage'`, `'presence'`, `'item'`) and the note-table density each reproduces at r = 1, empty slices, and the `'limits'` and `'duration'` arguments | — |
| `demo_scoreCategoricals` | Encoding a categorical column three ways — `'orderedMultiset'`, `'simplex'`, and no role at all — on one chorale: what question each asks of a re-voiced chord, the family relation between them as a `selectPreMaet` call, and the pairing trap of bound chords without a role | — |
| `demo_preMaetIo` | Showing, exporting, and importing a pre-MAET: markdown, LaTeX, CSV out and CSV back on one object, with parameter overrides, nesting, NA, a kernel covariance, and a pre-MAET from a score | — |
| `demo_repetitionHandling` | Repeated pitches under interval-scale and tempo invariance: three treatments (excise, prolong, count), which variants each identifies, and why uniform repetition confounds with a tempo change once tempo is quotiented out | — |
| `demo_tempoInvariance` | Tempo invariance by degree: `kernelCov`'s ridge against the exact relative quotient, on logarithmic inter-onset intervals, with a `sweptSimilarity` sweep | — |
| `demo_sweptSimilarity` | `sweptSimilarity` in depth, on one melody holding four statements related to a four-note query (exact, transposed, reordered, rhythm only), each setting finding a different subset: translation in time (the canonical `'sweep', a`) and in pitch and time, each match labelled by its key; `'both'`, a local comparison under cosine, where the window's width sets how much nearby unmatched material counts against a match; `'window'` with time dropped (pitch content per bar), and with the query translated in pitch as well (the transposition each bar holds, under the default normalization and under cosine), with time relative through bound onsets (rhythm per bar), and with time absolute (two voices compared in place, the bars summing to the whole); `'independent'` as a correlogram of a drifting lag; and what the calls compute, `sweepSimMaet` in one pass against translated copies one by one, timed | — |
| `demo_sigmaSpace` | Soft (`sigma > 0`) `sameness`, `coherence`, and `nTupleEntropy` on the diatonic scale, comparing `sigmaSpace = 'position'` against `'interval'` across a range of σ; also demonstrates the diatonic-tritone tie under positional jitter and the `n = 1` exactness relationship for `nTupleEntropy` | — |
| `demo_probeTone` | Probe-tone fit to a context: SPCS profiles of the C-major scale against the Krumhansl and Kessler (1982) ratings and of Porcupine[7] in 22-EDO; a melody context as a pre-MAET, with and without recency weighting by `weightEvents` (`'exponentialBefore'`) and under harmonic, stretched, and stiff-string spectra; and `continuity` of each probe after the melody | Milne (2026, TISMIR), Fig. 4 |
| `demo_softeningEquivalences` | Softening an equivalence by pairing attributes: a [rel] copy with an absolute copy (transposition), a [per] copy with a non-periodic copy (octave), and an exchangeable copy with an ordered copy (reordering), each swept over the unflagged copy's width σ_f; the transposition pairing checked against the `kernelCov` ridge under the Online Supplement's mapping | Milne (2026, JMM), Online Supplement |
| `demo_rhythmTensors` | Expectation tensors of rhythms in a 16-pulse cycle, the son clave against five other timelines: similarity at r = 1 absolute (phase-sensitive) and r = 2 relative (rotation-invariant inter-onset-interval content), the two densities drawn, Rényi-2 entropy of the relative tensor as rhythmic complexity, and circular `differenceEvents` and `bindEvents` reproducing `nTupleEntropy`, then compared across rhythms with `simMaet` | Milne & Dean (2016) |
| `demo_dftCircularSimulate` | Argand-DFT under positional jitter: balance / evenness sweeps with `sigma > 0`, the Rayleigh bias on a perfectly balanced multiset, full per-coefficient distributions via `dftCircularSimulate`, and the analytical α₁ damping for `projCentroid` | — |
| `demo_dispatchAndKernelControls` | Tour of the performance controls: method dispatch for flat attributes (Bulger's method, the Möbius method, the tuple centres) with `explainDispatch`; the level-by-level contraction of a nested (bound, spectrally enriched) attribute; one-pass sweeps with `sweepSimMaet` (mixture, orbit, and contraction routes) against a comparison per offset; kernel truncation (`truncationSigmas`); single-precision kernel arithmetic (`kernelPrecision`); the toolbox-wide defaults (`mptDefaults`), each explained; and the three entropy estimators (`'shannon'`, `'differential'`, `'renyi2'`) compared across the density's dimension | — |

### Python demos

Python equivalents of all demos are in `python/demos/`. They follow the same structure and produce the same results; the plotting demos require `matplotlib` (`pip install matplotlib`) and the audio demo requires `soundfile` (`pip install soundfile`).

| Script | MATLAB equivalent | Description |
|:---|:---|:---|
| `demo_0_start_here.py` | `demo_0_startHere` | A guide to the demos, printed when run: where to begin, how they fit together, and the demos by topic |
| `demo_overview.py` | `demo_overview` | Quick tour of all major function families |
| `demo_audio_analysis.py` | `demo_audioAnalysis` | Two-pass audio peak extraction and perceptual features |
| `demo_batch_processing.py` | `demo_batchProcessing` | Analysing experimental data: per-trial paired and single-set features, with internal dedup of the trial matrix; Workflow 3 contrasts that with a list of whole pre-MAETs |
| `demo_edo_approx.py` | `demo_edoApprox` | PCS of n-EDOs against a JI chord |
| `demo_maet_plots.py` | `demo_maetPlots` | Expectation tensor densities of the diatonic scale drawn by each of `plot_maet`'s methods (1–3D) |
| `demo_gen_chain_pcs.py` | `demo_genChainPcs` | Generator-chain PCS (linear and circular plots) |
| `demo_triad_consonance.py` | `demo_triadConsonance` | Five consonance measures over a triad grid |
| `demo_triad_spcs_grid.py` | `demo_triadSpcsGrid` | SPCS heatmap of triads with a fifth |
| `demo_virtual_pitches.py` | `demo_virtualPitches` | Virtual pitch salience profiles |
| `demo_helix_blend.py` | `demo_helixBlend` | Helix blend: pitch-class and register continuum |
| `demo_preprocessing.py` | `demo_preprocessing` | The pre-MAET preprocessing operations and their compositions |
| `demo_score_workflow.py` | `demo_scoreWorkflow` | From a score file to a result: the common path, end to end |
| `demo_score_grid.py` | `demo_scoreGrid` | Sampling a score on a grid: the step, the weighting, and empty slices |
| `demo_score_categoricals.py` | `demo_scoreCategoricals` | Encoding a categorical column: three ways, and what each asks |
| `demo_pre_maet_io.py` | `demo_preMaetIo` | Showing, exporting, and importing a pre-MAET (markdown, LaTeX, CSV) |
| `demo_repetition_handling.py` | `demo_repetitionHandling` | Repeated pitches: excise, prolong, or count, under interval-scale and tempo invariance |
| `demo_tempo_invariance.py` | `demo_tempoInvariance` | Tempo invariance by degree via the kernel covariance's ridge |
| `demo_swept_similarity.py` | `demo_sweptSimilarity` | `swept_similarity` in depth: translation, each match labelled by its key, a local comparison (`align='both'`), a window on a dropped or relative attribute (with translation on another), comparison in place, and a correlogram, on one melody |
| `demo_sigma_space.py` | `demo_sigmaSpace` | Soft sigma in `sameness`, `coherence`, `n_tuple_entropy` (position vs interval flag) |
| `demo_probe_tone.py` | `demo_probeTone` | Probe-tone profiles (C major against Krumhansl and Kessler, Porcupine[7] in 22-EDO), recency weighting and inharmonic spectra on a melody context, and probe continuity |
| `demo_softening_equivalences.py` | `demo_softeningEquivalences` | Softening an equivalence ([rel], [per], [exch]) by pairing attributes, with the `kernel_cov` identity |
| `demo_rhythm_tensors.py` | `demo_rhythmTensors` | Rhythmic similarity, density, and complexity; circular differencing and binding against `n_tuple_entropy` |
| `demo_dft_circular_simulate.py` | `demo_dftCircularSimulate` | Argand-DFT Monte Carlo: balance / evenness with σ, full per-coefficient distributions, projCentroid α₁ damping |
| `demo_dispatch_and_kernel_controls.py` | `demo_dispatchAndKernelControls` | Performance controls: method dispatch, nested attributes, one-pass sweeps, kernel truncation, single-precision kernel, the defaults, and the entropy estimators |

Both languages also carry the article's worked analyses as demos, in `matlab/demos/jmm/` and `python/demos/jmm/`: the windowed spectral entropy, voicing encodings, tuple size, and cadence localization of BWV 347, the motif analyses of *Acknowledgement*, the differencing, texture, and lag analyses of *Piano Phase*, and the supplied harmonic parse (retrieval, partial match, reduction, depth, and corpus marginals). Each folder's `README.md` maps the scripts to the article's analyses.

---

## 16. Known simplifications and future directions

### Flat-metric approximation

The expectation tensor framework uses a locally flat (Euclidean) metric: the Gaussian kernel is defined in terms of Euclidean distances in pitch (or time) space. In the periodic case, the domain is topologically circular (differences are wrapped modulo the period), but the metric within each period is still Euclidean — there is no curvature. For pitch-class sets with period 1200 (one octave in cents), this is well justified because the cents scale has uniform spacing in log-frequency, which closely approximates equal perceptual spacing over the range where most musical pitch perception occurs.

However, if one were to use a psychoacoustic pitch scale with non-constant spacing (e.g., mel, ERB-rate, or Bark — all available via `transformAttributes`), the Euclidean metric would introduce a systematic approximation: the effective smoothing would vary across the frequency range. The correct treatment would involve a Riemannian metric that accounts for the non-constant Jacobian of the pitch-scale mapping. In one dimension, this can be handled exactly by converting to the psychoacoustic scale before calling the toolbox (the Gaussian then has the correct width at every point). In higher dimensions (r ≥ 2), the full Riemannian treatment would require architectural changes. This is a potential future direction and is not implemented.

### Grid discretization for entropy

The differential entropy of a Gaussian mixture density has no known closed-form analytical solution, because the logarithm of a sum of Gaussians does not simplify. This is why the discrete entropy methods in `entropyMaet` and `spectralEntropy` (`method='shannon'` and `method='normalized'`) discretize the continuous density onto a finite grid and compute the entropy of the resulting probability mass function — unlike the cosine similarity, which *can* be computed analytically. The accuracy of this discretization depends on the ratio of $\sigma$ to the grid spacing.

There is no toolbox-wide default grid: `method='shannon'` and `method='normalized'` require the caller to pass an explicit `n_points_per_dim` (the right grid resolution is density- and sigma-dependent). Users can verify the discretization's accuracy by comparing results at different resolutions; with the normalized reading (`method='normalized'`), the result is independent of the arbitrary grid resolution to the extent that the grid is fine enough to capture the density's shape.

For comparisons across densities of different cardinality, spread, or support — where the grid-dependent normalizer $\log_b N$ becomes a confound — the toolbox also exposes the differential entropy $\hat h$ in closed adaptive form via `method='differential'` (adaptive nested-grid with Richardson extrapolation; no caller choice of grid) and the analytical Rényi-2 collision entropy via `method='renyi2'` (no grid at all). Both are scale-free in a sense that grid-normalized Shannon is not. Neither has a closed form for the full Shannon differential entropy of a Gaussian mixture; `'differential'` retains a discretization step internally and is therefore not exact, but it is grid-independent in the sense that the user does not pick the grid.

### Salience tensors

The expectation tensor framework as implemented returns the *expected density* of r-tuples at each point in the query space — a quantity whose scale depends on the number of elements, their weights, and $\sigma$. Many applications would benefit from a complementary **salience** reading of the same tensor: a transformation $S(\mathbf{x}) = 1 - \exp(-\mathrm{ET}(\mathbf{x})/\eta)$ that maps the density to a bounded $[0, 1)$ salience, with an interpretation as the probability that at least one $r$-tuple contributes at position $\mathbf{x}$ under an inhomogeneous Poisson point process model. This framing generalizes naturally to conditional forms (spectral masking, lateral inhibition; cf. Bulger, Milne & Dean, 2022) and can be used as a reading mode on any existing expectation tensor. A detailed specification has been drafted (`salience_specification.md`), and the feature is planned for a subsequent release.

---

## 17. References

Balzano, G. J. (1982). The pitch set as a level of description for studying musical pitch perception. In M. Clynes (Ed.), *Music, Mind, and Brain* (pp. 321–351). Plenum.

Carey, N. (2002). On coherence and sameness, and the evaluation of scale candidacy claims. *Journal of Music Theory*, 46(1/2), 1–56.

Carey, N. (2007). Coherence and sameness in well-formed and pairwise well-formed scales. *Journal of Mathematics and Music*, 1(2), 79–98.

Dean, R. T., Milne, A. J., & Bailes, F. (2019). Spectral pitch similarity is a predictor of perceived change in sound- as well as note-based music. *Music & Science*, 2, 1–14.

Duda, K., Barczentewicz, S. H., & Zieliński, T. P. (2016). Perfectly flat-top and equiripple flat-top cosine windows. *IEEE Transactions on Instrumentation and Measurement*, 65(5), 1129–1139.

Eck, D. (2006). Beat tracking using an autocorrelation phase matrix. *Proceedings of the International Computer Music Conference (ICMC)*.

Eerola, T. & Lahdelma, I. (2021). The anatomy of consonance/dissonance: Evaluating acoustic and cultural predictors across multiple datasets with chords. *Music & Science*, 4, 20592043211030471.

Eitel, M., Ruth, N., Harrison, P., Frieler, K., & Müllensiefen, D. (2024). Perception of chord sequences modeled with prediction by partial matching, voice-leading distance, and spectral pitch-class similarity: A new approach for testing individual differences in harmony perception. *Music & Science*, 7.

Harrison, P. M. C. & Pearce, M. T. (2020). Simultaneous consonance in music perception and composition. *Psychological Review*, 127(2), 216–244.

Hearne, L. M. (2020). *The Cognition of Harmonic Tonality in Microtonal Scales*. PhD thesis, Western Sydney University.

Hearne, L. M., Dean, R. T., & Milne, A. J. (2025). Acoustical and cultural explanations for contextual tonal stability. *Music Perception*, 43(3).

Homer, S., Harley, N., & Wiggins, G. (2024). Modelling of musical perception using spectral knowledge representation. *Journal of Cognition*, 7.

Huron, D. (2008). A comparison of average pitch height and interval size in major- and minor-key themes: Evidence consistent with affect-related pitch prosody. *Empirical Musicology Review*, 3, 59–63.

Krumhansl, C. L., & Kessler, E. J. (1982). Tracing the dynamic changes in perceived tonal organization in a spatial representation of musical keys. *Psychological Review*, 89(4), 334–368.

Mashinter, K. (2006). Calculating sensory dissonance: Some discrepancies arising from the models of Kameoka & Kuriyagawa, and Hutchinson & Knopoff. *Empirical Musicology Review*, 1(2), 65–84.

Milne, A. J. (2013). *A Computational Model of the Cognition of Tonality*. PhD thesis, The Open University.

Milne, A. J., Sethares, W. A., Laney, R., & Sharp, D. B. (2011). Modelling the similarity of pitch collections with expectation tensors. *Journal of Mathematics and Music*, 5(1), 1–20.

Milne, A. J., Laney, R., & Sharp, D. B. (2015). A spectral pitch class model of the probe tone data and scalic tonality. *Music Perception*, 32(4), 364–393.

Milne, A. J. & Holland, S. (2016). Empirically testing Tonnetz, voice-leading, and spectral models of perceived triadic distance. *Journal of Mathematics and Music*, 10(1), 59–85.

Milne, A. J. & Dean, R. T. (2016). Computational creation and morphing of multilevel rhythms by control of evenness. *Computer Music Journal*, 40(1), 35–53.

Milne, A. J., Laney, R., & Sharp, D. B. (2016). Testing a spectral model of tonal affinity with microtonal melodies and inharmonic spectra. *Musicae Scientiae*, 20(4), 465–494.

Milne, A. J., Bulger, D., & Herff, S. A. (2017). Exploring the space of perfectly balanced rhythms and scales. *Journal of Mathematics and Music*, 11(2–3), 101–133.

Milne, A. J. (2019). XronoMorph: Investigating paths through rhythmic space (pp. 95–113). Springer Series on Cultural Computing. Springer.

Milne, A. J. & Herff, S. A. (2020). The perceptual relevance of balance, evenness, and entropy in musical rhythms. *Cognition*, 203, 104233.

Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of rhythmic structure on tapping accuracy. *Attention, Perception, & Psychophysics*, 85, 2673–2699.

Milne, A. J., Smit, E. A., Sarvasy, H. S., & Dean, R. T. (2023). Evidence for a universal association of auditory roughness with musical stability. *PLOS ONE*, 18(9), e0291642.

Milne, A. J. (2024). Commentary on Buechele, Cooke, & Berezovsky (2024): Entropic models of scales and some extensions. *Empirical Musicology Review*, 19(2), 144–153.

Parncutt, R. (2006). Commentary on Keith Mashinter's "Calculating sensory dissonance: Some discrepancies arising from the models of Kameoka & Kuriyagawa, and Hutchinson & Knopoff." *Empirical Musicology Review*, 1(4), 201–204.

Plomp, R. & Levelt, W. J. M. (1965). Tonal consonance and critical bandwidth. *Journal of the Acoustical Society of America*, 38(4), 548–560.

Rothenberg, D. (1978). A model for pattern perception with musical applications. Part I. *Mathematical Systems Theory*, 11, 199–234.

Reljin, I. S., Reljin, B. D., & Papić, V. D. (2007). Extremely flat-top windows for harmonic analysis. *IEEE Transactions on Instrumentation and Measurement*, 56(3), 1025–1041.

Sethares, W. A. (1993). Local consonance and the relationship between timbre and scale. *Journal of the Acoustical Society of America*, 94(3), 1218–1228.

Sethares, W. A., Milne, A. J., Tiedje, S., Prechtl, A., & Plamondon, J. (2009). Spectral tools for Dynamic Tonality and audio morphing. *Computer Music Journal*, 33(2), 71–84.

Smit, E. A., Milne, A. J., Dean, R. T., & Weidemann, G. (2019). Perception of affect in unfamiliar musical chords. *PLOS ONE*, 14(6), e0218570.

Tymoczko, D. (2023). *Tonality: An Owner's Manual*. Oxford University Press.

---

## 18. Citation

If you use this toolbox in published work, please cite:

> Milne, A. J., Sethares, W. A., Laney, R., & Sharp, D. B. (2011). Modelling the similarity of pitch collections with expectation tensors. *Journal of Mathematics and Music*, 5(1), 1–20.

and the software itself using the DOI from Zenodo (see `CITATION.cff`).

For functions related to balance, evenness, and rhythmic structure, additionally cite:

> Milne, A. J., Bulger, D., & Herff, S. A. (2017). Exploring the space of perfectly balanced rhythms and scales. *Journal of Mathematics and Music*, 11(2–3), 101–133.

> Milne, A. J. & Herff, S. A. (2020). The perceptual relevance of balance, evenness, and entropy in musical rhythms. *Cognition*, 203, 104233.

For the rhythmic predictors (circApm, edges, projCentroid, meanOffset, markovS), additionally cite:

> Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of rhythmic structure on tapping accuracy. *Attention, Perception, & Psychophysics*, 85, 2673–2699.

---

## Acknowledgments

This work was supported, in part, by an Australian Research Council Discovery Early Career Researcher Award (project number DE170100353) funded by the Australian Government.
