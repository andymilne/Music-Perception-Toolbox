# Music Perception Toolbox – User Guide

Andrew J. Milne, Western Sydney University

---

## Contents

**Part I – Getting started**

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [The toolbox at a glance](#3-the-toolbox-at-a-glance)
4. [Quick start](#4-quick-start)

**Part II – From material to measures**

5. [Bringing material in](#5-bringing-material-in)
6. [The pre-MAET](#6-the-pre-maet)
7. [Preprocessing](#7-preprocessing)
8. [Densities and measures](#8-densities-and-measures)
9. [Measures on pitch and rhythm sets](#9-measures-on-pitch-and-rhythm-sets)

**Part III – Reference**

10. [API conventions](#10-api-conventions)
11. [Performance and numerical controls](#11-performance-and-numerical-controls)
12. [Function reference](#12-function-reference)
13. [Demo scripts](#13-demo-scripts)
14. [References](#14-references)
15. [Citation](#15-citation)

[Acknowledgements](#acknowledgements)

---

# Part I – Getting started

## 1. Introduction

The Music Perception Toolbox is an open-source toolbox, in MATLAB and Python, for computing perceptually and cognitively motivated measures of music: how similar two pieces of material are, from single chords to whole passages; where material recurs; how concentrated or dispersed, and so how predictable, it is; how consonant or harmonic a sound is; and how the pitches of a scale or the onsets of a rhythm are arranged around their cycle. Its input may be a score, an audio recording, material constructed to test a theoretical question, or the stimuli of an experiment.

Most of these measures are built on the expectation tensor (Milne, Sethares, Laney, & Sharp, 2011) and its multi-attribute generalization, the *multi-attribute expectation tensor* or MAET (Milne, 2026a). A MAET represents musical material as a smooth density: each event contributes a Gaussian at its values (its pitches, its onset, or its voice, say), whose width models the uncertainty with which each value is perceived. Comparing two densities gives the similarity of two passages, and the spread of one gives its entropy (§3.1). An analysis is specified by a *pre-MAET*: a table of the events, the values and weights each holds on each attribute, and how each attribute is to be read, such as pitch or pitch class, and single values or the intervals between them. The MAET follows from the pre-MAET mechanically, so the pre-MAET is where every analytical choice is made (§3.2).

A second family measures the arrangement of points on a cycle directly, without a density: balance and evenness, from the discrete Fourier transform (Milne, Bulger, & Herff, 2017), and further measures of scale and rhythm structure developed for modelling rhythmic perception and performance (Milne & Herff, 2020; Milne, Dean, & Bulger, 2023). A third group measures the consonance and harmonicity of a sound from its spectrum.

These measures have proven effective predictors of tonal fit and stability in conventional and microtonal tunings (Milne, Laney, & Sharp, 2015, 2016; Homer, Harley, & Wiggins, 2024; Hearne, Dean, & Milne, 2025), perceived change in music (Dean, Milne, & Bailes, 2019), perceived consonance and affect (Smit et al., 2019; Harrison & Pearce, 2020; Eerola & Lahdelma, 2021; Milne, Smit, Sarvasy, & Dean, 2023), individual differences in harmony perception (Eitel, Ruth, Harrison, Frieler, & Müllensiefen, 2024), and rhythmic complexity and tapping accuracy (Milne & Herff, 2020; Milne, Dean, & Bulger, 2023). They have also guided the design of music-computing interfaces (Sethares, Milne, Tiedje, Prechtl, & Plamondon, 2009; Milne & Dean, 2016; Milne, 2019). Published empirical validation is so far of the single-attribute case.

The two implementations give the same outputs to floating-point precision. Every function has full help text with examples (`help functionName` in MATLAB, `help(mpt.function_name)` in Python), and demo scripts in both languages cover the major uses, beginning with `demo_0_startHere` (§13). `CHANGELOG.md` lists the changes since the last release, `MIGRATION.md` maps code written for earlier versions onto the current names, and `ARCHITECTURE.md` describes the implementation.

**How this guide is organized.** Part I gets an analysis running. Part II follows an analysis from the material to the measures: bringing material in (§5), holding it as a pre-MAET (§6), preprocessing it (§7), and building and measuring its density (§8), with the measures that take a pitch or rhythm set directly (§9). Part III is reference: the conventions the two languages share (§10), performance controls (§11), and every function (§12).

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

MATLAB R2019b or later is required, for the `arguments` blocks most functions use; there are no other dependencies. Demo scripts are in `matlab/demos/`, and example audio in `matlab/audio/`.

### Python

The package is on PyPI:

```bash
pip install music-perception-toolbox
```

For audio file support (spectral peak extraction via `audio_peaks`):

```bash
pip install music-perception-toolbox[audio]
```

To work from a clone instead – to run the demos or the test suite, or to track the development version – install the local `python/` directory:

```bash
git clone https://github.com/andymilne/Music-Perception-Toolbox.git
pip install ./Music-Perception-Toolbox/python        # or "./Music-Perception-Toolbox/python[audio]"
```

Each release is archived with a DOI on Zenodo: <https://doi.org/10.5281/zenodo.19412254>.

The Python implementation requires Python 3.10 or later. Core dependencies (NumPy, SciPy, pandas) are installed automatically. The optional `soundfile` library is required only for `audio_peaks`.

All functions are accessible from the top-level `mpt` namespace:

```python
import mpt
s = mpt.sim_maet(...)
```

Python names are the snake_case forms of the MATLAB ones, with a few exceptions (§10.1).

---

## 3. The toolbox at a glance

This section is a map: what the toolbox computes, where material comes from, and which function does each job, with a pointer to where each is described. Names are given in MATLAB form; Python names follow §10.1 (`buildMaet` → `build_maet`).

Throughout this guide, a **Demos.** line names the demos that take a subsection's topic further, MATLAB name first and Python file second; a part number is one of the demo's own numbered sections, and *JMM* marks one of the analyses of the JMM article (Milne, 2026a) in `demos/jmm`, named alike in both languages.

### 3.1 What a MAET is

A listener does not register a musical event exactly. A tone is heard at a pitch that could plausibly have been any of a range of nearby pitches, an onset at a time that could have been any of a range of nearby times. Replacing each element of a collection by a Gaussian centred on its value, and adding these together, gives a continuous density whose value at any point is the expected number of elements perceived there. A single *kernel width*, $\sigma$, carries that perceptual uncertainty.

Two generalizations make the construction musically useful. First, the density can be over $r$-tuples of elements rather than single elements – pairs, triples, and larger groups – so that interval content, not just pitch content, is represented; $r$ is the attribute's *tuple size*. Second, the density may span several *attributes* – pitch, onset time, duration, a categorical voice label – each with its own tuple size, kernel width, and flags (below), whose values are tied together in *events* (a note ties its pitch to its onset and duration), and the density is over all of them jointly. That is the MAET.

Each attribute is independently *absolute* or *relative* (invariant to transposition of the whole tuple), *periodic* or not (pitch classes and metrical positions wrap; pitches and absolute times do not), and *exchangeable* or *ordered* (whether the order of an event's values matters, as it does for a chord's voicing but not for its pitches). These three *flags* – `[rel]`, `[per]`, and `[exch]` – together with the tuple size, are what turn one construction into the different measures the toolbox provides. Collections of unequal size compare directly, since each is embedded as a density before any comparison is made, and no correspondence between their elements is required.

Two things are computed from a density: the *similarity* of two of them, which measures how alike two collections are, or how much of one is present in the other (by default the *cosine similarity*: the inner product of the two densities, the integral of their product, divided by the product of their norms, so that it is 1 when the densities have the same shape and 0 when they do not overlap); and the *entropy* of one, which measures how evenly its mass is spread. A density's total *mass*, the number of tuples it holds where every weight is 1, turns a count into a share. The similarity, the mass, and the Rényi-2 entropy have closed forms, so no grid or resolution parameter enters; the other entropies need one (§11.2).

**Demos.** `demo_overview` / `demo_overview.py` (part 1: a chord as a density, with r and the three flags plotted); `demo_maetPlots` / `demo_maet_plots.py` (every combination of the parameters, plotted).

### 3.2 The MAET path

An analysis built on MAETs follows one path:

```
material  →  pre-MAET  →  preprocessing  →  density (MAET)  →  measures
```

- **Material.** What is to be analysed: a score, read from a MIDI or MusicXML file with `readScore` into an attribute table (one row per note, one column per attribute); spectral peaks, extracted from audio with `audioPeaks`; or values typed in, such as a chord, scale, or rhythm given as a vector (§3.3).
- **A pre-MAET.** The material is described by one or more *attributes* – pitch, onset, duration, voice – each holding a multiset of values, with optional weights, at each of a sequence of *events*. An event is what the attributes share: a chord's pitches and its onset belong to the same event, as do a melody note's pitch and duration. These values and weights, together with each attribute's parameters (its kernel width σ, its tuple size r, and its flags), form a pre-MAET: everything needed to build a density. `packPreMaet` gathers the values, weights, and parameters into one object, and `showPreMaet` displays the pre-MAET as a table, one row per attribute and one column per event (§6).
- **Preprocessing.** Six operations reshape the events before the density is built, each taking a pre-MAET and returning one, so they chain freely (§7):
  - *Event binding* (`bindEvents`) gathers consecutive events into one nested event, giving n-grams of successive material.
  - *Event differencing* (`differenceEvents`) replaces values with the change between successive events – intervals rather than pitches, inter-onset intervals rather than times.
  - *Attribute rescaling* (`transformAttributes`) maps values through a transform or a change of scale, which sets the scale on which σ is measured.
  - *Attribute translation* (`translateAttributes`) shifts an attribute's values by an offset, which is what a transposing or sliding comparison steps through.
  - *Event weighting* (`weightEvents`) multiplies a window or profile into the per-event weights, restricting the density to a local region or weighting events by salience.
  - *Spectral enrichment* (`addSpectra`) replaces each notated pitch with the partials of its sounded spectrum.
- **A density, and its measures.** `buildMaet` forms the density; `simMaet`, `evalMaet`, `entropyMaet`, and `massMaet` compare, evaluate, and measure it, and also take a pre-MAET, or for a single multiset the bare values, and build the density themselves. `sweptSimilarity`, `sweptEntropy`, and `sweptMass` package the standard sliding compositions: a query translated along a context, a window stepped through it, or both (§8).

**Demos.** `demo_overview` / `demo_overview.py` (part 2: the path once through, with a motif search); `demo_scoreWorkflow` / `demo_score_workflow.py` (the path from a score file to a result).

### 3.3 Where material comes from

Material enters the toolbox in one of four ways.

- **A score.** `readScore` reads a MIDI or MusicXML file into an *attribute table*, one row per note and one column per attribute (onset, duration, pitch, velocity, part, and so on). `gridAttrTable` samples the table on a regular time grid, and `preMaetFromAttrTable` turns it into a pre-MAET (§5.1–§5.3).
- **Audio.** `audioPeaks` extracts the spectral peaks of a recording, frequencies and amplitudes, which the spectral and consonance measures take directly (§5.4).
- **Material made by hand to answer a theoretical question.** A chord, scale, tuning, or rhythm is a vector of values – pitches in cents or semitones, positions in a cycle – and a set of them is the rows of a matrix. Most functions take these directly, and `packPreMaet` gathers several attributes into a pre-MAET (§5.5).
- **An experiment.** The stimuli of a study are a table of trials, one row per trial, and the batched forms of the measures compute a feature for every row, computing equivalent rows once (§5.5).

`transformAttributes` converts between pitch and frequency scales (Hz, MIDI, cents, octaves, mel, Bark, ERB-rate, Greenwood) and applies logarithmic, power, affine, and user-supplied transforms, at any of these entry points.

**Demos.** `demo_scoreWorkflow` / `demo_score_workflow.py` (a score); `demo_audioAnalysis` / `demo_audio_analysis.py` (audio); `demo_probeTone` / `demo_probe_tone.py` (chords and scales made by hand); `demo_rhythmTensors` / `demo_rhythm_tensors.py` (rhythms); `demo_batchProcessing` / `demo_batch_processing.py` (a table of experimental trials).

### 3.4 The MAET functions

Here is a small pre-MAET: four events, the first a C major triad at beat 0 and then the melody notes D, E, and F, with two attributes, pitch (in MIDI numbers) and onset (in beats). The pitches here are integers, but any value may be used, so a microtonal pitch is simply a non-integer such as 60.5. The pitch attribute is read as pitch class because its specs make it periodic (`[per] = 1`) with period 12 (`P = 12`), so values an octave apart are the same point.

```matlab
pAttr = {{[60 64 67], 62, 64, 65}, [0 1 1.5 2]};
specs = flatSpecs(pAttr, 'names', {'pitch', 'onset'}, 'sigma', [0.5 0.25], ...
                  'per', [true false], 'period', [12 0]);
pm = packPreMaet(pAttr, [], specs);
showPreMaet(pm)
```

```python
p_attr = [[[60, 64, 67], 62, 64, 65], [0, 1, 1.5, 2]]
specs = mpt.flat_specs(p_attr, names=['pitch', 'onset'], sigma=[0.5, 0.25],
                       per=[True, False], period=[12.0, 0.0])
pm = mpt.pack_pre_maet(p_attr, None, specs)
mpt.show_pre_maet(pm)
```

```
| attribute                                               |    n = 1     | n = 2 | n = 3 | n = 4 |
|:--------------------------------------------------------|:------------:|:-----:|:-----:|:-----:|
| pitch: sigma = 0.5, r = 1, [rel] = 0, [per] = 1, P = 12 | {60, 64, 67} |  62   |  64   |  65   |
| onset: sigma = 0.25, r = 1, [rel], [per] = 0            |      0       |   1   |  1.5  |   2   |
```

Each attribute is a row, headed by its parameters (`[rel], [per] = 0` means both flags are 0), and each event is a column; braces mark an unordered multiset, here the chord's three pitches. Each attribute's values are given event by event, as a cell (MATLAB) or list (Python) with one entry per event, and an attribute with one value per event may be a plain vector. The same pre-MAET can be read from a CSV file laid out like the table (§6.4), or made from a score (§5.3); §6 describes pre-MAETs in full.

**The pre-MAET** (§6)

| Function | What it does |
|:---|:---|
| `preMaetFromAttrTable` | Build a pre-MAET from an attribute table (§5.3) |
| `packPreMaet` / `unpackPreMaet` | Hold a pre-MAET's values, weights, and specs in one object, and split them again |
| `flatSpecs` | Make the per-attribute specifications for attributes without level structure |
| `showPreMaet` | Print a pre-MAET as a table (markdown, LaTeX, or CSV) |
| `readPreMaet` / `writePreMaet` | Read and write a pre-MAET as CSV |
| `kernelCov` | Build a matrix-valued kernel covariance for an ordered attribute (§8.1) |
| `simplexVertices` | Coordinates for a categorical attribute whose levels are all equally different |

**Preprocessing** (§7) – the six operations above, and three functions that reorganize a pre-MAET without changing its values: `selectPreMaet` keeps a selection of its attributes and events, and `bindAttributes` and `separateAttributes` join several attributes into one read as a tuple and split one again (§12.3).

**Densities and measures** (§8)

| Function | What it does |
|:---|:---|
| `buildMaet` | Build the density, the MAET |
| `evalMaet` | The density at query points |
| `plotMaet` | Plot a density of one, two, or three dimensions |
| `simMaet` | Cosine similarity of two densities; with spectral enrichment, spectral pitch (class) similarity (SPCS) |
| `entropyMaet` | Entropy of a density, by four estimators |
| `massMaet` | Total mass of a density: its number of tuples, where every weight is 1 |
| `sweptSimilarity`, `sweptEntropy`, `sweptMass` | Similarity, entropy, or total mass at each of a list of values on an attribute: a query translated along a context, a window stepped through it, or both |
| `sweepSimMaet` | Similarity of one density against translated copies of another, in one pass |
| `maetCentres` | The points at which a density places its kernels |

**Demos.** `demo_preMaetIo` / `demo_pre_maet_io.py` (a pre-MAET shown, written, and read back).

### 3.5 Measures outside the MAET framework

Not every function belongs to the MAET framework. The measures below take a pitch or rhythm set, or a spectrum, directly. Some of them use expectation tensors internally; several of the circular and structural measures do not use them at all. They are described in §9.

| Family | Functions | What they measure |
|:---|:---|:---|
| Consonance and harmonicity (§9.1) | `spectralEntropy`, `templateHarmonicity`, `tensorHarmonicity`, `roughness`, `virtualPitches` | How consonant or harmonic a chord or spectrum is, and the virtual pitches it evokes |
| Balance and evenness (§9.2) | `balanceCircular`, `evennessCircular`, `dftCircular`, `dftCircularSimulate` | How the points of a scale or rhythm are distributed around the cycle, from its discrete Fourier transform |
| Scale and rhythm structure (§9.2) | `coherence`, `sameness`, `nTupleEntropy`, `circApm`, `edges`, `projCentroid`, `meanOffset`, `markovS` | Structural features of points on a cycle, per collection or per position |
| Sequences (§9.3) | `continuity` | The recent trend of a sequence leading up to a point |

Three further functions support the others: `explainDispatch` reports how a call will be computed and why, `mptDefaults` inspects and sets the toolbox-wide defaults, and `estimateCompTime` estimates how long a call will take (§11, §12.9).

**Demos.** `demo_overview` / `demo_overview.py` (parts 3–5: consonance, balance and evenness, and scale structure); `demo_triadConsonance` / `demo_triad_consonance.py`; `demo_dftCircularSimulate` / `demo_dft_circular_simulate.py`; `demo_sigmaSpace` / `demo_sigma_space.py`.

---

## 4. Quick start

Short examples in both languages, each a complete call.

### Spectral pitch class similarity of two chords

*Spectral pitch class similarity* (SPCS) compares two chords by their spectra: each pitch is given its harmonic partials, the partials are taken as pitch classes (modulo the octave), and the cosine similarity of the two resulting densities is computed (§3.1, §8.3). Here the kernel width is σ = 10 cents, and the density is over single pitch classes (r = 1), absolute, and periodic with a period of 1200 cents.

**MATLAB:**
```matlab
% Define two weighted pitch multisets (in cents)
major = [0, 400, 700];       % 12-EDO major triad
minor = [0, 300, 700];       % 12-EDO minor triad

% Add harmonic spectra (12 partials, 1/n rolloff)
[maj_p, maj_w] = addSpectra(major, [], 'harmonic', 12, 'powerlaw', 1);
[min_p, min_w] = addSpectra(minor, [], 'harmonic', 12, 'powerlaw', 1);

% Compute SPCS (absolute, periodic, r = 1, sigma = 10)
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

A density can also be built once with `buildMaet` and passed in place of the raw values. Within one call this saves nothing, but the density outlives the call, for when the same reference is compared again later, or the density itself is wanted:

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

### Plotting a chord's density

`plotMaet` plots a density. The spectral pitch-class density of a major triad is one-dimensional, a line over the octave:

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

Densities of two and three dimensions, and the three ways of plotting them, are described in §8.6.

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
pm = preMaetFromAttrTable(t, 'specs', { ...
        struct('column', 'pitch', 'name', 'pitchClass', 'sigma', 0.5, ...
               'r', 1, 'exch', true, 'per', true, 'period', 12), ...
        struct('column', 'onset', 'sigma', 0.25)}, 'time', 'beats');
showPreMaet(pm, 'maxEvents', 6);
H = sweptEntropy(pm, 'sweep', 2, 'window', {2, {'rect', 'width', 4}}, ...
                 'drop', 2, 'method', 'renyi2');
```

**Python:**
```python
t = mpt.read_score('demos/jmm/data/bwv347.musicxml')     # from the python folder
pm = mpt.pre_maet_from_attr_table(t, specs=[
    {'column': 'pitch', 'name': 'pitchClass', 'sigma': 0.5, 'r': 1,
     'exch': True, 'per': True, 'period': 12},
    {'column': 'onset', 'sigma': 0.25}], time='beats')
mpt.show_pre_maet(pm, max_events=6)
H = mpt.swept_entropy(pm, sweep=1,
                      window={1: {'shape': 'rect', 'width': 4.0}}, drop=1,
                      method='renyi2')
```

Chords are bound into one event by default, so the pitch-class attribute holds the notes that start together, and σ is in semitones because the pitch column is in MIDI numbers. §5 covers scores, and §8.7 the swept functions.

**Demos.** `demo_scoreWorkflow` / `demo_score_workflow.py` (the path end to end); `demo_jmm_1_1_entropy` (JMM: windowed spectral entropy of the same chorale); `demo_jmm_3_1_texture` (JMM: local entropy tracking a phase process).

### Probe-tone fit to a context

In the probe-tone paradigm a listener hears a context followed by a probe, and rates how well the probe fits. The simplest model of the fit is the SPCS of the probe and the pooled spectrum of the context. Here the context's events are weighted by recency, later events being more salient in memory; `demo_probeTone` (part 3) does the same inside a pre-MAET with `weightEvents`.

The pitches are converted to cents because `addSpectra` works in cents by default, as the published spectral models do ($\sigma = 10$ cents, period 1200). To work in MIDI numbers, pass `'units', 12` to `addSpectra` and give $\sigma$ and the period in semitones (0.1 and 12).

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

This treats the context as an unordered multiset of weighted pitches. For fit as a function of time, carry time as a second attribute and step a window along it (§8.7), as `demo_overview` (part 2d) does with two triads.

**Demos.** `demo_probeTone` / `demo_probe_tone.py` (the published profiles, recency weighting, inharmonic spectra, and continuity).

### Finding where a pattern occurs within a melody

To find *where* a pattern occurs in a sequence, carry time as a second attribute: `sweptSimilarity` translates the query along time by each of a list of *sweep values* and compares it with the whole melody at each, giving a similarity profile (§8.7). The sweep values are by default the offsets added to the query as written; since the query and the melody are both written from time 0, a peak at $s$ means that the pattern occurs starting at $s$.

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
sigma = [0.1 0.2]; r = [1 1]; rel = [false false];
per = [true false]; period = [12 0];

% Translate the query along time (attribute 2) and compare it with the
% whole melody at each offset. Time is compared, so the query's internal
% timing must match. mu{2} holds the offsets.
[S, mu] = sweptSimilarity({melody_p, melody_t}, [], {query_p, query_t}, [], ...
                          sigma, r, rel, per, period, 'sweep', 2);
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

Naming only the swept attribute lets the toolbox choose the sweep values: every placement at which the query overlaps the melody (here offsets from −1 to 6), finely enough to resolve the peaks. A list can be given instead, `'sweep', {2, s}` / `sweep={1: s}`. The melody holds E–G at times 2–3 and 5–6, so the profile peaks at $s = 2$ and $s = 5$, with graded similarity between, as the kernel width on time allows. `demo_overview` (part 2c) repeats the search on the melody's intervals after `differenceEvents`, which finds the transposed statement too.

**Demos.** `demo_sweptSimilarity` / `demo_swept_similarity.py` (every setting, on one melody); `demo_helixBlend` / `demo_helix_blend.py` (a time-windowed sweep of a motif across registers); `demo_jmm_2_3_spectral` (JMM: a motif located in time and key); `demo_jmm_1_3_cadence_nesting` (JMM: cadences located with nested prototypes).

---

# Part II – From material to measures

An analysis built on MAETs follows the path of §3.2: material is brought in (§5) and held as a pre-MAET (§6), which may be preprocessed (§7) before its density is built and measured (§8). §9 covers the measures that take a pitch or rhythm set directly, without a density.

## 5. Bringing material in

This section covers the first stage of an analysis: reading a score and sampling it, turning the result into a pre-MAET, extracting peaks from audio, and working with material made by hand or taken from an experiment.

### 5.1 Reading a score

`readScore` parses a Standard MIDI File (format 0 or 1) or a MusicXML score (`.musicxml`, `.xml`, or `.mxl`) into an *attribute table*: a MATLAB `table` or pandas `DataFrame` with one row per sounding note, giving its onset and duration in quarter-note beats and in seconds, its MIDI pitch, velocity, part, and bar, and whatever else the source carries, such as a MIDI channel or a MusicXML voice (§12.1). A tie is merged into one note, a grace note is skipped, and a rest or unpitched note is not a note. Both parsers are self-contained, with no dependencies, and read a file to the same table in either language.

**Demos.** `demo_scoreWorkflow` / `demo_score_workflow.py` (parts 1–2: reading and looking at the table); `demo_jmm_1_1_entropy` (JMM: a chorale read and sampled).

### 5.2 Sampling a score on a grid

`gridAttrTable` samples an attribute table on a regular grid of time points, one event per point, a held note occupying every slice it sounds in and a note shorter than the step still occupying one. A grid gives the event index a uniform meaning in time, which a window measured in beats, or a comparison of what sounds at each moment, needs. `'weights'` says what a slice takes from a note overlapping it: `'coverage'` (the default) the fraction of the slice the note fills; `'presence'` the note's full weight in every slice it appears in; and `'item'` the note's weight spread over its slices, so that it counts once in all. A slice with nothing sounding is kept as an empty event, so the grid stays uniform. A gridded table can be regridded at a coarser step, and `ungridAttrTable` returns the table it was made from.

**Demos.** `demo_scoreGrid` / `demo_score_grid.py` (the step, the weighting, and empty slices); `demo_scoreWorkflow` / `demo_score_workflow.py` (part 3).

### 5.3 From an attribute table to a pre-MAET

`preMaetFromAttrTable` turns an attribute table into a pre-MAET. Each attribute names the column it reads together with its own parameters, so one pitch column listed twice gives two attributes, pitch class and pitch height, say. The conversion fills in what follows from the data and asks for the rest: σ always, and `r` and `exch` wherever an attribute holds several values at an event, since a score fixes the values but not how tolerant a match should be, nor how many of an event's values a tuple takes. Notes that start together are bound into one event by default (`'chords', 'bind'`), so that an `r = 2` reading of pitch gives the chords' dyads; `'chords', 'separate'` makes every note its own event. `'pitch'` sets the scale the pitch column is read on (`'midi'` by default), and `'time'` whether onsets and durations are in `'seconds'` or `'beats'`; these are arguments of the conversion, not the columns of the same names. Rows are selected with the host language's own indexing before converting, and `selectPreMaet` selects attributes and events afterwards.

**Encoding a categorical column.** `'roles'` says how a categorical column, such as the part, reaches the pre-MAET:

- `'separateAttributes'` – one attribute per level;
- `'orderedMultiset'` – one attribute with one position per level, read in order;
- `'simplex'` – the level as the coordinates of a vertex of a regular simplex (`simplexVertices`), carried as its own attribute, so that every level is equally different from every other; its σ, relative to the unit edge, sets how distinct the levels are, small keeping them apart and large blurring them together;
- `'drop'` – not encoded, as for a column with no entry (unrelated to the `'drop'` argument of the swept functions).

The first two are *structural*: they gather an event's notes into one event holding one position per level, so they need `'chords', 'bind'`, and an event that does not hold exactly one note per level is dropped, with a warning giving the count. Only one column may be structural. `'simplex'` gives each note its own event, and so needs `'chords', 'separate'`, unless a structural role is also given. A role may be given as a struct / mapping carrying `role` together with the new attribute's parameters, such as the simplex's σ. The three encodings the JMM article compares on BWV 347 are `'orderedMultiset'` on the part (voice-aware), `'simplex'` (simplex-voice), and no role with one event per note (voice-agnostic).

Without a role, `'chords', 'bind'` makes each attribute an unordered multiset at every event, so two attributes describing the same notes (pitch class and pitch height, say) are paired every value with every value, the soprano's pitch class with the bass's height included. To keep a note's attributes together, use one event per note.

`'groupBy'` names a column whose equal values in consecutive rows make one event, generalizing the binding of chords. The default is the grid position on a gridded table, and otherwise the onset, within `'chordTolerance'`.

**Demos.** `demo_scoreWorkflow` / `demo_score_workflow.py` (parts 4–5); `demo_scoreCategoricals` / `demo_score_categoricals.py` (a categorical column encoded three ways); `demo_preMaetIo` / `demo_pre_maet_io.py` (part 8: a pre-MAET from a score).

### 5.4 Audio

`audioPeaks` reads an audio file and returns the frequencies (in Hz) and normalized amplitudes of the peaks of its spectrum. The peaks are the sounded spectrum, so they go to the spectral and consonance measures without `addSpectra`: convert them to cents with `transformAttributes` for the pitch-based measures (§4), and keep them in Hz for `roughness`.

**Demos.** `demo_audioAnalysis` / `demo_audio_analysis.py` (two passes of peak extraction, then the features); `demo_virtualPitches` / `demo_virtual_pitches.py`.

### 5.5 Material made by hand, and experimental stimuli

A theoretical question usually starts from a vector: a chord or scale in cents or semitones, the steps of a tuning, or a rhythm as positions in a cycle. The single-multiset forms take it directly, with its weights (`[]` / `None` for all ones) and parameters: `simMaet(p1, w1, p2, w2, sigma, r, rel, per, period)` compares two chords, and `balanceCircular(p, [], period)` measures a rhythm. For several attributes per event, such as pitch and time, `packPreMaet` gathers their values, weights, and specs into a pre-MAET (§6.1), and `flatSpecs` makes the specs.

A set of chords, scales, or rhythms is the rows of a matrix, padded with `NaN` where they differ in size, and the batched forms return one value per row (§10.6). The stimuli of an experiment, one row per trial, go straight in, and rows that are equivalent under the measure's symmetries, such as transpositions or reorderings, are computed once (§10.7).

**Demos.** `demo_probeTone` / `demo_probe_tone.py`; `demo_edoApprox` / `demo_edo_approx.py`; `demo_genChainPcs` / `demo_gen_chain_pcs.py`; `demo_rhythmTensors` / `demo_rhythm_tensors.py`; `demo_batchProcessing` / `demo_batch_processing.py` (a table of trials, and a cell of pre-MAETs).

---

## 6. The pre-MAET

A pre-MAET is everything a density is built from: the values of each attribute at each event, their weights, and each attribute's specification (Milne, 2026a, Def. 2.6). The toolbox provides one object to hold them, one table to view them, and a CSV form to write and read them. Every preprocessing operation (§7) takes a pre-MAET and returns one, and every measure (§8) takes one wherever it takes a density.

### 6.1 Creating and modifying a pre-MAET

`packPreMaet` creates a pre-MAET from its three parts: the values, `pAttr`; the weights, `wAttr`; and the specs, `specs`. The result is a MATLAB struct with those fields, or a Python dict with the keys `p_attr`, `w_attr`, and `specs`. It is a plain struct or dict, so any part may be read or replaced directly.

`pAttr` holds one entry per attribute, in either of two forms. *Per event*, an attribute is a cell (MATLAB) or list (Python) with one entry per event, holding that event's values: a scalar, a vector, or empty. So `{[60 64 67], 62, [], 65}` / `[[60, 64, 67], 62, [], 65]` is a triad, a single note, a silence, and another note. *As a matrix*, it is $K_a \times N$, one column per event, each event's values at the top of its column and `NaN` below them where it holds fewer than the widest. On an ordered attribute a value's position within its event carries its identity (the first position holding the soprano's pitch, say), so a `NaN` may also mark a position with no value before or between values: `{[60 64 67], [62 NaN 67], [NaN 65]}` gives the second event no value in position 2 and the third none in position 1. A position with no value is read as a value of weight 0, so every tuple through it vanishes: an event holding fewer than r values contributes nothing, and `buildMaet` warns that it does, while an attribute with fewer than r positions, values and `NaN`s together, is an error. The per-event form is the one to write by hand; the toolbox stores the matrix, and every operation returns it. An attribute with one value per event is simply a vector. In Python a matrix is always a NumPy array, and a list is always read per event.

Weights may be given per event in the same way, each entry a scalar that weights all of the event's values or a vector with one weight per value: `{[1 0.5 0.5], 1, [], 1}` gives the triad's root twice the weight of its other notes. The matrix forms of §10.4 are accepted too.

Every function that takes a pre-MAET takes it whole or in its parts, and every preprocessing function returns it whole, so a composition needs no threading of the specs:

```matlab
pm  = packPreMaet(pAttr, wAttr, specs);
pmD = differenceEvents(pm, [1 0]);                             % pre-MAET in, pre-MAET out
pmD = differenceEvents(pAttr, wAttr, [1 0], 'specs', specs);   % the same
pm2 = differenceEvents(bindEvents(pm, [2 2]), [1 1]);          % composed
```

```python
pm  = mpt.pack_pre_maet(p_attr, w_attr, specs)
pmD = mpt.difference_events(pm, [1, 0])
pmD = mpt.difference_events(p_attr, w_attr, [1, 0], specs=specs)
pm2 = mpt.difference_events(mpt.bind_events(pm, [2, 2]), [1, 1])
```

`unpackPreMaet` splits a pre-MAET back into its parts. The measures take a pre-MAET wherever they take a density and build it themselves, so `simMaet(pmX, pmY)` compares two, and a cell or list of pre-MAETs stands for a list of densities: `simMaet(pmRef, {pm1, pm2, pm3})` builds and compares each in turn.

**Variants and overrides.** Replacing one part makes a variant: `packPreMaet(pm, [], specs2)`. To vary a parameter without changing the pre-MAET, as in a sweep, give it to `buildMaet` instead. All six per-attribute parameters, `sigma`, `per`, `period`, `r`, `rel`, and `exch`, may be given there, and take precedence over the specs. A cell (MATLAB) or list (Python) with one entry per attribute, whose empty entries keep the spec's value, overrides selected attributes only, so the call cannot fall out of step with the attributes:

```matlab
for s = [0.25 0.5 1]
    dens = buildMaet(pm, 'sigma', {[], s, []});   % attribute 2 only
end
```

```python
for s in [0.25, 0.5, 1.0]:
    dens = mpt.build_maet(pm, sigma=[None, s, None])   # attribute 2 only
```

`NaN` cannot mean "keep", since it already means NA (§6.3). On a nested attribute `r`, `rel`, and `exch` are per-level vectors, which a call cannot override; change the spec instead.

`sweptSimilarity`, `sweptEntropy`, and `sweptMass` take pre-MAETs on the same terms, with the same overrides. In these functions, the two pre-MAETs of a comparison must agree on `r`, `[rel]`, `[exch]`, and nesting; they may differ in `sigma`, `per`, and `period`, which are taken from the context. They may differ in the number of values per event, so a two-note query can be compared with a seven-note context.

**Demos.** `demo_preMaetIo` / `demo_pre_maet_io.py`; `demo_batchProcessing` / `demo_batch_processing.py` (Workflow 3, a cell of pre-MAETs).

### 6.2 Viewing a pre-MAET

`showPreMaet` prints a pre-MAET as a table in the layout of Milne (2026a): one row per attribute, headed by the parameters that determine its density (σ, the tuple size r, the `[rel]` and `[per]` flags, and the period where it is periodic), and one column per event. Since the density follows mechanically from the pre-MAET, the table is a complete statement of what a later `simMaet`, `entropyMaet`, or `evalMaet` call will compute.

A cell is in braces where the attribute is unordered (`[exch] = 1`) and in parentheses where it is ordered. A nested attribute is bracketed level by level, the outermost level's brackets outermost, so an ordered run of unordered chords reads `({62, 65, 69, 72}, {55, 59, 62, 65}, {60, 64, 67})`; a single element is written bare at the top level but keeps its brackets inside a nest. On an ordered attribute a position with no value before an event's last value is written as a blank, `(62, _, 67)`, so that every value keeps its position; on an unordered attribute, where position carries nothing, the values are written without gaps. Non-uniform weights appear as parenthesized superscripts, `60^(0.6)`, and a kernel covariance (§8.1) is named by its shape, `sigma = 3x3 covariance`.

It takes a pre-MAET, whole or in its parts with the kernel parameters alongside, or a density built by `buildMaet`:

```matlab
p = {[69 69 69 71 67 66 64], [1 2 3 4 5 6 7]};
w = {[1 0.5 0.75 0.5 1 0.5 0.75], [1 0.5 0.75 0.5 1 0.5 0.75]};
showPreMaet(p, w, [], 'names', {'pitch', 'time'}, 'sigma', [0.5 0.25], ...
            'per', [true false], 'period', [12 0]);
```

```
| attribute                                               | n = 1  |  n = 2   |   n = 3   |  n = 4   | n = 5  |  n = 6   |   n = 7   |
|:--------------------------------------------------------|:------:|:--------:|:---------:|:--------:|:------:|:--------:|:---------:|
| pitch: sigma = 0.5, r = 1, [rel] = 0, [per] = 1, P = 12 | 69^(1) | 69^(0.5) | 69^(0.75) | 71^(0.5) | 67^(1) | 66^(0.5) | 64^(0.75) |
| time: sigma = 0.25, r = 1, [rel], [per] = 0             | 1^(1)  | 2^(0.5)  | 3^(0.75)  | 4^(0.5)  | 5^(1)  | 6^(0.5)  | 7^(0.75)  |
```

When the specs carry `sigma`, `per`, and `period` as well as `r`, `rel`, and `exch`, the pre-MAET is complete, and `showPreMaet` and `buildMaet` need nothing further (§6.3).

`maxEvents` elides the middle columns of a long passage and `maxElements` the tail of a large multiset, as the article's tables do; `[]` / `None` shows everything. The default output is plain-ASCII markdown, so that its columns align identically in the two languages; `'format', 'latex'` gives a `booktabs` tabular in the article's markup, with optional `caption` and `label`, so that a manuscript's table can be generated by the code that runs the analysis.

**Demos.** `demo_preMaetIo` / `demo_pre_maet_io.py` (part 1: markdown and LaTeX); `demo_preprocessing` / `demo_preprocessing.py` (a pre-MAET shown after each operation).

### 6.3 Where the kernel parameters live

The `specs` hold every per-attribute parameter: the level-structured `r`, `rel`, and `exch` (per-level vectors on a nested attribute, alongside its `tags`), and the scalar `sigma`, `per`, and `period`. The three scalars are **optional in the pre-MAET and required at the density**. `buildMaet` resolves each per attribute:

- a `sigma`, `per`, or `period` given in the call takes precedence, silently, so that sweeping a width while the specs hold a baseline is an ordinary idiom;
- otherwise the spec supplies it;
- a value missing from both is an error naming the attribute, except `period`, which defaults to 0.

`NA` (`NaN`) is a third state, distinct from absent. A preprocessing step that cannot carry a parameter forward writes NA rather than a stale or invented value; `showPreMaet` prints `sigma = NA`, and `buildMaet` refuses it until the parameter is given in the call.

The preprocessing operations of §7 transform a pre-MAET before its density is built: gathering consecutive events into one (`bindEvents`), replacing values with the differences between successive events, such as pitches with melodic intervals (`differenceEvents`), changing an attribute's scale, such as frequency in Hz to cents (`transformAttributes`), shifting its values (`translateAttributes`), weighting events by a window (`weightEvents`), or replacing each pitch with its partials (`addSpectra`). Some of these change what a width or period means, so each carries the parameters forward as follows:

| operation | `sigma` | `period` |
|:--|:--|:--|
| `bindEvents`, `translateAttributes`, `weightEvents`, `addSpectra` | unchanged | unchanged |
| `differenceEvents`, order $k$ | $\times \sqrt{\binom{2k}{k}}$ | unchanged |
| `transformAttributes`, affine or within `midi`/`cents`/`octave` | $\times$ the scale factor | $\times$ the scale factor |
| `transformAttributes`, anything non-linear | NA | NA |

An affine map multiplies every distance by its scale factor, $|a|$ for $x \mapsto ax + b$, so the width is multiplied by the same factor to describe the same uncertainty in the new units: `midi` to `cents` multiplies by 100, so σ = 0.5 semitones becomes σ = 50 cents. A $k$-th difference of values with independent errors of width σ has width $\sigma\sqrt{\binom{2k}{k}}$: $\sqrt{2}\,\sigma$ for a first difference, $\sqrt{6}\,\sigma$ for a second. Independence is a modelling assumption, so `differenceEvents` announces the scaling. A kernel covariance, being in squared units, takes the factor itself. Under a non-linear map, NA does not mean that a width would be meaningless (a σ on a log axis expresses a ratio), but that no single value carries over, so the analyst supplies it.

Unless a spec gives them, `preMaetFromAttrTable` sets `per = 0` and `period = 0`, octave equivalence being an equivalence the analyst imposes, and leaves `sigma` unset, since a score implies no width.

**Demos.** `demo_preMaetIo` / `demo_pre_maet_io.py` (parts 3, 4, and 6: nothing supplied, overrides, and NA).

### 6.4 Reading and writing a pre-MAET as CSV

`readPreMaet` and `writePreMaet` move a pre-MAET between the toolbox and a CSV file that a spreadsheet can edit. The cells use the notation of `showPreMaet`, and a pre-MAET survives a round trip byte for byte; `showPreMaet(..., 'format', 'csv')` gives the same text without writing a file.

The header is fixed, `name, sigma, r, rel, per, P, exch`, followed by one column per event, whose headings are free text:

```
name,sigma,r,rel,per,P,exch,n = 1,n = 2,n = 3
pitch,0.5,2,0,1,12,1,"{60, 64, 67}","{62, 65, 69}","{60, 64, 67}"
onset,0.25,1,0,0,,1,0,1,2
```

This reads straight into a density, `buildMaet(readPreMaet(file))`, with nothing further supplied. When writing one by hand:

- `r`, `rel`, and `exch` take a parenthesized tuple on a nested attribute, innermost level first.
- Tags are never written: a nested attribute's levels are read from the brackets of its cells, so `"({60, 64, 67}, {62, 67, 71})"` at `r = "(1, 2)"` is enough.
- Events of different sizes are padded with `NaN` on reading and written back ragged. On an ordered attribute a blank `_` marks a position with no value before an event's last value, so `(_, 65)` puts 65 in the second position.
- A weight is written `60^(0.6)`; an attribute whose weights are all 1 is written bare.
- An empty parameter cell is absent; `NA` is NA.
- A kernel covariance is written as the flag and three widths that generate it, `cov(differenced=1, sd_value=0.2, sd_interval=0.3, sd_shift=0.5)`, the row's `r` giving its size; a covariance that `kernelCov` cannot generate is refused.
- A file elides nothing, whatever `maxEvents` says.

**Demos.** `demo_preMaetIo` / `demo_pre_maet_io.py` (part 2: the round trip).

---

## 7. Preprocessing

Before a pre-MAET is built into a density, it can be transformed to pose a particular musical question: comparing intervals rather than pitches, matching patterns of several notes, ignoring tempo, isolating a passage, or hearing each pitch as its spectrum. Following Milne (2026a), there are six such operations. Each takes a pre-MAET, whole or as its three parts (§6.1), and returns one, so they chain directly into one another and into any function that takes a pre-MAET: `buildMaet`, the measures (`simMaet`, `entropyMaet`, `massMaet`), and the swept functions.

| Operation | Function | What it does | Typical purpose |
|:---|:---|:---|:---|
| Event binding (§7.1) | `bindEvents` | Gathers consecutive events into one | Patterns of several notes: n-grams, motifs, progressions |
| Event differencing (§7.2) | `differenceEvents` | Replaces values with the differences between successive events | Melodic intervals, inter-onset intervals |
| Attribute rescaling (§7.3) | `transformAttributes` | Maps values through a transform or a change of scale | Frequency to cents; log inter-onset intervals, for tempo invariance |
| Attribute translation (§7.4) | `translateAttributes` | Shifts values by an offset | Transposition, time shift; the basis of translation sweeps |
| Event weighting (§7.5) | `weightEvents` | Multiplies a window or profile into the weights | Local analysis; recency and other salience profiles |
| Spectral enrichment (§7.6) | `addSpectra` | Replaces each pitch with its partials | Spectral pitch similarity, harmonicity |

Binding and differencing work *across* events: each output event combines an event with its neighbours, so the number of events changes and their order matters. The other four work on each event separately, keeping the events' number and order. Every operation acts per attribute, a scalar order, offset, or transform applying to every attribute. Three further functions reorganize a pre-MAET without changing its values: `selectPreMaet` keeps a selection of its attributes and events, and `bindAttributes` and `separateAttributes` join several attributes into one read as a tuple and split one again (§12.3).

The examples below use one short melody, with pitch (in MIDI numbers) and onset (in beats) as attributes:

```matlab
pAttr = {[60 62 64 65 67], [0 1 2 2.5 3]};
specs = flatSpecs(pAttr, 'names', {'pitch', 'onset'}, 'sigma', [0.5 0.1], ...
                  'per', [false false], 'period', [0 0]);
pm = packPreMaet(pAttr, [], specs);
```

```python
p_attr = [[60, 62, 64, 65, 67], [0, 1, 2, 2.5, 3]]
specs = mpt.flat_specs(p_attr, names=['pitch', 'onset'], sigma=[0.5, 0.1],
                       per=[False, False], period=[0.0, 0.0])
pm = mpt.pack_pre_maet(p_attr, None, specs)
```

Each example shows the MATLAB call, with the Python call in a comment, and the `showPreMaet` table of the result. Python attribute indices count from 0.

### 7.1 Event binding

`bindEvents` gathers each run of $L$ consecutive events into one event, so that a pattern of several notes becomes a single object that can be compared, counted, or located. Binding pitch over three events and onset over one gives three-note pitch patterns, each with the onset of its last note:

```matlab
pmB = bindEvents(pm, [3 1]);      % Python: mpt.bind_events(pm, [3, 1])
```

```
| attribute                                                 |       n = 1        |       n = 2        |       n = 3        |
|:----------------------------------------------------------|:------------------:|:------------------:|:------------------:|
| pitch: sigma = 0.5, r = (1, 3), [rel] = (0, 0), [per] = 0 | ({60}, {62}, {64}) | ({62}, {64}, {65}) | ({64}, {65}, {67}) |
| onset: sigma = 0.1, r = 1, [rel], [per] = 0               |         2          |        2.5         |         3          |
```

Each bound event is *nested*: an ordered outer level, the three positions, holding each original event's values at the inner level. The outer level is by default read whole (`'rOuter'`, $r = L$), ordered (`'exchOuter'`, false), and absolute (`'relOuter'`, false), so the density is over three-note patterns in register; making it relative (`'relOuter', true`) gives patterns invariant to transposition. A bound event's weight is the product of its events' weights, and there are $N - L + 1$ bound events, or $N$ with `'circular'`, which wraps around the end of a cyclic sequence. Unlike differencing, binding accepts an unordered attribute with several values per event, such as the notes of a chord. `'groupBy'` binds runs of events sharing a value (the notes of one bar, say) instead of a fixed number (§12.3).

**Demos.** `demo_preprocessing` / `demo_preprocessing.py` (parts 3 and 3b); `demo_rhythmTensors` / `demo_rhythm_tensors.py` (part 4: circular binding); `demo_jmm_1_3_cadence_nesting` (JMM); `demo_jmm_2_1_joint` (JMM).

### 7.2 Event differencing

`differenceEvents` replaces each attribute's values with their differences from one event to the next: pitches become melodic intervals and onsets become inter-onset intervals, so that a melody or rhythm is compared by its steps rather than its positions.

```matlab
pmD = differenceEvents(pm, [1 1]);   % Python: mpt.difference_events(pm, [1, 1])
```

```
| attribute                                        | n = 1 | n = 2 | n = 3 | n = 4 |
|:-------------------------------------------------|:-----:|:-----:|:-----:|:-----:|
| pitch: sigma = 0.707107, r = 1, [rel], [per] = 0 |   2   |   2   |   1   |   2   |
| onset: sigma = 0.141421, r = 1, [rel], [per] = 0 |   1   |   1   |  0.5  |  0.5  |
```

The orders are per attribute: 2 gives differences of differences, and 0 leaves an attribute unchanged. The first event is dropped, or with `'circular'` the differences wrap around, as suits a looped rhythm. A difference's weight is the product of its two events' weights, and σ is scaled by √2, the width of a difference of two independent values (§6.3).

Differencing is distinct from `rel = true`. Differencing compares *successive* events; a relative attribute at $r = 2$ compares *any* two values of an event (the intervals within a chord, say). They can be used together. An attribute can be differenced only if it holds one value per event or is ordered, so that a value at one event has a counterpart at the next; the voices of a chorale, for instance, are differenced as separate attributes or as one ordered attribute, not as an unordered chord.

**Demos.** `demo_preprocessing` / `demo_preprocessing.py` (part 2); `demo_rhythmTensors` / `demo_rhythm_tensors.py` (part 4: circular differencing); `demo_repetitionHandling` / `demo_repetition_handling.py`; `demo_jmm_3_2_diff` (JMM: joint differencing of pitch and time).

### 7.3 Attribute rescaling

`transformAttributes` maps an attribute's values through a transform, which sets the scale on which σ is measured: a constant σ in cents is a constant interval, and a constant σ on a logarithmic axis is a constant ratio. The transforms are `'log'`, `'power'`, and `'affine'`; conversions between the pitch scales `'hz'`, `'midi'`, `'cents'`, `'octave'`, `'mel'`, `'bark'`, `'erb'`, and `'greenwood'`; or a function. Given a bare array it converts the array, the one-line way to convert frequencies to cents: `transformAttributes(f, [], {'hz', 'cents'})`.

Its commonest musical use is tempo invariance. On a logarithmic scale a change of tempo, which multiplies every inter-onset interval by one factor, becomes a common shift. Taking the base-2 logarithm of the inter-onset intervals above:

```matlab
pmL = transformAttributes(pmD, {[], {'log', 'base', 2}});
% Python: mpt.transform_attributes(pmD, [None, ('log', {'base': 2})])
```

gives 0, 0, −1, −1; the same melody at half the tempo gives 1, 1, 0, 0, the same pattern shifted by 1. A relative attribute, or a translation sweep (§7.4), then disregards the shift, and so the tempo (`demo_tempoInvariance`).

The order of operations matters: `'log'` then differencing gives log ratios, whereas differencing then `'log'` gives the logarithms of the differences, as here. Values outside a transform's domain are refused with the remedies, the common case being a zero inter-onset interval under `'log'`; `'sign'` handles negative values (§12.3). Only `'affine'` suits a periodic attribute, and a non-linear transform leaves σ as NA, to be supplied anew (§6.3).

**Demos.** `demo_preprocessing` / `demo_preprocessing.py`; `demo_tempoInvariance` / `demo_tempo_invariance.py` (logarithmic inter-onset intervals); `demo_repetitionHandling` / `demo_repetition_handling.py` (part 4).

### 7.4 Attribute translation

`translateAttributes` shifts an attribute's values by an offset: a transposition of pitch, or a displacement in time. Transposing the melody up a perfect fourth:

```matlab
pmT = translateAttributes(pm, {5, []});   % Python: mpt.translate_attributes(pm, [5, None])
```

gives pitches 65, 67, 69, 70, and 72, the onsets unchanged. An offset may also be given per row, and on a relative attribute a uniform shift changes nothing, and warns.

One call makes one translation. Its main use is the *translation sweep*: a query translated over a range of offsets and compared with a context at each, which finds where, or at which transposition, the query occurs. `sweptSimilarity` computes every offset of a sweep in one pass (§8.7), so `translateAttributes` is needed only for a single shift within a pipeline. An offset is measured from the values as written, so for a sweep in time it is simplest to write the query and the context from a common origin, before any other preprocessing. Differencing and binding drop leading events but move no remaining value, so an offset keeps its meaning through them.

**Demos.** `demo_preprocessing` / `demo_preprocessing.py` (part 4); `demo_sweptSimilarity` / `demo_swept_similarity.py`.

### 7.5 Event weighting

`weightEvents` computes a factor for each event from its value on one attribute, the *input*, and multiplies it into the weights of another, the *target*, so that some events count more than others: those inside a window of time, say, or the most recent. A rectangular window two beats wide, centred at beat 2, on the onsets, applied to the pitches:

```matlab
pmW = weightEvents(pm, 2, 1, 2, 1, 'width', 2, 'dropInputAttr', true);
% Python: mpt.weight_events(pm, 1, 0, 2.0, 1.0, width=2.0, drop_input_attr=True)
```

```
| attribute                                   | n = 1  | n = 2  | n = 3  | n = 4  | n = 5  |
|:--------------------------------------------|:------:|:------:|:------:|:------:|:------:|
| pitch: sigma = 0.5, r = 1, [rel], [per] = 0 | 60^(0) | 62^(1) | 64^(1) | 65^(1) | 67^(0) |
```

The arguments after the pre-MAET are the input attribute, the target attribute, `alignAt` (Python `align_at`), the value of the input attribute at which the profile's reference value is placed (the centre of a symmetric window), and the shape. The shape γ runs from 0, a Gaussian, to 1, a rectangle, which is half-open, $[1, 3)$ here; the scale is given as `'sd'` or as `'width'`, the rectangle's full width. `'dropInputAttr'`, which is required, removes the input once its work is done, as here, where only pitch is to be measured. Other shapes are the exponentials (`'exponentialBefore'` aligned at the last onset gives a recency profile), the serial-position profiles anchored at the first and last events (for which `alignAt` is `NaN`, Python `None`), and any function of the displacement from `alignAt` (§12.3). Calling `weightEvents` twice on one target gives the product of two windows, such as one in time and one in register.

Windowing in the toolbox is always event weighting of this kind: the window acts on the events before the density is built, not on the finished density. So no smoothed mass leaks across the window's edge, and every measure applies to the windowed density unchanged. `sweptSimilarity`, `sweptEntropy`, and `sweptMass` step such a window through a passage (§8.7).

**Demos.** `demo_preprocessing` / `demo_preprocessing.py` (part 5); `demo_probeTone` / `demo_probe_tone.py` (part 3: recency weighting); `demo_jmm_1_1_entropy` (JMM: a window swept through a chorale).

### 7.6 Spectral enrichment

A musical pitch is not a pure tone but a spectrum of partials. `addSpectra` replaces each pitch with its partials, whose weights are the pitch's weight times their own, so that pitches are compared by their spectra: the basis of spectral pitch similarity (§8.3) and of the harmonicity measures (§9.1). Adding three harmonics, with weights falling as $1/n$, in semitones:

```matlab
pmS = addSpectra(pm, 'harmonic', 3, 'powerlaw', 1, 'attribute', 1, 'units', 12);
% Python: mpt.add_spectra(pm, 'harmonic', 3, 'powerlaw', 1, attribute=0, units=12)
```

turns the first pitch, 60, into `{60^(1), 72^(0.5), 79.0196^(0.3333)}`, and likewise for the others. Given values and weights rather than a pre-MAET, `addSpectra(p, w, ...)` returns the expanded values and weights.

The spectral modes are `'harmonic'`; `'stretched'`, with partials at $n^\beta$; `'freqlinear'`, stretched in frequency rather than log frequency; `'stiff'`, the sharpened partials of a stiff string such as a piano's; and `'custom'`, any partials and weights. The partials' weights fall as `'powerlaw'`, $1/n^\rho$ ($\rho = 1$ a sawtooth), or as `'geometric'`, $\tau^{n-1}$.

**Demos.** `demo_overview` / `demo_overview.py` (part 1a); `demo_triadConsonance` / `demo_triad_consonance.py`; `demo_jmm_2_3_spectral` (JMM).

### 7.7 Combining operations

The operations chain freely, and several common analyses are short chains:

- **n-grams of intervals:** differencing then binding gives patterns of successive intervals or inter-onset intervals. With both circular, on a cyclic rhythm or scale, their entropy is the n-tuple entropy of Milne and Dean (2016), which `nTupleEntropy` computes directly.
- **Tempo-invariant rhythm:** differencing onsets, then a logarithm, then binding with a relative outer level compares rhythms by their proportions alone (§7.3).
- **Spectral motifs:** spectral enrichment then binding gives patterns of spectra, so that a motif is matched by its sound rather than its notes (`demo_jmm_2_3_spectral`).

Operations on different attributes commute. On the same attribute their order can matter: a logarithm before differencing differs from one after (§7.3), whereas binding and differencing commute.

Two further analyses step an operation through a passage: a translation sweep (§7.4), which finds where a query occurs, and a window stepped through time (§7.5), which traces how similarity, entropy, or mass change. Both are done in one call, and for a sweep in one pass, by the swept functions of §8.7, which are the usual way to perform them.

**Demos.** `demo_preprocessing` / `demo_preprocessing.py` (the operations composed, with `selectPreMaet`, `bindAttributes`, and `separateAttributes`); `demo_rhythmTensors` / `demo_rhythm_tensors.py` (part 4: circular differencing and binding against `nTupleEntropy`); `demo_jmm_3_2_diff` (JMM).

---

## 8. Densities and measures

A density, once built, answers four questions:

| Question | Function | Section |
|:---|:---|:---|
| How much is there at a given point? | `evalMaet` | §8.2 |
| How alike are two densities, or how much of one is present in another? | `simMaet` | §8.3 |
| How spread out is it? | `entropyMaet` | §8.4 |
| How much does it hold in all? | `massMaet` | §8.5 |

`plotMaet` plots a density (§8.6), and the swept functions take these measures at each of a list of values along an attribute (§8.7). The mathematics is set out in Milne (2026a), and for a single attribute in Milne, Sethares, Laney, and Sharp (2011).

The examples of §8.1–§8.5 use the interval content of scales: their pitch classes in semitones, read two at a time and made relative and periodic, so that each density is over the intervals between every pair of the scale's notes, in either direction.

### 8.1 Building a density

`buildMaet` turns a pre-MAET into a density, the MAET. It enumerates the tuples each attribute admits at each event and stores their weight products, so that the density can be evaluated, compared, and measured without repeating that work. For a single multiset, the values, weights, and five parameters are given in turn: σ, r, whether the attribute is relative, whether it is periodic, and the period.

```matlab
dia   = buildMaet([0 2 4 5 7 9 11], [], 0.1, 2, true, true, 12);   % diatonic scale
wt    = buildMaet([0 2 4 6 8 10],   [], 0.1, 2, true, true, 12);   % whole-tone scale
fifth = buildMaet([0 7],            [], 0.1, 2, true, true, 12);   % a single fifth
```

```python
dia   = mpt.build_maet([0, 2, 4, 5, 7, 9, 11], None, 0.1, 2, True, True, 12)
wt    = mpt.build_maet([0, 2, 4, 6, 8, 10],    None, 0.1, 2, True, True, 12)
fifth = mpt.build_maet([0, 7],                 None, 0.1, 2, True, True, 12)
```

A pre-MAET is built whole, `buildMaet(pm)`, with any of the six per-attribute parameters (`sigma`, `per`, `period`, `r`, `rel`, `exch`) given alongside to override the specs (§6.1). Every measure below also takes a pre-MAET, or the raw values, in place of a density, and builds the density itself; building it once pays when the same density enters several calls (§10.5).

**Demos.** `demo_overview` / `demo_overview.py` (parts 1 and 2); `demo_maetPlots` / `demo_maet_plots.py`; `demo_dispatchAndKernelControls` / `demo_dispatch_and_kernel_controls.py` (parts 1–2: how a density is evaluated).

#### Matrix-valued kernel covariances

On an ordered attribute, `sigma` may be an $r \times r$ covariance matrix $\Sigma$ rather than a scalar, so that different directions in the tuple's space carry different uncertainty, and correlations between the tuple's coordinates, such as those successive intervals inherit from the note they share, enter the kernel directly. A scalar `sigma` is a standard deviation; a matrix is a covariance, in squared units.

The attribute must be ordered (`exch = false`), absolute, non-periodic, and not nested, with its tuple the whole event ($r = K$) and a value in every position; `'spectrum'` is not supported. A large variance along the all-ones direction expresses a graded tolerance of a common shift (a transposition, or a change of tempo), exact invariance remaining the province of `rel = true`. A matrix `sigma` is accepted by `buildMaet`, `evalMaet`, `simMaet`, `entropyMaet`, `sweptSimilarity`, and `sweptEntropy`, and the two sides of a comparison must share it. With $\Sigma = \sigma^2 I$ every result equals the scalar one.

`kernelCov` builds the common covariances from three sources of variance, each included when its width is given: `sdValue`, independent noise on each value (an onset, say); `sdInterval`, independent noise on each interval between successive values; and `sdShift`, a common shift of the whole tuple. The required `differenced` flag says whether the tuple holds values or their first differences, which decides how each source reaches it. With $\nabla$ the first-differencing map and $\nabla^{+}$ its pseudoinverse,

$$\Sigma = \mathrm{sd\_value}^2 \, I + \mathrm{sd\_interval}^2 \, \nabla^{+}(\nabla^{+})^{\top} + \mathrm{sd\_shift}^2 \, \mathbf{1}\mathbf{1}^{\top} \quad (\texttt{differenced = false}),$$

$$\Sigma = \mathrm{sd\_value}^2 \, \nabla\nabla^{\top} + \mathrm{sd\_interval}^2 \, I + \mathrm{sd\_shift}^2 \, \mathbf{1}\mathbf{1}^{\top} \quad (\texttt{differenced = true}).$$

On differenced times, value noise and interval noise are the two levels of the timing model of Wing and Kristofferson (1973); on logarithmic inter-onset intervals the shift is a change of tempo. The matrix must be positive-definite, which needs `sdValue` > 0, or both `sdInterval` and `sdShift`, on values, and `sdValue` or `sdInterval` on differences. The widths are standard deviations in the attribute's own coordinates, and the result goes straight into `sigma`:

```python
# Sliding rhythm-shape comparison, tempo-tolerant: ordered log-IOI
# triples, position noise from the shared onsets, graded tempo ridge.
# (A fragment: p_context, p_query, sigma_t, and onsets are defined
# beforehand; §8.7 explains the call.)
Sigma = mpt.kernel_cov(3, sd_value=0.03, sd_shift=0.4, differenced=True)
prof = mpt.swept_similarity(
    p_context, w_context, p_query, w_query,
    [Sigma, sigma_t], [3, 1], [False, False], [False, False], [0.0, 0.0],
    exch=[False, True], sweep={1: onsets}, align={1: 'window'},
    drop=[1], window={1: {'shape': 'rect', 'width': 2.0}})
```

**Demos.** `demo_tempoInvariance` / `demo_tempo_invariance.py`; `demo_softeningEquivalences` / `demo_softening_equivalences.py` (the kernelCov identity); `demo_repetitionHandling` / `demo_repetition_handling.py`.

### 8.2 Evaluating a density

`evalMaet` gives the density's value at query points. The diatonic density at intervals of 5, 6, and 7 semitones:

```matlab
evalMaet(dia, [5 6 7])          % Python: mpt.eval_maet(dia, [5, 6, 7])
```

gives 6, 2, and 6. Under `evalMaet`'s default normalization, `'none'`, each kernel peaks at its tuple's weight, so where kernels do not overlap a value counts the tuples there: the scale's six fifths (every pair a fifth apart but B–F), read in one direction at 7 and in the other at 5, and its one tritone, read both ways, at 6. `'gaussian'` makes each kernel integrate to 1 instead, so that the density integrates to its total mass, and `'pdf'` makes the whole density integrate to 1.

Query points are the columns of a matrix with one row per coordinate: `r` rows for an absolute attribute, and `r − 1` for a relative one, the tuple's values above its first. `maetCentres` returns the points at which the kernels sit.

**Demos.** `demo_preprocessing` / `demo_preprocessing.py`; `demo_jmm_2_1_joint` (JMM).

### 8.3 Comparing densities: similarity

The cosine similarity of two densities is the toolbox's central measure of resemblance, and `simMaet` computes it in closed form, with no grid:

```matlab
simMaet(dia, wt)                                     % 0.593
simMaet(dia, fifth)                                  % 0.626
simMaet(dia, fifth, 'normalize', 'oneSidedDenom')    % 6.000
```

`'normalize'` sets the denominator. `'cosine'` (the default) scores the match of shape alone, in [0, 1] since weights are non-negative, and is unchanged by rescaling either density's weights. `'oneSidedDenom'` divides by the second density's self inner product only, and so scores how much of the second is present in the first, in units of the second: the diatonic scale holds six fifths. It is 1 on a self-match, and more where the first holds more matching material. `'none'` returns the bare inner product.

The measures it generates are named by two independent choices: whether the pitches are spectrally enriched (S), and whether the domain is periodic, giving pitch *classes* (C): pitch similarity (PS), pitch class similarity (PCS), spectral pitch similarity (SPS), and spectral pitch class similarity (SPCS). SPCS is the best validated of these, predicting probe-tone ratings, tonal affinity in microtonal and inharmonic settings, and perceived triadic distance (Milne, Laney, & Sharp, 2015, 2016; Milne & Holland, 2016). The same measure applies unchanged to time, where periodicity gives metrical position rather than pitch class, and to any combination of attributes.

`simMaet` takes two densities, two pre-MAETs, or the raw values; one against a list; and batched matrices, one pair per row (§10.6). Two densities compared must have the same attributes, with the same `r`, σ, flags, and period on each; the number of values per event and of events may differ, so a monophonic query can be compared with a polyphonic context.

**Demos.** `demo_overview` / `demo_overview.py` (part 1b; part 1d: the cosine against `'oneSidedDenom'`, a scale against a fifth); `demo_triadSpcsGrid` / `demo_triad_spcs_grid.py`; `demo_scoreCategoricals` / `demo_score_categoricals.py` (what each encoding asks of a re-voiced chord); `demo_softeningEquivalences` / `demo_softening_equivalences.py`; `demo_jmm_1_2_similarity` (JMM); `demo_jmm_1_4_tonic_tuple_size` (JMM).

### 8.4 Entropy

`entropyMaet` measures how evenly a density's mass is spread:

```matlab
entropyMaet(dia, 'method', 'renyi2')    % 2.26 bits
entropyMaet(wt,  'method', 'renyi2')    % 1.33 bits
```

The whole-tone scale's interval content is the less spread, having only the even intervals. There are four estimators (§11.2): `'shannon'` and `'normalized'` on a grid, the latter the ratio in [0, 1] that reproduces published values; `'differential'`, which chooses its own grid; and `'renyi2'`, the collision entropy in closed form. The last two are scale-free, and so comparable across densities of different size, spread, or support; `'renyi2'` is also the fastest, and works at tuple sizes where any grid would exhaust memory. Since the kernels' width contributes to the spread, entropies are compared at a common σ.

**Demos.** `demo_overview` / `demo_overview.py` (part 1c); `demo_rhythmTensors` / `demo_rhythm_tensors.py` (part 3: entropy as rhythmic complexity); `demo_dispatchAndKernelControls` / `demo_dispatch_and_kernel_controls.py` (part 7: the estimators compared); `demo_jmm_1_1_entropy` (JMM).

### 8.5 Total mass

`massMaet` gives the total mass of a density: the sum of its tuples' weight products, each kernel carrying unit mass. With unit weights it is the number of tuples: `massMaet(dia)` is 42, the 7 × 6 ordered pairs of the scale's notes, and `massMaet(fifth)` is 2.

It turns a count into a share. The one-sided similarity of §8.3 counted the diatonic scale's fifths in units of the query; multiplied by the query's mass over the scale's, 6 × 2 / 42 = 0.286, it is their share of all the scale's pairs, 6 of 21. Because the match is by the relations among a tuple's values, this selects tuples in a way that no weighting of events can.

**Demos.** `demo_overview` / `demo_overview.py` (part 1d: a density's total mass, as the number of pairs a share is taken of; part 2d: a window's total mass, as the number of notes a share is taken of).

### 8.6 Plotting a density

`plotMaet` plots a density of one, two, or three dimensions (`dim = r - rel`); four or more cannot be plotted. It has three methods:

- **`'kernels'`** (the default) plots the model: one object per tuple, an ellipsoid, an ellipse, or a curve, coloured by the density at its centre. No grid is evaluated, so it is cheap, and it shows how many kernels make up each peak and, on a relative attribute, their shape, elongated along the all-ones diagonal. At one dimension the curves sum to the line that `'density'` plots.
- **`'points'`** (three dimensions only) samples the density on a grid, one translucent mark per point above a threshold. It evaluates the same grid as `'density'`, and plots more slowly.
- **`'density'`** plots the density itself: a line, a translucent surface, or at three dimensions a volume rendering. The three-dimensional volume is MATLAB only; Python raises an error naming `'points'`.

```matlab
pAttr = {{[0 200 400 500 700 900 1100]}};     % one event, seven pitches
specs = flatSpecs(pAttr, 'r', 4, 'rel', true, 'exch', true);
dens  = buildMaet(pAttr, [], 'specs', specs, 'sigma', 15, ...
                  'per', true, 'period', 1200);
plotMaet(dens);                                 % the kernels
figure; plotMaet(dens, 'method', 'density');    % the density
```

```python
p_attr = [[[0, 200, 400, 500, 700, 900, 1100]]]  # one event, seven pitches
specs = mpt.flat_specs(p_attr, r=4, rel=True, exch=True)
dens = mpt.build_maet(p_attr, None, specs=specs, sigma=[15.0],
                      per=[True], period=[1200.0])
mpt.plot_maet(dens)                              # the kernels
mpt.plot_maet(dens, method='points')             # the density sampled; the 3-D volume is MATLAB only
```

**The grid.** `'points'` and `'density'` evaluate a grid, set by `'step'`, a spacing in the density's units, or by `'nodes'`, a count of steps across the range plotted. What matters is the step against σ: a grid coarser than about one point per σ misses peaks rather than blurring them. The cost grows as the count to the power of the dimension, so the default is 1200 steps in one or two dimensions and 120 in three.

**Marks in MATLAB.** MATLAB plots overlapping marks of `'points'` one over another rather than blending them, so where they overlap the cloud can look different from opposite sides, with haloes seen from below. The marks are sized automatically to clear one another, `'markScale'` scales them, and `plotMaet` warns (`mpt:markOverlap`) when they overlap at the current view, naming a `markScale` or step that would keep them clear. If a plot looks haloed, lower `markScale`, view it from above, or use `'density'`. Python blends the marks, so its `marker_size` is used as given, with no warning.

**Appearance.** `'alphaPeak'` and `'alphaFloor'` set the opacity at the density's peak and where there is no density, and `'alphaGamma'` the curve between; `'colourGamma'` is the same curve for colour. MATLAB plots on dark panes by default (`'dark', true`), so that the dark low end of the colour map stays visible. Python follows matplotlib's defaults, and its `view` takes `(elev, azim)` where MATLAB's takes `[azimuth elevation]`. Python's plotting needs `matplotlib`, which is imported only when a plot is made.

**Demos.** `demo_maetPlots` / `demo_maet_plots.py` (every combination of r, [rel], [per], and [exch], by each method); `demo_overview` / `demo_overview.py` (part 1e); `demo_rhythmTensors` / `demo_rhythm_tensors.py` (part 2).

### 8.7 Swept similarity, entropy, and mass

The swept functions take a measure at each of a list of values on an attribute, the *sweep values*, and return a profile. They do this in two ways, which can be combined:

- **translating a query** along the attribute and comparing it with the whole context at each value, which finds *where*, or at which transposition, the query occurs (`sweptSimilarity`);
- **placing a window** on the context at each value, which measures *what is there*: the similarity of a query to the windowed context (`sweptSimilarity`), or the entropy or total mass of the windowed context itself (`sweptEntropy`, `sweptMass`).

The window is event weighting (§7.5), applied before each density is built. The two examples below set out the common cases; the rest of the section is reference.

#### Finding a motif: translation

A melody holds a three-note motif, C D E, at beat 0, and again a tone higher, D E F♯, at beat 5. Translating the motif in pitch class and in time finds both:

```matlab
specs = flatSpecs({0, 0}, 'names', {'pitch', 'time'}, 'sigma', [0.1 0.2], ...
                  'per', [true false], 'period', [12 0]);
pmMel = packPreMaet({[60 62 64 65 67 62 64 66 67 69], 0:9}, [], specs);
pmQ   = packPreMaet({[60 62 64], 0:2}, [], specs);
[S, mu] = sweptSimilarity(pmMel, pmQ, 'sweep', {1, 0:11; 2, -2:9});
```

```python
specs = mpt.flat_specs([[0], [0]], names=['pitch', 'time'], sigma=[0.1, 0.2],
                       per=[True, False], period=[12.0, 0.0])
pm_mel = mpt.pack_pre_maet([[60, 62, 64, 65, 67, 62, 64, 66, 67, 69], list(range(10))], None, specs)
pm_q = mpt.pack_pre_maet([[60, 62, 64], [0, 1, 2]], None, specs)
S, mu = mpt.swept_similarity(pm_mel, pm_q, sweep={0: np.arange(12), 1: np.arange(-2, 10)},
                             return_offsets=True)
```

`S` is a 12 × 12 profile, transposition by time offset. It is 1 at two points, a transposition of 0 at beat 0 and of 2 at beat 5, the two statements, and 2/3 wherever two of the motif's three notes match, such as the untransposed motif at beat 4, against G D E. The sweep values here are the offsets added to the query as written; naming an attribute alone, `'sweep', 2`, asks for default values covering every placement at which query and context overlap. Every offset is computed in one pass, not one comparison at a time.

#### Following a passage: windows

Two bars, the notes of a C major triad and then of an F major triad, are compared with a C major triad, bar by bar. A rectangular window four beats wide is placed at the start of each bar, and time is dropped once the window has weighted the events, so that the triad is compared with *what* each bar contains, not *where*:

```matlab
pmCtx = packPreMaet({[60 64 67 60 65 69 60 65], 0:7}, [], specs);   % C E G C | F A C F
pmTri = packPreMaet({{[60 64 67]}, 0}, [], specs);                  % a C major triad
win = {2, {'rect', 'width', 4, 'ref', 'start'}};
S = sweptSimilarity(pmCtx, pmTri, 'sweep', {2, [0 4]}, 'align', {2, 'window'}, ...
                    'window', win, 'drop', 2);                      % 1.333, 0.333
M = sweptMass(pmCtx, 'sweep', {2, [0 4]}, 'window', win, 'drop', 2); % 4, 4
share = S * massMaet(pmTri) ./ M;                                   % 1, 0.25
```

```python
pm_ctx = mpt.pack_pre_maet([[60, 64, 67, 60, 65, 69, 60, 65], list(range(8))], None, specs)
pm_tri = mpt.pack_pre_maet([[[60, 64, 67]], [0]], None, specs)
win = {1: {'shape': 'rect', 'width': 4.0, 'ref': 'start'}}
S = mpt.swept_similarity(pm_ctx, pm_tri, sweep={1: [0.0, 4.0]}, align={1: 'window'},
                         window=win, drop=[1])
M = mpt.swept_mass(pm_ctx, sweep={1: [0.0, 4.0]}, window=win, drop=[1])
share = S * mpt.mass_maet(pm_tri) / M
```

The one-sided similarity counts each bar's matching notes in units of the triad: four in the first bar, 4/3, and one in the second, 1/3. `sweptMass` gives the number of notes in each window, so dividing by it gives the share of each bar's notes that belong to the triad: all of the first, a quarter of the second. `sweptEntropy` takes the same sweep and window options and gives the entropy of each window's density. With more sweep values, `'sweep', {2, 0:0.25:4}`, the same calls trace a continuous profile as the window moves.

#### Reference

**Two attributes.** The *swept attribute* is the one the sweep values lie on: the query is translated along it, and a window is a function of position along it. The *target attribute* (`targetAttr`; by default the first attribute not dropped) is the one whose weights a window multiplies. They usually differ: a window over time weights the pitch events.

**The rule.** At each sweep value $s$:

- a translated query has its *reference value*, `queryRef`, at $s$: it is translated by $\mu = s - \mathrm{queryRef}$;
- a window has its reference value, the point $\delta = 0$ of its function $h(\delta)$, at $s$. This is its centre, unless `'ref'` places a rectangle's start or end there (below).

For each swept attribute, `'align'` says which of the two are placed:

| `align` | At each sweep value $s$ | `queryRef` by default |
|:---|:---|:---|
| `'query'` (default) | the query only: translation over the whole context | 0: sweep values are the offsets added to the query as written |
| `'both'` | the query and a window, both at $s$ | the query's middle, so the window is aligned at the query's middle |
| `'window'` | a window only; the query is left as written | – |
| `'independent'` | a window at each value of one list, the query at each value of another, in every combination (`'sweep'`, `{a, {windowValues, queryValues}}`) | the query's middle |

The query's middle is the mean of its events' values on the swept attribute. A window at $s$ weights each context event $n$ on the target attribute:
$$w'(n) = w(n)\,h\bigl(p_a(n) - s\bigr),$$
where $p_a(n)$ is event $n$'s value on the swept attribute $a$. Where an event holds several values there (the onsets of a bound event, say), `'locate'` says which stands for it: `'centroid'`, their mean (the default); `'start'` or `'end'`, the first or the last; or `'mid'`, the midpoint of those two. The window acts on the events before any density is built, so a relative or dropped swept attribute is still windowed by the values the pre-MAET holds.

**Choosing an alignment.**

- **`'query'`** – where in the context, or at which transposition, the query best matches the context as a whole. Only the kernel width σ on the swept attribute limits which context events count. The choice for transposition, for any periodic attribute, and for a whole-context cross-correlation.
- **`'both'`** – a local `'query'`: the window fixes the region of the context that counts. Under `'oneSidedDenom'`, the default, this changes little, the kernel already discounting distant events; under `'cosine'` unmatched material inside the window lowers the score, material outside it is ignored, and repeated material cannot stand in for missing content, as it can under the default.
- **`'window'`** – the query is not translated: the window steps through the context, and the query, as written, is compared with each region.
- **`'independent'`** – every window position against every translation of the query: a correlogram, for a best placement that changes across the context, such as a drifting lag between two parts. The query's list may hold one row per window value.

**Absolute, relative, or dropped.** What the swept attribute contributes depends on its `[rel]` flag and on whether it is dropped:

- **Absolute** – compared by position, so all four alignments apply. Under `'window'` the query is compared in place, showing where in the context its match comes from; under `'oneSidedDenom'`, windows that tile the context (half-open rectangles a width apart) give contributions that sum to the whole-context similarity.
- **Relative** – compared only through the values relative to the lowest (a bound event's onsets measured from its first, say), so the query's internal spacing must match but not its position. Translation changes nothing, so only `'window'` applies.
- **Dropped** (`'drop'`) – marginalized after the window has weighted the events, so the query is compared with *what* a region contains, not *where*: with time dropped, C–E–G matches any region containing those pitches, in any order or rhythm. Use it when the query has no meaningful arrangement on the attribute (a key profile, a chord). Only `'window'` applies.

A differenced attribute is absolute or relative like any other: swept, its windows select by interval size, and translation adds the same amount to every interval (on logarithmic inter-onset intervals, a change of tempo). Usually the differenced attribute (pitch) is not the swept one (time), which keeps each event's onset at order 0 (`demo_overview`, part 2c).

Several attributes can be swept at once, each with its own alignment. With each bar windowed on time, time dropped, and the query translated in pitch, the profile finds the bar and the transposition of each statement at once.

**When a window is needed.** On an absolute attribute, translation already localizes, the kernel letting the query match only material near where it is placed. A window on the context is indispensable where translation cannot localize: on a dropped or relative swept attribute, and in `sweptEntropy` and `sweptMass`, which have no query. On a translated attribute a window matters mainly under `'cosine'` (above). The query itself is not windowed: to weight its events, and so ask which part of it matches, apply `weightEvents` to it before the call.

**Windows.** A window is a shape followed by named options: MATLAB `{'gaussian', 'sd', 4}` or `{'rect', 'width', 2, 'edges', 'closed'}` (or a struct with these fields), Python `{'shape': 'gaussian', 'sd': 4}`. The shape is γ in [0, 1], `'gaussian'`, or `'rect'`, as in `weightEvents` (§7.5), and the scale is always named: `'width'`, the full width $W$ of the equivalent rectangle, or `'sd'`, $W/(2\sqrt{3})$. A window may also be one of `weightEvents`' exponentials, symmetric (`'exponential'`) or one-sided (`'exponentialBefore'`, `'exponentialAfter'`), with `'sd'` or `'decayRate'`, or a function of the displacement $p_a(n) - s$; the serial-position profiles are refused. On a periodic attribute the displacement wraps.

A rectangle is *half-open* by default (`'edges', 'halfOpen'`; not to be confused with the function `edges`), including its lower edge and not its upper, so that windows placed a width apart share no event; a *closed* rectangle (`'edges', 'closed'`) includes both, as a window that must hold a query needs. Its reference value is its centre; `'ref', 'start'` places its start at the sweep value, so that it covers $[s, s + W)$, and `'ref', 'end'` its end, covering $[s - W, s)$. A window is refused under `'query'`, and its width is required for `'window'` and `'independent'`, and in `sweptEntropy` and `sweptMass`, which have no query and so always place a window; a window that holds no event has a total mass of 0 and an entropy of `NaN`. For `'both'` it may be omitted, the window then being the smallest closed rectangle that holds the query, so that an exact match scores 1; a window given for `'both'` that leaves out some of the query gives a warning.

**Sweep values.** Naming the swept attribute alone (`'sweep', a`) asks for default sweep values, and `'start'`, `'stop'`, and `'step'` each replace one default, with a bare number where one attribute is swept: `'sweep', 2, 'step', 0.5` (Python `sweep=1, step=0.5`). Where the query is translated, the defaults cover every offset at which query and context overlap (one period, on a periodic attribute), at a step no wider than half the standard deviation of the profile's peaks: each pair of matching tuples contributes a peak of standard deviation $\sigma\sqrt{2/D}$ in the offset (Milne, 2026a, Eq. 10), where $D$ is the tuple size (for a nested attribute, the product of its per-level sizes). Where the values lie on a common lattice, such as onsets on whole beats, the step is chosen so that every exact match also falls on the grid. Where only a window is placed, the defaults cover the context's range at half the window's sd; for largely separate windows, give `'step'` as half the window's width. A rectangle's profile is constant between the points where an event enters or leaves it, so without a `'step'` its sweep values are those pieces, each sampled just inside its ends, and a line through them traces the steps exactly.

The offsets and the sweep values are returned on request, as the axes of the profile: MATLAB `[S, mu, sv] = sweptSimilarity(...)`, `[H, sv] = sweptEntropy(...)`, and `[M, sv] = sweptMass(...)`, each a $1 \times A$ cell indexed by attribute; Python `return_offsets=True` and `return_sweep_values=True`, each a dict. The offset $\mu = s - \mathrm{queryRef}$ is the query's shift from where it was written, so profiles of one query with and without differencing share one axis (§7.4). A profile sampled at the default step can be interpolated to a finer grid, `interp1(sv{a}, S, x, 'spline')` or `scipy.interpolate.CubicSpline(sv[a], S)(x)`, except across the jumps of a rectangular window.

**Choosing `queryRef`.** Under `'query'` it only relabels the axis; under `'both'` it also decides the point of the query at which the window is aligned. At 0, the default for `'query'`, sweep values are offsets from the query as written; at the query's middle, the default for `'both'`, they are where its middle lands; at a particular point of the query, such as its first onset or its root, they are where that point lands: the time at which a match starts, or the key of a transposition.

**Normalization.** The default for `sweptSimilarity`, `'oneSidedDenom'`, divides the inner product of the windowed context $h \cdot f_X$ and the query $f_Y$ by the query's own:
$$ s_{\text{one-sided}} = \frac{\langle h \cdot f_X, f_Y \rangle}{\langle f_Y, f_Y \rangle}. $$
It scores *how much of the query is present here*: 1 for the query itself, wholly inside the window; less where the window misses part of it; and more where the context holds more matching material than the query, as a motif-finding or probe-tone analysis wants. `'cosine'`, $\langle h \cdot f_X, f_Y \rangle / \sqrt{\langle h \cdot f_X, h \cdot f_X \rangle \langle f_Y, f_Y \rangle}$, scores *how well the local shape matches*, whatever the amount of material, in [0, 1]. `'none'` returns the inner product.

#### Sweeps from built densities

`sweepSimMaet` computes a translation sweep from two built densities and a matrix of offsets, one row per attribute and one column per step, so it can reuse densities across many sweeps and follow any path of offsets. Here the densities are the melody and query of the pattern search of §4, built with `buildMaet`:

**MATLAB:**
```matlab
melody  = buildMaet({melody_p, melody_t}, [], sigma, r, rel, per, period);
query   = buildMaet({query_p, query_t}, [], sigma, r, rel, per, period);
offsets = [zeros(1, 29); linspace(-1.0, 6.0, 29)];   % translate in time only
S = sweepSimMaet(melody, query, offsets);
```

**Python:**
```python
geom = ([0.1, 0.2], [1, 1], [False, False], [True, False], [12., 0.])
melody = mpt.build_maet([melody_p[None, :], melody_t[None, :]], None, *geom)
query = mpt.build_maet([query_p[None, :], query_t[None, :]], None, *geom)
offsets = np.vstack([np.zeros(29), np.linspace(-1.0, 6.0, 29)])
S = mpt.sweep_sim_maet(melody, query, offsets)
```

Like `sweptSimilarity`, it computes every offset in one pass, not one comparison per offset. Only an absolute attribute can be swept, a relative one being unchanged by translation. `'method'` chooses how (§11.1): `'mixture'`, a Gaussian mixture in the offset; `'orbit'`, which scales better to large tuple sizes and also covers a swept periodic attribute; `'contract'`, for densities with a nested attribute; or `'auto'` (the default), which chooses among them.

**Demos.** `demo_sweptSimilarity` / `demo_swept_similarity.py` (every alignment, on one melody holding four statements related to a query); `demo_overview` / `demo_overview.py` (part 2d: a window over time, with each triad's share of the notes in it); `demo_helixBlend` / `demo_helix_blend.py`; `demo_tempoInvariance` / `demo_tempo_invariance.py` (part 3: a sweep under a tempo-tolerant kernel); `demo_batchProcessing` / `demo_batch_processing.py` (`sweepSimMaet` against translated copies); `demo_dispatchAndKernelControls` / `demo_dispatch_and_kernel_controls.py` (part 3: sweeps in one pass); `demo_jmm_1_1_entropy` (JMM); `demo_jmm_1_3_cadence_nesting` (JMM); `demo_jmm_2_3_spectral` (JMM); `demo_jmm_3_1_texture` (JMM); `demo_jmm_3_2_diff` (JMM); `demo_jmm_3_3_xcorr` (JMM).

---

## 9. Measures on pitch and rhythm sets

The measures of this section take a pitch or rhythm set, or a spectrum, directly rather than through a pre-MAET. They were developed in separate literatures and are implemented here so that predictors from all of them can be applied to the same material within one analysis. The function entries are in §12.5–§12.8.

### 9.1 Consonance and harmonicity

Four complementary measures address how consonant or harmonic a collection sounds: spectral entropy (`spectralEntropy`), template harmonicity (`templateHarmonicity`), tensor harmonicity (`tensorHarmonicity`), and sensory roughness (`roughness`). They have been used singly and in combination to predict consonance ratings, perceived affect, and tonal stability. All but roughness are built on expectation-tensor densities: spectral entropy and template harmonicity smooth the spectrum into a density over pitch, and tensor harmonicity builds a relative density over intervals. Roughness works on the spectrum directly.

For a major and a minor triad, in cents, with harmonic spectra:

```matlab
maj = [0 400 700];  mnr = [0 300 700];
spec = {'harmonic', 12, 'powerlaw', 1};
spectralEntropy(maj, [], 12, 'spectrum', spec)    % 10.003
spectralEntropy(mnr, [], 12, 'spectrum', spec)    % 10.029
tensorHarmonicity(maj, [], 12)                    % 0.174
tensorHarmonicity(mnr, [], 12)                    % 0.031
```

Both rank the major triad as the more harmonic, by a lower spectral entropy and a higher tensor harmonicity; the difference is far greater in tensor harmonicity, which reads the triad's intervals directly against those of a harmonic series. The Python calls are the same, with `spectrum=` as a keyword.

#### templateHarmonicity and tensorHarmonicity compared

Both measure how far a chord's intervals resemble those of a harmonic series, but in different ways.

`templateHarmonicity` cross-correlates the chord's composite spectrum (its pitches enriched by `'chordSpectrum'`, or empirical peaks as given) with the spectrum of a single harmonic complex tone. It returns two measures of different things. hMax, the cosine similarity at the best-matching transposition, says how well the single best-fitting harmonic series accounts for the chord's spectrum: 1 for a perfect fit. hEntropy, the entropy of the whole cross-correlation treated as a distribution over transpositions (by default its differential entropy, in $\log_2$ cents; `'method'` selects the form), says how clearly one fit stands out from the others: low when the cross-correlation has one dominant peak, and high when it is flat, with many roughly equal candidate fundamentals. hMax reads only the peak and hEntropy the whole profile, so two chords with the same hMax can differ in hEntropy, although across typical sets of chords the two are highly (negatively) correlated. The chord enters only through its spectrum. Given pure tones, the best fit places the template's fundamental on one note, which with the default template (partial weights falling as $1/n$) outweighs any match of notes less than an octave apart to higher partials; such chords then share an hMax whenever they have the same number of notes. `'chordSpectrum'` gives the tones partials, through which hMax can compare their intervals. With `'per'` true, chord and template are folded into one octave (`'period'`, default 1200) and the cross-correlation is circular, so the candidates are virtual pitch classes.

`tensorHarmonicity` builds the relative tensor of a harmonic series at r = K, the chord's size, and evaluates it at the chord's intervals above its lowest pitch: how probable those intervals are within a harmonic series, given σ. The template is duplicated K times, every copy at 0 cents, so that one partial can fill several positions of a tuple; without this, a unison or two notes sharing a partial could not register. The copies are not placed at the chord's pitches: the chord enters only as the point at which the density is read.

`templateHarmonicity` is therefore a spectral measure, and `tensorHarmonicity` an interval measure. `virtualPitches` returns the whole cross-correlation profile from which `templateHarmonicity` takes its two values.

**Demos.** `demo_triadConsonance` / `demo_triad_consonance.py` (five measures over a grid of triads); `demo_virtualPitches` / `demo_virtual_pitches.py`; `demo_audioAnalysis` / `demo_audio_analysis.py`; `demo_overview` / `demo_overview.py` (part 3).

### 9.2 Scale and rhythm structure

These measures describe points distributed around a cycle: pitch classes in an octave, or onsets in a metrical cycle. They fall into three groups by output: *period-level* measures returning one value per collection (balance, evenness, the discrete Fourier transform itself, coherence, sameness, and $n$-tuple entropy); *integer-position* measures over an equal division of the period (the circular autocorrelation phase matrix, Markov prediction); and *continuous-position* measures evaluable anywhere on the cycle (edge detection, the projected centroid, mean offset). Seven of them also accept a positional uncertainty $\sigma$, which softens their discrete tallies in the same spirit as the expectation tensor's kernel. §12.6–§12.7 give each measure and what $\sigma$ means for it.

```matlab
coherence([0 2 4 5 7 9 11], 12)    % 0.993: diatonic
coherence([0 2 3 5 7 8 11], 12)    % 0.871: harmonic minor
```

The diatonic scale is coherent but for its one tie, the augmented fourth and the diminished fifth both spanning six semitones (`'strict', false` counts it as coherent, giving 1); the harmonic minor scale, with its augmented second, has eighteen failures. In Python, `mpt.coherence` returns the quotient and the count of failures together.

**Demos.** `demo_overview` / `demo_overview.py` (parts 4–5); `demo_dftCircularSimulate` / `demo_dft_circular_simulate.py`; `demo_sigmaSpace` / `demo_sigma_space.py` (soft sameness, coherence, and n-tuple entropy); `demo_rhythmTensors` / `demo_rhythm_tensors.py` (part 4).

### 9.3 Sequences

`continuity` summarizes the recent trend of a sequence of pitches, inter-onset intervals, or their differences, leading up to a query point (§12.8). Serial-position profiles are applied by `weightEvents` (§7.5).

**Demos.** `demo_probeTone` / `demo_probe_tone.py` (part 4: continuity).

---

# Part III – Reference

## 10. API conventions

The MATLAB and Python implementations give the same outputs to floating-point precision. This section covers the translation between the two languages and the calling conventions they share. §12 uses MATLAB syntax, from which the Python follows by the rules below.

### 10.1 Naming

MATLAB uses camelCase and Python snake_case (`simMaet` → `sim_maet`, `addSpectra` → `add_spectra`). The exceptions:

| MATLAB | Python |
|:---|:---|
| `balanceCircular` | `balance` |
| `evennessCircular` | `evenness` |
| `mptDefaults` | `set_default`, `get_default`, `get_defaults`, `reset_defaults`, `show_defaults` |

### 10.2 Terminology

- An **element** is one weighted entry in an attribute's multiset at one event: the unit from which every density is built.
- Its **position** is its value in the attribute's space (a pitch in cents, a time in beats, a category's coordinate), held in the `p` / `pAttr` arrays.
- Its **weight** is the non-negative mass at that position (salience, amplitude, or probability of perception), held in the `w` arrays.

An attribute's elements form a $K_a \times N$ matrix, one column per event. A **row** of that matrix runs across events and is not an element: a per-row weight or offset applies to one element in each event.

### 10.3 Calling conventions

| Concept | MATLAB | Python |
|:---|:---|:---|
| Default (all ones) weights | `[]` | `None` |
| Boolean flags | `true` / `false` | `True` / `False` |
| Spectrum arguments | Cell array: `{'harmonic', 12, 'powerlaw', 1}` | List or tuple: `('harmonic', 12, 'powerlaw', 1)` |
| Name-value pairs | `'name', value` | `name=value` |
| Precomputed density | Struct with `.tag = 'MaetDensity'` | `MaetDensity` dataclass |

### 10.4 Weight arguments

A weight argument `w` specifies a weight for each element, and may take any shape that broadcasts to one. For $N$ events with $K$ elements each ($K = 1$ for single-valued attributes):

| Input form | MATLAB | Python |
|:---|:---|:---|
| Default (uniform 1) | `[]` | `None` |
| Uniform scalar `c` | `c` | `c` |
| Per event (broadcast across rows) | length-$N$ row | 1-D length-$N$, or `(1, N)` |
| Per row (broadcast across events) | $K \times 1$ column | `(K, 1)`, or 1-D length-$K$ |
| Per row and event | $K \times N$ matrix | `(K, N)` |
| Per event, value by value | $1 \times N$ cell | list of length $N$ |

In the last form, each entry is a scalar that weights all of that event's values, or a vector with one weight per value (§6.1); positions holding no value take weight 0. A Python list of $K \neq N$ scalars is read per row. The per-row forms mean something only when $K > 1$, as for a chord's pitches or a pitch's partials; `continuity` accepts only the first three forms.

Forms that specify the same weights are interchangeable, so the output of one function can be passed to another. Weights are non-negative, and are read as the probability that an element is perceived; the preprocessing operations propagate them consistently with that reading. For several attributes, `w` is a cell (MATLAB) or list (Python) with one entry per attribute, each following these rules; a single `[]` / `None` or scalar applies to every attribute.

### 10.5 Densities and raw values

`simMaet`, `evalMaet`, and `entropyMaet` take a density built by `buildMaet`, a pre-MAET, or the raw values with their parameters, and detect the form from the first argument. The results are the same, and within one call so is the cost, since a batched call builds each distinct density once. Building a density yourself pays across calls, when the same reference is compared in several of them, or when the density itself is wanted, to inspect or plot. `'spectrum'` is accepted only in the raw forms, a built density's spectrum being part of it. `'precision'` applies to batched raw calls (§10.7), and `'verbose'` to every call.

### 10.6 Batching: rows as multisets

Most functions take a matrix of multisets as well as a single one:

- **Each row is one multiset**, and the function returns one result per row.
- **Each column is an element**: a voice of a chord, or a partial of a spectrum.
- **NaN-padding** allows rows of different sizes: a row's trailing `NaN` entries are dropped before it is processed.
- **Weights**, when given, have the same shape as the values; `[]` / `None` gives uniform weights.

These functions take the batched form: `simMaet` (one pair per row, a single row broadcasting against the others), `evalMaet`, `entropyMaet`, `tensorHarmonicity`, `templateHarmonicity`, `virtualPitches`, `spectralEntropy`, `dftCircular`, `meanOffset`, `edges`, `projCentroid`, `circApm`, `coherence`, `sameness`, `nTupleEntropy`, `balanceCircular`, and `evennessCircular`.

MATLAB reads a matrix with more than one row and column as a batch, and any vector as one multiset; Python reads any 2-D array as a batch, so a `(K, 1)` array is K one-element rows. A batch of one-element multisets is written the same way in both by padding to two columns with `NaN`: `[p(:), nan(K, 1)]` / `np.column_stack([p, np.full(K, np.nan)])`.

A pre-MAET or other multi-attribute item has no rows to collapse, so a cell / list of them is computed entry by entry instead (§6.1).

**Demos.** `demo_batchProcessing` / `demo_batch_processing.py`; `demo_edoApprox` / `demo_edo_approx.py`.

### 10.7 Canonical-form deduplication

A batched call finds the rows that are equivalent under the function's symmetries and computes each class once, so material with much repeated structure, such as generator-chain sweeps, EDO scans, or transposition orbits, costs little more than its distinct cases, with no change in the values returned. Which rows are equivalent depends on what the output is invariant to:

- **Invariant to transposition** (`coherence`, `sameness`, the `H` of `nTupleEntropy`, `simMaet` with `rel = true`): every transposition of a multiset is one class.
- **Equivariant under transposition** (`dftCircular`, `edges`, `projCentroid`, `circApm`): only reorderings and period-equivalents are merged.
- **Invariant to reordering** (every batched function): reorderings of the elements, and on a periodic attribute values a period apart, are merged.

`'precision'` sets the number of decimal places to which values are rounded when classes are found, so that values differing only by floating-point noise are merged: `'precision', 4` suffices for a 12-EDO grid, and 6 for fractional cents from just ratios. On an irrational grid, such as an EDO whose step in cents does not terminate, rounding cannot merge every transposition; express the pitches in EDO steps instead, scaling σ and the period to match.

**Demos.** `demo_batchProcessing` / `demo_batch_processing.py`.

### 10.8 Return values

The outputs are the same in both languages, except:

- `coherence` and `sameness` return both the quotient and the count in Python (`c, nc = mpt.coherence(...)`); in MATLAB the count is an optional second output.
- `audio_peaks` returns three values in Python (`f, w, detail = mpt.audio_peaks(...)`); MATLAB returns the detail struct only when a third output is requested.
- `balance` and `evenness` return `(mean, std)` with `return_std=True` in Python; MATLAB returns the standard deviation as an optional second output.

### 10.9 Query points

In MATLAB, query points (`X`, `x`) are the columns of a matrix. In Python, a 1-D density takes a 1-D array, and a density of more dimensions a `(dim, n_queries)` array.

---

## 11. Performance and numerical controls

Most analyses need nothing in this section. The defaults choose how each value is computed, and every choice gives the same value to within the accuracy that `truncationSigmas` sets.

### 11.1 Method selection

`simMaet`, `evalMaet`, and `sweepSimMaet` take a `'method'` keyword that chooses how a value is computed, never which value. The default, `'auto'`, chooses from the structure of the densities and a cost model, not from timing, so a call takes the same route on any machine and in either language. The alternatives:

| `method` | Functions | Route |
|:---|:---|:---|
| `'bulger'` | `simMaet` | Combinations of one side's tuples against permutations of the other's (Bulger's method); fastest at small r and K |
| `'mobius'` | `simMaet`, `evalMaet` | Möbius inversion on the partition lattice, with orbit collapse for the inner product; fastest at large r or K |
| `'centres'` | `simMaet`, `evalMaet` | Every tuple enumerated; the reference against which the others can be checked |
| `'contract'` | `simMaet`, `sweepSimMaet` | Level-by-level contraction of a nested attribute; nested densities only |
| `'mixture'`, `'orbit'` | `sweepSimMaet` | A translation sweep as a mixture in the offset, or by the orbit inner product, which also covers a swept periodic attribute |

Override `'auto'` only to benchmark, or to check one route against another. A forced method still respects the structural limits: an ordered attribute always takes Bulger's method in `simMaet`, and `'contract'` needs a nested density.

**Periodic kernels.** On a periodic attribute a kernel can wrap in two ways: summing over every periodic image (`'full-image'`, the default, set by `buildMaet`'s `'wrap'` argument) or taking the nearest image only (`'single-image'`). The two agree while σ is small relative to the period (below about 0.03 of it at the default truncation) and differ above that; there the attribute's `wrap` decides which is computed, and so which route is taken.

**Seeing the route.** `explainDispatch(densX, densY)`, or `explainDispatch(dens, nQueries)` for an evaluation, reports the route a call would take, the predicted time of each candidate, and the accuracy limits in force, without running the call. ARCHITECTURE §4 shows every routing rule as a decision tree.

**Demos.** `demo_dispatchAndKernelControls` / `demo_dispatch_and_kernel_controls.py` (parts 1–3).

### 11.2 The four entropy estimators

`entropyMaet` takes one of four estimators through `'method'`:

- **`'shannon'`** (default) – the discrete Shannon entropy of the density's mass in the cells of a grid of `'nPointsPerDim'` points per dimension.
- **`'normalized'`** (or `'normalised'`) – the same divided by $\log N$, into [0, 1]. It reproduces the values published in Milne et al. (2017) and Smit et al. (2019), but depends on the grid, so it compares densities only on a common grid.
- **`'differential'`** – the differential entropy, on a grid refined until the estimate converges, so no grid is chosen. It is scale-free, and so the one to compare densities of different size, spread, or support. Single densities only; undefined at σ = 0.
- **`'renyi2'`** – the Rényi-2 (collision) entropy, $H_2 = -\log(\langle T, T\rangle / Z^2)$, in closed form: exact, with no grid, and feasible at tuple sizes where any grid would exhaust memory. Single densities only; undefined at σ = 0. A relative attribute at r = 1 contributes 0.

The two grid methods require `'nPointsPerDim'`, which has no default, and `'xMin'` and `'xMax'` on a non-periodic attribute; the other two ignore them. `spectralEntropy` defaults to `'differential'`, and `nTupleEntropy` to `'normalized'`, the formulation of Milne and Dean (2016).

**Demos.** `demo_dispatchAndKernelControls` / `demo_dispatch_and_kernel_controls.py` (part 7).

### 11.3 Kernel-evaluation controls

Four settings trade accuracy, speed, and memory in the evaluation of Gaussian kernels.

| Setting | Default | Per call | Effect |
|:---|:---|:---|:---|
| `truncationSigmas` | 6 | yes | Kernel contributions beyond this many σ are skipped. The worst-case error is about 1e-3 at 4, 1e-5 at 5, 2e-8 at 6, and 1e-10 at 7; `Inf` gives the toolbox's accuracy floor of 1e-12, at a width of about 7.43σ. |
| `kernelPrecision` | `'double'` | yes | `'single'` computes kernel matrices in single precision, often about twice as fast, to about 7 significant figures. The Möbius inner product ignores it, since its alternating sum would magnify the rounding. |
| `kernelChunkBytes` | `'auto'` | no | The memory for each chunk of a large kernel computation; `'auto'` is half the available memory. Lower it to reduce peak memory; the result is unchanged. |
| `kernel_threads` (Python only) | `'auto'` | no | Threads for the kernel arithmetic; `'auto'` takes `OMP_NUM_THREADS`, or else the number of cores up to eight. Results agree to rounding at any count. Set it to 1 when parallelizing at a higher level, such as a process pool or a job array. MATLAB threads this arithmetic itself. |

A value given in a call overrides the toolbox default (§11.4), which overrides the factory default.

**Demos.** `demo_dispatchAndKernelControls` / `demo_dispatch_and_kernel_controls.py` (parts 4–5).

### 11.4 Toolbox defaults

`mptDefaults` (MATLAB) and `mpt.set_default`, `get_default`, `get_defaults`, `reset_defaults`, and `show_defaults` (Python) inspect and change the toolbox-wide defaults: the settings of §11.3, `showHints`, and `postHocGuards`. The defaults last for the session.

| MATLAB | Python | Effect |
|:---|:---|:---|
| `mptDefaults` | `mpt.show_defaults()` | Print the current values, with a summary of each |
| `S = mptDefaults` | `mpt.get_defaults()` | Return the current values |
| `mptDefaults('name')` | `mpt.get_default('name')` | Return one value |
| `prev = mptDefaults('name', val, ...)` | `prev = mpt.set_default(name=val, ...)` | Set values, returning the previous ones |
| `mptDefaults(prev)` | `mpt.set_default(**prev)` | Restore previous values |
| `mptDefaults('reset')` | `mpt.reset_defaults()` | Reset every value to its factory default |

Restoring the returned values, rather than resetting, keeps any other defaults the caller has set:

```matlab
prev = mptDefaults('truncationSigmas', Inf, 'kernelPrecision', 'single');
% ... work with the new defaults ...
mptDefaults(prev);                                 % restore
```

```python
import math
prev = mpt.set_default(truncation_sigmas=math.inf, kernel_precision='single')
# ... work with the new defaults ...
mpt.set_default(**prev)                            # restore
```

**Messages.** The first kernel evaluation of a session warns once, unless `truncationSigmas` has been set, that it is at its factory value (`mpt:truncationDefault`; Python `mpt.TruncationDefaultWarning`); the usual warning controls suppress it. `simMaet` and `evalMaet` name the route they chose, once per call and route, which `showHints = false` silences. Batched calls expected to take more than a few seconds print an estimated time and a progress count, which `'verbose', false` silences. `postHocGuards` (default true) re-runs a call by another route if its result is impossible, such as a non-finite value; switch it off only for calibration.

**Demos.** `demo_dispatchAndKernelControls` / `demo_dispatch_and_kernel_controls.py` (part 6).

---

## 12. Function reference

One entry per function, grouped by the stage of an analysis it serves: its signature in MATLAB form, what it does, and where it is described. `help functionName` (MATLAB) or `help(mpt.function_name)` (Python) gives every argument, default, and output. Python names follow §10.1.

### 12.1 Score and audio input

- **readScore(path)** – Reads a MIDI file (format 0 or 1) or a MusicXML score (`.musicxml`, `.xml`, or `.mxl`) into an attribute table, one row per sounding note (§5.1). Every table has `onsetBeats`, `onsetSeconds`, `durationBeats`, `durationSeconds`, `pitch` (a MIDI number), `velocity`, `part`, and `measure` (Python `onset_beats`, and so on). A MIDI file adds `channel`, `noteNumber`, `program`, `weight`, and sounding durations that include the sustain and sostenuto pedals; a MusicXML score adds `voice`, `staff`, `fermata`, and the articulations `staccato`, `accent`, and `tenuto`. Pitch bend is read into `pitch`, and channel volume and expression into `weight`. A beat is a quarter note, and seconds follow the tempo map.
- **gridAttrTable(T, step, ...)** – Samples an attribute table on a regular time grid, one event per grid point (§5.2). `'weights'` sets what a slice takes from a note overlapping it: `'coverage'` (default), `'presence'`, or `'item'`. Other options: `'time'` (`'beats'` or `'seconds'`), `'duration'` (`'duration'` or `'soundingDuration'`), and `'limits'`. Adds the columns `gridIndex`, `gridOnsetBeats`, `gridOnsetSeconds`, `noteId`, and `weight`. A gridded table can be regridded at a coarser step, a whole multiple of the first, with the same result as gridding at that step directly.
- **ungridAttrTable(G)** – Returns a gridded table to the table it was made from, less any notes that `'limits'` cut out or whose slices were all removed.
- **preMaetFromAttrTable(T, ...)** – Builds a pre-MAET from an attribute table (§5.3). `'specs'` (required) has one spec per attribute, a struct / mapping naming its `column` together with its parameters: `name`, `sigma`, `r`, `exch`, `rel`, `per`, and `period`. Other options: `'pitch'` (a pitch scale; default `'midi'`), `'time'` (`'seconds'` or `'beats'`), `'weights'` (`'velocity'`, `'ones'`, `'duration'`, or `'weight'`), `'roles'`, `'groupBy'`, `'parts'`, `'chords'` (`'bind'` or `'separate'`), and `'chordTolerance'`. The column names `'pitch'`, `'onset'`, `'duration'`, `'soundingDuration'`, `'velocity'`, `'weight'`, `'noteNumber'`, `'part'`, `'measure'`, and `'fermata'` are read with score-specific treatment (the pitch scale, beats or seconds, a grid's onset); any other column is read as it stands, and a categorical column only through `'roles'`. `sigma` must be given for every attribute, and `r` and `exch` wherever an attribute holds several values at an event.
- **audioPeaks(audioFile, ...)** – The spectral peaks of an audio file: their frequencies in Hz and amplitudes in [0, 1] (§5.4). `'sigma'` (in cents; default 0) smooths the spectrum on a log-frequency grid before the peaks are found, merging partials closer than about 2σ, which suits audio with vibrato; steady tones need none. Other options: `'resolution'`, `'rampDuration'`, `'fMin'`, `'fMax'`, `'minProminence'`, `'noiseFactor'`, and `'plot'`.

### 12.2 Pre-MAET objects

- **packPreMaet(pAttr [, wAttr] [, specs])** – Holds a pre-MAET's values, weights, and specs in one object, converting any attribute given per event to its matrix (§6.1). Given an existing pre-MAET, it replaces the parts supplied: `packPreMaet(pm, [], specs2)`.
- **unpackPreMaet(pm)** – Splits a pre-MAET into `pAttr`, `wAttr`, and `specs`.
- **flatSpecs(pAttr [, 'r', r] [, 'rel', rel] [, 'exch', exch] [, 'names', names] [, 'sigma', s] [, 'per', per] [, 'period', P])** – The specs of attributes without level structure, scalars broadcasting across attributes (defaults: `r` = 1, `rel` = false, `exch` = true). The preprocessing functions make flat specs themselves when none are given, so it is needed only to set other values.
- **kernelCov(r, 'differenced', tf [, 'sdValue', sv] [, 'sdInterval', si] [, 'sdShift', ss])** – A matrix-valued kernel covariance for an ordered tuple of r values, or of their first differences, from noise on the values, noise on the intervals, and a common shift (§8.1). Python: `kernel_cov(r, sd_value, sd_interval, sd_shift, differenced=...)`.
- **simplexVertices(N [, edgeLength])** – The N vertices of a regular simplex centred at the origin (edge length 1 by default), one row per level: coordinates for an N-level categorical attribute whose levels are all equally different (§5.3).
- **showPreMaet(pm, ...)** – Prints a pre-MAET as a table, in markdown, LaTeX, or CSV, and returns the text (§6.2). Takes a pre-MAET, whole or in its parts, or a density. Options include `'format'`, `'maxEvents'`, `'maxElements'`, `'decimals'`, `'weights'`, `'caption'`, and `'label'`.
- **readPreMaet(source)**, **writePreMaet(destination, pm, ...)** – Read and write a pre-MAET as CSV (§6.4). `source` may be a path or the CSV text; an empty `destination` returns the text without writing it.

### 12.3 Preprocessing

Each takes a pre-MAET, whole or as `pAttr` and `wAttr` with `'specs'`, and returns one (§7). Omitted specs are taken as flat.

- **differenceEvents(pm, diffOrders [, 'circular', tf])** – Replaces each attribute's values with their k-th differences across events, `diffOrders` giving k per attribute (0 leaves an attribute unchanged). A difference's weight is the product of the weights of the k + 1 events it spans. The first k events are dropped, or with `'circular'` the differences wrap and all N are kept. A differenced attribute must be ordered or hold one value per event (§7.2); the change in σ is announced (§6.3).
- **bindEvents(pm, bindOrders [, 'circular', tf] [, 'rOuter', r] [, 'exchOuter', tf] [, 'relOuter', tf] [, 'groupBy', a] [, 'groupAtol', tol])** – Gathers L consecutive events into one nested event, per attribute; the outer level is read whole, ordered, and absolute by default, and each bound event takes the position of its last event (§7.1). `'groupBy'` binds runs of equal values on one attribute instead.
- **transformAttributes(pm, transforms [, 'sign', tf])** – Maps each attribute's values through a named transform (`'log'`, `'power'`, `'affine'`), a conversion between the pitch scales `'hz'`, `'midi'`, `'cents'`, `'octave'`, `'mel'`, `'bark'`, `'erb'`, and `'greenwood'`, or a function (§7.3). Given a bare array it converts the array: `transformAttributes(f, [], {'hz', 'cents'})`. The `'cents'` scale is absolute, 100 × MIDI (A4 = 6900), not interval cents. `'sign'` transforms magnitudes and inserts a sign attribute after the source.
- **translateAttributes(pm, offsets)** – Shifts each attribute's values by an offset, one per attribute or one per row (§7.4). A uniform shift of a relative attribute changes nothing, and warns.
- **weightEvents(pm, inputAttr, targetAttr, alignAt, shape, 'sd' | 'width', s, 'dropInputAttr', tf [, 'per', per] [, 'period', P] [, 'locate', l] [, 'edges', e])** – Multiplies a window or profile, evaluated on one attribute's values, into another attribute's weights (§7.5). `shape` is γ in [0, 1] (0 a Gaussian, 1 a rectangle), a named exponential or serial-position profile, or a function. `'dropInputAttr'` is required.
- **addSpectra(p, w, mode, ...)** or **addSpectra(pm, mode, ..., 'attribute', a)** – Adds partials to each pitch (§7.6). `mode` is `'harmonic'`, `'stretched'`, `'freqlinear'`, `'stiff'`, or `'custom'`. All modes but `'custom'` take the number of partials N (including the fundamental), then the mode's own parameter where it has one (β, α, or B), then `'powerlaw', ρ` or `'geometric', τ`. `'units'` gives the pitch units per octave (default 1200, cents; 12 for semitones). Each partial's weight is its pitch's weight times its own. Given a pre-MAET, it expands attribute `a` of every event.
- **selectPreMaet(pm, 'attributes', a, 'events', e)** – Keeps the attributes (by index, name, or logical mask) and the events (by index or mask) given, in the order given.
- **bindAttributes(pm, attributes, 'name', n, 'r', r, 'exch', tf, ...)** – Joins several attributes into one whose value at an event is the tuple of their values, in the order listed: the coordinates of a simplex-coded level, say. Parameters on which the inputs agree are inherited; the others must be given.
- **separateAttributes(pm, attribute [, 'names', n])** – Splits one attribute into one attribute per position, the inverse of `bindAttributes`.

### 12.4 Densities and measures

- **buildMaet(pm [, overrides])**, **buildMaet(p, w, 'specs', specs [, overrides])**, or **buildMaet(p, w, sigma, r, rel, per, period)** – Builds the density (§8.1). The six per-attribute parameters, `sigma`, `per`, `period`, `r`, `rel`, and `exch`, may be given as name-value overrides of the specs, in full or for selected attributes (§6.1). The positional form takes a single multiset, or a cell of per-attribute $K_a \times N$ matrices with per-attribute parameter vectors, level structure then coming through `'specs'`.
- **evalMaet(dens, X [, normalize])** – The density at the columns of `X`, which has one row per dimension, r − rel (for a relative attribute, the values above the tuple's first), in per-attribute blocks for several attributes (§8.2). `normalize` is `'none'` (default), `'gaussian'`, or `'pdf'`.
- **simMaet(densX, densY)** or **simMaet(p1, w1, p2, w2, sigma, r, rel, per, period)** – The similarity of two densities (§8.3). `'normalize'` is `'cosine'` (default), `'oneSidedDenom'`, or `'none'`. Also takes pre-MAETs; one density against a list; two lists, pairwise (Python also `mode='cartesian'`); and matrices of multisets, one pair per row, a single row broadcasting (§10.6). `'spectrum'` adds the same partials to both sides in the raw forms.
- **sweepSimMaet(densX, densY, offsets [, 'method', m] [, 'normalize', n])** – The similarity of `densX` against `densY` translated by each column of the A × M matrix `offsets`, in one pass (§8.7). A swept attribute must be absolute and carry no anisotropic kernel covariance; otherwise translate with `translateAttributes` and compare offset by offset.
- **entropyMaet(dens, ...)** – The entropy of a density (§8.4, §11.2). `'method'` is `'shannon'` (default), `'normalized'`, `'differential'`, or `'renyi2'`; the first two need `'nPointsPerDim'`, and `'xMin'` and `'xMax'` on a non-periodic attribute. Other options: `'base'` (default 2), `'spectrum'`, and `'gridLimit'`. A density of zero mass gives `NaN`.
- **massMaet(dens)** – The total mass of a density, the sum of its tuples' weight products: with unit weights, its number of tuples (§8.5).
- **maetCentres(dens)** – The points at which a density places its kernels, one matrix per attribute.
- **plotMaet(dens [, 'method', m] [, ...])** – Plots a density of one to three dimensions as its kernels (default), as sampled points, or as the density itself (§8.6). Options include `'limits'`, `'step'` or `'nodes'`, `'alphaPeak'`, `'alphaFloor'`, `'alphaGamma'`, `'colourGamma'`, `'markerSize'`, and `'markScale'`.
- **sweptSimilarity(pmContext, pmQuery, ...)** or **sweptSimilarity(pContext, wContext, pQuery, wQuery, sigma, r, rel, per, period, ...)** – A similarity profile over sweep values (§8.7). Options: `'sweep'`, `'start'`, `'stop'`, `'step'`, `'align'` (`'query'`, `'both'`, `'window'`, or `'independent'`), `'window'`, `'drop'`, `'queryRef'`, `'locate'`, `'targetAttr'`, `'normalize'` (`'oneSidedDenom'`, the default, `'cosine'`, or `'none'`), `'specs'` or `'exch'` in the raw form (the latter needed for an ordered attribute with a kernel covariance), and the six parameter overrides. Returns the profile, and optionally the offsets and the sweep values (Python `return_offsets=True` and `return_sweep_values=True`, returned in that order after the profile).
- **sweptEntropy(pm, ...)** – An entropy profile over sweep values (§8.7). Takes the sweep, window, `'drop'`, `'locate'`, and `'targetAttr'` options of `sweptSimilarity` and the entropy options of `entropyMaet`, `'method'` defaulting to `'differential'`.
- **sweptMass(pm, ...)** – A total-mass profile over sweep values, with the sweep and window options of `sweptEntropy` (§8.7).

### 12.5 Consonance and harmonicity

These take absolute pitches in cents, except `roughness`, which takes Hz, and transpose internally so that the lowest pitch is 0. They accept `'spectrum'` and matrices of multisets (§10.6).

- **spectralEntropy(p, w, sigma, ...)** – The entropy of the smoothed composite spectrum; lower is more consonant. `'method'` defaults to `'differential'`; the grid methods (`'shannon'`, `'normalized'`) use a grid of `'resolution'` cents (default 1), part of the measure's definition. `'per'` (default false) works on pitch class, every partial folded into one `'period'` (default 1200), as Milne et al. (2017) computed it; there the grid is one period. On pitch, `'normalized'` warns, since its N grows with the spectrum's span.
- **templateHarmonicity(p, w, sigma, ...)** – Cross-correlates the chord's spectrum with a harmonic template, returning hMax, the greatest cosine similarity (Milne, 2013), and hEntropy, the entropy of the cross-correlation (Harrison & Pearce, 2020). `'spectrum'` sets the template's partials and `'chordSpectrum'` the chord's (§9.1). `'method'` sets hEntropy's form: `'differential'` (default), `'shannon'`, or `'normalized'` ($H/\log_2 N$, which warns on pitch, since N grows with the chord's span). `'per'` and `'period'` as for `spectralEntropy`.
- **tensorHarmonicity(p, w, sigma, ...)** – The density of a harmonic series' relative tensor at the chord's intervals above its lowest pitch (§9.1).
- **roughness(f, w, ...)** – Sensory roughness from frequencies in Hz (Plomp & Levelt, 1965; Sethares, 1993). Options: `'pNorm'` (default 1) and `'average'` (default false).
- **virtualPitches(p, w, sigma, ...)** – The whole template cross-correlation profile, whose peaks are candidate fundamentals. With `'per'` true, one value per pitch class, `0 : resolution : period − resolution`.

### 12.6 Balance and evenness

These take points on a cycle: pitch classes in a period, or onsets in a rhythmic cycle. With `sigma > 0`, `balanceCircular` and `evennessCircular` give the expected value under Gaussian jitter of the points, and optionally its standard deviation (§10.8).

- **dftCircular(p, w, period)** – The discrete Fourier transform of points on a circle: coefficients and magnitudes, the k = 0 coefficient first.
- **dftCircularSimulate(p, w, period, sigma, ...)** – The mean and standard deviation of each coefficient's magnitude under Gaussian jitter, by Monte Carlo (`'nDraws'`, default 10000; `'rngSeed'`). A third output gives every draw.
- **balanceCircular(p, w, period [, sigma])** – Balance, 1 − |F(0)|, in [0, 1]: 1 when the points' centre of gravity is the centre of the circle (Milne et al., 2017). Under jitter a perfectly balanced set scores slightly below 1, since the magnitude of a jittered centroid is positive on average. Python: `balance`.
- **evennessCircular(p, period [, sigma])** – Evenness, |F(1)|, in [0, 1]: 1 when the points are equally spaced. Unweighted. Python: `evenness`.

### 12.7 Scale and rhythm structure

At `sigma = 0`, `coherence`, `sameness`, and `nTupleEntropy` take integer positions and an integer period; `edges`, `projCentroid`, and `meanOffset` also take query points `x`, at which to evaluate between positions.

- **coherence(p, period [, sigma])** – The coherence quotient (Rothenberg, 1978; Carey, 2002), 1 when larger generic spans always have larger specific sizes; `'strict', false` uses Rothenberg's non-strict propriety. With `sigma > 0` each comparison becomes a probability under Gaussian uncertainty, of the positions (`'sigmaSpace', 'position'`, the default) or of each interval independently (`'interval'`), and the quotient may fall below 0. The number of failures is a second output.
- **sameness(p, period [, sigma])** – The sameness quotient (Carey, 2002, 2007), 1 when each specific interval size belongs to one generic span; `sigma` and `'sigmaSpace'` as for `coherence`.
- **nTupleEntropy(p, period [, n])** – The entropy of the n-tuples of consecutive step sizes (Milne & Dean, 2016); n = 1 (the default) is step-size entropy. Options: `'sigma'` (default 0; `sigma > 0` gives the smoothed extension of Milne, 2024), `'sigmaSpace'`, `'method'` (default `'normalized'`), `'base'`, and `'nPointsPerDim'`. It runs `differenceEvents`, `bindEvents`, and `entropyMaet` (§7.7); call them directly for non-integer, weighted, or non-periodic material, or for the density itself.
- **circApm(p, w, period, ...)** – The circular autocorrelation phase matrix, after Eck (2006), with its column sums, a metrical weight profile, and its row sums, the circular autocorrelation. `'decay'` adds exponential decay.
- **edges(p, w, period [, x])** – Edges: the points convolved with the derivative of a von Mises kernel, whose width `'kappa'` sets, giving absolute and signed edge weights.
- **projCentroid(p, w, period [, x] [, sigma])** – The projection of the circular centroid onto each position, with the centroid's magnitude and phase. With `sigma > 0`, the expected projection under jitter, in closed form: the deterministic value scaled by $\exp(-2\pi^2\sigma^2/P^2)$. Its magnitude is that of the mean centroid, where `balanceCircular` averages the centroid's magnitude.
- **meanOffset(p, w, period [, x])** – At each position, the weighted balance of upward over downward arcs to every point, normalized by the period: average pitch height (Huron, 2008) made a function of position, and related to mode height (Hearne, 2020; Tymoczko, 2023).
- **markovS(p, w, period [, S])** – An optimal S-step Markov predictor (default S = 3), after David Bulger: at each position, the average weight of the positions that share its next S steps.

### 12.8 Sequences

- **continuity(seq, x, sigma [, 'w', w] [, 'mode', m] [, 'theta', θ])** – The expected length and signed magnitude of the same-direction run leading up to each query point under Gaussian pitch uncertainty; their ratio is a trend slope. `'mode'`, `'strict'` (θ = 0) or `'lenient'` (θ = −1), sets when a run breaks. Weights scale each interval's contribution by the product of its two events' weights.

### 12.9 Diagnostics, defaults, and estimates

- **explainDispatch(densX, densY, ...)** or **explainDispatch(dens, nQueries, ...)** – The route a similarity or evaluation call would take, and why, without running it (§11.1).
- **mptDefaults(...)** – Inspects and sets the toolbox-wide defaults (§11.4). Python: `set_default`, `get_default`, `get_defaults`, `reset_defaults`, and `show_defaults`.
- **estimateCompTime(...)** – Estimates how long a call will take, from a short benchmark.

---

## 13. Demo scripts

The demos are in `matlab/demos/` and `python/demos/`, with adjustable parameters at the top of each, and give the same results in the two languages; the Python plotting demos need `matplotlib`, and the audio demo `soundfile`. Start with `demo_0_startHere` (Python `demo_0_start_here.py`), which runs no analysis but prints a guide to the others: a route for a beginner, how the demos fit together, and the demos by topic. The first on that route, `demo_overview`, tours every major function family and ends each section with pointers to the demos that go further.

| MATLAB | Python | What it shows | Based on |
|:---|:---|:---|:---|
| `demo_0_startHere` | `demo_0_start_here.py` | A printed guide to the demos | – |
| `demo_overview` | `demo_overview.py` | A tour of every major function family: SPCS, entropy, total mass, r and the three flags plotted, a multi-attribute motif search, a moving window's share of each of two triads, and the consonance and structural measures | – |
| `demo_audioAnalysis` | `demo_audio_analysis.py` | Peaks from audio in two passes, then similarity, harmonicity, roughness, and virtual pitches | – |
| `demo_batchProcessing` | `demo_batch_processing.py` | Features of a table of experimental trials in batched calls, contrasted with a list of pre-MAETs and a translation sweep | – |
| `demo_edoApprox` | `demo_edo_approx.py` | PCS of n-EDOs against a JI chord | Milne et al. (2011), Ex. 6.3 / Fig. 4 |
| `demo_genChainPcs` | `demo_gen_chain_pcs.py` | PCS of generator-chain tunings as the generator is swept | Milne et al. (2011), Ex. 6.4–6.5 / Figs. 5–7 |
| `demo_triadSpcsGrid` | `demo_triad_spcs_grid.py` | SPCS of every 12-EDO triad with a fifth | Milne et al. (2011), Fig. 3 |
| `demo_triadConsonance` | `demo_triad_consonance.py` | Five consonance measures over a grid of triads | – |
| `demo_virtualPitches` | `demo_virtual_pitches.py` | Virtual-pitch profiles of example chords | – |
| `demo_maetPlots` | `demo_maet_plots.py` | Every combination of r, [rel], [per], and [exch] on the diatonic scale, plotted by each method of `plotMaet` | – |
| `demo_helixBlend` | `demo_helix_blend.py` | A motif swept against a stream in pitch class and register, from pitch-class to pitch-height similarity | – |
| `demo_preprocessing` | `demo_preprocessing.py` | Every preprocessing operation on a chorale fragment, and how they compose | – |
| `demo_scoreWorkflow` | `demo_score_workflow.py` | From a score file to a result, end to end | – |
| `demo_scoreGrid` | `demo_score_grid.py` | Sampling a score on a grid: the step, the weightings, and empty slices | – |
| `demo_scoreCategoricals` | `demo_score_categoricals.py` | A categorical column encoded three ways, and what each asks of a re-voiced chord | – |
| `demo_preMaetIo` | `demo_pre_maet_io.py` | A pre-MAET shown, written, and read back: markdown, LaTeX, and CSV, with overrides, nesting, NA, and a kernel covariance | – |
| `demo_repetitionHandling` | `demo_repetition_handling.py` | Repeated pitches under interval and tempo invariance: excise, prolong, or count | – |
| `demo_tempoInvariance` | `demo_tempo_invariance.py` | Graded tempo invariance by a kernel covariance, against the exact relative quotient | – |
| `demo_sweptSimilarity` | `demo_swept_similarity.py` | `sweptSimilarity` in depth, every alignment on one melody holding four statements related to a query | – |
| `demo_sigmaSpace` | `demo_sigma_space.py` | Soft sameness, coherence, and n-tuple entropy, position against interval uncertainty | – |
| `demo_probeTone` | `demo_probe_tone.py` | Probe-tone profiles against the ratings of Krumhansl and Kessler (1982), recency weighting and inharmonic spectra on a melody, and continuity | Milne (2026b), Fig. 4 |
| `demo_softeningEquivalences` | `demo_softening_equivalences.py` | Softening transposition, octave, and reordering equivalence by pairing attributes | Milne (2026a), Online Supplement |
| `demo_rhythmTensors` | `demo_rhythm_tensors.py` | The son clave against five timelines: similarity, densities, entropy as complexity, and n-tuple entropy by circular differencing and binding | Milne & Dean (2016) |
| `demo_dftCircularSimulate` | `demo_dft_circular_simulate.py` | Balance and evenness under positional jitter | – |
| `demo_dispatchAndKernelControls` | `demo_dispatch_and_kernel_controls.py` | The performance controls: routes, nested contraction, one-pass sweeps, truncation, precision, defaults, and the entropy estimators | – |

The article's analyses are in `matlab/demos/jmm/` and `python/demos/jmm/`: the windowed spectral entropy, voicing encodings, tuple size, and cadence localization of BWV 347; the motif analyses of *Acknowledgement*; the differencing, texture, and lag analyses of *Piano Phase*; and the supplied harmonic parse. Each folder's `README.md` maps the scripts to the article.

---

## 14. References

Carey, N. (2002). On coherence and sameness, and the evaluation of scale candidacy claims. *Journal of Music Theory*, 46(1/2), 1–56.

Carey, N. (2007). Coherence and sameness in well-formed and pairwise well-formed scales. *Journal of Mathematics and Music*, 1(2), 79–98.

Dean, R. T., Milne, A. J., & Bailes, F. (2019). Spectral pitch similarity is a predictor of perceived change in sound- as well as note-based music. *Music & Science*, 2, 1–14.

Eck, D. (2006). Beat tracking using an autocorrelation phase matrix. *Proceedings of the International Computer Music Conference (ICMC)*.

Eerola, T., & Lahdelma, I. (2021). The anatomy of consonance/dissonance: Evaluating acoustic and cultural predictors across multiple datasets with chords. *Music & Science*, 4, 20592043211030471.

Eitel, M., Ruth, N., Harrison, P., Frieler, K., & Müllensiefen, D. (2024). Perception of chord sequences modeled with prediction by partial matching, voice-leading distance, and spectral pitch-class similarity: A new approach for testing individual differences in harmony perception. *Music & Science*, 7.

Harrison, P. M. C., & Pearce, M. T. (2020). Simultaneous consonance in music perception and composition. *Psychological Review*, 127(2), 216–244.

Hearne, L. M. (2020). *The Cognition of Harmonic Tonality in Microtonal Scales*. PhD thesis, Western Sydney University.

Hearne, L. M., Dean, R. T., & Milne, A. J. (2025). Acoustical and cultural explanations for contextual tonal stability. *Music Perception*, 43(3).

Homer, S., Harley, N., & Wiggins, G. (2024). Modelling of musical perception using spectral knowledge representation. *Journal of Cognition*, 7.

Huron, D. (2008). A comparison of average pitch height and interval size in major- and minor-key themes: Evidence consistent with affect-related pitch prosody. *Empirical Musicology Review*, 3, 59–63.

Krumhansl, C. L., & Kessler, E. J. (1982). Tracing the dynamic changes in perceived tonal organization in a spatial representation of musical keys. *Psychological Review*, 89(4), 334–368.

Milne, A. J. (2013). *A Computational Model of the Cognition of Tonality*. PhD thesis, The Open University.

Milne, A. J., Sethares, W. A., Laney, R., & Sharp, D. B. (2011). Modelling the similarity of pitch collections with expectation tensors. *Journal of Mathematics and Music*, 5(1), 1–20.

Milne, A. J., Laney, R., & Sharp, D. B. (2015). A spectral pitch class model of the probe tone data and scalic tonality. *Music Perception*, 32(4), 364–393.

Milne, A. J., & Holland, S. (2016). Empirically testing Tonnetz, voice-leading, and spectral models of perceived triadic distance. *Journal of Mathematics and Music*, 10(1), 59–85.

Milne, A. J., & Dean, R. T. (2016). Computational creation and morphing of multilevel rhythms by control of evenness. *Computer Music Journal*, 40(1), 35–53.

Milne, A. J., Laney, R., & Sharp, D. B. (2016). Testing a spectral model of tonal affinity with microtonal melodies and inharmonic spectra. *Musicae Scientiae*, 20(4), 465–494.

Milne, A. J., Bulger, D., & Herff, S. A. (2017). Exploring the space of perfectly balanced rhythms and scales. *Journal of Mathematics and Music*, 11(2–3), 101–133.

Milne, A. J. (2019). XronoMorph: Investigating paths through rhythmic space. In S. Holland, T. Mudd, K. Wilkie-McKenna, A. P. McPherson, & M. M. Wanderley (Eds.), *New directions in music and human-computer interaction* (pp. 95–113). Springer Series on Cultural Computing. Springer.

Milne, A. J., & Herff, S. A. (2020). The perceptual relevance of balance, evenness, and entropy in musical rhythms. *Cognition*, 203, 104233.

Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of rhythmic structure on tapping accuracy. *Attention, Perception, & Psychophysics*, 85, 2673–2699.

Milne, A. J., Smit, E. A., Sarvasy, H. S., & Dean, R. T. (2023). Evidence for a universal association of auditory roughness with musical stability. *PLOS ONE*, 18(9), e0291642.

Milne, A. J. (2024). Commentary on Buechele, Cooke, & Berezovsky (2024): Entropic models of scales and some extensions. *Empirical Musicology Review*, 19(2), 144–153.

Milne, A. J. (2026a). Multi-attribute expectation tensors: Smooth densities for modelling musical structure, complexity, and similarity. Manuscript submitted for publication.

Milne, A. J. (2026b). The Music Perception Toolbox: Analytical methods for pitch and rhythm similarity, consonance, complexity, and structure. *Transactions of the International Society for Music Information Retrieval*, 9(1), 526–543.

Plomp, R., & Levelt, W. J. M. (1965). Tonal consonance and critical bandwidth. *Journal of the Acoustical Society of America*, 38(4), 548–560.

Rothenberg, D. (1978). A model for pattern perception with musical applications. Part I. *Mathematical Systems Theory*, 11, 199–234.

Sethares, W. A. (1993). Local consonance and the relationship between timbre and scale. *Journal of the Acoustical Society of America*, 94(3), 1218–1228.

Sethares, W. A., Milne, A. J., Tiedje, S., Prechtl, A., & Plamondon, J. (2009). Spectral tools for Dynamic Tonality and audio morphing. *Computer Music Journal*, 33(2), 71–84.

Smit, E. A., Milne, A. J., Dean, R. T., & Weidemann, G. (2019). Perception of affect in unfamiliar musical chords. *PLOS ONE*, 14(6), e0218570.

Tymoczko, D. (2023). *Tonality: An Owner's Manual*. Oxford University Press.

Wing, A. M., & Kristofferson, A. B. (1973). Response delays and the timing of discrete motor responses. *Perception & Psychophysics*, 14(1), 5–12.

---

## 15. Citation

If you use this toolbox in published work, please cite the toolbox article, Milne (2026b), or, where your work rests mainly on expectation tensors themselves, the article that introduced them, Milne, Sethares, Laney, and Sharp (2011), instead or as well. Please also cite the software itself, by its Zenodo DOI (see `CITATION.cff`). For the balance and evenness measures, also cite Milne, Bulger, and Herff (2017) and Milne and Herff (2020); for the rhythmic predictors (`circApm`, `edges`, `projCentroid`, `meanOffset`, and `markovS`), Milne, Dean, and Bulger (2023). The full references are in §14.

---

## Acknowledgements

This work was supported, in part, by an Australian Research Council Discovery Early Career Researcher Award (project number DE170100353) funded by the Australian Government.
