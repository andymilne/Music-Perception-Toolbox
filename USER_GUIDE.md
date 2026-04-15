# Music Perception Toolbox — User Guide

Andrew J. Milne, Western Sydney University

---

## Contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Conceptual overview](#3-conceptual-overview)
4. [API conventions: MATLAB vs Python](#4-api-conventions-matlab-vs-python)
5. [Quick start](#5-quick-start)
6. [Function reference](#6-function-reference)
7. [Worked examples](#7-worked-examples)
8. [Demo scripts](#8-demo-scripts)
9. [Known simplifications and future directions](#9-known-simplifications-and-future-directions)
10. [References](#10-references)
11. [Citation](#11-citation)

---

## 1. Introduction

The Music Perception Toolbox is an open-source toolbox — available in MATLAB and Python — for computing psychoacoustically motivated similarity, consonance, and structural measures on pitch collections and rhythmic patterns.

The toolbox implements several original theoretical frameworks. Expectation tensors (Milne, Sethares, Laney, & Sharp, 2011) provide a unified framework — grounded in probability theory and Riemannian geometry — for quantifying the similarity of pitch collections and rhythmic patterns under different assumptions about perceptual equivalence and uncertainty, and at different orders of structural complexity (from individual pitches through dyads, triads, and beyond). The balance and evenness measures (Milne, Bulger, & Herff, 2017) draw on the discrete Fourier transform to characterise the distributional properties of scales and rhythms, including a novel class of perfectly balanced patterns. Additional structural and perceptual features — coherence, sameness, edge detection, projected centroid, mean offset, and Markov prediction — have been developed and empirically validated for modelling rhythmic perception and performance (Milne & Herff, 2020; Milne, Dean, & Bulger, 2023).

These measures have proven effective predictors across a range of music cognition contexts, including tonal fit and stability in conventional and microtonal tuning systems (Milne, Laney, & Sharp, 2015, 2016; Homer, Harley, & Wiggins, 2024; Hearne, Dean, & Milne, 2025), perceived consonance and affect (Smit et al., 2019; Harrison & Pearce, 2020; Eerola & Lahdelma, 2021), individual differences in harmony perception (Eitel, Ruth, Harrison, Frieler, & Müllensiefen, 2024), and rhythmic complexity and tapping accuracy (Milne & Herff, 2020; Milne, Dean, & Bulger, 2023). They have also guided the design of music-computing interfaces (Sethares, Milne, Tiedje, Prechtl, & Plamondon, 2009; Milne & Dean, 2016; Milne, 2019).

The toolbox operates on weighted multisets of pitches or time points. These can be entered directly (e.g., as cents values or integer pulse positions) or derived from audio recordings via the `audioPeaks` (`audio_peaks` in Python) function, which extracts spectral peaks and their amplitudes from audio files. The `convertPitch` (`convert_pitch`) function converts between seven pitch and frequency scales (Hz, MIDI, cents, mel, Bark, ERB-rate, Greenwood).

The toolbox addresses two broad domains:

**Pitch and rhythm via expectation tensors.** Expectation tensors and their cosine similarities provide a general framework for quantifying the similarity of any two multisets of weighted points — whether those points are pitches, pitch classes, time points, or time classes. For pitch, spectral enrichment (adding harmonics via `addSpectra`) yields spectral pitch class similarity (SPCS), a powerful predictor of perceived tonal fit and similarity. Related functions compute consonance via spectral entropy, template harmonicity, tensor harmonicity, and sensory roughness. The only component of this framework that is specific to pitch is the spectral enrichment itself; the underlying tensor machinery can apply equally to rhythmic patterns.

**Scale and rhythm structure.** A suite of functions computes structural and perceptual features of multisets of points on a circle, applicable to both pitches and positions (e.g., time onsets). These include Fourier-based measures (balance and evenness), interval-structure measures (coherence, sameness, n-tuple entropy), and further predictors (edge detection, projected centroid, mean offset, autocorrelation phase matrices, Markov prediction).

Version 2.0.0 is a major rewrite. Analytical methods have replaced the previous numerical approximations wherever feasible: in v1, analytical computation was available only for the cosine similarity inner product (`cosSimExpTens`, by David Bulger); in v2, this analytical approach has been extended to the construction and evaluation of individual expectation tensors via `buildExpTens` and `evalExpTens`, eliminating the discretization error and resolution trade-offs of v1's grid-based `expectationTensor`. The `cosSimExpTens` computation itself has also been substantially optimized — the original double loop over r-ad combinations has been replaced by fully vectorized operations over pre-calculated r-ads, with automatic memory-aware chunking for large problems. Entropy computation remains discretized, as the differential entropy of a Gaussian mixture has no known closed-form solution. The expectation tensor core has been restructured around precomputed density objects (`buildExpTens`), and the toolbox has been substantially expanded. Accessibility has also been a priority: every function now includes a full help text with runnable examples, this User Guide covers the conceptual foundations and provides a complete function reference for both languages, and nine demo scripts (in both MATLAB and Python) cover all major use cases. See `CHANGELOG.md` for a full list of changes and `MIGRATION.md` for guidance on updating v1 code.

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

The MATLAB implementation requires MATLAB R2019b or later (for the `arguments` block syntax used by most functions). It has no external dependencies — unlike v1, which required the [Sparse Array Toolbox](https://github.com/andymilne/Sparse-Array-Toolbox), v2 is entirely self-contained. Core functions are in the `matlab/` folder; demo scripts are in `matlab/demos/` and example audio files are in `matlab/audio/`. Only the `matlab/` folder needs to be added to the path.

### Python

1. Download or clone the repository from [GitHub](https://github.com/andymilne/Music-Perception-Toolbox).
2. Install from the local `python/` directory:

```bash
pip install ./python
```

For audio file support (spectral peak extraction via `audio_peaks`):

```bash
pip install ./python[audio]
```

The Python implementation requires Python 3.10 or later. Core dependencies (NumPy, SciPy) are installed automatically. The optional `soundfile` library is required only for `audio_peaks`.

All functions are accessible from the top-level `mpt` namespace:

```python
import mpt
s = mpt.cos_sim_exp_tens_raw(...)
```

The Python implementation uses snake_case naming and a few other syntactic differences from the MATLAB version; see [Section 4](#4-api-conventions-matlab-vs-python) for the full mapping.

---

## 3. Conceptual overview

### 3.1 Expectation tensors

An expectation tensor represents the distribution of r-tuples of pitches (or time points) that a listener expects to perceive, given a weighted multiset and a model of perceptual uncertainty. Formally, it is an unnormalized Gaussian mixture density: a weighted sum of Gaussian kernels centred at all ordered r-tuples drawn from the multiset, with standard deviation σ modelling perceptual uncertainty.

Given a multiset **p** = (p₁, p₂, ..., pₙ) with weights **w** = (w₁, w₂, ..., wₙ), the r-ad expectation tensor density at a query point **x** is:

> f(**x**) = Σⱼ wⱼ · exp(−(**x** − **cⱼ**)ᵀ M (**x** − **cⱼ**) / (2σ²))

where the sum is over all ordered r-tuples, **cⱼ** is the j-th tuple, and M is a quadratic form matrix determined by the mode (see below).

The toolbox implements this in two steps: `buildExpTens` precomputes the tuple indices and weight products into a density struct, and `evalExpTens` evaluates the density at query points. The cosine similarity between two such densities — which quantifies how similar the two weighted multisets are — is computed by `cosSimExpTens`. Crucially, the inner product underlying this cosine similarity has an analytical solution (a finite double sum of Gaussian kernel evaluations), so `cosSimExpTens` computes it exactly without discretization. This analytical computation was present in v1 (due to David Bulger); what is new in v2 is that individual tensor construction and evaluation (`buildExpTens` and `evalExpTens`) are also analytical, whereas v1's `expectationTensor` required discretization onto a grid. (Note, however, that the *entropy* of a Gaussian mixture density has no known closed-form solution, so the entropy functions `entropyExpTens` and `spectralEntropy` still use grid discretization — see [Section 9](#9-known-simplifications-and-future-directions).)

The term "expectation tensor" reflects two ideas: (a) the density represents the *expected* perceptual distribution of r-ads given a weighted multiset of pitches or time points, smoothed by perceptual uncertainty σ; and (b) "tensor" refers to the fact that the discretized density is a rank-r array (an r-dimensional grid of values), with its domain being the r-fold product of the pitch (or time) space with itself. This is "tensor" in the numerical/data sense (a multidimensional array) rather than the strict algebraic sense (a multilinear map with specific transformation properties).

#### The four modes

The expectation tensor has two logical flags that control its geometry:

**Periodic (isPer).** When true, the pitch (or time) line is wrapped into a circular domain of circumference `period`, so that values differing by the period are identified. For pitch, this implements pitch-class equivalence (e.g., with period = 1200, the pitch line is wrapped into a circle where 0 cents and 1200 cents are the same point). For rhythmic patterns, the period is the cycle length. When false, values are treated as points on an unbounded line.

**Relative (isRel).** When true, the density is invariant under transposition: shifting all values by the same amount does not change the density. Mathematically, this is achieved by projecting out the mean direction in R^r, reducing the effective dimensionality from r to r − 1. In the formula above, the quadratic form matrix M becomes the projection matrix I − **11**ᵀ/r (where **1** is the all-ones vector); the resulting quadratic form Q(**d**) = Σdᵢ² − (Σdᵢ)²/r is the squared distance in the quotient space R^r / R·**1** under the Riemannian metric induced by the standard Euclidean inner product. When false (absolute), M = I (the identity matrix) and the density depends on the actual values, not just intervals.

When both isPer and isRel are true, the cosine similarity computation (`cosSimExpTens`) uses an algebraically equivalent pairwise form of Q with wrapped pairwise differences — Q(**d**) = Σᵢ<ⱼ wrap(dᵢ − dⱼ)²/r — to maintain exact transposition invariance on the circle. This is necessary because the component-wise periodic wrapping is nonlinear and can otherwise introduce ±period artifacts in the pairwise differences between components of the wrapped difference vector.

The four combinations (absolute non-periodic, absolute periodic, relative non-periodic, relative periodic) cover a range of use cases. The choice depends on the question being asked: do we care about absolute positions or only intervals? Do we treat values a period apart as equivalent?

### 3.2 Spectral enrichment

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

### 3.3 (Spectral) pitch (class) similarity

The cosine similarity of two expectation tensor densities quantifies how similar two collections of weighted elements are — whether those elements are pitches, pitch classes, time points, or time classes. In the pitch domain, different combinations of spectral enrichment and periodicity yield a family of named similarity measures:

- **Pitch similarity (PS):** cosine similarity of absolute non-periodic tensors, without spectral enrichment. Compares two multisets of pitches (which can represent fundamental pitches without spectral content) on an unbounded pitch line (Milne, Sethares, Laney, & Sharp, 2011).
- **Pitch class similarity (PCS):** cosine similarity of absolute periodic tensors, without spectral enrichment. As above, but with octave (or other period) equivalence (Milne et al., 2011).
- **Spectral pitch similarity (SPS):** cosine similarity of absolute non-periodic tensors, with spectral enrichment via `addSpectra`. The term "spectral" indicates that harmonics are added to each pitch; "pitch" (without "class") indicates a non-periodic domain (Dean, Milne, & Bailes, 2019).
- **Spectral pitch class similarity (SPCS):** cosine similarity of absolute periodic tensors, with spectral enrichment. "Pitch class" indicates a periodic domain — pitches an octave apart are identified (Milne, Laney, & Sharp, 2015).

In all four cases, the standard parameters are r = 1 (monad tensor) and isRel = false (absolute). Each multiset is mapped to a density over pitch (or pitch class), and the cosine similarity between the two densities quantifies how similar they are. For Western-enculturated participants, SPCS has consistently outperformed SPS at modelling empirical results (e.g., probe tone ratings, perceived tonal fit), which is why the periodic form is the more commonly used measure. SPCS has also been validated as a predictor of perceived triadic distance (Milne & Holland, 2016), perceived change in sound-based music (Dean, Milne, & Bailes, 2019), affect in unfamiliar chords (Smit et al., 2019), consonance across multiple datasets (Eerola & Lahdelma, 2021), cross-cultural tonal stability (Milne, Smit, Sarvasy, & Dean, 2023), individual differences in harmony perception (Eitel et al., 2024), spectral models of musical knowledge (Homer et al., 2024), and contextual tonal stability in microtonal scales (Hearne, Dean, & Milne, 2025).

It is also possible to use higher-order tensors. Setting r = 2 with isRel = true gives a transposition-invariant density over intervals; r = 3 with isRel = true gives a density over ordered triples of intervals; and so on. These capture different aspects of the similarity and are useful in specific contexts — for example, the EDO approximation examples in Milne et al. (2011, Examples 6.3–6.5) use relative dyad tensors (r = 2, isRel = true) to produce one-dimensional interval-based approximations. Higher-order relative tensors are also central to `tensorHarmonicity`, which builds an r = K tensor (where K is the chord cardinality) from a harmonic series template and queries it at the chord's intervals.

These measures are not restricted to pitch. The same cosine similarity can be computed between two sets of time points (rhythmic patterns), yielding a measure of rhythmic similarity. Whether the "spectral" and "class" qualifiers apply depends on the domain and the parameter choices.

Typical SPCS parameters:
- σ = 6–15 cents (perceptual uncertainty; 10 is typical)
- r = 1, isRel = false, isPer = true, period = 1200
- Spectral enrichment: e.g., `'harmonic', 12, 'powerlaw', 1`

### 3.4 Consonance and harmonicity

The toolbox provides several complementary measures of consonance. These measures have been used individually and in combination to predict consonance ratings, perceived affect, and tonal stability across a variety of musical contexts including unfamiliar chords and microtonal tuning systems (Smit et al., 2019; Harrison & Pearce, 2020; Eerola & Lahdelma, 2021; Milne, Smit, Sarvasy, & Dean, 2023; Hearne, Dean, & Milne, 2025).

- **Spectral entropy** (`spectralEntropy`) measures the disorder of a chord's composite spectrum. Greater spectral overlap between partials (after smoothing) produces lower entropy, indicating greater consonance. This is a property of a single weighted multiset, unlike SPCS which compares two multisets (Milne, Bulger, & Herff, 2017; Smit et al., 2019).
- **Template harmonicity** (`templateHarmonicity`) cross-correlates a chord's spectrum with a harmonic template, returning both the maximum similarity (hMax; Milne, 2013) and the entropy of the cross-correlation (hEntropy; Harrison & Pearce, 2020). These measure how well the chord's partials align with *some* harmonic series (Milne, Laney, & Sharp, 2016).
- **Tensor harmonicity** (`tensorHarmonicity`) evaluates the expectation tensor of a harmonic series at the chord's intervals (measured from the lowest pitch to each of the remaining pitches), measuring how likely the chord's intervals are to occur within a single harmonic series (Smit et al., 2019).
- **Roughness** (`roughness`) computes sensory roughness from the beating of close partial pairs, using Sethares' (1993) parameterization of Plomp and Levelt's (1965) empirical dissonance curve. The implementation extends Sethares' original model by allowing pairwise roughnesses to be combined via a configurable p-norm (Mashinter, 2006; Parncutt, 2006), rather than a simple sum, and by supporting optional averaging over the number of partial pairs.
- **Virtual pitches** (`virtualPitches`) returns the full cross-correlation profile (pitch-indexed weights) from which template harmonicity extracts summary statistics. Peaks indicate strong virtual pitches — candidate fundamentals (Milne, Laney, & Sharp, 2016).

Template harmonicity and tensor harmonicity measure harmonicity in fundamentally different ways — see [Section 6.3](#63-consonance-and-harmonicity) for a detailed comparison.

### 3.5 Balance and evenness (Fourier-based measures)

Several functions compute properties of multisets of points distributed around a circle, applicable to both pitches and time points. The DFT here is computed directly from the positions of the points on the circle (mapping each point to the unit circle and taking the Fourier transform), rather than from an indicator vector over a discretized grid. The mathematical foundations — including the relationship between the DFT coefficients and balance, evenness, and perfect balance — are developed in Milne, Bulger, & Herff (2017). These measures have been validated as predictors of rhythm recognition and preference (Milne & Herff, 2020) and have informed the design of algorithmic rhythm generators (Milne & Dean, 2016; Milne, 2019).

- **Balance** (`balanceCircular`) measures how evenly the mass is distributed around the circle (1 = perfectly balanced, with the centre of gravity at the circle's centre; 0 = all weight at one point) (Milne, Bulger, & Herff, 2017).
- **Evenness** (`evennessCircular`) measures closeness to equal spacing (1 = equally spaced; 0 = maximally uneven) (Milne, Bulger, & Herff, 2017).
- The **DFT** (`dftCircular`) underlies both measures; its higher-order coefficients capture finer distributional structure (Milne, Bulger, & Herff, 2017; Milne & Herff, 2020).

### 3.6 Scale and rhythm structure

The following functions compute structural and perceptual features of multisets of points distributed around a periodic cycle, applicable to both scales and rhythms. These features have been validated as predictors of perceived rhythmic complexity and liking (Milne & Herff, 2020) and of tapping accuracy across a wide variety of rhythmic structures (Milne, Dean, & Bulger, 2023).

**Integer-position features.** These functions take integer positions within a discrete cycle of integer length (an equal division of the period). They are applicable to scales in an equal temperament (e.g., 12-EDO) and to rhythms quantized to a metrical grid:

- **Coherence** (`coherence`) — the coherence quotient (Balzano, 1982; Carey, 2002; Rothenberg, 1978) measures how consistently the ordering of intervals by generic span (number of scale steps) matches their ordering by specific size (number of chromatic steps). A coherent scale or rhythm is one where larger generic spans always correspond to larger specific sizes — hearing an interval's size uniquely identifies how many scale steps it spans.
- **Sameness** (`sameness`) — the sameness quotient (Carey, 2002, 2007) measures the proportion of interval sizes that are unambiguous: each specific size belongs to exactly one generic span. A scale with high sameness has a transparent relationship between its chromatic and generic interval structure.
- **n-tuple entropy** (`nTupleEntropy`) — entropy of the distribution of consecutive step-size sequences, capturing sequential predictability (Milne & Dean, 2016). When n = 1, this is IOI / step-size entropy. Gaussian smoothing is available for optimization applications (Milne, 2024).
- **Circular autocorrelation phase matrix** (`circApm`) — decomposes the circular autocorrelation by lag and phase, yielding a metrical weight profile. Adapted from Eck's (2006) non-circular autocorrelation phase matrix to the circular (periodic) case by Milne, Dean, & Bulger (2023).
- **Markov prediction** (`markovS`) — returns the optimal S-step lookahead prediction at each position in the cycle (Milne, Dean, & Bulger, 2023).

**Continuous-position features.** These functions accept non-integer event positions and can be evaluated at any position around the circle. They are applicable to scales and rhythms in any tuning or timing, not just equal divisions:

- **Edge detection** (`edges`) — adapts the standard edge-detection technique from image processing to the circular domain, identifying sharp transitions between event-dense and event-sparse regions of the cycle via the first derivative of a von Mises kernel (Milne, Dean, & Bulger, 2023).
- **Projected centroid** (`projCentroid`) — projects the circular centre of gravity onto each angular position, giving a position-level generalization of the rhythm- or pitch-class-set-level balance measure (Milne, Dean, & Bulger, 2023).
- **Mean offset** (`meanOffset`) — for each position, the net upward arc to all events. In a pitch-class context, this formalizes and generalizes Huron's (2008) "average pitch height," making the position-dependence explicit: it returns a value for every position around the circle, including non-scale-tone positions, capturing the "brightness" or "darkness" of a mode as seen from each chromatic position. The related concept of "mode height" is used by Hearne (2020) and Tymoczko (2023). Introduced as a rhythmic predictor in Milne, Dean, & Bulger (2023).


---

## 4. API conventions: MATLAB vs Python

The MATLAB and Python implementations are functionally identical: the same inputs produce the same outputs (to floating-point precision). The differences are syntactic, following the conventions of each language. This section is the primary reference for Python users. The Quick Start (Section 5) shows both languages side by side; the Function Reference (Section 6) and Worked Examples (Section 7) use MATLAB syntax, from which Python equivalents can be derived using the rules below.

### Naming

MATLAB uses camelCase; Python uses snake_case. A few names were shortened where the MATLAB name included a suffix that is redundant in a package namespace:

| MATLAB | Python |
|:---|:---|
| `balanceCircular` | `balance` |
| `evennessCircular` | `evenness` |
| `dftCircular` | `dft_circular` |
| `circApm` | `circ_apm` |
| `markovS` | `markov_s` |

All other names follow the mechanical camelCase → snake_case rule (e.g., `cosSimExpTens` → `cos_sim_exp_tens`, `addSpectra` → `add_spectra`).

### Calling conventions

| Concept | MATLAB | Python |
|:---|:---|:---|
| Default (uniform) weights | `[]` | `None` |
| Boolean flags | `true` / `false` | `True` / `False` |
| Spectrum arguments | Cell array: `{'harmonic', 12, 'powerlaw', 1}` | List: `['harmonic', 12, 'powerlaw', 1]` |
| Name-value pairs | `'name', value` | `name=value` |
| Precomputed density | Struct with `.tag = 'ExpTensDensity'` | `ExpTensDensity` dataclass |

### Struct vs raw-argument calling

In MATLAB, `cosSimExpTens` and `evalExpTens` accept either a precomputed struct or raw arguments in a single function, dispatching on the first argument's type. In Python, these are separate functions:

| MATLAB | Python |
|:---|:---|
| `cosSimExpTens(dens_x, dens_y)` | `cos_sim_exp_tens(dens_x, dens_y)` |
| `cosSimExpTens(p1, w1, p2, w2, ...)` | `cos_sim_exp_tens_raw(p1, w1, p2, w2, ...)` |
| `evalExpTens(dens, X)` | `eval_exp_tens(dens, x)` |
| `evalExpTens(p, w, sigma, r, ...)` | `eval_exp_tens_raw(p, w, sigma, r, ...)` |

### Return values

Most functions return identical outputs in both languages. Two exceptions:

- `coherence` and `sameness` always return both the quotient and the count as a tuple in Python (e.g., `c, nc = mpt.coherence(...)`), whereas in MATLAB the count is a second optional output (`[c, nc] = coherence(...)`).
- `audio_peaks` always returns three values in Python (`f, w, detail = mpt.audio_peaks(...)`) where the MATLAB version returns the detail struct only when a third output is requested.

### Query points

In MATLAB, query-point arguments (`X`, `x`) are row vectors or matrices whose columns are query points. In Python, 1-D query-point arrays are passed as 1-D arrays; for multi-dimensional densities (r − isRel > 1), query points are a `(dim, n_queries)` array. For the most common case — 1-D densities — the usage is identical in both languages: pass a 1-D array.

### Full name mapping

| MATLAB | Python | Category |
|:---|:---|:---|
| `addSpectra` | `add_spectra` | Spectra |
| `buildExpTens` | `build_exp_tens` | Tensor core |
| `evalExpTens` | `eval_exp_tens` / `eval_exp_tens_raw` | Tensor core |
| `cosSimExpTens` | `cos_sim_exp_tens` / `cos_sim_exp_tens_raw` | Tensor core |
| `batchCosSimExpTens` | `batch_cos_sim_exp_tens` | Tensor core |
| `entropyExpTens` | `entropy_exp_tens` | Entropy |
| `estimateCompTime` | `estimate_comp_time` | Utility |
| `dftCircular` | `dft_circular` | Circular |
| `balanceCircular` | `balance` | Circular |
| `evennessCircular` | `evenness` | Circular |
| `coherence` | `coherence` | Circular |
| `sameness` | `sameness` | Circular |
| `edges` | `edges` | Circular |
| `projCentroid` | `proj_centroid` | Circular |
| `meanOffset` | `mean_offset` | Circular |
| `circApm` | `circ_apm` | Circular |
| `markovS` | `markov_s` | Circular |
| `nTupleEntropy` | `n_tuple_entropy` | Entropy |
| `spectralEntropy` | `spectral_entropy` | Harmony |
| `templateHarmonicity` | `template_harmonicity` | Harmony |
| `tensorHarmonicity` | `tensor_harmonicity` | Harmony |
| `virtualPitches` | `virtual_pitches` | Harmony |
| `roughness` | `roughness` | Harmony |
| `audioPeaks` | `audio_peaks` | Audio |
| `convertPitch` | `convert_pitch` | Utility |

### Documentation

In MATLAB, use `help functionName`. In Python, use `help(mpt.function_name)` or access docstrings in your IDE.


---

## 5. Quick start

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
s = cosSimExpTens(maj_p, maj_w, min_p, min_w, 10, 1, false, true, 1200);
fprintf('SPCS(major, minor) = %.3f\n', s);
```

**Python:**
```python
import mpt

major = [0, 400, 700]
minor = [0, 300, 700]

maj_p, maj_w = mpt.add_spectra(major, None, 'harmonic', 12, 'powerlaw', 1)
min_p, min_w = mpt.add_spectra(minor, None, 'harmonic', 12, 'powerlaw', 1)

s = mpt.cos_sim_exp_tens_raw(maj_p, maj_w, min_p, min_w, 10, 1, False, True, 1200)
print(f'SPCS(major, minor) = {s:.3f}')
```

### Precomputing for repeated comparisons

**MATLAB:**
```matlab
% Build density object once
dens_ref = buildExpTens(maj_p, maj_w, 10, 1, false, true, 1200);

% Compare against many other chords efficiently
chords = {[0, 300, 700], [0, 400, 700], [0, 500, 700]};
for i = 1:numel(chords)
    [cp, cw] = addSpectra(chords{i}, [], 'harmonic', 12, 'powerlaw', 1);
    dens_cmp = buildExpTens(cp, cw, 10, 1, false, true, 1200);
    s(i) = cosSimExpTens(dens_ref, dens_cmp);
end
```

**Python:**
```python
dens_ref = mpt.build_exp_tens(maj_p, maj_w, 10, 1, False, True, 1200)

chords = [[0, 300, 700], [0, 400, 700], [0, 500, 700]]
s = []
for chord in chords:
    cp, cw = mpt.add_spectra(chord, None, 'harmonic', 12, 'powerlaw', 1)
    dens_cmp = mpt.build_exp_tens(cp, cw, 10, 1, False, True, 1200)
    s.append(mpt.cos_sim_exp_tens(dens_ref, dens_cmp))
```

### Visualising an expectation tensor

**MATLAB:**
```matlab
% Build and evaluate a 1-D absolute periodic density
p = [0; 400; 700];
[p_spec, w_spec] = addSpectra(p, [], 'harmonic', 12, 'powerlaw', 1);
dens = buildExpTens(p_spec, w_spec, 10, 1, false, true, 1200);
x = linspace(0, 1200, 1200);
vals = evalExpTens(dens, x);
plot(x, vals);
xlabel('Pitch class (cents)');
ylabel('Density');
title('Spectral pitch class density of a major triad');
```

**Python:**
```python
import numpy as np
import matplotlib.pyplot as plt

p_spec, w_spec = mpt.add_spectra([0, 400, 700], None, 'harmonic', 12, 'powerlaw', 1)
dens = mpt.build_exp_tens(p_spec, w_spec, 10, 1, False, True, 1200)
x = np.linspace(0, 1200, 1200)
vals = mpt.eval_exp_tens(dens, x)
plt.plot(x, vals)
plt.xlabel('Pitch class (cents)')
plt.ylabel('Density')
plt.title('Spectral pitch class density of a major triad')
plt.show()
```

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

### Extracting spectral peaks from audio

**MATLAB:**
```matlab
[f, w] = audioPeaks('audio/Piano_Cmin_open.wav');
p = convertPitch(f, 'hz', 'cents');
H = spectralEntropy(p, w, 12);
fprintf('Spectral entropy = %.3f\n', H);
```

**Python:**
```python
f, w, detail = mpt.audio_peaks('audio/Piano_Cmin_open.wav')
p = mpt.convert_pitch(f, 'hz', 'cents')
H = mpt.spectral_entropy(p, w, 12)
print(f'Spectral entropy = {H:.3f}')
```

---

## 6. Function reference

Functions are grouped by category. For full details, use `help functionName` in MATLAB or `help(mpt.function_name)` in Python. Code examples in this section use MATLAB syntax; see [Section 4](#4-api-conventions-matlab-vs-python) for the systematic Python equivalents.

### 7.1 Expectation tensor core

| MATLAB | Python | Description |
|:---|:---|:---|
| `buildExpTens` | `build_exp_tens` | Precompute an r-ad expectation tensor density object |
| `evalExpTens` | `eval_exp_tens` | Evaluate the density at query points |
| `cosSimExpTens` | `cos_sim_exp_tens` | Cosine similarity of two expectation tensor densities |
| `batchCosSimExpTens` | `batch_cos_sim_exp_tens` | Batch cosine similarity with deduplication |
| `entropyExpTens` | `entropy_exp_tens` | Shannon entropy of an expectation tensor |
| `estimateCompTime` | `estimate_comp_time` | Micro-benchmark-based computation time estimate |

**buildExpTens(p, w, sigma, r, isRel, isPer, period)**

Precomputes tuple indices, pitch matrices, and weight vectors for a weighted pitch multiset. Returns a struct (`tag = 'ExpTensDensity'`) that can be passed to `evalExpTens` and `cosSimExpTens`.

Key parameters:
- `p` — Pitch values (vector)
- `w` — Weights (vector, or empty for uniform)
- `sigma` — Standard deviation of the Gaussian kernel (cents)
- `r` — Tuple size (positive integer; r ≥ 2 if isRel = true)
- `isRel` — Transposition-invariant if true (effective dim = r − 1)
- `isPer` — Periodic wrapping if true
- `period` — Period for wrapping (e.g., 1200 for one octave in cents)

**evalExpTens(dens, X [, normalize])**

Evaluates the density at the query points given by the columns of X. Three normalization modes:
- `'none'` (default) — Raw weighted sum of Gaussian kernels. The absolute value depends on σ, the number of tuples, and the weight magnitudes. Only the relative values across query points are meaningful. Sufficient for visualization and cosine similarity, where any normalization cancels.
- `'gaussian'` — Each Gaussian component is normalized to integrate to 1. After this normalization, the total density integrates to the sum of all tuple weight products. Useful for comparing densities computed with different σ values: increasing σ spreads the same mass over a wider area rather than inflating the total integral.
- `'pdf'` — Full probability density normalization. Applies the Gaussian normalization above, then divides by the sum of all tuple weight products so the density integrates to 1 over the domain. Useful for comparing densities across pitch multisets of different sizes, or for computing entropy.

Query points X should have `dim` rows, where `dim = r − isRel`. For the relative case (dim = r − 1), each column of X specifies the r − 1 intervals that define an r-ad (e.g., for a triad with r = 3 and isRel = true, each column is a 2-element vector of intervals from the lowest pitch to the middle and highest pitches).

**cosSimExpTens(dens_x, dens_y)** or **cosSimExpTens(p1, w1, p2, w2, sigma, r, isRel, isPer, period)**

Computes the cosine similarity between two expectation tensor densities analytically. The precomputed-struct calling convention avoids recomputing tuple indices on each call. Both conventions support `'verbose', false`.

**batchCosSimExpTens(pMatA, pMatB, sigma, r, isRel, isPer, period, ...)**

Computes cosine similarity for many paired weighted multisets. Each row of pMatA and pMatB defines one pair. Automatically deduplicates rows with identical sorted content, computing `cosSimExpTens` only once per unique pair. Optional name-value pairs: `'weightsA'`, `'weightsB'`, `'spectrum'` (cell array of `addSpectra` arguments), `'verbose'`.

**entropyExpTens(p, w, sigma, r, isRel, isPer, period, ...)**

Shannon entropy of the expectation tensor, discretized on a fine grid. Also accepts a precomputed struct as the first argument. Optional name-value pairs: `'spectrum'`, `'normalize'` (default: true), `'base'` (default: 2), `'nPointsPerDim'` (default: 1200), `'xMin'`, `'xMax'` (required when isPer = false).

### 7.2 Spectral enrichment

| MATLAB | Python | Description |
|:---|:---|:---|
| `addSpectra` | `add_spectra` | Add spectral partials to a weighted pitch multiset |

**addSpectra(p, w, mode, ...)**

Adds partials to each pitch. Mode is one of `'harmonic'`, `'stretched'`, `'freqlinear'`, `'stiff'`, or `'custom'`. All non-custom modes take N (number of partials including the fundamental), followed by a weight-type specification (`'powerlaw', rho` or `'geometric', tau`). The `'stretched'`, `'freqlinear'`, and `'stiff'` modes each have one additional parameter (β, α, or B respectively) between N and the weight type.

The optional name-value pair `'units', U` specifies pitch units per octave (default: 1200, i.e., cents). When using semitones, set `'units', 12`.

Output weights are the product of each pitch's original weight and the spectral weight of each partial.

### 7.3 Consonance and harmonicity

| MATLAB | Python | Description |
|:---|:---|:---|
| `spectralEntropy` | `spectral_entropy` | Entropy of the smoothed composite spectrum |
| `templateHarmonicity` | `template_harmonicity` | Harmonicity via template cross-correlation |
| `tensorHarmonicity` | `tensor_harmonicity` | Harmonicity via expectation tensor lookup |
| `roughness` | `roughness` | Sensory roughness (Plomp–Levelt / Sethares) |
| `virtualPitches` | `virtual_pitches` | Virtual pitch salience profile |

These functions take pitches in cents as absolute pitches (not pitch classes). The functions transpose internally so the lowest pitch is 0.

**spectralEntropy(p, w, sigma, ...)** — Entropy of the composite spectrum (lower entropy = greater consonance). Pitches must be in cents; when using empirical spectral peaks from `audioPeaks` (which returns Hz), convert via `convertPitch(f, 'hz', 'cents')` first. Can apply spectral enrichment via `'spectrum'`, but this is unnecessary when using empirical peaks since they already represent the full spectrum.

**templateHarmonicity(p, w, sigma, ...)** — Cross-correlates the chord's spectrum with a harmonic template. Returns hMax (maximum cosine similarity; Milne, 2013) and hEntropy (entropy of the cross-correlation; Harrison, 2020). Separate `'spectrum'` (for the template) and `'chordSpectrum'` (for the chord) parameters. See the comparison with `tensorHarmonicity` below.

**tensorHarmonicity(p, w, sigma, ...)** — Evaluates the relative r-ad expectation tensor of a harmonic series at the chord's intervals (measured from the lowest pitch to each of the remaining pitches). See the comparison with `templateHarmonicity` below.

**roughness(f, w, ...)** — Frequencies must be in Hz (use `convertPitch` if needed). Optional name-value pairs: `'pNorm'` (default: 1), `'average'` (default: false).

**virtualPitches(p, w, sigma, ...)** — Returns the full cross-correlation profile (pitch-indexed weights) from which `templateHarmonicity` extracts summary statistics. Peaks in vp_w indicate strong virtual pitches (candidate fundamentals).

#### templateHarmonicity vs tensorHarmonicity in detail

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

### 7.4 Balance and evenness (Fourier-based measures)

| MATLAB | Python | Description |
|:---|:---|:---|
| `dftCircular` | `dft_circular` | DFT of points on a circle |
| `balanceCircular` | `balance` | Balance (1 − \|F(0)\|) |
| `evennessCircular` | `evenness` | Evenness (\|F(1)\|) |

These functions apply equally to pitches (where the circle is one octave or other period) and to positions (where the circle is one rhythmic cycle or other periodic domain).

**dftCircular(p, w, period)** — Returns complex Fourier coefficients F and their magnitudes. F(1) (MATLAB 1-based) is the k = 0 coefficient; F(2) is k = 1; etc.

**balanceCircular(p, w, period)** — Returns a value in [0, 1]. Balance = 1 means the centre of gravity is at the circle's centre (e.g., augmented triad, whole-tone scale, isochronous rhythm). Supports weighted events.

**evennessCircular(p, period)** — Returns a value in [0, 1]. Evenness = 1 means the events are equally spaced (e.g., whole-tone scale, chromatic scale). Always uses uniform (binary) weights, following Milne et al. (2017).

### 7.5 Scale and rhythm structure

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

**coherence(p, period, ...)** — Coherence quotient in [0, 1]. A value of 1 means no coherence failures (strict propriety: larger generic spans always have strictly larger specific sizes). The optional `'strict', false` flag uses non-strict propriety (Rothenberg's original definition: failures only when a larger span has a strictly *smaller* size).

**sameness(p, period)** — Sameness quotient in [0, 1]. A value of 1 means no ambiguities (each specific interval size belongs to exactly one generic span).

**nTupleEntropy(p, period [, n], ...)** — Shannon entropy of the distribution of n-tuples of consecutive step sizes. When n = 1 (the default), this is IOI / step-size entropy. Optional name-value pairs: `'sigma'` (Gaussian smoothing, default: 0), `'normalize'` (default: true), `'base'` (default: 2). With Gaussian smoothing, the entropy becomes a continuous function of event positions, suitable for optimization (Milne, 2024).

**circApm(p, w, period, ...)** — Returns the period × period autocorrelation phase matrix R, the metrical weight profile rPhase (column sum), and the circular autocorrelation rLag (row sum). Optional `'decay'` parameter for exponential decay weighting.

**edges(p, w, period [, x], ...)** — Circular edge detection via convolution with the first derivative of a von Mises kernel. Returns absolute and signed edge weights. Optional `'kappa'` parameter controls kernel width.

**projCentroid(p, w, period [, x])** — Projection of the circular centroid (k = 0 Fourier coefficient) onto each angular position. Returns the projection vector, centroid magnitude, and centroid phase.

**meanOffset(p, w, period [, x])** — For each query point, the weighted sum of (upward arc − downward arc) to all events, normalized by the period. In a pitch-class context, this formalizes and generalizes Huron's (2008) "average pitch height," making the position-dependence explicit: it returns a value for every position around the circle. The term "mode height" for a closely related concept is used by Hearne (2020) and Tymoczko (2023).

**markovS(p, w, period [, S])** — Optimal S-step Markov predictor (default S = 3). For each position in the cycle, finds all positions with an identical S-step future context and returns their average weight. Originally by David Bulger.

### 6.6 Utility

| MATLAB | Python | Description |
|:---|:---|:---|
| `convertPitch` | `convert_pitch` | Convert between pitch/frequency scales |
| `audioPeaks` | `audio_peaks` | Extract spectral peaks from audio |

**convertPitch(values, fromScale, toScale)** — Converts between seven scales: `'hz'`, `'midi'`, `'cents'`, `'mel'`, `'bark'`, `'erb'`, `'greenwood'`. All conversions route through Hz. Vectorized: accepts scalars, vectors, or matrices. Note that the `'cents'` scale is absolute MIDI cents (A4 = 6900, middle C = 6000), not relative interval cents.

**audioPeaks(audioFile, ...)** — Reads an audio file, computes the magnitude spectrum, and extracts peaks. Returns frequencies in Hz and normalized amplitudes in [0, 1]. Optional name-value pairs: `'sigma'` (smoothing in cents; default: 0), `'resolution'` (cents grid spacing; default: 1), `'rampDuration'` (onset/offset ramp in seconds; default: 0), `'fMin'`, `'fMax'`, `'minProminence'`, `'noiseFactor'`, `'plot'`.

When sigma > 0, the spectrum is resampled onto a uniform log-frequency (cents) grid before smoothing. This ensures the Gaussian kernel has a fixed perceptual width at all frequencies (a fixed-Hz kernel would over-smooth high partials and under-smooth low ones). Peak positions are converted back to Hz. This smoothing is useful for audio with vibrato or frequency jitter: partials separated by more than approximately 2σ cents are individually resolved, while closer partials are merged into a single peak. The sigma value should therefore be chosen to match the expected extent of frequency variation in the audio. Smoothing is unnecessary for steady-state tones, since the downstream toolbox functions already apply their own Gaussian smoothing via the expectation tensor framework.

Typical workflow:
```matlab
[f, w] = audioPeaks('audio/piano_C4.wav');
p = convertPitch(f, 'hz', 'cents');
H = spectralEntropy(p, w, 12);           % no addSpectra needed
r = roughness(f, w);                      % roughness needs Hz
[hMax, hEnt] = templateHarmonicity(p, w, 12);
```

---

## 7. Worked examples

The worked examples below use MATLAB syntax. Python equivalents are in `python/demos/` — start with `demo_overview.py` for a quick tour of all function families, or see the individual demo scripts listed in [Section 8](#8-demo-scripts) for the Python version of each example below.

### 7.1 EDO approximation via relative dyad tensors

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
dens_ref = buildExpTens(ref_p, ref_w, sigma, r, isRel, isPer, period);

% Sweep n-EDOs
nRange = 3:53;
s = zeros(size(nRange));
for i = 1:numel(nRange)
    n = nRange(i);
    edo = (0:n-1) * period / n;
    [edo_p, edo_w] = addSpectra(edo, [], 'harmonic', nHarm, 'powerlaw', rho);
    dens_edo = buildExpTens(edo_p, edo_w, sigma, r, isRel, isPer, period);
    s(i) = cosSimExpTens(dens_ref, dens_edo, 'verbose', false);
end

bar(nRange, s);
xlabel('n-EDO');
ylabel('SPCS');
title('SPCS of n-EDOs to JI major triad');
```

### 7.2 Consonance landscape of triads

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
    f_hz = convertPitch(fp, 'cents', 'hz');
    R(i) = roughness(f_hz, fw);
end

imagesc(ints, ints, -R);
axis xy;
xlabel('Interval 1 (cents)');
ylabel('Interval 2 (cents)');
title('Smoothness (negative roughness)');
colorbar;
```

### 7.3 Rhythmic structure features

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

### 7.4 Virtual pitch analysis

Identify the strongest virtual pitches of a chord.

```matlab
% C major triad in absolute cents (MIDI 60, 64, 67)
p = convertPitch([60 64 67], 'midi', 'cents');
spec = {'harmonic', 36, 'powerlaw', 1};

[vp_p, vp_w] = virtualPitches(p, [], 12, 'chordSpectrum', spec);

% Plot with MIDI pitch axis
vp_midi = convertPitch(vp_p, 'cents', 'midi');
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

### 7.5 Working with audio files

Extract peaks from audio and compute multiple features.

```matlab
% Extract peaks
[f, w] = audioPeaks('audio/music_sample.wav', 'sigma', 12, 'plot', true);

% Convert to cents
p = convertPitch(f, 'hz', 'cents');

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

---

## 8. Demo scripts

Nine demo scripts are included in each language, with user-adjustable parameters at the top. A good place to start is `demo_overview` (MATLAB) or `demo_overview.py` (Python), which exercises every major function family in a single script and follows the same section order as this guide.

### MATLAB demos

MATLAB demo scripts are in `matlab/demos/`. To run a demo, open it in the MATLAB editor or navigate to the `demos/` folder and run it from there.

| Script | Description | Based on |
|:---|:---|:---|
| `demo_overview` | Quick tour of all major function families: pitch conversion, spectral enrichment, SPCS, harmonicity, roughness, balance, evenness, coherence, sameness, entropy, mean offset, edges, and Markov | — |
| `demo_audioAnalysis` | Two-pass peak extraction (unsmoothed then smoothed) from audio files, with spectral similarity, harmonicity, roughness, and virtual pitch analysis | — |
| `demo_batchProcessing` | Batch feature computation with deduplication: paired SPCS via `batchCosSimExpTens`, and single-set measures (spectral entropy, harmonicity, roughness) via the unique/map pattern | — |
| `demo_edoApprox` | SPCS of n-EDOs against a JI chord | Milne et al. (2011), Ex. 6.3 / Fig. 4 |
| `demo_expTensorPlots` | Interactive visualisation of expectation tensors in 1–4 dimensions, with power sliders and projection toggles | — |
| `demo_genChainSpcs` | SPCS of generator-chain tunings as the generator is swept (linear and circular plots) | Milne et al. (2011), Ex. 6.4–6.5 / Figs. 5–7 |
| `demo_triadConsonance` | Five consonance measures over a grid of triad intervals | — |
| `demo_triadSpcsGrid` | SPCS heatmap of 12-EDO triads with a fifth | Milne et al. (2011), Fig. 3 |
| `demo_virtualPitches` | Virtual pitch salience profiles for example chords | — |

### Python demos

Python equivalents of all nine demos are in `python/demos/`. They follow the same structure and produce the same results; the plotting demos require `matplotlib` (`pip install matplotlib`) and the audio demo requires `soundfile` (`pip install soundfile`).

| Script | MATLAB equivalent | Description |
|:---|:---|:---|
| `demo_overview.py` | `demo_overview` | Quick tour of all major function families |
| `demo_audio_analysis.py` | `demo_audioAnalysis` | Two-pass audio peak extraction and perceptual features |
| `demo_batch_processing.py` | `demo_batchProcessing` | Batch feature computation with deduplication |
| `demo_edo_approx.py` | `demo_edoApprox` | SPCS of n-EDOs against a JI chord |
| `demo_exp_tensor_plots.py` | `demo_expTensorPlots` | Expectation tensor density visualisation (1–4D) |
| `demo_gen_chain_spcs.py` | `demo_genChainSpcs` | Generator-chain SPCS (linear and circular plots) |
| `demo_triad_consonance.py` | `demo_triadConsonance` | Five consonance measures over a triad grid |
| `demo_triad_spcs_grid.py` | `demo_triadSpcsGrid` | SPCS heatmap of triads with a fifth |
| `demo_virtual_pitches.py` | `demo_virtualPitches` | Virtual pitch salience profiles |

---

## 9. Known simplifications and future directions

### Flat-metric approximation

The expectation tensor framework uses a locally flat (Euclidean) metric: the Gaussian kernel is defined in terms of Euclidean distances in pitch (or time) space. In the periodic case, the domain is topologically circular (differences are wrapped modulo the period), but the metric within each period is still Euclidean — there is no curvature. For pitch-class sets with period 1200 (one octave in cents), this is well justified because the cents scale has uniform spacing in log-frequency, which closely approximates equal perceptual spacing over the range where most musical pitch perception occurs.

However, if one were to use a psychoacoustic pitch scale with non-constant spacing (e.g., mel, ERB-rate, or Bark — all available via `convertPitch`), the Euclidean metric would introduce a systematic approximation: the effective smoothing would vary across the frequency range. The correct treatment would involve a Riemannian metric that accounts for the non-constant Jacobian of the pitch-scale mapping. In one dimension, this can be handled exactly by converting to the psychoacoustic scale before calling the toolbox (the Gaussian then has the correct width at every point). In higher dimensions (r ≥ 2), the full Riemannian treatment would require architectural changes. This is documented as a potential future direction but is not implemented in v2.0.0.

### Grid discretization for entropy

The differential entropy of a Gaussian mixture density has no known closed-form analytical solution, because the logarithm of a sum of Gaussians does not simplify. This is why `entropyExpTens` and `spectralEntropy` discretize the continuous density onto a finite grid and compute Shannon entropy of the resulting probability mass function — unlike the cosine similarity, which *can* be computed analytically. The accuracy of this discretization depends on the ratio of σ to the grid spacing. Users can verify accuracy by comparing results at different resolutions (via `'nPointsPerDim'` or `'resolution'`). When the normalized entropy option is used (the default), the result is independent of the arbitrary grid resolution to the extent that the grid is fine enough to capture the density's shape.

---

## 10. References

Balzano, G. J. (1982). The pitch set as a level of description for studying musical pitch perception. In M. Clynes (Ed.), *Music, Mind, and Brain* (pp. 321–351). Plenum.

Carey, N. (2002). On coherence and sameness, and the evaluation of scale candidacy claims. *Journal of Music Theory*, 46(1/2), 1–56.

Carey, N. (2007). Coherence and sameness in well-formed and pairwise well-formed scales. *Journal of Mathematics and Music*, 1(2), 79–98.

Dean, R. T., Milne, A. J., & Bailes, F. (2019). Spectral pitch similarity is a predictor of perceived change in sound- as well as note-based music. *Music & Science*, 2, 1–14.

Eck, D. (2006). Beat tracking using an autocorrelation phase matrix. *Proceedings of the International Computer Music Conference (ICMC)*.

Eerola, T. & Lahdelma, I. (2021). The anatomy of consonance/dissonance: Evaluating acoustic and cultural predictors across multiple datasets with chords. *Music & Science*, 4, 20592043211030471.

Eitel, M., Ruth, N., Harrison, P., Frieler, K., & Müllensiefen, D. (2024). Perception of chord sequences modeled with prediction by partial matching, voice-leading distance, and spectral pitch-class similarity: A new approach for testing individual differences in harmony perception. *Music & Science*, 7.

Harrison, P. M. C. & Pearce, M. T. (2020). Simultaneous consonance in music perception and composition. *Psychological Review*, 127(2), 216–244.

Hearne, L. M. (2020). *The Cognition of Harmonic Tonality in Microtonal Scales*. PhD thesis, Western Sydney University.

Hearne, L. M., Dean, R. T., & Milne, A. J. (2025). Acoustical and cultural explanations for contextual tonal stability. *Music Perception*, 43(3).

Homer, S., Harley, N., & Wiggins, G. (2024). Modelling of musical perception using spectral knowledge representation. *Journal of Cognition*, 7.

Huron, D. (2008). A comparison of average pitch height and interval size in major- and minor-key themes: Evidence consistent with affect-related pitch prosody. *Empirical Musicology Review*, 3, 59–63.

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

Sethares, W. A. (1993). Local consonance and the relationship between timbre and scale. *Journal of the Acoustical Society of America*, 94(3), 1218–1228.

Sethares, W. A., Milne, A. J., Tiedje, S., Prechtl, A., & Plamondon, J. (2009). Spectral tools for Dynamic Tonality and audio morphing. *Computer Music Journal*, 33(2), 71–84.

Smit, E. A., Milne, A. J., Dean, R. T., & Weidemann, G. (2019). Perception of affect in unfamiliar musical chords. *PLOS ONE*, 14(6), e0218570.

Tymoczko, D. (2023). *Tonality: An Owner's Manual*. Oxford University Press.

---

## 11. Citation

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
