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

The Music Perception Toolbox is an open-source toolbox — available in MATLAB and Python — for computing perceptually and cognitively motivated measures of pitch and rhythmic similarity, consonance, structural features, and sequential analysis.

The toolbox implements several original theoretical frameworks. Expectation tensors (Milne, Sethares, Laney, & Sharp, 2011) provide a unified framework — grounded in probability theory and Riemannian geometry — for quantifying the similarity of collections of musical events under different assumptions about perceptual equivalence and uncertainty, and at different orders of structural complexity (from individual pitches through dyads, triads, and beyond). In v2.1.0, the framework has been extended to *multi-attribute* expectation tensors (MAETs), which represent events characterised by several perceptually relevant properties simultaneously — pitch together with time, with register, with timbre, with metrical position, with voice, and so on — each with its own uncertainty parameter and geometric structure. The balance and evenness measures (Milne, Bulger, & Herff, 2017) draw on the discrete Fourier transform to characterise the distributional properties of scales and rhythms, including a novel class of perfectly balanced patterns. Additional structural and perceptual features — coherence, sameness, edge detection, projected centroid, mean offset, and Markov prediction — have been developed and empirically validated for modelling rhythmic perception and performance (Milne & Herff, 2020; Milne, Dean, & Bulger, 2023).

These measures have proven effective predictors across a range of music cognition contexts, including tonal fit and stability in conventional and microtonal tuning systems (Milne, Laney, & Sharp, 2015, 2016; Homer, Harley, & Wiggins, 2024; Hearne, Dean, & Milne, 2025), perceived consonance and affect (Smit et al., 2019; Harrison & Pearce, 2020; Eerola & Lahdelma, 2021), individual differences in harmony perception (Eitel, Ruth, Harrison, Frieler, & Müllensiefen, 2024), and rhythmic complexity and tapping accuracy (Milne & Herff, 2020; Milne, Dean, & Bulger, 2023). They have also guided the design of music-computing interfaces (Sethares, Milne, Tiedje, Prechtl, & Plamondon, 2009; Milne & Dean, 2016; Milne, 2019). Published empirical validation is to date on the single-attribute case; MAETs enable a class of analyses for which existing single-attribute measures would be forced to either pool events into unordered multisets or abandon the framework entirely.

The toolbox operates on weighted multisets of events, each event being characterised by one or more attributes (most commonly pitch, or a combination of pitch and time). Input can be entered directly (e.g., as cents values or integer pulse positions) or derived from audio recordings via the `audioPeaks` (`audio_peaks` in Python) function, which extracts spectral peaks and their amplitudes from audio files. The `convertPitch` (`convert_pitch`) function converts between seven pitch and frequency scales (Hz, MIDI, cents, mel, Bark, ERB-rate, Greenwood).

The toolbox is organised around a **core framework** of expectation tensors (Section 3.1) and several **application areas** that build on it:

**Pitch (and pitch-class) similarity** (Section 3.3). Cosine similarity of expectation tensors quantifies how similar two collections of weighted events are — pitches, pitch classes, time points, rhythmic patterns, or combinations of these via MAETs. For pitch, spectral enrichment via `addSpectra` yields spectral pitch (class) similarity (SPS/SPCS), a robust predictor of perceived tonal fit, affect, and similarity.

**Consonance and harmonicity** (Section 3.4). Spectral entropy, template harmonicity, tensor harmonicity, and sensory roughness — complementary measures, usable singly or in combination, that together provide strong predictions of perceived consonance.

**Structural measures on periodic cycles** (Section 3.5). Structural and perceptual features of multisets of points on a circle, most obviously applicable to pitches and rhythmic positions, organised by output granularity: period-level measures (balance, evenness, DFT coefficients, coherence, sameness, n-tuple entropy), integer-position measures (circular autocorrelation phase matrix, Markov prediction), and continuous-position measures (edge detection, projected centroid, mean offset).

**Sequential-analysis utilities** (Section 3.6). Two utilities for analyses of ordered event sequences: a smoothed direction-continuity measure (`continuity`) that shares the difference-event substrate with the MAET-with-differencing pipeline but reads it as an ordered sequence with a directional gate rather than aggregating it into a tensor, and a flexible position-weight constructor (`seqWeights`) usable anywhere a weight argument is accepted.

**Version 2.0.0** was a major rewrite of the v1 toolbox. Analytical methods replaced the previous numerical approximations wherever feasible: in v1, analytical computation was available only for the cosine similarity inner product (`cosSimExpTens`, by David Bulger); in v2, this analytical approach was extended to the construction and evaluation of individual expectation tensors via `buildExpTens` and `evalExpTens`, eliminating the discretization error and resolution trade-offs of v1's grid-based `expectationTensor`. The `cosSimExpTens` computation itself was substantially optimised — the original double loop over r-ad combinations replaced by fully vectorised operations over pre-calculated r-ads, with automatic memory-aware chunking for large problems. Entropy computation remains discretized, as the differential entropy of a Gaussian mixture has no known closed-form solution. The expectation tensor core was restructured around precomputed density objects, and the toolbox was substantially expanded with new consonance, structural, and sequential measures.

**Version 2.1.0** introduced the multi-attribute expectation tensor and the post-tensor windowing mechanism that builds on it. MAETs represent weighted multisets of events whose attributes span several perceptual dimensions: pitch across voices, time, register, metric position, timbre, or any combination of these, with each group of attributes assigned its own uncertainty parameter and geometric structure. The analytical closed-form cosine similarity inner product that v2 established for single-attribute tensors has been extended to the multi-attribute case. `windowTensor` wraps a MAET with a shape-parameterised windowing specification, and `windowedSimilarity` sweeps a series of offsets from the query's centroid to produce a cross-correlation similarity profile; the closed-form inner product is preserved across the full window-shape family (from Gaussian to rectangular) for a wide class of tensor configurations. Every function includes full help text with runnable examples, this User Guide covers the conceptual foundations and provides a complete function reference for both languages, and demo scripts (in both MATLAB and Python) cover all major use cases. See `CHANGELOG.md` for a full list of changes and `MIGRATION.md` for guidance on updating v1 code.

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

This section introduces the measures conceptually, organised by topic; the function-by-function reference, with calling conventions and name-value options, is in Section 6.

The core mathematical framework of the toolbox is the *expectation tensor* (ET) and its multi-attribute generalisation (MAET), introduced in §3.1, which also describes the post-tensor windowing mechanism used for sliding-window template matching. Spectral enrichment (§3.2) is a preprocessing step, specific to pitched sounds, typically applied before the ET functions to obtain the spectral variants of the similarity and consonance measures. The ET cosine similarity underlies the similarity measures of §3.3 — which apply to multisets of pitches, pitch classes, time points, or any other axis, with only the spectral variants being pitch-specific — and several of the consonance measures in §3.4. Structural measures on periodic cycles, collected in §3.5, draw on the discrete Fourier transform (DFT) for the balance, evenness, and projected-centroid measures, and on various distinct methods for the remaining structural measures. Two utilities for ordered sequences (§3.6) sit outside the ET framework.

### 3.1 (Multi-attribute) expectation tensors

This section introduces the expectation tensor, the central mathematical object underlying most of the toolbox. The exposition proceeds in two stages: the single-attribute expectation tensor (ET) is introduced first, as the cleanest teaching unit and the form in which the framework has been empirically validated; the multi-attribute generalisation (MAET) is then introduced as a natural extension that accommodates events characterised by several perceptually relevant properties simultaneously. The single-attribute case is a special case of the multi-attribute form; all downstream toolbox functions are designed to handle either.

An expectation tensor represents the distribution of $r$-tuples — or *$r$-ads* (dyads when $r = 2$, triads when $r = 3$, and so on) — of pitches (or time points) that a listener expects to perceive, given a weighted multiset and a model of perceptual uncertainty. Formally, it is an unnormalized Gaussian mixture density: a weighted sum of Gaussian kernels centred at all ordered $r$-tuples drawn from the multiset, with standard deviation $\sigma$ modelling perceptual uncertainty.

Given a multiset $\mathbf{p} = (p_1, p_2, \ldots, p_N)$ with weights $\mathbf{w} = (w_1, w_2, \ldots, w_N)$, the $r$-ad expectation tensor density at a query point $\mathbf{x}$ is:

$$f(\mathbf{x}) = \sum_j w_j \, \exp\!\left(-\frac{(\mathbf{x} - \mathbf{c}_j)^{\mathsf{T}} \, \mathbf{M} \, (\mathbf{x} - \mathbf{c}_j)}{2\sigma^2}\right)$$

where the sum is over all ordered $r$-tuples, $\mathbf{c}_j$ is the $j$-th tuple, and $\mathbf{M}$ is a quadratic form matrix determined by the mode (see below).

The toolbox implements this in two steps: `buildExpTens` precomputes the tuple indices and weight products into a density struct, and `evalExpTens` evaluates the density at query points. The cosine similarity between two such densities — which quantifies how similar the two weighted multisets are — is computed by `cosSimExpTens`. Crucially, the inner product underlying this cosine similarity has an analytical solution (a finite double sum of Gaussian kernel evaluations), so `cosSimExpTens` computes it exactly without discretization. This analytical computation was present in v1 (due to David Bulger); what is new in v2 is that individual tensor construction and evaluation (`buildExpTens` and `evalExpTens`) are also analytical, whereas v1's `expectationTensor` required discretization onto a grid. (Note, however, that the *entropy* of a Gaussian mixture density has no known closed-form solution, so the entropy functions `entropyExpTens` and `spectralEntropy` still use grid discretization — see [Section 9](#9-known-simplifications-and-future-directions).)

The term "expectation tensor" reflects two ideas: (a) the density represents the *expected* perceptual distribution of r-ads given a weighted multiset of pitches or time points, smoothed by perceptual uncertainty σ; and (b) "tensor" refers to the fact that the discretized density is a rank-r array (an r-dimensional grid of values), with its domain being the r-fold product of the pitch (or time) space with itself. This is "tensor" in the numerical/data sense (a multidimensional array) rather than the strict algebraic sense (a multilinear map with specific transformation properties).

#### The four modes

The expectation tensor has two logical flags that control its geometry:

**Periodic (isPer).** When true, the pitch (or time) line is wrapped into a circular domain of circumference `period`, so that values differing by the period are identified. For pitch, this implements pitch-class equivalence (e.g., with period = 1200, the pitch line is wrapped into a circle where 0 cents and 1200 cents are the same point). For rhythmic patterns, the period is the cycle length. When false, values are treated as points on an unbounded line.

**Relative (isRel).** When true, the density is invariant under transposition: shifting all values by the same amount does not change the density. Mathematically, this is achieved by projecting out the mean direction in $\mathbb{R}^r$, reducing the *effective space* (the space in which the density is supported) from $\mathbb{R}^r$ to the $(r-1)$-dimensional quotient $\mathbb{R}^r / \mathbb{R} \cdot \mathbf{1}$. In the formula above, the quadratic form matrix $\mathbf{M}$ becomes the projection matrix $\mathbf{I} - \mathbf{1}\mathbf{1}^{\mathsf{T}}/r$ (where $\mathbf{1}$ is the all-ones vector); the resulting quadratic form $Q(\mathbf{d}) = \sum_i d_i^2 - (\sum_i d_i)^2 / r$ is the squared distance in the quotient space under the Riemannian metric induced by the standard Euclidean inner product. When false (absolute), $\mathbf{M} = \mathbf{I}$ (the identity matrix), the effective space is $\mathbb{R}^r$ itself, and the density depends on the actual values, not just intervals.

When both isPer and isRel are true, the cosine similarity computation (`cosSimExpTens`) uses an algebraically equivalent pairwise form of $Q$ with wrapped pairwise differences — $Q(\mathbf{d}) = \sum_{i<j} \mathrm{wrap}(d_i - d_j)^2 / r$ — to maintain exact transposition invariance on the circle. This is necessary because the component-wise periodic wrapping is nonlinear and can otherwise introduce $\pm\mathrm{period}$ artifacts in the pairwise differences between components of the wrapped difference vector.

The four combinations (absolute non-periodic, absolute periodic, relative non-periodic, relative periodic) cover a range of use cases. The choice depends on the question being asked: do we care about absolute positions or only intervals? Do we treat values a period apart as equivalent?

#### Multiple attributes

The description so far applies to a single attribute — pitch, or time, or another quantity treated on its own. Many music-cognition questions, though, are poorly served by reducing an event to a single attribute: a sequence of notes carries both *what* (pitch) and *when* (time); a polyphonic passage carries multiple pitches per event across voices; a piece may additionally carry register, timbre, metrical position, and spatial location for each event. Reducing all of this to one attribute is often a sensible modelling choice — the empirical literature referenced in Section 1 is largely in that regime — but there are questions it cannot ask. It cannot distinguish the same chord at beat 1 from the same chord at beat 3. It cannot locate recurrences of a motif within a longer passage. It cannot model contributions of timbral distance or register separation to perceived similarity. The multi-attribute expectation tensor (MAET) generalises the single-attribute framework to accommodate several attributes per event simultaneously, while preserving the analytical properties that make the single-attribute case tractable. The user retains full control over which attributes to include in any given model — MAETs do not obligate the user to include them, they enable their inclusion when useful.

**Generalised equation.** A MAET represents events via $A$ *attributes* (each a kind of property — pitches in a chord, pitch of a voice, time, register, timbre, …), grouped into $G \leq A$ *groups* that share perceptual parameters (*attributes*, *groups*, and *slots* are defined in detail below). At each event, each attribute $a$ carries $K_a$ value-*slots* (for example, a pitch attribute representing a three-pitch chord has $K_a = 3$); the tensor is built at order $r_a$ per attribute. The density at a query point $\mathbf{x} = (\mathbf{x}_1, \ldots, \mathbf{x}_A)$ is

$$f(\mathbf{x}_1, \ldots, \mathbf{x}_A) = \sum_j w_j \, \prod_a \exp\!\left(-\frac{(\mathbf{x}_a - \mathbf{c}_{a,j})^{\mathsf{T}} \, \mathbf{M}_a \, (\mathbf{x}_a - \mathbf{c}_{a,j})}{2\sigma_{g(a)}^2}\right)$$

where the sum is over multi-attribute tuples $j$ (formed by taking, per event, the Cartesian product across attributes of each attribute's $r_a$-combinations of its $K_a$ slots), $w_j$ is the tuple's combined weight (the product of its constituent per-slot weights), $\mathbf{x}_a$ is the query in attribute $a$'s effective space of dimension $r_a - \mathrm{isRel}_{g(a)}$, $\mathbf{c}_{a,j}$ is the attribute-$a$ centre for tuple $j$, $\mathbf{M}_a$ is the attribute-level quadratic form matrix ($\mathbf{I}$ for absolute, $\mathbf{I} - \mathbf{1}\mathbf{1}^{\mathsf{T}}/r_a$ for relative — determined by the attribute's $r_a$ and its group's isRel), and $\sigma_{g(a)}$ is the $\sigma$ of the group containing attribute $a$. The density factorises as a product across attributes, with the group membership supplying each attribute's σ, isRel, isPer, and period. The single-attribute tensor is the special case $A = 1$. The cosine-similarity inner product factors similarly and remains closed-form analytical in the multi-attribute case.

**Slots, attributes, and groups.** The three concepts each play a distinct role:

- A **slot** is a single value-position within an attribute at a single event. An attribute with $K_a$ slots per event carries $K_a$ values at each event. Slots within an attribute are treated as *exchangeable* — the tensor enumerates unordered *combinations* of $r_a$ slots taken from the $K_a$ available, so permuting the slots of one attribute at one event does not change the resulting tensor.
- An **attribute** is a kind of per-event property: several pitches in a chord, the pitch of voice 1, time, register, spectral centroid. Different attributes are *not* exchangeable — they are joined by Cartesian product across per-attribute combinations, so the tensor distinguishes "attribute-1 value paired with attribute-2 value" from the reverse pairing.
- A **group** is a set of attributes that share their perceptual parameters: σ, isRel, isPer, and period. Grouping is pure parameter-sharing — it expresses that several attributes are of the same kind and should use the same perceptual uncertainty and geometric flags — and confers no additional structure: it does not make the attributes exchangeable, and does not constrain their per-attribute r or K values.

The distinction between "several slots within one attribute" and "several attributes (possibly in one group)" is the modelling choice between exchangeable and distinguishable components. Consider a three-voice chord:

- Represented as **one pitch attribute with $K_a = 3$ slots** (and $r_a \in \{1, 2, 3\}$ depending on how we want to count intervals), the tensor treats the three pitches of each event as an unordered multiset — which pitches are sounding matters, which voice carries which does not.
- Represented as **three separate pitch attributes each with $K_a = 1$ slot** (and $r_a = 1$), the tensor distinguishes (soprano, alto, tenor) as an ordered triple — voice identity matters. If we still want all three voices to share their σ, isRel, isPer, and period (they are all pitches), we place all three attributes in the same group, letting grouping handle parameter-sharing while leaving voice identity intact.

In the common case of a sequence of events over time with one or more pitches per event, the pitches form a pitch attribute (or several, depending on exchangeability), and time is a separate attribute in its own group — the pitches may be exchangeable (a single pitch attribute with $K_a > 1$) or not (several voices each with their own pitches, sharing a group), but time is always its own group because events at different times carry different time values.

Within this structure, the parameters that can be configured per group versus per attribute are:

- **Per group**: σ (perceptual uncertainty), `isRel` (transposition invariance), `isPer` (periodicity), `period` (when periodic). All attributes within a group share these.
- **Per attribute**: r (tuple size); the number of slots per event (K) is implicit from the shape of the input.

**Use cases.** A common MAET configuration is pitch combined with time — a pitch attribute with its pitch group (a common σ, isPer, and isRel, each chosen to suit the analysis desired) and a time attribute in its own time group (with a time σ and isPer=false). This unifies analyses of sequences of events over time with the rest of the toolbox: a sliding probe-tone analysis, a motif-recurrence search, and a pairwise similarity between two sequences all become ordinary `cosSimExpTens` operations on time-attributed tensors. Harmonic analyses occur when the pitch attribute has $K_a > 1$ pitches per event (exchangeable voices); polyphonic (voice-aware) analyses occur when voices are provided as several attributes in one group (non-exchangeable voices). But MAETs are not limited to pitch × time. Register can be added as an additional pitch-like attribute in its own group with a broader σ. Timbre can be attached as a numerical descriptor (e.g., spectral centroid) with its own group, allowing two passages identical in pitch but different in instrumentation to be distinguished as such. Metrical position can be a periodic attribute with period equal to one bar. Spatial location in a stereo or surround context can be an angular (periodic) attribute with its own σ. Any quantity that can be represented as one or more values per event, and for which perceptual uncertainty is meaningful, can in principle become an attribute. The post-tensor windowing mechanism in the subsection below builds on MAETs, and Sections 7.4 and 7.5 show worked examples.

**Compatibility between multi-attribute densities.** Cosine similarity between two MAETs (via `cosSimExpTens`) requires the two densities to share the same attribute structure: the same number of attributes, the same group assignment of attributes to groups, the same per-attribute tuple order $r_a$, and the same per-group σ, isRel, isPer, and period. A mismatch on any of these raises an informative error (e.g., `cosSimExpTens:rMismatch`). What can differ between query and context are the per-event slot counts $K_a$ (so a monophonic query can be compared against a polyphonic context on the same pitch attribute) and of course the number of events itself. This is a stronger compatibility requirement than the single-attribute case, where the parameters are passed as scalar arguments at the call site and therefore trivially match; the MA form requires densities to be built with `buildExpTens` first, and the structural parameters are then read off the density objects.

#### Cross-event preprocessing

The MAET framework provides two preprocessing helpers that transform an input event sequence into a derived sequence before tensor construction: `differenceEvents` (replacing the sequence with inter-event differences) and `bindEvents` (gathering n consecutive events into a single super-event). Both act per group, restrict to single-value ($K_a = 1$) attributes for the same reason (within-event slot exchangeability does not license a cross-event slot alignment), and propagate per-event weights as rolling products under the toolbox's standard probability-of-perception reading. They differ in what the derived sequence represents and the two operations compose. The MAET built on the derived sequence then aggregates across its entries in the usual way, so the resulting density is a distribution over derived values rather than a function of position in the derived sequence.

**Event differencing.** `differenceEvents` replaces a sequence of events with a sequence of inter-event differences. Some analyses model events as relative to their predecessors — a sequence of melodic intervals rather than absolute pitches, a sequence of inter-onset intervals (IOIs) rather than absolute time points, a sequence of interval changes rather than intervals — so the tensor represents the distribution of these relative quantities. `differenceEvents` performs this transformation: for each group assigned a differencing order $k$ it replaces the input values with their $k$-th finite differences along the event axis (reducing the event count by $k$), and wraps differences to the shortest signed arc for periodic groups. Weights propagate as rolling products under the toolbox's standard broadcast convention (§4), so the weight of a $k$-th-order difference is the product of its $k + 1$ constituent events' weights — interpretable as the probability that all constituents are jointly perceived under the standard weights-as-salience reading. Scalar-$c$ and vector-of-$c$ inputs therefore produce equivalent downstream densities. The output `(p_attr_diff, w_diff)` feeds directly into `buildExpTens`.

Event differencing is distinct from `isRel = true`, and the two can be used together or independently. Event differencing is a preprocessing step that replaces absolute per-event values with differences between adjacent events, so the tensor is constructed from inter-event quantities. `isRel = true` is a property of the tensor construction itself — it makes the within-r-ad density translation-invariant, so that (for example) a dyad tensor at r = 2 represents the distribution of intervals between any two elements of the multiset (not only adjacent events). Using event differencing gives a density over sequential differences; using `isRel = true` gives a density over all r-ad interval patterns within the (possibly differenced) input. The two give different interval-based characterisations, and both are analytically supported.

Event differencing requires each attribute to have $K_a = 1$ slot per event. The operation is column-wise subtraction across adjacent events, which imposes a cross-event slot correspondence (slot $i$ at event $n-1$ is paired with slot $i$ at event $n$). Within-event slot exchangeability — the MAET's treatment of slots within an attribute as interchangeable — does not guarantee such a correspondence, so for multi-slot attributes the output would depend on an arbitrary slot-listing choice. `differenceEvents` therefore raises an error on any attribute with $K_a \ne 1$.

For analyses that might seem to require multi-slot differencing — for example, the voice-agnostic distribution of melodic step-sizes across a polyphonic texture — the principled route is to encode each voice as a separate $K_a = 1$ attribute in a shared pitch group, call `differenceEvents` (each attribute differences canonically), and then stack the differenced attributes into a single multi-slot attribute before `buildExpTens`. This makes the analytical choice explicit: step 1 preserves voice labels, step 2 produces well-defined per-voice step-sizes, and step 3 deliberately discards voice labels to yield a voice-exchangeable representation. A short MATLAB example:

```matlab
pAttr  = {pS, pA, pT, pB};
groups = [1 1 1 1];
[pDiff, ~] = differenceEvents(pAttr, [], groups, 1, 0);
pBundled   = { vertcat(pDiff{:}) };
dens = buildExpTens(pBundled, [], sigma, r, [], true, false, 0);
```

**Event binding.** `bindEvents` gathers n consecutive events into a single super-event, sliding a window of width $n$ across the input and emitting each window with $n$ separate attributes carrying the values at lags $0$ through $n-1$. The output is a length-$n$ list of single-value attribute matrices, suitable as the `pAttr` argument of `buildExpTens` with all $n$ attributes assigned to a single group sharing $\sigma$, `isRel`, `isPer`, and `period`. Lag slots are kept as separate attributes rather than packed into one $K_a = n$ attribute because lag identity is non-exchangeable: the within-attribute multiset symmetry would otherwise collapse ordered tuples to unordered ones. Per-event weights propagate as rolling products of width $n$, mirroring the differencing rule: the weight of a bound super-event is the product of the $n$ input-event weights it spans. The default output has $N - n + 1$ super-events; a `circular` option produces $N$ by wrapping around the end of the input. As with `differenceEvents`, the input is restricted to $K_a = 1$ for the same reason — within-event slot exchangeability does not license sliding-window slot alignment.

Differencing and binding compose. Differencing-then-binding gives joint distributions of $n$-tuples of consecutive inter-event differences: $n$-grams of melodic intervals when the attribute is pitch, of inter-onset intervals when it is time. The composition recovers the n-tuple entropy of Milne & Dean (2016) as a special case (uniform weights, integer-valued steps, periodic domain, $\sigma \to 0$, integer-step grid) while extending it to the smoothed continuous case, non-integer values, weighted events, and non-periodic domains; and for $n \ge 2$ the bound MAET is itself an $n$-dimensional density, supporting cosine-similarity comparison of $n$-tuple distributions across pieces and the rest of the toolbox pipeline. The convenience wrapper `nTupleEntropy` calls this pipeline with default arguments matching the original Milne & Dean formulation.

Binding alone, applied to the raw event values without a preceding differencing step, gives n-grams in absolute pitch or time register, useful when register or absolute timing carries musical information that the differenced view discards: a leitmotif identified by its specific octave, an onset pattern keyed to a metric position, a chord progression as a sequence of identified harmonic functions. To localise the n-grams in time, an explicit time or event-index attribute can be carried alongside the bound categorical or pitch attributes, with the time stamp travelling with each window (typically the centre or last constituent's onset, paralleling the end-alignment convention used by `differenceEvents`).

#### Post-tensor windowing

A common analysis question — "at each point in this longer passage, how similar is the query to the local content?" — is handled by two functions that operate on pre-built MAETs: `windowTensor` and `windowedSimilarity`. `windowTensor` wraps a MAET with a window specification (per-group widths, shapes, and centres), returning a `WindowedMaetDensity` that is consumed lazily by `evalExpTens` (pointwise multiplication by the window) and by `cosSimExpTens` (closed-form windowed inner product). `windowedSimilarity` sweeps a series of *offsets* and returns the windowed-similarity profile — one similarity value per offset. The most common application is a sliding time window — "show me where in this melody the motif recurs" — but the same machinery applies to any group, so windowing in register, in pitch class, or along any other attribute axis — or combinations of them — is equally possible.

**Cross-correlation semantics.** `windowedSimilarity` implements cross-correlation: at each offset $\delta_g$ in a windowed group, the query is translated so that its effective-space centroid $\mu_q$ within that group moves onto the window centre $\mu_q + \delta_g$, and then the windowed similarity between the translated query and the window-weighted context is computed. Offset zero therefore places the window on the query's own centroid, and a peak at offset $\delta$ means the query pattern is present in the context displaced by $\delta$ from the query's centroid — the standard convention for matched-filter cross-correlation, where the profile is indexed by lag rather than by absolute position. A relative group (`isRel = true`) is translation-invariant within its own effective space, so when a window is placed on that group it does not need the cross-correlation translation; windows on other groups (typically time, alongside a relative pitch group) operate on their own offsets independently, and do need the translation. Cross-correlation semantics apply only when one operand is a `WindowedMaetDensity`; unwindowed calls to `cosSimExpTens` compute the cosine similarity directly, with no translation involved.

**Reference point for the cross-correlation.** The cross-correlation shift is measured from a reference point to the window centre on each windowed attribute. `windowedSimilarity` provides two options. The default, used when no reference is supplied, is the unweighted column mean of the query's tuple centres on each attribute: $\mathrm{ref}_a^{\mathrm{def}} = \tfrac{1}{N}\sum_{j=1}^N \mathbf{c}_{a,j}^q$, where $\mathbf{c}_{a,j}^q$ is the attribute-$a$ centre of tuple $j$ in the query density. This is a purely geometric property of the tuple centres, independent of the tuple weights. The alternative is a user-supplied fixed reference, one vector per attribute.

The choice always matters if queries differ in any property that shifts peaks in the similarity profile. For single-slot attributes ($K_a = 1$), between-query variation of a slot's value is a translation of the tuple centres on that attribute; the default's reference then co-moves with the query and a fixed reference does not, and the choice depends on whether offsets should be read as query-relative displacement or as positions relative to an absolute anchor. The more substantive distinctions emerge when $K_a > 1$, because queries can then differ not only in slot *values* but also in slot *count* (adding or removing partials) and in slot *weights* (e.g., rolloff changes). In practice, multi-slot attributes are pitch attributes: either pitch attributes with $K_a > 1$ representing several pitches per event (e.g., chords with exchangeable voices), or pitch attributes expanded by `addSpectra` to place partials alongside the fundamentals. The remainder of this subsection discusses the reference-point choice in the context of multi-slot pitch attributes.

Whether the two reference methods give peak offsets that line up across a family of queries depends on how the queries vary. If they differ only in slot *weights* (for example, a rolloff sweep), both methods give identical and stable peak offsets, for any slot structure. If they differ in slot *values* (for example, a partial-stretching sweep), the default's peak offset drifts while a fixed reference's stays put, again for any slot structure. If they differ in slot *count* (for example, adding partials), both methods give drifting peak offsets in general, but one special case is distinguished: for queries whose slots lie at (or close to) integer-harmonic values — that is, at $f_e + 1200 \log_2 n_k$ cents for some set of positive integers $\{n_k\}$ — the default's peak offset is stable across slot-count changes, because the geometric centroid and the best-match absolute pitch co-move closely as partials are added. We refer to queries of this form as **harmonic queries** on the pitch attribute; the term here refers only to slot value placement on a log-frequency axis and is independent of any musical-harmony usage of the word. The demo `demo_windowingReference` documents these behaviours in detail, with plots of similarity profiles across incremental sweeps in slot weights, slot values, and slot count, and with the harmonic special case separately characterised.

**Practical guidance.**

*Default reference.* Appropriate when both queries being compared are harmonic queries on their multi-slot pitch attribute and the between-query variation is restricted to slot count and/or slot weights. The default then places peak offsets at the musical transposition interval between the queries and keeps this reading stable across the sweep.

*Fixed reference calibrated to a canonical harmonic query.* Appropriate for analyses that include any non-harmonic query, or any slot-value variation, or whenever the offset axis is to be calibrated to musical transposition intervals independent of the query being scored. Choose the fixed reference as the unweighted centroid of a canonical harmonic query (e.g., a 12-partial complex tone on the chosen fundamentals) and hold it fixed across all queries. Under this calibration, the harmonic baseline reproduces the default's musical-interval reading; non-harmonic perturbations then appear as changes in the profile's shape and peak amplitude rather than as drifts of the peak offset.

*Fixed reference calibrated to an arbitrary anchor.* Available for applications that need the offset axis anchored to a position with meaning outside the similarity computation. Offsets under this calibration have no automatic musical-interval reading but are comparable across arbitrary query variations.

The `reference` keyword (Python) / name-value pair (MATLAB) accepts one array per query attribute, each of length equal to that attribute's effective-space dimension; omitting it selects the default.

**A shape-parameterised window family.** The window is specified per group by two numbers: a width parameter `size` (in multiples of that group's σ) and a shape parameter `mix` in [0, 1]. At `mix = 0` the window is a pure Gaussian; at `mix = 1` it is a pure rectangular (boxcar) window; intermediate values produce flat-topped shapes formed by convolution of a rectangular window with a Gaussian. This one-parameter interpolation lets the user match the window shape to the problem at hand — soft edges when a smooth trade-off between locality and coverage is wanted, sharp edges when an in-or-out selection of events is wanted, or anything in between. The `size` parameter controls the window's effective width independently of its shape. The windowed inner product between a windowed MAET and an unwindowed MAET has a closed-form analytical solution across the entire `(size, mix)` family for all 1-D groups (e.g., time, which always has effective dimension 1) and for all multi-dimensional absolute groups; for multi-dimensional relative groups ($r \ge 3$ with `isRel = true`), the closed form is currently established for the pure-Gaussian case only, with a Gaussian-mixture approximation of the full family being a possible extension. Evaluation of a windowed MAET via `evalExpTens` (pointwise multiplication by the window) is closed-form across the full family in any group dimensionality — the inner-product restriction does not apply to pointwise evaluation, so `entropyExpTens` on a windowed MAET is also supported uniformly.

**Periodic groups: line-case approximation.** The closed-form expression evaluated by `cosSimExpTens` for a windowed × unwindowed pair is the *line-case* formula — exact for non-periodic groups, but only an approximation when applied to a periodic group whose window support is a non-negligible fraction of one period. Three regimes can be distinguished:

1. **Window much smaller than one period** ($\lambda\sigma \ll P/2$). The line-case formula approximates the true periodic windowed inner product accurately. This is the regime of intended use.

2. **Window comparable to one period** ($\lambda\sigma \sim P/2$). The line-case formula loses accuracy as periodic images of the window's support begin to overlap. The exact closed form in this regime is a finite sum over periodic images of the line-case kernel; this is left to future work.

3. **Window larger than one period** ($\lambda\sigma \gg P$). The windowed inner product collapses to the unwindowed form, which can be obtained directly from `cosSimExpTens` (MATLAB) or `cos_sim_exp_tens` (Python) without going through the windowing machinery.

`windowedSimilarity` (MATLAB) / `windowed_similarity` (Python) emits a warning whenever it is called on a windowed periodic group whose window standard deviation $\lambda\sigma$ is at least $P/4$ (regimes 2 and 3 above). The MATLAB warning identifier is `windowedSimilarity:periodicWindowApprox`; the Python warning class is `WindowedSimilarityPeriodicApproxWarning` (registered with an `"always"` filter so it fires on every offending call, matching the MATLAB per-call behaviour). The warning is suppressible via the standard MATLAB `warning('off', 'windowedSimilarity:periodicWindowApprox')` mechanism or the Python `warnings.filterwarnings('ignore', category=WindowedSimilarityPeriodicApproxWarning)` mechanism if the user has determined that the line-case approximation is acceptable for their use case.

**Absolute versus offset semantics.** `windowTensor` takes a `centre` field in its spec that sets the window's absolute position in effective space, for users who want to evaluate a single windowed density pointwise (e.g., to visualise the window's weighting of the context at a chosen location). `windowedSimilarity` is one level up: users supply *offsets* from the query's centroid, and the function resolves them to absolute centres internally before calling `windowTensor` for each sweep position. A `centre` field in the spec passed to `windowedSimilarity` is therefore ignored — offsets replace it. This separation reflects the natural scope of each function: `windowTensor` is a low-level primitive in context coordinates, while `windowedSimilarity` is a higher-level cross-correlation that is naturally indexed by lag.

**Magnitude-aware normalisation, not strict cosine similarity.** `windowedSimilarity` normalises the windowed inner product by the product of the *unwindowed* L2 norms of the two operands. The resulting profile value at each sweep position reflects both how well the local content matches the query and how much matching content is present: a region of the context with a small amount of near-perfectly-matching content produces a smaller profile value than a region with a large amount of equally-matching content, because the windowed inner product scales with the amount of matching mass and the denominator (unwindowed norms) does not. An alternative normalisation by the product of the query's norm with the *windowed* context norm — a strict shape-only cosine — would cancel out this magnitude information, which is undesirable for sliding-motif and probe-tone analyses where dense matching regions should score higher than sparse ones. The output is therefore not bounded in $[-1, 1]$ across sweep positions and does not correspond to an inner product on a single Hilbert space; the term *cosine similarity* is reserved for the strict shape-only form (which is not currently implemented in the toolbox), and *windowed similarity* is used for what `windowedSimilarity` returns. Multiplying either operand's weights by a constant leaves the profile unchanged, because both numerator and denominator scale together; profiles from different inputs whose weight scales differ are therefore directly comparable.

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

It is also possible to use higher-order tensors. Setting r = 2 with isRel = true gives a transposition-invariant density over intervals; r = 3 with isRel = true gives a density over ordered triples of intervals; and so on. These capture different aspects of the similarity and are useful in specific contexts — for example, the equal-division-of-the-octave (EDO) approximation examples in Milne et al. (2011, Examples 6.3–6.5) use relative dyad tensors (r = 2, isRel = true) to produce one-dimensional interval-based approximations. Higher-order relative tensors are also central to `tensorHarmonicity`, which builds an r = K tensor (where K is the chord cardinality) from a harmonic series template and queries it at the chord's intervals.

The P(C)S measures themselves are domain-agnostic — the same cosine similarity applies equally to multisets of time points (yielding a measure of rhythmic similarity) or to multisets on any other axis where the (isRel, isPer) choices correspond to meaningful equivalences. Spectral augmentation is naturally a pitch-only phenomenon — partials arise from the physics of pitched sounds — and is accordingly optional; the SP(C)S and P(C)S measures differ only in whether `addSpectra` is applied before the cosine similarity.

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

### 3.5 Structural measures on periodic cycles

Several toolbox functions compute structural and perceptual features of multisets of points distributed around a periodic cycle — most obviously pitches (with period = octave, giving pitch classes) and time points (with period = cycle length, giving rhythmic patterns). These measures have been validated as predictors of rhythm recognition and preference (Milne & Herff, 2020), perceived rhythmic complexity and liking, tapping accuracy across a wide variety of rhythmic structures (Milne, Dean, & Bulger, 2023), and have informed the design of algorithmic rhythm generators (Milne & Dean, 2016; Milne, 2019). The functions are organised below by the granularity of their output: *period-level measures* return a single value (or small coefficient vector) characterising the whole multiset; *integer-position measures* return a value at each integer position in a discrete cycle; *continuous-position measures* are evaluable at any real-valued position around the circle.

**Period-level measures.** These return a single value (or coefficient vector) per multiset, characterising a global property of its distribution around the cycle. Most accept real-valued event positions; `sameness` and `nTupleEntropy` require integer positions within a discrete cycle.

- **Balance** (`balanceCircular`) — Measures how evenly the mass is distributed around the circle (1 = perfectly balanced, with the centre of gravity at the circle's centre; 0 = all weight at one point) (Milne, Bulger, & Herff, 2017). Computed directly from the positions of the points on the circle — mapping each point to the unit circle and taking the Fourier transform — rather than from an indicator vector over a discretised grid.
- **Evenness** (`evennessCircular`) — Measures closeness to equal spacing (1 = equally spaced; 0 = maximally uneven) (Milne, Bulger, & Herff, 2017).
- **Discrete Fourier transform** (`dftCircular`) — Underlies balance and evenness; its higher-order coefficients capture finer distributional structure (Milne, Bulger, & Herff, 2017; Milne & Herff, 2020). The mathematical foundations — including the relationship between the DFT coefficients and balance, evenness, and perfect balance — are developed in Milne, Bulger, & Herff (2017).
- **Coherence** (`coherence`) — The coherence quotient (Balzano, 1982; Carey, 2002; Rothenberg, 1978) measures how consistently the ordering of intervals by generic span (number of scale steps) matches their ordering by specific size (number of chromatic steps). A coherent scale or rhythm is one where larger generic spans always correspond to larger specific sizes — hearing an interval's size uniquely identifies how many scale steps it spans.
- **Sameness** (`sameness`) — The sameness quotient (Carey, 2002, 2007) measures the proportion of interval sizes that are unambiguous: each specific size belongs to exactly one generic span. A scale with high sameness has a transparent relationship between its chromatic and generic interval structure.
- **n-tuple entropy** (`nTupleEntropy`) — Entropy of the distribution of consecutive step-size sequences, capturing sequential predictability (Milne & Dean, 2016). When n = 1, this is IOI / step-size entropy. With default arguments, replicates Milne & Dean's discrete formulation; optional Gaussian smoothing (Milne, 2024) and finer grid resolution are available. The function is a convenience wrapper around the `differenceEvents` + `bindEvents` + `entropyExpTens` pipeline (see §3.1, Cross-event preprocessing), which can be called directly for non-integer values, weighted events, non-periodic domains, or to obtain the n-tuple density itself for similarity comparison and other MAET operations.

**Integer-position measures.** These operate on integer event positions within a discrete cycle of integer length (an equal division of the period) and return one value per integer position, so they apply to scales in an equal temperament (e.g., 12-EDO) and to rhythms quantised to a metrical grid:

- **Circular autocorrelation phase matrix** (`circApm`) — Decomposes the circular autocorrelation by lag and phase, yielding a metrical weight profile. Adapted from Eck's (2006) non-circular autocorrelation phase matrix to the circular (periodic) case by Milne, Dean, & Bulger (2023).
- **Markov prediction** (`markovS`) — Returns the optimal S-step lookahead prediction at each position in the cycle (Milne, Dean, & Bulger, 2023).

**Continuous-position measures.** These accept non-integer event positions and can be evaluated at any position around the circle, so they apply to scales and rhythms in any tuning or timing, not just equal divisions:

- **Edge detection** (`edges`) — Adapts the standard edge-detection technique from image processing to the circular domain, identifying sharp transitions between event-dense and event-sparse regions of the cycle via the first derivative of a von Mises kernel (Milne, Dean, & Bulger, 2023).
- **Projected centroid** (`projCentroid`) — Projects the circular centre of gravity onto each angular position, giving a position-level generalisation of the rhythm- or pitch-class-set-level balance measure (Milne, Dean, & Bulger, 2023).
- **Mean offset** (`meanOffset`) — For each position, the net upward arc to all events. In a pitch-class context, this formalises and generalises Huron's (2008) "average pitch height," making the position-dependence explicit: it returns a value for every position around the circle, including non-scale-tone positions, capturing the "brightness" or "darkness" of a mode as seen from each chromatic position. The related concept of "mode height" is used by Hearne (2020) and Tymoczko (2023). Introduced as a rhythmic predictor in Milne, Dean, & Bulger (2023).


### 3.6 Sequential-analysis utilities

Two utilities support analyses of ordered event sequences: `continuity` summarises the recent trend in a pitch (or IOI, or second-difference, etc.) sequence leading up to a query point, and `seqWeights` constructs position-weight vectors that can be passed as weights to any tensor or spectral function. `continuity` shares the difference-event substrate with the MAET-with-differencing pipeline but reads it as an ordered sequence rather than aggregating it into a tensor; `seqWeights` is a generic weight-vector constructor used in serial, time-indexed settings.

#### Direction continuity

`continuity` provides a smoothed analogue of the backward same-direction run leading up to a query point — "was the melody rising or falling in the lead-up to this moment, and for how long?". Given a sequence $p_1, \ldots, p_N$ and a query $q$, let $i_k = p_{k+1} - p_k$ and $i_N = q - p_N$; under independent Gaussian pitch uncertainty with standard deviation $\sigma$, the expected product of the sign of $i_k$ with the sign of $i_N$ is $a_k = \mathrm{erf}(i_k / (2\sigma)) \cdot \mathrm{erf}(i_N / (2\sigma))$. The function walks backward from the most recent context interval, accumulating $\max(a_k, 0)$ (contributing to the expected run length, returned as `count`) and $\max(a_k, 0) \cdot i_k$ (contributing to a signed magnitude, returned as `magnitude`). Two break-threshold modes (`'strict'`, `'lenient'`) set $\theta$ to $0$ or $-1$ respectively; an explicit threshold in $[-1, +1]$ is accepted as an override. The ratio `magnitude / count` gives a trend-slope measure.

An optional `w` argument supplies per-event salience weights $w_1, \ldots, w_N$ (non-negative; a scalar broadcasts to a length-$N$ uniform vector, `[]` / `None` broadcasts weight 1 to every event). Under the standard weights-as-salience reading, the salience of difference event $k$ — the interval $(p_k, p_{k+1})$ — is the product $w_k \cdot w_{k+1}$, interpretable as the probability that both endpoints are perceived. `continuity` scales each $\max(a_k, 0)$ contribution by this difference-event salience, so $\mathrm{count} = \sum_k w_k w_{k+1} \max(a_k, 0)$ and $\mathrm{magnitude} = \sum_k w_k w_{k+1} \max(a_k, 0) \cdot i_k$ (summed over the backward run up to the break). The directional-break threshold $\theta$ acts on the unweighted sign-product $a_k$, so weights modulate contribution size without shifting where the walk halts. The rolling-product $w_k \cdot w_{k+1}$ is exactly what `differenceEvents` produces at differencing order 1; `continuity` consumes the same `(pAttrDiff, wDiff)` stream that the MAET-with-differencing pipeline does — the preprocessing is shared, the read-out is not.

Direction continuity is defined only on linearly ordered domains — those where the ordering is inherited from the real line. For pitch, these include pitch heights, pitch intervals (differences), signed interval changes (second differences), and higher-order differences as needed. For time, the analogous sequence starts one level higher — with IOIs (differences between successive event times), then signed IOI changes, and so on — because event times are by convention monotonically increasing, so direction on the raw time stamps is trivially always positive and carries no information relevant to continuity; only from IOIs onward can the sequence change direction. The function as currently implemented does not handle periodic data — pitch-class sequences, metric position, or any phase-like attribute — because raw differences without modular wrapping produce spurious "direction" jumps each time the sequence crosses the period boundary. Raw clock time is not periodic in this sense (event times are monotonically increasing), so this limitation applies only when time is modelled via a periodic quantity such as metric position. The shared preprocessing with `differenceEvents` does not extend to the read-out itself: `continuity` depends on the *sign* of successive intervals and performs a sequential backward walk with a break condition (a nonlinear, order-dependent, stateful operation on adjacent elements), whereas the expectation-tensor machinery compares marginal distributions via quadratic forms and does not expose the sequential-adjacency structure that a sign product between consecutive intervals and a break-on-reversal recurrence require. The two are complementary reads of the same difference-event stream.

#### Position weighting

`seqWeights` constructs length-N weight vectors that combine an intrinsic per-event salience with a named time-based memory-decay or attentional profile. Named specifications include `'flat'` (uniform), `'primacy'`, `'recency'`, `'exponentialFromStart'`, `'exponentialFromEnd'`, and `'uShape'` (a convex combination of the two exponentials controlled by an `alpha` parameter). An explicit numeric vector is accepted as a passthrough. When a strictly-increasing time index is supplied, decay operates over elapsed time from the relevant endpoint rather than over position index. `seqWeights` is a generic weight-vector constructor: its output is a plain numeric vector, usable with any function that accepts a weight argument. It is presented here because analyses of event sequences over time — where memory decay and metrical accent are natural considerations — are its most natural use case, but the same vector can be passed into an ordinary single-attribute `buildExpTens` call (for recency-weighted SPCS, for example), supplied as the `w` argument to `addSpectra` so that after spectral enrichment each partial's weight reflects the position weight of its source pitch, or passed as the `w` argument to `continuity` for a recency- or primacy-weighted directional run.

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
| Default (all ones) weights | `[]` | `None` |
| Boolean flags | `true` / `false` | `True` / `False` |
| Spectrum arguments | Cell array: `{'harmonic', 12, 'powerlaw', 1}` | List: `['harmonic', 12, 'powerlaw', 1]` |
| Name-value pairs | `'name', value` | `name=value` |
| Precomputed density | Struct with `.tag = 'ExpTensDensity'` | `ExpTensDensity` dataclass |

### Weight arguments

Functions that accept a weight argument `w` treat it as a *broadcast specification* of a weight function, not as a fixed-shape array. The accepted forms, for a multiset of $N$ events with $K$ slots per event (with $K = 1$ for single-slot attributes), are:

| Input form | MATLAB | Python |
|:---|:---|:---|
| Default (uniform 1) | `[]` | `None` |
| Uniform scalar `c` | `c` | `c` |
| Per-event (broadcast across slots) | length-$N$ row | 1-D length-$N$, or `(1, N)` |
| Per-slot (broadcast across events) | $K \times 1$ column | `(K, 1)`, or 1-D length-$K$ |
| Full per-slot-per-event | $K \times N$ matrix | `(K, N)` |

The per-slot and full forms are meaningful only when $K > 1$ — pitch attributes with multiple pitches per event (chords with exchangeable voices), or spectrally-enriched pitches where each fundamental is represented by $K$ partials. Functions operating on a 1-D event sequence (`continuity`, `seqWeights`, `differenceEvents`) accept only the first three forms.

Different inputs that specify the same underlying weight function are semantically equivalent: a scalar $c$ and a length-$N$ vector of $c$s produce identical downstream output, as do a $K \times 1$ column and its $K \times N$ broadcast. Functions preserve the compact representation where they can, for efficiency, but the output of one function fed into another is interpreted under this same convention, so the compact and broadcast-out forms are interchangeable in chained calls. Weights are non-negative; the standard reading is $w_i$ = probability that event (or slot-event) $i$ is perceived, with `differenceEvents` and `continuity` propagating weights consistently with this interpretation.

For the MAET calling form, `w` is a per-attribute cell (MATLAB) / list (Python) whose entries follow the single-attribute rules above. A top-level `[]` / `None` or scalar is accepted as a shortcut that applies the same default — all ones or the given scalar — to every attribute.

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
| `windowTensor` | `window_tensor` | Tensor core |
| `windowedSimilarity` | `windowed_similarity` | Tensor core |
| `differenceEvents` | `difference_events` | Tensor core (preprocessing) |
| `bindEvents` | `bind_events` | Tensor core (preprocessing) |
| `simplexVertices` | `simplex_vertices` | Tensor core (utility) |
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
| `continuity` | `continuity` | Serial |
| `seqWeights` | `seq_weights` | Serial |
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

### Probe-tone fit to a context

In the classic probe-tone paradigm a listener hears a context sequence followed by a probe, and rates how well the probe fits. The simplest model of this fit is a single similarity value (SPCS) between the probe and the pooled spectrum of the context. The example below computes this fit directly with `cosSimExpTens` on two single-attribute tensors. Context events are recency-weighted using `seqWeights`, reflecting the intuition that later events are more salient in working memory.

**MATLAB:**
```matlab
context = convertPitch([60 64 67 72], 'midi', 'cents');   % C E G C
probeE  = convertPitch(64, 'midi', 'cents');              % probe: E

% Recency decay: weights later events more heavily
w = seqWeights([], 'exponentialFromEnd', 'N', 4, 'decayRate', 0.5);

% Apply spectral enrichment, folding the decay weights into the pitch weights
spec = {'harmonic', 12, 'powerlaw', 1};
[ctx_p, ctx_w] = addSpectra(context, w, spec{:});
[probe_p, probe_w] = addSpectra(probeE, [], spec{:});

s = cosSimExpTens(ctx_p, ctx_w, probe_p, probe_w, 10, 1, false, true, 1200);
fprintf('Probe-E fit (recency-weighted): %.3f\n', s);
```

**Python:**
```python
import numpy as np
context = mpt.convert_pitch(np.array([60, 64, 67, 72]), 'midi', 'cents')
probe_E = mpt.convert_pitch(np.array([64]), 'midi', 'cents')

w = mpt.seq_weights(None, 'exponentialFromEnd', n=4, decay_rate=0.5)

spec = ('harmonic', 12, 'powerlaw', 1.0)
ctx_p, ctx_w = mpt.add_spectra(context, w, *spec)
probe_p, probe_w = mpt.add_spectra(probe_E, None, *spec)

s = mpt.cos_sim_exp_tens_raw(ctx_p, ctx_w, probe_p, probe_w,
                              10.0, 1, False, True, 1200.0)
print(f'Probe-E fit (recency-weighted): {s:.3f}')
```

This approach treats the context as an unordered pitch multiset with per-event salience weights. For questions that depend on *when* events occurred — probe fit as a function of time, for example, rather than a single fit value — time can be carried as a second attribute on a multi-attribute tensor; see [Section 7.4](#74-probe-tone-scanning-with-irregular-timing-and-event-weights) for a worked example.

### Finding where a pattern occurs within a melody

The previous subsection returned a single similarity value — the fit of a probe against the whole context. For longer sequences, it is often useful to identify *where* a query pattern most closely matches the content of the sequence, returning one similarity per position — a similarity profile. Using time as a second attribute of a multi-attribute expectation tensor, `windowedSimilarity` slides a time-centred window along the context and returns a windowed similarity at each offset. `windowedSimilarity` implements cross-correlation: offsets are measured from the query's centroid to the window centre, so a peak at offset $\delta$ means the query pattern is present in the melody displaced by $\delta$ from the query's own centroid (see Section 3.1, "Post-tensor windowing").

**MATLAB:**
```matlab
% Melody: C D E G C E G (in MIDI), with event times 0..6
melody_p = convertPitch([60 62 64 67 60 64 67], 'midi', 'cents');
melody_t = 0:6;

% Query: the 2-event pattern E G (centroid at t = 0.5)
query_p = convertPitch([64 67], 'midi', 'cents');
query_t = [0 1];

% Build MAETs for both. Pitch group: absolute periodic (σ = 10 cents).
% Time group: absolute non-periodic (σ = 0.2 time units).
melody = buildExpTens({melody_p, melody_t}, [], ...
                      [10 0.2], [1 1], [], ...
                      [false false], [true false], [1200 0]);
query  = buildExpTens({query_p, query_t}, [], ...
                      [10 0.2], [1 1], [], ...
                      [false false], [true false], [1200 0]);

% Sweep time offsets across the context with a Gaussian window on time.
% size = 2 corresponds to a window width of 2 σ_time = 0.4 time units.
% Offset 0 places the window on the query's centroid (t = 0.5); offset 2.5
% places it on absolute time t = 3, where the E-G pair recurs.
t_offsets = linspace(-1.0, 6.0, 29);
offsets = [zeros(1, numel(t_offsets)); t_offsets];   % 2 x M: pitch row unused
spec = struct('size', [Inf 2.0], 'mix', [0 0]);       % no window on pitch; Gaussian on time

S = windowedSimilarity(query, melody, spec, offsets);
disp(S);
```

**Python:**
```python
import numpy as np
melody_p = mpt.convert_pitch(np.array([60, 62, 64, 67, 60, 64, 67]), 'midi', 'cents')
melody_t = np.arange(7, dtype=float)

query_p = mpt.convert_pitch(np.array([64, 67]), 'midi', 'cents')
query_t = np.array([0., 1.])

melody = mpt.build_exp_tens([melody_p, melody_t], None,
                             [10., 0.2], [1, 1], None,
                             [False, False], [True, False], [1200., 0.])
query  = mpt.build_exp_tens([query_p, query_t], None,
                             [10., 0.2], [1, 1], None,
                             [False, False], [True, False], [1200., 0.])

t_offsets = np.linspace(-1.0, 6.0, 29)
offsets = np.vstack([np.zeros_like(t_offsets), t_offsets])
spec = {'size': [np.inf, 2.0], 'mix': [0.0, 0.0]}

S = mpt.windowed_similarity(query, melody, spec, offsets, verbose=False)
print(S)
```

The query E G, with mean time 0.5, is translated at each sweep centre $c$ to lie at times $(c-0.5, c+0.5)$. The melody contains the E–G pair at events $(t=2, t=3)$ and again at $(t=5, t=6)$, with respective mean times 2.5 and 5.5, so the profile peaks at $c = 2.5$ and $c = 5.5$. Positions between and outside these (including the passing-tone E at $t=2$ alone, without a G at its successor) produce graded similarity determined by how closely the translated query aligns with nearby melody events. See [Section 7.5](#75-recurrence-of-interval-content-across-a-melody) for a fuller worked example including transposition-invariant comparison of interval content via event differencing.

---

## 6. Function reference

Functions are grouped by category. For full details, use `help functionName` in MATLAB or `help(mpt.function_name)` in Python. Code examples in this section use MATLAB syntax; see [Section 4](#4-api-conventions-matlab-vs-python) for the systematic Python equivalents.

### 6.1 Expectation tensor core and MAET preprocessing

The functions below group naturally into three layers: *preprocessing* (transforms raw inputs before tensor construction), *core* (constructs, evaluates, or compares density objects), and *utility* (helpers that operate alongside the core functions).

| MATLAB | Python | Description |
|:---|:---|:---|
| `buildExpTens` | `build_exp_tens` | Precompute an r-ad expectation tensor density object |
| `evalExpTens` | `eval_exp_tens` | Evaluate the density at query points |
| `cosSimExpTens` | `cos_sim_exp_tens` | Cosine similarity of two expectation tensor densities |
| `batchCosSimExpTens` | `batch_cos_sim_exp_tens` | Batch cosine similarity with deduplication |
| `entropyExpTens` | `entropy_exp_tens` | Shannon entropy of an expectation tensor |
| `windowTensor` | `window_tensor` | Wrap a MAET with a post-tensor window specification |
| `windowedSimilarity` | `windowed_similarity` | Sliding-window similarity profile (magnitude-aware) |
| `differenceEvents` | `difference_events` | *Preprocessing:* replace event sequences with inter-event differences |
| `bindEvents` | `bind_events` | *Preprocessing:* gather n consecutive events into n-attribute super-events |
| `estimateCompTime` | `estimate_comp_time` | *Utility:* micro-benchmark-based computation time estimate |

**buildExpTens(p, w, sigma, r, isRel, isPer, period)**

Precomputes tuple indices, pitch matrices, and weight vectors for a weighted pitch multiset. Returns a struct (`tag = 'ExpTensDensity'`) that can be passed to `evalExpTens` and `cosSimExpTens`.

Key parameters:
- `p` — Pitch values (vector), or a cell array of per-attribute matrices for a multi-attribute tensor
- `w` — Weights (vector, or empty for all ones)
- `sigma` — Standard deviation of the Gaussian kernel (cents), or a per-group vector for MAETs
- `r` — Tuple size (positive integer; r ≥ 2 if isRel = true), or a per-attribute vector for MAETs
- `isRel` — Transposition-invariant if true (effective dim $= r - 1$)
- `isPer` — Periodic wrapping if true
- `period` — Period for wrapping (e.g., 1200 for one octave in cents)

Multi-attribute call form: `buildExpTens({p_1, …, p_A}, w, sigmaVec, rVec, groups, isRelVec, isPerVec, periodVec)`. Each `p_a` is a matrix of attribute values; `sigmaVec`, `isRelVec`, `isPerVec`, `periodVec` give per-group parameters; `groups` maps attributes to groups (can be omitted if each attribute is its own group). Returns a struct (`tag = 'MaetDensity'`) compatible with all downstream tensor functions.

**evalExpTens(dens, X [, normalize])**

Evaluates the density at the query points given by the columns of X. Three normalization modes:
- `'none'` (default) — Raw weighted sum of Gaussian kernels. The absolute value depends on σ, the number of tuples, and the weight magnitudes. Only the relative values across query points are meaningful. Sufficient for visualization and cosine similarity, where any normalization cancels.
- `'gaussian'` — Each Gaussian component is normalized to integrate to 1. After this normalization, the total density integrates to the sum of all tuple weight products. Useful for comparing densities computed with different σ values: increasing σ spreads the same mass over a wider area rather than inflating the total integral.
- `'pdf'` — Full probability density normalization. Applies the Gaussian normalization above, then divides by the sum of all tuple weight products so the density integrates to 1 over the domain. Useful for comparing densities across pitch multisets of different sizes, or for computing entropy.

Query points X should have `dim` rows, where $\mathrm{dim} = r - \mathrm{isRel}$. For the relative case ($\mathrm{dim} = r - 1$), each column of X specifies the $r - 1$ intervals that define an r-ad (e.g., for a triad with $r = 3$ and isRel = true, each column is a 2-element vector of intervals from the lowest pitch to the middle and highest pitches). For a MAET, X is either a cell array of per-attribute query matrices or a single matrix with per-attribute row blocks concatenated. A `WindowedMaetDensity` is also accepted and is evaluated as the underlying density multiplied pointwise by the window function.

**cosSimExpTens(dens_x, dens_y)** or **cosSimExpTens(p1, w1, p2, w2, sigma, r, isRel, isPer, period)**

Computes the cosine similarity between two expectation tensor densities analytically. The precomputed-struct calling convention avoids recomputing tuple indices on each call. Both conventions support `'verbose', false`. Accepts single-attribute `ExpTensDensity`, multi-attribute `MaetDensity`, or `WindowedMaetDensity` (the latter compares against an unwindowed counterpart using the magnitude-aware normalisation described in Section 3.1, "Post-tensor windowing"). For multi-attribute comparisons, the two densities must share the same attribute structure and per-group parameters; see the compatibility note at the end of Section 3.1.

**batchCosSimExpTens(pMatA, pMatB, sigma, r, isRel, isPer, period, ...)**

Computes cosine similarity for many paired weighted multisets. Each row of pMatA and pMatB defines one pair. Automatically deduplicates rows with identical sorted content, computing `cosSimExpTens` only once per unique pair. Optional name-value pairs: `'weightsA'`, `'weightsB'`, `'spectrum'` (cell array of `addSpectra` arguments), `'verbose'`.

**entropyExpTens(p, w, sigma, r, isRel, isPer, period, ...)**

Shannon entropy of the expectation tensor, discretized on a fine grid. Also accepts a precomputed struct as the first argument (`ExpTensDensity`, `MaetDensity`, or `WindowedMaetDensity`). Optional name-value pairs: `'spectrum'`, `'normalize'` (default: true), `'base'` (default: 2), `'nPointsPerDim'` (default: 1200), `'xMin'`, `'xMax'` (required when isPer = false), `'gridLimit'` (default: 10⁸ total grid points).

**windowTensor(dens, windowSpec)**

Wraps a `MaetDensity` with a post-tensor window specification, returning a `WindowedMaetDensity` struct. The window spec has three fields:
- `size` — Per-group window effective standard deviation in multiples of that group's σ. Scalar or length-G vector; NaN or Inf on an entry means the group is not windowed.
- `mix` — Per-group shape parameter in [0, 1]: 0 is pure Gaussian, 1 is pure rectangular, intermediate values are rectangular-convolved-with-Gaussian.
- `centre` — Length-A cell of per-attribute centre coordinates (each a column vector of length dim_per_attr(a)), or equivalently a flat column vector of total length `dim`.

No math is performed at construction time; the window is applied lazily by `evalExpTens` (pointwise) and `cosSimExpTens` (closed-form windowed inner product).

**windowedSimilarity(densQuery, densContext, windowSpec, offsets [, 'verbose', v])**

Sliding-window similarity profile (cross-correlation). For each column of `offsets` (a `dim × M` matrix), windows `densContext` with the given `windowSpec` at the corresponding centre and computes the windowed similarity against `densQuery` (unwindowed). Returns a 1 × M profile. Offsets are measured from the query's effective-space centroid to the window centre; the centroid is the unweighted column mean of the query's per-group tuple centres — a purely geometric property of the event positions, independent of the query's weight assignment (see Section 3.1 for the rationale for using the unweighted rather than the weighted mean). At offset 0 the window sits on the query's own centroid, so a peak at offset $\delta$ means the query pattern is present in the context displaced by $\delta$ from its centroid — the standard cross-correlation convention. Any `centre` field in the caller's `windowSpec` is ignored; offsets replace it. The normaliser uses the unwindowed L2 norms of both operands, so profile values reflect both how well the local content matches the query and how much matching content is present; the output is therefore a magnitude-aware *windowed similarity*, not a strict cosine similarity (see Section 3.1 "Magnitude-aware normalisation, not strict cosine similarity"). When applied to a periodic group whose window standard deviation $\lambda\sigma$ is at least $P/4$, a warning is emitted (identifier `windowedSimilarity:periodicWindowApprox` in MATLAB; class `WindowedSimilarityPeriodicApproxWarning` in Python; see Section 3.1 "Periodic groups: line-case approximation").

The analytical inner product is closed-form for the full `(size, mix)` family on 1-D groups (e.g., time) and on multi-D absolute groups. For multi-D relative groups (r ≥ 3 with isRel = true) only the pure-Gaussian case (mix = 0) is currently supported; a clear error is raised if a rectangular or raised-rectangular window is requested on such a group.

*MAET preprocessing.*

**differenceEvents(pAttr, w, groups, diffOrders, periods)**

Cross-event preprocessing for MAET input. Takes the `(pAttr, w)` pair that would otherwise be passed to `buildExpTens` and replaces the event sequences of selected groups with their $k$-th finite differences along the event axis, returning a `(pAttrDiff, wDiff)` pair in the same format. For each group, `diffOrders[g] = k` produces the $k$-th difference (reducing the event count by $k$); order 0 leaves a group unchanged. When `periods[g] > 0`, raw differences are wrapped to $[-P/2, P/2)$ using the shortest-arc convention. Weights follow the toolbox's standard broadcast convention (§4) and propagate as rolling products of width $k + 1$, so the weight of a differenced event is the product of the weights of the $k + 1$ constituent events it depends on (interpretable as the probability that all constituents are jointly perceived). When different groups use different orders, the output event count is $N' = N - \max_g k_g$, and lower-order groups have leading events dropped to keep columns aligned. The returned `(pAttrDiff, wDiff)` feeds directly into `buildExpTens` without further processing. Inputs are restricted to $K_a = 1$ per attribute; see §3.1 (Cross-event preprocessing) for the rationale and the voices-as-attributes pipeline for polyphonic analyses. Event differencing is distinct from `isRel = true`: the former replaces absolute per-event values with differences between adjacent events (a preprocessing step); the latter is a property of tensor construction that makes the within-r-ad density translation-invariant (no preprocessing involved). Both can be used together or independently. Composes with `bindEvents` (below) to produce n-grams of consecutive inter-event differences.

**bindEvents(p, w, n, ...)**

Cross-event preprocessing for MAET input. Slides a window of width $n$ across a single-attribute event sequence and emits each window as an $n$-attribute super-event whose $j$-th attribute holds the value at lag $j-1$. The output is a length-$n$ list of $1 \times N'$ matrices (where $N' = N - n + 1$ by default, or $N$ when `circular` is true), suitable as the `pAttr` argument of `buildExpTens` with all $n$ attributes assigned to a single group sharing $\sigma$, $r = 1$, `isRel`, `isPer`, and `period`. Per-event weights propagate as a rolling product of width $n$: the weight of a bound super-event is the product of the $n$ input-event weights it spans. Inputs are restricted to $K_a = 1$; lag slots must be carried as separate attributes, not as multi-slot values, because lag identity is non-exchangeable (within-attribute multiset symmetry would otherwise collapse ordered tuples to unordered ones). See §3.1 (Cross-event preprocessing) for the rationale.

Composes with `differenceEvents`: the standard pipeline `differenceEvents` $\to$ `bindEvents` $\to$ `buildExpTens` $\to$ `entropyExpTens` produces n-tuple entropy on the integer-step grid, recovering Milne & Dean (2016) at $\sigma \to 0$ and uniform weights, and extending it to the smoothed continuous case, weighted events, and non-periodic domains. The convenience function `nTupleEntropy` wraps this pipeline. Used without a preceding differencing step, `bindEvents` produces n-grams in absolute pitch or time register; pairing the bound block with an explicit time or event-index attribute (carried alongside, with the time stamp travelling with each window) localises the n-grams in time and supports time-windowed similarity for n-gram pattern matching.

**simplexVertices(N [, edgeLength])**

Returns the $N$ vertices of a regular $(N-1)$-simplex centred at the origin in $\mathbb{R}^{N-1}$, as an $N \times (N-1)$ matrix whose row $k$ is the coordinate vector for level $k$. All pairwise vertex distances equal `edgeLength` (default 1). Supports the simplex-coded encoding of an $N$-level categorical attribute (voice identity, instrument, etc.) for MAET input: each level becomes one row of the returned matrix, fed as values for the $N - 1$ numerical sub-attributes of a single categorical group (with `isPer = false`, `isRel = false`). Because all vertices are pairwise equidistant, no level is privileged over any other --- in contrast to the dummy or treatment codings familiar from regression. The categorical group's $\sigma$ then controls how sharply the levels are distinguished: at $\sigma = 0$ they are perfectly separated, and as $\sigma \to \infty$ they collapse to a level-blind representation. Construction uses the centred standard basis of $\mathbb{R}^N$ projected onto an orthonormal basis of $\mathbf{1}^\perp$; the result is rotation-equivalent across choices of basis, which is irrelevant for downstream MAET computations. Concrete shapes: $N = 2$ collapses to $\pm \tfrac{1}{2}$ on a line; $N = 3$ to an equilateral triangle in $\mathbb{R}^2$; $N = 4$ to a regular tetrahedron in $\mathbb{R}^3$. The complementary encoding --- one attribute per categorical level --- needs no helper, since it is constructed by simply assigning each level's events to a separate attribute.

### 6.2 Spectral enrichment

| MATLAB | Python | Description |
|:---|:---|:---|
| `addSpectra` | `add_spectra` | Add spectral partials to a weighted pitch multiset |

**addSpectra(p, w, mode, ...)**

Adds partials to each pitch. Mode is one of `'harmonic'`, `'stretched'`, `'freqlinear'`, `'stiff'`, or `'custom'`. All non-custom modes take N (number of partials including the fundamental), followed by a weight-type specification (`'powerlaw', rho` or `'geometric', tau`). The `'stretched'`, `'freqlinear'`, and `'stiff'` modes each have one additional parameter (β, α, or B respectively) between N and the weight type.

The optional name-value pair `'units', U` specifies pitch units per octave (default: 1200, i.e., cents). When using semitones, set `'units', 12`.

Output weights are the product of each pitch's original weight and the spectral weight of each partial.

### 6.3 Consonance and harmonicity

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

### 6.4 Balance and evenness (Fourier-based measures)

| MATLAB | Python | Description |
|:---|:---|:---|
| `dftCircular` | `dft_circular` | DFT of points on a circle |
| `balanceCircular` | `balance` | Balance (1 − \|F(0)\|) |
| `evennessCircular` | `evenness` | Evenness (\|F(1)\|) |

These functions apply equally to pitches (where the circle is one octave or other period) and to positions (where the circle is one rhythmic cycle or other periodic domain).

**dftCircular(p, w, period)** — Returns complex Fourier coefficients F and their magnitudes. F(1) (MATLAB 1-based) is the k = 0 coefficient; F(2) is k = 1; etc.

**balanceCircular(p, w, period)** — Returns a value in [0, 1]. Balance = 1 means the centre of gravity is at the circle's centre (e.g., augmented triad, whole-tone scale, isochronous rhythm). Supports weighted events.

**evennessCircular(p, period)** — Returns a value in [0, 1]. Evenness = 1 means the events are equally spaced (e.g., whole-tone scale, chromatic scale). Always uses uniform (binary) weights, following Milne et al. (2017).

### 6.5 Scale and rhythm structure

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

**nTupleEntropy(p, period [, n], ...)** — Shannon entropy of the distribution of n-tuples of consecutive step sizes. When n = 1 (the default), this is IOI / step-size entropy. Optional name-value pairs: `'sigma'` (Gaussian smoothing in step-size units, default: 0), `'normalize'` (default: true), `'base'` (default: 2), `'nPointsPerDim'` (grid resolution per effective dimension, default: 0 meaning *period*; the integer-step grid). With default arguments this exactly replicates Milne & Dean's (2016) discrete formulation; `sigma > 0` gives the smoothed extension of Milne (2024); `nPointsPerDim` finer than `period` discretises the underlying continuous density at finer resolution. As of v2.1.0, this function is a thin wrapper around the `differenceEvents` $\to$ `bindEvents` $\to$ `entropyExpTens` pipeline (see §6.1 and §3.1); call those primitives directly for non-integer values, weighted events, non-periodic domains, or to obtain the n-tuple density itself for similarity comparison and other MAET operations.

**circApm(p, w, period, ...)** — Returns the period × period autocorrelation phase matrix R, the metrical weight profile rPhase (column sum), and the circular autocorrelation rLag (row sum). Optional `'decay'` parameter for exponential decay weighting.

**edges(p, w, period [, x], ...)** — Circular edge detection via convolution with the first derivative of a von Mises kernel. Returns absolute and signed edge weights. Optional `'kappa'` parameter controls kernel width.

**projCentroid(p, w, period [, x])** — Projection of the circular centroid (k = 0 Fourier coefficient) onto each angular position. Returns the projection vector, centroid magnitude, and centroid phase.

**meanOffset(p, w, period [, x])** — For each query point, the weighted sum of (upward arc − downward arc) to all events, normalized by the period. In a pitch-class context, this formalizes and generalizes Huron's (2008) "average pitch height," making the position-dependence explicit: it returns a value for every position around the circle. The term "mode height" for a closely related concept is used by Hearne (2020) and Tymoczko (2023).

**markovS(p, w, period [, S])** — Optimal S-step Markov predictor (default S = 3). For each position in the cycle, finds all positions with an identical S-step future context and returns their average weight. Originally by David Bulger.

### 6.6 Ordered sequences

Utility functions for analyses of event sequences over time. Position-sensitive similarity of two sequences is computed via the core tensor functions on pitch-and-time-attributed multi-attribute tensors; see Sections 3.1 and 6.1 for `buildExpTens`, `cosSimExpTens`, `batchCosSimExpTens`, `windowTensor`, and `windowedSimilarity`.

| MATLAB | Python | Description |
|:---|:---|:---|
| `continuity` | `continuity` | Backward same-direction run |
| `seqWeights` | `seq_weights` | Position-weight vector constructor |

**continuity(seq, x, sigma [, 'w', w] [, 'mode', mode] [, 'theta', theta])** — Expected length and signed magnitude of the backward same-direction run leading up to each query, under Gaussian pitch uncertainty. Returns `[count, magnitude]`: `count` is non-negative, `magnitude` is signed (positive for ascending trends, negative for descending). The ratio `magnitude / count` gives a trend-slope measure. Modes `'strict'` (θ = 0) and `'lenient'` (θ = −1) set the break threshold; an explicit `'theta'` in [−1, +1] overrides. Optional per-event salience weights `w` (`[]` / `None` for all ones, a non-negative scalar, or a length-$N$ non-negative vector) scale each interval's contribution to `count` and `magnitude` by the difference-event salience $w_k \cdot w_{k+1}$ — the same rolling-product rule as `differenceEvents` at order 1. The break threshold acts on the unweighted sign-product, so weights modulate contribution size without shifting the halt condition. Defined only on linearly ordered domains.

**seqWeights(w, spec [, 'N', N] [, 'decayRate', d] [, 'alpha', a] [, 't', t])** — Apply a position-weighting profile to an existing weight vector. Constructs a length-N profile from `spec` and returns its pointwise product with `w`. `w` is a length-N vector of per-position weights, `[]` (MATLAB) / `None` (Python) for all ones — requires `'N'`, or a scalar broadcast to length N — requires `'N'`. The output length `N` is inferred from `numel(w)` when `w` is a non-empty, non-scalar vector; it must be supplied explicitly via the `'N'` name-value argument when `w` is empty or scalar. A mismatch between supplied `N` and `numel(w)` raises an error. `spec` is a named specification (`'flat'`, `'primacy'`, `'recency'`, `'exponentialFromStart'`, `'exponentialFromEnd'`, `'uShape'`) or an explicit length-N numeric vector (passthrough with length validation). `'flat'` produces a uniform profile; `'primacy'` and `'recency'` are point masses on the first and last positions. For exponential and uShape specs, `'decayRate'` is the non-negative decay (default 1; zero decay recovers `'flat'`), and `'alpha'` (default 0.5) controls the uShape mixing: `alpha = 1` recovers `'exponentialFromStart'`, `alpha = 0` recovers `'exponentialFromEnd'`. When a strictly-increasing time index `t` is supplied, decay operates over elapsed time from the relevant endpoint; when omitted, unit spacing is used. The returned vector can be passed as the event weights of `buildExpTens` (including in its MAET form, as per-event weights for a time-attributed tensor), or fed into `cosSimExpTens` via `addSpectra` for a position-weighted SPCS computation.

### 6.7 Utility

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

### 7.4 Probe-tone scanning with irregular timing and event weights

The Quick Start example of probe-tone fitting treated the context as an unordered pitch multiset with per-event salience weights, computing a single cosine similarity via `cosSimExpTens`. In realistic listening situations, context events are not uniformly spaced in time (ritardandi, held tones, metrical anacruses all break uniform spacing), and they are not equally salient (metrical position, loudness, duration, and phenomenal accent all modulate how strongly each event registers). A thorough probe-tone model accounts for both factors.

Two complementary approaches are available. The first is the pooled-context approach of the Quick Start: treat the context as a weighted multiset and compute a single similarity. The second is a *time-resolved* approach: represent the context as a multi-attribute tensor carrying both pitch and time, and sweep a window along the time axis to obtain a similarity *profile* showing how probe fit evolves moment by moment. The two approaches answer different questions — "what is the aggregate fit?" versus "where and when does the probe fit best?" — and both are naturally expressed in the v2.1.0 framework.

The example below scans twelve chromatic probes against a short melodic line `C F G C` representing a I–IV–V–I cadence, using the pooled-context approach. The final tonic is held for two beats (irregular time grid); the first and last events are metrically accented (irregular salience). Both effects are encoded by a pre-built weight vector from `seqWeights`, and each probe is compared against the same weighted context via `batchCosSimExpTens` for efficient scanning.

**MATLAB:**
```matlab
% Context: melodic line of a cadential progression
context = convertPitch([60 65 67 60], 'midi', 'cents');   % C F G C

% Irregular event times: final tonic held twice as long
t_events = [0 1 2 4];

% Per-event salience: accented first and last events
salience = [1.0; 0.6; 0.6; 1.3];

% Combine salience with exponential-from-end decay over elapsed time
w = seqWeights(salience, 'exponentialFromEnd', ...
               'decayRate', 0.4, 't', t_events);

% Apply spectral enrichment and build the context density once
spec = {'harmonic', 12, 'powerlaw', 1};
[ctx_p, ctx_w] = addSpectra(context, w, spec{:});
ctx_dens = buildExpTens(ctx_p, ctx_w, 10, 1, false, true, 1200);

% Scan 12 chromatic probes
probes = convertPitch(60:71, 'midi', 'cents');
fit = zeros(1, 12);
for i = 1:12
    [probe_p, probe_w] = addSpectra(probes(i), [], spec{:});
    probe_dens = buildExpTens(probe_p, probe_w, 10, 1, false, true, 1200);
    fit(i) = cosSimExpTens(ctx_dens, probe_dens);
end

bar(0:11, fit);
xlabel('Probe pitch class (semitones from C)');
ylabel('Spectral pitch-class fit');
title('Probe fit to C–F–G–C with time-aware, salience-weighted context');
```

**Python:**
```python
import numpy as np
context  = mpt.convert_pitch(np.array([60, 65, 67, 60]), 'midi', 'cents')
t_events = np.array([0.0, 1.0, 2.0, 4.0])
salience = np.array([1.0, 0.6, 0.6, 1.3])

w = mpt.seq_weights(salience, 'exponentialFromEnd',
                     decay_rate=0.4, t=t_events)

spec = ('harmonic', 12, 'powerlaw', 1.0)
ctx_p, ctx_w = mpt.add_spectra(context, w, *spec)
ctx_dens = mpt.build_exp_tens(ctx_p, ctx_w, 10., 1, False, True, 1200.)

probes = mpt.convert_pitch(np.arange(60, 72), 'midi', 'cents')
fit = np.empty(12)
for i in range(12):
    pp, pw = mpt.add_spectra(np.array([probes[i]]), None, *spec)
    probe_dens = mpt.build_exp_tens(pp, pw, 10., 1, False, True, 1200.)
    fit[i] = mpt.cos_sim_exp_tens(ctx_dens, probe_dens)
```

The resulting profile peaks at C (the tonic, present at both endpoints and carried by the most heavily weighted final event), with a secondary peak at G. The emphasis on the final tonic and the decay of the intermediate events fall out of the `seqWeights` profile — primacy and recency accents, plus time-aware exponential decay — composed into a single plain numeric vector that enters `buildExpTens` as the per-event pitch weights via `addSpectra`.

For a time-resolved view — how probe fit varies moment by moment along the context, rather than aggregated to a single number — the context can be represented as a multi-attribute tensor carrying both pitch and time as attributes, and scanned with `windowedSimilarity`. The following uses a single probe (C) to show the pattern. The probe is a single event at time 0, so its centroid is at 0, and the offsets passed to `windowedSimilarity` are absolute context times.

**MATLAB:**
```matlab
% Same context as above, now with explicit time attribute
ctx_maet = buildExpTens({context, t_events}, w, ...
                         [10 0.3], [1 1], [], ...
                         [false false], [true false], [1200 0]);

% Probe: single C event (time placeholder 0 -> centroid at 0)
probeC = convertPitch(60, 'midi', 'cents');
probe_maet = buildExpTens({probeC, 0}, [], ...
                           [10 0.3], [1 1], [], ...
                           [false false], [true false], [1200 0]);

% Sweep time offsets across the context. Because the probe's centroid
% is at t = 0, an offset of t here corresponds to placing the window at
% absolute time t in the context.
t_offsets = linspace(-0.5, 4.5, 51);
offsets = [zeros(1, numel(t_offsets)); t_offsets];
spec_w = struct('size', [Inf 1.0], 'mix', [0 0]);   % Gaussian window, width 1 time unit

profile = windowedSimilarity(probe_maet, ctx_maet, spec_w, offsets);
plot(t_offsets, profile);
xlabel('Time'); ylabel('Fit of probe C');
title('Time-resolved probe fit');
```

**Python:**
```python
ctx_maet = mpt.build_exp_tens([context, t_events], w,
                               [10., 0.3], [1, 1], None,
                               [False, False], [True, False], [1200., 0.])

probe_C = mpt.convert_pitch(np.array([60]), 'midi', 'cents')
probe_maet = mpt.build_exp_tens([probe_C, np.array([0.])], None,
                                 [10., 0.3], [1, 1], None,
                                 [False, False], [True, False], [1200., 0.])

t_offsets = np.linspace(-0.5, 4.5, 51)
offsets = np.vstack([np.zeros_like(t_offsets), t_offsets])
spec_w = {'size': [np.inf, 1.0], 'mix': [0.0, 0.0]}

profile = mpt.windowed_similarity(probe_maet, ctx_maet, spec_w, offsets,
                                 verbose=False)
```

The resulting profile peaks near times 0 and 4 (the C onsets) and is lower between them. The pooled-context scan and the time-resolved scan are complementary: the former asks "how well does the probe fit the context as a whole?", the latter "where within the context does the probe fit best?".

### 7.5 Recurrence of interval content across a melody

For locating recurrent interval content within a monophonic melody — *where does this motif reappear, regardless of its starting pitch?* — the analysis combines two steps: convert the pitch sequence into a sequence of inter-event intervals with `differenceEvents`, then sweep a time-windowed similarity over the differenced sequence with `windowedSimilarity`. The first step handles transposition invariance by construction (intervals are the same regardless of absolute pitch), and the second returns a similarity profile showing where the motif recurs.

The example below uses an eight-note melody, C D E♭ F A C D E♭, containing the interval pattern $(+2, +1)$ — a whole step followed by a semitone — at two places: positions 0–2 (C D E♭) and positions 5–7 (C D E♭ again, an octave higher). The query G A B♭ carries the same $(+2, +1)$ pattern but at a different absolute pitch. After `differenceEvents`, both melody and query live in interval space, and `windowedSimilarity` locates the motif by sliding a time window across the melody's interval sequence.

**MATLAB:**
```matlab
% Melody: C D Eb F A C D Eb, one event per time unit.
melody_p = convertPitch([60 62 63 65 69 72 74 75], 'midi', 'cents');
melody_t = 0:7;

% Query: G A Bb — same (+2, +1) pattern transposed.
query_p = convertPitch([67 69 70], 'midi', 'cents');
query_t = [0 1 2];

% Convert each event sequence into inter-event differences: pitch gets
% order-1 differencing (intervals); time gets order-0 (pass through, but
% the leading event is dropped so the event counts align with the pitch
% differences).
[mel_pd, ~] = differenceEvents({melody_p, melody_t}, [], [], [1 0], [0 0]);
[qry_pd, ~] = differenceEvents({query_p,  query_t},  [], [], [1 0], [0 0]);
% mel_pd{1} = [200 100 200 400 300 200 100]  (cents, signed)
% mel_pd{2} = [1 2 3 4 5 6 7]                (time of each interval event)

% Build r = 1 absolute non-periodic MAETs on the differenced sequences.
% Pitch group: sigma = 50 cents gives clean separation between 100-cent
% interval categories; non-periodic so that, e.g., +1300 is distinct
% from +100. Time group: sigma = 0.3 time units.
melody = buildExpTens(mel_pd, [], ...
                      [50 0.3], [1 1], [], ...
                      [false false], [false false], [0 0]);
query  = buildExpTens(qry_pd, [], ...
                      [50 0.3], [1 1], [], ...
                      [false false], [false false], [0 0]);

% Sweep offsets with a Gaussian window of width 2 sigma_time = 0.6.
% The query's differenced time centroid is 1.5 (events at t = 1, 2),
% so offset 0 places the window at absolute time 1.5; offset 5 places
% it at absolute time 6.5.
t_offsets = -1.5:0.25:6.5;
offsets = [zeros(1, numel(t_offsets)); t_offsets];
spec = struct('size', [Inf 2.0], 'mix', [0 0]);

S = windowedSimilarity(query, melody, spec, offsets);
bar(t_offsets, S);
xlabel('Offset from query centroid (time)');
ylabel('Interval-content similarity to query (+2, +1)');
title('Transposition-invariant interval recurrence profile');
```

**Python:**
```python
import numpy as np
melody_p = mpt.convert_pitch(np.array([60, 62, 63, 65, 69, 72, 74, 75]), 'midi', 'cents')
melody_t = np.arange(8, dtype=float)

query_p = mpt.convert_pitch(np.array([67, 69, 70]), 'midi', 'cents')
query_t = np.arange(3, dtype=float)

mel_pd, _ = mpt.difference_events([melody_p[None, :], melody_t[None, :]],
                                     None, None, [1, 0], [0.0, 0.0])
qry_pd, _ = mpt.difference_events([query_p[None, :],  query_t[None, :]],
                                     None, None, [1, 0], [0.0, 0.0])

melody = mpt.build_exp_tens(mel_pd, None,
                             [50., 0.3], [1, 1], None,
                             [False, False], [False, False], [0., 0.])
query  = mpt.build_exp_tens(qry_pd, None,
                             [50., 0.3], [1, 1], None,
                             [False, False], [False, False], [0., 0.])

t_offsets = np.arange(-1.5, 6.51, 0.25)
offsets = np.vstack([np.zeros_like(t_offsets), t_offsets])
spec = {'size': [np.inf, 2.0], 'mix': [0.0, 0.0]}

S = mpt.windowed_similarity(query, melody, spec, offsets, verbose=False)
```

The query's two differenced intervals have centroid time 1.5 (differencing drops the leading event, and the remaining events sit at t = 1 and t = 2), so offset 0 places the window on absolute time 1.5 and the sweep is indexed by displacement from that reference. The profile peaks sharply at offsets $\delta = 0$ and $\delta = 5$, corresponding to absolute times 1.5 and 6.5 — the centroids of the two $(+2, +1)$ occurrences in the melody — with identical heights, reflecting that the two occurrences are identical in interval content. Between the peaks, the profile is depressed where the melody's local interval content diverges from the query (most strongly around $\delta \approx 3$ — absolute time 4.5 — where the intervals $(+4, +3)$ span the leap from F to A to C), and modestly elevated where local intervals are similar in magnitude but in a different order (around $\delta \approx 1.25$ — absolute time 2.75 — where the melody has $(+100, +200, +400)$ vs the query's $(+200, +100)$, giving partial alignment of the $+200$ value).

Downstream aggregations on the returned profile are one-line computations:

```matlab
% Best match offset and strength
[sim, idx] = max(S);
best_offset = t_offsets(idx);

% Recency-weighted typicality over the profile
w_prof = seqWeights([], 'exponentialFromEnd', 'N', numel(S), 'decayRate', 0.5);
typicality = (w_prof(:).' * S(:)) / sum(w_prof);
```

Window width is a modelling choice: narrower windows are more selective (picking out single motif occurrences), wider windows smooth over multiple adjacent events and give a coarser view. The `mix` shape parameter controls the softness of the window edge (`mix = 0` is a pure Gaussian, `mix = 1` is a rectangular cutoff); intermediate values interpolate between the two extremes.

### 7.6 Virtual pitch analysis

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

### 7.7 Working with audio files

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

Demo scripts are included in each language, with user-adjustable parameters at the top. A good place to start is `demo_overview` (MATLAB) or `demo_overview.py` (Python), which exercises every major function family in a single script and follows the same section order as this guide.

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
| `demo_helixBlend` | Pitch-class and register routed simultaneously through periodic pitch-class and linear register groups; sweeping the register σ produces a continuum between pitch-class and pitch-height similarity | — |
| `demo_maetWindowing` | Time-windowed similarity sweep against a query motif: motif recurrence localisation in a melody | — |
| `demo_windowingReference` | Reference-point option for `windowedCosSim` / `windowed_similarity`: harmonic-baseline, slot-count, slot-value, and slot-weight sweeps with the harmonic-calibrated fixed-reference strategy | — |
| `demo_bindEvents` | Sliding-window binding into n-attribute super-events: n-tuple entropy via the `differenceEvents` $\to$ `bindEvents` pipeline (matches `nTupleEntropy` at $\sigma \to 0$ and extends to non-zero σ), cosine similarity between scales' n-tuple distributions, and binding raw pitch values to obtain melodic n-grams in absolute register | — |

### Python demos

Python equivalents of all demos are in `python/demos/`. They follow the same structure and produce the same results; the plotting demos require `matplotlib` (`pip install matplotlib`) and the audio demo requires `soundfile` (`pip install soundfile`).

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
| `demo_helix_blend.py` | `demo_helixBlend` | Helix blend: pitch-class and register continuum |
| `demo_maet_windowing.py` | `demo_maetWindowing` | Time-windowed motif-recurrence similarity sweep |
| `demo_windowing_reference.py` | `demo_windowingReference` | Reference-point option for windowed similarity |
| `demo_bindEvents.py` | `demo_bindEvents` | Sliding-window event binding for n-tuple analyses |

---

## 9. Known simplifications and future directions

### Flat-metric approximation

The expectation tensor framework uses a locally flat (Euclidean) metric: the Gaussian kernel is defined in terms of Euclidean distances in pitch (or time) space. In the periodic case, the domain is topologically circular (differences are wrapped modulo the period), but the metric within each period is still Euclidean — there is no curvature. For pitch-class sets with period 1200 (one octave in cents), this is well justified because the cents scale has uniform spacing in log-frequency, which closely approximates equal perceptual spacing over the range where most musical pitch perception occurs.

However, if one were to use a psychoacoustic pitch scale with non-constant spacing (e.g., mel, ERB-rate, or Bark — all available via `convertPitch`), the Euclidean metric would introduce a systematic approximation: the effective smoothing would vary across the frequency range. The correct treatment would involve a Riemannian metric that accounts for the non-constant Jacobian of the pitch-scale mapping. In one dimension, this can be handled exactly by converting to the psychoacoustic scale before calling the toolbox (the Gaussian then has the correct width at every point). In higher dimensions (r ≥ 2), the full Riemannian treatment would require architectural changes. This is documented as a potential future direction but is not implemented in v2.0.0.

### Grid discretization for entropy

The differential entropy of a Gaussian mixture density has no known closed-form analytical solution, because the logarithm of a sum of Gaussians does not simplify. This is why `entropyExpTens` and `spectralEntropy` discretize the continuous density onto a finite grid and compute Shannon entropy of the resulting probability mass function — unlike the cosine similarity, which *can* be computed analytically. The accuracy of this discretization depends on the ratio of σ to the grid spacing. Users can verify accuracy by comparing results at different resolutions (via `'nPointsPerDim'` or `'resolution'`). When the normalized entropy option is used (the default), the result is independent of the arbitrary grid resolution to the extent that the grid is fine enough to capture the density's shape.

### Salience tensors

The expectation tensor framework as implemented returns the *expected density* of r-tuples at each point in the query space — a quantity whose scale depends on the number of elements, their weights, and $\sigma$. Many applications would benefit from a complementary **salience** reading of the same tensor: a transformation $S(\mathbf{x}) = 1 - \exp(-\mathrm{ET}(\mathbf{x})/\eta)$ that maps the density to a bounded $[0, 1)$ salience, with an interpretation as the probability that at least one $r$-tuple contributes at position $\mathbf{x}$ under an inhomogeneous Poisson point process model. This framing generalises naturally to conditional forms (spectral masking, lateral inhibition; cf. Bulger, Milne & Dean, 2022) and can be used as a reading mode on any existing expectation tensor. A detailed specification has been drafted (`salience_specification.md`), and the feature is planned for a subsequent release.

---

## 10. References

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
