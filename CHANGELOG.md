# Changelog

All notable changes to the Music Perception Toolbox are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [2.0.0] — 2026-04-05

### Overview

Version 2.0.0 is a major rewrite of the Music Perception Toolbox. Analytical methods have replaced the previous numerical approximations wherever feasible (entropy computation, which has no known closed-form solution for Gaussian mixtures, remains discretized). The expectation tensor core has been restructured around precomputed density objects (`buildExpTens`), with fully vectorized computation and automatic memory-aware chunking. All existing functions have been given clearer names and a consistent interface, and a Python implementation has been added. The v1 dependency on the [Sparse Array Toolbox](https://github.com/andymilne/Sparse-Array-Toolbox) has been eliminated; v2 has no external dependencies.

Accessibility has been substantially improved: every function now includes a full help text with usage examples, a User Guide covers the conceptual foundations and provides a complete function reference for both languages, and nine demo scripts (in both MATLAB and Python) cover all major use cases.

The original `cosSimExpTens` function by David Bulger is preserved with full backward compatibility: the v1 nine-argument calling convention still works unchanged.

### Added

The following functions are entirely new in v2 (no v1 equivalent):

- **`buildExpTens`** — Precomputes an r-ad expectation tensor density object as a struct that can be reused across multiple calls to `evalExpTens` and `cosSimExpTens`.
- **`evalExpTens`** — Evaluates the expectation tensor density analytically at arbitrary query points with automatic memory-aware chunking and three normalisation modes. (Together, `buildExpTens` and `evalExpTens` replace `expectationTensor`, which built a discretized tensor on a grid.)
- **`batchCosSimExpTens`** — Batch cosine similarity of paired pitch multisets with automatic deduplication of repeated pairs, optional spectral enrichment, and progress reporting.
- **`estimateCompTime`** — Micro-benchmark-based computation time estimator, calibrated to the user's hardware.
- **`templateHarmonicity`** — Harmonicity via template cross-correlation (returns both hMax from Milne 2013 and hEntropy from Harrison & Pearce 2020).
- **`tensorHarmonicity`** — Harmonicity via expectation tensor lookup: evaluates the relative r-ad density of a harmonic series at the chord's intervals.
- **`virtualPitches`** — Virtual pitch (fundamental) salience profile via template cross-correlation.
- **`convertPitch`** — Converts between seven pitch/frequency scales (Hz, MIDI, cents, mel, Bark, ERB-rate, Greenwood cochlear position).
- **Python implementation** — A functionally identical Python package (`mpt`) using NumPy and SciPy. Python uses snake_case naming (e.g., `cos_sim_exp_tens`, `add_spectra`); a few names are shortened where the MATLAB suffix is redundant in a package namespace (e.g., `balanceCircular` → `balance`, `evennessCircular` → `evenness`). See the [User Guide](USER_GUIDE.md#4-api-conventions-matlab-vs-python) for the full mapping and calling convention differences.

### Added — Documentation

- **[User Guide](USER_GUIDE.md)** — Conceptual overview of the theoretical foundations, a complete function reference for both MATLAB and Python (with API convention mapping), worked examples, and demo descriptions.
- **Docstring examples** — Every function includes a full help text with runnable usage examples.
- **[MIGRATION.md](MIGRATION.md)** — Maps every v1 function to its v2 equivalent, with before/after code examples.
- **[CHANGELOG.md](CHANGELOG.md)** — Full list of changes from v1 to v2.
- **Example audio files** — Seven audio files (piano, violin, oboe, chords, and a music sample) in the `audio/` folder for use with `audioPeaks` and the demo scripts.

### Added — Demo scripts

Nine demo scripts are included in both MATLAB (`matlab/demos/`) and Python (`python/demos/`). A good starting point is `demo_overview` / `demo_overview.py`, which exercises every major function family.

- **`demo_overview`** — Quick tour of all function families: pitch conversion, spectral enrichment, SPCS, harmonicity, roughness, balance, evenness, coherence, sameness, entropy, mean offset, edges, and Markov.
- **`demo_audioAnalysis`** — Two-pass peak extraction (unsmoothed then smoothed) from audio files, with spectral similarity, harmonicity, roughness, and virtual pitch analysis.
- **`demo_batchProcessing`** — Batch feature computation with deduplication: paired SPCS via `batchCosSimExpTens`, and single-set measures (spectral entropy, harmonicity, roughness) via the unique/map pattern — a template for experimental data processing.
- **`demo_edoApprox`** — SPCS of n-EDOs against a JI reference chord (cf. Milne et al. 2011, Fig. 4).
- **`demo_expTensorPlots`** — Interactive visualisation of expectation tensors in 1–4 dimensions, with power sliders and projection toggles.
- **`demo_genChainSpcs`** — SPCS of generator-chain tunings (cf. Milne et al. 2011, Figs. 5–7).
- **`demo_triadConsonance`** — Five consonance measures plotted over a grid of triad intervals.
- **`demo_triadSpcsGrid`** — SPCS heatmap of 12-EDO triads with a fifth (cf. Milne et al. 2011, Fig. 3).
- **`demo_virtualPitches`** — Virtual pitch salience profiles for example chords.

### Rewritten and renamed

The following v1 functions have been rewritten with clearer names, a consistent `(p, w, period)` interface, and in many cases expanded functionality. All v1 functions previously requiring indicator vectors now accept event positions and weights directly. See [MIGRATION.md](MIGRATION.md) for before/after code examples.

| v1 name | v2 name | What changed |
|:---|:---|:---|
| `spectralize` | `addSpectra` | Expanded from one spectral mode (harmonic) to five (harmonic, stretched, freqlinear, stiff, custom) |
| `expectationTensor` | `buildExpTens` + `evalExpTens` | Split into precomputation and evaluation; evaluates analytically at arbitrary query points rather than on a discrete grid |
| `expTensorEntropy` / `rAdEntropy` | `entropyExpTens` | Supports precomputed structs and spectral enrichment |
| `fSetRoughness` | `roughness` | Added configurable p-norm and averaging options |
| `pSetSpectralEntropy` | `spectralEntropy` | Restructured to use the expectation tensor framework |
| `bal` | `balanceCircular` | Descriptive name |
| `eve` | `evennessCircular` | Descriptive name |
| `dft2sss` / `pitch2Argand` | `dftCircular` | Merged; computed directly from point positions |
| `modeHeight` | `meanOffset` | Descriptive name; now accepts query points |
| `projCent` | `projCentroid` | Descriptive name; now also returns centroid magnitude and phase |
| `stepEntropy` | `nTupleEntropy` | Generalized from 1-tuples to n-tuples; added optional Gaussian smoothing; removed `histcn` dependency |
| `coherence` | `coherence` | Same name; added `'strict'` option |
| `sameness` | `sameness` | Same name |
| `circApm` | `circApm` | Same name; now takes event positions (not indicator vector); added `'decay'` option |
| `edges` | `edges` | Same name; now takes event positions; added `'kappa'` option and continuous query points |
| `markovS` | `markovS` | Same name; now takes event positions |
| `peakPicker` | `audioPeaks` | Substantially expanded into a full audio analysis function with Gaussian smoothing in log-frequency space |

### Changed

- **`cosSimExpTens`** — Now additionally accepts precomputed density structs from `buildExpTens` (preferred for repeated comparisons). The original nine-argument calling convention is fully backward compatible. David Bulger's analytical algorithm — which inspired much of the v2 architecture — has been further optimized at the implementation level: the double loop over all pairs of r-ad index sets has been replaced by fully vectorized array operations (3D implicit expansion and matrix multiplication), the explicit Riemannian metric matrix has been eliminated in favour of direct scalar arithmetic, and automatic memory-aware chunking has been added for large problems. Tuple enumeration and weight computation are factored into the precomputed density struct returned by `buildExpTens`, so these are computed once and reused across the three inner product calls (⟨x,x⟩, ⟨y,y⟩, ⟨x,y⟩) and across multiple comparisons. Added `'verbose'` flag for suppressing console output.

### Breaking changes

- Function names, signatures, and argument ordering have changed throughout. Code that called only `cosSimExpTens` with its original nine-argument signature will continue to work without modification. Code that relied on other v1 functions will need to be updated — see [MIGRATION.md](MIGRATION.md) for a complete mapping.
- The v1 dependency on the [Sparse Array Toolbox](https://github.com/andymilne/Sparse-Array-Toolbox) has been eliminated. v2 has no external dependencies.

### Removed

- `cosSim` / `expTensorSim` → absorbed into `cosSimExpTens` (which computes the similarity analytically)
- `gaussianKernel`, `histEntropy`, `tensorSum`, `circConv` → absorbed as internal computations
- `pitch2Ind` / `ind2Pitch` → no longer needed (v2 evaluates densities at continuous query points)
- `contextProbeSpecSim` → superseded by the combination of `addSpectra` + `cosSimExpTens`
- `noiseSignal` → noise-floor estimation incorporated into `audioPeaks` via `'noiseFactor'`
- `nonLinDps`, `pDist` → not carried forward

---

## [1.0.0] — 2026-04-03 (retrospective tag)

The original Music Perception Toolbox as used in published work from 2011–2025. This version is permanently available as the `v1.0.0` tag and GitHub Release. It requires the [Sparse Array Toolbox](https://github.com/andymilne/Sparse-Array-Toolbox).

### Expectation tensor core

- `cosSimExpTens` — Cosine similarity of two r-ad expectation tensors. Originally by David Bulger (Macquarie University).
- `expectationTensor` — Discretized expectation tensor on a grid.
- `cosSim` / `expTensorSim` — Cosine similarity of precomputed tensors.
- `spectralize` — Harmonic spectral enrichment.

### Consonance and harmonicity

- `fSetRoughness` — Sensory roughness.
- `pSetSpectralEntropy` — Spectral entropy.
- `contextProbeSpecSim` — Context–probe spectral similarity.

### Balance and evenness (Fourier-based measures)

- `bal` — Balance.
- `eve` — Evenness.
- `dft2sss` / `pitch2Argand` — DFT of pitch-class sets.

### Scale and rhythm structure

- `coherence` — Coherence quotient.
- `sameness` — Sameness quotient.
- `stepEntropy` — Step-size entropy.
- `expTensorEntropy` / `rAdEntropy` — Expectation tensor entropy.
- `circApm` — Circular autocorrelation phase matrix.
- `edges` — Circular edge detection.
- `projCent` — Projected centroid.
- `modeHeight` — Mode height / mean offset.
- `markovS` — Optimal S-step Markov predictor. Originally by David Bulger.

### Other

- `gaussianKernel`, `histEntropy`, `tensorSum`, `circConv`, `pitch2Ind`, `ind2Pitch`, `peakPicker`, `noiseSignal`, `nonLinDps`, `pDist` — Helpers and utilities.
