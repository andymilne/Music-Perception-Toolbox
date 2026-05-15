# Music Perception Toolbox

An open-source package — available in **MATLAB** and **Python** — for computing perceptually and cognitively motivated measures of pitch similarity, consonance, and scale and rhythmic structure. It accepts inputs from symbolic pitch data or from spectral peaks extracted from audio recordings.

The toolbox implements several original theoretical frameworks grounded in probability theory, Riemannian geometry, and the discrete Fourier transform. Its measures have been validated as predictors of tonal fit, consonance, affect, and rhythmic complexity across diverse empirical studies, including experiments with microtonal and non-Western tuning systems. See the [User Guide](USER_GUIDE.md) for a full description of the theoretical foundations, function reference, and worked examples.

## What's in the toolbox

**Similarity and complexity via expectation tensors.** A unified framework — applicable to pitch, time, and other musical attributes individually, or to several together (pitch with time, with register, with timbre, with metrical position, with voice, ...) — for quantifying the similarity of any two weighted multisets of events, and the complexity of a single multiset, under configurable assumptions about perceptual equivalence, uncertainty, and structural order. With spectral enrichment, this yields spectral pitch class similarity (SPCS), a powerful predictor of perceived tonal fit. Sliding-window cross-correlation along any chosen attribute (typically time) supports motif and pattern matching in a context.

**Consonance and harmonicity.** Spectral entropy, template harmonicity, tensor harmonicity, sensory roughness, and virtual pitch analysis — complementary measures capturing different aspects of consonance. The toolbox models roughness and harmonicity; it does not model familiarity, which requires corpus-based or learning-based approaches.

**Scale and rhythm structure.** Fourier-based balance and evenness; coherence, sameness, n-tuple entropy, and event binding for n-gram analyses; edge detection, projected centroid, mean offset, circular autocorrelation phase matrices, and Markov prediction — applicable to both scales and rhythms. The deterministic measures additionally accept an optional Gaussian uncertainty parameter $\sigma$ that softens the underlying computation against perceptual jitter on each event, returning the *expected* feature value under that jitter (closed-form analytical for the linear-in-$F(0)$ measures, Monte Carlo otherwise).

**Utility.** Pitch scale conversion between seven scales (Hz, MIDI, cents, mel, Bark, ERB-rate, Greenwood) and spectral peak extraction from audio files.

## What's new in v2.2

A **performance release**. The headline gains come from two strands:

- **Mathematical: the Möbius method.** A new analytical decomposition for the three core quantities — inner product, point evaluation, and total mass — that complements the existing Bulger's method for the inner product. Bulger's method (v1, exposed as `method='bulger'`) is inner-product-only, has essentially no per-call overhead, and tends to win at small tensor order $r$ and small slot count $K$. The Möbius method (new in v2.2, exposed as `method='mobius'`) applies uniformly to all three quantities and tends to win at large $r$ or large $K$, where the slot-tuple loop in Bulger's method blows up. A per-call cost-model dispatcher picks the cheaper of the two for each call. The inclusion-exclusion technique at the core of the Möbius method was first published in Milne et al. 2011 and has been used in MPT since v1 for numerical computation of discrete tensors; v2.2 applies it as a closed-form analytical decomposition, combined with orbit collapse for the inner product.

- **Computational: dispatching, kernel precision, kernel truncation, and batching refinements.** Two new kernel-evaluation controls — `truncation_sigmas` (skip Gaussian contributions beyond $k$ standard deviations from a centre) and `kernel_precision` (`'single'` instead of `'double'` for ~2× speedup at ~7 significant figures of precision) — affect the centres-path consumers (eval, cos-sim, entropy) at user request. Several batching refinements replace v2.1 sequential per-item loops with grouped contractions: K-grouped direct enumeration for unsafe-event pairs in the MA inner product, a dedup-and-batch rewrite of `tensorHarmonicity`'s batched mode, and vectorised u-grid handling in the Möbius relative-mode evaluator.

A new **toolbox defaults API** (`mptDefaults` in MATLAB, `mpt.set_default` / `mpt.get_defaults` / `mpt.reset_defaults` / `mpt.show_defaults` in Python) lets the kernel-evaluation controls and other toolbox-wide settings be inspected, set, and reset per call or globally. Calling `mptDefaults` with no arguments (MATLAB) or `mpt.show_defaults()` (Python) prints the current values with brief descriptions of each setting. See the [User Guide](USER_GUIDE.md#toolbox-defaults-api-mptdefaults--mptset_default) for full details.

Other user-facing additions: closed-form **Rényi-2 differential entropy** (`method='renyi2'` on `entropyExpTens`; Shannon differential entropy continues to be evaluated on a numerical grid, since it has no closed form in any version); a **ragged-K hybrid** that lets the multi-attribute Möbius inner-product path handle variable-cardinality inputs natively; **`bindEvents`** now accepts $K_{a,n} > 1$ input, unblocking polyphonic-binding analyses with multi-slot attributes at the source.

The release is additive: existing v2.1 calling conventions are preserved at the floating-point level for the default routing in standard regimes. See [CHANGELOG.md](CHANGELOG.md) for the full list of changes and [MIGRATION.md](MIGRATION.md#v21--v22) for the v2.1 → v2.2 migration notes.

## What's new in v2.1

- **Multi-attribute expectation tensors (MAET).** `buildExpTens`, `evalExpTens`, `cosSimExpTens`, and `entropyExpTens` now accept a multi-attribute density specification alongside the v2.0 single-attribute form. Each attribute carries its own tuple order $r_a$ and per-event slot count $K_a$; groups share $\sigma$, periodicity, and relativity. The cosine-similarity inner product factorises analytically across attribute groups in the same way as the single-attribute case.
- **Cross-event preprocessing.** `differenceEvents` produces inter-event differences (e.g., interval content, IOIs, higher-order differences) and `bindEvents` gathers $n$ consecutive events into super-events with separate per-lag attributes. Composes with the MAET pipeline; recovers `nTupleEntropy` of Milne & Dean (2016) at $\sigma \to 0$ as a special case.
- **Post-tensor windowing.** `windowTensor` wraps a MAET with a per-group window specification; `windowedSimilarity` sweeps the window across a context and returns a windowed-similarity profile against a query, with a closed-form analytical inner product.
- **Unified dispatch and consumer-level batching.** `evalExpTens`, `cosSimExpTens`, and `entropyExpTens` accept a single density, a list of densities, or raw arrays (1-D for a single multiset, 2-D for a batch). `tensorHarmonicity`, `templateHarmonicity`, `virtualPitches`, and `spectralEntropy` accept a 2-D pitch matrix for batched evaluation, returning per-row results. The DFT-equivariant family (`dftCircular`, `meanOffset`, `edges`, `projCentroid`, `circApm`), the structural family (`coherence`, `sameness`, `nTupleEntropy`), and the Monte Carlo family (`balanceCircular`, `evennessCircular`) likewise gain batched-input dispatch. Every batched path uses canonical-form deduplication to collapse symmetry-equivalent rows onto a single cached computation. The standalone `cos_sim_exp_tens_raw`, `eval_exp_tens_raw`, `batch_cos_sim_exp_tens` (Python) and `batchCosSimExpTens` (MATLAB) are now deprecated shims forwarding to the unified entry points.
- **Soft (`sigma > 0`) structural measures.** `sameness` and `coherence` now accept an optional `sigma` argument that softens the discrete equality / ordering tests against Gaussian positional uncertainty; `nTupleEntropy`'s existing `sigma` argument gains a new `sigmaSpace` flag (shared across all three functions) controlling whether `sigma` describes positional uncertainty on each event (the new default) or independent per-interval uncertainty.
- **Argand-DFT Monte Carlo.** New `dftCircularSimulate` estimates the distribution of $|F(k)|$ under positional jitter; `balanceCircular`, `evennessCircular`, and `projCentroid` accept an optional `sigma` argument (Monte Carlo for the first two, closed-form analytical for the third).
- **Sequential-analysis utilities.** `continuity` summarises the recent direction trend leading up to a query; `seqWeights` constructs position-weight vectors with named time-based decay profiles.
- **Categorical-encoding utility.** `simplexVertices` returns equidistant vertex coordinates for simplex-coded categorical attributes (voice identity, instrument, etc.) suitable for MAET inputs.

The release is additive: existing v2.0 single-attribute calling conventions are preserved unchanged. One small documented numerical change applies to `nTupleEntropy` at `sigma > 0`; see [MIGRATION.md](MIGRATION.md#v20--v21).

For the full list of changes, see [CHANGELOG.md](CHANGELOG.md).

## What's new in v2

This was a major rewrite. Key changes:

- **Python implementation** — a functionally identical Python package (`mpt`) using snake_case naming. See the [User Guide](USER_GUIDE.md#4-api-conventions-matlab-vs-python) for the full name mapping.
- Analytical methods have replaced the previous numerical approximations wherever feasible. In v1, analytical computation was available only for the cosine similarity inner product (`cosSimExpTens`); in v2, individual tensor construction and evaluation (`buildExpTens` / `build_exp_tens` and `evalExpTens` / `eval_exp_tens`) are also analytical, eliminating grid discretization.
- The `cosSimExpTens` computation itself has been substantially optimized — the original double loop over r-ad combinations has been replaced by fully vectorized operations over pre-calculated r-ads.
- Precomputed density objects (`buildExpTens` / `build_exp_tens`) eliminate redundant computation across repeated comparisons.
- Spectral enrichment (`addSpectra` / `add_spectra`) expanded from one mode to five: harmonic, stretched, frequency-linear, stiff-string, and custom.
- All functions now accept event positions and weights directly (v1's indicator-vector inputs are no longer required).
- No external dependencies (v1 required the [Sparse Array Toolbox](https://github.com/andymilne/Sparse-Array-Toolbox)).
- **Comprehensive documentation** — every function includes a full help text with usage examples. A [User Guide](USER_GUIDE.md) covers the conceptual foundations, a complete function reference for both languages, worked examples, and nine demo scripts covering all major use cases. [MIGRATION.md](MIGRATION.md) maps every v1 function to its v2 equivalent.

The original `cosSimExpTens` calling convention is fully backward compatible.

**v1 users:** see [MIGRATION.md](MIGRATION.md#v1--v2) for a complete function mapping. The original toolbox is permanently available as the [v1.0.0 release](https://github.com/andymilne/Music-Perception-Toolbox/releases/tag/v1.0.0).

For a full list of changes, see [CHANGELOG.md](CHANGELOG.md).

## Installation

### MATLAB

1. Download or clone this repository.
2. Add the MATLAB folder to the MATLAB path:
   ```matlab
   addpath('/path/to/Music-Perception-Toolbox/matlab');
   ```

Requires MATLAB R2019b or later. No external dependencies.

### Python

1. Download or clone this repository.
2. Install from the local `python/` directory:
   ```bash
   pip install ./python
   ```

For audio file support (spectral peak extraction via `audio_peaks`):

```bash
pip install ./python[audio]
```

Requires Python 3.10+. Dependencies (NumPy, SciPy) are installed automatically.

## Quick example

The two implementations are functionally identical. The main differences are naming convention (camelCase → snake_case), `[]` → `None` for default weights, and cell arrays → lists for spectrum arguments. See the [User Guide](USER_GUIDE.md#api-conventions-matlab-vs-python) for a complete mapping.

### MATLAB

```matlab
% Define two chords (in cents)
major = [0, 400, 700];
minor = [0, 300, 700];

% Add harmonic spectra (12 partials, 1/n rolloff)
[maj_p, maj_w] = addSpectra(major, [], 'harmonic', 12, 'powerlaw', 1);
[min_p, min_w] = addSpectra(minor, [], 'harmonic', 12, 'powerlaw', 1);

% Compute spectral pitch class similarity
s = cosSimExpTens(maj_p, maj_w, min_p, min_w, 10, 1, false, true, 1200);
fprintf('SPCS(major, minor) = %.3f\n', s);
```

### Python

```python
import mpt

major = [0, 400, 700]
minor = [0, 300, 700]

maj_p, maj_w = mpt.add_spectra(major, None, 'harmonic', 12, 'powerlaw', 1)
min_p, min_w = mpt.add_spectra(minor, None, 'harmonic', 12, 'powerlaw', 1)

s = mpt.cos_sim_exp_tens(maj_p, maj_w, min_p, min_w, 10, 1, False, True, 1200)
print(f'SPCS(major, minor) = {s:.3f}')
```

Demo scripts are included in `matlab/demos/` and `python/demos/`. Start with `demo_overview` for a quick tour of all function families — see the [User Guide](USER_GUIDE.md#8-demo-scripts) for full descriptions.

## Repository structure

```
Music-Perception-Toolbox/
├── README.md, LICENSE, CITATION.cff
├── USER_GUIDE.md, CHANGELOG.md, MIGRATION.md
├── matlab/
│   ├── *.m                  (core toolbox functions)
│   ├── tests/               (test suite)
│   ├── demos/               (demo scripts)
│   └── audio/               (example audio files)
└── python/
    ├── mpt/                 (Python package)
    ├── tests/               (test suite)
    ├── demos/               (demo scripts)
    ├── audio/               (example audio files)
    └── pyproject.toml
```

## Documentation

- **[User Guide](USER_GUIDE.md)** — Conceptual overview, function reference (both languages), worked examples, and demo descriptions.
- **[CHANGELOG](CHANGELOG.md)** — Full list of changes from v1 to v2.
- **[MIGRATION](MIGRATION.md)** — Function-by-function mapping from v1 to v2 (MATLAB only) and a v2.0 → v2.1 migration covering the soft-sigma structural measures, DFT Monte Carlo additions, unified-dispatch entry points, and consumer-level batching.

## Citation

If you use this toolbox in published work, please cite:

> Milne, A. J., Sethares, W. A., Laney, R., & Sharp, D. B. (2011). Modelling the similarity of pitch collections with expectation tensors. *Journal of Mathematics and Music*, 5(1), 1–20.

and the software itself using the DOI from Zenodo (see [CITATION.cff](CITATION.cff)). GitHub will also display a "Cite this repository" button from the CITATION.cff metadata.

For functions related to balance, evenness, and rhythmic structure, additionally cite Milne, Bulger, & Herff (2017) and Milne & Herff (2020). For the rhythmic predictors, additionally cite Milne, Dean, & Bulger (2023). Full references are in the [User Guide](USER_GUIDE.md#9-references).

## Acknowledgments

This work was supported, in part, by an Australian Research Council Discovery Early Career Researcher Award (project number DE170100353) funded by the Australian Government.

The original `cosSimExpTens` algorithm and the `markovS` function were contributed by David Bulger (Department of Mathematics and Statistics, Macquarie University).

## License

MIT
