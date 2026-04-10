# Music Perception Toolbox

An open-source package — available in **MATLAB** and **Python** — for computing perceptually and cognitively motivated measures of pitch similarity, consonance, and scale and rhythmic structure. It accepts inputs from symbolic pitch data or from spectral peaks extracted from audio recordings.

The toolbox implements several original theoretical frameworks grounded in probability theory, Riemannian geometry, and the discrete Fourier transform. Its measures have been validated as predictors of tonal fit, consonance, affect, and rhythmic complexity across diverse empirical studies, including experiments with microtonal and non-Western tuning systems. See the [User Guide](USER_GUIDE.md) for a full description of the theoretical foundations, function reference, and worked examples.

## What's in the toolbox

**Similarity and complexity via expectation tensors.** A unified framework — applicable to both pitch and rhythm — for quantifying the similarity of any two weighted multisets of pitches or time points, and the complexity of a single multiset, under configurable assumptions about perceptual equivalence, uncertainty, and structural order. With spectral enrichment, this yields spectral pitch class similarity (SPCS), a powerful predictor of perceived tonal fit.

**Consonance and harmonicity.** Spectral entropy, template harmonicity, tensor harmonicity, sensory roughness, and virtual pitch analysis — complementary measures capturing different aspects of consonance. The toolbox models roughness and harmonicity; it does not model familiarity, which requires corpus-based or learning-based approaches.

**Scale and rhythm structure.** Fourier-based balance and evenness; coherence, sameness, and n-tuple entropy; edge detection, projected centroid, mean offset, circular autocorrelation phase matrices, and Markov prediction — applicable to both scales and rhythms.

**Utility.** Pitch scale conversion between seven scales (Hz, MIDI, cents, mel, Bark, ERB-rate, Greenwood) and spectral peak extraction from audio files.

## What's new in v2

This is a major rewrite. Key changes:

- **Python implementation** — a functionally identical Python package (`mpt`) using snake_case naming. See the [User Guide](USER_GUIDE.md#4-api-conventions-matlab-vs-python) for the full name mapping.
- Analytical methods have replaced the previous numerical approximations wherever feasible. In v1, analytical computation was available only for the cosine similarity inner product (`cosSimExpTens`); in v2, individual tensor construction and evaluation (`buildExpTens` / `build_exp_tens` and `evalExpTens` / `eval_exp_tens`) are also analytical, eliminating grid discretization.
- The `cosSimExpTens` computation itself has been substantially optimized — the original double loop over r-ad combinations has been replaced by fully vectorized operations over pre-calculated r-ads.
- Precomputed density objects (`buildExpTens` / `build_exp_tens`) eliminate redundant computation across repeated comparisons.
- Spectral enrichment (`addSpectra` / `add_spectra`) expanded from one mode to five: harmonic, stretched, frequency-linear, stiff-string, and custom.
- All functions now accept event positions and weights directly (v1's indicator-vector inputs are no longer required).
- No external dependencies (v1 required the [Sparse Array Toolbox](https://github.com/andymilne/Sparse-Array-Toolbox)).
- **Comprehensive documentation** — every function includes a full help text with usage examples. A [User Guide](USER_GUIDE.md) covers the conceptual foundations, a complete function reference for both languages, worked examples, and nine demo scripts covering all major use cases. [MIGRATION.md](MIGRATION.md) maps every v1 function to its v2 equivalent.

The original `cosSimExpTens` calling convention is fully backward compatible.

**v1 users:** see [MIGRATION.md](MIGRATION.md) for a complete function mapping. The original toolbox is permanently available as the [v1.0.0 release](https://github.com/andymilne/Music-Perception-Toolbox/releases/tag/v1.0.0).

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

s = mpt.cos_sim_exp_tens_raw(maj_p, maj_w, min_p, min_w, 10, 1, False, True, 1200)
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
- **[MIGRATION](MIGRATION.md)** — Function-by-function mapping from v1 to v2 (MATLAB only).

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
