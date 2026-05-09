# Migration Guide

This guide documents migration paths between major versions of the Music Perception Toolbox.

- [v2.0 → v2.1](#v20--v21) — soft (`sigma > 0`) structural measures, Argand-DFT Monte Carlo
- [v1 → v2](#v1--v2) — major rewrite (analytical methods, Python port, restructured core)

---

## v2.0 → v2.1

v2.1.0 is additive: existing v2.0 calling conventions are preserved unchanged at `sigma = 0` (or for functions that did not previously take `sigma`, with the unchanged signature). One numerical change exists at `sigma > 0` for `nTupleEntropy`. No code changes are required for v2.0 callers who do not touch the new features; users who do call `nTupleEntropy` with `sigma > 0` should read the *Numerical change in nTupleEntropy* note below.

### Numerical change in `nTupleEntropy` at `sigma > 0`

In v2.0, `nTupleEntropy(p, period, n, 'sigma', s)` (MATLAB) and `mpt.n_tuple_entropy(p, period, n, sigma=s)` (Python) treated `s` as independent uncertainty on each derived step size — interval-space semantics in v2.1.0 terminology. v2.1.0 introduces a `sigmaSpace` flag with default `'position'`: `s` is now treated as positional uncertainty on each event, and derived steps inherit a per-slot variance of $2s^2$ via the marginal-matched approximation.

At `n = 1`, the two semantics are exactly related: `sigmaSpace = 'position'` with $\sigma$ produces identical entropy to `sigmaSpace = 'interval'` with $\sigma\sqrt{2}$. To recover v2.0.0 numerical results, pass `sigmaSpace = 'interval'`:

```matlab
% v2.0 behaviour (interval-space sigma)
H = nTupleEntropy(p, period, n, 'sigma', s);

% v2.1: same numerical result via explicit flag
H = nTupleEntropy(p, period, n, 'sigma', s, 'sigmaSpace', 'interval');

% v2.1 default (position-space sigma) — different numerical result at sigma > 0
H = nTupleEntropy(p, period, n, 'sigma', s);
```

```python
# v2.0 behaviour (interval-space sigma)
H, _ = mpt.n_tuple_entropy(p, period, n, sigma=s)

# v2.1: same numerical result via explicit flag
H, _ = mpt.n_tuple_entropy(p, period, n, sigma=s, sigma_space='interval')

# v2.1 default (position-space sigma) — different numerical result at sigma > 0
H, _ = mpt.n_tuple_entropy(p, period, n, sigma=s)
```

The new default reflects the toolbox-wide convention that `sigma` describes uncertainty on the input quantity, which for `nTupleEntropy` is positions. The two semantics coincide at `sigma = 0`, so calls without an explicit `sigma` argument are unaffected.

At `n \ge 2` with `sigmaSpace = 'position'`, a one-time warning fires noting that the current implementation uses the marginal-matched approximation (slots independent at $\sigma_{\text{eff}} = \sigma\sqrt{2}$) and that full position-aware $n \ge 2$ support is planned for a future release. Suppress via the standard MATLAB / Python warning-filter mechanisms (`warning('off', 'nTupleEntropy:positionApprox')`; `warnings.filterwarnings('ignore', message='.*marginal-matched.*')`).

### `sameness` and `coherence` gain optional `sigma`

Both functions now accept an optional `sigma` argument and a `sigmaSpace` name-value flag. At `sigma = 0` (default), the v2.0 hard counts are recovered byte-for-byte; existing call sites are unaffected.

```matlab
% v2.0 — still works in v2.1
[sq, nDiff] = sameness(p, period);
[c, nc] = coherence(p, period);

% v2.1 — soft sigma version
[sq, nDiff] = sameness(p, period, sigma);
[c, nc] = coherence(p, period, sigma);

% v2.1 — interval-space sigma (different per-pair variance)
[sq, nDiff] = sameness(p, period, sigma, 'sigmaSpace', 'interval');
```

Float positions and float `period` are accepted when `sigma > 0`. The integer requirement (and rejection of non-integer input) applies only at `sigma = 0`.

### `balanceCircular`, `evennessCircular` gain optional `sigma`

Both functions now accept an optional `sigma` argument that triggers Monte Carlo estimation under positional jitter via the new `dftCircularSimulate`. At `sigma = 0` the v2.0 deterministic value is recovered exactly.

```matlab
% v2.0 — still works in v2.1
b = balanceCircular(p, w, period);
e = evennessCircular(p, period);

% v2.1 — expected balance / evenness under sigma jitter
b = balanceCircular(p, w, period, sigma);
e = evennessCircular(p, period, sigma);

% v2.1 — also request standard deviation (MATLAB nargout idiom)
[b, bStd] = balanceCircular(p, w, period, sigma);
[e, eStd] = evennessCircular(p, period, sigma);

% Optional name-value: nDraws (default 10000), rngSeed
[b, bStd] = balanceCircular(p, w, period, sigma, 'nDraws', 50000, 'rngSeed', 42);
```

In Python, the SD is requested via an explicit `return_std=True` flag (Python lacks `nargout`):

```python
# v2.0 — still works in v2.1
b = mpt.balance(p, None, period)
e = mpt.evenness(p, period)

# v2.1 — scalar mean (backward-compatible signature)
b = mpt.balance(p, None, period, sigma=s)
e = mpt.evenness(p, period, sigma=s)

# v2.1 — opt in to (mean, std) tuple
b, b_std = mpt.balance(p, None, period, sigma=s, return_std=True)
e, e_std = mpt.evenness(p, period, sigma=s, return_std=True)
```

### `projCentroid` gains optional `sigma` (analytical, no Monte Carlo)

Because $y(x)$ is linear in $F(0)$ and $F(0)$ is permutation-invariant under positional jitter, the mean projection has a clean closed form: $E[y(x)] = \alpha_1 \cdot y_{\text{deterministic}}(x)$ where $\alpha_1 = \exp(-2\pi^2 \sigma^2 / P^2)$ and $P$ is the period. No Monte Carlo is involved.

```matlab
% v2.0 — still works in v2.1
[y, centMag, centPhase] = projCentroid(p, w, period, x);

% v2.1 — expected projection under sigma jitter (analytical)
[y, centMag, centPhase] = projCentroid(p, w, period, x, sigma);
```

`centMag` returns $\alpha_1 \cdot |F(0)| = |E[\widetilde{F}(0)]|$, the magnitude of the *complex mean centroid* — consistent with the projection. The distinct scalar $E[|\widetilde{F}(0)|]$ — the *mean centroid magnitude under jitter*, picking up positive Rayleigh-style bias when the perturbation cloud straddles the origin — is what `balanceCircular(p, w, period, sigma)` returns (read as `1 - b`). The two answer different balance-related questions; see User Guide §3.5 "Two scalars, two balance-related questions" for the operational distinction. Notation: $\widetilde{F}(0)$ is the random variable $F(0)$ becomes when each $p_k$ is replaced by $\widetilde{p}_k = (p_k + \eta_k) \bmod P$ with $\eta_k \sim \mathcal{N}(0, \sigma^2)$.

`centPhase` is preserved in expectation (the argument of $E[\widetilde{F}(0)]$ equals the argument of $F(0)$).

### Summary of breaking changes

- `nTupleEntropy` at `sigma > 0`: default semantics changed from interval-space to position-space. Pass `sigmaSpace = 'interval'` for v2.0 numerical equivalence.

All other changes are additive: new optional arguments, new return-value options behind opt-in flags, no change to default behaviour at the v2.0 calling conventions.

---

## v1 → v2

This guide maps every v1 function to its v2 equivalent. If you used only `cosSimExpTens` with its original nine-argument signature, your code will work without modification — the old calling convention is fully preserved.

v2 also eliminates the v1 dependency on the [Sparse Array Toolbox](https://github.com/andymilne/Sparse-Array-Toolbox). All v2 functions are self-contained.

**Input convention change.** In v1, some functions (e.g., `circApm`, `markovS`, `edges`) required indicator vectors — a vector of length `period` with non-zero entries at event positions. In v2, all functions accept event positions `p` and weights `w` directly, giving a consistent calling convention across the entire toolbox.

---

## Complete function mapping

| v1 function | v2 equivalent | Notes |
|:---|:---|:---|
| `cosSimExpTens` | `cosSimExpTens` | Backward compatible; also accepts precomputed structs |
| `expectationTensor` | `buildExpTens` + `evalExpTens` | Split into precomputation and evaluation |
| `cosSim` | `cosSimExpTens` | Analytical computation replaces grid-based approach |
| `expTensorSim` | `cosSimExpTens` | Analytical computation replaces grid-based approach |
| `spectralize` | `addSpectra` | Expanded from one mode to five |
| `expTensorEntropy` | `entropyExpTens` | Supports precomputed structs and spectral enrichment |
| `rAdEntropy` | `entropyExpTens` | Merged with `expTensorEntropy` |
| `bal` | `balanceCircular` | Same algorithm, descriptive name |
| `eve` | `evennessCircular` | Same algorithm, descriptive name |
| `dft2sss` | `dftCircular` | Merged with `pitch2Argand` |
| `pitch2Argand` | `dftCircular` | Merged with `dft2sss` |
| `fSetRoughness` | `roughness` | Added p-norm and averaging options |
| `pSetSpectralEntropy` | `spectralEntropy` | Restructured; uses expectation tensor framework |
| `modeHeight` | `meanOffset` | Same algorithm, renamed for generality |
| `projCent` | `projCentroid` | Same algorithm, descriptive name |
| `stepEntropy` | `nTupleEntropy` | Generalized; removed `histcn` dependency |
| `coherence` | `coherence` | Same name; added `'strict'` option |
| `sameness` | `sameness` | Same name |
| `circApm` | `circApm` | Same name; now takes event positions (not indicator vector); added `'decay'` option |
| `edges` | `edges` | Same name; now takes event positions (not indicator vector); added `'kappa'` option and continuous query points |
| `markovS` | `markovS` | Same name; now takes event positions (not indicator vector) |
| `contextProbeSpecSim` | `addSpectra` + `cosSimExpTens` | Composed from general-purpose functions |
| `gaussianKernel` | *(internal)* | Absorbed into expectation tensor functions |
| `histEntropy` | *(internal)* | Absorbed into `nTupleEntropy` and `entropyExpTens` |
| `tensorSum` | *(internal)* | No longer needed |
| `circConv` | *(internal)* | No longer needed |
| `pitch2Ind` / `ind2Pitch` | *(removed)* | v2 evaluates at continuous query points |
| `peakPicker` | `audioPeaks` | Expanded into a full audio analysis function |
| `noiseSignal` | `audioPeaks` | Noise-floor estimation incorporated via `'noiseFactor'` parameter |
| `nonLinDps` | *(removed)* | Not carried forward |
| `pDist` | *(removed)* | Not carried forward |

---

## Detailed migration examples

### cosSimExpTens

**No changes required.** The original signature is still supported:

```matlab
% v1 (still works in v2)
s = cosSimExpTens(p1, w1, p2, w2, sigma, r, isRel, isPer, period);
```

The new preferred calling convention precomputes the density objects, which is faster when comparing a fixed reference against many sets:

```matlab
% v2 (preferred for repeated comparisons)
dens_ref = buildExpTens(p1, w1, sigma, r, isRel, isPer, period);
dens_cmp = buildExpTens(p2, w2, sigma, r, isRel, isPer, period);
s = cosSimExpTens(dens_ref, dens_cmp);
```

Both conventions support `'verbose', false` to suppress console output.

### expectationTensor → buildExpTens + evalExpTens

In v1, `expectationTensor` built a discretized tensor on a grid and returned a multidimensional array. In v2, this is split into two functions:

```matlab
% v1
T = expectationTensor(p, w, sigma, r, isRel, isPer, period, nPoints);
```

```matlab
% v2 equivalent
dens = buildExpTens(p, w, sigma, r, isRel, isPer, period);
x = linspace(0, period, nPoints + 1);
x = x(1:end-1);  % exclude duplicate endpoint for periodic case
vals = evalExpTens(dens, x);
```

The v2 approach has three advantages: (1) the density object is built once and reused across multiple queries; (2) query points can be arbitrary (not restricted to a uniform grid); (3) the density evaluation uses memory-aware chunking to handle large problems.

### cosSim / expTensorSim → cosSimExpTens

In v1, `cosSim` and `expTensorSim` computed cosine similarity from precomputed discretized tensors (built by `expectationTensor`). These grid-based functions have been removed in v2. Use `cosSimExpTens`, which has always computed similarity analytically (since v1) and has been substantially optimized in v2 — the original double loop over r-ad combinations has been replaced by fully vectorized operations over pre-calculated r-ads, with automatic memory-aware chunking.

```matlab
% v1
T_x = expectationTensor(p1, w1, sigma, r, isRel, isPer, period, nPoints);
T_y = expectationTensor(p2, w2, sigma, r, isRel, isPer, period, nPoints);
s = cosSim(T_x, T_y);
```

```matlab
% v2 equivalent
s = cosSimExpTens(p1, w1, p2, w2, sigma, r, isRel, isPer, period);
```

### spectralize → addSpectra

The v1 `spectralize` function added harmonic partials. The v2 `addSpectra` function generalises this to five spectral modes.

```matlab
% v1
[p2, w2] = spectralize(p, w, nHarmonics, rho);
```

```matlab
% v2 equivalent (harmonic mode with power-law decay)
[p2, w2] = addSpectra(p, w, 'harmonic', nHarmonics, 'powerlaw', rho);
```

v2 additionally supports `'stretched'`, `'freqlinear'`, `'stiff'`, and `'custom'` modes, as well as `'geometric'` weight decay. See `help addSpectra`.

### bal → balanceCircular

```matlab
% v1
b = bal(p, w, period);
```

```matlab
% v2
b = balanceCircular(p, w, period);
```

### eve → evennessCircular

```matlab
% v1
e = eve(p, period);
```

```matlab
% v2
e = evennessCircular(p, period);
```

### fSetRoughness → roughness

```matlab
% v1
r = fSetRoughness(f, w);
```

```matlab
% v2 (same basic call; new options available)
r = roughness(f, w);
r = roughness(f, w, 'pNorm', 2, 'average', true);  % new options
```

### modeHeight → meanOffset

```matlab
% v1
h = modeHeight(p, w, period);
```

```matlab
% v2 (same algorithm; also accepts query points)
h = meanOffset(p, w, period);
h = meanOffset(p, w, period, x);  % evaluate at specific positions
```

### projCent → projCentroid

```matlab
% v1
y = projCent(p, w, period);
```

```matlab
% v2 (also returns centroid magnitude and phase)
[y, centMag, centPhase] = projCentroid(p, w, period);
```

### stepEntropy → nTupleEntropy

```matlab
% v1
H = stepEntropy(p, period, k);
```

```matlab
% v2 (k renamed to n; new options)
H = nTupleEntropy(p, period, n);
H = nTupleEntropy(p, period, n, 'sigma', 0.5, 'normalize', true);
```

The v2 function also removes the dependency on the external `histcn` function.

### Batch processing (new in v2)

If your v1 code looped over trials calling `cosSimExpTens` on each:

```matlab
% v1 pattern
for i = 1:nTrials
    [pA, wA] = spectralize(A(i,:), [], nHarm, rho);
    [pB, wB] = spectralize(B(i,:), [], nHarm, rho);
    s(i) = cosSimExpTens(pA, wA, pB, wB, sigma, r, isRel, isPer, period);
end
```

v2 provides a vectorized alternative with automatic deduplication:

```matlab
% v2
s = batchCosSimExpTens(A, B, sigma, r, isRel, isPer, period, ...
                       'spectrum', {'harmonic', nHarm, 'powerlaw', rho});
```

---

## New functions with no v1 equivalent

The following v2 functions are entirely new. See the User Guide for full documentation.

**Expectation tensor core:** `batchCosSimExpTens`, `estimateCompTime`

**Consonance and harmonicity:** `templateHarmonicity`, `tensorHarmonicity`, `virtualPitches`

**Utility:** `convertPitch`
