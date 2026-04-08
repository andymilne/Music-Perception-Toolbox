# Migration Guide: v1 → v2

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
s = cosSimExpTens(x_p, x_w, y_p, y_w, sigma, r, isRel, isPer, period);
```

The new preferred calling convention precomputes the density objects, which is faster when comparing a fixed reference against many sets:

```matlab
% v2 (preferred for repeated comparisons)
dens_ref = buildExpTens(x_p, x_w, sigma, r, isRel, isPer, period);
dens_cmp = buildExpTens(y_p, y_w, sigma, r, isRel, isPer, period);
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
T_x = expectationTensor(x_p, x_w, sigma, r, isRel, isPer, period, nPoints);
T_y = expectationTensor(y_p, y_w, sigma, r, isRel, isPer, period, nPoints);
s = cosSim(T_x, T_y);
```

```matlab
% v2 equivalent
s = cosSimExpTens(x_p, x_w, y_p, y_w, sigma, r, isRel, isPer, period);
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
