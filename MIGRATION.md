# Migration Guide

This guide documents migration paths between major versions of the Music Perception Toolbox.

- [v2.0 → v3.0](#v20--v30) — multi-attribute expectation tensors, the Möbius method (alongside Bulger's), the four-method entropy API, soft (`sigma > 0`) structural measures, unified dispatch and batching
- [v1 → v2](#v1--v2) — major rewrite (analytical methods, Python port, restructured core)

---

## v2.0 → v3.0

v3.0.0 is a major release relative to the last public line (2.0.x). At the v2.0 calling conventions it is largely additive — `buildMaet`, `evalMaet`, `simMaet`, `entropyMaet`, and the circular, harmony, and structural families accept every v2.0 call unchanged — but a handful of defaults and one keyword have changed, so some v2.0 code needs attention. The breaking items, each detailed below, are:

- the `normalize` kwarg is removed from `entropyMaet`, `spectralEntropy`, and `nTupleEntropy` in favour of a four-method `method` API, and `entropyMaet`'s default `method='shannon'` now returns raw $H$ rather than $H / \log_b N$;
- `entropyMaet` requires an explicit `n_points_per_dim` for the discrete methods;
- `spectralEntropy`'s default method is `'differential'`, a different quantity from the v2.0 normalised Shannon entropy;
- `nTupleEntropy` at `sigma > 0` reads `sigma` as positional uncertainty (`sigmaSpace = 'position'`) by default;
- a periodic attribute's kernel sums every periodic image (`wrap='full-image'`), where v2.0 wrapped each difference to one image, and the discrete entropies integrate each grid cell's mass rather than sampling its centre;
- kernel truncation defaults to `truncation_sigmas = 6`, so default-configured output is a ~6-significant-figure approximation of the v2.0 untruncated value (`Inf` restores it to within $10^{-12}$);
- `evalMaet` in periodic-relative mode uses the corrected pairwise-wrap quadratic form.

Everything else — multi-attribute expectation tensors, the pre-MAET preprocessing primitives, unified dispatch and batching, the Möbius method and its dispatcher, the kernel-evaluation controls, Rényi-2 and differential entropy, anisotropic kernels, and translation sweeps — is new surface that v2.0 code does not touch.

### `entropyMaet` / `entropy_maet` four-method API and `n_points_per_dim` default (breaking)

The entropy API has been refactored into four distinct methods — `'shannon'` (raw discrete), `'normalized'` / `'normalised'` (the explicit name for $H / \log_b N$), `'differential'` (adaptive continuous $\hat h$), `'renyi2'` (analytical Rényi-2) — and the toolbox-wide default of `n_points_per_dim=1200` for `entropyMaet` has been dropped. Two breaking elements:

1. **`n_points_per_dim` is now required for the discrete methods** (`'shannon'`, `'normalized'`). A missing value at these methods raises `TypeError` with a message pointing to either supplying an explicit grid or switching to a grid-free method (`'differential'` or `'renyi2'`). The continuous methods ignore `n_points_per_dim` and need no migration.

   ```python
   # v2.0 — implicit grid via the 1200 default
   H = entropy_maet(dens)

   # v3 — either pass the grid explicitly
   H = entropy_maet(dens, n_points_per_dim=1200)
   # ... or switch to the grid-free continuous form
   h = entropy_maet(dens, method='differential')
   ```

2. **`spectralEntropy` / `spectral_entropy` default switched from `method='shannon'` (with `normalize=True`) to `method='differential'`.** The returned quantity is now the adaptive differential entropy $\hat h$ — a different quantity in different units, not a fourth-decimal numerical shift. To reproduce the v2.0 default behaviour (the Pielou-style ratio in $[0, 1]$ on the 1-cent grid of Milne et al. 2017 and Smit et al. 2019, agreeing with those values to about 1e-4 because v3 integrates cell masses where v2.0 summed point samples), pass `method='normalized'`; the grid spacing is the new `resolution` keyword (default 1 cent):

   ```python
   # v2.0 default (normalised Shannon in [0, 1])
   H = spectral_entropy(p, sigma=12)

   # v3 — to reproduce the v2.0 default exactly
   H = spectral_entropy(p, sigma=12, method='normalized')

   # v3 default — the principled grid-independent differential entropy
   h = spectral_entropy(p, sigma=12)
   ```

   The semantic shift is intentional. Differential entropy $\hat h$ is grid-independent and compares densities of different cardinality or spread on the same scale, whereas the normalised Shannon ratio is grid-dependent (its denominator $\log_b N$ depends on the discretisation). Cross-density comparisons published in the toolbox's existing literature (Milne et al. 2017, Smit et al. 2019) used the normalised form and are reproduced by `method='normalized'`; new analyses should generally prefer `method='differential'`.

### `normalize` kwarg removed from `entropyMaet`, `spectralEntropy`, and `nTupleEntropy`

The v2.0 `normalize` boolean is **removed** from all three entropy entry points. Pick the appropriate `method` instead: `'shannon'` for raw $H = -\sum q \log_b q$, `'normalized'` for $H/\log_b(N) \in [0, 1]$ (the v2.0 default behaviour). The continuous methods `'differential'` and `'renyi2'` have no $[0, 1]$ reference. Passing `normalize` to any of the three entry points raises a migration-error exception identifying the calling function and naming the replacement methods.

```python
# Python — v2.0
H = entropy_maet(T, normalize=False)         # raw
H = entropy_maet(T)                          # H/log_b(N) (default)
H = entropy_maet(T, method='renyi2',
                     normalize=False)            # renyi2 (required kwarg)
H, _ = n_tuple_entropy(p, period, n=2,
                       normalize=False)          # raw
H_spec = spectral_entropy(p, sigma=12,
                          normalize=False)       # raw

# Python — v3
H = entropy_maet(T, method='shannon')        # raw
H = entropy_maet(T, method='normalized')     # H/log_b(N)
H = entropy_maet(T, method='renyi2')         # renyi2 (no normalize needed)
H, _ = n_tuple_entropy(p, period, n=2,
                       method='shannon')         # raw
H_spec = spectral_entropy(p, sigma=12,
                          method='shannon')      # raw
```

```matlab
% MATLAB — v2.0
H = entropyMaet(T, 'normalize', false);                       % raw
H = entropyMaet(T);                                           % H/log_b(N)
H = entropyMaet(T, 'method', 'renyi2', 'normalize', false);   % renyi2
[H, tuples] = nTupleEntropy(p, period, 2, 'normalize', false);   % raw
H_spec = spectralEntropy(p, [], 12, 'normalize', false);         % raw

% MATLAB — v3
H = entropyMaet(T, 'method', 'shannon');                      % raw
H = entropyMaet(T, 'method', 'normalized');                   % H/log_b(N)
H = entropyMaet(T, 'method', 'renyi2');                       % renyi2
[H, tuples] = nTupleEntropy(p, period, 2, 'method', 'shannon');  % raw
H_spec = spectralEntropy(p, [], 12, 'method', 'shannon');        % raw
```

The continuous methods (`'differential'`, `'renyi2'`) are rejected at `sigma=0` with an explicit error rather than silently producing $-\infty$.

Note that the **default** of `entropy_maet` / `entropyMaet` is still `method='shannon'` but the semantics of `shannon` have changed: v2.0's default was effectively shannon + normalize=true (i.e. $H/\log_b(N)$); v3's `method='shannon'` returns raw $H$. To recover the v2.0 default value pass `method='normalized'` explicitly. The default of `n_tuple_entropy` already gives the v2.0 default value without changes (`'normalized'`); `spectral_entropy` defaults to `'differential'`, which gives the same ordering as the v2.0 normalised Shannon for consonance work, on a different scale (see above).

### `nTupleEntropy` at `sigma > 0`: `sigmaSpace` and the exact position model

In v2.0, `nTupleEntropy(p, period, n, 'sigma', s)` (MATLAB) and `mpt.n_tuple_entropy(p, period, n, sigma=s)` (Python) treated `s` as independent uncertainty on each derived step size — interval-space semantics in v3 terminology. v3 introduces a `sigmaSpace` flag with default `'position'`: `s` is now treated as positional uncertainty on each event, and the `n` steps of a tuple carry the exact correlated covariance $\sigma^2\,\mathrm{tridiag}(2, -1)$ (variance $2s^2$ per step, $-s^2$ between adjacent steps), obtained by binding the $n + 1$ underlying events and taking the window relative. To recover v2.0 numerical results, pass `sigmaSpace = 'interval'`:

```matlab
% v2.0 behaviour (interval-space sigma)
H = nTupleEntropy(p, period, n, 'sigma', s);

% v3: same numerical result via explicit flag
H = nTupleEntropy(p, period, n, 'sigma', s, 'sigmaSpace', 'interval');

% v3 default (position-space sigma) — different numerical result at sigma > 0
H = nTupleEntropy(p, period, n, 'sigma', s);
```

```python
# v2.0 behaviour (interval-space sigma)
H, _ = mpt.n_tuple_entropy(p, period, n, sigma=s)

# v3: same numerical result via explicit flag
H, _ = mpt.n_tuple_entropy(p, period, n, sigma=s, sigma_space='interval')

# v3 default (position-space sigma) — different numerical result at sigma > 0
H, _ = mpt.n_tuple_entropy(p, period, n, sigma=s)
```

The new default reflects the toolbox-wide convention that `sigma` describes uncertainty on the input quantity, which for `nTupleEntropy` is positions. The two semantics coincide at `sigma = 0`, so calls without an explicit `sigma` argument are unaffected, and `sigma = 0` continues to give the published integer-step histogram of Milne 2015bc / Milne & Dean 2016. Note that at $n = 1$ the position-mode value is reported on the relative-quotient grid, so `'position'` at $\sigma$ is not simply `'interval'` at $\sigma\sqrt{2}$.

### Periodic kernels sum every periodic image (`wrap`)

Under the default `wrap='full-image'`, a periodic attribute's kernel sums the contributions of every periodic image within the truncation width — in point evaluation, inner products, and the cell masses of the Shannon and normalized entropies — where v2.0 wrapped each difference to a single image. The two agree to within the accuracy floor while σ/period is small (for the entropies, below about 0.06) and differ above it. To recover the v2.0 numbers, declare `wrap='single-image'` on the attribute when building the density. Two operands that declare different `wrap` on the same attribute are refused (`ValueError` / `mpt:wrapMismatch`); build both with the same declaration.

### `convertPitch` / `convert_pitch` replaced by `transformAttributes` / `transform_attributes`

The pitch and frequency conversions are now one case of the new elementwise preprocessing primitive `transformAttributes`, and the old names are removed. The replacement is mechanical:

```matlab
p = convertPitch(f, 'hz', 'cents');                 % v2.0
p = transformAttributes(f, [], {'hz', 'cents'});    % v3.0
```

```python
p = mpt.convert_pitch(f, 'hz', 'cents')                     # v2.0
p = mpt.transform_attributes(f, None, ('hz', 'cents'))      # v3.0
```

The seven scales and their formulas are unchanged, so converted values are bit-identical. The same function applies logarithmic and other transforms to a pre-MAET's attributes, adds the `'octave'` pitch scale, and refuses out-of-domain values (a zero under `'log'`, a negative under `'power'`) with a message giving the remedies; see User Guide §7.3.

### Default kernel truncation (numerical change)

The factory default of `truncation_sigmas` / `truncationSigmas` is `6`, not `Inf`. Every centres-path consumer (`evalMaet`, `simMaet` on Bulger's method, `entropyMaet`, `spectralEntropy`, `templateHarmonicity`, `virtualPitches`) therefore returns a ~6-significant-figure approximation of the untruncated v2.0 value by default; the worst-case absolute error at the default is about $2 \times 10^{-8}$ and falls at low-density query points. A one-time warning (`mpt:truncationDefault` / `mpt.TruncationDefaultWarning`) says so on first use. To recover the v2.0 numerics to within $10^{-12}$, set `truncation_sigmas` / `truncationSigmas` to `math.inf` / `Inf`, which resolves to the accuracy floor (about 7.43σ, where the kernel falls below $10^{-12}$), per call or globally:

```matlab
mptDefaults('truncationSigmas', Inf);
```

```python
mpt.set_default(truncation_sigmas=math.inf)
```

### What's new at the surface

- **`method` keyword** on `simMaet`, `evalMaet`, `entropyMaet` (and Python equivalents). Default `'auto'` runs a per-call cost model that picks between **Bulger's method** (the v2.0 inner-product decomposition) and the new **Möbius method** (partition decomposition with orbit collapse in the IP case). Explicit values on `simMaet`: `'bulger'` (v2.0 decomposition; IP-only), `'mobius'` (new in v3; IP, eval, total mass), `'centres'` (the unrestricted enumeration; the reference route), and `'contract'` (nested densities); on `evalMaet`: `'centres'` and `'mobius'`. The Möbius and Bulger methods agree to floating-point precision in the regimes where both are valid (the IP case); the dispatcher chooses based on speed without changing answers.

- **`method='renyi2'`** on `entropyMaet`. Closed-form Rényi-2 differential entropy via the Möbius method's inner product and total mass. The analytical route was conceptually available in v2.0 (the inputs were both already analytical) but is newly exposed as a user-facing option in v3 and made efficient at high $r$ / $K$ via the Möbius method. Continuous-form: returns $H_2 \in (-\infty, \log_b V]$, no $[0, 1]$ reference. The legacy `normalize` kwarg is removed across all entropy entry points (see above).

- **Shipped orbit tables for $r \in \{2, \ldots, 8\}$.** Both Python and MATLAB ship pre-built tables for $r = 2$ through $r = 8$. The user-build path remains available for $r > 8$, gated by a cost-preview warning that prints the Bell-number scaling and estimated build time before construction begins. Set `MPT_NO_BUILD_WARN=1` (environment variable) to suppress the preview message in automation contexts. The user-build cache lives at `~/.mpt/orbit_tables/` (overridable via `MPT_CACHE_DIR`) and persists across sessions. The hard cap on $r$ is 12; beyond that, the build cost is prohibitive even for one-off use.

- **`kernel_chunk_bytes` (Python) / `kernelChunkBytes` (MATLAB) default.** Sets the per-chunk byte budget for the toolbox's memory-aware chunkers (the centres path, Bulger's method on `simMaet`, and the Möbius relative-mode evaluator). Factory value `'auto'` resolves at call time to half of currently available physical memory, queried from `/proc/meminfo` on Linux, `vm_stat` on macOS, and `memory().PhysicalMemory.Available` on Windows; a 4 GiB fallback covers the case where all platform queries fail. An explicit positive integer (in bytes) overrides globally via `mptDefaults('kernelChunkBytes', N)` / `mpt.set_default(kernel_chunk_bytes=N)`. v2.0 code requires no changes; the new default produces chunk sizes that differ from v2.0's fixed budget, so values differ from v2.0 at floating-point reduction order (relative differences below $\sim 10^{-13}$) — same answer, different bit pattern. Pin to a fixed integer if you need bit-identity across sessions or machines.

- **Pre-MAET preprocessing.** Events are carried, before any density is built, as a pre-MAET (`packPreMaet` / `pack_pre_maet`), and a set of primitives acts on it: `differenceEvents` (inter-event differences, with a `circular` flag), `bindEvents` (consecutive events bound into super-events), `translateAttributes` (an offset added to an attribute), `transformAttributes` (elementwise transforms and scale conversions, replacing `convertPitch`), and `weightEvents` (event weighting by a window on one attribute). `sweptSimilarity` / `sweptEntropy` compute a similarity or entropy profile across a list of sweep values, translating a query, aligning a window, or both, and `readScore` / `preMaetFromAttrTable` read MIDI and MusicXML. None of this touches v2.0 code; see User Guide §6 and §7.

### What's new under the hood

- **Lazy density-struct.** `buildMaet` now defaults to `lazy=true`: the expensive density fields (`U_perm`, `wJ`, `V_comb`, `wV_comb`) are deferred until a consumer needs them. User-level code that goes through `simMaet` / `evalMaet` / `entropyMaet` is unaffected. v2.0-era MATLAB code that reads `dens.U_perm` or the other deferred fields directly should build with `'lazy', false`; in Python the fields are properties that materialise on first access.

- **`tensorHarmonicity` rewrite.** The function now bypasses `buildMaet` entirely and routes through the Möbius relative-mode evaluator (`mobius.evalOrbitRel`) with per-template caching. Output is unchanged at the floating-point level.

- **`tensorHarmonicity` batched mode rewritten to dedup-and-batch.** The previous per-row loop has been replaced with a two-pass implementation that groups rows by `(nP, dup)`, deduplicates canonical chord intervals within each group, and issues a single batched call to the Möbius relative-mode evaluator per group. Same FP path as scalar mode (batched values now match scalar values to machine precision by construction rather than only by result cache). Verbose mode prints a one-line groups summary only for batches with ≥ 100 valid rows, matching the `min_print_sec=10` "silent for fast" semantics used by the other batched functions.

### Numerical equivalence

- v2.0 default routing chose Bulger's method implicitly. v3 default routing chooses `method='auto'`, which selects Bulger's method for the regimes where it dominates and the Möbius method elsewhere. In regimes where both methods are valid (the IP case), they agree to floating-point precision. User-visible cosine / entropy / eval values match v2.0 at default settings to floating-point reduction order — the dispatcher does not change the answer, only the route — with the exceptions noted below and the default kernel truncation described above. The `kernelChunkBytes` factory default (`'auto'`) produces chunk sizes that depend on the machine's available memory rather than v2.0's fixed budget, which alters reduction order and so introduces relative differences below $\sim 10^{-13}$ at otherwise-identical inputs. For bit-identity across machines or sessions, set `kernelChunkBytes` (or `kernel_chunk_bytes`) to an explicit integer.

- **`evalMaet` periodic-relative numerical change.** The centres-path quadratic form $Q$ in `evalMaet` for the `isRel = true, isPer = true` case is corrected to the pairwise-wrap form of Eq 6, matching `simMaet` (the same fix that was applied to `simMaet` in v2.0.1). At typical perceptual $\sigma/P \le 0.03$ the corrected and prior forms agree as $O((\sigma/P)^{\infty})$, so most existing rel+per callers will see numerical output indistinguishable from v2.0 at default settings; above the threshold the difference becomes measurable. Non-periodic eval, absolute eval, and all `simMaet` / `entropyMaet` modes are unchanged. See `CHANGELOG.md` under *Fixed* for the technical detail.

- **Cell-mass integration in discrete entropy.** `method='shannon'` and `method='normalized'` integrate an absolute-mode density over each grid cell (per-axis erf differences) where v2.0 point-sampled it at the cell centres. The two converge as the grid is refined but differ at coarse-to-moderate resolution (about $10^{-2}$ at 2.6 samples per σ, $10^{-5}$ at 12).

- A configuration v2.0 rejected is accepted: `buildMaet` with `r=1, isRel=true` now emits a warning (id `buildMaet:isRelDegenerate`) instead of raising. The configuration is well-defined under v3's framework (constant 0-D space, total mass = $\sum w$, Rényi-2 = 0), so callers exploring degenerate parameter combinations no longer need a `try/catch`.

### Demo migrations

- `demo_triadConsonance` / `demo_triad_consonance` calls `evalMaet`, `entropyMaet`, and `simMaet` on raw arrays rather than building densities first, reflecting v3's treatment of `buildMaet` as a less user-facing step.

### `sameness` and `coherence` gain optional `sigma`

Both functions now accept an optional `sigma` argument and a `sigmaSpace` name-value flag. At `sigma = 0` (default), the v2.0 hard counts are recovered byte-for-byte; existing call sites are unaffected.

```matlab
% v2.0 — still works in v3
[sq, nDiff] = sameness(p, period);
[c, nc] = coherence(p, period);

% v3 — soft sigma version
[sq, nDiff] = sameness(p, period, sigma);
[c, nc] = coherence(p, period, sigma);

% v3 — interval-space sigma (different per-pair variance)
[sq, nDiff] = sameness(p, period, sigma, 'sigmaSpace', 'interval');
```

Float positions and float `period` are accepted when `sigma > 0`. The integer requirement (and rejection of non-integer input) applies only at `sigma = 0`.

### `balanceCircular`, `evennessCircular` gain optional `sigma`

Both functions now accept an optional `sigma` argument that triggers Monte Carlo estimation under positional jitter via the new `dftCircularSimulate`. At `sigma = 0` the v2.0 deterministic value is recovered exactly.

```matlab
% v2.0 — still works in v3
b = balanceCircular(p, w, period);
e = evennessCircular(p, period);

% v3 — expected balance / evenness under sigma jitter
b = balanceCircular(p, w, period, sigma);
e = evennessCircular(p, period, sigma);

% v3 — also request standard deviation (MATLAB nargout idiom)
[b, bStd] = balanceCircular(p, w, period, sigma);
[e, eStd] = evennessCircular(p, period, sigma);

% Optional name-value: nDraws (default 10000), rngSeed
[b, bStd] = balanceCircular(p, w, period, sigma, 'nDraws', 50000, 'rngSeed', 42);
```

In Python, the SD is requested via an explicit `return_std=True` flag (Python lacks `nargout`):

```python
# v2.0 — still works in v3
b = mpt.balance(p, None, period)
e = mpt.evenness(p, period)

# v3 — scalar mean (backward-compatible signature)
b = mpt.balance(p, None, period, sigma=s)
e = mpt.evenness(p, period, sigma=s)

# v3 — opt in to (mean, std) tuple
b, b_std = mpt.balance(p, None, period, sigma=s, return_std=True)
e, e_std = mpt.evenness(p, period, sigma=s, return_std=True)
```

### `projCentroid` gains optional `sigma` (analytical, no Monte Carlo)

Because $y(x)$ is linear in $F(0)$ and $F(0)$ is permutation-invariant under positional jitter, the mean projection has a clean closed form: $E[y(x)] = \alpha_1 \cdot y_{\text{deterministic}}(x)$ where $\alpha_1 = \exp(-2\pi^2 \sigma^2 / P^2)$ and $P$ is the period. No Monte Carlo is involved.

```matlab
% v2.0 — still works in v3
[y, centMag, centPhase] = projCentroid(p, w, period, x);

% v3 — expected projection under sigma jitter (analytical)
[y, centMag, centPhase] = projCentroid(p, w, period, x, sigma);
```

`centMag` returns $\alpha_1 \cdot |F(0)| = |E[\widetilde{F}(0)]|$, the magnitude of the *complex mean centroid* — consistent with the projection. The distinct scalar $E[|\widetilde{F}(0)|]$ — the *mean centroid magnitude under jitter*, picking up positive Rayleigh-style bias when the perturbation cloud straddles the origin — is what `balanceCircular(p, w, period, sigma)` returns (read as `1 - b`). The two answer different balance-related questions; see the `projCentroid` entry, User Guide §12.7 for the operational distinction. Notation: $\widetilde{F}(0)$ is the random variable $F(0)$ becomes when each $p_k$ is replaced by $\widetilde{p}_k = (p_k + \eta_k) \bmod P$ with $\eta_k \sim \mathcal{N}(0, \sigma^2)$.

`centPhase` is preserved in expectation (the argument of $E[\widetilde{F}(0)]$ equals the argument of $F(0)$).

### Unified dispatch on `evalMaet`, `simMaet`, `entropyMaet`

In v3, the three core entry points are polymorphic. Each accepts:

- a single density object (the v2.0 case);
- raw arguments for a single multiset (the v2.0 case);
- a cell array (MATLAB) or list (Python) of density objects (new "list mode");
- a 2-D pitch matrix with both dimensions > 1 (new "batched-raw mode").

All forms produce byte-identical results to v2.0 for the v2.0 calling conventions; the new modes are dispatched on input shape and never collide with the existing paths.

```matlab
% v2.0 — still works in v3
s = simMaet(p1, w1, p2, w2, sigma, r, isRel, isPer, period);

% v3 — list mode (single context against many candidates)
densCtx = buildMaet(pCtx, wCtx, sigma, r, isRel, isPer, period);
densCands = arrayfun(@(i) buildMaet(pCands{i}, wCands{i}, sigma, r, isRel, isPer, period), ...
                     1:nCands, 'UniformOutput', false);
sims = simMaet(densCtx, densCands);   % 1-by-nCands cell

% v3 — batched-raw mode (paired multisets row-by-row)
sims = simMaet(P1, W1, P2, W2, sigma, r, isRel, isPer, period);   % length-nRows vector
```

```python
# v2.0 — still works in v3
s = mpt.sim_maet(p1, w1, p2, w2, sigma, r, is_rel, is_per, period)

# v3 — list mode
dens_ctx = mpt.build_maet(p_ctx, w_ctx, sigma, r, is_rel, is_per, period)
dens_cands = [mpt.build_maet(p, w, sigma, r, is_rel, is_per, period) for p, w in cands]
sims = mpt.sim_maet(dens_ctx, dens_cands)   # length-n_cands ndarray

# v3 — batched-raw mode
sims = mpt.sim_maet(P1, W1, P2, W2, sigma, r, is_rel, is_per, period)
```

The batched-raw mode also supports broadcasting: when one of `P1` / `P2` is an `M`-by-`K` matrix and the other is a length-`K` vector, the vector is broadcast across the matrix's rows. Eliminates the `repmat(refPitches, M, 1)` / `np.tile(ref_pitches, (M, 1))` idiom for the common "compare one reference against many candidates" use case.

In Python list × list mode, an additional `mode='cartesian'` returns the full `m`-by-`n` cross-product as a 2-D ndarray; MATLAB list mode is currently pairwise-only (with scalar broadcast).

### The core entry points are renamed on the MAET (breaking)

The entry points are named for the object they act on, the multi-attribute expectation tensor:

| v2.0 | v3 |
|:---|:---|
| `buildExpTens` / `build_exp_tens` | `buildMaet` / `build_maet` |
| `evalExpTens` / `eval_exp_tens` | `evalMaet` / `eval_maet` |
| `cosSimExpTens` / `cos_sim_exp_tens` | `simMaet` / `sim_maet` |
| `entropyExpTens` / `entropy_exp_tens` | `entropyMaet` / `entropy_maet` |
| `ExpTensDensity` (Python) | `MaetDensity` |

Signatures, arguments, and returned values are unchanged, so the migration is the name alone:

```matlab
dens = buildExpTens(p, w, sigma, r, isRel, isPer, period);   % v2.0
s    = cosSimExpTens(densX, densY);

dens = buildMaet(p, w, sigma, r, isRel, isPer, period);      % v3
s    = simMaet(densX, densY);
```

```python
dens = mpt.build_exp_tens(p, w, sigma, r, is_rel, is_per, period)   # v2.0
s = mpt.cos_sim_exp_tens(dens_x, dens_y)

dens = mpt.build_maet(p, w, sigma, r, is_rel, is_per, period)       # v3
s = mpt.sim_maet(dens_x, dens_y)
```

There are no deprecation shims: the old names are gone, and a call to one raises an unrecognized-name error. One thing beyond the names moves with them — the error and warning identifiers, so a `try` / `catch` matching `cosSimExpTens:badMethod` needs `simMaet:badMethod`, and likewise for the other four. The density struct's `MaetDensity` tag is unchanged, as are the already-deprecated `batchCosSimExpTens` and the `_raw` entry points.

### `batchCosSimExpTens`, `cos_sim_exp_tens_raw`, and `eval_exp_tens_raw` are removed (breaking)

These v2.0 names are gone in v3. Each call they served is a mode of the unified entry point:

| Removed (v2.0) | v3 replacement | Migration |
|:---|:---|:---|
| `batchCosSimExpTens` (MATLAB) | `simMaet` batched-raw mode | Move the weights from name-value pairs to the position after each pitch matrix: `simMaet(P1, W1, P2, W2, sigma, r, isRel, isPer, period)`. |
| `batch_cos_sim_exp_tens` (Python) | `sim_maet` batched-raw mode | Same, with `weights_a` / `weights_b` becoming positional. |
| `cos_sim_exp_tens_raw` (Python) | `sim_maet` | Drop the `_raw` suffix; the signature is unchanged. |
| `eval_exp_tens_raw` (Python) | `eval_maet` | Drop the `_raw` suffix; the signature is unchanged. |

### Batched-input dispatch on harmony, DFT, and structural families

The following functions gained batched-input dispatch in v3 and are fully backward-compatible at the v2.0 1-D calling convention. Pass a 2-D pitch matrix (rows are multisets) to get per-row results:

- **Harmony / consonance:** `tensorHarmonicity`, `templateHarmonicity`, `virtualPitches`, `spectralEntropy`.
- **DFT-equivariant:** `dftCircular`, `meanOffset`, `edges`, `projCentroid`, `circApm`.
- **Structural:** `coherence`, `sameness`, `nTupleEntropy`.
- **Monte Carlo:** `balanceCircular`, `evennessCircular` (with new `rngScope` name-value).

NaN-padded rows are accepted for variable-cardinality inputs. Per-row dedup uses a canonical-form key matched to each function's invariance class — see the CHANGELOG for the per-family details.

### Verbose / time-estimate options on harmony and entropy wrappers

`templateHarmonicity`, `tensorHarmonicity`, `virtualPitches`, `spectralEntropy`, and `entropyMaet` (single-multiset batched) all gain a `verbose` name-value argument (MATLAB) / keyword argument (Python), default `true`, controlling whether the function prints an upfront time estimate. The estimate is suppressed by `verbose=false`. Numerical results are unchanged. Combined with `estimateCompTime`'s new `minPrintSec` parameter (default 10 s), short workloads (typical interactive use) are silent by default; long workloads earn a one-line estimate with a `Ctrl+C` cancellation reminder.

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

### simMaet

**No changes required.** The original signature is still supported:

```matlab
% v1 (still works in v2)
s = simMaet(p1, w1, p2, w2, sigma, r, isRel, isPer, period);
```

The new preferred calling convention precomputes the density objects, which is faster when comparing a fixed reference against many sets:

```matlab
% v2 (preferred for repeated comparisons)
dens_ref = buildMaet(p1, w1, sigma, r, isRel, isPer, period);
dens_cmp = buildMaet(p2, w2, sigma, r, isRel, isPer, period);
s = simMaet(dens_ref, dens_cmp);
```

Both conventions support `'verbose', false` to suppress console output.

### expectationTensor → buildMaet + evalMaet

In v1, `expectationTensor` built a discretized tensor on a grid and returned a multidimensional array. In v2, this is split into two functions:

```matlab
% v1
T = expectationTensor(p, w, sigma, r, isRel, isPer, period, nPoints);
```

```matlab
% v2 equivalent
dens = buildMaet(p, w, sigma, r, isRel, isPer, period);
x = linspace(0, period, nPoints + 1);
x = x(1:end-1);  % exclude duplicate endpoint for periodic case
vals = evalMaet(dens, x);
```

The v2 approach has three advantages: (1) the density object is built once and reused across multiple queries; (2) query points can be arbitrary (not restricted to a uniform grid); (3) the density evaluation uses memory-aware chunking to handle large problems.

### cosSim / expTensorSim → simMaet

In v1, `cosSim` and `expTensorSim` computed cosine similarity from precomputed discretized tensors (built by `expectationTensor`). These grid-based functions have been removed in v2. Use `simMaet`, which has always computed similarity analytically (since v1) and has been substantially optimized in v2 — the original double loop over r-ad combinations has been replaced by fully vectorized operations over pre-calculated r-ads, with automatic memory-aware chunking.

```matlab
% v1
T_x = expectationTensor(p1, w1, sigma, r, isRel, isPer, period, nPoints);
T_y = expectationTensor(p2, w2, sigma, r, isRel, isPer, period, nPoints);
s = cosSim(T_x, T_y);
```

```matlab
% v2 equivalent
s = simMaet(p1, w1, p2, w2, sigma, r, isRel, isPer, period);
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
H = nTupleEntropy(p, period, n, 'sigma', 0.5, 'method', 'normalized');
```

The v2 function also removes the dependency on the external `histcn` function.

### Batch processing (new in v2)

If your v1 code looped over trials calling `simMaet` on each:

```matlab
% v1 pattern
for i = 1:nTrials
    [pA, wA] = spectralize(A(i,:), [], nHarm, rho);
    [pB, wB] = spectralize(B(i,:), [], nHarm, rho);
    s(i) = simMaet(pA, wA, pB, wB, sigma, r, isRel, isPer, period);
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

**Utility:** `transformAttributes` (absorbs `convertPitch`)
