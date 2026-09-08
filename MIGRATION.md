# Migration Guide

This guide documents migration paths between major versions of the Music Perception Toolbox.

- [v2.0 → v3.0](#v20--v30) — multi-attribute expectation tensors, the Möbius method (alongside Bulger's), the four-method entropy API, soft (`sigma > 0`) structural measures, unified dispatch and batching
- [v1 → v2](#v1--v2) — major rewrite (analytical methods, Python port, restructured core)

---

## v2.0 → v3.0

*Versions 2.1 and 2.2 were internal milestones that were never released; their changes are consolidated here.*

v3.0.0 is a major release relative to the last public line (2.0.x). At the v2.0 calling conventions it is largely additive — `buildExpTens`, `evalExpTens`, `cosSimExpTens`, `entropyExpTens`, and the circular, harmony, and structural families accept every v2.0 call unchanged — but a handful of defaults and one keyword have changed, so some v2.0 code needs attention. The breaking items, each detailed below, are:

- the `normalize` kwarg is removed from `entropyExpTens`, `spectralEntropy`, and `nTupleEntropy` in favour of a four-method `method` API, and `entropyExpTens`'s default `method='shannon'` now returns raw $H$ rather than $H / \log_b N$;
- `entropyExpTens` requires an explicit `n_points_per_dim` for the discrete methods;
- `spectralEntropy`'s default method is `'differential'`, a different quantity from the v2.0 normalised Shannon entropy;
- `nTupleEntropy` at `sigma > 0` reads `sigma` as positional uncertainty (`sigmaSpace = 'position'`) by default;
- kernel truncation defaults to `truncation_sigmas = 6`, so default-configured output is a ~6-significant-figure approximation of the v2.0 untruncated value;
- `evalExpTens` in periodic-relative mode uses the corrected pairwise-wrap quadratic form.

Everything else — multi-attribute expectation tensors, the pre-MAET preprocessing primitives, unified dispatch and batching, the Möbius method and its dispatcher, the kernel-evaluation controls, Rényi-2 and differential entropy, anisotropic kernels, and translation sweeps — is new surface that v2.0 code does not touch.

### `entropyExpTens` / `entropy_exp_tens` four-method API and `n_points_per_dim` default (breaking)

The entropy API has been refactored into four distinct methods — `'shannon'` (raw discrete), `'normalized'` / `'normalised'` (the explicit name for $H / \log_b N$), `'differential'` (adaptive continuous $\hat h$), `'renyi2'` (analytical Rényi-2) — and the toolbox-wide default of `n_points_per_dim=1200` for `entropyExpTens` has been dropped. Two breaking elements:

1. **`n_points_per_dim` is now required for the discrete methods** (`'shannon'`, `'normalized'`). A missing value at these methods raises `TypeError` with a message pointing to either supplying an explicit grid or switching to a grid-free method (`'differential'` or `'renyi2'`). The continuous methods ignore `n_points_per_dim` and need no migration.

   ```python
   # v2.0 — implicit grid via the 1200 default
   H = entropy_exp_tens(dens)

   # v3 — either pass the grid explicitly
   H = entropy_exp_tens(dens, n_points_per_dim=1200)
   # ... or switch to the grid-free continuous form
   h = entropy_exp_tens(dens, method='differential')
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

### `normalize` kwarg removed from `entropyExpTens`, `spectralEntropy`, and `nTupleEntropy`

The v2.0 `normalize` boolean is **removed** from all three entropy entry points. Pick the appropriate `method` instead: `'shannon'` for raw $H = -\sum q \log_b q$, `'normalized'` for $H/\log_b(N) \in [0, 1]$ (the v2.0 default behaviour). The continuous methods `'differential'` and `'renyi2'` have no $[0, 1]$ reference. Passing `normalize` to any of the three entry points raises a migration-error exception identifying the calling function and naming the replacement methods.

```python
# Python — v2.0
H = entropy_exp_tens(T, normalize=False)         # raw
H = entropy_exp_tens(T)                          # H/log_b(N) (default)
H = entropy_exp_tens(T, method='renyi2',
                     normalize=False)            # renyi2 (required kwarg)
H, _ = n_tuple_entropy(p, period, n=2,
                       normalize=False)          # raw
H_spec = spectral_entropy(p, sigma=12,
                          normalize=False)       # raw

# Python — v3
H = entropy_exp_tens(T, method='shannon')        # raw
H = entropy_exp_tens(T, method='normalized')     # H/log_b(N)
H = entropy_exp_tens(T, method='renyi2')         # renyi2 (no normalize needed)
H, _ = n_tuple_entropy(p, period, n=2,
                       method='shannon')         # raw
H_spec = spectral_entropy(p, sigma=12,
                          method='shannon')      # raw
```

```matlab
% MATLAB — v2.0
H = entropyExpTens(T, 'normalize', false);                       % raw
H = entropyExpTens(T);                                           % H/log_b(N)
H = entropyExpTens(T, 'method', 'renyi2', 'normalize', false);   % renyi2
[H, tuples] = nTupleEntropy(p, period, 2, 'normalize', false);   % raw
H_spec = spectralEntropy(p, [], 12, 'normalize', false);         % raw

% MATLAB — v3
H = entropyExpTens(T, 'method', 'shannon');                      % raw
H = entropyExpTens(T, 'method', 'normalized');                   % H/log_b(N)
H = entropyExpTens(T, 'method', 'renyi2');                       % renyi2
[H, tuples] = nTupleEntropy(p, period, 2, 'method', 'shannon');  % raw
H_spec = spectralEntropy(p, [], 12, 'method', 'shannon');        % raw
```

The continuous methods (`'differential'`, `'renyi2'`) are rejected at `sigma=0` with an explicit error rather than silently producing $-\infty$.

Note that the **default** of `entropy_exp_tens` / `entropyExpTens` is still `method='shannon'` but the semantics of `shannon` have changed: v2.0's default was effectively shannon + normalize=true (i.e. $H/\log_b(N)$); v3's `method='shannon'` returns raw $H$. To recover the v2.0 default value pass `method='normalized'` explicitly. The default of `n_tuple_entropy` already gives the v2.0 default value without changes (`'normalized'`); `spectral_entropy` defaults to `'differential'`, which gives the same ordering as the v2.0 normalised Shannon for consonance work, on a different scale (see above).

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

### Routing and measure changes (September 2026)

The routing-parity work that closed v3 changes a handful of numbers and rejects a handful of calls. Each item says what changed and what to do.

- **Shannon and normalized entropy on periodic attributes now honour `wrap`.** Under the default `wrap='full-image'` the cell masses sum the erf over every periodic image the truncation admits, so `method='shannon'` and `method='normalized'` values on a periodic attribute differ from earlier builds once σ/period exceeds about 0.06 (below that the two readings agree to within the accuracy floor). To recover the old numbers, declare `wrap='single-image'` on the attribute when building the density.

- **Rényi-2 entropy of a relative $r = 1$ attribute is 0.** A relative monad is a zero-dimensional point mass and contributes no entropy; the value is now 0 by convention where it was previously undefined. Code that special-cased this configuration can drop the special case.

- **`method='factored'` removed** (Python). The name is rejected with the usual bad-method error. Use `'auto'` (or one of the documented names): the route was an undocumented Python-only entry point that bypassed the selector, the measure rule, and the post-hoc guard, and it computed the same value the documented routes compute.

- **`cancellation_threshold` (Python) / `'cancellationThreshold'` (MATLAB) removed** from `cos_sim_exp_tens` / `cosSimExpTens`. The keyword had been inert; passing it now raises `TypeError` (Python) or the argument-count usage error `cosSimExpTens:wrongArgCount` (MATLAB). Delete it from the call — nothing replaces it, `'auto'` is the whole of the routing, and accuracy is governed by `truncation_sigmas`.

- **Mixed `wrap` across the two operands is now an error.** Declaring `'full-image'` on an attribute of one density and `'single-image'` on the same attribute of the other raises `ValueError` (Python) / `mpt:wrapMismatch` (MATLAB); previously the first operand's declaration was taken silently. Build both densities with the same declaration.

- **MATLAB list and batched forms now honour `method`, `truncationSigmas`, and `kernelPrecision`.** A call in list or batched form that passed these keywords and relied on their being ignored will now route as the keywords say — a forced `'mobius'` or `'centres'` is applied to every entry, and a per-call `truncationSigmas` governs every entry. Remove the keyword, or pass `'auto'`, to keep the earlier behaviour.

### The pre-MAET operators return one pre-MAET (breaking)

`differenceEvents`, `bindEvents`, `translateAttributes`, `transformAttributes`, `weightEvents`, `readPreMaet` and `eventsFromScore` returned the three parts of a pre-MAET as separate outputs. They now return the whole pre-MAET as one object — a MATLAB struct with the fields `pAttr`, `wAttr` and `specs`, a Python dict with the keys `p_attr`, `w_attr` and `specs` — since the parts always travel together and always describe the same pre-MAET. `translateAttributes` returns `[pm, sweep]`, the sweep struct unchanged.

Call sites that want the parts wrap the call in `unpackPreMaet` / `unpack_pre_maet`:

```matlab
[pD, wD, sD] = differenceEvents(pAttr, w, [1 0]);                    % before
[pD, wD, sD] = unpackPreMaet(differenceEvents(pAttr, w, [1 0]));     % now
```

```python
pD, wD, sD = mpt.difference_events(p_attr, w, [1, 0])                        # before
pD, wD, sD = mpt.unpack_pre_maet(mpt.difference_events(p_attr, w, [1, 0]))   # now
```

Call sites that pass the result straight on are shorter than before, since the pre-MAET goes in whole and the specs no longer have to be threaded by hand:

```matlab
pm  = preMaet(pAttr, w);
pm2 = differenceEvents(bindEvents(pm, [2 2]), [1 1]);
dens = buildExpTens(pm2, 'sigma', [0.5 0.25], 'isPer', [true false], 'period', [12 0]);
```

`windowedSimilarity` and `windowedEntropy` gain the pre-MAET form too, taking two pre-MAETs and one respectively in place of their operands and their five positional geometry vectors; their positional form is unchanged.

The loose triple still works as input everywhere it did: `differenceEvents(pAttr, wAttr, [1 0], 'specs', specs)` is the same call as `differenceEvents(pm, [1 0])`. The weights argument is now named `wAttr` in the signatures that name `pAttr`, which matters only for a Python call that passed it by keyword as `w=`. The bare-array form of `transformAttributes` is unchanged: an array in, the transformed array out. See User Guide §7.4.5.

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

The seven scales and their formulas are unchanged, so converted values are bit-identical. The same function applies logarithmic and other transforms to a pre-MAET's attributes, adds the `'octave'` pitch scale, and refuses out-of-domain values (a zero under `'log'`, a negative under `'power'`) with a message giving the remedies; see User Guide §7.3.3.

### Default kernel truncation (numerical change)

The factory default of `truncation_sigmas` / `truncationSigmas` is `6`, not `Inf`. Every centres-path consumer (`evalExpTens`, `cosSimExpTens` on Bulger's method, `entropyExpTens`, `spectralEntropy`, `templateHarmonicity`, `virtualPitches`) therefore returns a ~6-significant-figure approximation of the untruncated v2.0 value by default; the worst-case absolute error at the default is about $2 \times 10^{-8}$ and falls at low-density query points. A one-time warning (`mpt:truncationDefault` / `mpt.TruncationDefaultWarning`) says so on first use. To recover exact v2.0 numerics, set `truncation_sigmas` / `truncationSigmas` to `math.inf` / `Inf`, per call or globally:

```matlab
mptDefaults('truncationSigmas', Inf);
```

```python
mpt.set_default(truncation_sigmas=math.inf)
```

### What's new at the surface

- **`method` keyword** on `cosSimExpTens`, `evalExpTens`, `entropyExpTens` (and Python equivalents). Default `'auto'` runs a per-call cost model that picks between **Bulger's method** (the v2.0 inner-product decomposition) and the new **Möbius method** (partition decomposition with orbit collapse in the IP case). Explicit values on `cosSimExpTens`: `'bulger'` (v2.0 decomposition; IP-only), `'mobius'` (new in v3; IP, eval, total mass), `'centres'` (the unrestricted enumeration; the reference route), and `'contract'` (nested densities); on `evalExpTens`: `'centres'` and `'mobius'`. The Möbius and Bulger methods agree to floating-point precision in the regimes where both are valid (the IP case); the dispatcher chooses based on speed without changing answers.

- **`method='renyi2'`** on `entropyExpTens`. Closed-form Rényi-2 differential entropy via the Möbius method's inner product and total mass. The analytical route was conceptually available in v2.0 (the inputs were both already analytical) but is newly exposed as a user-facing option in v3 and made efficient at high $r$ / $K$ via the Möbius method. Continuous-form: returns $H_2 \in (-\infty, \log_b V]$, no $[0, 1]$ reference. The legacy `normalize` kwarg is removed across all entropy entry points (see above).

- **Shipped orbit tables for $r \in \{2, \ldots, 8\}$.** Both Python and MATLAB ship pre-built tables for $r = 2$ through $r = 8$. The user-build path remains available for $r > 8$, gated by a cost-preview warning that prints the Bell-number scaling and estimated build time before construction begins. Set `MPT_NO_BUILD_WARN=1` (environment variable) to suppress the preview message in automation contexts. The user-build cache lives at `~/.mpt/orbit_tables/` (overridable via `MPT_CACHE_DIR`) and persists across sessions. The hard cap on $r$ is 12; beyond that, the build cost is prohibitive even for one-off use.

- **`kernel_chunk_bytes` (Python) / `kernelChunkBytes` (MATLAB) default.** Sets the per-chunk byte budget for the toolbox's memory-aware chunkers (the centres path, Bulger's method on `cosSimExpTens`, and the Möbius relative-mode evaluator). Factory value `'auto'` resolves at call time to half of currently available physical memory, queried from `/proc/meminfo` on Linux, `vm_stat` on macOS, and `memory().PhysicalMemory.Available` on Windows; a 4 GiB fallback covers the case where all platform queries fail. An explicit positive integer (in bytes) overrides globally via `mptDefaults('kernelChunkBytes', N)` / `mpt.set_default(kernel_chunk_bytes=N)`. v2.0 code requires no changes; the new default produces chunk sizes that differ from v2.0's fixed budget, so values differ from v2.0 at floating-point reduction order (relative differences below $\sim 10^{-13}$) — same answer, different bit pattern. Pin to a fixed integer if you need bit-identity across sessions or machines.

- **`weightEvents` / `weight_events`.** New per-event preprocessing primitive. Computes a window factor from one attribute's values and multiplies it into the weight slot of another attribute, returning a transformed `(pAttr, wOut, groups)` three-tuple that feeds directly into `buildExpTens`. The signature names a single `inputAttr` (must have $K = 1$) supplying values to a window function specified by a centre $c$, a scale given as either `sd` (the window's standard deviation) or `width` (the full support of the rectangle at `shape = 1`; exactly one of the two must be supplied), and a shape $\gamma \in [0, 1]$ that interpolates between pure Gaussian and pure rectangle under the fixed-variance rect–Gaussian convolution family; the resulting $(1, N)$ factor is written into the slot of `targetAttr` (which may equal `inputAttr` or be a different attribute, and may itself carry $K_{\text{target}} > 1$). A mandatory keyword-only `dropInputAttr` flag (no default) selects whether the input attribute is dropped from the output (the usual idiom for windowed-entropy workflows where time scaffolds the window and is no longer needed downstream) or preserved. Multi-axis windowing is expressed as a sequence of calls with the same `targetAttr`. The canonical composition `weightEvents` (with `dropInputAttr=true`) $\to$ `buildExpTens` $\to$ `entropyExpTens` is the windowed-entropy construction — the principal new analysis pattern that this primitive supports. See USER_GUIDE §7.3 (Pre-MAET processing) for conceptual coverage and §6.1 for the API entry.

- **`differenceEvents` / `difference_events` and its `circular` flag.** `differenceEvents(pAttr, w, groups, diffOrders, 'circular', false)` takes four positional arguments plus a `circular` Name-Value (MATLAB) / keyword-only (Python) flag, paralleling the flag on `bindEvents`. Differences are emitted as raw signed subtractions regardless of group periodicity; the kernel applies the mod-period wrap downstream. Default `circular = false` drops the leading events at each order. Set `circular = true` for cyclic event sequences (looped rhythms, ostinati) where the boundary difference is a genuine inter-event interval; the function then wraps at the sequence boundary and returns $N$ events at every order.

### What's new under the hood

- **Lazy density-struct.** `buildExpTens` now defaults to `lazy=true`: the expensive density fields (`U_perm`, `wJ`, `V_comb`, `wV_comb`) are deferred until a consumer needs them. Consumers that read these fields directly should call `ensureExpTensExpensive(dens)` first; this is wired through the toolbox internally, so user-level code that goes through `cosSimExpTens` / `evalExpTens` / `entropyExpTens` is unaffected. If you have v2.0-era code that pokes at `dens.U_perm` directly, add an `ensureExpTensExpensive(dens)` call before the read.

- **`tensorHarmonicity` rewrite.** The function now bypasses `buildExpTens` entirely and routes through the Möbius relative-mode evaluator (`mobius.evalOrbitRel`) with per-template caching. Output is unchanged at the floating-point level. The previous "consider K_template > 3" warning is removed since the Möbius method handles arbitrary K-template without the centres-array memory footprint.

- **`tensorHarmonicity` batched mode rewritten to dedup-and-batch.** The previous per-row loop has been replaced with a two-pass implementation that groups rows by `(nP, dup)`, deduplicates canonical chord intervals within each group, and issues a single batched call to the Möbius relative-mode evaluator per group. Same FP path as scalar mode (batched values now match scalar values to machine precision by construction rather than only by result cache). Verbose mode prints a one-line groups summary only for batches with ≥ 100 valid rows, matching the `min_print_sec=10` "silent for fast" semantics used by the other batched functions.

- **`mobius.evalOrbitRel` u-grid vectorisation.** The sequential `for j = 1:N_u` loop in the Möbius relative-mode evaluator has been replaced by a single chunked, batched call to `mobius.evalOrbitAbs`. `mobius.evalOrbitAbs` now accepts query arrays of shape `(r, ...)` with arbitrary trailing dimensions; previously it required `(r, n_q)` exactly. Existing `(r, n_q)` callers see no change. The vectorisation eliminates the per-u-point MATLAB/Python function-call boundary; the effect is largest for small-`n_q` calls (where dispatch dominated). Output is bit-identical to the previous sequential implementation.

### Numerical equivalence

- v2.0 default routing chose Bulger's method implicitly. v3 default routing chooses `method='auto'`, which selects Bulger's method for the regimes where it dominates and the Möbius method elsewhere. In regimes where both methods are valid (the IP case), they agree to floating-point precision. User-visible cosine / entropy / eval values match v2.0 at default settings to floating-point reduction order — the dispatcher does not change the answer, only the route — with the exceptions noted below and the default kernel truncation described above. The `kernelChunkBytes` factory default (`'auto'`) produces chunk sizes that depend on the machine's available memory rather than v2.0's fixed budget, which alters reduction order and so introduces relative differences below $\sim 10^{-13}$ at otherwise-identical inputs. For bit-identity across machines or sessions, set `kernelChunkBytes` (or `kernel_chunk_bytes`) to an explicit integer.

- **`evalExpTens` periodic-relative numerical change.** The centres-path quadratic form $Q$ in `evalExpTens` for the `isRel = true, isPer = true` case is corrected to the pairwise-wrap form of Eq 6, matching `cosSimExpTens` (the same fix that was applied to `cosSimExpTens` in v2.0.1). At typical perceptual $\sigma/P \le 0.03$ the corrected and prior forms agree as $O((\sigma/P)^{\infty})$, so most existing rel+per callers will see numerical output indistinguishable from v2.0 at default settings; above the threshold the difference becomes measurable. Non-periodic eval, absolute eval, and all `cosSimExpTens` / `entropyExpTens` modes are unchanged. See `CHANGELOG.md` under *Fixed* for the technical detail.

- A previously-rejected configuration was relaxed: `buildExpTens` with `r=1, isRel=true` now emits a warning (id `buildExpTens:isRelDegenerate`) instead of raising. The configuration is well-defined under v3's framework (constant 0-D space, total mass = $\sum w$, Rényi-2 = 0), so callers exploring degenerate parameter combinations no longer need a `try/catch`.

### Demo migrations

- Five demos (`demo_triadConsonance` / `demo_triad_consonance`, `demo_bindEvents` / `demo_bind_events`, plus the Python `demo_bind_events` helper functions) were migrated from the `buildExpTens` + downstream pattern to direct raw-array calls on `evalExpTens`, `entropyExpTens`, and `cosSimExpTens`. This reflects the v3 principle of treating `buildExpTens` as a less user-facing entity.

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

`centMag` returns $\alpha_1 \cdot |F(0)| = |E[\widetilde{F}(0)]|$, the magnitude of the *complex mean centroid* — consistent with the projection. The distinct scalar $E[|\widetilde{F}(0)|]$ — the *mean centroid magnitude under jitter*, picking up positive Rayleigh-style bias when the perturbation cloud straddles the origin — is what `balanceCircular(p, w, period, sigma)` returns (read as `1 - b`). The two answer different balance-related questions; see User Guide §6.5 "Two scalars, two balance-related questions" for the operational distinction. Notation: $\widetilde{F}(0)$ is the random variable $F(0)$ becomes when each $p_k$ is replaced by $\widetilde{p}_k = (p_k + \eta_k) \bmod P$ with $\eta_k \sim \mathcal{N}(0, \sigma^2)$.

`centPhase` is preserved in expectation (the argument of $E[\widetilde{F}(0)]$ equals the argument of $F(0)$).

### Unified dispatch on `evalExpTens`, `cosSimExpTens`, `entropyExpTens`

In v3, the three core entry points are polymorphic. Each accepts:

- a single density object (the v2.0 case);
- raw arguments for a single multiset (the v2.0 case);
- a cell array (MATLAB) or list (Python) of density objects (new "list mode");
- a 2-D pitch matrix with both dimensions > 1 (new "batched-raw mode").

All forms produce byte-identical results to v2.0 for the v2.0 calling conventions; the new modes are dispatched on input shape and never collide with the existing paths.

```matlab
% v2.0 — still works in v3
s = cosSimExpTens(p1, w1, p2, w2, sigma, r, isRel, isPer, period);

% v3 — list mode (single context against many candidates)
densCtx = buildExpTens(pCtx, wCtx, sigma, r, isRel, isPer, period);
densCands = arrayfun(@(i) buildExpTens(pCands{i}, wCands{i}, sigma, r, isRel, isPer, period), ...
                     1:nCands, 'UniformOutput', false);
sims = cosSimExpTens(densCtx, densCands);   % 1-by-nCands cell

% v3 — batched-raw mode (paired multisets row-by-row)
sims = cosSimExpTens(P1, W1, P2, W2, sigma, r, isRel, isPer, period);   % length-nRows vector
```

```python
# v2.0 — still works in v3
s = mpt.cos_sim_exp_tens(p1, w1, p2, w2, sigma, r, is_rel, is_per, period)

# v3 — list mode
dens_ctx = mpt.build_exp_tens(p_ctx, w_ctx, sigma, r, is_rel, is_per, period)
dens_cands = [mpt.build_exp_tens(p, w, sigma, r, is_rel, is_per, period) for p, w in cands]
sims = mpt.cos_sim_exp_tens(dens_ctx, dens_cands)   # length-n_cands ndarray

# v3 — batched-raw mode
sims = mpt.cos_sim_exp_tens(P1, W1, P2, W2, sigma, r, is_rel, is_per, period)
```

The batched-raw mode also supports broadcasting: when one of `P1` / `P2` is an `M`-by-`K` matrix and the other is a length-`K` vector, the vector is broadcast across the matrix's rows. Eliminates the `repmat(refPitches, M, 1)` / `np.tile(ref_pitches, (M, 1))` idiom for the common "compare one reference against many candidates" use case.

In Python list × list mode, an additional `mode='cartesian'` returns the full `m`-by-`n` cross-product as a 2-D ndarray; MATLAB list mode is currently pairwise-only (with scalar broadcast).

### Deprecated entry points

The following function names are deprecated in v3 and emit warnings on direct use. They continue to work, forwarding to the unified entry points:

| Deprecated (v2.0)              | v3 replacement                                                         | Migration |
|:-------------------------------|:-----------------------------------------------------------------------|:----------|
| `batchCosSimExpTens` (MATLAB)  | `cosSimExpTens` batched-raw mode                                       | Move weights from name-value pairs to positional arguments after each pitch matrix; pass `[]` for uniform |
| `batch_cos_sim_exp_tens` (Python) | `cos_sim_exp_tens` batched-raw mode                                 | Same as above; `weights_a` / `weights_b` keyword-only → positional `w1` / `w2` |
| `cos_sim_exp_tens_raw` (Python) | `cos_sim_exp_tens`                                                    | Single-line edit: drop the `_raw` suffix |
| `eval_exp_tens_raw` (Python)   | `eval_exp_tens`                                                        | Single-line edit: drop the `_raw` suffix |

The deprecation warnings will be emitted for at least one minor release before removal.

### Batched-input dispatch on harmony, DFT, and structural families

The following functions gained batched-input dispatch in v3 and are fully backward-compatible at the v2.0 1-D calling convention. Pass a 2-D pitch matrix (rows are multisets) to get per-row results:

- **Harmony / consonance:** `tensorHarmonicity`, `templateHarmonicity`, `virtualPitches`, `spectralEntropy`.
- **DFT-equivariant:** `dftCircular`, `meanOffset`, `edges`, `projCentroid`, `circApm`.
- **Structural:** `coherence`, `sameness`, `nTupleEntropy`.
- **Monte Carlo:** `balanceCircular`, `evennessCircular` (with new `rngScope` name-value).

NaN-padded rows are accepted for variable-cardinality inputs. Per-row dedup uses a canonical-form key matched to each function's invariance class — see the CHANGELOG for the per-family details.

### Verbose / time-estimate options on harmony and entropy wrappers

`templateHarmonicity`, `tensorHarmonicity`, `virtualPitches`, `spectralEntropy`, and `entropyExpTens` (single-multiset batched) all gain a `verbose` name-value argument (MATLAB) / keyword argument (Python), default `true`, controlling whether the function prints an upfront time estimate. The estimate is suppressed by `verbose=false`. Numerical results are unchanged. Combined with `estimateCompTime`'s new `minPrintSec` parameter (default 10 s), short workloads (typical interactive use) are silent by default; long workloads earn a one-line estimate with a `Ctrl+C` cancellation reminder.

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
H = nTupleEntropy(p, period, n, 'sigma', 0.5, 'method', 'normalized');
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

**Utility:** `transformAttributes` (absorbs `convertPitch`)
