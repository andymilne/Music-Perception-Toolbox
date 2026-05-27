# Migration Guide

This guide documents migration paths between major versions of the Music Perception Toolbox.

- [v2.1 → v2.2](#v21--v22) — the Möbius method (alongside Bulger's), Rényi-2 entropy, ragged-K hybrid
- [v2.0 → v2.1](#v20--v21) — soft (`sigma > 0`) structural measures, Argand-DFT Monte Carlo
- [v1 → v2](#v1--v2) — major rewrite (analytical methods, Python port, restructured core)

---

## v2.1 → v2.2

v2.2.0 is largely additive: existing v2.1 calling conventions are preserved at the floating-point level for the default routing in standard regimes. One pre-MAET preprocessing primitive has a breaking signature change (see `differenceEvents` below); all other v2.1 code requires no changes.

### `differenceEvents` signature and periodicity convention (breaking)

The v2.1 signature `differenceEvents(pAttr, w, groups, diffOrders, periods)` is now `differenceEvents(pAttr, w, groups, diffOrders, 'circular', false)`: four positional arguments, with `'circular'` as a Name-Value (MATLAB) / keyword-only (Python) flag and no trailing `periods` positional argument. The `circular` flag controls whether the difference operator wraps at the event-sequence boundary (the natural choice for cyclic event sequences — looped rhythms, ostinati); the default `false` preserves the v2.1 leading-event-drop convention for non-cyclic inputs.

Periodicity is no longer handled inside `differenceEvents`. Differences are emitted as raw signed subtractions regardless of group periodicity; the mod-period wrap on periodic groups is applied by the kernel at MAET-construction time, downstream of `differenceEvents`. This aligns the function with the toolbox-wide convention that pre-MAET preprocessing primitives produce raw values and the kernel handles wrap downstream.

**Migration.** v2.1 callers passing a `periods` positional argument must remove it. Three cases:

```matlab
% v2.1 — periods positional
[pDiff, wDiff] = differenceEvents(pAttr, w, groups, 1, periods);

% v2.2 — drop periods; kernel handles wrap
[pDiff, wDiff] = differenceEvents(pAttr, w, groups, 1);
```

For analyses on cyclic event sequences (e.g. looped rhythms) that benefit from boundary wrap of the difference operator, opt in via the new flag:

```matlab
% v2.2 — circular differencing on a cyclic input
[pDiff, wDiff] = differenceEvents(pAttr, w, groups, 1, 'circular', true);
```

Numerically equivalent v2.1 / v2.2 outputs on the standard `differenceEvents` → `buildExpTens` → MAET-consumer pipeline: the v2.1 wrap was to the shortest signed arc on $[-P/2, P/2)$, while v2.2 leaves the signed subtraction unwrapped; the kernel applies the same mod-$P$ wrap on either input at evaluation time, so downstream densities and all MAET-consumer outputs (cosine similarities, entropies, evaluations) are identical at floating-point precision. Only direct consumers of the pre-MAET values themselves see the wrapping difference, and the toolbox does not ship any such consumer.

### What's new at the surface

- **`method` keyword** on `cosSimExpTens`, `evalExpTens`, `entropyExpTens` (and Python equivalents). Default `'auto'` runs a per-call cost model that picks between **Bulger's method** (the v2.1 inner-product decomposition) and the new **Möbius method** (partition decomposition with orbit collapse in the IP case). Explicit values: `'bulger'` (v2.1 decomposition; IP-only), `'mobius'` (new in v2.2; IP, eval, total mass), `'centres'` (eval only), `'direct'` (small problems). The Möbius and Bulger methods agree to floating-point precision in the regimes where both are valid (the IP case); the dispatcher chooses based on speed without changing answers.

- **`method='renyi2'`** on `entropyExpTens`. Closed-form Rényi-2 differential entropy via the Möbius method's inner product and total mass. The analytical route was conceptually available in v2.0 / v2.1 (the inputs were both already analytical) but is newly exposed as a user-facing option in v2.2 and made efficient at high $r$ / $K$ via the Möbius method. Default remains `method='shannon'` (v2.1 numerical-grid behaviour, unchanged — Shannon differential entropy has no closed form in any version). `normalize=True` with `method='renyi2'` raises `NotImplementedError` for now (the natural normaliser yields a $(-\infty, 1]$ range that doesn't compose with Shannon's $[0, 1]$); divide externally if needed.

- **`cancellationThreshold`** keyword on `cosSimExpTens` (default `1e-12`). Guards the Möbius method's alternating partition sum against catastrophic cancellation; if the cancellation ratio drops below the threshold, the dispatcher falls back to Bulger's method. Most callers will not need to touch it.

- **Shipped orbit tables for $r \in \{2, \ldots, 8\}$.** Both Python and MATLAB ship pre-built tables for $r = 2$ through $r = 8$. The user-build path remains available for $r > 8$, gated by a cost-preview warning that prints the Bell-number scaling and estimated build time before construction begins. Set `MPT_NO_BUILD_WARN=1` (environment variable) to suppress the preview message in automation contexts. The user-build cache lives at `~/.mpt/orbit_tables/` (overridable via `MPT_CACHE_DIR`) and persists across sessions. The hard cap on $r$ is 12; beyond that, the build cost is prohibitive even for one-off use.

- **`kernel_chunk_bytes` (Python) / `kernelChunkBytes` (MATLAB) default.** Sets the per-chunk byte budget for the toolbox's memory-aware chunkers (the centres path, Bulger's method on `cosSimExpTens`, and the Möbius relative-mode evaluator). Factory value `'auto'` resolves at call time to half of currently available physical memory, queried from `/proc/meminfo` on Linux, `vm_stat` on macOS, and `memory().PhysicalMemory.Available` on Windows; a 4 GiB fallback covers the case where all platform queries fail. An explicit positive integer (in bytes) overrides globally via `mptDefaults('kernelChunkBytes', N)` / `mpt.set_default(kernel_chunk_bytes=N)`. v2.1 code requires no changes; the new default produces chunk sizes that differ from v2.1's fixed budget, so values differ from v2.1 at floating-point reduction order (relative differences below $\sim 10^{-13}$) — same answer, different bit pattern. Pin to a fixed integer if you need bit-identity across sessions or machines.

- **`weightEvents` / `weight_events`.** New per-event preprocessing primitive. Computes a window factor from one attribute's values and multiplies it into the weight slot of another attribute, returning a transformed `(pAttr, wOut, groups)` three-tuple that feeds directly into `buildExpTens`. The signature names a single `inputAttr` (must have $K = 1$) supplying values to a window function specified by a centre $c$, a width $w$ (standard deviation), and a shape $\gamma \in [0, 1]$ that interpolates between pure Gaussian and pure rectangle under the fixed-variance rect–Gaussian convolution family; the resulting $(1, N)$ factor is written into the slot of `targetAttr` (which may equal `inputAttr` or be a different attribute, and may itself carry $K_{\text{target}} > 1$). A mandatory keyword-only `deleteInput` flag (no default) selects whether the input attribute is dropped from the output (the usual idiom for windowed-entropy workflows where time scaffolds the window and is no longer needed downstream) or preserved. Multi-axis windowing is expressed as a sequence of calls with the same `targetAttr`. The canonical composition `weightEvents` (with `deleteInput=true`) $\to$ `buildExpTens` $\to$ `entropyExpTens` is the windowed-entropy construction — the principal new analysis pattern that this primitive supports. See USER_GUIDE §3.7 (Pre-MAET processing) for conceptual coverage and §6.1 for the API entry.

- **`circular` flag on `differenceEvents` / `difference_events`.** New Name-Value (MATLAB) / keyword-only (Python) flag on `differenceEvents`, paralleling the existing flag on `bindEvents`. Default `false` preserves the v2.1 leading-event-drop convention. Set `circular = true` for cyclic event sequences (looped rhythms, ostinati) where the boundary difference is a genuine inter-event interval; the function then wraps at the sequence boundary and returns $N$ events at every order. Note that v2.2 also drops the v2.1 trailing `periods` positional argument from `differenceEvents`' signature; see the breaking-change section above.

### What's new under the hood

- **Lazy density-struct.** `buildExpTens` now defaults to `lazy=true`: the expensive density fields (`U_perm`, `wJ`, `V_comb`, `wV_comb`) are deferred until a consumer needs them. Consumers that read these fields directly should call `ensureExpTensExpensive(dens)` first; this is wired through the toolbox internally, so user-level code that goes through `cosSimExpTens` / `evalExpTens` / `entropyExpTens` is unaffected. If you have v2.1-era code that pokes at `dens.U_perm` directly, add an `ensureExpTensExpensive(dens)` call before the read.

- **Ragged-K hybrid for MA Möbius inner product.** The MA per-attribute IP wrapper now handles NaN-padded events natively via a per-event safe/unsafe partition. The dispatcher no longer routes on the presence of NaN entries. The v2.2.x unsafe-direct-enum path is K-grouped and batched (a single vectorised tensor contraction per `(K_eff_x, K_eff_y)` sub-block, replacing the v2.2.0 per-pair scalar loop), giving 1.5–2.4× end-to-end speedup on variable-cardinality MA workloads at no API change. Uniform-cardinality workloads are unaffected. No user-visible change unless you previously relied on the `has_nan` → Bulger fallback for some side-effect reason.

- **SA cosine-similarity dispatcher gains a timing probe.** When the cost-model pre-screen does not decisively favour one method (i.e., neither cost estimate dominates the other by ≥10× at the call's parameters), the dispatcher times each candidate method on a small subset of the workload and routes the full call accordingly. This is future-proof against incremental optimisations on either side. The user-visible default behaviour is unchanged in the regimes where the pre-screen was already decisive (which is the majority of typical workloads); only edge-case calls that previously got the wrong method now route correctly.

- **`tensorHarmonicity` rewrite.** The function now bypasses `buildExpTens` entirely and routes through the Möbius relative-mode evaluator (`mobius.evalOrbitRel`) with per-template caching. Output is unchanged at the floating-point level. The previous "consider K_template > 3" warning is removed since the Möbius method handles arbitrary K-template without the centres-array memory footprint.

- **`tensorHarmonicity` batched mode rewritten to dedup-and-batch.** The previous per-row loop has been replaced with a two-pass implementation that groups rows by `(nP, dup)`, deduplicates canonical chord intervals within each group, and issues a single batched call to the Möbius relative-mode evaluator per group. Same FP path as scalar mode (batched values now match scalar values to machine precision by construction rather than only by result cache). Verbose mode prints a one-line groups summary only for batches with ≥ 100 valid rows, matching the `min_print_sec=10` "silent for fast" semantics used by the other batched functions.

- **`mobius.evalOrbitRel` u-grid vectorisation.** The sequential `for j = 1:N_u` loop in the Möbius relative-mode evaluator has been replaced by a single chunked, batched call to `mobius.evalOrbitAbs`. `mobius.evalOrbitAbs` now accepts query arrays of shape `(r, ...)` with arbitrary trailing dimensions; previously it required `(r, n_q)` exactly. Existing `(r, n_q)` callers see no change. The vectorisation eliminates the per-u-point MATLAB/Python function-call boundary; the effect is largest for small-`n_q` calls (where dispatch dominated). Output is bit-identical to the previous sequential implementation.

### Numerical equivalence

- v2.1 default routing chose Bulger's method implicitly. v2.2 default routing chooses `method='auto'`, which selects Bulger's method for the regimes where it dominates and the Möbius method elsewhere. In regimes where both methods are valid (the IP case), they agree to floating-point precision. User-visible cosine / entropy / eval values match v2.1 at default settings to floating-point reduction order — the dispatcher does not change the answer, only the route — with one exception, noted below. The `kernelChunkBytes` factory default (`'auto'`) produces chunk sizes that depend on the machine's available memory rather than v2.1's fixed budget, which alters reduction order and so introduces relative differences below $\sim 10^{-13}$ at otherwise-identical inputs. For bit-identity across machines or sessions, set `kernelChunkBytes` (or `kernel_chunk_bytes`) to an explicit integer.

- **`evalExpTens` periodic-relative numerical change.** The centres-path quadratic form $Q$ in `evalExpTens` for the `isRel = true, isPer = true` case is corrected to the pairwise-wrap form of Eq 6, matching `cosSimExpTens` (the same fix that was applied to `cosSimExpTens` in v2.0.1). At typical perceptual $\sigma/P \le 0.03$ the corrected and prior forms agree as $O((\sigma/P)^{\infty})$, so most existing rel+per callers will see numerical output indistinguishable from v2.1 at default settings; above the threshold the difference becomes measurable. Non-periodic eval, absolute eval, and all `cosSimExpTens` / `entropyExpTens` modes are unchanged. See `CHANGELOG.md` under *Fixed* for the technical detail.

- A previously-rejected configuration was relaxed: `buildExpTens` with `r=1, isRel=true` now emits a warning (id `buildExpTens:isRelDegenerate`) instead of raising. The configuration is well-defined under v2.2's framework (constant 0-D space, total mass = $\sum w$, Rényi-2 = 0), so callers exploring degenerate parameter combinations no longer need a `try/catch`.

### Demo migrations

- Five demos (`demo_triadConsonance` / `demo_triad_consonance`, `demo_bindEvents` / `demo_bind_events`, plus the Python `demo_bind_events` helper functions) were migrated from the v2.1 `buildExpTens` + downstream pattern to direct raw-array calls on `evalExpTens`, `entropyExpTens`, and `cosSimExpTens`. This reflects the v2.x principle of treating `buildExpTens` as a less user-facing entity. Three further demos (`demo_helixBlend`, `demo_maetWindowing`, `demo_windowingReference`) keep the explicit `buildExpTens` until `windowedSimilarity` gains a raw-array overload (deferred; tracked as a TODO comment in the `windowedSimilarity` source).

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

### Unified dispatch on `evalExpTens`, `cosSimExpTens`, `entropyExpTens`

In v2.1.0, the three core entry points are polymorphic. Each accepts:

- a single density object (the v2.0 case);
- raw arguments for a single multiset (the v2.0 case);
- a cell array (MATLAB) or list (Python) of density objects (new "list mode");
- a 2-D pitch matrix with both dimensions > 1 (new "batched-raw mode").

All forms produce byte-identical results to v2.0 for the v2.0 calling conventions; the new modes are dispatched on input shape and never collide with the existing paths.

```matlab
% v2.0 — still works in v2.1
s = cosSimExpTens(p1, w1, p2, w2, sigma, r, isRel, isPer, period);

% v2.1 — list mode (single context against many candidates)
densCtx = buildExpTens(pCtx, wCtx, sigma, r, isRel, isPer, period);
densCands = arrayfun(@(i) buildExpTens(pCands{i}, wCands{i}, sigma, r, isRel, isPer, period), ...
                     1:nCands, 'UniformOutput', false);
sims = cosSimExpTens(densCtx, densCands);   % 1-by-nCands cell

% v2.1 — batched-raw mode (paired multisets row-by-row)
sims = cosSimExpTens(P1, W1, P2, W2, sigma, r, isRel, isPer, period);   % length-nRows vector
```

```python
# v2.0 — still works in v2.1
s = mpt.cos_sim_exp_tens(p1, w1, p2, w2, sigma, r, is_rel, is_per, period)

# v2.1 — list mode
dens_ctx = mpt.build_exp_tens(p_ctx, w_ctx, sigma, r, is_rel, is_per, period)
dens_cands = [mpt.build_exp_tens(p, w, sigma, r, is_rel, is_per, period) for p, w in cands]
sims = mpt.cos_sim_exp_tens(dens_ctx, dens_cands)   # length-n_cands ndarray

# v2.1 — batched-raw mode
sims = mpt.cos_sim_exp_tens(P1, W1, P2, W2, sigma, r, is_rel, is_per, period)
```

The batched-raw mode also supports broadcasting: when one of `P1` / `P2` is an `M`-by-`K` matrix and the other is a length-`K` vector, the vector is broadcast across the matrix's rows. Eliminates the `repmat(refPitches, M, 1)` / `np.tile(ref_pitches, (M, 1))` idiom for the common "compare one reference against many candidates" use case.

In Python list × list mode, an additional `mode='cartesian'` returns the full `m`-by-`n` cross-product as a 2-D ndarray; MATLAB list mode is currently pairwise-only (with scalar broadcast).

### Deprecated entry points

The following function names are deprecated in v2.1 and emit warnings on direct use. They continue to work, forwarding to the unified entry points:

| Deprecated (v2.0)              | v2.1 replacement                                                       | Migration |
|:-------------------------------|:-----------------------------------------------------------------------|:----------|
| `batchCosSimExpTens` (MATLAB)  | `cosSimExpTens` batched-raw mode                                       | Move weights from name-value pairs to positional arguments after each pitch matrix; pass `[]` for uniform |
| `batch_cos_sim_exp_tens` (Python) | `cos_sim_exp_tens` batched-raw mode                                 | Same as above; `weights_a` / `weights_b` keyword-only → positional `w1` / `w2` |
| `cos_sim_exp_tens_raw` (Python) | `cos_sim_exp_tens`                                                    | Single-line edit: drop the `_raw` suffix |
| `eval_exp_tens_raw` (Python)   | `eval_exp_tens`                                                        | Single-line edit: drop the `_raw` suffix |

The deprecation warnings will be emitted for at least one minor release before removal.

### Batched-input dispatch on harmony, DFT, and structural families

The following functions gained batched-input dispatch in v2.1.0 and are fully backward-compatible at the v2.0 1-D calling convention. Pass a 2-D pitch matrix (rows are multisets) to get per-row results:

- **Harmony / consonance:** `tensorHarmonicity`, `templateHarmonicity`, `virtualPitches`, `spectralEntropy`.
- **DFT-equivariant:** `dftCircular`, `meanOffset`, `edges`, `projCentroid`, `circApm`.
- **Structural:** `coherence`, `sameness`, `nTupleEntropy`.
- **Monte Carlo:** `balanceCircular`, `evennessCircular` (with new `rngScope` name-value).

NaN-padded rows are accepted for variable-cardinality inputs. Per-row dedup uses a canonical-form key matched to each function's invariance class — see the CHANGELOG for the per-family details.

### Verbose / time-estimate options on harmony and entropy wrappers

`templateHarmonicity`, `tensorHarmonicity`, `virtualPitches`, `spectralEntropy`, and `entropyExpTens` (SA batched) all gain a `verbose` name-value argument (MATLAB) / keyword argument (Python), default `true`, controlling whether the function prints an upfront time estimate. The estimate is suppressed by `verbose=false`. Numerical results are unchanged. Combined with `estimateCompTime`'s new `minPrintSec` parameter (default 10 s), short workloads (typical interactive use) are silent by default; long workloads earn a one-line estimate with a `Ctrl+C` cancellation reminder.

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
