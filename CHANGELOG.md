# Changelog

All notable changes to the Music Perception Toolbox are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Performance

- **Internal performance improvement for r = 1 single-attribute expectation tensors.** `buildExpTens` (MATLAB) and `build_exp_tens` (Python) now collapse equal-valued pitches/positions at build time when `r = 1`: multiset (p) elements with the same value are merged into a single element whose weight is the sum of the originals. This is mathematically exact at `r = 1`, since the Gaussian-mixture density depends on the source multiset only through its measure on the pitch line. No behavioural change, no API change. Speedup is proportional to the degree of repetition in the input; for probe-tone-style contexts (e.g., where 7 unique context pitch classes are repeated 3 times) the effective speedup on downstream `evalExpTens` and `cosSimExpTens` calls is approximately 9×. Not applied for `r ≥ 2`: at higher tuple orders the source multiplicity carries information about within-tuple structure (e.g., the `r = 2` relative density at interval 0 receives genuine contributions from pairs of equal-pitched events) and collapsing would alter the density.

### Fixed

- **Python packaging:** the `readme` field in `python/pyproject.toml` previously referenced `../README.md` (the toolbox-level README, one directory up from the Python package). Modern `setuptools` (>= 68) refuses paths outside the package directory for security reasons, causing `pip install` and `pip install -e .` to fail with "Cannot access '<path>/python/../README.md'". The field now uses an inline `text` description pointing readers to the GitHub repository for full documentation. Editable and non-editable installs now work without modification.



### Overview

Version 2.1.0 generalises the expectation tensor framework to **multiple attributes per event** (multi-attribute expectation tensors, MAETs), and adds a **post-tensor windowing** mechanism for sliding-window template matching along any MAET group. Together these let the toolbox handle questions that the single-attribute treatment of v2.0.0 could not: locating motif recurrences over time, probe-tone scanning on irregularly timed contexts, polyphonic voice-aware analyses, and multi-quantity similarity (e.g., pitch combined with register, timbre, or metrical position) with the same closed-form analytical cosine-similarity inner product that underlies the single-attribute case.

Two cross-event preprocessing primitives, `differenceEvents` and `bindEvents`, support analyses based on inter-event differences and on n-tuples of consecutive events. Two utilities for sequential analysis, `continuity` and `seqWeights`, also accompany the main additions.

This release also adds **soft (`sigma > 0`) versions of the structural measures `sameness`, `coherence`, and `nTupleEntropy`**, with a `sigmaSpace` flag controlling whether `sigma` is interpreted as positional uncertainty on each event (the new default, propagated through index sharing among derived intervals) or as independent uncertainty on each derived interval (matching the v2.0 numerical behaviour of `nTupleEntropy` at `sigma > 0`). A separate strand adds **Monte Carlo sigma support to the Argand-DFT family** via the new `dftCircularSimulate` function: `balanceCircular`, `evennessCircular`, and `projCentroid` accept an optional `sigma` argument, with closed-form analytical damping where the geometry permits it (`projCentroid`, since the projection is linear in $F(0)$) and Monte Carlo where it does not (`balanceCircular`, `evennessCircular`).

The release is additive: existing v2.0.0 single-attribute calling conventions for `buildExpTens`, `evalExpTens`, `cosSimExpTens`, and `entropyExpTens` are preserved unchanged. The new multi-attribute calling form is dispatched on input shape (a cell / list of per-attribute arrays for `p`) and is otherwise separate. The structural and Argand-DFT extensions are likewise additive at `sigma = 0`; a small documented numerical change to `nTupleEntropy` at `sigma > 0` is described in [MIGRATION.md](MIGRATION.md) and below.

### Added — Multi-attribute expectation tensors

- **MAET calling form for `buildExpTens`, `evalExpTens`, `cosSimExpTens`, `entropyExpTens`** — Each now accepts a multi-attribute density specification alongside the v2.0.0 single-attribute form. A MAET carries $A$ attributes in $G \le A$ groups; each attribute has its own tuple order $r_a$ and per-event slot count $K_a$, and each group has shared $\sigma$, `isRel`, `isPer`, and `period`. The density factorises as a product of per-attribute Gaussian-mixture kernels, and the cosine-similarity inner product factorises analytically in the same way. The single-attribute case is the special case $A = 1$. See User Guide §3.1 for the mathematical formulation and the post-tensor windowing mechanism built on top of it.

### Added — Cross-event preprocessing

- **`differenceEvents`** — Preprocessing helper that replaces each selected group's event sequence with inter-event differences before tensor construction. Per-group differencing order $k$ produces the $k$-th finite differences along the event axis (reducing the event count by $k$); periods supplied per group wrap each raw difference to the shortest signed arc. Weights propagate as rolling products of width $k + 1$, interpretable as the probability that all constituents of a difference are jointly perceived. Inputs are restricted to $K_a = 1$ slots per event; see User Guide §3.1 (Cross-event preprocessing) for the rationale and the voices-as-attributes pipeline for polyphonic step-size analyses. Output feeds directly into `buildExpTens`. This supports analyses over interval content, IOIs, and higher-order differences without needing a dedicated interval representation.

- **`bindEvents`** — Preprocessing helper that gathers $n$ consecutive events into a single super-event with $n$ separate attributes carrying the values at lags $0$ through $n-1$. Output is a length-$n$ list of $K_a \times N'$ matrices, suitable as the `pAttr` argument of `buildExpTens` with all $n$ attributes assigned to a single group sharing $\sigma$ and the other group-level parameters. Lag slots are kept as separate attributes (rather than a single $K_a = n$ multi-slot attribute) because lag identity is non-exchangeable — within-attribute multiset symmetry would otherwise collapse ordered tuples to unordered ones. Per-event weights propagate per-attribute (each output attribute inherits the slot weights of the underlying event at its lag); `buildExpTens` multiplies across attributes during tuple enumeration, so the end-to-end effective weight of a bound super-event equals the product of the $n$ constituent events' weights, matching the differencing rule. Inputs may have any $K_a \ge 1$: when the input attribute has multi-value slots, each output attribute carries the full $K_a$-slot vector of its underlying event, with within-attribute exchangeability preserved per output attribute. Default output has $N - n + 1$ super-events; a `circular` option produces $N$ by wrapping around the input sequence. Composes with `differenceEvents`: the standard pipeline `differenceEvents` $\to$ `bindEvents` $\to$ `buildExpTens` $\to$ `entropyExpTens` recovers the n-tuple entropy of Milne & Dean (2016) at $\sigma \to 0$ and uniform weights, and extends it to the smoothed continuous case, weighted events, and non-periodic domains; for $n \ge 2$ the bound MAET is itself an $n$-dimensional density supporting cosine-similarity comparison of n-tuple distributions across pieces and the rest of the MAET pipeline. Used without preceding differencing, produces n-grams in absolute pitch or time register, useful when register or absolute timing carries musical information that the differenced view discards.

### Added — Post-tensor windowing

- **`windowTensor`** — Wraps a MAET with a per-group window specification (widths, shapes, and centres), returning a `WindowedMaetDensity` that is consumed lazily by `evalExpTens` (pointwise multiplication by the window) and by `cosSimExpTens` (closed-form windowed inner product). Windows are specified in a one-parameter family by `size` (width, in multiples of group $\sigma$) and `mix` in $[0, 1]$ (shape: pure Gaussian at 0, pure rectangular at 1, rectangular-convolved-with-Gaussian in between).

- **`windowedSimilarity`** — Sweeps a series of window centres across a windowed context and returns the windowed-similarity profile against a query. Implements cross-correlation: at each sweep centre $c$ in a windowed group, the query is translated so that its effective-space mean within that group moves onto $c$, so a peak at $c$ means the query pattern is present in the context near $c$. The unwindowed $L^2$ norms are used in the normaliser, making the returned profile magnitude-aware (denser matching content gives a higher value than sparser content). The output is *not* a cosine similarity in the strict sense (the unwindowed denominator means it is not bounded in $[-1, 1]$ across sweep positions); "cosine similarity" is reserved for the strict shape-only form, which is not currently implemented in the toolbox. See User Guide §3.1 "Post-tensor windowing".

- **Periodic-window approximation warning.** `windowedSimilarity` (MATLAB) and `windowed_similarity` (Python) now emit a warning on every call involving a periodic windowed group, since the closed form in use is the line-case formula and is exact only for non-periodic groups. The message has two forms: a brief informational notice within the recommended bound $\lambda\sigma \le P/(2\sqrt{3}) \approx 0.289 P$ (where the line-case approximation is sub-percent-accurate across the full $(\mathrm{size}, \mathrm{mix})$ window-shape family), and a stronger notice past the bound (reporting the rectangular component's half-width $\phi = \lambda\sigma\sqrt{3\mathrm{mix}}$ against $P/2$ and noting that behaviour at the boundary depends on `mix`). For windows much larger than a period, the windowed inner product collapses to the unwindowed form, which can be obtained directly from `cosSimExpTens` / `cos_sim_exp_tens` without going through the windowing machinery. MATLAB warning identifier `windowedSimilarity:periodicWindowApprox`; Python warning class `WindowedSimilarityPeriodicApproxWarning`, registered with an "always" filter so it fires on every offending call (matching the MATLAB per-call behaviour). Suppressible via the standard MATLAB / Python warning-filter mechanisms. See User Guide §3.1 "Post-tensor windowing".

- **Analytical closed form.** The cosine-similarity inner product between a windowed MAET and an unwindowed MAET is closed-form analytical across the full `(size, mix)` family for all 1-D groups and all multi-dimensional absolute groups; for multi-dimensional relative groups, it is established for the pure-Gaussian window only, with approximate handling of the full family left as a future extension.

### Added — Sequential-analysis utilities

- **`continuity`** — Expected length and signed magnitude of the backward same-direction run leading up to each query, under Gaussian pitch uncertainty. Smoothed directional-agreement score $a_k = \mathrm{erf}(i_k/(2\sigma)) \cdot \mathrm{erf}(i_N/(2\sigma))$ drives a backward walk from the most recent context interval. Two break-threshold modes (`'strict'` with $\theta = 0$, `'lenient'` with $\theta = -1$) and an explicit-threshold override. Returns `(count, magnitude)`; `magnitude / count` is a trend-slope measure. An optional `w` argument supplies per-event salience weights; the salience of difference event $k$ is the rolling product $w_k \cdot w_{k+1}$, matching `differenceEvents` at order 1, and scales each interval's contribution to `count` and `magnitude`. Internally, the function routes through `differenceEvents` to compute both the context intervals and their weights, so the preprocessing is shared with the MAET-with-differencing pipeline; the read-out (a directional, order-preserving backward walk with a break condition) is not. Defined only on linearly ordered domains.

- **`seqWeights`** — Constructs length-$N$ position-weight vectors from a named specification (`'flat'`, `'primacy'`, `'recency'`, `'exponentialFromStart'`, `'exponentialFromEnd'`, `'uShape'`) or an explicit profile. Supports time-based decay via an optional time index. Combines an intrinsic per-event salience with a memory-decay or attentional profile in a single inspectable vector, usable anywhere a weight argument is accepted.

### Added — Categorical-encoding utility

- **`simplexVertices`** (MATLAB), **`simplex_vertices`** (Python) — Returns the $N$ vertices of a regular $(N-1)$-simplex centred at the origin in $\mathbb{R}^{N-1}$, with all pairwise vertex distances equal to a specified `edgeLength` (default 1). Supports the simplex-coded encoding of an $N$-level categorical attribute (voice identity, instrument, etc.) for MAET inputs: each level is represented by an $(N-1)$-dimensional coordinate row, fed as values for the $N-1$ numerical sub-attributes of a single categorical group. All levels are pairwise equidistant, so no level is privileged over any other --- in contrast to dummy or treatment coding. Construction uses the centred standard basis of $\mathbb{R}^N$ projected onto an orthonormal basis of $\mathbf{1}^\perp$; the result is rotation-equivalent across choices of basis, which is irrelevant for downstream MAET computations.

### Added — Data structures

- **`MaetDensity`** and **`WindowedMaetDensity`** — Tagged dataclasses (Python) / tagged structs (MATLAB) produced by `buildExpTens` (multi-attribute form) and `windowTensor` respectively. `evalExpTens`, `cosSimExpTens`, and `entropyExpTens` dispatch on the tag.

### Added — Soft (`sigma > 0`) structural measures

- **`sameness` / `coherence` — `sigma` argument with `sigmaSpace` flag.** Both functions now accept an optional positive `sigma` that softens the discrete equality / ordering tests against Gaussian positional uncertainty. With `sigma = 0` (default) the v2.0 hard counts are recovered byte-for-byte. With `sigma > 0`, `sameness` replaces each `[d1 = d2]` indicator with a Gaussian match kernel $\exp(-(d_2 - d_1)^2 / (2V))$, and `coherence` replaces each `[d2 \le d1]` indicator with the Gaussian-CDF probability $\Phi((d_1 - d_2)/\sqrt{V})$. The variance $V$ depends on the new `sigmaSpace` name-value flag: with `sigmaSpace = 'position'` (default), $V$ is computed per pair of intervals from the sharing structure of their endpoints (so the diatonic tritone, where the fourth and fifth share both endpoints with reinforcing signs, has $V = 8\sigma^2$ and the ordering test sees a tie at every $\sigma > 0$); with `sigmaSpace = 'interval'`, $V = 2\sigma^2$ uniformly (intervals treated as independent draws, ignoring index correlations). Float positions and float `period` are accepted when `sigma > 0`. See User Guide §3.5 (Soft sigma semantics) and §6.5.

- **`nTupleEntropy` — `sigmaSpace` flag.** Adds a `sigmaSpace` name-value pair with the same semantics as `sameness` and `coherence`, governing whether the supplied `sigma` is treated as positional uncertainty on each event (the new default, with derived steps inheriting variance $2\sigma^2$ per slot via the marginal-matched approximation) or as independent uncertainty on each derived step (matching v2.0 numerical behaviour). At `n = 1` the two flags are exactly related: `'position'` with $\sigma$ produces identical entropy to `'interval'` with $\sigma\sqrt{2}$. At `n \ge 2` with `sigmaSpace = 'position'`, the current implementation uses the marginal-matched approximation (slots independent at $\sigma_{\text{eff}} = \sigma\sqrt{2}$); a one-time `UserWarning` (Python warning class `UserWarning` with message containing `marginal-matched`; MATLAB warning identifier `nTupleEntropy:positionApprox`) flags this. Full position-aware $n \ge 2$ support, requiring a non-diagonal Gaussian-mixture entropy routine, is planned for a future release. See User Guide §3.5 and §6.5.

- **`positionVariance`** — Shared helper that computes the variance of a signed sum of independently jittered positions, $V = \sigma^2 \sum_a (\sum \text{signs at index } a)^2$. Used internally by the position-aware paths of `sameness` and `coherence`; documented and exported because users implementing their own position-aware structural measures may want to call it directly. The same logic is exposed in Python as `mpt._utils.position_variance`.

### Added — Argand-DFT Monte Carlo

- **`dftCircularSimulate` / `dft_circular_simulate`** — Monte Carlo estimator of the distribution of Argand-DFT coefficient magnitudes under independent Gaussian positional jitter $\widetilde{p}_k = (p_k + \eta_k) \bmod P$, $\eta_k \sim \mathcal{N}(0, \sigma^2)$, where $P$ is the period. For each draw, the perturbed positions are sorted (capturing the perceptual reordering when noise is comparable to the smallest event-to-event gap), the Argand vector is formed, and the DFT is computed. Returns the per-coefficient mean and standard deviation of $|\widetilde{F}(k)|$ across draws by default; a third optional output (`samples` / `return_samples=True`) yields the full $n_{\text{draws}} \times K$ magnitude matrix for histogram or quantile analysis. Name-value parameters `nDraws` (default 10000) and `rngSeed` for reproducibility.

- **`balanceCircular`, `evennessCircular` — `sigma` argument.** Both functions now accept an optional positive `sigma` that triggers Monte Carlo estimation via `dftCircularSimulate` and returns the *expected* balance / evenness under positional jitter. At `sigma = 0` the v2.0 deterministic value is recovered exactly. The MATLAB versions also return the per-coefficient standard deviation as an optional second left-hand-side output (idiomatic `nargout`-driven multi-return); the Python versions accept a `return_std=True` flag that switches the return type from scalar to `(mean, std)` tuple. Backward-compatible at `sigma = 0` (scalar return preserved). Name-value `nDraws` and `rngSeed` are forwarded to the simulator. The augmented triad — perfectly balanced at `sigma = 0` — exhibits the expected Rayleigh-bias positive shift of $E[|F(0)|]$ at `sigma > 0`. See User Guide §6.4.

- **`projCentroid` — `sigma` argument (analytical closed form).** Because $y(x) = \mathrm{Re}(F(0) \cdot e^{-2\pi i x / P})$ is linear in $F(0)$ and $F(0)$ is permutation-invariant under positional jitter, no Monte Carlo is needed for the mean projection: $E[y(x)] = \alpha_1 \cdot y_{\text{deterministic}}(x)$ where $\alpha_1 = \exp(-2\pi^2 \sigma^2 / P^2)$ is the standard kernel-smoothing damping factor. Phase is preserved exactly in expectation, so `centPhase` is unchanged; `centMag` returns $\alpha_1 \cdot |F(0)| = |E[\widetilde{F}(0)]|$, the magnitude of the *complex mean centroid* — consistent with the projection. The distinct scalar $E[|\widetilde{F}(0)|]$ — the *mean centroid magnitude under jitter*, which picks up positive Rayleigh-style bias — is what `balanceCircular(p, w, period, sigma)` returns; the two answer different balance-related questions (see User Guide §3.5 "Two scalars, two balance-related questions").

### Changed

- Python `mpt` package version bumped to `2.1.0` (`__version__` and `pyproject.toml`).

- Renamed `windowedCosSim` (MATLAB) and `windowed_cos_sim` (Python) to `windowedSimilarity` and `windowed_similarity` respectively. The function's output is a magnitude-aware *windowed similarity*, which is not a cosine similarity in the strict sense (the unwindowed denominator means it is not bounded in $[-1, 1]$ across sweep positions and does not correspond to an inner product on a single Hilbert space). "Cosine similarity" is reserved for the strict shape-only form, which is not currently implemented in the toolbox. This rename is a breaking change relative to the unreleased v2.1.0 development drafts; v2.0.0 was unaffected (the function did not exist there). See User Guide §3.1 "Post-tensor windowing".

- **`windowedSimilarity` / `windowed_similarity` `reference` option.** Added a `reference` keyword argument (Python) / name-value pair (MATLAB) that allows the offset-axis origin per attribute to be set explicitly. When absent or `None` / `[]` (default), the prior unweighted-centroid behaviour is preserved byte-for-byte. When supplied, the call uses the given reference as the offset-axis origin per attribute — for example, calibrated to a canonical harmonic query so that offsets retain a stable musical-interval reading across queries with different slot counts, slot values, or slot weights. See User Guide §3.1 "Post-tensor windowing" for the practical guidance on default versus fixed-reference choices and the `demo_windowingReference` / `demo_windowing_reference` demo for a worked walkthrough.

- **`nTupleEntropy` / `n_tuple_entropy` refactored as a thin wrapper.** Internally now calls the `differenceEvents` $\to$ `bindEvents` $\to$ `buildExpTens` $\to$ `entropyExpTens` pipeline. With default arguments (`sigma = 0`, `nPointsPerDim = period`, `normalize = true`) it exactly replicates the discrete n-tuple entropy of Milne & Dean (2016) byte-for-byte; existing call sites at `sigma = 0` are unaffected. New optional name-value pairs expose the underlying flexibility: `nPointsPerDim` controls the entropy-evaluation grid resolution per dimension (the integer-step grid by default, finer for better resolution of smoothed densities). For non-integer values, weighted events, non-periodic domains, or to obtain the n-tuple density itself for similarity comparison and other MAET operations, call the primitives directly.

  **Numerical change at `sigma > 0`.** As described above under *Added — Soft (`sigma > 0`) structural measures*, the new `sigmaSpace` flag changes the default semantics of `sigma` at `sigma > 0`: v2.1.0's default `sigmaSpace = 'position'` treats `sigma` as positional uncertainty on each event, with derived steps inheriting variance $2\sigma^2$ per slot. v2.0.0's effective semantics treated `sigma` as independent per-step uncertainty, equivalent to `sigmaSpace = 'interval'`. The two coincide at `sigma = 0`. To recover v2.0.0 numerical results at `sigma > 0`, pass `sigmaSpace = 'interval'`. See [MIGRATION.md](MIGRATION.md) for a worked example.

- **`bindEvents` / `bind_events` documentation.** Added a clarification to the `circular` argument: it describes the *event sequence* (whether the last event connects back to the first), and is independent of the *positional periodicity* set in `buildExpTens` via its `isPer` / `period` arguments. Both combinations are meaningful: a non-circular sequence on a periodic domain (a non-cyclic motif living in pitch-class space), and a circular sequence on a linear domain (a cyclic rhythm represented in linear time, e.g., for windowed analysis). The two flags are orthogonal. No behavioural change — clarification only.

### Design notes

**Additivity.** The v2.0.0 single-attribute calling conventions are preserved verbatim: `buildExpTens(p, w, sigma, r, isRel, isPer, period)`, `cosSimExpTens(p1, w1, p2, w2, sigma, r, isRel, isPer, period)`, and friends continue to work unchanged. The multi-attribute calling form is distinguished by passing `p` as a cell array (MATLAB) or list (Python) of per-attribute value matrices, so the single- and multi-attribute paths do not collide.

**Option Z normalisation.** `windowedSimilarity` normalises by the product of the *unwindowed* L2 norms of the two operands (rather than, say, the windowed norm of the context). This makes the profile value at each sweep position reflect both how well the local content matches the query and how much matching content is present. Multiplying either operand's weights by a constant leaves the profile unchanged.

**MaetDensity slot layout.** For a relative group, `buildExpTens` stores the effective-space Gaussian centres using a slot-0-anchored reduction: $v_i = u_{i+1} - u_0$ for $i = 1, \ldots, r-1$. This convention is inherited by all downstream consumers, including the cross-correlation shift lift in `windowedSimilarity`.

---

## [2.0.2] — 2026-04-15

### Fixed

- **`batchCosSimExpTens` / `batch_cos_sim_exp_tens`** — The absolute periodic deduplication path (`isRel = false`, `isPer = true`) now uses the cyclic canonical form to normalise A-sets, collapsing all transposition-modulo-period equivalences. Previously, only transpositions that preserved the sort order after mod-reduction were detected; transpositions that wrapped pitches across the period boundary produced distinct keys for the same equivalence class.

### Changed

- When `'precision'` is set, canonical forms are re-rounded after mod-reduction and subtraction to absorb arithmetic noise from these operations. Previously, rounding was applied only to the raw input values.
- Verbose output now reports which canonicalisation mode was applied and notes that, in absolute mode, B-set counts reflect position relative to A's canonical form.
- Docstring lists the equivalences exploited in all four `isRel` / `isPer` combinations, and documents the limitation of `'precision'` for pitches on irrational grids (e.g., N-EDO where 1200/N is a repeating decimal) with the workaround of converting to integer EDO steps.

---

## [2.0.1] — 2026-04-15

### Fixed

- **`cosSimExpTens` / `cos_sim_exp_tens`** — Fixed incorrect transposition invariance in the periodic relative case (`isPer = true`, `isRel = true`). The quadratic form $Q$ now wraps pairwise differences between components of the wrapped difference vector, restoring exact transposition invariance on the circle. Previously, component-wise wrapping could introduce $\pm$period artefacts, producing cosine similarity discrepancies of $\sim 5 \times 10^{-7}$ for typical parameters ($\sigma = 10$, period = 1200). This issue was present in the original v1 implementation. Non-periodic and absolute results are unchanged.

### Changed

- **`batchCosSimExpTens` / `batch_cos_sim_exp_tens`** — Rewritten with three optimisations:
  - **Canonical-form deduplication** under `isPer` (mod reduction), `isRel` (transposition normalisation), and both (cyclic canonical form), reducing redundant computation when input rows contain equivalent pitch sets.
  - **Individual-set density caching** — `buildExpTens` is called once per unique multiset rather than once per unique pair.
  - **New `'precision'` parameter** — optionally rounds pitch and weight values to a specified number of decimal places before processing, ensuring nominally identical multisets differing only by floating-point noise are correctly deduplicated.
- User Guide updated to document the wrapped-pairwise-difference form of $Q$ for the periodic relative case.

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
