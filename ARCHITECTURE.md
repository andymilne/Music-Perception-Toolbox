# Music Perception Toolbox – Architecture

A developer-facing map of what is in the toolbox, how the pieces relate, and the design rationale for the parts that are not obvious from a casual reading of the source. For *user-facing* documentation – what the functions do and how to call them – see [USER_GUIDE.md](USER_GUIDE.md). This document assumes the reader has either read USER_GUIDE §3 or is comfortable with the expectation-tensor framework from the source papers (Milne et al. 2011, 2015, 2016, 2020).

This document describes the toolbox as it currently exists. It was last verified line-by-line against the Python and MATLAB source in September 2026; the exhaustive, source-verified description of every routing decision is [ROUTING_MAP.md](ROUTING_MAP.md), which is the authoritative routing reference wherever this document and the map could be read differently.

## Contents

1. [Overview](#1-overview)
2. [Mathematical layering](#2-mathematical-layering)
3. [Code layering](#3-code-layering)
4. [The dispatcher pattern](#4-the-dispatcher-pattern)
5. [The orbit-table system](#5-the-orbit-table-system)
6. [Numerical guarantees](#6-numerical-guarantees)
7. [Twin-language conventions](#7-twin-language-conventions)
8. [How to add a new measure](#8-how-to-add-a-new-measure)
9. [Release discipline](#9-release-discipline)

---

## 1. Overview

The Music Perception Toolbox is implemented in parallel in MATLAB and Python with the same public function surface, the same routing rules, and the same internal architecture except where language conventions diverge. This twin-language parity is the toolbox's central organizing commitment – a refactor that would force the two sides apart is treated as a serious cost.

At the broadest level the toolbox stacks three computational tiers:

```
                      ┌────────────────────────────────────┐
                      │  Consumer wrappers                 │
                      │  (harmonicity, entropy, circular   │
                      │  measures, sequential utilities,   │
                      │  windowed and swept similarity, …) │
                      └──────────────┬─────────────────────┘
                                     │  consume
                                     ▼
                      ┌────────────────────────────────────┐
                      │  Tensor analysis primitives        │
                      │  (eval, inner product, total mass) │
                      └──────────────┬─────────────────────┘
                                     │  read
                                     ▼
                      ┌────────────────────────────────────┐
                      │  Density objects                   │
                      │  (MaetDensity – of which the       │
                      │  single multiset is the A = N = 1  │
                      │  corner)                           │
                      └────────────────────────────────────┘
```

Read top-down for *what calls what*; bottom-up for *what gets called by whom*. The density layer is the data structure shared by everything above it; the primitives layer is where the dispatchers live; the consumer layer is the published function set in §6 of USER_GUIDE.

There is one density type. A "single-multiset" density – one flat attribute holding one weighted multiset, the expectation tensor of Milne et al. (2011) – is not a separate class or a separate code path; it is the `A = N = 1` corner of `MaetDensity`, recognized by shape (`is_single_multiset` / `internal.isSingleMultiset`) and read through a flat-field view (`single_multiset_view` / `internal.singleMultisetView`) where a few faster kernels apply. The corner is a branch of the multi-attribute path and must return the same numbers as the general path; see §2.

Cross-cutting concerns – the Gaussian-kernel sum helper, the wrapped-Gaussian kernel, the Möbius package, the toolbox-wide defaults system, the cost models, the canonical-form deduplication machinery, and the `explain_dispatch` diagnostic – sit alongside this stack and are consumed at multiple tiers. Their organization is covered in §3.

The MATLAB and Python implementations share this layering. They diverge in mechanical translation only: function names (camelCase vs snake_case), absence sentinel (`[]` vs `None`), container types (cell array vs list), per-language convenience (MATLAB plotting features that have no Python equivalent without adding matplotlib as a dependency, Python dataclass conveniences with no clean MATLAB analogue), and the fitted constants of the cost models. The conventions for keeping the two sides aligned are documented in §7.

---

## 2. Mathematical layering

The three computational tiers each correspond to a distinct mathematical operation.

### Tier 1: Density objects

A weighted multiset $(\mathbf{p}, \mathbf{w})$ of $K$ values at tuple order $r$ defines an *r-ad expectation tensor density* – a Gaussian-mixture probability density on the $r$-fold product space, with one Gaussian centred at each $r$-tuple of distinct source values. A multi-attribute expectation tensor (MAET) density generalizes this to $N$ events, each carrying $A$ attributes; the density is the sum over events of the tensor product over attributes of the per-attribute densities. The density object stores the source multisets and geometry and precomputes – lazily – the tuple indices, weight products, and tuple centres; subsequent operations read from it without revisiting the source.

- `MaetDensity` (Python class; MATLAB struct with `tag == 'MaetDensity'`): $A$ attributes, $N$ events, per-attribute tuple order $r_a$, kernel width $\sigma_a$, periodicity flag and period, relativity flag, symmetry flag (`is_sym`: exchangeable tuples, or ordered), periodic measure declaration (`wrap`: `'full-image'` or `'single-image'`, §4), and an optional *nested* specification (a tag tree with per-level `r` and `sym` vectors, as produced by `bind_events`). An attribute may also carry an anisotropic kernel covariance (`kernel_cov`; the attribute's values are stored whitened with $\sigma = 1$). The effective space is the product of per-attribute effective spaces, each $\mathbb{R}^{r_a}$ (absolute) or its $(r_a - 1)$-dimensional translation quotient (relative).

- The **single-multiset corner** is `A = N = 1` with a flat first attribute. `is_single_multiset` tests exactly that shape; `single_multiset_view` exposes the flat names (`p`, `w`, scalar `sigma`, `r`, `is_rel`, `is_per`, `period`, and the per-tuple arrays) over the underlying `MaetDensity` so that the single-multiset kernels read one layout. The view is idempotent and, in Python, cached on the density by weak reference so that batch deduplication (which pairs operands by identity) stays stable. The `A = 1, r = 1, N > 1` case never reaches the corner as such: the build collapses it into one pooled event, so every downstream consumer meets the canonical `N = 1` form. Evaluation strategy for the corner is *shape-gated, not type-gated*: the corner offers speed (the direct kernel-sum leaf in evaluation, the single-attribute helper route in the Python inner product, canonical-key deduplication in batched forms), never a different value.

The build step (`build_exp_tens` / `buildExpTens`) is where the source multiset is consumed. The expensive per-tuple fields (`n_j`, `n_k`, `centres`, `u_perm`, `v_comb`, `w_j`, `wv_comb`, `event_of_j`, `event_of_k`) are *lazy*. In Python they are materialized on first attribute access and cached (`materialised` reports the state without triggering the build); in MATLAB, where structs have value semantics, `buildExpTens` returns a skinny struct by default (`'lazy', true`) and pairwise and centres consumers call `internal.ensureExpTensExpensive` before reading the per-tuple fields. Deferring the build keeps construction cheap and lets calls that route through the Möbius method – which reads only the cheap fields – skip the tuple-centres array entirely, whose footprint at high $r$ or high $K$ would otherwise dominate memory. See [§5](#5-the-orbit-table-system) for why this matters.

### Tier 2: Tensor analysis primitives

Three core analytical quantities operate on density objects:

- **Point evaluation** $T(\mathbf{x})$: the density's value at a query point. Single-density-single-query is the basic operation; batched evaluation (many queries, or many densities) is what most consumers actually need.

- **Inner product** $\langle T_x, T_y \rangle = \int T_x T_y \, \mathrm{d}\mathbf{z}$: the integral of the pointwise product of two densities over their shared support. Cosine similarity normalizes this by the L2 norms of both operands (`'oneSidedDenom'` divides by one of them; `'none'` returns the bare value on the canonical scale, `_ip_canonical_scale` / `internal.ipCanonicalScale` restoring the per-attribute prefactors each route drops); Rényi-2 differential entropy is computed from that bare self inner product and the total mass.

- **Total mass** $Z = \int T \, \mathrm{d}\mathbf{z}$: a scalar normalizer. Used to convert $T$ to a probability density via $T/Z$ and as a denominator in normalized quantities.

All three quantities have *closed-form analytical* expressions for the Gaussian-mixture density – no grid discretization is required. The expressions are sums over slot-tuples; the dispatchers described in [§4](#4-the-dispatcher-pattern) choose how to evaluate those sums. Because the density factorizes across attributes within an event, the inner product of two MAET densities is a sum over event pairs of a product over attributes of per-attribute inner matrices, and point evaluation is likewise a per-attribute product summed over events; every decomposition below is applied per attribute and the results are combined across attributes and events.

Four decompositions of the same sums coexist:

- **Bulger's method** (carried unchanged from v1) is specific to the inner product. It organizes the slot-tuple sum by combinations on one side and permutations on the other, exploiting within-tuple multinomial symmetry, and multiplies a truncated log-kernel over all attributes at once. It does not extend to point evaluation or total mass (the asymmetric combinations-vs-permutations organization requires two sides).

- **The centres route** enumerates tuples on both sides without restriction (permutations × permutations, $O(K^{2r})$ per attribute and event pair) and sums the kernel over every pair. For the inner product it is the slowest route at any appreciable $K$ and exists as the reference that reads the definition directly, and because it involves no alternating sum. For point evaluation the "centres" branch is the materialized-tuple-centres kernel sum – the joint materialization, or its factored per-attribute form when every $r_a \geq 2$ – and is the default at small problem sizes.

- **The Möbius method** applies uniformly to all three quantities via Möbius inversion on the partition lattice. For the inner product it composes with *orbit collapse* under joint slot-permutation symmetry, reducing $B_r^2$ partition pairs to $|\Omega_r|$ orbit equivalence classes; detailed in [§5](#5-the-orbit-table-system). Absolute attributes contract their orbit tables directly (densely, or sparsely when the kernel is large and sparse). Relative attributes need a translation marginal, and the method has three realizations of it: a **translation grid** of $N_u$ nodes (the alternating partition sum factorizes across slots only at fixed translation; integrating that product analytically would re-expand into the tuple enumeration the decomposition exists to avoid), a **spectral** Gram matrix on a mode grid for $2 \leq r \leq 4$, and a **tuple-centres closed form** that carries the relative Gaussian per tuple pair and is grid-free. The point evaluator in relative mode likewise integrates on a u-grid, with a *factored* strategy (tabulated smoothed event distributions read back per partition block, $K$-free per node) and a Fourier strategy at $2 \leq r \leq 4$ chosen by cost gates. The relative-periodic grid and spectral realizations compute the *all-image* measure; the tuple-centres and Bulger realizations compute the *minimum-image* measure (§4).

- **Nested contraction** (`_nested_contraction.py` / `internal.nestedContract`) is specific to attributes with a nested specification. Rather than enumerating every leaf-level tuple, it contracts the tag tree bottom-up: a recipe (tag tree plus per-node permutation/combination index arrays) is built once and reused across the three inner products and every quadrature node, and each symmetric level independently chooses the orbit reduction or explicit enumeration by a fitted cost model. The contraction composes with the Bulger and Möbius decompositions rather than replacing them. Absolute attributes need no quadrature; relative non-periodic attributes integrate the translation over a truncated line grid; relative-periodic attributes average over a $\tau$ grid on $[0, P)$ (the all-image measure, the one form in which the kernel factorizes per coordinate).

The decompositions are alternatives for the same analytical integral; their results agree to the accuracy floor in the regimes where each is admissible. They are not combined within a single quantity except as the nested plan combines them level by level – the dispatchers pick one route per attribute per call. The user-facing `method` knob defaults to `'auto'`; the accepted values are `'auto' | 'bulger' | 'centres' | 'mobius' | 'contract'` on the cosine (Python additionally accepts an undocumented `'factored'` route), `'auto' | 'centres' | 'mobius'` on evaluation, `'auto' | 'mixture' | 'orbit'` on the translation sweep, and `'differential' | 'shannon' | 'normalized' | 'renyi2'` on entropy, where the value names the estimator rather than a route. There is no `'direct'` method.

### Tier 3: Consumer wrappers

The consumer wrappers compose the tier-2 primitives into measures with musical interpretation:

- **Similarity**: `cos_sim_exp_tens`, `sweep_cos_sim_exp_tens` (one density against uniformly translated copies of another, as a Gaussian mixture in the offset), and `windowed_similarity` (a pre-MAET sliding window over raw events – the window reweights the events before each build – with each position routed through `cos_sim_exp_tens`) are themselves primitives or thin compositions of them; the spectral-enrichment wrapper (`add_spectra` applied before the cosine, producing spectral pitch-class similarity) is a consumer.

- **Harmonicity and consonance**: `template_harmonicity` cross-correlates a chord's composite spectrum against a harmonic template; `tensor_harmonicity` queries the density of interval patterns within a single harmonic series; `spectral_entropy` computes Shannon entropy of a spectral density; `roughness` (sensory roughness) is a direct frequency-pair calculation independent of the tensor framework; `virtual_pitches` extracts likely fundamentals via template harmonicity.

- **Entropy**: `entropy_exp_tens` evaluates the density's entropy (Shannon or normalized Shannon by cell masses on absolute densities or by grid evaluation on relative ones, differential by adaptive grid refinement with Richardson extrapolation, or Rényi-2 in closed form as the self inner product `cos_sim_exp_tens(dens, dens, normalize='none')` — through the inner-product selector, so every route and cost model of §4 serves it — over the squared closed-form total mass); `windowed_entropy` sweeps it across a window; `n_tuple_entropy` is a convenience wrapper composing `difference_events` + `bind_events` + `build_exp_tens` + `entropy_exp_tens` for the integer-step n-gram entropy of Milne & Dean (2016).

- **Circular measures**: `balance`, `evenness`, `coherence`, `sameness`, `edges`, `proj_centroid`, `mean_offset`, `circ_apm`, `markov_s`, with the DFT engine `dft_circular` and `dft_circular_simulate`. Some compose tensor primitives; others are direct DFT-based or symbolic computations independent of the tensor stack.

- **Sequential utilities**: `continuity` (smoothed direction-continuity), `seq_weights` (named time-based decay profiles), `interval_kernel_cov` (an anisotropic kernel covariance for an ordered tuple of consecutive differences).

- **Cross-event preprocessing**: `difference_events`, `bind_events`, `translate_attributes`, `transform_attributes`, `weight_events`, `flat_specs` – transform $(\mathbf{p}, \mathbf{w})$ or its specification before the tensor stack consumes them, supporting interval-based, n-gram, and swept analyses. `translate_attributes` can return a `TranslatedSweep` (Python only) that `cos_sim_exp_tens` recognizes and reduces to a sweep.

- **Utility and diagnostics**: `simplex_vertices` (categorical-attribute encoding), `add_spectra` (spectral enrichment), `audio_peaks` (spectral peak extraction), `read_score` / `events_from_score` (MIDI and MusicXML input, returning a pre-MAET), `estimate_comp_time`, `explain_dispatch` (reports how a call would be routed, without running it), and the defaults API (`get_default`, `set_default`, `get_defaults`, `reset_defaults`, `show_defaults` / `mptDefaults`).

The consumer layer is where measure-specific documentation belongs (see USER_GUIDE §6); the layering in this document stops at the tier-2 primitives.

---

## 3. Code layering

### Python module map

```
mpt/
├── __init__.py            Public API surface (re-exports + __all__); __version__
├── tensor.py              Re-export shim over _tensor/ (kept so existing
│                          `from mpt.tensor import X` imports – including
│                          developer-facing private names – continue to work)
├── _tensor/
│   ├── __init__.py        Re-exports the sub-modules' names
│   ├── density.py         MaetDensity, is_single_multiset,
│   │                      single_multiset_view, and the MA-input preprocessing
│   │                      helpers
│   ├── build.py           build_exp_tens (single-multiset and multi-attribute
│   │                      input forms; both produce a MaetDensity)
│   ├── premaet.py         pre_maet, unpack_pre_maet: the pre-MAET as one
│   │                      object, and the argument front end the operators share
│   ├── transform.py       transform_attributes (scale conversions, log and other elementwise maps)
│   ├── preprocessing.py   difference_events, bind_events, translate_attributes,
│   │                      weight_events, flat_specs, simplex_vertices
│   ├── aniso.py           Anisotropic (matrix-valued) kernel covariance support
│   ├── canonical.py       Canonical-form key helpers for batched dedup
│   ├── dispatch.py        Cost models and selectors: the flat inner-product
│   │                      selector, the eval selector and its cost model, the
│   │                      σ/P admissibility threshold, feasibility guards, and
│   │                      shared helpers (_normalize_density_input,
│   │                      _resolve_list_list_mode, _compute_Q)
│   ├── eval.py            eval_exp_tens: input forms, the centres branch
│   │                      (single-multiset leaf, factored MA, joint MA), and
│   │                      normalization
│   ├── _ma_eval_orbit.py  Factored Möbius point evaluator for a flat MAET
│   ├── cosine.py          cos_sim_exp_tens: input forms, the flat MA
│   │                      dispatcher, the Bulger, centres, and Möbius arms,
│   │                      the nested plan (_try_nested_contract), self-IP
│   │                      memoisation, and the deprecated shims
│   ├── _mobius_inner.py   Möbius per-attribute inner matrices: the absolute
│   │                      orbit contraction (dense / sparse), the relative
│   │                      translation grid, the spectral Gram matrix, the
│   │                      tuple-centres closed form, and the centres-vs-grid
│   │                      gate
│   ├── _centres_inner.py  Unrestricted tuple-centres inner product (test-only
│   │                      reference)
│   ├── _nested_contraction.py
│   │                      Tree contraction of the nested-attribute inner
│   │                      product: recipes, quadrature grids, per-level orbit
│   │                      vs enumeration, the accuracy guard
│   ├── _nested_cost.py    Fitted cost model for the nested plan and the
│   │                      plan-vs-enumeration race
│   ├── sweep.py           sweep_cos_sim_exp_tens: translation sweeps as a
│   │                      Gaussian mixture in the offset, with an orbit route
│   ├── windowed.py        Pre-MAET windowed sweeps: windowed_similarity,
│   │                      windowed_entropy
│   ├── explain.py         explain_dispatch: reports a call's routing and why
│   └── _timeest.py        Self-calibrated up-front time estimate for eval
├── circular.py            Re-export shim over _circular/
├── _circular/
│   ├── __init__.py        Re-exports the three sub-modules' names
│   ├── dft.py             DFT engine + DFT-based measures: dft_circular,
│   │                      dft_circular_simulate, balance, evenness,
│   │                      proj_centroid
│   ├── scale.py           Integer-position scale-theoretic measures
│   │                      (non-Fourier): coherence, sameness
│   └── pulse.py           Per-position pulse-level measures (non-Fourier):
│                          edges, mean_offset, circ_apm, markov_s
├── entropy.py             entropy_exp_tens, n_tuple_entropy
├── harmony.py             spectral_entropy, template_harmonicity,
│                          tensor_harmonicity, roughness, virtual_pitches
├── serial.py              continuity, seq_weights, interval_kernel_cov
├── spectra.py             add_spectra
├── audio.py               audio_peaks, AudioPeaksDetail
├── score.py               read_score, events_from_score (MIDI, MusicXML)
├── _kernel.py             Gaussian-kernel sum helper: the single centres-path
│                          numerical primitive (truncated 1-D, bucket-grid,
│                          circular, and exact chunked branches)
├── _wrapped_kernel.py     Wrapped-Gaussian 1-D kernel with automatic
│                          image-sum vs Fourier dispatch (the abs-per
│                          full-image measure)
├── _mobius.py             Möbius machinery: orbit-table build/load/cache, the
│                          orbit-contracted inner products, the Möbius point
│                          evaluators, total mass
├── _orbit_cost.py         Power-law cost model for the per-level orbit vs
│                          enumeration choice (mirror of internal.orbitCostModel)
├── _defaults.py           Toolbox-wide defaults API, the accuracy floor,
│                          truncation resolution, dispatch-scope guard
├── _utils.py              Small helpers (estimate_comp_time, validation,
│                          memory-aware chunk-budget resolution and pinning)
└── _orbit_tables/         Shipped orbit tables (pickle, r = 2..8)
```

### MATLAB layout

```
matlab/
├── *.m                    One file per public function (cosSimExpTens,
│                          evalExpTens, entropyExpTens, sweepCosSimExpTens,
│                          windowedSimilarity, explainDispatch, mptDefaults,
│                          …) – MATLAB requires one top-level function per
│                          file. The large entry points (cosSimExpTens.m,
│                          evalExpTens.m, entropyExpTens.m) carry their
│                          arms and kernel cores as local functions.
├── +internal/             Helpers reachable only via internal.helperName,
│                          mirroring Python's _-prefix convention: the
│                          selectors (selectMaInnerProductMethod,
│                          selectMaEval), cost models (relRouteCostMs,
│                          predictOrbitCostMs, maEvalCostsMs, nestedCost,
│                          orbitCostModel), guards and thresholds
│                          (relPerSigmaOverPThreshold, accuracyFloor,
│                          guardForcedBulgerFeasible, dispatchMemBudget),
│                          the nested contraction (nestedContract), the
│                          kernels (gaussianKernelSum, wrappedGaussian1d),
│                          the single-multiset corner
│                          (isSingleMultiset, singleMultisetView,
│                          ensureExpTensExpensive), memo keys (selfIpKey,
│                          selfIpMemoised), canonical keys, and the
│                          call-scope guards (callGuard, dispatchScope,
│                          kernelChunkBytesResolved)
├── +mobius/               Möbius machinery: orbit-table build/load
│   │                      (getOrbitTable, buildOrbitTable, recipeVersion),
│   │                      contraction recipes (buildContractRecipe,
│   │                      executeRecipe), the orbit inner products, the
│   │                      per-attribute matrices (maPerAttrInnerMatrix,
│   │                      relInnerBatched, spectralRelInnerMatrix,
│   │                      closedFormAttrCentres / closedFormAttrMatrixFrom,
│   │                      maRelAttrPrefersCentres), the point evaluators
│   │                      (evalMaOrbit, evalOrbitAbs, evalOrbitRel), and
│   │                      total mass
│   └── _orbit_tables/     Shipped orbit tables (.mat, r = 2..8)
├── demos/                 Demo scripts
├── tools/                 Calibration and benchmark scripts
└── tests/                 MATLAB test suite (test_*.m, bench_*.m, and the
                           mptTestIsolateDefaults harness)
```

MATLAB's lack of a per-file private-namespace convention is partly compensated by the `+package` mechanism, but with constraints: a `+package` member is reached as `package.member` from any caller on the path, so `+internal` is *internal by convention* rather than enforced. The convention is documented in §7.

### Public/private convention

In Python, the public surface is exactly what `mpt/__init__.py` re-exports and lists in `__all__`. Every other module-level name is private – including names without `_`-prefix in modules like `_mobius.py` or `_utils.py`, which are public *within their module* but not on the package surface. The `_`-prefix is reserved for names that are private *within their module* (private to private modules, and intra-module private to public modules).

In MATLAB, the public surface is the set of top-level `.m` filenames. Anything inside `+internal/` or `+mobius/` is by convention not for external use, but enforcement is via convention only.

`_mobius.py` is *internal* from the toolbox's perspective (no `mpt._mobius` name appears in `__all__`), but its contents are organized as if it were a public module – with documented public entry points and the conventional underscore prefix for helpers. This is because `_mobius.py` is large enough (~2,300 lines) that its own internal/external boundary is useful. The MATLAB counterpart `+mobius/` has the same character. `_tensor/_mobius_inner.py` (the per-attribute matrices) and `_tensor/_nested_contraction.py` follow the same pattern.

### Test layout

```
python/tests/              ~110 test_*.py files, flat, organized by feature
                           area (see tests/README.md for the groupings)

python/tests/precision_audit/   Numerical precision and incidence sweeps
                                (19 scripts with .out.txt output snapshots;
                                not part of CI)

python/tools/              Calibration scripts for the cost models
                           (calibrate_rel_ip_cost.py, fit_ma_eval_cost.py,
                           calibrate_nested_cost.py, …) and benches

matlab/tests/              ~90 test_*.m files mirroring the Python tests
                           where applicable, plus bench_*.m cross-language
                           and calibration benches (not CI)

matlab/tools/              Calibration scripts (calibrateRelIpCost.m, …)
```

Tests are predominantly organized by feature rather than by module – e.g. the canonical-key tests cover deduplication across the single-multiset inner product, batched cosine, and harmony wrappers. `python/tests/README.md` documents the file groupings.

---

## 4. The dispatcher pattern

The dispatchers are the toolbox's most distinctive design feature. USER_GUIDE §4 ("Method selection") describes the user-facing API – the `method` keyword, the `wrap` declaration, `truncation_sigmas`, the kernel-evaluation controls, and when to override defaults. This section covers the *internals* that make `method='auto'` work, at architecture level; [ROUTING_MAP.md](ROUTING_MAP.md) gives every rule, guard, and constant in full and is the reference when a detail here is not enough.

Four points are useful to internalize before reading the dispatch code:

1. A dispatcher is *exclusively* about routing the same analytical computation between decomposition methods. It never changes the value computed, only how that value is reached. The one apparent exception is the periodic *measure* (below), which is declared by the user on the density and constrains which routes are admissible; the dispatcher honours the declaration, it does not choose the measure.

2. Every selector applies the same three tests, in order: is a route *feasible* (memory, table availability, structural admissibility); is it *accurate enough* for the accuracy the caller asked for (the σ/P threshold derived from `truncation_sigmas`); and, if more than one route survives, which is *fastest* (a fitted cost model). `explain_dispatch` reports the three for a given call by invoking the very same selectors.

3. There is **no timing probe**. Routing is decided entirely from structure and the cost models; the hardware scale factor cancels in the cost *ratio*, which is why the dispatchers need no timing (the one place a timing is taken – `_timeest.py`'s self-calibration – serves the absolute up-front time estimate, not routing). The former `cancellation_threshold` keyword on `cos_sim_exp_tens` / `cosSimExpTens` has been removed; the alternating-sum cancellation it once guarded is handled by the accuracy floor (§6) and by the post-hoc guards below.

4. Two guards act *after* a route has run. On the cosine Möbius arm, an *impossible value* – a non-finite inner product, a negative self-inner-product, or $|\langle X, Y \rangle| > 1.000001\sqrt{\langle X, X \rangle \langle Y, Y \rangle}$ – triggers a warning, purges the Möbius memo entries on both densities, and re-runs the call through Bulger's method. On the nested contraction, a per-level orbit reduction whose error bound exceeds the accuracy floor falls back to enumeration when the enumeration is affordable and otherwise keeps the value with a warning. On the point-evaluation Möbius route, non-finite output triggers a warning and a re-run through the centres branch, in both languages and for every density shape (the single-multiset corner inherits the guard from the general path). All post-hoc guards are disabled by the `post_hoc_guards` default.

### The measure rule

Periodic attributes admit two readings of the Gaussian kernel: the *minimum-image* (single-image) reading, which wraps each difference to its nearest image, and the *all-image* (full-image) reading, which sums the kernel over every periodic image (the wrapped Gaussian $\theta$ of `_wrapped_kernel.py` / `internal.wrappedGaussian1d`, computed as an image sum or a Poisson-summed Fourier series, whichever converges faster). The two coincide numerically when $\sigma \ll P$ and diverge as $\sigma$ approaches $P$. The user declares the reading per attribute with `wrap` (`'full-image'`, the default, or `'single-image'`).

On absolute-periodic attributes every leaf branches on `wrap` directly. On relative-periodic attributes the reading is fixed by the *route*: Bulger's method and the tuple-centres closed form compute the pairwise-wrapped (minimum-image) quadratic form; the translation grid, the spectral Gram matrix, and the nested $\tau$-grid compute the all-image average. Below the admissibility threshold the two readings agree to within the accuracy floor and the selectors are free to race the routes on cost; above it the declaration decides the route – `'single-image'` forces the Bulger or centres arm, `'full-image'` forces the Möbius or contraction route – and a density that mixes the two declarations across relative attributes is rejected.

The threshold is `_orbit_sigma_over_p_threshold` / `internal.relPerSigmaOverPThreshold`. It is not a constant: a calibration table of the departure between the two readings against σ/P is consulted and the largest σ/P whose departure sits inside the floor implied by the caller's `truncation_sigmas` is taken (0.03 at the factory default of 6), capped by a fixed positive-definiteness ceiling of 0.05 that binds at `truncation_sigmas ≤ 4`. The threshold therefore tightens as the caller asks for more accuracy, where a single constant could only be right at one setting.

### The dispatchers

The same pattern recurs in five places; the flat cosine selector is the canonical instance.

**The flat selector** (`_select_ma_inner_product_method` / `internal.selectMaInnerProductMethod`) chooses among the Bulger, centres, and Möbius arms for a density pair without nested attributes. Its rules fire in order: (1) a user `method` other than `'auto'` is returned unchanged, bypassing everything below; (2) if every $r_a \leq 1$, Bulger; (3) if any $r_a$ exceeds the shipped-table ceiling of 8, Bulger, after a feasibility guard that raises `SingleImageInfeasibleError` / `mpt:dispatch:singleImageInfeasible` when the joint tuple-pair kernel would exceed the memory budget (4 GiB fixed in Python; half of available memory clamped to 1–4 GiB in MATLAB); (4) Python only: a working-set guard sends a large flat density to Möbius when its tuple-centres working set would exceed 256 MiB; (5) the measure rule above; (6) the cost race – the predicted Bulger cost from the fitted law against the predicted Möbius cost, which itself takes, per relative attribute, the cheaper of the tuple-centres and grid realizations, floored by a per-order set-up cost. Ties go to Bulger. After the selector, an *ordered* attribute (`is_sym == false`, $r_a > 1$) on either side silently overrides the answer to Bulger, including an explicit `'mobius'` or `'centres'`.

**The Möbius arm** (`_cos_sim_exp_tens_ma_orbit` / `localCosSimMAOrbit`) decides, per relative attribute, between the tuple-centres closed form and the grid (`_ma_rel_attr_prefers_centres` / `mobius.maRelAttrPrefersCentres`): the closed form is inadmissible above the σ/P threshold (it carries the minimum-image reading) and when either side has fewer values than $r_a$; otherwise a small wall-time model races the two, an explicit `method='mobius'` pins the grid, and the calibration lever `rel_attr_route` pins either. Inside the grid branch a *spectral gate* substitutes the Fourier Gram matrix for $2 \leq r \leq 4$ when its mode grid is at most $4 \times 10^6$ points (a memory guard that is never bypassed) and cheaper than the grid by a cost gate (bypassed by `SPECTRAL_IP_FORCE` for testing). A *sparse gate* switches the absolute contraction and the periodic grid to sparse kernels when the kernel is large ($K_x K_y \geq 200{,}000$) and at most 20 % dense.

**The nested plan** (`_try_nested_contract` / `internal.nestedContract`, priced by `_nested_cost.py` / `internal.nestedCost`) handles densities with a nested attribute. Per attribute it lists the admissible routes under the measure rule – `contract` for absolute attributes; `centres` and `contract_relnonper` for relative non-periodic; for relative-periodic, `centres` and `taugrid` below the threshold and, above it, whichever the `wrap` declaration admits – and picks by a fitted law with a 256 MiB memory guard on the centres route. It then races the whole plan against joint-tuple enumeration (Bulger), choosing enumeration only when it is predicted cheaper by a safety factor of 2 and is itself admissible. A forced `method` raises when the route it names has no carrier rather than silently substituting. Within a contraction, each symmetric level re-decides orbit-vs-enumeration with the power-law `orbit_cost_model` (`_orbit_cost.py` / `internal.orbitCostModel`), whose intercept is the `orbit_cost_intercept` default.

**The eval selector** (`_select_ma_eval` / `internal.selectMaEval`) chooses between the centres branch and the Möbius evaluator for `eval_exp_tens`: `'centres'` returns at once; `'mobius'` returns after rejecting ordered attributes (on a nested density it runs the per-level Möbius evaluator, `_nested_mobius_eval.py` / `mobius.evalNestedAttrOrbit`, which applies the set-partition identity at every symmetric level of the tag tree and a dynamic programme at every ordered one, integrating a co-transposition unit over its own translation grid inside the recursion; it touches no tuple centre); under `'auto'`, ordered or all-$r_a \leq 1$ densities go to centres, and a nested density is decided by its own cost row (`_nested_eval_costs_ms` / `internal.nestedEvalCostsMs`: the tag-tree centres enumeration, counted by `nested_tuple_count` / `internal.nestedTupleCount`, against the per-level evaluator, fitted by `tools/fit_nested_eval_cost.py` on the `bench_nested_eval` grid; a mixed density adds that row to the flat law); $r_a > 10$ goes to centres after a feasibility guard on the joint working set; the measure rule on the first relative-periodic attribute above the threshold forces the arm; otherwise the cost model `_ma_eval_costs_ms` / `internal.maEvalCostsMs` prices both, and Möbius wins when it is predicted cheaper by a safety factor of 1.5 (when the centres working set exceeds 256 MiB) or 1.0. Inside the centres branch the shape rules pick the single-multiset kernel-sum leaf, the factored per-attribute form (every $r_a \geq 2$, no kernel covariance), or the joint materialization.

**The sweep chooser** (`_choose_sweep_route` in `sweep.py` / `sweepCosSimExpTens`) decides between the mixture and orbit routes on eligibility and a work-ratio rule; it is the one selector whose routes have different admissibility sets rather than different costs alone.

### Cost models

Every "fastest" decision reads a fitted cost model rather than an operation count. The models share one shape: for each route and coarse structure key (tuple order for the flat model, total tuple order for the nested model) a power law $t_{\mathrm{ms}} = e^{a} \cdot \mathrm{term}^{b}$ in the quantity the route actually works over – tuple-pair entries for Bulger and the tuple-centres route, node count times value count for the grid – with the exponents fitted rather than pinned at their structural values, because they absorb amortization that an explicit overhead term does not capture. Around the laws sit floors (a per-order set-up cost the Möbius route cannot go under, applied with `max`, which matters only at $r = 2$), memory guards (the 4 GiB feasibility budget, the 256 MiB working-set soft budget), and safety factors (2.0 on the nested enumeration race, 1.5 on the eval race when memory is tight).

The constants are per-language by design: the two implementations amortize differently, so each carries its own fitted row (`_REL_COST_LAW` / `internal.relRouteCostMs`, `_NESTED_COST_LAW` / `internal.nestedCost`, the eval constants in `dispatch.py` / `internal.maEvalCostsMs`), while the rule structure that consumes them is identical. The rows were fitted on the maintainer's machine from timed sweeps (the flat law on 666 cells across orders, value counts, event counts, widths, periodicities, and weight profiles, each route timed in isolation) and cross-validated on the *routing decision* – held-out routing regret over random halves – rather than on absolute time, so that they generalize to other hardware: the hardware factor scales every route alike and cancels in the comparison. The calibration scripts under `python/tools/` and `matlab/tools/` regenerate them, and their header comments record the validation figures and the earlier fits that scored well and shipped badly because an axis was missing from the sweep.

### Self-inner-product memoisation

Both self inner products are memoised on their densities (`_self_ip_cache` / the MATLAB `selfIP` map), keyed per route and per truncation width (and, on the Bulger and centres routes, per kernel precision) so that a route never reads another route's numbers. The *pricing* flag the selectors consult (`_self_ip_memoised` / `internal.selfIpMemoised`) is nonetheless shared across routes, deliberately: pricing each route against its own memo would seed only the winner's memo on the first call and lock the choice in on the second, even where the loser, once warm, is cheaper. Sharing the flag prices the comparison on per-matrix costs, at the price of one mispriced call per crossover. The post-hoc impossible-value guard purges only the Möbius entries it distrusts.

### `explain_dispatch`

`explain_dispatch` / `explainDispatch` is a diagnostic, not a decision: it calls the same selectors and cost models with the same inputs and reports the chosen route, the predicted time for each candidate, the accuracy floor in force, the σ/P limit that follows from it and which test set it, and where the call sits relative to them. On the flat cosine path it builds the selector's inputs with the same helper the call itself uses (`_flat_selector_inputs` / `internal.flatSelectorInputs`: the `wrap` vector, the geometry-derived node counts, the periodicity and symmetry vectors, and the memo flags read from the densities' caches) and applies the empty-operand and ordered-attribute rules first, so the route it names is the route the call takes; a test in each language pins the agreement.

### User-facing knob

The `method` keyword on `cos_sim_exp_tens`, `eval_exp_tens`, `sweep_cos_sim_exp_tens`, and `entropy_exp_tens` (and the matching MATLAB functions) is listed in §2. Most users should leave it at `'auto'`. Hand-overriding is useful for benchmarking or for tests that need a specific method to exercise specific code paths; a forced method bypasses the cost race but never the structural guards, and on the cosine path it remains subject to the ordered-attribute override and the post-hoc guard. The density-list and batched forms forward `method`, `truncation_sigmas`, and `kernel_precision` to every entry in both languages.

---

## 5. The orbit-table system

The Möbius method's per-call cost in the inner product is dominated by $|\Omega_r|$ tensor contractions. For these contractions to run in microseconds rather than milliseconds, two pieces of precomputation are required, both encoded in the *orbit table*.

### What an orbit is

Möbius inversion on the partition lattice rewrites the distinct-index $r$-tuple sum as an alternating sum over set partitions of the slot indices, with each partition's term factorizing across blocks. For the inner product specifically, the slot-permutation symmetry $S_r$ acts jointly on both sides; many partition pairs are equivalent under this action.

Concretely, a partition pair $(\pi_A, \pi_B)$ is identified by:

- $m_A$: the integer partition giving the block sizes of $\pi_A$.
- $m_B$: the integer partition giving the block sizes of $\pi_B$.
- $M$: a contingency matrix with row sums $m_A$ and column sums $m_B$, encoding how the blocks of $\pi_A$ and $\pi_B$ overlap.

The pair $(\pi_A, \pi_B)$ and its image under the $S_r$ action share the same $(m_A, m_B, M)$ triple – up to within-size-group row and column permutations of $M$. Each equivalence class is an *orbit*; the orbit set is $\Omega_r$.

The unsymmetrized partition-pair count is $B_r^2$ where $B_r$ is the Bell number; the orbit count $|\Omega_r|$ is much smaller. Concretely:

| $r$ | $B_r$ | $B_r^2$ | $|\Omega_r|$ |
|---:|---:|---:|---:|
| 2 | 2 | 4 | 4 |
| 3 | 5 | 25 | 10 |
| 4 | 15 | 225 | 33 |
| 5 | 52 | 2,704 | 92 |
| 6 | 203 | 41,209 | 306 |
| 7 | 877 | 769,129 | 948 |
| 8 | 4,140 | 17,139,600 | 3,210 |

The reduction grows rapidly with $r$.

### What the orbit table stores

For each orbit (`OrbitEntry` in Python; a struct-array element in MATLAB), the orbit table stores:

1. **The orbit identifier** – the canonical block sizes $m_A$, $m_B$ (with their block counts) and the non-zero entries of $M$ as `(row, column, multiplicity)` edge triples.

2. **The orbit's weight** – the number of partition pairs in this orbit's equivalence class; the weights sum to $B_r^2$.

3. **The orbit's Möbius coefficient** $\mu(\hat 0, \pi_A)\,\mu(\hat 0, \pi_B)$, shared across the orbit because both factors depend only on block sizes.

4. **A contraction recipe**. In Python, three pre-built `einsum` subscript strings – one each for the single-pair inner product, the gridded (translation-node) form, and the batched per-event-pair form – together with **precomputed contraction paths** (`np.einsum_path(..., optimize='greedy')` run once at table-build time, so runtime calls bypass path-finding). In MATLAB, three step-wise recipes (`recipeIP`, `recipeGrid`, `recipeBatched`) built by `mobius.buildContractRecipe` and executed by `mobius.executeRecipe`; precomputing them removed the runtime set-operation overhead in the pair-picker that had once made the MATLAB orbit path many times slower than Python's.

Both loaders repair a table that predates the current layout on first load: Python's `_ensure_paths` computes missing subscript strings and paths, and MATLAB's `ensureRecipes` rebuilds any recipe whose embedded `version` is below `mobius.recipeVersion()` (currently 2: a two-phase pair picker that bounds every intermediate to non-free rank 2 for all orbits up to $r = 6$). Improvements to the recipe builder therefore take effect without regenerating the shipped files; the version must be incremented when the ordering algorithm or the recipe layout changes.

### Build, cache, and discipline

Orbit tables for $r = 2 \ldots 8$ ship pre-built in the toolbox:

- Python: `python/mpt/_orbit_tables/orbit_r{N}.pkl` (pickle format; ~2.3 MB across the seven tables, 1.7 MB of it at $r = 8$).
- MATLAB: `matlab/+mobius/_orbit_tables/orbit_r{N}.mat` (`-v7` `.mat` format; ~4.6 MB).

`get_orbit_table` / `mobius.getOrbitTable` looks a table up in order: the in-memory cache, the shipped tables, the per-user disk cache, and finally a fresh build. Beyond $r = 8$:

- Default cache location: `~/.mpt/orbit_tables/` (overridable via the `MPT_CACHE_DIR` environment variable). The two languages share the directory but not the files (`.pkl` vs `.mat`); they do not interoperate. A freshly built table is written to the cache only for $r \geq 5$, and only best-effort.
- Build cost scales with $B_r^2$. The estimates embedded in the code put the shipped $r = 8$ table at a few minutes and $r = 9$ at an hour or more in Python (MATLAB's extrapolation is more optimistic at about eight minutes); $r \geq 10$ runs to days.
- Each build beyond the shipped range is preceded by a cost preview on stderr giving $B_r$ and the estimated build time. `MPT_NO_BUILD_WARN=1` suppresses it (intended for automation contexts where the warning is noise).
- The hard cap is $r = 12$; beyond that the build cost is prohibitive even for one-off use, and the toolbox refuses to attempt it. Only the inner-product routes (layer 2, below) read orbit tables; the flat cosine selector never routes to Möbius above $r = 8$ on its own, and the nested contraction uses the orbit reduction only for $2 \leq r \leq 8$ per level (§4), so a build beyond the shipped range happens only under a user-forced `'mobius'` at $9 \leq r \leq 12$. The Möbius point evaluator and total mass use the set-partition lists (layer 1) and need no orbit table, which is why the eval selector's own ceiling is $r = 10$.

The pickle format is chosen for speed and structural fidelity (each `OrbitEntry` holds tuples and numpy-compatible paths). If a future Python upgrade breaks the shipped pickles, the build path is available to regenerate them, and the loader's augmentation step means an older table layout is repaired rather than rejected.

### The two-layer reformulation

A reader landing in `_mobius.py` cold should know that the file implements *two* layers of mathematical reformulation, which are sometimes conflated:

1. **Möbius inversion on the partition lattice.** A sum over distinct ordered $r$-tuples is rewritten as an alternating sum over set partitions, with each partition's term factorizing across blocks. This step alone reduces a $K!/(K-r)!$ sum to $B_r$ partition contributions, each of which is a product of single-source-sum quantities.

2. **Orbit collapse under joint slot symmetry.** This applies only to the inner product (which has two sides and so a joint $S_r$ action). The $B_r^2$ partition pairs collapse to $|\Omega_r|$ orbit classes; each orbit is computed once with its weight.

Layer 1 alone is enough for point evaluation and total mass (which have only one side). Layer 2 is the inner-product-specific further reduction.

This distinction matters for development: the eval-side Möbius code (`eval_orbit_abs`, `eval_orbit_rel`) is layer-1-only; the inner-product code (`inner_product_orbit`, `inner_product_orbit_grid`, `inner_product_orbit_pw_batched`, `inner_product_orbit_sparse`) is layers 1 + 2.

### Public-in-the-module entry points

`_mobius.py` is internal from the toolbox's perspective but is organized as a self-contained sub-package. The public-in-the-module entry points (called from `_tensor/` and consumer wrappers, never directly by user code) are:

| Function | Layer | Purpose |
|:---|:---|:---|
| `get_orbit_table(r)` | – | Load or build the orbit table for tensor order $r$ |
| `inner_product_orbit(K, w_A, w_B, r, ...)` | 1+2 | Distinct-index inner product from one kernel matrix, single pair |
| `inner_product_orbit_grid(K, w_A, w_B, r, ...)` | 1+2 | Inner product over a stack of kernel matrices (translation nodes), shared weights |
| `inner_product_orbit_pw_batched(...)` | 1+2 | Batched inner product over a stack of kernel matrices with per-batch weights (event pairs) |
| `inner_product_orbit_sparse(K_sp, w_A, w_B, r, ...)` | 1+2 | Sparse-kernel variant for large, sparse kernels |
| `eval_orbit_abs(p, w, X, sigma, r, ...)` | 1 | Point evaluation in absolute mode (per-block helper reduction or direct broadcast) |
| `eval_orbit_rel(p, w, X, sigma, r, ...)` | 1 | Point evaluation in relative mode (u-grid; direct, factored, or Fourier strategy) |
| `total_mass_abs(p, w, sigma, r)` | 1 | Total-mass scalar in absolute mode |
| `total_mass_rel(p, w, sigma, r)` | 1 | Total-mass scalar in relative mode |

The per-attribute matrices that the cosine path consumes (`_ma_per_attr_inner_matrix`, `_rel_inner_batched`, `_spectral_rel_inner_matrix`, `_closed_form_attr_matrix_from`) live in `_tensor/_mobius_inner.py` and call these entry points. The MATLAB twin `+mobius` package has matching entry points (`mobius.getOrbitTable`, `mobius.innerProductOrbit`, `mobius.maPerAttrInnerMatrix`, etc.); semantics match.

---

## 6. Numerical guarantees

The toolbox's numerical commitments fall into four categories.

### The accuracy floor and `truncation_sigmas`

Every kernel-based computation is governed by one knob, `truncation_sigmas` (factory default 6; per-call keyword or toolbox default). A kernel centred more than `truncation_sigmas · σ` from an evaluation point is dropped; at that radius the kernel has decayed to $\exp(-k^2/2)$ of its peak, which is therefore the scale of the relative error a truncated sum can incur ($1.5 \times 10^{-8}$ at the default). The value `inf` does *not* mean exhaustive summation into the denormal tail: it resolves to the finite width (≈ 7.43 σ) at which the kernel reaches the toolbox's **accuracy floor** of $10^{-12}$ (`_ACCURACY_FLOOR_EPS` / `internal.accuracyFloor`), the tightest meaningful accuracy the toolbox targets, below which the Möbius decomposition's own cancellation error would dominate a true value already far below the floor. `truncation_floor(ts)` returns the tolerance that follows from a width – $\exp(-k^2/2)$, or exactly $10^{-12}$ at the floor width – and is the single tolerance every truncation path reads.

The governing principle is that *work never exceeds what the requested precision needs*. The same width sets the kernel cutoff of the centres and Bulger routes, the image count $L$ and mode count of the wrapped Gaussian, the σ/P admissibility threshold (§4), the grid-node density of the translation grid (`resolve_samples_per_sigma`), the quadrature tolerance of the nested contraction, the accuracy guard on per-level orbit reductions, and the convergence tolerance of differential entropy. Asking for a tighter width buys more images, more nodes, and a stricter threshold; nothing is computed beyond that. A few places still read the *global* default rather than the per-call width (the τ-node count `auto_ntau_default`, the selector's node-count margin, the centres-vs-grid estimate, the Rényi-2 path, and in Python the whole nested path); ROUTING_MAP §1.10 and §10 list them.

`kernel_precision` (`'double'` default, or `'single'`) casts the hot-loop arrays of the kernel-sum helper to single precision for speed. It is honoured on the centres-path routes that go through `gaussian_kernel_sum` / `internal.gaussianKernelSum` and on the joint evaluation route; which secondary forms forward it differs between the languages (ROUTING_MAP A-9).

### Post-hoc guards

Described in §4: the impossible-value guard on the cosine Möbius arm, the accuracy guard on the nested per-level orbit reduction, and the non-finite guard on Möbius point evaluation. The impossible-value check is exact (it inspects the computed inner products, not an estimate); the fallback re-runs through Bulger's method without reusing any Möbius intermediate, and purges the Möbius memo entries so that a later call does not read the distrusted numbers. Incidence is measured by the `precision_audit` sweeps rather than asserted here.

### σ → 0 discipline

At very small σ relative to the period $P$, the Gaussian kernel approaches a delta function and several toolbox computations are limit cases. The toolbox handles these limits structurally rather than numerically:

- Small σ/P is the *admissible* side of the measure threshold, so the selectors are free to route by cost; the wrapped Gaussian's image count $L$ is 0 there and the single-image and full-image readings coincide exactly.

- The kernel-sum helper's truncation makes entries outside the kernel's support exactly zero, not exponentially small floats that propagate noise.

- Discrete-limit measures (`coherence`, `sameness`) have explicit σ = 0 branches that compute the integer-position limit directly rather than relying on the kernel evaluation to converge; Rényi-2 entropy rejects σ = 0 outright.

### Cross-language parity

The cross-language golden tests (`test_cross_language_golden.py` and `.m`) hard-code the outputs of a fixed set of deterministic cases – single-multiset and multi-attribute cosine similarity on the Möbius method, Rényi-2 entropy, orbit-path `tensor_harmonicity`, orbit-path `eval_exp_tens`, and Shannon entropy – and require both languages to reproduce them to $10^{-8}$ relative ($10^{-12}$ absolute), the standard tolerance used throughout the v3 suite for orbit-vs-pairwise agreement on shared regimes. The cosine cases force `method='mobius'` so the Möbius machinery is genuinely exercised rather than the cost model's fallback. Either language drifting fails its own suite.

Beyond the goldens, the parity commitment is that the two languages apply the *same rules and routes* to the same input (§7); the cases where they currently do not are enumerated with evidence in ROUTING_MAP §10.

---

## 7. Twin-language conventions

The MATLAB and Python implementations are intentionally parallel. USER_GUIDE §4 ("API conventions") covers the *user-facing* mapping (function-name mapping table, weight-argument convention, query-point convention, etc.). This section adds the *developer-facing* rules for keeping the two sides aligned – what to mirror, what is allowed to diverge, and how the test parity discipline works.

### The parity principle

Parity is defined at the level of *decisions*, not of constants. Every selector, guard, override, and post-hoc check exists in both languages with the same predicate and fires in the same order; every route computes the same measure; and the leaf that finally does the arithmetic may differ in implementation (a vectorized kernel here, a log-kernel product there) provided the numbers agree within the accuracy floor. The fitted constants of the cost models are *per-language by design* – the two implementations amortize differently – but they are measured the same way, on the maintainer's machine by twin calibration scripts, and cross-validated on held-out routing regret so that they generalize to other hardware. A parity audit therefore compares rule structure first and constants last; ROUTING_MAP.md and `routing_parity.md` are the record of the most recent audit.

### Naming

| Style | MATLAB | Python |
|:---|:---|:---|
| Function names | `cosSimExpTens` (camelCase) | `cos_sim_exp_tens` (snake_case) |
| Density tag / class | struct with `tag == 'MaetDensity'` | `MaetDensity` (CapWords) |
| Constants in user code | `PERIOD = 1200` (UPPER_SNAKE) | `PERIOD = 1200` (UPPER_SNAKE) |
| Private helpers | `+internal/helperName.m` | `_helper_name` |
| Package internals | `+mobius/canonicalForm.m` | `_mobius.canonical_form` |
| Local functions of an entry point | `localCosSimMA` inside `cosSimExpTens.m` | `_cos_sim_exp_tens_ma` in `_tensor/cosine.py` |

The function-name mapping is the most consequential. Every public Python function `mpt.foo_bar` should have a MATLAB sibling `fooBar` with semantically identical inputs, outputs, and side effects (after applying the absence-sentinel and container-type conventions below). The known exceptions are the circular measures `balance` / `evenness`, whose MATLAB files are `balanceCircular.m` / `evennessCircular.m`, and `TranslatedSweep`, which has no MATLAB twin.

### Absence sentinels

| Concept | MATLAB | Python |
|:---|:---|:---|
| Default weights | `[]` | `None` |
| Default period | omitted | omitted |
| Unset name-value | not passed | not in kwargs |

`None`/`[]` round-trip cleanly through any wrapper that respects the convention. Code that explicitly checks `if w is None` (Python) should have an `if isempty(w)` (MATLAB) counterpart; vice versa.

### Container types

| Concept | MATLAB | Python |
|:---|:---|:---|
| Sequence of densities | cell array `{d1, d2, ...}` | list `[d1, d2, ...]` |
| Per-attribute matrices (MAET) | cell array of matrices | list of arrays |
| Per-row spectrum kwargs | cell array | list/tuple |
| Density | struct with `.tag == 'MaetDensity'` | `MaetDensity` instance |
| Single-multiset view | struct with `.tag == 'SingleMultisetView'` | `_SingleMultisetView` instance |
| Self-IP memo | `selfIP` field keyed by `'route|ts|extra'` strings; the updated structs are returned as extra outputs (`[s, densXOut, densYOut] = cosSimExpTens(...)`) because structs have value semantics | `_self_ip_cache` dict keyed by `(route, ts, kp, extra)` tuples, updated in place |

MATLAB structs and Python instances are used interchangeably for the density data structure; field names match up to case convention (`n_attrs` ↔ `nAttrs`, `p_attr` ↔ `pAttr`, `is_rel` ↔ `isRel`). This is non-idiomatic Python (a real Python implementation would use `__slots__` and properties throughout), but is the path of least resistance for twin-language parity. The one structural difference follows from value semantics: Python densities materialize lazy fields in place, whereas MATLAB consumers must call `internal.ensureExpTensExpensive` and keep the returned struct.

### Weight broadcast

Both languages follow the same broadcast convention for weights:

- Scalar weight: broadcast to all events.
- `[]` (MATLAB) / `None` (Python): broadcast to all-ones.
- Per-event vector (length $N$): one weight per source event.
- Per-slot vector (length $K_a$ for MA inputs): one weight per slot, broadcast across events.
- Full per-event-per-slot matrix ($K_a \times N$): explicit per-slot per-event weights.

The same rules apply to MAET inputs at the per-attribute level. The full mapping is documented in USER_GUIDE §4 ("Weight arguments").

### Call-scope guards

Both languages wrap every user-facing entry point in a scope guard (`_dispatch_scope` / `_with_dispatch_scope` in `_defaults.py`; `internal.callGuard`, combining `internal.dispatchScope` and the kernel-chunk-bytes pin, in MATLAB). On the outermost entry the guard resets the once-per-call throttle on dispatch messages and pins the `kernel_chunk_bytes` resolution (`'auto'` = half of currently available physical memory) so that recursive inner calls neither repeat announcements nor re-query the operating system. New entry points must take the guard as their first executable line.

### Twin-language test parity

Every Python `test_*.py` should have a MATLAB `test_*.m` counterpart with the same name suffix. Where the counterpart does not exist, the gap is either:

1. Deliberate: the feature is Python-only or MATLAB-only (rare; should be documented).
2. A known gap: the test exists but is not yet ported.

The cross-language golden tests are the ultimate parity check: any value that drifts between MATLAB and Python on the golden cases fails the suite on both sides.

### When languages should diverge

Twin-language parity is the default but not a rule. Acceptable divergences:

- Plotting features that one language has and the other does not (e.g., `audioPeaks.m`'s `'plot'` option, which has no Python equivalent because `matplotlib` would be a heavy optional dependency for one convenience feature).
- Language-idiomatic conveniences that do not change semantics (e.g., Python dataclass defaults, MATLAB `arguments` block validation, the Python `TranslatedSweep` reduction and per-density caches, which are speed-only).
- Performance optimizations that exploit per-language strengths (e.g., Python's single-attribute helper route in the inner-product core versus MATLAB's log-kernel form). The output must agree within the accuracy floor.
- The fitted constants of the cost models (above).

Divergences that *do* change semantics or output values are never acceptable without explicit documentation; the ones that currently exist are recorded in ROUTING_MAP §10 with the side whose behaviour the other should adopt.

---

## 8. How to add a new measure

A walk-through for adding a new measure to the toolbox. The example: imagine adding a hypothetical `tonality_index` – a single scalar describing how strongly a pitch multiset suggests a tonal centre.

### 1. Decide which tier the measure sits at

- *Tier 1 (density)*: no. A `tonality_index` is computed from a density, not a different density structure.
- *Tier 2 (primitive)*: no. It does not introduce a new operation on densities – it composes existing primitives.
- *Tier 3 (consumer wrapper)*: yes. Composes `cos_sim_exp_tens` with a reference template.

### 2. Decide which file it lives in

In Python: `harmony.py` if the measure is harmony-flavoured; a new module otherwise. In MATLAB: a new top-level `.m` file. Match the categorization USER_GUIDE §6 uses.

### 3. Decide which existing primitives it consumes

For `tonality_index`: presumably it computes `cos_sim_exp_tens` against one or more reference tonal-centre templates and returns a scalar derived from the resulting similarity values. It should consume `cos_sim_exp_tens` rather than implementing its own inner product, and it should pass `method`, `truncation_sigmas`, and `kernel_precision` through rather than dropping them.

### 4. Single-multiset corner only, or general multi-attribute input?

There is one density type, so the question is not which *path* to write but which *shapes* the measure accepts. Most consumer wrappers are introduced for single-multiset input (one attribute, one event: a chord, a rhythm) and reach the primitives through `build_exp_tens` on a 1-D array, which returns a `MaetDensity` at the `A = N = 1` corner. Write the measure against that density; do not special-case the corner yourself – the primitives do so where it buys speed, and the numbers are the same either way. If the measure has a natural interpretation on general multi-attribute input, accept the list-of-attribute-matrices form as well and let the same primitives handle it. If it does not, validate the shape at entry and raise `ValueError` (Python) / `error('tonalityIndex:singleMultisetOnly', ...)` (MATLAB) for input that is not a single multiset, so that a caller learns the restriction from the message rather than from a wrong number.

### 5. Decide whether canonical-form dedup applies

Canonical-form dedup is for batched evaluations where structurally-identical chords map to the same output. For `tonality_index` operating on chord pitches:

- If the measure depends only on intervals (relative), then transpositions are equivalent and dedup by interval canonical form is appropriate.
- If the measure depends on absolute pitch class, transpositions are distinguishable and dedup is by chord canonical form.

The existing dedup helpers (`_chord_canonical_key`, `_pair_canonical_key` in `_tensor/canonical.py`; `internal.chordCanonicalKey`, `internal.pairCanonicalKey`) are designed to be reused. New measures that need a different dedup key should add a new key function in `canonical.py` and document its semantics.

### 6. Add to public API surface

Python: edit `mpt/__init__.py` to re-export `tonality_index` and add it to `__all__`. MATLAB: place `tonalityIndex.m` at `matlab/`'s top level (no other action needed – MATLAB's path makes it public) and take `internal.callGuard()` as its first executable line.

### 7. Mirror in the other language

Write the MATLAB and Python implementations side-by-side. The function signatures should be mechanical translations of each other (snake_case ↔ camelCase, `None` ↔ `[]`, list ↔ cell). The internal logic should also match step-for-step where reasonable.

### 8. Add tests in both languages

- A `test_tonality_index.py` and `test_tonality_index.m` with parallel test cases.
- Add at least one entry to the cross-language golden test corpus exercising the new measure on a known input/output pair.

### 9. Document

- Per-function docstring: full NumPy-doc style in Python, full H1 style in MATLAB. Document every parameter, return value, and side effect.
- USER_GUIDE entry: add to the appropriate section in §6 (function reference).

### 10. Update CHANGELOG

Add a `### Added` entry to the unreleased section of `CHANGELOG.md`. If the new measure changes any existing behaviour, also add a `### Changed` entry and an entry in `MIGRATION.md` for the next release.

### 11. Demo

If the measure has obvious teaching value, add a demo to `matlab/demos/` (and a Python counterpart) following the naming conventions (`demo_tonalityIndex.m` / `demo_tonality_index.py`). Include user-adjustable parameters at the top and prose comments oriented to learners.

### 12. Architecture and routing documents

If the new measure adds new structural patterns (e.g., a new selector, a new caching system, a new route through the primitives), update this document *and* ROUTING_MAP.md. Routine additions (consumer wrappers that compose existing primitives) need neither.

---

## 9. Release discipline

### Orbit tables

The shipped orbit tables (`_orbit_tables/orbit_r{N}.pkl` for Python, `.mat` for MATLAB) are versioned with the toolbox release. The pickle/`.mat` format is considered stable but not frozen: if a future Python/numpy version breaks the shipped pickle, the build path (`_build_orbit_table` / `mobius.buildAndSavePrebuiltTables`) regenerates them in place. The Python package requires Python ≥ 3.10 and numpy ≥ 1.24; the shipped tables are expected to load on any interpreter the package supports.

### Format provenance

Each shipped orbit table is built and validated against the corresponding `_mobius.py` / `+mobius` source at release time. The build process is deterministic; identical source should produce identical tables. A diff in the table file is taken as a release-blocking signal that the source has changed in a way that requires the table to be rebuilt. Changes to the MATLAB contraction-recipe builder do not require rebuilding the shipped files – incrementing `mobius.recipeVersion` makes the loader regenerate recipes on first use – but the shipped files should nonetheless be regenerated at release so that first load is not slowed by the repair.

### Cost-model constants

The fitted cost-model rows (§4) are release artefacts in the same sense as the orbit tables: they are regenerated by the calibration scripts under `python/tools/` and `matlab/tools/`, and each row's header comment records the sweep it was fitted on and its held-out validation figure. A refit must use the same terms the selector passes (the laws are fitted against the quantity the caller passes, not an idealization of it), and must vary every axis the sweep claims to vary; `calibrate_rel_ip_cost.py --check` asserts the latter. Refitting one language does not require refitting the other.

### CHANGELOG conventions

The CHANGELOG follows Keep-a-Changelog conventions:

- One section per release, ordered newest-first.
- Sub-sections: `### Added`, `### Changed`, `### Deprecated`, `### Removed`, `### Fixed`, `### Internal`.
- The unreleased section is dated `[X.Y.Z] – Unreleased` and is appended to throughout development.
- On release, the date is filled in and a new unreleased section is opened.

### MIGRATION conventions

`MIGRATION.md` has one section per `vN.X → vN.X+1` boundary. Each section documents:

- Numerical changes at default settings (rare; high bar).
- New keyword arguments or kwargs with changed defaults.
- Deprecation paths for renamed or replaced functions.
- One-paragraph migration example for each non-trivial change.

### Deprecation policy

Deprecations follow a two-release cycle:

1. *Release N*: deprecate the function/argument. Emit a `DeprecationWarning` (Python) or `<funcName>:deprecated` warning (MATLAB) on every call. Keep the function working unchanged otherwise. Document the replacement in the deprecation message.

2. *Release N+1 or later*: remove the deprecated function. The CHANGELOG `### Removed` section documents the removal.

Active examples in v3:

- `batch_cos_sim_exp_tens` (Python) / `batchCosSimExpTens` (MATLAB): deprecated as of v3, still functional as thin shims forwarding to `cos_sim_exp_tens` batched-raw mode.
- `cos_sim_exp_tens_raw` and `eval_exp_tens_raw` (Python): deprecated as of v3, still functional as thin shims forwarding to the unified entry points.

These shims will be removed in a future release; the timing is left open.

### Release artefact list

A release includes:

- Source code at the tagged commit on GitHub.
- A Zenodo deposit minted from the GitHub release, with the concept DOI preserved across versioned deposits.
- Updated CHANGELOG and MIGRATION docs.
- Updated CITATION.cff with the new version metadata.
- USER_GUIDE updated for new features.
- ARCHITECTURE and ROUTING_MAP updated if new structural patterns or routes are introduced.
