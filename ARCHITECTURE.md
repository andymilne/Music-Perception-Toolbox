# Music Perception Toolbox --- Architecture

A developer-facing map of what's in the toolbox, how the pieces relate, and the design rationale for the bits that aren't obvious from a casual reading of the source. For *user-facing* documentation --- what the functions do and how to call them --- see [USER_GUIDE.md](USER_GUIDE.md). This document assumes the reader has either read USER_GUIDE §3 or is comfortable with the expectation-tensor framework from the source papers (Milne et al. 2011, 2015, 2016, 2020).

This document describes the toolbox as it currently exists.

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

The Music Perception Toolbox is implemented in parallel in MATLAB and Python with the same public function surface, the same semantics at the floating-point level for default routing in standard regimes, and the same internal architecture except where language conventions diverge. This twin-language parity is the toolbox's central organising commitment --- a refactor that would force the two sides apart is treated as a serious cost.

At the broadest level the toolbox stacks three computational tiers:

```
                      ┌────────────────────────────────────┐
                      │  Consumer wrappers                 │
                      │  (harmonicity, entropy, circular   │
                      │  measures, sequential utilities,   │
                      │  spectral enrichment, …)           │
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
                      │  (ExpTensDensity, MaetDensity,     │
                      │  WindowedMaetDensity)              │
                      └────────────────────────────────────┘
```

Read top-down for *what calls what*; bottom-up for *what gets called by whom*. The density layer is the data structure shared by everything above it; the primitives layer is where the dispatcher lives; the consumer layer is the published function set in §6 of USER_GUIDE.

Cross-cutting concerns --- the Gaussian-kernel sum helper, the Möbius package, the toolbox-wide defaults system, the canonical-form deduplication machinery --- sit alongside this stack and are consumed at multiple tiers. Their organisation is covered in §3.

The MATLAB and Python implementations share this layering. They diverge in mechanical translation only: function names (camelCase vs snake_case), absence sentinel (`[]` vs `None`), container types (cell array vs list), per-language convenience (MATLAB plotting features that have no Python equivalent without adding matplotlib as a dependency, Python dataclass conveniences with no clean MATLAB analogue). The conventions for keeping the two sides aligned are documented in §7.

---

## 2. Mathematical layering

The three computational tiers each correspond to a distinct mathematical operation.

### Tier 1: Density objects

A weighted multiset $(\mathbf{p}, \mathbf{w})$ of $N$ events at tuple order $r$ defines an *r-ad expectation tensor density* --- a Gaussian-mixture probability density on the $r$-fold product space, with one Gaussian centred at each ordered $r$-tuple of source events. The density object precomputes the tuple indices, weight products, and tuple centres; subsequent operations read from it without revisiting the source multiset.

Single-attribute and multi-attribute densities share the same role but differ in structure:

- `ExpTensDensity`: single attribute (typically pitch or time), tensor order $r$, periodicity flag, relativity flag, period, σ. The density is supported on $\mathbb{R}^r$ (absolute) or its $(r-1)$-dimensional translation quotient (relative).

- `MaetDensity`: multiple attributes with per-attribute $r_a$ and per-group ($\sigma$, isPer, period, isRel). Effective space is the product of per-group effective spaces.

- `WindowedMaetDensity`: a `MaetDensity` paired with a per-group window specification (size and mix). The window is applied lazily at evaluation or inner-product time; the windowed density is *not* materialised at construction.

The build step (`build_exp_tens` / `buildExpTens`) is where the source multiset is consumed and the density is precomputed. After v2.2 the expensive density fields are *lazy* --- they are not materialised until a consumer needs them. The Möbius method's centres-array footprint at high $r$ would dominate memory if eagerly built; deferring lets calls that route through the Möbius method skip the centres array entirely. See [§5](#5-the-orbit-table-system) for why this matters.

### Tier 2: Tensor analysis primitives

Three core analytical quantities operate on density objects:

- **Point evaluation** $T(\mathbf{x})$: the density's value at a query point. Single-density-single-query is the basic operation; batched evaluation (many queries, or many densities) is what most consumers actually need.

- **Inner product** $\langle T_x, T_y \rangle = \int T_x T_y \, \mathrm{d}\mathbf{z}$: the integral of the pointwise product of two densities over their shared support. Cosine similarity normalises this by the L2 norms of both operands; Rényi-2 differential entropy is computed from the self-IP and total mass.

- **Total mass** $Z = \int T \, \mathrm{d}\mathbf{z}$: a scalar normaliser. Used to convert $T$ to a probability density via $T/Z$ and as a denominator in normalised quantities.

All three quantities have *closed-form analytical* expressions for the Gaussian-mixture density --- no grid discretisation is required. The expressions are sums over slot-tuples; the dispatcher described in [§4](#4-the-dispatcher-pattern) chooses how to evaluate those sums.

Two methods for evaluating the sums coexist:

- **Bulger's method** (carried unchanged from v1) is specific to the inner product. It organises the slot-tuple sum by combinations on one side and permutations on the other, exploiting within-tuple multinomial symmetry. Does not extend to point evaluation or total mass (the asymmetric combinations-vs-permutations organisation requires two sides).

- **The Möbius method** (new in v2.2) applies uniformly to all three quantities via Möbius inversion on the partition lattice. For the inner product, this composes with *orbit collapse* under joint slot-permutation symmetry, reducing $B_r^2$ partition-pairs to $|\Omega_r|$ orbit equivalence classes. Detailed in [§5](#5-the-orbit-table-system).

The two methods are alternative decompositions of the same analytical integral; their results agree to floating-point precision in the regimes where both are valid. They are not combined within a single call --- the dispatcher picks one per call. The user-facing `method` knob (`'auto' | 'bulger' | 'mobius' | 'centres' | 'direct'`) controls the choice; defaults to `'auto'`.

### Tier 3: Consumer wrappers

The consumer wrappers compose the tier-2 primitives into measures with musical interpretation:

- **Similarity and complexity**: `cos_sim_exp_tens` and `windowed_similarity` are themselves primitives, but the spectral-enrichment wrapper (`add_spectra` applied before cos-sim, producing spectral pitch-class similarity) is a consumer.

- **Harmonicity and consonance**: `template_harmonicity` cross-correlates a chord's composite spectrum against a harmonic template; `tensor_harmonicity` queries the density of interval patterns within a single harmonic series; `spectral_entropy` computes Shannon entropy of a spectral density; `roughness` (sensory roughness) is a direct frequency-pair calculation independent of the tensor framework; `virtual_pitches` extracts likely fundamentals via template harmonicity.

- **Entropy**: `entropy_exp_tens` evaluates the density's differential entropy (Shannon by grid discretisation, or Rényi-2 by closed-form Möbius); `n_tuple_entropy` is a convenience wrapper composing `difference_events` + `bind_events` + `build_exp_tens` + `entropy_exp_tens` for the integer-step n-gram entropy of Milne & Dean (2016).

- **Circular measures**: `balance`, `evenness`, `coherence`, `sameness`, `edges`, `proj_centroid`, `mean_offset`, `circ_apm`, `markov_s`. Some compose tensor primitives; others are direct DFT-based or symbolic computations independent of the tensor stack.

- **Sequential utilities**: `continuity` (smoothed direction-continuity), `seq_weights` (named time-based decay profiles).

- **Cross-event preprocessing**: `difference_events`, `bind_events` --- transform $(\mathbf{p}, \mathbf{w})$ to $(\mathbf{p}', \mathbf{w}')$ before the tensor stack consumes them, supporting interval-based and n-gram analyses respectively.

- **Utility**: `simplex_vertices` (categorical-attribute encoding), `convert_pitch` (seven-scale conversion), `add_spectra` (spectral enrichment), `audio_peaks` (spectral peak extraction).

The consumer layer is where measure-specific documentation belongs (see USER_GUIDE §6); the layering in this document stops at the tier-2 primitives.

---

## 3. Code layering

### Python module map

```
mpt/
├── __init__.py            Public API surface (re-exports + __all__)
├── tensor.py              Re-export shim over _tensor/ (kept so
│                          existing `from mpt.tensor import X` imports
│                          --- including developer-facing private names
│                          --- continue to work unchanged)
├── _tensor/
│   ├── __init__.py        Re-exports the eight sub-modules' names
│   ├── density.py         ExpTensDensity, MaetDensity, WindowedMaetDensity,
│   │                      and the MA-input preprocessing helpers
│   ├── build.py           build_exp_tens (SA + MA paths)
│   ├── preprocessing.py   difference_events, bind_events, simplex_vertices
│   ├── canonical.py       Canonical-form key helpers for batched dedup
│   ├── dispatch.py        Path-selection cost model + shared dispatch
│   │                      helpers (_normalize_density_input,
│   │                      _resolve_list_list_mode, _compute_Q, probes)
│   ├── eval.py            eval_exp_tens (SA centres / orbit / fast, MA)
│   ├── cosine.py          cos_sim_exp_tens + batch_cos_sim_exp_tens
│   │                      and all inner-product cores
│   └── windowing.py       window_tensor, windowed_similarity, and the
│                          windowed inner-product machinery
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
├── serial.py              continuity, seq_weights
├── spectra.py             add_spectra
├── audio.py               audio_peaks, AudioPeaksDetail
├── convert.py             convert_pitch
├── _kernel.py             Gaussian-kernel sum helper (single centres-path
│                          numerical primitive for the toolbox)
├── _mobius.py             Möbius-Bulger inner product machinery
├── _defaults.py           Toolbox-wide defaults API + dispatch-scope guard
├── _utils.py              Small helpers (estimate_comp_time, validation)
└── _orbit_tables/         Shipped orbit tables (pickle, r = 2..8)
```

### MATLAB layout

```
matlab/
├── *.m                    One file per public function (cosSimExpTens,
│                          evalExpTens, entropyExpTens, …) --- MATLAB
│                          requires one top-level function per file
├── +internal/             Helpers reachable only via internal.helperName,
│                          mirroring Python's _-prefix convention
├── +mobius/               Möbius-Bulger inner product machinery
│   └── _orbit_tables/     Shipped orbit tables (.mat, r = 2..8)
├── audio/                 Example audio files for demos
├── demos/                 Demo scripts
└── tests/                 MATLAB test suite (one test_*.m file per
                           test target, run via the mptTestIsolateDefaults
                           harness)
```

MATLAB's lack of a per-file private-namespace convention is partly compensated by the `+package` mechanism, but with constraints: a `+package` member is reached as `package.member` from any caller on the path, so `+internal` is *internal by convention* rather than enforced. The convention is documented in §7.

### Public/private convention

In Python, the public surface is exactly what `mpt/__init__.py` re-exports and lists in `__all__`. Every other module-level name is private --- including names without `_`-prefix in modules like `_mobius.py` or `_utils.py`, which are public *within their module* but not on the package surface. The `_`-prefix is reserved for names that are private *within their module* (private to private modules, and intra-module private to public modules).

In MATLAB, the public surface is the set of top-level `.m` filenames. Anything inside `+internal/` is by convention not for external use, but enforcement is via convention only.

`_mobius.py` is *internal* from the toolbox's perspective (no `mpt._mobius` name appears in `__all__`), but its contents are organised as if it were a public module --- with documented public entry points and conventional underscore-prefix for helpers. This is because `_mobius.py` is large enough (~1,400 lines) that its own internal/external boundary is useful. The MATLAB counterpart `+mobius/` has the same character.

### Test layout

```
python/tests/              ~58 test_*.py files, organised by feature area
                           (test_dispatcher_*, test_*_dispatcher,
                           test_canonical_keys, test_maet, …)

python/tests/precision_audit/   Numerical precision sweeps (12 scripts +
                                output snapshots; not part of CI)

matlab/tests/              ~42 test_*.m files mirroring the Python tests
                           where applicable

matlab/tests/bench_orbit_xlang.m   Cross-language benchmark (not CI)
```

Tests are predominantly organised by feature rather than by module --- e.g. `test_canonical_keys.py` covers canonical-form deduplication across the SA inner-product, batched cos-sim, and harmony wrappers. A `tests/README.md` documenting the test-file groupings is planned as part of the v2.2 documentation work.

---

## 4. The dispatcher pattern

The dispatcher is the v2.2 release's most distinctive design feature. USER_GUIDE §4 ("Method selection") describes the user-facing API --- the `method` keyword, the cancellation-threshold knob, the kernel-evaluation controls, and when to override defaults. This section covers the *internals* that make `method='auto'` work: the structural pre-screen, the cost model, the timing probe, and the cancellation guard.

Three points are useful to internalise before reading the dispatch code:

1. The dispatcher is *exclusively* about routing the same analytical computation between two (or more) decomposition methods. It never changes the value computed, only how that value is reached.

2. The dispatcher has three lines of defence: (a) a *structural pre-screen* that rejects methods that are mathematically inapplicable or numerically unsafe at the call's parameters; (b) a *cost model* that estimates the relative wall-time of the surviving methods at the call's parameters; (c) a *timing probe* that runs each surviving method on a small subset of the workload and times them.

3. The dispatcher can fall back *after* the call has run: if the chosen method's result shows pathological numerical loss (e.g. Möbius's alternating-sum cancellation crossing a threshold), the result is discarded and the call re-runs through the unaffected method.

### The dispatch flow

For the SA inner product (the canonical case; other dispatchers follow the same shape with different details):

```
                    user call with method='auto'
                              │
                              ▼
            ┌──── structural pre-screen ────┐
            │  K < r + 2?         → Bulger only (Möbius unsafe)
            │  σ/P > 0.03 in PR?  → Bulger only (Möbius unsafe at small σ)
            │  no Möbius table?   → Bulger only
            └────────────────┬──────────────┘
                              │
                              ▼ (both viable)
            ┌──── cost model (pre-screen) ────┐
            │  estimate pairwise cost  (ms)
            │  estimate Möbius cost    (ms)
            │  ratio ≥ 10× either way? → pick winner
            └────────────────┬─────────────────┘
                              │
                              ▼ (within 10×)
            ┌──── timing probe ────┐
            │  run each method on a
            │  small probe subset
            │  pick the faster
            └─────────┬────────────┘
                      │
                      ▼
            ┌──── run on full workload ────┐
            │  with chosen method
            └──────────┬───────────────────┘
                       │
                       ▼ (Möbius only)
            ┌──── cancellation check ────┐
            │  |sum| / max(|term|) below
            │  cancellation_threshold? → fallback to Bulger
            │  Möbius output non-finite? → fallback to centres
            └──────────────────────────────┘
                       │
                       ▼
                    return value
```

### Why two methods exist

The two methods have inverse cost-scaling regimes. Bulger's method has near-zero per-call fixed overhead but its slot-tuple loop scales as $K!/(K-r)!$. The Möbius method has fixed per-call overhead (orbit-table lookup, $|\Omega_r|$ tensor contractions) but the per-orbit contraction cost is independent of $K$.

Crossover happens roughly where the slot-tuple loop becomes comparable to $|\Omega_r|$ contractions. Concretely: at $r = 2$, Bulger's method usually wins because $|\Omega_2| = 4$ overhead beats any $K!/(K-2)!$; at $r = 4$ with $K = 30$, the slot-tuple loop is $657,720$ terms while the Möbius method runs $|\Omega_4| = 33$ tensor contractions on a $30 \times 30$ kernel matrix --- the Möbius method dominates. The cost model in `_predict_pairwise_kernel_size` and `_predict_orbit_cost_ms` estimates these costs analytically using calibrated per-operation timings.

### Structural pre-screen conditions

The pre-screen rejects the Möbius method when its mathematical assumptions are violated or its numerical behaviour is unreliable:

- **$K < r + 2$**: the Möbius method's alternating partition sum has at most $B_r$ terms, but the slot-tuple loop has $K!/(K-r)!$ terms. When the latter is comparable to or smaller than the former, the Möbius method has nothing to optimise and the cancellation guard becomes the dominant cost. Bulger's method handles this regime natively without overhead.

- **σ/P > 0.03 in periodic-relative mode**: at large σ/P the periodic wrapping makes the Gaussian kernel essentially flat on the period, and the Möbius alternating sum becomes a difference of near-equal large quantities --- catastrophic cancellation. The threshold is empirically determined; Bulger's method handles wrapping exactly via the pairwise-wrap form.

- **No Möbius table available**: at $r > 8$ (the shipped-table ceiling) without a user-built cache, Möbius falls back to Bulger's method silently. The hard cap at $r = 12$ is the prohibitive-build-cost limit.

### Cost model

Two cost functions are calibrated against measured timings:

- `_pw_per_entry_ms`: cost per slot-tuple in Bulger's method, multiplied by the slot-tuple count.

- `_predict_orbit_cost_ms`: cost per orbit contraction multiplied by $|\Omega_r|$, with the per-contraction cost depending on the orbit's einsum shape (the orbit table stores precomputed einsum paths to avoid runtime path-finding overhead).

The cost model is calibrated for typical hardware; it errs on the side of conservatism (i.e., requires a 10× margin before bypassing the timing probe). The timing probe is the safety net for any miscalibration.

### Timing probe

When the cost model's verdict is within 10× either way, the dispatcher runs each surviving method on a small probe subset of the workload (e.g. the first 32 candidate pairs in a batched cos-sim) and routes the full workload to the faster method. The probe runs the actual method, not a calibrated estimate, so per-machine variance is absorbed.

Probes are only run when the cost model is genuinely indeterminate. For calls clearly in one regime, the probe is skipped.

### Cancellation guard

The Möbius method's partition-pair sum is alternating: blocks contribute with sign $(-1)^{|\pi| - 1} (|\pi| - 1)!$. When the kernel matrix $K$ has structure that makes the partition contributions near-equal in magnitude but cancelling in sign, the result digit-count is set by `(precision in input) - (cancellation digits)`. The cancellation guard monitors

$$
\rho = \frac{|\sum_\pi \mathrm{sign}(\pi) \cdot \mathrm{contrib}_\pi|}{\max_\pi |\mathrm{contrib}_\pi|}
$$

across orbit classes. A value near 1 indicates no cancellation; a value much smaller than 1 indicates digits of precision lost to catastrophic cancellation. When $\rho$ falls below `cancellation_threshold` (default $10^{-12}$), the call falls back to Bulger's method, which has no comparable cancellation pathology.

The post-hoc non-finite-output check is a separate safety net for any pathology the cancellation ratio misses (e.g. NaN propagating from a structurally bad input).

### The same pattern recurs

The SA inner product is the canonical dispatcher. Three other dispatchers follow the same pattern with different details:

- `_select_sa_eval_method`: centres-array path vs Möbius for point evaluation.
- `_select_ma_inner_product_method`: pairwise vs orbit for the MA inner product.
- `_select_and_estimate_sa` / `_select_and_estimate_sa_ip`: wrapper functions that produce both a chosen method and a cost estimate, the latter used for verbose-mode time-remaining displays.

### User-facing knob

The `method` keyword on `cos_sim_exp_tens`, `eval_exp_tens`, `entropy_exp_tens` (and matching MATLAB functions) accepts `'auto'` (default), `'bulger'`, `'mobius'`, `'centres'` (eval only), `'direct'` (raw enumeration, small problems). Most users should leave it at `'auto'`. Hand-overriding is useful for benchmarking or for tests that need a specific method to exercise specific code paths.

---

## 5. The orbit-table system

The Möbius method's per-call cost in the inner product is dominated by $|\Omega_r|$ tensor contractions. For these contractions to run in microseconds rather than milliseconds, two pieces of precomputation are required, both encoded in the *orbit table*.

### What an orbit is

Möbius inversion on the partition lattice rewrites the distinct-index $r$-tuple sum as an alternating sum over set partitions of the slot indices, with each partition's term factorising across blocks. For the inner product specifically, the slot-permutation symmetry $S_r$ acts jointly on both sides; many partition pairs are equivalent under this action.

Concretely, a partition pair $(\pi_A, \pi_B)$ is identified by:

- $m_A$: the integer partition giving the block sizes of $\pi_A$.
- $m_B$: the integer partition giving the block sizes of $\pi_B$.
- $M$: a contingency matrix with row sums $m_A$ and column sums $m_B$, encoding how the blocks of $\pi_A$ and $\pi_B$ overlap.

The pair $(\pi_A, \pi_B)$ and its image under the $S_r$ action share the same $(m_A, m_B, M)$ triple --- up to within-size-group row and column permutations of $M$. Each equivalence class is an *orbit*; the orbit set is $\Omega_r$.

The unsymmetrised partition-pair count is $B_r^2$ where $B_r$ is the Bell number; the orbit count $|\Omega_r|$ is much smaller. Concretely:

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

For each orbit, the orbit table stores:

1. **The orbit identifier** $(m_A, m_B, M)$ in canonical form.

2. **The orbit's multiplicity** --- the number of partition pairs in this orbit's equivalence class.

3. **The orbit's Möbius coefficient** --- the alternating-sum sign and factorial coefficient from the partition lattice.

4. **A contraction recipe** --- a sequence of `einsum`-style tensor contractions that computes the orbit's contribution from the pairwise kernel matrix $K$ and the per-side source weights $w_A$, $w_B$. The recipe is parameterised by the orbit's structure; precomputed at table-build time.

5. **Precomputed einsum paths** for typical input shapes --- `np.einsum_path` is run once at table-build time and cached, so runtime calls bypass the path-finding overhead.

In v2.2, the MATLAB side gained an additional Stage-A optimisation: contraction recipes are precomputed at table-build time (`buildContractRecipe` + `executeRecipe`), avoiding the runtime set-operation overhead in the pair-picker that had made the MATLAB orbit path 14--32× slower than Python. After the Stage-A fix, MATLAB and Python run at comparable speeds (MATLAB/Python ratio is currently median 0.53× --- MATLAB is faster).

### Build, cache, and discipline

Orbit tables for $r = 2 \ldots 8$ ship pre-built in the toolbox:

- Python: `python/mpt/_orbit_tables/orbit_r{N}.pkl` (pickle format; total ~1.7 MB across the seven tables).
- MATLAB: `matlab/+mobius/_orbit_tables/orbit_r{N}.mat` (`.mat` format).

For $r > 8$, the table is built on first use and cached:

- Default cache location: `~/.mpt/orbit_tables/` (overridable via the `MPT_CACHE_DIR` environment variable).
- Build cost grows roughly as $B_r$, so $r = 9$ builds in seconds, $r = 10$ in minutes, $r = 11$ in hours.
- Each build is gated by a cost-preview warning that prints the Bell-number scaling and an estimated build time before construction begins.
- `MPT_NO_BUILD_WARN=1` suppresses the preview message (intended for automation contexts where the warning is noise).
- The hard cap is $r = 12$; beyond that, the build cost is prohibitive even for one-off use, and the toolbox refuses to attempt it.

The pickle format is chosen for speed and structural fidelity (each `OrbitEntry` is a dataclass-like object with numpy arrays inside). The brittleness of pickle across Python versions has been monitored; if a future Python upgrade breaks the shipped pickles, the build path is available to regenerate them. The format-stability commitment is "any LTS Python with numpy ≥ 1.21".

### The two-layer reformulation

A reader landing in `_mobius.py` cold should know that the file implements *two* layers of mathematical reformulation, which are sometimes conflated:

1. **Möbius inversion on the partition lattice.** A sum over distinct ordered $r$-tuples is rewritten as an alternating sum over set partitions, with each partition's term factorising across blocks. This step alone reduces a $K!/(K-r)!$ sum to $B_r$ partition contributions, each of which is a product of single-source-sum quantities.

2. **Orbit collapse under joint slot symmetry.** This applies only to the inner product (which has two sides and so a joint $S_r$ action). The $B_r^2$ partition pairs collapse to $|\Omega_r|$ orbit classes; each orbit is computed once with its multiplicity.

Layer 1 alone is enough for point evaluation and total mass (which have only one side). Layer 2 is the inner-product-specific further reduction.

This distinction matters for development: the eval-side Möbius code (`eval_orbit_abs`, `eval_orbit_rel`) is layer-1-only; the inner-product code (`inner_product_orbit`, `inner_product_orbit_grid`, `inner_product_orbit_pw_batched`) is layers 1 + 2.

### Public-in-the-module entry points

`_mobius.py` is internal from the toolbox's perspective but is organised as a self-contained sub-package. The public-in-the-module entry points (called from `tensor.py` and consumer wrappers, never directly by user code) are:

| Function | Layer | Purpose |
|:---|:---|:---|
| `get_orbit_table(r)` | --- | Load or build the orbit table for tensor order $r$ |
| `inner_product_orbit(K, w_A, w_B, r, ...)` | 1+2 | Unwindowed distinct-index inner product, single pair |
| `inner_product_orbit_grid(K, w_A, w_B, r, ...)` | 1+2 | Inner-product grid evaluation for windowed contributions |
| `inner_product_orbit_pw_batched(...)` | 1+2 | Batched inner product over MA per-attribute slot variations |
| `eval_orbit_abs(p, w, X, sigma, r, ...)` | 1 | Point-evaluation in absolute mode |
| `eval_orbit_rel(p, w, X, sigma, r, ...)` | 1 | Point-evaluation in relative mode (with u-grid vectorisation) |
| `total_mass_abs(p, w, sigma, r)` | 1 | Total-mass scalar in absolute mode |
| `total_mass_rel(p, w, sigma, r)` | 1 | Total-mass scalar in relative mode |

The MATLAB twin `+mobius` package has matching entry points (`mobius.getOrbitTable`, `mobius.innerProductOrbit`, etc.); semantics match exactly.

---

## 6. Numerical guarantees

The toolbox's numerical commitments fall into three categories.

### FP-bit-identical between releases (default settings)

User code that calls the toolbox with default kwargs in v2.1 should produce floating-point-identical output when run against v2.2 (and so on for future v2.x releases). The mechanism is conservative:

- Default routing is unchanged across releases for the regimes where the v2.x release introduced new methods. v2.2's `method='auto'` dispatcher selects Bulger's method in the regimes where v2.1 used Bulger's method, and the Möbius method only where v2.1 had no Möbius option. The two methods agree to floating-point precision in the regimes where both are valid, so the user-visible value is preserved.

- New default-on behaviour requires deliberate justification. The bar is "the previous behaviour was a bug, and the fix would have been ported back as a v2.x.y patch."

The cross-language golden tests (`test_cross_language_golden.py` and `.m`) hold a corpus of standard-regime calls and their expected floating-point outputs to ~14 significant decimal digits. They run in CI for both languages and fail any release that drifts.

The commitment is to *default-routing* behaviour. Non-default `method='mobius'` outputs in v2.2 might shift to FP-bit-identical with v2.1's Bulger output in a future release (if the dispatcher is recalibrated and `'mobius'` becomes default in some new regime); the commitment is to what `'auto'` produces, not to every code path.

### σ → 0 fallback discipline

At very small σ relative to the period $P$, the Gaussian kernel becomes a delta function and several toolbox computations are limit cases. The toolbox handles these limits structurally rather than numerically:

- The Möbius method's structural pre-screen (σ/P > 0.03 in periodic-relative mode) routes σ → 0 to Bulger's method or the centres path, both of which handle small σ correctly.

- The centres path's kernel evaluator (`gaussian_kernel_sum`) handles σ → 0 via the truncation_sigmas option (zero entries outside the kernel's support become exactly zero, not exponentially-small floats that propagate noise).

- Discrete-limit measures (`sameness`, `coherence`, `n_tuple_entropy`) have explicit σ = 0 branches that compute the limit directly rather than relying on the soft-mode kernel evaluation to converge.

### Möbius cancellation guard

Detailed in [§4](#4-the-dispatcher-pattern). The cancellation ratio is monitored *after* each Möbius call; the post-hoc check is exact (i.e. the ratio is computed from the actual orbit-class contributions, not estimated) and the fallback re-runs through Bulger's method without re-using any Möbius intermediate. This costs a re-run but the fallback is rare in practice (< 1% of standard-regime calls).

### Cross-language parity

The cross-language golden tests hold expected outputs to ~14 significant decimal digits for a standard regime corpus (1475 cells covering $r \in \{2..6\}$, $K$ up to 100, σ down to $10^{-5}$ cents, all four mode combinations, multi-attribute self-IPs, adversarial pitch configurations, and harmonic spectra up to $K = 64$). MATLAB and Python pass the same goldens to the same precision.

The mechanism is: each language computes the value via its full dispatcher stack (with `method='auto'`), and both are compared to a stored expected value computed at golden-generation time using `method='auto'` in Python. Any drift in either language is caught.

---

## 7. Twin-language conventions

The MATLAB and Python implementations are intentionally parallel. USER_GUIDE §4 ("API conventions") covers the *user-facing* mapping (function-name mapping table, weight-argument convention, query-point convention, etc.). This section adds the *developer-facing* rules for keeping the two sides aligned --- what to mirror, what's allowed to diverge, and how the test parity discipline works.

### Naming

| Style | MATLAB | Python |
|:---|:---|:---|
| Function names | `cosSimExpTens` (camelCase) | `cos_sim_exp_tens` (snake_case) |
| Class names | `MaetDensity` (struct field convention) | `MaetDensity` (CapWords) |
| Constants in user code | `PERIOD = 1200` (UPPER_SNAKE) | `PERIOD = 1200` (UPPER_SNAKE) |
| Private helpers | `+internal/helperName.m` | `_helper_name` |
| Package internals | `+mobius/canonicalForm.m` | `_mobius.canonical_form` |

The function-name mapping is the most consequential. Every public Python function `mpt.foo_bar` should have a MATLAB sibling `fooBar` with semantically identical inputs, outputs, and side effects (after applying the absence-sentinel and container-type conventions below).

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
| Single density | struct | `ExpTensDensity` instance |
| Multi-attribute density | struct with `.tag == 'MaetDensity'` | `MaetDensity` instance |

MATLAB structs and Python instances are used interchangeably for the density data structure; field names match exactly (with the conversion `obj.field` ↔ `obj.field`). This is non-idiomatic Python (a real Python implementation would use `__slots__` and properties), but is the path of least resistance for twin-language parity.

### Weight broadcast

Both languages follow the same broadcast convention for weights:

- Scalar weight: broadcast to all events.
- `[]` (MATLAB) / `None` (Python): broadcast to all-ones.
- Per-event vector (length $N$): one weight per source event.
- Per-slot vector (length $K_a$ for MA inputs): one weight per slot, broadcast across events.
- Full per-event-per-slot matrix ($K_a \times N$): explicit per-slot per-event weights.

The same rules apply to MAET inputs at the per-attribute level. The full mapping is documented in USER_GUIDE §4 ("Weight arguments").

### Twin-language test parity

Every Python `test_*.py` should have a MATLAB `test_*.m` counterpart with the same name suffix. Where the counterpart doesn't exist, the gap is either:

1. Deliberate: the feature is Python-only or MATLAB-only (rare; should be documented).
2. A known gap: the test exists but isn't yet ported.

The cross-language golden tests are the ultimate parity check: any value that drifts between MATLAB and Python at default routing fails CI on both sides.

### When languages should diverge

Twin-language parity is the default but not a rule. Acceptable divergences:

- Plotting features that one language has and the other doesn't (e.g., `audioPeaks.m`'s `'plot'` option, which has no Python equivalent because `matplotlib` would be a heavy optional dependency for one convenience feature).
- Language-idiomatic conveniences that don't change semantics (e.g., Python dataclass defaults, MATLAB `arguments` block validation).
- Performance optimisations that exploit per-language strengths (e.g., MATLAB's `pagefun` vs Python's broadcasting). The output must remain FP-bit-identical at default routing.

Divergences that *do* change semantics or output values are never acceptable without explicit documentation.

---

## 8. How to add a new measure

A walk-through for adding a new measure to the toolbox. The example: imagine adding a hypothetical `tonality_index` --- a single scalar describing how strongly a pitch multiset suggests a tonal centre.

### 1. Decide which tier the measure sits at

- *Tier 1 (density)*: no. A `tonality_index` is computed from a density, not a different density structure.
- *Tier 2 (primitive)*: no. It doesn't introduce a new operation on densities --- it composes existing primitives.
- *Tier 3 (consumer wrapper)*: yes. Composes `cos_sim_exp_tens` with a reference template.

### 2. Decide which file it lives in

In Python: `harmony.py` if the measure is harmony-flavoured; a new module otherwise. In MATLAB: a new top-level `.m` file. Match the categorisation USER_GUIDE §6 uses.

### 3. Decide which existing primitives it consumes

For `tonality_index`: presumably it computes `cos_sim_exp_tens` against one or more reference tonal-centre templates and returns a scalar derived from the resulting similarity values. It should consume `cos_sim_exp_tens` rather than implementing its own inner product.

### 4. SA-only or also MA?

Most consumer wrappers are SA-only at first introduction. MA support can be added later if the measure has a natural MA interpretation. The cleanest way to handle this in code is to write the SA path and raise `NotImplementedError` (Python) / `error('tonalityIndex:notImplementedForMA', ...)` (MATLAB) for MA input until MA is wired through.

### 5. Decide whether canonical-form dedup applies

Canonical-form dedup is for batched evaluations where structurally-identical chords map to the same output. For `tonality_index` operating on chord pitches:

- If the measure depends only on intervals (relative), then transpositions are equivalent and dedup by interval canonical form is appropriate.
- If the measure depends on absolute pitch class, transpositions are distinguishable and dedup is by chord canonical form.

The existing dedup helpers (`_chord_canonical_key`, `_pair_canonical_key`, etc.) are designed to be reused. New measures that need a different dedup key should add a new key function in `canonical.py` (after the planned `_tensor/` refactor; currently in `tensor.py`) and document its semantics.

### 6. Add to public API surface

Python: edit `mpt/__init__.py` to re-export `tonality_index` and add it to `__all__`. MATLAB: place `tonalityIndex.m` at `matlab/`'s top level (no other action needed --- MATLAB's path makes it public).

### 7. Mirror in the other language

Write the MATLAB and Python implementations side-by-side. The function signatures should be mechanical translations of each other (snake_case ↔ camelCase, `None` ↔ `[]`, list ↔ cell). The internal logic should also match step-for-step where reasonable.

### 8. Add tests in both languages

- A `test_tonality_index.py` and `test_tonality_index.m` with parallel test cases.
- Add at least one entry to the cross-language golden test corpus exercising the new measure on a known input/output pair.

### 9. Document

- Per-function docstring: full NumPy-doc style in Python, full H1 style in MATLAB. Document every parameter, return value, and side effect.
- USER_GUIDE entry: add to the appropriate section in §6 (function reference). The Guide's pointer-only format (after the planned restructuring) means one-line summary plus pointer to `help` for full reference --- but the summary should be accurate and complete.

### 10. Update CHANGELOG

Add a `### Added` entry to the unreleased v2.x section of `CHANGELOG.md`. If the new measure changes any existing behaviour, also add a `### Changed` entry and an entry in `MIGRATION.md` for the next release.

### 11. Demo

If the measure has obvious teaching value, add a demo to both `matlab/demos/` and `python/demos/` following the naming conventions (`demo_tonalityIndex.m` / `demo_tonality_index.py`). Include user-adjustable parameters at the top, prose comments oriented to learners (per the demo-comment conventions documented in style preferences).

### 12. Architecture document

If the new measure adds new structural patterns (e.g., a new dispatcher, a new caching system), update this document. Routine additions (consumer wrappers that compose existing primitives) need no ARCHITECTURE update.

---

## 9. Release discipline

### Orbit tables

The shipped orbit tables (`_orbit_tables/orbit_r{N}.pkl` for Python, `.mat` for MATLAB) are versioned with the toolbox release. The pickle/`.mat` format is considered stable but not frozen: if a future Python/numpy version breaks the shipped pickle, the build path is available to regenerate them in-place. The commitment is "any LTS Python with numpy ≥ 1.21 reads the shipped tables".

### Format provenance

Each shipped orbit table is built and validated against the corresponding `_mobius.py` source at release time. The build process is deterministic; identical source should produce identical tables. A diff in the table file is taken as a release-blocking signal that the source has changed in a way that requires the table to be rebuilt.

### CHANGELOG conventions

The CHANGELOG follows Keep-a-Changelog conventions:

- One section per release, ordered newest-first.
- Sub-sections: `### Added`, `### Changed`, `### Deprecated`, `### Removed`, `### Fixed`, `### Internal`.
- The unreleased section is dated `[X.Y.Z] --- Unreleased` and is appended to throughout development.
- On release, the date is filled in and a new unreleased section is opened.

### MIGRATION conventions

`MIGRATION.md` has one section per `vN.X → vN.X+1` boundary. Each section documents:

- Numerical changes at default settings (rare; high bar).
- New keyword arguments or kwargs with changed defaults.
- Deprecation paths for renamed or replaced functions.
- One-paragraph migration example for each non-trivial change.

The default routing FP-bit-identical commitment (see [§6](#6-numerical-guarantees)) means most MIGRATION entries are about *new opt-in features* rather than mandatory migration.

### Deprecation policy

Deprecations follow a two-release cycle:

1. *Release N*: deprecate the function/argument. Emit a `DeprecationWarning` (Python) or `<funcName>:deprecated` warning (MATLAB) on every call. Keep the function working unchanged otherwise. Document the replacement in the deprecation message.

2. *Release N+1 or later*: remove the deprecated function. The CHANGELOG `### Removed` section documents the removal.

Active examples in v2.2:

- `batch_cos_sim_exp_tens` (Python) / `batchCosSimExpTens` (MATLAB): deprecated as of v2.1, still functional as thin shims forwarding to `cos_sim_exp_tens` batched-raw mode.
- `cos_sim_exp_tens_raw` and `eval_exp_tens_raw` (Python): deprecated as of v2.1, still functional as thin shims forwarding to the unified entry points.

These shims will be removed in a future release; the timing is left open.

### Release artifact list

A v2.x release includes:

- Source code at the tagged commit on GitHub.
- A Zenodo deposit minted from the GitHub release, with the concept DOI preserved across versioned deposits.
- Updated CHANGELOG and MIGRATION docs.
- Updated CITATION.cff with the new version metadata.
- USER_GUIDE updated for new features.
- ARCHITECTURE updated if new structural patterns are introduced (this document).
