# Routing-map specification (shared by the Python and MATLAB mappers)

Goal: an exhaustive, source-verified map of every routing decision in the
Music Perception Toolbox's analytical computations — where a call can go,
what decides it, and which leaf routine does the arithmetic. It will be
diffed against the other language's map, so follow this format exactly.

## Scope (entry points)

Trace each of these user-facing entry points, in the order listed, down to
the leaf routines that compute numbers:

1. cosine similarity: `cos_sim_exp_tens` / `cosSimExpTens` — every input
   shape it accepts (single density pair; density lists / batched;
   scalar-pair shortcuts), single-multiset (SA) vs multi-attribute (MA) vs
   nested (specs) densities, the all-r = 1 fast path, and every `method`
   value ('auto', 'bulger', 'centres', 'mobius', 'contract', and anything
   else the parser accepts, including retired synonyms).
2. sweep: `sweep_cos_sim_exp_tens` / `sweepCosSimExpTens`.
3. point evaluation: `eval_exp_tens` / `evalExpTens` — every `method`
   value; SA vs MA vs nested; batched queries.
4. entropy: `entropy_exp_tens` / `entropyExpTens` — every `method` value
   and what each delegates to (inner product, total mass, grid, …).
5. windowed similarity: `windowed_similarity` / `windowedSimilarity`
   (and `windowed_tensor_similarity` / `windowedTensorSimilarity` if it
   routes differently).
6. `explain_dispatch` / `explainDispatch` — only to record which of the
   above decisions it reports (it must not be a separate routing).

Within those, trace the sub-dispatchers all the way down:
- the flat MA selector (`_select_ma_inner_product_method` /
  `internal.selectMaInnerProductMethod`), including hard rules (guards,
  feasibility, sigma/P threshold + `wrap`), the cost comparison, and the
  timing probe if any;
- the per-attribute Möbius matrix (`_mobius_inner` /
  `+mobius/maPerAttrInnerMatrix.m` and friends): tuple-centres vs
  translation grid (`_ma_rel_attr_prefers_centres` /
  `maRelAttrPrefersCentres`), the spectral gate
  (`_SPECTRAL_IP_COST_C`, `spectralIpEnabled/Force`), comb-side
  restriction, the abs-per full-image branch and its L = 0 short-circuit,
  the Y-side guard, block Q forms;
- the wrapped Gaussian (`wrapped_gaussian_1d` / `internal.wrappedGaussian1d`):
  image-sum vs Fourier, L = 0;
- the nested path (`_try_nested_contract`, `_nested_admissible_routes`,
  `_nested_attr_plan`, `_nested_cost` / `internal.nestedContract`,
  `internal.nestedCost`): plan vs enumeration, per-attribute routes
  (contract, centres, contract_relnonper, taugrid), forced routes, the
  measure rule (`wrap` + sigma/P threshold), memory guard, safety factor;
- the eval selector (`_select_ma_eval` / `internal.selectMaEval`):
  centres vs Möbius, ordered-attribute rejection, precision guard,
  working-set gate on the safety factor, the per-node direct vs factored
  strategy in relative mode;
- kernel evaluation chunks (`_kernel._eval_chunk` /
  `internal.gaussianKernelSum` evalChunk): abs-per full-image vs
  nearest-image at L = 0, tabulated form, single-image opt-in;
- the single-multiset (SA) inner-product paths (direct, orbit, sparse,
  pw-batched, grid) and what selects among them;
- post-hoc guards (impossible-value check, cancellation threshold) and
  where they divert to;
- self-IP memoisation: which routes read/write which keys, and the
  shared pricing flag (`_self_ip_memoised` / `internal.selfIpMemoised`).

Do NOT trace: preprocessing, density construction, audio, canonicalisation,
windows construction, plotting. Only routing that changes which arithmetic
runs (or which measure is computed).

## Method

Read the source; do not guess from docstrings. For every branch record the
exact predicate as written (variable names as in the code). Where a
predicate depends on a fitted constant, name the constant and its value.
Where a decision is "cheaper of A and B by cost model X", record X's
function name and the terms it prices, not the arithmetic. Where a user
override bypasses a decision, say so explicitly. Where a branch raises an
error rather than routing, record the error identifier / message stem.
Note any dead code: a branch whose predicate can never be true, a routine
defined but never called from any traced path, a `method` value accepted
by the parser but never acted on. Check reachability of every routine in
the routing modules listed above (grep for callers).

## Output — two files

### 1. `routing_map_<lang>.md` — the human-readable map

One section per entry point, as a decision tree in nested bullets:

```
- ENTRY cos_sim_exp_tens(x, y, method, ...)
  - [parse] method ∈ {...}; retired synonyms: ...
  - IF <predicate as in code> → NODE <name>          (file:line)
    - IF ... → LEAF <routine>                          (file:line)
    - ELSE → ...
  - OVERRIDE method='bulger': skips <which decision>, still subject to <guard>
  - ERROR <identifier> when <predicate>
```

Then per entry point:
- a table of every LEAF routine reached (name, file, one-line role);
- a list of overrides and what each bypasses / does not bypass;
- a list of guards and fallbacks (with direction of fallback);
- the memo keys read/written on each route.

Finish the file with:
- **Orphans**: routines in the routing modules with no caller on any
  traced path (with your evidence: the grep you ran);
- **Suspicious**: predicates that look wrong, unreachable, duplicated,
  or inconsistent with a docstring; do not fix anything — report;
- **Constants table**: every fitted constant / threshold on any path,
  with value and defining file.

### 2. `routing_edges_<lang>.json` — the machine-readable graph

A JSON object `{"nodes": [...], "edges": [...]}`.
Node: `{"id": "<language-neutral id>", "kind": "entry|decision|leaf|error|override", "label": "<short>", "src": "<file>:<line>"}`.
Edge: `{"from": "<id>", "to": "<id>", "when": "<predicate, language-neutral>", "override": "<method value or null>"}`.

Use language-neutral ids so the two graphs can be joined: lower snake_case
of the Python name where a Python twin exists, and the same for MATLAB
(e.g. `select_ma_inner_product_method`, `ma_rel_attr_prefers_centres`,
`wrapped_gaussian_1d`, `eval_chunk`, `nested_attr_plan`). Predicates in
`when` should be written in a neutral notation: `sigma/P > threshold(ts)`,
`method == 'bulger'`, `K - r >= 2`, `L == 0`, `cost_bulger <= cost_mobius`.

Write both files to /home/claude/. Be exhaustive and precise; this is a
verification document, so it is better to be long than to omit a branch.
Report back with a short summary (entry points traced, node/edge counts,
orphans found, suspicious items) — the files are the deliverable.
