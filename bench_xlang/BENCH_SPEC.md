# Cross-language benchmark specification

Both `bench_xlang.py` and `bench_xlang.m` iterate the same set of cases,
generate identical inputs from deterministic formulae, time each call,
and write a CSV with per-row wall-clock time and a numerical checksum.

## Input generation

For a case with parameters `(A, N, K, period)`, positions and weights
are deterministic functions of the index, identical in both languages:

```
p[j, n, a] = period * ((j + 3*n + 7*a) mod (K + N + A)) / (K + N + A)
w[j, n, a] = 0.7 + 0.3 * cos((j + 2*n + 5*a) / (K + N + A) * pi)
```

Query positions for eval:

```
x[d, q] = period * (0.5 + 0.4 * sin((d + 2*q + 1) / (dim + nQ + 1) * pi))
```

These give identical inputs in Python and MATLAB with no shared file
and no RNG-alignment problem.

## Benchmark grid

Base case: `A=1, N=1, K=8, r=2, sigma_over_P=0.10, isRel=false,
isPer=true, wrap='full-image', nQ=10`.

Each axis varied independently from the base:

- `sigma_over_P` in {0.002, 0.005, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50}
- `r` in {1, 2, 3, 4}
- `(isRel, isPer)` in {(F,F), (F,T), (T,F), (T,T)}
- `wrap` in {'full-image', 'single-image'}
- `A` in {1, 2, 3}
- `N` in {1, 5, 20, 50, 100}
- `K` in {4, 8, 16, 50, 100}
- `nQ` (eval only) in {1, 10, 100}

Each unique configuration is measured for `eval_exp_tens` and
`cos_sim_exp_tens`. Duplicates (e.g. the base row appears in every
sweep) are dropped in the runner.

## CSV output

Columns:
- `label` — short identifier `"axis=value"` for the swept axis
  (or `"base"` for the reference row).
- `sigma_over_P, r, isRel, isPer, wrap, A, N, K, nQ` — full parameter
  set for the row.
- `operation` — `"eval"` or `"cossim"`.
- `elapsed_s` — best-of-3 wall-clock in seconds.
- `checksum` — `sum(abs(v))` for eval; cosine value for cossim.

## Comparison

`compare_bench.py` joins the two CSVs on
`(label, operation, sigma_over_P, r, isRel, isPer, wrap, A, N, K, nQ)`
and prints:

- Value agreement: relative difference in `checksum` per row.
- Time ratio: `matlab_elapsed_s / python_elapsed_s` per row.
- Flags rows where value disagreement exceeds a tolerance or time
  ratio exceeds a factor-of-10 in either direction.

## Sweep and cost-model benches (v2.2 sweep work)

`bench_sweep.{py,m}` time the point-set query sweep (all-r = 1,
one value per event) in three variants — `broadcast` (the batched
kernel pass), `loop_memo` (scalar calls with the self-IP memo carried:
object persistence in Python, the cache-carrying outputs in MATLAB),
and `loop_fresh` (scalar calls with no cross-pair memo) — under both
normalisations, at N in {300, 1200, 5000} with M = 100 three-event
queries. Inputs are deterministic formulae shared by both languages;
the `checksum` column (sum of the sweep's similarities) is the
cross-language value-parity check, and `compare_sweep.py` enforces it
at 1e-9 relative when joining the CSVs. Timing uses each language's
repeated-measurement helper (`adaptive_time` / `internal.timeRepeated`);
every Python variant's timed closure is one complete operation from a
cold memo — broadcast and `loop_memo` clear the caches at the top of
each timed call, `loop_fresh` before every pair — matching the MATLAB
bench, whose value-semantics memo cannot cross calls. Python's
object-attached memo would otherwise persist across the timer's
repetitions, and the `cosine` rows would compare Python-warm against
MATLAB-cold (the state of the first delivered revision, whose cosine
ratios were confounded exactly so). In the MATLAB cost-model bench the
warm regimes seed the memo once outside the timed closure via the
cache-carrying outputs.

`bench_cost_model.{py,m}` audit the self-matrix skip flags added to the
Bulger-vs-Möbius pricing. Part 1 is deterministic: the selector is
called across symmetric and asymmetric (broadcast-shaped: a large
shared operand against a small fixed query) grids with the flags off
and on, and the rows where the routing flips are the cells the flags
exist for — on the symmetric grid both prices shrink near-
proportionally, so flips concentrate on the asymmetric grid, where the
shared self matrix dominates the full-triple Bulger price. Part 2
times `method='mobius'` under three memo regimes (three, two, and one
matrices computed) to measure per-matrix costs against the pricing's
`n_matrices / 3` scaling of the fitted whole-triple constants. Part 3
times both forced routes at each flip cell in the warm-selves regime;
a `MISPICK` verdict means the flags-on choice was not the measured
faster route at that near-crossover cell. Flip cells are the model's
weakest points by construction and the deliberate bias is toward
Bulger (the cheap-to-mispick side), so isolated small-magnitude
mispicks are expected; systematic large ones on the reference machine
indicate the per-language constants want a refit.

The per-language fitted constants make the two languages' flip cells
legitimately different; `compare_sweep.py` reports routing differences
as informative, not as defects.
