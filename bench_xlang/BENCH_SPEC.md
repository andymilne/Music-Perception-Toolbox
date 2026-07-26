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
