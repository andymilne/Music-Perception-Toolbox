"""Join the Python and MATLAB sweep / cost-model CSVs.

Usage (from bench_xlang/, after running both languages' benches):

    python compare_sweep.py

For the sweep bench: prints per-(N, normalize, variant) timings side by
side with the MATLAB/Python ratio, and verifies the checksums agree to
1e-9 relative (a cross-language value-parity failure is reported
loudly — timings are meaningless if the values differ).

For the cost-model bench: prints selector-audit rows where the two
languages' routing decisions differ (expected wherever the per-language
fitted constants place a crossover differently — informative, not a
defect), and the Möbius per-matrix timings side by side.
"""
import csv
import os


def read_rows(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    # ---- Sweep ----------------------------------------------------
    py = read_rows(os.path.join(here, "bench_sweep_python.csv"))
    ml = read_rows(os.path.join(here, "bench_sweep_matlab.csv"))
    if py and ml:
        key = lambda r: (r["N"], r["normalize"], r["variant"])
        ml_by = {key(r): r for r in ml}
        print(f"{'N':>6} {'normalize':>14} {'variant':>11} "
              f"{'py ms/off':>10} {'ml ms/off':>10} {'ml/py':>7}  parity")
        for r in py:
            m = ml_by.get(key(r))
            if m is None:
                continue
            t_py = float(r["ms_per_offset"])
            t_ml = float(m["ms_per_offset"])
            c_py = float(r["checksum"])
            c_ml = float(m["checksum"])
            rel = abs(c_py - c_ml) / max(abs(c_py), 1e-300)
            parity = "OK" if rel < 1e-9 else f"FAIL ({rel:.1e})"
            print(f"{r['N']:>6} {r['normalize']:>14} {r['variant']:>11} "
                  f"{t_py:10.4f} {t_ml:10.4f} {t_ml / t_py:7.2f}  {parity}")
    else:
        print("sweep: one or both CSVs missing; run bench_sweep.py and "
              "bench_sweep.m first.")

    # ---- Cost model -----------------------------------------------
    py = read_rows(os.path.join(here, "bench_cost_model_python.csv"))
    ml = read_rows(os.path.join(here, "bench_cost_model_matlab.csv"))
    if py and ml:
        key = lambda r: (r["part"], r["r"], r["K"], r["N"], r["regime"])
        ml_by = {key(r): r for r in ml}
        print("\nselector rows where the languages route differently "
              "(per-language constants; informative):")
        any_diff = False
        for r in py:
            if not r["part"].startswith("selector"):
                continue
            m = ml_by.get(key(r))
            if m is None:
                continue
            for col in ("chosen_off", "chosen_on"):
                if r[col] != m[col]:
                    any_diff = True
                    print(f"  {r['part']} r={r['r']} K={r['K']} "
                          f"N={r['N']} {col}: py={r[col]} ml={m[col]}")
        if not any_diff:
            print("  (none)")
        print("\nMöbius per-matrix timings (ms):")
        print(f"{'r':>3} {'N':>4} {'regime':>16} {'python':>9} "
              f"{'matlab':>9}")
        for r in py:
            if r["part"] != "mobius_matrices":
                continue
            m = ml_by.get(key(r))
            if m is None:
                continue
            print(f"{r['r']:>3} {r['N']:>4} {r['regime']:>16} "
                  f"{float(r['ms']):9.4f} {float(m['ms']):9.4f}")
    else:
        print("\ncost model: one or both CSVs missing; run "
              "bench_cost_model.py and bench_cost_model.m first.")


if __name__ == "__main__":
    main()
