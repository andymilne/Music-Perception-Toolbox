"""Compare bench_python.csv and bench_matlab.csv.

Joins on (label, operation) and reports value agreement (relative
error in checksum) and time ratio (matlab / python) per row.

Usage:
    python3 compare_bench.py                                # defaults
    python3 compare_bench.py -p bench_py.csv -m bench_ml.csv
    python3 compare_bench.py --value-tol 1e-9 --time-flag 10
"""
import argparse
import csv
from pathlib import Path


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def key(row):
    return (row['label'], row['operation'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--python-csv', default='bench_python.csv')
    ap.add_argument('-m', '--matlab-csv', default='bench_matlab.csv')
    ap.add_argument('--value-tol', type=float, default=1e-9,
                    help='Relative error threshold on checksum (default 1e-9)')
    ap.add_argument('--time-flag', type=float, default=10.0,
                    help='Flag rows where |log(matlab/python)| > log(flag) '
                         '(default 10x either way)')
    args = ap.parse_args()

    py_rows = read_csv(args.python_csv)
    ml_rows = read_csv(args.matlab_csv)
    py_by_key = {key(r): r for r in py_rows}
    ml_by_key = {key(r): r for r in ml_rows}

    all_keys = sorted(set(py_by_key) | set(ml_by_key))
    print(f"Loaded {len(py_rows)} Python rows and {len(ml_rows)} MATLAB rows")
    print(f"Joined on (label, operation): {len(all_keys)} unique keys\n")

    hdr = f"{'label':<28} {'op':<6} {'py_s':>10} {'ml_s':>10} " \
          f"{'ratio':>8} {'py_cs':>14} {'ml_cs':>14} {'rel_err':>10}   flags"
    print(hdr)
    print('-' * len(hdr))

    max_rel_err = 0.0
    max_ratio = 1.0
    n_val_bad = 0
    n_time_bad = 0
    missing = []

    for k in all_keys:
        pyr = py_by_key.get(k)
        mlr = ml_by_key.get(k)
        label, op = k
        if pyr is None:
            missing.append((k, 'python'))
            continue
        if mlr is None:
            missing.append((k, 'matlab'))
            continue

        py_t = float(pyr['elapsed_s'])
        ml_t = float(mlr['elapsed_s'])
        py_cs = float(pyr['checksum'])
        ml_cs = float(mlr['checksum'])

        # Value comparison
        denom = max(abs(py_cs), abs(ml_cs), 1e-30)
        rel_err = abs(py_cs - ml_cs) / denom
        val_flag = rel_err > args.value_tol

        # Time comparison
        ratio = ml_t / py_t if py_t > 0 else float('inf')
        time_flag = (ratio > args.time_flag) or (ratio < 1.0 / args.time_flag)

        max_rel_err = max(max_rel_err, rel_err)
        max_ratio = max(max_ratio, ratio, 1.0 / max(ratio, 1e-30))
        if val_flag:
            n_val_bad += 1
        if time_flag:
            n_time_bad += 1

        flags = ''
        if val_flag:
            flags += 'V'
        if time_flag:
            flags += 'T'

        print(f"{label:<28} {op:<6} {py_t:>10.4g} {ml_t:>10.4g} "
              f"{ratio:>8.2f} {py_cs:>14.6g} {ml_cs:>14.6g} "
              f"{rel_err:>10.2e}   {flags}")

    print()
    print(f"Value agreement: max rel_err = {max_rel_err:.3e}")
    print(f"  {n_val_bad} rows exceed tolerance {args.value_tol:.1e} "
          f"(flag V)")
    print(f"Time ratio: max |ratio| = {max_ratio:.2f}x")
    print(f"  {n_time_bad} rows exceed {args.time_flag:.0f}x either way "
          f"(flag T)")
    if missing:
        print(f"\n{len(missing)} rows missing on one side:")
        for k, side in missing:
            print(f"  missing from {side}: {k}")


if __name__ == '__main__':
    main()
