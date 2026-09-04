"""Python twin of matlab/tests/bench_spectral_ip_gate.m.

Times the spectral (Fourier) branch of the relative-mode inner product
against the translation grid, over the same grid of shapes the MATLAB
benchmark uses, and prints a CSV block with identical columns so the two
languages' output can be compared line for line.

It times ``_rel_inner_batched`` directly with the spectral
branch toggled on and off via ``_SPECTRAL_IP_ENABLED`` --- the same
object the MATLAB harness times through ``relInnerBatched`` --- so the
comparison is like for like. Both routes compute the full-image measure,
so the toggle changes cost, not value.

Run from the ``python`` directory of the toolbox, with the package
importable:

    python3 tools/bench_spectral_ip_gate.py

Run it warm on an otherwise idle machine. The first call in each cell is
discarded to absorb any one-off cost; the reported figure is the median
of the rest. Expect a few minutes. Send back the block between
CSV_BEGIN and CSV_END.

Columns: r, K, N, sigmaOverP, isPer, gridSize, msSpectral, msGrid, ratio
where ratio = msGrid / msSpectral (branch worth taking where ratio > 1).
"""
import sys
import time

import numpy as np

from mpt._tensor import cosine as _c
import mpt._tensor._mobius_inner as _mobius_inner

PERIOD = 1200.0
MODE_SIGMAS = _mobius_inner._SPECTRAL_IP_MODE_SIGMAS
REPS = 5

# Same grid as bench_spectral_ip_gate.m.
RS = (2, 3, 4)
KS = (4, 8, 16, 30)
NS = (1, 2, 4, 8, 16)
SOPS = (0.002, 0.005, 0.0125, 0.05, 0.20)
IS_PERS = (True, False)


def _grid_size(sigma, r, is_per):
    # Mirrors the MATLAB harness's reported sizing exactly: the
    # non-periodic embedding length assumes each multiset spans the full
    # period (span_x + span_y = 2 * PERIOD), so the reported gridSize is
    # a fixed function of (sigma, r, is_per) and does not depend on the
    # particular points. This is the sizing bench_spectral_ip_gate.m
    # prints, not necessarily the one the branch computes internally from
    # the actual span; it exists so the two languages' CSV rows carry the
    # same gridSize column and line up.
    if is_per:
        L = PERIOD
    else:
        L = 2.0 * PERIOD + 2.0 * (MODE_SIGMAS + 2.0) * sigma
    M = int(np.ceil(MODE_SIGMAS / np.sqrt(2.0) * L / (2.0 * np.pi * sigma))) + 2
    return (2 * M + 1) ** (r - 1)


def _time(Px, Wx, Py, Wy, sigma, r, is_per, spectral):
    prev = _mobius_inner._SPECTRAL_IP_ENABLED
    prev_force = _mobius_inner._SPECTRAL_IP_FORCE
    _mobius_inner._SPECTRAL_IP_ENABLED = spectral
    # The spectral arm must bypass the cost gate, not merely enable the
    # branch: with the gate in force every cell the shipped constant
    # declines would run the grid in both arms and read a ratio of 1.0
    # by construction (the MATLAB twin was found doing exactly that on
    # all 80 of its declined cells). _SPECTRAL_IP_FORCE bypasses the
    # comparison only, never the _SPECTRAL_IP_MAX_POINTS memory guard.
    _mobius_inner._SPECTRAL_IP_FORCE = bool(spectral)
    try:
        period = PERIOD if is_per else 0.0
        # One warm-up call, discarded, which also gauges the cost.
        t0 = time.perf_counter()
        _mobius_inner._rel_inner_batched(
            Px, Wx, Py, Wy, sigma, r, is_per, period,
            truncation_sigmas=float('inf'))
        first = time.perf_counter() - t0
        # Cost-aware repetition: cheap cells get the full REPS for a
        # stable median; cells already costing seconds are timed fewer
        # times, since their own magnitude dwarfs timing jitter and the
        # extended low-sigma/P grid has a few very heavy ones.
        if first > 2.0:
            reps = 1
        elif first > 0.2:
            reps = 2
        else:
            reps = REPS
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            _mobius_inner._rel_inner_batched(
                Px, Wx, Py, Wy, sigma, r, is_per, period,
                truncation_sigmas=float('inf'))
            ts.append(time.perf_counter() - t0)
        return 1e3 * float(np.median(ts))
    finally:
        _mobius_inner._SPECTRAL_IP_ENABLED = prev
        _mobius_inner._SPECTRAL_IP_FORCE = prev_force


def main():
    sys.stderr.write(
        "\n=== bench_spectral_ip_gate (Python) ===\n"
        "Measuring the spectral branch against the translation grid.\n"
        "Send the CSV block below back for comparison with MATLAB.\n\n")
    sys.stderr.flush()

    print("CSV_BEGIN")
    print("r,K,N,sigmaOverP,isPer,gridSize,msSpectral,msGrid,ratio")
    for is_per in IS_PERS:
        for r in RS:
            for K in KS:
                if K < r:
                    continue
                for N in NS:
                    for sop in SOPS:
                        sigma = sop * PERIOD
                        rng = np.random.default_rng(1000 * r + 10 * K + N)
                        Px = np.sort(rng.uniform(0, PERIOD, (K, N)), axis=0)
                        Py = np.sort(rng.uniform(0, PERIOD, (K, N)), axis=0)
                        Wx = np.ones((K, N))
                        Wy = np.ones((K, N))
                        gs = _grid_size(sigma, r, is_per)
                        # When the mode grid exceeds the memory guard the
                        # branch always declines, so both toggle states run
                        # the grid path and the ratio is 1 by construction.
                        # Timing that adds nothing about the cost gate and
                        # these are the slowest cells, so record the decline
                        # without paying for a multi-second grid contraction.
                        if gs > _mobius_inner._SPECTRAL_IP_MAX_POINTS:
                            print(f"{r},{K},{N},{sop:.4f},{int(is_per)},{gs},"
                                  f"nan,nan,nan")
                            sys.stdout.flush()
                            continue
                        ms_s = _time(Px, Wx, Py, Wy, sigma, r, is_per, True)
                        ms_g = _time(Px, Wx, Py, Wy, sigma, r, is_per, False)
                        ratio = ms_g / max(ms_s, 1e-12)
                        print(f"{r},{K},{N},{sop:.4f},{int(is_per)},{gs},"
                              f"{ms_s:.4f},{ms_g:.4f},{ratio:.4f}")
                        sys.stdout.flush()
    print("CSV_END")
    sys.stderr.write(
        "\nDone. Send the block between CSV_BEGIN and CSV_END.\n\n")


if __name__ == "__main__":
    main()
