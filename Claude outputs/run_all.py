"""Reproduce every computed figure and number in "The Music Perception Toolbox".

This is a runner: the work is done by the scripts alongside it, each of which
can also be run on its own. Everything here uses version 3.0 of the toolbox
and nothing else.

    worked_example.py           the chordal-probe worked application
    make_figures.py             its scatter and correlation matrix
    make_th_zoom.py             tensor harmonicity around the triad peaks
    make_consonance.py          five consonance measures over triad space
    generate_article_figures.py expectation tensor modes, generator-chain
                                tunings, SPCS probe-tone profiles, triad
                                similarity grids, balance, and position-level
                                rhythmic features
    sensitivity.py              what the consonance measures respond to

The dataflow diagram is not produced by any script: it is drawn in TikZ in
the manuscript source.

Requirements
------------
    music-perception-toolbox 3.0, numpy, scipy, matplotlib

If the toolbox is not installed, point MPT_PYTHON at the `python` directory
of a checkout; every script honours it:

    MPT_PYTHON=/path/to/Music-Perception-Toolbox/python python3 run_all.py

Usage
-----
    python3 run_all.py                    # everything, in dependency order
    python3 run_all.py consonance th-zoom  # only those
    python3 run_all.py sensitivity         # only that analysis
    python3 run_all.py --list              # what can be run

Running everything takes a few hours on a typical machine, almost all of it
in the consonance comparison and the tensor harmonicity zoom. Both cache
their grids beside their scripts, so a second
run only redraws; delete the .npz files to force recomputation. If you want
the figures sooner, both scripts take a resolution argument -- see their
docstrings -- and doubling either value quarters the work.

Parameters
----------
The article uses 16 harmonics with power-law rolloff rho = 1 throughout,
sigma = 10 cents for the (S)P(C)S family, and sigma = 12 cents for the
consonance measures, with tensor harmonicity at sigma * sqrt(2). Each script
sets these as named constants at the top of its own file rather than sharing
them, so that any script can be read and run on its own.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# Each task names the script to run and the arguments to pass. Order matters
# in one place only: the scatter and correlation matrix are drawn from the
# results file that
# worked_example.py writes, so the worked example must run first.
TASKS = [
    ("worked-example", "the chordal-probe worked application",
     ["worked_example.py"]),
    ("worked-example-plot", "its scatter and correlation matrix",
     ["make_figures.py"]),
    ("th-zoom", "tensor harmonicity around the triad peaks",
     ["make_th_zoom.py"]),
    ("consonance", "five consonance measures over triad space",
     ["make_consonance.py"]),
    ("standard-figures", "the six figures with no separate script",
     ["generate_article_figures.py"]),
    ("sensitivity", "what the consonance measures respond to",
     ["sensitivity.py"]),
]
BY_NAME = {name: (desc, cmd) for name, desc, cmd in TASKS}


def run(cmd):
    """Run one script in this directory, passing the environment through."""
    proc = subprocess.run([sys.executable] + cmd, cwd=HERE)
    if proc.returncode != 0:
        sys.exit(f"failed: {' '.join(cmd)}")


def main():
    args = sys.argv[1:]
    if "--list" in args or "-l" in args:
        for name, desc, _ in TASKS:
            print(f"  {name:22s} {desc}")
        return

    if args:
        unknown = [a for a in args if a not in BY_NAME]
        if unknown:
            sys.exit(f"unknown task(s): {', '.join(unknown)}\n"
                     f"try --list")
        # de-duplicate while preserving the order given
        seen, chosen = set(), []
        for a in args:
            desc, cmd = BY_NAME[a]
            key = tuple(cmd)
            if key not in seen:
                seen.add(key)
                chosen.append((a, desc, cmd))
        # the plot needs the worked example's results file
        if any(c[0] == "make_figures.py" for _, _, c in chosen) and \
                not os.path.exists(os.path.join(HERE,
                                                "worked_example_results.json")):
            chosen.insert(0, BY_NAME["worked-example"][0] and
                          ("worked_example", *BY_NAME["worked-example"]))
    else:
        chosen = TASKS

    for name, desc, cmd in chosen:
        print(f"\n=== {name}: {desc}")
        t0 = time.time()
        run(cmd)
        print(f"=== {name} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
