#!/usr/bin/env bash
# bench_exp_tensor_plots.sh
#
# Cross-language timing harness for demo_expTensorPlots (MATLAB) and
# its Python port demo_exp_tensor_plots.py. Times each at default
# kernel settings and at truncationSigmas = 6, several reps per
# condition. Also captures available physical memory and the resolved
# kernel-chunk byte budget — both relevant to interpretation, since
# the chunk count depends on the budget and so does per-call
# overhead.
#
# Usage:
#   ./bench_exp_tensor_plots.sh                    # auto-detect REPO
#   REPO=/path/to/repo ./bench_exp_tensor_plots.sh
#   PYTHON=/path/to/python ./bench_exp_tensor_plots.sh
#   REPS=5 ./bench_exp_tensor_plots.sh
#   ./bench_exp_tensor_plots.sh --skip-matlab      # Python only
#   ./bench_exp_tensor_plots.sh --skip-python      # MATLAB only

set -euo pipefail

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
REPS="${REPS:-3}"
PYTHON="${PYTHON:-python3}"
MATLAB="${MATLAB:-matlab}"
REPO="${REPO:-$(cd "$(dirname "$0")" && pwd)}"

SKIP_PYTHON=0
SKIP_MATLAB=0
for arg in "$@"; do
    case "$arg" in
        --skip-python) SKIP_PYTHON=1 ;;
        --skip-matlab) SKIP_MATLAB=1 ;;
        -h|--help)
            sed -n '2,18p' "$0"; exit 0 ;;
    esac
done

# Sanity checks
if [[ ! -d "$REPO/python/mpt" || ! -d "$REPO/matlab" ]]; then
    echo "Error: REPO=$REPO does not look like the toolbox root." >&2
    echo "Pass REPO=/path/to/Music-Perception-Toolbox" >&2
    exit 1
fi

# ---------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------
echo "=== System ==="
uname -srm
case "$(uname)" in
    Darwin)
        sysctl -n machdep.cpu.brand_string 2>/dev/null || true
        echo "Total memory:     $(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024)) GB"
        # Available memory: (free + inactive + speculative) * pagesize
        vm_stat | awk -v pgsz="$(sysctl -n hw.pagesize)" '
            /Pages free/             { free=$3+0 }
            /Pages inactive/         { inact=$3+0 }
            /Pages speculative/      { spec=$3+0 }
            END {
                printf "Available memory: %.2f GB\n",
                       (free + inact + spec) * pgsz / 1024 / 1024 / 1024
            }'
        ;;
    Linux)
        grep -E 'model name' /proc/cpuinfo | head -1 | sed 's/.*: //'
        awk '/MemTotal/ {printf "Total memory:     %.2f GB\n", $2/1024/1024}' /proc/meminfo
        awk '/MemAvailable/ {printf "Available memory: %.2f GB\n", $2/1024/1024}' /proc/meminfo
        ;;
esac
echo "CPU cores:        $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo unknown)"
echo "Repo:             $REPO"
echo "Reps per cond:    $REPS"
echo

# ---------------------------------------------------------------------
# Python timing
# ---------------------------------------------------------------------
if [[ $SKIP_PYTHON -eq 0 ]]; then
    echo "=== Python (demo_exp_tensor_plots.py) ==="
    PY_RUNNER=$(mktemp /tmp/mpt_bench_py.XXXXXX.py)
    trap 'rm -f $PY_RUNNER' EXIT

    cat > "$PY_RUNNER" << 'PYEOF'
import argparse, os, runpy, sys, time
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda *a, **kw: None  # demo has interactive widgets — disable

ap = argparse.ArgumentParser()
ap.add_argument('demo')
ap.add_argument('--trunc', type=float, default=None)
args = ap.parse_args()

import mpt
mpt.set_default(show_hints=False)
if args.trunc is not None:
    mpt.set_default(truncation_sigmas=args.trunc)

# Report the resolved chunk budget — chunk count depends on it
from mpt._utils import kernel_chunk_bytes_resolved
print(f"  chunk_budget: {kernel_chunk_bytes_resolved() / 1e9:.2f} GB", flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(args.demo)))
t = time.perf_counter()
runpy.run_path(args.demo, run_name='__main__')
print(f"  elapsed:      {time.perf_counter() - t:.3f} s")
PYEOF

    DEMO_PY="$REPO/python/demos/demo_exp_tensor_plots.py"

    for cond in "default" "trunc=6"; do
        echo "  condition: $cond"
        for i in $(seq 1 "$REPS"); do
            echo "  rep $i:"
            if [[ "$cond" == "default" ]]; then
                "$PYTHON" "$PY_RUNNER" "$DEMO_PY" 2>&1 | grep -E "chunk_budget|elapsed"
            else
                "$PYTHON" "$PY_RUNNER" "$DEMO_PY" --trunc 6 2>&1 | grep -E "chunk_budget|elapsed"
            fi
        done
        echo
    done
fi

# ---------------------------------------------------------------------
# MATLAB timing
# ---------------------------------------------------------------------
if [[ $SKIP_MATLAB -eq 0 ]]; then
    echo "=== MATLAB (demo_expTensorPlots.m) ==="

    ML_RUNNER=$(mktemp /tmp/mpt_bench_ml_XXXXXX)
    ML_RUNNER_M="${ML_RUNNER}.m"
    mv "$ML_RUNNER" "$ML_RUNNER_M"
    ML_FUNC=$(basename "$ML_RUNNER_M" .m)
    ML_DIR=$(dirname "$ML_RUNNER_M")

    cat > "$ML_RUNNER_M" << MLEOF
function $ML_FUNC(demoPath, truncSigmas)
    set(0, 'DefaultFigureVisible', 'off');
    addpath('$REPO/matlab');
    mptDefaults('reset');
    mptDefaults('showHints', false);
    if isfinite(truncSigmas)
        mptDefaults('truncationSigmas', truncSigmas);
    end
    fprintf('  chunk_budget: %.2f GB\n', ...
            internal.kernelChunkBytesResolved() / 1e9);
    addpath(fileparts(demoPath));
    tStart = tic;
    run(demoPath);
    fprintf('  elapsed:      %.3f s\n', toc(tStart));
    close all force;
end
MLEOF

    DEMO_ML="$REPO/matlab/demos/demo_expTensorPlots.m"

    for cond in "default" "trunc=6"; do
        echo "  condition: $cond"
        if [[ "$cond" == "default" ]]; then
            TRUNC="Inf"
        else
            TRUNC="6"
        fi
        for i in $(seq 1 "$REPS"); do
            echo "  rep $i:"
            "$MATLAB" -batch \
                "addpath('$ML_DIR'); $ML_FUNC('$DEMO_ML', $TRUNC);" 2>&1 \
                | grep -E "chunk_budget|elapsed"
        done
        echo
    done

    rm -f "$ML_RUNNER_M"
fi

echo "=== Done ==="
