#!/usr/bin/env bash
# Align local matlab/ to Claude's baseline (tree 3d03ef28...).
# Run from the repository root. Review before running; it deletes files.
# NOTE: demo_edoApprox.m is handled SEPARATELY (see notes) because your
# local copy has edits worth reviewing first. This script aligns
# everything else; after it runs, the tree hash will NOT yet match until
# you also resolve demo_edoApprox.m.
set -e

echo "== removing SA-unification shim + diagnostic strays =="
git rm -f matlab/+internal/isSaShaped.m matlab/+internal/saView.m
git rm -f matlab/tests/diag_sa_flip.m matlab/tests/test_sa_unified.m

echo "== removing misplaced root-level bench copy =="
git rm -f matlab/bench_ip_dispatch.m

echo "== moving bench_ip_unit_cost.m into tests/ (identical content) =="
git mv matlab/bench_ip_unit_cost.m matlab/tests/bench_ip_unit_cost.m

echo "== overwriting the 2 content-differing bench files with baseline versions =="
cp align_payload/matlab/tests/bench_ip_dispatch.m       matlab/tests/bench_ip_dispatch.m
cp align_payload/matlab/tests/bench_ma_eval_dispatch.m  matlab/tests/bench_ma_eval_dispatch.m

echo ""
echo "== DONE with automatic part. Now resolve demo_edoApprox.m: =="
echo "   Your version differs from baseline AND you had local edits."
echo "   To take Claude's baseline version:"
echo "     cp align_payload/matlab/demos/demo_edoApprox.m matlab/demos/demo_edoApprox.m"
echo "   Then: git add matlab/ && git write-tree --prefix=matlab/"
echo "   Expect: 3d03ef282797729b862e18cb72584287b7b48d0c"
