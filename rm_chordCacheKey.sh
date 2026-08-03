#!/usr/bin/env bash
# Remove the orphaned matlab/+internal/chordCacheKey.m.
#
# chordCacheKey was renamed to chordCanonicalKey in an earlier session
# ("cache" was the wrong word: the function computes a canonical key, and
# caching is what callers do with it). The new file shipped and the call
# sites were updated, but the deletion of the old file was never issued,
# so both have been sitting in +internal since. The orphan has no callers.
#
# Run from the repository root, on sym-dev.

set -euo pipefail

OLD="matlab/+internal/chordCacheKey.m"
NEW="matlab/+internal/chordCanonicalKey.m"

cd "$(git rev-parse --show-toplevel)"

# 1. Both files must be where we think they are.
test -f "$OLD" || { echo "ABORT: $OLD not found (already removed?)"; exit 1; }
test -f "$NEW" || { echo "ABORT: $NEW not found — do not delete the old one"; exit 1; }

# 2. The replacement must actually be the same function. Compare the two
#    with the name substituted, so only genuine content differences show.
echo "== diff (old renamed to new) =="
sed -e 's/chordCacheKey/chordCanonicalKey/g' \
    -e 's/CHORDCACHEKEY/CHORDCANONICALKEY/g' "$OLD" \
    | diff - "$NEW" || true
echo "== end diff =="

# 3. No live caller may reference the old name. The CHANGELOG mentions it
#    in a historical release entry, which is correct as written and is
#    excluded here; anything else is a blocker.
HITS=$(grep -rn "chordCacheKey" \
         --include=*.m --include=*.py --include=*.mlx \
         . | grep -v "^\./$OLD:" || true)
if [ -n "$HITS" ]; then
    echo "ABORT: live references to chordCacheKey remain:"
    echo "$HITS"
    exit 1
fi
echo "No live callers. Removing $OLD."

git rm "$OLD"

echo
echo "Done. Stage nothing else with this; commit it on its own:"
echo "  git commit -m 'Remove the orphaned internal.chordCacheKey'"
