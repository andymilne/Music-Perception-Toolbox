"""Cross-language gate agreement, driven from a shared fixture.

The existing gate tests pin decisions independently in each language
against hand-written expectations. That catches a regression in one
language but not a shared misunderstanding: if both sides were wrong in
the same way, both would pass. This reads the same fixture the MATLAB
twin reads, so a decision can only be marked correct once.

Why gate agreement is worth a dedicated test. These gates choose between
routes, and before the full-image work those routes computed different
measures --- the centres route the pairwise-wrap (single-image) form,
the grid route a transposition average. A gate disagreement would then
have shown up as a value difference that looks like a numerical bug
rather than a routing mismatch. The routes now agree in value, so a
disagreement costs only time; the constants are still deliberately
matched, and this test is what holds them matched.
"""
import json
import pathlib

import numpy as np
import pytest

from mpt._tensor._mobius_inner import (_ma_rel_attr_prefers_centres)

_FIXTURE = (pathlib.Path(__file__).resolve().parents[2]
            / "matlab" / "tests" / "gate_agreement_parity.json")


def _load():
    if not _FIXTURE.exists():
        pytest.skip(f"gate fixture not present at {_FIXTURE}")
    with open(_FIXTURE) as f:
        return json.load(f)["cases"]


@pytest.fixture(scope="module")
def cases():
    return _load()


def test_fixture_is_not_degenerate(cases):
    # A fixture that is all-centres or all-grid would pass trivially
    # while testing nothing about where the boundary sits.
    decisions = [c["decision"] for c in cases]
    n_true = sum(decisions)
    assert len(decisions) >= 100
    assert 0.05 < n_true / len(decisions) < 0.95, (
        f"fixture is lopsided: {n_true}/{len(decisions)} centres"
    )


def test_every_gate_decision_reproduces(cases):
    mismatches = []
    for c in cases:
        px = np.asarray(c["px"], dtype=float).reshape(-1, 1)
        py = np.asarray(c["py"], dtype=float).reshape(-1, 1)
        got = bool(_ma_rel_attr_prefers_centres(
            px, py, c["sigma"], c["r"], True, c["isPer"], c["period"]
        ))
        if got != c["decision"]:
            mismatches.append(
                f"r={c['r']} K={c['K']} sigma={c['sigma']} "
                f"isPer={c['isPer']} period={c['period']}: "
                f"expected {c['decision']}, got {got}"
            )
    assert not mismatches, (
        f"{len(mismatches)}/{len(cases)} gate decisions differ:\n  "
        + "\n  ".join(mismatches[:10])
    )


def test_decisions_are_monotone_in_K_within_each_shape(cases):
    """Within a fixed (r, sigma, period, isPer), the gate must not
    oscillate as K grows: centres cost rises as K^(2r) while grid cost
    rises far more slowly, so once grid wins it must keep winning.
    A non-monotone boundary would mean the cost model is not ordering
    the two paths consistently."""
    groups = {}
    for c in cases:
        key = (c["r"], c["sigma"], c["period"], c["isPer"])
        groups.setdefault(key, []).append((c["K"], c["decision"]))
    bad = []
    for key, items in groups.items():
        items.sort()
        seen_grid = False
        for K, dec in items:
            if not dec:
                seen_grid = True
            elif seen_grid:
                bad.append(f"{key}: centres reappears at K={K}")
    assert not bad, "non-monotone gate boundary:\n  " + "\n  ".join(bad)
