"""The orbit count |Omega_r| and the orbit-table build warning.

Cost models need only |Omega_r|, so pricing a route must never build an
orbit table (hours at r = 9). ``orbit_count`` reads the count from a
closed table, which is checked here against an independent Burnside
count and against the shipped tables' lengths. When the orbit route is
actually taken beyond the shipped range, the build announces its cost
on stderr, whatever ``verbose`` says, and says that it may be
interrupted.
"""
import numpy as np
import pytest

import mpt._mobius as mob
from mpt import build_maet, sweep_sim_maet
from mpt._tensor.sweep import (
    _choose_sweep_route, orbit_sweep_supported, sweep_eligibility,
)


@pytest.mark.parametrize("r", range(2, 13))
def test_closed_table_matches_burnside(r):
    assert mob._ORBIT_COUNTS[r] == mob._orbit_count_burnside(r)


@pytest.mark.parametrize("r", range(2, 9))
def test_count_matches_shipped_tables(r):
    assert mob.orbit_count(r) == len(mob.get_orbit_table(r))


def test_count_rejects_out_of_range():
    with pytest.raises(ValueError):
        mob.orbit_count(1)
    with pytest.raises(ValueError):
        mob.orbit_count(mob._R_HARD_CAP + 1)


def _r9_pair():
    rng = np.random.default_rng(9)
    p_x = [rng.normal(0.0, 3.0, (9, 1))]
    p_y = [rng.normal(0.0, 3.0, (9, 1))]
    geom = ([0.9], [9], [False], [False], [None], [True])
    return (build_maet(p_x, None, *geom, verbose=False),
            build_maet(p_y, None, *geom, verbose=False))


@pytest.fixture
def no_build(monkeypatch):
    """Fail on any orbit-table build, and start from an empty cache."""
    def refuse(*args, **kwargs):
        raise AssertionError("an orbit table was built")
    monkeypatch.setattr(mob, "_orbit_cache", {})
    monkeypatch.setattr(mob, "_build_orbit_table", refuse)
    monkeypatch.setattr(mob, "_maybe_warn_build_cost", refuse)


def test_pricing_at_r9_does_not_build(no_build):
    dx, dy = _r9_pair()
    off = np.linspace(-2.0, 2.0, 5).reshape(1, -1)
    mixture_ok, _ = sweep_eligibility(dx, dy, off)
    orbit_ok = orbit_sweep_supported(dx, dy, off)
    assert mixture_ok and orbit_ok
    # Both routes are admissible, so the chooser prices the orbit route.
    route = _choose_sweep_route(dx, dy, off, mixture_ok, orbit_ok)
    assert route == "mixture"
    vals = sweep_sim_maet(dx, dy, off, verbose=False)
    assert np.all(np.isfinite(vals))


def test_build_beyond_shipped_range_warns(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("MPT_NO_BUILD_WARN", raising=False)
    monkeypatch.setattr(mob, "_orbit_cache", {})
    monkeypatch.setattr(mob, "_USER_CACHE_DIR", tmp_path)
    monkeypatch.setattr(mob, "_build_orbit_table", lambda r: [])
    mob.get_orbit_table(9)
    err = capsys.readouterr().err
    assert "building orbit table for r=9" in err
    assert "interrupted" in err


def test_orbit_route_at_r9_warns_with_verbose_off(monkeypatch, tmp_path,
                                                  capsys):
    monkeypatch.delenv("MPT_NO_BUILD_WARN", raising=False)
    monkeypatch.setattr(mob, "_orbit_cache", {})
    monkeypatch.setattr(mob, "_USER_CACHE_DIR", tmp_path)
    monkeypatch.setattr(mob, "_build_orbit_table", lambda r: [])
    dx, dy = _r9_pair()
    off = np.linspace(-2.0, 2.0, 3).reshape(1, -1)
    try:
        sweep_sim_maet(dx, dy, off, method="orbit", verbose=False)
    except Exception:
        pass  # the stand-in table is empty; only the warning matters
    assert "building orbit table for r=9" in capsys.readouterr().err
