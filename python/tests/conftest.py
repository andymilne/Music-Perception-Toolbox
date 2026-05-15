"""Pytest configuration and autouse fixtures for the MPT test suite.

Provides two autouse fixtures that together isolate test runs from the
user's interactive session state and from cross-test state leakage:

  * **Session-scoped** (`_mpt_save_user_defaults`): snapshots the
    caller's ``mpt.get_defaults()`` at session start, restores it at
    session end. Pytest is often invoked from interactive sessions
    where the user has set non-default truncation / precision values;
    this fixture means running the suite has no lasting side-effect on
    those settings.

  * **Function-scoped** (`_mpt_reset_each_test`): resets defaults to
    factory state before every test. Within a single pytest run, tests
    that set defaults (``mpt.set_default(...)``) should not leak that
    state into later tests. This fixture guarantees independence
    regardless of test execution order.

Rationale: many sections of the v2.2 suite — mobius vs direct
enumeration, A<->B-swap symmetry, broadcast vs explicit-tile
equivalence, MA cell-form vs matrix-form equivalence, cross-language
goldens — assert agreement at 1e-10 / 1e-12 tolerance under the
factory defaults (un-truncated kernels at double precision). Under
``truncation_sigmas`` < ``inf`` or ``kernel_precision='single'``, those
tolerances fail by construction. This is the Python analogue of the
MATLAB ``mptTestIsolateDefaults`` helper called at the top of
``test_mpt.m``.

Some test files (notably ``v22/test_kernel_truncation.py`` and
``v22/test_eval_routing.py``) define their own file-level autouse
``reset_defaults`` fixtures. Those become redundant once this conftest
is in place but remain harmless — the function-scoped fixture below
runs before them, and both call ``mpt.reset_defaults()``. A follow-up
cleanup pass could remove the file-level duplicates.
"""

from __future__ import annotations

import pytest

import mpt


@pytest.fixture(scope="session", autouse=True)
def _mpt_save_user_defaults():
    """Save the user's pre-session defaults; restore at session end."""
    prev = mpt.get_defaults()
    yield
    mpt.reset_defaults()
    if prev:
        mpt.set_default(**prev)


@pytest.fixture(autouse=True)
def _mpt_reset_each_test():
    """Reset to factory defaults before every test, then silence the
    one-time hints / dispatch messages.

    Tests that want to verify the hint/dispatch console UX call
    ``mpt.reset_defaults()`` themselves at test entry — that
    restores the factory ``show_hints=True``, and the hint fires as
    a user would see it. Tests that don't care about the UX get
    silent execution by default, regardless of test execution
    order. This matches the spirit of the existing per-test
    ``mpt.reset_defaults()`` calls in the dispatch/hint tests while
    sparing every other test from the historical accident that
    those messages happened to be throttled across the suite.
    """
    mpt.reset_defaults()
    mpt.set_default(show_hints=False)
    yield
