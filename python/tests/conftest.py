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

Rationale: many sections of the suite — mobius vs direct enumeration,
A<->B-swap symmetry, broadcast vs explicit-tile equivalence, MA
cell-form vs matrix-form equivalence, cross-language goldens — assert
exact algebraic agreement at 1e-10 / 1e-12 tolerance. Those identities
hold on the *un-truncated* kernel path; the factory default
``truncation_sigmas=6`` is a deliberate ~6-significant-figure
approximation that would break such tolerances by construction. The
function-scoped fixture therefore pins ``truncation_sigmas=inf`` as the
suite baseline, so goldens and identities are verified at full
precision (this is where the 1e-12 guarantee is observed). Tests that
exercise truncation itself (e.g. ``test_kernel_truncation.py``,
``test_eval_routing.py``) set ``truncation_sigmas`` explicitly per
call and so are unaffected by the baseline. This is the Python analogue
of the MATLAB ``mptTestIsolateDefaults`` helper called at the top of
``test_mpt.m``.

Some test files (notably ``v22/test_kernel_truncation.py`` and
``v22/test_eval_routing.py``) define their own file-level autouse
``reset_defaults`` fixtures. Those become redundant once this conftest
is in place but remain harmless — the function-scoped fixture below
runs before them, and both call ``mpt.reset_defaults()``. A follow-up
cleanup pass could remove the file-level duplicates.
"""

from __future__ import annotations

import math

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
    # Re-arm the one-time truncation notice so running the suite in a
    # long-lived interpreter (e.g. Jupyter) does not permanently silence
    # it for subsequent interactive use.
    from mpt import _defaults as _mpt_defaults
    _mpt_defaults._rearm_truncation_notice()


@pytest.fixture(autouse=True)
def _mpt_reset_each_test():
    """Reset to factory defaults before every test, then pin the exact
    kernel path and silence the dispatch messages.

    Tests that want to verify the dispatch console UX call
    ``mpt.reset_defaults()`` / re-enable ``show_hints`` themselves at
    test entry. Tests that don't care about the UX get silent execution
    by default, regardless of test execution order.
    """
    mpt.reset_defaults()
    # Pin the exact (un-truncated) kernel path as the suite baseline.
    # The factory default (``truncation_sigmas=6``) is a ~6-sig-fig
    # approximation; exact-algebra identities and goldens are verified
    # untruncated, where the 1e-12 guarantee holds. Truncation tests set
    # ``truncation_sigmas`` explicitly per call and so override this.
    mpt.set_default(show_hints=False, truncation_sigmas=math.inf)
    # Keep the one-time truncation-default notice out of test output.
    from mpt import _defaults as _mpt_defaults
    _mpt_defaults._suppress_truncation_notice()
    yield
