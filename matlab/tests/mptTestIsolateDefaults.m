function cleanup = mptTestIsolateDefaults()
%MPTTESTISOLATEDEFAULTS  Save+reset MPT defaults; restore on cleanup.
%
%   cleanup = mptTestIsolateDefaults()
%       Captures the caller's current mptDefaults struct, resets the
%       defaults to the factory state (mptDefaults('reset')), and
%       returns an onCleanup token that restores the saved struct
%       when destroyed.
%
%   Because MATLAB scripts keep their variables in base workspace
%   after the script exits, the token does NOT fire automatically
%   on script exit — callers must `clear` the holding variable
%   explicitly at script end, otherwise the user's defaults remain
%   at factory (the suite's running state) instead of being
%   restored. The expected pattern in a test script is:
%
%       % --- top of script ---
%       clear cleanupDefaults
%       cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
%
%       % ... test body ...
%
%       % --- end of script ---
%       clear cleanupDefaults
%
%   The opening `clear` drains any stale token left over from a
%   previous run of the same script in this session (see "stale
%   token gotcha" below). The closing `clear` fires this run's
%   token immediately, restoring the user's pre-test defaults
%   before the script exits.
%
%   Rationale (why isolate at all): many sections of the MPT test
%   suite — mobius vs direct enumeration, A<->B-swap symmetry,
%   broadcast vs explicit repmat, MA cell-form vs matrix-form
%   equivalence, cross-language goldens — assert exact algebraic
%   agreement at 1e-10 / 1e-12 tolerance. Those identities hold on
%   the un-truncated kernel path; the factory default
%   truncationSigmas = 6 is a deliberate ~6-sig-fig approximation
%   that would fail such tolerances by construction. This helper
%   therefore pins truncationSigmas = Inf as the suite baseline, so
%   goldens and identities are verified at full precision. Tests
%   that exercise truncation itself set truncationSigmas explicitly
%   per call and so override the baseline.
%
%   Stale token gotcha: when this helper is called a second time
%   in the same session (e.g., user runs test_mpt twice), the
%   natural reassignment `cleanupDefaults = mptTestIsolateDefaults()`
%   releases the old binding AFTER this function has captured prev
%   and reset to factory. The old token's destructor then fires,
%   restoring its captured-at-previous-run state — which corrupts
%   the new suite. The opening `clear cleanupDefaults` in the
%   pattern above defeats this by firing the stale destructor
%   BEFORE prev is captured.
%
%   The %#ok<NASGU> on the assignment suppresses MATLAB's "value
%   never used" warning (the token's effect is on its lifetime,
%   not its value).
%
%   See also: MPTDEFAULTS, ONCLEANUP.

    prev = mptDefaults();
    mptDefaults('reset');
    % Pin the exact (un-truncated) kernel path as the suite baseline.
    % The factory default (truncationSigmas = 6) is a ~6-sig-fig
    % approximation; exact-algebra identities and goldens are verified
    % un-truncated, where the 1e-12 parity guarantee holds. Truncation
    % tests set truncationSigmas explicitly per call and so override this.
    mptDefaults('truncationSigmas', Inf);
    internal.maybeShowTruncationNotice('suppress');
    cleanup = onCleanup(@() iRestoreAndRearm(prev));
end


function iRestoreAndRearm(prev)
%IRESTOREANDREARM  Restore saved defaults and re-arm the truncation notice.
    mptDefaults(prev);
    internal.maybeShowTruncationNotice('rearm');
end
