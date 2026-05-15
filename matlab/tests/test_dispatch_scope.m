function test_dispatch_scope()
%TEST_DISPATCH_SCOPE  Top-level scope detection and per-call throttle reset.
%
%   Verifies that internal.dispatchScope, when placed as the first
%   executable line of a user-facing function, gives users a fresh
%   dispatch announce on every top-level call while keeping inner
%   sub-call announces throttled within one top-level call.
%
%   The mechanism: dispatchScope maintains a persistent depth
%   counter. When the counter is 0 (we are entering the toolbox from
%   outside), it resets the maybeShowDispatchMsg throttle. The
%   returned onCleanup guard decrements the counter on function exit.
%   Nested entries see depth > 0 and do not touch the throttle, so
%   inner repeated dispatches are suppressed as before.
%
%   evalc requires a text scalar (a char row vector or a scalar
%   string), so the multi-statement command strings below are built
%   as char arrays with escaped single quotes (e.g. ''foo'' for the
%   literal 'foo').

    results = cell(0, 2);

    % Ensure a clean state. Explicitly enable showHints in case an
    % earlier test left it off (some tests in test_harmony.m suppress
    % it temporarily).
    internal.dispatchScope('reset');
    internal.maybeShowDispatchMsg('reset');
    prevSH = mptDefaults('showHints', true);
    cleanupSH = onCleanup(@() mptDefaults(prevSH)); %#ok<NASGU>

    % Build a reusable triple-emitting command and a top-level-call
    % command as char arrays.  The triple-call body always emits one
    % unprobed dispatch message for ('foo', 'bulger', 'r=1').
    emit = 'internal.maybeShowDispatchMsg(''foo'', ''bulger'', ''r=1'', 0, false);';

    % --- Test 1: depth starts at 0, increments on entry, returns to 0
    %     when the guard goes out of scope.
    d0 = internal.dispatchScope('query');
    results{end+1, 1} = 'depth starts at 0';
    results{end, 2} = (d0 == 0);

    localOpenScope();   % opens and closes a scope inside a helper
    d2 = internal.dispatchScope('query');
    results{end+1, 1} = 'depth returns to 0 after helper exits';
    results{end, 2} = (d2 == 0);

    % --- Test 2: top-level entry resets the throttle.
    % Prime the throttle with one announce.
    evalc(emit);
    % Now simulate a fresh top-level call. The scope guard's reset
    % should clear the seen-set, so the same triple announces again.
    cmd = ['guard = internal.dispatchScope(); ' emit ' clear guard'];
    out = evalc(cmd);
    results{end+1, 1} = 'top-level entry resets throttle';
    results{end, 2} = ~isempty(strtrim(out)) && contains(out, 'foo');

    % --- Test 3: inner sub-call within one top-level call stays
    %     throttled. (Same triple should print once, not twice.)
    internal.dispatchScope('reset');
    internal.maybeShowDispatchMsg('reset');
    cmd = ['guard_outer = internal.dispatchScope(); ' ...
           emit ' ' ...
           'guard_inner = internal.dispatchScope(); ' ...
           emit ' ' ...
           'clear guard_inner; clear guard_outer'];
    out = evalc(cmd);
    nLines = numel(strsplit(strtrim(out), newline));
    results{end+1, 1} = 'inner sub-call within one top-level call stays throttled';
    results{end, 2} = (nLines == 1) && contains(out, 'foo');

    % --- Test 4: two consecutive top-level calls each announce.
    internal.dispatchScope('reset');
    internal.maybeShowDispatchMsg('reset');
    cmd1 = ['guard1 = internal.dispatchScope(); ' emit ' clear guard1'];
    cmd2 = ['guard2 = internal.dispatchScope(); ' emit ' clear guard2'];
    out1 = evalc(cmd1);
    out2 = evalc(cmd2);
    results{end+1, 1} = 'consecutive top-level calls each announce';
    results{end, 2} = ~isempty(strtrim(out1)) && ~isempty(strtrim(out2));

    % --- Test 5: nested entry does NOT reset (proves the depth check).
    %     If we prime the throttle, then enter a scope at depth > 0
    %     (i.e. while already inside another scope), the throttle should
    %     NOT be reset and the second call should stay silent.
    internal.dispatchScope('reset');
    internal.maybeShowDispatchMsg('reset');
    cmd = ['guard_outer = internal.dispatchScope(); ' ...
           emit ' ' ...
           'guard_nested = internal.dispatchScope(); ' ...   % depth 1->2
           emit ' ' ...
           'clear guard_nested; clear guard_outer'];
    out = evalc(cmd);
    nLines = numel(strsplit(strtrim(out), newline));
    results{end+1, 1} = 'nested scope does not reset throttle';
    results{end, 2} = (nLines == 1);

    % --- Report.
    nTests = size(results, 1);
    nPassed = sum([results{:, 2}]);
    if nPassed == nTests
        fprintf('  test_dispatch_scope: %d/%d passed\n', nPassed, nTests);
    else
        fprintf('  test_dispatch_scope: %d/%d passed\n', nPassed, nTests);
        for k = 1:nTests
            if ~results{k, 2}
                fprintf('    FAILED: %s\n', results{k, 1});
            end
        end
    end

    internal.dispatchScope('reset');
    internal.maybeShowDispatchMsg('reset');
end


function localOpenScope()
%LOCALOPENSCOPE  Helper for the depth-tracking test.
%   Opens a scope and immediately returns; the onCleanup guard fires
%   on exit, decrementing the depth.
    guard = internal.dispatchScope(); %#ok<NASGU>
end
