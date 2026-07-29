function test_dispatch_msg_throttle()
%TEST_DISPATCH_MSG_THROTTLE  Throttling and print-format for dispatch msgs.
%
%   Verifies that internal.maybeShowDispatchMsg:
%     1. Prints the first occurrence of a (funcName, chosen) pair.
%     2. Suppresses subsequent occurrences with the same (funcName,
%        chosen), regardless of the routingReason argument (the reason
%        is no longer part of the throttle key — two different reasons
%        leading to the same chosen path collapse to a single announce).
%     3. Prints all occurrences afresh after a 'reset' call.
%     4. Uses the documented print format: "<func>: chose '<chosen>'
%        path." --- the message announces the routing DECISION only and
%        never carries a time estimate (estimates are emitted separately
%        by estimateCompTime under the per-call verbose flag).
%
%   The throttle is the basis for the user-facing behaviour where a
%   tight loop produces one informational message per unique dispatch
%   instead of one per iteration.

    results = cell(0, 2);

    % Ensure a clean throttle state to start. Explicitly enable
    % showHints in case an earlier test in the runner left it off.
    internal.maybeShowDispatchMsg('reset');
    prevSH = mptDefaults('showHints', true);
    cleanupSH = onCleanup(@() mptDefaults(prevSH)); %#ok<NASGU>

    % --- Test 1: first call fires, second identical call is silent.
    %     New print format: "foo: chose 'bulger' path." (no reason).
    out1 = evalc("internal.maybeShowDispatchMsg('foo', 'bulger', 'r = 1');");
    out2 = evalc("internal.maybeShowDispatchMsg('foo', 'bulger', 'r = 1');");
    results{end+1, 1} = 'first call fires with documented format';
    results{end, 2} = ~isempty(strtrim(out1)) ...
                    && contains(out1, 'foo') && contains(out1, 'bulger') ...
                    && contains(out1, 'path') ...
                    && ~contains(out1, 'r = 1');  % reason not user-visible
    results{end+1, 1} = 'repeated identical call is silent';
    results{end, 2} = isempty(strtrim(out2));

    % --- Test 2: differing reason is silent (same (funcName, chosen)
    %     already in seen-set; reason is NOT part of the throttle key).
    out3 = evalc("internal.maybeShowDispatchMsg('foo', 'bulger', 'r = 9 > 8');");
    results{end+1, 1} = 'differing reason collapses to same throttle key';
    results{end, 2} = isempty(strtrim(out3));

    % --- Test 3: differing function name fires fresh.
    out4 = evalc("internal.maybeShowDispatchMsg('bar', 'bulger', 'r = 1');");
    results{end+1, 1} = 'differing function name fires fresh';
    results{end, 2} = ~isempty(strtrim(out4)) && contains(out4, 'bar');

    % --- Test 4: differing chosen method fires fresh.
    out5 = evalc("internal.maybeShowDispatchMsg('foo', 'mobius', 'r = 1');");
    results{end+1, 1} = 'differing chosen method fires fresh';
    results{end, 2} = ~isempty(strtrim(out5)) && contains(out5, 'mobius');

    % --- Test 5: the dispatch message announces the DECISION only and
    %     never carries a time estimate. Time estimates are a separate
    %     concern emitted by estimateCompTime under the per-call verbose
    %     flag, so the announce must contain neither 'estimated' nor the
    %     'Ctrl+C' cancellation hint.
    internal.maybeShowDispatchMsg('reset');
    out6 = evalc("internal.maybeShowDispatchMsg('foo', 'bulger', 'probe');");
    results{end+1, 1} = 'dispatch message is decision-only (no estimate, no Ctrl+C)';
    results{end, 2} = ~isempty(strtrim(out6)) ...
                    && contains(out6, 'foo') && contains(out6, 'bulger') ...
                    && contains(out6, 'path') ...
                    && ~contains(out6, 'estimated') ...
                    && ~contains(out6, 'Ctrl');

    % --- Test 6: after reset, the previously-seen tuple fires again.
    out7 = evalc("internal.maybeShowDispatchMsg('foo', 'bulger', 'r = 1');");
    internal.maybeShowDispatchMsg('reset');
    out8 = evalc("internal.maybeShowDispatchMsg('foo', 'bulger', 'r = 1');");
    results{end+1, 1} = 'reset clears seen-set';
    % out7 was already cached from Test 1, so should be silent; out8 after
    % reset should fire.
    results{end, 2} = isempty(strtrim(out7)) && ~isempty(strtrim(out8));

    % --- Test 7: mptDefaults('reset') also clears the throttle.
    out9 = evalc("internal.maybeShowDispatchMsg('foo', 'bulger', 'r = 1');");
    mptDefaults('reset');
    out10 = evalc("internal.maybeShowDispatchMsg('foo', 'bulger', 'r = 1');");
    results{end+1, 1} = 'mptDefaults(''reset'') clears the throttle';
    results{end, 2} = isempty(strtrim(out9)) && ~isempty(strtrim(out10));

    % --- Test 8: 'reset' alone is a no-op print (no message emitted).
    internal.maybeShowDispatchMsg('reset');
    out11 = evalc("internal.maybeShowDispatchMsg('reset');");
    results{end+1, 1} = '''reset'' prints nothing itself';
    results{end, 2} = isempty(strtrim(out11));

    % --- Report.
    nTests = size(results, 1);
    nPassed = sum([results{:, 2}]);
    if nPassed == nTests
        fprintf('  test_dispatch_msg_throttle: %d/%d passed\n', nPassed, nTests);
    else
        fprintf('  test_dispatch_msg_throttle: %d/%d passed\n', nPassed, nTests);
        for k = 1:nTests
            if ~results{k, 2}
                fprintf('    FAILED: %s\n', results{k, 1});
            end
        end
    end

    % Clean up so subsequent tests / interactive use start clean.
    internal.maybeShowDispatchMsg('reset');
end
