function test_dispatch_msg_throttle()
%TEST_DISPATCH_MSG_THROTTLE  Once-per-session throttling for dispatch msgs.
%
%   Verifies that internal.maybeShowDispatchMsg:
%     1. Prints the first occurrence of a (funcName, chosen, reason).
%     2. Suppresses subsequent identical occurrences.
%     3. Prints a new occurrence with a different reason.
%     4. Prints all occurrences afresh after a 'reset' call.
%
%   The throttle is the basis for the user-facing behaviour where a
%   tight loop (e.g., demo_edoApprox with 101 cosSimExpTens calls)
%   produces one informational message instead of 101.

    results = cell(0, 2);

    % Ensure a clean throttle state to start.
    internal.maybeShowDispatchMsg('reset');

    % --- Test 1: first call fires, second identical call is silent.
    out1 = evalc("internal.maybeShowDispatchMsg('foo', 'bulger', 'r = 1', 0, false);");
    out2 = evalc("internal.maybeShowDispatchMsg('foo', 'bulger', 'r = 1', 0, false);");
    results{end+1, 1} = 'first call fires';
    results{end, 2} = ~isempty(strtrim(out1)) && contains(out1, 'foo') ...
                    && contains(out1, 'bulger') && contains(out1, 'r = 1');
    results{end+1, 1} = 'repeated identical call is silent';
    results{end, 2} = isempty(strtrim(out2));

    % --- Test 2: differing reason fires fresh.
    out3 = evalc("internal.maybeShowDispatchMsg('foo', 'bulger', 'K - r < 2', 0, false);");
    results{end+1, 1} = 'differing reason fires fresh';
    results{end, 2} = ~isempty(strtrim(out3)) && contains(out3, 'K - r < 2');

    % --- Test 3: differing function name fires fresh.
    out4 = evalc("internal.maybeShowDispatchMsg('bar', 'bulger', 'r = 1', 0, false);");
    results{end+1, 1} = 'differing function name fires fresh';
    results{end, 2} = ~isempty(strtrim(out4)) && contains(out4, 'bar');

    % --- Test 4: differing chosen method fires fresh.
    out5 = evalc("internal.maybeShowDispatchMsg('foo', 'mobius', 'r = 1', 0, false);");
    results{end+1, 1} = 'differing chosen method fires fresh';
    results{end, 2} = ~isempty(strtrim(out5)) && contains(out5, 'mobius');

    % --- Test 5: probed-form message includes time estimate.
    internal.maybeShowDispatchMsg('reset');
    out6 = evalc("internal.maybeShowDispatchMsg('foo', 'bulger', 'probe', 4.5, true);");
    results{end+1, 1} = 'probed form includes time estimate';
    results{end, 2} = contains(out6, 'estimated') && contains(out6, 'Ctrl');

    % --- Test 6: after reset, the previously-seen tuple fires again.
    out7 = evalc("internal.maybeShowDispatchMsg('foo', 'bulger', 'r = 1', 0, false);");
    internal.maybeShowDispatchMsg('reset');
    out8 = evalc("internal.maybeShowDispatchMsg('foo', 'bulger', 'r = 1', 0, false);");
    results{end+1, 1} = 'reset clears seen-set';
    % out7 was already cached from Test 1, so should be silent; out8 after
    % reset should fire.
    results{end, 2} = isempty(strtrim(out7)) && ~isempty(strtrim(out8));

    % --- Test 7: mptDefaults('reset') also clears the throttle.
    out9 = evalc("internal.maybeShowDispatchMsg('foo', 'bulger', 'r = 1', 0, false);");
    mptDefaults('reset');
    out10 = evalc("internal.maybeShowDispatchMsg('foo', 'bulger', 'r = 1', 0, false);");
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
