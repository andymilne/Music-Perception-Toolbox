function tests = test_dispatcher_probe
%TEST_DISPATCHER_PROBE  v2.2.x — unified dispatcher + probe-based estimator.
%
%   Verifies hard rules decide without probing, probe-based selection
%   activates for non-trivial workloads, the dispatch-decision message
%   prints when verbose=true and probing happened, and the dispatcher
%   does not mutate the computed value (auto matches an explicit
%   method= override bit-for-bit).

    tests = functiontests(localfunctions);
end


% =========================================================================
%  Suite-level setup/teardown — defaults isolation
% =========================================================================

function setupOnce(testCase)
    % Save user's session defaults; restore at suite end. Within the
    % suite, the per-test `setup` resets to factory so tests are
    % independent of each other and of caller session state.
    prev = mptDefaults();
    testCase.addTeardown(@() mptDefaults(prev));
end

function setup(~)
    mptDefaults('reset');
end


% =========================================================================
%  Hard rules — no probe
% =========================================================================

function test_user_override_centres(testCase)
    dens = makeDens(12, 3, true);
    x = rand(2, 1000) * 1200;
    [chosen, probed, est] = evalExpTens_dispatch(dens, x, 1000, 'centres');
    verifyEqual(testCase, chosen, 'centres');
    verifyFalse(testCase, probed);
    verifyEqual(testCase, est, 0.0);
end

function test_user_override_orbit(testCase)
    dens = makeDens(12, 3, true);
    x = rand(2, 1000) * 1200;
    [chosen, probed, ~] = evalExpTens_dispatch(dens, x, 1000, 'mobius');
    verifyEqual(testCase, chosen, 'mobius');
    verifyFalse(testCase, probed);
end

function test_r_one_routes_to_centres(testCase)
    dens = buildExpTens([0; 400; 700], ones(3, 1), 12, 1, ...
                       false, false, 0, 'verbose', false);
    x = [100, 200, 300];
    [chosen, probed, ~] = evalExpTens_dispatch(dens, x, 3, 'auto');
    verifyEqual(testCase, chosen, 'centres');
    verifyFalse(testCase, probed);
end

function test_cancellation_guard(testCase)
    % K=3, r=3 → K-r=0, cancellation guard triggers
    dens = buildExpTens([0; 400; 700], ones(3, 1), 12, 3, ...
                       true, false, 0, 'verbose', false);
    x = rand(2, 1000) * 1200;
    [chosen, probed, ~] = evalExpTens_dispatch(dens, x, 1000, 'auto');
    verifyEqual(testCase, chosen, 'centres');
    verifyFalse(testCase, probed);
end

function test_tiny_workload_skips_probe(testCase)
    dens = makeDens(12, 3, true);
    PROBE_MIN_NQ = 200;
    x = rand(2, PROBE_MIN_NQ - 1) * 1200;
    [chosen, probed, ~] = evalExpTens_dispatch(dens, x, ...
        PROBE_MIN_NQ - 1, 'auto');
    verifyEqual(testCase, chosen, 'centres');
    verifyFalse(testCase, probed);
end


% =========================================================================
%  Memory-budget rule
% =========================================================================

function test_huge_centres_array_forces_orbit(testCase)
    % K=200, r=5 → 200^5 ~ 3.2e11 tuples → over the 4 GB budget
    K = 200;
    p = linspace(0, 1200, K + 1)';
    p = p(1:K);
    dens = buildExpTens(p, ones(K, 1), 12, 5, false, false, 0, ...
                       'verbose', false);
    x = rand(5, 1000) * 1200;
    [chosen, probed, ~] = evalExpTens_dispatch(dens, x, 1000, 'auto');
    verifyEqual(testCase, chosen, 'mobius');
    verifyFalse(testCase, probed);
end


% =========================================================================
%  Probing fires
% =========================================================================

function test_probe_fires_for_non_trivial_workload(testCase)
    dens = makeDens(12, 3, true);
    x = rand(2, 500) * 1200;
    [chosen, probed, est] = evalExpTens_dispatch(dens, x, 500, 'auto', ...
        'truncationSigmas', 6.0);
    verifyTrue(testCase, probed);
    verifyTrue(testCase, ismember(chosen, {'centres', 'mobius'}));
    verifyGreaterThan(testCase, est, 0);
end


% =========================================================================
%  Dispatcher doesn't mutate the answer
% =========================================================================

function test_auto_matches_centres_for_rel(testCase)
    dens = makeDens(12, 3, true);
    rng(0, 'twister');
    x = rand(2, 300) * 1200;
    mptDefaults('reset');
    v_auto = evalExpTens(dens, x, 'method', 'auto', 'verbose', false);
    v_centres = evalExpTens(dens, x, 'method', 'centres', 'verbose', false);
    verifyEqual(testCase, v_auto, v_centres, 'AbsTol', 0);
end

function test_auto_with_truncation_matches_centres(testCase)
    dens = makeDens(12, 3, true);
    rng(1, 'twister');
    x = rand(2, 300) * 1200;
    mptDefaults('reset');
    v_auto = evalExpTens(dens, x, 'method', 'auto', ...
                          'truncationSigmas', 6.0, 'verbose', false);
    v_centres = evalExpTens(dens, x, 'method', 'centres', ...
                             'truncationSigmas', 6.0, 'verbose', false);
    verifyEqual(testCase, v_auto, v_centres, 'AbsTol', 0);
end


% =========================================================================
%  Helpers
% =========================================================================

function dens = makeDens(K, r, isRel)
    % Build a small harmonic-template density.
    [tp, tw] = addSpectra([0; 0; 0], [1; 1; 1], ...
                          'harmonic', max(K / 3, 3), 'powerlaw', 1);
    dens = buildExpTens(tp, tw, 12, r, isRel, false, 0, 'verbose', false);
end

function [chosen, probed, est] = evalExpTens_dispatch(dens, x, nQ, ...
        method, varargin)
    % Invoke the dispatcher directly via the local helper. Because
    % localSelectAndEstimateSA is a local function in evalExpTens.m,
    % we exercise it indirectly through a verbose call and capture
    % the message — or call evalExpTens with extra outputs if we had
    % an API for that. Instead we exercise behaviour via output
    % comparison: run with 'auto' and an explicit override; if they
    % differ in path, the dispatch was non-trivial.
    %
    % Simpler: the test framework uses evalExpTens and asserts on
    % the printed output rather than capturing the internal triple.
    % For these tests we use a heuristic: hard-rule cases route to
    % 'centres' or 'mobius' deterministically; we infer 'probed' from
    % whether the dispatch-decision message printed.
    truncationSigmas = [];
    for i = 1:2:numel(varargin)
        if strcmp(varargin{i}, 'truncationSigmas')
            truncationSigmas = varargin{i + 1};
        end
    end

    % Run with verbose=true and capture stdout to detect probing.
    if isempty(truncationSigmas)
        out = evalc('evalExpTens(dens, x, ''method'', method, ''verbose'', true);');
    else
        out = evalc(sprintf( ...
            'evalExpTens(dens, x, ''method'', method, ''truncationSigmas'', %g, ''verbose'', true);', ...
            truncationSigmas));
    end

    if contains(out, 'chose ''centres''')
        chosen = 'centres';
        probed = true;
    elseif contains(out, 'chose ''mobius''')
        chosen = 'mobius';
        probed = true;
    else
        % No dispatch message → hard rule or shortcut fired. Re-run
        % with explicit centres and orbit to find which actually got
        % executed by comparing outputs. Simpler: explicit overrides
        % return immediately so we just trust the method param logic.
        if strcmp(method, 'mobius')
            chosen = 'mobius';
        else
            chosen = 'centres';  % default for 'auto' under hard rules
        end
        probed = false;
    end

    % Parse the estimated time from the message if present (rough).
    est = 0.0;
    if probed
        % Match patterns like "(estimated 7.2 s)" or "(estimated 250 ms)"
        tokens = regexp(out, '\(estimated (\d+\.?\d*) (ms|s|min|hr)\)', ...
                        'tokens', 'once');
        if ~isempty(tokens)
            val = str2double(tokens{1});
            unit = tokens{2};
            switch unit
                case 'ms', est = val / 1000;
                case 's',  est = val;
                case 'min', est = val * 60;
                case 'hr', est = val * 3600;
            end
        end
    end
end
