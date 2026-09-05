%% test_dispatcher_ip_probe.m — v3 single-multiset IP dispatcher
%
%  Mirrors python/tests/test_dispatcher_ip_probe.py. The dispatcher
%  (internal.selectMaInnerProductMethod) decides between the Möbius
%  method and Bulger's method. Hard rules decide first (correctness /
%  feasibility); otherwise the fitted cost model prices both routes and
%  the cheaper is picked.
%
%  Standalone-runnable.

if ~exist('results', 'var')
    results = {};
    % Defaults isolation when run standalone (when invoked from
    % test_mpt.m the outer wrapper has already isolated defaults).
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults_dip
    cleanupDefaults_dip = mptTestIsolateDefaults(); %#ok<NASGU>
end

% This file's tests assert that the dispatch-announce message appears
% in the captured output of cosSimExpTens calls (via evalc). The
% announce is gated by mptDefaults('showHints'), which test_mpt.m
% silences at suite level. Explicitly enable it here and restore the
% previous state on script exit.
prevSH_dip = mptDefaults('showHints', true);
cleanupSH_dip = onCleanup(@() mptDefaults(prevSH_dip)); %#ok<NASGU>

% --- Semantic equivalence: explicit orbit matches explicit Bulger --
rng(0, 'twister');
ipprobe_px = sort(rand(20, 1) * 100);
rng(1, 'twister');
ipprobe_py = sort(rand(20, 1) * 100);
ipprobe_dx = buildExpTens(ipprobe_px, ones(20, 1), 1, 3, false, false, 1200, ...
    'verbose', false);
ipprobe_dy = buildExpTens(ipprobe_py, ones(20, 1), 1, 3, false, false, 1200, ...
    'verbose', false);
ipprobe_simOrbit = cosSimExpTens(ipprobe_dx, ipprobe_dy, ...
    'method', 'mobius', 'verbose', false);
ipprobe_simPair  = cosSimExpTens(ipprobe_dx, ipprobe_dy, ...
    'method', 'bulger', 'verbose', false);
results{end+1, 1} = 'ip_probe: explicit orbit matches explicit Bulger';
results{end, 2}   = abs(ipprobe_simOrbit - ipprobe_simPair) < 1e-10;

% --- Auto matches explicit paths --------------------------------------
ipprobe_simAuto = cosSimExpTens(ipprobe_dx, ipprobe_dy, 'verbose', false);
results{end+1, 1} = 'ip_probe: auto routing matches explicit orbit';
results{end, 2}   = abs(ipprobe_simAuto - ipprobe_simOrbit) < 1e-10;

% --- Hard rule: r=1 routes pairwise (no orbit machinery) -------------
ipprobe_dx1 = buildExpTens(ipprobe_px, ones(20, 1), 1, 1, false, false, 1200, ...
    'verbose', false);
ipprobe_dy1 = buildExpTens(ipprobe_py, ones(20, 1), 1, 1, false, false, 1200, ...
    'verbose', false);
ipprobe_simR1 = cosSimExpTens(ipprobe_dx1, ipprobe_dy1, ...
    'method', 'auto', 'verbose', false);
results{end+1, 1} = 'ip_probe: r=1 hard rule routes pairwise (call ok)';
results{end, 2}   = isfinite(ipprobe_simR1);

% --- Hard rule: K_y too small for orbit safety (n_min - r < 2) -------
rng(1, 'twister');
ipprobe_pyS = sort(rand(4, 1) * 100);
ipprobe_dyS = buildExpTens(ipprobe_pyS, ones(4, 1), 1, 3, false, false, 1200, ...
    'verbose', false);
ipprobe_simSafety = cosSimExpTens(ipprobe_dx, ipprobe_dyS, ...
    'method', 'auto', 'verbose', false);
results{end+1, 1} = 'ip_probe: n_min-r<2 hard rule routes pairwise';
results{end, 2}   = isfinite(ipprobe_simSafety);

% --- Verbose dispatch message: probe-firing region (r=2, K=5) --------
rng(0, 'twister');
ipprobe_pxP = sort(rand(5, 1) * 100);
rng(1, 'twister');
ipprobe_pyP = sort(rand(5, 1) * 100);
ipprobe_dxP = buildExpTens(ipprobe_pxP, ones(5, 1), 1, 2, false, false, 1200, ...
    'verbose', false);
ipprobe_dyP = buildExpTens(ipprobe_pyP, ones(5, 1), 1, 2, false, false, 1200, ...
    'verbose', false);
ipprobe_evalStr = evalc( ...
    'cosSimExpTens(ipprobe_dxP, ipprobe_dyP, ''verbose'', true);');
results{end+1, 1} = 'ip_probe: verbose message printed when probe fires';
results{end, 2}   = contains(ipprobe_evalStr, 'cosSimExpTens') ...
                 && contains(ipprobe_evalStr, 'chose');

% --- Verbose dispatch message at r=1 (hard rule) ---------------------
% Under the showHints + dispatchScope policy the announce fires for
% both probe-firing AND hard-rule decisions; the only suppressor is
% the once-per-top-level-call throttle and the master showHints
% switch. Verify it announces.
ipprobe_evalStrR1 = evalc( ...
    'cosSimExpTens(ipprobe_dx1, ipprobe_dy1, ''verbose'', true);');
results{end+1, 1} = 'ip_probe: verbose message present at r=1 (hard rule)';
results{end, 2}   = contains(ipprobe_evalStrR1, 'cosSimExpTens') ...
                 && contains(ipprobe_evalStrR1, 'chose');

% --- Verbose dispatch message when user method set -------------------
% Same logic: even when method=bulger is set explicitly, the
% dispatcher still announces the chosen path.
ipprobe_evalStrUser = evalc( ...
    ['cosSimExpTens(ipprobe_dx, ipprobe_dy, ''method'', ''bulger'', ' ...
     '''verbose'', true);']);
results{end+1, 1} = 'ip_probe: verbose message present when method set';
results{end, 2}   = contains(ipprobe_evalStrUser, 'cosSimExpTens') ...
                 && contains(ipprobe_evalStrUser, 'chose');

clear ipprobe_px ipprobe_py ipprobe_dx ipprobe_dy ipprobe_simOrbit ...
      ipprobe_simPair ipprobe_simAuto ipprobe_dx1 ipprobe_dy1 ipprobe_simR1 ...
      ipprobe_pyS ipprobe_dyS ipprobe_simSafety ipprobe_pxP ipprobe_pyP ...
      ipprobe_dxP ipprobe_dyP ipprobe_evalStr ipprobe_evalStrR1 ...
      ipprobe_evalStrUser

% Restore caller's pre-test defaults eagerly when run
% standalone (fires the helper's onCleanup destructor on
% script exit; guarded so we don't clear a like-named
% variable when this file was run from test_mpt.m, where
% the standalone branch was skipped).
if exist('cleanupDefaults_dip', 'var')
    clear cleanupDefaults_dip
end
