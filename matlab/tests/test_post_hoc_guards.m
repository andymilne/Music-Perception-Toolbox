%% test_post_hoc_guards.m — the postHocGuards default and its wiring
%
%  Two checks in the toolbox inspect a route's output after computing it
%  and may then recompute by another route: the nested accuracy guard in
%  nestedContract/combinePair and the corruption check in the flat cosine
%  path. With either active the measured cost of the Mobius route is not
%  the cost of choosing it, because a diverting check pays for both
%  routes. The default switches them off so the routes can be timed as
%  the alternatives they are.
%
%  Python twin: tests/test_post_hoc_guards.py. The Python file also
%  exercises combinePair directly; that function is local to
%  nestedContract.m and unreachable from here, so this file covers the
%  default's registration, validation, and the flat cosine wiring.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    % Defaults isolation when run standalone (when invoked from
    % test_mpt.m the outer wrapper has already isolated defaults).
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

% --- 1. Default is on ---
phg_prev = mptDefaults('postHocGuards');
results{end+1, 1} = 'postHocGuards: factory default is true'; %#ok<*AGROW>
results{end, 2}   = islogical(phg_prev) && isscalar(phg_prev) && phg_prev;

% --- 2. Round-trips ---
mptDefaults('postHocGuards', false);
phg_ok = ~mptDefaults('postHocGuards');
mptDefaults('postHocGuards', true);
phg_ok = phg_ok && mptDefaults('postHocGuards');
results{end+1, 1} = 'postHocGuards: set/get round-trips';
results{end, 2}   = phg_ok;

% --- 3. Rejects a non-logical value ---
phg_ok = false;
try
    mptDefaults('postHocGuards', 'yes');
catch phg_err
    phg_ok = strcmp(phg_err.identifier, 'mptDefaults:badValue');
end
results{end+1, 1} = 'postHocGuards: rejects a non-logical value';
results{end, 2}   = phg_ok;

% --- 4. Survives a reset ---
mptDefaults('postHocGuards', false);
mptDefaults('reset');
results{end+1, 1} = 'postHocGuards: reset restores true';
results{end, 2}   = logical(mptDefaults('postHocGuards'));

% --- 5. Flat cosine agrees with the guard off ---
%  The corruption check fires only on a broken value (non-finite,
%  negative auto-IP, or |cosine| > 1), so on a healthy input the two
%  settings must give the same answer bit for bit. Fixed pitches rather
%  than rand(), so this file leaves the global RNG stream untouched for
%  the files that run after it.
phg_P  = 1200;
phg_p1 = [0; 137; 291; 452; 613; 761; 908; 1063];
phg_p2 = [24; 160; 318; 470; 640; 788; 931; 1085];
phg_w  = ones(8, 1);

mptDefaults('postHocGuards', true);
phg_sOn  = cosSimExpTens(phg_p1, phg_w, phg_p2, phg_w, 30, 3, ...
                         false, true, phg_P, 'verbose', false);
mptDefaults('postHocGuards', false);
phg_sOff = cosSimExpTens(phg_p1, phg_w, phg_p2, phg_w, 30, 3, ...
                         false, true, phg_P, 'verbose', false);
mptDefaults('postHocGuards', phg_prev);

results{end+1, 1} = 'postHocGuards: flat cosine unchanged on a healthy input';
results{end, 2}   = isequal(phg_sOn, phg_sOff);

if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:,2)));
    nFail = numel(results(:,1)) - nPass;
    fprintf('\n=== test_post_hoc_guards: %d passed, %d failed (of %d) ===\n', ...
        nPass, nFail, numel(results(:,1)));
    % Restore caller's pre-test defaults eagerly
    % (fires the helper's onCleanup destructor before
    % the conditional error below halts execution).
    clear cleanupDefaults
    if nFail > 0
        for ii = 1:size(results, 1)
            if ~isequal(results{ii, 2}, true)
                fprintf('  FAIL  %s\n', results{ii, 1});
            end
        end
    end
end
