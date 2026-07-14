%% test_dispatch_oom_guard.m — single-image infeasibility guard (inner product)
%
%  The cosine dispatch forces the single-image Bulger route when the
%  Möbius method is unavailable (r above the shipped orbit order, or the
%  K-r precision floor). At high tuple order the Bulger tuple-pair kernel
%  can be infeasibly large; auto-dispatch must then raise
%  mpt:dispatch:singleImageInfeasible rather than risk an out-of-memory
%  crash. Explicit method='bulger' is the user's own choice and is
%  honoured without the guard.
%
%  Twin of python tests/test_ma_eval_cost_model.py IP-guard cases.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults_og
    cleanupDefaults_og = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

% 1. r > 8 with large K -> forced bulger, infeasible -> raises
ok = false;
try
    cosSimExpTens(mk(15, 9, false, false, 0), mk(15, 9, false, false, 0), ...
        'method', 'auto', 'verbose', false);
catch err
    ok = strcmp(err.identifier, 'mpt:dispatch:singleImageInfeasible');
end
results{end+1, 1} = 'OOM guard: r=9 K=15 auto raises singleImageInfeasible'; %#ok<*AGROW>
results{end, 2} = ok;

% 2. Precision floor (K = r+1) at high r -> forced bulger, infeasible -> raises
ok = false;
try
    cosSimExpTens(mk(9, 8, false, false, 0), mk(9, 8, false, false, 0), ...
        'method', 'auto', 'verbose', false);
catch err
    ok = strcmp(err.identifier, 'mpt:dispatch:singleImageInfeasible');
end
results{end+1, 1} = 'OOM guard: r=8 K=9 (precision floor) auto raises';
results{end, 2} = ok;

% 3. User override method='bulger' at infeasible shape -> honoured (no raise
%    from the dispatch guard; a real run may still fail, but the selector
%    must not second-guess an explicit request). We only check the guard
%    does not fire by confirming a different outcome than the auto raise:
%    the call proceeds past dispatch. Use a shape that is infeasible to
%    actually compute, so we expect either success or a NON-guard error.
ok = true;
try
    cosSimExpTens(mk(15, 9, false, false, 0), mk(15, 9, false, false, 0), ...
        'method', 'bulger', 'verbose', false);
catch err
    % Any error other than the dispatch guard is acceptable here (the
    % explicit request was honoured by the selector; downstream may fail).
    ok = ~strcmp(err.identifier, 'mpt:dispatch:singleImageInfeasible');
end
results{end+1, 1} = 'OOM guard: user override bulger bypasses the guard';
results{end, 2} = ok;

% 4. Feasible normal case (r=3, small K) -> no raise, normal result
ok = false;
try
    s = cosSimExpTens(mk(8, 3, false, false, 0), mk(8, 3, false, false, 0), ...
        'method', 'auto', 'verbose', false);
    ok = isfinite(s);
catch
    ok = false;
end
results{end+1, 1} = 'OOM guard: r=3 K=8 feasible does not raise';
results{end, 2} = ok;

% 5. estimateMaJointWorkingSetBytes sanity: grows past budget at high r
b_small = internal.estimateMaJointWorkingSetBytes([3], [8], false);
b_huge  = internal.estimateMaJointWorkingSetBytes([11], [20], false);
results{end+1, 1} = 'OOM guard: joint working-set estimate ordering';
results{end, 2} = b_huge > b_small && b_huge > internal.dispatchMemBudget();

%% ---- Standalone reporting ----
if standalone
    nFail = 0;
    for ii = 1:size(results, 1)
        if ~isequal(results{ii, 2}, true), nFail = nFail + 1; end
    end
    fprintf('test_dispatch_oom_guard: %d passed, %d failed (of %d)\n', ...
        size(results, 1) - nFail, nFail, size(results, 1));
    for ii = 1:size(results, 1)
        if ~isequal(results{ii, 2}, true)
            fprintf('  FAIL  %s\n', results{ii, 1});
        end
    end
    if exist('cleanupDefaults_og', 'var'), clear cleanupDefaults_og; end
end

% ---- Local helpers (script-scope; must follow all executable code) ----
function d = mk(K, r, rel, per, P)
    g = (1:K)' * 37.0;           % deterministic distinct values
    if per
        g = mod(g, P);
        d = buildExpTens(g, ones(K, 1), 40.0, r, rel, true, P, ...
            'verbose', false);
    else
        d = buildExpTens(g, ones(K, 1), 6.0, r, rel, false, 0.0, ...
            'verbose', false);
    end
end
