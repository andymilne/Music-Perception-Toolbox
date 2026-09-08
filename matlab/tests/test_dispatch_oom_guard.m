%% test_dispatch_oom_guard.m — single-image infeasibility guard (inner product)
%
%  The cosine dispatch forces the single-image Bulger route when the
%  Möbius method is unavailable (r above the shipped orbit order, or the
%  feasibility bound). At high tuple order the Bulger tuple-pair kernel
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

% 2. A collection barely larger than the tuple size (K = r+1) is no
%    longer a reason to refuse the Mobius method: accuracy is governed
%    by truncationSigmas, so the route is chosen on cost and the call
%    completes rather than raising.
ok = false;
try
    cosSimExpTens(mk(9, 8, false, false, 0), mk(9, 8, false, false, 0), ...
        'method', 'auto', 'verbose', false);
    ok = true;
catch
    ok = false;
end
results{end+1, 1} = 'Dispatch: r=8 K=9 chooses on cost, no infeasibility raise';
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

% 6. estimateMaJointWorkingSetBytes on an ordered attribute counts
%    C(K, r) joint tuples, not K!/(K-r)!: at r = K that is one tuple.
b_ord     = internal.estimateMaJointWorkingSetBytes([11], [20], false, false);
b_ordFull = internal.estimateMaJointWorkingSetBytes([11], [11], false, false);
results{end+1, 1} = 'OOM guard: ordered attr counts combinations';
results{end, 2} = abs(b_ord - nchoosek(20, 11) * 11 * 2 * 8) < 0.5 ...
    && b_ordFull == 1 * 11 * 2 * 8 && b_ord < b_huge;

% 7. A bound ordered 9-tuple density (three attributes, one read
%    relative; one tuple per attribute per event) must pass
%    auto-dispatch -- its enumerated tuple set is one tuple per
%    attribute, not 9! -- and match the closed-form single-pair cosine
%    exp(-sum_a Q_a(delta_a) / (4 sigma_a^2)), the relative attribute's
%    delta projected off the common shift.
ok = false;
try
    L = 9;
    sigC = [0.3, 0.4, 0.5];
    rng9 = RandStream('mt19937ar', 'Seed', 3);
    xs = cell(1, 3); ys = cell(1, 3);
    for a = 1:3
        xs{a} = randn(rng9, 1, L);
        ys{a} = xs{a} + 0.2 * randn(rng9, 1, L);
    end
    dx = localOrderedBound9(xs, sigC);
    dy = localOrderedBound9(ys, sigC);
    got = cosSimExpTens(dx, dy, 'verbose', false);
    q = 0.0;
    for a = 1:3
        d = xs{a} - ys{a};
        if a == 2                      % relative attribute
            d = d - mean(d);
        end
        q = q + (d * d') / (4 * sigC(a)^2);
    end
    ok = abs(got - exp(-q)) < 1e-10;
catch
    ok = false;
end
results{end+1, 1} = 'OOM guard: ordered bound r=9 cosine matches closed form';
results{end, 2} = ok;

% 8. A bound ordered 11-tuple routes evalExpTens to centres (ordered
%    hard rule) and evaluates: its joint tuple set is one tuple, so no
%    infeasibility guard may fire.
ok = false;
try
    x11 = (0:10);
    [pB11, wB11, spB11] = unpackPreMaet(bindEvents({x11}, [], 11));
    d11 = buildExpTens(pB11, wB11, 'specs', spB11, 'sigma', 0.3, ...
        'isPer', false, 'period', 0, 'verbose', false);
    v11 = evalExpTens(d11, x11(:), 'verbose', false);
    ok = all(isfinite(v11)) && max(v11) > 0;
catch
    ok = false;
end
results{end+1, 1} = 'OOM guard: ordered bound r=11 eval routes and runs';
results{end, 2} = ok;

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

function d = localOrderedBound9(vals, sigC)
%LOCALORDEREDBOUND9  One bound super-event: three ordered 9-tuples,
%   the second read relative (a common shift quotiented).
    L = numel(vals{1});
    [pB, wB, spB] = unpackPreMaet(bindEvents(vals, [], L, ...
        'relOuter', [false, true, false]));
    d = buildExpTens(pB, wB, 'specs', spB, 'sigma', sigC, ...
        'isPer', [false, false, false], 'period', [0, 0, 0], ...
        'verbose', false);
end
