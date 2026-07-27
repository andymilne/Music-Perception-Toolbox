%% test_ma_factored.m
%  Tests the factored multi-attribute centres path (localMaEvalFactored in
%  evalExpTens), the twin of Python's _ma_eval_factored. The joint density
%  factors within each event as a product across attributes, so the centres
%  route evaluates sum_events prod_attributes S_a^(event) --- each factor
%  through the culled kernel --- without accumulating the joint tuple set.
%
%  Tests (flat attributes; nested attributes are covered by test_nested):
%    - method='centres' (factored) agrees with method='mobius' across
%      attribute counts, event counts, read-arities, and geometry modes.
%      Möbius is the independent reference: it never builds the joint and
%      shares the flat query convention.
%    - Ragged events (NaN-dropped slots, differing valid-slot patterns)
%      agree: absent slots are zero-weight on a shared enumeration.
%    - Fall-back: an r_a = 1 attribute routes to the joint accumulator
%      (factored returns []); the eval still matches Möbius.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

rng(9);

% Max relative error over non-negligible query points, centres vs mobius.
relErr = @(d, X) localRelErr(d, X);

% --- Flat relative: A, K, N sweep ---
cfgs = {2, 8, 1; 3, 8, 1; 2, 20, 1; 2, 7, 5; 2, 8, 20};
for i = 1:size(cfgs, 1)
    A = cfgs{i,1}; K = cfgs{i,2}; N = cfgs{i,3};
    d = localBuildFlat(A, K, N, 2, true, false, 0.0);
    X = localQuery(d, 60);
    results{end+1,1} = sprintf(...
        'MA factored flat rel A=%d K=%d N=%d: centres matches mobius (1e-4)', ...
        A, K, N);
    results{end,2} = relErr(d, X) < 1e-4;
end

% --- Flat absolute ---
d = localBuildFlat(2, 10, 3, 2, false, false, 0.0);
results{end+1,1} = 'MA factored flat abs: centres matches mobius (1e-4)';
results{end,2}   = relErr(d, localQuery(d, 60)) < 1e-4;

% --- Flat periodic ---
d = localBuildFlat(2, 8, 3, 2, true, true, 1200.0);
results{end+1,1} = 'MA factored flat per: centres matches mobius (1e-4)';
results{end,2}   = relErr(d, localQuery(d, 60)) < 1e-4;

% --- Higher tuple size r = 3, 4 ---
for r = [3, 4]
    d = localBuildFlat(2, 7, 2, r, true, false, 0.0);
    results{end+1,1} = sprintf(...
        'MA factored flat rel r=%d: centres matches mobius (1e-4)', r);
    results{end,2} = relErr(d, localQuery(d, 60)) < 1e-4;
end

% --- Ragged events (NaN-dropped slots, differing valid patterns) ---
A = 2; K = 8; N = 6;
P = cell(1, A);
for a = 1:A
    Pa = sort(3600 * rand(K, N), 1);
    for n = 1:N
        nd = randi([0, 2]);
        if nd > 0
            drop = randperm(K, nd);
            Pa(drop, n) = NaN;
        end
    end
    P{a} = Pa;
end
d = buildExpTens(P, [], repmat(15.0, 1, A), repmat(2, 1, A), ...
    true(1, A), false(1, A), zeros(1, A), 'verbose', false);
results{end+1,1} = 'MA factored ragged rel: centres matches mobius (1e-4)';
results{end,2}   = relErr(d, localQuery(d, 60)) < 1e-4;

% --- Fall-back: r = 1 attribute routes to the joint accumulator ---
d = localBuildFlat(2, 8, 1, 1, true, false, 0.0);
X = localQuery(d, 20);
vC = evalExpTens(d, X, 'method', 'centres', 'verbose', false);
vM = evalExpTens(d, X, 'method', 'mobius',  'verbose', false);
mask = vM > 1e-3 * max(vM);
results{end+1,1} = 'MA factored r=1 fall-back: centres matches mobius (1e-4)';
results{end,2}   = max(abs((vC(mask) - vM(mask)) ./ vM(mask))) < 1e-4;

if standalone
    pass = cellfun(@(x) x, results(:,2));
    fprintf('%s: %d/%d passed\n', mfilename, sum(pass), numel(pass));
    for i = 1:size(results,1)
        if ~results{i,2}, fprintf('  FAIL: %s\n', results{i,1}); end
    end
end


% ======================================================================
%  Local helpers
% ======================================================================

function d = localBuildFlat(A, K, N, r, isRelP, isPerP, period)
    P = cell(1, A);
    for a = 1:A
        P{a} = sort(3600 * rand(K, N), 1);
    end
    d = buildExpTens(P, [], repmat(15.0, 1, A), repmat(r, 1, A), ...
        repmat(logical(isRelP), 1, A), repmat(logical(isPerP), 1, A), ...
        repmat(period, 1, A), 'verbose', false);
end

function X = localQuery(d, nQ)
    % Queries near the joint centres (jittered) so the density is
    % non-negligible at the compared points.
    de  = internal.ensureExpTensExpensive(d);
    idx = randi(de.nJ, 1, nQ);
    X = zeros(de.dim, nQ);
    rs = 1;
    for a = 1:numel(de.dimPerAttr)
        da = de.dimPerAttr(a);
        if da == 0, continue; end
        re = rs + da - 1;
        X(rs:re, :) = de.Centres{a}(:, idx) + 7.5 * randn(da, nQ);
        rs = re + 1;
    end
end

function e = localRelErr(d, X)
    vC = evalExpTens(d, X, 'method', 'centres', 'verbose', false);
    vM = evalExpTens(d, X, 'method', 'mobius',  'verbose', false);
    mask = vM > 1e-3 * max(vM);
    if ~any(mask)
        e = 0;
        return;
    end
    e = max(abs((vC(mask) - vM(mask)) ./ vM(mask)));
end
