%% test_nested_mobius_eval.m — per-level Möbius point evaluation of nested densities
%
%  evalExpTens('method', 'mobius') on a nested density runs the per-level
%  Möbius evaluator (mobius.evalNestedAttrOrbit), which must agree with
%  the tuple-centres route on every nested shape: any depth, symmetric or
%  ordered levels, absolute or any co-transposition unit, periodic or not,
%  either wrap, ragged (NaN-padded) events, and a nested attribute
%  tensored with flat ones. Mirror of Python
%  tests/test_nested_mobius_eval.py; the reference block shares its
%  numbers with that file.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.
%  mptTestIsolateDefaults sets truncationSigmas = Inf (accuracy floor).

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults_nme
    cleanupDefaults_nme = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

nme_P = 12.0;
nme_T2 = repelem(0:1, 3);
nme_T3 = repelem(0:2, 3);
nme_T3L = [0 0; 0 0; 1 0; 1 0; 2 1; 2 1; 3 1; 3 1];

% --- two levels: every shape agrees with the centres route ---
nme_shapes = { nme_T2, [1 2], [0 0], 6; ...
               nme_T3, [2 2], [0 0], 9; ...
               nme_T3, [2 2], [1 0], 9; ...    % inner unit
               nme_T3, [2 2], [0 1], 9; ...    % outer unit
               nme_T3, [2 3], [0 0], 9 };
nme_syms = { [true true], [true false], [false true] };
for nme_i = 1:size(nme_shapes, 1)
    for nme_per = [false true]
        for nme_j = 1:numel(nme_syms)
            nme_tags = nme_shapes{nme_i, 1}; nme_r = nme_shapes{nme_i, 2};
            nme_rel = nme_shapes{nme_i, 3}; nme_K = nme_shapes{nme_i, 4};
            nme_sym = nme_syms{nme_j};
            d = nmeDensity(nme_tags, nme_r, nme_sym, nme_rel, nme_per, nme_K, 1, 0.7, 'full-image', nme_P);
            X = nmeQueries(d, 1);
            if nme_per && any(nme_rel), nme_tol = 1e-7; else, nme_tol = 1e-10; end
            results{end+1, 1} = sprintf( ...
                'nested mobius eval: r=[%s] sym=[%s] rel=[%s] per=%d matches centres', ...
                num2str(nme_r), num2str(nme_sym), num2str(nme_rel), nme_per); %#ok<*SAGROW>
            results{end, 2} = nmeRoutesAgree(d, X, nme_tol);
        end
    end
end

% --- three levels ---
nme_rels = { [0 0 0], [0 0 1], [0 1 0], [1 0 0] };
nme_cfg = { [true true true], false; [true false true], true; [false true false], false };
for nme_i = 1:numel(nme_rels)
    for nme_j = 1:size(nme_cfg, 1)
        nme_rel = nme_rels{nme_i}; nme_sym = nme_cfg{nme_j, 1}; nme_per = nme_cfg{nme_j, 2};
        d = nmeDensity(nme_T3L, [2 2 2], nme_sym, nme_rel, nme_per, 8, 2, 0.6, 'full-image', nme_P);
        if d.dim == 0
            continue;
        end
        X = nmeQueries(d, 2);
        if nme_per && any(nme_rel), nme_tol = 1e-7; else, nme_tol = 1e-10; end
        results{end+1, 1} = sprintf( ...
            'nested mobius eval: three levels rel=[%s] sym=[%s] per=%d matches centres', ...
            num2str(nme_rel), num2str(nme_sym), nme_per);
        results{end, 2} = nmeRoutesAgree(d, X, nme_tol);
    end
end

% --- single-image wrap honoured ---
d = nmeDensity(nme_T3, [2 2], [true true], [0 0], true, 9, 3, 0.4, 'single-image', nme_P);
results{end+1, 1} = 'nested mobius eval: single-image wrap honoured';
results{end, 2} = nmeRoutesAgree(d, nmeQueries(d, 3), 1e-10);

% --- ragged events with NaN padding ---
rng(4, 'twister');
p = sort(nme_P * rand(6, 3), 1);
p(6, 1) = NaN;
p(5:6, 3) = NaN;
spec = struct('tags', nme_T2, 'r', [1 2], 'sym', [true true], 'rel', [0 0]);
d = buildExpTens({p}, {[]}, 'specs', {spec}, 'sigma', 0.7, 'isPer', false, ...
                 'period', 0, 'verbose', false);
results{end+1, 1} = 'nested mobius eval: ragged events with NaN padding';
results{end, 2} = nmeRoutesAgree(d, nmeQueries(d, 4), 1e-10);

% --- nested tensored with flat attributes ---
rng(5, 'twister');
p0 = sort(nme_P * rand(6, 3), 1);
p1 = sort(nme_P * rand(3, 3), 1);
p2 = 10 * rand(1, 3);
specs = { struct('tags', nme_T2, 'r', [1 2], 'sym', [true true], 'rel', [0 1]), ...
          struct('r', 2, 'sym', true, 'rel', false), ...
          struct('r', 1, 'sym', true, 'rel', false) };
d = buildExpTens({p0, p1, p2}, {[], [], []}, 'specs', specs, ...
                 'sigma', [0.5 0.8 1.0], 'isPer', [true false false], ...
                 'period', [nme_P 0 0], 'verbose', false);
results{end+1, 1} = 'nested mobius eval: nested tensored with flat attributes';
results{end, 2} = nmeRoutesAgree(d, nmeQueries(d, 5), 1e-7);

% --- auto keeps centres; mobius accepted ---
d = nmeDensity(nme_T2, [1 2], [true true], [0 0], false, 6, 6, 0.7, 'full-image', nme_P);
X = nmeQueries(d, 6);
va = evalExpTens(d, X, 'verbose', false);
vc = evalExpTens(d, X, 'method', 'centres', 'verbose', false);
vm = evalExpTens(d, X, 'method', 'mobius', 'verbose', false);
results{end+1, 1} = 'nested mobius eval: auto keeps centres and mobius is accepted';
results{end, 2} = max(abs(va(:) - vc(:))) <= 1e-12 * max(1, max(abs(vc(:)))) ...
                  && max(abs(vm(:) - vc(:))) <= 1e-10 * max(abs(vc(:)));

% --- ordered flat attribute still refuses mobius ---
rng(7, 'twister');
p = sort(nme_P * rand(4, 2), 1);
d = buildExpTens({p}, {[]}, 'specs', {struct('r', 2, 'sym', false, 'rel', false)}, ...
                 'sigma', 0.5, 'isPer', false, 'period', 0, 'verbose', false);
results{end+1, 1} = 'nested mobius eval: ordered flat attribute still refuses mobius';
results{end, 2} = throwsErrorWithId(@() evalExpTens(d, zeros(d.dim, 2), 'method', 'mobius', 'verbose', false), ...
                                    'mpt:evalExpTens:orderedMobius');

% --- reference values shared with the Python test ---
p = [1.0; 2.5; 4.0; 7.0; 8.2; 11.0];
w = [1; 0.5; 1; 1; 0.7; 1];
tags = [0; 0; 0; 1; 1; 1];
x = [0.5 3 7 11; 2 2.7 8 1; 1.5 4 9 3; 7 8 6 4];
v1 = mobius.evalNestedAttrOrbit(p, w, tags, [2 2], [true true], [], 0.7, x, 'truncationSigmas', Inf);
v2 = mobius.evalNestedAttrOrbit(p, w, tags, [2 2], [true false], 1, 0.7, x(1:2, :), 'truncationSigmas', Inf);
v3 = mobius.evalNestedAttrOrbit(p, w, tags, [2 2], [false true], 2, 0.7, x(1:3, :), ...
                                'is_per', true, 'period', 12, 'truncationSigmas', Inf);
ref1 = [2.00328713036504e-15 2.01827912258021e-05 1.38777878078145e-17 4.71134784883262e-17];
ref2 = [0.884583285039345 1.76658427991777 8.14454249730343e-08 5.86413326215107e-15];
ref3 = [1.44730864634257e-05 0.00179025660699691 0.00635877431113622 0.000329925782403053];
results{end+1, 1} = 'nested mobius eval: cross-language reference values';
results{end, 2} = all(abs(v1(:).' - ref1) <= 1e-9 * abs(ref1) + 1e-16) ...
                  && all(abs(v2(:).' - ref2) <= 1e-9 * abs(ref2)) ...
                  && all(abs(v3(:).' - ref3) <= 1e-9 * abs(ref3));

% --- much cheaper than the centres route at r = (3, 3) ---
rng(8, 'twister');
tags = repelem(0:3, 3);
p = sort(nme_P * rand(12, 4), 1);
spec = struct('tags', tags, 'r', [3 3], 'sym', [true true], 'rel', [0 0]);
d = buildExpTens({p}, {[]}, 'specs', {spec}, 'sigma', 0.7, 'isPer', false, ...
                 'period', 0, 'verbose', false);
X = nme_P * rand(d.dim, 50);
tic; vc = evalExpTens(d, X, 'method', 'centres', 'verbose', false); tc = toc;
tic; vm = evalExpTens(d, X, 'method', 'mobius', 'verbose', false); tm = toc;
results{end+1, 1} = 'nested mobius eval: per-level route is cheaper than centres at r=(3,3)';
results{end, 2} = max(abs(vm(:) - vc(:))) <= 1e-7 * max(abs(vc(:))) && tm < tc;

if standalone
    nPass = sum([results{:, 2}]);
    nFail = size(results, 1) - nPass;
    for ii = 1:size(results, 1)
        if results{ii, 2}
            fprintf('  PASS  %s\n', results{ii, 1});
        else
            fprintf('  FAIL  %s\n', results{ii, 1});
        end
    end
    fprintf('\n=== test_nested_mobius_eval: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults_nme
    if nFail > 0
        error('test_nested_mobius_eval:failed', '%d test(s) failed.', nFail);
    end
end


function d = nmeDensity(tags, r, sym, rel, per, K, seed, sigma, wrap, P)
    rng(seed, 'twister');
    p = sort(P * rand(K, 2), 1);
    spec = struct('tags', tags, 'r', r, 'sym', sym, 'rel', rel);
    d = buildExpTens({p}, {[]}, 'specs', {spec}, 'sigma', sigma, 'isPer', per, ...
                     'period', P, 'wrap', {wrap}, 'verbose', false);
end


function X = nmeQueries(d, seed)
    rng(seed + 100, 'twister');
    % Three queries far from the data and three on actual tuple centres
    % (all attributes stacked, in slot order), so the density carries
    % mass at the near queries. A query assembled from independently
    % drawn coordinates lands at ~1e-14 of the peak, where both routes
    % are at their noise floor and the relative comparison is meaningless.
    X = -2 + 16 * rand(d.dim, 6);
    if d.dim > 0
        dm = internal.ensureExpTensExpensive(d);
        c = vertcat(dm.Centres{:});
        X(:, 1:3) = c(:, randi(size(c, 2), 1, 3)) + 0.3 * randn(d.dim, 3);
    end
end


function ok = nmeRoutesAgree(d, X, tol)
    % Both routes at a 40-sigma width: at the accuracy-floor width the
    % centres route drops every tuple kernel below 1e-12, and over a few
    % thousand tuples the dropped mass reaches 1e-9 of the maximum, which
    % is truncation, not disagreement (diag: 5.6e-14 at 40 sigma against
    % 1.8e-9 at the floor on the r = [2 3] row).
    vc = evalExpTens(d, X, 'method', 'centres', 'truncationSigmas', 40, 'verbose', false);
    vm = evalExpTens(d, X, 'method', 'mobius', 'truncationSigmas', 40, 'verbose', false);
    scale = max(max(abs(vc(:))), 1e-300);
    err = max(abs(vm(:) - vc(:))) / scale;
    ok = err <= tol;
    if ~ok
        fprintf('    routes differ: max|mobius - centres| / max|centres| = %.3e (tol %.0e)\n', err, tol);
        fprintf('    centres: %s\n    mobius:  %s\n', mat2str(vc(:).', 8), mat2str(vm(:).', 8));
    end
end
