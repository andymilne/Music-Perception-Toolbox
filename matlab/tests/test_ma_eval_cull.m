%% test_ma_eval_cull.m — culling and cache-sized chunks on the joint-centres path
%
%  The joint-centres path of evalMaet (taken where an attribute is at
%  r = 1 or carries a kernel covariance) evaluates each query against
%  only the centres within the truncation width on one culling
%  coordinate (internal.maCullPlan), and holds a dense chunk to a
%  cache-sized working set. Neither may change a value: a culled pair is
%  one the truncation would have set to zero, so culled and dense
%  evaluations agree to rounding. Pinned here: that agreement on every
%  kind of culling coordinate (absolute, periodic on a single image,
%  relative, nested inner unit, kernel covariance), beside an attribute
%  that is never culled on (periodic on the full image), at the boundary
%  of the window, across the wrap of a periodic coordinate, and across
%  group and chunk boundaries; and that the decision to cull follows the
%  kernel width. Mirror of Python tests/test_ma_eval_cull.py.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults_mc
    cleanupDefaults_mc = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end
mc_prevMode = internal.maCullMode();
mc_restoreMode = onCleanup(@() internal.maCullMode(mc_prevMode));

% --- every kind of culling coordinate, each beside an onset at r = 1 ---
rng(11, 'twister');
mc_N = 300;
mc_on = sort(60 * rand(1, mc_N));
mc_one = @(p) ones(size(p));
mc_shapes = struct('name', {}, 'dens', {});
p1 = 60 + 12 * rand(1, mc_N);
mc_shapes(end + 1) = struct('name', 'absolute, r = 1 twice', 'dens', ...
    buildMaet({mc_on, p1}, {mc_one(mc_on), mc_one(p1)}, [0.1 0.3], [1 1], ...
              [false false], [false false], [0 0], 'verbose', false));
p1 = 60 + 12 * rand(3, mc_N);
mc_shapes(end + 1) = struct('name', 'absolute at r = 2', 'dens', ...
    buildMaet({mc_on, p1}, {mc_one(mc_on), mc_one(p1)}, [0.1 0.3], [1 2], ...
              [false false], [false false], [0 0], 'verbose', false));
p1 = 60 + 12 * rand(4, mc_N);
mc_shapes(end + 1) = struct('name', 'relative at r = 3', 'dens', ...
    buildMaet({mc_on, p1}, {mc_one(mc_on), mc_one(p1)}, [0.1 0.3], [1 3], ...
              [false true], [false false], [0 0], 'verbose', false));
p1 = 12 * rand(2, mc_N);
mc_shapes(end + 1) = struct('name', 'periodic absolute, single image', 'dens', ...
    buildMaet({mc_on, p1}, {mc_one(mc_on), mc_one(p1)}, [0.1 0.3], [1 2], ...
              [false false], [false true], [0 12], ...
              'wrap', {'full-image', 'single-image'}, 'verbose', false));
p1 = round(12 * rand(2, mc_N));
mc_shapes(end + 1) = struct('name', 'periodic absolute, full image, tabulated', 'dens', ...
    buildMaet({mc_on, p1}, {mc_one(mc_on), mc_one(p1)}, [0.1 0.3], [1 2], ...
              [false false], [false true], [0 12], 'verbose', false));
p1 = 12 * rand(3, mc_N);
mc_shapes(end + 1) = struct('name', 'periodic relative', 'dens', ...
    buildMaet({mc_on, p1}, {mc_one(mc_on), mc_one(p1)}, [0.1 0.3], [1 2], ...
              [false true], [false true], [0 12], 'verbose', false));
mc_A = randn(2, 2);
mc_Sig = mc_A * mc_A.' * 0.05 + 0.02 * eye(2);
p1 = 10 * rand(2, mc_N);
mc_shapes(end + 1) = struct('name', 'kernel covariance', 'dens', ...
    buildMaet({p1, mc_on}, {mc_one(p1), mc_one(mc_on)}, {mc_Sig, 0.1}, [2 1], ...
              [false false], [false false], [0 0], [false true], ...
              'verbose', false));
p1 = sort(12 * rand(6, mc_N), 1);
mc_specs = {struct('tags', repelem(0:1, 3), 'r', [1 2], 'exch', [true true], ...
                   'rel', [0 1]), ...
            struct('r', 1, 'exch', true, 'rel', false)};
mc_shapes(end + 1) = struct('name', 'nested inner unit', 'dens', ...
    buildMaet({p1, mc_on}, {[], []}, 'specs', mc_specs, 'sigma', [0.5 0.1], ...
              'per', [false false], 'period', [0 0], 'verbose', false));
p1 = 60 + 12 * rand(3, mc_N);
p1(3, 1:3:end) = NaN;
mc_w = rand(1, mc_N);
mc_w(1:5:end) = 0;
mc_shapes(end + 1) = struct('name', 'absent values and zero weights', 'dens', ...
    buildMaet({mc_on, p1}, {mc_w, ones(3, mc_N)}, [0.1 0.3], [1 2], ...
              [false false], [false false], [0 0], 'verbose', false));

for mc_k = [6, Inf]
    for mc_i = 1:numel(mc_shapes)
        mc_d = mc_shapes(mc_i).dens;
        mc_X = mcQueries(mc_d, 400, 1);
        results{end + 1, 1} = sprintf('ma_eval_cull: culled equals dense, %s, k = %g', ...
            mc_shapes(mc_i).name, mc_k);
        results{end, 2} = mcCullMatchesDense(mc_d, mc_X, 1e-12, ...
            'truncationSigmas', mc_k);
    end
end

for mc_i = [2, 3, 6]
    mc_d = mc_shapes(mc_i).dens;
    mc_X = mcQueries(mc_d, 400, 2);
    results{end + 1, 1} = sprintf('ma_eval_cull: culled equals dense in single precision, %s', ...
        mc_shapes(mc_i).name);
    results{end, 2} = mcCullMatchesDense(mc_d, mc_X, 1e-5, ...
        'truncationSigmas', 6, 'kernelPrecision', 'single');
end

% --- the relative window is sqrt(2) k sigma ---
% The culling coordinate is the relative attribute's one difference
% (r = 2, where Q equals D^2 / 2 exactly), so pairs whose difference lies
% between k sigma and sqrt(2) k sigma carry weight between exp(-k^2 / 2)
% and exp(-k^2 / 4) and are lost to a window without the sqrt(2). The
% attribute at r = 1 puts the density on the joint-centres path and, with
% a wide kernel over one value, adds nothing to the exponent.
rng(3, 'twister');
mc_kr = 3;
mc_sr = 0.5;
p1 = 100 * rand(2, 400);
mc_d = buildMaet({p1, zeros(1, 400)}, {ones(2, 400), ones(1, 400)}, ...
    [mc_sr 50], [2 1], [true false], [false false], [0 0], 'verbose', false);
mc_dm = internal.ensureMaetExpensive(mc_d);
mc_c = mc_dm.Centres{1};
mc_pick = randi(size(mc_c, 2), 1, 600);
mc_X = [mc_c(:, mc_pick) + (2.8 * rand(1, 600) - 1.4) * mc_kr * mc_sr; zeros(1, 600)];
results{end + 1, 1} = 'ma_eval_cull: the relative window is wide enough';
results{end, 2} = mcCullMatchesDense(mc_d, mc_X, 1e-12, 'truncationSigmas', mc_kr);

% --- the boundary of the window ---
% Queries at exactly k sigma from a centre on the culling coordinate, and
% a hair either side, are kept or dropped as the truncation itself would
% keep or drop them.
mc_kb = 3;
mc_sb = 0.5;
mc_onb = 0:10:190;
mc_d = buildMaet({mc_onb, zeros(1, 20)}, {ones(1, 20), ones(1, 20)}, ...
    [mc_sb 5], [1 1], [false false], [false false], [0 0], 'verbose', false);
mc_offs = mc_kb * mc_sb * [1 - 1e-12, 1, 1 + 1e-12];
mc_xs = mc_onb(6) + [mc_offs, -mc_offs];
mc_X = repmat([mc_xs; zeros(1, 6)], 1, 8);
results{end + 1, 1} = 'ma_eval_cull: the boundary of the window';
results{end, 2} = mcCullMatchesDense(mc_d, mc_X, 0, 'truncationSigmas', mc_kb);

% --- a periodic window wraps ---
% Centres and queries crowd both ends of the cycle, so each window
% crosses the wrap.
rng(5, 'twister');
mc_P = 4;
mc_ph = [0.2 * rand(1, 100), mc_P - 0.2 * rand(1, 100)];
p1 = 60 + 12 * rand(1, 200);
mc_d = buildMaet({mc_ph, p1}, {ones(1, 200), ones(1, 200)}, [0.02 0.3], ...
    [1 1], [false false], [true false], [mc_P 0], ...
    'wrap', {'single-image', 'full-image'}, 'verbose', false);
mc_X = [0.1 * rand(1, 200), mc_P - 0.1 * rand(1, 200); 60 + 12 * rand(1, 400)];
results{end + 1, 1} = 'ma_eval_cull: a periodic window wraps';
results{end, 2} = mcCullMatchesDense(mc_d, mc_X, 1e-12, 'truncationSigmas', 6);

% --- groups and chunks ---
% A tiny chunk budget splits the culled pairs into many groups and the
% dense evaluation into many chunks; neither changes a value.
mc_d = mc_shapes(2).dens;
mc_X = mcQueries(mc_d, 300, 6);
internal.maCullMode('never');
mc_whole = evalMaet(mc_d, mc_X, 'method', 'centres', 'truncationSigmas', 6, ...
                    'verbose', false);
mc_prevBytes = mptDefaults('kernelChunkBytes');
mptDefaults('kernelChunkBytes', 4096);
mc_dense = evalMaet(mc_d, mc_X, 'method', 'centres', 'truncationSigmas', 6, ...
                    'verbose', false);
internal.maCullMode('always');
mc_culled = evalMaet(mc_d, mc_X, 'method', 'centres', 'truncationSigmas', 6, ...
                     'verbose', false);
mptDefaults('kernelChunkBytes', mc_prevBytes);
internal.maCullMode(mc_prevMode);
results{end + 1, 1} = 'ma_eval_cull: chunks and groups do not change a value';
results{end, 2} = max(abs(mc_dense - mc_whole)) <= 1e-12 * max(abs(mc_whole)) ...
    && max(abs(mc_culled - mc_whole)) <= 1e-12 * max(abs(mc_whole));

% --- queries beyond every centre ---
internal.maCullMode('always');
mc_v = evalMaet(mc_shapes(1).dens, [1e4 * ones(1, 50); 66 * ones(1, 50)], ...
                'method', 'centres', 'truncationSigmas', 6, 'verbose', false);
internal.maCullMode(mc_prevMode);
results{end + 1, 1} = 'ma_eval_cull: queries beyond every centre evaluate to zero';
results{end, 2} = all(mc_v == 0);

% --- full-image periodic attributes are never culled on ---
rng(7, 'twister');
p0 = 4 * rand(1, 500);
p1 = 12 * rand(1, 500);
mc_d = buildMaet({p0, p1}, {ones(1, 500), ones(1, 500)}, [0.02 0.1], [1 1], ...
    [false false], [true true], [4 12], 'verbose', false);
internal.maCullMode('always');
mc_plan = mcPlan(mc_d, 4 * rand(2, 400), 6);
internal.maCullMode(mc_prevMode);
results{end + 1, 1} = 'ma_eval_cull: full-image periodic attributes are never culled on';
results{end, 2} = isempty(mc_plan);

% --- the decision to cull follows the kernel width ---
rng(8, 'twister');
mc_onD = 100 * rand(1, 3000);
mc_pD = 60 + 12 * rand(1, 3000);
mc_X = [100 * rand(1, 500); 60 + 12 * rand(1, 500)];
mc_narrow = buildMaet({mc_onD, mc_pD}, {ones(1, 3000), ones(1, 3000)}, ...
    [0.05 0.3], [1 1], [false false], [false false], [0 0], 'verbose', false);
mc_wide = buildMaet({mc_onD, mc_pD}, {ones(1, 3000), ones(1, 3000)}, ...
    [8 3], [1 1], [false false], [false false], [0 0], 'verbose', false);
internal.maCullMode('auto');
mc_planN = mcPlan(mc_narrow, mc_X, 6);
mc_planW = mcPlan(mc_wide, mc_X, 6);
internal.maCullMode(mc_prevMode);
results{end + 1, 1} = 'ma_eval_cull: a narrow kernel is culled, each query meeting few centres';
results{end, 2} = ~isempty(mc_planN) && mean(mc_planN.hi - mc_planN.lo) < 0.05 * 3000;
results{end + 1, 1} = 'ma_eval_cull: a wide kernel is evaluated dense';
results{end, 2} = isempty(mc_planW);

clear mc_N mc_on mc_one mc_shapes mc_A mc_Sig mc_specs mc_w mc_k mc_i mc_d ...
      mc_X mc_kr mc_sr mc_dm mc_c mc_pick mc_kb mc_sb mc_onb mc_offs mc_xs ...
      mc_P mc_ph mc_whole mc_prevBytes mc_dense mc_culled mc_v mc_plan ...
      mc_onD mc_pD mc_narrow mc_wide mc_planN mc_planW p0 p1
clear mc_restoreMode mc_prevMode

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
    fprintf('\n=== test_ma_eval_cull: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults_mc
    if nFail > 0
        error('test_ma_eval_cull:failed', '%d test(s) failed.', nFail);
    end
end


function X = mcQueries(d, n, seed)
    % Half near actual centres, half spread over their range.
    rng(seed, 'twister');
    dm = internal.ensureMaetExpensive(d);
    c = vertcat(dm.Centres{:});
    h = floor(n / 2);
    nearX = c(:, randi(size(c, 2), 1, h)) + 0.2 * randn(size(c, 1), h);
    lo = min(c, [], 2) - 1;
    hi = max(c, [], 2) + 1;
    farX = lo + (hi - lo) .* rand(size(c, 1), n - h);
    X = [nearX, farX];
end


function ok = mcCullMatchesDense(d, X, tol, varargin)
    % Culled ('always') against dense ('never'), relative to the largest
    % dense value, which must be positive for the comparison to mean
    % anything.
    prev = internal.maCullMode('never');
    dense = evalMaet(d, X, 'method', 'centres', 'verbose', false, varargin{:});
    internal.maCullMode('always');
    culled = evalMaet(d, X, 'method', 'centres', 'verbose', false, varargin{:});
    internal.maCullMode(prev);
    scale = max(max(abs(dense)), 1e-300);
    err = max(abs(culled - dense)) / scale;
    ok = err <= tol && any(dense > 0);
    if ~ok
        fprintf('    culled and dense differ: %.3e relative (tol %.0e)\n', err, tol);
    end
end


function plan = mcPlan(d, X, k)
    % The culling plan for flat density d at queries X (stacked rows).
    dm = internal.ensureMaetExpensive(d);
    A = dm.nAttrs;
    Xc = cell(1, A);
    row = 0;
    for a = 1:A
        Xc{a} = X(row + (1:dm.dimPerAttr(a)), :);
        row = row + dm.dimPerAttr(a);
    end
    if isfield(dm, 'wrap') && ~isempty(dm.wrap)
        wrapCell = dm.wrap;
    else
        wrapCell = repmat({'full-image'}, 1, A);
    end
    plan = internal.maCullPlan(dm.Centres, Xc, dm.nJ, size(X, 2), ...
        dm.dimPerAttr, dm.sigma, dm.rel, dm.per, dm.period, zeros(1, A), ...
        wrapCell, k, 'double');
end
