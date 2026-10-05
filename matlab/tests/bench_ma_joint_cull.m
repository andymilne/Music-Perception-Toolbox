%% bench_ma_joint_cull.m — timing of the joint-centres path of evalMaet
%
%  Twin of python/tools/bench_ma_joint_cull.py. The joint-centres path
%  (taken where an attribute is at r = 1 or carries a kernel covariance)
%  holds a dense chunk to a cache-sized working set and culls on one
%  coordinate where that pays (internal.maCullPlan). This script measures,
%  on densities that take the path, the time per query of the dense
%  evaluation at several chunk sizes and of the culled evaluation, to
%  check two constants on a given machine:
%
%    * cacheChunkBytes (8 MB): the dense chunk is min(kernelChunkBytes,
%      8 MB), so the columns at 1, 2, 4 and 8 MB set kernelChunkBytes; if
%      the 8 MB column is not the fastest, the cap should come down;
%    * pairCost (2.5): the cost of a culled pair relative to a dense one,
%      printed as RATIO (culled time per kept pair over dense time per
%      pair, at 8 MB). Culling is chosen when the kept pairs, at that
%      cost, undercut the dense ones.
%
%  Run from matlab/tests with the toolbox on the path:
%      clear all; rehash; bench_ma_joint_cull
%  Each cell is the minimum of five runs after a warm-up.

bj_capsMB = [1 2 4 8];
bj_nQ = 500;
bj_prevMode = internal.maCullMode();
bj_prevBytes = mptDefaults('kernelChunkBytes');
bj_prevHints = mptDefaults('showHints');
mptDefaults('showHints', false);
bj_cleanup = onCleanup(@() bjRestore(bj_prevMode, bj_prevBytes, bj_prevHints));

rng(0, 'twister');
bj_cells = {};
for bj_cfg = {[2 0.1], [2 0.3], [3 0.1]}
    A = bj_cfg{1}(1); s = bj_cfg{1}(2); N = 10000;
    ps = cell(1, A); ws = cell(1, A);
    for a = 1:A
        ps{a} = 10 * rand(1, N); ws{a} = ones(1, N);
    end
    d = buildMaet(ps, ws, repmat(s, 1, A), ones(1, A), false(1, A), ...
                  false(1, A), zeros(1, A), 'verbose', false);
    bj_cells(end + 1, :) = {sprintf('A=%d r=1 N=%d sigma=%g', A, N, s), ...
                            d, 10 * rand(A, bj_nQ)}; %#ok<SAGROW>
end
for N = [300 3000]
    on = sort(100 * rand(1, N));
    pitch = 60 + 12 * rand(3, N);
    d = buildMaet({on, pitch}, {ones(1, N), ones(3, N)}, [0.3 0.3], [1 2], ...
                  [false false], [false false], [0 0], 'verbose', false);
    bj_cells(end + 1, :) = {sprintf('onset r=1 + 3 pitches r=2, N=%d', N), ...
                            d, [100 * rand(1, bj_nQ); 60 + 12 * rand(2, bj_nQ)]}; %#ok<SAGROW>
end

fprintf(['microseconds per query; dense at each chunk size, culled, auto, ' ...
         'and the culled-to-dense pair-cost ratio\n']);
fprintf('%-36s %s   culled     auto  ratio\n', 'cell', ...
        sprintf('%6dMB ', bj_capsMB));
for ci = 1:size(bj_cells, 1)
    [label, d, X] = bj_cells{ci, :};
    dense = zeros(1, numel(bj_capsMB));
    for k = 1:numel(bj_capsMB)
        mptDefaults('kernelChunkBytes', bj_capsMB(k) * 2^20);
        dense(k) = bjTime(@() bjRun(d, X, 'never'));
    end
    mptDefaults('kernelChunkBytes', bj_prevBytes);
    culled = bjTime(@() bjRun(d, X, 'always'));
    auto = bjTime(@() bjRun(d, X, 'auto'));
    kept = bjKeptPairs(d, X);
    dm = internal.ensureMaetExpensive(d);
    ratio = (culled / kept) / (dense(end) / (double(dm.nJ) * size(X, 2)));
    perQ = 1e3 / size(X, 2);
    fprintf('%-36s %s %8.1f %8.1f %6.2f\n', label, ...
            sprintf('%8.1f', dense * perQ), culled * perQ, auto * perQ, ratio);
end
clear bj_capsMB bj_nQ bj_prevMode bj_prevBytes bj_prevHints bj_cells bj_cfg ...
      A s N ps ws a d on pitch ci label X dense k culled auto kept dm ratio perQ
clear bj_cleanup


function v = bjRun(d, X, mode)
    prev = internal.maCullMode(mode);
    v = evalMaet(d, X, 'method', 'centres', 'truncationSigmas', 6, ...
                 'verbose', false);
    internal.maCullMode(prev);
end


function ms = bjTime(fn)
    fn();
    ms = Inf;
    for i = 1:5
        t0 = tic;
        fn();
        ms = min(ms, toc(t0) * 1e3);
    end
end


function n = bjKeptPairs(d, X)
    dm = internal.ensureMaetExpensive(d);
    A = dm.nAttrs;
    Xc = cell(1, A);
    row = 0;
    for a = 1:A
        Xc{a} = X(row + (1:dm.dimPerAttr(a)), :);
        row = row + dm.dimPerAttr(a);
    end
    prev = internal.maCullMode('always');
    plan = internal.maCullPlan(dm.Centres, Xc, dm.nJ, size(X, 2), ...
        dm.dimPerAttr, dm.sigma, dm.rel, dm.per, dm.period, zeros(1, A), ...
        repmat({'full-image'}, 1, A), 6, 'double');
    internal.maCullMode(prev);
    n = sum(plan.hi - plan.lo);
end


function bjRestore(mode, bytes, hints)
    internal.maCullMode(mode);
    mptDefaults('kernelChunkBytes', bytes);
    mptDefaults('showHints', hints);
end
