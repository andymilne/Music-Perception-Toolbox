function benchNestedEval(outFile)
%BENCHNESTEDEVAL  Timing grid for pricing the per-level Möbius evaluator.
%
%   benchNestedEval()            prints the CSV to the console
%   benchNestedEval('file.csv')  writes it to a file
%
%   Twin of python/tools/bench_nested_eval.py. evalExpTens on a nested
%   density has two routes: the tuple-centres route (materialise every
%   nested tuple, M_perm per event, and sum a Gaussian per centre) and the
%   per-level Möbius evaluator (mobius.evalNestedAttrOrbit), which touches
%   no tuple. The eval selector keeps the centres route under 'auto' until
%   the Möbius route has a fitted cost row; this script MEASURES the grid
%   on which that row is to be fitted. It does not fit.
%
%   Run it on a quiet machine. Each cell reports the median of repeats
%   after a warm-up. Columns as in the Python twin: the structural
%   quantities the two routes work over (m_perm, d, k, n, n_q, bell_sum,
%   n_u), the shape flags, and the two medians in ms.

    if nargin < 1
        outFile = '';
    end
    here = fileparts(mfilename('fullpath'));
    addpath(fileparts(here));
    prevHints = mptDefaults('showHints');
    mptDefaults('showHints', false);
    cleanup = onCleanup(@() mptDefaults('showHints', prevHints)); %#ok<NASGU>
    ts = mptDefaults('truncationSigmas');
    P = 12.0;
    rng(0, 'twister');

    if isempty(outFile)
        fid = 1;
    else
        fid = fopen(outFile, 'w');
    end
    fprintf(fid, 'BEGIN_CSV\n');
    fprintf(fid, ['groups,group_size,r_levels,sym,rel_unit,per,k,n,n_q,d,m_perm,' ...
                  'bell_sum,n_u,centres_ms,mobius_ms\n']);

    shapes = {2, 3; 3, 3; 4, 3; 3, 4; 4, 4; 2, 5};
    rls = {[1 2], [2 2], [2 3], [3 2], [3 3], [1 3], [2 4]};
    syms = {[true true], [true false], [false true]};
    relUnits = {[], 2, 1};             % [] absolute; 1-based level
    for iS = 1:size(shapes, 1)
        groups = shapes{iS, 1}; gsize = shapes{iS, 2};
        for iR = 1:numel(rls)
            rLevels = rls{iR};
            if rLevels(1) > gsize || rLevels(2) > groups
                continue;
            end
            for iY = 1:numel(syms)
                sym = syms{iY};
                for iU = 1:numel(relUnits)
                    relUnit = relUnits{iU};
                    if isequal(relUnit, 1) && rLevels(1) < 2
                        continue;
                    end
                    for per = [false true]
                        K = groups * gsize;
                        tags = repelem(0:groups - 1, gsize);
                        rel = [0 0];
                        if ~isempty(relUnit)
                            rel(relUnit) = 1;
                        end
                        for N = [4 32]
                            p = sort(P * rand(K, N), 1);
                            spec = struct('tags', tags, 'r', rLevels, 'sym', sym, 'rel', rel);
                            sigma = 0.6;
                            d = buildExpTens({p}, {[]}, 'specs', {spec}, 'sigma', sigma, ...
                                             'isPer', per, 'period', P, 'verbose', false);
                            if d.dim == 0
                                continue;
                            end
                            % buildExpTens is lazy: nJ is an expensive field.
                            % Materialise once, as Python's cached d.n_j does,
                            % and time the centres route on the materialised
                            % density so that neither language pays the tuple
                            % enumeration inside the timed call.
                            dm = internal.ensureExpTensExpensive(d);
                            mPerm = dm.nJ / N;
                            bellSum = 0;
                            for l = 1:2
                                if sym(l) && rLevels(l) >= 2
                                    bellSum = bellSum + numel(mobius.getSetPartitionsWithMobius(rLevels(l)));
                                end
                            end
                            nU = localGridNodes(d, relUnit, rLevels, sigma, per, ts, P);
                            for nQ = [1 20 200]
                                X = P * rand(d.dim, nQ);
                                if mPerm * N * nQ > 4e7
                                    tC = NaN;
                                else
                                    tC = localMedianMs(@() evalExpTens(dm, X, 'method', 'centres', 'verbose', false));
                                end
                                tM = localMedianMs(@() evalExpTens(d, X, 'method', 'mobius', 'verbose', false));
                                if isempty(relUnit), ru = -1; else, ru = relUnit - 1; end
                                fprintf(fid, '%d,%d,%dx%d,%d%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.4f,%.4f\n', ...
                                        groups, gsize, rLevels(1), rLevels(2), sym(1), sym(2), ...
                                        ru, per, K, N, nQ, d.dim, mPerm, bellSum, nU, tC, tM);
                            end
                        end
                    end
                end
            end
        end
    end
    fprintf(fid, 'END_CSV\n');
    if fid ~= 1
        fclose(fid);
    end
end


function ms = localMedianMs(fn)
    fn();
    t = zeros(1, 5);
    for i = 1:5
        tic; fn(); t(i) = toc * 1e3;
    end
    ms = median(t);
end


function nU = localGridNodes(d, relUnit, rLevels, sigma, per, ts, P)
    if isempty(relUnit)
        nU = 1;
        return;
    end
    s = prod(rLevels(1:relUnit));
    spp = internal.resolveSamplesPerSigma([], max(2, s), ts);
    if per
        nU = max(64, ceil(P / sigma * spp));
    else
        v = d.pAttr{1}(:);
        span = max(v(~isnan(v))) - min(v(~isnan(v))) + 16 * sigma;
        nU = max(64, ceil(max(span, 1) / sigma * spp));
    end
end
