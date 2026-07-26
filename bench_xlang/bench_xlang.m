%% bench_xlang.m
%  Cross-language benchmark runner (MATLAB side).
%
%  Generates deterministic inputs (identical to bench_xlang.py), times
%  evalExpTens and cosSimExpTens across a small grid varying one axis
%  at a time from a base configuration, and writes CSV.
%
%  Usage from the matlab/ directory (or after addpath):
%    bench_xlang;                         % writes bench_matlab.csv
%    bench_xlang('bench_matlab_v2.csv');  % custom output path
%
%  See BENCH_SPEC.md for the grid and input formulae.

function bench_xlang(outPath)
    if nargin < 1
        outPath = 'bench_matlab.csv';
    end
    PERIOD = 1200;

    % Silence the abs-per single-image opt-in warning throughout: several
    % benchmark rows opt into single-image at sigma/P above the threshold.
    prevW = warning('off', 'buildExpTens:absPerSingleImage');
    cleanupW = onCleanup(@() warning(prevW));

    cases = localBuildCases();
    nCases = numel(cases);
    fprintf('Running %d unique configurations, each measured for eval and cossim (best-of-3)...\n', ...
        nCases);

    rows = {};
    for idx = 1:nCases
        c = cases{idx};
        label = c.label;
        fprintf('  [%d/%d] %s: ', idx, nCases, label);

        % eval
        try
            [tEval, vEval, nJEval] = localRunEval(c, PERIOD);
            csEval = sum(abs(vEval(:)));
            rows{end+1} = localMakeRow(label, 'eval', tEval, csEval, nJEval, c); %#ok<AGROW>
            fprintf('eval %.1fms  ', tEval * 1000);
        catch e
            fprintf('eval FAIL (%s)  ', e.message);
        end

        % cossim
        try
            [tCos, vCos, nJCos] = localRunCossim(c, PERIOD);
            rows{end+1} = localMakeRow(label, 'cossim', tCos, vCos, nJCos, c); %#ok<AGROW>
            fprintf('cossim %.1fms\n', tCos * 1000);
        catch e
            fprintf('cossim FAIL (%s)\n', e.message);
        end
    end

    localWriteCsv(outPath, rows);
    fprintf('\nWrote %d rows to %s\n', numel(rows), outPath);
end


% =========================================================================
% Deterministic input generation (matches bench_xlang.py)
% =========================================================================

function [pAll, wAll] = localMakeInputs(A, N, K, period)
    S = K + N + A;
    pAll = cell(1, A);
    wAll = cell(1, A);
    for a = 1:A
        pa = zeros(K, N);
        wa = zeros(K, N);
        for n = 1:N
            for j = 1:K
                % Python uses 0-based indices; map to 0-based here for parity.
                jp = j - 1;
                np_ = n - 1;
                ap_ = a - 1;
                idx = mod(jp + 3 * np_ + 7 * ap_, S);
                pa(j, n) = period * idx / S;
                wa(j, n) = 0.7 + 0.3 * cos( ...
                    (jp + 2 * np_ + 5 * ap_) / S * pi);
            end
        end
        pAll{a} = pa;
        wAll{a} = wa;
    end
end


function X = localMakeQueries(dim, nQ, period)
    S = dim + nQ + 1;
    X = zeros(dim, nQ);
    for d = 1:dim
        for q = 1:nQ
            dp = d - 1;
            qp = q - 1;
            X(d, q) = period * (0.5 + 0.4 * sin( ...
                (dp + 2 * qp + 1) / S * pi));
        end
    end
end


% =========================================================================
% Case iteration
% =========================================================================

function cases = localBuildCases()
    base = struct( ...
        'sigma_over_P', 0.10, 'r', 2, 'isRel', false, 'isPer', true, ...
        'wrap', 'full-image', 'A', 1, 'N', 1, 'K', 8, 'nQ', 10);

    cases = {};
    seenFP = {};

    cases = localAdd(cases, seenFP, 'base', base);
    seenFP = {localFingerprint(base)};

    for v = [0.002, 0.005, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
        c = base; c.sigma_over_P = v;
        [cases, seenFP] = localTryAdd(cases, seenFP, sprintf('sigma_over_P=%g', v), c);
    end

    for v = [1, 2, 3, 4]
        c = base; c.r = v;
        [cases, seenFP] = localTryAdd(cases, seenFP, sprintf('r=%d', v), c);
    end

    relPer = {[false false], [false true], [true false], [true true]};
    relPerLbl = {'F,F', 'F,T', 'T,F', 'T,T'};
    for kk = 1:4
        c = base; c.isRel = relPer{kk}(1); c.isPer = relPer{kk}(2);
        [cases, seenFP] = localTryAdd(cases, seenFP, ...
            sprintf('isRel=%s&isPer=%s', localBoolStr(c.isRel), localBoolStr(c.isPer)), c);
    end

    for wr = {'full-image', 'single-image'}
        c = base; c.wrap = wr{1};
        [cases, seenFP] = localTryAdd(cases, seenFP, sprintf('wrap=%s', wr{1}), c);
    end

    for v = [1, 2, 3]
        c = base; c.A = v;
        [cases, seenFP] = localTryAdd(cases, seenFP, sprintf('A=%d', v), c);
    end

    for v = [1, 5, 20, 50, 100]
        c = base; c.N = v;
        [cases, seenFP] = localTryAdd(cases, seenFP, sprintf('N=%d', v), c);
    end

    for v = [4, 8, 16, 50, 100]
        c = base; c.K = v;
        [cases, seenFP] = localTryAdd(cases, seenFP, sprintf('K=%d', v), c);
    end

    for v = [1, 10, 100]
        c = base; c.nQ = v;
        [cases, seenFP] = localTryAdd(cases, seenFP, sprintf('nQ=%d', v), c);
    end
end


function s = localBoolStr(b)
    if b, s = 'True'; else, s = 'False'; end
end


function fp = localFingerprint(c)
    fp = sprintf('%.6f|%d|%d|%d|%s|%d|%d|%d|%d', ...
        c.sigma_over_P, c.r, c.isRel, c.isPer, c.wrap, ...
        c.A, c.N, c.K, c.nQ);
end


function cases = localAdd(cases, ~, label, cfg)
    cfg.label = label;
    cases{end+1} = cfg;
end


function [cases, seenFP] = localTryAdd(cases, seenFP, label, cfg)
    fp = localFingerprint(cfg);
    if any(strcmp(seenFP, fp))
        return
    end
    seenFP{end+1} = fp;
    cases = localAdd(cases, seenFP, label, cfg);
end


% =========================================================================
% Timing and measurement
% =========================================================================

function [tBest, result] = localBestOf3(fn)
    result = fn();   % warm
    ts = zeros(1, 3);
    for i = 1:3
        t0 = tic;
        r = fn(); %#ok<NASGU>
        ts(i) = toc(t0);
    end
    tBest = min(ts);
end


function [tBest, v, nJ] = localRunEval(c, PERIOD)
    sigma = c.sigma_over_P * PERIOD;
    if c.isPer, periodVal = PERIOD; else, periodVal = 0; end

    [pAll, wAll] = localMakeInputs(c.A, c.N, c.K, PERIOD);
    dimPer = c.r - double(c.isRel);
    dim = c.A * dimPer;
    X = localMakeQueries(dim, c.nQ, PERIOD);

    d = buildExpTens(pAll, wAll, ...
        repmat(sigma, 1, c.A), repmat(c.r, 1, c.A), ...
        repmat(c.isRel, 1, c.A), repmat(c.isPer, 1, c.A), ...
        repmat(periodVal, 1, c.A), ...
        'wrap', c.wrap, 'verbose', false, 'lazy', false);
    if isfield(d, 'nJ'); nJ = double(d.nJ); else; nJ = -1; end

    % Force the centres path on both sides so any residual value
    % disagreement is genuinely in the centres kernel and not a
    % dispatch-routing difference across languages.
    [tBest, v] = localBestOf3(@() evalExpTens(d, X, ...
        'method', 'centres', 'verbose', false));
end


function [tBest, v, nJ] = localRunCossim(c, PERIOD)
    sigma = c.sigma_over_P * PERIOD;
    if c.isPer, periodVal = PERIOD; else, periodVal = 0; end

    [pAll, wAll] = localMakeInputs(c.A, c.N, c.K, PERIOD);
    pAll2 = cell(1, c.A);
    for a = 1:c.A
        pAll2{a} = pAll{a} + 37.5;
    end

    dx = buildExpTens(pAll, wAll, ...
        repmat(sigma, 1, c.A), repmat(c.r, 1, c.A), ...
        repmat(c.isRel, 1, c.A), repmat(c.isPer, 1, c.A), ...
        repmat(periodVal, 1, c.A), ...
        'wrap', c.wrap, 'verbose', false, 'lazy', false);
    dy = buildExpTens(pAll2, wAll, ...
        repmat(sigma, 1, c.A), repmat(c.r, 1, c.A), ...
        repmat(c.isRel, 1, c.A), repmat(c.isPer, 1, c.A), ...
        repmat(periodVal, 1, c.A), ...
        'wrap', c.wrap, 'verbose', false, 'lazy', false);
    if isfield(dx, 'nJ'); nJ = double(dx.nJ); else; nJ = -1; end

    [tBest, v] = localBestOf3(@() cosSimExpTens(dx, dy, 'verbose', false));
end


% =========================================================================
% Row assembly and CSV output
% =========================================================================

function row = localMakeRow(label, operation, elapsed, checksum, nJ, c)
    row = struct( ...
        'label', label, ...
        'operation', operation, ...
        'elapsed_s', elapsed, ...
        'checksum', checksum, ...
        'n_j', nJ, ...
        'sigma_over_P', c.sigma_over_P, ...
        'r', c.r, ...
        'isRel', localBoolStr(c.isRel), ...
        'isPer', localBoolStr(c.isPer), ...
        'wrap', c.wrap, ...
        'A', c.A, 'N', c.N, 'K', c.K, 'nQ', c.nQ);
end


function localWriteCsv(outPath, rows)
    fid = fopen(outPath, 'w');
    if fid < 0
        error('bench_xlang:write', 'Cannot open %s for writing', outPath);
    end
    fprintf(fid, ['label,operation,elapsed_s,checksum,n_j,' ...
                  'sigma_over_P,r,isRel,isPer,wrap,A,N,K,nQ\n']);
    for i = 1:numel(rows)
        r = rows{i};
        fprintf(fid, '%s,%s,%.9g,%.15g,%d,%.6g,%d,%s,%s,%s,%d,%d,%d,%d\n', ...
            r.label, r.operation, r.elapsed_s, r.checksum, r.n_j, ...
            r.sigma_over_P, r.r, r.isRel, r.isPer, r.wrap, ...
            r.A, r.N, r.K, r.nQ);
    end
    fclose(fid);
end
