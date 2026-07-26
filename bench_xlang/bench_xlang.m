%% bench_xlang.m
%  Cross-language benchmark runner (MATLAB side).
%
%  Generates deterministic inputs (identical to bench_xlang.py), times
%  evalExpTens and cosSimExpTens across a small grid varying one axis
%  at a time from a base configuration, and writes CSV.
%
%  Usage from the matlab/ directory (or after addpath):
%    bench_xlang;                                       % bench_matlab.csv, auto method
%    bench_xlang('bench_matlab_v2.csv');                % custom output path
%    bench_xlang('bench_matlab_bulger.csv', 'bulger');  % force cossim to bulger
%    bench_xlang('bench_matlab_mobius.csv', 'mobius');  % force cossim to mobius
%
%  Forcing a cossim method isolates route-vs-routing discrepancies:
%  if the two languages disagree at method='auto' but agree at
%  method='bulger', the disagreement is which route the two auto
%  dispatchers picked, not the routes themselves. If they still
%  disagree at method='bulger', the bulger path itself has a
%  cross-language mismatch.
%
%  See BENCH_SPEC.md for the grid and input formulae.

function bench_xlang(outPath, method)
    if nargin < 1 || isempty(outPath)
        outPath = 'bench_matlab.csv';
    end
    if nargin < 2 || isempty(method)
        method = 'auto';
    end
    if ~ismember(method, {'auto', 'bulger', 'mobius'})
        error('bench_xlang:badMethod', ...
            "method must be 'auto', 'bulger', or 'mobius'; got '%s'.", method);
    end
    PERIOD = 1200;

    % Silence the abs-per single-image opt-in warning throughout: several
    % benchmark rows opt into single-image at sigma/P above the threshold.
    prevW = warning('off', 'buildExpTens:absPerSingleImage');
    cleanupW = onCleanup(@() warning(prevW));

    % Silence dispatch messages: with adaptive-inner-loop timing the
    % bench runs each call thousands of times, and 'chose X path' fires
    % on every top-level entry (the throttle resets per top-level
    % call). showHints=false disables the whole dispatch-message
    % facility globally for this run; correctness is unaffected. Saved
    % previous state and restored via onCleanup so the caller's session
    % keeps whatever they had before.
    prevDefaults = mptDefaults('showHints', false);
    cleanupD = onCleanup(@() mptDefaults(prevDefaults));

    cases = localBuildCases();
    nCases = numel(cases);
    fprintf(['Running %d unique configurations with cossim method=''%s'', ' ...
             'each measured for eval and cossim (adaptive inner loop, ~30 ms window, min of 7)...\n'], ...
        nCases, method);

    rows = {};
    for idx = 1:nCases
        c = cases{idx};
        label = c.label;
        fprintf('  [%d/%d] %s: ', idx, nCases, label);

        % eval
        try
            [tEval, vEval, nJEval, nInEval] = localRunEval(c, PERIOD);
            csEval = sum(abs(vEval(:)));
            rows{end+1} = localMakeRow(label, 'eval', tEval, csEval, ...
                nJEval, nInEval, c); %#ok<AGROW>
            fprintf('eval %.1fus(x%d)  ', tEval * 1e6, nInEval);
        catch e
            fprintf('eval FAIL (%s)  ', e.message);
        end

        % cossim
        % Graceful skip: forcing bulger at A>=3 materialises the joint
        % tuple set (n_j^A pair matrix), 250+ GB at A=3, K=8, r=2. auto
        % and mobius handle A=3 fine (mobius factorises across attributes).
        if strcmp(method, 'bulger') && c.A >= 3
            fprintf('cossim SKIP (bulger joint tuple set too large at A>=3)\n');
        else
            try
                [tCos, vCos, nJCos, nInCos] = localRunCossim(c, PERIOD, method);
                rows{end+1} = localMakeRow(label, 'cossim', tCos, vCos, ...
                    nJCos, nInCos, c); %#ok<AGROW>
                fprintf('cossim %.1fus(x%d)\n', tCos * 1e6, nInCos);
            catch e
                fprintf('cossim FAIL (%s)\n', e.message);
            end
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

function [tBest, result, nInner] = localAdaptiveTime(fn)
%LOCALADAPTIVETIME  Adaptive-inner-loop timing with outer-minimum.
%
%   For sub-millisecond calls, best-of-N on single-call measurements is
%   dominated by timer-resolution noise and OS scheduling jitter. Fix:
%   loop N times inside a single tic-toc so the measured window is
%   ~30 ms, drowning out per-measurement jitter, then divide by N.
%   n_outer = 7 independent measurements are collected and the minimum
%   is returned as the estimate (minimum reflects the intrinsic
%   per-call cost when the CPU is not contested).
%
%   Twin of adaptive_time() in bench_xlang.py.
    TARGET_INNER_S = 0.030;   % ~30 ms per outer window
    N_OUTER        = 7;
    N_WARMUP       = 2;
    MAX_INNER      = 1e6;

    % First call warms caches, does JIT compilation, and returns the
    % result we hand back to the caller.
    result = fn();

    % Extra warm-ups so the sizing measurement isn't polluted by cold
    % cache/JIT effects.
    for k = 1:N_WARMUP
        fn();
    end

    % Quick size: one measurement to estimate single-call time, then
    % compute nInner so a single outer window is ~TARGET_INNER_S.
    t0 = tic;
    fn();
    singleS = max(toc(t0), 1e-9);
    nInner = max(1, floor(TARGET_INNER_S / singleS));
    nInner = min(nInner, MAX_INNER);

    % Timed measurements
    ts = zeros(1, N_OUTER);
    for j = 1:N_OUTER
        t0 = tic;
        for k = 1:nInner
            fn();
        end
        ts(j) = toc(t0) / nInner;
    end
    tBest = min(ts);
end


function [tBest, result] = localBestOf3(fn)
%LOCALBESTOF3  Legacy: single-call best-of-3. Retained for reference;
%   new callers should use localAdaptiveTime.
    result = fn();   % warm
    ts = zeros(1, 3);
    for i = 1:3
        t0 = tic;
        r = fn(); %#ok<NASGU>
        ts(i) = toc(t0);
    end
    tBest = min(ts);
end


function [tBest, v, nJ, nInner] = localRunEval(c, PERIOD)
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

    [tBest, v, nInner] = localAdaptiveTime( ...
        @() evalExpTens(d, X, 'verbose', false));
end


function [tBest, v, nJ, nInner] = localRunCossim(c, PERIOD, method)
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

    [tBest, v, nInner] = localAdaptiveTime(@() cosSimExpTens(dx, dy, ...
        'method', method, 'verbose', false));
end


% =========================================================================
% Row assembly and CSV output
% =========================================================================

function row = localMakeRow(label, operation, elapsed, checksum, nJ, nInner, c)
    row = struct( ...
        'label', label, ...
        'operation', operation, ...
        'elapsed_s', elapsed, ...
        'checksum', checksum, ...
        'n_j', nJ, ...
        'n_inner', nInner, ...
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
    fprintf(fid, ['label,operation,elapsed_s,checksum,n_j,n_inner,' ...
                  'sigma_over_P,r,isRel,isPer,wrap,A,N,K,nQ\n']);
    for i = 1:numel(rows)
        r = rows{i};
        fprintf(fid, '%s,%s,%.9g,%.15g,%d,%d,%.6g,%d,%s,%s,%s,%d,%d,%d,%d\n', ...
            r.label, r.operation, r.elapsed_s, r.checksum, r.n_j, ...
            r.n_inner, ...
            r.sigma_over_P, r.r, r.isRel, r.isPer, r.wrap, ...
            r.A, r.N, r.K, r.nQ);
    end
    fclose(fid);
end
