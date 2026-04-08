function estSec = estimateCompTime(nPairs, dim, label, verbose)
%ESTIMATECOMPTIME Estimate computation time for kernel evaluation.
%
%   estSec = estimateCompTime(nPairs, dim, label):
%   estSec = estimateCompTime(nPairs, dim, label, verbose):
%   Estimates how long a kernel evaluation involving 'nPairs' total
%   (tuple, query) pair evaluations will take, where each pair involves
%   a 'dim'-dimensional difference vector, quadratic form, exp, and
%   weighted accumulation.
%
%   The estimate is based on a micro-benchmark of the actual computation
%   pattern (implicit expansion, squaring, summing, exp, matrix-vector
%   multiply), calibrated separately for each dimensionality. The
%   calibration result is cached across calls within a MATLAB session.
%
%   Inputs:
%     nPairs  — Total number of (tuple, query) pair evaluations (scalar).
%               For evalExpTens: nJ * nQ.
%               For cosSimExpTens: sum of nJ*nK across the three inner
%               products.
%     dim     — Dimensionality of the difference vectors (positive integer)
%     label   — Description string for the console message. If empty (''),
%               the function runs silently (useful for accumulating
%               estimates across multiple calls).
%     verbose — Optional logical (default: true). If false, suppresses
%               all console output regardless of label.
%
%   Output:
%     estSec  — Estimated time in seconds
%
%   This function is used internally by evalExpTens, cosSimExpTens, and
%   plotExpTens_demo.

if nargin < 4
    verbose = true;
end

persistent rateCache;  % containers.Map: dim -> pairsPerSec

% Initialize the cache on first call
if isempty(rateCache)
    rateCache = containers.Map('KeyType', 'int32', 'ValueType', 'double');
end

% Calibrate for this dimensionality if not already cached
dimKey = int32(dim);
if ~rateCache.isKey(dimKey)
    % Run a small representative workload that exactly mirrors the
    % dominant operations in evalExpTens / cosSimExpTens:
    %   1. Implicit-expansion subtraction  (dim x nCal x nCal)
    %   2. Element-wise squaring + sum     (quadratic form)
    %   3. exp                             (Gaussian kernel)
    %   4. Matrix-vector multiply          (weighted accumulation)
    %
    % nCal is chosen to be large enough for stable timing but small
    % enough to complete in well under a second.
    nCal = 1000;
    U_cal = randn(dim, nCal);
    V_cal = randn(dim, nCal);
    w_cal = randn(1, nCal);

    % Warm-up run (avoids JIT and memory allocation skewing the timing)
    D_ = reshape(U_cal, dim, nCal, 1) - reshape(V_cal, dim, 1, nCal);
    Q_ = sum(D_.^2, 1);
    E_ = reshape(exp(-Q_(:)), nCal, nCal);
    v_ = w_cal * E_; %#ok<NASGU>

    % Timed run
    t = tic;
    D = reshape(U_cal, dim, nCal, 1) - reshape(V_cal, dim, 1, nCal);
    Q = sum(D.^2, 1);
    E = reshape(exp(-Q(:)), nCal, nCal);
    v = w_cal * E; %#ok<NASGU>
    elapsed = toc(t);

    % Pairs processed in the benchmark
    calPairs = double(nCal) * double(nCal);
    rateCache(dimKey) = calPairs / elapsed;
end

% Estimate time
pairsPerSec = rateCache(dimKey);
estSec = double(nPairs) / pairsPerSec;

% Print estimate (skip if not verbose or if label is empty)
if verbose && ~isempty(label)
    if estSec < 1
        timeStr = sprintf('%.0f ms', estSec * 1000);
    elseif estSec < 60
        timeStr = sprintf('%.1f s', estSec);
    elseif estSec < 3600
        timeStr = sprintf('%.1f min', estSec / 60);
    else
        timeStr = sprintf('%.1f hr', estSec / 3600);
    end

    if estSec > 2
        fprintf('%s: estimated time ~%s (Ctrl+C to cancel).\n', label, timeStr);
    else
        fprintf('%s: estimated time ~%s.\n', label, timeStr);
    end
end

end