function [b, b_std] = balanceCircular(p, w, period, sigma, nvArgs)
%BALANCECIRCULAR Balance of a weighted circular multiset.
%
%   b = balanceCircular(p, w, period):
%
%   Computes the balance of a weighted multiset of points on a circle
%   (p represents pitches or positions), defined as:
%
%     b = 1 - |F(0)|
%
%   where F(k) is the k-th DFT coefficient of the multiset (see
%   dftCircular). |F(0)| is the magnitude of the weighted centre of
%   gravity on the unit circle (the mean of exp(2*pi*1i*p/period)).
%
%   Balance ranges from 0 to 1:
%     b = 1: perfectly balanced — the centre of gravity is at the centre
%            of the circle. Examples: the whole-tone scale, the augmented
%            triad, or any equal-step scale in pitch; isochronous rhythms.
%     b = 0: maximally unbalanced — all weight concentrated at one point.
%
%   Perfect balance is a necessary condition for maximal evenness but is
%   not sufficient: a multiset can be perfectly balanced without being
%   maximally even (see evennessCircular).
%
%   b = balanceCircular(p, w, period, sigma) returns the expected
%   balance under independent Gaussian positional jitter on each event,
%   estimated by Monte Carlo simulation:
%
%     P_k = (p_k + eta_k) mod period,   eta_k ~ N(0, sigma^2)
%
%   The perturbed positions are sorted (resort) before computing the
%   DFT — though for balance specifically, F(0) is permutation-invariant
%   so the sort step has no effect on this coefficient. At sigma = 0 the
%   v2.0 deterministic value is recovered exactly.
%
%   [b, b_std] = balanceCircular(...) also returns the standard
%   deviation of (1 - |F(0)|) under the jitter model. b_std = 0 at
%   sigma = 0.
%
%   For further information, see:
%     Milne, A. J., Bulger, D., & Herff, S. A. (2017). Exploring the
%       space of perfectly balanced rhythms and scales. Journal of
%       Mathematics and Music, 11(2-3), 101-133.
%     Milne, A. J. & Herff, S. A. (2020). The perceptual relevance of
%       balance, evenness, and entropy in musical rhythms. Cognition,
%       203, 104233.
%
%   Inputs:
%     p      — Pitch or position values (vector of length K).
%     w      — Weights (vector of length K, or empty for all ones).
%     period — Period of the circular domain.
%     sigma  — (Optional) Positional jitter standard deviation
%              (non-negative scalar; default 0). In the same units as
%              p and period.
%
%   Name-Value Arguments:
%     'nDraws'  — Number of Monte Carlo draws when sigma > 0 (default
%                 10000).
%     'rngSeed' — Optional non-negative integer for reproducibility.
%
%   Outputs:
%     b      — Balance (mean under jitter, scalar in [0, 1]).
%     b_std  — Standard deviation of 1 - |F(0)| under jitter.
%
%   Examples:
%     % Perfectly balanced: augmented triad (deterministic)
%     b = balanceCircular([0, 400, 800], [], 1200);     % b = 1.000
%
%     % Same triad, expected balance under sigma = 25 cents jitter
%     [b, bs] = balanceCircular([0, 400, 800], [], 1200, 25);
%
%   See also evennessCircular, dftCircular, dftCircularSimulate.
%
%   Batched (v2.1+):
%   bVec = balanceCircular(P, W, period, sigma, ...) with P an
%   nRows-by-K matrix returns an nRows-by-1 vector of balance
%   values; with two output arguments it also returns an
%   nRows-by-1 vector of per-row standard deviations.
%
%   Per-row dedup is over permutation + period symmetries via a
%   sorted-modular canonical key. For sigma > 0 the new
%   ``'rngScope'`` NV pair controls how each row's RNG seed is
%   derived from the base ``'rngSeed'``:
%     'canonical' (default): seed = base + fnv1a32(canonical_key).
%        Canonical-form-equivalent rows get identical seeds and
%        identical Monte-Carlo realisations, enabling full dedup.
%     'row': seed = base + row_index. Each row gets an independent
%        reproducible realisation; dedup is disabled.
%   If ``'rngSeed'`` is empty in batched mode, a session-random
%   base is generated once per call so within-call dedup remains
%   reproducible while across-call results differ.

    arguments
        p
        w
        period (1,1) {mustBeNumeric, mustBePositive}
        sigma (1,1) {mustBeNumeric, mustBeNonnegative} = 0
        nvArgs.nDraws (1,1) {mustBePositive, mustBeInteger} = 10000
        nvArgs.rngSeed = []
        nvArgs.rngScope (1,:) char ...
            {mustBeMember(nvArgs.rngScope, {'canonical', 'row'})} ...
            = 'canonical'
    end

    % --- Batched dispatch (v2.1+) ---
    if size(p, 1) > 1 && size(p, 2) > 1
        if nargout > 1
            [b, b_std] = localBatchedBalance(p, w, period, sigma, nvArgs);
        else
            b = localBatchedBalance(p, w, period, sigma, nvArgs);
        end
        return;
    end

    if sigma == 0
        [~, mag] = dftCircular(p, w, period);
        b = 1 - mag(1);
        if nargout > 1
            b_std = 0;
        end
        return;
    end

    % --- Monte Carlo path ---
    [magMean, magStd] = dftCircularSimulate(p, w, period, sigma, ...
        'nDraws', nvArgs.nDraws, 'rngSeed', nvArgs.rngSeed);
    b = 1 - magMean(1);
    if nargout > 1
        b_std = magStd(1);  % SD of |F(0)| equals SD of (1 - |F(0)|)
    end
end


% =====================================================================
%  v2.1 unified dispatch helper: batched-raw mode.
% =====================================================================

function [bVec, bStdVec] = localBatchedBalance(P, W, period, sigma, nvArgs)
%LOCALBATCHEDBALANCE Per-row balance from a 2-D pitch matrix.

    nRows = size(P, 1);
    bVec = nan(nRows, 1);
    bStdVec = nan(nRows, 1);

    haveRowWeights = ~isempty(W) && isequal(size(W), size(P));
    if ~isempty(W) && ~haveRowWeights
        if isvector(W) && numel(W) == size(P, 2)
            W_broadcast = W(:).';
        else
            error('balanceCircular:weightShape', ...
                ['In batched mode, w must be empty, a matrix the same size as p, ' ...
                 'or a vector matching the number of pitch columns.']);
        end
    end

    baseSeed = localResolveBaseSeed(nvArgs.rngSeed);
    useCache = (sigma == 0) || strcmp(nvArgs.rngScope, 'canonical');
    cache = containers.Map('KeyType', 'char', 'ValueType', 'any');

    for k = 1:nRows
        pRow = P(k, :);
        validMask = ~isnan(pRow);
        pK = pRow(validMask);
        if haveRowWeights
            wK = W(k, validMask);
        elseif ~isempty(W)
            wK = W_broadcast(validMask);
        else
            wK = [];
        end
        if isempty(pK)
            continue;
        end

        if isempty(wK)
            wKcol = ones(numel(pK), 1);
        else
            wKcol = wK(:);
        end
        pMod = mod(pK(:), period);
        [pSorted, sortIdx] = sort(pMod);
        wSorted = wKcol(sortIdx);
        keyStr = sprintf('%.12g,', pSorted, wSorted);

        if useCache && isKey(cache, keyStr)
            stored = cache(keyStr);
            bVec(k)    = stored(1);
            bStdVec(k) = stored(2);
            continue;
        end

        % Per-row seed
        if sigma == 0
            rowSeed = [];
        elseif strcmp(nvArgs.rngScope, 'canonical')
            rowSeed = mod(double(baseSeed) + double(localFnv1a32(keyStr)), 2^32);
        else
            rowSeed = mod(double(baseSeed) + (k - 1), 2^32);
        end

        [bk, bsk] = balanceCircular(pK(:), wKcol, period, sigma, ...
            'nDraws', nvArgs.nDraws, 'rngSeed', rowSeed);
        bVec(k)    = bk;
        bStdVec(k) = bsk;
        if useCache
            cache(keyStr) = [bk, bsk];
        end
    end
end


function baseSeed = localResolveBaseSeed(rngSeed)
%LOCALRESOLVEBASESEED 32-bit base seed for batched MC.
%   When rngSeed is empty, generates a session-random base from a
%   shuffled, isolated RandStream so the global RNG state is not
%   disturbed.
    if isempty(rngSeed)
        rs = RandStream('mt19937ar', 'Seed', 'shuffle');
        baseSeed = uint32(floor(rand(rs) * 2^32));
    else
        baseSeed = uint32(mod(double(rngSeed), 2^32));
    end
end


function h = localFnv1a32(s)
%LOCALFNV1A32 32-bit FNV-1a hash of a char/byte sequence.
    h = uint32(2166136261);  % FNV offset basis
    p = uint64(16777619);    % FNV prime
    mask = uint64(2^32 - 1);
    bytes = uint8(s);
    for i = 1:numel(bytes)
        h = bitxor(h, uint32(bytes(i)));
        h = uint32(bitand(uint64(h) * p, mask));
    end
end
