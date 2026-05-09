function [e, e_std] = evennessCircular(p, period, sigma, nvArgs)
%EVENNESSCIRCULAR Evenness of a circular multiset.
%
%   e = evennessCircular(p, period):
%
%   Computes the evenness of a multiset of K points on a circle
%   (p represents pitches or positions), defined as:
%
%     e = |F(1)|
%
%   where F(k) is the k-th DFT coefficient of the multiset
%   (see dftCircular). The k = 1 coefficient captures the extent
%   to which the K sorted elements match a maximally even
%   (equal-step) distribution around the circle. For a maximally
%   even multiset, each sorted element j (0-indexed) is at position
%   approximately j * period / K, so:
%     z(j) * exp(-2*pi*1i*j/K) = exp(2*pi*1i*j/K) * exp(-2*pi*1i*j/K) = 1
%   and |F(1)| = 1.
%
%   Evenness ranges from 0 to 1:
%     e = 1: maximally even — the multiset consists of K equally spaced
%            points. Examples: the whole-tone scale, the chromatic
%            scale, an isochronous rhythm.
%     e = 0: maximally uneven for this cardinality.
%
%   Maximal evenness implies perfect balance, but perfect balance does
%   not imply maximal evenness: a multiset can be perfectly balanced
%   without being maximally even (see balanceCircular).
%
%   Evenness always uses uniform (binary) weights, following Milne et
%   al. (2017): "we focus on binary-weighted patterns, whose weights
%   are all zero or one." Evenness is a property of the spatial
%   distribution of elements around the circle, not of their relative
%   saliences. See balanceCircular for a measure that supports
%   non-uniform weights.
%
%   e = evennessCircular(p, period, sigma) returns the expected
%   evenness under independent Gaussian positional jitter on each
%   event, estimated by Monte Carlo simulation:
%
%     P_k = (p_k + eta_k) mod period,   eta_k ~ N(0, sigma^2)
%
%   The perturbed positions are sorted (resort) before computing the
%   DFT, capturing the perceptual reordering that occurs when noise
%   is comparable to the smallest event-to-event gap. At sigma = 0
%   the v2.0 deterministic value is recovered exactly.
%
%   [e, e_std] = evennessCircular(...) also returns the standard
%   deviation of |F(1)| under the jitter model. e_std = 0 at
%   sigma = 0.
%
%   For further information, see:
%     Milne, A. J., Bulger, D., & Herff, S. A. (2017). Exploring the
%       space of perfectly balanced rhythms and scales. Journal of
%       Mathematics and Music, 11(2-3), 101-133.
%     Milne, A. J. & Herff, S. A. (2020). The perceptual relevance of
%       balance, evenness, and entropy in musical rhythms. Cognition,
%       203, 104233. (Section 5.2.1.1 documents the per-coefficient
%       coefficient-of-variation pattern under jitter.)
%
%   Inputs:
%     p      — Pitch or position values (vector of length K).
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
%     e      — Evenness (mean under jitter, scalar in [0, 1]).
%     e_std  — Standard deviation of |F(1)| under jitter.
%
%   Examples:
%     % Maximally even (deterministic): whole-tone scale
%     e = evennessCircular([0, 200, 400, 600, 800, 1000], 1200);   % 1.000
%
%     % Same scale, expected evenness under sigma = 25 cents
%     [e, es] = evennessCircular([0, 200, 400, 600, 800, 1000], 1200, 25);
%
%   See also balanceCircular, dftCircular, dftCircularSimulate.
%
%   Batched (v2.1+):
%   eVec = evennessCircular(P, period, sigma, ...) with P an
%   nRows-by-K matrix returns an nRows-by-1 vector of evenness
%   values; with two output arguments it also returns an
%   nRows-by-1 vector of per-row standard deviations. See
%   balanceCircular's batched note for the ``'rngScope'`` semantics.

    arguments
        p
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
            [e, e_std] = localBatchedEvenness(p, period, sigma, nvArgs);
        else
            e = localBatchedEvenness(p, period, sigma, nvArgs);
        end
        return;
    end

    if sigma == 0
        [~, mag] = dftCircular(p, [], period);
        e = mag(2);
        if nargout > 1
            e_std = 0;
        end
        return;
    end

    % --- Monte Carlo path ---
    [magMean, magStd] = dftCircularSimulate(p, [], period, sigma, ...
        'nDraws', nvArgs.nDraws, 'rngSeed', nvArgs.rngSeed);
    e = magMean(2);
    if nargout > 1
        e_std = magStd(2);
    end
end


% =====================================================================
%  v2.1 unified dispatch helper: batched-raw mode.
% =====================================================================

function [eVec, eStdVec] = localBatchedEvenness(P, period, sigma, nvArgs)
%LOCALBATCHEDEVENNESS Per-row evenness from a 2-D pitch matrix.

    nRows = size(P, 1);
    eVec = nan(nRows, 1);
    eStdVec = nan(nRows, 1);

    baseSeed = localResolveBaseSeed(nvArgs.rngSeed);
    useCache = (sigma == 0) || strcmp(nvArgs.rngScope, 'canonical');
    cache = containers.Map('KeyType', 'char', 'ValueType', 'any');

    for k = 1:nRows
        pRow = P(k, :);
        validMask = ~isnan(pRow);
        pK = pRow(validMask);
        if isempty(pK)
            continue;
        end

        pMod = mod(pK(:), period);
        pSorted = sort(pMod);
        keyStr = sprintf('%.12g,', pSorted);

        if useCache && isKey(cache, keyStr)
            stored = cache(keyStr);
            eVec(k)    = stored(1);
            eStdVec(k) = stored(2);
            continue;
        end

        if sigma == 0
            rowSeed = [];
        elseif strcmp(nvArgs.rngScope, 'canonical')
            rowSeed = mod(double(baseSeed) + double(localFnv1a32(keyStr)), 2^32);
        else
            rowSeed = mod(double(baseSeed) + (k - 1), 2^32);
        end

        [ek, esk] = evennessCircular(pK(:), period, sigma, ...
            'nDraws', nvArgs.nDraws, 'rngSeed', rowSeed);
        eVec(k)    = ek;
        eStdVec(k) = esk;
        if useCache
            cache(keyStr) = [ek, esk];
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
    h = uint32(2166136261);
    p = uint64(16777619);
    mask = uint64(2^32 - 1);
    bytes = uint8(s);
    for i = 1:numel(bytes)
        h = bitxor(h, uint32(bytes(i)));
        h = uint32(bitand(uint64(h) * p, mask));
    end
end
