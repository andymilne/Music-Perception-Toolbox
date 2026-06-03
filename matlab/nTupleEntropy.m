function [H, tuples] = nTupleEntropy(p, period, n, nvArgs)
%NTUPLEENTROPY Entropy of n-tuples of consecutive step sizes.
%
%   H = nTupleEntropy(p, period) returns the normalized entropy of
%   the distribution of step sizes (interonset intervals or pitch
%   intervals) in the set p within an equal division of size period.
%   This is the n = 1 case (the default).
%
%   H = nTupleEntropy(p, period, n) generalizes to n-tuples: ordered
%   sequences of n consecutive step sizes drawn from the circular
%   sequence of events.
%
%   [H, tuples] = nTupleEntropy(...) also returns the K x n matrix
%   of n-tuples.
%
%   With default arguments (sigma = 0 and nPointsPerDim = period, the
%   integer-step grid), this exactly replicates the discrete n-tuple
%   entropy of Milne & Dean (2016): the Shannon entropy of the
%   integer step-size n-tuple histogram.
%
%   Inputs
%       p      - Pitch or position values (vector of length K).
%                Non-negative; values less than period.
%                Must be integer when sigma = 0; may be float when
%                sigma > 0.
%                Duplicates (modulo period) are not allowed.
%       period - Size of the equal division (positive number; must
%                be integer when sigma = 0).
%       n      - (Optional) Tuple size (positive integer, default 1).
%                Must satisfy 1 <= n <= K - 1.
%
%   Name-Value Arguments
%       'sigma'         - Smoothing bandwidth (non-negative scalar;
%                         default 0). In the same units as p and
%                         period.
%       'sigmaSpace'    - How sigma is interpreted (default
%                         'position'). 'position' treats sigma as
%                         positional uncertainty on each p_k;
%                         'interval' treats sigma as independent
%                         uncertainty per derived step. See "Sigma
%                         semantics" below.
%       'method'        - Entropy variant (default 'normalized').
%                         One of {'normalized', 'shannon',
%                         'differential', 'renyi2'} (or the British
%                         alias 'normalised'). See entropyExpTens for
%                         the four-method API. The continuous methods
%                         ('differential', 'renyi2') require sigma > 0.
%       'base'          - Logarithm base (default 2).
%       'nPointsPerDim' - Grid resolution per dimension (used by
%                         'normalized' and 'shannon'; ignored by
%                         'differential' and 'renyi2'). Default 0
%                         means use period (the Milne & Dean 2016
%                         mass-conserving Gaussian-confusion grid).
%
%   Sigma semantics
%       Under the toolbox convention, sigma applies to the input
%       quantity. For nTupleEntropy the input is positions p, so
%       sigmaSpace = 'position' is the default and matches behavior
%       elsewhere in the toolbox (sameness, coherence, etc.).
%
%       For sigmaSpace = 'position':
%         - Each p_k is treated as N(p_k, sigma^2).
%         - Derived steps d_k = p_{k+1} - p_k have variance 2 sigma^2
%           per step, with anti-correlation -sigma^2 between adjacent
%           steps (they share an endpoint with opposite signs).
%         - At n = 1, only the marginal step variance matters, and
%           the entropy is identical to sigmaSpace = 'interval' with
%           sigma_eff = sigma * sqrt(2). This case is handled exactly.
%         - At n >= 2, the cross-step anti-correlation in principle
%           shifts the entropy. The current implementation uses the
%           marginal-matched approximation (sigma_eff = sigma * sqrt(2)
%           per slot, slots independent). Full cross-slot covariance
%           handling at n >= 2 is planned for a future release; a
%           warning is issued when this approximation is in effect.
%
%       For sigmaSpace = 'interval':
%         - Each step d_k is treated as N(d_k, sigma^2) independently.
%         - This is the legacy "step-size" interpretation: each step
%           is the primitive, with its own independent uncertainty.
%         - Use this if your psychological model treats per-step
%           uncertainty as the primitive (rather than positional
%           uncertainty).
%
%       At sigma = 0 the two flags coincide (no smoothing).
%
%   Examples
%       % Diatonic scale in 12-EDO: 1-tuple entropy (sigma = 0)
%       H = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12)
%
%       % Same scale: 2-tuple entropy (sigma = 0)
%       H = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 2)
%
%       % Position-aware soft 1-tuple with positional uncertainty 0.5
%       H = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 1, ...
%                          'sigma', 0.5)
%
%       % Son clave rhythm (16-step cycle)
%       H = nTupleEntropy([0, 3, 6, 10, 12], 16)
%
%       % Raw 2-tuple Shannon entropy in bits (1.56 bits, matching
%       % Milne & Dean 2016, p. 50)
%       H = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 2, ...
%                          'method', 'shannon')
%
%   References
%     Milne, A. J. & Dean, R. T. (2016). Computational creation
%       and morphing of multilevel rhythms by control of evenness.
%       Computer Music Journal, 40(1), 35-53.
%     Milne, A. J. (2024). Commentary on Buechele, Cooke, &
%       Berezovsky (2024). Empirical Musicology Review, 19(2),
%       143-152.
%     Milne, A. J. & Herff, S. A. (2020). The perceptual relevance
%       of balance, evenness, and entropy in musical rhythms.
%       Cognition, 203, 104233.
%
%   See also BINDEVENTS, BUILDEXPTENS, ENTROPYEXPTENS,
%   DIFFERENCEEVENTS, SAMENESS, COHERENCE.
%
%   Batched:
%   [HVec, tuplesCell] = nTupleEntropy(P, period, n) with P an
%   nRows-by-K matrix returns an nRows-by-1 vector of entropies and
%   a 1-by-nRows cell of per-row tuple matrices. NaN-padded rows
%   are accepted; rows with no valid pitches give NaN H and empty
%   cell entries. Per-row dedup is over **permutation and period**
%   symmetries (sorted-modular canonical key), but **not
%   transposition** — the H value alone is transposition-invariant,
%   but the per-row tuples output reflects the input's cyclic order
%   from sort-min and so differs across transposed inputs.

    arguments
        p {mustBeNumeric}
        period (1,1) {mustBePositive}
        n (1,1) {mustBePositive, mustBeInteger} = 1
        nvArgs.sigma (1,1) {mustBeNumeric, mustBeNonnegative} = 0
        nvArgs.sigmaSpace (1,:) char ...
            {mustBeMember(nvArgs.sigmaSpace, {'position', 'interval'})} ...
            = 'position'
        nvArgs.method (1,:) char ...
            {mustBeMember(nvArgs.method, ...
                {'differential','shannon','normalized','normalised','renyi2'})} ...
            = 'normalized'
        nvArgs.base (1,1) {mustBePositive} = 2
        nvArgs.nPointsPerDim (1,1) {mustBeNonnegative, mustBeInteger} = 0
    end

    % Canonicalise British 'normalised' alias.
    if strcmp(nvArgs.method, 'normalised')
        nvArgs.method = 'normalized';
    end

    % Continuous-form methods diverge at sigma=0; reject explicitly
    % before the internal sigma=0 -> sigma=1e-12 nudge that supports
    % the categorical Gaussian-confusion path on a pinned integer
    % grid (which is valid only for the discrete methods).
    if nvArgs.sigma == 0 && any(strcmp(nvArgs.method, {'differential','renyi2'}))
        error('nTupleEntropy:continuousNeedsSigmaPositive', ...
              ['nTupleEntropy: method=''%s'' requires sigma > 0 ' ...
               '(the continuous form diverges at sigma=0). For ' ...
               'categorical sigma=0 n-tuple entropy use ' ...
               'method=''shannon'' or method=''normalized'' (the ' ...
               'default).'], nvArgs.method);
    end

    % --- Batched dispatch ---
    if size(p, 1) > 1 && size(p, 2) > 1
        [H, tuples] = localBatchedNTupleEntropy(p, period, n, nvArgs);
        return;
    end

    % --- Input validation ---

    p = p(:);
    if any(p < 0)
        error('nTupleEntropy:negativePitch', ...
            'p must contain only nonnegative values.');
    end
    p = sort(mod(p, period));
    K = numel(p);

    if numel(unique(p)) ~= K
        error('nTupleEntropy:duplicates', ...
              'p must not contain duplicate values (modulo period).');
    end
    if K < 2
        error('nTupleEntropy:tooFewEvents', ...
              'At least 2 events are required (got %d).', K);
    end
    if n > K - 1
        error('nTupleEntropy:nTooLarge', ...
              ['n must not exceed K - 1 = %d, where K is the number ' ...
               'of events (got n = %d).'], K - 1, n);
    end

    if nvArgs.sigma == 0
        if any(abs(p - round(p)) > 0)
            error('nTupleEntropy:nonIntegerPositions', ...
                  ['For sigma = 0, p must contain integers. ' ...
                   'Use sigma > 0 for non-integer positions.']);
        end
        if abs(period - round(period)) > 0
            error('nTupleEntropy:nonIntegerPeriod', ...
                  ['For sigma = 0, period must be integer ' ...
                   '(got %g). Use sigma > 0 for non-integer periods.'], ...
                  period);
        end
    end

    if nvArgs.nPointsPerDim == 0
        nGrid = round(period);
    else
        nGrid = nvArgs.nPointsPerDim;
    end

    % --- Cyclic first differences via the framework's circular mode ---
    % differenceEvents with 'circular' = true wraps at the sequence
    % boundary (output position 1 holds p(1) - p(N)); the downstream
    % periodic kernel handles mod-period wrapping at evaluation time,
    % so no explicit mod is needed here. The resulting multiset of
    % consecutive-difference n-grams is invariant under the cyclic
    % rotation that distinguishes this ordering from the equivalent
    % "diff first, wrap difference at position N" convention.
    pRow = p(:).';
    [pDiffCell, ~, ~] = differenceEvents({pRow}, [], [], 1, ...
                                          'circular', true);
    diffsRow = pDiffCell{1};

    % --- Bind n consecutive cyclic step sizes ---

    [pBound, wBound, specs] = bindEvents({diffsRow}, [], n, 1, false, true, ...
                                         'circular', true);

    % --- Resolve sigma per the sigmaSpace flag ---
    %
    % 'interval': sigma is per-step uncertainty (legacy step-size mode);
    %             slots are independent with variance sigma^2 each.
    %
    % 'position': sigma is positional uncertainty; each step inherits
    %             variance 2 sigma^2 (since step = p_{k+1} - p_k with
    %             two independent positional jitters). The full
    %             position model also includes -sigma^2 anti-
    %             correlation between adjacent slots, but this is not
    %             yet implemented; the marginal-matched approximation
    %             (sigma_eff = sigma * sqrt(2), slots independent) is
    %             used at n >= 2. Exact at n = 1.

    if strcmp(nvArgs.sigmaSpace, 'position')
        sigmaUse = nvArgs.sigma * sqrt(2);
        if n >= 2 && nvArgs.sigma > 0
            warning('nTupleEntropy:positionApprox', ...
                    ['sigmaSpace=''position'' at n >= 2 currently uses ' ...
                     'a marginal-matched approximation; cross-slot ' ...
                     'anti-correlations are not yet captured. Full ' ...
                     'position-aware n-tuple support is planned for a ' ...
                     'future release. Suppress this warning with ' ...
                     'warning(''off'', ''nTupleEntropy:positionApprox'').']);
        end
    else  % 'interval'
        sigmaUse = nvArgs.sigma;
    end

    if sigmaUse <= 0
        sigmaUse = 1e-12;
    end

    % --- Build MAET ---
    % The bound events nest into a single attribute (outer r = n reads the
    % whole window, rel absolute), reproducing the old tensor join of n
    % single-step attributes (spec §6.5); sigma/isPer/period are scalar.

    T = buildExpTens(pBound, wBound, 'specs', specs, 'sigma', sigmaUse, ...
                     'isPer', true, 'period', period, 'verbose', false);

    % --- Entropy on the chosen grid / via the chosen method ---
    % Grid-based methods ('shannon', 'normalized') use the pinned
    % period grid nGrid. 'differential' and 'renyi2' bypass the grid
    % (adaptive and analytical respectively).
    switch nvArgs.method
        case 'shannon'
            H = entropyExpTens(T, ...
                               'method', 'shannon', ...
                               'base', nvArgs.base, ...
                               'nPointsPerDim', nGrid);
        case 'normalized'
            H = entropyExpTens(T, ...
                               'method', 'normalized', ...
                               'base', nvArgs.base, ...
                               'nPointsPerDim', nGrid);
        case 'differential'
            H = entropyExpTens(T, ...
                               'method', 'differential', ...
                               'base', nvArgs.base);
        otherwise  % 'renyi2'
            H = entropyExpTens(T, ...
                               'method', 'renyi2', ...
                               'base', nvArgs.base);
    end

    % --- Tuples matrix (N', n) for compatibility with the prior API ---
    % pBound is one stacked attribute; its columns are the n-grams, so the
    % tuples matrix is its transpose.

    if nargout > 1
        tuples = pBound{1}.';
    end
end


% =====================================================================
%  Unified dispatch helper: batched-raw mode.
% =====================================================================

function [HVec, tuplesCell] = localBatchedNTupleEntropy(P, period, n, nvArgs)
%LOCALBATCHEDNTUPLEENTROPY Per-row n-tuple entropy from a 2-D matrix.

    nRows = size(P, 1);
    HVec = nan(nRows, 1);
    tuplesCell = cell(1, nRows);

    cache = containers.Map('KeyType', 'char', 'ValueType', 'any');

    for k = 1:nRows
        pRow = P(k, :);
        validMask = ~isnan(pRow);
        pK = pRow(validMask);
        if isempty(pK)
            continue;
        end
        if any(pK < 0)
            error('nTupleEntropy:negativePitch', ...
                'Row %d contains negative pitches; all valid (non-NaN) entries must be nonnegative.', k);
        end

        pCanon = sort(mod(pK(:), period));
        keyStr = sprintf('%.12g,', pCanon);

        if isKey(cache, keyStr)
            stored = cache(keyStr);
            HVec(k)       = stored{1};
            tuplesCell{k} = stored{2};
            continue;
        end

        [Hk, tk] = nTupleEntropy(pK(:), period, n, ...
            'sigma', nvArgs.sigma, 'sigmaSpace', nvArgs.sigmaSpace, ...
            'method', nvArgs.method, 'base', nvArgs.base, ...
            'nPointsPerDim', nvArgs.nPointsPerDim);
        HVec(k)       = Hk;
        tuplesCell{k} = tk;
        cache(keyStr) = {Hk, tk};
    end
end
