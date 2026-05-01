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
%       'normalize'     - Logical (default true). Divide by
%                         log_base(nPointsPerDim^n).
%       'base'          - Logarithm base (default 2).
%       'nPointsPerDim' - Grid resolution per dimension. Default 0
%                         means use period.
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
%         - This is exactly the v2.0 behavior of this function.
%         - Use this if you want the v2 numerical results, or if your
%           psychological model treats per-step uncertainty as the
%           primitive (rather than positional uncertainty).
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
%       % Unnormalized 2-tuple entropy in bits (1.56 bits, matching
%       % Milne & Dean 2016, p. 50)
%       H = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 2, ...
%                          'normalize', false)
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

    arguments
        p (:,1) {mustBeNumeric, mustBeNonnegative}
        period (1,1) {mustBePositive}
        n (1,1) {mustBePositive, mustBeInteger} = 1
        nvArgs.sigma (1,1) {mustBeNumeric, mustBeNonnegative} = 0
        nvArgs.sigmaSpace (1,:) char ...
            {mustBeMember(nvArgs.sigmaSpace, {'position', 'interval'})} ...
            = 'position'
        nvArgs.normalize (1,1) logical = true
        nvArgs.base (1,1) {mustBePositive} = 2
        nvArgs.nPointsPerDim (1,1) {mustBeNonnegative, mustBeInteger} = 0
    end

    % --- Input validation ---

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

    % --- Cyclic step sizes (K events -> K cyclic differences) ---

    pCol = p(:);
    diffs = mod(diff([pCol; pCol(1) + period]), period);
    diffsRow = diffs(:).';

    % --- Bind n consecutive cyclic step sizes ---

    [pBound, wBound] = bindEvents(diffsRow, [], n, 'circular', true);

    % --- Resolve sigma per the sigmaSpace flag ---
    %
    % 'interval': sigma is per-step uncertainty (v2.0 semantics);
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

    T = buildExpTens(pBound, wBound, sigmaUse, ones(1, n), ...
                     ones(1, n), false, true, period, ...
                     'verbose', false);

    % --- Shannon entropy on the chosen grid ---

    H = entropyExpTens(T, ...
                       'normalize', nvArgs.normalize, ...
                       'base', nvArgs.base, ...
                       'nPointsPerDim', nGrid);

    % --- Tuples matrix (K, n) for compatibility with the prior API ---

    if nargout > 1
        tuples = zeros(K, n);
        for j = 1:n
            tuples(:, j) = pBound{j}(:);
        end
    end
end
