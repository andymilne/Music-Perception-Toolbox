function [H, tuples] = nTupleEntropy(p, period, n, nvArgs)
%NTUPLEENTROPY Entropy of n-tuples of consecutive step sizes.
%
%   H = nTupleEntropy(p, period) returns the normalized entropy of
%   the distribution of step sizes (interonset intervals or pitch
%   intervals) in the set p within an equal division of size period.
%   This is the n = 1 case (the default): each step size is a
%   1-tuple, and the entropy measures how predictable the step sizes
%   are.
%
%   H = nTupleEntropy(p, period, n) generalizes to n-tuples: ordered
%   sequences of n consecutive step sizes drawn from the circular
%   sequence of events. For n = 2, each starting event yields a pair
%   of consecutive step sizes; for n = 3, an ordered triple; and so
%   on.
%
%   [H, tuples] = nTupleEntropy(...) also returns the K x n matrix
%   of n-tuples. Each row is one n-tuple of consecutive step sizes,
%   ordered by starting event.
%
%   Convenience wrapper. As of v2.1.0, this function is a thin
%   wrapper around the bind-and-compute pipeline of bindEvents,
%   buildExpTens, and entropyExpTens. With default arguments
%   (sigma = 0 and nPointsPerDim = period, the integer-step grid),
%   it exactly replicates the discrete n-tuple entropy of Milne &
%   Dean (2016): the Shannon entropy of the integer step-size
%   n-tuple histogram. Setting sigma > 0 gives the smoothed
%   extension of Milne (2024); setting nPointsPerDim finer than
%   period discretizes the underlying continuous density at finer
%   resolution. For more flexibility — non-integer values, non-
%   uniform weights, non-periodic domains, or access to the
%   n-tuple density itself for similarity comparison and other
%   MAET operations — call bindEvents and the rest of the pipeline
%   directly. This function remains as a convenient entry point for
%   the common case of integer-valued, uniformly weighted, periodic
%   step-size analyses.
%
%   Inputs
%       p      - Pitch or position values (vector of length K).
%                Must be non-negative integers less than period.
%                Duplicates (modulo period) are not allowed.
%       period - Size of the equal division (positive integer).
%       n      - (Optional) Tuple size: the number of consecutive
%                step sizes in each n-tuple (positive integer,
%                default: 1). Must satisfy 1 <= n <= K - 1.
%
%   Name-Value Arguments
%       'sigma'         - Standard deviation of the Gaussian
%                         smoothing kernel, in units of step sizes
%                         (i.e., chromatic steps or pulses).
%                         Default: 0 (no smoothing).
%       'normalize'     - Logical (default: true). If true, divides
%                         the entropy by log_base(nPointsPerDim^n)
%                         to give a value in [0, 1]. With the
%                         default nPointsPerDim = period this
%                         matches the log_base(period^n)
%                         normalization of Milne & Dean (2016).
%       'base'          - Logarithm base (default: 2, "bits"). When
%                         'normalize' is true, the base cancels.
%       'nPointsPerDim' - Grid resolution per effective dimension.
%                         Default: 0 (interpreted as period; the
%                         integer-step grid). Any positive integer
%                         is accepted; finer grids resolve the
%                         underlying continuous density more
%                         accurately when sigma > 0.
%
%   Outputs
%       H      - Shannon entropy of the n-tuple step-size
%                distribution.
%       tuples - K x n matrix of n-tuples. Row i is the n-tuple
%                starting from the i-th event of the sorted
%                circular sequence; column j is the value at
%                lag j-1.
%
%   Examples
%       % Diatonic scale in 12-EDO: 1-tuple (step / IOI) entropy
%       H = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12)
%
%       % Same scale: 2-tuple entropy (1.56 bits unnormalized,
%       % matching Milne & Dean 2016, p. 50)
%       H = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 2, ...
%                          'normalize', false)
%
%       % Son clave rhythm (16-step cycle)
%       H = nTupleEntropy([0, 3, 6, 10, 12], 16)
%
%       % With Gaussian smoothing
%       H = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 2, ...
%                          'sigma', 0.5, 'normalize', false)
%
%       % With finer grid (resolves the smoothed density better)
%       H = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 2, ...
%                          'sigma', 0.5, 'nPointsPerDim', 120, ...
%                          'normalize', false)
%
%   References
%     Milne, A. J. & Dean, R. T. (2016). Computational creation
%       and morphing of multilevel rhythms by control of evenness.
%       Computer Music Journal, 40(1), 35-53.
%     Milne, A. J. (2024). Commentary on Buechele, Cooke, &
%       Berezovsky (2024): Entropic models of scales and some
%       extensions. Empirical Musicology Review, 19(2), 143-152.
%     Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects
%       of rhythmic structure on tapping accuracy. Attention,
%       Perception, & Psychophysics, 85, 2673-2699.
%     Milne, A. J. & Herff, S. A. (2020). The perceptual relevance
%       of balance, evenness, and entropy in musical rhythms.
%       Cognition, 203, 104233.
%
%   See also BINDEVENTS, BUILDEXPTENS, ENTROPYEXPTENS,
%   DIFFERENCEEVENTS.

    arguments
        p (:,1) {mustBeNonnegative, mustBeInteger}
        period (1,1) {mustBePositive, mustBeInteger}
        n (1,1) {mustBePositive, mustBeInteger} = 1
        nvArgs.sigma (1,1) {mustBeNonnegative} = 0
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

    if nvArgs.nPointsPerDim == 0
        nGrid = period;
    else
        nGrid = nvArgs.nPointsPerDim;
    end

    % --- Cyclic step sizes (K events -> K cyclic differences) ---

    pCol = p(:);
    diffs = mod(diff([pCol; pCol(1) + period]), period);
    diffsRow = diffs(:).';

    % --- Bind n consecutive cyclic step sizes ---

    [pBound, wBound] = bindEvents(diffsRow, [], n, 'circular', true);

    % --- Build MAET ---
    %   n attributes (one per lag), all in one group sharing sigma,
    %   isPer = true, period; r = 1 per attribute (each is an
    %   absolute monad).

    sigmaUse = nvArgs.sigma;
    if sigmaUse <= 0
        sigmaUse = 1e-12;
    end
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
