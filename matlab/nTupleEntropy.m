function [H, tuples] = nTupleEntropy(p, period, n, nvArgs)
%NTUPLEENTROPY Entropy of n-tuples of consecutive step sizes.
%
%   H = nTupleEntropy(p, period) returns the normalized entropy of the
%   distribution of step sizes (seconds / interonset intervals) in the
%   set p within an equal division of size period (p represents
%   pitches or positions). This is the n = 1 case (the default):
%   each step size is a 1-tuple, and the entropy measures how
%   predictable the step sizes are.
%
%   H = nTupleEntropy(p, period, n) generalizes to n-tuples: ordered
%   sequences of n consecutive step sizes drawn from the circular
%   sequence of events. For n = 2, each starting event yields a pair
%   of consecutive step sizes; for n = 3, an ordered triple; and so
%   on. The function counts all K such n-tuples (one per starting
%   element in the K-element set), builds a probability mass function
%   over the space of all possible n-tuples, and returns the Shannon
%   entropy.
%
%   [H, tuples] = nTupleEntropy(...) also returns the K x n matrix of
%   n-tuples. Each row is one n-tuple of consecutive step sizes,
%   ordered by starting event.
%
%   Higher values of n capture progressively finer sequential
%   structure. Whereas n = 1 measures only which step sizes occur
%   (ignoring order), n = 2 measures which ordered pairs of
%   consecutive step sizes occur, and so on. As shown in Milne and
%   Dean (2016, Fig. 9), n-tuple entropy distinguishes scales that
%   share the same step-size inventory but arrange them differently:
%   well-formed arrangements of two step sizes have lower n-tuple
%   entropy than non-well-formed arrangements.
%
%   When n = 1, this reduces to interonset interval entropy (IOI
%   entropy) or scale-step entropy, which has proven useful for
%   predicting the recognizability and liking of rhythms (Milne &
%   Herff, 2020), modelling tapping accuracy (Milne, Dean, & Bulger,
%   2023), and optimizing scale structures (Milne, 2024). Note that
%   the n = 1 case considers only step sizes between consecutive
%   events (generic-span-1 intervals); it is not equivalent to the
%   entropy of a relative dyad tensor from entropyExpTens, which
%   considers all pairwise intervals. For the n = 1 case
%   specifically, the smoothed entropy could alternatively be
%   computed by passing the step sizes to entropyExpTens as an
%   absolute periodic monad tensor; for n > 1, however,
%   entropyExpTens does not currently support multi-dimensional
%   grids, so the smoothing is handled internally here.
%
%   By default, the entropy is normalized to [0, 1] by dividing by
%   log_base(period^n), the maximum entropy of a uniform distribution
%   over all possible n-tuples. This removes dependence on both the
%   period and the logarithm base.
%
%   Gaussian Smoothing
%   ------------------
%   Milne (2024) describes applying Gaussian smoothing to model
%   perceptual inaccuracy in step-size estimation, making the entropy
%   a continuous function of the event positions suitable for
%   optimization. When 'sigma' > 0, the discrete histogram of
%   n-tuples is convolved with a separable circular Gaussian kernel
%   before the entropy is computed. The smoothing is applied
%   independently along each tuple dimension, modelling independent
%   perceptual inaccuracy for each step-size measurement.
%
%   Inputs:
%     p      — Pitch or position values (vector of
%              length K). Must be non-negative integers less than
%              period. Duplicates (modulo period) are not allowed.
%     period — Size of the equal division (positive integer).
%     n      — (Optional) Tuple size: the number of consecutive step
%              sizes in each n-tuple (positive integer, default: 1).
%              Must satisfy 1 <= n <= K - 1. When n = 1, the function
%              computes IOI / step-size entropy.
%
%   Name-Value Arguments:
%     'sigma'     — Standard deviation of the Gaussian smoothing
%                   kernel, in units of step sizes (i.e., chromatic
%                   steps or pulses). Default: 0 (no smoothing).
%                   When sigma > 0, a circular Gaussian kernel is
%                   convolved independently along each tuple
%                   dimension before the entropy is computed.
%     'normalize' — Logical (default: true). If true, divides the
%                   entropy by log_base(period^n) to give a value in
%                   [0, 1]. If false, returns the raw entropy in
%                   units determined by 'base'.
%     'base'      — Logarithm base (default: 2, giving bits). When
%                   'normalize' is true, the base cancels and has
%                   no effect on the result.
%
%   Outputs:
%     H      — Shannon entropy of the n-tuple step-size distribution.
%              When 'normalize' is true (default), H is in [0, 1]:
%              0 when all n-tuples are identical, and approaching 1
%              as the distribution becomes more uniform. When
%              'normalize' is false, H is in the units determined
%              by 'base'.
%     tuples — K x n matrix of n-tuples. Each row is a sequence of
%              n consecutive step sizes starting from one event.
%              Row i corresponds to the n-tuple starting from the
%              i-th event in the sorted circular sequence. Values
%              are in {1, ..., period - 1}.
%
%   Examples:
%     % Diatonic scale in 12-EDO: 1-tuple (step / IOI) entropy
%     H = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12)
%
%     % Same scale: 2-tuple entropy (1.56 bits unnormalized,
%     % matching Milne & Dean 2016, p. 50)
%     H = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 2, ...
%                        'normalize', false)
%
%     % Same scale: 3-tuple entropy (1.95 bits unnormalized)
%     H = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 3, ...
%                        'normalize', false)
%
%     % Whole-tone scale (single step size, entropy = 0)
%     H = nTupleEntropy([0, 2, 4, 6, 8, 10], 12)
%
%     % Son clave rhythm (16-step cycle)
%     H = nTupleEntropy([0, 3, 6, 10, 12], 16)
%
%     % With Gaussian smoothing (sigma = 0.2 steps)
%     H = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 1, ...
%                        'sigma', 0.2, 'normalize', false)
%
%     % Smoothed 2-tuple entropy
%     H = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 2, ...
%                        'sigma', 0.2, 'normalize', false)
%
%     % Extract tuples for inspection
%     [H2, tuples2] = nTupleEntropy( ...
%         [0, 2, 4, 5, 7, 9, 11], 12, 2);
%     disp(tuples2)
%
%   References:
%     Milne, A. J. & Dean, R. T. (2016). Computational creation
%       and morphing of multilevel rhythms by control of evenness.
%       Computer Music Journal, 40(1), 35-53.
%       (Introduced n-tuple entropy for scales and rhythms;
%       see p. 50 and Fig. 9.)
%     Milne, A. J. (2024). Commentary on Buechele, Cooke, &
%       Berezovsky (2024): Entropic models of scales and some
%       extensions. Empirical Musicology Review, 19(2), 143-152.
%       (Applied scale-step entropy with Gaussian smoothing as a
%       cost function for scale optimization.)
%     Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects
%       of rhythmic structure on tapping accuracy. Attention,
%       Perception, & Psychophysics, 85, 2673-2699.
%       (Applied IOI entropy — the n = 1 case — to rhythmic
%       patterns.)
%     Milne, A. J. & Herff, S. A. (2020). The perceptual relevance
%       of balance, evenness, and entropy in musical rhythms.
%       Cognition, 203, 104233.
%       (Demonstrated IOI entropy as a predictor of rhythm
%       perception.)
%
%   See also ENTROPYEXPTENS, COHERENCE, SAMENESS, SPECTRALENTROPY.

    arguments
        p (:,1) {mustBeNonnegative, mustBeInteger}
        period (1,1) {mustBePositive, mustBeInteger}
        n (1,1) {mustBePositive, mustBeInteger} = 1
        nvArgs.sigma (1,1) {mustBeNonnegative} = 0
        nvArgs.normalize (1,1) logical = true
        nvArgs.base (1,1) {mustBePositive} = 2
    end

    % === Input validation ===

    p = sort(mod(p, period));
    K = numel(p);

    if numel(unique(p)) ~= K
        error('nTupleEntropy:duplicates', ...
              'p must not contain duplicate pitch classes (modulo period).');
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

    N = period;

    % Guard against excessive memory usage.
    totalBins = N^n;
    if totalBins > 1e9
        error('nTupleEntropy:tooManyBins', ...
              ['period^n = %d^%d = %.2e exceeds 10^9. ' ...
               'Reduce n or period.'], N, n, totalBins);
    end

    % === Build circulant matrix of sorted events ===
    % Each column j is the sorted set rotated by j - 1 positions.

    idx = mod((0:K-1)' + (0:K-1), K) + 1;
    rotations = p(idx);  % K x K

    % === Compute step sizes between consecutive events ===
    % diff along dimension 1: step from event i to event i+1 in
    % each rotation. mod(..., period) wraps the final step (from
    % the last event back to the first) into [0, period - 1].

    allStepSizes = mod(diff(rotations, 1, 1), N);  % (K-1) x K

    % === Extract the n step sizes that define each n-tuple ===

    steps = allStepSizes(1:n, :);  % n x K

    % === Optional output: K x n matrix of n-tuples ===

    if nargout > 1
        tuples = steps';  % K x n
    end

    % === Histogram via linear indexing ===
    % Each column of steps is an n-tuple in {0, ..., N-1}^n.
    % Map to a 1-based linear index: sum_i steps(i,:) * N^(i-1) + 1.

    multipliers = N .^ (0:n-1)';        % n x 1
    linIdx = (steps' * multipliers) + 1; % K x 1

    counts = accumarray(linIdx, 1, [totalBins, 1]);

    % === Gaussian smoothing ===

    if nvArgs.sigma > 0
        counts = smoothHistogram(counts, N, n, nvArgs.sigma);
    end

    % === Shannon entropy ===

    total = sum(counts);
    if total == 0
        H = 0;
        return;
    end

    q = counts / total;          % normalize to PMF
    q(q == 0) = [];              % 0 * log(0) = 0 convention

    H = -sum(q .* (log(q) / log(nvArgs.base)));

    if nvArgs.normalize
        H = H / (log(totalBins) / log(nvArgs.base));
    end

end


% =====================================================================
%  LOCAL FUNCTION
% =====================================================================

function counts = smoothHistogram(counts, N, n, sigma)
%SMOOTHHISTOGRAM Separable circular Gaussian convolution.
%
%   Smooths an n-dimensional histogram (stored as a flat vector of
%   length N^n) by convolving with a circular Gaussian kernel along
%   each dimension independently via FFT.

    % Build 1-D circular Gaussian kernel of length N.
    % The circular distance from 0 for each bin index is
    % min(x, N - x), giving a wrapped Gaussian.
    x = (0:N-1)';
    d = min(x, N - x);
    kernel = exp(-d.^2 / (2 * sigma^2));
    kernel = kernel / sum(kernel);  % normalize to unit sum

    kernelFFT = fft(kernel);

    % Reshape flat histogram to an n-dimensional array.
    % For n = 1, use [N, 1] to keep it a column vector.
    if n == 1
        histND = counts;
    else
        histND = reshape(counts, repmat(N, 1, n));
    end

    % Apply separable circular convolution along each dimension.
    % The Gaussian kernel is the same in every dimension (isotropic
    % smoothing). Reshaping kernelFFT so that its N elements lie
    % along dimension dim ensures correct broadcasting with the
    % n-dimensional FFT of the histogram.
    for dim = 1:n
        sz = ones(1, max(n, 2));
        sz(dim) = N;
        kFFT = reshape(kernelFFT, sz);
        histND = real(ifft( ...
            fft(histND, [], dim) .* kFFT, [], dim));
    end

    % Clamp any small negative values from floating-point
    % arithmetic and flatten back to a column vector.
    counts = max(histND(:), 0);

end