function Sigma = intervalKernelCov(r, nvArgs)
%INTERVALKERNELCOV Kernel covariance for an ordered tuple of consecutive
%differences.
%
%   Sigma = intervalKernelCov(r)
%   Sigma = intervalKernelCov(r, 'sdPosition', sp)
%   Sigma = intervalKernelCov(r, 'sdInterval', si)
%   Sigma = intervalKernelCov(r, 'sdShift', ss)
%   Sigma = intervalKernelCov(..., Name, Value)
%
%   Builds the r x r covariance matrix
%
%       Sigma = sdPosition^2 * (D * D') + sdInterval^2 * eye(r)
%               + sdShift^2 * ones(r)
%
%   for an ordered attribute whose event tuples are r consecutive
%   differences (intervals) of r + 1 underlying positions, where D is
%   the r x (r + 1) first-differencing map (so D * D' is tridiagonal:
%   2 on the diagonal, -1 on the first off-diagonals).
%
%   This parametrization is meaningful ONLY for first-differenced
%   multisets. sdPosition builds D * D', whose off-diagonal entries
%   encode the endpoints that neighbouring differences share; an
%   undifferenced multiset has no such shared endpoints, so on one the
%   term imposes correlations the data do not contain. Nothing here
%   inspects the multiset, so passing the result for an undifferenced
%   attribute raises no error: the caller is responsible for applying
%   it only to interval tuples. For an undifferenced attribute,
%   independent per-value noise is what the ordinary scalar sigma
%   already provides, and a matrix covariance is warranted only for a
%   common-shift ridge.
%
%   The three terms are three independently specified sources of
%   perceptual uncertainty, added because their sources are
%   independent:
%
%     sdPosition — uncertainty on the underlying *positions* from
%       which the differences are formed. Shared endpoints propagate
%       it to the tridiagonal sdPosition^2 * D * D': perturbing one
%       interior position lengthens one interval and shortens its
%       neighbour. This is the exact counterpart of
%       sigmaSpace = 'position' in nTupleEntropy.
%     sdInterval — uncertainty on each *interval* itself, independent
%       across intervals (sigmaSpace = 'interval').
%     sdShift — graded tolerance for a *common shift* of the whole
%       tuple, the rank-one ridge sdShift^2 * ones(r). A common shift
%       of an interval tuple is a transposition when the values are
%       pitch intervals and a tempo change when they are log
%       inter-onset intervals. As sdShift grows the kernel's precision
%       tends to the relative-mode projector, so isRel = true is the
%       exact (infinite-sdShift) limit; a matrix covariance expresses
%       the graded counterpart.
%
%   In the time reading, the first two terms are the two levels of the
%   Wing & Kristofferson (1973) timing model: motor implementation
%   delays attach to onsets (sdPosition), central timekeeper variance
%   attaches to intervals (sdInterval).
%
%   The covariance is expressed in whatever coordinates the attribute
%   carries: log inter-onset intervals for multiplicative tempo
%   tolerance, semitones (or cents) for pitch steps. All three
%   arguments are standard deviations in those coordinates; they are
%   squared internally.
%
%   Inputs:
%     r          — Tuple size (number of consecutive differences);
%                  a positive integer.
%     sdPosition — Standard deviation of independent noise on each
%                  underlying position (default 0).
%     sdInterval — Standard deviation of independent noise on each
%                  interval (default 0).
%     sdShift    — Standard deviation of a common shift of the whole
%                  tuple (default 0).
%
%   Output:
%     Sigma      — r x r symmetric positive-definite covariance, ready
%                  to be passed as the sigma argument of buildExpTens,
%                  evalExpTens, cosSimExpTens, entropyExpTens,
%                  windowedSimilarity, or windowedEntropy for an
%                  ordered (isSym = false), absolute (isRel = false),
%                  non-periodic (isPer = false) attribute with r == K.
%
%   Errors if r < 1, any argument is negative or non-finite, or the
%   resulting matrix is singular (sdShift alone is rank one for
%   r >= 2, so at least one of sdPosition and sdInterval must be
%   positive).
%
%   Reference: Wing, A. M., & Kristofferson, A. B. (1973). Response
%   delays and the timing of discrete motor responses. Perception &
%   Psychophysics, 14(1), 5-12.
%
%   See also buildExpTens, cosSimExpTens, entropyExpTens,
%   windowedSimilarity, nTupleEntropy.

    arguments
        r (1, 1) double
        nvArgs.sdPosition (1, 1) double = 0.0
        nvArgs.sdInterval (1, 1) double = 0.0
        nvArgs.sdShift (1, 1) double = 0.0
    end

    if rem(r, 1) ~= 0 || r < 1
        error('mpt:intervalKernelCov:badR', ...
            'intervalKernelCov: r must be a positive integer.');
    end
    if r == 1
        error('mpt:intervalKernelCov:rOne', ...
            ['intervalKernelCov: at r = 1 the covariance reduces to ' ...
             'a scalar variance, which is indistinguishable from a ' ...
             'scalar sigma; pass the equivalent standard deviation ' ...
             'sqrt(2*sdPosition^2 + sdInterval^2 + sdShift^2) as the ' ...
             'ordinary sigma argument instead.']);
    end
    sds = {nvArgs.sdPosition, nvArgs.sdInterval, nvArgs.sdShift};
    names = {'sdPosition', 'sdInterval', 'sdShift'};
    for i = 1:3
        if ~isfinite(sds{i}) || sds{i} < 0
            error('mpt:intervalKernelCov:badSd', ...
                ['intervalKernelCov: %s must be a finite non-negative ' ...
                 'standard deviation; got %g. For exact common-shift ' ...
                 'invariance use isRel = true rather than an infinite ' ...
                 'sdShift.'], names{i}, sds{i});
        end
    end

    % First-differencing map D: r x (r + 1); D * D' is tridiagonal with
    % 2 on the diagonal and -1 on the first off-diagonals.
    ddt = 2 * eye(r) - diag(ones(r - 1, 1), 1) - diag(ones(r - 1, 1), -1);
    Sigma = nvArgs.sdPosition^2 * ddt ...
        + nvArgs.sdInterval^2 * eye(r) ...
        + nvArgs.sdShift^2 * ones(r);

    % Definiteness check (PSD validation always errors): sdShift alone
    % is rank one for r >= 2.
    [~, flag] = chol(Sigma, 'lower');
    if flag ~= 0
        error('mpt:intervalKernelCov:singular', ...
            ['intervalKernelCov: the resulting covariance is ' ...
             'singular. sdShift alone is rank one, so at least one ' ...
             'of sdPosition and sdInterval must be positive.']);
    end
end
