function Sigma = kernelCov(r, nvArgs)
%KERNELCOV Kernel covariance for an ordered tuple, from three sources of
%variance.
%
%   Sigma = kernelCov(r, 'differenced', tf)
%   Sigma = kernelCov(r, 'differenced', tf, 'sdValue', sp)
%   Sigma = kernelCov(r, 'differenced', tf, 'sdInterval', si)
%   Sigma = kernelCov(r, 'differenced', tf, 'sdShift', ss)
%   Sigma = kernelCov(..., Name, Value)
%
%   Builds the r x r covariance matrix of an ordered attribute's tuples
%   from three independent sources of perceptual uncertainty, each
%   included only when its width is set: independent noise on
%   the *values* themselves (onsets, or pitches), sdValue; independent noise on
%   the *intervals* between consecutive values, sdInterval; and a
%   *common shift* of the whole tuple, sdShift. How each reaches the
%   tuple depends on whether the tuple holds the values themselves
%   or their first differences, which the mandatory 'differenced'
%   flag declares.
%
%   With Del the first-differencing map, (Del p)_i = p_{i+1} - p_i,
%   taken at the size its operand requires -- (r - 1) x r on a tuple of
%   r values, r x (r + 1) on a tuple of the r differences of r + 1
%   values -- Del+ its pseudoinverse, and J = ones(r):
%
%     differenced = false:
%       Sigma = sdValue^2 * I + sdInterval^2 * (Del+ * Del+')
%               + sdShift^2 * J
%     differenced = true:
%       Sigma = sdValue^2 * (Del * Del') + sdInterval^2 * I
%               + sdShift^2 * J
%
%   Del+ * Del+' equals the centred cumulative sum P * S * S' * P, with
%   S the r x (r - 1) cumulative-sum map and P = I - J/r the centring
%   projector, which is how it is built here. The two cases are one
%   model: differencing a tuple of r values carries the first to the
%   second at tuple size r - 1, since Del * 1 = 0 annihilates the ridge
%   and Del * Del+ = I.
%
%   Undifferenced values (r values). Interval noise accumulates from
%   one value to the next, a random walk that P centres on the
%   tuple's mean so that no value is privileged (P * S * S' * P is
%   the covariance of the centred cumulative sums of r - 1 independent
%   interval errors; S is fixed only up to a base point, and the
%   choices differ by a multiple of the all-ones vector, which P
%   removes). The ridge tolerates a common shift of every
%   value: a transposition of pitches, a displacement of onsets. As
%   sdShift grows the kernel's precision tends to the relative-mode
%   projector, so isRel = true is the exact (infinite-sdShift) limit
%   and the ridge its graded counterpart.
%
%   Differenced values (r consecutive differences of r + 1 values).
%   Value noise reaches each interval through its two endpoints:
%   D * D' is tridiagonal, 2 on the diagonal and -1 beside it, since
%   adjacent intervals share an endpoint (perturbing one interior
%   value lengthens one interval and shortens its neighbour).
%   Interval noise is independent per interval. On times the two are
%   the two levels of the Wing & Kristofferson (1973) model, motor
%   delay variance on onsets and central timekeeper variance on
%   intervals. The ridge adds a constant to every interval, seldom the
%   equivalence wanted for uneven rhythms, so sdShift is usually
%   omitted here; on LOG-differenced values it becomes a common factor
%   on the intervals (a tempo change, or intervallic augmentation),
%   and is wanted again. sdValue corresponds to
%   sigmaSpace = 'position' and sdInterval to sigmaSpace = 'interval'
%   in nTupleEntropy.
%
%   The covariance is expressed in whatever coordinates the attribute
%   carries: cents or semitones for pitch, seconds for onsets, log
%   inter-onset intervals for multiplicative tempo tolerance. All three
%   widths are standard deviations in those coordinates; they are
%   squared internally.
%
%   Inputs:
%     r            — Tuple size; an integer >= 2.
%     differenced  — (mandatory Name-Value) false when the tuple holds
%                    values, true when it holds their first
%                    differences (as differenceEvents produces). There
%                    is no safe default, so it must be given.
%     sdValue   — Standard deviation of independent noise on each
%                    value (default 0).
%     sdInterval   — Standard deviation of independent noise on each
%                    interval between consecutive values (default 0).
%     sdShift      — Standard deviation of a common shift of the whole
%                    tuple (default 0).
%
%   Output:
%     Sigma        — r x r symmetric positive-definite covariance, ready
%                    to be passed as the sigma argument of buildMaet,
%                    evalMaet, simMaet, entropyMaet, windowedSimilarity,
%                    or windowedEntropy for an ordered (isExch = false),
%                    absolute (isRel = false), non-periodic
%                    (isPer = false) attribute with r == K.
%
%   Errors if r < 2, any width is negative or non-finite, or the result
%   is not positive-definite. On undifferenced values that needs
%   sdValue > 0, or sdInterval and sdShift both non-zero: the centred
%   walk annihilates the all-ones direction and the ridge is rank one,
%   so neither serves alone. On differenced values it needs
%   sdValue > 0 or sdInterval > 0, only the ridge alone failing.
%
%   Reference: Wing, A. M., & Kristofferson, A. B. (1973). Response
%   delays and the timing of discrete motor responses. Perception &
%   Psychophysics, 14(1), 5-12.
%
%   See also buildMaet, simMaet, entropyMaet, windowedSimilarity,
%   differenceEvents, nTupleEntropy.

    arguments
        r (1, 1) double
        nvArgs.differenced = []
        nvArgs.sdValue (1, 1) double = 0.0
        nvArgs.sdInterval (1, 1) double = 0.0
        nvArgs.sdShift (1, 1) double = 0.0
    end

    if rem(r, 1) ~= 0 || r < 1
        error('mpt:kernelCov:badR', ...
            'kernelCov: r must be a positive integer.');
    end
    if r == 1
        error('mpt:kernelCov:rOne', ...
            ['kernelCov: at r = 1 the covariance reduces to a scalar ' ...
             'variance, which is indistinguishable from a scalar ' ...
             'sigma; pass the equivalent standard deviation as the ' ...
             'ordinary sigma argument instead (sqrt(2*sdValue^2 + ' ...
             'sdInterval^2 + sdShift^2) if differenced, ' ...
             'sqrt(sdValue^2 + sdShift^2) if not).']);
    end
    d = nvArgs.differenced;
    if isempty(d) || ~isscalar(d) || ~(islogical(d) || isnumeric(d)) ...
            || (isnumeric(d) && ~any(d == [0 1]))
        error('mpt:kernelCov:differenced', ...
            ['kernelCov: ''differenced'' must be given, true (the ' ...
             'tuple holds first differences) or false (it holds ' ...
             'values).']);
    end
    differenced = logical(d);
    sds = {nvArgs.sdValue, nvArgs.sdInterval, nvArgs.sdShift};
    names = {'sdValue', 'sdInterval', 'sdShift'};
    for i = 1:3
        if ~isfinite(sds{i}) || sds{i} < 0
            error('mpt:kernelCov:badSd', ...
                ['kernelCov: %s must be a finite non-negative ' ...
                 'standard deviation; got %g. For exact common-shift ' ...
                 'invariance use isRel = true rather than an infinite ' ...
                 'sdShift.'], names{i}, sds{i});
        end
    end
    sp2 = nvArgs.sdValue^2;
    si2 = nvArgs.sdInterval^2;
    ss2 = nvArgs.sdShift^2;
    J = ones(r);
    if differenced
        % D * D': tridiagonal, 2 on the diagonal, -1 on the first
        % off-diagonals.
        ddt = 2 * eye(r) - diag(ones(r - 1, 1), 1) - diag(ones(r - 1, 1), -1);
        Sigma = sp2 * ddt + si2 * eye(r) + ss2 * J;
        why = ['Positive-definiteness needs sdValue > 0 or ' ...
               'sdInterval > 0; the ridge alone is rank one.'];
    else
        % S: r x (r - 1) cumulative sums (value i carries the first
        % i - 1 interval errors); P centres them on the tuple's mean.
        S = tril(ones(r, r - 1), -1);
        P = eye(r) - J / r;
        pssp = P * (S * S') * P;
        Sigma = sp2 * eye(r) + si2 * pssp + ss2 * J;
        why = ['Positive-definiteness needs sdValue > 0, or ' ...
               'sdInterval and sdShift both non-zero: the centred ' ...
               'walk annihilates the common-shift direction and the ' ...
               'ridge is rank one, so neither serves alone.'];
    end
    Sigma = (Sigma + Sigma') / 2;

    % Definiteness check (PSD validation always errors). An eigenvalue
    % test rather than a Cholesky attempt: the singular cases are
    % exactly singular, and rounding can let a factorization of one
    % through.
    ev = eig(Sigma);
    if min(ev) <= 1e-12 * max(max(ev), realmin)
        error('mpt:kernelCov:singular', ...
            ['kernelCov: the resulting covariance is singular. ' why]);
    end
end
