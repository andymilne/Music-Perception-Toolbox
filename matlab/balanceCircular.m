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

    arguments
        p
        w
        period (1,1) {mustBeNumeric, mustBePositive}
        sigma (1,1) {mustBeNumeric, mustBeNonnegative} = 0
        nvArgs.nDraws (1,1) {mustBePositive, mustBeInteger} = 10000
        nvArgs.rngSeed = []
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
