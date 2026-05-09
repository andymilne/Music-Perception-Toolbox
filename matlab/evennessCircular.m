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

    arguments
        p
        period (1,1) {mustBeNumeric, mustBePositive}
        sigma (1,1) {mustBeNumeric, mustBeNonnegative} = 0
        nvArgs.nDraws (1,1) {mustBePositive, mustBeInteger} = 10000
        nvArgs.rngSeed = []
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
