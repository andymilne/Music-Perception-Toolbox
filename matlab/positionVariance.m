function V = positionVariance(idx, signs, sigma)
%POSITIONVARIANCE Variance of a signed sum of independently jittered
%positions.
%
%   Used by the position-aware paths of SAMENESS, COHERENCE, and
%   other measures whose readouts depend on differences of intervals
%   derived from a shared position set under independent positional
%   jitter p_k ~ N(p_k, sigma^2).
%
%   Computes:
%       V = sigma^2 * sum_a (sum of signs at index a)^2
%
%   Repeated indices add their signs algebraically before squaring,
%   so a position that enters once with sign +1 and once with sign
%   -1 contributes nothing to the variance, while one that enters
%   twice with the same sign contributes 4 * sigma^2.
%
%   Inputs
%       idx   - Position indices (with possible repeats).
%       signs - Signed contributions, same length as idx (typically
%               +1 or -1).
%       sigma - Per-position standard deviation.
%
%   Output
%       V     - Variance of the signed sum.

    [uIdx, ~, grp] = unique(idx);
    nU = numel(uIdx);
    netSigns = zeros(1, nU);
    for m = 1:numel(idx)
        netSigns(grp(m)) = netSigns(grp(m)) + signs(m);
    end
    V = sigma^2 * sum(netSigns.^2);
end
