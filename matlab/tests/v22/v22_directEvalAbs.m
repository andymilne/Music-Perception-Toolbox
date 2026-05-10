function vals = v22_directEvalAbs(p, w, sigma, r, x, is_per, period)
%V22_DIRECTEVALABS  Brute-force T_abs(x_q) by enumeration of distinct r-tuples.
%
%   VALS = V22_DIRECTEVALABS(P, W, SIGMA, R, X) computes T_abs at each
%   query column of X by direct summation over distinct ordered
%   r-tuples of source indices:
%
%     T_abs(x) = sum_{distinct r-tuples (i_1..i_r)}
%                  prod_s w[i_s] * exp(-(x[s] - p[i_s])^2 / (2 sigma^2))
%
%   VALS = V22_DIRECTEVALABS(..., IS_PER, PERIOD) wraps differences to
%   [-PERIOD/2, PERIOD/2) before squaring (periodic mode).
%
%   Test helper for verifying mobius.evalOrbitAbs at small (n, r);
%   cost is O(n!/(n-r)! * r * n_q), practical only for n <= 8, r <= 4.

    if nargin < 6; is_per = false; end
    if nargin < 7; period = 0.0; end

    p = p(:); w = w(:);
    n_q = size(x, 2);
    tuples = v22_orderedTuples(numel(p), r);
    vals = zeros(n_q, 1);
    inv2s2 = 1 / (2 * sigma^2);
    for ip = 1:size(tuples, 1)
        tup = tuples(ip, :);
        wp = prod(w(tup));
        cp = p(tup);  % (r, 1)
        % Differences (r, n_q): cp_repeated minus x.
        diffs = x - cp;
        if is_per
            diffs = diffs - period * floor(diffs / period + 0.5);
        end
        kernel = exp(-sum(diffs .* diffs, 1) * inv2s2);  % (1, n_q)
        vals = vals + wp * kernel(:);
    end
end
