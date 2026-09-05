function ipval = innerProductDirectAbsSingleMultiset(p_x, w_x, p_y, w_y, sigma, r, ...
                                            isPer, period)
%REFERENCE.INNERPRODUCTDIRECTABSSINGLEMULTISET  Direct-enumeration single-multiset IP, abs mode.
%
%   Test oracle only (tests/reference); nothing shipped calls it.
%
%   IPVAL = REFERENCE.INNERPRODUCTDIRECTABSSINGLEMULTISET(P_X, W_X, P_Y, W_Y, SIGMA, R,
%                                            IS_PER, PERIOD)
%   computes <T_X, T_Y> for two single-multiset absolute-mode densities at order
%   R >= 1 by enumerating ordered R-tuples on each side and summing
%       <T_X, T_Y> = (sigma * sqrt(pi))^R *
%                    sum_{J, K} wJ_x[J] * wJ_y[K] *
%                               exp(-||centres_x[:,J] - centres_y[:,K]||^2 / (4 sigma^2))
%
%   No Möbius alternating sum is involved, so the result is exact
%   (no catastrophic cancellation) for any K_x, K_y >= R. It is the
%   enumerated comparison point for the batched Möbius route of
%   MOBIUS.MAPERATTRINNERMATRIX, which no longer partitions events by
%   K_eff (accuracy is governed by truncationSigmas).
%
%   NaN tolerance: NaN entries in P_X / W_X / P_Y / W_Y are dropped
%   per event before enumeration. If the dropped count leaves either
%   side with fewer than R valid values, IPVAL = 0 by convention
%   (can't form an R-tuple).
%
%   Cost: O(K_x! / (K_x - R)! * K_y! / (K_y - R)! * R) per call.
%
%   See also MOBIUS.ORBITINNERABSSINGLEMULTISET, MOBIUS.MAPERATTRINNERMATRIX.

    p_x = p_x(:); w_x = w_x(:);
    p_y = p_y(:); w_y = w_y(:);

    valid_x = ~(isnan(p_x) | isnan(w_x));
    valid_y = ~(isnan(p_y) | isnan(w_y));
    p_x = p_x(valid_x); w_x = w_x(valid_x);
    p_y = p_y(valid_y); w_y = w_y(valid_y);
    K_x = numel(p_x);
    K_y = numel(p_y);

    if K_x < r || K_y < r
        ipval = 0;
        return;
    end

    if r == 1
        % Direct kernel sum at r=1 (no enumeration needed).
        diffs = p_x - p_y.';
        if isPer
            diffs = diffs - period * floor(diffs / period + 0.5);
        end
        K_mat = exp(-(diffs.^2) / (4 * sigma^2));
        ipval = sigma * sqrt(pi) * sum(sum((w_x * w_y.') .* K_mat));
        return;
    end

    % r >= 2: enumerate ordered r-tuples and contract.
    [U_x, wJ_x] = localBuildPerms(p_x, w_x, r);   % U_x: (r, nJ_x)
    [U_y, wJ_y] = localBuildPerms(p_y, w_y, r);

    nJ_x = size(U_x, 2);
    nJ_y = size(U_y, 2);

    % Pairwise differences across ordered tuples: (r, nJ_x, nJ_y).
    diffs = reshape(U_x, r, nJ_x, 1) - reshape(U_y, r, 1, nJ_y);
    if isPer
        diffs = diffs - period * floor(diffs / period + 0.5);
    end
    Q = reshape(sum(diffs.^2, 1), nJ_x, nJ_y);
    K_mat = exp(-Q / (4 * sigma^2));

    ipval = (sigma * sqrt(pi))^r * (wJ_x * (K_mat * wJ_y.'));
end


function [U, wJ] = localBuildPerms(p, w, r)
%LOCALBUILDPERMS  Mirror of buildExpTens's ordered-r-tuple construction.

    K = numel(p);
    nC = nchoosek(K, r);
    nP = factorial(r);
    nJ = nP * nC;

    nck = nchoosek(1:K, r)';     % (r, nC)
    allPerms = perms(1:r)';      % (r, nP)

    Ju = zeros(r, nJ);
    offset = 0;
    for i = 1:nP
        Ju(:, offset + 1 : offset + nC) = nck(allPerms(:, i), :);
        offset = offset + nC;
    end

    U  = reshape(p(Ju), r, nJ);
    wJ = reshape(prod(reshape(w(Ju), r, nJ), 1), 1, nJ);
end
