function val = v22_directIP(p_A, w_A, p_B, w_B, sigma, r)
%V22_DIRECTIP  Brute-force <T_A, T_B> by enumeration of distinct r-tuples.
%
%   VAL = V22_DIRECTIP(P_A, W_A, P_B, W_B, SIGMA, R) computes the
%   distinct-index inner product
%       sum over distinct r-tuples (tA, tB) of
%         prod(w_A[tA]) * prod(w_B[tB]) * prod_l exp(-(p_A[tA(l)] - p_B[tB(l)])^2 / (4 sigma^2))
%   times the conventional (sigma * sqrt(pi))^r prefactor.
%
%   Test helper for verifying mobius.innerProductOrbit at small (n, r);
%   cost is O(n!^2 / (n-r)!^2 * r), practical only for n <= 8, r <= 4.

    p_A = p_A(:); w_A = w_A(:); p_B = p_B(:); w_B = w_B(:);
    permA = v22_orderedTuples(numel(p_A), r);
    permB = v22_orderedTuples(numel(p_B), r);
    total = 0.0;
    for ia = 1:size(permA, 1)
        tA = permA(ia, :);
        wAp = prod(w_A(tA));
        cA = p_A(tA);
        for ib = 1:size(permB, 1)
            tB = permB(ib, :);
            wBp = prod(w_B(tB));
            cB = p_B(tB);
            kernel = prod(exp(-((cA - cB).^2) / (4 * sigma^2)));
            total = total + wAp * wBp * kernel;
        end
    end
    val = total * (sigma * sqrt(pi))^r;
end
