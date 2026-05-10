function [vals, ratios] = evalOrbitAbs(p, w, sigma, r, x, opts)
%MOBIUS.EVALORBITABS  Möbius point evaluator for SA absolute-mode tensor.
%
%   VALS = MOBIUS.EVALORBITABS(P, W, SIGMA, R, X) computes T_abs(x_q)
%   for each column of X without materialising the (r, n_distinct_tuples)
%   centres array. Uses the set-partition Möbius decomposition
%
%     T_abs(x) = sum_pi mu(pi) * prod_l [ sum_i w_i^{m_l} *
%                                          exp(-sum_{k in B_l} d^2(x_k, p_i)/(2 sigma^2)) ]
%
%   where pi ranges over set partitions of {1..r} with blocks
%   B_1, ..., B_q of sizes m_1, ..., m_q, and d is plain difference
%   (non-periodic) or wrapped difference (periodic).
%
%   VALS = MOBIUS.EVALORBITABS(..., 'is_per', true, 'period', P)
%   wraps differences into [-P/2, P/2) before squaring (periodic mode).
%
%   [VALS, RATIOS] = MOBIUS.EVALORBITABS(..., 'returnCancellationRatio', true)
%   additionally returns per-query cancellation ratios; values near 1
%   indicate no cancellation, values << 1 indicate digits lost. Same
%   interpretation as in the orbit-IP machinery.
%
%   Inputs:
%     P                       (N, 1) double — source positions.
%     W                       (N, 1) double — source weights.
%     SIGMA                   (1, 1) positive double.
%     R                       integer >= 1.
%     X                       (R, n_q) double — query points, one per column.
%     opts.is_per             logical (default false).
%     opts.period             double (default 0; consulted only when is_per).
%     opts.returnCancellationRatio  logical (default false).
%
%   Memory: O(B_r * m_max * N * n_q) per partition (transient, freed
%   between partitions). For r=4, N=20, n_q=1000: ~5 MB. The
%   centre-array brute-force path's peak would be ~3.7 GB.
%
%   See also MOBIUS.EVALORBITREL, MOBIUS.GETSETPARTITIONSWITHMOBIUS.

    arguments
        p (:,1) double
        w (:,1) double
        sigma (1,1) double {mustBePositive}
        r (1,1) {mustBeInteger, mustBePositive}
        x (:,:) double
        opts.is_per (1,1) logical = false
        opts.period (1,1) double = 0.0
        opts.returnCancellationRatio (1,1) logical = false
    end

    if size(x, 1) ~= r
        error('mobius:evalOrbitAbs:queryShape', ...
            'x must have size (r, n_q) with r=%d; got %dx%d.', ...
            r, size(x, 1), size(x, 2));
    end

    n_q = size(x, 2);
    N = numel(p);
    inv_2s2 = 1.0 / (2 * sigma^2);

    partitions = mobius.getSetPartitionsWithMobius(r);
    total = zeros(n_q, 1);
    maxAbsTerm = zeros(n_q, 1);

    for pi = 1:numel(partitions)
        blocks = partitions(pi).blocks;
        mu = partitions(pi).mu;

        blockFactor = ones(n_q, 1);
        for b = 1:numel(blocks)
            B = blocks{b};
            m = numel(B);
            % x_B: (m, n_q); p: (N, 1).
            % Build diffs of shape (m, N, n_q) via implicit expansion.
            x_B = x(B, :);
            x_B_re = reshape(x_B, m, 1, n_q);
            p_re = reshape(p, 1, N, 1);
            diffs = x_B_re - p_re;
            if opts.is_per
                diffs = diffs - opts.period * floor(diffs / opts.period + 0.5);
            end
            % Sum-of-squares over the block-slot axis (dim 1).
            sqSum = reshape(sum(diffs .* diffs, 1), N, n_q);
            kernel = exp(-sqSum * inv_2s2);  % (N, n_q)
            if m == 1
                wm = w;
            else
                wm = w .^ m;
            end
            % blockFactor[q] = sum_i w_i^m * kernel[i, q]
            blockFactor = blockFactor .* (kernel' * wm);
        end

        term = mu * blockFactor;
        total = total + term;
        maxAbsTerm = max(maxAbsTerm, abs(term));
    end

    vals = total;
    if opts.returnCancellationRatio
        ratios = ones(n_q, 1);
        nz = maxAbsTerm > 0;
        ratios(nz) = abs(total(nz)) ./ maxAbsTerm(nz);
    else
        ratios = [];
    end
end
