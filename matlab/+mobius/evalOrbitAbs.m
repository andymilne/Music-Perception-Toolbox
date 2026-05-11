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
%   X may be a 2-D matrix of shape (r, n_q) or a higher-dimensional
%   array of shape (r, ...) where ... is any product of trailing
%   dimensions. The output VALS has shape (...) with trailing
%   dimensions preserved (a column vector when ... is (n_q,)). The
%   higher-dimensional form is the entry point used by
%   MOBIUS.EVALORBITREL when it batches its u-grid loop into a single
%   vectorised call.
%
%   VALS = MOBIUS.EVALORBITABS(..., 'is_per', true, 'period', P)
%   wraps differences into [-P/2, P/2) before squaring (periodic mode).
%
%   [VALS, RATIOS] = MOBIUS.EVALORBITABS(..., 'returnCancellationRatio', true)
%   additionally returns per-query cancellation ratios; values near 1
%   indicate no cancellation, values << 1 indicate digits lost. RATIOS
%   has the same shape as VALS.
%
%   Inputs:
%     P                       (N, 1) double — source positions.
%     W                       (N, 1) double — source weights.
%     SIGMA                   (1, 1) positive double.
%     R                       integer >= 1.
%     X                       (R, ...) double — query points; the
%                              first dimension must equal R, the
%                              remaining dimensions are query indices.
%     opts.is_per             logical (default false).
%     opts.period             double (default 0; consulted only when is_per).
%     opts.returnCancellationRatio  logical (default false).
%
%   Memory: O(B_r * m_max * N * n_q_total) per partition (transient,
%   freed between partitions), where n_q_total is the product of
%   trailing dimensions. Callers responsible for sizing the trailing
%   dims to fit in available memory; MOBIUS.EVALORBITREL chunks the
%   query axis when it batches its u-grid loop.
%
%   See also MOBIUS.EVALORBITREL, MOBIUS.GETSETPARTITIONSWITHMOBIUS.

    arguments
        p (:,1) double
        w (:,1) double
        sigma (1,1) double {mustBePositive}
        r (1,1) {mustBeInteger, mustBePositive}
        x double
        opts.is_per (1,1) logical = false
        opts.period (1,1) double = 0.0
        opts.returnCancellationRatio (1,1) logical = false
        opts.truncationSigmas (1,1) double = mptDefaults('truncationSigmas')
        opts.kernelPrecision (1,:) char ...
            {mustBeMember(opts.kernelPrecision, {'double','single'})} ...
            = mptDefaults('kernelPrecision')
    end

    sz = size(x);
    if sz(1) ~= r
        error('mobius:evalOrbitAbs:queryShape', ...
            'x must have size (r, ...) with r=%d; got first dim %d.', ...
            r, sz(1));
    end

    % Collapse trailing dimensions to a single query axis. Restore
    % shape on output. This lets the inner loop stay 2-D while the
    % API accepts any (r, ...) shape.
    is2D = (numel(sz) == 2);
    if numel(sz) >= 2
        outShape = sz(2:end);
    else
        outShape = [1, 1];
    end
    n_q_total = max(prod(outShape), 1);
    if n_q_total == 0
        vals = zeros(outShape);
        if opts.returnCancellationRatio
            ratios = ones(outShape);
        else
            ratios = [];
        end
        return;
    end
    x = reshape(x, r, n_q_total);

    N = numel(p);
    inv_2s2 = 1.0 / (2 * sigma^2);

    % Note: opts.truncationSigmas / opts.kernelPrecision are accepted
    % so this function can be called uniformly from Stage 4 wrappers,
    % but they are currently NO-OP in the orbit path. The helper's
    % truncated kernel sum carries per-query loop overhead that
    % exceeds the savings at typical orbit-path N (~50–300 partials
    % per template). Routing through it would be a regression for the
    % regimes where the orbit path is selected. A vectorised 1-D
    % truncated kernel sum (planned follow-up) will unlock real
    % speedup here; until then, the orbit-path stays on the exact
    % tensor-broadcast code below.

    partitions = mobius.getSetPartitionsWithMobius(r);
    total = zeros(n_q_total, 1);
    maxAbsTerm = zeros(n_q_total, 1);

    for pi = 1:numel(partitions)
        blocks = partitions(pi).blocks;
        mu = partitions(pi).mu;

        blockFactor = ones(n_q_total, 1);
        for b = 1:numel(blocks)
            B = blocks{b};
            m = numel(B);
            % x_B: (m, n_q_total); p: (N, 1).
            % Build diffs of shape (m, N, n_q_total) via implicit expansion.
            x_B = x(B, :);
            x_B_re = reshape(x_B, m, 1, n_q_total);
            p_re = reshape(p, 1, N, 1);
            diffs = x_B_re - p_re;
            if opts.is_per
                diffs = diffs - opts.period * floor(diffs / opts.period + 0.5);
            end
            % Sum-of-squares over the block-slot axis (dim 1).
            sqSum = reshape(sum(diffs .* diffs, 1), N, n_q_total);
            kernel = exp(-sqSum * inv_2s2);  % (N, n_q_total)
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

    % Restore shape: trailing dims as in the input.
    if is2D
        % Caller passed (r, n_q); preserve column-vector return for
        % v2.0/v2.1 compatibility.
        vals = total;
    else
        vals = reshape(total, outShape);
    end
    if opts.returnCancellationRatio
        ratiosFlat = ones(n_q_total, 1);
        nz = maxAbsTerm > 0;
        ratiosFlat(nz) = abs(total(nz)) ./ maxAbsTerm(nz);
        if is2D
            ratios = ratiosFlat;
        else
            ratios = reshape(ratiosFlat, outShape);
        end
    else
        ratios = [];
    end
end
