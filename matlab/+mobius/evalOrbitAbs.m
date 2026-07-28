function [vals, ratios] = evalOrbitAbs(p, w, sigma, r, x, opts)
%MOBIUS.EVALORBITABS  Möbius point evaluator for single multiset absolute-mode tensor.
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
        opts.wrap (1,:) char ...
            {mustBeMember(opts.wrap, {'full-image', 'single-image'})} ...
            = 'full-image'
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
    % Resolve Inf to the accuracy-floor width so the absolute orbit
    % truncates at the 1e-12 parity floor rather than summing exactly
    % into the far tail (uniform with Python and the kernel path).
    opts.truncationSigmas = internal.accuracyFloor('resolve', ...
                                                   opts.truncationSigmas);

    inv_2s2 = 1.0 / (2 * sigma^2);

    % Per-block factoring. Non-periodic always uses the 1-D variance/mean
    % reduction (Q_B = var(x_B) + m*(mean(x_B) - p)^2), which reduces the
    % m-D block sum to a 1-D Gaussian kernel sum at effective sigma_eff =
    % sigma/sqrt(m) times a per-query prefactor exp(-var/2sigma^2), and
    % whose source sum is culled by internal.gaussianKernelSum. Periodic
    % uses the same reduction where it survives wrapping --- the block span
    % and the truncation window both within half the circle --- which is
    % exactly the regime where the culled (circular) kernel sum applies;
    % elsewhere it falls back to the exact direct broadcast.
    if opts.is_per
        rWin = sqrt(2.0) * opts.truncationSigmas * sigma;
        perHelperGlobal = opts.period > 2.0 * rWin;
    else
        perHelperGlobal = false;
    end

    [uniqueBlocks, partBlockIdx, mus] = mobius.getPartitionBlockStructure(r);

    % Each block (a subset of the r positions) recurs across the set
    % partitions, and its factor --- a 1-D Gaussian kernel sum over the N
    % sources --- is the dominant cost. Evaluate each distinct block's
    % factor once here; the reuse across partitions and the alternating sum
    % are shared with the relative evaluator through
    % mobius.mobiusPartitionCombine.
    blockContrib = cell(1, numel(uniqueBlocks));
    for k = 1:numel(uniqueBlocks)
        B = uniqueBlocks{k};
        m = numel(B);
        x_B = x(B, :);
        if m == 1
            wm = w;
        else
            wm = w .^ m;
        end
        sigmaEff = sigma / sqrt(m);

        if ~opts.is_per
            if m == 1
                mean_x = x_B;
                var_x  = zeros(1, n_q_total);
            else
                mean_x = sum(x_B, 1) / m;
                var_x  = sum((x_B - mean_x).^2, 1);
            end
            useReduction = true;
        else
            % Circular mean/variance relative to the block's reference coordinate
            % (translation-invariant offsets), so a block sitting on the
            % period seam is handled correctly.
            if m == 1
                mean_x = x_B;
                var_x  = zeros(1, n_q_total);
                spanOk = true;
            else
                off = x_B - x_B(1, :);
                off = off - opts.period * floor(off / opts.period + 0.5);
                meanOff = sum(off, 1) / m;
                var_x = sum((off - meanOff).^2, 1);
                mean_x = x_B(1, :) + meanOff;
                spanVals = max(off, [], 1) - min(off, [], 1);
                spanOk = all(spanVals < 0.5 * opts.period);
            end
            useReduction = perHelperGlobal && spanOk;
        end

        if useReduction
            % Factored 1-D path.
            prefactor = exp(-var_x(:) * inv_2s2);
            if opts.is_per
                kw = {'truncationSigmas', opts.truncationSigmas, ...
                      'kernelPrecision', opts.kernelPrecision, ...
                      'isPer', true, 'period', opts.period, ...
                      'wrap', opts.wrap};
            else
                kw = {'truncationSigmas', opts.truncationSigmas, ...
                      'kernelPrecision', opts.kernelPrecision};
            end
            kernelSum = internal.gaussianKernelSum( ...
                p(:).', wm(:), mean_x(:).', sigmaEff, kw{:});
            blockContrib{k} = prefactor .* kernelSum(:);
        else
            % Direct (m, N, n_q) broadcast: exact fallback for the periodic
            % small-circle / wide-block case. Honour the density's wrap:
            % full-image via wrappedGaussian1d, single-image via nearest-
            % image reduction (the pre-v3 behaviour).
            x_B_re = reshape(x_B, m, 1, n_q_total);
            p_re = reshape(p, 1, N, 1);
            diffs = x_B_re - p_re;
            if strcmp(opts.wrap, 'full-image')
                % Per-coordinate 1-D wrapped Gaussian at the original sigma
                % (density-kernel convention, exponent_denominator = 2);
                % the r-tuple wrapped kernel over the block factors as
                % prod_k theta_1D(d_k; sigma). The sigma/sqrt(m)
                % effective width belongs to the *useReduction* branch,
                % where the m-D block sum is collapsed to a 1-D
                % Gaussian at mean_x; it is not correct here where each
                % coordinate is broadcast separately.
                theta = internal.wrappedGaussian1d(diffs, sigma, ...
                    opts.period, opts.truncationSigmas, 2);
                kernel = reshape(prod(theta, 1), N, n_q_total);
                blockContrib{k} = kernel' * wm;
            else
                diffs = diffs - opts.period * floor(diffs / opts.period + 0.5);
                sqSum = reshape(sum(diffs .* diffs, 1), N, n_q_total);
                kernel = exp(-sqSum * inv_2s2);
                blockContrib{k} = kernel' * wm;
            end
        end
    end

    [total, maxAbsTerm] = mobius.mobiusPartitionCombine( ...
        blockContrib, partBlockIdx, mus, opts.returnCancellationRatio);

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
