function [vals, ratios] = evalOrbitRel(p, w, sigma, r, x_rel, opts)
%MOBIUS.EVALORBITREL  Möbius point evaluator for single multiset relative-mode tensor.
%
%   VALS = MOBIUS.EVALORBITREL(P, W, SIGMA, R, X_REL) computes T_rel
%   at each column of X_REL via u-grid quadrature of the translation
%   marginal:
%
%     T_rel(Δ) = (1/Z_t) * ∫ T_abs(u, u+Δ_1, ..., u+Δ_{r-1}) du
%
%   where Z_t = sigma * sqrt(2*pi/r) is the translation-mode normaliser.
%   The quadrature is intrinsic to the Möbius realisation of relative
%   mode: the alternating partition sum only factorises across {1, ..., r} at
%   fixed u, and integrating it analytically re-expands into the O(K^r)
%   tuple enumeration the decomposition exists to avoid. The grid
%   mirrors the tensor module's _orbit_inner_rel: periodic uses [0, P)
%   sampled at SAMPLES_PER_SIGMA points per sigma; non-periodic uses a
%   Gaussian-supported window extending 8*sigma beyond the alignment of
%   source positions and query trajectory.
%
%   Two integrand-evaluation strategies are available:
%
%   * Direct — each u-node costs one MOBIUS.EVALORBITABS evaluation;
%     per-query cost O(B_r * r * K * N_u).
%   * Factored (non-periodic only) — each partition block's factor
%     separates exactly as exp(-var(δ_B)/2σ²) * S_m(u + mean(δ_B)),
%     with S_m(v) = Σ_i w_i^m exp(-m (v - p_i)²/2σ²) a query-
%     independent smoothed event distribution at width σ/√m. Tabulating
%     S_1..S_r once and reading them back by local quintic (6-point
%     Lagrange) interpolation removes the K factor from the per-node
%     cost: per-query cost O(B_r * r * N_u) after a one-time
%     O(Σ_m N_fine_m * K) tabulation. The read-back accuracy is tied to
%     TRUNCATIONSIGMAS: the tabulation step targets a relative error at
%     the kernel-truncation floor exp(-k²/2), clamped to [1e-12, 1e-3]
%     (Inf targets 1e-12, the noise level of the u-grid quadrature;
%     kernelPrecision 'single' floors the target at 1e-7). Read-back
%     error is bounded relative to partition-TERM scales; relative
%     error of the assembled value degrades by the query's cancellation
%     ratio, exactly as truncation and roundoff do on the direct
%     strategy.
%
%   By default ('factored', 'auto') a cost gate picks the cheaper
%   strategy per call (direct for a single query at modest K, factored
%   for batches or large K). Periodic relative mode always uses the
%   direct strategy: with per-component wrapping a block whose offsets
%   straddle an image boundary does not separate into variance and mean
%   parts, so the factorisation identity does not hold on the circle.
%
%   VALS = MOBIUS.EVALORBITREL(..., 'is_per', true, 'period', P) selects
%   periodic mode.
%
%   VALS = MOBIUS.EVALORBITREL(..., 'samplesPerSigma', N) overrides the
%   u-grid density (points per sigma). By default ([]) the count is
%   derived from TRUNCATIONSIGMAS and R via
%   internal.resolveSamplesPerSigma, so the quadrature error sits at or
%   below the kernel truncation floor.
%
%   VALS = MOBIUS.EVALORBITREL(..., 'factored', F) with F one of
%   'auto' (default; cost gate), 'on' (force factored; errors in
%   periodic mode), 'off' (force direct). Intended for testing and
%   benchmarking; the gate is the supported default.
%
%   [VALS, RATIOS] = MOBIUS.EVALORBITREL(..., 'returnCancellationRatio', true)
%   additionally returns per-query worst-case cancellation ratios
%   (minimum across the u-grid for each query). Worst-case is the right
%   summary statistic since a single bad u-point corrupts the integral.
%   Ratio semantics are identical in both strategies.
%
%   Inputs:
%     P                       (N, 1) double — source positions.
%     W                       (N, 1) double — source weights.
%     SIGMA                   (1, 1) positive double.
%     R                       integer >= 1 (R=1 is degenerate; see below).
%     X_REL                   (R-1, n_q) double — relative query points;
%                             column q is interpreted as (Δ_1..Δ_{r-1})
%                             with the implicit reference coordinate at u.
%     opts.is_per             logical (default false).
%     opts.period             double (default 0; consulted only when is_per).
%     opts.samplesPerSigma    integer >= 1, or [] to derive (default []).
%     opts.returnCancellationRatio  logical (default false).
%     opts.truncationSigmas   positive scalar or Inf (default from mptDefaults).
%     opts.kernelPrecision    'double' or 'single' (default from mptDefaults).
%     opts.factored           'auto' | 'on' | 'off' (default 'auto').
%
%   For R=1 the relative space is 0-dim and T_rel is a constant; by
%   convention this returns sum(W) at each query. Tests should not
%   exercise this case.
%
%   The factored strategy is the twin of the Python implementation in
%   python/mpt/_mobius.py (eval_orbit_rel); the two share calibration
%   constants and the closed-form quintic stencil for exact
%   cross-language parity.
%
%   See also MOBIUS.EVALORBITABS, MOBIUS.GETSETPARTITIONSWITHMOBIUS.

    arguments
        p (:,1) double
        w (:,1) double
        sigma (1,1) double {mustBePositive}
        r (1,1) {mustBeInteger, mustBePositive}
        x_rel (:,:) double
        opts.is_per (1,1) logical = false
        opts.period (1,1) double = 0.0
        opts.samplesPerSigma double {mustBeNonnegative} = []
        opts.returnCancellationRatio (1,1) logical = false
        opts.truncationSigmas (1,1) double = mptDefaults('truncationSigmas')
        opts.kernelPrecision (1,:) char ...
            {mustBeMember(opts.kernelPrecision, {'double','single'})} ...
            = mptDefaults('kernelPrecision')
        opts.factored (1,:) char ...
            {mustBeMember(opts.factored, {'auto','on','off'})} = 'auto'
    end

    if r < 2
        % Degenerate: rel space is 0-dim; T_rel is constant. By
        % convention return sum(w) at each query column.
        n_q = size(x_rel, 2);
        if n_q == 0
            n_q = numel(x_rel);
        end
        vals = repmat(sum(w), n_q, 1);
        if opts.returnCancellationRatio
            ratios = ones(n_q, 1);
        else
            ratios = [];
        end
        return
    end

    if size(x_rel, 1) ~= r - 1
        error('mobius:evalOrbitRel:queryShape', ...
            'x_rel must have size (r-1, n_q) with r-1=%d; got %dx%d.', ...
            r - 1, size(x_rel, 1), size(x_rel, 2));
    end
    n_q = size(x_rel, 2);

    % Resolve the u-grid density: empty derives the accuracy-tied count
    % from truncationSigmas and r (see internal.resolveSamplesPerSigma);
    % an explicit value is honoured unchanged.
    opts.samplesPerSigma = internal.resolveSamplesPerSigma( ...
        opts.samplesPerSigma, r, opts.truncationSigmas);

    % Build the u-grid (mirrors mpt.tensor._orbit_inner_rel).
    if opts.is_per
        N_u = max(64, ceil(opts.period / sigma * opts.samplesPerSigma));
        % Endpoint-exclusive uniform grid on [0, period).
        u_grid = (0:N_u - 1) * (opts.period / N_u);
        du = opts.period / N_u;
    else
        % Non-periodic alignment window: u must let some source p_i
        % land near each query coordinate.
        if isempty(x_rel)
            x_min = 0; x_max = 0;
        else
            x_min = min(x_rel(:));
            x_max = max(x_rel(:));
        end
        u_min = min(p) - max(0, x_max) - 8 * sigma;
        u_max = max(p) - min(0, x_min) + 8 * sigma;
        N_u = max(64, ceil(max(u_max - u_min, 1) / sigma * opts.samplesPerSigma));
        u_grid = linspace(u_min, u_max, N_u);
    end

    % ---- Strategy selection --------------------------------------
    K = numel(p);
    eps_target = factoredTargetEps(opts.truncationSigmas, opts.kernelPrecision);
    spp = factoredSpp(eps_target);

    % ---- Spectral (Fourier) strategy, r = 2..4 (auto selection only) ----
    % Partition evaluation dispatched by block count: analytic masses,
    % 1-D mode series, and total-mode-zero spectral products (see
    % localEvalOrbitRelFourier). Engagement thresholds are calibrated
    % from measured wall times (Python side, K in 7..300 and n_q in
    % 5..5000): at r = 2 the spectral form wins every measured cell
    % from n_q ~ 16 up; at r = 3 from n_q ~ 32 (K >= 8); at r = 4 it
    % matches the factored strategy and beats direct from n_q ~ 64; at
    % r = 5 the factored strategy remains faster at scale, so the gate
    % stands down there. Explicit 'on'/'off' force their strategies;
    % cancellation-ratio requests fall through (the spectral form has
    % no per-node terms matching that diagnostic). Guards: the periodic
    % principal-image window protects the analytic variance prefactors,
    % and (as in the factored-periodic gate) every query's position
    % span must fit inside half the circle, under which the
    % principal-image wrap of the deltas is a no-op and the per-block
    % statistics are exact.
    fourMinQ = [16, 32, 64];
    fourMinK = [2, 8, 16];
    if r >= 2 && r <= 4 && strcmp(opts.factored, 'auto') ...
            && ~opts.returnCancellationRatio ...
            && n_q >= fourMinQ(r - 1) && K >= fourMinK(r - 1)
        kResF = internal.accuracyFloor('resolve', opts.truncationSigmas);
        spanOk = true;
        if opts.is_per && ~isempty(x_rel)
            pLoF = min(0, min(x_rel, [], 1));
            pHiF = max(0, max(x_rel, [], 1));
            spanOk = max(pHiF - pLoF) < 0.5 * opts.period;
        end
        if spanOk && (~opts.is_per ...
                      || opts.period > 2 * sqrt(2) * kResF * sigma)
            vals = localEvalOrbitRelFourier(p, w, sigma, r, x_rel, ...
                opts.is_per, opts.period, opts.truncationSigmas);
            ratios = [];
            return;
        end
    end

    if opts.is_per
        % Factored-periodic is valid only where the circular variance/mean
        % block reduction survives wrapping: the truncation window must fit
        % within half the circle and each query's position span must too,
        % so no source that matters lies on the wrapped-far side. This is
        % exactly the regime where culling helps (window < half-circle);
        % below it, fall back to the direct strategy.
        truncEff = internal.accuracyFloor('resolve', opts.truncationSigmas);
        rWin = sqrt(2.0) * truncEff * sigma;
        if r >= 2
            posLo = min(0, min(x_rel, [], 1));
            posHi = max(0, max(x_rel, [], 1));
            maxSpread = max(posHi - posLo);
        else
            maxSpread = 0.0;
        end
        factoredPerValid = (opts.period > 2.0 * rWin) ...
                        && (maxSpread < 0.5 * opts.period);
        if strcmp(opts.factored, 'on') && ~factoredPerValid
            error('mobius:evalOrbitRel:factoredPeriodic', ...
                ['''factored'', ''on'' is not available for this ' ...
                 'periodic relative case: the truncation window or ' ...
                 'query span exceeds half the period, so the circular ' ...
                 'variance/mean block factorisation wraps. Use ' ...
                 '''auto'' or ''off''.']);
        end
        switch opts.factored
            case 'auto'
                nFineTotal = 0;
                for m = 1:r
                    h_m = (sigma / sqrt(m)) / spp;
                    nFineTotal = nFineTotal + max(6, round(opts.period / h_m));
                end
                useFactored = factoredPerValid ...
                    && factoredWorthwhile(K, r, n_q, N_u, nFineTotal);
            case 'on'
                useFactored = factoredPerValid;   % true (raise above if not)
            otherwise
                useFactored = false;
        end
    else
        dmin = min(0, x_min);
        dmax = max(0, x_max);
        nFineTotal = 0;
        for m = 1:r
            h_m = (sigma / sqrt(m)) / spp;
            nFineTotal = nFineTotal + ceil((u_max + dmax - u_min - dmin) / h_m) + 12;
        end
        switch opts.factored
            case 'auto'
                useFactored = factoredWorthwhile(K, r, n_q, N_u, nFineTotal);
            case 'on'
                useFactored = true;
            otherwise
                useFactored = false;
        end
    end

    BUDGET_BYTES = internal.kernelChunkBytesResolved();

    if useFactored
        % ---- Factored strategy: tabulate S_1..S_r, read back -----
        kw = {'truncationSigmas', opts.truncationSigmas, ...
              'kernelPrecision', opts.kernelPrecision};
        tabLo = zeros(r, 1);
        tabH = zeros(r, 1);
        tabY = cell(r, 1);
        for m = 1:r
            sigmaEff = sigma / sqrt(m);
            if m == 1
                wm = w;
            else
                wm = w .^ m;
            end
            if opts.is_per
                % Uniform grid over exactly one period; the read-back wraps
                % its stencil, so no padding is needed. Tabulate the
                % circular S_m directly (gaussianKernelSum's periodic path
                % is exact over the circle).
                n_m = max(6, round(opts.period / (sigmaEff / spp)));
                h_m = opts.period / n_m;
                lo = 0.0;
                grid_m = h_m * (0:n_m - 1);
                kwPer = [kw, {'isPer', true, 'period', opts.period}];
                ym = internal.gaussianKernelSum( ...
                    p(:).', wm(:), grid_m, sigmaEff, kwPer{:});
            else
                h_m = sigmaEff / spp;
                lo = u_min + dmin - 3 * h_m;
                hi = u_max + dmax + 3 * h_m;
                n_m = ceil((hi - lo) / h_m) + 7;
                grid_m = lo + h_m * (0:n_m - 1);
                ym = internal.gaussianKernelSum( ...
                    p(:).', wm(:), grid_m, sigmaEff, kw{:});
            end
            tabLo(m) = lo;
            tabH(m) = h_m;
            tabY{m} = ym(:);
        end

        inv_2s2 = 1.0 / (2 * sigma^2);
        deltasAll = [zeros(1, n_q); x_rel];

        % Each block (a subset of the r positions) recurs across the set
        % partitions --- a singleton, for instance, reappears in every
        % partition that isolates it --- and the block read-back is the
        % dominant cost. Evaluate each distinct block's contribution once
        % per query chunk (here) and reuse it across partitions through the
        % shared mobius.mobiusPartitionCombine.
        [uniqueBlocks, partBlockIdx, mus] = mobius.getPartitionBlockStructure(r);
        nUniqueBlocks = numel(uniqueBlocks);

        % Chunk queries: dominant transient is the (N_u, nQc, 6) stencil
        % workspace, the per-block contribution cache
        % (nUniqueBlocks × (N_u, nQc)), and a few accumulators.
        perQueryBytes = 8 * N_u * (40 + nUniqueBlocks);
        chunkSize = max(1, floor(BUDGET_BYTES / max(perQueryBytes, 1)));
        chunkSize = min(chunkSize, n_q);

        F = zeros(N_u, n_q);
        if opts.returnCancellationRatio
            R = ones(N_u, n_q);
        end
        u_col = u_grid(:);
        for c0 = 1:chunkSize:n_q
            c1 = min(c0 + chunkSize - 1, n_q);
            idx = c0:c1;
            nQc = numel(idx);
            dl = deltasAll(:, idx);
            blockContrib = cell(1, nUniqueBlocks);
            for k = 1:nUniqueBlocks
                B = uniqueBlocks{k};
                m = numel(B);
                dB = dl(B, :);
                mean_d = sum(dB, 1) / m;
                var_d = sum((dB - mean_d).^2, 1);
                pts = u_col + mean_d;                           % (N_u, nQc)
                if opts.is_per
                    Sm = lagrange6Circular(tabY{m}, tabLo(m), tabH(m), pts);
                else
                    Sm = lagrange6Uniform(tabY{m}, tabLo(m), tabH(m), pts);
                end
                blockContrib{k} = exp(-var_d * inv_2s2) .* Sm;
            end
            [total, maxAbs] = mobius.mobiusPartitionCombine( ...
                blockContrib, partBlockIdx, mus, opts.returnCancellationRatio);
            F(:, idx) = total;
            if opts.returnCancellationRatio
                Rchunk = ones(N_u, nQc);
                nz = maxAbs > 0;
                Rchunk(nz) = abs(total(nz)) ./ maxAbs(nz);
                R(:, idx) = Rchunk;
            end
        end
    else
        % ---- Direct strategy: evalOrbitAbs at each u-node --------
        % The intermediate (m, N, N_u, n_q) array can be very large for
        % fine grids; chunk along the query axis to bound peak memory.
        perChunkBytesPerQuery = 8 * r * N_u * numel(p) * 4;   % m_max <= r, fudge x4
        chunkSize = max(1, floor(BUDGET_BYTES / max(perChunkBytesPerQuery, 1)));
        chunkSize = min(chunkSize, n_q);

        F = zeros(N_u, n_q);
        if opts.returnCancellationRatio
            R = ones(N_u, n_q);
        end
        for c0 = 1:chunkSize:n_q
            c1 = min(c0 + chunkSize - 1, n_q);
            idx = c0:c1;
            nQc = numel(idx);

            % Build (r, N_u, nQc) query stack: row 1 is u (shared across
            % queries), rows 2..r are u + x_rel.
            x_full = zeros(r, N_u, nQc);
            u_re = reshape(u_grid, 1, N_u, 1);
            x_full(1, :, :) = repmat(u_re, 1, 1, nQc);
            if r >= 2
                % x_rel(:, idx) is (r-1, nQc); add u_grid (broadcast over query)
                x_rel_chunk = reshape(x_rel(:, idx), r - 1, 1, nQc);
                x_full(2:end, :, :) = x_rel_chunk + u_re;
            end

            if opts.returnCancellationRatio
                [vals_chunk, ratios_chunk] = mobius.evalOrbitAbs( ...
                    p, w, sigma, r, x_full, ...
                    'is_per', opts.is_per, 'period', opts.period, ...
                    'returnCancellationRatio', true, ...
                    'truncationSigmas', opts.truncationSigmas, ...
                    'kernelPrecision', opts.kernelPrecision);
                F(:, idx) = reshape(vals_chunk, N_u, nQc);
                R(:, idx) = reshape(ratios_chunk, N_u, nQc);
            else
                vals_chunk = mobius.evalOrbitAbs( ...
                    p, w, sigma, r, x_full, ...
                    'is_per', opts.is_per, 'period', opts.period, ...
                    'truncationSigmas', opts.truncationSigmas, ...
                    'kernelPrecision', opts.kernelPrecision);
                F(:, idx) = reshape(vals_chunk, N_u, nQc);
            end
        end
    end

    % Integrate over u: rectangle rule for periodic (closes the loop),
    % trapezoidal for non-periodic.
    if opts.is_per
        integral = sum(F, 1) * du;
    else
        integral = trapz(u_grid, F, 1);
    end

    Z_t = sigma * sqrt(2 * pi / r);
    vals = (integral(:)) / Z_t;

    if opts.returnCancellationRatio
        % Worst case across u-grid for each query.
        ratios = min(R, [], 1)';
    else
        ratios = [];
    end
end

% ---- Factored-strategy helpers (twins of python/mpt/_mobius.py) -----

function eps_target = factoredTargetEps(truncationSigmas, kernelPrecision)
%FACTOREDTARGETEPS  Read-back accuracy target from the truncation floor.
%   Twin of _factored_target_eps: eps = exp(-k^2/2) clamped to
%   [floor, 1e-3]; Inf targets the dynamic accuracy floor; 'single'
%   kernels floor at 1e-7. The floor is read from
%   internal.accuracyFloor so a golden-regeneration override widens the
%   read-back accuracy here too, matching Python.
    EPS_CEIL = 1e-3;
    floorEps = internal.accuracyFloor('eps');
    k = truncationSigmas;
    if isfinite(k)
        eps_target = exp(-0.5 * k * k);
    else
        eps_target = floorEps;
    end
    eps_target = min(max(eps_target, floorEps), EPS_CEIL);
    if strcmp(kernelPrecision, 'single')
        eps_target = max(eps_target, 1e-7);
    end
end

function spp = factoredSpp(eps_target)
%FACTOREDSPP  Samples per sigma/sqrt(m) for the S_m tabulation.
%   Twin of _factored_spp: err ~= CALIB_A6 * spp^-6 (calibrated with the
%   measured Möbius cancellation amplification and a x10 safety margin).
    CALIB_A6 = 1600.0;
    SPP_MIN = 8;
    SPP_MAX = 512;
    spp = ceil((CALIB_A6 / eps_target)^(1/6));
    spp = min(max(spp, SPP_MIN), SPP_MAX);
end

function tf = factoredWorthwhile(K, r, n_q, N_u, nFineTotal)
%FACTOREDWORTHWHILE  Cost gate: factored cheaper than direct?
%   Twin of _factored_worthwhile. Direct ~ B_r*r*K*N_u*n_q kernel
%   evaluations; factored ~ K*nFineTotal tabulation plus READBACK_COST
%   kernel-equivalents per (partition-block, u-node, query). The
%   constant is measured (one stencil evaluation, 6 gathers plus
%   degree-5 weights, costs ~10 vectorised kernel operations).
    READBACK_COST = 10.0;
    partitions = mobius.getSetPartitionsWithMobius(r);
    B_r = numel(partitions);
    direct = B_r * r * K * N_u * n_q;
    fact = K * nFineTotal + READBACK_COST * B_r * r * N_u * n_q;
    tf = fact < direct;
end

function vals = lagrange6Uniform(y, x0, h, pts)
%LAGRANGE6UNIFORM  Quintic (6-point Lagrange) read-back on a uniform grid.
%   Twin of _lagrange6_uniform: identical closed-form stencil weights
%   for exact cross-language parity. Stencils are clamped at the grid
%   ends; callers pad the grid so clamping only occurs where y has
%   decayed below the truncation floor.
    n = numel(y);
    if n < 6
        error('mobius:evalOrbitRel:readbackGrid', ...
            'read-back grid must have >= 6 nodes; got %d.', n);
    end
    s = (pts - x0) / h;
    base = floor(s) - 2;                      % zero-based stencil start
    base = min(max(base, 0), n - 6);
    t = s - base;                             % stencil coordinate
    d = t - reshape(0:5, 1, 1, 6);            % (..., 6)
    % prod_{k~=j}(t - k) via prefix/suffix products (no division by d,
    % which may be exactly zero at grid nodes).
    pref = ones(size(d));
    pref(:, :, 2:end) = cumprod(d(:, :, 1:end - 1), 3);
    suff = ones(size(d));
    suff(:, :, 1:end - 1) = flip(cumprod(flip(d(:, :, 2:end), 3), 3), 3);
    denom = reshape([-120, 24, -12, 12, -24, 120], 1, 1, 6);
    wgt = pref .* suff ./ denom;
    idx = base + reshape(0:5, 1, 1, 6) + 1;   % 1-based gather indices
    vals = sum(wgt .* y(idx), 3);
end


function vals = lagrange6Circular(y, x0, h, pts)
%LAGRANGE6CIRCULAR  Quintic (6-point Lagrange) read-back on a periodic grid.
%   Circular twin of lagrange6Uniform. The samples Y cover exactly one
%   period on the grid x0 + i*h, i = 0..n-1 (so n*h is the period); the
%   interpolant is periodic and the six-node stencil wraps around the ends
%   modulo n rather than clamping. Identical closed-form stencil weights,
%   for exact cross-language parity with _lagrange6_circular. Query points
%   may lie anywhere on the line; they map onto the circle through the
%   wrap.
    n = numel(y);
    if n < 6
        error('mobius:evalOrbitRel:readbackGrid', ...
            'read-back grid must have >= 6 nodes; got %d.', n);
    end
    s = (pts - x0) / h;
    i0 = floor(s);
    base = i0 - 2;                            % zero-based stencil start
    t = s - i0 + 2.0;                         % stencil coordinate in [2, 3)
    d = t - reshape(0:5, 1, 1, 6);            % (..., 6)
    % prod_{k~=j}(t - k) via prefix/suffix products (no division by d).
    pref = ones(size(d));
    pref(:, :, 2:end) = cumprod(d(:, :, 1:end - 1), 3);
    suff = ones(size(d));
    suff(:, :, 1:end - 1) = flip(cumprod(flip(d(:, :, 2:end), 3), 3), 3);
    denom = reshape([-120, 24, -12, 12, -24, 120], 1, 1, 6);
    wgt = pref .* suff ./ denom;
    idx = mod(base + reshape(0:5, 1, 1, 6), n) + 1;   % wrap mod n, 1-based
    vals = sum(wgt .* y(idx), 3);
end


function vals = localEvalOrbitRelFourier(p, w, sigma, r, x_rel, ...
        isPer, period, truncationSigmas)
%LOCALEVALORBITRELFOURIER  Relative evaluation, spectral form, r >= 2.
%   Every partition's translation integral constrains the total mode
%   index of its block product to zero. The evaluation dispatches by
%   block count: single-block partitions are the analytic mass
%   sum(w.^r)*sigma*sqrt(2*pi/r); two-block partitions are 1-D mode
%   series L*sum_m conj(cA(m))*cB(m)*exp(1i*om*m*(meanB-meanA)),
%   evaluated with a multiplicative phase ladder; partitions with three
%   or more blocks are total-mode-zero coefficients of per-block
%   spectral products, evaluated as cached per-block FFTs (one per
%   distinct block appearing in any such partition, shared across
%   partitions exactly as blocks recur) and one pointwise product per
%   partition. The block spectra are the closed-form Fourier
%   coefficients of the wrapped Gaussian mixture --- no sampling, no
%   kernel truncation, no interpolation --- truncated per block size
%   where the Gaussian envelope reaches machine precision (the mode
%   width 8.6 below): the Mobius cancellation amplifies per-term error,
%   so per-term error must sit near machine eps for the combined value
%   to reach the requested floor. The non-periodic case embeds in a
%   virtual period padded to the requested truncation floor. Queries
%   are chunked so the cached spectra respect the kernel chunk budget.
%   kernelPrecision does not apply: the coefficients are closed-form in
%   double precision. Twin of the Python _eval_orbit_rel_fourier.
    kRes = internal.accuracyFloor('resolve', truncationSigmas);
    % Mode-cutoff width in sigmas: exp(-8.6^2/2) ~ 8e-17 (machine
    % precision); mode count scales as sqrt(log(1/eps)), so this costs
    % only ~16% more modes than the 1e-12 floor would.
    kM = 8.6;
    x_rel = double(x_rel);
    n_q = size(x_rel, 2);
    p = double(p(:));
    w = double(w(:));
    if isPer
        L = period;
    else
        pad = (kRes + 2) * sigma;
        lo = min(p) - pad + min(0, min([x_rel(:); 0]));
        hi = max(p) + pad + max(0, max([x_rel(:); 0]));
        L = hi - lo;
    end
    om = 2 * pi / L;
    MBy = zeros(1, r);
    for mSz = 1:r
        MBy(mSz) = ceil(kM * L * sqrt(mSz) / (2 * pi * sigma)) + 2;
    end
    c = cell(1, r);
    for mSz = 1:r
        Mm = MBy(mSz);
        mm = (-Mm:Mm).';
        sigM = sigma / sqrt(mSz);
        env = sigM * sqrt(2 * pi) / L ...
            * exp(-2 * pi^2 * sigM^2 * mm.^2 / L^2);
        c{mSz} = env .* (exp(-1i * om * mm * p.') * w.^mSz);
    end
    partitions = mobius.getSetPartitionsWithMobius(r);
    maxSpan = 0;
    nFftBlocks = 0;
    seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    for ip = 1:numel(partitions)
        blocks = partitions(ip).blocks;
        if numel(blocks) >= 3
            spanSum = 0;
            for ib = 1:numel(blocks)
                spanSum = spanSum + MBy(numel(blocks{ib}));
                key = localBlockKey(blocks{ib});
                if ~isKey(seen, key)
                    seen(key) = true;
                    nFftBlocks = nFftBlocks + 1;
                end
            end
            maxSpan = max(maxSpan, spanSum);
        end
    end
    Nf = max(2 * maxSpan + 2, 2);
    masses = zeros(1, r);
    for mSz = 1:r
        masses(mSz) = sum(w.^mSz) * sigma * sqrt(2 * pi / mSz);
    end
    perQueryBytes = (nFftBlocks + 3) * Nf * 16;
    budget = max(internal.kernelChunkBytesResolved(), 1);
    chunk = max(1, min(n_q, floor(budget / max(perQueryBytes, 1))));
    inv2s2 = 1 / (2 * sigma^2);
    vals = zeros(n_q, 1);
    for q0 = 1:chunk:n_q
        q1 = min(q0 + chunk - 1, n_q);
        nqc = q1 - q0 + 1;
        deltas = [zeros(1, nqc); x_rel(:, q0:q1)];
        if isPer
            deltas = deltas - period .* round(deltas ./ period);
        end
        stats = containers.Map('KeyType', 'char', 'ValueType', 'any');
        ghat = containers.Map('KeyType', 'char', 'ValueType', 'any');
        total = complex(zeros(1, nqc));
        for ip = 1:numel(partitions)
            blocks = partitions(ip).blocks;
            mu = partitions(ip).mu;
            qn = numel(blocks);
            pref = ones(1, nqc);
            for ib = 1:qn
                key = localBlockKey(blocks{ib});
                if ~isKey(stats, key)
                    dB = deltas(blocks{ib}, :);
                    meanB = mean(dB, 1);
                    varB = sum((dB - meanB).^2, 1);
                    stats(key) = {meanB, exp(-varB * inv2s2)};
                end
                sv = stats(key);
                pref = pref .* sv{2};
            end
            if qn == 1
                total = total + mu * pref * masses(r);
                continue;
            end
            if qn == 2
                BA = blocks{1};
                BB = blocks{2};
                mA = numel(BA);
                mB = numel(BB);
                Mp = max(MBy(mA), MBy(mB));
                ca = complex(zeros(2 * Mp + 1, 1));
                cb = complex(zeros(2 * Mp + 1, 1));
                ca(Mp - MBy(mA) + 1:Mp + MBy(mA) + 1) = c{mA};
                cb(Mp - MBy(mB) + 1:Mp + MBy(mB) + 1) = c{mB};
                G = L * conj(ca) .* cb;
                svA = stats(localBlockKey(BA));
                svB = stats(localBlockKey(BB));
                b = svB{1} - svA{1};
                E = localPhaseLadder(Mp, b, om);
                total = total + mu * (pref .* (G.' * E));
                continue;
            end
            prodG = [];
            for ib = 1:qn
                B = blocks{ib};
                key = localBlockKey(B);
                if ~isKey(ghat, key)
                    Mm = MBy(numel(B));
                    mm = (-Mm:Mm).';
                    sv = stats(key);
                    g = complex(zeros(Nf, nqc));
                    g(mod(mm, Nf) + 1, :) = c{numel(B)} ...
                        .* localPhaseLadder(Mm, sv{1}, om);
                    ghat(key) = fft(g, [], 1);
                end
                gh = ghat(key);
                if isempty(prodG)
                    prodG = gh;
                else
                    prodG = prodG .* gh;
                end
            end
            total = total + mu * (pref .* ((L / Nf) * sum(prodG, 1)));
        end
        vals(q0:q1) = real(total).';
    end
    Z_t = sigma * sqrt(2 * pi / r);
    vals = vals / Z_t;
end


function E = localPhaseLadder(Mm, b, om)
%LOCALPHASELADDER  E(m, q) = exp(1i*om*m*b(q)) for m = -Mm..Mm via one
%   exponential per query and cumulative products.
    z = exp(1i * om * b(:).');
    pos = cumprod(repmat(z, Mm, 1), 1);
    E = [conj(pos(end:-1:1, :)); ones(1, numel(b)); pos];
end


function key = localBlockKey(B)
%LOCALBLOCKKEY  Char key for a block (its member indices).
    key = sprintf('%d,', B);
end
