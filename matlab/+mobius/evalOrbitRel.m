function [vals, ratios] = evalOrbitRel(p, w, sigma, r, x_rel, opts)
%MOBIUS.EVALORBITREL  Möbius point evaluator for SA relative-mode tensor.
%
%   VALS = MOBIUS.EVALORBITREL(P, W, SIGMA, R, X_REL) computes T_rel
%   at each column of X_REL via u-grid quadrature of the translation
%   marginal:
%
%     T_rel(Δ) = (1/Z_t) * ∫ T_abs(u, u+Δ_1, ..., u+Δ_{r-1}) du
%
%   where Z_t = sigma * sqrt(2*pi/r) is the translation-mode normaliser.
%   The quadrature is intrinsic to the Möbius realisation of relative
%   mode: the alternating partition sum only factorises across slots at
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
%   default u-grid density of 10 points per sigma. Reduce to 5 for
%   speed at the cost of ~1e-9 relative precision.
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
%                             with the implicit reference slot at u.
%     opts.is_per             logical (default false).
%     opts.period             double (default 0; consulted only when is_per).
%     opts.samplesPerSigma    integer >= 1 (default 10).
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
        opts.samplesPerSigma (1,1) {mustBeInteger, mustBePositive} = 10
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
    if strcmp(opts.factored, 'on') && opts.is_per
        error('mobius:evalOrbitRel:factoredPeriodic', ...
            ['''factored'', ''on'' is not available in periodic ', ...
             'relative mode: per-component wrapping breaks the ', ...
             'variance/mean block factorisation. Use ''auto'' or ''off''.']);
    end

    n_q = size(x_rel, 2);

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
    if opts.is_per
        useFactored = false;
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
            h_m = sigmaEff / spp;
            lo = u_min + dmin - 3 * h_m;
            hi = u_max + dmax + 3 * h_m;
            n_m = ceil((hi - lo) / h_m) + 7;
            grid_m = lo + h_m * (0:n_m - 1);
            if m == 1
                wm = w;
            else
                wm = w .^ m;
            end
            ym = internal.gaussianKernelSum( ...
                p(:).', wm(:), grid_m, sigmaEff, kw{:});
            tabLo(m) = lo;
            tabH(m) = h_m;
            tabY{m} = ym(:);
        end

        partitions = mobius.getSetPartitionsWithMobius(r);
        inv_2s2 = 1.0 / (2 * sigma^2);
        deltasAll = [zeros(1, n_q); x_rel];

        % Chunk queries: dominant transient is the (N_u, nQc, 6)
        % stencil workspace plus a few (N_u, nQc) accumulators.
        perQueryBytes = 8 * N_u * 40;
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
            total = zeros(N_u, nQc);
            if opts.returnCancellationRatio
                maxAbs = zeros(N_u, nQc);
            end
            for iPart = 1:numel(partitions)
                blocks = partitions(iPart).blocks;
                mu = partitions(iPart).mu;
                blockProd = ones(N_u, nQc);
                for b = 1:numel(blocks)
                    B = blocks{b};
                    m = numel(B);
                    dB = dl(B, :);
                    mean_d = sum(dB, 1) / m;
                    var_d = sum((dB - mean_d).^2, 1);
                    pts = u_col + mean_d;                       % (N_u, nQc)
                    Sm = lagrange6Uniform(tabY{m}, tabLo(m), tabH(m), pts);
                    blockProd = blockProd .* (exp(-var_d * inv_2s2) .* Sm);
                end
                term = mu * blockProd;
                total = total + term;
                if opts.returnCancellationRatio
                    maxAbs = max(maxAbs, abs(term));
                end
            end
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
