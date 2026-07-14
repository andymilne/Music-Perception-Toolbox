function [I, ratio] = relInnerBatched(Px, Wx, Py, Wy, sigma, r, ...
                                       isPer, period, opts)
%MOBIUS.RELINNERBATCHED  Batched relative-mode inner products, all pairs.
%
%   I = MOBIUS.RELINNERBATCHED(PX, WX, PY, WY, SIGMA, R, ISPER, PERIOD)
%   returns the (N_x, N_y) matrix of relative-mode inner products
%   between the two densities' events: every pair's inner product
%   marginalises a translation u over a grid. This is the single
%   relative-mode evaluator: the single-collection form (one event per
%   side) is its N = 1 specialisation via MOBIUS.ORBITINNERRELSA.
%
%   Periodic mode uses the shared uniform grid over [0, P) with
%   INTERNAL.AUTONTAUDEFAULT nodes — the single node-count source
%   shared with the flat and nested relative-periodic paths.
%   Non-periodic mode gives each pair a grid of the same shape centred
%   on its own weighted-mean offset: by translation invariance the
%   pair integrand depends on u only through u + (mean_Y - mean_X), so
%   shifting each pair's window by that offset lets all pairs share
%   one grid whose extent is the maximum within-pair spread plus the
%   INTERNAL.RELWINDOWMARGIN margin per side, which places every
%   kernel entry strictly outside the truncation radius at the window
%   edges: the endpoint integrand is exactly zero for any finite
%   truncation, so the plain Riemann sum equals the trapezoidal rule
%   exactly (with truncation disabled the fixed 8-sigma margin leaves
%   an endpoint tail far below FP noise for r >= 2).
%
%   [I, RATIO] = ... additionally returns the minimum across event
%   pairs of the per-pair mass-aware cancellation diagnostic
%   |sum_u F_u| / sum_u max_orb(|term_orb_u|) — each pair's integrated
%   alternating sum relative to the integral of its worst-magnitude
%   partition term, which bounds the relative error of that pair's
%   integral. A pointwise worst-case over (pair, u) cells is the wrong
%   aggregate: translation bands with a true integrand of exactly zero
%   arise from exact cancellation of nonzero orbit terms. Pairs with
%   zero accumulated term mass report 1.0.
%
%   The pair-and-grid batch is processed in slabs of at most
%   SLAB_ELEMS kernel entries, built directly in the batch-first
%   (batch, Kx, Ky) layout of the orbit primitives — no permute copies
%   — so the working set stays memory-resident and the per-op
%   contraction cost is flat in N and K. When every event carries the
%   same weight vector the cheaper shared-weights
%   MOBIUS.INNERPRODUCTORBITGRID contraction applies. Zero-padded
%   (NaN) slots carry zero weight, contributing a zero factor to every
%   Möbius term touching them, so the result is exact for the K_eff
%   events.
%
%   Name-value options:
%     'truncationSigmas'  kernel truncation (default mptDefaults)
%     'samplesPerSigma'   non-periodic grid nodes per sigma (default 10)

    arguments
        Px double
        Wx double
        Py double
        Wy double
        sigma (1,1) double {mustBePositive}
        r (1,1) double {mustBeInteger, mustBePositive}
        isPer (1,1) logical
        period (1,1) double
        opts.truncationSigmas (1,1) double = mptDefaults('truncationSigmas')
        opts.samplesPerSigma (1,1) double {mustBePositive} = 10
    end

    truncationSigmas = opts.truncationSigmas;
    wantRatio = nargout > 1;
    ratio = 1.0;

    SLAB_ELEMS = 2^19;   % mirror of Python _ORBIT_GRID_SLAB_ELEMS

    [K, Nx] = size(Px);
    [~, Ny] = size(Py);

    nanX = isnan(Px) | isnan(Wx);
    if any(nanX(:))
        Px(nanX) = 0;
        Wx(nanX) = 0;
    end
    nanY = isnan(Py) | isnan(Wy);
    if any(nanY(:))
        Py(nanY) = 0;
        Wy(nanY) = 0;
    end

    sharedW = all(all(Wx == Wx(:, 1))) && all(all(Wy == Wy(:, 1)));

    if isPer
        N_u = internal.autoNtauDefault(period, sigma);
        uGrid = (0:N_u-1) * (period / N_u);
        du = period / N_u;
        centres = zeros(Nx, Ny);
    else
        wsx = sum(Wx, 1);
        wsy = sum(Wy, 1);
        mx = sum(Px .* Wx, 1) ./ max(wsx, realmin);
        my = sum(Py .* Wy, 1) ./ max(wsy, realmin);
        centres = my - mx.';                       % (Nx, Ny)

        spreadX = localWeightedSpread(Px, Wx);
        spreadY = localWeightedSpread(Py, Wy);
        margin = internal.relWindowMargin(truncationSigmas);
        span = max(spreadX) + max(spreadY) + 2 * margin * sigma;
        N_u = max(64, ceil(max(span, 1.0) / sigma * opts.samplesPerSigma));
        uGrid = linspace(-0.5 * span, 0.5 * span, N_u);
        du = span / (N_u - 1);
    end

    PxT = Px.';                                    % (Nx, K)
    PyT = Py.';                                    % (Ny, K)

    I = zeros(Nx, Ny);
    perPair = K * K;
    ncx = max(1, min(Nx, floor(SLAB_ELEMS / max(Ny * perPair, 1))));

    for nStart = 1:ncx:Nx
        nEnd = min(nStart + ncx - 1, Nx);
        idxX = nStart:nEnd;
        nc = numel(idxX);
        nPairs = nc * Ny;
        n_uc = max(1, floor(SLAB_ELEMS / max(nPairs * perPair, 1)));

        % Base differences with per-pair centres, built in
        % (u, ix, iy, kx, ky) dims and flattened to (1, pairs, K, K)
        % so the u broadcast below is 4-D with no singleton axes
        % (mirror of the Python staging; the (ix, iy) merge is a free
        % column-major reshape and downstream page ordering is
        % unchanged).
        baseD = reshape( ...
            reshape(PxT(idxX, :), [1, nc, 1, K, 1]) ...
          - reshape(PyT, [1, 1, Ny, 1, K]) ...
          + reshape(centres(idxX, :), [1, nc, Ny, 1, 1]), ...
            [1, nPairs, K, K]);

        if ~sharedW
            wA = reshape(repmat(reshape(Wx(:, idxX).', [nc, 1, K]), ...
                                 1, Ny, 1), [nPairs, K]);
            wB = reshape(repmat(reshape(Wy.', [1, Ny, K]), ...
                                 nc, 1, 1), [nPairs, K]);
        end

        F_sum = zeros(nPairs, 1);
        massSum = zeros(nPairs, 1);
        for uStart = 1:n_uc:N_u
            uEnd = min(uStart + n_uc - 1, N_u);
            u_s = uGrid(uStart:uEnd);
            nu = numel(u_s);

            diffs = baseD + reshape(u_s, [nu, 1, 1, 1]);
            if isPer
                diffs = diffs - period * floor(diffs / period + 0.5);
            end
            K_uc = internal.truncKernelExp(diffs.^2, sigma, ...
                                           truncationSigmas);
            K_uc = reshape(K_uc, [nu * nPairs, K, K]);

            if sharedW
                if wantRatio
                    [flat, ~, mass] = mobius.innerProductOrbitGrid( ...
                        K_uc, Wx(:, 1), Wy(:, 1), r, ...
                        'returnCancellationRatio', true);
                else
                    flat = mobius.innerProductOrbitGrid( ...
                        K_uc, Wx(:, 1), Wy(:, 1), r);
                    mass = [];
                end
            else
                wA_uc = reshape(repmat(reshape(wA, [1, nPairs, K]), ...
                                        nu, 1, 1), [nu * nPairs, K]);
                wB_uc = reshape(repmat(reshape(wB, [1, nPairs, K]), ...
                                        nu, 1, 1), [nu * nPairs, K]);
                if wantRatio
                    [flat, ~, mass] = mobius.innerProductOrbitPwBatched( ...
                        K_uc, wA_uc, wB_uc, r, ...
                        'returnCancellationRatio', true);
                else
                    flat = mobius.innerProductOrbitPwBatched( ...
                        K_uc, wA_uc, wB_uc, r);
                    mass = [];
                end
            end
            if ~isempty(mass)
                massSum = massSum + reshape( ...
                    sum(reshape(mass, [nu, nPairs]), 1), [nPairs, 1]);
            end
            F_sum = F_sum + reshape( ...
                sum(reshape(flat, [nu, nPairs]), 1), [nPairs, 1]);
        end

        if wantRatio
            pairRatios = ones(nPairs, 1);
            nzMass = massSum > 0;
            pairRatios(nzMass) = abs(F_sum(nzMass)) ./ massSum(nzMass);
            ratio = min(ratio, min(pairRatios));
        end
        integralChunk = reshape(F_sum, [nc, Ny]) * du;
        I(idxX, :) = integralChunk;
    end

    c = sigma * sqrt(2 * pi / r);
    I = (sigma * sqrt(pi))^r * I / c^2;
end


function s = localWeightedSpread(P, W)
%LOCALWEIGHTEDSPREAD  Per-event max-minus-min over positive-weight slots.
    masked = P;
    masked(W <= 0) = NaN;
    s = max(masked, [], 1, 'omitnan') - min(masked, [], 1, 'omitnan');
    s(~isfinite(s)) = 0;
end
