function [I, ratio] = relInnerBatched(Px, Wx, Py, Wy, sigma, r, ...
                                       isPer, period, opts)
%MOBIUS.RELINNERBATCHED  Batched relative-mode inner products, all pairs.
%
%   I = MOBIUS.RELINNERBATCHED(PX, WX, PY, WY, SIGMA, R, ISPER, PERIOD)
%   returns the (N_x, N_y) matrix of relative-mode inner products
%   between the two densities' events: every pair's inner product
%   marginalises a translation u over a grid. This is the single
%   relative-mode evaluator: the single-multiset form (one event per
%   side) is its N = 1 specialisation via MOBIUS.ORBITINNERRELSINGLEMULTISET.
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
%   (NaN) values carry zero weight, contributing a zero factor to every
%   Möbius term touching them, so the result is exact for the K_eff
%   events.
%
%   Name-value options:
%     'truncationSigmas'  kernel truncation (default mptDefaults)
%     'samplesPerSigma'   non-periodic grid nodes per sigma; [] (the
%                         default) derives the accuracy-tied count from
%                         truncationSigmas and r via
%                         internal.resolveSamplesPerSigma

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
        opts.samplesPerSigma double {mustBeNonnegative} = []
    end

    truncationSigmas = opts.truncationSigmas;
    opts.samplesPerSigma = internal.resolveSamplesPerSigma( ...
        opts.samplesPerSigma, r, truncationSigmas);
    wantRatio = nargout > 1;
    ratio = 1.0;

    SLAB_ELEMS = 2^19;   % mirror of Python _ORBIT_GRID_SLAB_ELEMS

    [Kx, Nx] = size(Px);
    [Ky, Ny] = size(Py);

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

    % ---- Spectral (Fourier) branch, r = 2..4 ------------------------
    % Replaces the translation grid with a mode sum: each event's
    % spectrum is built once and the matrix over event pairs is their
    % Gram matrix, so the per-pair cost carries no K and no grid nodes.
    % Cancellation-ratio requests fall through (the spectral form has no
    % per-node terms matching that diagnostic), as do configurations
    % whose mode grid would be too large (the helper returns []).
    if internal.spectralIpEnabled() && r >= 2 && r <= 4 && ~wantRatio
        Ispec = mobius.spectralRelInnerMatrix(Px, Wx, Py, Wy, sigma, r, ...
                                              isPer, period);
        if ~isempty(Ispec)
            I = Ispec;
            return;
        end
    end

    if isPer
        N_u = internal.autoNtauDefault(period, sigma);
        uGrid = (0:N_u-1) * (period / N_u);
        du = period / N_u;
        centres = zeros(Nx, Ny);
    else
        % Each pair's window is centred on the midpoint of its own
        % difference range, not on its weighted-mean offset. The cross
        % integrand's support runs from (min_y - max_x) to
        % (max_y - min_x), whose midpoint is the midrange offset; the
        % weighted mean can sit far from it when the weights are
        % lopsided -- by over 100 cents on ordinary random data, against
        % a margin of 8 sigma -- and a window of the right width centred
        % there clips one end of the support. The clipped mass is the
        % extreme pairs' contribution, so the resulting error is
        % data-dependent and survives grid refinement; it reached 4.8e-4
        % against direct enumeration, four orders above the truncation
        % floor.
        [loX, hiX] = localExtremes(Px, Wx);
        [loY, hiY] = localExtremes(Py, Wy);
        midX = 0.5 * (loX + hiX);
        midY = 0.5 * (loY + hiY);
        midX(~isfinite(midX)) = 0;
        midY(~isfinite(midY)) = 0;
        centres = midY - midX.';                   % (Nx, Ny)

        spreadX = localWeightedSpread(Px, Wx);
        spreadY = localWeightedSpread(Py, Wy);
        margin = internal.relWindowMargin(truncationSigmas);
        span = max(spreadX) + max(spreadY) + 2 * margin * sigma;
        N_u = max(64, ceil(max(span, 1.0) / sigma * opts.samplesPerSigma));
        uGrid = linspace(-0.5 * span, 0.5 * span, N_u);
        du = span / (N_u - 1);
    end

    % Sparse-orbit fast path (periodic): when the value kernel is large
    % and the truncation window fits inside the circle, each u-node's
    % kernel is a circular band of width 2R out of the period, so a
    % spatially-culled per-node orbit beats the dense slab contraction;
    % the win repeats across every u-node while the sort is paid once
    % per pair. Same size/density thresholds as the absolute gate, with
    % a cheap density probe at u = 0 from the first pair (the band
    % fraction is u-independent, so one node is representative). The
    % strict margin keeps the builder's padded candidate window from
    % ever admitting both period-copies of a centre (both would pass
    % the exact wrapped-distance filter and be double-counted).
    if isPer && r >= 2
        [minKernel, maxDensity] = localOrbitSparseThresholds();
        kRes = internal.accuracyFloor('resolve', truncationSigmas);
        cutoff = 2 * (kRes * sigma)^2;
        if Kx * Ky >= minKernel && 2 * sqrt(cutoff) < period * (1 - 1e-8)
            vx0 = Wx(:, 1) ~= 0;
            vy0 = Wy(:, 1) ~= 0;
            [c3p, j3p] = localRelPerSparsePrep(Py(vy0, 1), period);
            K0 = localBuildSparseKernelRelPer(Px(vx0, 1), c3p, j3p, ...
                Py(vy0, 1), sigma, cutoff, period, 0.0);
            if nnz(K0) <= maxDensity * Kx * Ky
                [I, worstR] = localRelPerInnerSparse(Px, Wx, Py, Wy, ...
                    sigma, r, period, uGrid, du, cutoff, wantRatio);
                if wantRatio
                    ratio = min(ratio, worstR);
                end
                c = sigma * sqrt(2 * pi / r);
                I = (sigma * sqrt(pi))^r * I / c^2;
                return;
            end
        end
    end

    PxT = Px.';                                    % (Nx, Kx)
    PyT = Py.';                                    % (Ny, Ky)

    I = zeros(Nx, Ny);
    perPair = Kx * Ky;
    ncx = max(1, min(Nx, floor(SLAB_ELEMS / max(Ny * perPair, 1))));

    for nStart = 1:ncx:Nx
        nEnd = min(nStart + ncx - 1, Nx);
        idxX = nStart:nEnd;
        nc = numel(idxX);
        nPairs = nc * Ny;
        n_uc = max(1, floor(SLAB_ELEMS / max(nPairs * perPair, 1)));

        % Base differences with per-pair centres, built in
        % (u, ix, iy, kx, ky) dims and flattened to (1, pairs, Kx, Ky)
        % so the u broadcast below is 4-D with no singleton axes
        % (mirror of the Python staging; the (ix, iy) merge is a free
        % column-major reshape and downstream page ordering is
        % unchanged). The two collections may differ in size, so the
        % A-side carries Kx pitches on axis 4 and the B-side Ky on axis 5.
        baseD = reshape( ...
            reshape(PxT(idxX, :), [1, nc, 1, Kx, 1]) ...
          - reshape(PyT, [1, 1, Ny, 1, Ky]) ...
          + reshape(centres(idxX, :), [1, nc, Ny, 1, 1]), ...
            [1, nPairs, Kx, Ky]);

        if ~sharedW
            wA = reshape(repmat(reshape(Wx(:, idxX).', [nc, 1, Kx]), ...
                                 1, Ny, 1), [nPairs, Kx]);
            wB = reshape(repmat(reshape(Wy.', [1, Ny, Ky]), ...
                                 nc, 1, 1), [nPairs, Ky]);
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
                nImg = internal.relPerImageCount(sigma, period, ...
                                                 truncationSigmas);
                % Full-image kernel. The transposition average of the
                % wrapped (theta) kernel equals the lattice-sum form
                % exactly, so summing images here upgrades the whole
                % contraction to the full-image measure with the orbit
                % reduction, the grid, and the slabbing untouched.
                % Accumulated in a loop rather than on a trailing
                % dimension so peak memory stays flat in the image count.
                K_uc = internal.truncKernelExp(diffs.^2, sigma, ...
                                               truncationSigmas);
                for lImg = 1:nImg
                    shiftL = lImg * period;
                    K_uc = K_uc + internal.truncKernelExp( ...
                        (diffs + shiftL).^2, sigma, truncationSigmas);
                    K_uc = K_uc + internal.truncKernelExp( ...
                        (diffs - shiftL).^2, sigma, truncationSigmas);
                end
            else
                K_uc = internal.truncKernelExp(diffs.^2, sigma, ...
                                               truncationSigmas);
            end
            K_uc = reshape(K_uc, [nu * nPairs, Kx, Ky]);

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
                wA_uc = reshape(repmat(reshape(wA, [1, nPairs, Kx]), ...
                                        nu, 1, 1), [nu * nPairs, Kx]);
                wB_uc = reshape(repmat(reshape(wB, [1, nPairs, Ky]), ...
                                        nu, 1, 1), [nu * nPairs, Ky]);
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


function [lo, hi] = localExtremes(P, W)
%LOCALEXTREMES  Per-event min and max over positive-weight values.
    masked = P;
    masked(W <= 0) = NaN;
    lo = min(masked, [], 1, 'omitnan');
    hi = max(masked, [], 1, 'omitnan');
end


function s = localWeightedSpread(P, W)
%LOCALWEIGHTEDSPREAD  Per-event max-minus-min over positive-weight values.
    [lo, hi] = localExtremes(P, W);
    s = hi - lo;
    s(~isfinite(s)) = 0;
end


function [minKernel, maxDensity] = localOrbitSparseThresholds()
%LOCALORBITSPARSETHRESHOLDS  Sparse-orbit cost model (mirror of the Python
%   _ORBIT_SPARSE_MIN_KERNEL / _ORBIT_SPARSE_MAX_DENSITY, shared with the
%   absolute gate in mobius.maPerAttrInnerMatrix). The sparse per-pair
%   orbit undercuts the dense contraction only when the value kernel is
%   both large and sparse; below these thresholds the dense
%   contraction's constant factors win. Tunable.
    minKernel = 200000;   % Kx * Ky floor
    maxDensity = 0.20;    % nnz / (Kx * Ky) ceiling
end


function [c3, j3] = localRelPerSparsePrep(pY, period)
%LOCALRELPERSPARSEPREP  One-time sorted-tripled centre arrays.
%   Folds the B-side values into [0, P), sorts them, and replicates
%   each at c-P, c, c+P so a wrapped window maps to a contiguous range
%   of the sorted array. j3 carries the original column index of each
%   copy. Shared across all u-nodes of a pair. Twin of the Python
%   _rel_per_sparse_prep.
    pYm = mod(pY(:), period);
    [pYs, order] = sort(pYm);
    c3 = [pYs - period; pYs; pYs + period];
    j3 = [order; order; order];
end


function K = localBuildSparseKernelRelPer(pX, c3, j3, pY, sigma, ...
        cutoff, period, u)
%LOCALBUILDSPARSEKERNELRELPER  Circular sparse kernel at shift u.
%   Entries exp(-wrap(pX(i)+u-pY(j))^2/(4*sigma^2)) for wrapped squared
%   distance at most CUTOFF; the caller guards 2*sqrt(cutoff) strictly
%   inside the period, so each row's padded window covers at most one
%   copy of any centre. Candidates are located via the sorted-tripled
%   window (with a hair of padding), then retained and evaluated with
%   internal.truncKernelExp's own arithmetic --- the raw difference, its
%   floor-wrap, the expArg <= cutoff retention, and
%   exp(-expArg/(4*sigma^2)) --- so the sparse kernel densifies to the
%   truncated dense kernel bit-for-bit. Twin of the Python
%   _build_sparse_kernel_rel_per; the reshape guards on the gathered
%   index matrices protect the single-row and single-column corners
%   (see the header note in internal.gaussianKernelSum's
%   localTruncatedKernelSum1D).
    pX = pX(:);
    pY = pY(:);
    nX = numel(pX);
    nY = numel(pY);
    R = sqrt(cutoff);
    Rpad = R * (1 + 1e-9) + 1e-9 * period;
    x = mod(pX + u, period);                       % (nX, 1)
    lo0 = sum(c3 < (x.' - Rpad), 1);               % (1, nX), 0-indexed
    hi0 = sum(c3 <= (x.' + Rpad), 1);              % (1, nX), one-past-last
    win = hi0 - lo0;
    maxWin = max(win);
    if isempty(maxWin) || maxWin == 0
        K = sparse(nX, nY);
        return;
    end
    idx = lo0(:) + (0:maxWin - 1) + 1;             % (nX, maxWin), 1-based
    mask = idx <= hi0(:);
    idxC = min(idx, numel(c3));
    cols = reshape(j3(idxC), nX, maxWin);
    rows = repmat((1:nX).', 1, maxWin);
    % Dense-path arithmetic on the candidates: raw difference,
    % floor-wrap, inclusive cutoff, exp.
    d = reshape(pX(rows), nX, maxWin) + u - reshape(pY(cols), nX, maxWin);
    d = d - period * floor(d / period + 0.5);
    expArg = d.^2;
    keep = mask & (expArg <= cutoff);
    if ~any(keep(:))
        K = sparse(nX, nY);
        return;
    end
    vals = exp(-expArg(keep) / (4 * sigma^2));
    K = sparse(rows(keep), cols(keep), vals, nX, nY);
end


function [I, worst] = localRelPerInnerSparse(Px, Wx, Py, Wy, sigma, r, ...
        period, uGrid, du, cutoff, wantRatio)
%LOCALRELPERINNERSPARSE  Periodic relative inner products, sparse route.
%   Per pair: the B-side sorted-tripled arrays are prepared once, then
%   each u-node builds its circular sparse kernel and runs the sparse
%   orbit collapse. Values match the dense slab route (the circular
%   window retains precisely the entries the truncated dense kernel
%   keeps, and zero-weight values contribute zero to every orbit term),
%   and the mass-aware pair ratio |sum_u F_u| / sum_u max|term_u|
%   matches the dense diagnostic. The caller applies the shared
%   normalisation tail. Twin of the Python _rel_per_inner_sparse.
    Nx = size(Px, 2);
    Ny = size(Py, 2);
    I = zeros(Nx, Ny);
    worst = 1.0;
    yPre = cell(Ny, 1);
    for j = 1:Ny
        vy = Wy(:, j) ~= 0;
        [c3, j3] = localRelPerSparsePrep(Py(vy, j), period);
        yPre{j} = {c3, j3, Py(vy, j), Wy(vy, j)};
    end
    for i = 1:Nx
        vx = Wx(:, i) ~= 0;
        pxi = Px(vx, i);
        wxi = Wx(vx, i);
        for j = 1:Ny
            pj = yPre{j};
            c3 = pj{1}; j3 = pj{2}; pyj = pj{3}; wyj = pj{4};
            F = 0.0;
            M = 0.0;
            for uu = uGrid
                Ks = localBuildSparseKernelRelPer(pxi, c3, j3, pyj, ...
                    sigma, cutoff, period, uu);
                if wantRatio
                    [v, ~, m] = mobius.innerProductOrbitSparse( ...
                        Ks, wxi, wyj, r, ...
                        'returnCancellationRatio', true, ...
                        'returnTermMass', true);
                    M = M + m;
                else
                    v = mobius.innerProductOrbitSparse(Ks, wxi, wyj, r);
                end
                F = F + v;
            end
            I(i, j) = F * du;
            if wantRatio
                if M > 0
                    pr = abs(F) / M;
                else
                    pr = 1.0;
                end
                if pr < worst
                    worst = pr;
                end
            end
        end
    end
end
