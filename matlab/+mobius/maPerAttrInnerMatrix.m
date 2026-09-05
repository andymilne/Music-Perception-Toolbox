function I = maPerAttrInnerMatrix(Px, Wx, Py, Wy, sigma, r, isRel, ...
                                    isPer, period, opts)
%MOBIUS.MAPERATTRINNERMATRIX  Per-attribute (event_X, event_Y) IP matrix.
%
%   I = MOBIUS.MAPERATTRINNERMATRIX(PX, WX, PY, WY, SIGMA, R, IS_REL,
%                                    IS_PER, PERIOD)
%   computes the (N_X, N_Y) per-attribute inner product matrix used by
%   the multi-attribute Möbius factorisation
%       <T_X, T_Y> = sum_{n_x, n_y} prod_a I_a[n_x, n_y]
%   on a single attribute.
%
%   Inputs:
%     PX, WX     (K_x, N_x) per-event positions and weights for
%                density X. NaN entries indicate ragged events; they
%                are carried as zero-weight padding.
%     PY, WY     (K_y, N_y) same for Y.
%     SIGMA      positive scalar.
%     R          integer >= 1.
%     IS_REL     logical.
%     IS_PER     logical.
%     PERIOD     positive scalar; periodic mode only.
%
%   Name-Value options:
%     truncationSigmas         (1,1) double, default mptDefaults('truncationSigmas').
%                              When finite, kernel entries whose squared
%                              distance exceeds the cutoff are zeroed
%                              without evaluating exp() in every
%                              kernel-evaluation branch (r=1 abs, r>=2
%                              abs via the batched Möbius method, and
%                              r>=2 rel via the translation grid).
%                              Threshold matches: kernel value falls
%                              below exp(-truncationSigmas^2 / 2).
%     pruneZeroWeightEvents    (1,1) logical, default true. Drops
%                              events with column-wise weight
%                              identically zero (treating NaN as
%                              missing) before dispatching to any
%                              sub-helper. Such events contribute zero
%                              to every kernel entry, so the result is
%                              mathematically unchanged; the saving is
%                              wall-clock. Combines naturally with
%                              truncationSigmas since WEIGHTEVENTS
%                              hard-zeros the factor outside the cutoff.
%                              Result is scattered back into the full
%                              (N_x, N_y) output shape with zeros in
%                              the dropped rows / columns.
%
%   Output:
%     I          (N_x, N_y) double; entry (n_X, n_Y) is the
%                per-attribute inner product over the atoms of
%                event n_X (X-side) against those of n_Y (Y-side).
%
%   Strategy:
%
%   - r = 1: direct kernel sum (no Möbius decomposition; cancellation impossible).
%     Zero-pad NaN entries (zero weight kills any contribution).
%
%   - r >= 2 abs: every event takes the vectorised batched Möbius
%     method, with zero-weight padding for NaN entries. Accuracy is
%     governed by truncationSigmas rather than by how close K_eff is to
%     R, so no size-based partition is applied; an event with fewer
%     than R non-NaN values contributes no R-tuples and its entries
%     come out as zero. A sparse per-pair orbit is taken instead when
%     the value kernel is large and (non-periodic) well separated.
%
%   - r >= 2 rel: batched translation-grid integration with zero-pad
%     (all event pairs at once, slab-bounded; MOBIUS.RELINNERBATCHED,
%     the single relative-mode evaluator, of which the
%     single-multiset form is the N = 1 specialisation). The caller
%     may take the tuple-centres closed form instead where
%     MOBIUS.MARELATTRPREFERSCENTRES says it is cheaper.
%
%   See also MOBIUS.INNERPRODUCTORBITPWBATCHED, MOBIUS.RELINNERBATCHED,
%            MOBIUS.INNERPRODUCTORBITSPARSE, INTERNAL.TRUNCKERNELEXP.

    arguments
        Px double
        Wx double
        Py double
        Wy double
        sigma (1,1) double {mustBePositive}
        r (1,1) double {mustBeInteger, mustBePositive}
        isRel (1,1) logical
        isPer (1,1) logical
        period (1,1) double
        opts.truncationSigmas (1,1) double = mptDefaults('truncationSigmas')
        opts.pruneZeroWeightEvents (1,1) logical = true
        opts.wrap (1,:) char ...
            {mustBeMember(opts.wrap, {'full-image', 'single-image'})} ...
            = 'full-image'
    end

    [Kx, Nx] = size(Px);
    [Ky, Ny] = size(Py);

    % --- Zero-weight-event pruning (auto, before dispatch) ---
    % An event contributes zero to every output entry iff its weight
    % column is identically zero in this attribute (NaN entries are
    % missing values — equivalent to zero in the IP). Drop such events,
    % recurse on the smaller matrices, scatter the result back.
    if opts.pruneZeroWeightEvents && Nx > 0 && Ny > 0
        colMaxX = max(abs(Wx), [], 1, 'omitnan');
        colMaxY = max(abs(Wy), [], 1, 'omitnan');
        keepX = isfinite(colMaxX) & colMaxX > 0;
        keepY = isfinite(colMaxY) & colMaxY > 0;
        if ~(all(keepX) && all(keepY))
            if ~any(keepX) || ~any(keepY)
                % Every event on at least one side has zero weight;
                % the IP is the all-zero matrix.
                I = zeros(Nx, Ny);
                return;
            end
            subIp = mobius.maPerAttrInnerMatrix( ...
                Px(:, keepX), Wx(:, keepX), ...
                Py(:, keepY), Wy(:, keepY), ...
                sigma, r, isRel, isPer, period, ...
                'truncationSigmas', opts.truncationSigmas, ...
                'wrap', opts.wrap, ...
                'pruneZeroWeightEvents', false);   % avoid infinite recursion
            I = zeros(Nx, Ny);
            I(keepX, keepY) = subIp;
            return;
        end
    end

    truncationSigmas = opts.truncationSigmas;
    wrap = opts.wrap;

    % --- r = 1: direct kernel sum, zero-pad fine (no cancellation) ---
    if r == 1
        I = localR1ZeroPad(Px, Wx, Py, Wy, sigma, isPer, period, ...
            truncationSigmas, wrap);
        return;
    end

    % --- r >= 2 rel: batched translation-grid integration, zero-pad ---
    if isRel
        I = mobius.relInnerBatched(Px, Wx, Py, Wy, sigma, r, ...
            isPer, period, 'truncationSigmas', truncationSigmas);
        return;
    end

    % --- r >= 2 abs: every event takes the batched Mobius route ---
    %
    % Accuracy is governed by truncationSigmas, not by the collection
    % size, so no size-based partition is applied: every event takes the
    % vectorised batched Mobius route, which is also the faster one.
    % Events whose non-NaN value count falls below r contribute no
    % r-tuples; zero-weight padding makes every orbit term containing a
    % padded value vanish, so those entries come out as zero.
    if Nx == 0 || Ny == 0
        I = zeros(Nx, Ny);
        return;
    end
    % Sparse-orbit fast path: when the value kernel is large and the
    % (non-periodic) values are well-separated, a spatially-culled
    % per-pair orbit beats the dense batched contraction. Gate on a
    % cheap density probe from one representative pair.
    [minKernel, maxDensity] = localOrbitSparseThresholds();
    useSparse = false;
    if ~isPer && r >= 2 && Kx * Ky >= minKernel
        vx0 = ~isnan(Px(:, 1)) & ~isnan(Wx(:, 1)) & (Wx(:, 1) ~= 0);
        vy0 = ~isnan(Py(:, 1)) & ~isnan(Wy(:, 1)) & (Wy(:, 1) ~= 0);
        K0 = localBuildSparseKernelAbs( ...
            Px(vx0, 1), Py(vy0, 1), sigma, truncationSigmas);
        if nnz(K0) <= maxDensity * Kx * Ky
            useSparse = true;
        end
    end

    if useSparse
        I = localOrbitSparse(Px, Wx, Py, Wy, sigma, r, truncationSigmas);
    else
        I = localOrbit(Px, Wx, Py, Wy, sigma, r, isPer, period, ...
            truncationSigmas, wrap);
    end
end


% =========================================================================
%  Local helpers
% =========================================================================

function I = localR1ZeroPad(Px, Wx, Py, Wy, sigma, isPer, period, ...
                              truncationSigmas, wrap)
%LOCALR1ZEROPAD  r=1 direct kernel sum with NaN -> zero-weight padding.

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

    absPerFullImage = isPer && strcmp(wrap, 'full-image');

    % Memory: each of diffs, diffs.^2, K_tens is
    % (Kx, chunk_Nx, Ky, Ny) * 8 bytes. Up to ~3 live arrays during
    % evaluation; budget accordingly.
    perRowBytes = 3 * Kx * Ky * Ny * 8;
    memLimit = internal.kernelChunkBytesResolved();
    chunkNx = max(1, min(Nx, floor(memLimit / max(perRowBytes, 1))));

    I = zeros(Nx, Ny);
    for nStart = 1:chunkNx:Nx
        nEnd = min(nStart + chunkNx - 1, Nx);
        idxX = nStart:nEnd;
        nc = numel(idxX);

        diffs = reshape(Px(:, idxX), Kx, nc, 1, 1) ...
              - reshape(Py, 1, 1, Ky, Ny);
        if absPerFullImage
            % Full-image 1-D wrapped Gaussian in overlap convention
            % (exponent_denominator = 4). Same shape as diffs; the
            % r=1 kernel factor is this theta directly (no product
            % across coordinates at r = 1).
            K_tens = internal.wrappedGaussian1d(diffs, sigma, period, ...
                                                 truncationSigmas, 4);
        else
            if isPer
                diffs = diffs - period * floor(diffs / period + 0.5);
            end
            K_tens = internal.truncKernelExp(diffs.^2, sigma, truncationSigmas);
        end
        for n_local = 1:nc
            n_x = idxX(n_local);
            slab = squeeze(K_tens(:, n_local, :, :));   % (Kx, Ky, Ny)
            tmp = reshape(Wx(:, n_x).' * reshape(slab, Kx, Ky*Ny), Ky, Ny);
            I(n_x, :) = sum(tmp .* Wy, 1);
        end
    end
    I = I * sigma * sqrt(pi);
end


function I = localOrbit(Px, Wx, Py, Wy, ...
                          sigma, r, isPer, period, truncationSigmas, wrap)
%LOCALORBIT  Vectorised Möbius-method IP over the whole event-pair grid.
%
%   K still varies per event; zero-pad to the slab Kx / Ky dimensions.
%   The waste factor is K_max/mean(K).

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

    prefactor = (sigma * sqrt(pi))^r;
    absPerFullImage = isPer && strcmp(wrap, 'full-image');

    % Memory: each of diffs, diffs.^2, K_tens is
    % (Kx, chunk_Nx, Ky, Ny) * 8 bytes; ~3 live arrays.
    % The K_pairs reshape adds another N_pairs * Kx * Ky * 8.
    perRowBytes = 4 * Kx * Ky * Ny * 8;
    memLimit = internal.kernelChunkBytesResolved();
    chunkNx = max(1, min(Nx, floor(memLimit / max(perRowBytes, 1))));

    I = zeros(Nx, Ny);
    for nStart = 1:chunkNx:Nx
        nEnd = min(nStart + chunkNx - 1, Nx);
        idxX = nStart:nEnd;
        nc = numel(idxX);

        Px_chunk = Px(:, idxX);
        Wx_chunk = Wx(:, idxX);

        diffs = reshape(Px_chunk, Kx, nc, 1, 1) ...
              - reshape(Py, 1, 1, Ky, Ny);
        if absPerFullImage
            % Full-image 1-D wrapped Gaussian in overlap convention.
            % innerProductOrbitPwBatched consumes the per-position pair
            % kernel unchanged; the r-tuple full-image kernel factors
            % across coordinates as prod_a theta(d_a), delivered by the orbit
            % reduction over the 1-D theta values.
            K_tens = internal.wrappedGaussian1d(diffs, sigma, period, ...
                                                 truncationSigmas, 4);
        else
            if isPer
                diffs = diffs - period * floor(diffs / period + 0.5);
            end
            K_tens = internal.truncKernelExp(diffs.^2, sigma, truncationSigmas);
        end

        K_perm  = permute(K_tens, [2, 4, 1, 3]);     % (nc, Ny, Kx, Ky)
        K_pairs = reshape(K_perm, nc*Ny, Kx, Ky);

        Wx_t = Wx_chunk.';                            % (nc, Kx)
        Wx_pairs = reshape(repmat(reshape(Wx_t, nc, 1, Kx), 1, Ny, 1), ...
                            nc*Ny, Kx);
        Wy_t = Wy.';                            % (Ny, Ky)
        Wy_pairs = reshape(repmat(reshape(Wy_t, 1, Ny, Ky), nc, 1, 1), ...
                            nc*Ny, Ky);

        flat = mobius.innerProductOrbitPwBatched( ...
            K_pairs, Wx_pairs, Wy_pairs, r, ...
            'prefactor', prefactor);
        I(idxX, :) = reshape(flat, nc, Ny);
    end
end


function [minKernel, maxDensity] = localOrbitSparseThresholds()
%LOCALORBITSPARSETHRESHOLDS  Sparse-orbit cost model (mirror of the Python
%   _ORBIT_SPARSE_MIN_KERNEL / _ORBIT_SPARSE_MAX_DENSITY). The sparse
%   per-pair orbit undercuts the dense batched contraction only when the
%   value kernel is both large and sparse; below these thresholds the
%   dense contraction's constant factors win. Tunable.
    minKernel = 200000;   % Kx * Ky floor
    maxDensity = 0.20;    % nnz / (Kx * Ky) ceiling
end


function I = localOrbitSparse(Px, Wx, Py, Wy, ...
                                sigma, r, truncationSigmas)
%LOCALORBITSPARSE  Whole event-pair grid via the sparse per-pair orbit.
%
%   Absolute mode only. Each event uses just its non-zero-weight,
%   non-NaN values, so variable cardinality is handled naturally (a
%   zero-weight value contributes zero to every orbit term). Mirrors the
%   dense LOCALORBIT output.

    [~, Nx] = size(Px);
    [~, Ny] = size(Py);
    prefactor = (sigma * sqrt(pi))^r;

    yPts = cell(Ny, 1);
    yWts = cell(Ny, 1);
    for j = 1:Ny
        py = Py(:, j); wy = Wy(:, j);
        vy = ~isnan(py) & ~isnan(wy) & (wy ~= 0);
        yPts{j} = py(vy);
        yWts{j} = wy(vy);
    end

    I = zeros(Nx, Ny);
    for i = 1:Nx
        px = Px(:, i); wx = Wx(:, i);
        vx = ~isnan(px) & ~isnan(wx) & (wx ~= 0);
        pxi = px(vx); wxi = wx(vx);
        for j = 1:Ny
            Ksp = localBuildSparseKernelAbs( ...
                pxi, yPts{j}, sigma, truncationSigmas);
            I(i, j) = mobius.innerProductOrbitSparse( ...
                Ksp, wxi, yWts{j}, r, 'prefactor', prefactor);
        end
    end
end


function Ksp = localBuildSparseKernelAbs(pX, pY, sigma, truncationSigmas)
%LOCALBUILDSPARSEKERNELABS  Spatially-culled absolute-mode kernel as a
%   sparse matrix. Keeps only value pairs within the truncation radius via
%   a 1-D sorted window, matching INTERNAL.TRUNCKERNELEXP's cutoff
%   (|d|^2 > 2 (truncationSigmas * sigma)^2) without the dense O(n^2) pass.

    nx = numel(pX);
    ny = numel(pY);
    if nx == 0 || ny == 0
        Ksp = sparse(nx, ny);
        return;
    end
    R = sqrt(2) * truncationSigmas * sigma;
    [pYs, order] = sort(pY(:));
    rowsC = cell(nx, 1);
    colsC = cell(nx, 1);
    valsC = cell(nx, 1);
    for i = 1:nx
        x = pX(i);
        lo = localLowerBound(pYs, x - R);
        hi = localLowerBound(pYs, x + R);
        if hi > lo
            jj = order(lo:hi - 1);
            d = x - pY(jj);
            rowsC{i} = repmat(i, numel(jj), 1);
            colsC{i} = jj(:);
            valsC{i} = exp(-(d(:) .^ 2) / (4 * sigma^2));
        end
    end
    rows = vertcat(rowsC{:});
    cols = vertcat(colsC{:});
    vals = vertcat(valsC{:});
    if isempty(rows)
        Ksp = sparse(nx, ny);
    else
        Ksp = sparse(rows, cols, vals, nx, ny);
    end
end


function idx = localLowerBound(sortedVec, val)
%LOCALLOWERBOUND  First index i with sortedVec(i) >= val (numel+1 if none).
    lo = 1;
    hi = numel(sortedVec) + 1;
    while lo < hi
        mid = floor((lo + hi) / 2);
        if sortedVec(mid) < val
            lo = mid + 1;
        else
            hi = mid;
        end
    end
    idx = lo;
end
