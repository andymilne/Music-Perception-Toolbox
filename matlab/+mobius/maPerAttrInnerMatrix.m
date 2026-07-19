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
%     PX, WX     (K_x, N_x) per-event slot positions and weights for
%                density X. NaN entries indicate ragged events; they
%                are routed and handled per-pair as appropriate.
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
%                              abs safe via batched Möbius, r>=2 abs
%                              unsafe via direct enumeration, and the
%                              r>=2 rel per-pair fallback). Threshold
%                              matches: kernel value falls below
%                              exp(-truncationSigmas^2 / 2).
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
%                per-attribute inner product over the slot values of
%                event n_X (X-side) against those of n_Y (Y-side).
%
%   Strategy:
%
%   - r = 1: direct kernel sum (no Möbius decomposition; cancellation impossible).
%     Zero-pad NaN entries (zero weight kills any contribution).
%
%   - r >= 2 abs: hybrid safe/unsafe partition. An event is "safe" on
%     this attribute iff its non-NaN slot count K_eff satisfies
%     K_eff - R >= _ORBIT_K_MINUS_R_MIN = 2 (the precision margin
%     used elsewhere in the Möbius machinery). Safe-vs-safe pairs flow
%     through the vectorised batched Möbius method with within-safe-group
%     zero-padding. Pairs involving any unsafe event flow through the
%     direct-enumeration helper MOBIUS.INNERPRODUCTDIRECTABSSINGLEMULTISET, which
%     is exact for any K >= R (no Möbius alternating sum).
%
%   - r >= 2 rel: batched translation-grid integration with zero-pad
%     (all event pairs at once, slab-bounded; MOBIUS.RELINNERBATCHED,
%     the single relative-mode evaluator, of which the
%     single-collection form is the N = 1 specialisation).
%     Auto dispatch routes most small-K rel groups to
%     Bulger globally; this path runs only on explicit
%     method='mobius' opt-in. Events with K_eff - R below the precision
%     margin in this niche regime may lose precision in the Möbius
%     alternating sum; users who care about exact rel + ragged
%     Möbius-mode behaviour should either filter events to K_eff >=
%     R + 2 or use method='auto' (which routes to Bulger's method).
%
%   See also MOBIUS.INNERPRODUCTORBITPWBATCHED, MOBIUS.INNERPRODUCTDIRECTABSSINGLEMULTISET,
%            MOBIUS.ORBITINNERRELSINGLEMULTISET, INTERNAL.TRUNCKERNELEXP.

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
    end

    [Kx, Nx] = size(Px); %#ok<ASGLU>
    [Ky, Ny] = size(Py); %#ok<ASGLU>

    % --- Zero-weight-event pruning (auto, before dispatch) ---
    % An event contributes zero to every output entry iff its weight
    % column is identically zero in this attribute (NaN entries are
    % missing slots — equivalent to zero in the IP). Drop such events,
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
                'pruneZeroWeightEvents', false);   % avoid infinite recursion
            I = zeros(Nx, Ny);
            I(keepX, keepY) = subIp;
            return;
        end
    end

    truncationSigmas = opts.truncationSigmas;

    % --- r = 1: direct kernel sum, zero-pad fine (no cancellation) ---
    if r == 1
        I = localR1ZeroPad(Px, Wx, Py, Wy, sigma, isPer, period, ...
            truncationSigmas);
        return;
    end

    % --- r >= 2 rel: batched translation-grid integration, zero-pad ---
    if isRel
        I = mobius.relInnerBatched(Px, Wx, Py, Wy, sigma, r, ...
            isPer, period, 'truncationSigmas', truncationSigmas);
        return;
    end

    % --- r >= 2 abs: hybrid safe/unsafe partition ---

    K_MARGIN_MIN = 2;   % matches _ORBIT_K_MINUS_R_MIN

    % Per-event K_eff (count of non-NaN slots), per side.
    K_eff_x = sum(~isnan(Px) & ~isnan(Wx), 1);   % (1, Nx)
    K_eff_y = sum(~isnan(Py) & ~isnan(Wy), 1);   % (1, Ny)

    safe_x_mask = (K_eff_x - r) >= K_MARGIN_MIN;
    safe_y_mask = (K_eff_y - r) >= K_MARGIN_MIN;
    safe_x_idx   = find(safe_x_mask);
    unsafe_x_idx = find(~safe_x_mask);
    safe_y_idx   = find(safe_y_mask);
    unsafe_y_idx = find(~safe_y_mask);

    I = zeros(Nx, Ny);

    % --- Safe x Safe submatrix: vectorised batched Möbius method ---
    if ~isempty(safe_x_idx) && ~isempty(safe_y_idx)
        I(safe_x_idx, safe_y_idx) = localSafeSafeOrbit( ...
            Px(:, safe_x_idx), Wx(:, safe_x_idx), ...
            Py(:, safe_y_idx), Wy(:, safe_y_idx), ...
            sigma, r, isPer, period, truncationSigmas);
    end

    % --- Pairs involving any unsafe event: K-grouped batched direct ---
    % Under v2.2.0 this was a pair-by-pair MATLAB double-loop calling
    % mobius.innerProductDirectAbsSingleMultiset per (n_x, n_y); for variable-K_a
    % workloads with many unsafe events that dominated runtime by
    % 10-100x over the actual numerical work.
    %
    % K_a grouping: partition the unsafe-involved event-index union by
    % K_eff value per side, then batch direct enumeration per
    % (K_eff_x, K_eff_y) sub-block. Within a sub-block every event
    % shares an ordered-r-tuple shape, so the IP matrix is a single
    % contracted tensor op. Coverage is (unsafe_x, all_y) plus
    % (safe_x, unsafe_y), the same partition as v2.2.0.
    if ~isempty(unsafe_x_idx)
        I = localFillDirectEnumGroups(I, ...
            Px, Wx, Py, Wy, unsafe_x_idx, 1:Ny, ...
            K_eff_x, K_eff_y, sigma, r, isPer, period, truncationSigmas);
    end
    if ~isempty(unsafe_y_idx) && ~isempty(safe_x_idx)
        I = localFillDirectEnumGroups(I, ...
            Px, Wx, Py, Wy, safe_x_idx, unsafe_y_idx, ...
            K_eff_x, K_eff_y, sigma, r, isPer, period, truncationSigmas);
    end
end


% =========================================================================
%  Local helpers
% =========================================================================

function I = localR1ZeroPad(Px, Wx, Py, Wy, sigma, isPer, period, ...
                              truncationSigmas)
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
        if isPer
            diffs = diffs - period * floor(diffs / period + 0.5);
        end
        K_tens = internal.truncKernelExp(diffs.^2, sigma, truncationSigmas);
        for n_local = 1:nc
            n_x = idxX(n_local);
            slab = squeeze(K_tens(:, n_local, :, :));   % (Kx, Ky, Ny)
            tmp = reshape(Wx(:, n_x).' * reshape(slab, Kx, Ky*Ny), Ky, Ny);
            I(n_x, :) = sum(tmp .* Wy, 1);
        end
    end
    I = I * sigma * sqrt(pi);
end


function I = localSafeSafeOrbit(Px_safe, Wx_safe, Py_safe, Wy_safe, ...
                                  sigma, r, isPer, period, truncationSigmas)
%LOCALSAFESAFEORBIT  Vectorised Möbius-method IP on the safe submatrix.
%
%   Within the safe group K still varies per event; zero-pad to the
%   slab Kx / Ky dimensions. The waste factor K_max/mean(K) is smaller
%   here than in the all-events approach because the safe group has
%   more uniform K (everyone has K_eff >= r + 2).

    [Kx, Nx_safe] = size(Px_safe);
    [Ky, Ny_safe] = size(Py_safe);

    nanX = isnan(Px_safe) | isnan(Wx_safe);
    if any(nanX(:))
        Px_safe(nanX) = 0;
        Wx_safe(nanX) = 0;
    end
    nanY = isnan(Py_safe) | isnan(Wy_safe);
    if any(nanY(:))
        Py_safe(nanY) = 0;
        Wy_safe(nanY) = 0;
    end

    prefactor = (sigma * sqrt(pi))^r;

    % Memory: each of diffs, diffs.^2, K_tens is
    % (Kx, chunk_Nxs, Ky, Ny_safe) * 8 bytes; ~3 live arrays.
    % The K_pairs reshape adds another N_pairs * Kx * Ky * 8.
    perRowBytes = 4 * Kx * Ky * Ny_safe * 8;
    memLimit = internal.kernelChunkBytesResolved();
    chunkNxs = max(1, min(Nx_safe, floor(memLimit / max(perRowBytes, 1))));

    I = zeros(Nx_safe, Ny_safe);
    for nStart = 1:chunkNxs:Nx_safe
        nEnd = min(nStart + chunkNxs - 1, Nx_safe);
        idxX = nStart:nEnd;
        nc = numel(idxX);

        Px_chunk = Px_safe(:, idxX);
        Wx_chunk = Wx_safe(:, idxX);

        diffs = reshape(Px_chunk, Kx, nc, 1, 1) ...
              - reshape(Py_safe, 1, 1, Ky, Ny_safe);
        if isPer
            diffs = diffs - period * floor(diffs / period + 0.5);
        end
        K_tens = internal.truncKernelExp(diffs.^2, sigma, truncationSigmas);

        K_perm  = permute(K_tens, [2, 4, 1, 3]);     % (nc, Ny_safe, Kx, Ky)
        K_pairs = reshape(K_perm, nc*Ny_safe, Kx, Ky);

        Wx_t = Wx_chunk.';                            % (nc, Kx)
        Wx_pairs = reshape(repmat(reshape(Wx_t, nc, 1, Kx), 1, Ny_safe, 1), ...
                            nc*Ny_safe, Kx);
        Wy_t = Wy_safe.';                            % (Ny_safe, Ky)
        Wy_pairs = reshape(repmat(reshape(Wy_t, 1, Ny_safe, Ky), nc, 1, 1), ...
                            nc*Ny_safe, Ky);

        flat = mobius.innerProductOrbitPwBatched( ...
            K_pairs, Wx_pairs, Wy_pairs, r, ...
            'prefactor', prefactor);
        I(idxX, :) = reshape(flat, nc, Ny_safe);
    end
end


function I = localFillDirectEnumGroups(I, Px, Wx, Py, Wy, x_idx, y_idx, ...
                                         K_eff_x, K_eff_y, sigma, r, ...
                                         isPer, period, truncationSigmas)
%LOCALFILLDIRECTENUMGROUPS  K-grouped batched direct-enum fill.
%
%   Partitions x_idx by K_eff_x value and y_idx by K_eff_y value, then
%   computes each (K_x_val, K_y_val) sub-block via a single vectorised
%   tensor contraction in localBatchedDirectEnumAbsSingleMultiset. Replaces the
%   v2.2.0 per-pair MATLAB double loop.

    if isempty(x_idx) || isempty(y_idx)
        return;
    end

    uniqueKx = unique(K_eff_x(x_idx));
    uniqueKy = unique(K_eff_y(y_idx));

    for Kx_val = uniqueKx
        xMask = K_eff_x(x_idx) == Kx_val;
        x_grp = x_idx(xMask);
        if isempty(x_grp) || Kx_val < r
            continue;
        end
        [Px_grp, Wx_grp] = localPackNanTop(Px(:, x_grp), Wx(:, x_grp));
        Px_grp = Px_grp(1:double(Kx_val), :);
        Wx_grp = Wx_grp(1:double(Kx_val), :);
        for Ky_val = uniqueKy
            yMask = K_eff_y(y_idx) == Ky_val;
            y_grp = y_idx(yMask);
            if isempty(y_grp) || Ky_val < r
                continue;
            end
            [Py_grp, Wy_grp] = localPackNanTop(Py(:, y_grp), Wy(:, y_grp));
            Py_grp = Py_grp(1:double(Ky_val), :);
            Wy_grp = Wy_grp(1:double(Ky_val), :);
            sub_ip = localBatchedDirectEnumAbsSingleMultiset( ...
                Px_grp, Wx_grp, Py_grp, Wy_grp, ...
                sigma, r, isPer, period, truncationSigmas);
            I(x_grp, y_grp) = sub_ip;
        end
    end
end


function [Pp, Wp] = localPackNanTop(P, W)
%LOCALPACKNANTOP  Pack non-NaN slots to the top of each column.
%
%   Returns P_packed, W_packed where for each column n the first
%   K_eff(n) rows hold the valid slots (preserving their original
%   order) and the rest are NaN. The buildExpTens convention already
%   places NaN at the bottom, in which case this is mathematically a
%   no-op; per-column packing handles user-constructed densities with
%   arbitrary NaN positions.

    [K, N] = size(P);
    Pp = nan(K, N);
    Wp = nan(K, N);
    for n = 1:N
        valid = ~(isnan(P(:, n)) | isnan(W(:, n)));
        k = sum(valid);
        if k == 0
            continue;
        end
        Pp(1:k, n) = P(valid, n);
        Wp(1:k, n) = W(valid, n);
    end
end


function I = localBatchedDirectEnumAbsSingleMultiset(Px, Wx, Py, Wy, sigma, r, ...
                                          isPer, period, truncationSigmas)
%LOCALBATCHEDDIRECTENUMABSSINGLEMULTISET  Batched direct r-tuple enumeration IP.
%
%   Vectorised replacement for repeated calls to
%   mobius.innerProductDirectAbsSingleMultiset when every event in Px has the
%   same K_x = K_eff_x and every event in Py has the same
%   K_y = K_eff_y (no NaN within the first K rows of either side).
%
%   Inputs:
%       Px : (K_x, N_x), no NaN
%       Wx : (K_x, N_x), no NaN
%       Py : (K_y, N_y), no NaN
%       Wy : (K_y, N_y), no NaN
%   Returns:
%       I  : (N_x, N_y) inner-product matrix
%
%   No Möbius alternating sum; exact for any K_x, K_y >= r.

    [Kx, Nx] = size(Px);
    [Ky, Ny] = size(Py);

    if Kx < r || Ky < r
        I = zeros(Nx, Ny);
        return;
    end

    if r == 1
        % r=1: direct kernel sum without r-tuple enumeration.
        diffs = reshape(Px, Kx, Nx, 1, 1) - reshape(Py, 1, 1, Ky, Ny);
        if isPer
            diffs = diffs - period * floor(diffs / period + 0.5);
        end
        K_tens = internal.truncKernelExp(diffs.^2, sigma, truncationSigmas);
        I = zeros(Nx, Ny);
        for n_x = 1:Nx
            slab = squeeze(K_tens(:, n_x, :, :));   % (Kx, Ky, Ny)
            tmp = reshape(Wx(:, n_x).' * reshape(slab, Kx, Ky*Ny), Ky, Ny);
            I(n_x, :) = sum(tmp .* Wy, 1);
        end
        I = I * sigma * sqrt(pi);
        return;
    end

    % --- r >= 2: enumerate ordered r-tuple indices ---
    idx_x = perms(1:Kx);
    % perms returns rows in reverse lex order; keep only the leading r
    % columns to match Python's permutations(range(K_x), r).
    idx_x = idx_x(:, 1:r);
    % perms gives all permutations of K_x elements; we need K_x! / (K_x-r)!
    % ordered r-tuples. Drop duplicates that arise from permutations
    % differing only in trailing columns.
    idx_x = unique(idx_x, 'rows', 'stable');
    idx_y = perms(1:Ky);
    idx_y = idx_y(:, 1:r);
    idx_y = unique(idx_y, 'rows', 'stable');

    nJ_x = size(idx_x, 1);                % K_x! / (K_x - r)!
    nJ_y = size(idx_y, 1);

    % Gather tuple slot positions per event. Px(idx_x.', :) is
    % (r, nJ_x, N_x); we want U_x of shape (r, N_x, nJ_x).
    U_x = permute(Px(idx_x.', :), [1 3 2]);  % (r, N_x, nJ_x)... wait
    % MATLAB: Px(idx_x.', :) with idx_x.' (r, nJ_x) — fancy index is
    % (r*nJ_x, N_x). Need to reshape carefully.
    Pxgath = Px(idx_x.', :);                 % ((r*nJ_x), N_x)
    Pxgath = reshape(Pxgath, r, nJ_x, Nx);   % (r, nJ_x, N_x)
    U_x = permute(Pxgath, [1 3 2]);          % (r, N_x, nJ_x)

    Pygath = Py(idx_y.', :);
    Pygath = reshape(Pygath, r, nJ_y, Ny);
    U_y = permute(Pygath, [1 3 2]);          % (r, N_y, nJ_y)

    Wxgath = Wx(idx_x.', :);                 % ((r*nJ_x), N_x)
    Wxgath = reshape(Wxgath, r, nJ_x, Nx);   % (r, nJ_x, N_x)
    Wj_x = reshape(prod(Wxgath, 1), nJ_x, Nx).';  % (N_x, nJ_x)

    Wygath = Wy(idx_y.', :);
    Wygath = reshape(Wygath, r, nJ_y, Ny);
    Wj_y = reshape(prod(Wygath, 1), nJ_y, Ny).';  % (N_y, nJ_y)

    % Differences: (r, N_x, nJ_x, N_y, nJ_y)
    diffs = reshape(U_x, r, Nx, nJ_x, 1, 1) ...
          - reshape(U_y, r, 1, 1, Ny, nJ_y);
    if isPer
        diffs = diffs - period * floor(diffs / period + 0.5);
    end
    Q = reshape(sum(diffs.^2, 1), Nx, nJ_x, Ny, nJ_y);
    Kmat = internal.truncKernelExp(Q, sigma, truncationSigmas);

    % Contract: ip(n_x, n_y) = sum_{jx, jy}
    %               Wj_x(n_x, jx) * Kmat(n_x, jx, n_y, jy) * Wj_y(n_y, jy)
    % MATLAB has no named tensor-contraction primitive; do it as two reduction steps.
    %   step 1: T1(n_x, n_y, jy) = sum_jx Wj_x(n_x, jx) * Kmat(n_x, jx, n_y, jy)
    %   step 2: I(n_x, n_y)      = sum_jy T1(n_x, n_y, jy) * Wj_y(n_y, jy)
    % Reshape for compact bsxfun-style products.

    Kperm = permute(Kmat, [1 3 4 2]);        % (N_x, N_y, nJ_y, nJ_x)
    Wj_x_b = reshape(Wj_x, Nx, 1, 1, nJ_x);  % (N_x, 1, 1, nJ_x)
    T1 = sum(Kperm .* Wj_x_b, 4);            % (N_x, N_y, nJ_y)

    Wj_y_b = reshape(Wj_y, 1, Ny, nJ_y);     % (1, N_y, nJ_y)
    I = sum(T1 .* Wj_y_b, 3);                % (N_x, N_y)

    I = I * (sigma * sqrt(pi))^r;
end
