function I = maPerAttrInnerMatrix(Px, Wx, Py, Wy, sigma, r, isRel, ...
                                    isPer, period)
%MOBIUS.MAPERATTRINNERMATRIX  Per-attribute (event_X, event_Y) IP matrix.
%
%   I = MOBIUS.MAPERATTRINNERMATRIX(PX, WX, PY, WY, SIGMA, R, IS_REL,
%                                    IS_PER, PERIOD)
%   computes the (N_X, N_Y) per-attribute inner product matrix used by
%   the multi-attribute orbit-Möbius factorisation
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
%   Output:
%     I          (N_x, N_y) double; entry (n_X, n_Y) is the
%                per-attribute inner product over the slot values of
%                event n_X (X-side) against those of n_Y (Y-side).
%
%   Strategy:
%
%   - r = 1: direct kernel sum (no orbit; cancellation impossible).
%     Zero-pad NaN entries (zero weight kills any contribution).
%
%   - r >= 2 abs: hybrid safe/unsafe partition. An event is "safe" on
%     this attribute iff its non-NaN slot count K_eff satisfies
%     K_eff - R >= _ORBIT_K_MINUS_R_MIN = 2 (the precision margin
%     used elsewhere in the orbit machinery). Safe-vs-safe pairs flow
%     through the vectorised batched orbit path with within-safe-group
%     zero-padding. Pairs involving any unsafe event flow through the
%     direct-enumeration helper MOBIUS.INNERPRODUCTDIRECTABSSA, which
%     is exact for any K >= R (no Möbius alternating sum).
%
%   - r >= 2 rel: per-event-pair loop with zero-pad, calling
%     MOBIUS.ORBITINNERRELSA. Auto dispatch routes any rel group to
%     pairwise globally; this path runs only on explicit
%     method='orbit' opt-in. Events with K_eff - R below the precision
%     margin in this niche regime may lose precision in the orbit
%     alternating sum; users who care about exact rel + ragged
%     orbit-mode behaviour should either filter events to K_eff >=
%     R + 2 or use method='auto' (which routes to pairwise).
%
%   See also MOBIUS.INNERPRODUCTORBITPWBATCHED, MOBIUS.INNERPRODUCTDIRECTABSSA,
%            MOBIUS.ORBITINNERRELSA.

    [Kx, Nx] = size(Px); %#ok<ASGLU>
    [Ky, Ny] = size(Py); %#ok<ASGLU>

    % --- r = 1: direct kernel sum, zero-pad fine (no cancellation) ---
    if r == 1
        I = localR1ZeroPad(Px, Wx, Py, Wy, sigma, isPer, period);
        return;
    end

    % --- r >= 2 rel: per-pair loop, zero-pad (rare regime) ---
    if isRel
        I = localR2RelPerPair(Px, Wx, Py, Wy, sigma, r, isPer, period);
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

    % --- Safe x Safe submatrix: vectorised batched orbit ---
    if ~isempty(safe_x_idx) && ~isempty(safe_y_idx)
        I(safe_x_idx, safe_y_idx) = localSafeSafeOrbit( ...
            Px(:, safe_x_idx), Wx(:, safe_x_idx), ...
            Py(:, safe_y_idx), Wy(:, safe_y_idx), ...
            sigma, r, isPer, period);
    end

    % --- Pairs involving any unsafe event: K-grouped batched direct ---
    % Under v2.2.0 this was a pair-by-pair MATLAB double-loop calling
    % mobius.innerProductDirectAbsSA per (n_x, n_y); for variable-K_a
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
            K_eff_x, K_eff_y, sigma, r, isPer, period);
    end
    if ~isempty(unsafe_y_idx) && ~isempty(safe_x_idx)
        I = localFillDirectEnumGroups(I, ...
            Px, Wx, Py, Wy, safe_x_idx, unsafe_y_idx, ...
            K_eff_x, K_eff_y, sigma, r, isPer, period);
    end
end


% =========================================================================
%  Local helpers
% =========================================================================

function I = localR1ZeroPad(Px, Wx, Py, Wy, sigma, isPer, period)
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

    diffs = reshape(Px, Kx, Nx, 1, 1) - reshape(Py, 1, 1, Ky, Ny);
    if isPer
        diffs = diffs - period * floor(diffs / period + 0.5);
    end
    K_tens = exp(-(diffs.^2) / (4 * sigma^2));
    I = zeros(Nx, Ny);
    for n_x = 1:Nx
        slab = squeeze(K_tens(:, n_x, :, :));     % (Kx, Ky, Ny)
        tmp = reshape(Wx(:, n_x).' * reshape(slab, Kx, Ky*Ny), Ky, Ny);
        I(n_x, :) = sum(tmp .* Wy, 1);
    end
    I = I * sigma * sqrt(pi);
end


function I = localR2RelPerPair(Px, Wx, Py, Wy, sigma, r, isPer, period)
%LOCALR2RELPERPAIR  r>=2 rel: per-pair loop with zero-pad.

    [~, Nx] = size(Px);
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

    I = zeros(Nx, Ny);
    for n_x = 1:Nx
        for n_y = 1:Ny
            I(n_x, n_y) = mobius.orbitInnerRelSA( ...
                Px(:, n_x), Wx(:, n_x), Py(:, n_y), Wy(:, n_y), ...
                sigma, r, isPer, period);
        end
    end
end


function I = localSafeSafeOrbit(Px_safe, Wx_safe, Py_safe, Wy_safe, ...
                                  sigma, r, isPer, period)
%LOCALSAFESAFEORBIT  Vectorised orbit IP on the safe submatrix.
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

    diffs = reshape(Px_safe, Kx, Nx_safe, 1, 1) ...
          - reshape(Py_safe, 1, 1, Ky, Ny_safe);
    if isPer
        diffs = diffs - period * floor(diffs / period + 0.5);
    end
    K_tens = exp(-(diffs.^2) / (4 * sigma^2));   % (Kx, Nx_safe, Ky, Ny_safe)

    K_perm  = permute(K_tens, [2, 4, 1, 3]);     % (Nx_safe, Ny_safe, Kx, Ky)
    K_pairs = reshape(K_perm, Nx_safe*Ny_safe, Kx, Ky);

    Wx_t = Wx_safe.';                            % (Nx_safe, Kx)
    Wx_pairs = reshape(repmat(reshape(Wx_t, Nx_safe, 1, Kx), 1, Ny_safe, 1), ...
                        Nx_safe*Ny_safe, Kx);
    Wy_t = Wy_safe.';                            % (Ny_safe, Ky)
    Wy_pairs = reshape(repmat(reshape(Wy_t, 1, Ny_safe, Ky), Nx_safe, 1, 1), ...
                        Nx_safe*Ny_safe, Ky);

    flat = mobius.innerProductOrbitPwBatched( ...
        K_pairs, Wx_pairs, Wy_pairs, r, ...
        'prefactor', (sigma * sqrt(pi))^r);
    I = reshape(flat, Nx_safe, Ny_safe);
end


function I = localFillDirectEnumGroups(I, Px, Wx, Py, Wy, x_idx, y_idx, ...
                                         K_eff_x, K_eff_y, sigma, r, ...
                                         isPer, period)
%LOCALFILLDIRECTENUMGROUPS  K-grouped batched direct-enum fill.
%
%   Partitions x_idx by K_eff_x value and y_idx by K_eff_y value, then
%   computes each (K_x_val, K_y_val) sub-block via a single vectorised
%   tensor contraction in localBatchedDirectEnumAbsSA. Replaces the
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
            sub_ip = localBatchedDirectEnumAbsSA( ...
                Px_grp, Wx_grp, Py_grp, Wy_grp, ...
                sigma, r, isPer, period);
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


function I = localBatchedDirectEnumAbsSA(Px, Wx, Py, Wy, sigma, r, ...
                                          isPer, period)
%LOCALBATCHEDDIRECTENUMABSSA  Batched direct r-tuple enumeration IP.
%
%   Vectorised replacement for repeated calls to
%   mobius.innerProductDirectAbsSA when every event in Px has the
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
        K_tens = exp(-(diffs.^2) / (4 * sigma^2));
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
    Kmat = exp(-Q / (4 * sigma^2));

    % Contract: ip(n_x, n_y) = sum_{jx, jy}
    %               Wj_x(n_x, jx) * Kmat(n_x, jx, n_y, jy) * Wj_y(n_y, jy)
    % MATLAB lacks named einsum; do it as two reduction steps.
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
