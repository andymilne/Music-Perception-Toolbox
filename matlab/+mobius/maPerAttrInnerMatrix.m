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

    % --- Pairs involving any unsafe event: direct enumeration ---
    % unsafe_x x all_y: covers the unsafe-x rows.
    % safe_x x unsafe_y: covers the unsafe-y columns within safe-x rows.
    % Together these cover all pairs not in (safe_x, safe_y).

    for ii = 1:numel(unsafe_x_idx)
        n_x = unsafe_x_idx(ii);
        for n_y = 1:Ny
            I(n_x, n_y) = mobius.innerProductDirectAbsSA( ...
                Px(:, n_x), Wx(:, n_x), Py(:, n_y), Wy(:, n_y), ...
                sigma, r, isPer, period);
        end
    end
    for jj = 1:numel(unsafe_y_idx)
        n_y = unsafe_y_idx(jj);
        for ii = 1:numel(safe_x_idx)
            n_x = safe_x_idx(ii);
            I(n_x, n_y) = mobius.innerProductDirectAbsSA( ...
                Px(:, n_x), Wx(:, n_x), Py(:, n_y), Wy(:, n_y), ...
                sigma, r, isPer, period);
        end
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
