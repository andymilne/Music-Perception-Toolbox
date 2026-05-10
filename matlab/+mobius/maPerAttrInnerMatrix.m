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
%                are zero-weight padded internally.
%     PY, WY     (K_y, N_y) same for Y.
%     SIGMA      positive scalar.
%     R          integer >= 1.
%     IS_REL     logical.
%     IS_PER     logical.
%     PERIOD     positive scalar; periodic mode only.
%
%   Output:
%     I          (N_x, N_y) double; entry (n_X, n_Y) is the
%                per-attribute inner product over the K slot values of
%                event n_X (X-side) against those of n_Y (Y-side).
%
%   Vectorisation strategy:
%   - r = 1     : direct (K_x, K_y) kernel slab per event-X, vectorised
%                 over event-Y.
%   - r >= 2 abs: build (N_x*N_y, K_x, K_y) kernel tensor and dispatch
%                 to MOBIUS.INNERPRODUCTORBITPWBATCHED.
%   - r >= 2 rel: per-event-pair loop calling MOBIUS.ORBITINNERRELSA;
%                 un-vectorised in v2.2 (rel-MA dispatch routes to
%                 pairwise unless the user opts in to orbit explicitly).
%
%   Ragged K_{a,n} (NaN-padded events) is handled via zero-weight
%   padding: NaN entries in P or W are replaced with arbitrary p (0)
%   and zero weight, which kills any orbit term involving the padded
%   slot and yields the mathematically correct event IP.
%
%   See also MOBIUS.INNERPRODUCTORBITPWBATCHED, MOBIUS.ORBITINNERRELSA.

    [Kx, Nx] = size(Px);
    [Ky, Ny] = size(Py);

    % Zero-pad: replace NaN entries (in P or W) with 0.
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

    if r == 1
        % Direct sum (no orbit machinery needed at r=1):
        %   I[n_x, n_y] = (sigma*sqrt(pi))^r
        %                * sum_{i,j} Wx[i,n_x] * Wy[j,n_y] * K[i,n_x;j,n_y]
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
        I = I * (sigma * sqrt(pi))^r;
        return;
    end

    % r >= 2 absolute: vectorised across event pairs via per-batch weights.
    if ~isRel
        diffs = reshape(Px, Kx, Nx, 1, 1) - reshape(Py, 1, 1, Ky, Ny);
        if isPer
            diffs = diffs - period * floor(diffs / period + 0.5);
        end
        K_tens = exp(-(diffs.^2) / (4 * sigma^2));   % (Kx, Nx, Ky, Ny)

        % Reshape to (Nx*Ny, Kx, Ky):
        K_perm  = permute(K_tens, [2, 4, 1, 3]);     % (Nx, Ny, Kx, Ky)
        K_pairs = reshape(K_perm, Nx*Ny, Kx, Ky);

        % Per-batch A-side weights: (Nx, Ny, Kx) reshaped to (Nx*Ny, Kx).
        Wx_t = Wx.';                                  % (Nx, Kx)
        Wx_pairs = reshape(repmat(reshape(Wx_t, Nx, 1, Kx), 1, Ny, 1), ...
                            Nx*Ny, Kx);
        % Per-batch B-side weights: (Nx, Ny, Ky) reshaped to (Nx*Ny, Ky).
        Wy_t = Wy.';                                  % (Ny, Ky)
        Wy_pairs = reshape(repmat(reshape(Wy_t, 1, Ny, Ky), Nx, 1, 1), ...
                            Nx*Ny, Ky);

        flat = mobius.innerProductOrbitPwBatched( ...
            K_pairs, Wx_pairs, Wy_pairs, r, ...
            'prefactor', (sigma * sqrt(pi))^r);
        I = reshape(flat, Nx, Ny);
        return;
    end

    % r >= 2 relative: per-pair loop (un-vectorised in v2.2).
    I = zeros(Nx, Ny);
    for n_x = 1:Nx
        for n_y = 1:Ny
            I(n_x, n_y) = mobius.orbitInnerRelSA( ...
                Px(:, n_x), Wx(:, n_x), Py(:, n_y), Wy(:, n_y), ...
                sigma, r, isPer, period);
        end
    end
end
