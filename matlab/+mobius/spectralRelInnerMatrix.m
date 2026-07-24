function I = spectralRelInnerMatrix(Px, Wx, Py, Wy, sigma, r, isPer, period)
%MOBIUS.SPECTRALRELINNERMATRIX  Relative inner matrix by the spectral form.
%
%   I = MOBIUS.SPECTRALRELINNERMATRIX(PX, WX, PY, WY, SIGMA, R, ISPER,
%   PERIOD) returns the (N_x, N_y) per-attribute inner matrix for a
%   relative attribute, or [] when the branch declines (the caller then
%   falls through to the translation-grid contraction).
%
%   The relative inner product is the integral of the absolute inner
%   product under a rigid diagonal shift, and transforming along that
%   shift constrains each partition's total mode index to zero. Every
%   partition therefore contributes a product of per-block spectra
%   evaluated at block-summed frequencies, so each event carries a
%   single spectrum
%
%       S_n(xi) = sum_pi mu(pi) prod_B A_{|B|}(eta_B),
%       A_m(eta) = sum_i w_{i,n}^m exp(-i eta p_{i,n}),
%
%   and the matrix over event pairs is the Gram matrix of those spectra
%   against the envelope exp(-sigma^2 sum_s xi_s^2), scaled by
%
%       C_r = r * (sigma^2 * dxi)^(r-1).
%
%   The spectra are per-event objects, so the K-dependence is paid once
%   per event rather than once per pair, and the per-pair cost is
%   independent of K. Frequencies are exact on the circle
%   (xi_m = 2 pi m / P, valid at any sigma/period, since the
%   wrapped-Gaussian coefficients are closed form); on the line they are
%   spaced 2 pi / L for an embedding period L covering both sides' spans
%   plus a truncation margin.
%
%   Hermitian symmetry cuts the working sets in half on two sides:
%     (i)  env is even under xi -> -xi and S_n(-xi) = conj(S_n(xi)),
%          since the underlying Gaussian mixture is real, so only the DC
%          point plus one point per (xi, -xi) pair is evaluated on the
%          grid, with the off-DC envelope doubled to account for the
%          conjugate partner;
%     (ii) real event weights give A_m(-eta) = conj(A_m(eta)), so
%          per-event phases are evaluated only on non-negative modes and
%          the full A_m table is assembled by reflection and
%          conjugation.
%   The first halving cuts the partition loop and the final matmul; the
%   second halves the per-event phase matmul, which dominates at r = 2
%   where the partition loop is trivial.
%
%   Returns [] when the mode grid would exceed MAXPOINTS (a memory
%   guard), or when the cost gate judges the mode grid unrepaid against
%   the grid route's K^2 per event pair.
%
%   COST_C is calibrated on 497 measured cells spanning both periodic
%   modes, r = 2..4, K = 4..30, event counts 1..16 and sigma/period from
%   0.001 to 0.2, scored by routing regret -- the wall time actually
%   paid against an oracle that always picks the faster route. 3160
%   gives 1.043x of oracle; the earlier 1000 gave 1.127x, and was
%   one-sided: 29 of its 31 errors declined a route that would have won,
%   spending 2378 ms to avoid 775 ms. Raising it improves the mean and
%   the tail together (worst misroute 11.0x -> 9.2x).
%
%   The form was selected rather than assumed. Every subset of
%   {log gridSize, log K, log N, log N_u, log P(r), log B(r), isPer} was
%   fitted as a log-linear model of log(t_grid / t_spectral) and scored
%   by BIC and by cross-validated regret over 40 random halves. No
%   fitted subset beat this form, whose exponents (1, -2, -2) come from
%   the cost algebra rather than estimation, and BIC's own optimum
%   decides worse than most of the family -- likelihood weights cells
%   far from the boundary, where the decision is easy.
%
%   Constants match Python cosine._SPECTRAL_IP_* exactly, so both
%   languages decline on the same shapes.
%
%   Constants match Python cosine._SPECTRAL_IP_* exactly, so both
%   languages decline on the same shapes: route parity here is a
%   correctness matter, not merely a performance one, because the two
%   routes must return the same measure.
%
%   Mirror of Python cosine._spectral_rel_inner_matrix.

    MODE_SIGMAS = 8.6;      % _SPECTRAL_IP_MODE_SIGMAS
    MAX_POINTS  = 4e6;      % _SPECTRAL_IP_MAX_POINTS
    COST_C      = 3160.0;   % _SPECTRAL_IP_COST_C
    ENV_FLOOR   = 1e-18;

    I = [];

    Px = double(Px);  Py = double(Py);
    Wx = double(Wx);  Wy = double(Wy);

    Kx = size(Px, 1);  Nx = size(Px, 2);
    Ny = size(Py, 2);

    % ---- Embedding period ------------------------------------------
    if isPer
        L = double(period);
    else
        [loX, hiX] = localSpan(Px, Wx);
        [loY, hiY] = localSpan(Py, Wy);
        L = (hiX - loX) + (hiY - loY) + 2 * (MODE_SIGMAS + 2) * sigma;
        if ~isfinite(L) || L <= 0
            return;
        end
    end

    dxi = 2 * pi / L;
    M   = ceil(MODE_SIGMAS / sqrt(2) * L / (2 * pi * sigma)) + 2;
    gridSize = (2 * M + 1)^(r - 1);
    if gridSize > MAX_POINTS
        return;
    end
    % Cost gate: the grid path pays K^2 per event pair, the branch pays
    % the mode grid. Decline where the mode grid is not repaid.
    nPairs = double(Nx) * double(Ny);
    if gridSize > COST_C * double(Kx)^2 * nPairs
        return;
    end

    % ---- Mode grid: r-1 free axes, last slot fixed by the constraint
    axis1 = (-M:M).';
    if r == 2
        xs = {axis1};
    else
        grids = cell(1, r - 1);
        [grids{:}] = ndgrid(axis1);
        xs = cell(1, r - 1);
        for ii = 1:r-1
            xs{ii} = grids{ii}(:);
        end
        clear grids;
    end
    last = zeros(size(xs{1}));
    for ii = 1:r-1
        last = last + xs{ii};
    end
    xs{r} = -last;

    a = (dxi * sigma)^2;
    quad = zeros(size(xs{1}));
    for ii = 1:r
        quad = quad + double(xs{ii}).^2;
    end
    env = exp(-a * quad);

    keep = env > ENV_FLOOR;
    for ii = 1:r
        xs{ii} = xs{ii}(keep);
    end
    env = env(keep);
    if isempty(env)
        I = zeros(Nx, Ny);
        return;
    end

    % ---- Grid-side Hermitian halving --------------------------------
    % Lex-positivity via a scalar key with key(-m) = -key(m); base
    % 2M+1 keeps the key in comfortable integer range across the
    % regime the branch operates in.
    base = 2 * M + 1;
    key = zeros(size(xs{1}));
    for ii = 1:r-1
        key = key + double(xs{ii}) * base^(ii - 1);
    end
    isDC  = (key == 0);
    keepH = (key > 0) | isDC;
    for ii = 1:r
        xs{ii} = xs{ii}(keepH);
    end
    dcH = isDC(keepH);
    env = env(keepH);
    env(~dcH) = env(~dcH) * 2;
    nPts = numel(env);

    % ---- Per-event spectra ------------------------------------------
    Wax        = r * M;
    axModesPos = dxi * (0:Wax).';
    partitions = mobius.getSetPartitionsWithMobius(r);

    SX = localSpectra(Px, Wx, xs, r, nPts, Wax, axModesPos, partitions);
    SY = localSpectra(Py, Wy, xs, r, nPts, Wax, axModesPos, partitions);

    C_r = r * (sigma^2 * dxi)^(r - 1);
    I = C_r * real((SX .* env.') * SY');
end


function [lo, hi] = localSpan(P_, W_)
%LOCALSPAN  Min and max over slots carrying nonzero weight.
    live = abs(W_) > 0;
    if ~any(live(:))
        lo = 0; hi = 0;
        return;
    end
    vals = P_(live);
    lo = min(vals(:));
    hi = max(vals(:));
end


function S = localSpectra(P_, W_, xs, r, nPts, Wax, axModesPos, partitions)
%LOCALSPECTRA  (N, nPts) complex spectra, one row per event.
    N = size(P_, 2);
    S = zeros(N, nPts);
    S = complex(S, 0);
    for n = 1:N
        Pn = P_(:, n);
        Wn = W_(:, n);
        % Phase-side Hermitian: W_ is real, so A_m(-eta) = conj(A_m(eta)).
        % Evaluate phases on non-negative modes only and assemble the
        % full table by reflecting the positive half and conjugating.
        phasePos = exp(-1i * (axModesPos * Pn.'));      % (Wax+1, K)
        A = cell(1, r);
        for m = 1:r
            AposM = phasePos * (Wn.^m);                 % (Wax+1, 1)
            Am = zeros(2 * Wax + 1, 1);
            Am = complex(Am, 0);
            % Index Wax+1 .. 2*Wax+1 carries modes 0 .. Wax;
            % index 1 .. Wax carries modes -Wax .. -1 by conjugation.
            Am(Wax + 1 : end) = AposM;
            Am(1 : Wax)       = flip(conj(AposM(2 : Wax + 1)));
            A{m} = Am;
        end
        tot = zeros(nPts, 1);
        tot = complex(tot, 0);
        for pp = 1:numel(partitions)
            blocks = partitions(pp).blocks;
            term = complex(repmat(double(partitions(pp).mu), nPts, 1), 0);
            for bb = 1:numel(blocks)
                B = blocks{bb};
                eta = xs{B(1)};
                for kk = 2:numel(B)
                    eta = eta + xs{B(kk)};
                end
                term = term .* A{numel(B)}(eta + Wax + 1);
            end
            tot = tot + term;
        end
        S(n, :) = tot.';
    end
end
