function I = spectralRelInnerMatrix(Px, Wx, Py, Wy, sigma, r, isPer, period, forceBranch)
%MOBIUS.SPECTRALRELINNERMATRIX  Relative inner matrix by the spectral form.
%
%   I = MOBIUS.SPECTRALRELINNERMATRIX(PX, WX, PY, WY, SIGMA, R, ISPER,
%   PERIOD) returns the (N_x, N_y) per-attribute inner matrix for a
%   relative attribute, or [] when the branch declines (the caller then
%   falls through to the translation-grid contraction).
%
%   ...(..., FORCEBRANCH) with FORCEBRANCH true skips the cost gate (but
%   not the MAXPOINTS memory guard), so the branch runs wherever it is
%   representable. Used only by the parity test, which must compare the
%   spectral VALUE against the Python reference on cells the two
%   languages' cost gates route differently. The memory guard is never
%   bypassed.
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
%   COST_C is PER-LANGUAGE in principle -- both routes compute the same
%   full-image measure, so which one runs affects only time, and the
%   multiplier absorbs BLAS, JIT and layout differences -- but the
%   September 2026 refit landed on the Python value. The form is shared:
%   the same K^2 * N_x * N_y product and the same exponents (1, -2, -2),
%   pinned by the cost algebra and selected on the Python data against
%   every subset of {log gridSize, log K, log N, log N_u, log P(r),
%   log B(r), isPer} by BIC and by cross-validated regret.
%
%   1100 is fitted by routing regret against an oracle on measured MATLAB
%   wall times: the bench_spectral_ip_gate grid (r = 2..4, K = 4..30,
%   N = 1..16, both periodic modes, 480 cells) plus bench_spectral_ip_gate_ext
%   (r = 3, sigma = 10 cents over three octaves, K = 24..140, N = 1..2),
%   with the gate bypassed on the spectral arm. The optimum is C = 1096
%   (geometric-mean regret 1.029 of oracle, worst 6.5x); 282, the
%   previous value, scores 1.048 with a worst case of 6.9x and misroutes
%   the extension sweep badly (1.75 of oracle there: at K = 80 it runs
%   the grid at 298 ms where the branch takes 43 ms, and it declines the
%   branch until K = 86 at that shape). The earlier 282 fit was made on a
%   harness that enabled the branch without bypassing the gate, so every
%   cell the gate declined timed the grid twice and read as a tie; that
%   is why the constant sat at the bottom of an apparent plateau.
%
%   The residual is at r = 4: the branch loses on every r = 4 cell with
%   K <= 8 and a large mode grid (worst 811 ms against 125 ms at K = 4,
%   N = 16, sigma/P = 0.05 non-periodic), and the r-wise optima are
%   ~1600 at r = 3 and ~220 at r = 4 -- the branch's cost grows with r
%   faster than gridSize alone carries. A per-r constant ({1600, 1600,
%   250}) scores 1.018 with a worst case of 3.4x and cross-validates
%   marginally better (1.032 against 1.035 over random halves). It was
%   not adopted, to keep one form and one constant across the languages;
%   the r = 4 loss is bounded and confined to tiny K.
%
%   MODE_SIGMAS and MAX_POINTS match Python _SPECTRAL_IP_MODE_SIGMAS and
%   _SPECTRAL_IP_MAX_POINTS exactly (they size the mode grid, which sets
%   the value's accuracy). The parity test checks value agreement only,
%   on cells both languages fire.
%
%   Mirror of Python cosine._spectral_rel_inner_matrix.

    if nargin < 9 || isempty(forceBranch)
        forceBranch = false;
    end
    MODE_SIGMAS = 8.6;      % _SPECTRAL_IP_MODE_SIGMAS
    MAX_POINTS  = 4e6;      % _SPECTRAL_IP_MAX_POINTS
    COST_C      = 1100.0;   % refit Sep 2026; see the note in the header
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
    % the mode grid. Decline where the mode grid is not repaid. The
    % parity test passes forceBranch to skip this (never the memory
    % guard above) so it can compare values on cells the two languages
    % route differently.
    if ~forceBranch
        nPairs = double(Nx) * double(Ny);
        if gridSize > COST_C * double(Kx)^2 * nPairs
            return;
        end
    end

    % ---- Mode grid: r-1 free axes, last coordinate fixed by the constraint
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

    plan = localBuildPlan(xs, r, Wax, partitions);
    SX = localSpectra(Px, Wx, r, nPts, Wax, axModesPos, plan);
    % Self inner product: when the two event sets are identical their
    % spectra are identical, so compute them once. The cosine forms three
    % inner matrices per call, two of which -- <X, X> and <Y, Y> -- are
    % self inner products, so this halves their spectra work. MATLAB has
    % no identity test, so compare by value behind a cheap size guard.
    if isequal(size(Px), size(Py)) && isequal(Px, Py) ...
            && isequal(size(Wx), size(Wy)) && isequal(Wx, Wy)
        SY = SX;
    else
        SY = localSpectra(Py, Wy, r, nPts, Wax, axModesPos, plan);
    end

    C_r = r * (sigma^2 * dxi)^(r - 1);
    I = C_r * real((SX .* env.') * SY');
end


function plan = localBuildPlan(xs, r, Wax, partitions)
%LOCALBUILDPLAN  Gather indices and block sizes, once per call.
%   The indices eta + Wax + 1 depend only on the mode grid xs, not on the
%   events, so they are built once here rather than rebuilt inside the
%   per-event loop of localSpectra. At r = 4 rebuilding them cost several
%   times a single assembly, so this hoist is the dominant saving on the
%   branch's most expensive shapes. Returns a struct array with fields
%   mu, sizes (1 x nBlocks), and idx (nPts x nBlocks) of 1-based gather
%   indices into the A tables.
    nP = numel(partitions);
    plan(nP) = struct('mu', 0, 'sizes', [], 'idx', []);
    for pp = 1:nP
        blocks = partitions(pp).blocks;
        nB = numel(blocks);
        sizes = zeros(1, nB);
        idx = zeros(numel(xs{1}), nB);
        for bb = 1:nB
            B = blocks{bb};
            eta = xs{B(1)};
            for kk = 2:numel(B)
                eta = eta + xs{B(kk)};
            end
            sizes(bb) = numel(B);
            idx(:, bb) = eta + Wax + 1;
        end
        plan(pp).mu = double(partitions(pp).mu);
        plan(pp).sizes = sizes;
        plan(pp).idx = idx;
    end
end


function [lo, hi] = localSpan(P_, W_)
%LOCALSPAN  Min and max over values carrying nonzero weight.
    live = abs(W_) > 0;
    if ~any(live(:))
        lo = 0; hi = 0;
        return;
    end
    vals = P_(live);
    lo = min(vals(:));
    hi = max(vals(:));
end


function S = localSpectra(P_, W_, r, nPts, Wax, axModesPos, plan)
%LOCALSPECTRA  (N, nPts) complex spectra, one row per event.
%   plan carries the gather indices and block sizes, built once by
%   localBuildPlan and shared across events and across the three inner
%   products, so this loop does no index arithmetic.
    N = size(P_, 2);
    S = complex(zeros(N, nPts), 0);
    nP = numel(plan);
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
            Am = complex(zeros(2 * Wax + 1, 1), 0);
            % Index Wax+1 .. 2*Wax+1 carries modes 0 .. Wax;
            % index 1 .. Wax carries modes -Wax .. -1 by conjugation.
            Am(Wax + 1 : end) = AposM;
            Am(1 : Wax)       = flip(conj(AposM(2 : Wax + 1)));
            A{m} = Am;
        end
        tot = complex(zeros(nPts, 1), 0);
        for pp = 1:nP
            sizes = plan(pp).sizes;
            idx   = plan(pp).idx;
            term = complex(repmat(plan(pp).mu, nPts, 1), 0);
            for bb = 1:numel(sizes)
                term = term .* A{sizes(bb)}(idx(:, bb));
            end
            tot = tot + term;
        end
        S(n, :) = tot.';
    end
end
