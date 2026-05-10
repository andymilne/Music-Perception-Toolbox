function [vals, ratios] = evalOrbitRel(p, w, sigma, r, x_rel, opts)
%MOBIUS.EVALORBITREL  Möbius point evaluator for SA relative-mode tensor.
%
%   VALS = MOBIUS.EVALORBITREL(P, W, SIGMA, R, X_REL) computes T_rel
%   at each column of X_REL via u-grid quadrature wrapping
%   MOBIUS.EVALORBITABS:
%
%     T_rel(Δ) = (1/Z_t) * ∫ T_abs(u, u+Δ_1, ..., u+Δ_{r-1}) du
%
%   where Z_t = sigma * sqrt(2*pi/r) is the translation-mode normaliser.
%   The integration grid mirrors the tensor module's _orbit_inner_rel:
%   periodic uses [0, P) sampled at SAMPLES_PER_SIGMA points per sigma;
%   non-periodic uses a Gaussian-supported window extending 8*sigma
%   beyond the alignment of source positions and query trajectory.
%
%   VALS = MOBIUS.EVALORBITREL(..., 'is_per', true, 'period', P) selects
%   periodic mode.
%
%   VALS = MOBIUS.EVALORBITREL(..., 'samplesPerSigma', N) overrides the
%   default u-grid density of 10 points per sigma. Reduce to 5 for
%   speed at the cost of ~1e-9 relative precision.
%
%   [VALS, RATIOS] = MOBIUS.EVALORBITREL(..., 'returnCancellationRatio', true)
%   additionally returns per-query worst-case cancellation ratios
%   (minimum across the u-grid for each query). Worst-case is the right
%   summary statistic since a single bad u-point corrupts the integral.
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
%
%   For R=1 the relative space is 0-dim and T_rel is a constant; by
%   convention this returns sum(W) at each query. Tests should not
%   exercise this case.
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

    % Evaluate T_abs at each u-grid point and accumulate.
    F = zeros(N_u, n_q);
    if opts.returnCancellationRatio
        R = ones(N_u, n_q);
    end
    for j = 1:N_u
        u = u_grid(j);
        x_full = zeros(r, n_q);
        x_full(1, :) = u;
        x_full(2:end, :) = u + x_rel;
        if opts.returnCancellationRatio
            [vals_j, ratios_j] = mobius.evalOrbitAbs(p, w, sigma, r, x_full, ...
                'is_per', opts.is_per, 'period', opts.period, ...
                'returnCancellationRatio', true);
            F(j, :) = vals_j(:)';
            R(j, :) = ratios_j(:)';
        else
            vals_j = mobius.evalOrbitAbs(p, w, sigma, r, x_full, ...
                'is_per', opts.is_per, 'period', opts.period);
            F(j, :) = vals_j(:)';
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
