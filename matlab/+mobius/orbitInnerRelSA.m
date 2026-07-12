function [val, ratio] = orbitInnerRelSA(p_a, w_a, p_b, w_b, sigma, r, ...
                                          isPer, period, opts)
%MOBIUS.ORBITINNERRELSA  <T_A, T_B>_rel via translation-grid integration.
%
%   [VAL, RATIO] = MOBIUS.ORBITINNERRELSA(P_A, W_A, P_B, W_B, SIGMA, R,
%                                          IS_PER, PERIOD)
%   computes the relative-mode SA inner product by marginalising a
%   translation u over [0, period) (periodic) or a Gaussian-supported
%   window around the alignment of A and B (non-periodic), and
%   integrating the orbit-evaluated kernel against u via
%   MOBIUS.INNERPRODUCTORBITGRID.
%
%   In the periodic case the node count comes from
%   INTERNAL.AUTONTAUDEFAULT, the single shared source used by the flat
%   and nested relative-periodic paths so the same level returns the same
%   value whichever path computes it. In the non-periodic case the line
%   grid has samplesPerSigma = 10 points per sigma and the truncation
%   extends 8*sigma beyond the natural overlap.
%
%   Inputs:
%     P_A, W_A   (n_a, 1) source positions and weights for density A.
%     P_B, W_B   (n_b, 1) source positions and weights for density B.
%     SIGMA      positive scalar; Gaussian smoothing in pitch space.
%     R          integer >= 2.
%     IS_PER     logical.
%     PERIOD     positive scalar; periodic mode only.
%
%   Name-Value options:
%     truncationSigmas  (1,1) double, default mptDefaults('truncationSigmas').
%                       Kernel entries whose squared distance exceeds
%                       the truncation cutoff are zeroed without
%                       evaluating exp().
%
%   Outputs:
%     VAL        scalar inner product.
%     RATIO      mass-aware global cancellation diagnostic in (0, 1]:
%                |sum(F)| / sum(termMass), the magnitude of the
%                integrated alternating sum relative to the integral of
%                the worst-magnitude partition term. This bounds the
%                relative error of the integral (absolute error ~
%                eps * denominator * du), which is the quantity the
%                acceptance threshold protects. A pointwise worst case
%                over the u-grid is the wrong aggregate here: at sharp
%                sigma, translation bands where only one event pair
%                falls inside the kernel support have a true integrand
%                of exactly zero produced by exact cancellation of
%                nonzero orbit terms, so a pointwise ratio at such a
%                band is ~0 while the band contributes nothing to the
%                integral.
%
%   See also MOBIUS.INNERPRODUCTORBITGRID, MOBIUS.ORBITINNERABSSA,
%            INTERNAL.TRUNCKERNELEXP.

    arguments
        p_a double
        w_a double
        p_b double
        w_b double
        sigma (1,1) double {mustBePositive}
        r (1,1) double {mustBeInteger, mustBePositive}
        isPer (1,1) logical
        period (1,1) double
        opts.truncationSigmas (1,1) double = mptDefaults('truncationSigmas')
    end

    samplesPerSigma = 10;
    p_a = p_a(:); p_b = p_b(:);
    n_a = numel(p_a); n_b = numel(p_b);

    if isPer
        N_u = internal.autoNtauDefault(period, sigma);
        u_grid = (0:N_u-1)' * (period / N_u);
        du = period / N_u;
    else
        u_min = min(p_b) - max(p_a) - 8 * sigma;
        u_max = max(p_b) - min(p_a) + 8 * sigma;
        N_u = max(64, ceil(max(u_max - u_min, 1.0) / sigma * samplesPerSigma));
        u_grid = linspace(u_min, u_max, N_u)';
    end

    % Build N_u x n_a x n_b stack of differences and Gaussian kernels.
    diffs = reshape(u_grid, N_u, 1, 1) ...
          + reshape(p_a, 1, n_a, 1) ...
          - reshape(p_b, 1, 1, n_b);
    if isPer
        diffs = diffs - period * floor(diffs / period + 0.5);
    end
    K_u = internal.truncKernelExp(diffs.^2, sigma, opts.truncationSigmas);

    [F, ~, termMass] = mobius.innerProductOrbitGrid(K_u, w_a(:), w_b(:), r, ...
        'returnCancellationRatio', true);

    if isPer
        integral = sum(F) * du;
    else
        integral = trapz(u_grid, F);
    end
    c = sigma * sqrt(2 * pi / r);
    val = (sigma * sqrt(pi))^r * integral / c^2;
    denom = sum(termMass);
    if denom > 0
        ratio = abs(sum(F)) / denom;
    else
        ratio = 1;
    end
end
