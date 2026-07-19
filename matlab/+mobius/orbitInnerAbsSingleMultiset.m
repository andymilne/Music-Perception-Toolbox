function [val, ratio] = orbitInnerAbsSingleMultiset(p_a, w_a, p_b, w_b, sigma, r, ...
                                          isPer, period, opts)
%MOBIUS.ORBITINNERABSSINGLEMULTISET  <T_A, T_B>_abs via mobius.innerProductOrbit.
%
%   [VAL, RATIO] = MOBIUS.ORBITINNERABSSINGLEMULTISET(P_A, W_A, P_B, W_B, SIGMA, R,
%                                          IS_PER, PERIOD)
%   computes the absolute-mode SA inner product between two
%   weighted-multiset densities at tensor order R via the orbit-Möbius
%   identity, by building the (n_a, n_b) Gaussian-kernel matrix and
%   delegating to MOBIUS.INNERPRODUCTORBIT.
%
%   Inputs:
%     P_A, W_A   (n_a, 1) source positions and weights for density A.
%     P_B, W_B   (n_b, 1) source positions and weights for density B.
%     SIGMA      positive scalar; Gaussian smoothing in pitch space.
%     R          integer >= 2 (R=1 reduces to a direct kernel sum;
%                callers handle that case directly).
%     IS_PER     logical; wrap differences modulo PERIOD when true.
%     PERIOD     positive scalar; periodic mode only.
%
%   Name-Value options:
%     truncationSigmas  (1,1) double, default mptDefaults('truncationSigmas').
%                       Kernel entries whose squared distance exceeds
%                       the truncation cutoff are zeroed without
%                       evaluating exp().
%
%   Outputs:
%     VAL        scalar inner product, including the (sigma * sqrt(pi))^r
%                prefactor.
%     RATIO      worst per-batch cancellation ratio in (0, 1]; values
%                near 1 mean no cancellation, low values mean digits
%                lost in the orbit alternating sum.
%
%   See also MOBIUS.INNERPRODUCTORBIT, MOBIUS.ORBITINNERRELSINGLEMULTISET,
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

    p_a = p_a(:); p_b = p_b(:);
    diffs = p_a - p_b.';                              % n_a x n_b
    if isPer
        diffs = diffs - period * floor(diffs / period + 0.5);
    end
    K = internal.truncKernelExp(diffs.^2, sigma, opts.truncationSigmas);
    [val, ratio] = mobius.innerProductOrbit(K, w_a(:), w_b(:), r, ...
        'prefactor', (sigma * sqrt(pi))^r, ...
        'returnCancellationRatio', true);
end
