function [val, ratio] = orbitInnerAbsSingleMultiset(p_a, w_a, p_b, w_b, sigma, r, ...
                                          isPer, period, opts)
%MOBIUS.ORBITINNERABSSINGLEMULTISET  <T_A, T_B>_abs via mobius.innerProductOrbit.
%
%   [VAL, RATIO] = MOBIUS.ORBITINNERABSSINGLEMULTISET(P_A, W_A, P_B, W_B, SIGMA, R,
%                                          IS_PER, PERIOD)
%   computes the absolute-mode single-multiset inner product between two
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
%                       evaluating exp() (single-image mode only;
%                       ignored in full-image mode where the wrapped
%                       Gaussian is evaluated directly).
%     wrap              char, default 'full-image'. Selects the abs-per
%                       measure. 'full-image' (default) uses the torus
%                       (all-image) 1-D wrapped Gaussian per coordinate,
%                       delivered by INTERNAL.WRAPPEDGAUSSIAN1D in
%                       overlap convention. The r-tuple full-image
%                       kernel factors as prod_a theta(d_a), delivered
%                       by the orbit reduction over the 1-D theta
%                       values. 'single-image' opts into the nearest-
%                       image kernel unchanged. Ignored when
%                       IS_PER = false.
%
%   Outputs:
%     VAL        scalar inner product, including the (sigma * sqrt(pi))^r
%                prefactor. The prefactor is the same in both measures
%                because the 1-D wrapped Gaussian's integral over the
%                circle equals the single Gaussian's integral over the
%                line, so the r-tuple normalisation is identical.
%     RATIO      worst per-batch cancellation ratio in (0, 1]; values
%                near 1 mean no cancellation, low values mean digits
%                lost in the orbit alternating sum.
%
%   Mirror of Python cosine._orbit_inner_abs.
%
%   See also MOBIUS.INNERPRODUCTORBIT, MOBIUS.ORBITINNERRELSINGLEMULTISET,
%            INTERNAL.TRUNCKERNELEXP, INTERNAL.WRAPPEDGAUSSIAN1D.

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
        opts.wrap (1,:) char ...
            {mustBeMember(opts.wrap, {'full-image', 'single-image'})} ...
            = 'full-image'
    end

    p_a = p_a(:); p_b = p_b(:);
    diffs = p_a - p_b.';                              % n_a x n_b
    if isPer && strcmp(opts.wrap, 'full-image')
        % Overlap-kernel convention (exponent_denominator = 4). The
        % (sigma sqrt(pi))^r prefactor stays: the 1-D wrapped Gaussian
        % integrates to sigma sqrt(pi) over the circle, matching the
        % line integral of the single Gaussian, so the r-tuple
        % normalisation is identical to single-image.
        K = internal.wrappedGaussian1d(diffs, sigma, period, ...
                                        opts.truncationSigmas, 4);
    else
        if isPer
            diffs = diffs - period * floor(diffs / period + 0.5);
        end
        K = internal.truncKernelExp(diffs.^2, sigma, opts.truncationSigmas);
    end
    [val, ratio] = mobius.innerProductOrbit(K, w_a(:), w_b(:), r, ...
        'prefactor', (sigma * sqrt(pi))^r, ...
        'returnCancellationRatio', true);
end
