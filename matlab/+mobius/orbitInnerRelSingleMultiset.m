function [val, ratio] = orbitInnerRelSingleMultiset(p_a, w_a, p_b, w_b, sigma, r, ...
                                          isPer, period, opts)
%MOBIUS.ORBITINNERRELSINGLEMULTISET  <T_A, T_B>_rel: single-multiset wrapper.
%
%   [VAL, RATIO] = MOBIUS.ORBITINNERRELSINGLEMULTISET(P_A, W_A, P_B, W_B, SIGMA, R,
%                                          IS_PER, PERIOD)
%   computes the relative-mode inner product of two single-multiset densities
%   as the N = 1 specialisation of MOBIUS.RELINNERBATCHED, which is
%   the single relative-mode evaluator. All grid, slab, truncation,
%   and cancellation-diagnostic conventions are the core's: the shared
%   [0, P) grid in periodic mode; in non-periodic mode a window of
%   width spread_A + spread_B + 2*INTERNAL.RELWINDOWMARGIN*sigma
%   centred on the weighted-mean offset, evaluated by plain Riemann
%   sum (the margin places every kernel entry strictly outside the
%   truncation radius at the window edges, so for finite truncation
%   the endpoint integrand is exactly zero and the Riemann sum equals
%   the trapezoidal rule exactly).
%
%   RATIO is the mass-aware cancellation diagnostic
%   |sum_u F_u| / sum_u max_orb(|term_orb_u|), computed only when
%   requested (nargout > 1).
%
%   Name-value options:
%     'truncationSigmas'  kernel truncation (default mptDefaults)

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

    Pa = p_a(:);  Wa = w_a(:);
    Pb = p_b(:);  Wb = w_b(:);
    if nargout > 1
        [I, ratio] = mobius.relInnerBatched(Pa, Wa, Pb, Wb, sigma, r, ...
            isPer, period, 'truncationSigmas', opts.truncationSigmas);
    else
        I = mobius.relInnerBatched(Pa, Wa, Pb, Wb, sigma, r, ...
            isPer, period, 'truncationSigmas', opts.truncationSigmas);
    end
    val = I(1, 1);
end
