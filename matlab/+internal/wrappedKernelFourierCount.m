function M = wrappedKernelFourierCount(sigma, period, ...
                                        truncationSigmas, ...
                                        exponentDenominator)
%WRAPPEDKERNELFOURIERCOUNT  Fourier-mode truncation for the wrapped Gaussian.
%
%   M = INTERNAL.WRAPPEDKERNELFOURIERCOUNT(SIGMA, PERIOD, TRUNCATIONSIGMAS,
%   EXPONENTDENOMINATOR) returns the number of Fourier modes per side
%   for the Poisson-summed form of the 1-D wrapped Gaussian to reach
%   the caller's accuracy floor.
%
%   The envelope at mode m is exp(-alpha m^2) with
%   alpha = pi^2 e_d sigma^2 / P^2. Requiring exp(-alpha M^2) < tol:
%
%       M > sqrt(-log tol / alpha)
%         = (P / (pi sigma)) sqrt(-log tol / e_d).
%
%   EXPONENTDENOMINATOR is 4 for overlap-kernel and 2 for density-kernel
%   conventions.
%
%   Mirror of Python mpt._wrapped_kernel._fourier_count_M.
    M = 1;
    if ~isfinite(sigma) || ~isfinite(period) || period <= 0 || sigma <= 0
        return
    end
    k = internal.accuracyFloor('resolve', truncationSigmas);
    if k == internal.accuracyFloor('sigmas')
        tol = internal.accuracyFloor('eps');
    else
        tol = exp(-0.5 * k * k);
    end
    if ~(tol > 0 && tol < 1)
        return
    end
    alpha = pi * pi * exponentDenominator * sigma * sigma / (period * period);
    if ~(alpha > 0)
        return
    end
    val = sqrt(-log(tol) / alpha);
    if ~isfinite(val) || val < 1
        return
    end
    M = ceil(val);
end
