function L = wrappedKernelImageCount(sigma, period, ...
                                      truncationSigmas, ...
                                      exponentDenominator)
%WRAPPEDKERNELIMAGECOUNT  Image-sum truncation for the 1-D wrapped Gaussian.
%
%   L = INTERNAL.WRAPPEDKERNELIMAGECOUNT(SIGMA, PERIOD, TRUNCATIONSIGMAS,
%   EXPONENTDENOMINATOR) returns the number of periodic images per side
%   for the truncated lattice sum
%
%       theta_L(d) = sum_{n = -L..L} exp(-(d + n P)^2 / (e_d sigma^2))
%
%   to reach the caller's accuracy floor. After nearest-image reduction
%   the first-omitted image at |d + n P| >= (L + 1/2) P is bounded by
%
%       exp(-((L + 1/2) P)^2 / (e_d sigma^2)) < tol,
%
%   solving for L:
%
%       L > (sigma / P) sqrt(-e_d log tol) - 1/2.
%
%   EXPONENTDENOMINATOR is 4 for overlap-kernel and 2 for density-kernel
%   conventions. Returns 0 whenever the nearest image alone already
%   meets the floor.
%
%   Mirror of Python mpt._wrapped_kernel._image_count_L.
    L = 0;
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
    rhs = (sigma / period) * sqrt(-log(tol) * exponentDenominator);
    if ~isfinite(rhs) || rhs - 0.5 <= 0
        return
    end
    L = ceil(rhs - 0.5);
end
