function tf = wrappedKernelPreferFourier(sigma, period, ...
                                          truncationSigmas, ...
                                          exponentDenominator)
%WRAPPEDKERNELPREFERFOURIER  True when Fourier's grid is narrower than image-sum's.
%
%   Compares the two truncation widths at the caller's accuracy floor.
%   Each per-component 1-D kernel evaluation costs 2 L + 1 terms for
%   image-sum and M for Fourier (plus the constant DC term). Prefer
%   Fourier when its grid is strictly narrower.
%
%   Crossover falls at sigma/P ~ 0.24 for the overlap convention
%   (exponentDenominator = 4) and sigma/P ~ 0.20 for the density
%   convention (exponentDenominator = 2), tolerance-independent.
%
%   Mirror of Python mpt._wrapped_kernel._prefer_fourier.
    L = internal.wrappedKernelImageCount(sigma, period, ...
                                         truncationSigmas, ...
                                         exponentDenominator);
    M = internal.wrappedKernelFourierCount(sigma, period, ...
                                            truncationSigmas, ...
                                            exponentDenominator);
    tf = M < (2 * L + 1);
end
