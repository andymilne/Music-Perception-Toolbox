function theta = wrappedGaussian1d(d, sigma, period, ...
                                    truncationSigmas, exponentDenominator)
%WRAPPEDGAUSSIAN1D  Wrapped Gaussian with image-sum vs Fourier dispatch.
%
%   THETA = INTERNAL.WRAPPEDGAUSSIAN1D(D, SIGMA, PERIOD, TRUNCATIONSIGMAS,
%   EXPONENTDENOMINATOR) evaluates the 1-D wrapped Gaussian
%
%       theta(d) = sum_{n in Z} exp(-(d + n P)^2 / (e_d sigma^2))
%
%   at each entry of D and returns a same-shape array of theta values.
%   EXPONENTDENOMINATOR is 4 for the overlap-kernel convention (inner
%   products) and 2 for the density-kernel convention (density
%   evaluation).
%
%   The abs-per full-image r-tuple kernel is prod_a theta(d_a) --- Q
%   factors across slots in absolute mode, so the product-of-theta form
%   is the cheap representation of the all-image kernel.
%
%   Two exact representations of the theta function converge at
%   reciprocal rates as sigma/period varies:
%
%   - Image-sum: truncate the lattice at |n| <= L. Cheap when sigma/P
%     is small (few images have mass). Each term is one exp.
%   - Fourier (Poisson-summed):
%       theta(d) = (sqrt(e_d pi) sigma / P) *
%                  (1 + 2 * sum_m exp(-e_d pi^2 sigma^2 m^2 / P^2)
%                             * cos(2 pi m d / P))
%     truncated at |m| <= M. Cheap when sigma/P is large (envelope
%     drops quickly).
%
%   The 1-D crossover between image-sum and Fourier sits at
%   sigma/P ~ 0.20 for the density convention and sigma/P ~ 0.24 for
%   the overlap convention. Below the crossover image-sum wins;
%   above it, Fourier wins. Both compute the same wrapped Gaussian, so
%   the routing is transparent to callers.
%
%   Nearest-image reduction is applied before image-sum so the
%   truncation bound is correct for any input. Fourier is periodic and
%   needs no reduction.
%
%   Mirror of Python mpt._wrapped_kernel.wrapped_gaussian_1d.
%
%   Implementation note. Two MATLAB semantics require care.
%
%   (1) Trailing-singleton stripping: ``ndims`` and ``size`` strip
%       trailing 1s, so a reshape target that ends in ``, 1]`` is
%       reported with a lower dimension count than requested. The
%       image / Fourier axis is captured as ``ndims(d) + 1`` up front
%       rather than read back from a reshaped array.
%
%   (2) Broadcasting is position-based, not right-aligned: MATLAB pads
%       shorter operands with trailing singletons before matching
%       dimensions position-by-position. A row vector reshaped to
%       ``(1, M)`` pairs with a 4-D array's dimension 2, not its last
%       dimension. To broadcast against ``d`` along the intended
%       reduce axis, the Fourier envelope and mode indices must share
%       the same trailing-``M`` shape ``[ones(1, ndims(d)), M]``.

    nDimsIn = ndims(d);
    reduceAxis = nDimsIn + 1;

    if internal.wrappedKernelPreferFourier(sigma, period, ...
                                            truncationSigmas, ...
                                            exponentDenominator)
        M = internal.wrappedKernelFourierCount(sigma, period, ...
                                                truncationSigmas, ...
                                                exponentDenominator);
        alpha = pi * pi * exponentDenominator * sigma * sigma ...
                / (period * period);
        twoPiOverP = 2 * pi / period;
        prefactor = sqrt(pi * exponentDenominator) * sigma / period;
        modeShape = ones(1, nDimsIn);
        modeShape(reduceAxis) = M;
        mShaped = reshape((1:M), modeShape);
        envShaped = reshape(exp(-alpha * (1:M) .* (1:M)), modeShape);
        phase = twoPiOverP * d .* mShaped;
        contribSum = sum(envShaped .* cos(phase), reduceAxis);
        theta = prefactor * (1 + 2 * contribSum);
        return
    end

    L = internal.wrappedKernelImageCount(sigma, period, ...
                                         truncationSigmas, ...
                                         exponentDenominator);
    dRed = d - period * floor(d / period + 0.5);
    inv2s2 = 1 / (exponentDenominator * sigma * sigma);
    if L == 0
        theta = exp(-dRed .* dRed * inv2s2);
        return
    end
    imageShape = ones(1, nDimsIn);
    imageShape(reduceAxis) = 2 * L + 1;
    nShaped = reshape((-L:L), imageShape);
    dShift = dRed + period * nShaped;
    theta = sum(exp(-dShift .* dShift * inv2s2), reduceAxis);
end
