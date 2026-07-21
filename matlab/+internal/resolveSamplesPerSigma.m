function spp = resolveSamplesPerSigma(samplesPerSigma, r, truncationSigmas)
%RESOLVESAMPLESPERSIGMA  Accuracy-tied relative-mode u-grid density.
%
%   spp = internal.resolveSamplesPerSigma(samplesPerSigma, r, truncationSigmas)
%
%   An explicit positive integer passes through unchanged. Empty ([])
%   derives the count from the truncation floor, so the single toolbox
%   accuracy knob (truncationSigmas) governs the translation-integral
%   quadrature as well as the kernel floor. Mirrors the Python
%   mpt._defaults.resolve_samples_per_sigma.
%
%   Derivation. The relative-mode value is a 1-D integral over the
%   translation u. Every Mobius partition's integrand is a Gaussian
%   mixture in u of width exactly sigma/sqrt(r): each block of size m
%   contributes a factor of width sigma/sqrt(m), and precisions add
%   across the product (sum m = r). The trapezoidal rule on a
%   Gaussian-decaying (or circle-periodic) integrand is spectrally
%   accurate; by Poisson summation the error is the aliased spectrum,
%   ~exp(-2*pi^2*(w/h)^2) for feature width w and step h. With
%   h = sigma/spp and w = sigma/sqrt(r) the error is
%   ~exp(-2*pi^2*spp^2/r), so accuracy eps = exp(-k^2/2) (k the
%   resolved truncation width) needs
%
%       spp >= sqrt(r * log(1/eps)) / (pi * sqrt(2)) = k*sqrt(r)/(2*pi).
%
%   One extra sample absorbs the mixture-mass prefactor (measured to be
%   within ~2x of the bound), and the floor of 2 guards degenerate
%   inputs. At the factory 6-sigma floor this resolves to 3 for r <= 5
%   and 4 up to r = 12; at the tightest accuracy (Inf, the 1e-12 floor)
%   to 3-5 over the same range.

    if nargin < 3
        truncationSigmas = [];
    end
    if ~isempty(samplesPerSigma)
        spp = double(samplesPerSigma);
        return;
    end
    k = internal.accuracyFloor('resolve', truncationSigmas);
    rEff = max(double(r), 1);
    spp = max(2, ceil(k * sqrt(rEff) / (2 * pi)) + 1);
end
