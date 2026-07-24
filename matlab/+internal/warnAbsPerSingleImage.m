function warnAbsPerSingleImage(sigmaOverP)
%WARNABSPERSINGLEIMAGE  Warn that an absolute-periodic attribute uses the
%   single-image measure at a sigma/period where that departs from the
%   full-image measure.
%
%   INTERNAL.WARNABSPERSINGLEIMAGE(SIGMAOVERP) emits the toolbox-wide
%   warning that an absolute-periodic attribute wraps each difference to
%   its nearest image, which above sigma/period = 0.05 departs from the
%   full-image measure that sums the kernel over every periodic image.
%
%   Raised at density construction rather than at any one operation,
%   because the choice is a property of the density: evaluation, inner
%   product, and entropy all inherit it.
%
%   Unlike the relative-periodic case, absolute-periodic mode currently
%   offers no full-image route, so this warning names no alternative
%   method -- it reports a limitation rather than announcing a
%   substitution. The two measures agree below the threshold, so the
%   warning is silent in the range musical work normally occupies.
%
%   Threshold set at 0.05 on two independent grounds, whichever binds
%   first. Accuracy: below it the two measures agree to within the
%   toolbox's own floor (the cosine differs by 0 up to sigma/period =
%   0.03 and by 2.5e-13 at 0.05, against ~1e-12 at truncationSigmas =
%   inf and ~1.5e-8 at the default of 6), rising to 2.6e-6 at 0.08.
%   Positive definiteness: the single-image kernel's Fourier coefficients
%   on the circle are non-negative only up to about this point, going
%   negative above it (-4.9e-5 at 0.10, -1.4e-2 at 0.20), so the form it
%   induces is not an inner product and Cauchy-Schwarz fails -- cosines
%   of 1.07 at 0.20 and 1.12 at 0.30 are reachable with ordinary
%   non-negative weights.
%
%   Mirror of Python dispatch._warn_abs_per_single_image.
    warning('buildExpTens:absPerSingleImage', ...
        ['sigma/period = %.3f exceeds 0.05: this absolute-periodic ' ...
         'attribute uses the single-image (minimum-image) measure, ' ...
         'which above this sigma/period departs from the full-image ' ...
         'measure that sums the kernel over every periodic image (the ' ...
         'two agree below it). Everything computed from this density ' ...
         'inherits the choice, and no full-image route is available in ' ...
         'absolute-periodic mode at present. Above roughly ' ...
         'sigma/period = 0.15 the single-image kernel also stops being ' ...
         'positive definite, so a cosine similarity computed from it ' ...
         'is not bounded by 1. Reduce sigma relative to the period if ' ...
         'the measure matters at this scale.'], sigmaOverP);
end
