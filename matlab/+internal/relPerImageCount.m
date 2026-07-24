function n = relPerImageCount(sigma, period, truncationSigmas)
%RELPERIMAGECOUNT  Periodic images each side for the relative-periodic kernel.
%
%   N = INTERNAL.RELPERIMAGECOUNT(SIGMA, PERIOD, TRUNCATIONSIGMAS) returns
%   the number of periodic images needed on each side for the
%   relative-periodic kernel to reach the caller's own accuracy floor.
%
%   The relative-periodic inner product marginalises a rigid common
%   shift. Taking that average over a kernel that carries every periodic
%   image yields the lattice-sum (full-image) measure exactly; taking it
%   over a nearest-image kernel yields a different measure, which departs
%   from it as sigma/period grows. Summing images restores the identity,
%   and the count needed is set by the accuracy already being asked for
%   elsewhere rather than by a fixed constant.
%
%   After the nearest-image reduction the difference satisfies
%   |d| <= period/2, so the image at offset l is bounded by
%   exp(-((|l| - 1/2) period)^2 / (4 sigma^2)). Requiring the first
%   omitted image to fall below the kernel-value floor gives
%
%       n > 2 (sigma/period) sqrt(ln(1/tol)) - 1/2 .
%
%   Returns 0 whenever the nearest image alone already meets the floor,
%   which is the case throughout the range musical work normally occupies
%   (0 up to sigma/period ~ 0.05 at the default truncation, where the two
%   measures agree to 2.5e-13). In that regime the full-image kernel and
%   the nearest-image kernel are the same object and this costs nothing.
%
%   Mirror of Python cosine._rel_per_image_count.
    n = 0;
    if ~isfinite(sigma) || ~isfinite(period) || period <= 0
        return;
    end
    if nargin < 3
        truncationSigmas = [];
    end
    % Resolve through the shared accuracy-floor helper so None/[] takes
    % the default and Inf takes the finite accuracy-floor width, exactly
    % as Python's resolve_truncation_sigmas does. The 'sigmas' width has
    % 'eps' as its floor by construction, so return that directly rather
    % than through a log/exp round trip.
    k = internal.accuracyFloor('resolve', truncationSigmas);
    if k == internal.accuracyFloor('sigmas')
        tol = internal.accuracyFloor('eps');
    else
        tol = exp(-0.5 * k * k);
    end
    if ~(tol > 0 && tol < 1)
        return;
    end
    nRaw = 2 * (sigma / period) * sqrt(log(1 / tol)) - 0.5;
    if ~isfinite(nRaw) || nRaw <= 0
        return;
    end
    n = ceil(nRaw);
end
