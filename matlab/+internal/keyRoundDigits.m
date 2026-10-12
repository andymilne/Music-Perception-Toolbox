function digits = keyRoundDigits(sigma, r, truncationSigmas)
%KEYROUNDDIGITS Decimal places for canonical forms, set by the accuracy floor.
%   digits = internal.keyRoundDigits(sigma, r, truncationSigmas)
%
%   Rounding a canonical form moves each value by at most
%   delta = 0.5 * 10^-digits, so a tuple centre moves by at most
%   sqrt(r) * delta and a difference of two centres by twice that. A
%   Gaussian kernel of width sigma then changes by at most
%   sqrt(2) * exp(-1/2) * 2 * sqrt(r) * delta / sigma of its peak (the
%   largest slope of the pairwise inner-product kernel; the point kernel's
%   is smaller). digits is the fewest places that keep this within the
%   truncation floor exp(-k^2/2) (k = the resolved truncationSigmas;
%   internal.truncationFloor), so rounding perturbs no kernel contribution
%   by more than truncation may omit, and the accuracy bound
%   truncationSigmas states is kept. Rounding as coarsely as that allows
%   absorbs as much floating-point noise as possible, so equivalent
%   inputs computed by different arithmetic share a key.
%
%   truncationSigmas may be [] (the toolbox default). A non-finite or
%   non-positive sigma gives the structural default of 9 places.
%
%   Twin of _key_round_digits in python/mpt/_tensor/canonical.py.
%
%   See also internal.truncationFloor, internal.pairCanonicalKey,
%   internal.chordCanonicalKey.

    if nargin < 3
        truncationSigmas = [];
    end
    sigma = double(sigma);
    if ~(isfinite(sigma) && sigma > 0)
        digits = 9;
        return
    end
    kernelSlope = sqrt(2) * exp(-0.5);
    tau = internal.truncationFloor(truncationSigmas);
    deltaMax = tau * sigma / (2 * kernelSlope * sqrt(max(double(r), 1)));
    digits = ceil(log10(0.5 / deltaMax));
    digits = min(max(digits, -12), 15);
end
