function f = truncationFloor(truncationSigmas)
%TRUNCATIONFLOOR  Kernel-value floor exp(-k^2/2) at the resolved width.
%   Mirror of Python _defaults.truncation_floor. The largest normalised
%   Gaussian value truncation discards -- the same for the density and
%   inner-product kernels, since it is stated on the value scale. This is
%   the single measure the toolbox judges accuracy by.
%
%   The argument is resolved through internal.accuracyFloor('resolve'),
%   so [] takes the default and Inf takes the finite accuracy-floor
%   width. Always returns a finite positive value.
%
%   Example:
%       f = internal.truncationFloor([]);    % ~1.523e-08 at the default
%
%   See also internal.accuracyFloor, internal.truncKernelExp.
    if nargin < 1
        truncationSigmas = [];
    end
    k = internal.accuracyFloor('resolve', truncationSigmas);
    % The "exact" sentinel resolves to accuracyFloor('sigmas'), whose
    % floor is by construction exactly accuracyFloor('eps'); return that
    % directly rather than through a log/exp round trip, which would
    % perturb it in the last ULP.
    if k == internal.accuracyFloor('sigmas')
        f = internal.accuracyFloor('eps');
        return;
    end
    f = exp(-0.5 * k * k);
end
