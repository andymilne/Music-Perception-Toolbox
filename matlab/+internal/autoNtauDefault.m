function n = autoNtauDefault(period, sigma, truncationSigmas)
%AUTONTAUDEFAULT  Transposition-average node count from the truncation width.
%
%   N = INTERNAL.AUTONTAUDEFAULT(PERIOD, SIGMA) returns the trapezoidal node
%   count for the all-image (relative-periodic) transposition average over
%   [0, PERIOD), with the quadrature tolerance derived from the toolbox-wide
%   truncationSigmas default (mptDefaults).
%
%   N = INTERNAL.AUTONTAUDEFAULT(PERIOD, SIGMA, TRUNCATIONSIGMAS) derives
%   the tolerance from the given per-call width instead ([] = the
%   default). Every route that honours a per-call width on its kernel
%   cutoff must size its tau grid from the same width, or the grid
%   density and the route price stay pinned to the default while the
%   kernel moves; the Python twin takes the same optional argument.
%
%   This is the single shared source of the relative-periodic node count for
%   every path that evaluates it -- the flat single-attribute and
%   multi-attribute Mobius integrators (mobius.orbitInnerRelSingleMultiset, reused by
%   mobius.maPerAttrInnerMatrix) and the nested contraction
%   (internal.nestedContract) -- so their transposition grids coincide exactly
%   and the same level returns the same value whichever path computes it.
%
%   Mirror of Python _nested_contraction.auto_ntau_default. The integrand is
%   periodic-smooth with bandwidth ~ period/sigma, so the trapezoidal rule
%   converges geometrically past ~pi*period/sigma nodes; the margin keeps the
%   quadrature error well under the tolerance.
    if nargin < 3
        truncationSigmas = [];
    end
    % internal.truncationFloor resolves [] -> the default and Inf -> the
    % accuracy-floor width, and returns the floor epsilon exactly at that
    % width (no log/exp round trip), so a caller passing the resolved
    % width and one passing Inf size the same grid --- and the same grid
    % as the Python auto_ntau_default, whose truncation_floor does the
    % same.
    tol = internal.truncationFloor(truncationSigmas);
    base = 2 * pi * period / sigma;
    margin = 1 + 0.5 * max(0, -log10(max(tol, 1e-16))) / 12;
    n = max(64, ceil(base * margin));
end
