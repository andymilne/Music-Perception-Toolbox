function n = autoNtauDefault(period, sigma)
%AUTONTAUDEFAULT  Transposition-average node count; tol from the global default.
%
%   N = INTERNAL.AUTONTAUDEFAULT(PERIOD, SIGMA) returns the trapezoidal node
%   count for the all-image (relative-periodic) transposition average over
%   [0, PERIOD), with the quadrature tolerance derived from the toolbox-wide
%   truncationSigmas default (mptDefaults).
%
%   This is the single shared source of the relative-periodic node count for
%   every path that evaluates it -- the flat single-attribute and
%   multi-attribute Mobius integrators (mobius.orbitInnerRelSA, reused by
%   mobius.maPerAttrInnerMatrix) and the nested contraction
%   (internal.nestedContract) -- so their transposition grids coincide exactly
%   and the same level returns the same value whichever path computes it.
%
%   Mirror of Python _nested_contraction.auto_ntau_default. The integrand is
%   periodic-smooth with bandwidth ~ period/sigma, so the trapezoidal rule
%   converges geometrically past ~pi*period/sigma nodes; the margin keeps the
%   quadrature error well under the tolerance.
    ts = mptDefaults('truncationSigmas');
    if isempty(ts) || ~isfinite(ts)
        tol = 1e-12;
    else
        tol = max(exp(-0.5 * double(ts)^2), 1e-12);
    end
    base = 2 * pi * period / sigma;
    margin = 1 + 0.5 * max(0, -log10(max(tol, 1e-16))) / 12;
    n = max(64, ceil(base * margin));
end
