function out = truncKernelExp(expArg, sigma, truncationSigmas)
%INTERNAL.TRUNCKERNELEXP  exp(-expArg / (4 sigma^2)) with optional truncation.
%
%   OUT = INTERNAL.TRUNCKERNELEXP(EXPARG, SIGMA, TRUNCATIONSIGMAS)
%   evaluates exp(-expArg / (4 * sigma^2)) with optional kernel
%   truncation. EXPARG is the non-negative quantity entering the
%   kernel exponent: for a 1-D pairwise Gaussian IP kernel,
%   EXPARG = (p_x - p_y)^2; for an r-D r-tuple kernel,
%   EXPARG = ||d||^2 = sum_a d_a^2.
%
%   When TRUNCATIONSIGMAS is finite, entries with
%   EXPARG > 2 * (TRUNCATIONSIGMAS * SIGMA)^2 are zeroed without
%   evaluating exp(), saving work proportional to the pruned
%   fraction. The threshold uniformly matches "the inner-product
%   kernel value falls below exp(-TRUNCATIONSIGMAS^2 / 2)": the IP
%   kernel is G(d; sigma * sqrt(2)), so its value at distance |d| is
%   exp(-|d|^2 / (4 sigma^2)), and the cutoff condition is
%   |d|^2 > 2 (TRUNCATIONSIGMAS * sigma)^2.
%
%   For r-tuple kernels the same threshold on sum_a d_a^2 is correct
%   because the r-D kernel is prod_a G(d_a; sigma * sqrt(2)) =
%   exp(-sum_a d_a^2 / (4 sigma^2)).
%
%   When TRUNCATIONSIGMAS is non-finite (Inf), the full exponential
%   is computed and no masking work is done.
%
%   See also INTERNAL.TRUNCLOGKERNELEXP, MPTDEFAULTS.

    if ~isfinite(truncationSigmas)
        out = exp(-expArg / (4 * sigma^2));
        return;
    end
    cutoff = 2 * (truncationSigmas * sigma)^2;
    mask = expArg <= cutoff;
    out = zeros(size(expArg));
    out(mask) = exp(-expArg(mask) / (4 * sigma^2));
end
