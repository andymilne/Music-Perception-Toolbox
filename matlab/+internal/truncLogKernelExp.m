function out = truncLogKernelExp(logKernel, truncationSigmas, nTerms)
%INTERNAL.TRUNCLOGKERNELEXP  exp(logKernel) with optional truncation in log space.
%
%   OUT = INTERNAL.TRUNCLOGKERNELEXP(LOGKERNEL, TRUNCATIONSIGMAS)
%   evaluates exp(LOGKERNEL) with optional kernel truncation.
%   LOGKERNEL is the (non-positive) log of the kernel - typically
%   -sum_a d_a^2 / (4 sigma_a^2) accumulated across attributes (the
%   Bulger-method MA log-kernel pattern), where each attribute may
%   have its own sigma.
%
%   When TRUNCATIONSIGMAS is finite, entries with
%   LOGKERNEL < -TRUNCATIONSIGMAS^2 / 2 are zeroed without evaluating
%   exp(). The threshold uniformly matches: kernel value falls below
%   exp(-TRUNCATIONSIGMAS^2 / 2). This is the log-space counterpart
%   of INTERNAL.TRUNCKERNELEXP and applies cleanly to the MA
%   log-kernel case where per-attribute sigma values differ (so a
%   single quadratic-form cutoff doesn't apply).
%
%   TRUNCATIONSIGMAS is resolved through the accuracy floor: the Inf
%   ("exact") sentinel becomes the finite parity-floor width, so
%   truncation always applies (uniform with Python).
%
%   NTERMS (optional) states how many entries the caller will sum. The
%   floor is stated per entry, but the inner product is a sum, so
%   discarding NTERMS entries each just under the floor admits an error
%   of NTERMS times the floor on the summed value. This holds in every
%   mode: the periodic wrap makes the tail broadest, so the shortfall is
%   largest there, but an untightened threshold overshoots the stated
%   accuracy wherever the block is large enough. Passing the count
%   lowers the per-entry threshold to floor / NTERMS, bounding the total
%   discarded mass by the floor itself -- the scale the accuracy is
%   stated on. In log space this is a shift of -log(NTERMS), so the
%   equivalent width is sqrt(k^2 + 2 log(NTERMS)) and the cost is a
%   modestly wider kernel window rather than a different algorithm.
%
%   See also INTERNAL.TRUNCKERNELEXP, INTERNAL.ACCURACYFLOOR.

    truncationSigmas = internal.accuracyFloor('resolve', truncationSigmas);
    threshold = -0.5 * truncationSigmas^2;
    if nargin >= 3 && ~isempty(nTerms) && nTerms > 1
        threshold = threshold - log(double(nTerms));
    end
    mask = logKernel >= threshold;
    out = zeros(size(logKernel));
    out(mask) = exp(logKernel(mask));
end
