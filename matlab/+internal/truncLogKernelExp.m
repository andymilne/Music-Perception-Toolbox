function out = truncLogKernelExp(logKernel, truncationSigmas)
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
%   When TRUNCATIONSIGMAS is non-finite (Inf), the full exponential
%   is computed and no masking work is done.
%
%   See also INTERNAL.TRUNCKERNELEXP, MPTDEFAULTS.

    if ~isfinite(truncationSigmas)
        out = exp(logKernel);
        return;
    end
    threshold = -0.5 * truncationSigmas^2;
    mask = logKernel >= threshold;
    out = zeros(size(logKernel));
    out(mask) = exp(logKernel(mask));
end
