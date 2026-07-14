function m = relWindowMargin(truncationSigmas)
%RELWINDOWMARGIN  Margin (sigmas per side) of the non-periodic rel window.
%   Mirror of Python cosine._rel_window_margin. The inner-product
%   kernel is G(d; sigma*sqrt(2)) (a convolution of two sigma-kernels),
%   and the toolbox truncation zeroes entries whose kernel value falls
%   below exp(-t^2/2), i.e. at distances beyond sqrt(2)*t*sigma.
%
%   The margin's contract is subordination to the kernel-truncation
%   contract: the toolbox's accuracy guarantee is stated in
%   truncation-sigmas terms (t = 6 accepts ~2e-8 worst-case error;
%   t = 8 sits below the 1e-12 cross-language parity floor), so the
%   window need only keep its own error comfortably below the kernel
%   error the caller has already accepted. A margin of 8 achieves that
%   for every t: each kernel factor is at least e^-16 down at the
%   window edge, an r-tuple term needs r such factors, so the window
%   tail is bounded near 1e-14 relative independent of t (and measures
%   at ~1e-27 in practice). The rule is therefore
%   min(sqrt(2)*t + 0.1, 8): for t below ~5.59 the integrand's compact
%   support (which ends exactly sqrt(2)*t*sigma beyond the extreme
%   pair difference) fits inside the cap, so the cheaper exact window
%   is taken -- the endpoint integrand vanishes identically and the
%   plain Riemann sum equals the trapezoidal rule exactly (the
%   0.1-sigma clearance is needed because the extreme pair sits
%   exactly at margin*sigma at the endpoint and the truncation mask is
%   inclusive). For larger or disabled t the margin caps at the
%   empirically validated 8, where the window error is subdominant to
%   the kernel contract at every t.
    if ~isfinite(truncationSigmas)
        m = 8;
    else
        m = min(sqrt(2) * truncationSigmas + 0.1, 8);
    end
end
