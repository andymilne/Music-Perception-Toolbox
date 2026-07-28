function bundle = closedFormAttrCentres(dens, a)
%CLOSEDFORMATTRCENTRES  Materialised tuple-centres bundle for attribute a.
%
%   Mirror of Python cosine._closed_form_attr_centres, restricted to
%   flat attributes (the multi-attribute Möbius orchestrator routes
%   only flat attributes here; nested attributes take their own
%   contraction paths before reaching it).
%
%   The attribute is rebuilt in isolation as a single-attribute
%   density: the multi-attribute density stores tuple-centres on the
%   joint perm-side index with joint weight products, whereas the
%   per-attribute inner matrix needs the attribute's own tuple
%   enumeration with its own weights. The rebuild is a skinny
%   single-attribute construction (milliseconds), eagerly
%   materialised.
%
%   Relative attributes store centres in the first-coordinate reduced
%   convention (r_a - 1 rows: U_perm(2:r) - U_perm(1)); absolute
%   attributes store full r_a-tuples. MOBIUS.CLOSEDFORMATTRMATRIXFROM
%   evaluates the matching quadratic form for each convention.
%
%   Ragged (NaN-padded) values are carried by the rebuild, which drops
%   every tuple touching a padded value to zero weight.
%
%   Returns a struct with fields:
%     Centres   (d x nJ)  tuple-centres (d = r_a - 1 rel, r_a abs)
%     wJ        (1 x nJ)  per-tuple weights
%     eventOfJ  (1 x nJ)  event index per tuple
%     isPer, period, r, isRel, sigma, N — attribute parameters.

    Pa = dens.pAttr{a};
    Wa = dens.w{a};
    sigma_a = dens.sigma(a);
    r_a = dens.r(a);
    isRel_a = logical(dens.isRel(a));
    isPer_a = logical(dens.isPer(a));
    period_a = dens.period(a);

    da = buildExpTens({Pa}, {Wa}, sigma_a, r_a, isRel_a, isPer_a, ...
        period_a, 'lazy', false, 'verbose', false);

    bundle = struct();
    bundle.Centres  = da.Centres{1};
    bundle.wJ       = da.wJ;
    bundle.eventOfJ = da.eventOfJ;
    bundle.isPer    = isPer_a;
    bundle.period   = period_a;
    bundle.r        = r_a;
    bundle.isRel    = isRel_a;
    bundle.sigma    = sigma_a;
    bundle.N        = da.N;
end
