function tf = maRelAttrPrefersCentres(Px, Py, sigma, r_a, isRel, ...
                                        isPer, period)
%MARELATTRPREFERSCENTRES  Centres route vs translation grid, per attribute.
%
%   Mirror of Python cosine._ma_rel_attr_prefers_centres. True when a
%   relative attribute's (event_X, event_Y) inner matrices should use
%   the pairwise closed form over materialised tuple-centres
%   (MOBIUS.CLOSEDFORMATTRMATRIXFROM) rather than the batched
%   translation-grid contraction in MOBIUS.MAPERATTRINNERMATRIX.
%
%   The centres route costs (r_a! * C(K, r_a))^2 kernel ops per event
%   pair; the grid route costs N_u * K^2 (N_u from the shared
%   node-count source in periodic mode, or the centred-window rule in
%   non-periodic mode). The centres route is chosen when it is cheaper
%   AND measure-safe: it computes the minimum-image relative-periodic
%   reading (the toolbox's defined measure), which coincides with the
%   grid route's all-image reading only below the sigma/P threshold —
%   above it the grid route is kept so an explicit method='mobius'
%   opt-in preserves the all-image measure. The non-periodic closed
%   form is exact, so it is always measure-safe.

    SIGMA_OVER_P_THRESHOLD = 0.03;   % _ORBIT_SIGMA_OVER_P_THRESHOLD

    tf = false;
    if ~isRel || r_a < 2
        return;
    end
    if isPer && (sigma / period) > SIGMA_OVER_P_THRESHOLD
        return;
    end
    K = size(Px, 1);
    if K < r_a
        return;
    end
    centresPairOps = (factorial(r_a) * nchoosek(K, r_a))^2;
    if isPer
        n_u = internal.autoNtauDefault(period, sigma);
    else
        span = (max(Px(:), [], 'omitnan') - min(Px(:), [], 'omitnan')) ...
             + (max(Py(:), [], 'omitnan') - min(Py(:), [], 'omitnan')) ...
             + 16 * sigma;
        n_u = max(64, ceil(max(span, 1.0) / sigma * 10));
    end
    gridPairOps = n_u * K * K;
    tf = centresPairOps < gridPairOps;
end
