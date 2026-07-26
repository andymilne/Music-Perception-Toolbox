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
%   The gate compares predicted wall time on the two paths rather than
%   raw kernel-op counts. Raw counts were misleading here because the
%   centres path pays roughly one exp per op (~50 ns) while the grid
%   path — served by the spectral branch of maPerAttrInnerMatrix at
%   r_a in {2, 3, 4} — pays a sub-K^2 per-op cost after a fixed setup,
%   so the per-op cost ratio between them varies with r_a and
%   configuration and is far from unity. The old raw-count comparison
%   consequently over-selected centres for r_a = 2 across a wide K
%   band (roughly 15..90 at sigma/P ~ 1/240), where centres was up to
%   100x slower than grid.
%
%   Decision-safety layer: below the sigma/P threshold the centres
%   route's minimum-image relative-periodic reading coincides with the
%   grid route's all-image reading, so the two agree numerically;
%   above it, the grid route is kept regardless of cost so that
%   method='mobius' on that side opts into the all-image measure
%   without the gate flipping to a different reading. The non-periodic
%   closed form is exact for the non-periodic reading and always
%   measure-safe.
%
%   Cost-model constants match the Python calibration exactly so that
%   both languages route the same (K, r, sigma, span, isPer) cells to
%   the same path (route parity across the two implementations). See
%   Python cosine._CENTRES_NS_BASE etc. for the calibration notes; the
%   fit was measured on a 168-cell Python sweep spanning
%   r_a in {2, 3, 4}, K in {5..80}, sigma in {5, 10, 15, 20, 25, 30},
%   span in {1200, 2400, 3000, 3600, 4800}, and both periodic modes.

    SIGMA_OVER_P_THRESHOLD = 0.03;   % _ORBIT_SIGMA_OVER_P_THRESHOLD

    % Cost-model constants (nanoseconds). Cross-language route parity
    % requires these to match Python cosine._CENTRES_NS_*, _GRID_NS_*.
    CENTRES_NS_BASE  = 45.0;
    CENTRES_NS_LIN   = 15.0;
    CENTRES_NS_WRAP  = 10.0;
    GRID_NS_FLOOR    = 1e6;      % 1 ms per-pair setup
    % gridNsPerOp(r_a): per-(N_u * K) coefficient for r_a in {2, 3, 4}.
    % Small fixed table --- a containers.Map (previously used here)
    % was being reconstructed on every call; an inline switch is
    % essentially free and gives identical semantics.

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

    % Centres wall (ns): per-element base + linear in (r_a-1); periodic
    % adds an (r_a-1)(r_a-2) inner pairwise-wrap term (zero at r_a = 2).
    n_e = (factorial(r_a) * nchoosek(K, r_a))^2;
    perEl = CENTRES_NS_BASE + CENTRES_NS_LIN * (r_a - 1);
    if isPer
        perEl = perEl + CENTRES_NS_WRAP * (r_a - 1) * (r_a - 2);
    end
    cWallNs = perEl * n_e;

    % Grid wall (ns): fixed setup + per-r_a coefficient * N_u * K.
    if isPer
        n_u = internal.autoNtauDefault(period, sigma);
    else
        margin = internal.relWindowMargin(mptDefaults('truncationSigmas'));
        span = (max(Px(:), [], 'omitnan') - min(Px(:), [], 'omitnan')) ...
             + (max(Py(:), [], 'omitnan') - min(Py(:), [], 'omitnan')) ...
             + 2 * margin * sigma;
        n_u = max(64, ceil(max(span, 1.0) / sigma * 10));
    end
    switch r_a
        case 2
            gOp = 30.0;
        case 3
            gOp = 700.0;
        case 4
            gOp = 2000.0;
        otherwise
            % Extrapolate calibrated r_a in {2, 3, 4} to r_a >= 5 by
            % tripling per r_a increment; centres cost grows faster than
            % that in K, so the extrapolation only affects the tiny-K
            % corner.
            gOp = 2000.0 * 3.0^(r_a - 4);
    end
    gWallNs = GRID_NS_FLOOR + gOp * double(n_u) * double(K);

    tf = cWallNs < gWallNs;
end
