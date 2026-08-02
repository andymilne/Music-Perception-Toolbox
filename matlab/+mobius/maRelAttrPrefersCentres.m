function tf = maRelAttrPrefersCentres(Px, Py, sigma, r_a, isRel, ...
                                        isPer, period, truncationSigmas)
%MARELATTRPREFERSCENTRES  Centres route vs translation grid, per attribute.
%
%   Mirror of Python _mobius_inner._ma_rel_attr_prefers_centres. True
%   when a relative attribute's (event_X, event_Y) inner matrices should use
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
%   The two densities need not carry the same number of values in the
%   attribute, and a chord against a scale, or a reference tuning
%   against an equal division, is the ordinary case. The centres
%   estimate therefore spans all three matrices the route computes --
%   the cross matrix and one self matrix per density -- reading each
%   density's own value count; the grid estimate reads Px's count
%   alone, for the reasons recorded below at gOp.
%
%   Cost-model constants match the Python calibration exactly so that
%   both languages route the same (K_x, K_y, r, sigma, span, isPer)
%   cells to the same path (route parity across the two
%   implementations). See Python _mobius_inner._CENTRES_NS_BASE etc.
%   for the calibration notes; the fit was measured on a 168-cell
%   Python sweep spanning r_a in {2, 3, 4}, K in {5..80},
%   sigma in {5, 10, 15, 20, 25, 30}, span in {1200, 2400, 3000, 3600,
%   4800}, and both periodic modes. Every cell of that sweep gave both
%   densities the same value count, where the three matrices carry
%   3*M^2 elements between them; the fit was performed against an
%   element count of M^2, so the fitted figures (45, 15, 10 ns) each
%   absorb that factor of three. They appear below as those figures
%   divided by three, which leaves every equal-value-count cell
%   predicting exactly what it predicted when the fit was made.
%   Writing the division out keeps the fitted figures visible and gives
%   both languages the identical double.

    if nargin < 8, truncationSigmas = []; end
    SIGMA_OVER_P_THRESHOLD = ...
        internal.relPerSigmaOverPThreshold(truncationSigmas);

    % Cost-model constants (nanoseconds). Cross-language route parity
    % requires these to match Python _mobius_inner._CENTRES_NS_*,
    % _GRID_NS_*.
    CENTRES_NS_BASE  = 45.0 / 3.0;
    CENTRES_NS_LIN   = 15.0 / 3.0;
    CENTRES_NS_WRAP  = 10.0 / 3.0;
    GRID_NS_FLOOR    = 1e6;      % 1 ms per-pair setup
    % gridNsPerOp(r_a): per-(N_u * K) coefficient for r_a in {2, 3, 4}.
    % Small fixed table --- a containers.Map (previously used here)
    % was being reconstructed on every call; an inline switch is
    % essentially free and gives identical semantics.

    % Admissibility first. These returns are not cost judgements: no
    % choice exists at r_a < 2 or for an absolute attribute; the centres
    % route is measure-blocked above the sigma/period threshold, where
    % the wrapped-difference kernel it evaluates is no longer positive
    % definite; and a value count below r_a leaves an empty tuple set.
    % The relAttrRoute lever below overrides the cost judgement only, so
    % none of these may be forced past.
    tf = false;
    if ~isRel || r_a < 2
        return;
    end
    blockedByMeasure = isPer && (sigma / period) > SIGMA_OVER_P_THRESHOLD;
    K = size(Px, 1);
    K_y = size(Py, 1);
    emptyTupleSet = K < r_a || K_y < r_a;

    forced = mptDefaults('relAttrRoute');
    switch forced
        case 'grid'
            return;
        case 'centres'
            if blockedByMeasure
                error('mpt:relAttrRouteBlocked', ...
                    ['relAttrRoute=''centres'' cannot be honoured at ' ...
                     'sigma/period = %.4g: above %g the tuple-centres ' ...
                     'route evaluates a kernel that is not positive ' ...
                     'definite, so the translation grid is the only ' ...
                     'admissible route. Lower sigma/period or use ' ...
                     '''auto''.'], sigma / period, SIGMA_OVER_P_THRESHOLD);
            end
            if emptyTupleSet
                error('mpt:relAttrRouteBlocked', ...
                    ['relAttrRoute=''centres'' cannot be honoured with ' ...
                     'value counts (%d, %d) at r_a = %d: the tuple set ' ...
                     'is empty.'], K, K_y, r_a);
            end
            tf = true;
            return;
    end

    if blockedByMeasure || emptyTupleSet
        return;
    end

    % Centres wall (ns): per-element base + linear in (r_a-1); periodic
    % adds an (r_a-1)(r_a-2) inner pairwise-wrap term (zero at r_a = 2).
    % The element count spans all three matrices the route computes,
    % M_x*M_y + M_x^2 + M_y^2, with M taken from each density's own
    % value count. The three-matrix count matters: a five-partial
    % reference against an 80-value tuning has M_x = 20 and M_y = 6320,
    % and the second self matrix supplies 4.0e7 of the 4.0e7 elements.
    M_x = factorial(r_a) * nchoosek(K, r_a);
    M_y = factorial(r_a) * nchoosek(K_y, r_a);
    n_e = M_x * M_y + M_x * M_x + M_y * M_y;
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
    % The grid estimate reads Px's value count alone, so it is low
    % whenever Py carries more values. That is deliberate on two
    % grounds. Choosing the grid route where centres is faster costs at
    % most the setup floor, whereas choosing centres where the grid
    % route is faster costs a factor rising as the fourth power of the
    % larger value count, so a low grid estimate errs on the cheap
    % side. And whether this estimate should depend on the value count
    % at all is an open question: measured grid times on this workload
    % do not grow with it, which is the subject of the pending
    % bench_ip_unit_cost extension. Raising the estimate now would
    % pre-empt that measurement.
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
