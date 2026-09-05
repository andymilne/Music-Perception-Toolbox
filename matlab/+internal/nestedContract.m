function [triple, routes, cacheX, cacheY] = nestedContract( ...
        densX, densY, normalize, truncationSigmas, force, opts)
%NESTEDCONTRACT Per-attribute dispatch for a nested attribute's IP.
%   triple = internal.nestedContract(densX, densY, normalize, truncationSigmas)
%   [triple, routes, cacheX, cacheY] = internal.nestedContract( ...
%       densX, densY, normalize, truncationSigmas, force, opts)
%
%   Returns a struct with fields xy, xx, yy (the bare inner-product triple)
%   when the case is covered -- exactly one nested attribute, outer or no
%   [rel] (inner_r == 0), cosine or one-sided normalisation, NaN-padded
%   (variable-K) values included -- and [] otherwise, when the caller routes
%   to the exact Bulger enumeration. ROUTES is a cell row naming the route
%   each attribute took ('-' for an attribute that took neither a nested nor
%   an ordered-flat route), also recorded in INTERNAL.LASTNESTEDROUTES.
%
%   OPTS is an optional struct; every field is optional:
%     forceRoute  the route name to force for every nested attribute:
%                 'centres' (method = 'centres'), 'contract',
%                 'contract_relnonper' or 'taugrid'; '' (default) lets the
%                 route rule decide. Forcing overrides the cost race but
%                 not the measure rule, which errors rather than silently
%                 returning a different measure, and a route whose mode
%                 does not apply (e.g. 'taugrid' on an absolute attribute)
%                 errors too. Only 'centres' is reachable from the public
%                 method keyword; the other names exist for benchmarking
%                 (tests/bench_nested_cost.m), which must time every route
%                 that carries a cell's measure.
%     termsOnly   true to return, instead of a triple, the ANALYTIC COST
%                 TERMS of one nested attribute -- the quantities the cost
%                 model in python/mpt/_tensor/_nested_cost.py is fitted
%                 against -- computed from this file's own tupleCounts,
%                 recipeWork and quadNodes so the harness and the dispatch
%                 read one set of counts. TRIPLE is then a struct with
%                 fields terms (centres / taugrid / contract_relnonper /
%                 contract), bulger (the joint tuple-pair kernel size of
%                 the whole density) and info (the counts the terms were
%                 built from). Reached through INTERNAL.NESTEDCOSTTERMS.
%     termsAttr   the attribute index termsOnly reports on (default 1).
%     termsSkipXX, termsSkipYY
%                 true to leave the corresponding self matrix out of the
%                 termsOnly sums (default false, which is what the
%                 harness wants: it times cold densities and so computes
%                 all three matrices). The bulger total is unaffected;
%                 its per-matrix components ride in info instead,
%                 because the enumeration memoises under its own key and
%                 so carries its own skip flags.
%     methodName  the user-facing method name to quote in a decline
%                 message ('contract' by default).
%     cacheX, cacheY
%                 self-IP memo structs (COSSIMEXPTENS's 'keys'/'vals'
%                 shape, plus the optional 'nestedCentres' cell holding
%                 the centres route's memoised tuple-centres bundles,
%                 see INTERNAL.NESTEDCENTRESMEMOISED); returned updated.
%     routesOnly  true to plan and return ROUTES without computing
%                 anything (used by EXPLAINDISPATCH).
%
%   Route set (per nested attribute, decided once so xy, xx and yy share one
%   measure and one quadrature grid):
%     'contract'            absolute and absolute-periodic: the event-pair
%                           contraction, exact, per-level Moebius/Bulger.
%     'centres'             the materialised tuple centres
%                           (MOBIUS.CLOSEDFORMATTRCENTRES /
%                           CLOSEDFORMATTRMATRIXFROM): the exact analytic
%                           quadratic for relative-non-periodic, the
%                           minimum-image measure (A) for relative-periodic.
%     'contract_relnonper'  relative-non-periodic on the line translation
%                           grid; the same measure as the centres
%                           quadratic to grid accuracy, so a pure speed
%                           choice.
%     'taugrid'             relative-periodic on the all-image tau grid:
%                           the transposition average over the period,
%                           measure (C).
%
%   The measure is declared by wrap, not by the dispatch, mirroring the rule
%   INTERNAL.SELECTMAINNERPRODUCTMETHOD already enforces on the flat path,
%   and the rule is symmetric in the two declarations (NESTEDADMISSIBLEROUTES
%   is where it lives):
%
%     * above sigma/P = INTERNAL.RELPERSIGMAOVERPTHRESHOLD the two readings
%       differ by more than the truncation floor, so each declaration has
%       exactly one carrier and that route is taken whatever it costs --- the
%       tau grid under wrap = 'full-image' (the default, declaring (C)), the
%       centres route under wrap = 'single-image' (declaring (A)). A cheaper
%       route to a different number is not a cheaper route.
%     * at or below the threshold the two agree inside the floor, so BOTH
%       routes serve EITHER declaration and the price decides. This is what
%       the flat selector does with Bulger's method and the Moebius method:
%       its wrap override is reached only above the threshold, and below it
%       the two are raced whatever wrap says.
%
%   Relative-non-periodic and absolute attributes offer only same-measure
%   choices and stay cost-driven throughout. Where more than one route
%   survives the measure rule, INTERNAL.NESTEDCOST prices each survivor in
%   milliseconds from its fitted law and diverts the materialising centres
%   route where its bundle exceeds the working-set soft budget; that model
%   replaced a raw operation-count comparison which weighed a kernel entry
%   against a unit of quadrature work as though the two cost the same.
%
%   The whole plan is then priced against the joint-tuple enumeration, as the
%   flat selector prices Bulger's method against the Moebius method. Where
%   the enumeration wins, this returns [] and the caller runs it --- which is
%   what a [] has always meant here. A forced method is never diverted.
%
%   Mirror of the Python mpt/_tensor/_nested_contraction.py module and the
%   _try_nested_contract / _nested_attr_route / _nested_attr_plan /
%   _nested_attr_matrix dispatcher. Quadrature node counts, recipe structure,
%   tuple counts and the cost estimate are integer-identical to the Python
%   path so the two languages take the same route; reduction order differs,
%   so values agree to floating-point (not bit-for-bit).

    if nargin < 5 || isempty(force); force = false; end
    if nargin < 6 || isempty(opts); opts = struct(); end
    forceRoute = localOptField(opts, 'forceRoute', '');
    methodName = localOptField(opts, 'methodName', 'contract');
    cacheX     = localOptField(opts, 'cacheX', struct('keys', {{}}, 'vals', []));
    cacheY     = localOptField(opts, 'cacheY', struct('keys', {{}}, 'vals', []));
    routesOnly = localOptField(opts, 'routesOnly', false);
    termsOnly  = localOptField(opts, 'termsOnly', false);
    termsAttr  = localOptField(opts, 'termsAttr', 1);
    termsSkipXX = localOptField(opts, 'termsSkipXX', false);
    termsSkipYY = localOptField(opts, 'termsSkipYY', false);
    triple = [];
    routes = {};
    if ~isequal(termsOnly, false)
        % Analytic terms only: no matrix is formed and no route is taken,
        % so this returns before every decline path below (a case the plan
        % declines still has well-defined counts).
        triple = localNestedTerms(densX, densY, double(termsAttr), ...
            internal.accuracyFloor('resolve', truncationSigmas), ...
            termsSkipXX, termsSkipYY);
        return;
    end
    if ~any(strcmp(normalize, {'cosine', 'oneSidedDenom'}))
        declineContractIfForced(force, methodName, ...
            sprintf('unsupported normalisation ''%s''', normalize));
        return;
    end
    if densX.nAttrs ~= densY.nAttrs
        declineContractIfForced(force, methodName, ...
            'the two densities have different attribute counts');
        return;
    end
    if densX.nAttrs ~= 1
        % Nested attribute(s) tensored with further attributes: the cosine
        % factorises per event-pair across attributes (JMM Eq 3.4), so each
        % nested factor goes through the contraction and each plain factor
        % through the per-attribute MA matrix, instead of enumerating the
        % joint tuple.
        [triple, routes, cacheX, cacheY] = nestedContractMA(densX, densY, ...
            normalize, truncationSigmas, force, forceRoute, methodName, ...
            cacheX, cacheY, routesOnly);
        return;
    end
    if ~isfield(densX, 'nested') || ~iscell(densX.nested) ...
            || ~isfield(densY, 'nested') || ~iscell(densY.nested)
        declineContractIfForced(force, methodName, ...
            'both densities must carry the same nested attribute');
        return;
    end
    specX = densX.nested{1};
    specY = densY.nested{1};
    if isempty(specX) || isempty(specY)
        declineContractIfForced(force, methodName, ...
            'both densities must carry the same nested attribute');
        return;
    end
    if localInnerR(specX) ~= 0 || localInnerR(specY) ~= 0
        declineContractIfForced(force, methodName, ...
            'an inner/intermediate [rel] unit is not yet covered');
        return;   % inner [rel] unit not covered by the contraction yet
    end

    PX = double(densX.pAttr{1});
    PY = double(densY.pAttr{1});
    WX = densX.w{1}; if isempty(WX); WX = ones(size(PX)); end
    WY = densY.w{1}; if isempty(WY); WY = ones(size(PY)); end
    WX = double(WX);
    WY = double(WY);
    % Variable-K (NaN-padded) values: a padded value is exactly equivalent
    % to a zero-weight value at any finite position (every tuple touching it
    % carries zero weight), so the contraction covers it by filling each
    % padded value with an in-range value at weight zero -- the same
    % NaN -> zero-weight idiom as mobius.maPerAttrInnerMatrix.
    mX = isnan(PX);
    mY = isnan(PY);
    if any(mX(:)) || any(mY(:))
        fillVal = min(min(PX(:), [], 'omitnan'), min(PY(:), [], 'omitnan'));
        PX(mX) = fillVal;  WX(mX | isnan(WX)) = 0;
        PY(mY) = fillVal;  WY(mY | isnan(WY)) = 0;
    end

    rLevels   = double(specX.r(:)).';
    symLevels = logical(specX.sym(:)).';
    % tags: rows index values, columns the inner tag levels (L-1 of them). A
    % MATLAB literal tag vector is a row, whereas buildRecipe takes the value
    % count from dimension 1 (matching the Python 1-D convention), so orient
    % single-level (L=2) tags as a column and undo any transposed matrix.
    tagsX = orientTags(double(specX.tags), size(PX, 1), numel(rLevels));
    tagsY = orientTags(double(specY.tags), size(PY, 1), numel(rLevels));
    % The two densities must agree on the per-level read-arities and [sym]
    % flags (same nested attribute); only the leaf cardinalities (tags shape)
    % may differ -- a 4-pitch prototype against an 8-pitch window, say.
    if ~isequal(rLevels, double(specY.r(:)).') ...
            || ~isequal(symLevels, logical(specY.sym(:)).')
        declineContractIfForced(force, methodName, ...
            'the two nested attributes differ in [r]/[sym]');
        return;
    end
    sameStruct = isequal(size(tagsX), size(tagsY)) && isequal(tagsX, tagsY);
    isRel  = logical(densX.isRel(1));
    isPer  = logical(densX.isPer(1));
    period = double(densX.period(1));
    sigma  = double(densX.sigma(1));
    % Resolve the truncation width once, at entry, through the shared
    % accuracy-floor resolver: [] -> the mptDefaults default, Inf -> the
    % finite accuracy-floor width (~7.43 sigma, the 1e-12 floor), a finite
    % value passes through unchanged. Per the toolbox contract Inf means
    % "accuracy-floor accuracy", NOT unbounded exact summation, so every
    % downstream isfinite(ts) gate here receives a finite width. (Genuinely
    % exhaustive summation is reachable only by widening the floor eps via
    % internal.accuracyFloor('setEps', ...), as golden regeneration does.)
    ts = internal.accuracyFloor('resolve', truncationSigmas);
    % Bind the accuracy budget for this call; warnings fire once per
    % scope rather than once per block.
    orbitGuard('begin', ts);
    guardCleanup = onCleanup(@() orbitGuard('end', []));  %#ok<NASGU>

    % No sigma/period warning here any more. The mpt:nestedSurrogateResolution
    % warning fired above 0.85 of the sigma/P at which the accuracy floor
    % first asks for a second periodic image -- exactly where the
    % nearest-image kernel this branch used to average over tau stopped
    % being the wrapped Gaussian. relPerKernel now averages the wrapped
    % Gaussian itself, so the transposition average is the all-image
    % measure at every sigma/P, to spectral quadrature accuracy; the
    % surrogate the warning described no longer exists. Python has no such
    % warning.

    % Per-attribute route and shared quadrature grid (see the header and
    % nestedAttrPlan). The skip flags say which self inner products this
    % call will actually compute, so a memoised (or unconsumed) one is not
    % priced; they are derived from the caches the caller threaded in.
    [skipXX, skipYY] = selfIpSkipFlags(cacheX, cacheY, normalize);
    [route, quad] = nestedAttrPlan(densX, densY, 1, forceRoute, ts, ...
                                   skipXX, skipYY);
    % The plan is a candidate, not a conclusion: under method = 'auto' it
    % is priced against the joint-tuple enumeration and the cheaper is
    % taken, mirroring the flat selector's Bulger-versus-Moebius
    % comparison. Declining here returns the caller to the enumeration,
    % which is what a [] has always meant. A forced method is never
    % diverted, and neither is a routes-only plan (EXPLAINDISPATCH prices
    % the two sides itself, and wants the plan's routes either way).
    if ~force && ~routesOnly && nestedPrefersEnumeration(densX, densY, ...
            {route}, ts, skipXX, skipYY)
        internal.lastNestedRoutes({});
        triple = [];
        routes = {};
        return;
    end
    routes = {route};
    internal.lastNestedRoutes(routes);
    if routesOnly
        return;
    end

    % The two self inner products are memoised on their densities, as the
    % flat Bulger, centres and Moebius routes already do -- a sweep against
    % one prototype, or any repeated call on the same pair, then pays for
    % the cross term alone. The key carries the route *and* the shared
    % quadrature grid, because the grid routes discretise the self inner
    % product too: a different partner can widen the grid (the relative-
    % non-periodic line spans both densities' values), and a value taken
    % under one grid must never be reused under another. wrap is part of
    % each density's own contents, so it needs no separate key entry.
    key = internal.selfIpKey('contract', ts, routeSignature(route, quad));
    [xxHit, xxVal] = cacheGet(cacheX, key);
    [yyHit, yyVal] = cacheGet(cacheY, key);
    % <X,X> is consumed by the cosine only: under 'oneSidedDenom' it is
    % neither computed nor memoised, as on the flat routes, and the
    % caller's finaliser receives [] for it.
    needXX = strcmp(normalize, 'cosine');
    formXX = needXX && ~xxHit;

    if strcmp(route, 'centres')
        % The materialised tuple centres. The per-attribute Gaussian
        % prefactor closedFormAttrMatrixFrom drops is identical across the
        % cross and both self matrices, so it cancels in the cosine and the
        % one-sided ratios -- the same argument the flat centres route
        % makes.
        % The two bundles are built once and shared by the cross and the
        % self matrices, and memoised on each density's memo struct
        % (INTERNAL.NESTEDCENTRESMEMOISED), which the caller threads
        % back into the density's 'selfIP' field, so a later call
        % against the same density finds them built: the rebuild
        % materialises the attribute's permutation arrays, which at
        % small r is the dominant cost of the whole route. Twin of the
        % Python _nested_centres_cache.
        wrapA = wrapPair(densX, densY, 1);
        [cxB, cacheX] = internal.nestedCentresMemoised(cacheX, densX, 1);
        [cyB, cacheY] = internal.nestedCentresMemoised(cacheY, densY, 1);
        ipxy = sum(sum(mobius.closedFormAttrMatrixFrom(cxB, cyB, wrapA, ts)));
        if xxHit
            ipxx = xxVal;
        elseif ~needXX
            ipxx = [];
        else
            ipxx = sum(sum(mobius.closedFormAttrMatrixFrom( ...
                cxB, cxB, wrapA, ts)));
        end
        if yyHit
            ipyy = yyVal;
        else
            ipyy = sum(sum(mobius.closedFormAttrMatrixFrom( ...
                cyB, cyB, wrapA, ts)));
        end
    else
        % One recipe per side: the X recipe indexes the X axis of the
        % rectangular leaf kernel, the Y recipe the Y axis. They coincide
        % when the densities share a nesting structure (the common case,
        % incl. all XX/YY products).
        recipeX = buildRecipe(rLevels, symLevels, tagsX, isRel, isPer);
        if sameStruct
            recipeY = recipeX;
        else
            recipeY = buildRecipe(rLevels, symLevels, tagsY, isRel, isPer);
        end
        ipxy = tripSum(recipeX, recipeY, PX, WX, PY, WY, sigma, period, ...
                       ts, quad, false);
        if xxHit
            ipxx = xxVal;
        elseif ~needXX
            ipxx = [];
        else
            ipxx = tripSum(recipeX, recipeX, PX, WX, PX, WX, sigma, ...
                           period, ts, quad, true);
        end
        if yyHit
            ipyy = yyVal;
        else
            ipyy = tripSum(recipeY, recipeY, PY, WY, PY, WY, sigma, ...
                           period, ts, quad, true);
        end
    end
    if formXX; cacheX = cacheSet(cacheX, key, ipxx); end
    if ~yyHit; cacheY = cacheSet(cacheY, key, ipyy); end
    triple = struct('xy', ipxy, 'xx', ipxx, 'yy', ipyy);
end


% ----------------------------------------------------------------------
%  Per-attribute route rule, plan, and the cost race behind it
% ----------------------------------------------------------------------
function [route, quad] = nestedAttrPlan(densX, densY, a, forceRoute, ts, ...
                                        skipXX, skipYY)
%NESTEDATTRPLAN  Route plus the *shared* quadrature grid for a nested
%   attribute, decided once from the (x, y) pair. Mirror of the Python
%   cosine._nested_attr_plan.
%
%   Returning the grid here -- rather than recomputing it inside each of
%   xy, xx and yy -- is what makes the cosine normalise exactly: the
%   relative grids are value-dependent, so a per-call grid would
%   discretise the three inner products differently and the ratio would
%   drift off 1 (breaking, e.g., transposition invariance). One grid
%   spanning both densities serves all three.
%
%   The route names map onto makeQuadrature's modes one-for-one --
%   'taugrid' is the relative-periodic period grid, 'contract_relnonper'
%   the relative-non-periodic line grid, 'contract' the absolute
%   (single-node) mode -- so one call covers all three; 'centres'
%   materialises tuples instead and needs no grid.
%
%   SKIPXX / SKIPYY are passed to the cost model so a memoised (or
%   unconsumed) self inner product is not priced, mirroring the flat
%   selector's per-route skip flags.
    if nargin < 6 || isempty(skipXX); skipXX = false; end
    if nargin < 7 || isempty(skipYY); skipYY = false; end
    PXa = double(densX.pAttr{a});
    PYa = double(densY.pAttr{a});
    vmin = min(min(PXa(:), [], 'omitnan'), min(PYa(:), [], 'omitnan'));
    vmax = max(max(PXa(:), [], 'omitnan'), max(PYa(:), [], 'omitnan'));
    route = nestedAttrRoute(densX, densY, a, forceRoute, ts, skipXX, skipYY);
    if strcmp(route, 'centres')
        quad = [];
        return;
    end
    quad = makeQuadrature(logical(densX.isRel(a)), logical(densX.isPer(a)), ...
                          double(densX.sigma(a)), double(densX.period(a)), ...
                          vmin, vmax, ts, wrapPair(densX, densY, a));
end


function admissible = nestedAdmissibleRoutes(densX, densY, a, ts)
%NESTEDADMISSIBLEROUTES  The routes for nested attribute A that carry its
%   declared measure. Mirror of the Python
%   cosine._nested_admissible_routes. The wrap is read from both
%   densities (WRAPPAIR), which error where they disagree.
%
%   This is the measure rule and nothing else: it says which routes are
%   on offer, never which is taken. NESTEDATTRROUTE applies FORCEROUTE
%   and, where more than one route survives, INTERNAL.NESTEDCOST picks
%   among them.
%
%   An absolute attribute reports {'contract'} alone. The materialised
%   centres carry the absolute measure too (both routes read the
%   attribute's declared wrap), but 'auto' has always kept absolute
%   attributes on the contraction and still does; method = 'centres'
%   reaches the centres route there through FORCEROUTE.
%
%   A relative-periodic attribute is governed by wrap and by the
%   sigma/period threshold together, and the rule is symmetric in the
%   two declarations, exactly as on the flat path:
%
%     * above the threshold the two readings differ by more than the
%       truncation floor, so only the route that computes the declared
%       one is admissible --- the tau grid under wrap = 'full-image',
%       the centres under wrap = 'single-image';
%     * below it they agree inside the floor, so BOTH routes are
%       admissible under EITHER declaration and the price decides. This
%       is what INTERNAL.SELECTMAINNERPRODUCTMETHOD does with Bulger's
%       method and the Moebius method: its wrap override is reached only
%       above the threshold, and below it the two are raced whatever
%       wrap says.
    isRel = logical(densX.isRel(a));
    isPer = logical(densX.isPer(a));
    if ~isRel
        admissible = {'contract'};
        return;
    end
    if ~isPer
        admissible = {'centres', 'contract_relnonper'};
        return;
    end
    sigma  = double(densX.sigma(a));
    period = double(densX.period(a));
    limit  = internal.relPerSigmaOverPThreshold(ts);
    if period > 0 && sigma / period > limit
        % Beyond the floor the declared measure has exactly one carrier.
        if strcmp(wrapPair(densX, densY, a), 'single-image')
            admissible = {'centres'};
        else
            admissible = {'taugrid'};
        end
        return;
    end
    admissible = {'centres', 'taugrid'};
end


function route = nestedAttrRoute(densX, densY, a, forceRoute, ts, ...
                                 skipXX, skipYY)
%NESTEDATTRROUTE  Per-attribute route for a nested attribute, decided once
%   so xy, xx and yy share a single measure. Mirror of the Python
%   cosine._nested_attr_route; the route meanings and the measure rule are
%   in this file's header and in NESTEDADMISSIBLEROUTES.
%
%   The measure rule settles the admissible set first. Where it leaves
%   more than one route, INTERNAL.NESTEDCOST prices each survivor in
%   milliseconds from its fitted law and diverts the materialising
%   centres route where its bundle exceeds the working-set soft budget.
%   That model replaced RELCONTRACTCHEAPER's raw operation-count
%   comparison, which weighed materialised kernel entries against
%   quadrature nodes times contraction work as though the two cost the
%   same per unit. (The centres term is still counted exactly as that
%   function counted it, Bulger orbit restriction included; see the term
%   notes in INTERNAL.NESTEDCOST and LOCALNESTEDTERMS.)
%
%   FORCEROUTE names a route the caller has forced (method = 'centres'
%   forces 'centres'; the benchmark harness also forces 'contract',
%   'contract_relnonper' and 'taugrid'); it overrides the cost race but
%   not the measure rule, which errors instead of silently returning a
%   different measure, and not the mode rule, which errors on a route
%   that has no meaning for this attribute's (rel, per) pair.
%
%   SKIPXX / SKIPYY name the self inner products this call will not
%   compute, so the cost model does not price them.
    if nargin < 6 || isempty(skipXX); skipXX = false; end
    if nargin < 7 || isempty(skipYY); skipYY = false; end
    isRel = logical(densX.isRel(a));
    isPer = logical(densX.isPer(a));
    forced = '';
    if ~isempty(forceRoute)
        forced = char(forceRoute);
        localCheckForcedMode(forced, a, isRel, isPer);
    end
    if ~isRel
        if strcmp(forced, 'centres')
            route = 'centres';
        else
            route = 'contract';
        end
        return;
    end
    sigma  = double(densX.sigma(a));
    period = double(densX.period(a));
    admissible = nestedAdmissibleRoutes(densX, densY, a, ts);
    if isPer && numel(admissible) == 1
        % Above the threshold the declared measure has one carrier, and
        % that route is taken whatever it costs.
        if strcmp(admissible{1}, 'taugrid')
            if strcmp(forced, 'centres')
                error('cosSimExpTens:centresUnavailable', ...
                    ['method=''centres'' cannot be honoured on relative-' ...
                     'periodic nested attribute %d at sigma/period = ' ...
                     '%.4g: above %g the minimum-image centres route no ' ...
                     'longer computes the declared full-image measure. ' ...
                     'Pass wrap = ''single-image'' on this attribute to ' ...
                     'ask for the minimum-image measure, or use ' ...
                     'method = ''auto''.'], a, sigma / period, ...
                    internal.relPerSigmaOverPThreshold(ts));
            end
        elseif ~isempty(forced) && ~strcmp(forced, 'centres')
            % wrap = 'single-image' above the threshold. Python has no
            % such branch because 'centres' is the only route its public
            % API can force; this harness can force the grid routes, and
            % forcing one here would silently return the other measure.
            error('cosSimExpTens:centresUnavailable', ...
                ['forceRoute = ''%s'' cannot be honoured on relative-' ...
                 'periodic nested attribute %d at sigma/period = %.4g ' ...
                 'under wrap = ''single-image'': above %g that wrap ' ...
                 'declares the minimum-image measure, which only the ' ...
                 'centres route computes.'], forced, a, sigma / period, ...
                internal.relPerSigmaOverPThreshold(ts));
        end
        route = admissible{1};
        return;
    end
    if strcmp(forced, 'centres')
        route = 'centres';
        return;
    end
    % Below the threshold both readings agree inside the truncation floor,
    % so every route of the right mode carries the declared measure and a
    % forced name is honoured as given.
    if ~isempty(forced)
        route = forced;
        return;
    end
    if numel(admissible) == 1
        route = admissible{1};
        return;
    end
    route = internal.nestedCost('priceNestedAttr', densX, densY, a, ...
                                admissible, ts, skipXX, skipYY);
end


function localCheckForcedMode(forced, a, isRel, isPer)
%LOCALCHECKFORCEDMODE  Reject a forced route that this attribute's mode has
%   no carrier for. 'centres' applies in every mode; 'contract' is the
%   absolute single-node contraction; 'contract_relnonper' the relative
%   line grid; 'taugrid' the relative-periodic period grid. Forcing a
%   route across modes would silently change the measure, so it errors.
    switch forced
        case 'centres',            ok = true;
        case 'contract',           ok = ~isRel;
        case 'contract_relnonper', ok = isRel && ~isPer;
        case 'taugrid',            ok = isRel && isPer;
        otherwise
            error('cosSimExpTens:centresUnavailable', ...
                ['unknown forceRoute ''%s''; expected ''centres'', ' ...
                 '''contract'', ''contract_relnonper'' or ''taugrid''.'], ...
                forced);
    end
    if ~ok
        if isRel
            relStr = 'relative';
        else
            relStr = 'absolute';
        end
        if isPer
            perStr = 'periodic';
        else
            perStr = 'non-periodic';
        end
        error('cosSimExpTens:centresUnavailable', ...
            ['forceRoute = ''%s'' cannot be honoured on nested attribute ' ...
             '%d: the attribute is %s-%s, whose measure that route does ' ...
             'not carry.'], forced, a, relStr, perStr);
    end
end


% ----------------------------------------------------------------------
%  The plan against the joint-tuple enumeration
% ----------------------------------------------------------------------
function [skipXX, skipYY] = selfIpSkipFlags(cacheX, cacheY, normalize)
%SELFIPSKIPFLAGS  Which self inner products this call need not compute.
%   Mirror of the Python cosine._nested_self_ip_skip_flags.
%
%   A self inner product that is already memoised on its density, or that
%   the requested normalisation does not consume, costs nothing at call
%   time and must not be priced. As on the flat path the flags are shared
%   by the two sides of the comparison --- the contraction plan and the
%   joint-tuple enumeration --- rather than read off each side's own
%   memo; see INTERNAL.SELFIPMEMOISED for why an asymmetric flag locks
%   the first winner in, and what the shared flag trades for that.
    needXX = strcmp(normalize, 'cosine');
    skipXX = ~needXX || internal.selfIpMemoised(cacheX);
    skipYY = internal.selfIpMemoised(cacheY);
end


function tf = nestedEnumerationAdmissible(densX, densY, ts)
%NESTEDENUMERATIONADMISSIBLE  True when the joint-tuple enumeration
%   carries the declared measure. Mirror of the Python
%   cosine._nested_enumeration_admissible. The wrap is read from both
%   densities (WRAPPAIR), which error where they disagree.
%
%   The enumeration evaluates the minimum-image wrapped-difference
%   kernel on a relative-periodic attribute --- measure (A). It may
%   therefore serve a wrap = 'full-image' attribute only below the
%   sigma/period threshold, where the two readings agree inside the
%   truncation floor, and serves a wrap = 'single-image' attribute at any
%   sigma/period. This is the rule
%   INTERNAL.SELECTMAINNERPRODUCTMETHOD applies to Bulger's method on the
%   flat path, read off the density instead of a wrap vector.
    tf = true;
    limit = internal.relPerSigmaOverPThreshold(ts);
    for a = 1:double(densX.nAttrs)
        if ~(logical(densX.isRel(a)) && logical(densX.isPer(a)))
            continue;
        end
        if strcmp(wrapPair(densX, densY, a), 'single-image')
            continue;
        end
        period = double(densX.period(a));
        if period > 0 && double(densX.sigma(a)) / period > limit
            tf = false;
            return;
        end
    end
end


function tf = nestedPrefersEnumeration(densX, densY, routesByAttr, ...
                                       ts, skipXX, skipYY)
%NESTEDPREFERSENUMERATION  True when the enumeration is priced cheaper
%   than the planned routes. Mirror of the Python
%   cosine._nested_prefers_enumeration.
%
%   ROUTESBYATTR is a 1-by-A cell array holding, for each nested
%   attribute, the route already chosen for it by the measure rule and
%   the per-attribute cost model, and '' or '-' for an attribute that is
%   not nested; the plan is therefore priced as what would actually run
%   rather than re-raced here. The comparison mirrors the flat selector's
%   Bulger-versus-Moebius one and records its prices in
%   INTERNAL.LASTNESTEDCOSTS.
    A = double(densX.nAttrs);
    admByAttr = cell(1, A);
    for a = 1:min(A, numel(routesByAttr))
        rt = routesByAttr{a};
        if isempty(rt) || strcmp(rt, '-')
            continue;
        end
        admByAttr{a} = {rt};
    end
    % The plan and the enumeration are priced against one pair of skip
    % flags, as the flat selector's two routes are: they memoise into
    % different cache keys, so a warm plan memo does not literally spare
    % the enumeration its self matrices, but pricing each side against
    % its own memo decides the comparison on which side ran first and
    % locks that first choice in (INTERNAL.SELFIPMEMOISED).
    [chosen, planMs, enumMs, detail] = internal.nestedCost( ...
        'selectNestedMethod', densX, densY, admByAttr, ...
        nestedEnumerationAdmissible(densX, densY, ts), ts, skipXX, skipYY);
    internal.lastNestedCosts(struct('chosen', chosen, 'planMs', planMs, ...
                                    'enumMs', enumMs, 'detail', detail));
    tf = strcmp(chosen, 'bulger');
end


function s = routeSignature(route, quad)
%ROUTESIGNATURE  Route plus the shared tau grid's identity, for the
%   self-IP memo key. Twin of the Python (route, tau_sig) tuple: the node
%   count and the two endpoints pin the grid, and a value taken under one
%   grid must never be reused under another.
    if isempty(quad) || ~isfield(quad, 'taus') || isempty(quad.taus)
        s = route;
        return;
    end
    t = quad.taus;
    s = sprintf('%s|%d|%.17g|%.17g', route, numel(t), t(1), t(end));
end


% ----------------------------------------------------------------------
%  Self-IP memo access. The cache is COSSIMEXPTENS's 'keys'/'vals' struct;
%  the key format lives in INTERNAL.SELFIPKEY so the two files agree.
% ----------------------------------------------------------------------
function [hit, val] = cacheGet(cache, key)
    hit = false;
    val = [];
    if ~isstruct(cache) || ~isfield(cache, 'keys') || ~isfield(cache, 'vals')
        return;
    end
    idx = find(strcmp(cache.keys, key), 1);
    if ~isempty(idx)
        hit = true;
        val = cache.vals(idx);
    end
end


function cache = cacheSet(cache, key, val)
    if ~isstruct(cache) || ~isfield(cache, 'keys') || ~isfield(cache, 'vals')
        cache = struct('keys', {{}}, 'vals', []);
    end
    idx = find(strcmp(cache.keys, key), 1);
    if isempty(idx)
        cache.keys{end + 1} = key;
        cache.vals(end + 1) = val;
    else
        cache.vals(idx) = val;
    end
end


function v = localOptField(opts, name, default)
%LOCALOPTFIELD  Defensive read of an optional opts field.
    v = default;
    if isstruct(opts) && isfield(opts, name) && ~isempty(opts.(name))
        v = opts.(name);
    end
end


% ----------------------------------------------------------------------
function s = tripSum(recipeA, recipeB, PA, WA, PB, WB, sigma, period, ts, quad, sym)
    % Sum of the per-event-pair inner products over the pair grid.
    %
    % sym=true (self inner products): <e_i,e_j> = <e_j,e_i>, so evaluate
    % only the upper triangle and double the off-diagonal terms. recipeA /
    % recipeB index the PA / PB axes of the rectangular kernel.
    %
    % The contraction machinery is already batched over its leading axis
    % (contractNode reads Q = size(K, 1) and every helper below it is
    % generic in Q), which the absolute mode uses with Q = 1 and the
    % relative-periodic mode with Q = the transposition count. Folding the
    % event pairs into that same axis therefore needs no change to the
    % contraction itself: the pair grid is assembled into one kernel and
    % reduced in a single call per chunk, as on the Python side.
    %
    % The relative-non-periodic mode keeps the per-pair route, because its
    % factored shortcut (ipRelNonperFactored) is chosen per pair and has no
    % batched form.
    if nargin < 11; sym = false; end
    if batchableMode(recipeA, recipeB, PA, WA, PB, WB, quad)
        s = tripSumBatched(recipeA, recipeB, PA, WA, PB, WB, ...
                           sigma, period, ts, quad, sym);
    else
        s = tripSumLooped(recipeA, recipeB, PA, WA, PB, WB, ...
                          sigma, period, ts, quad, sym);
    end
end


% ----------------------------------------------------------------------
function s = tripSumBatched(recipeA, recipeB, PA, WA, PB, WB, ...
                            sigma, period, ts, quad, sym)
    % Sum over the event-pair grid, evaluated in batches.
    [mi, ni] = pairIndices(size(PA, 2), size(PB, 2), sym);
    mult = ones(numel(mi), 1);
    if sym
        % Upper triangle only, so off-diagonal pairs stand for two terms.
        mult(:) = 2.0;
        mult(mi == ni) = 1.0;
    end
    v = pairValuesBatched(recipeA, recipeB, PA, WA, PB, WB, ...
                          sigma, period, ts, quad, mi, ni);
    s = sum(v .* mult);
end


% ----------------------------------------------------------------------
function [mi, ni] = pairIndices(nA, nB, sym)
    % Event-pair index lists. Under sym only the upper triangle is listed;
    % the caller supplies the multiplicity or scatters the transpose.
    if sym
        spans = max(nB - (1:nA) + 1, 0);
        nPairs = sum(spans);
        mi = zeros(nPairs, 1);
        ni = zeros(nPairs, 1);
        p = 0;
        for i = 1:nA
            span = spans(i);
            if span == 0; continue; end
            idx = p + (1:span);
            mi(idx) = i;
            ni(idx) = i:nB;
            p = p + span;
        end
    else
        mi = repelem((1:nA).', nB, 1);
        ni = repmat((1:nB).', nA, 1);
    end
end


% ----------------------------------------------------------------------
function v = pairValuesBatched(recipeA, recipeB, PA, WA, PB, WB, ...
                               sigma, period, ts, quad, mi, ni)
    % Per-event-pair inner products for the listed pairs, with the pairs
    % folded into the contraction's leading batch axis.
    %
    % contractNode and everything below it read the batch extent from
    % size(K, 1), so a batch carrying pairs (and, in relative-periodic mode,
    % pairs times transpositions) needs no change to the contraction. This
    % is the MATLAB form of the Python nested_attr_matrix, which folds the
    % same grid into the leading axis of its contraction.
    nPairs = numel(mi);
    nX = size(PA, 1);
    nY = size(PB, 1);
    v = zeros(nPairs, 1);
    if nPairs == 0
        return;
    end

    isRelPer = strcmp(quad.mode, 'relper');
    isRelNon = strcmp(quad.mode, 'relnonper');
    if isRelPer || isRelNon
        taus = quad.taus(:).';
        T = numel(taus);
    else
        taus = [];
        T = 1;
    end

    % Chunk so the assembled kernel stays within budget: it holds
    % T * nX * nY doubles per pair.
    memBudget = 16e6;
    chunk = max(1, min(nPairs, floor(memBudget / max(T * nX * nY, 1))));

    for c0 = 1:chunk:nPairs
        c1 = min(c0 + chunk - 1, nPairs);
        nb = c1 - c0 + 1;
        cm = mi(c0:c1);
        cn = ni(c0:c1);

        vx = PA(:, cm).';               % (nb, nX)
        vy = PB(:, cn).';               % (nb, nY)
        wx = WA(:, cm).';
        wy = WB(:, cn).';

        if isRelNon
            % Per-pair window into the line grid: taus outside it give
            % kernel entries below the truncation floor, which are zeroed
            % and then summed as exact zeros, so the window changes nothing
            % but the amount of arithmetic.
            [tw, valid] = tauWindow(vx, vy, taus, sigma, ts);
            if isempty(tw)
                tw = repmat(taus, nb, 1);
                valid = [];
            end
            W = size(tw, 2);
            d = reshape(vx, [nb, nX, 1, 1]) ...
                - (reshape(vy, [nb, 1, nY, 1]) + reshape(tw, [nb, 1, 1, W]));
            K = exp(-d.^2 / (4 * sigma^2));
            K = K .* (reshape(wx, [nb, nX, 1, 1]) .* reshape(wy, [nb, 1, nY, 1]));
            K = permute(K, [1, 4, 2, 3]);
            K = reshape(K, [nb * W, nX, nY]);
            K = truncK(K, ts);
            vc = contractNode(recipeA, recipeB, K);
            vc = reshape(vc, [nb, W]);
            if ~isempty(valid); vc = vc .* valid; end
            vc = sum(vc, 2);                     % common dtau cancels
        elseif isRelPer
            d = reshape(vx, [nb, nX, 1, 1]) ...
                - (reshape(vy, [nb, 1, nY, 1]) + reshape(taus, [1, 1, 1, T]));
            K = relPerKernel(d, sigma, period, ts, quad);
            K = K .* (reshape(wx, [nb, nX, 1, 1]) .* reshape(wy, [nb, 1, nY, 1]));
            % (nb, nX, nY, T) -> (nb, T, nX, nY) -> (nb*T, nX, nY). The
            % merge is column-major, so the pair index runs fastest; the
            % reshape below inverts it the same way.
            K = permute(K, [1, 4, 2, 3]);
            K = reshape(K, [nb * T, nX, nY]);
            K = truncK(K, ts);
            vc = contractNode(recipeA, recipeB, K);
            vc = sum(reshape(vc, [nb, T]), 2);   % common dtau cancels
        else
            d = reshape(vx, [nb, nX, 1]) - reshape(vy, [nb, 1, nY]);
            K = absKernel(d, sigma, period, ts, quad);
            K = K .* (reshape(wx, [nb, nX, 1]) .* reshape(wy, [nb, 1, nY]));
            K = truncK(K, ts);
            vc = contractNode(recipeA, recipeB, K);
            vc = vc(:);
        end

        v(c0:c1) = vc;
    end
end


% ----------------------------------------------------------------------
function tf = batchableMode(recipeX, recipeY, PA, WA, PB, WB, quad)
    % Whether the pair grid can be evaluated in batches.
    %
    % Absolute and relative-periodic modes always can. The relative-non-
    % periodic mode can whenever its closed-form shortcut cannot apply,
    % since that shortcut is chosen per pair and has no batched form: the
    % two structural gates are read from the recipes, and the third asks
    % whether every cell carries a shared leaf template. Where the shortcut
    % could fire the per-pair route is kept, so nothing is given up.
    tf = true;
    if any(strcmp(quad.mode, {'abs', 'relper'}))
        return;
    end
    if ~strcmp(quad.mode, 'relnonper')
        tf = false;
        return;
    end
    if recipeX.sym || recipeY.sym
        return;                        % shortcut needs ordered cells
    end
    if recipeX.r ~= numel(recipeX.children) ...
            || recipeY.r ~= numel(recipeY.children)
        return;                        % shortcut needs the whole cell
    end
    for i = 1:size(PA, 2)
        if isempty(sharedLeafTemplate(recipeX, PA(:, i), WA(:, i)))
            return;
        end
    end
    for j = 1:size(PB, 2)
        if isempty(sharedLeafTemplate(recipeY, PB(:, j), WB(:, j)))
            return;
        end
    end
    tf = false;                        % every cell can use the shortcut
end


% ----------------------------------------------------------------------
function [tw, valid] = tauWindow(vxT, vyT, taus, sigma, ts)
    % Per-pair slice of a uniform line tau-grid, or [] when not worthwhile.
    %
    % On the line the grid spans the whole value range, because any two
    % events may be that far apart, but one event pair aligns only over the
    % taus near its own offset. A kernel entry survives truncation when
    % |v_x - v_y - tau| <= 2*sigma*sqrt(-log(floor)), so the window runs
    % from min(v_x) - max(v_y) to max(v_x) - min(v_y), widened by that
    % margin at each end. Every node outside is zeroed by truncK and then
    % summed as an exact zero.
    %
    % All windows share one width so the pairs stay in a single batch: only
    % the start index varies, and valid masks the tail where a window
    % overruns its own end at the grid edges.
    tw = [];
    valid = [];
    T = numel(taus);
    if T < 3
        return;
    end
    step = taus(2) - taus(1);
    if ~isfinite(step) || step <= 0
        return;                        % not a uniform ascending grid
    end
    if isempty(ts) || ~isfinite(ts)
        return;                        % without truncation no node is removable
    end
    floorv = exp(-0.5 * ts^2);
    if ~(floorv > 0 && floorv < 1)
        return;
    end
    margin = 2 * sigma * sqrt(-log(floorv));

    t0 = taus(1);
    lo = min(vxT, [], 2) - max(vyT, [], 2) - margin;      % (nb, 1)
    hi = max(vxT, [], 2) - min(vyT, [], 2) + margin;
    startI = min(max(floor((lo - t0) / step), 0), T - 1);
    stopI  = min(max(ceil((hi - t0) / step), 0), T - 1);
    W = max(stopI - startI) + 1;
    if W >= T
        return;                        % the window is the grid; nothing saved
    end

    idx = startI + (0:W - 1);                            % (nb, W), 0-based
    valid = double(idx <= stopI);
    idx = min(idx, T - 1) + 1;                           % to 1-based
    tw = taus(idx);                                      % (nb, W)
end


% ----------------------------------------------------------------------
function s = tripSumLooped(recipeA, recipeB, PA, WA, PB, WB, ...
                           sigma, period, ts, quad, sym)
    % Per-event-pair route, retained for the relative-non-periodic mode.
    s = 0.0;
    nA = size(PA, 2);
    nB = size(PB, 2);
    for i = 1:nA
        ai = PA(:, i);
        wi = WA(:, i);
        if sym; j0 = i; else; j0 = 1; end
        for j = j0:nB
            v = nestedIp(recipeA, recipeB, ai, PB(:, j), wi, WB(:, j), ...
                         sigma, period, ts, quad);
            if sym && j ~= i
                s = s + 2.0 * v;
            else
                s = s + v;
            end
        end
    end
end


% ----------------------------------------------------------------------
%  Multi-attribute: nested factor(s) via the contraction, plain factors via
%  the per-attribute MA matrix; combine per event-pair (JMM Eq 3.4).
% ----------------------------------------------------------------------
function [triple, routes, cacheX, cacheY] = nestedContractMA( ...
        densX, densY, normalize, truncationSigmas, force, forceRoute, ...
        methodName, cacheX, cacheY, routesOnly)
%NESTEDCONTRACTMA  MA cosine when one or more attributes are nested or
%   ordered. Mirror of the Python cosine._try_nested_contract_ma.
%
%   The MAET cross-event inner product factorises per event-pair across
%   attributes (JMM Eq 3.4): <X,Y> = sum_{i,j} prod_a I_a(i,j). Each
%   attribute contributes an (N_x, N_y) per-event-pair inner matrix. A
%   nested attribute goes through the per-attribute dispatch
%   (NESTEDATTRPLAN / NESTEDATTRMATRICES), with the route decided once so
%   its xy, xx and yy share one measure and, on a relative-periodic
%   attribute, decided by the declared wrap rather than by cost wherever
%   the two differ. An ordered-flat attribute ([sym] = false, r > 1, not
%   nested) goes through the centres path (MOBIUS.CLOSEDFORMATTRCENTRES /
%   CLOSEDFORMATTRMATRIXFROM): a single ordered level has no symmetric
%   orbit to reduce, and routing it through the orbit/Moebius matrix would
%   wrongly symmetrise it (summing its full S_r orbit). Flat-symmetric and
%   r = 1 attributes go through the orbit/Moebius per-attribute matrix
%   (MOBIUS.MAPERATTRINNERMATRIX). The matrices multiply element-wise then
%   sum, mirroring LOCALCOSSIMMAORBIT in COSSIMEXPTENS.

    triple = [];
    routes = {};
    if ~any(strcmp(normalize, {'cosine', 'oneSidedDenom'}))
        declineContractIfForced(force, methodName, ...
            sprintf('unsupported normalisation ''%s''', normalize));
        return;
    end
    A = densX.nAttrs;
    % Resolve the truncation width through the shared accuracy-floor
    % resolver (see the entry note above): Inf -> the finite accuracy-floor
    % width, not unbounded exact summation.
    ts = internal.accuracyFloor('resolve', truncationSigmas);
    % Bind the accuracy budget for this call; warnings fire once per
    % scope rather than once per block.
    orbitGuard('begin', ts);
    guardCleanup = onCleanup(@() orbitGuard('end', []));  %#ok<NASGU>
    N_x = densX.N;
    N_y = densY.N;

    % ---- Pass 1: validate and plan. The per-attribute routes and shared
    % quadrature grids are settled before any matrix is formed, so the two
    % self inner products can be looked up in the memo before the work that
    % would produce them is done. Splitting the pass is what makes the
    % memoisation possible at all: the MA self inner product is a product
    % across *all* attributes, so its cache key is not known until every
    % attribute has been planned.
    kinds = cell(1, A);
    attrRoutes = cell(1, A);
    quads = cell(1, A);
    % The skip flags say which self inner products this call will
    % actually compute, so a memoised (or unconsumed) one is not priced.
    [skipXX, skipYY] = selfIpSkipFlags(cacheX, cacheY, normalize);
    for a = 1:A
        isNestedX = isfield(densX, 'nested') && iscell(densX.nested) ...
            && numel(densX.nested) >= a && ~isempty(densX.nested{a});
        isNestedY = isfield(densY, 'nested') && iscell(densY.nested) ...
            && numel(densY.nested) >= a && ~isempty(densY.nested{a});
        if isNestedX ~= isNestedY
            declineContractIfForced(force, methodName, ...
                'an attribute is nested on only one side');
            return;
        end
        if isNestedX
            specX = densX.nested{a};
            specY = densY.nested{a};
            if localInnerR(specX) ~= 0 || localInnerR(specY) ~= 0
                declineContractIfForced(force, methodName, ...
                    'an inner/intermediate [rel] unit is not yet covered');
                return;
            end
            if ~isequal(double(specX.r(:)).', double(specY.r(:)).') ...
                    || ~isequal(logical(specX.sym(:)).', ...
                                logical(specY.sym(:)).')
                declineContractIfForced(force, methodName, ...
                    'the two nested attributes differ in [r]/[sym]');
                return;
            end
            kinds{a} = 'nested';
            [attrRoutes{a}, quads{a}] = nestedAttrPlan(densX, densY, a, ...
                                           forceRoute, ts, skipXX, skipYY);
        elseif localIsOrderedFlat(densX, a) || localIsOrderedFlat(densY, a)
            kinds{a} = 'ordered';
            attrRoutes{a} = '-';
        else
            kinds{a} = 'flat';
            attrRoutes{a} = '-';
        end
    end
    % The plan is priced against the joint-tuple enumeration, as in the
    % one-attribute plan: the per-attribute routes chosen above are
    % what the plan would run, and their prices plus the flat companions'
    % are what the enumeration has to beat. Declining here returns the
    % caller to the enumeration. A forced method is never diverted, and
    % neither is a routes-only plan (EXPLAINDISPATCH prices the two sides
    % itself, and wants the plan's routes either way).
    if ~force && ~routesOnly && nestedPrefersEnumeration(densX, densY, ...
            attrRoutes, ts, skipXX, skipYY)
        internal.lastNestedRoutes({});
        triple = [];
        routes = {};
        return;
    end
    routes = attrRoutes;
    internal.lastNestedRoutes(routes);
    if routesOnly
        return;
    end

    % The self-IP memo key carries every attribute's kind, route and shared
    % grid, for the reason the one-attribute plan records: a grid route
    % discretises the self inner product too, and a different partner can
    % widen the grid, so a value taken under one grid must never be reused
    % under another. wrap is part of each density's own contents, so it
    % needs no separate key entry.
    sigParts = cell(1, A);
    for a = 1:A
        sigParts{a} = sprintf('%s:%d:%s', kinds{a}, a, ...
                              routeSignature(attrRoutes{a}, quads{a}));
    end
    key = internal.selfIpKey('contract_ma', ts, strjoin(sigParts, ','));
    [xxHit, xxVal] = cacheGet(cacheX, key);
    [yyHit, yyVal] = cacheGet(cacheY, key);
    % <X,X> is consumed by the cosine only: under 'oneSidedDenom' it is
    % neither formed nor memoised, as on the flat routes.
    needXX = strcmp(normalize, 'cosine');
    formXX = needXX && ~xxHit;

    P_xy = ones(N_x, N_y);
    P_xx = ones(N_x, N_x);
    P_yy = ones(N_y, N_y);

    % ---- Pass 2: form the matrices.
    for a = 1:A
        sigma  = densX.sigma(a);
        isRel  = logical(densX.isRel(a));
        isPer  = logical(densX.isPer(a));
        period = densX.period(a);
        r_a    = densX.r(a);
        wrapA  = wrapPair(densX, densY, a);

        switch kinds{a}
            case 'flat'
                % Flat-symmetric or r = 1: the orbit/Moebius per-attribute
                % matrix, which correctly symmetrises these readings. The
                % attribute's declared wrap rides through, as it does at
                % the flat MA call site in COSSIMEXPTENS; omitting it here
                % read every periodic attribute as full-image whatever the
                % user had declared.
                Px = densX.pAttr{a}; Wx = densX.w{a};
                Py = densY.pAttr{a}; Wy = densY.w{a};
                P_xy = P_xy .* mobius.maPerAttrInnerMatrix(Px, Wx, Py, Wy, ...
                    sigma, r_a, isRel, isPer, period, ...
                    'truncationSigmas', ts, 'wrap', wrapA);
                if formXX
                    P_xx = P_xx .* mobius.maPerAttrInnerMatrix(Px, Wx, ...
                        Px, Wx, sigma, r_a, isRel, isPer, period, ...
                        'truncationSigmas', ts, 'wrap', wrapA);
                end
                if ~yyHit
                    P_yy = P_yy .* mobius.maPerAttrInnerMatrix(Py, Wy, ...
                        Py, Wy, sigma, r_a, isRel, isPer, period, ...
                        'truncationSigmas', ts, 'wrap', wrapA);
                end

            case 'ordered'
                % Ordered flat ([sym] = false, r > 1, not nested): the
                % materialised centres, which read the attribute's own
                % stored tuples. MOBIUS.MAPERATTRINNERMATRIX would sum the
                % full S_r orbit and so symmetrise an attribute the user
                % asked to keep ordered -- the defect this branch fixes.
                % Mirror of the ordered-flat branch of the Python
                % _try_nested_contract_ma.
                [cxB, cacheX] = internal.nestedCentresMemoised( ...
                    cacheX, densX, a);
                [cyB, cacheY] = internal.nestedCentresMemoised( ...
                    cacheY, densY, a);
                P_xy = P_xy .* mobius.closedFormAttrMatrixFrom( ...
                    cxB, cyB, wrapA, ts);
                if formXX
                    P_xx = P_xx .* mobius.closedFormAttrMatrixFrom( ...
                        cxB, cxB, wrapA, ts);
                end
                if ~yyHit
                    P_yy = P_yy .* mobius.closedFormAttrMatrixFrom( ...
                        cyB, cyB, wrapA, ts);
                end

            otherwise   % 'nested'
                [Ixy, Ixx, Iyy, cacheX, cacheY] = nestedAttrMatrices( ...
                    densX, densY, a, attrRoutes{a}, quads{a}, ts, ...
                    formXX, ~yyHit, cacheX, cacheY);
                P_xy = P_xy .* Ixy;
                if formXX; P_xx = P_xx .* Ixx; end
                if ~yyHit; P_yy = P_yy .* Iyy; end
        end
    end

    if xxHit
        ip_xx = xxVal;
    elseif ~needXX
        ip_xx = [];
    else
        ip_xx = sum(P_xx(:));
        cacheX = cacheSet(cacheX, key, ip_xx);
    end
    if yyHit
        ip_yy = yyVal;
    else
        ip_yy = sum(P_yy(:));
        cacheY = cacheSet(cacheY, key, ip_yy);
    end

    % The enumeration was already offered its chance, above: the plan is
    % priced against it before any matrix is formed, and a plan that
    % reaches here won that comparison (or was forced). method = 'bulger'
    % on an MA nested density agrees with this route to floating point in
    % every mode, which tests/test_nested_measure_rule.m checks, so the
    % choice between them is a cost choice and nothing else --- with the
    % one measure caveat NESTEDENUMERATIONADMISSIBLE carries, that the
    % enumeration computes the minimum-image reading of a
    % relative-periodic attribute.
    triple = struct('xy', sum(P_xy(:)), 'xx', ip_xx, 'yy', ip_yy);
end


function tf = localIsOrderedFlat(dens, a)
%LOCALISORDEREDFLAT  A non-nested attribute the user asked to keep
%   ordered ([sym] = false) at r > 1. Densities built before isSym
%   existed default to the symmetric reading, unchanged.
    tf = false;
    if double(dens.r(a)) <= 1
        return;
    end
    if isfield(dens, 'isSym') && numel(dens.isSym) >= a
        tf = ~logical(dens.isSym(a));
    end
end


function [Ixy, Ixx, Iyy, cacheX, cacheY] = nestedAttrMatrices( ...
        densX, densY, a, route, quad, ts, wantXX, wantYY, cacheX, cacheY)
%NESTEDATTRMATRICES  The (N_x, N_y), (N_x, N_x) and (N_y, N_y) inner
%   matrices for one nested attribute, on the ROUTE and shared QUAD that
%   NESTEDATTRPLAN settled (passed in so xy, xx and yy share one measure
%   and one grid). Mirror of three calls to the Python
%   cosine._nested_attr_matrix; grouped here so the recipes, the NaN-fill
%   and the centres bundles are built once.
%
%   WANTXX / WANTYY false skips a self matrix whose value is already
%   memoised; the skipped output is [] and the caller must not read it.
    Ixy = [];  Ixx = [];  Iyy = [];
    if nargin < 9;  cacheX = struct('keys', {{}}, 'vals', []); end
    if nargin < 10; cacheY = struct('keys', {{}}, 'vals', []); end
    wrapA = wrapPair(densX, densY, a);
    if strcmp(route, 'centres')
        % Centres bundles memoised on each density's memo struct (see
        % INTERNAL.NESTEDCENTRESMEMOISED); returned so the caller can
        % thread them back to the density.
        [cxB, cacheX] = internal.nestedCentresMemoised(cacheX, densX, a);
        [cyB, cacheY] = internal.nestedCentresMemoised(cacheY, densY, a);
        Ixy = mobius.closedFormAttrMatrixFrom(cxB, cyB, wrapA, ts);
        if wantXX
            Ixx = mobius.closedFormAttrMatrixFrom(cxB, cxB, wrapA, ts);
        end
        if wantYY
            Iyy = mobius.closedFormAttrMatrixFrom(cyB, cyB, wrapA, ts);
        end
        return;
    end

    specX = densX.nested{a};
    specY = densY.nested{a};
    isRel  = logical(densX.isRel(a));
    isPer  = logical(densX.isPer(a));
    sigma  = densX.sigma(a);
    period = densX.period(a);
    PXa = double(densX.pAttr{a});
    PYa = double(densY.pAttr{a});
    rLevels   = double(specX.r(:)).';
    symLevels = logical(specX.sym(:)).';
    tagsX = orientTags(double(specX.tags), size(PXa, 1), numel(rLevels));
    tagsY = orientTags(double(specY.tags), size(PYa, 1), numel(rLevels));
    sameStruct = isequal(size(tagsX), size(tagsY)) && isequal(tagsX, tagsY);
    WXa = densX.w{a}; if isempty(WXa); WXa = ones(size(PXa)); end
    WYa = densY.w{a}; if isempty(WYa); WYa = ones(size(PYa)); end
    WXa = double(WXa);
    WYa = double(WYa);
    % Variable-K (NaN-padded) values: fill with an in-range value at
    % weight zero (exactly equivalent; see the one-attribute plan).
    mXa = isnan(PXa);
    mYa = isnan(PYa);
    if any(mXa(:)) || any(mYa(:))
        fillVal = min(min(PXa(:), [], 'omitnan'), ...
            min(PYa(:), [], 'omitnan'));
        PXa(mXa) = fillVal;  WXa(mXa | isnan(WXa)) = 0;
        PYa(mYa) = fillVal;  WYa(mYa | isnan(WYa)) = 0;
    end
    recipeX = buildRecipe(rLevels, symLevels, tagsX, isRel, isPer);
    if sameStruct
        recipeY = recipeX;
    else
        recipeY = buildRecipe(rLevels, symLevels, tagsY, isRel, isPer);
    end
    % (The former mpt:nestedSurrogateResolution warning is gone: see the
    % note at the single-attribute site above.)
    Ixy = nestedAttrInnerMatrix(recipeX, recipeY, PXa, PYa, WXa, WYa, ...
                                sigma, period, ts, quad, false);
    if wantXX
        Ixx = nestedAttrInnerMatrix(recipeX, recipeX, PXa, PXa, WXa, WXa, ...
                                    sigma, period, ts, quad, true);
    end
    if wantYY
        Iyy = nestedAttrInnerMatrix(recipeY, recipeY, PYa, PYa, WYa, WYa, ...
                                    sigma, period, ts, quad, true);
    end
end


function M = nestedAttrInnerMatrix(recipeA, recipeB, Pa, Pb, Wa, Wb, ...
                                   sigma, period, ts, quad, symmetric)
    % (N_a, N_b) per-event-pair inner matrix for one nested attribute via the
    % tree contraction. symmetric exploits <e_i,e_j> = <e_j,e_i> for the self
    % matrices. The per-attribute prefactor is constant and cancels in the
    % cosine when the attribute matrices are multiplied and summed.
    %
    % The absolute and relative-periodic modes evaluate the whole pair grid
    % in batches; the relative-non-periodic mode keeps the per-pair route,
    % whose factored shortcut is chosen per pair and has no batched form.
    na = size(Pa, 2);
    nb = size(Pb, 2);
    M = zeros(na, nb);

    if batchableMode(recipeA, recipeB, Pa, Wa, Pb, Wb, quad)
        [mi, ni] = pairIndices(na, nb, symmetric);
        v = pairValuesBatched(recipeA, recipeB, Pa, Wa, Pb, Wb, ...
                              sigma, period, ts, quad, mi, ni);
        M(sub2ind([na, nb], mi, ni)) = v;
        if symmetric
            M(sub2ind([na, nb], ni, mi)) = v;
        end
        return;
    end

    for i = 1:na
        ai = Pa(:, i); wi = Wa(:, i);
        if symmetric; j0 = i; else; j0 = 1; end
        for j = j0:nb
            v = nestedIp(recipeA, recipeB, ai, Pb(:, j), wi, Wb(:, j), ...
                         sigma, period, ts, quad);
            M(i, j) = v;
            if symmetric && j ~= i; M(j, i) = v; end
        end
    end
end


function declineContractIfForced(force, methodName, reason)
%DECLINECONTRACTIFFORCED  Raise on a case the plan does not cover, when
%   the user forced a method that names the plan; stay silent under
%   method = 'auto', where the caller falls back to the enumeration.
%   METHODNAME is quoted so the message names the method the user asked
%   for ('contract', 'mobius' or 'centres'), not always 'contract'.
    if force
        error('cosSimExpTens:contractUnavailable', ...
            ['method=''%s'' is not available here: %s. Use ' ...
             'method=''auto'' or method=''bulger''.'], methodName, reason);
    end
end

function r = localInnerR(spec)
    r = 0;
    if isstruct(spec) && isfield(spec, 'proj') && isfield(spec, 'relUnit') ...
            && (strcmp(spec.proj, 'inner') || strcmp(spec.proj, 'intermediate'))
        u = spec.relUnit;
        r = prod(spec.r(1:u));
    end
end


% ----------------------------------------------------------------------
%  Recipe: tag tree + permutation/combination index arrays (built once)
% ----------------------------------------------------------------------
function recipe = buildRecipe(rLevels, symLevels, tags, isRel, isPer)
    L = numel(rLevels);
    Ktot = size(tags, 1);
    recipe = buildNode(L - 1, (1:Ktot).', rLevels, symLevels, tags, ...
                       isRel, isPer);
end


function node = buildNode(level, valIdx, rLevels, symLevels, tags, isRel, isPer)
    valIdx = valIdx(:);
    if level == 0
        r0 = rLevels(1);
        sy0 = symLevels(1);
        useOrb = orbitEligible(numel(valIdx), r0, sy0, isRel, isPer);
        if useOrb
            xt = zeros(0, r0); yt = zeros(0, r0);   % lazy: orbit needs no tuples
        else
            [xt, yt] = tupleIndices(numel(valIdx), r0, sy0);
        end
        node = struct('level', 0, 'valIdx', valIdx, 'children', {{}}, ...
                      'xtup', xt, 'ytup', yt, 'r', r0, 'sym', sy0, ...
                      'useOrbit', useOrb);
        return;
    end
    col = level;                       % 1-based tag column (Python col=level-1)
    keys = tags(valIdx, col);
    uk = unique(keys);                 % ascending
    children = cell(1, numel(uk));
    for c = 1:numel(uk)
        sub = valIdx(keys == uk(c));
        children{c} = buildNode(level - 1, sub, rLevels, symLevels, tags, ...
                                isRel, isPer);
    end
    rl = rLevels(level + 1);           % Python r_levels[level]
    syl = symLevels(level + 1);
    useOrb = orbitEligible(numel(children), rl, syl, isRel, isPer);
    if useOrb
        xt = zeros(0, rl); yt = zeros(0, rl);
    else
        [xt, yt] = tupleIndices(numel(children), rl, syl);
    end
    node = struct('level', level, 'valIdx', valIdx, 'children', {children}, ...
                  'xtup', xt, 'ytup', yt, 'r', rl, 'sym', syl, ...
                  'useOrbit', useOrb);
end


function tf = orbitEligible(K, r, sym, isRel, isPer) %#ok<INUSD>
    % Is the Mobius reduction *structurally* available at this level?
    %
    % Structure only: the level must be symmetric and r within the shipped
    % orbit order. Whether the Mobius route is also the *faster* one is a
    % separate question, settled in combinePair, because it turns on the
    % batch extent and no block exists yet when the recipe is built. The
    % crossover moves by up to 11 in K across the batch range, so a
    % decision taken here could not express it.
    %
    % Precision is not judged here either. A size margin between K and r
    % is the wrong variable: at r = 2, K = r + 1 the orbit error is
    % 4e-16. The estimate in combinePair measures the error the
    % computation actually incurred and compares it against the accuracy
    % the caller asked for.
    ORBIT_R_MAX_SHIPPED = 8;     % match Python _ORBIT_MAX_R
    tf = sym && r >= 2 && r <= ORBIT_R_MAX_SHIPPED;
end




function ts = admittingSigmas(bound)
    % Largest truncationSigmas whose floor would admit an error bound.
    %
    % The guard admits the orbit route when truncationFloor(ts) >= bound,
    % both being absolute quantities on the value scale, and that floor is
    % exp(-ts^2 / 2), so the condition inverts to ts <= sqrt(-2 log(bound)).
    % Empty when bound is at or above 1 -- the top of the normalised value
    % scale -- where no positive setting satisfies it.
    if ~(bound > 0 && bound < 1)
        ts = [];
        return;
    end
    ts = sqrt(-2 * log(bound));
end


function [xt, yt] = tupleIndices(n, r, sym)
    % X side: permutations of r-combinations if sym, else combinations.
    % Y side: combinations. 1-based indices into 1:n.
    if r > n
        xt = zeros(0, r);
        yt = zeros(0, r);
        return;
    end
    if r == 1
        C = (1:n).';
    else
        C = nchoosek(1:n, r);          % (nC x r), each row a combination
    end
    yt = C;
    if sym && r > 1
        P = perms(1:r);                % (r! x r)
        nC = size(C, 1);
        nP = size(P, 1);
        xt = zeros(nC * nP, r);
        idx = 1;
        for p = 1:nP
            xt(idx:idx + nC - 1, :) = C(:, P(p, :));
            idx = idx + nC;
        end
    else
        xt = C;
    end
end


% ----------------------------------------------------------------------
%  Contraction (vectorised over the quadrature batch, dim 1)
% ----------------------------------------------------------------------
function v = contractNode(xn, yn, K)
    % Bottom-up, batched over the quadrature (dim 1) AND over sibling pairs.
    % K is (Q, nX, nY): the X axis is indexed by xn values, the Y axis by yn
    % values. For XX/YY (and equal-cardinality XY) xn and yn coincide and this
    % is the original single-tree walk; differing leaf spans are handled per
    % level by combinePair's per-size tuple sourcing.
    if xn.level == 0
        block = K(:, xn.valIdx, yn.valIdx);
        v = combinePair(block, xn.r, xn.sym, xn.useOrbit && yn.useOrbit);
    else
        Mc = subtreeOverlaps(xn.children, yn.children, K);
        v = combinePair(Mc, xn.r, xn.sym, xn.useOrbit && yn.useOrbit);
    end
end


function s = nodeSpan(node)
    if node.level == 0
        s = numel(node.valIdx);
    else
        s = numel(node.children);
    end
end


function tf = siblingsUniform(nodes)
    rep = nodes{1};
    span = nodeSpan(rep);
    tf = true;
    for k = 1:numel(nodes)
        nd = nodes{k};
        if nodeSpan(nd) ~= span || nd.r ~= rep.r || nd.sym ~= rep.sym ...
                || nd.useOrbit ~= rep.useOrbit
            tf = false; return;
        end
    end
end


function M = leafOverlaps(xnodes, ynodes, K)
    % (Q, gx, gy) pairwise overlaps among leaf siblings, X-side vs Y-side.
    gx = numel(xnodes);
    gy = numel(ynodes);
    Q = size(K, 1);
    nX = size(K, 2);
    nY = size(K, 3);
    r = xnodes{1}.r;
    sym = xnodes{1}.sym;
    if r == 1
        % r0 = 1: M(q,a,b) = sum_{i in Sxa, j in Syb} K(q,i,j) (weights folded).
        Gx = zeros(gx, nX);
        for a = 1:gx
            Gx(a, xnodes{a}.valIdx) = 1.0;
        end
        Gy = zeros(gy, nY);
        for b = 1:gy
            Gy(b, ynodes{b}.valIdx) = 1.0;
        end
        KG = reshape(reshape(K, [Q * nX, nY]) * Gy.', [Q, nX, gy]);  % (q,i,b)
        KGp = reshape(permute(KG, [2, 1, 3]), [nX, Q * gy]);          % (i, q*b)
        MG = Gx * KGp;                                                % (a, q*b)
        M = permute(reshape(MG, [gx, Q, gy]), [2, 1, 3]);            % (Q,gx,gy)
        return;
    end
    ux = siblingsUniform(xnodes);
    uy = siblingsUniform(ynodes);
    if ux && uy
        mx = numel(xnodes{1}.valIdx);
        my = numel(ynodes{1}.valIdx);
        blocks = zeros(gx, gy, Q, mx, my);
        for a = 1:gx
            sa = K(:, xnodes{a}.valIdx, :);
            for b = 1:gy
                blocks(a, b, :, :, :) = reshape(sa(:, :, ynodes{b}.valIdx), ...
                                                [1, 1, Q, mx, my]);
            end
        end
        useOrbit = xnodes{1}.useOrbit && ynodes{1}.useOrbit;
        vals = combinePair(reshape(blocks, [gx * gy * Q, mx, my]), ...
                           r, sym, useOrbit);
        M = permute(reshape(vals, [gx, gy, Q]), [3, 1, 2]);
        return;
    end
    M = zeros(Q, gx, gy);
    for a = 1:gx
        for b = 1:gy
            uo = xnodes{a}.useOrbit && ynodes{b}.useOrbit;
            M(:, a, b) = combinePair(K(:, xnodes{a}.valIdx, ynodes{b}.valIdx), ...
                                     r, sym, uo);
        end
    end
end


function M = subtreeOverlaps(xnodes, ynodes, K)
    % (Q, gx, gy) pairwise overlaps among sibling subtrees, X-side vs Y-side.
    if xnodes{1}.level == 0
        M = leafOverlaps(xnodes, ynodes, K);
        return;
    end
    gx = numel(xnodes);
    gy = numel(ynodes);
    Q = size(K, 1);
    r = xnodes{1}.r;
    sym = xnodes{1}.sym;
    xsizes = zeros(1, gx);
    xflat = {};
    for k = 1:gx
        xsizes(k) = numel(xnodes{k}.children);
        xflat = [xflat, xnodes{k}.children];   %#ok<AGROW>
    end
    ysizes = zeros(1, gy);
    yflat = {};
    for k = 1:gy
        ysizes(k) = numel(ynodes{k}.children);
        yflat = [yflat, ynodes{k}.children];   %#ok<AGROW>
    end
    xoffs = [0, cumsum(xsizes)];
    yoffs = [0, cumsum(ysizes)];
    Mc = subtreeOverlaps(xflat, yflat, K);          % (Q, Gcx, Gcy)
    ux = siblingsUniform(xnodes);
    uy = siblingsUniform(ynodes);
    if ux && uy
        gcx = xsizes(1);
        gcy = ysizes(1);
        blocks = zeros(gx, gy, Q, gcx, gcy);
        for a = 1:gx
            ra = xoffs(a) + 1 : xoffs(a) + gcx;
            for b = 1:gy
                cb = yoffs(b) + 1 : yoffs(b) + gcy;
                blocks(a, b, :, :, :) = reshape(Mc(:, ra, cb), ...
                                                [1, 1, Q, gcx, gcy]);
            end
        end
        useOrbit = xnodes{1}.useOrbit && ynodes{1}.useOrbit;
        vals = combinePair(reshape(blocks, [gx * gy * Q, gcx, gcy]), ...
                           r, sym, useOrbit);
        M = permute(reshape(vals, [gx, gy, Q]), [3, 1, 2]);
        return;
    end
    M = zeros(Q, gx, gy);
    for a = 1:gx
        ra = xoffs(a) + 1 : xoffs(a + 1);
        for b = 1:gy
            cb = yoffs(b) + 1 : yoffs(b + 1);
            uo = xnodes{a}.useOrbit && ynodes{b}.useOrbit;
            M(:, a, b) = combinePair(Mc(:, ra, cb), r, sym, uo);
        end
    end
end


% ----------------------------------------------------------------------
function v = combineChunked(M, xtup, ytup)
    % Enumerated combine, chunked over the batch to bound peak memory.
    % Feasibility is judged on the work before this is called; chunking
    % only keeps the materialised array within reach.
    maxElems = 16e6;
    Q = size(M, 1);
    per = max(size(xtup, 1) * size(ytup, 1), 1);
    step = max(1, floor(maxElems / per));
    if step >= Q
        v = combine(M, xtup, ytup);
        return;
    end
    v = zeros(Q, 1);
    for s = 1:step:Q
        e = min(s + step - 1, Q);
        v(s:e) = combine(M(s:e, :, :), xtup, ytup);
    end
end


% ----------------------------------------------------------------------
function out = orbitGuard(cmd, arg)
    % Accuracy budget for one batched call, and the once-per-call warning
    % flags. The Moebius (orbit) reduction sums signed terms that largely
    % cancel, so its answer carries fewer digits than the terms it was built
    % from; enumeration sums only non-negative terms and loses nothing.
    % The rounding error is estimated as
    %
    %     eps * sum|term| / r!
    %
    % on the scale combineOrbit returns, where sum|term| is reported by
    % innerProductOrbitGrid, so the estimate costs nothing to evaluate.
    % It is an estimate rather than a guaranteed limit; see the note at
    % combineOrbit's return site.
    %
    % Its conservatism has been measured on the Python implementation
    % only, by tools/calibrate_orbit_cancellation.py. Rounding error
    % depends on the order in which the terms are accumulated, and the
    % two implementations do not accumulate in the same order, so those
    % figures do not transfer to this side. Calibrate with
    % tools/calibrateOrbitCancellation.m before relying on them here.
    persistent state
    switch cmd
        case 'begin'
            state = struct('floor', arg, 'sigmas', arg, ...
                           'warnedCost', false, 'warnedAcc', false, ...
                           'enabled', logical(mptDefaults('postHocGuards')));
            out = [];
        case 'end'
            state = [];
            out = [];
        case 'get'
            out = state;
        case 'set'
            state = arg;
            out = [];
    end
end


% ----------------------------------------------------------------------
function w = enumWork(Q, gx, gy, r)
    % Kernel products the enumerated route performs for this block. Peak
    % memory can be capped by chunking, but the work cannot, so feasibility
    % is judged on the work.
    if r > gx || r > gy
        w = Inf;
        return;
    end
    w = Q * nchoosek(gx, r) * factorial(r) * nchoosek(gy, r);
end


function v = combinePair(M, r, sym, useOrbit)
    % Combine a (Q, gx, gy) block at one level: X-side perm tuples over gx,
    % Y-side comb tuples over gy (the r!-cancelled perm x comb form, same
    % scale as combine). gx and gy are read from the block, so unequal X/Y
    % spans -- ragged siblings *or* two densities whose nested cardinalities
    % differ -- are handled directly. For a square block with gx == gy this
    % reproduces the old combineNode exactly, so the X == Y path is unchanged.
    % Both enumerated routes -- the one taken when the level is not
    % orbit-eligible and the guard's fallback -- chunk over the batch, so
    % peak memory is bounded whatever the tuple counts.
    gx = size(M, 2);
    gy = size(M, 3);
    if useOrbit
        % The recipe said the Mobius route is structurally available here.
        % Whether it is also the faster one depends on the batch extent,
        % which only exists now, so the cost question is settled here. K
        % is taken as max(gx, gy), matching how the warning strings report
        % the shape; the model was fitted on square blocks, so a markedly
        % ragged level is outside what it was validated on.
        useOrbit = internal.orbitCostModel(r, max(gx, gy), size(M, 1));
    end
    if useOrbit
        [v, bound] = combineOrbit(M, r);
        budget = orbitGuard('get', []);
        if isempty(budget); return; end
        if ~budget.enabled
            % postHocGuards is off. The check below inspects a result that
            % has already been computed and, when it diverts, pays for the
            % enumerated route on top of this one -- so with it active the
            % measured cost of the Mobius route is not the cost of choosing
            % it. Calibration runs switch it off so the two routes can be
            % timed as the alternatives they are.
            return;
        end
        % The bound and the truncation floor are both absolute quantities on
        % the value scale, which is the single error measure the toolbox
        % judges accuracy by. Comparing them directly is the whole test; a
        % ratio to the returned value would reintroduce a denominator that
        % legitimately approaches zero.
        if bound <= budget.floor
            return;                      % inside the requested accuracy
        end
        admit = admittingSigmas(bound);
        work = enumWork(size(M, 1), gx, gy, r);
        if work <= 16e6
            if ~budget.warnedCost
                budget.warnedCost = true;
                orbitGuard('set', budget);
                head = sprintf(['The Mobius route''s error bound (%.1e) ' ...
                    'exceeds the accuracy implied by truncationSigmas = ' ...
                    '%.4g (%.1e), so enumeration was used instead.'], ...
                    bound, budget.sigmas, budget.floor);
                if isempty(admit)
                    tail = [' The bound is at or above the value scale ' ...
                            'itself, so no truncationSigmas setting would ' ...
                            'admit the Mobius route here.'];
                elseif internal.orbitCostModel(r, max(gx, gy), size(M, 1))
                    % The Mobius route is the cheaper one at this level's
                    % sizes, so trading accuracy for it does buy speed.
                    tail = sprintf([' Setting truncationSigmas to %.3g or ' ...
                        'below would admit the Mobius route, which is the ' ...
                        'faster of the two at r = %d, K = %d, at the cost ' ...
                        'of an error that may exceed the tighter figure.'], ...
                        admit, r, max(gx, gy));
                else
                    % Enumeration is also the cheaper route at these sizes,
                    % so there is nothing to be gained by loosening the
                    % budget; saying otherwise would offer a false trade.
                    tail = sprintf([' Loosening truncationSigmas would not ' ...
                        'help: enumeration is also the faster of the two ' ...
                        'at r = %d, K = %d.'], r, max(gx, gy));
                end
                warning('mpt:nestedOrbitCost', '%s%s', head, tail);
            end
            [xt, ~] = tupleIndices(gx, r, sym);
            [~, yt] = tupleIndices(gy, r, sym);
            v = combineChunked(M, xt, yt);
            return;
        end
        if ~budget.warnedAcc
            budget.warnedAcc = true;
            orbitGuard('set', budget);
            head = sprintf(['The Mobius route''s error bound (%.1e) ' ...
                'exceeds the accuracy implied by truncationSigmas = ' ...
                '%.4g (%.1e), and enumeration is not feasible at r = %d, ' ...
                'K = %d. The returned value may carry an error above ' ...
                '%.1e.'], bound, budget.sigmas, budget.floor, r, ...
                max(gx, gy), budget.floor);
            if isempty(admit)
                % The bound is at or above the value scale itself, so no
                % truncation setting can accommodate it; saying otherwise
                % would offer a lever that cannot help.
                tail = [' The bound is at or above the value scale itself, ' ...
                        'so no truncationSigmas setting would admit this ' ...
                        'route; the weight profile is too steeply peaked ' ...
                        'for the Mobius reduction at this tuple size.'];
            else
                tail = sprintf([' Setting truncationSigmas to %.3g or ' ...
                    'below would bring the requested accuracy within the ' ...
                    'bound this is judged against.'], admit);
            end
            warning('mpt:nestedOrbitAccuracy', '%s%s', head, tail);
        end
        return;
    end
    [xt, ~] = tupleIndices(gx, r, sym);
    [~, yt] = tupleIndices(gy, r, sym);
    v = combineChunked(M, xt, yt);
end


function [v, bound] = combineOrbit(M, r)
    % (B,) = Sum_{cX,cY} perm(M[cX,cY]) via the partition-lattice orbit
    % reduction (= innerProductOrbitGrid / r!), vectorised over the leading
    % batch. Supports rectangular M (gx ~= gy).
    %
    % ratios is requested only because termMass depends on it; the
    % cancellation ratio informs no decision. Accuracy is judged solely by
    % the absolute bound returned here, against the truncation floor on the
    % value scale.
    gx = size(M, 2);
    gy = size(M, 3);
    [vals, ~, ~, termMassSum] = mobius.innerProductOrbitGrid(M, ...
        ones(gx, 1), ones(gy, 1), r, 'prefactor', 1.0, ...
        'returnCancellationRatio', true);
    vals = vals(:) / factorial(r);
    if isempty(termMassSum)
        bound = 0.0;
    else
        % Estimated rounding error of the alternating sum, as eps times
        % the sum of the terms' magnitudes.
        %
        % This is NOT a guaranteed upper limit. Sequential summation of
        % n terms admits (n-1) * eps * sum|term| in the worst case, and
        % both this estimate and the |Omega_r| * eps * max|term| it
        % replaces sit below that. Each is a heuristic; this one was
        % measured against enumeration over four weight profiles and
        % r = 3..7 and came out 2x to 44x above the true error, where
        % its predecessor ran 5x to 1100x above. Preferred because it is
        % the better-validated of the two, not because it is provably
        % safe. Widening to the guaranteed limit would refuse the Mobius
        % route almost everywhere.
        bound = eps * max(abs(termMassSum(:))) / factorial(r);
    end
    v = vals;
end


function v = combine(M, xtup, ytup)
    Tx = size(xtup, 1);
    Ty = size(ytup, 1);
    if Tx == 0 || Ty == 0
        v = zeros(size(M, 1), 1);
        return;
    end
    r = size(xtup, 2);
    P = M(:, xtup(:, 1), ytup(:, 1));              % B x Tx x Ty
    for t = 2:r
        P = P .* M(:, xtup(:, t), ytup(:, t));
    end
    v = sum(sum(P, 3), 2);                          % B x 1
    v = v(:);
end


% ----------------------------------------------------------------------
%  Leaf-kernel batches + per-event-pair bare inner product
% ----------------------------------------------------------------------
function ipv = nestedIp(recipeX, recipeY, vX, vY, wX, wY, sigma, period, ts, quad)
    nX = numel(vX);
    nY = numel(vY);
    switch quad.mode
        case 'abs'
            % Periodicity comes from the density's [per] flag (carried in the
            % quadrature struct), not from whether period happens to be
            % finite: an absolute non-periodic attribute may carry a finite
            % period.
            d = reshape(vX, [1, nX, 1]) - reshape(vY, [1, 1, nY]);   % 1 x nX x nY
            K = absKernel(d, sigma, period, ts, quad);
            K = K .* (reshape(wX, [1, nX, 1]) .* reshape(wY, [1, 1, nY]));
            K = truncK(K, ts);
            v = contractNode(recipeX, recipeY, K);
            ipv = v(1);
        case 'relper'
            taus = quad.taus(:);
            T = numel(taus);
            d = reshape(vX, [1, nX, 1]) ...
                - (reshape(vY, [1, 1, nY]) + reshape(taus, [T, 1, 1]));  % T x nX x nY
            K = relPerKernel(d, sigma, period, ts, quad);
            K = K .* (reshape(wX, [1, nX, 1]) .* reshape(wY, [1, 1, nY]));
            K = truncK(K, ts);
            ipv = sum(contractNode(recipeX, recipeY, K));   % common dtau cancels
        case 'relnonper'
            taus = quad.taus(:);
            ipv = ipRelNonperFactored(recipeX, recipeY, vX, vY, wX, wY, ...
                                      sigma, ts, taus);
            if isempty(ipv)
                T = numel(taus);
                d = reshape(vX, [1, nX, 1]) ...
                    - (reshape(vY, [1, 1, nY]) + reshape(taus, [T, 1, 1]));
                K = exp(-d.^2 / (4 * sigma^2));               % no wrap
                K = K .* (reshape(wX, [1, nX, 1]) .* reshape(wY, [1, 1, nY]));
                K = truncK(K, ts);
                ipv = sum(contractNode(recipeX, recipeY, K));
            end
    end
end


function tpl = sharedLeafTemplate(node, v, w)
%SHAREDLEAFTEMPLATE Detect a spectral-augmentation leaf (mirror of the
%   Python _shared_leaf_template): a two-level node whose children are all
%   r == 1 leaves sharing one partial template (a common offset and weight
%   profile, translated per child by a single reference value). Returns a struct with
%   fields refVals/off/wt, or [] when the node is not of this form.
    tpl = [];
    if node.level ~= 1 || isempty(node.children)
        return;
    end
    rep = node.children{1};
    if rep.level ~= 0 || rep.r ~= 1 || ~isempty(rep.children)
        return;
    end
    s0 = rep.valIdx(:);
    width = numel(s0);
    if width < 2                       % Kp == 1 is a plain fundamental: leave
        return;                        % it on the generic path (no change)
    end
    v0 = v(s0);
    w0 = w(s0);
    off = v0 - v0(1);
    nChildren = numel(node.children);
    refVals = zeros(nChildren, 1);
    for a = 1:nChildren
        ch = node.children{a};
        if ch.level ~= 0 || ch.r ~= 1 || ~isempty(ch.children)
            return;
        end
        sa = ch.valIdx(:);
        if numel(sa) ~= width
            return;
        end
        va = v(sa);
        if ~isequal(va - va(1), off) || ~isequal(w(sa), w0)
            return;
        end
        refVals(a) = va(1);
    end
    tpl = struct('refVals', refVals, 'off', off(:), 'wt', w0(:));
end


function ipv = ipRelNonperFactored(recipeX, recipeY, vX, vY, wX, wY, ...
                                   sigma, ts, taus)
%IPRELNONPERFACTORED Closed-form inner-partial reduction of the relative-non-
%   periodic inner product for spectrally-augmented ordered cells (mirror of
%   the Python _ip_rel_nonper_factored). The inner partial index sums into the
%   template cross-correlation g, and the cell overlap reduces to the reference-value
%   differences: sum_tau prod_a g(refX_a - refY_a - tau). Returns [] when
%   the structure is not of this form (then the caller uses the generic path).
    ipv = [];
    if recipeX.sym || recipeY.sym
        return;                        % need ordered cells (outer [sym] = 0)
    end
    if recipeX.r ~= numel(recipeX.children) ...
            || recipeY.r ~= numel(recipeY.children)
        return;                        % need the whole cell as one ordered tuple
    end
    tx = sharedLeafTemplate(recipeX, vX, wX);
    ty = sharedLeafTemplate(recipeY, vY, wY);
    if isempty(tx) || isempty(ty)
        return;
    end
    r = numel(tx.refVals);
    if r ~= numel(ty.refVals)         % diagonal needs equal cell lengths
        return;
    end
    dpq = tx.off - ty.off.';                          % Kx x Ky
    wpq = tx.wt * ty.wt.';                            % Kx x Ky
    Kx = size(dpq, 1);
    Ky = size(dpq, 2);
    T = numel(taus);
    delta = reshape(tx.refVals - ty.refVals, [1, r]) ...
            - reshape(taus, [T, 1]);                  % T x r
    arg = reshape(delta, [T, r, 1, 1]) ...
          + reshape(dpq, [1, 1, Kx, Ky]);             % T x r x Kx x Ky
    K = exp(-arg.^2 / (4 * sigma^2)) .* reshape(wpq, [1, 1, Kx, Ky]);
    if ~isempty(ts) && isfinite(ts)
        floorv = exp(-0.5 * ts^2);     % per-term floor, matching truncK exactly
        K(K < floorv) = 0;
    end
    mDiag = sum(sum(K, 4), 3);                        % T x r
    ipv = sum(prod(mDiag, 2));         % common dtau cancels in the cosine
end


function K = truncK(K, ts)
    if isempty(ts) || ~isfinite(ts)
        return;
    end
    floorv = exp(-0.5 * ts^2);
    K(K < floorv) = 0;
end


% ----------------------------------------------------------------------
%  Quadrature (shared across event-pairs and the IP triple)
% ----------------------------------------------------------------------
function K = absKernel(d, sigma, period, ts, quad)
    % Per-coordinate absolute kernel on the differences d. Periodicity comes
    % from the density's [per] flag (carried in the quadrature struct), not
    % from whether period happens to be finite.
    %
    % Absolute-periodic full-image: the per-coordinate 1-D kernel is the
    % wrapped Gaussian theta(d) = sum_n exp(-(d + n P)^2 / (4 sigma^2)), the
    % same object the flat abs-per path and mobius.closedFormAttrMatrixFrom
    % compute through internal.wrappedGaussian1d. Reducing d to the nearest
    % image and exponentiating drops every image but the nearest, which is
    % the 'single-image' measure: it agrees with the full-image kernel only
    % while the accuracy floor puts the image count at zero (sigma/P below
    % ~0.05 at the default truncation) and departs above it (5e-5 in the
    % cosine at sigma/P = 0.1, 4e-2 at 0.2, measured in Python). Going
    % through the shared helper keeps the image-sum / Fourier choice
    % identical to the flat path's. Mirror of the Python
    % _nested_contraction._nested_attr_matrix_impl.
    if quad.isPer && ~strcmp(quad.wrap, 'single-image')
        K = internal.wrappedGaussian1d(d, sigma, period, ts, 4);
        return;
    end
    if quad.isPer
        d = d - period * round(d / period);
    end
    K = exp(-d.^2 / (4 * sigma^2));
end


function K = relPerKernel(d, sigma, period, ts, quad)
    % Per-coordinate kernel under one transposition for the relative-periodic
    % all-image route. The transposition average of the wrapped Gaussian
    % theta is the lattice-sum (full-image) measure this route is
    % documented to compute, and the one the flat Moebius integrator
    % computes (it sums relPerImageCount images before averaging for the
    % same reason). Averaging the nearest-image Gaussian instead is a
    % different measure once the floor asks for one image or more: 1.4e-5
    % in the cosine at sigma/P = 0.1, 1.8e-3 at 0.2 (measured in Python,
    % whose branch this mirrors). theta is also smooth in tau, where the
    % nearest-image kernel has a kink at |d| = P/2 that costs the
    % trapezoidal rule its spectral convergence (1e-6 at sigma/P = 0.15 on
    % the auto grid). The 'single-image' opt-in keeps the nearest image.
    if ~strcmp(quad.wrap, 'single-image')
        K = internal.wrappedGaussian1d(d, sigma, period, ts, 4);
        return;
    end
    d = d - period * round(d / period);
    K = exp(-d.^2 / (4 * sigma^2));
end


function w = wrapOf(dens, a)
    % The attribute's declared wrap, 'full-image' where the density carries
    % none (legacy structs).
    w = 'full-image';
    if isfield(dens, 'wrap') && iscell(dens.wrap) && numel(dens.wrap) >= a ...
            && ~isempty(dens.wrap{a})
        w = char(dens.wrap{a});
    end
end


function w = wrapPair(densX, densY, a)
    % The wrap the two densities declare on attribute A. The wrap declares
    % a measure, and the measure of an inner product must be one thing,
    % so the two must agree wherever the wrap is read --- on a periodic
    % attribute. A disagreement there errors rather than being resolved
    % in favour of densX, which made the result depend on operand order.
    % On a non-periodic attribute the wrap axis has no meaning and is not
    % compared. Twin of the Python cosine._declared_wrap.
    w = wrapOf(densX, a);
    wy = wrapOf(densY, a);
    if ~strcmp(w, wy) && logical(densX.isPer(a))
        error('mpt:wrapMismatch', ...
            ['wrap mismatch on attribute %d: densX declares ''%s'' and ' ...
             'densY declares ''%s''. The wrap declares the measure, so ' ...
             'the two densities must declare the same wrap on every ' ...
             'periodic attribute.'], a, w, wy);
    end
end


function quad = makeQuadrature(isRel, isPer, sigma, period, vmin, vmax, ts, wrapA)
    % wrapA is the attribute's declared wrap ('full-image' by default, or
    % 'single-image'); it is read only by the absolute-periodic kernel.
    if nargin < 8 || isempty(wrapA)
        wrapA = 'full-image';
    end
    if ~isRel
        quad = struct('mode', 'abs', 'isPer', logical(isPer), ...
                      'wrap', char(wrapA));
        return;
    end
    if isfinite(ts)
        tol = max(exp(-0.5 * ts^2), 1e-12);
    else
        tol = 1e-12;
    end
    if isPer
        ntau = internal.autoNtauDefault(period, sigma, ts);
        t = linspace(0, period, ntau + 1);
        quad = struct('mode', 'relper', 'taus', t(1:end - 1), ...
                      'wrap', char(wrapA));                     % endpoint=false
    else
        spread = vmax - vmin;
        pad = (6 + 0.5 * max(0, -log10(max(tol, 1e-16)))) * sigma;
        hi = spread + pad;
        n = max(64, ceil(2 * hi / (sigma / 4)));
        quad = struct('mode', 'relnonper', 'taus', linspace(-hi, hi, n));
    end
end


function out = localNestedTerms(densX, densY, a, ts, skipXX, skipYY)
%LOCALNESTEDTERMS  Analytic per-route cost terms for attribute A.
%   Twin of the Python _nested_cost.nested_attr_terms plus
%   predict_nested_pairwise_kernel_size, built from this file's own
%   tupleCounts / recipeWork / quadNodes so the harness that fits the cost
%   model and the dispatch that consumes it read one set of counts.
%
%   Every term is summed over the inner matrices a cosine computes: the
%   cross matrix always, each self matrix unless its skip flag is set,
%   each carrying its own event-pair count. Both flags default false,
%   which is what the calibration harness wants -- it times cold
%   densities and so computes all three.
%
%   A may be a nested attribute or a flat one. A flat attribute reads
%   the r!*C(K, r) / C(K, r) counts (the r! dropped when it is ordered)
%   and takes their product as its contraction work, mirroring the
%   Python _attr_tuple_counts / works fallback; that is what prices an
%   ordered flat companion on the centres law.
    if nargin < 5 || isempty(skipXX); skipXX = false; end
    if nargin < 6 || isempty(skipYY); skipYY = false; end
    isRel  = logical(densX.isRel(a));
    isPer  = logical(densX.isPer(a));
    sigma  = double(densX.sigma(a));
    period = double(densX.period(a));
    Nx = double(densX.N);
    Ny = double(densY.N);

    isNestedA = isfield(densX, 'nested') && iscell(densX.nested) ...
        && numel(densX.nested) >= a && ~isempty(densX.nested{a});
    if isNestedA
        specX = densX.nested{a};
        specY = densY.nested{a};
        rLevels   = double(specX.r(:)).';
        symLevels = logical(specX.sym(:)).';
        tagsX = orientTags(double(specX.tags), size(densX.pAttr{a}, 1), numel(rLevels));
        tagsY = orientTags(double(specY.tags), size(densY.pAttr{a}, 1), numel(rLevels));
        [mPermX, mCombX] = tupleCounts(rLevels, symLevels, tagsX);
        [mPermY, mCombY] = tupleCounts(rLevels, symLevels, tagsY);
        mult = internal.nestedOrbitMult(rLevels, symLevels);
        rx = buildRecipe(rLevels, symLevels, tagsX, isRel, isPer);
        if isequal(size(tagsX), size(tagsY)) && isequal(tagsX, tagsY)
            ry = rx;
        else
            ry = buildRecipe(rLevels, symLevels, tagsY, isRel, isPer);
        end
        workX = recipeWork(rx);
        workY = recipeWork(ry);
    else
        % Flat attribute: no recipe to build, so the contraction work is
        % the tuple-count product, as Python's works fallback takes it.
        [mPermX, mCombX] = localAttrCounts(densX, a);
        [mPermY, mCombY] = localAttrCounts(densY, a);
        rA = double(densX.r(a));
        isSymA = true;
        if isfield(densX, 'isSym') && numel(densX.isSym) >= a
            isSymA = logical(densX.isSym(a));
        end
        if isSymA && rA > 1
            mult = factorial(rA);
        else
            mult = 1;
        end
        workX = mPermX * mCombX;
        workY = mPermY * mCombY;
    end
    restrOn = internal.combRestrictionEnabled();
    restrX = restrOn && mult >= 2 && mCombX > 0 && abs(mPermX - mult * mCombX) < 0.5;
    restrY = restrOn && mult >= 2 && mCombY > 0 && abs(mPermY - mult * mCombY) < 0.5;

    PXa = double(densX.pAttr{a});
    PYa = double(densY.pAttr{a});
    vmin = min(min(PXa(:), [], 'omitnan'), min(PYa(:), [], 'omitnan'));
    vmax = max(max(PXa(:), [], 'omitnan'), max(PYa(:), [], 'omitnan'));
    if isRel && isPer
        nTau = quadNodes(true, true, sigma, period, 0, period, ts);
    else
        nTau = 1;
    end
    if isRel && ~isPer
        nLine = quadNodes(true, false, sigma, period, vmin, vmax, ts);
    else
        nLine = 1;
    end

    % (side of X, side of Y, event-pair count) for the matrices this call
    % computes: the cross matrix always, each self matrix unless skipped.
    sides = [1 2 Nx * Ny];
    if ~skipXX; sides = [sides; 1 1 Nx * Nx]; end %#ok<AGROW>
    if ~skipYY; sides = [sides; 2 2 Ny * Ny]; end %#ok<AGROW>
    mPerm = [mPermX mPermY];
    restr = [restrX restrY];
    mComb = [mCombX mCombY];
    works = [workX workY];
    terms = struct('centres', 0, 'taugrid', 0, ...
                   'contract_relnonper', 0, 'contract', 0);
    for k = 1:size(sides, 1)
        sx = sides(k, 1); sy = sides(k, 2); pairs = sides(k, 3);
        if restr(sx)
            xSide = mComb(sx);
        else
            xSide = mPerm(sx);
        end
        w = max(works(sx), works(sy));
        terms.centres = terms.centres + pairs * xSide * mPerm(sy);
        terms.taugrid = terms.taugrid + pairs * nTau * w;
        terms.contract_relnonper = terms.contract_relnonper + pairs * nLine * w;
        terms.contract = terms.contract + pairs * w;
    end

    % Joint tuple-pair kernel size of the WHOLE density (every attribute),
    % which is what the enumeration builds: a nested attribute contributes
    % its tupleCounts pair, a flat one r! * C(K, r) / C(K, r) with the r!
    % dropped on an ordered attribute.
    permX = Nx; combX = Nx; permY = Ny; combY = Ny;
    for b = 1:densX.nAttrs
        [pxB, cxB] = localAttrCounts(densX, b);
        [pyB, cyB] = localAttrCounts(densY, b);
        permX = permX * pxB; combX = combX * cxB;
        permY = permY * pyB; combY = combY * cyB;
    end
    bulgerXY = permX * combY;
    bulgerXX = permX * combX;
    bulgerYY = permY * combY;
    % Always the full three matrices: the enumeration memoises its self
    % inner products under its own cache key, so it carries its own skip
    % flags, which the cost model applies to the components below rather
    % than to this total. (A warm contraction memo says nothing about
    % what the enumeration would have to compute.)
    bulger = bulgerXY + bulgerXX + bulgerYY;

    info = struct('mPermX', mPermX, 'mCombX', mCombX, 'mPermY', mPermY, ...
                  'mCombY', mCombY, 'restrictedX', restrX, ...
                  'restrictedY', restrY, 'workX', workX, 'workY', workY, ...
                  'nTau', nTau, 'nLine', nLine, ...
                  'totalOrder', double(densX.r(a)), 'Nx', Nx, 'Ny', Ny, ...
                  'bulgerXY', bulgerXY, 'bulgerXX', bulgerXX, ...
                  'bulgerYY', bulgerYY);
    out = struct('terms', terms, 'bulger', bulger, 'info', info);
end


function [mPerm, mComb] = localAttrCounts(dens, a)
%LOCALATTRCOUNTS  Per-event (perm, comb) tuple counts for any attribute.
    isNested = isfield(dens, 'nested') && iscell(dens.nested) ...
        && numel(dens.nested) >= a && ~isempty(dens.nested{a});
    if isNested
        spec = dens.nested{a};
        rLevels   = double(spec.r(:)).';
        symLevels = logical(spec.sym(:)).';
        tg = orientTags(double(spec.tags), size(dens.pAttr{a}, 1), numel(rLevels));
        [mPerm, mComb] = tupleCounts(rLevels, symLevels, tg);
        return;
    end
    rA = double(dens.r(a));
    K = size(dens.pAttr{a}, 1);
    isSym = true;
    if isfield(dens, 'isSym') && numel(dens.isSym) >= a
        isSym = logical(dens.isSym(a));
    end
    mComb = nchoosekCount(K, rA);
    if isSym && rA > 1
        mPerm = mComb * factorial(rA);
    else
        mPerm = mComb;
    end
end


function Q = quadNodes(isRel, isPer, sigma, period, vmin, vmax, ts)
    if ~isRel
        Q = 1;
        return;
    end
    if isfinite(ts)
        tol = max(exp(-0.5 * ts^2), 1e-12);
    else
        tol = 1e-12;
    end
    if isPer
        Q = internal.autoNtauDefault(period, sigma, ts);
    else
        spread = vmax - vmin;
        pad = (6 + 0.5 * max(0, -log10(max(tol, 1e-16)))) * sigma;
        Q = max(64, ceil(2 * (spread + pad) / (sigma / 4)));
    end
end


% ----------------------------------------------------------------------
%  Analytic tuple counts (elementary symmetric polynomials) + tree work
% ----------------------------------------------------------------------
function [mPerm, mComb] = tupleCounts(rLevels, symLevels, tags)
    L = numel(rLevels);
    Ktot = size(tags, 1);
    mPerm = countSide((1:Ktot).', L - 1, true, rLevels, symLevels, tags);
    mComb = countSide((1:Ktot).', L - 1, false, rLevels, symLevels, tags);
end


function c = countSide(valIdx, level, useSym, rLevels, symLevels, tags)
    if level == 0
        r0 = rLevels(1);
        c = nchoosekCount(numel(valIdx), r0);
        if useSym && symLevels(1)
            c = c * factorial(r0);
        end
        return;
    end
    col = level;
    keys = tags(valIdx, col);
    uk = unique(keys);
    subs = zeros(1, numel(uk));
    for iGrp = 1:numel(uk)
        sub = valIdx(keys == uk(iGrp));
        subs(iGrp) = countSide(sub, level - 1, useSym, rLevels, symLevels, tags);
    end
    rl = rLevels(level + 1);
    c = elemSym(subs, rl);
    if useSym && symLevels(level + 1)
        c = c * factorial(rl);
    end
end


function v = elemSym(xs, k)
    % Elementary symmetric polynomial e_k of xs (e_0 = 1; 0 if k > numel).
    e = zeros(1, k + 1);
    e(1) = 1;
    for ii = 1:numel(xs)
        for j = k + 1:-1:2
            e(j) = e(j) + e(j - 1) * xs(ii);
        end
    end
    v = e(k + 1);
end


function c = nchoosekCount(nn, kk)
    if kk < 0 || kk > nn
        c = 0;
        return;
    end
    c = 1;
    for ii = 0:kk - 1
        c = c * (nn - ii) / (ii + 1);
    end
    c = round(c);
end


function w = recipeWork(node)
    % Orbit-eligible symmetric levels are costed at the orbit reduction's
    % |Omega_r| * K^2 * r rather than the enumerated r! * C(K,r)^2, so the
    % dispatch reflects the route actually taken at each level (mirrors the
    % Python recipe_work).
    if node.useOrbit
        K = nodeSpan(node);
        w = numel(mobius.getOrbitTable(node.r)) * K * K * max(1, node.r);
    else
        w = size(node.xtup, 1) * size(node.ytup, 1) * max(1, node.r);
    end
    if node.level ~= 0
        K = numel(node.children);
        w = w + K * K;
        for c = 1:K
            w = w + recipeWork(node.children{c});
        end
    end
end


function tg = orientTags(tg, nValues, L)
    % Orient a tag array so values index dimension 1 and the L-1 inner tag
    % levels index dimension 2 (the convention buildRecipe expects, matching
    % the Python 1-D tags). A single-level (L == 2) spec is a vector -- a
    % MATLAB literal makes it a row -- so reshape it to a column; an already
    % oriented matrix is left as is, a transposed one is corrected.
    if isvector(tg)
        tg = tg(:);
    elseif size(tg, 1) ~= nValues && size(tg, 2) == nValues
        tg = tg.';
    end
    if size(tg, 1) ~= nValues && L >= 2   %#ok<BDLGI> defensive: keep values on dim 1
        tg = reshape(tg, nValues, []);
    end
end
