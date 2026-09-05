function varargout = nestedCost(cmd, varargin)
%INTERNAL.NESTEDCOST  Cost model for the nested-attribute inner product.
%
%   The nested path of COSSIMEXPTENS chooses, per attribute, among routes
%   that all carry the attribute's *declared* measure, and then chooses,
%   for the density as a whole, between the contraction plan and the
%   joint-tuple enumeration. Until this file existed those choices were
%   made on raw analytic operation counts --- mComb * mPerm against
%   nTau * work --- with no fitted constants, no memory guard, and no
%   enumeration candidate. This supplies the missing half, in the same
%   shape as the flat model of INTERNAL.SELECTMAINNERPRODUCTMETHOD:
%
%     * one fitted power law t_ms = exp(a) * term ^ b per route and
%       coarse structure key, keyed the way the flat RELROUTECOSTMS law
%       is keyed (there by tuple order, here by the nested attribute's
%       *total* tuple order);
%     * a per-route setup floor applied with MAX (never added), because
%       a multiplicative law carries no fixed cost and extrapolates
%       below what the route can do once the term is small;
%     * a working-set guard that diverts away from the materialising
%       route above CENTRES_WORKING_SET_SOFT_BUDGET, as the flat eval
%       selector's guard does;
%     * the self-inner-product skip flags, so a memoised <X,X> or <Y,Y>
%       is not priced.
%
%   THE MEASURE RULE IS NOT PART OF THIS MODEL. The routes admissible
%   for an attribute are settled first, by NESTEDADMISSIBLEROUTES inside
%   INTERNAL.NESTEDCONTRACT; the cost model only orders the survivors. A
%   cheaper route to a different number is not a cheaper route.
%
%   COMMANDS (mirroring the public functions of the Python twin
%   mpt/_tensor/_nested_cost.py):
%
%     MS = INTERNAL.NESTEDCOST('routeCostMs', ROUTE, TOTALORDER, TERM,
%                              NMATRICES)
%         Predicted wall time in ms for one route from its fitted law.
%         TERM is the route's own term (see the term notes below); it
%         already carries the event-pair counts and the matrix count, so
%         NMATRICES is used only by the floor. Twin of
%         nested_route_cost_ms.
%
%     BYTES = INTERNAL.NESTEDCOST('centresWorkingSetBytes', DENSX, DENSY,
%                                 A, TS)
%         Peak bundle size of the nested centres route for attribute A.
%         Twin of nested_centres_working_set_bytes.
%
%     [ROUTE, COSTMS, PRICES, INFO] = INTERNAL.NESTEDCOST( ...
%         'priceNestedAttr', DENSX, DENSY, A, ADMISSIBLE, TS, SKIPXX,
%         SKIPYY)
%         Price the admissible routes (a cell row, in the order the
%         caller wants ties broken) for nested attribute A. PRICES is a
%         struct with one field per admissible route; Inf there marks a
%         materialising route diverted by the working-set guard, which
%         is how the guard shows up in a report. Twin of
%         price_nested_attr.
%
%     [CHOSEN, PLANMS, ENUMMS, DETAIL] = INTERNAL.NESTEDCOST( ...
%         'selectNestedMethod', DENSX, DENSY, ADMISSIBLEBYATTR,
%         ENUMERATIONOK, TS, SKIPXX, SKIPYY)
%         Choose between the contraction plan ('contract') and the
%         joint-tuple enumeration ('bulger'). ADMISSIBLEBYATTR is a
%         1-by-A cell array whose entry A is the cell row of routes the
%         measure rule admits for nested attribute A and {} for an
%         attribute that is not nested. ENUMERATIONOK says whether the
%         enumeration carries the declared measure at all --- it
%         computes the minimum-image reading of a relative-periodic
%         attribute, so a wrap = 'full-image' attribute above the
%         sigma/period threshold rules it out, exactly as the flat
%         selector rules Bulger's method out there. DETAIL carries the
%         per-attribute (route, costMs, prices, info) in DETAIL.attr and
%         the flat companions' price in DETAIL.flat. Twin of
%         select_nested_method with return_costs.
%
%         The two sides share one pair of self-inner-product skip flags,
%         as the flat selector's two routes do. They memoise under
%         different cache keys, so a warm plan memo does not literally
%         spare the enumeration its self matrices; pricing each side
%         against its own memo nonetheless decides the comparison on
%         which side happened to run first rather than on what the two
%         sides cost, and locks that first choice in. See
%         INTERNAL.SELFIPMEMOISED.
%
%     S = INTERNAL.NESTEDCOST('enumSafety')
%         The near-tie safety factor favouring the contraction plan.
%
%   TERMS
%   -----
%   Every term is summed over the inner-product matrices this call will
%   actually compute --- the cross matrix always, each self matrix
%   unless its skip flag is set --- with each matrix carrying its own
%   event-pair count (Nx*Ny, Nx^2, Ny^2). They are computed by
%   INTERNAL.NESTEDCOSTTERMS, i.e. by INTERNAL.NESTEDCONTRACT's own
%   tupleCounts / recipeWork / quadNodes, so the calibration harness
%   (TESTS/BENCH_NESTED_COST), the dispatch and this model cannot drift
%   apart. Writing mPerm and mComb for the per-event permutation- and
%   combination-side tuple counts of the nested attribute:
%
%     'centres'   kernel entries: per event pair the route forms one
%                 (mCombX, mPermY) array where Bulger's orbit
%                 restriction holds and one (mPermX, mPermY) array where
%                 it does not, and the restriction is priced only in the
%                 cases MOBIUS.CLOSEDFORMATTRCENTRES admits it.
%     'taugrid'   nTau * work: transposition-average nodes times the
%                 per-level contraction work of the recipe (the larger
%                 of the two sides').
%     'contract_relnonper'
%                 the same product with the line grid's node count in
%                 place of nTau.
%     'contract'  the same product with one node: the absolute kernel
%                 needs no quadrature.
%     'bulger'    the joint tuple-pair kernel of the whole density,
%                 mirroring PREDICTPAIRWISEKERNELSIZE but with each
%                 nested attribute contributing its own mPerm / mComb in
%                 place of r!*C(K, r) / C(K, r).
%
%   The laws are per-language and per-machine: they absorb interpreter
%   overhead, array layout and BLAS constants, exactly as the flat laws
%   do.
%
%   See also INTERNAL.NESTEDCOSTTERMS, INTERNAL.NESTEDCONTRACT,
%   INTERNAL.LASTNESTEDCOSTS, INTERNAL.PREDICTORBITCOSTMS.

    % ------------------------------------------------------------------
    %  Calibrated cost-model constants
    % ------------------------------------------------------------------
    %  Fitted September 2026 on one run of TESTS/BENCH_NESTED_COST (258
    %  cells) on the maintainer's Mac, by
    %  python/tools/fit_nested_cost.py --lang matlab --min-cells 60 (a
    %  structure key with fewer than 60 measured cells for a route takes
    %  that route's pooled law, which is why the grid, contract and most
    %  centres rows repeat one value). Predicted over measured, geometric
    %  mean / spread / worst: centres 1.04 / 1.42 / 3.8, tau-grid 1.00 /
    %  1.48 / 3.1, line grid 1.00 / 1.56 / 3.1, contract 1.00 / 1.53 / 8.7
    %  (its law is flat in the term; the floor carries it), enumeration
    %  1.02 / 1.50 / 3.6. Routing regret against the measured oracle:
    %  per-attribute route 1.029 (13 cells beyond the oracle, worst 2.1x);
    %  plan against enumeration 1.055 (worst 2.5x). The Python-seeded
    %  values these replaced scored 1.133 and 1.155 on the same cells. The
    %  fitter's paste block emits exactly the names below, in this order,
    %  so a refit is a block replacement rather than an edit. What the
    %  dispatch consumes is the RATIO between two routes, which is far
    %  less sensitive to the machine than either number.
    %
    %  Cross-validated September 2026 with
    %  python/tools/fit_nested_cost.py --compare-pooling, which scores
    %  three pooling levels --- these per-key laws, one law per route
    %  with the structure key dropped, and one exponent shared by all
    %  five routes with per-route intercepts --- by held-out routing
    %  regret over 40 random halves. Per key wins on these MATLAB cells,
    %  1.044 +- 0.018 against 1.105 +- 0.035 (per route) and 1.128 +-
    %  0.035 (shared exponent), and wins by the same margin on the
    %  Python VM and sandbox grids, so the form generalises across all
    %  three datasets and both languages. Refitting these cells at the
    %  held-out-optimal --min-cells (30 to 40 for 258 cells) moves the
    %  regrets by 0.002 or less while adding law rows, so the values
    %  below were left as they are: the simplest fit within noise of the
    %  best. Prediction error does NOT transfer across machines --- the
    %  log-ratio error of one machine's constants on another's cells is
    %  around 1.4, a factor of four in absolute time --- but the routing
    %  regret does, at 1.03, which is the whole reason the model is
    %  consumed as a ratio.
    %
    %  Structure key: the nested attribute's *total* tuple order,
    %  prod(rLevels) --- the number of leaf positions a tuple carries,
    %  which is what dens.r(a) holds. The per-entry and per-node costs
    %  both grow with it (the relative quadratic form is O(R^2) per
    %  kernel entry, the centres array O(R) per tuple), so it is the one
    %  coarse index the routes share. Rows exist for the orders the
    %  calibration grid reaches; a lookup takes the largest row at or
    %  below the requested order, and orders below the smallest row use
    %  it. That is the nested analogue of the flat laws'
    %  min(max(r, 2), 4) clamp, generalised because a nested attribute's
    %  total order is a product of level arities and so skips values.
    NESTED_LAW_KEYS = [2, 3, 4, 6];

    % Fitted per-route laws, t_ms = exp(a) * term ^ b, by structure key.
    NESTED_COST_LAW_A_CENTRES = [-1.7847, -2.3985, -2.3985, -2.4964];
    NESTED_COST_LAW_B_CENTRES = [0.2427, 0.3487, 0.3487, 0.3964];
    NESTED_COST_LAW_A_TAUGRID = [-2.9453, -2.9453, -2.9453, -2.9453];
    NESTED_COST_LAW_B_TAUGRID = [0.2995, 0.2995, 0.2995, 0.2995];
    NESTED_COST_LAW_A_CONTRACT_RELNONPER = [-1.6570, -1.6570, -1.6570, -1.6570];
    NESTED_COST_LAW_B_CONTRACT_RELNONPER = [0.2085, 0.2085, 0.2085, 0.2085];
    NESTED_COST_LAW_A_CONTRACT = [-1.0484, -1.0484, -1.0484, -1.0484];
    NESTED_COST_LAW_B_CONTRACT = [0.0036, 0.0036, 0.0036, 0.0036];
    NESTED_COST_LAW_A_BULGER = [-2.4458, -2.6540, -2.6540, -2.4903];
    NESTED_COST_LAW_B_BULGER = [0.3305, 0.3791, 0.3791, 0.3982];

    % Per-route setup floor in milliseconds, as (fixed, perMatrix) by
    % structure key: a call computing NMATRICES of the three inner
    % matrices cannot go under fixed + perMatrix * nMatrices. Applied
    % with MAX, not added, for the reason the flat orbit floor gives:
    % the laws above are multiplicative in their term and so carry no
    % fixed cost, while the routes pay a recipe build, an orbit-table
    % fetch and a per-attribute dispatch whatever the term is. Taken
    % from the smallest cells of the calibration grid, where each
    % route's measured time is flat in the term. The harness always
    % computes all three inner matrices, so the split between the fixed
    % and the per-matrix half is not identified by the measurements and
    % the whole floor sits in the per-matrix half --- the conservative
    % reading, which discounts a memoised-self call rather than
    % over-charging it. Fitted with the laws above.
    NESTED_FLOOR_FIXED_CENTRES = [0, 0, 0, 0];
    NESTED_FLOOR_PER_MATRIX_CENTRES = [0.1373, 0.1437, 0.1437, 0.2328];
    NESTED_FLOOR_FIXED_TAUGRID = [0, 0, 0, 0];
    NESTED_FLOOR_PER_MATRIX_TAUGRID = [0.3356, 0.3356, 0.3356, 0.3356];
    NESTED_FLOOR_FIXED_CONTRACT_RELNONPER = [0, 0, 0, 0];
    NESTED_FLOOR_PER_MATRIX_CONTRACT_RELNONPER = [0.2399, 0.2399, 0.2399, 0.2399];
    NESTED_FLOOR_FIXED_CONTRACT = [0, 0, 0, 0];
    NESTED_FLOOR_PER_MATRIX_CONTRACT = [0.09213, 0.09213, 0.09213, 0.09213];
    NESTED_FLOOR_FIXED_BULGER = [0, 0, 0, 0];
    NESTED_FLOOR_PER_MATRIX_BULGER = [0.1652, 0.1696, 0.1696, 0.3078];

    % Safety factor favouring the contraction plan at near-ties: the
    % joint-tuple enumeration is taken only when its estimate is below
    % the plan's estimate divided by this factor. The asymmetry points
    % toward the route that does not materialise the joint tuple set,
    % for two reasons. The enumeration builds n_J = N * prod_a mPerm_a
    % tuples of sum_a r_a rows on each side, which the contraction never
    % does; a near-tie in predicted time is not a near-tie in memory.
    % And where the plan's relative-periodic attribute runs on the tau
    % grid, the plan computes the declared all-image measure EXACTLY
    % while the enumeration computes the minimum-image reading,
    % admissible below the sigma/period threshold only because the two
    % agree inside the truncation floor. Trading an exact measure for an
    % approximate one is worth a real saving, not a marginal one.
    %
    % Set at the scale of the near-crossover cells the calibration found
    % rather than fitted: on the small multi-attribute shapes where the
    % three routes are measured within about 1.5x of one another, the
    % model's ranking is not to be trusted to the percent.
    NESTED_ENUM_SAFETY = 2.0;

    % Working-set soft budget, in bytes. Same value and same units as
    % the flat eval selector's CENTRES_WORKING_SET_SOFT_BUDGET, so the
    % two guards read the same quantity the same way.
    CENTRES_WORKING_SET_SOFT_BUDGET = 256 * 1024^2;

    laws = struct( ...
        'centres', struct('a', NESTED_COST_LAW_A_CENTRES, ...
                          'b', NESTED_COST_LAW_B_CENTRES, ...
                          'f', NESTED_FLOOR_FIXED_CENTRES, ...
                          'pm', NESTED_FLOOR_PER_MATRIX_CENTRES), ...
        'taugrid', struct('a', NESTED_COST_LAW_A_TAUGRID, ...
                          'b', NESTED_COST_LAW_B_TAUGRID, ...
                          'f', NESTED_FLOOR_FIXED_TAUGRID, ...
                          'pm', NESTED_FLOOR_PER_MATRIX_TAUGRID), ...
        'contract_relnonper', ...
            struct('a', NESTED_COST_LAW_A_CONTRACT_RELNONPER, ...
                   'b', NESTED_COST_LAW_B_CONTRACT_RELNONPER, ...
                   'f', NESTED_FLOOR_FIXED_CONTRACT_RELNONPER, ...
                   'pm', NESTED_FLOOR_PER_MATRIX_CONTRACT_RELNONPER), ...
        'contract', struct('a', NESTED_COST_LAW_A_CONTRACT, ...
                           'b', NESTED_COST_LAW_B_CONTRACT, ...
                           'f', NESTED_FLOOR_FIXED_CONTRACT, ...
                           'pm', NESTED_FLOOR_PER_MATRIX_CONTRACT), ...
        'bulger', struct('a', NESTED_COST_LAW_A_BULGER, ...
                         'b', NESTED_COST_LAW_B_BULGER, ...
                         'f', NESTED_FLOOR_FIXED_BULGER, ...
                         'pm', NESTED_FLOOR_PER_MATRIX_BULGER));

    switch char(cmd)
        case 'enumSafety'
            varargout{1} = NESTED_ENUM_SAFETY;

        case 'lawKeys'
            varargout{1} = NESTED_LAW_KEYS;

        case 'routeCostMs'
            varargout{1} = localRouteCostMs(laws, NESTED_LAW_KEYS, ...
                varargin{1}, varargin{2}, varargin{3}, ...
                localArg(varargin, 4, 3));

        case 'centresWorkingSetBytes'
            [dX, dY, a] = deal(varargin{1}, varargin{2}, varargin{3});
            ts = localArg(varargin, 4, []);
            [~, ~, info] = internal.nestedCostTerms(dX, dY, a, ts);
            varargout{1} = localWorkingSetBytes(dX, a, info);

        case 'priceNestedAttr'
            [dX, dY, a, admissible] = deal(varargin{1}, varargin{2}, ...
                                           varargin{3}, varargin{4});
            ts     = localArg(varargin, 5, []);
            skipXX = localArg(varargin, 6, false);
            skipYY = localArg(varargin, 7, false);
            [route, costMs, prices, info] = localPriceAttr(laws, ...
                NESTED_LAW_KEYS, CENTRES_WORKING_SET_SOFT_BUDGET, ...
                dX, dY, a, admissible, ts, skipXX, skipYY);
            varargout = {route, costMs, prices, info};

        case 'selectNestedMethod'
            [dX, dY, admByAttr, enumOk] = deal(varargin{1}, varargin{2}, ...
                                               varargin{3}, varargin{4});
            ts         = localArg(varargin, 5, []);
            skipXX     = localArg(varargin, 6, false);
            skipYY     = localArg(varargin, 7, false);
            [chosen, planMs, enumMs, detail] = localSelect(laws, ...
                NESTED_LAW_KEYS, CENTRES_WORKING_SET_SOFT_BUDGET, ...
                NESTED_ENUM_SAFETY, dX, dY, admByAttr, enumOk, ts, ...
                skipXX, skipYY);
            varargout = {chosen, planMs, enumMs, detail};

        otherwise
            error('mpt:nestedCost:badCommand', ...
                  'Unknown command ''%s''.', char(cmd));
    end
end


% ----------------------------------------------------------------------
function v = localArg(args, k, default)
    v = default;
    if numel(args) >= k && ~isempty(args{k})
        v = args{k};
    end
end


% ----------------------------------------------------------------------
function idx = localKeyIndex(keys, totalOrder)
%LOCALKEYINDEX  Largest law row at or below TOTALORDER (never below the
%   first). Twin of the Python _nested_cost_key.
    R = double(totalOrder);
    idx = 1;
    for k = 1:numel(keys)
        if keys(k) <= R
            idx = k;
        end
    end
end


% ----------------------------------------------------------------------
function ms = localRouteCostMs(laws, keys, route, totalOrder, term, nMatrices)
%LOCALROUTECOSTMS  t_ms = exp(a) * term^b, floored by max.
%
%   TEST-ONLY HOOK: an override installed by INTERNAL.NESTEDCOSTOVERRIDE
%   replaces the law outright. Nothing in the toolbox installs one; it
%   exists so TESTS/TEST_NESTED_COST_MODEL can stub the prices, as the
%   Python tests monkeypatch nested_route_cost_ms.
    if nargin < 6 || isempty(nMatrices); nMatrices = 3; end
    ov = internal.nestedCostOverride();
    if ~isempty(ov) && isfield(ov, route)
        ms = double(ov.(route));
        return;
    end
    law = laws.(route);
    idx = localKeyIndex(keys, totalOrder);
    t = exp(law.a(idx)) * max(double(term), 1.0) ^ law.b(idx);
    ms = max(t, law.f(idx) + law.pm(idx) * double(nMatrices));
end


% ----------------------------------------------------------------------
function bytes = localWorkingSetBytes(densX, a, info)
%LOCALWORKINGSETBYTES  Peak bundle size of the nested centres route.
%
%   The route rebuilds each side as a single-attribute density and holds
%   its materialised perm-side centres (dim, N * mPerm) together with
%   the matching weight and event-index arrays; DIM is R for an absolute
%   attribute and R - 1 for a relative one, whose centres are anchored
%   at position 0. The estimate is the larger side's centres array with
%   the row factor of two the flat guard uses for its index and weight
%   companions, so the two guards read the same quantity in the same
%   units. Twin of nested_centres_working_set_bytes.
    isRel = logical(densX.isRel(a));
    R = double(densX.r(a));
    dim = max(1, R - double(isRel));
    nJx = info.Nx * info.mPermX;
    nJy = info.Ny * info.mPermY;
    bytes = max(nJx, nJy) * (2.0 * dim) * 8.0;
end


% ----------------------------------------------------------------------
function [best, bestMs, prices, info] = localPriceAttr(laws, keys, budget, ...
        densX, densY, a, admissible, ts, skipXX, skipYY)
%LOCALPRICEATTR  Price the admissible routes for nested attribute A.
%
%   ADMISSIBLE has already been settled by the measure rule, so nothing
%   here can change the number computed. Twin of price_nested_attr.
    if ischar(admissible); admissible = {admissible}; end
    [terms, ~, info] = internal.nestedCostTerms(densX, densY, a, ts, ...
                                                skipXX, skipYY);
    nMatrices = 1 + double(~skipXX) + double(~skipYY);
    ws = localWorkingSetBytes(densX, a, info);
    info.centresWorkingSetBytes = ws;
    prices = struct();
    for k = 1:numel(admissible)
        route = admissible{k};
        cost = localRouteCostMs(laws, keys, route, info.totalOrder, ...
                                terms.(route), nMatrices);
        % Memory guard: the centres route is the only materialising
        % route here, so it is the only one the soft budget can divert.
        % It is diverted only when another admissible route is on offer
        % --- above the sigma/period threshold under
        % wrap = 'single-image' centres is the sole carrier of the
        % declared measure and no budget may move it.
        if strcmp(route, 'centres') && numel(admissible) > 1 && ws > budget
            cost = Inf;
        end
        prices.(route) = cost;
    end
    best = admissible{1};
    bestMs = prices.(best);
    for k = 2:numel(admissible)
        if prices.(admissible{k}) < bestMs
            best = admissible{k};
            bestMs = prices.(best);
        end
    end
end


% ----------------------------------------------------------------------
function [chosen, planMs, enumMs, detail] = localSelect(laws, keys, budget, ...
        safety, densX, densY, admByAttr, enumOk, ts, skipXX, skipYY)
%LOCALSELECT  The contraction plan against the joint-tuple enumeration.
%   Twin of select_nested_method with return_costs.
%
%   The plan and the enumeration share one pair of self-inner-product
%   skip flags. They memoise under different cache keys, so a warm plan
%   memo does not literally spare the enumeration its self matrices;
%   pricing each side against its own memo nonetheless decides the
%   comparison on which side happened to run first rather than on what
%   the two sides cost, and locks that first choice in. See
%   INTERNAL.SELFIPMEMOISED.
    A = double(densX.nAttrs);
    detail = struct('attr', {cell(1, A)}, 'flat', 0.0);
    planMs = 0.0;
    firstNested = 0;
    for a = 1:A
        if a > numel(admByAttr) || isempty(admByAttr{a})
            continue;
        end
        if firstNested == 0; firstNested = a; end
        [route, cost, prices, info] = localPriceAttr(laws, keys, budget, ...
            densX, densY, a, admByAttr{a}, ts, skipXX, skipYY);
        detail.attr{a} = struct('route', route, 'costMs', cost, ...
                                'prices', prices, 'info', info);
        planMs = planMs + cost;
    end

    % The plan's non-nested attributes. Flat-symmetric and r = 1
    % attributes go through the flat per-attribute orbit matrix, so they
    % are priced by INTERNAL.PREDICTORBITCOSTMS on exactly those
    % attributes. An ordered flat attribute goes through the
    % materialised centres, so it is priced by this file's 'centres' law
    % on the same kernel-entry term the nested centres route uses. Twin
    % of _flat_companion_cost_ms.
    flatA = [];
    orderedA = [];
    for a = 1:A
        if a <= numel(admByAttr) && ~isempty(admByAttr{a})
            continue;
        end
        isSymA = true;
        if isfield(densX, 'isSym') && numel(densX.isSym) >= a
            isSymA = logical(densX.isSym(a));
        end
        if ~isSymA && double(densX.r(a)) > 1
            orderedA(end + 1) = a; %#ok<AGROW>
        else
            flatA(end + 1) = a; %#ok<AGROW>
        end
    end
    companions = 0.0;
    if ~isempty(flatA)
        nF = numel(flatA);
        rVec = zeros(1, nF); kVec = zeros(1, nF); kVecY = zeros(1, nF);
        relVec = false(1, nF); nuVec = ones(1, nF);
        sop = 0.0;
        for i = 1:nF
            a = flatA(i);
            rVec(i)  = double(densX.r(a));
            kVec(i)  = size(densX.pAttr{a}, 1);
            kVecY(i) = size(densY.pAttr{a}, 1);
            relVec(i) = logical(densX.isRel(a));
            if relVec(i) && logical(densX.isPer(a))
                nuVec(i) = internal.autoNtauDefault(double(densX.period(a)), ...
                                                    double(densX.sigma(a)), ts);
                sop = max(sop, double(densX.sigma(a)) ...
                               / max(double(densX.period(a)), 1e-300));
            elseif relVec(i)
                nuVec(i) = 2000.0;
            end
        end
        centresOk = sop <= internal.relPerSigmaOverPThreshold(ts);
        companions = companions + internal.predictOrbitCostMs( ...
            rVec, kVec, nF, double(densX.N), double(densY.N), relVec, ...
            nuVec, centresOk, kVecY, skipXX, skipYY);
    end
    nMatrices = 1 + double(~skipXX) + double(~skipYY);
    for i = 1:numel(orderedA)
        a = orderedA(i);
        [terms, ~, info] = internal.nestedCostTerms(densX, densY, a, ts, ...
                                                    skipXX, skipYY);
        companions = companions + localRouteCostMs(laws, keys, 'centres', ...
            info.totalOrder, terms.centres, nMatrices);
    end
    detail.flat = companions;
    planMs = planMs + companions;

    if enumOk && firstNested > 0
        % Joint tuple-pair kernel entries for the enumeration on this
        % pair, under the shared skip flags. The components come from
        % INTERNAL.NESTEDCOSTTERMS, which builds them over every
        % attribute of the density. Twin of
        % predict_nested_pairwise_kernel_size.
        [~, ~, binfo] = internal.nestedCostTerms(densX, densY, firstNested, ts);
        size_ = binfo.bulgerXY;
        if ~skipXX; size_ = size_ + binfo.bulgerXX; end
        if ~skipYY; size_ = size_ + binfo.bulgerYY; end
        rMax = max(double(densX.r(:)).');
        if isempty(rMax); rMax = 2; end
        enumMatrices = 1 + double(~skipXX) + double(~skipYY);
        enumMs = localRouteCostMs(laws, keys, 'bulger', rMax, size_, ...
                                  enumMatrices);
    else
        enumMs = Inf;
    end

    % Ties, and near-ties, favour the contraction: see NESTED_ENUM_SAFETY.
    if enumMs * safety < planMs
        chosen = 'bulger';
    else
        chosen = 'contract';
    end
end
