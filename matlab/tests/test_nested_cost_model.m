%% test_nested_cost_model.m — the nested IP is routed by a priced cost model
%
%  Mirror of the Python tests/test_nested_cost_model.py.
%
%  The nested path used to choose its per-attribute route by comparing raw
%  analytic operation counts --- materialised kernel entries against
%  quadrature nodes times contraction work --- as though a kernel entry and
%  a unit of contraction work cost the same, and it never considered the
%  joint-tuple enumeration on price at all. It now prices every candidate
%  in milliseconds from the fitted laws of INTERNAL.NESTEDCOST, guards the
%  materialising route with the same working-set budget the flat selector
%  uses, and compares the whole contraction plan against the enumeration
%  the way INTERNAL.SELECTMAINNERPRODUCTMETHOD compares the Moebius method
%  against Bulger's.
%
%  What these tests pin is the COST MODEL. The measure rule is pinned by
%  tests/test_nested_measure_rule.m and is deliberately upstream of
%  everything here: the prices are stubbed to absurd values throughout, and
%  no stub is allowed to move a route that carries a different measure.
%
%  Where the Python tests monkeypatch nested_route_cost_ms, this file
%  installs a price stub through INTERNAL.NESTEDCOSTOVERRIDE --- a
%  TEST-ONLY persistent that INTERNAL.NESTEDCOST consults before
%  evaluating a law, and that nothing in the toolbox ever installs. It is
%  cleared on cleanup below (and after every block that sets it), so a
%  failure part-way through cannot leave the session priced by a stub.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.
%  mptTestIsolateDefaults sets truncationSigmas = Inf (accuracy floor).

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults_ncm
    cleanupDefaults_ncm = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

% The stub is a persistent, so it must be cleared however this script ends.
internal.nestedCostOverride([]);
clear cleanupStub_ncm
cleanupStub_ncm = onCleanup(@() internal.nestedCostOverride([])); %#ok<NASGU>

ncm_P = 12.0;
% Same soft budget as the flat eval selector's
% CENTRES_WORKING_SET_SOFT_BUDGET (INTERNAL.SELECTMAEVAL) and as the
% Python dispatch._CENTRES_WORKING_SET_SOFT_BUDGET: 256 MiB.
ncm_BUDGET = 256 * 1024^2;
ncm_limit  = internal.relPerSigmaOverPThreshold([]);
ncm_below  = 0.5 * ncm_limit * ncm_P;     % a sigma below the threshold
ncm_above  = 0.3 * ncm_P;                 % and one well above it

rng(7, 'twister');
ncm_VX = sort(rand(9, 1) * ncm_P, 1);
ncm_VY = sort(rand(9, 1) * ncm_P, 1);

% A pair whose materialised centres bundle is huge: twelve values per
% chord, ten chords, read two-at-two, relative non-periodic. Every count
% the guard reads is analytic --- nothing here enumerates a tuple.
rng(11, 'twister');
ncm_bigX = sort(rand(120, 4) * 40.0, 1);
ncm_bigY = sort(rand(120, 4) * 40.0, 1);


%% --- the working-set guard -------------------------------------------
ncm_dx = ncmDens(ncm_bigX, 0.5, 12, true, false, ncm_P, []);
ncm_dy = ncmDens(ncm_bigY, 0.5, 12, true, false, ncm_P, []);
ncm_ws = internal.nestedCost('centresWorkingSetBytes', ncm_dx, ncm_dy, 1);
results{end+1, 1} = sprintf( ...
    ['nested cost model: the big shape is over the working-set budget ' ...
     '(%.3g > %.3g bytes)'], ncm_ws, ncm_BUDGET); %#ok<*SAGROW>
results{end, 2} = ncm_ws > ncm_BUDGET;

% Price centres as free and the contraction as ruinous: only the guard can
% move the route now.
internal.nestedCostOverride(struct('centres', 1e-6, ...
                                   'contract_relnonper', 1e9));
[ncm_route, ~, ncm_prices] = internal.nestedCost('priceNestedAttr', ...
    ncm_dx, ncm_dy, 1, {'centres', 'contract_relnonper'});
ncm_planned = ncmRoute(ncm_dx, ncm_dy);
internal.nestedCostOverride([]);
results{end+1, 1} = ['nested cost model: the memory guard diverts a huge ' ...
                     'shape however cheaply centres is priced'];
results{end, 2} = isinf(ncm_prices.centres) ...
    && strcmp(ncm_route, 'contract_relnonper') ...
    && strcmp(ncm_planned, 'contract_relnonper');

% The guard is a guard, not a policy: below the budget the price decides.
ncm_sx = ncmDens(ncm_VX, 0.1, 3, true, false, ncm_P, []);
ncm_sy = ncmDens(ncm_VY, 0.1, 3, true, false, ncm_P, []);
ncm_wsSmall = internal.nestedCost('centresWorkingSetBytes', ncm_sx, ncm_sy, 1);
internal.nestedCostOverride(struct('centres', 1e-6, ...
                                   'contract_relnonper', 1e9));
ncm_planned = ncmRoute(ncm_sx, ncm_sy);
internal.nestedCostOverride([]);
results{end+1, 1} = 'nested cost model: a small shape stays priced';
results{end, 2} = ncm_wsSmall < ncm_BUDGET && strcmp(ncm_planned, 'centres');


%% --- the per-attribute race, below the threshold ---------------------
% Both routes carry the full-image measure inside the truncation floor
% there, so the cheaper one is taken --- either way round.
ncm_dx = ncmDens(ncm_VX, ncm_below, 3, true, true, ncm_P, []);
ncm_dy = ncmDens(ncm_VY, ncm_below, 3, true, true, ncm_P, []);
internal.nestedCostOverride(struct('centres', 100.0, 'taugrid', 1.0));
ncm_r1 = ncmRoute(ncm_dx, ncm_dy);
internal.nestedCostOverride(struct('centres', 1.0, 'taugrid', 100.0));
ncm_r2 = ncmRoute(ncm_dx, ncm_dy);
internal.nestedCostOverride([]);
results{end+1, 1} = ['nested cost model: full-image below the threshold ' ...
                     'races the two routes'];
results{end, 2} = strcmp(ncm_r1, 'taugrid') && strcmp(ncm_r2, 'centres');

% Above the threshold the full-image measure admits the tau grid alone, so
% pricing centres at nothing changes nothing.
ncm_hx = ncmDens(ncm_VX, ncm_above, 3, true, true, ncm_P, []);
ncm_hy = ncmDens(ncm_VY, ncm_above, 3, true, true, ncm_P, []);
internal.nestedCostOverride(struct('centres', 1e-9, 'taugrid', 1e9));
ncm_r1 = ncmRoute(ncm_hx, ncm_hy);
internal.nestedCostOverride([]);
results{end+1, 1} = ['nested cost model: above the threshold no price ' ...
                     'reaches the centres route'];
results{end, 2} = strcmp(ncm_r1, 'taugrid');

% Above the threshold the minimum-image measure has one carrier, so
% pricing the tau grid at nothing changes nothing.
ncm_ok = true;
for ncm_sop = [0.05 0.2 0.4]
    ncm_sx2 = ncmDens(ncm_VX, ncm_sop * ncm_P, 3, true, true, ncm_P, ...
                      'single-image');
    ncm_sy2 = ncmDens(ncm_VY, ncm_sop * ncm_P, 3, true, true, ncm_P, ...
                      'single-image');
    internal.nestedCostOverride(struct('centres', 1e9, 'taugrid', 1e-9));
    ncm_ok = ncm_ok && strcmp(ncmRoute(ncm_sx2, ncm_sy2), 'centres');
    internal.nestedCostOverride([]);
end
results{end+1, 1} = ['nested cost model: single-image above the threshold ' ...
                     'is centres at any price'];
results{end, 2} = ncm_ok;

% Below the threshold the two readings agree inside the floor, so
% single-image is priced exactly as full-image is.
ncm_sx2 = ncmDens(ncm_VX, ncm_below, 3, true, true, ncm_P, 'single-image');
ncm_sy2 = ncmDens(ncm_VY, ncm_below, 3, true, true, ncm_P, 'single-image');
internal.nestedCostOverride(struct('centres', 100.0, 'taugrid', 1.0));
ncm_r1 = ncmRoute(ncm_sx2, ncm_sy2);
internal.nestedCostOverride(struct('centres', 1.0, 'taugrid', 100.0));
ncm_r2 = ncmRoute(ncm_sx2, ncm_sy2);
internal.nestedCostOverride([]);
results{end+1, 1} = ['nested cost model: single-image below the threshold ' ...
                     'is raced'];
results{end, 2} = strcmp(ncm_r1, 'taugrid') && strcmp(ncm_r2, 'centres');

% The soft budget diverts the centres route only where another route
% carries the same measure; above the threshold under single-image there
% is none, so the guard stands down.
rng(11, 'twister');
ncm_bx = ncmDens(sort(rand(120, 4) * ncm_P, 1), ncm_above, 12, true, ...
                 true, ncm_P, 'single-image');
ncm_by = ncmDens(sort(rand(120, 4) * ncm_P, 1), ncm_above, 12, true, ...
                 true, ncm_P, 'single-image');
ncm_ws = internal.nestedCost('centresWorkingSetBytes', ncm_bx, ncm_by, 1);
[ncm_route, ~, ncm_prices] = internal.nestedCost('priceNestedAttr', ...
    ncm_bx, ncm_by, 1, {'centres'});
results{end+1, 1} = ['nested cost model: the guard never moves ' ...
                     'single-image above the threshold'];
results{end, 2} = ncm_ws > ncm_BUDGET && strcmp(ncm_route, 'centres') ...
    && isfinite(ncm_prices.centres);

% An absolute attribute stays on the contraction.
ncm_ax = ncmDens(ncm_VX, 0.2, 3, false, false, ncm_P, []);
ncm_ay = ncmDens(ncm_VY, 0.2, 3, false, false, ncm_P, []);
internal.nestedCostOverride(struct('contract', 1e9, 'centres', 1e-9));
ncm_r1 = ncmRoute(ncm_ax, ncm_ay);
internal.nestedCostOverride([]);
results{end+1, 1} = ['nested cost model: an absolute attribute stays on ' ...
                     'the contraction'];
results{end, 2} = strcmp(ncm_r1, 'contract');


%% --- the plan against the enumeration --------------------------------
% Price the enumeration at a millionth of everything else.
ncm_cheapEnum = struct('bulger', 1e-6, 'centres', 1.0, 'taugrid', 1.0, ...
                       'contract', 1.0, 'contract_relnonper', 1.0);

ncm_ref = cosSimExpTens( ...
    ncmDens(ncm_VX, ncm_below, 3, true, true, ncm_P, []), ...
    ncmDens(ncm_VY, ncm_below, 3, true, true, ncm_P, []), ...
    'method', 'bulger', 'verbose', false);
internal.nestedCostOverride(ncm_cheapEnum);
ncm_got = cosSimExpTens( ...
    ncmDens(ncm_VX, ncm_below, 3, true, true, ncm_P, []), ...
    ncmDens(ncm_VY, ncm_below, 3, true, true, ncm_P, []), 'verbose', false);
ncm_costs = internal.lastNestedCosts();
internal.nestedCostOverride([]);
results{end+1, 1} = sprintf( ...
    ['nested cost model: the enumeration is taken where it is priced ' ...
     'cheaper (%.2e)'], abs(ncm_got - ncm_ref));
results{end, 2} = strcmp(ncm_costs.chosen, 'bulger') ...
    && ncm_costs.enumMs < ncm_costs.planMs ...
    && abs(ncm_got - ncm_ref) <= 1e-9 * max(abs(ncm_ref), 1);

internal.nestedCostOverride(struct('bulger', 1e9, 'centres', 1.0, ...
    'taugrid', 1.0, 'contract', 1.0, 'contract_relnonper', 1.0));
cosSimExpTens( ...
    ncmDens(ncm_VX, ncm_below, 3, true, true, ncm_P, []), ...
    ncmDens(ncm_VY, ncm_below, 3, true, true, ncm_P, []), 'verbose', false);
ncm_costs = internal.lastNestedCosts();
ncm_r1 = internal.lastNestedRoutes();
internal.nestedCostOverride([]);
results{end+1, 1} = ['nested cost model: the plan is kept where the ' ...
                     'enumeration is priced dearer'];
results{end, 2} = strcmp(ncm_costs.chosen, 'contract') ...
    && numel(ncm_r1) == 1 && any(strcmp(ncm_r1{1}, {'centres', 'taugrid'}));

% The enumeration has to be decisively cheaper, not marginally: the plan
% costs 1 ms and the enumeration 1/safety ms, which ties exactly.
ncm_safety = internal.nestedCost('enumSafety');
internal.nestedCostOverride(struct('bulger', 1.0 / ncm_safety, ...
    'centres', 1.0, 'taugrid', 1.0, 'contract', 1.0, ...
    'contract_relnonper', 1.0));
cosSimExpTens( ...
    ncmDens(ncm_VX, ncm_below, 3, true, true, ncm_P, []), ...
    ncmDens(ncm_VY, ncm_below, 3, true, true, ncm_P, []), 'verbose', false);
ncm_costs = internal.lastNestedCosts();
internal.nestedCostOverride([]);
results{end+1, 1} = 'nested cost model: a near tie keeps the plan';
results{end, 2} = strcmp(ncm_costs.chosen, 'contract');

% Above the threshold the enumeration computes the minimum-image reading,
% so no price buys it.
internal.nestedCostOverride(ncm_cheapEnum);
cosSimExpTens( ...
    ncmDens(ncm_VX, ncm_above, 3, true, true, ncm_P, []), ...
    ncmDens(ncm_VY, ncm_above, 3, true, true, ncm_P, []), 'verbose', false);
ncm_costs = internal.lastNestedCosts();
internal.nestedCostOverride([]);
results{end+1, 1} = ['nested cost model: the enumeration is inadmissible ' ...
                     'above the threshold'];
results{end, 2} = strcmp(ncm_costs.chosen, 'contract') ...
    && isinf(ncm_costs.enumMs);

% A forced method is never diverted.
internal.nestedCostOverride(ncm_cheapEnum);
internal.lastNestedRoutes({});
cosSimExpTens( ...
    ncmDens(ncm_VX, ncm_below, 3, true, true, ncm_P, []), ...
    ncmDens(ncm_VY, ncm_below, 3, true, true, ncm_P, []), ...
    'method', 'contract', 'verbose', false);
ncm_r1 = internal.lastNestedRoutes();
internal.nestedCostOverride([]);
results{end+1, 1} = 'nested cost model: a forced method is never diverted';
results{end, 2} = ~isempty(ncm_r1);

% The multi-attribute plan is priced too, companions included. A flat
% SYMMETRIC r = 2 companion: the flat model absorbs an r = 1 attribute
% into its base and prices it at nothing, so an r = 1 companion could not
% show that companions are priced at all.
ncm_ex = [0.3 1.1 2.0; 0.7 1.5 2.6; 1.2 2.1 3.3];
ncm_ey = ncm_ex + 0.2;
ncm_maX = repmat(ncm_VX, 1, 3);
ncm_maY = repmat(ncm_VY, 1, 3);
ncm_ref = cosSimExpTens( ...
    ncmDensMA(ncm_maX, ncm_ex, ncm_below, ncm_P), ...
    ncmDensMA(ncm_maY, ncm_ey, ncm_below, ncm_P), ...
    'method', 'bulger', 'verbose', false);
internal.nestedCostOverride(ncm_cheapEnum);
ncm_got = cosSimExpTens( ...
    ncmDensMA(ncm_maX, ncm_ex, ncm_below, ncm_P), ...
    ncmDensMA(ncm_maY, ncm_ey, ncm_below, ncm_P), 'verbose', false);
ncm_costs = internal.lastNestedCosts();
internal.nestedCostOverride([]);
results{end+1, 1} = sprintf( ...
    'nested cost model: the multi-attribute plan is priced too (%.2e)', ...
    abs(ncm_got - ncm_ref));
results{end, 2} = strcmp(ncm_costs.chosen, 'bulger') ...
    && ncm_costs.detail.flat > 0 ...
    && abs(ncm_got - ncm_ref) <= 1e-9 * max(abs(ncm_ref), 1);


%% --- the terms and the skip flags ------------------------------------
% The centres route forms one (mCombX, mPermY) array per event pair where
% Bulger's restriction holds, and the term says so.
ncm_dx = ncmDens(ncm_VX, ncm_below, 3, true, true, ncm_P, []);
ncm_dy = ncmDens(ncm_VY, ncm_below, 3, true, true, ncm_P, []);
[ncm_terms, ~, ncm_info] = internal.nestedCostTerms(ncm_dx, ncm_dy, 1, []);
ncm_expect = 3.0 * ncm_info.mCombX * ncm_info.mPermY;
results{end+1, 1} = 'nested cost model: the centres term counts the restricted side';
results{end, 2} = ncm_info.restrictedX && ncm_info.restrictedY ...
    && abs(ncm_terms.centres - ncm_expect) <= 1e-9 * ncm_expect;

ncm_cross = internal.nestedCostTerms(ncm_dx, ncm_dy, 1, [], true, true);
results{end+1, 1} = 'nested cost model: a skipped self inner product is not priced';
results{end, 2} = ncm_cross.centres < ncm_terms.centres ...
    && ncm_cross.taugrid < ncm_terms.taugrid;


%% --- the report ------------------------------------------------------
ncm_rep = explainDispatch(ncm_dx, ncm_dy);
ncm_text = evalc('explainDispatch(ncm_dx, ncm_dy);');
ncm_byName = containers.Map(ncm_rep.routeNames, num2cell(ncm_rep.routeMs));
results{end+1, 1} = 'nested cost model: explainDispatch shows the prices';
results{end, 2} = ncm_rep.priced ...
    && ncm_byName('contract') > 0 && ncm_byName('bulger') > 0 ...
    && contains(ncm_rep.decidedBy, 'nested cost model') ...
    && contains(ncm_rep.routeWhy{1}, 'centres') ...
    && contains(ncm_rep.routeWhy{1}, 'taugrid') ...
    && contains(ncm_rep.routeWhy{1}, 'ms') ...
    && contains(ncm_text, 'ms');

ncm_hx = ncmDens(ncm_VX, ncm_above, 3, true, true, ncm_P, []);
ncm_hy = ncmDens(ncm_VY, ncm_above, 3, true, true, ncm_P, []);
ncm_text = evalc('explainDispatch(ncm_hx, ncm_hy);');
results{end+1, 1} = ['nested cost model: explainDispatch reports an ' ...
                     'inadmissible enumeration'];
results{end, 2} = contains(ncm_text, 'minimum-image measure');


if standalone
    nPass = sum([results{:, 2}]);
    nFail = size(results, 1) - nPass;
    for ii = 1:size(results, 1)
        if results{ii, 2}
            fprintf('  PASS  %s\n', results{ii, 1});
        else
            fprintf('  FAIL  %s\n', results{ii, 1});
        end
    end
    fprintf('\n=== test_nested_cost_model: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupStub_ncm cleanupDefaults_ncm
    if nFail > 0
        error('test_nested_cost_model:failed', '%d test(s) failed.', nFail);
    end
end


% ----------------------------------------------------------------------
function d = ncmDens(v, sigma, chord, isRel, isPer, P, wrap)
    % A single nested attribute, r = [2 2], both levels symmetric, values
    % grouped CHORD at a time. Twin of the Python test's _dens.
    nGroup = size(v, 1) / chord;
    tags = repelem((0:nGroup - 1).', chord, 1);
    if isRel
        relVec = [0, 1];
    else
        relVec = [0, 0];
    end
    spec = struct('tags', tags, 'r', [2 2], 'sym', [true true], ...
                  'rel', relVec);
    if isPer
        period = P;
    else
        period = 0.0;
    end
    args = {{v}, {[]}, 'specs', {spec}, 'sigma', sigma, 'isPer', isPer, ...
            'period', period, 'verbose', false};
    if ~isempty(wrap)
        args = [args, {'wrap', {wrap}}];
    end
    d = buildExpTens(args{:});
end


% ----------------------------------------------------------------------
function d = ncmDensMA(v, extra, sigma, P)
    % The same nested attribute tensored with a flat SYMMETRIC r = 2
    % companion, which the flat orbit model prices at more than nothing.
    nGroup = size(v, 1) / 3;
    tags = repelem((0:nGroup - 1).', 3, 1);
    spec = struct('tags', tags, 'r', [2 2], 'sym', [true true], 'rel', [0 1]);
    specs = {spec, struct('r', 2, 'rel', false, 'sym', true)};
    d = buildExpTens({v, extra}, {[], []}, 'specs', specs, ...
                     'sigma', [sigma, 1.0], 'isPer', [true, false], ...
                     'period', [P, 0.0], 'verbose', false);
end


% ----------------------------------------------------------------------
function route = ncmRoute(dx, dy)
    % The route the plan would take for attribute 1, read without
    % computing anything and without the plan-versus-enumeration race
    % (which routesOnly skips, as EXPLAINDISPATCH needs it to).
    [~, routes] = internal.nestedContract(dx, dy, 'cosine', [], false, ...
                                          struct('routesOnly', true));
    route = routes{1};
end
