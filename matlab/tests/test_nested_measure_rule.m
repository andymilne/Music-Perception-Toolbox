%% test_nested_measure_rule.m — the nested rel-per measure is declared by wrap
%
%  Mirror of the Python tests/test_nested_measure_rule.py.
%
%  A relative-periodic attribute admits two readings of "periodic":
%  (A) the minimum-image pairwise-wrap form the materialised-centres route
%  evaluates, and (C) the all-image transposition average the tau-grid
%  contraction evaluates. They agree for sigma << P and diverge as sigma
%  approaches it. The flat path has always let wrap decide between them;
%  the nested path used to let the *cost race* decide, so the same input
%  returned one number or the other depending on how many values or levels
%  it happened to carry.
%
%  These tests pin the rule the nested path now follows, the same one
%  INTERNAL.SELECTMAINNERPRODUCTMETHOD enforces on the flat path:
%
%    * wrap = 'full-image' (the default) declares (C); wrap =
%      'single-image' declares (A).
%    * Above sigma/P = INTERNAL.RELPERSIGMAOVERPTHRESHOLD each declaration
%      has exactly one carrier and that route is taken whatever it costs:
%      the tau grid under (C), the centres route under (A).
%    * At or below the threshold the two agree inside the truncation
%      floor, so both routes serve either declaration and the cost model
%      decides between them --- the same rule the flat path follows, whose
%      wrap override is likewise reached only above the threshold.
%
%  They also pin what the method keyword now means on a nested density
%  ('centres' and 'mobius' no longer fall through to 'bulger'), the
%  memoisation of the multi-attribute self inner products, the ordered-flat
%  companion attribute (which must not be symmetrised), and the agreement of
%  method = 'bulger' with the contraction that retires the claim that the
%  joint-tuple enumeration mis-shapes a nested attribute.
%
%  Where the Python tests monkeypatch the route hook to name a reference
%  measure, this file uses an independent brute force instead: the nested
%  tuple set is enumerated explicitly and the transposition average of the
%  wrapped Gaussian taken on a 4000-node grid, so the test says which
%  measure the swept value follows rather than only that it is consistent.
%  The route actually taken is read back from INTERNAL.LASTNESTEDROUTES,
%  the MATLAB twin of Python's _LAST_NESTED_ROUTES.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.
%  mptTestIsolateDefaults sets truncationSigmas = Inf (accuracy floor).

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults_nmr
    cleanupDefaults_nmr = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

nmr_P = 12.0;
% Two units of two values: small enough to enumerate by brute force, and
% large enough that the outer level's orbit and each unit's own orbit both
% contribute to the wreath-product group the comb restriction collapses.
nmr_TAGS = [0 0 1 1];
nmr_SPEC = struct('tags', nmr_TAGS, 'r', [2 2], 'sym', [true true], ...
                  'rel', [0 1]);
% Nine values in three units of three: a shape whose tuple counts are large
% enough that the cost race is a live question at every sigma/P.
nmr_TAGS9 = repelem(0:2, 3);
nmr_SPEC9 = struct('tags', nmr_TAGS9, 'r', [2 2], 'sym', [true true], ...
                   'rel', [0 1]);

rng(7, 'twister');
nmr_px = sort(rand(4, 2) * nmr_P, 1);
nmr_py = sort(rand(4, 2) * nmr_P, 1);
nmr_px9 = sort(rand(9, 2) * nmr_P, 1);
nmr_py9 = sort(rand(9, 2) * nmr_P, 1);

nmr_limit = internal.relPerSigmaOverPThreshold([]);
nmr_floor = internal.truncationFloor([]);
nmr_sops  = [0.005 0.02 0.05 0.1 0.2 0.3];

% --- the measure rule: which route is taken, and why -------------------
% Above the threshold the centres route is not admissible at any price, so
% the route must be the tau grid whatever the tuple counts say; below it
% either route carries the declared measure and the cost race decides.
for nmr_ii = 1:numel(nmr_sops)
    nmr_sop = nmr_sops(nmr_ii);
    nmr_sigma = nmr_sop * nmr_P;
    cosSimExpTens(nmrDens(nmr_px9, nmr_sigma, nmr_SPEC9, [], nmr_P), ...
                  nmrDens(nmr_py9, nmr_sigma, nmr_SPEC9, [], nmr_P), ...
                  'verbose', false);
    nmr_r = internal.lastNestedRoutes();
    if nmr_sop > nmr_limit
        nmr_ok = isequal(nmr_r, {'taugrid'});
    else
        nmr_ok = numel(nmr_r) == 1 ...
            && any(strcmp(nmr_r{1}, {'centres', 'taugrid'}));
    end
    results{end+1, 1} = sprintf( ...
        ['nested measure rule: full-image route at sigma/P = %.3f is %s ' ...
         '(limit %g)'], nmr_sop, nmr_r{1}, nmr_limit); %#ok<*SAGROW>
    results{end, 2} = nmr_ok;
end

% ABOVE the threshold wrap = 'single-image' declares the minimum-image
% measure and the centres route is the only route that computes it, so it
% is taken however the cost model would price it.
nmr_ok = true;
for nmr_sop = [0.05 0.2 0.4]
    nmr_sigma = nmr_sop * nmr_P;
    cosSimExpTens( ...
        nmrDens(nmr_px9, nmr_sigma, nmr_SPEC9, 'single-image', nmr_P), ...
        nmrDens(nmr_py9, nmr_sigma, nmr_SPEC9, 'single-image', nmr_P), ...
        'verbose', false);
    nmr_ok = nmr_ok && isequal(internal.lastNestedRoutes(), {'centres'});
end
results{end+1, 1} = ['nested measure rule: single-image takes the centres ' ...
                     'route above the threshold'];
results{end, 2} = nmr_ok;

% BELOW the threshold the two readings agree inside the floor, so
% 'single-image' is raced exactly as 'full-image' is -- the same rule the
% flat path follows. What is pinned is that the choice is free and cannot
% change the answer: whichever route the price picks, the value is the
% minimum-image (forced-centres) value to within the truncation floor.
nmr_ok = true;
nmr_worst = 0;
for nmr_sop = [0.005 0.01 0.02]
    nmr_sigma = nmr_sop * nmr_P;
    nmr_auto = cosSimExpTens( ...
        nmrDens(nmr_px9, nmr_sigma, nmr_SPEC9, 'single-image', nmr_P), ...
        nmrDens(nmr_py9, nmr_sigma, nmr_SPEC9, 'single-image', nmr_P), ...
        'verbose', false);
    nmr_r = internal.lastNestedRoutes();
    nmr_ok = nmr_ok && numel(nmr_r) == 1 ...
        && any(strcmp(nmr_r{1}, {'centres', 'taugrid'}));
    nmr_ref = cosSimExpTens( ...
        nmrDens(nmr_px9, nmr_sigma, nmr_SPEC9, 'single-image', nmr_P), ...
        nmrDens(nmr_py9, nmr_sigma, nmr_SPEC9, 'single-image', nmr_P), ...
        'method', 'centres', 'verbose', false);
    nmr_worst = max(nmr_worst, abs(nmr_auto - nmr_ref));
end
results{end+1, 1} = sprintf( ...
    ['nested measure rule: single-image below the threshold admits ' ...
     'either route, to one value (%.2e)'], nmr_worst);
results{end, 2} = nmr_ok && nmr_worst <= max(nmr_floor, 1e-9);

% --- the swept value follows one measure throughout --------------------
% Testing the step sizes directly would not separate a route flip from the
% genuine curvature of the cosine, which is steep at small sigma/P. What
% the rule guarantees is stronger and exactly checkable: the value equals
% the all-image reference throughout -- exactly above the threshold, and to
% within the truncation floor below it, where the minimum-image centres
% route is admitted. A cost-driven flip broke that wherever the shape
% crossed the price crossover.
for nmr_ii = 1:numel(nmr_sops)
    nmr_sop = nmr_sops(nmr_ii);
    nmr_sigma = nmr_sop * nmr_P;
    nmr_got = cosSimExpTens( ...
        nmrDens(nmr_px, nmr_sigma, nmr_SPEC, [], nmr_P), ...
        nmrDens(nmr_py, nmr_sigma, nmr_SPEC, [], nmr_P), 'verbose', false);
    nmr_ref = nmrRefCos(nmr_px, nmr_py, nmr_sigma, nmr_P);
    results{end+1, 1} = sprintf( ...
        ['nested measure rule: auto == all-image transposition average at ' ...
         'sigma/P = %.3f (%.2e)'], nmr_sop, abs(nmr_got - nmr_ref));
    results{end, 2} = abs(nmr_got - nmr_ref) <= 1e-11 * max(abs(nmr_ref), 1) ...
        + max(nmr_floor, 1e-12);
end

% Below the threshold the two measures agree inside the floor, which is
% exactly what makes the centres route admissible there; above it they do
% not, which is what the rule exists to protect.
nmr_lowSig = 0.02 * nmr_P;
nmr_lowFull = cosSimExpTens( ...
    nmrDens(nmr_px, nmr_lowSig, nmr_SPEC, [], nmr_P), ...
    nmrDens(nmr_py, nmr_lowSig, nmr_SPEC, [], nmr_P), 'verbose', false);
nmr_lowSingle = cosSimExpTens( ...
    nmrDens(nmr_px, nmr_lowSig, nmr_SPEC, 'single-image', nmr_P), ...
    nmrDens(nmr_py, nmr_lowSig, nmr_SPEC, 'single-image', nmr_P), ...
    'verbose', false);
results{end+1, 1} = sprintf( ...
    ['nested measure rule: the two measures agree below the threshold ' ...
     '(%.2e)'], abs(nmr_lowFull - nmr_lowSingle));
results{end, 2} = abs(nmr_lowFull - nmr_lowSingle) <= max(nmr_floor, 1e-9);

nmr_hiSig = 0.2 * nmr_P;
nmr_hiFull = cosSimExpTens( ...
    nmrDens(nmr_px, nmr_hiSig, nmr_SPEC, [], nmr_P), ...
    nmrDens(nmr_py, nmr_hiSig, nmr_SPEC, [], nmr_P), 'verbose', false);
nmr_hiSingle = cosSimExpTens( ...
    nmrDens(nmr_px, nmr_hiSig, nmr_SPEC, 'single-image', nmr_P), ...
    nmrDens(nmr_py, nmr_hiSig, nmr_SPEC, 'single-image', nmr_P), ...
    'verbose', false);
results{end+1, 1} = sprintf( ...
    ['nested measure rule: the two measures differ above the threshold ' ...
     '(%.2e)'], abs(nmr_hiFull - nmr_hiSingle));
results{end, 2} = abs(nmr_hiFull - nmr_hiSingle) > 1e-3;

% --- method keywords on a nested density -------------------------------
% All three formerly reached chosen = 'bulger', so the method name
% described something other than what ran. Each now names a route of the
% contraction plan, which the route hook records.
nmr_lowDx = nmrDens(nmr_px9, nmr_lowSig, nmr_SPEC9, [], nmr_P);
nmr_lowDy = nmrDens(nmr_py9, nmr_lowSig, nmr_SPEC9, [], nmr_P);
nmr_methods = {'auto', 'contract', 'mobius', 'centres'};
for nmr_ii = 1:numel(nmr_methods)
    internal.lastNestedRoutes({});
    cosSimExpTens(nmr_lowDx, nmr_lowDy, 'method', nmr_methods{nmr_ii}, ...
                  'verbose', false);
    nmr_r = internal.lastNestedRoutes();
    nmr_ok = numel(nmr_r) == 1;
    if strcmp(nmr_methods{nmr_ii}, 'centres')
        nmr_ok = nmr_ok && strcmp(nmr_r{1}, 'centres');
    end
    results{end+1, 1} = sprintf( ...
        'nested measure rule: method ''%s'' runs the plan, not bulger', ...
        nmr_methods{nmr_ii});
    results{end, 2} = nmr_ok;
end

% Forcing the centres route above the threshold must error rather than
% silently return the other measure, and the message must name the opt-in.
nmr_hiDx = nmrDens(nmr_px9, nmr_hiSig, nmr_SPEC9, [], nmr_P);
nmr_hiDy = nmrDens(nmr_py9, nmr_hiSig, nmr_SPEC9, [], nmr_P);
results{end+1, 1} = ['nested measure rule: forced centres above the ' ...
                     'threshold raises centresUnavailable'];
results{end, 2} = throwsErrorWithId(@() cosSimExpTens(nmr_hiDx, nmr_hiDy, ...
    'method', 'centres', 'verbose', false), 'cosSimExpTens:centresUnavailable') ...
    && errorMessageContains(@() cosSimExpTens(nmr_hiDx, nmr_hiDy, ...
        'method', 'centres', 'verbose', false), 'single-image');

% method = 'contract' still applies to nested densities only.
nmr_flat = buildExpTens([0 4 7], [], 1.0, 2, false, false, 0, ...
                        'verbose', false);
results{end+1, 1} = 'nested measure rule: contract is rejected on a flat density';
results{end, 2} = throwsErrorWithId(@() cosSimExpTens(nmr_flat, nmr_flat, ...
    'method', 'contract', 'verbose', false), 'cosSimExpTens:contractUnavailable');

% --- the dispatch message announces the route that ran -----------------
internal.maybeShowDispatchMsg('reset');
nmr_prevHints = mptDefaults('showHints', true);
nmr_out = evalc('cosSimExpTens(nmr_hiDx, nmr_hiDy, ''verbose'', false);');
mptDefaults(nmr_prevHints);
results{end+1, 1} = ['nested measure rule: the nested route is announced ' ...
                     'as ''contract'', not ''bulger'''];
results{end, 2} = contains(nmr_out, 'chose ''contract'' path') ...
    && ~contains(nmr_out, 'chose ''bulger'' path');

% --- multi-attribute: memoisation, ordered companion, bulger parity ----
nmr_maSpecs = {nmr_SPEC, struct('r', 1, 'rel', false, 'sym', true)};
nmr_v2x = rand(1, 2) * 5;
nmr_v2y = nmr_v2x + 0.3;
nmr_maDx = buildExpTens({nmr_px, nmr_v2x}, {[], []}, 'specs', nmr_maSpecs, ...
    'sigma', [nmr_lowSig, 1.0], 'isPer', [true, false], ...
    'period', [nmr_P, 0.0], 'verbose', false);
nmr_maDy = buildExpTens({nmr_py, nmr_v2y}, {[], []}, 'specs', nmr_maSpecs, ...
    'sigma', [nmr_lowSig, 1.0], 'isPer', [true, false], ...
    'period', [nmr_P, 0.0], 'verbose', false);
[nmr_ma1, nmr_maXo, nmr_maYo] = cosSimExpTens(nmr_maDx, nmr_maDy, ...
                                              'verbose', false);
nmr_keysX = nmr_maXo.selfIP.keys;
nmr_keysY = nmr_maYo.selfIP.keys;
nmr_nX = sum(strncmp(nmr_keysX, 'contract_ma|', 12));
nmr_nY = sum(strncmp(nmr_keysY, 'contract_ma|', 12));
results{end+1, 1} = 'nested measure rule: MA nested self inner products are memoised';
results{end, 2} = nmr_nX == 1 && nmr_nY == 1;

% The cached values are consumed, not merely stored: a second call with the
% caches warm returns the same number.
nmr_ma2 = cosSimExpTens(nmr_maXo, nmr_maYo, 'verbose', false);
results{end+1, 1} = 'nested measure rule: the warm MA memo returns the same value';
results{end, 2} = nmr_ma1 == nmr_ma2;

% The key names the route, so a value taken under one measure can never be
% reused under another.
nmr_siSpecs = nmr_maSpecs;
nmr_maSiX = buildExpTens({nmr_px, nmr_v2x}, {[], []}, 'specs', nmr_siSpecs, ...
    'sigma', [nmr_hiSig, 1.0], 'isPer', [true, false], ...
    'period', [nmr_P, 0.0], 'wrap', {'single-image', 'full-image'}, ...
    'verbose', false);
nmr_maSiY = buildExpTens({nmr_py, nmr_v2y}, {[], []}, 'specs', nmr_siSpecs, ...
    'sigma', [nmr_hiSig, 1.0], 'isPer', [true, false], ...
    'period', [nmr_P, 0.0], 'wrap', {'single-image', 'full-image'}, ...
    'verbose', false);
[~, nmr_siXo] = cosSimExpTens(nmr_maSiX, nmr_maSiY, 'verbose', false);
nmr_maFiX = buildExpTens({nmr_px, nmr_v2x}, {[], []}, 'specs', nmr_siSpecs, ...
    'sigma', [nmr_hiSig, 1.0], 'isPer', [true, false], ...
    'period', [nmr_P, 0.0], 'verbose', false);
nmr_maFiY = buildExpTens({nmr_py, nmr_v2y}, {[], []}, 'specs', nmr_siSpecs, ...
    'sigma', [nmr_hiSig, 1.0], 'isPer', [true, false], ...
    'period', [nmr_P, 0.0], 'verbose', false);
[~, nmr_fiXo] = cosSimExpTens(nmr_maFiX, nmr_maFiY, 'verbose', false);
results{end+1, 1} = 'nested measure rule: the MA memo key separates the routes';
results{end, 2} = any(contains(nmr_siXo.selfIP.keys, 'centres')) ...
    && any(contains(nmr_fiXo.selfIP.keys, 'taugrid'));

% An ordered flat companion ([sym] = false, r > 1) must go through the
% materialised centres, not the orbit per-attribute matrix, which would
% sum its full S_r orbit and so symmetrise an attribute the user asked to
% keep ordered. The joint-tuple enumeration reads the stored ordered
% tuples, so it is the reference.
nmr_ordSpecs = {nmr_SPEC, struct('r', 2, 'rel', false, 'sym', false)};
nmr_o2x = rand(3, 2) * 5;
nmr_o2y = nmr_o2x + 0.3;
nmr_ordKw = {'sigma', [nmr_lowSig, 1.0], 'isPer', [true, false], ...
             'period', [nmr_P, 0.0], 'verbose', false};
nmr_ordC = cosSimExpTens( ...
    buildExpTens({nmr_px, nmr_o2x}, {[], []}, 'specs', nmr_ordSpecs, nmr_ordKw{:}), ...
    buildExpTens({nmr_py, nmr_o2y}, {[], []}, 'specs', nmr_ordSpecs, nmr_ordKw{:}), ...
    'verbose', false);
nmr_ordB = cosSimExpTens( ...
    buildExpTens({nmr_px, nmr_o2x}, {[], []}, 'specs', nmr_ordSpecs, nmr_ordKw{:}), ...
    buildExpTens({nmr_py, nmr_o2y}, {[], []}, 'specs', nmr_ordSpecs, nmr_ordKw{:}), ...
    'method', 'bulger', 'verbose', false);
results{end+1, 1} = sprintf( ...
    ['nested measure rule: an ordered flat companion is not symmetrised ' ...
     '(%.2e)'], abs(nmr_ordC - nmr_ordB));
results{end, 2} = abs(nmr_ordC - nmr_ordB) <= 1e-9 * max(abs(nmr_ordB), 1);

% method = 'bulger' on a multi-attribute nested density agrees with the
% contraction to floating point, in every mode, at a sigma/P where the two
% measures coincide. This retires the claim that the joint-tuple
% enumeration mis-shapes a nested attribute's per-event tuples in the MA
% tensor build. The fallback policy is unchanged --- the contraction still
% always returns its triple rather than deferring to the enumeration ---
% but the reason is cost and per-attribute measure control, not a shape bug.
% label, outer rel, isPer, period, sigma
nmr_bulgerCases = { ...
    'abs non-periodic', 0, false, 0.0,   2.0; ...
    'rel non-periodic', 1, false, 0.0,   2.0; ...
    'abs periodic',     0, true,  nmr_P, 0.24; ...
    'rel periodic',     1, true,  nmr_P, 0.24};
for nmr_ii = 1:size(nmr_bulgerCases, 1)
    nmr_spec = struct('tags', nmr_TAGS, 'r', [2 2], 'sym', [true true], ...
                      'rel', [0 nmr_bulgerCases{nmr_ii, 2}]);
    nmr_specsB = {nmr_spec, struct('r', 1, 'rel', false, 'sym', true)};
    nmr_kwB = {'sigma', [nmr_bulgerCases{nmr_ii, 5}, 1.0], ...
               'isPer', [nmr_bulgerCases{nmr_ii, 3}, false], ...
               'period', [nmr_bulgerCases{nmr_ii, 4}, 0.0], 'verbose', false};
    nmr_bC = cosSimExpTens( ...
        buildExpTens({nmr_px, nmr_v2x}, {[], []}, 'specs', nmr_specsB, nmr_kwB{:}), ...
        buildExpTens({nmr_py, nmr_v2y}, {[], []}, 'specs', nmr_specsB, nmr_kwB{:}), ...
        'verbose', false);
    nmr_bB = cosSimExpTens( ...
        buildExpTens({nmr_px, nmr_v2x}, {[], []}, 'specs', nmr_specsB, nmr_kwB{:}), ...
        buildExpTens({nmr_py, nmr_v2y}, {[], []}, 'specs', nmr_specsB, nmr_kwB{:}), ...
        'method', 'bulger', 'verbose', false);
    results{end+1, 1} = sprintf( ...
        'nested measure rule: bulger == contraction on MA nested, %s (%.2e)', ...
        nmr_bulgerCases{nmr_ii, 1}, abs(nmr_bC - nmr_bB));
    results{end, 2} = abs(nmr_bC - nmr_bB) <= 1e-9 * max(abs(nmr_bB), 1);
end

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
    fprintf('\n=== test_nested_measure_rule: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults_nmr
    if nFail > 0
        error('test_nested_measure_rule:failed', '%d test(s) failed.', nFail);
    end
end


% ----------------------------------------------------------------------
function d = nmrDens(p, sigma, spec, wrap, P)
    % A nested relative-periodic density on one attribute.
    args = {{p}, {[]}, 'specs', {spec}, 'sigma', sigma, 'isPer', true, ...
            'period', P, 'verbose', false};
    if ~isempty(wrap)
        args = [args, {'wrap', {wrap}}];
    end
    d = buildExpTens(args{:});
end


function T = nmrTuples(col)
    % Every nested tuple of the r = [2, 2], sym = [true true] attribute on
    % two units of two values: the outer level picks an ordered pair of
    % units and each unit contributes an ordered pair of its values, so the
    % tuple set is the full wreath-product orbit (|G| = 2!^2 * 2! = 8) of
    % the single combination. Independent of INTERNAL.NESTEDENUMINDICES: the
    % point of the reference is to share no code with the route under test.
    units = {col(1:2).', col(3:4).'};
    orders = [1 2; 2 1];
    T = zeros(8, 4);
    row = 0;
    for a = 1:2
        ua = units{orders(a, 1)};
        ub = units{orders(a, 2)};
        for p = 1:2
            for q = 1:2
                row = row + 1;
                T(row, :) = [ua(orders(p, :)), ub(orders(q, :))];
            end
        end
    end
end


function v = nmrRefIp(px, py, sigma, P)
    % All-image transposition average over the period: the wrapped Gaussian
    % per coordinate, product across the tuple, summed over tuple pairs and
    % averaged over a 4000-node tau grid.
    ntau = 4000;
    taus = (0:ntau - 1) * (P / ntau);
    v = 0;
    for i = 1:size(px, 2)
        for j = 1:size(py, 2)
            X = nmrTuples(px(:, i));      % (Tx, 4)
            Y = nmrTuples(py(:, j));      % (Ty, 4)
            Tx = size(X, 1);  Ty = size(Y, 1);  r = size(X, 2);
            d = reshape(X, [Tx, 1, r, 1]) ...
              - reshape(Y, [1, Ty, r, 1]) ...
              - reshape(taus, [1, 1, 1, ntau]);
            K = internal.wrappedGaussian1d(d, sigma, P, Inf, 4);
            v = v + mean(sum(sum(prod(K, 3), 1), 2), 4);
        end
    end
end


function c = nmrRefCos(px, py, sigma, P)
    c = nmrRefIp(px, py, sigma, P) / sqrt( ...
        nmrRefIp(px, px, sigma, P) * nmrRefIp(py, py, sigma, P));
end
