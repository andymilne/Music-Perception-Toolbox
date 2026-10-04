%% test_massMaet.m — the total mass of a density
%
%  massMaet's total mass is the sum of the density's tuples' weight
%  products, each kernel taken with unit mass. It is pinned to hand-
%  computed weight sums; sweptMass is pinned to the composition it stands
%  for (weightEvents -> buildMaet -> massMaet); and its use as the
%  normalizer of a one-sided similarity is pinned on the diatonic scale's
%  fifths. Mirror of Python's tests/test_mass_maet.py.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end
prevHints = mptDefaults('showHints', false);

pM = {[60; 64; 67]};
wM = {[1; 0.5; 2]};
flatM = @(sigma, r, rel, per, period, varargin) buildMaet(pM, wM, sigma, r, ...
    rel, per, period, varargin{:}, 'verbose', false);
closeM = @(a, b, tol) abs(a - b) <= tol;

% ---- totals -----------------------------------------------------------------
% Weights 1, 0.5, and 2: r = 1 sums them (3.5); an exchangeable pair takes
% both orders of each pair of values, 2 (0.5 + 2 + 1) = 7; an ordered pair
% one order only, 3.5; an exchangeable triple all six orders of the one
% triple, 6.
results(end+1, :) = {'massMaet total is the sum of weight products', ...
    closeM(massMaet(flatM(1, 1, false, false, 0)), 3.5, 1e-13) ...
    && closeM(massMaet(flatM(1, 2, false, false, 0)), 7, 1e-13) ...
    && closeM(massMaet(flatM(1, 2, true, false, 0)), 7, 1e-13) ...
    && closeM(massMaet(flatM(1, 2, true, true, 1200)), 7, 1e-13) ...
    && closeM(massMaet(buildMaet(pM, wM, 1, 2, false, false, 0, false, ...
        'verbose', false)), 3.5, 1e-13) ...
    && closeM(massMaet(flatM(1, 3, true, false, 0)), 6, 1e-13)}; %#ok<SAGROW>

% ---- several attributes: a product per event ------------------------------------
d = buildMaet({[60 62 64 65], [0 1 2 3]}, {[1 2 1 0.5], [1 1 3 2]}, ...
    [1 0.2], [1 1], [false false], [false false], [0 0], 'verbose', false);
results(end+1, :) = {'massMaet multi-attribute product per event', ...
    closeM(massMaet(d), 1 + 2 + 3 + 1, 1e-13)}; %#ok<SAGROW>

% ---- nested and kernel covariance ------------------------------------------------
pit = [60 62 64 65 67];
[pb, wb, sb] = unpackPreMaet(bindEvents({pit, 0:4}, [], [2 1], 'step', 1));
dN = buildMaet(pb, wb, 'sigma', [0.5 0.2], 'per', [false false], ...
    'period', [0 0], 'specs', sb, 'verbose', false);
% Four bound events, each an ordered pair of pitches (weight 1) and an onset.
dc = buildMaet([0; 1; 2], ones(3, 1), 0.25 * eye(3), 3, false, false, 0, false, ...
    'verbose', false);
results(end+1, :) = {'massMaet nested and kernel-covariance attributes', ...
    closeM(massMaet(dN), 4, 1e-12) && closeM(massMaet(dc), 1, 1e-13)}; %#ok<SAGROW>

% ---- pre-MAET and cell inputs ---------------------------------------------------
pmM = packPreMaet(pM, wM, flatSpecs(pM, 'sigma', 1, 'per', false, 'period', 0));
dP = buildMaet(pmM, 'verbose', false);
outM = massMaet({dP, pmM});
results(end+1, :) = {'massMaet pre-MAET and cell inputs', ...
    massMaet(pmM) == massMaet(dP) && isequal(size(outM), [1 2])}; %#ok<SAGROW>

% ---- no region --------------------------------------------------------------------
results(end+1, :) = {'massMaet takes no region', ...
    localRefusedAny(@() massMaet(dP, 'region', {1, [61 66]}))}; %#ok<SAGROW>

% ---- the normalizer of a one-sided similarity -------------------------------------
% A relative, periodic, exchangeable dyad density: the one-sided similarity
% with a lone fifth counts the scale's pairs a fifth or a fourth apart, and
% the ratio of masses makes it their share of all pairs (6 of 21), up to the
% kernels' overlap with neighbouring intervals (about 2e-5 at sigma = 10).
diatM = [0 200 400 500 700 900 1100];
fifthM = [0 700];
countM = simMaet(diatM, [], fifthM, [], 10, 2, true, true, 1200, ...
    'normalize', 'oneSidedDenom', 'verbose', false);
mD = massMaet(buildMaet(diatM, [], 10, 2, true, true, 1200, 'verbose', false));
mF = massMaet(buildMaet(fifthM, [], 10, 2, true, true, 1200, 'verbose', false));
results(end+1, :) = {'massMaet normalizes a one-sided similarity', ...
    closeM(countM, 6, 1e-4) && closeM(mD, 42, 1e-12) && closeM(mF, 2, 1e-12) ...
    && closeM(countM * mF / mD, 6 / 21, 1e-5)}; %#ok<SAGROW>

% ---- sweptMass ----------------------------------------------------------------
pitS = [60 62 64 65 67 65 64 62 60 67 72 67];
onS = 0:11;
pmS = packPreMaet({pitS, onS}, [], flatSpecs({pitS, onS}, 'sigma', [0.5 0.1], ...
    'per', [false false], 'period', [0 0]));
valsS = 0:2:10;
gotS = sweptMass(pmS, 'sweep', {2, valsS}, 'window', {2, {'gaussian', 'sd', 1.5}}, ...
    'drop', 2);
okS = true;
for i = 1:numel(valsS)
    [pw, ww] = unpackPreMaet(weightEvents(pmS, 2, 1, valsS(i), 0, ...
        'sd', 1.5, 'dropInputAttr', true));
    dS = buildMaet(pw, ww, 0.5, 1, false, false, 0, 'verbose', false);
    okS = okS && closeM(gotS(i), massMaet(dS), 1e-12);
end
results(end+1, :) = {'sweptMass matches weightEvents -> buildMaet -> massMaet', okS}; %#ok<SAGROW>
results(end+1, :) = {'sweptMass counts the window', ...
    all(abs(sweptMass(pmS, 'sweep', {2, [2 5]}, 'window', {2, {'rect', 'width', 4}}, ...
    'drop', 2) - 4) < 1e-12)}; %#ok<SAGROW>
results(end+1, :) = {'sweptMass takes no region', ...
    localRefusedAny(@() sweptMass(pmS, 'sweep', {2, 2}, ...
    'window', {2, {'rect', 'width', 4}}, 'drop', 2, 'region', {1, [60 62]}))}; %#ok<SAGROW>

mptDefaults(prevHints);

if standalone
    nPass = sum([results{:, 2}]);
    fprintf('test_massMaet: %d/%d passed\n', nPass, size(results, 1));
    for i = 1:size(results, 1)
        if ~results{i, 2}
            fprintf('  FAIL: %s\n', results{i, 1});
        end
    end
end

function tf = localRefusedAny(f)
try
    f();
    tf = false;
catch
    tf = true;
end
end
