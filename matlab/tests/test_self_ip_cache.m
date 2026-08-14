%% test_self_ip_cache.m — memoised self IPs, oneSidedDenom skip, r = 1 route
%
%  Tests for the self-inner-product memoisation, the 'oneSidedDenom'
%  <X,X> skip, and the direct all-r = 1 inner-product route in
%  cosSimExpTens. Covers:
%    - Sweep (scalar-vs-cell broadcast) returns exactly the values that
%      fresh scalar calls return, under both normalisations (the memo
%      is a pure optimisation; values are unchanged).
%    - The same equality on the Möbius route (method = 'mobius'
%      forced), exercising the orbit-side memo and choices key.
%    - The cache-carrying outputs: a manual scalar loop threading
%      [s, densRef] = cosSimExpTens(densRef, ...) matches the sweep,
%      and the returned struct carries a populated 'selfIP' field.
%    - Cross-language goldens: r = 1 two-attribute similarities on
%      formula-based inputs match the Python toolbox to 1e-12 relative,
%      in all three abs-mode combinations and both normalisations
%      (this pins the r = 1 direct route against the independently
%      validated Python path), plus one Möbius abs r = 3 golden.
%    - The cache-carrying outputs are refused outside the
%      density-struct scalar form.
%
%  Standalone-runnable. When invoked from test_mpt.m the existing
%  `results` cell is appended to; when run alone, results print on the
%  fly and a summary follows.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

%% ---- Shared deterministic inputs (formula-based; identical in Python) ----

iVec = 1:12;
P = 50 + mod(7 * iVec.^2, 37);
T = cumsum(mod(3 * iVec, 5) / 4 + 0.2);
qP = [60 63.5 68.25];
qT = [0.5 1.25 2.0];

modesPer  = {[false false], [true false], [true true]};
modesPrd  = {[0 0], [12 0], [12 4]};
modesName = {'nonper', 'perA', 'perAB'};

% Python-computed goldens (goldens_inf.py; formula-based inputs above),
% computed with truncation_sigmas = inf --- the un-truncated
% accuracy-floor path that mptTestIsolateDefaults pins as the suite
% baseline (the factory default of 6 is a ~6-sig-fig approximation,
% which a 1e-12 golden would fail by construction; the perAB mode
% coincides at either setting because every wrapped distance there
% falls inside the effective truncation window).
golden = struct();
golden.nonper.cosine        = [6.682242385236272e-03, 1.968354486993972e-03, 1.730893616999029e-05];
golden.nonper.oneSidedDenom = [1.364172585654229e-02, 4.018374484498330e-03, 3.533600676040842e-05];
golden.perA.cosine          = [5.737534202374079e-02, 1.245711904636613e-01, 2.814032744888095e-01];
golden.perA.oneSidedDenom   = [1.306592271612683e-01, 2.836824130095172e-01, 6.408316372239535e-01];
golden.perAB.cosine         = [3.162529342551098e-01, 2.510893754426644e-01, 4.565179404859586e-01];
golden.perAB.oneSidedDenom  = [8.376450969950919e-01, 6.650492737482304e-01, 1.209158787535208e+00];

%% ---- Sweep == fresh scalars, plus cross-language goldens (r = 1) ----

for mi = 1:3
    isPerM = modesPer{mi};
    prdM   = modesPrd{mi};
    dX = buildExpTens({P, T}, [], [0.9 0.35], [1 1], [false false], ...
        isPerM, prdM, 'verbose', false);
    dYs = cell(1, 3);
    for k = 0:2
        dYs{k+1} = buildExpTens({qP + 0.6*k, qT + 0.9*k}, [], ...
            [0.9 0.35], [1 1], [false false], isPerM, prdM, ...
            'verbose', false);
    end
    for nrmC = {'cosine', 'oneSidedDenom'}
        nrm = nrmC{1};
        sweep = cell2mat(cosSimExpTens(dX, dYs, ...
            'normalize', nrm, 'verbose', false));
        fresh = zeros(1, 3);
        for k = 1:3
            dXf = buildExpTens({P, T}, [], [0.9 0.35], [1 1], ...
                [false false], isPerM, prdM, 'verbose', false);
            fresh(k) = cosSimExpTens(dXf, dYs{k}, ...
                'normalize', nrm, 'verbose', false);
        end
        results{end+1,1} = sprintf( ...
            'selfIP.r1 %s %s: sweep matches fresh scalars (0 diff)', ...
            modesName{mi}, nrm); %#ok<*SAGROW>
        results{end,2}   = max(abs(sweep - fresh)) == 0;

        g = golden.(modesName{mi}).(nrm);
        results{end+1,1} = sprintf( ...
            'selfIP.r1 %s %s: matches Python golden (1e-12 rel)', ...
            modesName{mi}, nrm);
        results{end,2}   = max(abs(sweep - g) ./ max(abs(g), 1e-300)) < 1e-12;
    end
end

%% ---- Cache-carrying outputs: threaded loop matches sweep ----

dX = buildExpTens({P, T}, [], [0.9 0.35], [1 1], [false false], ...
    [false false], [0 0], 'verbose', false);
dYs = cell(1, 3);
for k = 0:2
    dYs{k+1} = buildExpTens({qP + 0.6*k, qT + 0.9*k}, [], [0.9 0.35], ...
        [1 1], [false false], [false false], [0 0], 'verbose', false);
end
sweep = cell2mat(cosSimExpTens(dX, dYs, 'verbose', false));
dRef = dX;
sLoop = zeros(1, 3);
for k = 1:3
    [sLoop(k), dRef] = cosSimExpTens(dRef, dYs{k}, 'verbose', false);
end
results{end+1,1} = 'selfIP.threaded loop via 2nd output matches sweep (0 diff)';
results{end,2}   = max(abs(sLoop - sweep)) == 0;
results{end+1,1} = 'selfIP.returned struct carries a populated selfIP field';
results{end,2}   = isfield(dRef, 'selfIP') && isstruct(dRef.selfIP) ...
    && numel(dRef.selfIP.keys) >= 1;

%% ---- Möbius route: sweep == fresh, and abs r = 3 golden ----

jVec = 1:15;
xFlat = 100 * mod(11 * jVec.^2, 19) / 19 + 10 * jVec;
yFlat = 100 * mod(13 * jVec.^2, 23) / 23 + 9 * jVec;
% MATLAB reshape is column-major; NumPy's is row-major. Transpose the
% 3-by-5 column-major reshape to obtain NumPy's (5, 3) row-major array.
X3 = reshape(xFlat, 3, 5).';
Y3 = reshape(yFlat, 3, 5).';
bld3 = @(v) buildExpTens({v}, [], 8.0, 3, false, false, 0, 'verbose', false);

goldMob = struct('cosine', 1.566062218777721e-01, ...
                 'oneSidedDenom', 1.497935199140709e-01);
for nrmC = {'cosine', 'oneSidedDenom'}
    nrm = nrmC{1};
    dX3 = bld3(X3);
    sMemo1 = cosSimExpTens(dX3, bld3(Y3), 'method', 'mobius', ...
        'normalize', nrm, 'verbose', false);
    sFresh = cosSimExpTens(bld3(X3), bld3(Y3), 'method', 'mobius', ...
        'normalize', nrm, 'verbose', false);
    results{end+1,1} = sprintf( ...
        'selfIP.mobius %s: repeated-operand call matches fresh (0 diff)', nrm);
    results{end,2}   = abs(sMemo1 - sFresh) == 0;
    results{end+1,1} = sprintf( ...
        'selfIP.mobius %s: matches Python golden (1e-9 rel)', nrm);
    results{end,2}   = abs(sFresh - goldMob.(nrm)) ...
        / abs(goldMob.(nrm)) < 1e-9;
end

%% ---- Cache-carrying outputs refused outside the scalar struct form ----

okListRefused = false;
try
    [~, ~] = cosSimExpTens(dX, dYs, 'verbose', false); %#ok<ASGLU>
catch ME
    okListRefused = strcmp(ME.identifier, ...
        'cosSimExpTens:selfIpOutputsUnavailable');
end
results{end+1,1} = 'selfIP.2nd output in list mode raises selfIpOutputsUnavailable';
results{end,2}   = okListRefused;

%% ---- Summary (standalone only) ----

if standalone
    nPass = sum(cell2mat(results(:,2)));
    fprintf('\ntest_self_ip_cache: %d/%d passed\n', nPass, size(results, 1));
    for ri = 1:size(results, 1)
        if ~results{ri,2}
            fprintf('  FAIL: %s\n', results{ri,1});
        end
    end
end
