%% test_entropy_four_method_api.m — four-method entropy API
%
%  Tests for entropyExpTens's method kwarg in its post-Stage-2 form:
%  'differential', 'shannon', 'normalized' (with British 'normalised'
%  alias), 'renyi2'. Also exercises the policy guards (sigma=0
%  rejection for continuous methods; explicit grid required for
%  discrete methods) and propagation through spectralEntropy and
%  nTupleEntropy.
%
%  Standalone-runnable; appends to `results` when called from
%  test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

% =====================================================================
% Method canonicalization and validation
% =====================================================================

dens1d = buildExpTens([100, 200, 300], ones(1, 3), 20, 1, false, false, 0, ...
                       'verbose', false);

% --- British 'normalised' alias matches 'normalized' ---
H_us = entropyExpTens(dens1d, 'method', 'normalized', ...
                       'nPointsPerDim', 200, 'xMin', 50, 'xMax', 350, ...
                       'verbose', false);
H_uk = entropyExpTens(dens1d, 'method', 'normalised', ...
                       'nPointsPerDim', 200, 'xMin', 50, 'xMax', 350, ...
                       'verbose', false);
results{end+1,1} = 'four-method: normalised alias == normalized';
results{end,2}   = abs(H_us - H_uk) < 1e-14;

% --- Bogus method rejected with informative error ---
results{end+1,1} = 'four-method: bogus method rejected';
results{end,2}   = throwsErrorWithId(@() entropyExpTens(dens1d, ...
    'method', 'bogus', 'nPointsPerDim', 200, 'verbose', false), ...
    'entropyExpTens:badMethod');

% =====================================================================
% Grid-requirement policy
% =====================================================================

dens = buildExpTens([100, 200, 300], ones(1, 3), 20, 1, false, false, 0, ...
                     'verbose', false);

% --- Shannon without explicit grid errors ---
results{end+1,1} = 'gridReq: shannon without nPointsPerDim errors';
results{end,2}   = throwsErrorWithId(@() entropyExpTens(dens, ...
    'method', 'shannon', 'verbose', false), ...
    'entropyExpTens:gridRequired');

% --- Normalized without explicit grid errors ---
results{end+1,1} = 'gridReq: normalized without nPointsPerDim errors';
results{end,2}   = throwsErrorWithId(@() entropyExpTens(dens, ...
    'method', 'normalized', 'verbose', false), ...
    'entropyExpTens:gridRequired');

% --- Differential works without grid ---
hDiff = entropyExpTens(dens, 'method', 'differential', 'verbose', false);
results{end+1,1} = 'gridReq: differential works without nPointsPerDim';
results{end,2}   = isfinite(hDiff);

% --- Renyi2 works without grid ---
hRny = entropyExpTens(dens, 'method', 'renyi2', ...
                       'verbose', false);
results{end+1,1} = 'gridReq: renyi2 works without nPointsPerDim';
results{end,2}   = isfinite(hRny);

% =====================================================================
% sigma=0 policy on continuous methods
% =====================================================================

densSigmaZero = buildExpTens([100, 200, 300], ones(1, 3), 0, 1, ...
                              false, false, 0, 'verbose', false);

% --- Differential at sigma=0 errors ---
results{end+1,1} = 'sigma=0: differential rejects sigma=0';
results{end,2}   = throwsErrorWithId(@() entropyExpTens(densSigmaZero, ...
    'method', 'differential', 'verbose', false), ...
    'entropyExpTens:sigmaZeroNotSupported');

% --- Renyi2 at sigma=0 errors ---
results{end+1,1} = 'sigma=0: renyi2 rejects sigma=0';
results{end,2}   = throwsErrorWithId(@() entropyExpTens(densSigmaZero, ...
    'method', 'renyi2', 'verbose', false), ...
    'entropyExpTens:sigmaZeroNotSupported');

% =====================================================================
% 'normalized' equals 'shannon' (raw) / log_b(N)
% =====================================================================

N = 200;
xMin = 50; xMax = 350;
H_shan = entropyExpTens(dens, 'method', 'shannon', ...
                         'nPointsPerDim', N, 'xMin', xMin, 'xMax', xMax, ...
                         'verbose', false);
H_norm = entropyExpTens(dens, 'method', 'normalized', ...
                         'nPointsPerDim', N, 'xMin', xMin, 'xMax', xMax, ...
                         'verbose', false);
results{end+1,1} = 'four-method: normalized == shannon / log2(N)';
results{end,2}   = abs(H_norm - H_shan / log2(N)) < 1e-12;

% =====================================================================
% Differential entropy: convergence and grid-independence
% =====================================================================

% --- 1-D non-periodic: adaptive vs fine fixed grid ---
densSE = buildExpTens([6000, 6300, 6700], ones(1, 3), 20, 1, ...
                       false, false, 0, 'verbose', false);
hDiff1D = entropyExpTens(densSE, 'method', 'differential', 'verbose', false);
N_ref = 25000;
xMinRef = 5500; xMaxRef = 7200;
H_shanRef = entropyExpTens(densSE, 'method', 'shannon', ...
                            'nPointsPerDim', N_ref, ...
                            'xMin', xMinRef, 'xMax', xMaxRef, ...
                            'verbose', false);
h_hat_ref = H_shanRef + log2((xMaxRef - xMinRef) / (N_ref - 1));
results{end+1,1} = 'differential 1D: adaptive matches fixed-grid reference';
results{end,2}   = abs(hDiff1D - h_hat_ref) < 1e-5;

% --- Grid-independence: concentrated vs spread ---
densA = buildExpTens([6500, 6600], ones(1, 2), 15, 1, ...
                      false, false, 0, 'verbose', false);
densB = buildExpTens([6000, 6400, 6800, 7200], ones(1, 4), 15, 1, ...
                      false, false, 0, 'verbose', false);
h_A = entropyExpTens(densA, 'method', 'differential', 'verbose', false);
h_B = entropyExpTens(densB, 'method', 'differential', 'verbose', false);
results{end+1,1} = 'differential: grid-indep ordering (spread > concentrated)';
results{end,2}   = h_B > h_A;

% --- truncationSigmas tightening converges ---
h_ts6 = entropyExpTens(densSE, 'method', 'differential', ...
                        'truncationSigmas', 6.0, 'verbose', false);
h_ts8 = entropyExpTens(densSE, 'method', 'differential', ...
                        'truncationSigmas', 8.0, 'verbose', false);
results{end+1,1} = 'differential: ts=6 and ts=8 agree to 1e-4';
results{end,2}   = isfinite(h_ts6) && isfinite(h_ts8) && abs(h_ts8 - h_ts6) < 1e-4;

% --- truncationSigmas=Inf resolves to the accuracy-floor width. Under ---
% --- the truncation contract Inf (the user-facing "exact" sentinel)   ---
% --- resolves to the finite accuracy-floor width (~7.43 sigma, the    ---
% --- 1e-12 floor), including the differential span and tolerance      ---
% --- anchoring. It must therefore equal truncationSigmas set to the   ---
% --- accuracy-floor width bit-for-bit, and differ from ts=6.0 by the  ---
% --- ~6-sigma truncation-error scale (~2e-8).                         ---
h_tsInf = entropyExpTens(densSE, 'method', 'differential', ...
                          'truncationSigmas', Inf, 'verbose', false);
h_tsFloor = entropyExpTens(densSE, 'method', 'differential', ...
                          'truncationSigmas', internal.accuracyFloor('sigmas'), ...
                          'verbose', false);
results{end+1,1} = 'differential: truncationSigmas=Inf resolves to accuracy floor';
results{end,2}   = isfinite(h_tsInf) && (h_tsInf == h_tsFloor) ...
                   && abs(h_tsInf - h_ts6) > 1e-9;

% --- 2-D periodic (the prior OOM case before Richardson) ---
P2 = [1, 2, 4, 5];
W2 = [1, 1, 1, 1];
period = 12.0;
sigmaEff = sqrt(2);
densMA = buildExpTens({P2, P2}, {W2, W2}, [sigmaEff sigmaEff], ...
                       [1, 1], [false false], [true true], [period period], ...
                       'verbose', false);
hDiffMA = entropyExpTens(densMA, 'method', 'differential', ...
                          'truncationSigmas', 6, 'verbose', false);
H_shanMA = entropyExpTens(densMA, 'method', 'shannon', ...
                           'nPointsPerDim', 500, ...
                           'truncationSigmas', 6, 'verbose', false);
h_hat_ref_MA = H_shanMA + 2.0 * log2(period / 500.0);
results{end+1,1} = 'differential 2D: adaptive matches N=500 reference';
results{end,2}   = isfinite(hDiffMA) && abs(hDiffMA - h_hat_ref_MA) < 1e-3;

% =====================================================================
% n_tuple_entropy with method kwarg
% =====================================================================

p_diatonic = [0, 2, 4, 5, 7, 9, 11];

% --- Default equals 'normalized' ---
H_def = nTupleEntropy(p_diatonic, 12, 2);
H_norm = nTupleEntropy(p_diatonic, 12, 2, 'method', 'normalized');
results{end+1,1} = 'nTupleEntropy: default == method=normalized';
results{end,2}   = abs(H_def - H_norm) < 1e-14;

% --- 'normalised' alias ---
H_uk = nTupleEntropy(p_diatonic, 12, 2, 'method', 'normalised');
results{end+1,1} = 'nTupleEntropy: normalised alias == normalized';
results{end,2}   = abs(H_norm - H_uk) < 1e-14;

% --- Differential at sigma=0 rejected ---
results{end+1,1} = 'nTupleEntropy: differential rejects sigma=0';
results{end,2}   = throwsErrorWithId(@() nTupleEntropy(p_diatonic, 12, 2, ...
    'method', 'differential'), ...
    'nTupleEntropy:continuousNeedsSigmaPositive');

% --- Renyi2 at sigma=0 rejected ---
results{end+1,1} = 'nTupleEntropy: renyi2 rejects sigma=0';
results{end,2}   = throwsErrorWithId(@() nTupleEntropy(p_diatonic, 12, 2, ...
    'method', 'renyi2'), ...
    'nTupleEntropy:continuousNeedsSigmaPositive');

% --- Continuous methods work at sigma > 0 ---
H_diff = nTupleEntropy(p_diatonic, 12, 2, 'sigma', 0.5, ...
                       'method', 'differential');
H_rny = nTupleEntropy(p_diatonic, 12, 2, 'sigma', 0.5, ...
                      'method', 'renyi2');
results{end+1,1} = 'nTupleEntropy: continuous methods finite at sigma>0';
results{end,2}   = isfinite(H_diff) && isfinite(H_rny);

% =====================================================================
% spectralEntropy four-method dispatch
% =====================================================================

majorTriad = [0, 400, 700];

% --- Default is differential ---
H_seDef = spectralEntropy(majorTriad, [], 12, 'verbose', false);
H_seDiff = spectralEntropy(majorTriad, [], 12, 'method', 'differential', ...
                            'verbose', false);
results{end+1,1} = 'spectralEntropy: default == differential';
results{end,2}   = abs(H_seDef - H_seDiff) < 1e-14;

% --- All four methods run ---
H_seNorm = spectralEntropy(majorTriad, [], 12, 'method', 'normalized', ...
                            'verbose', false);
H_seShan = spectralEntropy(majorTriad, [], 12, 'method', 'shannon', ...
                            'verbose', false);
H_seRny = spectralEntropy(majorTriad, [], 12, 'method', 'renyi2', ...
                           'verbose', false);
results{end+1,1} = 'spectralEntropy: all four methods return finite values';
results{end,2}   = isfinite(H_seDiff) && isfinite(H_seNorm) ...
                   && isfinite(H_seShan) && isfinite(H_seRny);

% --- 'normalised' alias ---
H_seUK = spectralEntropy(majorTriad, [], 12, 'method', 'normalised', ...
                          'verbose', false);
results{end+1,1} = 'spectralEntropy: normalised alias == normalized';
results{end,2}   = abs(H_seNorm - H_seUK) < 1e-14;

% --- 'normalized' in [0, 1] ---
results{end+1,1} = 'spectralEntropy: normalized in [0,1]';
results{end,2}   = H_seNorm >= 0 && H_seNorm <= 1;

% --- JI lower entropy than EDO under differential (with partials) ---
spec_h = {'harmonic', 12, 'powerlaw', 1};
H_ji = spectralEntropy([0, 386.31, 701.96], [], 12, ...
                        'spectrum', spec_h, 'method', 'differential', ...
                        'verbose', false);
H_edo = spectralEntropy([0, 400, 700], [], 12, ...
                         'spectrum', spec_h, 'method', 'differential', ...
                         'verbose', false);
results{end+1,1} = 'spectralEntropy differential: JI < EDO with partials';
results{end,2}   = H_ji < H_edo;


%% ---- Per-method input-form coverage ----------------------------------
% entropyExpTens supports five input forms; shannon and normalized
% support all five, differential and renyi2 support three (scalar
% density, raw single multiset scalar, raw MA scalar) and reject list / batched.

% Set up input fixtures.
sa_p     = [100, 200, 300];
sa_w     = [1, 1, 1];
sa_sigma = 20.0;
sa_dens  = buildExpTens(sa_p, sa_w, sa_sigma, 1, false, false, 0, ...
    'verbose', false);
sa_dens_list = {
    buildExpTens(sa_p,        sa_w, sa_sigma, 1, false, false, 0, 'verbose', false), ...
    buildExpTens(sa_p +  50,  sa_w, sa_sigma, 1, false, false, 0, 'verbose', false), ...
    buildExpTens(sa_p + 100,  sa_w, sa_sigma, 1, false, false, 0, 'verbose', false)};
sa_P = [sa_p; sa_p + 50; sa_p + 100; sa_p + 150];
sa_W = ones(size(sa_P));

% MA fixture: 2 attributes, 4 events, 2 groups, both non-periodic absolute.
ma_pAttr  = {[100, 200, 300, 400], [10, 20, 30, 40]};
ma_w      = {ones(1, 4), ones(1, 4)};
ma_dens   = buildExpTens(ma_pAttr, ma_w, [20, 5], [1, 1], ...
    [false, false], [false, false], [0, 0], 'verbose', false);

% --- Shannon: 5 input forms ---
h_sa_dens   = entropyExpTens(sa_dens, 'method', 'shannon', ...
    'nPointsPerDim', 200, 'xMin', 0, 'xMax', 500, 'verbose', false);
h_sa_list   = entropyExpTens(sa_dens_list, 'method', 'shannon', ...
    'nPointsPerDim', 200, 'xMin', 0, 'xMax', 500, 'verbose', false);
h_sa_raw    = entropyExpTens(sa_p, sa_w, sa_sigma, 1, false, false, 0, ...
    'method', 'shannon', 'nPointsPerDim', 200, ...
    'xMin', 0, 'xMax', 500, 'verbose', false);
h_sa_batch  = entropyExpTens(sa_P, sa_W, sa_sigma, 1, false, false, 0, ...
    'method', 'shannon', 'nPointsPerDim', 200, ...
    'xMin', 0, 'xMax', 500, 'verbose', false);
h_ma_dens   = entropyExpTens(ma_dens, 'method', 'shannon', ...
    'nPointsPerDim', 80, 'xMin', [0, 0], 'xMax', [500, 50], 'verbose', false);
results{end+1,1} = 'shannon input forms: all five return finite values';
results{end,2}   = isfinite(h_sa_dens) ...
                   && iscell(h_sa_list) && numel(h_sa_list) == 3 ...
                   && all(cellfun(@isfinite, h_sa_list)) ...
                   && isfinite(h_sa_raw) ...
                   && numel(h_sa_batch) == 4 && all(isfinite(h_sa_batch)) ...
                   && isfinite(h_ma_dens);

% --- Normalized: 5 input forms, values in [0, 1] ---
n_sa_dens   = entropyExpTens(sa_dens, 'method', 'normalized', ...
    'nPointsPerDim', 200, 'xMin', 0, 'xMax', 500, 'verbose', false);
n_sa_list   = entropyExpTens(sa_dens_list, 'method', 'normalized', ...
    'nPointsPerDim', 200, 'xMin', 0, 'xMax', 500, 'verbose', false);
n_sa_raw    = entropyExpTens(sa_p, sa_w, sa_sigma, 1, false, false, 0, ...
    'method', 'normalized', 'nPointsPerDim', 200, ...
    'xMin', 0, 'xMax', 500, 'verbose', false);
n_sa_batch  = entropyExpTens(sa_P, sa_W, sa_sigma, 1, false, false, 0, ...
    'method', 'normalized', 'nPointsPerDim', 200, ...
    'xMin', 0, 'xMax', 500, 'verbose', false);
n_ma_dens   = entropyExpTens(ma_dens, 'method', 'normalized', ...
    'nPointsPerDim', 80, 'xMin', [0, 0], 'xMax', [500, 50], 'verbose', false);
inUnit = @(x) x >= 0 && x <= 1;
results{end+1,1} = 'normalized input forms: all five return values in [0, 1]';
results{end,2}   = inUnit(n_sa_dens) ...
                   && all(cellfun(inUnit, n_sa_list)) ...
                   && inUnit(n_sa_raw) ...
                   && all(arrayfun(inUnit, n_sa_batch)) ...
                   && inUnit(n_ma_dens);

% --- Differential: 3 input forms, list and batched rejected ---
% The MA form is D==2; at the tightest accuracy a 2-D differential grid is
% infeasible, so pin a feasible truncationSigmas (the scalar forms are 1-D
% and converge at any accuracy). The check is only that a finite value
% comes back.
d_sa_dens = entropyExpTens(sa_dens, 'method', 'differential', 'verbose', false);
d_sa_raw  = entropyExpTens(sa_p, sa_w, sa_sigma, 1, false, false, 0, ...
    'method', 'differential', 'verbose', false);
d_ma_dens = entropyExpTens(ma_dens, 'method', 'differential', ...
    'truncationSigmas', 5, 'verbose', false);
results{end+1,1} = 'differential input forms: 3 scalar forms return finite';
results{end,2}   = isfinite(d_sa_dens) && isfinite(d_sa_raw) && isfinite(d_ma_dens);

results{end+1,1} = 'differential rejects list input';
results{end,2}   = throwsError(@() entropyExpTens( ...
    sa_dens_list, 'method', 'differential', 'verbose', false));

results{end+1,1} = 'differential rejects 2-D batched input';
results{end,2}   = throwsError(@() entropyExpTens( ...
    sa_P, sa_W, sa_sigma, 1, false, false, 0, ...
    'method', 'differential', 'verbose', false));

% --- Rényi-2: 3 input forms, list and batched rejected ---
r_sa_dens = entropyExpTens(sa_dens, 'method', 'renyi2', 'verbose', false);
r_sa_raw  = entropyExpTens(sa_p, sa_w, sa_sigma, 1, false, false, 0, ...
    'method', 'renyi2', 'verbose', false);
r_ma_dens = entropyExpTens(ma_dens, 'method', 'renyi2', 'verbose', false);
results{end+1,1} = 'renyi2 input forms: 3 scalar forms return finite';
results{end,2}   = isfinite(r_sa_dens) && isfinite(r_sa_raw) && isfinite(r_ma_dens);

results{end+1,1} = 'renyi2 rejects list input';
results{end,2}   = throwsError(@() entropyExpTens( ...
    sa_dens_list, 'method', 'renyi2', 'verbose', false));

results{end+1,1} = 'renyi2 rejects 2-D batched input';
results{end,2}   = throwsError(@() entropyExpTens( ...
    sa_P, sa_W, sa_sigma, 1, false, false, 0, ...
    'method', 'renyi2', 'verbose', false));


%% ---- v2.2 migration error: 'normalize' kwarg removed ------------------

% Passing 'normalize' (any value) to entropyExpTens raises a TypeError-
% equivalent (id entropyExpTens:normalizeRemoved). Tested across each
% method and each value, plus a check that the message points to the
% replacement methods.
methods_to_test = {'shannon', 'normalized', 'differential', 'renyi2'};
values_to_test  = {true, false};
all_migration_ok = true;
last_msg = '';
for mi = 1:numel(methods_to_test)
    for vi = 1:numel(values_to_test)
        ok_one = false;
        try
            entropyExpTens(sa_dens, 'method', methods_to_test{mi}, ...
                'normalize', values_to_test{vi}, ...
                'nPointsPerDim', 100, 'xMin', 0, 'xMax', 500, ...
                'verbose', false);
        catch ME
            ok_one = strcmp(ME.identifier, 'entropyExpTens:normalizeRemoved');
            last_msg = ME.message;
        end
        all_migration_ok = all_migration_ok && ok_one;
    end
end
results{end+1,1} = 'entropyExpTens: legacy normalize kwarg raises migration error';
results{end,2}   = all_migration_ok;

% Migration message points to replacement methods.
results{end+1,1} = 'entropyExpTens: migration message names ''normalized'' and ''shannon''';
results{end,2}   = contains(last_msg, 'method=''normalized''') ...
                   && contains(last_msg, 'method=''shannon''');

% spectralEntropy: same migration behaviour.
ok_se_false = false;
try
    spectralEntropy([0, 400, 700], [], 12, ...
        'method', 'shannon', 'normalize', false, 'verbose', false);
catch ME
    ok_se_false = strcmp(ME.identifier, 'spectralEntropy:normalizeRemoved');
end
ok_se_true = false;
try
    spectralEntropy([0, 400, 700], [], 12, ...
        'method', 'normalized', 'normalize', true, 'verbose', false);
catch ME
    ok_se_true = strcmp(ME.identifier, 'spectralEntropy:normalizeRemoved');
end
results{end+1,1} = 'spectralEntropy: legacy normalize kwarg raises migration error';
results{end,2}   = ok_se_false && ok_se_true;


%% ---- truncationSigmas=3 fast-path contract (differential) -------------

% truncationSigmas=3 loosens the differential convergence tolerance and
% should return a value within a fifth-decimal of the ts=6 reference,
% while preserving consonance ordering of two clearly-separated densities.
dens_diff_ref = buildExpTens([6000, 6300, 6700], ones(1,3), 20.0, 1, ...
    false, false, 0, 'verbose', false);
h_ts3 = entropyExpTens(dens_diff_ref, 'method', 'differential', ...
    'truncationSigmas', 3, 'verbose', false);
h_ts6 = entropyExpTens(dens_diff_ref, 'method', 'differential', ...
    'truncationSigmas', 6, 'verbose', false);
results{end+1,1} = 'differential ts=3: finite and within 1e-2 of ts=6';
results{end,2}   = isfinite(h_ts3) && abs(h_ts3 - h_ts6) < 1e-2;

dens_concentrated = buildExpTens([6500, 6600], ones(1, 2), 15.0, 1, ...
    false, false, 0, 'verbose', false);
dens_spread       = buildExpTens([6000, 6400, 6800, 7200], ones(1, 4), 15.0, ...
    1, false, false, 0, 'verbose', false);
h_conc   = entropyExpTens(dens_concentrated, 'method', 'differential', ...
    'truncationSigmas', 3, 'verbose', false);
h_spread = entropyExpTens(dens_spread, 'method', 'differential', ...
    'truncationSigmas', 3, 'verbose', false);
results{end+1,1} = 'differential ts=3: preserves consonance ordering';
results{end,2}   = h_spread > h_conc;




if standalone
    nPass = sum([results{:, 2}]);
    nFail = size(results, 1) - nPass;
    for i = 1:size(results, 1)
        if results{i, 2}
            fprintf('  PASS  %s\n', results{i, 1});
        else
            fprintf('  FAIL  %s\n', results{i, 1});
        end
    end
    fprintf('\n=== test_entropy_four_method_api: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_entropy_four_method_api:failed', '%d test(s) failed.', nFail);
    end
end
