%% test_dft_montecarlo.m — dftCircularSimulate; balance/evenness/projCentroid sigma (v2.1)
%
%  Tests for dftCircularSimulate.
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


%
%  Tests for the MC sigma additions to balance, evenness, projCentroid,
%  and the new dftCircularSimulate function. Insert after the existing
%  Circular measures section (before sigmaSpace section, or just after
%  the existing dftCircular tests).

% --- dftCircularSimulate: sigma=0 exact -----------------------------

[m, s] = dftCircularSimulate([0, 200, 400, 500, 700, 900, 1100], [], ...
                              1200, 0, 'nDraws', 100, 'rngSeed', 42);
[~, magDet] = dftCircular([0, 200, 400, 500, 700, 900, 1100], [], 1200);
results{end+1,1} = 'dftCircularSimulate: sigma=0 mean = deterministic';
results{end,2}   = max(abs(m - magDet)) < 1e-12;
results{end+1,1} = 'dftCircularSimulate: sigma=0 SD = 0';
results{end,2}   = max(abs(s)) < 1e-12;

% --- dftCircularSimulate: small sigma -> deterministic --------------

[m, s] = dftCircularSimulate([0, 200, 400, 500, 700, 900, 1100], [], ...
                              1200, 1e-3, 'nDraws', 2000, 'rngSeed', 42);
results{end+1,1} = 'dftCircularSimulate: small sigma close to deterministic';
results{end,2}   = max(abs(m - magDet)) < 1e-3 && max(s) < 1e-3;

% --- dftCircularSimulate: closed-form E[|F(0)|^2] for augmented triad ---
%   Augmented triad has F_det(0) = 0, so:
%   E[|F(0)|^2] = (1 - alpha_1^2) * sum(w^2) / sum(w)^2 = (1 - alpha_1^2) / K
period = 1200; sigma = 50; K = 3;
alpha1 = exp(-2 * pi^2 * sigma^2 / period^2);
expectedF0sq = (1 - alpha1^2) / K;
[~, ~, samples] = dftCircularSimulate([0, 400, 800], [], period, sigma, ...
                                      'nDraws', 50000, 'rngSeed', 42);
mcF0sq = mean(samples(:, 1).^2);
results{end+1,1} = 'dftCircularSimulate: closed-form E[|F(0)|^2] (aug triad)';
results{end,2}   = abs(mcF0sq - expectedF0sq) < 5e-3;

% --- dftCircularSimulate: rngSeed reproducibility -------------------

m1 = dftCircularSimulate([0, 200, 400, 500, 700, 900, 1100], [], 1200, ...
                         50, 'nDraws', 1000, 'rngSeed', 42);
m2 = dftCircularSimulate([0, 200, 400, 500, 700, 900, 1100], [], 1200, ...
                         50, 'nDraws', 1000, 'rngSeed', 42);
results{end+1,1} = 'dftCircularSimulate: rngSeed reproducible';
results{end,2}   = isequal(m1, m2);

% --- dftCircularSimulate: returnSamples shape -----------------------

[~, ~, samples] = dftCircularSimulate([0, 200, 400, 500, 700, 900, 1100], ...
                                      [], 1200, 50, ...
                                      'nDraws', 500, 'rngSeed', 42);
results{end+1,1} = 'dftCircularSimulate: samples shape [nDraws x K]';
results{end,2}   = isequal(size(samples), [500, 7]);

% --- balanceCircular: sigma=0 backward compatibility ----------------

b = balanceCircular([0, 400, 800], [], 1200);
results{end+1,1} = 'balanceCircular: sigma=0 default scalar = 1 (aug triad)';
results{end,2}   = abs(b - 1) < 1e-10;

% --- balanceCircular: sigma=0 with explicit sigma argument ----------

b = balanceCircular([0, 400, 800], [], 1200, 0);
results{end+1,1} = 'balanceCircular: sigma=0 explicit = 1 (aug triad)';
results{end,2}   = abs(b - 1) < 1e-10;

% --- balanceCircular: sigma>0 returns Rayleigh bias ------------------

[b, bs] = balanceCircular([0, 400, 800], [], 1200, 50, ...
                          'nDraws', 50000, 'rngSeed', 42);
expectedRayleigh = sqrt((1 - alpha1^2) * pi / (4 * K));
results{end+1,1} = 'balanceCircular: sigma>0 reveals Rayleigh bias';
results{end,2}   = b < 1 && bs > 0 && ...
                   abs((1 - b) - expectedRayleigh) < 5e-3;

% --- balanceCircular: nDraws name-value works -----------------------

b = balanceCircular([0, 400, 800], [], 1200, 25, 'nDraws', 5000, 'rngSeed', 7);
results{end+1,1} = 'balanceCircular: accepts nDraws/rngSeed name-value args';
results{end,2}   = isfinite(b) && b >= 0 && b <= 1;

% --- evennessCircular: sigma=0 backward compatibility ---------------

e = evennessCircular([0, 200, 400, 600, 800, 1000], 1200);
results{end+1,1} = 'evennessCircular: sigma=0 default = 1 (whole-tone)';
results{end,2}   = abs(e - 1) < 1e-10;

% --- evennessCircular: sigma>0 returns scalar -----------------------

e = evennessCircular([0, 200, 400, 600, 800, 1000], 1200, 50, ...
                     'nDraws', 5000, 'rngSeed', 42);
results{end+1,1} = 'evennessCircular: sigma>0 returns valid scalar';
results{end,2}   = isfinite(e) && e >= 0 && e <= 1;

% --- evennessCircular: smoothing reduces evenness for irregular pattern ---

eDet = evennessCircular([0, 200, 400, 500, 700, 900, 1100], 1200);
eSmooth = evennessCircular([0, 200, 400, 500, 700, 900, 1100], 1200, 100, ...
                           'nDraws', 20000, 'rngSeed', 42);
results{end+1,1} = 'evennessCircular: smoothing reduces |F(1)| for diatonic';
results{end,2}   = eSmooth < eDet;

% --- evennessCircular: SD output ------------------------------------

[e, es] = evennessCircular([0, 200, 400, 500, 700, 900, 1100], 1200, 50, ...
                           'nDraws', 5000, 'rngSeed', 42);
results{end+1,1} = 'evennessCircular: SD output positive when sigma>0';
results{end,2}   = es > 0;

% --- projCentroid: alpha_1 damping is exact (closed-form) -----------

period = 12; sigma = 0.5;
alpha1 = exp(-2 * pi^2 * sigma^2 / period^2);
yDet = projCentroid([0, 4, 7], [], period);
ySmooth = projCentroid([0, 4, 7], [], period, [], sigma);
results{end+1,1} = 'projCentroid: y_smooth = alpha_1 * y_det (exact)';
results{end,2}   = max(abs(ySmooth - alpha1 * yDet)) < 1e-12;

% --- projCentroid: phase preserved ----------------------------------

[~, ~, cpDet] = projCentroid([0, 4, 7], [], 12);
[~, ~, cpSmooth] = projCentroid([0, 4, 7], [], 12, [], 1.0);
results{end+1,1} = 'projCentroid: phase preserved under sigma';
results{end,2}   = abs(cpDet - cpSmooth) < 1e-12;

% --- projCentroid: cent_mag damped by alpha_1 -----------------------

period = 1200; sigma = 100;
alpha1 = exp(-2 * pi^2 * sigma^2 / period^2);
[~, cmDet] = projCentroid([0, 200, 400, 500, 700, 900, 1100], [], period);
[~, cmSmooth] = projCentroid([0, 200, 400, 500, 700, 900, 1100], [], period, ...
                              [], sigma);
results{end+1,1} = 'projCentroid: cent_mag damped by alpha_1';
results{end,2}   = abs(cmSmooth - alpha1 * cmDet) < 1e-12;

% --- projCentroid: sigma=0 recovers v2 ------------------------------

[y0, cm0, cp0] = projCentroid([0, 4, 7], [], 12, [], 0);
[y1, cm1, cp1] = projCentroid([0, 4, 7], [], 12);
results{end+1,1} = 'projCentroid: sigma=0 explicit = v2';
results{end,2}   = isequal(y0, y1) && cm0 == cm1 && cp0 == cp1;


%% ---- Standalone summary ----

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
    fprintf('\n=== test_dft_montecarlo: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_dft_montecarlo:failed', '%d test(s) failed.', nFail);
    end
end
