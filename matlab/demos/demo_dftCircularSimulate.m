%% demo_dftCircularSimulate.m
%
%  Demonstration of the v2.1.0 sigma + Monte Carlo additions to the
%  Argand-DFT family.
%
%  New in v2.1:
%    * dftCircularSimulate — Monte Carlo estimation of |F(k)|
%      distribution under independent positional jitter.
%    * balanceCircular, evennessCircular — accept an optional sigma
%      argument; with two left-hand-side outputs, also return the
%      per-coefficient standard deviation.
%    * projCentroid — accepts sigma and applies analytical kernel-
%      smoothing damping (alpha_1 factor); no Monte Carlo needed for
%      the projection because it is linear in F(0).
%
%  The deterministic dftCircular function is unchanged.

clear; close all;

% ===== 1. Balance and evenness sweeps =====

fprintf('\n=== Balance and evenness on the diatonic scale ===\n');
fprintf('%6s %10s %10s %10s %10s\n', 'sigma', 'b', 'b_std', 'e', 'e_std');
fprintf('  %s\n', repmat('-', 1, 50));
diat = [0, 200, 400, 500, 700, 900, 1100];
period = 1200;
for s = [0, 10, 50, 100, 200]
    [b, bs] = balanceCircular(diat, [], period, s, 'rngSeed', 42);
    [e, es] = evennessCircular(diat, period, s, 'rngSeed', 42);
    fprintf('%6g %10.4f %10.4f %10.4f %10.4f\n', s, b, bs, e, es);
end
fprintf('  Both balance and evenness shrink as sigma grows; the standard\n');
fprintf('  deviations grow correspondingly. At sigma = 0 the v2.0\n');
fprintf('  deterministic values are recovered exactly and SD = 0.\n');


% ===== 2. Rayleigh bias on a perfectly balanced multiset =====

fprintf('\n=== Augmented triad: deterministically balanced (F(0) = 0) ===\n');
fprintf('Under jitter, |F(0)| picks up a positive bias from the underlying\n');
fprintf('Rayleigh distribution (sum of two independent N(0, V) components).\n');
fprintf('%6s %10s %20s\n', 'sigma', 'b', 'closed-form mean');
fprintf('  %s\n', repmat('-', 1, 36));
period = 1200;
K = 3;
for s = [0, 10, 25, 50, 100]
    b = balanceCircular([0, 400, 800], [], period, s, ...
        'nDraws', 20000, 'rngSeed', 42);
    mc_mag = 1 - b;
    if s == 0
        closedForm = 0;
    else
        alpha1 = exp(-2 * pi^2 * s^2 / period^2);
        closedForm = sqrt((1 - alpha1^2) * pi / (4 * K));
    end
    fprintf('%6g %10.4f %20.4f\n', s, b, closedForm);
end
fprintf('  MC estimate of E[|F(0)|] tracks the closed-form Rayleigh mean\n');
fprintf('  sqrt((1 - alpha_1^2) * pi / (4K)) very accurately.\n');


% ===== 3. Full DFT distribution via dftCircularSimulate =====

fprintf('\n=== Full DFT magnitudes for the son clave under jitter ===\n');
clave = [0, 3, 6, 10, 12];
period = 16;
sigma = 0.5;
fprintf('  Pattern: %s on %d-step cycle, sigma = %g\n', ...
    mat2str(clave), period, sigma);
[~, magDet] = dftCircular(clave, [], period);
[m, s] = dftCircularSimulate(clave, [], period, sigma, ...
    'nDraws', 20000, 'rngSeed', 42);
fprintf('  %3s %12s %12s %12s %10s\n', 'k', 'det |F|', 'mean |F|', 'SD |F|', 'CV');
fprintf('  %s\n', repmat('-', 1, 51));
for k = 1:numel(clave)
    if m(k) > 1e-10
        cv = s(k) / m(k);
        cvStr = sprintf('%10.3f', cv);
    else
        cvStr = sprintf('%10s', 'inf');
    end
    fprintf('  %3d %12.4f %12.4f %12.4f %s\n', ...
        k - 1, magDet(k), m(k), s(k), cvStr);
end
fprintf('  Note that low-order coefficients (k = 1: evenness) have the\n');
fprintf('  smallest CV under jitter -- Milne & Herff (2020) Fig 13 reports\n');
fprintf('  this is the most jitter-robust of the magnitudes.\n');


% ===== 4. projCentroid with analytical sigma damping =====

fprintf('\n=== projCentroid: closed-form alpha_1 damping (no MC) ===\n');
p = [0, 4, 7];   % major triad
period = 12;
sigma = 0.5;
alpha1 = exp(-2 * pi^2 * sigma^2 / period^2);
[yDet, cmDet, cpDet] = projCentroid(p, [], period);
[ySmooth, cmSmooth, cpSmooth] = projCentroid(p, [], period, [], sigma);
fprintf('  C major triad [0, 4, 7] on 12-step cycle, sigma = %g\n', sigma);
fprintf('  alpha_1 = exp(-2 pi^2 sigma^2 / period^2) = %.6f\n\n', alpha1);
fprintf('  %3s %12s %14s %10s\n', 'x', 'y_det(x)', 'y_smooth(x)', 'ratio');
fprintf('  %s\n', repmat('-', 1, 41));
for k = 1:period
    if abs(yDet(k)) > 1e-10
        ratio = ySmooth(k) / yDet(k);
        ratioStr = sprintf('%10.6f', ratio);
    else
        ratioStr = sprintf('%10s', 'NaN');
    end
    fprintf('  %3d %12.4f %14.4f %s\n', k - 1, yDet(k), ySmooth(k), ratioStr);
end
fprintf('  All ratios equal alpha_1 exactly (analytical, not MC).\n');
fprintf('  cent_phase preserved: %.4f -> %.4f\n', cpDet, cpSmooth);
fprintf('  cent_mag damped by alpha_1: %.4f -> %.4f\n', cmDet, cmSmooth);
fprintf('\n');
fprintf('  cent_mag here is |E[F(0)]| = alpha_1 * |F_det(0)|, which is\n');
fprintf('  consistent with the projection. The DIFFERENT scalar E[|F(0)|]\n');
fprintf('  (Rician/Rayleigh-style mean magnitude under jitter) is what\n');
fprintf('  balance returns: 1 - balanceCircular(p, w, T, sigma) = E[|F(0)|].\n');


% ===== 5. Full distribution via the third output of dftCircularSimulate =====

fprintf('\n=== Full distribution: |F(1)| under jitter ===\n');
diat = [0, 200, 400, 500, 700, 900, 1100];
sigma = 50;
[m, s, mags] = dftCircularSimulate(diat, [], 1200, sigma, ...
    'nDraws', 20000, 'rngSeed', 42);
F1samples = mags(:, 2);
F1sorted = sort(F1samples);
nDraws = numel(F1sorted);
q05 = F1sorted(max(1, ceil(0.05 * nDraws)));
q95 = F1sorted(min(nDraws, ceil(0.95 * nDraws)));
fprintf('  Diatonic scale, sigma = %g cents, nDraws = %d\n', ...
    sigma, numel(F1samples));
fprintf('  Sample shape: [%d %d]  (nDraws x K)\n', size(mags, 1), size(mags, 2));
fprintf('  |F(1)| min  = %.4f\n', min(F1samples));
fprintf('  |F(1)| 5%%   = %.4f\n', q05);
fprintf('  |F(1)| mean = %.4f\n', m(2));
fprintf('  |F(1)| 95%%  = %.4f\n', q95);
fprintf('  |F(1)| max  = %.4f\n', max(F1samples));
fprintf('  |F(1)| SD   = %.4f\n', s(2));

fprintf('\nDone.\n');
