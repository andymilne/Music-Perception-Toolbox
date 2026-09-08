%% demo_jmm_3_1_diff.m
% Analysis 3.1 (Section 4.3.1 of the JMM article).
%
% A demo of the Music Perception Toolbox reproducing the analysis from the
% JMM article; lightly edited from the article's own script. Data come
% from the jmm package (BWV 347 read from the bundled MusicXML) or
% jmm.pianoPhase (the rendered Piano Phase voices); a figure is written to
% figures/.
%
% Analysis 3.1: joint differencing on pitch and time in Reich's Piano Phase.
%
% The phasing voice (Piano 2) is differenced jointly on pitch and time via
% differenceEvents with per-attribute orders [1, 1, 0]: the pitch and onset
% attributes are first-differenced to (dp, dt), while a third copy of the
% onset attribute is passed through (order 0) to carry absolute time for
% the windowing sweep (after alignment all three share the N-1 grid). A
% broad Gaussian window is swept over the piece and the windowed Renyi-2
% entropy of the (dp, dt) density is read at each centre.
%
% The analysis is run at two values of the time-difference kernel width:
%
%   * sigma_t = 6 ms --- the just-noticeable difference for inter-onset
%     intervals in an isochronous sequence (Friberg & Sundberg 1995). The
%     accelerandi shift the IOI over a 135.4-137.8 ms range (a 2.4 ms
%     excursion, below the JND), so at this width the (dp, dt) fingerprint
%     is indistinguishable everywhere and the entropy is flat while the
%     phase staircase climbs 0 -> 12 pulses --- the foil that motivates
%     Analyses 3.2 (phase as texture) and 3.3 (phase as lag).
%
%   * sigma_t = 0.1 ms --- far below the JND. At this super-human
%     resolution the sub-JND IOI excursion is resolved: the entropy
%     fluctuates strongly, rising where Piano 2's tempo is modulating (the
%     accelerandi by which it advances its phase). This is voice 2's own
%     tempo change becoming visible, not the inter-voice phase (which
%     single-voice differencing quotients out).
%
% Plotting both on a shared scale shows that matching sigma_t to the
% perceptual JND is what aligns the analysis with what a listener hears.
%
% Pre-MAET structure (after differencing):
%
%     attribute   order  sigma            rel    per
%     ---------   -----  ---------------  -----  -----
%     dp          1      0.5 semitone     no     no
%     dt          1      6 ms / 0.1 ms    no     no
%     abs onset   0      --- (window axis, deleted after weighting)
%
%     r = (1, 1); estimator: windowed Renyi-2 (Gaussian window, s.d. 6 s).
%
% The Python mirror is demos/jmm/demo_jmm_3_1_diff.py.

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
addpath(fullfile(thisDir, '..', '..'));
mptDefaults('showHints', false, 'truncationSigmas', 4.0, 'kernelPrecision', 'double');

SIGMA_DP    = 0.5;           % semitones
SIGMA_JND   = 0.006;         % seconds: IOI JND in an isochronous sequence
SIGMA_FINE  = 0.0001;        % seconds: 0.1 ms, far below the JND
WINDOW_SD   = 6.0;           % seconds; Gaussian time window for the entropy sweep
N_SWEEP     = 200;
pe = jmm.pianoPhase();
IOI = pe.baseIoi;

C_JND  = [0.122 0.306 0.722];   % blue   --- perceptual (JND-matched) line
C_FINE = [0.557 0.184 0.620];   % purple --- sub-JND (super-human) line

% --- joint differencing of the phasing voice --------------------------------
pitch = pe.voice2.pitch;
onset = pe.voice2.onset;
pAttr = {pitch, onset, onset};                   % each 1 x N
% Per-attribute difference orders: pitch and onset first-differenced, the
% third (onset copy) passed through at order 0 as the windowing axis.
[pd, wd, ~] = unpackPreMaet(differenceEvents(pAttr, [], [1 1 0]));
dp = pd{1}(:).'; dt = pd{2}(:).'; tAbs = pd{3}(:).';
fprintf('Differenced events: %d; dp distinct: [%s]\n', numel(dp), ...
        strjoin(arrayfun(@(v) sprintf('%d', v), unique(round(dp)), ...
                         'UniformOutput', false), ', '));
fprintf('IOI (=dt) min/max: %.2f / %.2f ms  (excursion %.2f ms)\n', ...
        min(dt) * 1000, max(dt) * 1000, (max(dt) - min(dt)) * 1000);

showPreMaet({pd{1}, pd{2}}, [], [], 'names', {'dp', 'dt'}, ...
    'sigma', [SIGMA_DP, SIGMA_JND], 'isPer', [false false], ...
    'maxEvents', 4, 'decimals', 3);

% --- (a) static (dp, dt) density over the whole voice (at the JND width) ---
static = buildExpTens({pd{1}, pd{2}}, [], [SIGMA_DP, SIGMA_JND], [1 1], ...
                      [false false], [false false], [0 0], 'verbose', false);
dpGrid = linspace(min(dp) - 2, max(dp) + 2, 200);
dtGrid = linspace(min(dt) - 0.04, max(dt) + 0.04, 120);
[DP, DT] = meshgrid(dpGrid, dtGrid);
Z = evalExpTens(static, [DP(:).'; DT(:).'], 'verbose', false);
Z = reshape(Z, size(DP));
fprintf('static density evaluated\n');

% --- (b) windowed (dp, dt) Renyi-2 entropy across the piece, two widths ---
centres = linspace(min(tAbs), max(tAbs), N_SWEEP);
phaseAt = pe.lagAt(centres / (pe.nc * IOI));     % continuous lag

% Windowed (dp, dt) Renyi-2 entropy at each sweep centre. A single
% windowedEntropy sweep: a Gaussian window (shape 0) on the absolute-onset
% axis (attribute 3) modulates the event weights, and that onset axis is
% dropped from the entropy density ('dropWindowAttr', true), leaving the
% two-attribute (dp, dt) density whose Renyi-2 entropy is returned. The
% window standard deviation WINDOW_SD maps to the variance-matched
% rectangular width 2*sqrt(3)*sd. (The placeholder onset sigma is unused:
% that axis is dropped.)
sweep = @(sig) windowedEntropy( ...
    pd, wd, ...
    [SIGMA_DP, sig, 1.0], [1 1 1], ...
    [false false false], [false false false], [0 0 0], ...
    centres, ...
    'contextWindow', {0.0, WINDOW_SD * 2.0 * sqrt(3.0)}, ...
    'method', 'renyi2', ...
    'windowAttr', 3, 'dropWindowAttr', true, ...
    'verbose', false);

H_jnd  = sweep(SIGMA_JND);
H_fine = sweep(SIGMA_FINE);
tags = {'6 ms (JND)', '0.1 ms'}; Hs = {H_jnd, H_fine};
for k = 1:2
    h = Hs{k}(~isnan(Hs{k}));
    fprintf('sigma_t = %10s: entropy %.3f..%.3f nats (range %.4f)\n', ...
            tags{k}, min(h), max(h), max(h) - min(h));
end

% --- figure -----------------------------------------------------------------
fig = figure('Position', [50 50 1500 440], 'Color', 'w');
axD = axes('Parent', fig, 'Position', [0.05 0.16 0.30 0.72]);
axH = axes('Parent', fig, 'Position', [0.45 0.16 0.44 0.72]);

contourf(axD, DP, DT * 1000, Z, 24, 'LineStyle', 'none');
colormap(axD, jmm.colourMap('magma'));
hold(axD, 'on');
plot(axD, dp, dt * 1000, '.', 'Color', [0.85 0.85 0.85], 'MarkerSize', 4);
xlabel(axD, '\Deltap (semitones)', 'FontSize', 15);
set(axD, 'XTick', -10:2:10, 'FontSize', 13, 'Box', 'off');
ylabel(axD, '\Deltat (ms)', 'FontSize', 15);
title(axD, 'Static (\Deltap, \Deltat) density (whole voice)', 'FontSize', 16);
cb = colorbar(axD);
ylabel(cb, 'density', 'FontSize', 13);

hold(axH, 'on');
lo = min(min(H_jnd), min(H_fine));
hi = max(max(H_jnd), max(H_fine));
pad = 0.12 * (hi - lo);
for s = pe.shiftCentres
    plot(axH, [s s], [lo - pad, hi + pad], 'Color', [0.88 0.66 0.52], 'LineWidth', 0.7);
end
hJnd = plot(axH, centres, H_jnd,  'Color', C_JND,  'LineWidth', 1.9);
hFine = plot(axH, centres, H_fine, 'Color', C_FINE, 'LineWidth', 1.6);
ylim(axH, [lo - pad, hi + pad]);
xlim(axH, [centres(1), centres(end)]);
xlabel(axH, 'window-centre offset (s); accelerandi marked orange, phase grey', 'FontSize', 15);
ylabel(axH, 'Renyi-2 entropy (nats)', 'FontSize', 15);
title(axH, 'Windowed (\Deltap, \Deltat) entropy at two kernel widths', 'FontSize', 16);
legend([hJnd, hFine], {'\sigma_t = 6 ms (IOI JND): flat', ...
                       '\sigma_t = 0.1 ms: resolves tempo modulation'}, ...
       'Location', 'west', 'FontSize', 12);
grid(axH, 'on');
set(axH, 'FontSize', 13, 'Box', 'off');
% Phase k on a second axis at the right.
ax2 = axes('Parent', fig, 'Position', get(axH, 'Position'), 'Color', 'none', ...
           'YAxisLocation', 'right', 'XTick', [], 'YColor', [0.6 0.6 0.6]);
hold(ax2, 'on');
plot(ax2, centres, phaseAt, 'Color', [0.733 0.733 0.733], 'LineWidth', 1.0);
set(ax2, 'YTick', 0:3:12, 'YLim', [-1 13], 'XLim', [centres(1), centres(end)], ...
         'FontSize', 13, 'Box', 'off');
ylabel(ax2, 'phase k (pulses)', 'Color', [0.6 0.6 0.6], 'FontSize', 15);

figDir = fullfile(thisDir, 'figures');
if ~exist(figDir, 'dir'), mkdir(figDir); end
print(fig, '-dpng', '-r140', fullfile(figDir, 'demo_jmm_3_1_diff.png'));
fprintf('saved figures/demo_jmm_3_1_diff.png\n');
