%% demo_jmm_3_2_texture.m
% Analysis 3.2 (Section 4.3.2 of the JMM article).
%
% A demo of the Music Perception Toolbox reproducing the analysis from the
% JMM article; lightly edited from the article's own script. Data come
% from the jmm package (BWV 347 read from the bundled MusicXML) or
% jmm.pianoPhase (the rendered Piano Phase voices); a figure is written to
% figures/.
%
% Analysis 3.2: phase as local texture in Reich's Piano Phase.
%
% The two pianos are pooled into a single (pitch, time) event stream ---
% the voice label is not an attribute, so the measure reads the combined
% sounding texture rather than either line on its own. A broad Gaussian
% localization window (s.d. 3 s) is swept over the piece; at each sweep
% centre the windowed Renyi-2 entropy of the joint (pitch, time) density
% is read off. Because the window is smooth and wide, there is no
% rectangular-edge artefact and every window has ample mass.
%
% The single controlling parameter is the time kernel sigma_t:
%
%   * sigma_t = 15 ms  --- within the precedence-effect fusion window
%     (~5-30 ms), far narrower than the 138 ms pulse, so the density
%     resolves every event. Only exact vertical coincidences (the two
%     pianos striking the same pitch at the same instant) merge. This is
%     the coincidence reading: a high plateau cut by dips wherever the
%     canonical near-period-6 structure forces pitches to align (phase
%     k = 4, 6, 8).
%
%   * sigma_t = 100 ms --- a kernel about 0.7 of a pulse wide, so a pitch
%     the two pianos play one pulse apart now overlaps and merges. This is
%     the redundancy reading: the one-pulse canonic echo is counted as
%     repetition, giving a smooth two-humped profile (peaks at the
%     maximally de-correlated phases k ~ 2-3 and k ~ 9-10, a valley at the
%     half-cycle k ~ 6, troughs at the unisons k = 0, 12).
%
% Same surface, same window, same estimator; only the kernel width
% differs, and that single change carries the reading from coincidence to
% redundancy.
%
% Pre-MAET structure (both panels):
%
%     attribute   sigma            rel    per
%     ---------   ---------------  -----  -----
%     pitch       0.15 semitone    no     no
%     time        sigma_t (s)      no     no
%
%     r = (1, 1);  voices pooled (voice is not an attribute);
%     window: Gaussian (shape 0) on the time attribute, s.d. 3 s, centred
%     at the sweep offset, time retained ('dropWindowAttr', false);
%     estimator: Renyi-2.
%
% The Python mirror is demos/jmm/demo_jmm_3_2_texture.py.

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
addpath(fullfile(thisDir, '..', '..'));
mptDefaults('showHints', false, 'truncationSigmas', 3.0, 'kernelPrecision', 'double');

% --- fixed parameters --------------------------------------------------------
SIGMA_PITCH = 0.15;          % semitone (= 15 cents)
WIN_SD      = 3.0;           % localization-window s.d. (s)
PRUNE       = 4.0;           % keep events within PRUNE * WIN_SD of the centre
N_SWEEP     = 300;
SIGMAS_T    = [0.015, 0.100];   % coincidence (precedence/fusion window), redundancy
pe = jmm.pianoPhase();
IOI = pe.baseIoi;

% --- two-voice surface, voices pooled ----------------------------------------
pitch = pe.piece.pitch;
onset = pe.piece.onset;
tLo = min(onset); tHi = max(onset);
edge = 2 * WIN_SD;                                  % unreliable near the ends
centres = linspace(tLo, tHi, N_SWEEP);
phaseAt = pe.lagAt(centres / (pe.nc * IOI));        % continuous lag

% Windowed joint (pitch, time) Renyi-2 entropy at each sweep centre. A
% single windowedEntropy sweep: a Gaussian localization window (shape 0)
% on the time axis (attribute 2) modulates the event weights, with the
% time axis retained ('dropWindowAttr', false) so the joint (pitch, time)
% density is built and its Renyi-2 entropy returned. The window standard
% deviation WIN_SD maps to the variance-matched rectangular width
% 2*sqrt(3)*sd. The previous explicit prune to +/- PRUNE * WIN_SD is
% unnecessary here: the global truncationSigmas (set to 3.0 above, tighter
% than PRUNE = 4.0) already zeros every event the prune would have
% removed, so the result is identical.
sweep = @(sigmaT) windowedEntropy( ...
    {pitch, onset}, [], ...
    [SIGMA_PITCH, sigmaT], [1 1], ...
    [false false], [false false], [0 0], ...
    centres, ...
    'contextWindow', {0.0, WIN_SD * 2.0 * sqrt(3.0)}, ...
    'method', 'renyi2', ...
    'windowAttr', 2, 'dropWindowAttr', false, ...
    'verbose', false);

H = cell(1, numel(SIGMAS_T));
for k = 1:numel(SIGMAS_T)
    H{k} = sweep(SIGMAS_T(k));
end

% --- figure ------------------------------------------------------------------
fig = figure('Position', [50 50 1300 520], 'Color', 'w');
labels = {'\sigma_t = 15 ms', '\sigma_t = 100 ms'};
notes = {'coincidence: dips at k = 4, 6, 8', ...
         'redundancy: humps at k ~ 2-3, 9-10; valley at k ~ 6'}; %#ok<NASGU>
shifts = pe.shiftCentres;
interior = (centres > tLo + edge) & (centres < tHi - edge);

for k = 1:numel(SIGMAS_T)
    y0 = 0.14 + (2 - k) * 0.41;
    ax = axes('Parent', fig, 'Position', [0.07, y0, 0.86, 0.34]);
    hold(ax, 'on');
    yl = [min(H{k}(interior)) - 0.1, max(H{k}(interior)) + 0.1];
    fill(ax, [tLo, tLo + edge, tLo + edge, tLo], [yl(1) yl(1) yl(2) yl(2)], ...
         [0.6 0.6 0.6], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    fill(ax, [tHi - edge, tHi, tHi, tHi - edge], [yl(1) yl(1) yl(2) yl(2)], ...
         [0.6 0.6 0.6], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    for s = shifts
        plot(ax, [s s], yl, 'Color', [0.88 0.66 0.52], 'LineWidth', 0.6);
    end
    plot(ax, centres, H{k}, 'Color', [0.122 0.306 0.722], 'LineWidth', 1.7);
    ylim(ax, yl);
    xlim(ax, [tLo, tHi]);
    ylabel(ax, sprintf('%s\nRenyi-2', labels{k}), 'FontSize', 15);
    grid(ax, 'on');
    set(ax, 'FontSize', 13, 'Box', 'off');
    if k == numel(SIGMAS_T)
        xlabel(ax, 'window-centre offset (s); grey = phase k', 'FontSize', 15);
    else
        set(ax, 'XTickLabel', {});
    end
    % Phase k on a second axis at the right.
    ax2 = axes('Parent', fig, 'Position', get(ax, 'Position'), 'Color', 'none', ...
               'YAxisLocation', 'right', 'XTick', [], 'YColor', [0.6 0.6 0.6]);
    hold(ax2, 'on');
    plot(ax2, centres, phaseAt, 'Color', [0.733 0.733 0.733], 'LineWidth', 0.9);
    set(ax2, 'YTick', 0:3:12, 'YLim', [-1 13], 'XLim', [tLo, tHi], ...
             'FontSize', 13, 'Box', 'off');
end
annotation(fig, 'textbox', [0.05 0.93 0.9 0.06], 'String', ...
           'Texture entropy (voices pooled, Gaussian 3 s window): coincidence vs redundancy', ...
           'HorizontalAlignment', 'center', 'FontSize', 18, 'EdgeColor', 'none');

figDir = fullfile(thisDir, 'figures');
if ~exist(figDir, 'dir'), mkdir(figDir); end
print(fig, '-dpng', '-r140', fullfile(figDir, 'demo_jmm_3_2_texture.png'));
fprintf('saved figures/demo_jmm_3_2_texture.png\n');
