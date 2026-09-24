%% demo_jmm_3_3_xcorr.m
% Analysis 3.3 (Online Supplement, Section 10): phase as lag, a windowed
% cross-correlogram.
%
% A demo of the Music Perception Toolbox reproducing the analysis from the
% JMM article; lightly edited from the article's own script. Data come
% from the jmm package (BWV 347 read from the bundled MusicXML) or
% jmm.pianoPhase (the rendered Piano Phase voices); the figures stay on screen unless
% SAVE_FIGURES is set.
%
% Analysis 3.3: phase as lag in Reich's Piano Phase.
%
% The query is a single canonical cell (Piano 1's repeating pattern); the
% context is Piano 2's event stream. At each anchor (a Piano-1 cell
% boundary) the query is translated in time by a lag tau spanning one
% whole cell, and the one-sided matched-filter response against Piano 2
% is read off (simMaet with 'normalize', 'oneSidedDenom', the
% Analysis-1.4 idiom). Collecting these rows gives a cross-correlogram
% R(anchor, tau); its bright ridge tracks the running phase offset
% between the two pianos, climbing the staircase 0 -> 12 pulses across
% the piece. Secondary ridges a half-cell away are expected from the
% cell's near-period-6 internal self-similarity.
%
% Pre-MAET structure:
%
%     attribute   sigma           rel   per
%     ---------   --------------  ----  ----
%     pitch       0.15 semitone   no    no
%     time        0.015 s         no    no
%
%     r = (1, 1); query = 12-event canonical cell; context = Piano 2;
%     one-sided normalization (divide by the query self-overlap).
%
% Data: jmm.pianoPhase (the rendered Piano Phase voices). Toolbox:
% preMaetFromAttrTable, packPreMaet, windowedSimilarity. Runtime: half a
% minute.

% The demo folder is located from the toolbox root, and adding it puts
% the +jmm helper package in scope.
mptRoot = which('buildMaet');
if isempty(mptRoot)
    error('demoJmm:toolboxNotFound', ...
        ['The toolbox is not on the path. Add the matlab folder of the ' ...
         'Music Perception Toolbox, then run this demo again.']);
end
thisDir = fullfile(fileparts(mptRoot), 'demos', 'jmm');
addpath(thisDir);
clear mptRoot

% Set true to write the figures to a
% figures/ folder beside this script; false leaves them on screen only.
SAVE_FIGURES = false;

prevDefaults = mptDefaults('showHints', false, 'truncationSigmas', 4.0);

% --- parameters --------------------------------------------------------------
SIGMA_PITCH = 0.15;
SIGMA_TIME  = 0.015;                % 15 ms
pe = jmm.pianoPhase();
IOI         = pe.baseIoi;
CELL_DUR    = pe.nc * IOI;          % one cell in seconds
N_TAU       = 241;                  % lag samples over one cell (resolves ~33 ms ridge)
ASTEP       = 2;                    % anchor every ASTEP Piano-1 cells

isRel   = [false false];
isPer   = [false false];
periods = [0 0];
sigma   = [SIGMA_PITCH, SIGMA_TIME];
rVec    = [1 1];

% --- query: one canonical cell (Piano 1's pattern), cell starting at t = 0 --
qPitch = double(pe.cell);                       % 1 x 12
qTime  = (0:(pe.nc - 1)) * IOI;                 % 1 x 12
queryPAttr = {qPitch, qTime};

% --- context: Piano 2 ---------------------------------------------------------
context = preMaetFromAttrTable(pe.voice2Table, 'attributes', { ...
    struct('column', 'pitch', 'sigma', sigma(1)), ...
    struct('column', 'onset', 'name', 'time', 'sigma', sigma(2))}, ...
    'time', 'seconds', 'chords', 'separate', 'weights', 'ones');
ctxPAttr = unpackPreMaet(context);
t2 = ctxPAttr{2};

% --- anchors and lag grid -----------------------------------------------------
nCells = pe.nRepsV1;
anchors = (0:ASTEP:(nCells - 1)) * CELL_DUR;
tauGrid = (0:(N_TAU - 1)) * (CELL_DUR / N_TAU);  % linspace without the endpoint

% Single windowedSimilarity call producing the whole (anchor, lag)
% correlogram. The context (Piano 2) is localized by a rectangular window
% of full support 2 * HALF at each anchor; the query is translated to each
% lag. The output shape follows queryCentres: an (anchors x N_TAU) matrix
% of absolute query centres gives an (anchors x N_TAU) response, with the
% context broadcast along the lag axis.
%
% Query placement. windowedSimilarity translates the query so its mean on
% the time axis lands at queryCentres(i, j); the original loop translated
% the query by (a - tau) directly, i.e. landed its mean at (a - tau) + mu_q.
% So queryCentres = (a - tau) + mu_q reproduces the original placement.
% Piano 2 leads, so retarding the query by tau makes the ridge read the
% running phase k directly.
HALF = CELL_DUR + 2 * IOI;
muQ = mean(qTime);
queryCentres = (anchors(:) - tauGrid) + muQ;    % (anchors, N_TAU)
% The query is read under the context's geometry, so it carries the same
% specs.
[~, ~, ctxSpecs] = unpackPreMaet(context);
query = packPreMaet(queryPAttr, [], ctxSpecs);
showPreMaet(context, 'maxEvents', 4);
showPreMaet(query, 'maxEvents', 4);

R = windowedSimilarity( ...
    context, query, anchors, ...
    'queryCentres', queryCentres, ...
    'contextWindow', {1.0, 2 * HALF}, ...       % rectangle, full support 2 * HALF
    'normalize', 'oneSidedDenom', ...
    'windowAttr', 2, ...
    'dropWindowAttr', false, ...
    'verbose', false);
% Reproduce the original sparse-context skip: blank anchors whose
% localization window holds fewer than one full cell of context events.
ctxCounts = arrayfun(@(a) sum((t2 >= a - HALF) & (t2 <= a + HALF)), anchors);
R(ctxCounts < pe.nc, :) = NaN;

% --- true lag staircase for overlay ------------------------------------------
trueLag = pe.lagAt(anchors / CELL_DUR);

% --- figure ------------------------------------------------------------------
fig = figure('Position', [50 50 1300 460], 'Color', 'w');
ax = axes('Parent', fig, 'Position', [0.06 0.15 0.80 0.75]);
imagesc(ax, [anchors(1), anchors(end)], [0, pe.nc], R.');
set(ax, 'YDir', 'normal');
colormap(ax, jmm.colourMap('magma'));
caxis(ax, [0, max(R(:))]);
hold(ax, 'on');
hLag = plot(ax, anchors, trueLag, 'Color', [0.224 0.827 1.0], 'LineWidth', 1.1);
for s = pe.shiftCentres
    plot(ax, [s s], [0, pe.nc], 'Color', [0.55 0.55 0.55], 'LineWidth', 0.4);
end
xlabel(ax, 'anchor time \alpha (s)', 'FontSize', 15);
ylabel(ax, 'lag \tau (pulses)', 'FontSize', 15);
set(ax, 'YTick', 0:2:pe.nc, 'FontSize', 13);
title(ax, ['Lag cross-correlation: canonical cell (Piano 1) vs Piano 2 ' ...
           '(one-sided matched filter)'], 'FontSize', 16);
legend(hLag, {'true phase k'}, 'Location', 'northwest', 'FontSize', 12);
cb = colorbar(ax);
ylabel(cb, 'matched-filter response', 'FontSize', 13);

figDir = fullfile(thisDir, 'figures');
if SAVE_FIGURES && ~exist(figDir, 'dir'), mkdir(figDir); end
if SAVE_FIGURES
    print(fig, '-dpng', '-r140', fullfile(figDir, 'demo_jmm_3_3_xcorr.png'));
    fprintf('Saved figures/demo_jmm_3_3_xcorr.png\n');
end
Rf = R(~isnan(R));
fprintf('R range %.3f-%.3f, %d anchors x %d lags\n', ...
        min(Rf), max(Rf), numel(anchors), N_TAU);

% The demo leaves the toolbox as it found it: the defaults it set at the
% top are restored here.
mptDefaults(prevDefaults);
