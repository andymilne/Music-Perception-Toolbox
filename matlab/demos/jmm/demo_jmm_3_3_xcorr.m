%% demo_jmm_3_3_xcorr.m
% Analysis 3.3 (Online Supplement, Section 10): phase as lag, a
% cross-correlogram.
%
% A demo of the Music Perception Toolbox reproducing the analysis from the
% JMM article; lightly edited from the article's own script. Data come
% from jmm.pianoPhase (the rendered Piano Phase voices); the figures stay
% on screen unless SAVE_FIGURES is set.
%
% Analysis 3.3: phase as lag in Reich's Piano Phase.
%
% The query is a single canonical cell (Piano 1's repeating pattern); the
% context is Piano 2's event stream. At each anchor alpha (a Piano-1 cell
% boundary) the query is translated in time by a lag tau spanning one
% whole cell, and its one-sided similarity against Piano 2 is read off
% ('normalize', 'oneSidedDenom', as in Analysis 1.3). Collecting these rows
% gives the supplement's cross-correlogram R(alpha, tau) =
% <f_X, f_Y^(alpha - tau)> / <f_Y, f_Y>: a translation sweep of the query
% by alpha - tau, which sweepSimMaet computes in one pass. R = 1 where
% Piano 2 holds one cell-aligned copy of the query. The bright ridge
% (R = 1) tracks the running phase k between the two pianos, climbing the
% staircase 0 -> 12 pulses across the piece. Three fainter ridges
% (R = 0.5) run at lags k + 4, k + 6, and k + 8 pulses (mod 12): rotated
% by 4, 6, or 8 positions, the cell agrees with itself at 6 of its 12
% positions, and at every other nonzero rotation at none of them. The
% maximum therefore fixes the lag uniquely.
%
% Pre-MAET structure:
%
%     attribute   sigma           rel   per
%     ---------   --------------  ----  ----
%     pitch       0.15 semitone   no    no
%     time        0.015 s         no    no
%
%     r = (1, 1); query = 12-event canonical cell; context = Piano 2;
%     one-sided normalization (divide by the query's self inner product).
%
% Data: jmm.pianoPhase (the rendered Piano Phase voices). Toolbox:
% preMaetFromAttrTable, packPreMaet, buildMaet, sweepSimMaet. Runtime: a
% few seconds.

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

% --- query: one canonical cell (Piano 1's pattern), cell starting at t = 0 --
qPitch = double(pe.cell);                       % 1 x 12
qTime  = (0:(pe.nc - 1)) * IOI;                 % 1 x 12
queryPAttr = {qPitch, qTime};

% --- context: Piano 2 ---------------------------------------------------------
context = preMaetFromAttrTable(pe.voice2Table, 'attributes', { ...
    struct('column', 'pitch', 'sigma', SIGMA_PITCH), ...
    struct('column', 'onset', 'name', 'time', 'sigma', SIGMA_TIME)}, ...
    'time', 'seconds', 'chords', 'separate', 'weights', 'ones');
ctxPAttr = unpackPreMaet(context);
t2 = ctxPAttr{2};

% --- anchors and lag grid -----------------------------------------------------
nCells = pe.nRepsV1;
anchors = (0:ASTEP:(nCells - 1)) * CELL_DUR;
tauGrid = (0:(N_TAU - 1)) * (CELL_DUR / N_TAU);  % linspace without the endpoint

% The correlogram is a translation sweep: at anchor alpha and lag tau the
% query (the cell, written from t = 0) is translated in time by
% alpha - tau and compared with the whole of Piano 2. Piano 2 leads, so
% retarding the query by tau makes the ridge read the running phase k
% directly. sweepSimMaet takes every (anchor, lag) offset in one pass: the
% pitch row of the offsets is zero (pitch is not translated), the time row
% holds alpha - tau, and the result is reshaped to (anchors, N_TAU).
offsets = anchors(:) - tauGrid;                 % (anchors, N_TAU)
% The query is read under the context's geometry, so it carries the same
% specs.
[~, ~, ctxSpecs] = unpackPreMaet(context);
query = packPreMaet(queryPAttr, [], ctxSpecs);
showPreMaet(context, 'maxEvents', 4);
showPreMaet(query, 'maxEvents', 4);

R = sweepSimMaet(buildMaet(context, 'verbose', false), ...
    buildMaet(query, 'verbose', false), ...
    [zeros(1, numel(offsets)); offsets(:).'], ...
    'normalize', 'oneSidedDenom', 'verbose', false);
R = reshape(R, size(offsets));
% Blank the anchors near the ends of the piece, where the span a cell can
% reach from the anchor (one cell and two pulses either side) holds fewer
% than one full cell of Piano 2's events.
HALF = CELL_DUR + 2 * IOI;
ctxCounts = arrayfun(@(a) sum((t2 >= a - HALF) & (t2 <= a + HALF)), anchors);
R(ctxCounts < pe.nc, :) = NaN;

% --- true lag staircase for overlay ------------------------------------------
trueLag = pe.lagAt(anchors / CELL_DUR);

% --- figure ------------------------------------------------------------------
fig = figure('Position', [50 50 1300 460], 'Color', 'w');
ax = axes('Parent', fig, 'Position', [0.06 0.15 0.80 0.70]);
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
% The title is raised clear of the tick labels and the colour bar, so a
% crop can remove it cleanly.
hTitle = title(ax, ['Lag cross-correlation: canonical cell (Piano 1) vs Piano 2 ' ...
                    '(one-sided similarity)'], 'FontSize', 16);
set(hTitle, 'Units', 'normalized', 'VerticalAlignment', 'bottom');
hTitle.Position(2) = 1.09;
legend(hLag, {'known phase k'}, 'Location', 'northwest', 'FontSize', 12);
cb = colorbar(ax);
ylabel(cb, 'one-sided similarity R(\alpha, \tau)', 'FontSize', 13);

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
