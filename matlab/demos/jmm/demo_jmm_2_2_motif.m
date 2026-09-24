%% demo_jmm_2_2_motif.m
% Analysis 2.2 (Online Supplement, Section 8.1): motif discovery in
% Acknowledgement from pitch alone.
%
% A demo of the Music Perception Toolbox reproducing the analysis from
% the JMM article; lightly edited from the article's own script. Data
% come from the jmm package (Acknowledgement read from a MIDI
% transcription you supply); the figures stay on screen unless
% SAVE_FIGURES is set.
%
% Analysis 2.2: data-driven motif discovery in Coltrane's Acknowledgement.
%
% The melody is read as one event per note carrying pitch. The recurring
% four-note cell is then recovered as a peak in the density of short
% interval patterns, without being supplied in advance. Two routes reach
% the same transposition-invariant cell.
%
% Differenced route. Pitch is first-differenced to melodic intervals,
% collapsing every transposition of a figure onto the same interval
% pattern. Three consecutive intervals are then bound into an ordered
% super-event: each is a point in a three-dimensional interval space, and
% the density over those points is a smoothed recurrence count. A figure
% stated k times contributes k near-coincident points, so its interval
% triple stands out as a local maximum. Reading the density at every
% observed triple and ranking turns motif discovery into peak finding.
%
% Relative route. The same cell is reached without differencing, by
% binding four consecutive pitches into an ordered super-event taken
% relative: the common transposition is removed, so transposed statements
% again coincide, and three degrees of freedom remain --- the dimension
% of the differenced route's interval triple. The two densities therefore
% describe the same object through different internal metrics. The
% differenced form treats consecutive intervals independently; the
% relative form couples them through the pitch they share. With the
% interval kernel set to sqrt(2) times the per-pitch kernel --- an
% interval being a difference of two pitches, which is the scaling
% differenceEvents applies of its own accord --- the two rank the motifs
% identically, and differ only in that coupling: a faint shear in the
% relative density's slice, absent from the differenced one.
%
% A relative density is read in translation-reduced coordinates: an
% r-tuple minus its first value, so a cell's coordinates are its
% cumulative intervals. Both the cells and the slice grid are written
% that way for the relative route below.
%
% Pre-MAET structure:
%
%     attribute    order  sigma              rel  per
%     ----------   -----  -----------------  ---  ---
%     dp           3      sqrt(2) * 0.15 st  no   no     differenced route
%     pitch        4      0.15 st            yes  no     relative route
%
%     Ordered (exch = 0) in both. Estimator: the density read at each cell.
%
% Data: jmm.acknowledgement (the solo, from your own MIDI transcription at
% data/AwakeningSolo.mid). Toolbox: preMaetFromAttrTable, differenceEvents,
% bindEvents, buildMaet, evalMaet, showPreMaet. Runtime: a few seconds.

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

% Set true to write the figures to a figures/ folder beside this script;
% false leaves them on screen only.
SAVE_FIGURES = false;

prevDefaults = mptDefaults('showHints', false);

SIGMA_PITCH = 0.15;        % semitones (15 cents): the per-pitch uncertainty
R_DIFF      = 3;           % bound interval triples (a four-note cell)
R_REL       = 4;           % bound pitch quadruples (the same cell)
TOP         = 6;           % motifs shown in the ranking
ALS         = [3 -3 5];    % +m3, -m3, +P4: the "A Love Supreme" cell

C_ALS   = [0.761 0.314 0.031];   % the recurring "A Love Supreme" cell
C_OTHER = [0.122 0.306 0.722];   % the surrounding recurring cells

% --- the melody as a pre-MAET, one event per note ---------------------------
notes = jmm.acknowledgement();
fprintf('%d notes; pitch range %.0f-%.0f (MIDI); span %.1f QN\n', ...
        height(notes), min(notes.pitch), max(notes.pitch), ...
        max(notes.onsetBeats));
melody = preMaetFromAttrTable(notes, 'attributes', { ...
    struct('column', 'pitch', 'name', 'pitch', 'sigma', SIGMA_PITCH)}, ...
    'time', 'beats', 'chords', 'separate', 'weights', 'ones');

% --- differenced route: intervals bound into ordered triples -----------------
% differenceEvents widens the kernel by sqrt(2) itself, so the bound
% pre-MAET already carries the interval width and buildMaet needs no sigma.
diffRoute = bindEvents(differenceEvents(melody, 1), R_DIFF, 'step', 1);
showPreMaet(diffRoute, 'maxEvents', 3);
[pd, ~, ~] = unpackPreMaet(diffRoute);
cellsD = pd{1};                                   % 3 x nCells intervals
densD = evalMaet(buildMaet(diffRoute, 'verbose', false), cellsD, ...
                 'verbose', false);
[classD, countD, meanD] = localRank(cellsD, densD);

% --- relative route: pitches bound into ordered relative quadruples ---------
relRoute = bindEvents(melody, R_REL, 'step', 1, 'relOuter', true);
showPreMaet(relRoute, 'maxEvents', 3);
[pr, ~, ~] = unpackPreMaet(relRoute);
cellsR = pr{1};                                   % 4 x nCells pitches
ivR = diff(cellsR, 1, 1);                         % its interval triples
densR = evalMaet(buildMaet(relRoute, 'verbose', false), cumsum(ivR, 1), ...
                 'verbose', false);
[classR, countR, meanR] = localRank(ivR, densR);

% --- report -----------------------------------------------------------------
fprintf('\ndifferenced: %d ordered interval triples, %d distinct classes\n', ...
        size(cellsD, 2), size(classD, 1));
fprintf('relative:    %d ordered pitch quadruples, %d distinct classes\n\n', ...
        size(cellsR, 2), size(classR, 1));
fprintf('%4s  %16s  %5s  %11s  %9s   contour\n', ...
        'rank', 'interval class', 'count', 'differenced', 'relative');
for i = 1:TOP
    k = classD(i, :);
    [~, j] = ismember(k, classR, 'rows');
    if isequal(k, ALS)
        mark = '  <- A Love Supreme cell';
    else
        mark = '';
    end
    fprintf('%4d  %16s  %5d  %11.1f  %9.1f   %s%s\n', i, ...
            localTriple(k), countD(i), meanD(i), meanR(j), ...
            localContour(k), mark);
end
same = isempty(setdiff(classD(1:TOP + 2, :), classR(1:TOP + 2, :), 'rows'));
fprintf('\ntop-%d interval-class set identical: %d\n', TOP + 2, same);
[~, rankD] = ismember(ALS, classD, 'rows');
[~, rankR] = ismember(ALS, classR, 'rows');
fprintf('A Love Supreme cell %s: rank %d (differenced), rank %d (relative)\n', ...
        localTriple(ALS), rankD, rankR);

% --- the (+3, i2, i3) slice of each density ---------------------------------
% The plane of cells sharing the leading motif's first interval. The grid
% is written as interval triples for the differenced route and as their
% cumulative sums for the relative one.
gridStep = SIGMA_PITCH * sqrt(2) / 3;
gridAxis = -7:gridStep:7;
[I2, I3] = meshgrid(gridAxis, gridAxis);
gridIv = [repmat(ALS(1), 1, numel(I2)); I2(:).'; I3(:).'];
sliceD = reshape(evalMaet(buildMaet(diffRoute, 'verbose', false), gridIv, ...
                          'verbose', false), size(I2));
sliceR = reshape(evalMaet(buildMaet(relRoute, 'verbose', false), ...
                          cumsum(gridIv, 1), 'verbose', false), size(I2));
[~, iD] = max(sliceD(:));  [rD, cD] = ind2sub(size(sliceD), iD);
[~, iR] = max(sliceR(:));  [rR, cR] = ind2sub(size(sliceR), iR);
fprintf('\nslice peak (i2, i3): differenced (%.2f, %.2f), relative (%.2f, %.2f)\n', ...
        gridAxis(cD), gridAxis(rD), gridAxis(cR), gridAxis(rR));

% --- figure -----------------------------------------------------------------
fig = figure('Position', [50 50 1500 1070], 'Color', 'w');
axB1 = axes('Parent', fig, 'Position', [0.10 0.60 0.33 0.32]);
axB2 = axes('Parent', fig, 'Position', [0.58 0.60 0.33 0.32]);
axS1 = axes('Parent', fig, 'Position', [0.10 0.07 0.28 0.40]);
axS2 = axes('Parent', fig, 'Position', [0.58 0.07 0.28 0.40]);

localBars(axB1, classD, countD, meanD, TOP, ALS, C_ALS, C_OTHER, ...
          'Differenced interval triples');
localBars(axB2, classR, countR, meanR, TOP, ALS, C_ALS, C_OTHER, ...
          'Relative pitch quadruples');
localSlice(axS1, gridAxis, sliceD, classD, meanD, TOP, ALS, C_ALS, ...
           'Density through (+3, i_2, i_3): differenced');
localSlice(axS2, gridAxis, sliceR, classR, meanR, TOP, ALS, C_ALS, ...
           'Density through (+3, i_2, i_3): relative');
annotation(fig, 'textbox', [0.05 0.94 0.9 0.05], 'String', ...
    'Coltrane, Acknowledgement: motif density', ...
    'HorizontalAlignment', 'center', 'EdgeColor', 'none', 'FontSize', 15);

figDir = fullfile(thisDir, 'figures');
if SAVE_FIGURES
    if ~exist(figDir, 'dir'), mkdir(figDir); end
    print(fig, '-dpng', '-r140', fullfile(figDir, 'demo_jmm_2_2_motif.png'));
    fprintf('Saved figures/demo_jmm_2_2_motif.png\n');
end

% The demo leaves the toolbox as it found it: the defaults it set at the
% top are restored here.
mptDefaults(prevDefaults);


function [classes, counts, means] = localRank(intervals, density)
%LOCALRANK  Group cells by their interval triple and rank the classes.
%
%   The density at a cell already equals the class's recurrence --- every
%   member of a class sits at the same point, so each sees all c copies.
%   Summing over the members would square that, so the class mean is what
%   recovers the count.
    [classes, ~, idx] = unique(round(intervals).', 'rows');
    counts = accumarray(idx, 1);
    means = accumarray(idx, density(:)) ./ counts;
    [means, order] = sort(means, 'descend');
    classes = classes(order, :);
    counts = counts(order);
end

function s = localTriple(k)
%LOCALTRIPLE  An interval triple as "(3, -3, 5)".
    s = ['(' strjoin(arrayfun(@(v) sprintf('%d', v), k, ...
                              'UniformOutput', false), ', ') ')'];
end

function s = localContour(k)
%LOCALCONTOUR  An interval triple as the scale degrees it traces from 0.
    s = strjoin(arrayfun(@(v) sprintf('%d', v), cumsum([0 k]), ...
                         'UniformOutput', false), char(8594));
end

function localBars(ax, classes, counts, means, top, alsClass, alsColour, ...
                   otherColour, titleText)
%LOCALBARS  The top motifs as a horizontal bar chart, the named class
%   highlighted. A local function cannot see the script's variables, so the
%   count, the class to highlight, and the two colours are passed in.
    sel = top:-1:1;
    yy = 1:top;
    hold(ax, 'on');
    for i = 1:top
        if isequal(classes(sel(i), :), alsClass)
            col = alsColour;
        else
            col = otherColour;
        end
        barh(ax, yy(i), means(sel(i)), 0.56, 'FaceColor', col, ...
             'EdgeColor', 'none');
        text(ax, means(sel(i)), yy(i), ...
             sprintf('  %.1f (%d)', means(sel(i)), counts(sel(i))), ...
             'VerticalAlignment', 'middle', 'FontSize', 12, ...
             'Color', [0.27 0.27 0.27]);
    end
    set(ax, 'YTick', yy, 'YTickLabel', ...
        arrayfun(@(i) localContour(classes(sel(i), :)), 1:top, ...
                 'UniformOutput', false), ...
        'FontSize', 12, 'Box', 'off');
    xlim(ax, [0, max(means(1:top)) * 1.20]);
    ylim(ax, [0.3, top + 0.7]);
    xlabel(ax, 'density (and count)', 'FontSize', 15);
    title(ax, titleText, 'FontSize', 16);
    grid(ax, 'on');
end

function localSlice(ax, gridAxis, field, classes, means, top, alsClass, ...
                    alsColour, titleText)
%LOCALSLICE  A density slice through the leading motif's first interval,
%   with the top classes lying in the slice marked.
    pcolor(ax, gridAxis, gridAxis, field);
    shading(ax, 'interp');
    colormap(ax, jmm.colourMap('magma'));
    cb = colorbar(ax);
    ylabel(cb, 'density', 'FontSize', 13);
    hold(ax, 'on');
    for i = 1:top
        k = classes(i, :);
        if k(1) ~= alsClass(1), continue; end
        als = isequal(k, alsClass);
        if als
            col = alsColour; sz = 130; lw = 2.4;
            label = sprintf('  A Love Supreme  %.1f', means(i));
        else
            col = [1 1 1]; sz = 60; lw = 1.4;
            label = sprintf('  %.1f', means(i));
        end
        scatter(ax, k(2), k(3), sz, 'MarkerEdgeColor', col, 'LineWidth', lw);
        text(ax, k(2), k(3) + 0.35, label, 'Color', col, 'FontSize', 12);
    end
    axis(ax, 'equal');
    xlim(ax, [gridAxis(1), gridAxis(end)]);
    ylim(ax, [gridAxis(1), gridAxis(end)]);
    set(ax, 'XTick', -6:2:6, 'YTick', -6:2:6, 'FontSize', 13, 'Box', 'off');
    xlabel(ax, 'second interval  i_2  (semitones)', 'FontSize', 15);
    ylabel(ax, 'third interval  i_3  (semitones)', 'FontSize', 15);
    title(ax, titleText, 'FontSize', 16);
end
