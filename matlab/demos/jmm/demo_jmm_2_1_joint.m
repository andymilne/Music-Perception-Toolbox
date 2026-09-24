%% demo_jmm_2_1_joint.m
% Analysis 2.1 (JMM article, Section 4.2.1): the motif of Acknowledgement
% as a joint pitch-and-rhythm object.
%
% A demo of the Music Perception Toolbox reproducing the analysis from
% the JMM article; lightly edited from the article's own script. Data
% come from the jmm package (Acknowledgement read from a MIDI
% transcription you supply); the figures stay on screen unless
% SAVE_FIGURES is set.
%
% Analysis 2.1: the motif of Coltrane's Acknowledgement as a joint
% object --- the interval pattern together with the rhythm it is set in.
%
% Analysis 2.2 recovers the four-note cell from pitch alone. Adding onset
% time as a second attribute asks for agreement in pitch and rhythm at
% once: the two attributes are joined by the tensor product, so the joint
% density over interval and inter-onset-interval super-events ranks a
% motif by both. Both routes of Analysis 2.2 carry over, and the
% attributes are handled alike within each:
%
%   differenced route  pitch to melodic intervals and onset time to
%                      inter-onset intervals, three consecutive of each
%                      bound into ordered super-events in step;
%   relative route     four consecutive pitches and four consecutive
%                      onset times bound into ordered super-events, each
%                      taken relative.
%
% The attributes need not be handled alike --- pitch may be differenced
% while onset time is taken relative, or the reverse, each attribute
% carrying its own choice --- but taking them in step keeps the two
% routes comparable. The motif is set in one rhythm almost always:
% eighth, quarter, eighth (0.5, 1.0, 0.5 QN), in 35 of its 36 statements,
% the remaining one holding the first note long before two quick
% sixteenths (2.5, 0.25, 0.25 QN). The joint motif is therefore sharply
% defined and leads both routes.
%
% Because the attributes are tensored, either may be marginalized --- by
% omitting it from the density, exactly so where its total mass per event
% is constant, as here --- or conditioned on, by fixing its coordinates
% in the point at which the density is evaluated. The lower panels do
% both for rhythm: the inter-onset-interval density with pitch
% marginalized away, and the same density conditioned on the motif's
% intervals.
%
% A relative density is read in translation-reduced coordinates: an
% r-tuple minus its first value, so a cell's coordinates are its
% cumulative intervals. Both the cells and the slice grids are written
% that way for the relative route below.
%
% Pre-MAET structure:
%
%     attribute    order  sigma                rel  per
%     ----------   -----  -------------------  ---  ---
%     dp           3      sqrt(2) * 0.15 st    no   no   differenced route
%     dt           3      sqrt(2) * 0.125 QN   no   no   differenced route
%     pitch        4      0.15 st              yes  no   relative route
%     onset        4      0.125 QN             yes  no   relative route
%
%     Ordered (exch = 0) throughout; the two attributes are tensored.
%     Estimator: the density read at each cell.
%
% Data: jmm.acknowledgement (the solo, from your own MIDI transcription at
% data/AwakeningSolo.mid). Toolbox: preMaetFromAttrTable, differenceEvents,
% bindEvents, selectPreMaet, buildMaet, evalMaet, showPreMaet. Runtime:
% under a minute.

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

SIGMA_PITCH = 0.15;            % semitones (15 cents): the per-pitch uncertainty
SIGMA_TIME  = 0.125;           % QN (a thirty-second note): the per-onset one
R_DIFF      = 3;               % bound interval triples (a four-note cell)
R_REL       = 4;               % bound pitch quadruples (the same cell)
TOP         = 6;               % motifs shown in the ranking
ALS_IV      = [3 -3 5];        % +m3, -m3, +P4: the "A Love Supreme" cell
ALS_RHYTHM  = [0.5 1.0 0.5];   % its rhythm: eighth, quarter, eighth
FIRST_IOI   = 0.5;             % the motif's first IOI: the slice's fixed value

C_ALS   = [0.761 0.314 0.031];   % the recurring "A Love Supreme" motif
C_OTHER = [0.122 0.306 0.722];   % the surrounding recurring motifs

% --- the melody as a pre-MAET, one event per note ---------------------------
notes = jmm.acknowledgement();
fprintf('%d notes; pitch range %.0f-%.0f (MIDI); span %.1f QN\n', ...
        height(notes), min(notes.pitch), max(notes.pitch), ...
        max(notes.onsetBeats));
melody = preMaetFromAttrTable(notes, 'attributes', { ...
    struct('column', 'pitch', 'name', 'pitch', 'sigma', SIGMA_PITCH), ...
    struct('column', 'onset', 'name', 'onset', 'sigma', SIGMA_TIME)}, ...
    'time', 'beats', 'chords', 'separate', 'weights', 'ones');

% --- differenced route: intervals and IOIs bound in step --------------------
% One bindEvents call takes both attributes at order 3, so the two stay in
% step. differenceEvents widens each kernel by sqrt(2) itself --- a
% difference of two values of width sigma has width sqrt(2) sigma --- so
% the bound pre-MAET already carries both interval widths.
diffRoute = bindEvents(differenceEvents(melody, [1 1]), [R_DIFF R_DIFF], ...
                       'step', 1);
showPreMaet(diffRoute, 'maxEvents', 2, 'decimals', 3);
[pd, ~, ~] = unpackPreMaet(diffRoute);
ivD = pd{1};  ioiD = pd{2};
jointD = buildMaet(diffRoute, 'verbose', false);
densD = evalMaet(jointD, {ivD, ioiD}, 'verbose', false);
[classD, rhythmD, countD, meanD] = localRank(ivD, ioiD, densD);

% --- relative route: pitches and onsets bound, each taken relative ----------
relRoute = bindEvents(melody, [R_REL R_REL], 'step', 1, 'relOuter', true);
showPreMaet(relRoute, 'maxEvents', 2, 'decimals', 3);
[pr, ~, ~] = unpackPreMaet(relRoute);
ivR = diff(pr{1}, 1, 1);  ioiR = diff(pr{2}, 1, 1);
jointR = buildMaet(relRoute, 'verbose', false);
densR = evalMaet(jointR, {cumsum(ivR, 1), cumsum(ioiR, 1)}, 'verbose', false);
[classR, rhythmR, countR, meanR] = localRank(ivR, ioiR, densR);

% --- report -----------------------------------------------------------------
fprintf('\ndifferenced: %d cells, %d distinct (interval, rhythm) classes\n', ...
        size(ivD, 2), size(classD, 1));
fprintf('relative:    %d cells, %d distinct classes\n\n', ...
        size(pr{1}, 2), size(classR, 1));
fprintf('%4s  %16s  %14s  %5s  %11s  %9s\n', 'rank', 'interval class', ...
        'rhythm (QN)', 'count', 'differenced', 'relative');
for i = 1:TOP
    j = find(ismember(classR, classD(i, :), 'rows') & ...
             ismember(rhythmR, rhythmD(i, :), 'rows'), 1);
    if isequal(classD(i, :), ALS_IV) && isequal(rhythmD(i, :), ALS_RHYTHM)
        mark = '  <- A Love Supreme motif';
    else
        mark = '';
    end
    fprintf('%4d  %16s  %14s  %5d  %11.1f  %9.1f%s\n', i, ...
            localTriple(classD(i, :)), localRhythm(rhythmD(i, :)), ...
            countD(i), meanD(i), meanR(j), mark);
end
isAls = ismember(classD, ALS_IV, 'rows');
fprintf('\nthe cell %s is stated %d times, in %d rhythms: %s\n', ...
        localTriple(ALS_IV), sum(countD(isAls)), sum(isAls), ...
        strjoin(arrayfun(@(k) sprintf('(%s) x %d', ...
            localRhythm(rhythmD(k, :)), countD(k)), find(isAls).', ...
            'UniformOutput', false), ', '));

% --- rhythm with pitch marginalized away, and conditioned on the motif -----
% Marginalizing is dropping the pitch attribute from the density (exact
% here, the mass per event being constant); conditioning is fixing its
% coordinates in the evaluation point.
timeD = buildMaet(selectPreMaet(diffRoute, 'attributes', {'onset'}), ...
                  'verbose', false);
timeR = buildMaet(selectPreMaet(relRoute, 'attributes', {'onset'}), ...
                  'verbose', false);
gridStep = SIGMA_TIME * sqrt(2) / 3;
gridAxis = 0:gridStep:2.5;
[T2, T3] = meshgrid(gridAxis, gridAxis);
gridIoi = [repmat(FIRST_IOI, 1, numel(T2)); T2(:).'; T3(:).'];
gridIv = repmat(ALS_IV(:), 1, numel(T2));
margD = reshape(evalMaet(timeD, gridIoi, 'verbose', false), size(T2));
condD = reshape(evalMaet(jointD, {gridIv, gridIoi}, 'verbose', false), size(T2));
margR = reshape(evalMaet(timeR, cumsum(gridIoi, 1), 'verbose', false), size(T2));
condR = reshape(evalMaet(jointR, {cumsum(gridIv, 1), cumsum(gridIoi, 1)}, ...
                         'verbose', false), size(T2));
sliceTags = {'marginalized, differenced', 'marginalized, relative', ...
        'conditioned, differenced', 'conditioned, relative'};
slices = {margD, margR, condD, condR};
for k = 1:4
    [~, idx] = max(slices{k}(:));
    [r, c] = ind2sub(size(T2), idx);
    fprintf('%26s: peak at (second, third) IOI = (%.2f, %.2f) QN\n', ...
            sliceTags{k}, gridAxis(c), gridAxis(r));
end

% --- figure -----------------------------------------------------------------
fig = figure('Position', [50 50 1400 1500], 'Color', 'w');
axB1 = axes('Parent', fig, 'Position', [0.13 0.745 0.30 0.185]);
axB2 = axes('Parent', fig, 'Position', [0.62 0.745 0.30 0.185]);
axM1 = axes('Parent', fig, 'Position', [0.13 0.405 0.24 0.26]);
axM2 = axes('Parent', fig, 'Position', [0.62 0.405 0.24 0.26]);
axC1 = axes('Parent', fig, 'Position', [0.13 0.055 0.24 0.26]);
axC2 = axes('Parent', fig, 'Position', [0.62 0.055 0.24 0.26]);

localBars(axB1, classD, rhythmD, countD, meanD, TOP, ALS_IV, ALS_RHYTHM, ...
          C_ALS, C_OTHER, 'Differenced interval triples');
localBars(axB2, classR, rhythmR, countR, meanR, TOP, ALS_IV, ALS_RHYTHM, ...
          C_ALS, C_OTHER, 'Relative pitch quadruples');
localSlice(axM1, gridAxis, margD, ALS_RHYTHM, C_ALS, ...
           'Pitch marginalized: differenced');
localSlice(axM2, gridAxis, margR, ALS_RHYTHM, C_ALS, ...
           'Pitch marginalized: relative');
localSlice(axC1, gridAxis, condD, ALS_RHYTHM, C_ALS, ...
           'Conditioned on (+3, -3, +5): differenced');
localSlice(axC2, gridAxis, condR, ALS_RHYTHM, C_ALS, ...
           'Conditioned on (+3, -3, +5): relative');
annotation(fig, 'textbox', [0.05 0.955 0.9 0.04], 'String', ...
    ['Coltrane, Acknowledgement: joint pitch-and-rhythm motifs ' ...
     '(first IOI fixed at an eighth note)'], ...
    'HorizontalAlignment', 'center', 'EdgeColor', 'none', 'FontSize', 15);

figDir = fullfile(thisDir, 'figures');
if SAVE_FIGURES
    if ~exist(figDir, 'dir'), mkdir(figDir); end
    print(fig, '-dpng', '-r140', fullfile(figDir, 'demo_jmm_2_1_joint.png'));
    fprintf('Saved figures/demo_jmm_2_1_joint.png\n');
end

% The demo leaves the toolbox as it found it: the defaults it set at the
% top are restored here.
mptDefaults(prevDefaults);


function [classes, rhythms, counts, means] = localRank(intervals, iois, density)
%LOCALRANK  Group cells by interval triple and rhythm, and rank the classes.
%
%   The density at a cell already equals the class's recurrence --- every
%   member of a class sits at the same point, so each sees all c copies.
%   Summing over the members would square that, so the class mean is what
%   recovers the count. Inter-onset intervals are grouped to the sixteenth
%   note, the passage's shortest written value.
    labels = [round(intervals).' round(iois.' * 4) / 4];
    [labels, ~, idx] = unique(labels, 'rows');
    counts = accumarray(idx, 1);
    means = accumarray(idx, density(:)) ./ counts;
    [means, order] = sort(means, 'descend');
    labels = labels(order, :);
    counts = counts(order);
    classes = labels(:, 1:3);
    rhythms = labels(:, 4:6);
end

function s = localTriple(k)
%LOCALTRIPLE  An interval triple as "(3, -3, 5)".
    s = ['(' strjoin(arrayfun(@(v) sprintf('%d', v), k, ...
                              'UniformOutput', false), ', ') ')'];
end

function s = localRhythm(k)
%LOCALRHYTHM  A cell's inter-onset intervals in QN, compactly.
    s = strjoin(arrayfun(@(v) sprintf('%g', v), k, ...
                         'UniformOutput', false), char(183));
end

function s = localContour(k)
%LOCALCONTOUR  An interval triple as the scale degrees it traces from 0.
    s = strjoin(arrayfun(@(v) sprintf('%d', v), cumsum([0 k]), ...
                         'UniformOutput', false), char(8594));
end

function localBars(ax, classes, rhythms, counts, means, top, alsClass, ...
                   alsRhythm, alsColour, otherColour, titleText)
%LOCALBARS  The top joint motifs as a horizontal bar chart, the named motif
%   highlighted. A local function cannot see the script's variables, so the
%   count, the motif to highlight, and the two colours are passed in.
    sel = top:-1:1;
    yy = 1:top;
    hold(ax, 'on');
    labels = cell(1, top);
    for i = 1:top
        k = sel(i);
        if isequal(classes(k, :), alsClass) && isequal(rhythms(k, :), alsRhythm)
            col = alsColour;
        else
            col = otherColour;
        end
        barh(ax, yy(i), means(k), 0.72, 'FaceColor', col, 'EdgeColor', 'none');
        text(ax, means(k), yy(i), sprintf('  %.1f (%d)', means(k), counts(k)), ...
             'VerticalAlignment', 'middle', 'FontSize', 12, ...
             'Color', [0.27 0.27 0.27]);
        labels{i} = sprintf('%s\\newline%s', localContour(classes(k, :)), ...
                            localRhythm(rhythms(k, :)));
    end
    set(ax, 'YTick', yy, 'YTickLabel', labels, 'FontSize', 11, 'Box', 'off');
    xlim(ax, [0, max(means(1:top)) * 1.20]);
    ylim(ax, [0.3, top + 0.7]);
    xlabel(ax, 'density (and count)', 'FontSize', 15);
    title(ax, titleText, 'FontSize', 16);
    grid(ax, 'on');
end

function localSlice(ax, gridAxis, field, alsRhythm, alsColour, titleText)
%LOCALSLICE  An inter-onset-interval density slice, first IOI fixed.
    pcolor(ax, gridAxis, gridAxis, field);
    shading(ax, 'interp');
    colormap(ax, jmm.colourMap('magma'));
    cb = colorbar(ax);
    ylabel(cb, 'density', 'FontSize', 13);
    hold(ax, 'on');
    scatter(ax, alsRhythm(2), alsRhythm(3), 150, ...
            'MarkerEdgeColor', alsColour, 'LineWidth', 2.4);
    axis(ax, 'equal');
    xlim(ax, [gridAxis(1), gridAxis(end)]);
    ylim(ax, [gridAxis(1), gridAxis(end)]);
    set(ax, 'FontSize', 13, 'Box', 'off');
    xlabel(ax, 'second IOI (QN)', 'FontSize', 15);
    ylabel(ax, 'third IOI (QN)', 'FontSize', 15);
    title(ax, titleText, 'FontSize', 16);
end
