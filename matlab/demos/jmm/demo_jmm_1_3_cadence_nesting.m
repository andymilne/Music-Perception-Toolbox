%% demo_jmm_1_3_cadence_nesting.m
% Analysis 1.3 (JMM article, Section 4.1.3; its minor-mode rows Online
% Supplement, Section 6): cadence localization with nested multisets.
%
% A demo of the Music Perception Toolbox reproducing the analysis from the
% JMM article ("Cadence localization using nested multisets"), lightly
% edited from the article's own
% scripts. Data come from the jmm package (BWV 347 read from the bundled
% MusicXML); the figures stay on screen unless SAVE_FIGURES is set.
%
% What the analysis asks. Where in the chorale do cadences, and
% cadence-like progressions, occur — of which type, and in any key? A
% prototypical cadential progression (a short ordered run of chords) is
% slid across the chorale, and its one-sided similarity to the local
% harmony is read at every candidate resolution moment.
%
% How it is computed. Query and context alike are single events whose
% pitch attribute is a nested multiset two levels deep: the inner level is
% each chord's pitch multiset ([exch] = 1, at inner tuple size r = 1, 2, or
% 3, the parameter this analysis varies), the outer level the chords in
% order ([exch] = 0, r = the progression length). The comparison is taken
% relative at the outer level alone ([rel] = (0, 1)), removing one common
% transposition of the whole progression while leaving each chord's own
% pitch classes absolute, and periodic at the octave (sigma = 0.15
% semitones, P = 12). The chorale is reduced beat by beat to weighted
% pitch aggregates — its two eighth-note events weighted by metrical
% position (1 on the beat, 0.5 off it, times 1.5 under a fermata) and by
% the fraction of the eighth each note sounds, merged and normalized by
% the 1.5 a beat carries — and the aligned span of L consecutive beats,
% the resolution on the last, is bound into one nested super-event
% (bindEvents) and compared with the query under the one-sided
% similarity (windowedSimilarity(..., 'normalize', 'oneSidedDenom')), so a
% peak of 1 is one isolated exact match. An optional inversion flag — a
% second, simplex-coded attribute at +/-0.5 with sigma_flag = 0.1 — marks
% whether a chosen chord is a root-position triad (the dyad skeleton's
% resolution) or a second-inversion triad (the six-four's antepenult); the
% context's flag value is derived from the pitch content of the sonority
% at that beat, no harmonic labels being consulted.
%
% Eight queries are swept at inner r = 1, 2, 3: the tritone-to-major-third
% dyad skeleton B–F -> C–E, plain and with the root-position flag on its
% resolution; the maximal prototypes ii7–V7–I and its minor counterpart;
% and the major and minor cadential six-fours, each plain and flagged.
% Three figures are written from the one computation: the article's
% five major-mode rows, the three minor-mode rows of the Online
% Supplement, and all eight together. The dyad rows are blank at r = 3,
% where a two-pitch chord has no inner triple.
%
% The encodings live in the jmm package: the beat aggregates
% (jmm.bwvWindowState), the nested context and query builders
% (jmm.boundContext, jmm.query, jmm.windowStarts, jmm.asCompared,
% jmm.dyadQuery, jmm.prototypeQuery), and the pitch-derived flags
% (jmm.isRootPosition, jmm.isSixFour). The two sweeps are local
% functions at the foot of this file. Toolbox: gridAttrTable (twice, the
% second regridding the first), preMaetFromAttrTable, bindEvents (with
% per-attribute orders, so the window's time and flag stay flat),
% windowedSimilarity, flatSpecs, selectPreMaet. Runtime: a few seconds
% (eight queries at three inner tuple sizes, each sweep one call).

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

prevDefaults = mptDefaults('showHints', false);

% Similarity normalization for all sweeps: 'oneSidedDenom' (query
% self-overlap alone) or 'cosine' (symmetric; penalizes context content
% the query does not match).
NORMALIZE = 'oneSidedDenom';

CAD_TIMES = [7, 15, 23, 31, 45, 65];
CAD_NAMES = {'C1', 'C2', 'C1''', 'C2''', 'C3', 'C4'};
COL = [0.122 0.306 0.475];
RS = [1 2 3];

% ---------------------------------------------------------------------------
% The six three-chord prototype queries (MIDI semitones, as in the article's
% tables): each chord enters as its distinct pitch classes, with no doubling.
% Doubling is material at inner r >= 2 (two unison-weighted entries of a
% pitch class are not one double-weighted entry: only the former contributes
% repeated-PC tuples), so entering doubled voicings would change the sweep
% values substantially. The flagged six-fours share their unflagged twins'
% pitch content — the inversion flag is the only difference.
% ---------------------------------------------------------------------------
QUERY_NAMES = {'ii7-V7-I', 'iio7-V7-i', 'I-V-I', 'Ic-V-I', 'i-V-i', 'ic-V-i'};
QUERIES = struct('chords', ...
    {{[62 65 69 72], [55 59 62 65], [60 64 67]}, ...
     {[62 65 68 72], [55 59 62 65], [60 63 67]}, ...
     {[60 64 67], [62 67 71], [60 64 67]}, ...
     {[60 64 67], [62 67 71], [60 64 67]}, ...
     {[60 63 67], [62 67 71], [60 63 67]}, ...
     {[60 63 67], [62 67 71], [60 63 67]}}, ...
    'flagged', {false, false, false, true, false, true});

STEP = 1.0;                                     % sweep step, QN (one beat)
MUS = 2.0:STEP:67.0;                            % candidate resolution moments
WIN_BAR = jmm.b2bar(MUS);

% Rows of the figures: label, kind ('dyad' or 'proto'), and its
% specification (the dyad's flag, or the prototype query's index).
ROW_LABELS = {'d5/A4–M3/m6', 'd5/A4–*M3', 'ii^7–V^7–I', 'ii^{\oslash7}–V^7–i', ...
              'I–V–I', '*I_c–V–I', 'i–V–i', '*i_c–V–i'};
ROW_KINDS = {'dyad', 'dyad', 'proto', 'proto', 'proto', 'proto', 'proto', 'proto'};
ROW_SPECS = {false, true, 1, 2, 3, 4, 5, 6};
nRows = numel(ROW_LABELS);

VARIANT_FILES = {'demo_jmm_1_3_cadence_sweeps', ...            % the article's figure
                 'demo_jmm_1_3_cadence_sweeps_minor', ...      % Online Supplement
                 'demo_jmm_1_3_cadence_sweeps_all8'};
VARIANT_ROWS = {[1 2 3 5 6], [4 7 8], 1:nRows};

% ---------------------------------------------------------------------------
% Compute: data{ri, r} is {x, y} (or [] for the dyad at r = 3)
% ---------------------------------------------------------------------------
protoProf = cell(1, numel(RS));
for r = RS
    if r == RS(1)
        for qi = 1:numel(QUERIES)
            qPm = jmm.prototypeQuery(QUERIES(qi).chords, ...
                QUERIES(qi).flagged, r);
            % The query as the cosine receives it: the placement axis has
            % done its work and is not part of the comparison.
            showPreMaet(jmm.asCompared(qPm), ...
                'title', sprintf('  query: %s (rInner = %d)', ...
                QUERY_NAMES{qi}, r));
            fprintf('\n');
        end
    end
    protoProf{r} = localPrototypeSweep(r, QUERIES, MUS, NORMALIZE);
end
data = cell(nRows, numel(RS));
for ri = 1:nRows
    for r = RS
        if strcmp(ROW_KINDS{ri}, 'dyad')
            if r == 3
                data{ri, r} = [];
            else
                [x, y] = localDyadSweep(r, ROW_SPECS{ri}, MUS, NORMALIZE);
                data{ri, r} = {x, y};
            end
        else
            data{ri, r} = {WIN_BAR, protoProf{r}{ROW_SPECS{ri}}};
        end
    end
end

% ---------------------------------------------------------------------------
% Report
% ---------------------------------------------------------------------------
fprintf('panel maxima (one-sided similarity; 1 = one isolated exact match),\n');
fprintf('at inner r = 1, 2, 3:\n');
for ri = 1:nRows
    ms = cell(1, numel(RS));
    for r = RS
        d = data{ri, r};
        if isempty(d), ms{r} = '   --  ';
        else,          ms{r} = sprintf('%6.3f', max(d{2}(~isnan(d{2})))); end
    end
    fprintf('  %-48s  %s\n', ROW_LABELS{ri}, strjoin(ms, '  '));
end
% Where each query peaks at inner r = 2, the discriminating size.
fprintf('\nlocation of the r = 2 maximum (played-through bar):\n');
for ri = 1:nRows
    d = data{ri, 2};
    x = d{1}; y = d{2};
    y(isnan(y)) = -Inf;
    [~, k] = max(y);
    fprintf('  %-48s  bar %5.2f\n', ROW_LABELS{ri}, x(k));
end

% ---------------------------------------------------------------------------
% Figures
% ---------------------------------------------------------------------------
figDir = fullfile(thisDir, 'figures');
if SAVE_FIGURES && ~exist(figDir, 'dir'), mkdir(figDir); end
for v = 1:numel(VARIANT_FILES)
    rowIds = VARIANT_ROWS{v};
    n = numel(rowIds);
    fig = figure('Position', [100 100 1150 round(100 * (1.275 * n + 0.4))], ...
                 'Color', 'w');
    for vi = 1:n
        ri = rowIds(vi);
        for ci = 1:numel(RS)
            r = RS(ci);
            ax = subplot(n, 3, (vi - 1) * 3 + ci);
            d = data{ri, r};
            if isempty(d)                            % dyad at r = 3: blank cell
                axis(ax, 'off');
                if vi == 1
                    title(ax, sprintf('Inner r = %d', r), 'FontSize', 15);
                end
                continue;
            end
            x = d{1}; y = d{2};
            m = max(y(~isnan(y)));
            if m > 1e-3, top = m * 1.10; else, top = 1.0; end   % per-panel scale
            hold(ax, 'on');
            for bar = 1:18
                plot(ax, [bar bar], [0 top], 'Color', [0.86 0.86 0.86], 'LineWidth', 0.6);
            end
            for c = 1:numel(CAD_TIMES)
                xc = jmm.b2bar(CAD_TIMES(c));
                plot(ax, [xc xc], [0 top], '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.0);
                if vi == 1
                    text(ax, xc, top * 1.02, CAD_NAMES{c}, 'FontSize', 11, ...
                         'HorizontalAlignment', 'center', ...
                         'VerticalAlignment', 'bottom', 'Color', [0.3 0.3 0.3]);
                end
            end
            yf = y; yf(isnan(yf)) = 0;
            fill(ax, [x, fliplr(x)], [yf, zeros(size(yf))], COL, ...
                 'FaceAlpha', 0.10, 'EdgeColor', 'none');
            plot(ax, x, y, 'LineWidth', 1.2, 'Color', COL);
            set(ax, 'XTick', [1 5 9 13 17], 'FontSize', 12, 'Box', 'off');
            xlim(ax, [1, 17.6]);
            ylim(ax, [0, top]);
            grid(ax, 'on');
            set(ax, 'XGrid', 'off');
            if ci == 1
                ylabel(ax, ROW_LABELS{ri}, 'FontSize', 13);
            end
            if vi == 1
                title(ax, sprintf('Inner r = %d', r), 'FontSize', 15);
            end
            if vi == n
                xlabel(ax, 'bar', 'FontSize', 13);
            else
                set(ax, 'XTickLabel', {});
            end
        end
    end
    if SAVE_FIGURES
        print(fig, '-dpdf', fullfile(figDir, [VARIANT_FILES{v}, '.pdf']));
        print(fig, '-dpng', '-r150', fullfile(figDir, [VARIANT_FILES{v}, '.png']));
        fprintf('Saved figures/%s.pdf\n', VARIANT_FILES{v});
    end
end

% The demo leaves the toolbox as it found it: the defaults it set at the
% top are restored here.
mptDefaults(prevDefaults);


%% Local functions

function profiles = localPrototypeSweep(rInner, queries, mus, normalize)
    %localPrototypeSweep One-sided similarity profiles of the three-chord
    %queries at one inner r: a 1 x Q cell, entry q the (1 x numel(mus))
    %profile of query q. queries is a 1 x Q struct array with fields
    %.chords (1 x 3 cell of MIDI pitch vectors) and .flagged (logical);
    %mus the candidate resolution moments (QN); normalize 'oneSidedDenom'
    %or 'cosine'. Positions whose windows lack events score 0. The
    %chorale's aligned span at resolution moment mu is the three beat
    %aggregates [mu-2, mu-1), [mu-1, mu), [mu, mu+1).
    %
    % One bindEvents call nests every window of three beats across the
    % whole chorale, carrying the window's own start time and the
    % inversion flag flat alongside the nested pitch. The sweep is then
    % one call per query: a rectangle of one beat admits exactly one
    % window at each centre. The flag is pitch-derived --- a predicate on
    % the sonority at the antepenult beat, which is the window's first ---
    % and no harmonic labels are consulted.
    ctxPlain = jmm.boundContext(3, rInner);
    ctxFlag = jmm.boundContext(3, rInner, 'sixFour');
    [idxs, centres] = jmm.windowStarts(mus, 2.0);
    profiles = cell(1, numel(queries));
    for q = 1:numel(queries)
        qd = jmm.prototypeQuery(queries(q).chords, queries(q).flagged, rInner);
        if queries(q).flagged, ctx = ctxFlag; else, ctx = ctxPlain; end
        % A rectangle of full support one beat, centred on each window's
        % own start time, admits exactly that window and no other --- its
        % neighbours sit exactly a beat away. The time axis (attribute 2)
        % is dropped from the comparison, having done its work in placing
        % the window.
        prof = zeros(1, numel(mus));
        raw = windowedSimilarity(ctx, qd, centres(:).', ...
            'contextWindow', {1.0, 1.0}, 'windowAttr', 2, ...
            'dropWindowAttr', true, 'normalize', normalize, 'verbose', false);
        prof(idxs) = raw(:).';
        profiles{q} = prof;
    end
end

function [x, so] = localDyadSweep(rInner, useFlag, mus, normalize)
    %localDyadSweep One-sided similarity of the dyad-skeleton query
    %(jmm.dyadQuery) against the chorale, swept over candidate resolution
    %moments mus (every beat, QN); useFlag adds the pitch-derived
    %root-position flag. Returns the played-through bar coordinate
    %x = jmm.b2bar(mus) and the profile so (NaN where the sweep does not
    %reach; 0 where a window lacks events).
    %
    % One bindEvents call nests every pair of adjacent beats --- the
    % approach beat [mu-1, mu) and the resolution beat [mu, mu+1) ---
    % carrying the window's own start time and the inversion flag flat
    % alongside the nested pitch. The optional inversion attribute is
    % pitch-derived: a predicate on the sonority sounding at mu, which is
    % the window's second beat, and no harmonic labels are consulted.
    S = jmm.bwvWindowState();
    if useFlag, qFlag = S.rootYes; else, qFlag = []; end
    qd = jmm.dyadQuery(qFlag, rInner);
    if useFlag, flagName = 'rootPositionNext'; else, flagName = ''; end
    ctx = jmm.boundContext(2, rInner, flagName);
    [idxs, centres] = jmm.windowStarts(mus, 1.0);
    so = nan(1, numel(mus));
    raw = windowedSimilarity(ctx, qd, centres(:).', ...
        'contextWindow', {1.0, 1.0}, 'windowAttr', 2, ...
        'dropWindowAttr', true, 'normalize', normalize, 'verbose', false);
    so(idxs) = raw(:).';
    x = jmm.b2bar(mus);
end
