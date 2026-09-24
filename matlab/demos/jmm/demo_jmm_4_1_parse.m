%% demo_jmm_4_1_parse.m
% Analysis 4.1 (Online Supplement, Section 11): a supplied parse carried as
% a nested attribute.
%
% A demo of the Music Perception Toolbox reproducing the analysis from the
% JMM article's Online Supplement. Data come from jmm.derivations (the
% rule-labelled derivations of Ren, Rammos, and Rohrmeier 2024, which you
% supply); the figures stay on screen unless SAVE_FIGURES is set.
%
% Analysis 4.1: an expert harmonic analysis, supplied as input, carried
% into the framework, and operated on by it.
%
% The material is a rule-labelled derivation: each surface chord of a tune
% is assigned a root-to-leaf path of rule labels, the grammar's account of
% how that chord is reached. The framework does not produce the derivation
% --- the rules that license it do that, outside --- and its part begins
% once it is given one.
%
% The encoding. Each chord is one event and its path one nested attribute:
% the inner level a rule label's simplex coordinates, read whole and in
% order, so that two labels are either the same or equally different; the
% outer level the positions of the path, in order, so that position carries
% depth. Paths differ in length, so the positions of a chord are bound by
% 'groupBy' rather than by a fixed window, and the outer tuple size rOuter
% then says how much of a path a comparison takes at once. At rOuter = 2 a
% tuple is an ordered pair of positions, matched wherever it occurs in
% order, contiguous or not --- which is depth-shift and elaboration
% tolerance without a level coordinate.
%
% Four things the framework then does with it:
%
%   retrieval     a configuration is the query, and the one-sided
%                 similarity counts its occurrences --- exactly, since a
%                 label either matches or does not at this kernel width;
%   partial match a wider kernel extends retrieval to labels that match
%                 only approximately, and on a regular simplex every
%                 substitution is equally wrong;
%   reduction     per-position weights g^(level - 3) grade a path by
%                 degree, and the graded tune is compared with the hard
%                 reduction in which every path is truncated at level 3;
%   depth         one further inner coordinate, the level scaled by
%                 sLevel, puts a displacement of one level at kernel
%                 distance sLevel, so that depth enters the comparison
%                 itself rather than being ignored.
%
% Pre-MAET structure:
%
%     attribute   order       sigma          rel   per
%     ---------   ----------  -------------  ----  ----
%     label       (V-1, 2)    0.1 (or 0.3)   no    no    the nested path
%     label       (V, 2)      0.1            no    no    with the level
%
%     V is the number of rule labels in the alphabet, so a label is a
%     point of a regular (V-1)-simplex of unit edge. Ordered (exch = 0) at
%     both levels. Estimator: one-sided similarity for retrieval, cosine
%     for the reduction.
%
% Data: jmm.derivations (from your own copy of ParseTrees.json at
% data/ParseTrees.json). Toolbox: simplexVertices, packPreMaet, flatSpecs,
% bindAttributes, bindEvents ('groupBy'), selectPreMaet, buildMaet,
% simMaet, showPreMaet. Runtime: a few seconds.

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

% --- parameters --------------------------------------------------------------
TUNE = '(Valid)Solar';            % Miles Davis, as the corpus names it
CONTROL = '(Valid)Interplay';     % holds no instance of the query
SIGMA_LABEL = 0.1;                % a label either matches or does not
SIGMA_WIDE = 0.3;                 % wide enough for a substitution to count
R_OUTER = 2;                      % an ordered pair of path positions
QUERY = {'V_I', 'Descending5th'};         % a dominant prepared by fifths
QUERY_LEVELS = [3 4];                     % where it first occurs in Solar
SUBSTITUTES = {'IV_V', 'Backdoor_I'};     % neither occurs in Solar
REDUCTION_LEVEL = 3;              % paths are graded, and cut, beyond this
G_VALUES = [1.0 0.5 0.2 0.0];     % the grading's decay per level
S_RATIOS = [0.0 0.2 0.5 1.0 3.0]; % sLevel / sigma

C_TUNE = [0.122 0.306 0.722];
C_QUERY = [0.761 0.314 0.031];

% --- the derivations, and the alphabet of rule labels ------------------------
parseRows = jmm.derivations({TUNE, CONTROL});
alphabet = unique([parseRows.label; QUERY(:); SUBSTITUTES(:)]);
vertices = simplexVertices(numel(alphabet));    % one row per label
E = struct('alphabet', {alphabet}, 'vertices', vertices, ...
           'dim', numel(alphabet) - 1, ...      % coordinates of a unit-edge simplex
           'reductionLevel', REDUCTION_LEVEL);
fprintf('%d tunes, %d path positions, %d rule labels: %s\n', ...
        numel(unique(parseRows.tune)), height(parseRows), ...
        numel(alphabet), strjoin(alphabet(:).', ', '));
fprintf('each label is a vertex of a unit-edge %d-simplex\n', E.dim);

tuneRows = parseRows(strcmp(parseRows.tune, TUNE), :);
controlRows = parseRows(strcmp(parseRows.tune, CONTROL), :);

% The column-preparation options are a struct, so that each analysis below
% changes one field and leaves the rest at their defaults. No toolbox
% function is called in that preparation: the encoding itself stays in the
% open here.
base = struct('levelScale', [], 'decay', [], 'truncate', []);

% --- the encoding ------------------------------------------------------------
% Four calls turn a table of path positions into the pre-MAET, and every
% analysis below makes the same four: pack the columns with their kernel
% parameters; bind the coordinates of a label into one attribute read whole
% and in order; bind the positions of one chord, which the run-length form
% reads from the chord index rather than from a window width; and keep the
% bound attribute, the chord index having done its work.
[values, weights, names] = localPathColumns( ...
    localQueryTable(QUERY, [1 2]), E, base);
pm = packPreMaet(values, weights, ...
                 flatSpecs(values, 'sigma', SIGMA_LABEL, 'isPer', false, ...
                           'period', 0, 'name', names));
pm = bindAttributes(pm, names(1:end-1), 'name', 'label', ...
                    'r', numel(names) - 1, 'exch', false);
pm = bindEvents(pm, [], 'groupBy', 'chord', 'rOuter', R_OUTER);
queryPm = selectPreMaet(pm, 'attributes', {'label'});
showPreMaet(queryPm, 'maxEvents', 1, 'decimals', 2);

contextTables = {tuneRows, controlRows};
contextPm = cell(1, 2);
for iCtx = 1:2
    [values, weights, names] = localPathColumns(contextTables{iCtx}, E, base);
    pm = packPreMaet(values, weights, ...
                     flatSpecs(values, 'sigma', SIGMA_LABEL, 'isPer', false, ...
                               'period', 0, 'name', names));
    pm = bindAttributes(pm, names(1:end-1), 'name', 'label', ...
                        'r', numel(names) - 1, 'exch', false);
    pm = bindEvents(pm, [], 'groupBy', 'chord', 'rOuter', R_OUTER);
    contextPm{iCtx} = selectPreMaet(pm, 'attributes', {'label'});
end

% --- retrieval ---------------------------------------------------------------
% One-sided normalization divides by the query's self inner product, so a
% tune scores the number of occurrences of the configuration: at this
% kernel width a label either matches (1) or does not (0), and a tuple's
% weight is the product across its values.
qry = buildMaet(queryPm, 'verbose', false);
ctx = buildMaet(contextPm{1}, 'verbose', false);
ctl = buildMaet(contextPm{2}, 'verbose', false);
fprintf('\nretrieval of (%s):\n', strjoin(QUERY, ', '));
fprintf('  sOne(%-18s) = %.4f\n', TUNE, ...
        simMaet(ctx, qry, 'normalize', 'oneSidedDenom', 'verbose', false));
fprintf('  sOne(%-18s) = %.4f\n', CONTROL, ...
        simMaet(ctl, qry, 'normalize', 'oneSidedDenom', 'verbose', false));
fprintf('  cos(tune, tune)    = %.4f\n', simMaet(ctx, ctx, 'verbose', false));
fprintf('  cos(tune, control) = %.4f\n', simMaet(ctx, ctl, 'verbose', false));

% --- partial match -----------------------------------------------------------
% On a regular simplex every pair of labels is the same distance apart, so
% every substitution costs the same: the two scores below are equal by
% construction, which is what makes the simplex the neutral coding. Only the
% kernel widens, so the pre-MAET is the one already encoded, built at a sigma
% of its own: buildMaet's own argument takes precedence over the width the
% spec carries.
wideCtx = buildMaet(contextPm{1}, 'sigma', SIGMA_WIDE, 'verbose', false);
fprintf('\nat sigma = %g, substituting the query''s second label:\n', SIGMA_WIDE);
for iSub = 1:numel(SUBSTITUTES)
    [values, weights, names] = localPathColumns( ...
        localQueryTable({QUERY{1}, SUBSTITUTES{iSub}}, [1 2]), E, base);
    pm = packPreMaet(values, weights, ...
                     flatSpecs(values, 'sigma', SIGMA_LABEL, 'isPer', false, ...
                               'period', 0, 'name', names));
    pm = bindAttributes(pm, names(1:end-1), 'name', 'label', ...
                        'r', numel(names) - 1, 'exch', false);
    pm = bindEvents(pm, [], 'groupBy', 'chord', 'rOuter', R_OUTER);
    qSub = buildMaet(selectPreMaet(pm, 'attributes', {'label'}), ...
                     'sigma', SIGMA_WIDE, 'verbose', false);
    fprintf('  %-12s -> %.3f\n', SUBSTITUTES{iSub}, ...
        simMaet(wideCtx, qSub, 'normalize', 'oneSidedDenom', 'verbose', false));
end

% --- reduction by degree -----------------------------------------------------
% The hard reduction cuts every path at the reduction level; the graded ones
% keep the deeper positions at a weight that decays with depth.
cutOpts = base;
cutOpts.truncate = REDUCTION_LEVEL;
[values, weights, names] = localPathColumns(tuneRows, E, cutOpts);
pm = packPreMaet(values, weights, ...
                 flatSpecs(values, 'sigma', SIGMA_LABEL, 'isPer', false, ...
                           'period', 0, 'name', names));
pm = bindAttributes(pm, names(1:end-1), 'name', 'label', ...
                    'r', numel(names) - 1, 'exch', false);
pm = bindEvents(pm, [], 'groupBy', 'chord', 'rOuter', R_OUTER);
hardCut = buildMaet(selectPreMaet(pm, 'attributes', {'label'}), 'verbose', false);

redCos = zeros(1, numel(G_VALUES));
for iG = 1:numel(G_VALUES)
    gradedOpts = base;
    gradedOpts.decay = G_VALUES(iG);
    [values, weights, names] = localPathColumns(tuneRows, E, gradedOpts);
    pm = packPreMaet(values, weights, ...
                     flatSpecs(values, 'sigma', SIGMA_LABEL, 'isPer', false, ...
                               'period', 0, 'name', names));
    pm = bindAttributes(pm, names(1:end-1), 'name', 'label', ...
                        'r', numel(names) - 1, 'exch', false);
    pm = bindEvents(pm, [], 'groupBy', 'chord', 'rOuter', R_OUTER);
    graded = buildMaet(selectPreMaet(pm, 'attributes', {'label'}), ...
                       'verbose', false);
    redCos(iG) = simMaet(graded, hardCut, 'verbose', false);
end
fprintf('\nreduction: the graded tune against the hard cut at level %d\n', ...
        REDUCTION_LEVEL);
for iG = 1:numel(G_VALUES)
    fprintf('  g = %-4g cos = %.4f\n', G_VALUES(iG), redCos(iG));
end

% --- depth in the comparison -------------------------------------------------
% The level enters as one further inner coordinate, scaled so that a
% displacement of one level sits at kernel distance sLevel. Query and context
% take the same scale, so both are encoded inside the sweep.
depthS = zeros(1, numel(S_RATIOS));
for iS = 1:numel(S_RATIOS)
    depthOpts = base;
    depthOpts.levelScale = S_RATIOS(iS) * SIGMA_LABEL;
    depthTables = {localQueryTable(QUERY, QUERY_LEVELS), tuneRows};
    built = cell(1, 2);
    for iSide = 1:2
        [values, weights, names] = localPathColumns( ...
            depthTables{iSide}, E, depthOpts);
        pm = packPreMaet(values, weights, ...
                         flatSpecs(values, 'sigma', SIGMA_LABEL, ...
                                   'isPer', false, 'period', 0, 'name', names));
        pm = bindAttributes(pm, names(1:end-1), 'name', 'label', ...
                            'r', numel(names) - 1, 'exch', false);
        pm = bindEvents(pm, [], 'groupBy', 'chord', 'rOuter', R_OUTER);
        built{iSide} = buildMaet(selectPreMaet(pm, 'attributes', {'label'}), ...
                                 'verbose', false);
    end
    depthS(iS) = simMaet(built{2}, built{1}, ...
                         'normalize', 'oneSidedDenom', 'verbose', false);
end
fprintf('\ndepth in the comparison: the query at levels %s\n', ...
        strjoin(arrayfun(@(v) sprintf('%d', v), QUERY_LEVELS, ...
                         'UniformOutput', false), '/'));
for iS = 1:numel(S_RATIOS)
    fprintf('  sLevel / sigma = %-4g sOne = %.2f\n', S_RATIOS(iS), depthS(iS));
end

% --- figure ------------------------------------------------------------------
fig = figure('Color', 'w', 'Position', [100 100 1300 480]);

axR = subplot(1, 2, 1);
plot(axR, G_VALUES, redCos, 'o-', 'Color', C_TUNE, ...
     'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', C_TUNE);
xlabel(axR, sprintf('grading g (weight per level beyond %d)', REDUCTION_LEVEL), ...
       'FontSize', 15);
ylabel(axR, 'cosine against the hard cut', 'FontSize', 15);
title(axR, 'Reduction by degree', 'FontSize', 17);
ylim(axR, [0 1.05]);
set(axR, 'XDir', 'reverse', 'FontSize', 13, 'Box', 'off');
grid(axR, 'on');

axD = subplot(1, 2, 2);
plot(axD, S_RATIOS, depthS, 'o-', 'Color', C_QUERY, ...
     'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', C_QUERY);
xlabel(axD, 's_{level} / \sigma', 'FontSize', 15);
ylabel(axD, 'one-sided similarity', 'FontSize', 15);
title(axD, 'Depth in the comparison', 'FontSize', 17);
ylim(axD, [0 max(depthS) * 1.1]);
set(axD, 'FontSize', 13, 'Box', 'off');
grid(axD, 'on');

sgtitle(fig, 'A supplied parse: reduction by degree, and depth as a coordinate', ...
        'FontSize', 17);

figDir = fullfile(thisDir, 'figures');
if SAVE_FIGURES && ~exist(figDir, 'dir'), mkdir(figDir); end
if SAVE_FIGURES
    print(fig, '-dpng', '-r140', fullfile(figDir, 'demo_jmm_4_1_parse.png'));
    fprintf('Saved figures/demo_jmm_4_1_parse.png\n');
end

% The demo leaves the toolbox as it found it: the defaults it set at the
% top are restored here.
mptDefaults(prevDefaults);


% --- local functions ---------------------------------------------------------

function t = localQueryTable(labels, levels)
%LOCALQUERYTABLE  The query as a table of the same shape as a derivation:
%   one chord, one row per position, at the levels given.
    t = table(repmat({'query'}, numel(labels), 1), zeros(numel(labels), 1), ...
              double(levels(:)), labels(:), ...
              'VariableNames', {'tune', 'chord', 'level', 'label'});
end


function [values, weights, names] = localPathColumns(t, E, opts)
%LOCALPATHCOLUMNS  The value rows, weights, and names of a table of positions.
%   No toolbox function is called here: this only turns the table into the
%   arrays the pre-MAET is packed from. opts.levelScale appends the level as
%   one further coordinate, scaled, so that depth enters the comparison;
%   opts.decay weights a position by decay^(level - reductionLevel) beyond
%   that level, grading the path by degree; opts.truncate drops the positions
%   beyond a level outright, which is the hard reduction the grading
%   approaches.
    if ~isempty(opts.truncate)
        t = t(t.level <= opts.truncate, :);
    end
    [~, whichLabel] = ismember(t.label, E.alphabet);
    coords = E.vertices(whichLabel, :).';         % (V-1) x N
    lev = double(t.level(:)).';                   % 1 x N
    chordIdx = double(t.chord(:)).';              % 1 x N

    values = cell(1, E.dim);
    names = cell(1, E.dim);
    for i = 1:E.dim
        values{i} = coords(i, :);
        names{i} = sprintf('coord%d', i);
    end
    if ~isempty(opts.levelScale)
        values{end + 1} = opts.levelScale * lev;
        names{end + 1} = sprintf('coord%d', E.dim + 1);
    end
    values{end + 1} = chordIdx;
    names{end + 1} = 'chord';

    % A position's weight multiplies into every tuple that reads it; the
    % coordinates of one position share it, so it is carried by the first
    % and the others weigh 1.
    weights = [];
    if ~isempty(opts.decay)
        beyond = max(lev - E.reductionLevel, 0);
        weights = [{opts.decay .^ beyond}, ...
                   repmat({ones(1, numel(lev))}, 1, numel(values) - 1)];
    end
end
