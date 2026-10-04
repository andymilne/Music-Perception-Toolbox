%% demo_jmm_4_1_parse.m
% Analysis 4.1 (Online Supplement, Section 11): a supplied parse carried as
% a nested multiset.
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
% The nested encoding. Each chord is one event and its path one nested
% multiset within one attribute: the inner level a rule label's simplex
% coordinates, read whole and in order, so that two labels are either the
% same or equally different; the outer level the positions of the path, in
% order, so that position carries depth. Paths differ in length, so the
% positions of a chord are bound into one super-event by 'groupBy'
% (consecutive rows sharing a chord index) rather than by a fixed window,
% and the outer tuple size rOuter then says how much of a path a
% comparison takes at once. At rOuter = 2 a tuple is an ordered pair of
% positions, matched wherever it occurs in order, contiguous or not, so
% that a configuration is found at any depth and across intervening
% elaboration without a level coordinate.
%
% Five things the framework then does with it, the last on a second,
% unrolled encoding:
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
%                 itself rather than being ignored;
%   marginals     unrolled into one event per (chord, position) across
%                 the whole corpus, each weighted 1/m with m the number
%                 of surface chords its node governs, the label marginal
%                 returns the corpus's rule frequencies, and carrying each
%                 chord's quality alongside gives the joint distribution
%                 of rule and surface.
%
% Pre-MAET structure:
%
%     attribute   r (inner, outer)  sigma          rel   per
%     ---------   ----------------  -------------  ----  ----
%     label       (V-1, 2)          0.1 (or 0.3)   no    no    nested
%     label       (V, 2)            0.1            no    no    nested, with level
%     label       V-1               0.1            no    no    unrolled
%     quality     Q-1               0.1            no    no    unrolled
%
%     V is the number of rule labels in the alphabet, so a label is a
%     point of a regular (V-1)-simplex of unit edge; Q is the number of
%     chord qualities, likewise. Ordered (exch = 0) at every level.
%     Estimator: one-sided similarity for retrieval and the marginals,
%     cosine for the reduction.
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
DOMINANT_SEVENTH = 'Maj Min Min'; % a major third, then two minor thirds

C_TUNE = [0.122 0.306 0.722];
C_QUERY = [0.761 0.314 0.031];

% --- the derivations, and the alphabet of rule labels ------------------------
parseRows = jmm.derivations({TUNE, CONTROL});
alphabet = unique([parseRows.label; QUERY(:); SUBSTITUTES(:)]);
fprintf('%d tunes, %d path positions, %d rule labels: %s\n', ...
        numel(unique(parseRows.tune)), height(parseRows), ...
        numel(alphabet), strjoin(alphabet(:).', ', '));
fprintf('each label is a vertex of a unit-edge %d-simplex\n', numel(alphabet) - 1);

tuneRows = parseRows(strcmp(parseRows.tune, TUNE), :);
controlRows = parseRows(strcmp(parseRows.tune, CONTROL), :);

% The parameters every nested encoding shares, and its options as a struct,
% so that each analysis below changes one field and leaves the rest unset.
P = struct('alphabet', {alphabet}, 'sigma', SIGMA_LABEL, ...
           'rOuter', R_OUTER, 'reductionLevel', REDUCTION_LEVEL);
base = struct('levelScale', [], 'decay', [], 'truncate', []);

% --- the nested encoding -----------------------------------------------------
queryPm = localEncodePaths(localQueryTable(QUERY, [1 2]), P, base);
showPreMaet(queryPm, 'maxEvents', 1, 'decimals', 2);
tunePm = localEncodePaths(tuneRows, P, base);

% --- retrieval ---------------------------------------------------------------
% One-sided normalization divides by the query's self inner product, so a
% tune scores the number of occurrences of the configuration: at this
% kernel width a label either matches (1) or does not (0), and a tuple's
% weight is the product across its values.
qry = buildMaet(queryPm, 'verbose', false);
ctx = buildMaet(tunePm, 'verbose', false);
ctl = buildMaet(localEncodePaths(controlRows, P, base), 'verbose', false);
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
% kernel widens, so the pre-MAETs are the ones already encoded, built at a
% sigma of their own: buildMaet's own argument takes precedence over the
% width the spec carries.
wideCtx = buildMaet(tunePm, 'sigma', SIGMA_WIDE, 'verbose', false);
fprintf('\nat sigma = %g, substituting the query''s second label:\n', SIGMA_WIDE);
for iSub = 1:numel(SUBSTITUTES)
    qSub = buildMaet(localEncodePaths( ...
        localQueryTable({QUERY{1}, SUBSTITUTES{iSub}}, [1 2]), P, base), ...
        'sigma', SIGMA_WIDE, 'verbose', false);
    fprintf('  %-12s -> %.3f\n', SUBSTITUTES{iSub}, ...
        simMaet(wideCtx, qSub, 'normalize', 'oneSidedDenom', 'verbose', false));
end

% --- reduction by degree -----------------------------------------------------
% The hard reduction cuts every path at the reduction level; the graded ones
% keep the deeper positions at a weight that decays with depth.
cutOpts = base;
cutOpts.truncate = REDUCTION_LEVEL;
hardCut = buildMaet(localEncodePaths(tuneRows, P, cutOpts), 'verbose', false);
redCos = zeros(1, numel(G_VALUES));
for iG = 1:numel(G_VALUES)
    gradedOpts = base;
    gradedOpts.decay = G_VALUES(iG);
    graded = buildMaet(localEncodePaths(tuneRows, P, gradedOpts), 'verbose', false);
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
    qDepth = buildMaet(localEncodePaths( ...
        localQueryTable(QUERY, QUERY_LEVELS), P, depthOpts), 'verbose', false);
    tDepth = buildMaet(localEncodePaths(tuneRows, P, depthOpts), 'verbose', false);
    depthS(iS) = simMaet(tDepth, qDepth, ...
                         'normalize', 'oneSidedDenom', 'verbose', false);
end
fprintf('\ndepth in the comparison: the query at levels %s\n', ...
        strjoin(arrayfun(@(v) sprintf('%d', v), QUERY_LEVELS, ...
                         'UniformOutput', false), '/'));
for iS = 1:numel(S_RATIOS)
    fprintf('  sLevel / sigma = %-4g sOne = %.2f\n', S_RATIOS(iS), depthS(iS));
end

% --- marginals, on the unrolled encoding -------------------------------------
% The unrolled encoding makes one event of each (chord, position) of every
% derivation in the corpus, the label and the chord's quality becoming two
% flat attributes, each holding its simplex coordinates read whole and in
% order. Weighting each event 1/m, with m the number of surface chords its
% node governs, gives every rule application unit total weight, so the
% one-sided similarity of the corpus against a one-event query holding a
% label (retrieval, as above, now of single positions) is that rule's
% frequency in the corpus. At unit weights, the same reading against a
% (label, quality) query, divided by the reading against the label alone,
% is the share of that quality among the chords the rule governs.
corpus = jmm.derivations();
U = struct('rules', {unique(corpus.label)}, ...
           'qualities', {unique(corpus.quality)}, 'sigma', SIGMA_LABEL);

byRule = localLabelsOnly(localEncodeUnrolled(corpus, U, 1 ./ corpus.governed));
freq = zeros(1, numel(U.rules));
for iR = 1:numel(U.rules)
    freq(iR) = simMaet(byRule, ...
        localLabelsOnly(localPosition(U.rules{iR}, DOMINANT_SEVENTH, U)), ...
        'normalize', 'oneSidedDenom', 'verbose', false);
end
fprintf('\nunrolled: %d derivations, %d events\n', ...
        numel(unique(corpus.tune)), height(corpus));
fprintf('rule frequencies, read from the label marginal:\n');
[~, order] = sort(freq, 'descend');
for iR = order
    fprintf('  %-15s %7.1f\n', U.rules{iR}, freq(iR));
end

plainPm = localEncodeUnrolled(corpus, U, []);
jointDens = buildMaet(plainPm, 'verbose', false);
byLabel = localLabelsOnly(plainPm);
fprintf(['share of the dominant-seventh quality (%s) among the chords a ' ...
         'rule governs:\n'], DOMINANT_SEVENTH);
for rule = {'V_I', 'Repeat'}
    qPm = localPosition(rule{1}, DOMINANT_SEVENTH, U);
    both = simMaet(jointDens, buildMaet(qPm, 'verbose', false), ...
                   'normalize', 'oneSidedDenom', 'verbose', false);
    alone = simMaet(byLabel, localLabelsOnly(qPm), ...
                    'normalize', 'oneSidedDenom', 'verbose', false);
    fprintf('  under %-7s %.0f%%\n', rule{1}, 100 * both / alone);
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


function rows = localSimplexRows(categories, alphabet)
%LOCALSIMPLEXROWS  Each category as its vertex of the unit-edge regular
%   simplex whose vertices are the alphabet, returned as one 1 x N value
%   row per coordinate. No toolbox function beyond simplexVertices is
%   called here: this only turns a column of the table into the arrays a
%   pre-MAET is packed from, so that the encoding itself stays in the open.
    vertices = simplexVertices(numel(alphabet));   % one row per category
    [~, which] = ismember(categories, alphabet);
    rows = num2cell(vertices(which, :).', 2).';    % 1 x (V-1) cell of rows
end


function pm = localEncodePaths(t, P, opts)
%LOCALENCODEPATHS  A table of path positions as the nested pre-MAET, one
%   event per chord. Four calls make the encoding, and every analysis of
%   the nested form makes the same four: pack the columns with their kernel
%   parameters; bind the coordinates of a label into one attribute, read
%   whole and in order; bind the positions of one chord into one
%   super-event, the group read from the chord index ('groupBy') rather
%   than from a window width; and keep the bound attribute, the chord index
%   having done its work. opts.levelScale appends the level as one further
%   coordinate, scaled, so that depth enters the comparison; opts.decay
%   weights a position by decay^(level - reductionLevel) beyond that level,
%   grading the path by degree; opts.truncate drops the positions beyond a
%   level outright, which is the hard reduction the grading approaches.
    if ~isempty(opts.truncate)
        t = t(t.level <= opts.truncate, :);
    end
    lev = double(t.level(:)).';                   % 1 x N
    values = localSimplexRows(t.label, P.alphabet);
    if ~isempty(opts.levelScale)
        values{end + 1} = opts.levelScale * lev;
    end
    inner = arrayfun(@(i) sprintf('coord%d', i), 1:numel(values), ...
                     'UniformOutput', false);
    values{end + 1} = double(t.chord(:)).';

    % A position's weight multiplies into every tuple that reads it; the
    % coordinates of one position share it, so it is carried by the first
    % and the others weigh 1.
    weights = [];
    if ~isempty(opts.decay)
        beyond = max(lev - P.reductionLevel, 0);
        weights = [{opts.decay .^ beyond}, ...
                   repmat({ones(1, numel(lev))}, 1, numel(inner))];
    end
    pm = packPreMaet(values, weights, ...
                     flatSpecs(values, 'sigma', P.sigma, 'per', false, ...
                               'period', 0, 'names', [inner, {'chord'}]));
    pm = bindAttributes(pm, inner, 'name', 'label', ...
                        'r', numel(inner), 'exch', false);
    pm = bindEvents(pm, [], 'groupBy', 'chord', 'rOuter', P.rOuter);
    pm = selectPreMaet(pm, 'attributes', {'label'});
end


function pm = localEncodeUnrolled(t, U, weights)
%LOCALENCODEUNROLLED  A table of path positions as the unrolled pre-MAET,
%   one event per (chord, position): its label and its chord's quality,
%   each one attribute holding a simplex vertex, read whole and in order.
%   weights (empty for unit weights) is carried by the first coordinate.
    label = localSimplexRows(t.label, U.rules);
    quality = localSimplexRows(t.quality, U.qualities);
    namesLabel = arrayfun(@(i) sprintf('rule%d', i), 1:numel(label), ...
                          'UniformOutput', false);
    namesQuality = arrayfun(@(i) sprintf('quality%d', i), 1:numel(quality), ...
                            'UniformOutput', false);
    values = [label, quality];
    w = [];
    if ~isempty(weights)
        w = [{double(weights(:)).'}, ...
             repmat({ones(1, height(t))}, 1, numel(values) - 1)];
    end
    pm = packPreMaet(values, w, ...
                     flatSpecs(values, 'sigma', U.sigma, 'per', false, ...
                               'period', 0, 'names', [namesLabel, namesQuality]));
    pm = bindAttributes(pm, namesLabel, 'name', 'label', ...
                        'r', numel(namesLabel), 'exch', false);
    pm = bindAttributes(pm, namesQuality, 'name', 'quality', ...
                        'r', numel(namesQuality), 'exch', false);
end


function pm = localPosition(label, quality, U)
%LOCALPOSITION  A one-event query: one position, its label and its chord's
%   quality.
    pm = localEncodeUnrolled(table({label}, {quality}, ...
                                   'VariableNames', {'label', 'quality'}), U, []);
end


function dens = localLabelsOnly(pm)
%LOCALLABELSONLY  The density of the label attribute alone.
    dens = buildMaet(selectPreMaet(pm, 'attributes', {'label'}), 'verbose', false);
end
