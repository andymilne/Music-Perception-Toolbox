%% demo_jmm_1_3_tonic_tuple_size.m
% Analysis 1.3 (Section 4.1.3 of the JMM article; the article calls it
% Analysis 1.2 in the reduced version).
%
% A demo of the Music Perception Toolbox reproducing the analysis from the
% JMM article; lightly edited from the article's own script. Data come
% from the jmm package (BWV 347 read from the bundled MusicXML) or
% jmm.pianoPhase (the rendered Piano Phase voices); a figure is written to
% figures/.
%
% Analysis 1.3: structural matching of the four cadence tonics of BWV 347
% at increasing tuple size r, on a single chord (no nesting).
%
% Each cadence tonic (the final chord of cadences C1-C4) is one event with
% a single pitch attribute, the unordered chord multiset (sym = 1, K = 4).
% The four tonics are compared pairwise under the cross of absolute vs
% relative mode and non-periodic vs periodic, each swept over r in {1,2,3}.
% Because the comparison is on one attribute (not a role product), raising
% r tightens the match informatively rather than annihilating it: pitch
% content (r = 1) -> dyad/interval content (r = 2) -> triad content (r = 3).
%
% The payoff is the relative, periodic row. At r = 2 the interval-class
% content cannot separate a major triad from a minor one (they are
% inversionally related, and the unordered relative pair content is
% inversion-invariant): the three major tonics and the minor tonic all read
% as near-identical. At r = 3 the triadic structure separates them: the
% three majors stay mutually 1 (transposition-equivalent) and the minor
% isolates. Relative mode at r = 1 is a constant (degenerate) density and is
% shown only for completeness.
%
% These are the same four tonics compared as whole nested cadences in
% Analysis 1.4, in the same panel layout, so the two figures read together.
%
% Toolbox-dependency notes
% ------------------------
% Uses
%     buildExpTens, cosSimExpTens; jmm.bwv347Grid.
%
% The Python mirror is demos/jmm/demo_jmm_1_3_tonic_tuple_size.py.

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
addpath(fullfile(thisDir, '..', '..'));
mptDefaults('showHints', false);

% ---------------------------------------------------------------------------
% Parameters
% ---------------------------------------------------------------------------
SIGMA_PITCH = 0.15;
PERIOD = 12.0;
R_VALUES = [1 2 3];
TONIC_TIMES = [7.0, 15.0, 45.0, 65.0];       % cadence finals C1..C4
CADENCE_NAMES = {'C1', 'C2', 'C3', 'C4'};

% ---------------------------------------------------------------------------
% Extract the cadence tonics
% ---------------------------------------------------------------------------
fprintf('Loading BWV 347 and extracting cadence tonics...\n');
[times, pitchesSatb, ~] = jmm.bwv347Grid();
tonics = zeros(4, 4);                       % row cid: (4,) pitch vector
for cid = 1:4
    [~, k] = min(abs(times - TONIC_TIMES(cid)));
    tonics(cid, :) = pitchesSatb(k, :);
    fprintf('  tonic C%d: [%s]\n', cid, ...
            strjoin(arrayfun(@(x) sprintf('%.1f', x), tonics(cid, :), ...
                             'UniformOutput', false), ', '));
end

% ---------------------------------------------------------------------------
% Figure: 2 x 6 landscape. Rows are absolute / relative; the left half
% (cols 1-3) is non-periodic and the right half (cols 4-6) periodic, with
% r ascending 1 -> 3 within each half. The two relative r = 1 cells are
% degenerate (a single pitch has no relative content) and are omitted.
% Cadence names sit on each row's leftmost visible cell and each column's
% lowest visible cell; r headers sit on the top row.
% ---------------------------------------------------------------------------
modeRows = {false, 'Absolute'; true, 'Relative'};
halves = [false true];                      % non-periodic, periodic

fig = figure('Position', [100 100 1290 540], 'Color', 'w');
% Panel geometry (normalized figure units): left 0.065, right 0.99, top
% 0.80, bottom 0.085, with gaps of a tenth of a panel between panels.
axW = (0.99 - 0.065) / 6.5;
axH = (0.80 - 0.085) / 2.1;
axesPos = cell(2, 6);
for row = 1:2
    isRel = modeRows{row, 1};
    for half = 1:2
        isPer = halves(half);
        for ri = 1:numel(R_VALUES)
            r = R_VALUES(ri);
            col = (half - 1) * 3 + ri;
            ax = axes('Parent', fig, 'Position', ...
                      [0.065 + (col - 1) * 1.1 * axW, ...
                       0.085 + (2 - row) * 1.1 * axH, axW, axH]);
            axesPos{row, col} = ax;
            if isRel && r == 1                      % degenerate: omit
                axis(ax, 'off');
                continue;
            end
            % All pairwise tonic similarities: each tonic is a single-event
            % density (one pitch attribute, sym = 1); the 4 x 4 matrix (unit
            % diagonal, symmetric) comes from one broadcast density-list
            % call per row.
            dens = cell(1, 4);
            for c = 1:4
                dens{c} = buildExpTens({tonics(c, :).'}, [], SIGMA_PITCH, r, ...
                                       isRel, isPer, PERIOD, 'verbose', false);
            end
            M = zeros(4, 4);
            for i = 1:4
                M(i, :) = cell2mat(cosSimExpTens(dens{i}, dens, 'verbose', false));
            end
            imagesc(ax, M, [0 1]);
            colormap(ax, jmm.colourMap('viridis'));
            axis(ax, 'image');
            for i = 1:4
                for j = 1:4
                    if M(i, j) < 0.5, c = 'w'; else, c = 'k'; end
                    text(ax, j, i, sprintf('%.2f', M(i, j)), ...
                         'HorizontalAlignment', 'center', ...
                         'VerticalAlignment', 'middle', 'FontSize', 10, ...
                         'Color', c);
                end
            end
            set(ax, 'XTick', 1:4, 'YTick', 1:4, 'FontSize', 12);
            % x labels on each column's lowest visible cell (cols 1 and 4
            % have their relative r = 1 cell omitted, so they carry the
            % cadence names on the absolute row)
            xEdge = (row == 2) || any(col == [1 4]);
            if xEdge, set(ax, 'XTickLabel', CADENCE_NAMES);
            else,     set(ax, 'XTickLabel', {}); end
            % y labels on each row's leftmost visible cell
            if isRel, leftCol = 2; else, leftCol = 1; end
            if col == leftCol, set(ax, 'YTickLabel', CADENCE_NAMES);
            else,              set(ax, 'YTickLabel', {}); end
            if row == 1
                title(ax, sprintf('r = %d', r), 'FontSize', 14);
            end
        end
    end
end
% half-headers spanning each block of three columns, and row labels
annotation(fig, 'textbox', [0.20 0.855 0.20 0.05], 'String', 'Non-periodic', ...
           'HorizontalAlignment', 'center', 'FontSize', 15, 'EdgeColor', 'none');
annotation(fig, 'textbox', [0.645 0.855 0.20 0.05], 'String', 'Periodic', ...
           'HorizontalAlignment', 'center', 'FontSize', 15, 'EdgeColor', 'none');
for row = 1:2
    pos = get(axesPos{row, 2}, 'Position');
    annotation(fig, 'textbox', [0.0, pos(2), 0.06, pos(4)], ...
               'String', modeRows{row, 2}, 'FontSize', 15, ...
               'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
               'VerticalAlignment', 'middle');
end
annotation(fig, 'textbox', [0.05 0.905 0.9 0.09], 'String', ...
           {'BWV 347 cadence-tonic pair similarity (single chords), \sigma_{pitch} = 15 cents', ...
            ['single pitch attribute, sym 1 (unordered chord); cosine in [0,1], ' ...
             'diagonals 1, symmetric; relative r = 1 omitted (degenerate); ' ...
             'major/minor separation appears at r = 3']}, ...
           'HorizontalAlignment', 'center', 'FontSize', 11, 'EdgeColor', 'none');

figDir = fullfile(thisDir, 'figures');
if ~exist(figDir, 'dir'), mkdir(figDir); end
outPng = fullfile(figDir, 'demo_jmm_1_3_tonic_tuple_size.png');
print(fig, '-dpng', '-r140', outPng);
fprintf('Saved figures/demo_jmm_1_3_tonic_tuple_size.png\n');

