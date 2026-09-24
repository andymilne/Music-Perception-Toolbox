%% demo_jmm_1_2_similarity.m
% Analysis 1.2 (JMM article, Section 4.1.2): voice-aware versus
% voice-agnostic similarity across the
% pitch–pitch-class blend.
%
% A demo of the Music Perception Toolbox reproducing the analysis from the
% JMM article (Section 4.1.2, "Voice-aware versus voice-agnostic across
% the pitch–pitch-class blend"), lightly edited from the article's own
% scripts. Data come from the jmm package (BWV 347 read from the bundled
% MusicXML); the figures stay on screen unless SAVE_FIGURES is set.
%
% What the analysis asks. When do two chords count as alike? Six chord
% pairs from BWV 347 — an identical voicing, a bass octave shift, a full
% re-voicing, root position against first inversion, and two
% different-chord baselines — are compared under three encodings of
% voice information, each embodying a different answer, across the
% pitch–pitch-class continuum of Shepard's helix stretched or compressed.
%
% How it is computed. Every pitch is routed through two attributes at
% once: a periodic pitch-class attribute (sigma_pc = 50 cents, P = 1200)
% and a non-periodic pitch-height attribute whose width sigma_ph is swept
% from one semitone to several octaves — narrow, and pitches must agree
% in octave to count as similar; wide, and pitch-class equivalence
% dominates. Voice information enters in one of three ways:
%   (i)   Voice-aware: one event per chord; each attribute holds the
%         ordered (S, A, T, B) voicing — K = 4, [exch] = 0, r = 4 — so
%         matching is voice by voice, a multiplicative AND across voices.
%   (ii)  Simplex-voice: one event per note (N = 4 single-pitch events);
%         pitch class and pitch height at r = 1, plus a voice attribute
%         holding each note's vertex of a regular tetrahedron
%         (the simplex role, its three coordinates taken in order:
%         [exch] = 0, r = 3, sigma_voice = 0.2), so matching accrues
%         additive partial credit, voice by voice.
%   (iii) Voice-agnostic: one event per note (N = 4, K = 1, r = 1) on the
%         same two attributes -- the simplex-voice encoding without its
%         voice attribute, so each note's pitch class stays bound to its
%         own height; voice identity is not encoded.
% Each encoding is one preMaetFromAttrTable call on the gridded chorale --
% the voice enters through 'roles', and 'chords' sets the grain -- so the
% three differ only in those two arguments. A chord's density is then one
% selectPreMaet (its events, and the pitch attributes alone) and one
% buildMaet, whose 'sigma' override carries the sweep; the six pair
% similarities come from one batched simMaet call on density lists
% (elementwise list mode).
%
% An appendix figure (HEATMAPS = true) extends the same three encodings
% to every event of the chorale: N x N cosine-similarity matrices over the
% 272 sixteenth-note grid points, one broadcast simMaet call per
% row and encoding, at three pitch-height widths.
%
% Data: jmm.bwv347Notes. Toolbox: gridAttrTable, preMaetFromAttrTable,
% selectPreMaet, buildMaet, simMaet. Runtime: seconds for the sweep; a few
% minutes more for the heat maps.

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

HEATMAPS = false;            % true: also compute the appendix heat maps

% ---------------------------------------------------------------------------
% Parameters (cents; the article's tables quote the same values in semitones)
% ---------------------------------------------------------------------------
SIGMA_PC = 50.0;
SIGMA_VOICE = 0.2;
SIGMA_PHS = logspace(log10(100), log10(8000), 50);
SIGMA_PH_SNAPSHOTS = [200.0, 600.0, 3000.0];       % heat-map widths

% ---------------------------------------------------------------------------
% Load chorale, identify reference chord pairs by score time
% ---------------------------------------------------------------------------
GRID_STEP_QN = jmm.gridStepQn();
g = gridAttrTable(jmm.bwv347Notes(), GRID_STEP_QN);
eventAt = @(t) round(t / GRID_STEP_QN) + 1;         % 1-based grid index

% Six reference chord pairs (label, t_i, t_j), times in the played-through
% chorale (bars 1–4 repeat, so events from bar 5 on sit 16 QN later than in
% the unexpanded score). Chord locations (bar.beat):
%   (a) A maj b3.b1 vs b16.b1            (d) A maj b3.b1 vs b3.b3
%   (b) E maj cad1 b2.b3 vs cad2 b4.b3   (e) E maj cad1 b2.b3 vs B min cad3 b12.b1
%   (c) E maj b2.b3 vs b2.b4             (f) E maj cad1 b2.b3 vs A maj cad4 b17.b1
PAIR_LABELS = {'(a) identical voicing', '(b) bass octave shift', ...
               '(c) full re-voicing', '(d) root vs first inv', ...
               '(e) E maj vs B min', '(f) E maj vs A maj'};
PAIR_TIMES = [9.0, 61.0; 7.0, 15.0; 7.0, 8.0; 9.0, 11.0; 7.0, 45.0; 7.0, 65.0];
PAIR_COLOURS = [0.122 0.306 0.722; 0.169 0.541 0.243; 0.761 0.314 0.031; ...
                0.690 0.188 0.376; 0.400 0.400 0.400; 0.667 0.667 0.667];
nPairs = numel(PAIR_LABELS);

% ---------------------------------------------------------------------------
% The three encodings
% ---------------------------------------------------------------------------
% One conversion each, from the same gridded table and the same attributes.
% Every pitch is routed through two attributes of the one pitch column, read
% in cents: a periodic pitch-class attribute and a non-periodic pitch-height
% attribute. The onset attribute locates a chord in the piece and is dropped
% before any density is built, so its width never enters; the pitch-height
% width is the sweep's, which buildMaet overrides per call.
ATTRIBUTES = {struct('column', 'pitch', 'name', 'pitchClass', ...
                     'sigma', SIGMA_PC, 'isPer', true, 'period', 1200), ...
              struct('column', 'pitch', 'name', 'pitchHeight', ...
                     'sigma', SIGMA_PHS(1)), ...
              struct('column', 'onset', 'sigma', 1.0)};

% The voice attribute the simplex role builds carries its own width rather
% than taking one from the list above.
VOICE = struct('role', 'simplex', 'sigma', SIGMA_VOICE);

% Voice-aware binds the chord into one event and reads it as the ordered
% (S, A, T, B) voicing on both pitch attributes (r = 4, exch = 0);
% simplex-voice takes one event per note and adds the voice as a simplex
% vertex; voice-agnostic is the same grain with no voice attribute. The
% three differ only in 'roles' and 'chords'.
BUILDER_TITLES = {'Voice-aware encoding', ...
                  sprintf('Simplex-voice encoding (\\sigma_{voice} = %g)', SIGMA_VOICE), ...
                  'Voice-agnostic encoding'};
% One conversion each, from the same table and the same attributes.
ENCODINGS = { ...
    preMaetFromAttrTable(g, 'attributes', ATTRIBUTES, 'time', 'beats', ...
        'pitch', 'cents', 'weights', 'ones', ...
        'roles', struct('part', 'orderedMultiset')), ...
    preMaetFromAttrTable(g, 'attributes', ATTRIBUTES, 'time', 'beats', ...
        'pitch', 'cents', 'weights', 'ones', ...
        'chords', 'separate', 'roles', struct('part', VOICE)), ...
    preMaetFromAttrTable(g, 'attributes', ATTRIBUTES, 'time', 'beats', ...
        'pitch', 'cents', 'weights', 'ones', 'chords', 'separate')};
% The sigmas of each encoding's kept attributes, given the swept width.
ENCODING_SIGMAS = {@(sph) [SIGMA_PC, sph], ...
                   @(sph) [SIGMA_PC, sph, SIGMA_VOICE], ...
                   @(sph) [SIGMA_PC, sph]};
nBuilders = numel(ENCODINGS);

% The grid points, read off the voice-aware encoding, whose events are the
% grid points themselves.
awarePAttr = unpackPreMaet(ENCODINGS{1});
times = awarePAttr{3};
N = numel(times);

% The three encodings carry the same chord differently, so each is shown
% as the pre-MAET the cosine actually receives, on the cadence-1 tonic.
for bIdx = 1:nBuilders
    showPreMaet(selectPreMaet(ENCODINGS{bIdx}, ...
        'attributes', localKept(ENCODINGS{bIdx}), ...
        'events', localChordEvents(ENCODINGS{bIdx}, 7.0)), ...
        'title', BUILDER_TITLES{bIdx});
    fprintf('\n');
end

% ---------------------------------------------------------------------------
% Sweep: (3 encodings, 6 pairs, numel(SIGMA_PHS)) cosine similarities
% ---------------------------------------------------------------------------
% The six pairs draw on eight distinct chords, several shared between
% pairs (the cadence-1 tonic at t = 7 QN appears in four of them), so each
% chord's density is built once per encoding and sigma_ph and the pair
% similarities are read from that cache.
chordTimes = unique(PAIR_TIMES(:)).';
[~, pairI] = ismember(PAIR_TIMES(:, 1), chordTimes);
[~, pairJ] = ismember(PAIR_TIMES(:, 2), chordTimes);
sims = zeros(nBuilders, nPairs, numel(SIGMA_PHS));
for spIdx = 1:numel(SIGMA_PHS)
    sigmaPh = SIGMA_PHS(spIdx);
    for bIdx = 1:nBuilders
        dens = cell(1, numel(chordTimes));
        for c = 1:numel(chordTimes)
            dens{c} = buildMaet(selectPreMaet(ENCODINGS{bIdx}, ...
                'attributes', localKept(ENCODINGS{bIdx}), ...
                'events', localChordEvents(ENCODINGS{bIdx}, chordTimes(c))), ...
                'sigma', ENCODING_SIGMAS{bIdx}(sigmaPh), 'verbose', false);
        end
        % One batched call per encoding: list-vs-list elementwise mode
        % returns all six pair similarities at once.
        sims(bIdx, :, spIdx) = cell2mat(simMaet(dens(pairI), dens(pairJ), ...
                                                      'verbose', false));
    end
end

% ---------------------------------------------------------------------------
% Report
% ---------------------------------------------------------------------------
cols = [100.0, 1200.0, 8000.0];
fprintf('cosine similarity at σ_ph = %s cents (σ_pc = 50 cents):\n', ...
        strjoin(arrayfun(@(c) sprintf('%g', c), cols, 'UniformOutput', false), ', '));
for bIdx = 1:nBuilders
    title_ = BUILDER_TITLES{bIdx};
    k = strfind(title_, ' (');
    if ~isempty(k), title_ = title_(1:k(1) - 1); end
    fprintf('  %s\n', title_);
    for pIdx = 1:nPairs
        vals = '';
        for c = cols
            [~, at] = min(abs(SIGMA_PHS - c));
            vals = [vals, sprintf('%5.3f  ', sims(bIdx, pIdx, at))]; %#ok<AGROW>
        end
        fprintf('    %-24s %s\n', PAIR_LABELS{pIdx}, strtrim(vals));
    end
end

% ---------------------------------------------------------------------------
% Figure: the sweep
% ---------------------------------------------------------------------------
figDir = fullfile(thisDir, 'figures');
if SAVE_FIGURES && ~exist(figDir, 'dir'), mkdir(figDir); end

fig = figure('Position', [100 100 1800 600], 'Color', 'w');
for bIdx = 1:nBuilders
    ax = axes('Parent', fig, 'Position', [0.06 + (bIdx - 1) * 0.315, 0.13, 0.27, 0.72]);
    hold(ax, 'on');
    for pIdx = 1:nPairs
        plot(ax, SIGMA_PHS, squeeze(sims(bIdx, pIdx, :)), ...
             'Color', PAIR_COLOURS(pIdx, :), 'LineWidth', 2);
    end
    set(ax, 'XScale', 'log', 'FontSize', 14, 'Box', 'off');
    xlim(ax, [SIGMA_PHS(1), SIGMA_PHS(end)]);
    ylim(ax, [-0.02, 1.02]);
    xlabel(ax, '\sigma_{ph} (cents)', 'FontSize', 17);
    title(ax, BUILDER_TITLES{bIdx}, 'FontSize', 18);
    grid(ax, 'on');
    if bIdx == 1
        ylabel(ax, 'cosine similarity', 'FontSize', 17);
        legend(ax, PAIR_LABELS, 'Location', 'southeast', 'FontSize', 11, 'Box', 'off');
    else
        set(ax, 'YTickLabel', {});
    end
end
annotation(fig, 'textbox', [0.1 0.92 0.8 0.07], 'String', ...
           sprintf('BWV 347 chord-pair similarity vs \\sigma_{ph} (\\sigma_{pc} = %g cents fixed)', SIGMA_PC), ...
           'HorizontalAlignment', 'center', 'FontSize', 20, 'EdgeColor', 'none');
if SAVE_FIGURES
    print(fig, '-dpng', '-r140', fullfile(figDir, 'demo_jmm_1_2_sweep.png'));
    print(fig, '-dpdf', fullfile(figDir, 'demo_jmm_1_2_sweep.pdf'));
    fprintf('Saved figures/demo_jmm_1_2_sweep.png\n');
end

% ---------------------------------------------------------------------------
% Appendix: N x N event-pair heat maps
% ---------------------------------------------------------------------------
if HEATMAPS
    maps = cell(numel(SIGMA_PH_SNAPSHOTS), nBuilders);
    for row = 1:numel(SIGMA_PH_SNAPSHOTS)
        sigmaPh = SIGMA_PH_SNAPSHOTS(row);
        fprintf('  σ_ph = %g: building %d densities x 3 encodings ...\n', sigmaPh, N);
        for bIdx = 1:nBuilders
            dens = cell(1, N);
            for i = 1:N
                dens{i} = buildMaet(selectPreMaet(ENCODINGS{bIdx}, ...
                    'attributes', localKept(ENCODINGS{bIdx}), ...
                    'events', localChordEvents(ENCODINGS{bIdx}, times(i))), ...
                    'sigma', ENCODING_SIGMAS{bIdx}(sigmaPh), 'verbose', false);
            end
            % The N x N matrix is symmetric with unit diagonal: one
            % broadcast call per row against the densities from that row
            % on, mirrored below the diagonal.
            S = zeros(N, N);
            for i = 1:N
                S(i, i:N) = cell2mat(simMaet(dens{i}, dens(i:N), ...
                                                   'verbose', false));
                S(i:N, i) = S(i, i:N).';
            end
            maps{row, bIdx} = S;
        end
    end

    barDownbeats = [1, 9, 17, 25, 33, 41, 49, 57, 65];
    tickPositions = arrayfun(eventAt, barDownbeats);
    tickLabels = arrayfun(@(t) sprintf('%d', floor((t - 1) / 4) + 1), ...
                          barDownbeats, 'UniformOutput', false);
    fig = figure('Position', [100 100 1600 1600], 'Color', 'w');
    for row = 1:numel(SIGMA_PH_SNAPSHOTS)
        sigmaPh = SIGMA_PH_SNAPSHOTS(row);
        for col = 1:nBuilders
            ax = axes('Parent', fig, 'Position', ...
                      [0.06 + (col - 1) * 0.29, 0.07 + (3 - row) * 0.285, 0.25, 0.25]);
            imagesc(ax, maps{row, col}, [0 1]);
            colormap(ax, jmm.colourMap('magma'));
            set(ax, 'YDir', 'normal');
            axis(ax, 'image');
            hold(ax, 'on');
            if row == 1
                title(ax, sprintf('%s\n(\\sigma_{ph} = %g cents)', ...
                                  BUILDER_TITLES{col}, sigmaPh), 'FontSize', 16);
            else
                title(ax, sprintf('\\sigma_{ph} = %g cents', sigmaPh), 'FontSize', 16);
            end
            for pIdx = 1:nPairs
                i = eventAt(PAIR_TIMES(pIdx, 1));
                j = eventAt(PAIR_TIMES(pIdx, 2));
                plot(ax, [j, i], [i, j], 's', 'Color', PAIR_COLOURS(pIdx, :), ...
                     'MarkerSize', 12, 'LineWidth', 2.0);
            end
            set(ax, 'XTick', tickPositions, 'XTickLabel', tickLabels, ...
                    'YTick', tickPositions, 'FontSize', 13);
            if col == 1
                set(ax, 'YTickLabel', tickLabels);
                ylabel(ax, sprintf('event index i\n(ticks = bar number)'), 'FontSize', 15);
            else
                set(ax, 'YTickLabel', {});
            end
            if row == 3
                xlabel(ax, sprintf('event index j\n(ticks = bar number)'), 'FontSize', 15);
            end
        end
    end
    cb = colorbar(ax);
    set(cb, 'Position', [0.935 0.07 0.015 0.82]);
    ylabel(cb, 'similarity', 'FontSize', 15);
    annotation(fig, 'textbox', [0.05 0.94 0.9 0.06], 'String', ...
               {sprintf(['BWV 347 N x N event similarity heat maps (N = %d at ' ...
                         '\\Delta = %g QN; \\sigma_{pc} = %g cents fixed).'], ...
                        N, GRID_STEP_QN, SIGMA_PC), ...
                'Rows: \sigma_{ph} snapshots. Columns: encodings.'}, ...
               'HorizontalAlignment', 'center', 'FontSize', 18, 'EdgeColor', 'none');
    if SAVE_FIGURES
        print(fig, '-dpng', '-r140', fullfile(figDir, 'demo_jmm_1_2_heatmaps.png'));
        print(fig, '-dpdf', fullfile(figDir, 'demo_jmm_1_2_heatmaps.pdf'));
        fprintf('Saved figures/demo_jmm_1_2_heatmaps.png\n');
    end
end

% The demo leaves the toolbox as it found it: the defaults it set at the
% top are restored here.
mptDefaults(prevDefaults);

%% Local functions

function names = localKept(pm)
    %localKept The attributes a density is built on: the pitch content and
    %the voice encoding, not when the chord happens.
    [~, ~, specs] = unpackPreMaet(pm);
    all_ = cellfun(@(s) s.name, specs, 'UniformOutput', false);
    names = all_(~strcmp(all_, 'onset'));
end

function events = localChordEvents(pm, t)
    %localChordEvents The events of the chord sounding at T: one where the
    %chord is bound, one per note where it is not.
    [pAttr, ~, specs] = unpackPreMaet(pm);
    all_ = cellfun(@(s) s.name, specs, 'UniformOutput', false);
    onsets = pAttr{find(strcmp(all_, 'onset'), 1)}(1, :);
    events = find(abs(onsets - t) < 1e-9);
end
