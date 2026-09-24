%% demo_jmm_2_3_spectral.m
% Analysis 2.3 (Online Supplement, Section 8.2): windowed similarity of the
% motif, with spectral augmentation.
%
% A demo of the Music Perception Toolbox reproducing the analysis from
% the JMM article; lightly edited from the article's own script. Data
% come from the jmm package (Acknowledgement read from a MIDI
% transcription you supply); the figures stay on screen unless
% SAVE_FIGURES is set.
%
% Analysis 2.3: windowed similarity of the "A Love Supreme" motif across
% Coltrane's Acknowledgement, with and without spectral augmentation.
%
% Analyses 2.1 and 2.2 recover the motif from the passage. Here it is
% supplied instead, as a query, and slid along the time axis: at each
% position it is compared with the passage in the surrounding window.
% Query and passage are encoded alike, as bound super-events of four
% consecutive notes, with two attributes bound at order 4 --- the
% four-note pitch pattern and the group's four onsets, that is its
% rhythm. Onset time is taken relative, and is then either dropped from
% the comparison, in which case it only places the window on the group's
% first onset, or compared, demanding the motif's rhythm as well.
%
% Four readings of the same cross-correlation:
%
%   A1, A2  relative pitch, time dropped: a transposition-invariant
%           similarity against time. Every literal statement scores 1, in
%           whatever key it is played.
%   B1, B2  absolute pitch, time dropped: a pitch-offset by time map,
%           each statement resolving at the offset of its transposition.
%
% The fundamental readings (A1, B1) carry one pitch per note. The
% spectral readings (A2, B2) replace each pitch by its twelve harmonic
% partials --- the hth at p + 12 log2 h semitones, weight h^-rho --- an
% inner multiset within the ordered four, which adds graded harmonic
% affinity between related transpositions. addSpectra takes the pre-MAET
% and expands the pitch attribute of every event at once, so the two
% readings differ by one line of the encoding.
%
% A fifth reading, not in the article's figure, compares the onset
% pattern rather than dropping it. The motif's single early statement,
% long before the closing run, states its pitches in an unrelated
% rhythm, so under A1 it scores a full match --- a false positive of a
% pitch-only comparison. Comparing the rhythm as well sends it to nearly
% zero while leaving the closing run alone; this is the same call with
% 'dropWindowAttr', false.
%
% Pre-MAET structure:
%
%     attribute  order   sigma      rel       per
%     ---------  ------  ---------  --------  ---
%     pitch      4       0.15 st    0 or 1    no    fundamental
%     pitch      (1, 4)  0.15 st    (0, 0/1)  no    spectral
%     onset      4       0.125 QN   yes       no    dropped (A1, A2, B1, B2)
%
%     Ordered (exch = 0) at the outer level, the spectral inner multiset
%     unordered. Estimator: windowed similarity, rectangular window of
%     0.6 QN full support, one-sided normalization.
%
% Data: jmm.acknowledgement (the solo, from your own MIDI transcription at
% data/AwakeningSolo.mid). Toolbox: preMaetFromAttrTable, addSpectra,
% bindEvents, windowedSimilarity. Runtime: a few minutes (about four on
% two cores), three of them the two spectral panels; the spectral
% pitch-offset sweep alone is some 80,000 windowed comparisons of
% twelve-partial super-events.

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

% The kernels here are narrow (0.15 semitones), so the article truncates
% them at four standard deviations rather than the toolbox's six: past
% 0.6 semitones the kernel bears on nothing musical, and the spectral
% sweeps are the heaviest calls in these demos.
prevDefaults = mptDefaults('showHints', false, 'truncationSigmas', 4.0);

SIGMA_PITCH   = 0.15;    % semitones (15 cents): the pitch-matching tolerance
SIGMA_TIME    = 0.125;   % QN: the onset-matching tolerance
N_PART        = 12;      % harmonic partials per note (spectral readings)
RHO           = 0.67;    % power-law roll-off (Milne et al. 2015)
WIN           = 0.6;     % full support of the rectangular time window (QN)
Q_ROOT        = 56;      % query root (MIDI); offset 0 reads as this root
ALS_IV        = [0 3 0 5];           % the motif, from its root
MOTIF_ONSETS  = [0 0.5 1.5 2.0];     % its rhythm: (0.5, 1.0, 0.5) QN
CENTRE_STEP   = 0.5;     % QN between window centres; below the window's
                         % support, so every position of the passage is covered
OFFSETS       = -14:0.5:14;          % semitones, for B1 and B2
BEATS_PER_BAR = 4;       % 4/4 throughout

C_FUND = [0.122 0.306 0.722];
C_SPEC = [0.761 0.314 0.031];

% --- the passage, the query, and the window centres -------------------------
notes = jmm.acknowledgement();
onset = notes.onsetBeats;
centres = min(onset):CENTRE_STEP:max(onset);
fprintf('%d notes; span %.1f QN; %d window centres\n', ...
        height(notes), max(onset), numel(centres));

% The query is a four-note score carrying the motif's own rhythm, and goes
% through the same steps as the passage below. Where onset time is dropped
% from the comparison that rhythm only fixes the window's placement on the
% query's first onset; where it is compared it is the rhythm the passage
% must match.
queryNotes = table(Q_ROOT + ALS_IV(:), MOTIF_ONSETS(:), ...
                   'VariableNames', {'pitch', 'onsetBeats'});

% One conversion each, pitch and onset, one event per note.
ATTRIBUTES = { ...
    struct('column', 'pitch', 'name', 'pitch', 'sigma', SIGMA_PITCH), ...
    struct('column', 'onset', 'name', 'onset', 'sigma', SIGMA_TIME)};
passage = preMaetFromAttrTable(notes, 'attributes', ATTRIBUTES, ...
    'time', 'beats', 'chords', 'separate', 'weights', 'ones');
query = preMaetFromAttrTable(queryNotes, 'attributes', ATTRIBUTES, ...
    'time', 'beats', 'chords', 'separate', 'weights', 'ones');

% The spectral reading replaces each pitch by its N_PART harmonic partials,
% an inner multiset within the note. This one call is the whole difference
% between the fundamental panels and the spectral ones.
passageSpectral = addSpectra(passage, 'harmonic', N_PART, 'powerlaw', RHO, ...
                             'attribute', 'pitch', 'units', 12);
querySpectral = addSpectra(query, 'harmonic', N_PART, 'powerlaw', RHO, ...
                           'attribute', 'pitch', 'units', 12);

% Four consecutive notes bound into one super-event, 'step' = 1 advancing
% the group by one note at a time. Both attributes are bound at order 4:
% the four-note pitch pattern, and the group's four onsets, that is its
% rhythm. The relative panels take both attributes relative, so any
% transposition of the motif matches; the absolute panels take pitch
% absolute, so that each statement resolves at the offset of its own
% transposition, and leave onset relative.
RELATIVE       = [true true];    % pitch relative, onset relative
ABSOLUTE_PITCH = [false true];   % pitch absolute, onset relative

ctxRelFund = bindEvents(passage, [4 4], 'step', 1, 'relOuter', RELATIVE);
ctxRelSpec = bindEvents(passageSpectral, [4 4], 'step', 1, ...
                        'relOuter', RELATIVE);
ctxAbsFund = bindEvents(passage, [4 4], 'step', 1, 'relOuter', ABSOLUTE_PITCH);
ctxAbsSpec = bindEvents(passageSpectral, [4 4], 'step', 1, ...
                        'relOuter', ABSOLUTE_PITCH);
qryRelFund = bindEvents(query, [4 4], 'step', 1, 'relOuter', RELATIVE);
qryRelSpec = bindEvents(querySpectral, [4 4], 'step', 1, 'relOuter', RELATIVE);
qryAbsFund = bindEvents(query, [4 4], 'step', 1, 'relOuter', ABSOLUTE_PITCH);
qryAbsSpec = bindEvents(querySpectral, [4 4], 'step', 1, ...
                        'relOuter', ABSOLUTE_PITCH);

showPreMaet(qryRelFund, 'maxEvents', 1);
showPreMaet(qryRelSpec, 'maxEvents', 1, 'decimals', 2);

% --- A1 and A2: transposition-invariant similarity against time ------------
% The swept axis is time, attribute 2: a rectangular window of full support
% WIN slides over the passage. Onset time is dropped from the comparison
% ('dropWindowAttr', true), so it only places the window, on the group's
% first onset ('locate', 'start'), which lands each peak on the statement's
% onset. Pitch is then the sole compared attribute.
fprintf('computing A1 (fundamental, relative) ...\n');
A1 = windowedSimilarity(ctxRelFund, qryRelFund, centres, ...
    'contextWindow', {'rect', WIN}, 'windowAttr', 2, ...
    'dropWindowAttr', true, 'locate', 'start', ...
    'normalize', 'oneSidedDenom');
fprintf('computing A2 (spectral, relative) ...\n');
A2 = windowedSimilarity(ctxRelSpec, qryRelSpec, centres, ...
    'contextWindow', {'rect', WIN}, 'windowAttr', 2, ...
    'dropWindowAttr', true, 'locate', 'start', ...
    'normalize', 'oneSidedDenom');

% --- B1 and B2: pitch offset by time ---------------------------------------
% One multi-axis sweep slides the query over both attributes at once: pitch
% (axis 1) over the transposition offsets, compared ('drop' false); time
% (axis 2) over the window centres, carrying the window and dropped, exactly
% as in A1 and A2. The pitch positions start from the query's own pitch
% centroid, so that offset 0 reads as the untransposed query.
[qpFund, ~, ~] = unpackPreMaet(qryAbsFund);
[qpSpec, ~, ~] = unpackPreMaet(qryAbsSpec);
qPitchFund = mean(qpFund{1}(:), 'omitnan');
qPitchSpec = mean(qpSpec{1}(:), 'omitnan');
fprintf('computing B1 (fundamental, absolute) ...\n');
B1 = windowedSimilarity(ctxAbsFund, qryAbsFund, [], ...
    'sweep', {1, qPitchFund + OFFSETS; 2, centres}, ...
    'drop', {1, false; 2, true}, ...
    'contextWindow', {2, struct('shape', 'rect', 'width', WIN)}, ...
    'locate', {2, 'start'}, 'normalize', 'oneSidedDenom');
fprintf('computing B2 (spectral, absolute) ...\n');
B2 = windowedSimilarity(ctxAbsSpec, qryAbsSpec, [], ...
    'sweep', {1, qPitchSpec + OFFSETS; 2, centres}, ...
    'drop', {1, false; 2, true}, ...
    'contextWindow', {2, struct('shape', 'rect', 'width', WIN)}, ...
    'locate', {2, 'start'}, 'normalize', 'oneSidedDenom');

panelTags = {'A1', 'A2', 'B1', 'B2'};
panels = {A1, A2, B1, B2};
for k = 1:4
    v = panels{k}(:);
    fprintf('%s: max %.3f; cells above 0.99: %d; above 0.005: %d\n', ...
            panelTags{k}, max(v), sum(v > 0.99), sum(v > 0.005));
end
rho12 = corrcoef(A1(:), A2(:));
fprintf('A1 vs A2 correlation: %.4f; max|A2 - A1| = %.3f\n', ...
        rho12(1, 2), max(abs(A2 - A1)));

% --- the rhythm-aware reading -----------------------------------------------
% A1's call again, with the onset attribute compared rather than dropped
% ('dropWindowAttr', false): a match must now reproduce the motif's rhythm,
% which the query carries, as well as its pitch pattern.
Aj = windowedSimilarity(ctxRelFund, qryRelFund, centres, ...
    'contextWindow', {'rect', WIN}, 'windowAttr', 2, ...
    'dropWindowAttr', false, 'locate', 'start', ...
    'normalize', 'oneSidedDenom');
early = centres < 250;
[bestEarly, iEarly] = max(A1(early));
earlyCentres = centres(early);
fprintf(['\nthe early statement, bar %.0f: pitch-only match %.3f, match ' ...
         'with the rhythm compared %.3f\n'], ...
        earlyCentres(iEarly) / BEATS_PER_BAR, bestEarly, max(Aj(early)));
fprintf(['whole passage: %d unit matches on pitch alone, %d with the ' ...
         'rhythm compared\n'], sum(A1 > 0.99), sum(Aj > 0.99));

% --- figure -----------------------------------------------------------------
bars = centres / BEATS_PER_BAR;
fig = figure('Position', [50 50 1300 620], 'Color', 'w');
axA1 = axes('Parent', fig, 'Position', [0.09 0.62 0.38 0.29]);
axA2 = axes('Parent', fig, 'Position', [0.55 0.62 0.38 0.29]);
axB1 = axes('Parent', fig, 'Position', [0.09 0.10 0.38 0.42]);
axB2 = axes('Parent', fig, 'Position', [0.55 0.10 0.38 0.42]);

yTop = max(1.05, 1.05 * max([A1, A2]));
localStems(axA1, bars, A1, C_FUND, 'A1  fundamental, relative', yTop);
localStems(axA2, bars, A2, C_SPEC, 'A2  spectral, relative', yTop);
ylabel(axA1, 'similarity', 'FontSize', 13);

% Each statement is a single narrow cell against a several-hundred-bar
% axis, so the absolute panels are drawn as a scatter of the non-zero
% cells; a surface would render them sub-pixel and invisible. The colour
% scale is square-rooted (gamma 0.5) to lift the low spectral matches.
vmaxB = max(max(B1(:)), max(B2(:)));
localScatter(axB1, bars, OFFSETS, B1, vmaxB, 'B1  fundamental, absolute');
localScatter(axB2, bars, OFFSETS, B2, vmaxB, 'B2  spectral, absolute');
ylabel(axB1, 'pitch offset from query root (st)', 'FontSize', 13);
cb = colorbar(axB2);
ticks = [0 0.1 0.25 0.5 0.75 1] * vmaxB;
set(cb, 'Ticks', sqrt(ticks), 'TickLabels', ...
    arrayfun(@(v) sprintf('%.2f', v), ticks, 'UniformOutput', false));
annotation(fig, 'textbox', [0.05 0.955 0.9 0.04], 'String', ...
    sprintf(['Coltrane, Acknowledgement: windowed similarity of the ' ...
             'A Love Supreme motif (rho = %g, per = 0)'], RHO), ...
    'HorizontalAlignment', 'center', 'EdgeColor', 'none', 'FontSize', 14);

figDir = fullfile(thisDir, 'figures');
if SAVE_FIGURES
    if ~exist(figDir, 'dir'), mkdir(figDir); end
    print(fig, '-dpng', '-r160', fullfile(figDir, 'demo_jmm_2_3_spectral.png'));
    fprintf('Saved figures/demo_jmm_2_3_spectral.png\n');
end

% The demo leaves the toolbox as it found it: the defaults it set at the
% top are restored here.
mptDefaults(prevDefaults);


function localStems(ax, bars, values, colour, titleText, yTop)
%LOCALSTEMS  One panel of the relative row: similarity against time.
    hold(ax, 'on');
    plot(ax, [bars; bars], [zeros(size(values)); values], ...
         'Color', colour, 'LineWidth', 0.8);
    xlim(ax, [bars(1), bars(end)]);
    ylim(ax, [0, yTop]);
    set(ax, 'FontSize', 12, 'Box', 'off', 'XTickLabel', []);
    title(ax, titleText, 'FontSize', 13);
end

function localScatter(ax, bars, offsets, M, vmax, titleText)
%LOCALSCATTER  One panel of the absolute row: the non-zero cells only.
    [TG, OG] = meshgrid(bars, offsets);
    nz = M > 0.005;
    set(ax, 'Color', 'k');
    hold(ax, 'on');
    scatter(ax, TG(nz), OG(nz), 11, sqrt(M(nz)), 'filled', 'Marker', 's');
    colormap(ax, jmm.colourMap('magma'));
    caxis(ax, [0, sqrt(vmax)]);
    xlim(ax, [bars(1), bars(end)]);
    ylim(ax, [offsets(1), offsets(end)]);
    set(ax, 'YTick', -12:3:12, 'FontSize', 12, 'Box', 'off');
    xlabel(ax, 'bar', 'FontSize', 13);
    title(ax, titleText, 'FontSize', 13);
end
