%% demo_jmm_1_1_entropy.m
% Analysis 1.1 (JMM article, Section 4.1.1): windowed pitch entropy across
% BWV 347.
%
% Reproduces Analysis 1.1 of the JMM article (Section 4.1.1): the temporal
% evolution of pitch entropy across Bach's chorale BWV 347, read through a
% window that slides along the piece.
%
% What the analysis asks. Where in the chorale is the sounding pitch
% content most and least concentrated? A cadence resolves onto a triad
% whose partials cohere, so the spectral pitch density there is peaked and
% its differential entropy low; passing sonorities between cadences spread
% the density and raise the entropy. Read at every grid point and grouped
% by metric class, the profile bears on the article's prediction that
% spectral entropy (a model for dissonance) is on average higher at times
% of lower metrical weight, which a permutation test then tests directly.
%
% How it is computed. The chorale is gridded (gridAttrTable) and converted
% to a two-attribute pre-MAET (pitch, time) in one call
% (preMaetFromAttrTable): each grid point is an event holding its chord as
% an unordered pitch multiset in MIDI semitones (sigma = 0.1), alongside
% the point's own time. Every chord is then spectrally enriched (addSpectra
% on that attribute: twelve harmonics, partial h at p + 12 log2 h
% semitones weighted h^-0.67), so the pitch attribute carries 48 partials
% per event. sweptEntropy then sweeps a window along the time
% attribute: at each sweep value the events are reweighted by the window
% (event weighting, as weightEvents does, the window factor multiplied
% into the pitch weights), the time attribute is dropped, and the
% differential entropy of the remaining pitch density is returned
% ('method', 'differential': adaptive grid with Richardson extrapolation,
% in bits, with pitch in semitones). Two windows are compared: a tight
% rectangle of one sixteenth note (one event per window, so the profile is
% the per-event entropy) and a Gaussian of standard deviation one quarter
% note. The metric-class panels give each class's mean over its grid
% points; their error bars are cluster-robust standard errors with the
% sonority as the cluster, so that the grid points of a chord held across
% several of them are not treated as independent observations.
%
% The test. The prediction is tested on the rectangular-window profile,
% one observation per sonority: its entropy and the metric class of its
% onset, ranked downbeat > medium > weak > offbeat. The repeat of bars
% 1-4 is omitted, so that it does not count twice. The statistic is
% Kendall's tau-b between rank and entropy, which the prediction makes
% negative; its one-sided p-value comes from permuting the entropies
% among the sonorities of each phrase (the stretches closed by the
% score's fermatas), which leaves any difference between phrases intact.
%
% Data: jmm.bwv347Notes (the bundled MusicXML read with readScore,
% repeats expanded). Toolbox: gridAttrTable, preMaetFromAttrTable,
% addSpectra, sweptEntropy. Runtime: under a minute (the differential
% estimator refines its grid at every sweep value).
% The figures stay on screen unless SAVE_FIGURES is set.

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

% Set true to write the figures and the checkpoint data to a
% figures/ folder beside this script; false leaves them on screen only.
SAVE_FIGURES = false;

prevDefaults = mptDefaults('showHints', false);

% ---------------------------------------------------------------------------
% Parameters
% ---------------------------------------------------------------------------
SIGMA_PITCH = 0.1;           % semitones (10 cents)
H_PARTIALS = 12;
ROLLOFF = 0.67;            % partial h weighted h^-0.67 (Milne et al. 2015)
SPECTRUM = {'harmonic', H_PARTIALS, 'powerlaw', ROLLOFF};

% Window specifications, as weightEvents and sweptEntropy take them.
% Each window is specified through one of two interchangeable parameters:
% 'sd' (the window's standard deviation) or 'width' (the full support of
% the rectangle at shape = 1). Across the shape family the standard
% deviation is held constant whichever parameter is supplied.
WINDOWS = struct( ...
    'kind',  {'width', 'sd'}, ...
    'value', {0.25, 1.0}, ... % rect: full support 0.25 QN; Gaussian: sd = 1 QN
    'shape', {1.0, 0.0}, ...
    'label', {'Rect, support 0.25 QN', 'Gaussian sigma = 1 QN'});
nWindows = numel(WINDOWS);

% ---------------------------------------------------------------------------
% Load chorale, spectral enrichment
% ---------------------------------------------------------------------------
fprintf('Loading BWV 347 and enriching it spectrally...\n');
% The chorale on the sixteenth-note grid, converted to a two-attribute
% pre-MAET: the grid point's chord as the pitch attribute, read in MIDI
% semitones as an unordered multiset, and the grid point's own time as the
% attribute the window will slide along. Each chord's four pitches then
% take their partials ('units', 12: twelve units to the octave), which
% multiplies the pitch attribute's K by twelve and leaves the events alone.
g = gridAttrTable(jmm.bwv347Notes(), jmm.gridStepQn());
pm = preMaetFromAttrTable(g, 'specs', { ...
        struct('column', 'pitch', 'sigma', SIGMA_PITCH, 'r', 1, ...
               'exch', true), ...
        struct('column', 'onset', 'name', 'time', 'sigma', 1.0)}, ...
        'time', 'beats', 'pitch', 'midi', 'weights', 'ones');
% Each grid point's chord (its pitches sorted), read before enrichment: a
% run of consecutive grid points holding the same chord is one sonority.
pAttrPre = unpackPreMaet(pm);
chords = sort(pAttrPre{1}, 1);
newChord = any(chords(:, 2:end) ~= chords(:, 1:end - 1), 1);
sonority = [0, cumsum(newChord)];
pm = addSpectra(pm, SPECTRUM{:}, 'attribute', 'pitch', 'units', 12);

times = pAttrPre{2};
N = numel(times);
tEnd = times(end) + jmm.gridStepQn();

showPreMaet(pm, 'maxEvents', 4, 'maxElements', 4, 'decimals', 2);

% ---------------------------------------------------------------------------
% Compute differential entropy at each event time
% ---------------------------------------------------------------------------

fprintf('Computing windowed differential entropy at %d sweep values over %d window(s)...\n', ...
        N, nWindows);

H = zeros(nWindows, N);

% Each window is a single sweptEntropy sweep over all sweep values.
% The time attribute (attribute 2) is the window attribute: it supplies
% the window and is dropped from the entropy density ('drop', 2; for
% an r = 1 absolute attribute, dropping it equals marginalizing it out),
% leaving the pitch density whose differential entropy is returned. The
% window is given as the specification above, by its width or by its
% standard deviation. (The placeholder time sigma is unused: the time
% attribute is dropped before any density is built.)
t0 = tic;
for wi = 1:nWindows
    window = WINDOWS(wi);
    fprintf('Window %d/%d: %s\n', wi, nWindows, window.label);
    H(wi, :) = sweptEntropy( ...
        pm, 'sweep', {2, times}, 'drop', 2, ...
        'window', {2, struct('shape', window.shape, ...
                             window.kind, window.value)}, ...
        'method', 'differential', 'verbose', false);
    fprintf('  done (%.0fs elapsed)\n', toc(t0));
end

for wi = 1:nWindows
    arr = H(wi, :);
    finite = arr(isfinite(arr));
    if ~isempty(finite)
        fprintf('  %s: range [%.4f, %.4f] bits\n', WINDOWS(wi).label, ...
                min(finite), max(finite));
    end
end

% Checkpoint: save H so the figure can be rebuilt without re-computing.
figDir = fullfile(thisDir, 'figures');
if SAVE_FIGURES && ~exist(figDir, 'dir'), mkdir(figDir); end
if SAVE_FIGURES
    H_rect = H(1, :); H_gauss = H(2, :); %#ok<NASGU>
    window_labels = {WINDOWS.label}; %#ok<NASGU>
    save(fullfile(figDir, 'demo_jmm_1_1_H.mat'), 'times', 'H_rect', 'H_gauss', ...
         'window_labels');
end

% ---------------------------------------------------------------------------
% Metric-class classification
% ---------------------------------------------------------------------------
classNames = {'downbeat', 'medium', 'weak', 'offbeat'};
% Two-line metric-class labels, drawn as text below the bar axis (a tick
% label does not reliably honour a line break).
classLabels = {{'down', '(b1)'}, {'med', '(b3)'}, {'weak', '(b2,4)'}, ...
               {'off-', 'beat'}};
classColours = [0.122 0.306 0.722; 0.353 0.541 0.796; 0.659 0.718 0.839; 0.8 0.8 0.8];
classesPerEvent = zeros(1, N);       % index into classNames
for n = 1:N
    t = times(n);
    if abs(t - round(t)) > 1e-6
        classesPerEvent(n) = 4;                         % offbeat
    else
        beatInBar = mod(round(t) - 1, 4) + 1;
        switch beatInBar
            case 1, classesPerEvent(n) = 1;             % downbeat
            case 3, classesPerEvent(n) = 2;             % medium
            otherwise, classesPerEvent(n) = 3;          % weak
        end
    end
end

% Mean over the grid points of each metric class, with its cluster-robust
% standard error (CR1), the sonority being the cluster: the residuals of
% the grid points holding one sonority are summed before squaring, so that
% a chord held across grid points is not counted as independent
% observations. With one grid point per sonority this is the ordinary
% standard error of the mean.
classMeans = zeros(nWindows, 4); classSems = zeros(nWindows, 4);
classCounts = zeros(1, 4);
fprintf(['Mean differential entropy (bits) by metric class, ' ...
         '+/- cluster-robust SE (number of sonorities):\n']);
for wi = 1:nWindows
    fprintf('  %s\n', WINDOWS(wi).label);
    for cls = 1:4
        mask = classesPerEvent == cls;
        x = H(wi, mask).';
        [~, ~, grp] = unique(sonority(mask));
        nClusters = max(grp);
        classMeans(wi, cls) = mean(x);
        if nClusters > 1
            score = accumarray(grp(:), x - mean(x));
            classSems(wi, cls) = sqrt(nClusters / (nClusters - 1) ...
                * sum(score .^ 2)) / numel(x);
        end
        classCounts(cls) = nClusters;
        fprintf('    %-9s %.4f +/- %.4f (%d)\n', classNames{cls}, ...
                classMeans(wi, cls), classSems(wi, cls), classCounts(cls));
    end
end

% ---------------------------------------------------------------------------
% Test of the prediction
% ---------------------------------------------------------------------------
% One observation per sonority, from the rectangular-window profile: its
% entropy (the same at every grid point it holds) and the metric class of
% its onset, ranked downbeat 4 > medium 3 > weak 2 > offbeat 1. The grid
% points from 16 to 32 QN repeat those from 0 to 16 QN (bars 1-4 and
% their upbeat), so they are omitted rather than counted twice.
REPEAT_SPAN = [16, 32];         % QN: the expanded repeat of bars 1-4
N_PERM = 20000;
CLASS_RANK = [4, 3, 2, 1];      % downbeat, medium, weak, offbeat

% Phrases: each closes with the last grid point of a fermata.
fermataPts = false(1, N);
fermataPts(unique(g.gridIndex(logical(g.fermata)))) = true;
phrase = [0, cumsum(fermataPts(1:end - 1) & ~fermataPts(2:end))];

first = [true, sonority(2:end) ~= sonority(1:end - 1)];
inRepeat = times >= REPEAT_SPAN(1) & times < REPEAT_SPAN(2);
obs = find(first & ~inRepeat);
rankObs = CLASS_RANK(classesPerEvent(obs));
hObs = H(1, obs);
phraseObs = phrase(obs);

% Kendall's tau-b from the pairwise sign matrices. Permuting the
% entropies changes only the numerator: the tie counts, and so the
% denominator, stay fixed.
nObs = numel(obs);
sRank = sign(rankObs(:) - rankObs(:).');
sH = sign(hObs(:) - hObs(:).');
nPairs = nObs * (nObs - 1) / 2;
untiedRank = nPairs - (nnz(sRank == 0) - nObs) / 2;
untiedH = nPairs - (nnz(sH == 0) - nObs) / 2;
concord = sum(sRank .* sH, 'all') / 2;
tauB = concord / sqrt(untiedRank * untiedH);

% One-sided p-value: entropies permuted among the sonorities of each
% phrase, so that differences between phrases cannot produce the result.
rs = RandStream('mt19937ar', 'Seed', 347);
phraseIds = unique(phraseObs);
members = arrayfun(@(k) find(phraseObs == k), phraseIds, ...
                   'UniformOutput', false);
nAsLow = 0;
for b = 1:N_PERM
    perm = 1:nObs;
    for m = 1:numel(members)
        idx = members{m};
        perm(idx) = idx(randperm(rs, numel(idx)));
    end
    nAsLow = nAsLow + (sum(sRank .* sH(perm, perm), 'all') / 2 <= concord);
end
pPerm = (1 + nAsLow) / (1 + N_PERM);
fprintf(['Test (rectangular window, %d sonorities, repeat omitted): ' ...
         'Kendall tau-b = %.3f, one-sided permutation p = %.2g ' ...
         '(%d permutations within %d phrases)\n'], ...
        nObs, tauB, pPerm, N_PERM, numel(members));

% ---------------------------------------------------------------------------
% Plot: one row per window
% ---------------------------------------------------------------------------
fig = figure('Position', [50 50 1500 400 * nWindows], 'Color', 'w');

cadenceSpans = {7.0,  8.0,  sprintf('tonic 1\n(E maj)'); ...
                15.0, 16.0, sprintf('tonic 2\n(E maj)'); ...
                23.0, 24.0, sprintf('tonic 1''\n(E maj)'); ...
                31.0, 32.0, sprintf('tonic 2''\n(E maj)'); ...
                45.0, 48.0, sprintf('tonic 3\n(B min)'); ...
                65.0, 68.0, sprintf('tonic 4\n(A maj)')};
barDownbeats = 1:4:floor(tEnd);

% Panel geometry: a wide profile panel and a narrow bar panel per row.
rowH = 0.78 / nWindows;
for wi = 1:nWindows
    window = WINDOWS(wi);
    HRow = H(wi, :);
    % Window-edge band (Gaussian only; tight rect has no useful edge band).
    if window.shape == 0.0, edge = 2 * window.value; else, edge = 0.0; end

    y0 = 0.10 + (nWindows - wi) * rowH;
    ax = axes('Parent', fig, 'Position', [0.07, y0, 0.66, rowH * 0.78]);
    axBar = axes('Parent', fig, 'Position', [0.80, y0, 0.17, rowH * 0.78]);
    hold(ax, 'on'); hold(axBar, 'on');

    yl = [min(HRow(isfinite(HRow))), max(HRow(isfinite(HRow)))];
    yl = yl + [-0.05, 0.05] * diff(yl);
    for t = barDownbeats
        fill(ax, [t - 0.06, t + 0.06, t + 0.06, t - 0.06], [yl(1) yl(1) yl(2) yl(2)], ...
             'k', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
        fill(ax, [t + 1.94, t + 2.06, t + 2.06, t + 1.94], [yl(1) yl(1) yl(2) yl(2)], ...
             'k', 'FaceAlpha', 0.07, 'EdgeColor', 'none');
    end
    if edge > 0.0
        fill(ax, [0, edge, edge, 0], [yl(1) yl(1) yl(2) yl(2)], ...
             [0.6 0.6 0.6], 'FaceAlpha', 0.22, 'EdgeColor', 'none');
        fill(ax, [tEnd - edge, tEnd, tEnd, tEnd - edge], [yl(1) yl(1) yl(2) yl(2)], ...
             [0.6 0.6 0.6], 'FaceAlpha', 0.22, 'EdgeColor', 'none');
    end
    for c = 1:size(cadenceSpans, 1)
        ta = cadenceSpans{c, 1}; tb = cadenceSpans{c, 2};
        fill(ax, [ta, tb, tb, ta], [yl(1) yl(1) yl(2) yl(2)], ...
             [0.761 0.314 0.031], 'FaceAlpha', 0.18, 'EdgeColor', 'none');
    end
    % Step plot for the rectangular per-event case, smooth plot for Gaussian.
    if window.shape == 1.0
        stairs(ax, [times, tEnd], [HRow, HRow(end)], 'Color', [0.122 0.306 0.722], ...
               'LineWidth', 1.2);
    else
        plot(ax, times, HRow, 'Color', [0.122 0.306 0.722], 'LineWidth', 1.4);
    end
    xlim(ax, [0, tEnd]);
    ylim(ax, yl);
    set(ax, 'XTick', 1:8:floor(tEnd), 'FontSize', 13, 'Box', 'off');
    grid(ax, 'on');
    ylabel(ax, sprintf('%s\n\ndifferential entropy (bits)', window.label), 'FontSize', 15);
    if wi == nWindows
        xlabel(ax, 'time (quarter notes)', 'FontSize', 15);
    end
    if wi == 1
        for c = 1:size(cadenceSpans, 1)
            ta = cadenceSpans{c, 1}; tb = cadenceSpans{c, 2};
            text(ax, (ta + tb) / 2, yl(1) + diff(yl) * 0.02, cadenceSpans{c, 3}, ...
                 'Color', [0.761 0.314 0.031], 'FontSize', 12, ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                 'FontWeight', 'bold');
        end
    end

    % Bar panel: metric-class means +/- cluster-robust SE
    means = classMeans(wi, :); sems = classSems(wi, :);
    for cls = 1:4
        bar(axBar, cls, means(cls), 'FaceColor', classColours(cls, :), ...
            'EdgeColor', 'k', 'LineWidth', 0.7);
    end
    for cls = 1:4                               % error bars with caps
        plot(axBar, [cls cls], means(cls) + [-1 1] * sems(cls), 'k', 'LineWidth', 0.8);
        plot(axBar, cls + [-0.12 0.12], [1 1] * (means(cls) + sems(cls)), 'k', 'LineWidth', 0.8);
        plot(axBar, cls + [-0.12 0.12], [1 1] * (means(cls) - sems(cls)), 'k', 'LineWidth', 0.8);
    end
    set(axBar, 'XTick', 1:4, 'XTickLabel', {}, 'FontSize', 13, 'Box', 'off');
    xlim(axBar, [0.4, 4.6]);
    ylabel(axBar, 'mean ± SE', 'FontSize', 15);
    span = max(means) - min(means);
    pad = max(sems) * 2 + span * 0.1 + 0.001;
    ylim(axBar, [min(means) - pad, max(means) + pad]);
    ylBar = ylim(axBar);
    for cls = 1:4
        text(axBar, cls, ylBar(1) - 0.03 * diff(ylBar), classLabels{cls}, ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
             'FontSize', 11);
    end
    grid(axBar, 'on'); set(axBar, 'XGrid', 'off');
    if wi == 1
        title(axBar, 'By metric class', 'FontSize', 17);
    end
end

annotation(fig, 'textbox', [0.05 0.93 0.9 0.06], 'String', ...
           sprintf(['BWV 347 windowed differential pitch entropy (\\sigma_{pitch} = %g semitones, ' ...
                    'harmonic \\times %d, weight h^{-%.2f})'], SIGMA_PITCH, H_PARTIALS, ROLLOFF), ...
           'HorizontalAlignment', 'center', 'FontSize', 18, 'EdgeColor', 'none');
if SAVE_FIGURES
    print(fig, '-dpng', '-r140', fullfile(figDir, 'demo_jmm_1_1_entropy.png'));
    fprintf('Saved figures/demo_jmm_1_1_entropy.png.\n');
end

% The demo leaves the toolbox as it found it: the defaults it set at the
% top are restored here.
mptDefaults(prevDefaults);
