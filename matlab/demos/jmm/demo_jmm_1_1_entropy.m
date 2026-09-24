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
% by metric class, the profile also shows the on-beat / off-beat contrast
% that the Online Supplement tests across a corpus of chorales.
%
% How it is computed. The chorale is gridded (gridAttrTable) and converted
% to a two-attribute pre-MAET (pitch, time) in one call
% (preMaetFromAttrTable): each grid point is an event holding its chord as
% an unordered pitch multiset in cents, alongside the point's own time.
% Every chord is then spectrally augmented (addSpectra on that attribute:
% twelve harmonics, partial h weighted h^-0.67), so the pitch attribute
% carries 48 partials per event. windowedEntropy then sweeps a
% window along the time attribute: at each centre the events are
% reweighted by the window (weightEvents under the hood, the window
% factor multiplied into the pitch weights), the time axis is dropped, and
% the differential entropy of the remaining pitch density is returned
% ('method', 'differential': adaptive grid with Richardson extrapolation,
% in bits). Two windows are compared: a tight rectangle of one sixteenth
% note (one event per window, so the profile is the per-event entropy) and
% a Gaussian of one quarter note.
%
% Data: jmm.bwv347Notes (the bundled MusicXML read with readScore,
% repeats expanded). Toolbox: gridAttrTable, preMaetFromAttrTable,
% addSpectra, windowedEntropy. Runtime: under a minute (the differential
% estimator refines its grid at every centre).
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
SIGMA_PITCH = 10.0;          % cents
H_PARTIALS = 12;
ROLLOFF = 0.67;            % partial h weighted h^-0.67 (Milne et al. 2015)
SPECTRUM = {'harmonic', H_PARTIALS, 'powerlaw', ROLLOFF};

% Window specifications for weightEvents. Each window is specified through
% one of two interchangeable parameters: 'sd' (the window's standard
% deviation) or 'width' (the full support of the rectangle at shape = 1).
% Across the shape family the SD is held constant regardless of which
% parameter the caller supplies.
WINDOWS = struct( ...
    'kind',  {'width', 'sd'}, ...
    'value', {0.25, 1.0}, ...                   % rect: full support 0.25 QN; Gaussian: sd = 1 QN
    'shape', {1.0, 0.0}, ...
    'label', {'Rect, support 0.25 QN', 'Gaussian sigma = 1 QN'});
nWindows = numel(WINDOWS);

% ---------------------------------------------------------------------------
% Load chorale, spectral enrichment
% ---------------------------------------------------------------------------
fprintf('Loading BWV 347 and expanding partials...\n');
% The chorale on the sixteenth-note grid, converted to a two-attribute
% pre-MAET: the grid point's chord as the pitch attribute, read in cents as
% an unordered multiset, and the grid point's own time as the axis the
% window will slide along. Each chord's four pitches then take their
% partials, which multiplies the pitch attribute's K by twelve and leaves
% the events alone.
g = gridAttrTable(jmm.bwv347Notes(), jmm.gridStepQn());
pm = preMaetFromAttrTable(g, 'attributes', { ...
        struct('column', 'pitch', 'sigma', SIGMA_PITCH, 'r', 1, ...
               'exch', true), ...
        struct('column', 'onset', 'name', 'time', 'sigma', 1.0)}, ...
        'time', 'beats', 'pitch', 'cents', 'weights', 'ones');
pm = addSpectra(pm, SPECTRUM{:}, 'attribute', 'pitch');

pAttrPre = unpackPreMaet(pm);
times = pAttrPre{2};
N = numel(times);
tEnd = times(end) + jmm.gridStepQn();

showPreMaet(pm, 'maxEvents', 4, 'maxElements', 4, 'decimals', 2);

% ---------------------------------------------------------------------------
% Compute differential entropy at each event time
% ---------------------------------------------------------------------------
sweepCentres = times;
nSweep = numel(sweepCentres);

fprintf('Computing windowed differential entropy at %d sweep centres over %d window(s)...\n', ...
        nSweep, nWindows);

H = zeros(nWindows, nSweep);

% Each window is a single windowedEntropy sweep over all centres. The
% time axis (attribute 2) supplies the window and is dropped from the
% entropy density ('dropWindowAttr', true; for an r = 1 absolute axis,
% dropping the axis equals marginalizing it out), leaving the pitch
% density whose differential entropy is returned. The window width is the
% rectangular full support; a Gaussian window given by its standard
% deviation s maps to the variance-matched width 2*sqrt(3)*s. (The
% placeholder time sigma is unused: the time axis is dropped before any
% density is built.)
RT3 = 2.0 * sqrt(3.0);
t0 = tic;
for wi = 1:nWindows
    window = WINDOWS(wi);
    fprintf('Window %d/%d: %s\n', wi, nWindows, window.label);
    if strcmp(window.kind, 'width'), width = window.value;
    else,                             width = window.value * RT3; end
    H(wi, :) = windowedEntropy( ...
        pm, sweepCentres, ...
        'contextWindow', {window.shape, width}, ...
        'method', 'differential', ...
        'windowAttr', 2, 'dropWindowAttr', true, ...
        'verbose', false);
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
classLabels = {sprintf('down\n(b1)'), sprintf('med\n(b3)'), ...
               sprintf('weak\n(b2,4)'), sprintf('off-\nbeat')};
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

    % Bar panel: metric-class means +/- SE
    means = zeros(1, 4); sems = zeros(1, 4);
    for cls = 1:4
        vals = HRow(classesPerEvent == cls);
        n = numel(vals);
        means(cls) = mean(vals);
        if n > 1, sems(cls) = std(vals) / sqrt(n); end
    end
    for cls = 1:4
        bar(axBar, cls, means(cls), 'FaceColor', classColours(cls, :), ...
            'EdgeColor', 'k', 'LineWidth', 0.7);
    end
    for cls = 1:4                               % error bars with caps
        plot(axBar, [cls cls], means(cls) + [-1 1] * sems(cls), 'k', 'LineWidth', 0.8);
        plot(axBar, cls + [-0.12 0.12], [1 1] * (means(cls) + sems(cls)), 'k', 'LineWidth', 0.8);
        plot(axBar, cls + [-0.12 0.12], [1 1] * (means(cls) - sems(cls)), 'k', 'LineWidth', 0.8);
    end
    set(axBar, 'XTick', 1:4, 'XTickLabel', classLabels, 'FontSize', 13, 'Box', 'off');
    xlim(axBar, [0.4, 4.6]);
    ylabel(axBar, 'mean ± SE', 'FontSize', 15);
    span = max(means) - min(means);
    pad = max(sems) * 2 + span * 0.1 + 0.001;
    ylim(axBar, [min(means) - pad, max(means) + pad]);
    grid(axBar, 'on'); set(axBar, 'XGrid', 'off');
    if wi == 1
        title(axBar, 'By metric class', 'FontSize', 17);
    end
end

annotation(fig, 'textbox', [0.05 0.93 0.9 0.06], 'String', ...
           sprintf(['BWV 347 windowed differential pitch entropy (\\sigma_{pitch} = %.0f cents, ' ...
                    'harmonic \\times %d, weight h^{-%.2f})'], SIGMA_PITCH, H_PARTIALS, ROLLOFF), ...
           'HorizontalAlignment', 'center', 'FontSize', 18, 'EdgeColor', 'none');
if SAVE_FIGURES
    print(fig, '-dpng', '-r140', fullfile(figDir, 'demo_jmm_1_1_entropy.png'));
    fprintf('Saved figures/demo_jmm_1_1_entropy.png.\n');
end

% The demo leaves the toolbox as it found it: the defaults it set at the
% top are restored here.
mptDefaults(prevDefaults);
