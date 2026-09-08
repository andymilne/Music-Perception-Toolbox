%% demo_jmm_1_1_entropy.m
% Analysis 1.1: windowed pitch entropy across BWV 347.
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
% How it is computed. Every grid-point chord is spectrally augmented
% (addSpectra: twelve harmonics with 1/n roll-off), so the pitch
% attribute carries 48 partials per event. The chorale is then a two-
% attribute pre-MAET (pitch, time). windowedEntropy sweeps a
% window along the time attribute: at each centre the events are
% reweighted by the window (weightEvents under the hood, the window
% factor multiplied into the pitch weights), the time axis is dropped, and
% the differential entropy of the remaining pitch density is returned
% ('method', 'differential': adaptive grid with Richardson extrapolation,
% in nats). Two windows are compared: a tight rectangle of one sixteenth
% note (one event per window, so the profile is the per-event entropy) and
% a Gaussian of one quarter note.
%
% Data: jmm.bwv347Grid (the score sampled on the sixteenth-note grid,
% repeats expanded). Toolbox: addSpectra, windowedEntropy. Runtime: a few
% minutes (the differential estimator refines its grid at every centre).
% A figure is written to figures/.
%
% The Python mirror is demos/jmm/demo_jmm_1_1_entropy.py.

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
addpath(fullfile(thisDir, '..', '..'));
mptDefaults('showHints', false, ...
            'truncationSigmas', 3.0, ...        % truncate Gaussian tails at 3 sigma
            'kernelPrecision', 'single');       % 32-bit kernel arithmetic

% ---------------------------------------------------------------------------
% Parameters
% ---------------------------------------------------------------------------
SIGMA_PITCH = 10.0;          % cents
H_PARTIALS = 12;
ROLLOFF = 1.0;
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
% Load chorale, spectrally expand partials
% ---------------------------------------------------------------------------
fprintf('Loading BWV 347 and expanding partials...\n');
[times, pitchesSatb, ~] = jmm.bwv347Grid();
N = numel(times);
tEnd = times(end) + 0.25;
pitchesCents = transformAttributes(pitchesSatb, [], {'midi', 'cents'});           % (N, 4)
K = 4 * H_PARTIALS;                           % 48 partials per event

% addSpectra operates on one weighted multiset (one event) at a time, so
% the expansion runs as a per-event loop.
pPartials = zeros(N, K);
wPartials = zeros(N, K);
for n = 1:N
    [pAug, wAug] = addSpectra(pitchesCents(n, :), [], SPECTRUM{:});
    pPartials(n, :) = pAug(:).';
    wPartials(n, :) = wAug(:).';
end

% Pre-MAET inputs: 2 attributes (pitch K=48 partials, time K=1 events).
% Pitch is attribute 1, time is attribute 2.
pAttrPre = {pPartials.', times};
wPre = {wPartials.', ones(1, N)};

showPreMaet(pAttrPre, wPre, [], 'names', {'pitch', 'time'}, ...
    'sigma', [SIGMA_PITCH, 1.0], 'isPer', [false false], ...
    'maxEvents', 4, 'maxElements', 4, 'decimals', 2);

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
        pAttrPre, wPre, ...
        [SIGMA_PITCH, 1.0], [1, 1], ...
        [false, false], [false, false], [0.0, 0.0], ...
        sweepCentres, ...
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
        fprintf('  %s: range [%.4f, %.4f] nats\n', WINDOWS(wi).label, ...
                min(finite), max(finite));
    end
end

% Checkpoint: save H so the figure can be rebuilt without re-computing.
figDir = fullfile(thisDir, 'figures');
if ~exist(figDir, 'dir'), mkdir(figDir); end
H_rect = H(1, :); H_gauss = H(2, :); %#ok<NASGU>
window_labels = {WINDOWS.label}; %#ok<NASGU>
save(fullfile(figDir, 'demo_jmm_1_1_H.mat'), 'times', 'H_rect', 'H_gauss', ...
     'window_labels');

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
    ylabel(ax, sprintf('%s\n\ndifferential entropy (nats)', window.label), 'FontSize', 15);
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
                    'harmonic \\times %d with 1/n rolloff)'], SIGMA_PITCH, H_PARTIALS), ...
           'HorizontalAlignment', 'center', 'FontSize', 18, 'EdgeColor', 'none');
print(fig, '-dpng', '-r140', fullfile(figDir, 'demo_jmm_1_1_entropy.png'));
fprintf('Saved figures/demo_jmm_1_1_entropy.png.\n');
