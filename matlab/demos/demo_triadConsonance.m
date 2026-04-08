%% demo_triadConsonance.m
%  Computes and plots consonance-related features for triads
%  [0, interval1, interval2] over a grid of intervals.
%
%  Five measures are available (select which to plot below):
%
%    'tmplMax'   — Template harmonicity hMax (Milne 2013): maximum of
%                  the normalized cross-correlation with a harmonic
%                  template.
%
%    'tmplEnt'   — Template harmonicity hEntropy (Harrison 2020):
%                  entropy of the normalized cross-correlation. Plotted
%                  as -hEntropy so that peaks = consonance.
%
%    'tensor'    — Tensor harmonicity (Milne 2013 / Smit et al. 2019):
%                  density of the relative triad expectation tensor of
%                  a harmonic series, evaluated at the chord's interval
%                  vector.
%
%    'specEnt'   — Spectral entropy (Milne et al. 2017): entropy of the
%                  smoothed composite spectrum. Plotted as -entropy so
%                  that peaks = consonance.
%
%    'rough'     — Sensory roughness (Sethares 1993 / Plomp-Levelt
%                  1965): total pairwise roughness of the chord's
%                  partials. Plotted as -roughness so that
%                  peaks = consonance.
%
%  Each plot has interval1 on the x-axis and interval2 on the y-axis.
%  The plots are symmetric about the diagonal (swapping the two
%  intervals gives the same chord).
%
%  Uses: templateHarmonicity, tensorHarmonicity, spectralEntropy,
%        roughness, addSpectra, buildExpTens, evalExpTens, convertPitch
%  (from the Music Perception Toolbox).

%% === User-adjustable parameters ===

% === Select which measures to plot ===
% Comment out or remove entries to skip them. The subplot grid adapts
% automatically to the number of selected measures.
plotMeasures = {
    'tmplMax'     % Template harmonicity: hMax (Milne 2013)
    'tmplEnt'     % Template harmonicity: -hEntropy (Harrison 2020)
    'tensor'      % Tensor harmonicity (Smit et al. 2019)
    'specEnt'     % -Spectral entropy (Milne et al. 2017)
    'rough'       % -Roughness (Sethares 1993)
};

% Grid
step    = 10;       % grid spacing in cents (smaller = finer but slower)
maxInt  = 2400;     % maximum interval in cents (1200 = one octave)

% Reference pitch for roughness calculation (Hz)
f0 = 261.63;        % middle C (C4)

% Smoothing widths (one per measure that uses sigma)
sigma_tmpl  = 10;   % templateHarmonicity
sigma_tens  = 10;   % tensorHarmonicity
sigma_ent   = 10;   % spectralEntropy

% Spectral parameters (addSpectra arguments) for each measure.
% Each is a cell array passed to addSpectra (after p and w). The
% rolloff type (powerlaw/geometric) and all other spectral parameters
% are explicit here, ensuring consistency across measures.
spec_tmpl  = {'harmonic', 24, 'powerlaw', 1};   % templateHarmonicity
spec_tens  = {'harmonic', 24, 'powerlaw', 1};   % tensorHarmonicity
spec_ent   = {'harmonic', 24, 'powerlaw', 1};   % spectralEntropy
spec_rough = {'harmonic', 24, 'powerlaw', 1};   % roughness

% Tensor harmonicity: spectrum duplication.
% The template spectrum is duplicated so that each partial can fill
% multiple positions in an r-tuple. This allows unisons to contribute to
% the harmonicity. Set to 0 for automatic (= chord cardinality, i.e., 3 for
% triads). Set to 1 to disable duplication.
dup_tens = 0;

% Gamma (power compression) for visualization: displayed = data.^gamma
%   gamma < 1 compresses the dynamic range, revealing structure in
%   lower-valued regions. An interactive slider is provided.
gamma = 1.0;

%% === Determine which measures are selected ===

doTmplMax = ismember('tmplMax', plotMeasures);
doTmplEnt = ismember('tmplEnt', plotMeasures);
doTmpl    = doTmplMax || doTmplEnt;  % both share one function call
doTensor  = ismember('tensor',  plotMeasures);
doSpecEnt = ismember('specEnt', plotMeasures);
doRough   = ismember('rough',   plotMeasures);

%% === Build grid ===

ints  = 0:step:maxInt;
nInts = numel(ints);
[Ga, Gb] = meshgrid(ints, ints);

% Preallocate only the selected measures
if doTmplMax, tmplHarmMax = NaN(nInts, nInts); end
if doTmplEnt, tmplHarmEnt = NaN(nInts, nInts); end
if doTensor,  tensHarm    = NaN(nInts, nInts); end
if doSpecEnt, specEnt     = NaN(nInts, nInts); end
if doRough,   rough       = NaN(nInts, nInts); end

% Reference pitch in absolute cents (for roughness Hz conversion)
refCents = convertPitch(f0, 'hz', 'cents');

%% === Precompute tensor harmonicity template (if selected) ===

if doTensor
    % Resolve auto-duplication: 0 means match chord cardinality (triads = 3)
    if dup_tens == 0
        dup_tens = 3;
    end

    fprintf('Precomputing tensor harmonicity template (r=3, dup=%d, spectrum: %s)...\n', ...
        dup_tens, ...
        strjoin(cellfun(@num2str, spec_tens, 'UniformOutput', false), ', '));
    if dup_tens > 3
        warning('demo_triadConsonance:largeDuplicate', ...
                'dup_tens = %d: computation time grows rapidly. Consider reducing to 3 or fewer.', ...
                dup_tens);
    end
    [tp, tw] = addSpectra(zeros(dup_tens, 1), ones(dup_tens, 1), spec_tens{:});
    T = buildExpTens(tp, tw, sigma_tens, 3, true, false, 1200, 'verbose', false);
    fprintf('  Done (%d ordered triples).\n', T.nJ);
end

%% === Compute features ===
% Exploit symmetry: features are invariant to swapping interval1 and
% interval2, so compute only the upper triangle (j >= i) and mirror.

nTotal = nInts * (nInts + 1) / 2;
nDone  = 0;
t0     = tic;

fprintf('Computing features for %d triads (step = %d cents)...\n', ...
    nTotal, step);

for i = 1:nInts
    for j = i:nInts
        int1 = ints(i);
        int2 = ints(j);

        % --- Tensor harmonicity ---
        if doTensor
            intVec = [int1; int2];
            tensHarm(j, i) = evalExpTens(T, intVec, 'verbose', false);
            tensHarm(i, j) = tensHarm(j, i);
        end

        % --- Template harmonicity (computes both outputs in one call) ---
        if doTmpl
            [hMax, hEnt] = templateHarmonicity([0, int1, int2], [], ...
                sigma_tmpl, 'spectrum', spec_tmpl, ...
                'chordSpectrum', spec_tmpl);
            if doTmplMax
                tmplHarmMax(j, i) = hMax;
                tmplHarmMax(i, j) = hMax;
            end
            if doTmplEnt
                tmplHarmEnt(j, i) = hEnt;
                tmplHarmEnt(i, j) = hEnt;
            end
        end

        % --- Spectral entropy ---
        if doSpecEnt
            specEnt(j, i) = spectralEntropy([0, int1, int2], [], ...
                sigma_ent, 'spectrum', spec_ent);
            specEnt(i, j) = specEnt(j, i);
        end

        % --- Roughness ---
        if doRough
            chordCents = [refCents, refCents + int1, refCents + int2];
            [ep, ew] = addSpectra(chordCents(:), [], spec_rough{:});
            fHz = convertPitch(ep, 'cents', 'hz');
            rough(j, i) = roughness(fHz, ew);
            rough(i, j) = rough(j, i);
        end

        % Progress
        nDone = nDone + 1;
        if mod(nDone, 500) == 0 || nDone == nTotal
            elapsed = toc(t0);
            rate    = nDone / elapsed;
            remain  = (nTotal - nDone) / rate;
            fprintf('  %d / %d triads (%.1f s elapsed, ~%.0f s remaining)\n', ...
                nDone, nTotal, elapsed, remain);
        end
    end
end

fprintf('All features computed in %.1f s.\n', toc(t0));

%% === Assemble selected measures for plotting ===

allData   = {};
allTitles = {};

if doTmplMax
    allData{end+1}   = tmplHarmMax;
    allTitles{end+1} = sprintf('Template harmonicity: hMax (Milne 2013)\n%s, \\sigma=%d', ...
        strjoin(cellfun(@num2str, spec_tmpl, 'UniformOutput', false), ', '), sigma_tmpl);
end
if doTmplEnt
    allData{end+1}   = -tmplHarmEnt;
    allTitles{end+1} = sprintf('Template harmonicity: -hEntropy (Harrison 2020)\n%s, \\sigma=%d', ...
        strjoin(cellfun(@num2str, spec_tmpl, 'UniformOutput', false), ', '), sigma_tmpl);
end
if doTensor
    allData{end+1}   = tensHarm;
    allTitles{end+1} = sprintf('Tensor harmonicity (Smit et al. 2019)\n%s, \\sigma=%d, dup=%d', ...
        strjoin(cellfun(@num2str, spec_tens, 'UniformOutput', false), ', '), sigma_tens, dup_tens);
end
if doSpecEnt
    allData{end+1}   = -specEnt;
    allTitles{end+1} = sprintf('-Spectral entropy (Milne et al. 2017)\n%s, \\sigma=%d', ...
        strjoin(cellfun(@num2str, spec_ent, 'UniformOutput', false), ', '), sigma_ent);
end
if doRough
    allData{end+1}   = -rough;
    allTitles{end+1} = sprintf('-Roughness (Sethares 1993)\n%s, f_0=%.1f Hz', ...
        strjoin(cellfun(@num2str, spec_rough, 'UniformOutput', false), ', '), f0);
end

nPlots = numel(allData);

if nPlots == 0
    fprintf('No measures selected — nothing to plot.\n');
    return;
end

%% === Plots ===

% Determine subplot grid
nCols = min(nPlots, 3);
nRows = ceil(nPlots / nCols);

fig = figure('Name', 'Triad consonance', ...
    'Position', [50, 50, min(400 * nCols + 200, 1400), min(400 * nRows, 1000)]);

hSurfs  = gobjects(nPlots, 1);
hAxes   = gobjects(nPlots, 1);
rawData = cell(nPlots, 1);

for mi = 1:nPlots
    hAxes(mi) = subplot(nRows, nCols, mi);

    data = allData{mi};
    rawData{mi} = data;

    V = applyGamma(data, gamma);

    hSurfs(mi) = surf(Ga, Gb, V, 'EdgeColor', 'none');
    colormap(hAxes(mi), parula);
    colorbar;
    xlabel('Interval 1 (cents)');
    ylabel('Interval 2 (cents)');
    title(allTitles{mi});
    xlim([0 maxInt]);
    ylim([0 maxInt]);
    rangeV = max(V(:)) - min(V(:));
    if rangeV > 0
        daspect([1 1 rangeV / maxInt]);
    end
    set(hAxes(mi), 'Projection', 'orthographic');
    view(0, 90);
end

sgtitle(sprintf('Triad consonance (step = %d cents)', step), ...
    'FontWeight', 'bold');

%% === Vertical sliders (power + cmap shift) and projection toggle ===

% Store plot info in figure appdata for callbacks
pInfo.hSurfs  = hSurfs;
pInfo.hAxes   = hAxes;
pInfo.rawData = rawData;
pInfo.maxInt  = maxInt;
setappdata(fig, 'plotInfo', pInfo);

% Force draw so subplot positions are finalized
drawnow;

% Determine slider region: to the right of the rightmost colorbars
allCBs = findobj(fig, 'Type', 'ColorBar');
maxRight = 0;
for ci = 1:numel(allCBs)
    cbPos = allCBs(ci).Position;
    maxRight = max(maxRight, cbPos(1) + cbPos(3));
end

% Slider layout
sliderW    = 0.025;
labelH     = 0.03;
readoutH   = 0.025;
sliderGap  = 0.015;
cbGap      = 0.05;

sliderBot = 0.10;
sliderH   = 0.78;

% --- Power slider ---
powerX = maxRight + cbGap;

uicontrol(fig, 'Style', 'slider', ...
    'Min', 0.01, 'Max', 1, 'Value', gamma, ...
    'Units', 'normalized', ...
    'Position', [powerX, sliderBot, sliderW, sliderH], ...
    'Tag', 'powerSlider', ...
    'Callback', @(src, ~) gammaCallback(src, fig));

uicontrol(fig, 'Style', 'text', ...
    'String', 'Power', ...
    'Units', 'normalized', ...
    'Position', [powerX - 0.01, sliderBot + sliderH + 0.002, ...
        sliderW + 0.02, labelH], ...
    'FontSize', 8, ...
    'HorizontalAlignment', 'center', ...
    'BackgroundColor', get(fig, 'Color'));

uicontrol(fig, 'Style', 'text', ...
    'String', sprintf('%.2f', gamma), ...
    'Units', 'normalized', ...
    'Position', [powerX - 0.005, sliderBot - readoutH - 0.002, ...
        sliderW + 0.01, readoutH], ...
    'FontSize', 8, ...
    'Tag', 'powerReadout', ...
    'HorizontalAlignment', 'center', ...
    'BackgroundColor', get(fig, 'Color'));

% --- Colormap shift slider ---
cmapX = powerX + sliderW + sliderGap;

uicontrol(fig, 'Style', 'slider', ...
    'Min', 0, 'Max', 0.95, 'Value', 0, ...
    'Units', 'normalized', ...
    'Position', [cmapX, sliderBot, sliderW, sliderH], ...
    'Tag', 'cmapShiftSlider', ...
    'Callback', @(src, ~) cmapShiftCallback(src, fig));

uicontrol(fig, 'Style', 'text', ...
    'String', 'Cmap', ...
    'Units', 'normalized', ...
    'Position', [cmapX - 0.01, sliderBot + sliderH + 0.002, ...
        sliderW + 0.02, labelH], ...
    'FontSize', 8, ...
    'HorizontalAlignment', 'center', ...
    'BackgroundColor', get(fig, 'Color'));

uicontrol(fig, 'Style', 'text', ...
    'String', '0.00', ...
    'Units', 'normalized', ...
    'Position', [cmapX - 0.005, sliderBot - readoutH - 0.002, ...
        sliderW + 0.01, readoutH], ...
    'FontSize', 8, ...
    'Tag', 'cmapShiftReadout', ...
    'HorizontalAlignment', 'center', ...
    'BackgroundColor', get(fig, 'Color'));

% --- Projection toggle ---
toggleW = cmapX + sliderW - powerX;
toggleH = 0.035;
toggleY = sliderBot - readoutH - toggleH - 0.01;

uicontrol(fig, 'Style', 'togglebutton', ...
    'String', 'Perspective', ...
    'Units', 'normalized', ...
    'Position', [powerX, toggleY, toggleW, toggleH], ...
    'FontSize', 8, ...
    'Tag', 'projToggle', ...
    'Value', 0, ...
    'Callback', @(src, ~) projCallback(src, fig));

fprintf('Done. Adjust sliders and toggle to explore the data.\n');

%% === Helper functions ===

function vg = applyGamma(vals, gamma)
%APPLYGAMMA Apply power compression, normalized to [0, 1] first.
    mn = min(vals(:));
    mx = max(vals(:));
    if mx > mn
        vn = (vals - mn) / (mx - mn);
        vg = vn .^ gamma * (mx - mn) + mn;
    else
        vg = vals;
    end
end

function gammaCallback(src, fig)
    g = get(src, 'Value');
    hReadout = findobj(fig, 'Tag', 'powerReadout');
    set(hReadout, 'String', sprintf('%.2f', g));

    pInfo = getappdata(fig, 'plotInfo');
    for k = 1:numel(pInfo.hSurfs)
        Vg = applyGamma(pInfo.rawData{k}, g);
        set(pInfo.hSurfs(k), 'ZData', Vg, 'CData', Vg);
        rangeVg = max(Vg(:)) - min(Vg(:));
        if rangeVg > 0
            daspect(pInfo.hAxes(k), [1 1 rangeVg / pInfo.maxInt]);
        end
    end

    % Reapply colormap shift to the new gamma-transformed data
    hShift = findobj(fig, 'Tag', 'cmapShiftSlider');
    if ~isempty(hShift)
        cmapShiftCallback(hShift, fig);
    end

    drawnow;
end

function cmapShiftCallback(src, fig)
    shiftFrac = get(src, 'Value');
    hReadout = findobj(fig, 'Tag', 'cmapShiftReadout');
    set(hReadout, 'String', sprintf('%.2f', shiftFrac));

    pInfo = getappdata(fig, 'plotInfo');
    for k = 1:numel(pInfo.hSurfs)
        cdata = get(pInfo.hSurfs(k), 'CData');
        minC  = min(cdata(:));
        maxC  = max(cdata(:));
        if maxC > minC
            newLow = minC + shiftFrac * (maxC - minC);
            set(pInfo.hAxes(k), 'CLim', [newLow, maxC]);
        end
    end

    drawnow;
end

function projCallback(src, fig)
    pInfo = getappdata(fig, 'plotInfo');

    if get(src, 'Value') == 1
        proj = 'perspective';
        label = 'Orthographic';
    else
        proj = 'orthographic';
        label = 'Perspective';
    end
    set(src, 'String', label);

    for k = 1:numel(pInfo.hAxes)
        set(pInfo.hAxes(k), 'Projection', proj);
    end

    drawnow;
end