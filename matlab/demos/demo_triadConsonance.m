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
%    'tensor'    — Tensor harmonicity (Smit et al. 2019):
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
%        roughness, addSpectra, evalExpTens, convertPitch
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

% Default transform mode and parameter values for visualization.
%   - 'off'  : no transform; data shown as-is
%   - 'gamma': power compression  v -> v.^gamma  (gamma in [0.01, 1])
%   - 'sat'  : saturation         v -> 1 - exp(-v / eta)  (eta log-scale in
%              [0.001, 5], applied to data normalised to [0, 1] then
%              rescaled back to the original range)
% Both gamma and eta have per-mode memory: switching between modes
% restores each mode's last slider value. An interactive radio group
% selects the mode; the slider beside it adapts.
gamma = 1.0;
eta   = 5;

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

    fprintf('Tensor harmonicity template setup (r=3, dup=%d, spectrum: %s)...\n', ...
        dup_tens, ...
        strjoin(cellfun(@num2str, spec_tens, 'UniformOutput', false), ', '));
    if dup_tens > 3
        warning('demo_triadConsonance:largeDuplicate', ...
                'dup_tens = %d: computation time grows rapidly. Consider reducing to 3 or fewer.', ...
                dup_tens);
    end
    [tp, tw] = addSpectra(zeros(dup_tens, 1), ones(dup_tens, 1), spec_tens{:});
    nJ_template = factorial(3) * nchoosek(numel(tp), 3);
    fprintf('  Template: %d partials, %d ordered triples.\n', numel(tp), nJ_template);
end

%% === Compute features ===
% Exploit symmetry: features are invariant to swapping interval1 and
% interval2, so we build a linear list of unordered (int1, int2) pairs
% (one per upper-triangle entry, j >= i) and compute each feature once
% per unique pair, then mirror into the symmetric output matrix.
%
% Loop structure: each unique triad {0, ints(i), ints(j)} (j >= i) is
% computed once and mirrored into the symmetric (nInts x nInts) result
% grids. The upper-triangle pattern is recommended for *roughness*,
% which has no batched-input dispatch and no internal dedup — every
% iteration of its loop does the full computation from scratch, so
% halving the iteration count halves the actual work. For the three
% batched features (tensor harmonicity via evalExpTens, template
% harmonicity, and spectral entropy), the upper triangle is a
% code-organization choice only: passing the full (nInts^2) grid would
% do the same amount of internal ET work, because the canonical-form
% dedup in the batched dispatch collapses permutation-equivalent inputs
% (i, j) and (j, i) onto a single cached density. Keeping the
% upper-triangle pattern across all four features makes the unique-
% triad structure explicit in the demo code.

nUpper = nInts * (nInts + 1) / 2;

% Build the linear list of (i, j) pairs with j >= i.
iLin    = zeros(nUpper, 1);
jLin    = zeros(nUpper, 1);
int1Lin = zeros(nUpper, 1);
int2Lin = zeros(nUpper, 1);
k = 0;
for i = 1:nInts
    for j = i:nInts
        k = k + 1;
        iLin(k)    = i;
        jLin(k)    = j;
        int1Lin(k) = ints(i);
        int2Lin(k) = ints(j);
    end
end

% Linear indices into the (nInts x nInts) result matrices for the upper
% triangle and its mirror. The matrices use the convention
% rows = int2 (= ints(j)), cols = int1 (= ints(i)).
linIdxUpper = sub2ind([nInts, nInts], jLin, iLin);   % row = j, col = i
linIdxLower = sub2ind([nInts, nInts], iLin, jLin);   % mirror

fprintf('Computing features for %d unique triads (step = %d cents)...\n', ...
    nUpper, step);
t0_total = tic;

% --- Tensor harmonicity ---
% One evalExpTens call: the harmonic-template arrays (tp, tw) are
% queried at all upper-triangle interval pairs in a single 2 x nUpper
% query matrix. evalExpTens builds the template tensor internally and
% prints its own time estimate via estimateCompTime when called with
% 'verbose', true.
if doTensor
    intMat  = [int1Lin'; int2Lin'];   % 2 x nUpper
    t0 = tic;
    tensLin = evalExpTens(tp, tw, sigma_tens, 3, true, false, 1200, ...
        intMat, 'verbose', true);
    fprintf('  Tensor harmonicity:   %.2f s actual (%d triads, batched)\n', ...
        toc(t0), nUpper);
    tensHarm(linIdxUpper) = tensLin;
    tensHarm(linIdxLower) = tensLin;
end

% --- Template harmonicity ---
% One templateHarmonicity call: stack chords as rows of an nUpper x 3
% matrix; the function returns hMax and hEntropy as nUpper-element
% column vectors. templateHarmonicity prints its own time
% estimate via estimateCompTime when called with 'verbose', true.
if doTmpl
    chordMat = [zeros(nUpper, 1), int1Lin, int2Lin];
    t0 = tic;
    [hMaxLin, hEntLin] = templateHarmonicity(chordMat, [], sigma_tmpl, ...
        'spectrum', spec_tmpl, ...
        'chordSpectrum', spec_tmpl, ...
        'verbose', true);
    fprintf('  Template harmonicity: %.2f s actual (%d triads, batched)\n', ...
        toc(t0), nUpper);
    if doTmplMax
        tmplHarmMax(linIdxUpper) = hMaxLin;
        tmplHarmMax(linIdxLower) = hMaxLin;
    end
    if doTmplEnt
        tmplHarmEnt(linIdxUpper) = hEntLin;
        tmplHarmEnt(linIdxLower) = hEntLin;
    end
end

% --- Spectral entropy ---
% One spectralEntropy call on a stacked chord matrix.
%
% Method choice: we pass 'method', 'normalized' explicitly to
% reproduce the consonance ordering and absolute values reported in
% Smit et al. (2019) and Milne et al. (2017), which use the
% normalised Shannon entropy H / log_b(N) in [0, 1]. The toolbox
% default for spectralEntropy is 'differential' (adaptive nested-
% grid differential entropy h_hat) which gives the same ordering of
% chords by consonance but in different units and at higher per-call
% cost (the adaptive evaluator doubles the grid to convergence,
% which is several times slower than a single discrete pass).
% 'method', 'renyi2' (analytical Rényi-2 via the inner-product /
% Möbius machinery) is also available; it agrees on ordering but,
% like differential, is in different units.
if doSpecEnt
    chordMatSE = [zeros(nUpper, 1), int1Lin, int2Lin];
    t0 = tic;
    specEntLin = spectralEntropy(chordMatSE, [], sigma_ent, ...
        'spectrum', spec_ent, 'method', 'normalized', 'verbose', true);
    fprintf('  Spectral entropy:     %.2f s actual (%d triads, batched)\n', ...
        toc(t0), nUpper);
    specEnt(linIdxUpper) = specEntLin;
    specEnt(linIdxLower) = specEntLin;
end

% --- Roughness (no batched mode; explicit loop) ---
if doRough
    roughLin = NaN(nUpper, 1);

    fprintf('  Roughness: looping over %d triads...\n', nUpper);
    t0 = tic;
    nDone = 0;
    for k = 1:nUpper
        int1k = int1Lin(k);
        int2k = int2Lin(k);

        chordCents = [refCents, refCents + int1k, refCents + int2k];
        [ep, ew] = addSpectra(chordCents(:), [], spec_rough{:});
        fHz = convertPitch(ep, 'cents', 'hz');
        roughLin(k) = roughness(fHz, ew);

        nDone = nDone + 1;
        if mod(nDone, 500) == 0 || nDone == nUpper
            elapsed = toc(t0);
            rate    = nDone / elapsed;
            remain  = (nUpper - nDone) / rate;
            fprintf('    %d / %d triads (%.1f s elapsed, ~%.0f s remaining)\n', ...
                nDone, nUpper, elapsed, remain);
        end
    end
    fprintf('  Roughness:            %.2f s\n', toc(t0));

    rough(linIdxUpper) = roughLin;
    rough(linIdxLower) = roughLin;
end

fprintf('All features computed in %.1f s.\n', toc(t0_total));

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

% Figure width: enough columns AND enough room on the right for the
% slider region (which sits in the rightmost 14% of figure width).
fig = figure('Name', 'Triad consonance', ...
    'Position', [50, 50, min(420 * nCols + 320, 1700), ...
                          min(420 * nRows + 80,  1100)]);

hSurfs  = gobjects(nPlots, 1);
hAxes   = gobjects(nPlots, 1);
hCbars  = gobjects(nPlots, 1);
rawData = cell(nPlots, 1);

% Manual subplot+colorbar layout. Reserve the rightmost `sliderRegion`
% fraction of the figure width for the slider/toggle UI; lay out the
% subplots in the remaining left part. The colorbar is deliberately
% narrow so that when MATLAB switches to perspective projection (which
% reflows the y-axis labels to the right of the plot area, pushing
% rendered content rightwards), the colorbar still does not collide
% with the slider region. Both axes and colorbars are pinned; the
% slider region is at a fixed x.
sliderRegion = 0.18;
leftMargin   = 0.06;
rightMargin  = sliderRegion + 0.02;     % right edge of subplot block
topMargin    = 0.86;
bottomMargin = 0.10;
gutterX      = 0.085;
gutterY      = 0.10;
plotW = (1 - leftMargin - rightMargin - (nCols - 1) * gutterX) / nCols;
plotH = (topMargin - bottomMargin - (nRows - 1) * gutterY) / nRows;
% Width given to a colorbar (within its subplot's allocated horizontal
% slot). Kept narrow to leave perspective-mode rendering room.
cbarFrac     = 0.045;

for mi = 1:nPlots
    hAxes(mi) = subplot(nRows, nCols, mi);

    data = allData{mi};
    rawData{mi} = data;

    V = applyTransform(data, 'off', gamma, eta);

    hSurfs(mi) = surf(Ga, Gb, V, 'EdgeColor', 'none');
    colormap(hAxes(mi), parula);
    hCbars(mi) = colorbar(hAxes(mi));
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

    % Compute this subplot's slot in the grid (row-major)
    [rIdx, cIdx] = ind2sub([nCols, nRows], mi);  % column-major: swap
    rIdx = floor((mi - 1) / nCols) + 1;
    cIdx = mod(mi - 1, nCols) + 1;
    slotX = leftMargin + (cIdx - 1) * (plotW + gutterX);
    slotY = topMargin  - rIdx       * plotH ...
                       - (rIdx - 1) * gutterY;

    % Split the slot horizontally: the surface plot on the left,
    % the colorbar on the right. Lock both with PositionConstraint
    % so projection changes do not move them.
    cbarW   = cbarFrac * plotW;
    cbarGap = 0.012;
    plotInnerW = plotW - cbarW - cbarGap;
    set(hAxes(mi), 'Units', 'normalized', ...
        'Position', [slotX, slotY, plotInnerW, plotH]);
    if isprop(hAxes(mi), 'PositionConstraint')
        set(hAxes(mi), 'PositionConstraint', 'innerposition');
    end
    set(hCbars(mi), 'Units', 'normalized', ...
        'Position', [slotX + plotInnerW + cbarGap, ...
                     slotY + 0.05 * plotH, ...
                     cbarW, 0.90 * plotH]);
end

sgtitle(sprintf('Triad consonance (step = %d cents)', step), ...
    'FontWeight', 'bold');

%% === Vertical sliders, transform-mode toggle, and projection toggle ===

% Store plot info in figure appdata for callbacks
pInfo.hSurfs        = hSurfs;
pInfo.hAxes         = hAxes;
pInfo.hCbars        = hCbars;
pInfo.cbarPosOrtho  = cell(numel(hCbars), 1);   % saved (x, y, w, h) per cbar
for k = 1:numel(hCbars)
    pInfo.cbarPosOrtho{k} = get(hCbars(k), 'Position');
end
pInfo.rawData       = rawData;
pInfo.maxInt        = maxInt;
pInfo.mode          = 'off';      % 'off' | 'gamma' | 'sat'
pInfo.gamma         = gamma;      % per-mode memory: gamma slider value
pInfo.eta           = eta;        % per-mode memory: eta slider value
pInfo.cmapShiftOff   = 0;         % per-mode memory: cmap shift for 'off'
pInfo.cmapShiftGamma = 0;         % per-mode memory: cmap shift for 'gamma'
pInfo.cmapShiftSat   = 0;         % per-mode memory: cmap shift for 'sat'
setappdata(fig, 'plotInfo', pInfo);

% Force draw so subplot positions are finalized
drawnow;

% Slider region is anchored at a fixed x in the figure: the rightmost 
% sliderRegion fraction of the figure (set up earlier when laying out 
% subplots). xformX is the left edge of the transform slider.
sliderW    = 0.025;
labelH     = 0.025;
readoutH   = 0.025;
sliderGap  = 0.015;
modeGroupH = 0.025;          % height for the radio circles themselves
labelTopH  = 0.020;          % height for the "Off"/"Saturation" row above
labelBotH  = 0.020;          % height for the "Gamma" row below
modeBlockH = labelTopH + modeGroupH + labelBotH + 0.005;

sliderBot = 0.10;
sliderH   = 0.78 - modeBlockH - 0.005;

% Anchor sliders inside the reserved right region. Place the transform
% slider near the left edge of the slider region; cmap to its right.
xformX = 1 - sliderRegion + 0.025;

% Mode selector: a button group containing three empty-string radios
% (just the circles), plus separate text labels positioned above (for
% "Off" and "Saturation") and below (for "Gamma"). This zigzag layout
% lets the labels be wider than 1/3 of the bargroup width without
% truncation.
modeBlockY = sliderBot + sliderH + labelH + 0.005;

% Top label row: "Off" above column 1, "Saturation" above column 3.
% Sit a bit closer to the radio row than before.
topLabelY = modeBlockY + modeGroupH + labelBotH - 0.005;
modeBgW   = 2 * sliderW + sliderGap + 0.01;

uicontrol(fig, 'Style', 'text', ...
    'String', 'Off', ...
    'Units', 'normalized', ...
    'Position', [xformX - 0.020, topLabelY, 0.040, labelTopH], ...
    'FontSize', 8, 'HorizontalAlignment', 'center', ...
    'Tag', 'modeOff_label', ...
    'BackgroundColor', get(fig, 'Color'));

uicontrol(fig, 'Style', 'text', ...
    'String', 'Saturation', ...
    'Units', 'normalized', ...
    'Position', [xformX + modeBgW - 0.045, topLabelY, ...
                 0.060, labelTopH], ...
    'FontSize', 8, 'HorizontalAlignment', 'center', ...
    'Tag', 'modeSat_label', ...
    'BackgroundColor', get(fig, 'Color'));

% Radio row: three radios with empty Strings, just showing their circles
radioY = modeBlockY + labelBotH;

modeGroup = uibuttongroup(fig, ...
    'Units', 'normalized', ...
    'Position', [xformX - 0.005, radioY, modeBgW, modeGroupH], ...
    'BorderType', 'none', ...
    'BackgroundColor', get(fig, 'Color'), ...
    'Tag', 'modeGroup', ...
    'SelectionChangedFcn', @(src, evt) modeChangedCallback(src, evt, fig));

uicontrol(modeGroup, 'Style', 'radiobutton', ...
    'String', '', ...
    'Units', 'normalized', ...
    'Position', [0.05, 0, 0.28, 1], ...
    'Tag', 'modeOff', ...
    'BackgroundColor', get(fig, 'Color'), ...
    'Value', 1);

uicontrol(modeGroup, 'Style', 'radiobutton', ...
    'String', '', ...
    'Units', 'normalized', ...
    'Position', [0.39, 0, 0.28, 1], ...
    'Tag', 'modeGamma', ...
    'BackgroundColor', get(fig, 'Color'), ...
    'Value', 0);

uicontrol(modeGroup, 'Style', 'radiobutton', ...
    'String', '', ...
    'Units', 'normalized', ...
    'Position', [0.72, 0, 0.28, 1], ...
    'Tag', 'modeSat', ...
    'BackgroundColor', get(fig, 'Color'), ...
    'Value', 0);

% Bottom label row: "Gamma" below column 2 (centre)
botLabelY = modeBlockY;

uicontrol(fig, 'Style', 'text', ...
    'String', 'Gamma', ...
    'Units', 'normalized', ...
    'Position', [xformX + modeBgW/2 - 0.026, botLabelY, ...
                 0.040, labelBotH], ...
    'FontSize', 8, 'HorizontalAlignment', 'center', ...
    'Tag', 'modeGamma_label', ...
    'BackgroundColor', get(fig, 'Color'));

% Slider (initial state: 'off' -> disabled). Min/Max/Value updated by 
% modeChangedCallback when the user selects Gamma or Sat. Slimmer 
% slider thumb via reduced SliderStep.
uicontrol(fig, 'Style', 'slider', ...
    'Min', 0.01, 'Max', 1, 'Value', gamma, ...
    'Units', 'normalized', ...
    'Position', [xformX, sliderBot, sliderW, sliderH], ...
    'Tag', 'xformSlider', ...
    'Enable', 'off', ...
    'SliderStep', [0.005, 0.03], ...
    'Callback', @(src, ~) xformSliderCallback(src, fig));

% Slider readout below
uicontrol(fig, 'Style', 'text', ...
    'String', '', ...
    'Units', 'normalized', ...
    'Position', [xformX - 0.005, sliderBot - readoutH - 0.002, ...
        sliderW + 0.01, readoutH], ...
    'FontSize', 8, ...
    'Tag', 'xformReadout', ...
    'HorizontalAlignment', 'center', ...
    'BackgroundColor', get(fig, 'Color'));

% --- Colormap shift slider (unchanged in role) ---
cmapX = xformX + sliderW + sliderGap;

uicontrol(fig, 'Style', 'slider', ...
    'Min', 0, 'Max', 0.95, 'Value', 0, ...
    'Units', 'normalized', ...
    'Position', [cmapX, sliderBot, sliderW, sliderH], ...
    'Tag', 'cmapShiftSlider', ...
    'SliderStep', [0.005, 0.03], ...
    'Callback', @(src, ~) cmapShiftCallback(src, fig));

uicontrol(fig, 'Style', 'text', ...
    'String', 'Cmap', ...
    'Units', 'normalized', ...
    'Position', [cmapX - 0.01, sliderBot + sliderH + 0.002, ...
        sliderW + 0.02, labelH], ...
    'FontSize', 8, ...
    'Tag', 'cmapShiftLabel', ...
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
toggleW = cmapX + sliderW - xformX;
toggleH = 0.035;
toggleY = sliderBot - readoutH - toggleH - 0.01;

uicontrol(fig, 'Style', 'togglebutton', ...
    'String', 'Perspective', ...
    'Units', 'normalized', ...
    'Position', [xformX, toggleY, toggleW, toggleH], ...
    'FontSize', 8, ...
    'Tag', 'projToggle', ...
    'Value', 0, ...
    'Callback', @(src, ~) projCallback(src, fig));

% Capture the orthographic-mode X position of every UI element in the 
% slider region, so projCallback can restore-then-shift them when 
% switching projection. The order here must match the sliderTags list 
% in projCallback.
sliderTags = {'modeGroup', 'modeOff_label', 'modeSat_label', ...
               'modeGamma_label', 'xformSlider', 'xformReadout', ...
               'cmapShiftSlider', 'cmapShiftLabel', ...
               'cmapShiftReadout', 'projToggle'};
pInfo = getappdata(fig, 'plotInfo');
pInfo.sliderXOrtho = cell(1, numel(sliderTags));
for ti = 1:numel(sliderTags)
    h = findobj(fig, 'Tag', sliderTags{ti});
    xs = zeros(numel(h), 1);
    for hi = 1:numel(h)
        pos = get(h(hi), 'Position');
        xs(hi) = pos(1);
    end
    pInfo.sliderXOrtho{ti} = xs;
end
setappdata(fig, 'plotInfo', pInfo);

fprintf(['Done. Pick a transform (Off / Gamma / Sat) and adjust ' ...
         'sliders to explore the data.\n']);

%% === Helper functions ===

function vt = applyTransform(vals, mode, gamma, eta)
%APPLYTRANSFORM Dispatch on mode.
%
%  'off'   identity; output range = input range.
%  'gamma' power compression: data normalised by the empirical
%          (min, max) so the slider stays responsive across measures
%          with very different ranges. Output is in [0, 1]. Gamma is
%          a display-cosmetic knob with no perceptual interpretation
%          tied to absolute scale, so anchoring at the empirical min
%          is appropriate.
%  'sat'   saturation: anchored at 0 (a meaningful baseline of "no
%          density / no roughness / no entropy"). For non-negative
%          data vn = vals / max. For non-positive data (e.g.,
%          -roughness, -spec_entropy) vn = (vals - min) / (-min).
%          Mixed-sign data falls back to min/max. The saturation
%          curve (1 - exp(-vn/eta)) / (1 - exp(-1/eta)) is then
%          applied. Output is in [0, 1].
    if strcmp(mode, 'off')
        vt = vals;
        return;
    end

    mn = min(vals(:));
    mx = max(vals(:));

    switch mode
        case 'gamma'
            if mx > mn
                vn = (vals - mn) / (mx - mn);
                vt = vn .^ gamma;
            else
                vt = vals;
            end
        case 'sat'
            if mn >= 0 && mx > 0
                vn = vals / mx;
            elseif mx <= 0 && mn < 0
                vn = (vals - mn) / (-mn);
            elseif mx > mn
                vn = (vals - mn) / (mx - mn);
            else
                vt = vals;
                return;
            end
            num = 1 - exp(-vn / eta);
            den = 1 - exp(-1 / eta);
            if den > 0
                vt = num / den;
            else
                vt = vn;
            end
        otherwise
            vt = vals;
    end
end

function applyToAllSurfaces(fig)
%APPLYTOALLSURFACES Recompute the transformed data for every surface in
% the figure based on the current mode and parameter values, then update
% ZData/CData and refresh the colormap shift.
    pInfo = getappdata(fig, 'plotInfo');
    for k = 1:numel(pInfo.hSurfs)
        Vt = applyTransform(pInfo.rawData{k}, pInfo.mode, ...
                             pInfo.gamma, pInfo.eta);
        set(pInfo.hSurfs(k), 'ZData', Vt, 'CData', Vt);
        rangeVt = max(Vt(:)) - min(Vt(:));
        if rangeVt > 0
            daspect(pInfo.hAxes(k), [1 1 rangeVt / pInfo.maxInt]);
        end
    end

    % Reapply colormap shift to the new transformed data
    hShift = findobj(fig, 'Tag', 'cmapShiftSlider');
    if ~isempty(hShift)
        cmapShiftCallback(hShift, fig);
    end

    drawnow;
end

function modeChangedCallback(~, evt, fig)
    pInfo = getappdata(fig, 'plotInfo');
    hSlider     = findobj(fig, 'Tag', 'xformSlider');
    hReadout    = findobj(fig, 'Tag', 'xformReadout');
    hCmap       = findobj(fig, 'Tag', 'cmapShiftSlider');
    hCmapRdout  = findobj(fig, 'Tag', 'cmapShiftReadout');

    % Save the current slider value into the *outgoing* mode's storage
    % before switching (gamma/eta), and likewise the cmap shift.
    switch pInfo.mode
        case 'gamma'
            pInfo.gamma = get(hSlider, 'Value');
            pInfo.cmapShiftGamma = get(hCmap, 'Value');
        case 'sat'
            pInfo.eta = 10 ^ get(hSlider, 'Value');
            pInfo.cmapShiftSat = get(hCmap, 'Value');
        case 'off'
            pInfo.cmapShiftOff = get(hCmap, 'Value');
    end

    % Determine the new mode from the selected radio's tag
    switch evt.NewValue.Tag
        case 'modeOff',   newMode = 'off';
        case 'modeGamma', newMode = 'gamma';
        case 'modeSat',   newMode = 'sat';
        otherwise,        newMode = 'off';
    end
    pInfo.mode = newMode;

    % Update the slider and readout to reflect the new mode. To prevent
    % the slider thumb from disappearing when Min/Max change, first
    % clamp the current Value into the *new* range, THEN set Min/Max,
    % THEN set Value to the desired position.
    switch newMode
        case 'off'
            set(hSlider, 'Enable', 'off');
            set(hReadout, 'String', '');
        case 'gamma'
            newMin   = 0.01;
            newMax   = 1.0;
            curValue = get(hSlider, 'Value');
            safeVal  = max(min(curValue, newMax), newMin);
            set(hSlider, 'Value', safeVal);
            set(hSlider, 'Min', newMin, 'Max', newMax);
            set(hSlider, 'Value', pInfo.gamma);
            set(hSlider, 'Enable', 'on');
            set(hReadout, 'String', sprintf('%.2f', pInfo.gamma));
        case 'sat'
            newMin   = log10(0.002);
            newMax   = log10(5);
            curValue = get(hSlider, 'Value');
            safeVal  = max(min(curValue, newMax), newMin);
            set(hSlider, 'Value', safeVal);
            set(hSlider, 'Min', newMin, 'Max', newMax);
            set(hSlider, 'Value', log10(pInfo.eta));
            set(hSlider, 'Enable', 'on');
            set(hReadout, 'String', sprintf('%.3f', pInfo.eta));
    end

    % Restore the cmap shift saved for the incoming mode
    switch newMode
        case 'off',   newCmap = pInfo.cmapShiftOff;
        case 'gamma', newCmap = pInfo.cmapShiftGamma;
        case 'sat',   newCmap = pInfo.cmapShiftSat;
    end
    set(hCmap, 'Value', newCmap);
    set(hCmapRdout, 'String', sprintf('%.2f', newCmap));

    setappdata(fig, 'plotInfo', pInfo);
    applyToAllSurfaces(fig);
end

function xformSliderCallback(src, fig)
    pInfo = getappdata(fig, 'plotInfo');
    hReadout = findobj(fig, 'Tag', 'xformReadout');

    switch pInfo.mode
        case 'gamma'
            pInfo.gamma = get(src, 'Value');
            set(hReadout, 'String', sprintf('%.2f', pInfo.gamma));
        case 'sat'
            pInfo.eta = 10 ^ get(src, 'Value');
            set(hReadout, 'String', sprintf('%.3f', pInfo.eta));
        otherwise
            % 'off': slider disabled; this should not fire
            return;
    end

    setappdata(fig, 'plotInfo', pInfo);
    applyToAllSurfaces(fig);
end

function cmapShiftCallback(src, fig)
    shiftFrac = get(src, 'Value');
    hReadout = findobj(fig, 'Tag', 'cmapShiftReadout');
    set(hReadout, 'String', sprintf('%.2f', shiftFrac));

    pInfo = getappdata(fig, 'plotInfo');
    % Persist this shift to the current mode's per-mode memory
    switch pInfo.mode
        case 'off',   pInfo.cmapShiftOff   = shiftFrac;
        case 'gamma', pInfo.cmapShiftGamma = shiftFrac;
        case 'sat',   pInfo.cmapShiftSat   = shiftFrac;
    end
    setappdata(fig, 'plotInfo', pInfo);

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
        % In perspective view, MATLAB renders the y-axis labels and 
        % numbers to the right of the plot area instead of the left, 
        % so the colorbar needs to move further right to clear them. 
        % Shift each colorbar right by a fraction of its plot's width; 
        % shift the sliders the same amount so they don't end up over 
        % the colorbars.
        cbarShift   = 0.030;
        sliderShift = cbarShift;
    else
        proj = 'orthographic';
        label = 'Perspective';
        cbarShift   = 0;
        sliderShift = 0;
    end
    set(src, 'String', label);

    for k = 1:numel(pInfo.hAxes)
        set(pInfo.hAxes(k), 'Projection', proj);
        % Restore the saved orthographic position, then add the 
        % perspective shift if needed.
        cbPos = pInfo.cbarPosOrtho{k};
        cbPos(1) = cbPos(1) + cbarShift;
        set(pInfo.hCbars(k), 'Position', cbPos);
    end

    % Shift every UI element in the slider region (radios, labels, 
    % both sliders, both readouts, and the projection toggle itself) 
    % horizontally by sliderShift.
    sliderTags = {'modeGroup', 'modeOff_label', 'modeSat_label', ...
                   'modeGamma_label', 'xformSlider', 'xformReadout', ...
                   'cmapShiftSlider', 'cmapShiftLabel', ...
                   'cmapShiftReadout', 'projToggle'};
    for ti = 1:numel(sliderTags)
        h = findobj(fig, 'Tag', sliderTags{ti});
        if isempty(h), continue; end
        for hi = 1:numel(h)
            pos = get(h(hi), 'Position');
            pos(1) = pInfo.sliderXOrtho{ti}(hi) + sliderShift;
            set(h(hi), 'Position', pos);
        end
    end

    drawnow;
end