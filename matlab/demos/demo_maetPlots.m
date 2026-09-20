%% demo_maetPlots.m
%  Visualizes expectation tensor densities in 1 to 3 dimensions for
%  user-specified combinations of r, isRel, isPer, and isExch, each
%  configuration drawn both unordered and ordered -- including r = 1,
%  where the two are necessarily the same picture.
%
%  Uses buildMaet to precompute the density object once per configuration,
%  then passes it to evalMaet for one and two dimensions, and to
%  plotMaet3d for three.
%
%  Each figure includes an interactive transform-mode selector (Off /
%  Gamma / Saturation) for real-time adjustment of dynamic-range
%  compression: gamma applies v -> v.^gamma, saturation applies
%  v -> 1 - exp(-v / eta), both with per-plot normalisation. Surface
%  plots (dim = 2) additionally include a colormap shift slider and a
%  perspective/orthographic projection toggle.
%
%  Edit the parameters below to experiment with different multisets,
%  smoothing widths, plot configurations, and visualization modes.
%
%  The three-dimensional plots are drawn by the toolbox's own
%  plotMaet3d, whose method is chosen by plot3Dmode below.
%
%  Uses: buildMaet, evalMaet, plotMaet3d (from the Music Perception
%  Toolbox).

%% === User-editable parameters ===
close all
% Pitch set and weights
p = [0; 200; 400; 500; 700; 900; 1100];
w = [];

% Gaussian smoothing width
sigma = 15;

% Normalization mode for density evaluation: 'none', 'gaussian', or 'pdf'
%   'none'     — Raw weighted proximity scores (default). Only relative
%                values across query points are meaningful.
%   'gaussian' — Each Gaussian component integrates to 1. Useful for
%                comparing densities computed with different sigma values.
%   'pdf'      — Full probability density (integrates to 1 over the domain).
%                Useful for comparing across multisets of different sizes.
normalize = 'none';

% Default transform-mode and parameter values for visualization.
%   - 'off'  : no transform; data shown as-is
%   - 'gamma': power compression  v -> v.^gamma  (gamma in [0.01, 1])
%   - 'sat'  : saturation         v -> 1 - exp(-v / eta)  (eta log-scale 
%              in [0.001, 5], applied to data normalised to [0, 1] then 
%              rescaled back to the original range)
% Both gamma and eta have per-mode memory inside each figure: switching 
% between modes via the radio selector restores each mode's last 
% slider value.
gamma = 1;
eta   = 5;

% Period for periodic configurations (in the same units as p)
period = 1200;

% === Plot configurations ===
% Each row specifies one plot: [r, isRel, isPer, isExch]
%   r      — tuple size
%   isRel  — 0 = absolute, 1 = relative (transposition-invariant)
%   isPer  — 0 = non-periodic, 1 = periodic
%   isExch — 1 = unordered (exchangeable tuples), 0 = ordered
%
% The effective query dimensionality is dim = r - isRel, and only one
% to three dimensions are drawn: a four-dimensional density has no
% honest picture, and the grid of two-dimensional slices this demo used
% to draw for it showed three arbitrary cuts rather than the density.
%
% Every configuration appears twice, unordered and then ordered. An
% ordered density counts each arrangement of a tuple separately, so it
% is unsymmetric in its arguments and has the fuller support; the
% unordered one is its symmetrization.
%
% Add, remove, or reorder rows to control which plots are produced.
%
% r = 1 is drawn both ways too, though it has one slot and so nothing
% to order: the two plots come out identical, which is the point of
% including them. Exchangeability is a statement about the arrangement
% of a tuple's elements, and a tuple of one has only the one.
configs = [
1,  0,  0,  1;
1,  0,  0,  0;
1,  0,  1,  1;
1,  0,  1,  0;
2,  0,  0,  1;
2,  0,  0,  0;
2,  0,  1,  1;
2,  0,  1,  0;
2,  1,  0,  1;
2,  1,  0,  0;
2,  1,  1,  1;
2,  1,  1,  0;
3,  0,  0,  1;
3,  0,  0,  0;
3,  0,  1,  1;
3,  0,  1,  0;
3,  1,  0,  1;
3,  1,  0,  0;
3,  1,  1,  1;
3,  1,  1,  0;
4,  1,  0,  1;
4,  1,  0,  0;
4,  1,  1,  1;
4,  1,  1,  0;
];

% === Grid resolution as step size ===
% Specify the grid spacing (in the same units as p and period) for each
% effective dimensionality. The number of grid points per axis is computed
% automatically from the axis range and step size.
% Smaller step = finer grid = slower computation (scales as step^(-dim)).
%
% The step that matters is the step measured against sigma, not against
% the axis range: a blob is a few sigma across, so a grid coarser than
% sigma steps straight over it and the density appears to have peaks
% missing rather than blurred. A step of about sigma puts one sample
% per sigma, which is the least that shows the shape.
% 'ellipsoids' evaluates no grid and ignores step_3d entirely, so it is
% the mode to check a blob count against.
step_1d = 1;     % e.g., 1 cent per grid point
step_2d = 5;     % 5 cents per dimension
step_3d = 10;    % about one sample per sigma at the sigma set above

% === Axis range for non-periodic configurations ===
% For periodic configurations, the range is always [0, period].
% For non-periodic configurations, set the range here. A range centred
% on zero shows an interval and its inversion either side of the
% unison, which is what a relative density is symmetric about; an
% absolute density is drawn over the same range so that the two can be
% read against each other.
axMinNonPer = -1200;
axMaxNonPer = 1200;

% === Colour map ===
% Brightening of parula for the two-dimensional surfaces, as MATLAB's
% brighten: positive lifts its low end, which is where most of a
% density's material sits. The three-dimensional plots are left to
% plotMaet3d's own default, that being the function's to set.
surfBrighten = 0.7;

% === Figure window style ===
% true docks every figure, so the configurations arrive as tabs of one
% window rather than as a window each. Docked figures take their size
% from the dock, so the widening the surface plots and the controls
% would otherwise ask for is skipped.
dockFigures = true;

% === 3D visualization settings ===

% 3D plot mode, passed straight to plotMaet3d as its 'method':
%   'ellipsoids' — one ellipsoid per tuple centre, shaped by the
%                  kernel's covariance and coloured by the density
%                  there. No grid is evaluated, so it is by far the
%                  cheapest, and it shows the kernels rather than the
%                  sum they make.
%   'points'     — one translucent mark per grid node above a
%                  threshold, coloured and made translucent by the
%                  value. Shows what lies between the peaks, and is the
%                  only mode the transform controls can drive.
%   'slices'     — the volume as a stack of textured planes, a true
%                  volume rendering: what a ray accumulates along its
%                  length is what the picture shows.
plot3Dmode = 'slices';

%% === Estimate total runtime ===

nConfigs = size(configs, 1);
totalEstSec = 0;

fprintf('\n--- Plot summary ---\n');
for ci = 1:nConfigs
    rc      = configs(ci, 1);
    isRelC  = logical(configs(ci, 2));
    isPerC  = logical(configs(ci, 3));
    isExchC = logical(configs(ci, 4));

    % Skip invalid configs (same logic as main loop)
    if rc > numel(p), continue; end
    if isRelC && rc < 2, continue; end

    % Effective dimensionality
    dimC = rc - isRelC;

    % Axis range for this config
    if isPerC
        axMinC = 0;
        axMaxC = period;
    else
        axMinC = axMinNonPer;
        axMaxC = axMaxNonPer;
    end

    % Grid resolution from step size
    switch dimC
        case 1, stepC = step_1d;
        case 2, stepC = step_2d;
        otherwise, stepC = step_3d;
    end
    resC = max(2, round((axMaxC - axMinC) / stepC) + 1);

    % Only one to three dimensions are drawn, and the main loop stops
    % on anything more, so a row that asks for more is reported here
    % rather than costed.
    if dimC > 3
        fprintf(['  Config %d: r=%d, dim=%d -- not drawn, only one to ' ...
                 'three dimensions are.\n'], ci, rc, dimC);
        continue
    end
    nQc = double(resC)^dimC;

    % Problem sizes
    nc      = numel(p);
    nPermsC = factorial(rc);
    nCombsC = nchoosek(nc, rc);
    nJc     = nPermsC * nCombsC;

    % Estimate this config's eval time (silent call — empty label)
    nPairsC    = double(nJc) * nQc;
    configEst  = estimateCompTime(nPairsC, dimC, '');
    totalEstSec = totalEstSec + configEst;

    % Mode labels
    if isRelC, mStr = 'rel'; else, mStr = 'abs'; end
    if isPerC, pStr = 'per'; else, pStr = 'non-per'; end

    fprintf('  Config %d: r=%d, %s, %s, dim=%d, res=%d, queries=%.2g, tuples=%d\n', ...
        ci, rc, mStr, pStr, dimC, resC, nQc, nJc);
end

fprintf('---\n');
% Format and print the accumulated total
if totalEstSec < 1
    totalTimeStr = sprintf('%.0f ms', totalEstSec * 1000);
elseif totalEstSec < 60
    totalTimeStr = sprintf('%.1f s', totalEstSec);
elseif totalEstSec < 3600
    totalTimeStr = sprintf('%.1f min', totalEstSec / 60);
else
    totalTimeStr = sprintf('%.1f hr', totalEstSec / 3600);
end
if totalEstSec > 2
    fprintf('plotMaet (total): estimated time ~%s (Ctrl+C to cancel).\n\n', ...
        totalTimeStr);
else
    fprintf('plotMaet (total): estimated time ~%s.\n\n', totalTimeStr);
end


%% === Iterate through configurations ===

for ci = 1:nConfigs
    % Fresh for each configuration, so that no field set for one plot
    % is left behind for the next.
    info = struct();

    r      = configs(ci, 1);
    isRelR = logical(configs(ci, 2));
    isPerR = logical(configs(ci, 3));
    isExchR = logical(configs(ci, 4));

    % --- Validation ---
    if r > numel(p)
        fprintf('Config %d: r = %d skipped (multiset has only %d elements).\n', ...
            ci, r, numel(p));
        continue;
    end
    if isRelR && r < 2
        fprintf('Config %d: r = %d with isRel = true skipped (requires r >= 2).\n', ...
            ci, r);
        continue;
    end

    % --- Axis range ---
    if isPerR
        axMin = 0;
        axMax = period;
    else
        axMin = axMinNonPer;
        axMax = axMaxNonPer;
    end

    % --- Effective dimensionality ---
    dim = r - isRelR;

    % --- Grid resolution from step size ---
    switch dim
        case 1, stepSize = step_1d;
        case 2, stepSize = step_2d;
        otherwise, stepSize = step_3d;
    end
    res = max(2, round((axMax - axMin) / stepSize) + 1);

    % --- Labels for plot titles ---
    if isRelR
        modeStr = 'relative';
        axLabel = 'Interval';
    else
        modeStr = 'absolute';
        axLabel = 'Pitch';
    end

    if isPerR
        perStr = 'periodic';
    else
        perStr = 'non-periodic';
    end

    if isExchR
        ordStr = 'unordered';
    else
        ordStr = 'ordered';
    end

    titleStr = sprintf('r = %d, %s, %s, %s, \\sigma = %.2f', ...
        r, modeStr, perStr, ordStr, sigma);

    fprintf(['Config %d: r = %d (%s, %s, %s, dim = %d, res = %d): ' ...
             'precomputing...'], ...
        ci, r, modeStr, perStr, ordStr, dim, res);

    % --- Precompute the density object ---
    dens = buildMaet(p, w, sigma, r, isRelR, isPerR, period, isExchR);

    fprintf(' evaluating...');

    % -----------------------------------------------------------------
    %  Dispatch to the appropriate plotting routine based on dim
    % -----------------------------------------------------------------
    switch dim

        % =============================================================
        %  dim = 1: line plot with power slider
        % =============================================================
        case 1
            x = linspace(axMin, axMax, res);
            X = x;  % 1 x res

            vals = evalMaet(dens, X, normalize);

            fig = newDemoFigure(ci, r, dim, dockFigures);
            hLine = plot(x, applyTransform(vals, 'off', gamma, eta), 'LineWidth', 1.5);
            xlabel(sprintf('%s 1', axLabel));
            ylabel('Density');
            title(titleStr);
            xlim([axMin axMax]);
            set(gca, 'XGrid', 'on', 'YGrid', 'on');
            setDemoTicks(gca, [axMin axMax], 1);

            % Store raw data and add slider
            info.mode    = 'line';
            info.rawVals = vals;
            info.hLine   = hLine;
            addPlotControls(fig, info, gamma, eta);

        % =============================================================
        %  dim = 2: surface plot (top-down X-Y view) with controls
        % =============================================================
        case 2
            x = linspace(axMin, axMax, res);
            [Ga, Gb] = meshgrid(x, x);

            % An exchangeable density is symmetric in its arguments,
            % so half the grid can be evaluated and mirrored. An
            % ordered one is not -- being unsymmetric is the whole of
            % what distinguishes it -- so it is evaluated whole.
            if isExchR
                upperMask = triu(true(res));
                Xu = [Ga(upperMask)'; Gb(upperMask)'];
                valsU = evalMaet(dens, Xu, normalize);
                Vraw = zeros(res, res);
                Vraw(upperMask) = valsU;
                Vraw = Vraw + Vraw.' - diag(diag(Vraw));
                vals = Vraw(:).';
            else
                vals = evalMaet(dens, [Ga(:)'; Gb(:)'], normalize);
                vals = vals(:).';
            end

            V = reshape(applyTransform(vals, 'off', gamma, eta), res, res);

            fig = newDemoFigure(ci, r, dim, dockFigures);

            % Widen figure to accommodate plot + colorbar + controls.
            % A docked figure has no say in its size, so it is left be.
            if strcmp(get(fig, 'WindowStyle'), 'normal')
                figPos = get(fig, 'Position');
                set(fig, 'Position', [figPos(1), figPos(2), ...
                    max(figPos(3), 900), figPos(4)]);
            end

            hSurf = surf(Ga, Gb, V, 'EdgeColor', 'none');
            hAx = gca;

            % Shrink axes to leave room for 3D labels, colorbar, and controls.
            % The right margin (~50% of figure width) accommodates:
            %   - 3D axis tick labels in perspective view
            %   - Colorbar + its tick labels
            %   - Power and Cmap sliders
            %   - Projection toggle
            set(hAx, 'Position', [0.08 0.12 0.48 0.78]);
            xlabel(sprintf('%s 1', axLabel));
            ylabel(sprintf('%s 2', axLabel));
            zlabel('Density');
            title(titleStr);
            colormap(hAx, brighten(parula(256), surfBrighten));
            colorbar;
            xlim([axMin axMax]);
            ylim([axMin axMax]);
            maxV = max(applyTransform(vals, 'off', gamma, eta));
            if maxV > 0
                daspect([1 1 maxV / (axMax - axMin)]);
            end
            set(hAx, 'Projection', 'orthographic');
            view(0, 90);
            setDemoTicks(hAx, [axMin axMax], 2);

            % Opacity following the density, over dark panes, as the
            % three-dimensional plots do. A surface painted opaque
            % covers its whole plane in the map's low colour, so the
            % ground a density is read against is that colour rather
            % than the background, and brightening the map lifts it
            % along with everything else. Let the low material fade
            % out instead and the two kinds of picture agree.
            % Texture mapping for both the colour and the opacity.
            % FaceColor and FaceAlpha have to agree, and of the pairs
            % that do, only this one renders: per-vertex opacity over
            % a grid this size comes out blank. It is also what
            % plotMaet3d's slices use, for the same reason.
            set(hSurf, 'FaceColor', 'texturemap', ...
                       'FaceAlpha', 'texturemap', ...
                       'AlphaData', surfAlpha(V), ...
                       'AlphaDataMapping', 'none');
            set(hAx, 'ALim', [0 1], ...
                     'Color', [0.06 0.06 0.06], ...
                     'GridColor', [0.22 0.22 0.22], 'GridAlpha', 1, ...
                     'XGrid', 'on', 'YGrid', 'on', 'ZGrid', 'on');

            % Store raw data and add controls
            info.mode    = 'surf';
            info.rawVals = vals;
            info.hSurf   = hSurf;
            info.hAx     = hAx;
            info.res     = res;
            info.axRange = [axMin axMax];
            addPlotControls(fig, info, gamma, eta);

        % =============================================================
        %  dim = 3: drawn by the toolbox's own plotMaet3d
        % =============================================================
        case 3
            fig = newDemoFigure(ci, r, dim, dockFigures);
            h3 = plotMaet3d(dens, 'method', plot3Dmode, ...
                            'limits', [axMin axMax], 'step', stepSize, ...
                            'upsample', 2);
            xlabel(sprintf('%s 1', axLabel));
            ylabel(sprintf('%s 2', axLabel));
            zlabel(sprintf('%s 3', axLabel));
            title(sprintf('%s — %s', titleStr, plot3Dmode));
            setDemoTicks(gca, [axMin axMax], 3);

            % These are meant to be turned, so the figure opens ready
            % to. plotMaet3d leaves the interaction mode alone, that
            % being the caller's to set rather than a plotting
            % function's to impose.
            rotate3d(fig, 'on');

            % The transform controls apply to 'points' alone.
            if strcmp(plot3Dmode, 'points')
                info.mode     = 'scatter3';
                info.rawVals  = h3.AlphaData(:);
                info.hScatter = h3;
                addPlotControls(fig, info, gamma, eta);
            end

        % =============================================================
        %  dim >= 4: not drawn
        % =============================================================
        otherwise
            error('demo_maetPlots:tooManyDimensions', ...
                  ['Config %d has dim = %d. Only one to three ' ...
                   'dimensions are drawn: a four-dimensional density ' ...
                   'has no honest picture.'], ci, dim);

    end

    fprintf(' done.\n');
end

fprintf('All plots complete.\n');


%% === Helper functions ===

function setDemoTicks(ax, lims, dim)
%SETDEMOTICKS Ticks at the smallest tidy interval that is not crowded.
%
%  The interval is one of 100, 200, 300, 400, or 600, so that the
%  labels fall on musically legible values and every plot is read the
%  same way. The smallest of those is taken that still leaves each
%  label room, which depends on the axes as drawn rather than on the
%  range alone: a docked tab is narrower than a window, and a
%  three-dimensional cube gives each of its axes a fraction of the box.
%
%  One interval serves every drawn axis, the largest any of them needs.
%  The axes cover the same range as each other, so ticking them
%  differently would make a square plot read as though they did not.
    wh = demoAxesPoints(ax);
    span = diff(lims);

    % A label such as -1200 is five characters wide, so tick labels
    % along a horizontal axis need far more room than the stacked
    % labels of a vertical one.
    switch dim
        case 1
            avail = wh(1);
            room  = 50;
        case 2
            avail = [wh(1), wh(2)];
            room  = [50, 30];
        otherwise
            % No orientation projects the cube wider than its space
            % diagonal, and all three axes share the box.
            avail = [1 1 1] * min(wh) / sqrt(3);
            room  = [50 50 50];
    end

    step = 0;
    for k = 1:numel(avail)
        step = max(step, tidyTickStep(avail(k), span, room(k)));
    end
    ticks = lims(1):step:lims(2);

    names = {'XTick', 'YTick', 'ZTick'};
    for k = 1:numel(avail)
        set(ax, names{k}, ticks);
    end
end


function a = surfAlpha(V)
%SURFALPHA Opacity for a surface, the density against its own peak.
%
%  Proportional to the density, as a mark's is in plotMaet3d at its
%  default alphaGamma of 1. Clamped at zero: evaluation can return
%  values a billionth below it.
    mx = max(V(:));
    if mx > 0
        a = max(V, 0) / mx;
    else
        a = ones(size(V));
    end
end


function step = tidyTickStep(availPts, span, minPts)
%TIDYTICKSTEP The smallest tidy interval whose labels still have room.
    candidates = [100 200 300 400 600];
    for k = 1:numel(candidates)
        nLabels = span / candidates(k) + 1;
        if availPts / nLabels >= minPts
            step = candidates(k);
            return
        end
    end
    step = candidates(end);
end


function wh = demoAxesPoints(ax)
%DEMOAXESPOINTS The axes box in points, whatever its Units are set to.
    drawnow limitrate
    px = getpixelposition(ax, true);
    wh = px(3:4) * 72 / get(groot, 'ScreenPixelsPerInch');
end


function fig = newDemoFigure(ci, r, dim, docked)
%NEWDEMOFIGURE One figure per configuration, docked or free.
%
%  Docking is set on every figure rather than left to MATLAB's own
%  preference, so that the configurations arrive the same way whatever
%  that preference is.
    if docked
        style = 'docked';
    else
        style = 'normal';
    end
    fig = figure('Name', sprintf('Config %d: r=%d dim=%d', ci, r, dim), ...
                 'WindowStyle', style);
end


function vt = applyTransform(vals, mode, gamma, eta)
%APPLYTRANSFORM Dispatch on mode.
%
%  'off'   identity; output range = input range.
%  'gamma' power compression: data normalised by the empirical
%          (min, max), then raised to gamma. Output is in [0, 1].
%          Gamma is a display-cosmetic knob, so anchoring at the
%          empirical min keeps the slider responsive regardless of
%          where the data sits.
%  'sat'   saturation: anchored at 0 (a meaningful baseline of "no
%          density"). For tensor density, which is always non-
%          negative, vn = vals / max. The saturation curve
%          (1 - exp(-vn/eta)) / (1 - exp(-1/eta)) is then applied.
%          Output is in [0, 1].
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

function addPlotControls(fig, info, gammaInit, etaInit)
%ADDPLOTCONTROLS Add interactive controls to a figure.
%
%  All plot modes get a transform-mode selector (radio buttons:
%  Off / Gamma / Saturation) and an adaptive slider whose meaning
%  depends on the chosen mode. 'gamma' applies v -> v.^gamma in
%  [0.01, 1]; 'sat' applies v -> 1 - exp(-v / eta) with eta on a
%  log10 scale in [0.001, 5]. Both gamma and eta have per-mode
%  memory.
%
%  For 'surf' mode, the controls are placed vertically to the right
%  of the colorbar. There is also a colormap shift slider (with
%  per-mode memory) and a perspective/orthographic projection toggle.
%  In perspective mode the colorbar and slider region shift right to
%  clear the y-axis labels.
%
%  For the other modes, the controls are placed in a horizontal row at
%  the bottom of the figure.
%
%  Supported info.mode values:
%    'line'     — updates YData of a line plot
%    'surf'     — updates ZData/CData; includes projection + cmap controls
%    'scatter3' — updates CData and AlphaData of a scatter3 plot

    isSurf = strcmp(info.mode, 'surf');

    info.mode_xform = 'off';        % active transform mode
    info.gamma      = gammaInit;    % per-mode memory
    info.eta        = etaInit;      % per-mode memory

    if isSurf
        % === Surf mode: vertical layout to the right of the colorbar ===

        drawnow;

        % Position the colorbar narrowly so perspective mode has room
        hCB = findobj(fig, 'Type', 'ColorBar');
        if ~isempty(hCB)
            axPos = get(info.hAx, 'Position');
            cbLeft   = axPos(1) + axPos(3) + 0.14;
            cbBottom = axPos(2);
            cbWidth  = 0.018;
            cbHeight = axPos(4);
            hCB(1).Location = 'manual';
            hCB(1).Position = [cbLeft, cbBottom, cbWidth, cbHeight];
            cbPos = hCB(1).Position;
        else
            cbPos = [0.72, 0.12, 0.018, 0.78];
        end

        sliderW    = 0.025;
        labelH     = 0.025;
        readoutH   = 0.025;
        cbGap      = 0.055;
        sliderGap  = 0.015;
        modeGroupH = 0.025;
        labelTopH  = 0.020;
        labelBotH  = 0.020;
        modeBlockH = labelTopH + modeGroupH + labelBotH + 0.005;

        % Slider region anchored to the colorbar
        sliderH    = cbPos(4) - modeBlockH - 0.005;
        sliderBot  = cbPos(2);
        xformX     = cbPos(1) + cbPos(3) + cbGap;
        modeBgW    = 2 * sliderW + sliderGap + 0.01;
        modeBlockY = sliderBot + sliderH + labelH + 0.005;

        % Per-mode cmap shift memory
        info.cmapShiftOff   = 0;
        info.cmapShiftGamma = 0;
        info.cmapShiftSat   = 0;

        % Top labels: "Off" above column 1, "Saturation" above column 3
        topLabelY = modeBlockY + modeGroupH + labelBotH - 0.005;
        uicontrol(fig, 'Style', 'text', 'String', 'Off', ...
            'Units', 'normalized', ...
            'Position', [xformX - 0.020, topLabelY, 0.040, labelTopH], ...
            'FontSize', 8, 'HorizontalAlignment', 'center', ...
            'Tag', 'modeOff_label', ...
            'BackgroundColor', get(fig, 'Color'));
        uicontrol(fig, 'Style', 'text', 'String', 'Saturation', ...
            'Units', 'normalized', ...
            'Position', [xformX + modeBgW - 0.045, topLabelY, ...
                         0.060, labelTopH], ...
            'FontSize', 8, 'HorizontalAlignment', 'center', ...
            'Tag', 'modeSat_label', ...
            'BackgroundColor', get(fig, 'Color'));

        % Radio row
        radioY = modeBlockY + labelBotH;
        modeGroup = uibuttongroup(fig, ...
            'Units', 'normalized', ...
            'Position', [xformX - 0.005, radioY, modeBgW, modeGroupH], ...
            'BorderType', 'none', ...
            'BackgroundColor', get(fig, 'Color'), ...
            'Tag', 'modeGroup', ...
            'SelectionChangedFcn', @(src, evt) modeChangedCallback(src, evt, fig));
        uicontrol(modeGroup, 'Style', 'radiobutton', 'String', '', ...
            'Units', 'normalized', 'Position', [0.05, 0, 0.28, 1], ...
            'Tag', 'modeOff', 'BackgroundColor', get(fig, 'Color'), ...
            'Value', 1);
        uicontrol(modeGroup, 'Style', 'radiobutton', 'String', '', ...
            'Units', 'normalized', 'Position', [0.39, 0, 0.28, 1], ...
            'Tag', 'modeGamma', 'BackgroundColor', get(fig, 'Color'), ...
            'Value', 0);
        uicontrol(modeGroup, 'Style', 'radiobutton', 'String', '', ...
            'Units', 'normalized', 'Position', [0.72, 0, 0.28, 1], ...
            'Tag', 'modeSat', 'BackgroundColor', get(fig, 'Color'), ...
            'Value', 0);

        % "Gamma" label below column 2
        botLabelY = modeBlockY;
        uicontrol(fig, 'Style', 'text', 'String', 'Gamma', ...
            'Units', 'normalized', ...
            'Position', [xformX + modeBgW/2 - 0.026, botLabelY, ...
                         0.040, labelBotH], ...
            'FontSize', 8, 'HorizontalAlignment', 'center', ...
            'Tag', 'modeGamma_label', ...
            'BackgroundColor', get(fig, 'Color'));

        % Transform slider (initial: 'off' -> disabled)
        uicontrol(fig, 'Style', 'slider', ...
            'Min', 0.01, 'Max', 1, 'Value', gammaInit, ...
            'Units', 'normalized', ...
            'Position', [xformX, sliderBot, sliderW, sliderH], ...
            'Tag', 'xformSlider', 'Enable', 'off', ...
            'SliderStep', [0.005, 0.03], ...
            'Callback', @(src, ~) xformSliderCallback(src, fig));
        uicontrol(fig, 'Style', 'text', 'String', '', ...
            'Units', 'normalized', ...
            'Position', [xformX - 0.005, sliderBot - readoutH - 0.002, ...
                sliderW + 0.01, readoutH], ...
            'FontSize', 8, ...
            'Tag', 'xformReadout', ...
            'HorizontalAlignment', 'center', ...
            'BackgroundColor', get(fig, 'Color'));

        % Cmap slider
        cmapX = xformX + sliderW + sliderGap;
        uicontrol(fig, 'Style', 'slider', ...
            'Min', 0, 'Max', 0.95, 'Value', 0, ...
            'Units', 'normalized', ...
            'Position', [cmapX, sliderBot, sliderW, sliderH], ...
            'Tag', 'cmapShiftSlider', ...
            'SliderStep', [0.005, 0.03], ...
            'Callback', @(src, ~) cmapShiftCallback(src, fig));
        uicontrol(fig, 'Style', 'text', 'String', 'Cmap', ...
            'Units', 'normalized', ...
            'Position', [cmapX - 0.01, sliderBot + sliderH + 0.002, ...
                sliderW + 0.02, labelH], ...
            'FontSize', 8, 'HorizontalAlignment', 'center', ...
            'Tag', 'cmapShiftLabel', ...
            'BackgroundColor', get(fig, 'Color'));
        uicontrol(fig, 'Style', 'text', 'String', '0.00', ...
            'Units', 'normalized', ...
            'Position', [cmapX - 0.005, sliderBot - readoutH - 0.002, ...
                sliderW + 0.01, readoutH], ...
            'FontSize', 8, ...
            'Tag', 'cmapShiftReadout', ...
            'HorizontalAlignment', 'center', ...
            'BackgroundColor', get(fig, 'Color'));

        % Projection toggle
        toggleW = cmapX + sliderW - xformX;
        toggleH = 0.035;
        toggleY = sliderBot - readoutH - toggleH - 0.01;
        uicontrol(fig, 'Style', 'togglebutton', 'String', 'Perspective', ...
            'Units', 'normalized', ...
            'Position', [xformX, toggleY, toggleW, toggleH], ...
            'FontSize', 8, ...
            'Tag', 'projToggle', 'Value', 0, ...
            'Callback', @(src, ~) projCallback(src, fig));

        % Save initial colorbar position and slider-region X positions
        % so projCallback can shift them in perspective mode.
        info.hCbar = hCB(1);
        info.cbarPosOrtho = hCB(1).Position;
        sliderTags = {'modeGroup', 'modeOff_label', 'modeSat_label', ...
                       'modeGamma_label', 'xformSlider', 'xformReadout', ...
                       'cmapShiftSlider', 'cmapShiftLabel', ...
                       'cmapShiftReadout', 'projToggle'};
        info.sliderTags    = sliderTags;
        info.sliderXOrtho  = cell(1, numel(sliderTags));
        for ti = 1:numel(sliderTags)
            h = findobj(fig, 'Tag', sliderTags{ti});
            xs = zeros(numel(h), 1);
            for hi = 1:numel(h)
                p = get(h(hi), 'Position');
                xs(hi) = p(1);
            end
            info.sliderXOrtho{ti} = xs;
        end

    else
        % === Non-surf modes: horizontal row at bottom ===
        % The controls sit in normalized units, so a docked figure
        % needs no extra height and would ignore the request anyway.
        if strcmp(get(fig, 'WindowStyle'), 'normal')
            figPos = get(fig, 'Position');
            set(fig, 'Position', [figPos(1), figPos(2), figPos(3), ...
                                  figPos(4) + 50]);
        end

        rowH = 0.030;
        y_xform = 0.015;        % the one row, reserved for transform UI

        % Mode-selector buttongroup on the left, with full inline labels
        modeBgX = 0.04;
        modeBgW = 0.28;
        modeGroup = uibuttongroup(fig, ...
            'Units', 'normalized', ...
            'Position', [modeBgX, y_xform, modeBgW, rowH], ...
            'BorderType', 'none', ...
            'BackgroundColor', get(fig, 'Color'), ...
            'Tag', 'modeGroup', ...
            'SelectionChangedFcn', @(src, evt) modeChangedCallback(src, evt, fig));
        uicontrol(modeGroup, 'Style', 'radiobutton', 'String', 'Off', ...
            'Units', 'normalized', 'Position', [0, 0, 1/3, 1], ...
            'Tag', 'modeOff', 'BackgroundColor', get(fig, 'Color'), ...
            'FontSize', 8, 'Value', 1);
        uicontrol(modeGroup, 'Style', 'radiobutton', 'String', 'Gamma', ...
            'Units', 'normalized', 'Position', [1/3, 0, 1/3, 1], ...
            'Tag', 'modeGamma', 'BackgroundColor', get(fig, 'Color'), ...
            'FontSize', 8, 'Value', 0);
        uicontrol(modeGroup, 'Style', 'radiobutton', 'String', 'Sat', ...
            'Units', 'normalized', 'Position', [2/3, 0, 1/3, 1], ...
            'Tag', 'modeSat', 'BackgroundColor', get(fig, 'Color'), ...
            'FontSize', 8, 'Value', 0);

        % Transform slider (initially disabled)
        uicontrol(fig, 'Style', 'slider', ...
            'Min', 0.01, 'Max', 1, 'Value', gammaInit, ...
            'Units', 'normalized', ...
            'Position', [modeBgX + modeBgW + 0.04, y_xform, 0.50, rowH], ...
            'Tag', 'xformSlider', 'Enable', 'off', ...
            'SliderStep', [0.005, 0.03], ...
            'Callback', @(src, ~) xformSliderCallback(src, fig));

        uicontrol(fig, 'Style', 'text', 'String', '', ...
            'Units', 'normalized', ...
            'Position', [modeBgX + modeBgW + 0.55, y_xform - 0.002, ...
                         0.10, rowH], ...
            'FontSize', 8, ...
            'Tag', 'xformReadout', ...
            'HorizontalAlignment', 'left', ...
            'BackgroundColor', get(fig, 'Color'));

    end

    % Store the plot info in the figure's application data
    setappdata(fig, 'plotInfo', info);

    % === Callbacks (nested) ===

    function modeChangedCallback(~, evt, fig)
        pInfo = getappdata(fig, 'plotInfo');
        hSlider     = findobj(fig, 'Tag', 'xformSlider');
        hReadout    = findobj(fig, 'Tag', 'xformReadout');
        hCmap       = findobj(fig, 'Tag', 'cmapShiftSlider');
        hCmapRdout  = findobj(fig, 'Tag', 'cmapShiftReadout');

        % Save outgoing slider value (and cmap shift, if in surf mode)
        switch pInfo.mode_xform
            case 'gamma'
                pInfo.gamma = get(hSlider, 'Value');
                if ~isempty(hCmap)
                    pInfo.cmapShiftGamma = get(hCmap, 'Value');
                end
            case 'sat'
                pInfo.eta = 10 ^ get(hSlider, 'Value');
                if ~isempty(hCmap)
                    pInfo.cmapShiftSat = get(hCmap, 'Value');
                end
            case 'off'
                if ~isempty(hCmap)
                    pInfo.cmapShiftOff = get(hCmap, 'Value');
                end
        end

        switch evt.NewValue.Tag
            case 'modeOff',   newMode = 'off';
            case 'modeGamma', newMode = 'gamma';
            case 'modeSat',   newMode = 'sat';
            otherwise,        newMode = 'off';
        end
        pInfo.mode_xform = newMode;

        switch newMode
            case 'off'
                set(hSlider, 'Enable', 'off');
                set(hReadout, 'String', '');
            case 'gamma'
                newMin = 0.01; newMax = 1.0;
                cur = get(hSlider, 'Value');
                set(hSlider, 'Value', max(min(cur, newMax), newMin));
                set(hSlider, 'Min', newMin, 'Max', newMax);
                set(hSlider, 'Value', pInfo.gamma);
                set(hSlider, 'Enable', 'on');
                set(hReadout, 'String', sprintf('%.2f', pInfo.gamma));
            case 'sat'
                newMin = log10(0.002); newMax = log10(5);
                cur = get(hSlider, 'Value');
                set(hSlider, 'Value', max(min(cur, newMax), newMin));
                set(hSlider, 'Min', newMin, 'Max', newMax);
                set(hSlider, 'Value', log10(pInfo.eta));
                set(hSlider, 'Enable', 'on');
                set(hReadout, 'String', sprintf('%.3f', pInfo.eta));
        end

        % Restore incoming mode's cmap shift (surf only)
        if ~isempty(hCmap)
            switch newMode
                case 'off',   newCmap = pInfo.cmapShiftOff;
                case 'gamma', newCmap = pInfo.cmapShiftGamma;
                case 'sat',   newCmap = pInfo.cmapShiftSat;
            end
            set(hCmap, 'Value', newCmap);
            set(hCmapRdout, 'String', sprintf('%.2f', newCmap));
        end

        setappdata(fig, 'plotInfo', pInfo);
        applyTransformToPlot(fig);
    end

    function xformSliderCallback(src, fig)
        pInfo = getappdata(fig, 'plotInfo');
        hReadout = findobj(fig, 'Tag', 'xformReadout');

        switch pInfo.mode_xform
            case 'gamma'
                pInfo.gamma = get(src, 'Value');
                set(hReadout, 'String', sprintf('%.2f', pInfo.gamma));
            case 'sat'
                pInfo.eta = 10 ^ get(src, 'Value');
                set(hReadout, 'String', sprintf('%.3f', pInfo.eta));
            otherwise
                return;
        end

        setappdata(fig, 'plotInfo', pInfo);
        applyTransformToPlot(fig);
    end

    function applyTransformToPlot(fig)
    %APPLYTRANSFORMTOPLOT Apply the current transform to all plot elements.
        pInfo = getappdata(fig, 'plotInfo');
        m = pInfo.mode_xform;
        g = pInfo.gamma;
        e = pInfo.eta;

        switch pInfo.mode
            case 'line'
                set(pInfo.hLine, 'YData', applyTransform(pInfo.rawVals, m, g, e));
            case 'surf'
                Vt = reshape(applyTransform(pInfo.rawVals, m, g, e), ...
                    pInfo.res, pInfo.res);
                set(pInfo.hSurf, 'ZData', Vt, 'CData', Vt, ...
                    'AlphaData', surfAlpha(Vt));
                maxVt = max(Vt(:));
                if maxVt > 0
                    axR = pInfo.axRange;
                    daspect(pInfo.hAx, [1 1 maxVt / (axR(2) - axR(1))]);
                end
                hShift = findobj(fig, 'Tag', 'cmapShiftSlider');
                if ~isempty(hShift)
                    cmapShiftCallback(hShift, fig);
                end
            case 'scatter3'
                vT = applyTransform(pInfo.rawVals, m, g, e);
                % In 'off' mode, vT is in the input's native range;
                % normalise to [0, 1] for color/alpha. In 'gamma'/'sat'
                % modes vT is already in [0, 1] so use directly.
                if strcmp(m, 'off')
                    M = max(pInfo.rawVals(:));
                    if M > 0
                        vT = pInfo.rawVals / M;
                    else
                        vT = pInfo.rawVals;
                    end
                end
                set(pInfo.hScatter, 'CData', vT, ...
                    'SizeData', 10 * ones(size(vT)));
                pInfo.hScatter.AlphaData = vT;
        end

        drawnow;
    end

    function cmapShiftCallback(src, fig)
        shiftFrac = get(src, 'Value');
        hReadout = findobj(fig, 'Tag', 'cmapShiftReadout');
        set(hReadout, 'String', sprintf('%.2f', shiftFrac));

        pInfo = getappdata(fig, 'plotInfo');
        switch pInfo.mode_xform
            case 'off',   pInfo.cmapShiftOff   = shiftFrac;
            case 'gamma', pInfo.cmapShiftGamma = shiftFrac;
            case 'sat',   pInfo.cmapShiftSat   = shiftFrac;
        end
        setappdata(fig, 'plotInfo', pInfo);

        cdata = get(pInfo.hSurf, 'CData');
        minC  = min(cdata(:));
        maxC  = max(cdata(:));
        if maxC > minC
            newLow = minC + shiftFrac * (maxC - minC);
            set(pInfo.hAx, 'CLim', [newLow, maxC]);
        end
        drawnow;
    end

    function projCallback(src, fig)
        pInfo = getappdata(fig, 'plotInfo');

        if get(src, 'Value') == 1
            proj = 'perspective';
            label = 'Orthographic';
            cbarShift   = 0.030;
            sliderShift = cbarShift;
        else
            proj = 'orthographic';
            label = 'Perspective';
            cbarShift   = 0;
            sliderShift = 0;
        end
        set(src, 'String', label);

        set(pInfo.hAx, 'Projection', proj);
        cbPos = pInfo.cbarPosOrtho;
        cbPos(1) = cbPos(1) + cbarShift;
        set(pInfo.hCbar, 'Position', cbPos);

        % Shift slider region
        for ti = 1:numel(pInfo.sliderTags)
            h = findobj(fig, 'Tag', pInfo.sliderTags{ti});
            for hi = 1:numel(h)
                p = get(h(hi), 'Position');
                p(1) = pInfo.sliderXOrtho{ti}(hi) + sliderShift;
                set(h(hi), 'Position', p);
            end
        end

        drawnow;
    end

end