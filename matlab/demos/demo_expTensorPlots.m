%% demo_expTensorPlots.m
%  Visualizes expectation tensor densities in 1 to 4 dimensions for
%  user-specified combinations of r, isRel, and isPer.
%
%  Uses buildExpTens to precompute the density object once per configuration,
%  then passes it to evalExpTens.
%
%  Each figure includes an interactive power slider for real-time
%  adjustment of the dynamic range compression: displayed = data.^gamma.
%  Surface plots (dim = 2) additionally include a colormap shift slider
%  and a perspective/orthographic projection toggle.
%
%  Edit the parameters below to experiment with different pitch sets,
%  smoothing widths, plot configurations, and visualization modes.
%
%  Uses: buildExpTens, evalExpTens (from the Music Perception Toolbox).

%% === User-editable parameters ===

% Pitch set and weights
p = [0; 200; 400; 500; 700; 900; 1100];
w = [];

% Gaussian smoothing width
sigma = 10;

% Normalization mode for density evaluation: 'none', 'gaussian', or 'pdf'
%   'none'     — Raw weighted proximity scores (default). Only relative
%                values across query points are meaningful.
%   'gaussian' — Each Gaussian component integrates to 1. Useful for
%                comparing densities computed with different sigma values.
%   'pdf'      — Full probability density (integrates to 1 over the domain).
%                Useful for comparing across pitch sets of different sizes.
normalize = 'none';

% Gamma (power compression) for visualization: displayed = data.^gamma
%   gamma = 1 shows the original data (no compression).
%   gamma < 1 compresses the dynamic range, making lower-density regions
%   more visible by reducing the dominance of high peaks.
%   gamma -> 0 tends toward a binary (nonzero / zero) representation.
%   An interactive slider is added to each figure for real-time adjustment.
gamma = 1;

% Period for periodic configurations (in the same units as p)
period = 1200;

% === Plot configurations ===
% Each row specifies one plot: [r, isRel, isPer]
%   r     — tuple size
%   isRel — 0 = absolute, 1 = relative (transposition-invariant)
%   isPer — 0 = non-periodic, 1 = periodic
%
% The effective query dimensionality is dim = r - isRel.
% Add, remove, or reorder rows to control which plots are produced.
configs = [
1,  0,  0;
1,  0,  1;
2,  0,  0;
2,  0,  1;
2,  1,  0;
2,  1,  1;
3,  0,  0;
3,  0,  1;
3,  1,  0;
3,  1,  1;
4,  0,  0;
4,  0,  1;
4,  1,  0;
4,  1,  1;
];

% === Grid resolution as step size ===
% Specify the grid spacing (in the same units as p and period) for each
% effective dimensionality. The number of grid points per axis is computed
% automatically from the axis range and step size.
% Smaller step = finer grid = slower computation (scales as step^(-dim)).
step_1d = 1;     % e.g., 1 cent per grid point
step_2d = 5;     % e.g., 5 cents per dimension
step_3d = 20;     % e.g., 20 cents per dimension
step_4d = 50;    % e.g., 50 cents per dimension

% === Axis range for non-periodic configurations ===
% For periodic configurations, the range is always [0, period].
% For non-periodic configurations, set the range here.
axMinNonPer = 0;
axMaxNonPer = 1200;

% === 3D visualization settings ===

% Isosurface threshold (fraction of max density)
isoFrac = 0.3;

% 3D plot mode: 'isosurface', 'volumetric', or 'scatter'
%   'isosurface'  — Single isosurface at isoFrac of max (fast, clean).
%                   Note: power slider is not available for this mode
%                   (would require recomputing isosurfaces on each change).
%   'volumetric'  — Stacked isosurfaces at graded alpha levels. Same
%                   limitation as isosurface regarding the power slider.
%   'scatter'     — scatter3 with per-point alpha mapped to density value
%                   (most literal, but slower and noisier). Power slider
%                   updates color and alpha in real time.
% All three modes set explicit axis limits to [axMin, axMax].
plot3Dmode = 'scatter';

% Number of isosurface layers for 'volumetric' mode (more = smoother)
nIsoLayers = 8;

% Minimum density threshold for 'scatter' mode (fraction of max).
% Points below this are not plotted, to reduce clutter and speed things up.
scatterThreshFrac = 0.05;


%% === Estimate total runtime ===

nConfigs = size(configs, 1);
totalEstSec = 0;

fprintf('\n--- Plot summary ---\n');
for ci = 1:nConfigs
    rc      = configs(ci, 1);
    isRelC  = logical(configs(ci, 2));
    isPerC  = logical(configs(ci, 3));

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
        case 3, stepC = step_3d;
        otherwise, stepC = step_4d;
    end
    resC = max(2, round((axMaxC - axMinC) / stepC) + 1);

    % Number of query points = res^dim (for dims 1-3)
    % For dim >= 4, the 2D slice approach uses res^2 per slice
    if dimC <= 3
        nQc = double(resC)^dimC;
    else
        % Estimate number of slices (same logic as main loop)
        if isRelC
            nFixedVals = min(3, numel(unique(diff(sort(p)))));
        else
            nFixedVals = min(3, numel(p));
        end
        nExtra   = dimC - 2;
        nSlices  = nFixedVals^nExtra;
        nQc      = double(resC)^2 * nSlices;
    end

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
    fprintf('plotExpTens (total): estimated time ~%s (Ctrl+C to cancel).\n\n', ...
        totalTimeStr);
else
    fprintf('plotExpTens (total): estimated time ~%s.\n\n', totalTimeStr);
end


%% === Iterate through configurations ===

for ci = 1:nConfigs
    r      = configs(ci, 1);
    isRelR = logical(configs(ci, 2));
    isPerR = logical(configs(ci, 3));

    % --- Validation ---
    if r > numel(p)
        fprintf('Config %d: r = %d skipped (pitch set has only %d elements).\n', ...
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
        case 3, stepSize = step_3d;
        otherwise, stepSize = step_4d;
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

    titleStr = sprintf('r = %d, %s, %s, \\sigma = %.2f', ...
        r, modeStr, perStr, sigma);

    fprintf('Config %d: r = %d (%s, %s, dim = %d, res = %d): precomputing...', ...
        ci, r, modeStr, perStr, dim, res);

    % --- Precompute the density object ---
    dens = buildExpTens(p, w, sigma, r, isRelR, isPerR, period);

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

            vals = evalExpTens(dens, X, normalize);

            fig = figure('Name', sprintf('Config %d: r=%d dim=%d', ci, r, dim));
            hLine = plot(x, applyGamma(vals, gamma), 'LineWidth', 1.5);
            xlabel(sprintf('%s 1', axLabel));
            ylabel('Density');
            title(titleStr);
            if isPerR
                xlim([axMin axMax]);
            end
            grid on;

            % Store raw data and add slider
            info.mode    = 'line';
            info.rawVals = vals;
            info.hLine   = hLine;
            addPlotControls(fig, info, gamma);

        % =============================================================
        %  dim = 2: surface plot (top-down X-Y view) with controls
        % =============================================================
        case 2
            x = linspace(axMin, axMax, res);
            [Ga, Gb] = meshgrid(x, x);

            % Exploit symmetry: density at (a,b) = density at (b,a).
            % Evaluate only the upper triangle (including diagonal),
            % then mirror to fill the full matrix.
            upperMask = triu(true(res));
            Xu = [Ga(upperMask)'; Gb(upperMask)'];

            valsU = evalExpTens(dens, Xu, normalize);

            Vraw = zeros(res, res);
            Vraw(upperMask) = valsU;
            Vraw = Vraw + Vraw.' - diag(diag(Vraw));
            vals = Vraw(:).';  % 1 x res^2 for compatibility with rawVals

            V = reshape(applyGamma(vals, gamma), res, res);

            fig = figure('Name', sprintf('Config %d: r=%d dim=%d', ci, r, dim));

            % Widen figure to accommodate plot + colorbar + controls
            figPos = get(fig, 'Position');
            set(fig, 'Position', [figPos(1), figPos(2), ...
                max(figPos(3), 900), figPos(4)]);

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
            colorbar;
            xlim([axMin axMax]);
            ylim([axMin axMax]);
            maxV = max(applyGamma(vals, gamma));
            if maxV > 0
                daspect([1 1 maxV / (axMax - axMin)]);
            end
            set(hAx, 'Projection', 'orthographic');
            view(0, 90);

            % Store raw data and add controls
            info.mode    = 'surf';
            info.rawVals = vals;
            info.hSurf   = hSurf;
            info.hAx     = hAx;
            info.res     = res;
            info.axRange = [axMin axMax];
            addPlotControls(fig, info, gamma);

        % =============================================================
        %  dim = 3: volumetric / isosurface / scatter
        % =============================================================
        case 3
            x = linspace(axMin, axMax, res);
            [Ga, Gb, Gc] = ndgrid(x, x, x);
            X = [Ga(:)'; Gb(:)'; Gc(:)'];  % 3 x (res^3)

            vals = evalExpTens(dens, X, normalize);
            V = reshape(vals, res, res, res);
            maxVal = max(vals);

            fig = figure('Name', sprintf('Config %d: r=%d dim=%d', ci, r, dim));

            switch plot3Dmode

                case 'isosurface'
                    isoVal = isoFrac * maxVal;
                    ptch = patch(isosurface(x, x, x, V, isoVal));
                    set(ptch, 'FaceColor', [0.2 0.5 0.8], ...
                        'EdgeColor', 'none', 'FaceAlpha', 0.6);
                    lighting gouraud;
                    camlight headlight;
                    title(sprintf('%s — isosurface at %.0f%%', ...
                        titleStr, isoFrac * 100));

                case 'volumetric'
                    thresholds = linspace(0.05, 0.95, nIsoLayers);
                    alphas     = linspace(0.05, 0.6, nIsoLayers);
                    cmap       = parula(nIsoLayers);

                    for li = 1:nIsoLayers
                        isoVal = thresholds(li) * maxVal;
                        fv = isosurface(x, x, x, V, isoVal);
                        if isempty(fv.vertices)
                            continue;
                        end
                        ptch = patch(fv);
                        set(ptch, ...
                            'FaceColor', cmap(li, :), ...
                            'EdgeColor', 'none', ...
                            'FaceAlpha', alphas(li));
                    end
                    lighting gouraud;
                    camlight headlight;
                    title(sprintf('%s — volumetric (%d layers)', ...
                        titleStr, nIsoLayers));

                case 'scatter'
                    thresh = scatterThreshFrac * maxVal;
                    mask   = vals > thresh;
                    gx     = Ga(mask);  gx = gx(:);
                    gy     = Gb(mask);  gy = gy(:);
                    gz     = Gc(mask);  gz = gz(:);
                    vMask  = vals(mask); vMask = vMask(:);

                    vGamma = applyGamma(vMask, gamma);
                    vNorm  = normalizeForDisplay(vGamma);

                    sc = scatter3(gx, gy, gz, 10, vNorm, 'filled');
                    sc.MarkerFaceAlpha = 'flat';
                    sc.AlphaData = vNorm;

                    colormap(gca, parula);
                    colorbar;
                    title(sprintf('%s — scatter', titleStr));

                otherwise
                    error('Unknown plot3Dmode: ''%s''.', plot3Dmode);
            end

            xlabel(sprintf('%s 1', axLabel));
            ylabel(sprintf('%s 2', axLabel));
            zlabel(sprintf('%s 3', axLabel));
            xlim([axMin axMax]);
            ylim([axMin axMax]);
            zlim([axMin axMax]);
            daspect([1 1 1]);
            grid on;
            view([-30 30]);

            % Power slider for scatter mode only (isosurface/volumetric
            % would require regenerating patch objects, which is slow)
            if strcmp(plot3Dmode, 'scatter')
                info.mode    = 'scatter3';
                info.rawVals = vMask;
                info.hScatter = sc;
                addPlotControls(fig, info, gamma);
            end

        % =============================================================
        %  dim >= 4: grid of 2D slices (fix all but first two dims)
        % =============================================================
        otherwise
            x = linspace(axMin, axMax, res);
            [Ga, Gb] = meshgrid(x, x);

            % Choose fixed values for the extra dimensions
            if isRelR
                allIntervals = sort(unique(diff(sort(p))));
                if numel(allIntervals) >= 3
                    fixedVals = allIntervals(1:3)';
                else
                    fixedVals = linspace(axMin, axMax, 3);
                end
            else
                if numel(p) >= 3
                    fixedVals = p(1:min(3, numel(p)))';
                else
                    fixedVals = linspace(axMin, axMax, 3);
                end
            end

            % Number of extra dimensions beyond the first two
            nExtra = dim - 2;

            % Build all combinations of fixed values for extra dims
            fixedGrid = fixedVals(:);
            for d = 2:nExtra
                nPrev = size(fixedGrid, 1);
                nNew  = numel(fixedVals);
                fixedGrid = [repmat(fixedGrid, nNew, 1), ...
                    kron(fixedVals(:), ones(nPrev, 1))];
            end
            nSlices = size(fixedGrid, 1);

            % Determine subplot grid layout
            nCols = ceil(sqrt(nSlices));
            nRows = ceil(nSlices / nCols);

            fig = figure('Name', sprintf('Config %d: r=%d dim=%d', ci, r, dim));
            sgtitle(sprintf('%s — 2D slices', titleStr));

            % Collect raw data and image handles for the power slider
            allSliceVals   = cell(nSlices, 1);
            allSliceImages = gobjects(nSlices, 1);

            nPts = numel(Ga);
            for si = 1:nSlices
                Xq = [Ga(:)'; Gb(:)'];
                for d = 1:nExtra
                    Xq = [Xq; fixedGrid(si, d) * ones(1, nPts)]; %#ok<AGROW>
                end

                sliceVals = evalExpTens(dens, Xq, normalize);
                allSliceVals{si} = sliceVals;
                Vs = reshape(applyGamma(sliceVals, gamma), res, res);

                subplot(nRows, nCols, si);
                allSliceImages(si) = imagesc(x, x, Vs);
                axis xy equal tight;

                fixStr = '';
                for d = 1:nExtra
                    if d > 1
                        fixStr = [fixStr, ', ']; %#ok<AGROW>
                    end
                    fixStr = [fixStr, sprintf('%s %d=%.1f', ...
                        lower(axLabel), d + 2, fixedGrid(si, d))]; %#ok<AGROW>
                end
                title(fixStr, 'FontSize', 8);

                if si > (nRows - 1) * nCols
                    xlabel(sprintf('%s 1', axLabel));
                end
                if mod(si - 1, nCols) == 0
                    ylabel(sprintf('%s 2', axLabel));
                end
            end

            colormap(gca, 'parula');

            % Store raw data and add slider
            info.mode        = 'slices';
            info.rawVals     = allSliceVals;
            info.hImages     = allSliceImages;
            info.res         = res;
            addPlotControls(fig, info, gamma);
    end

    fprintf(' done.\n');
end

fprintf('All plots complete.\n');


%% === Helper functions ===

function vg = applyGamma(vals, gamma)
%APPLYGAMMA Apply power compression: data.^gamma, normalized to [0, 1].
%  Normalizes to [0, 1] first so that the gamma has a consistent effect
%  regardless of the absolute scale of the data.
    maxV = max(vals(:));
    if maxV > 0
        vg = (vals / maxV) .^ gamma * maxV;  % preserve original scale
    else
        vg = vals;
    end
end

function vn = normalizeForDisplay(vals)
%NORMALIZEFORDISPLAY Normalize to [0, 1] for color/alpha mapping.
    mn = min(vals(:));
    mx = max(vals(:));
    if mx > mn
        vn = (vals - mn) / (mx - mn);
    else
        vn = ones(size(vals));
    end
end

function addPlotControls(fig, info, gammaInit)
%ADDPLOTCONTROLS Add interactive controls to a figure.
%
%  Adds a power (gamma compression) slider to all plot types.
%  For 'surf' mode, also adds:
%    - A colormap shift slider that adjusts CLim
%    - A projection toggle (orthographic / perspective)
%
%  Layout:
%    For 'surf' mode, the two sliders are placed as vertical sliders to
%    the right of the colorbar (same height), with labels above and
%    readouts below. The projection toggle sits beneath the sliders.
%    For other modes, a horizontal power slider is placed at the bottom.
%
%  Supported info.mode values:
%    'line'     — updates YData of a line plot
%    'surf'     — updates ZData/CData; includes projection + cmap controls
%    'scatter3' — updates CData and AlphaData of a scatter3 plot
%    'slices'   — updates CData of multiple imagesc subplots

    isSurf = strcmp(info.mode, 'surf');

    if isSurf
        % === Surf mode: vertical sliders to the right of the colorbar ===

        % Force a draw so axes position is finalized
        drawnow;

        % Explicitly position the colorbar well to the right of the axes,
        % leaving room for 3D axis labels in perspective view
        hCB = findobj(fig, 'Type', 'ColorBar');
        if ~isempty(hCB)
            axPos = get(info.hAx, 'Position');
            % Place colorbar starting at 72% of figure width, aligned
            % vertically with the axes
            cbLeft   = axPos(1) + axPos(3) + 0.14;
            cbBottom = axPos(2);
            cbWidth  = 0.02;
            cbHeight = axPos(4);
            hCB(1).Location = 'manual';
            hCB(1).Position = [cbLeft, cbBottom, cbWidth, cbHeight];
            cbPos = hCB(1).Position;
        else
            cbPos = [0.72, 0.12, 0.02, 0.78];
        end

        % Slider dimensions: same height as colorbar, narrow, stacked to
        % the right. Each slider gets a label above and readout below.
        sliderW    = 0.025;
        labelH     = 0.03;
        readoutH   = 0.025;
        cbGap      = 0.055;  % gap between colorbar tick labels and first slider
        sliderGap  = 0.015;  % gap between the two sliders

        sliderH   = cbPos(4);
        sliderBot = cbPos(2);

        % Power slider: first column to the right of the colorbar
        powerX = cbPos(1) + cbPos(3) + cbGap;

        uicontrol(fig, 'Style', 'slider', ...
            'Min', 0.01, 'Max', 1, 'Value', gammaInit, ...
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
            'String', sprintf('%.2f', gammaInit), ...
            'Units', 'normalized', ...
            'Position', [powerX - 0.005, sliderBot - readoutH - 0.002, ...
                sliderW + 0.01, readoutH], ...
            'FontSize', 8, ...
            'Tag', 'powerReadout', ...
            'HorizontalAlignment', 'center', ...
            'BackgroundColor', get(fig, 'Color'));

        % Colormap shift slider: second column
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

        % Projection toggle: beneath the sliders, spanning both columns
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

    else
        % === Non-surf modes: horizontal power slider at the bottom ===

        figPos = get(fig, 'Position');
        set(fig, 'Position', [figPos(1), figPos(2), figPos(3), figPos(4) + 30]);

        rowH = 0.03;
        y1   = 0.01;

        uicontrol(fig, 'Style', 'slider', ...
            'Min', 0.01, 'Max', 1, 'Value', gammaInit, ...
            'Units', 'normalized', ...
            'Position', [0.15, y1, 0.55, rowH], ...
            'Tag', 'powerSlider', ...
            'Callback', @(src, ~) gammaCallback(src, fig));

        uicontrol(fig, 'Style', 'text', ...
            'String', 'Power:', ...
            'Units', 'normalized', ...
            'Position', [0.02, y1 - 0.002, 0.12, rowH], ...
            'HorizontalAlignment', 'right', ...
            'BackgroundColor', get(fig, 'Color'));

        uicontrol(fig, 'Style', 'text', ...
            'String', sprintf('%.2f', gammaInit), ...
            'Units', 'normalized', ...
            'Position', [0.72, y1 - 0.002, 0.08, rowH], ...
            'Tag', 'powerReadout', ...
            'HorizontalAlignment', 'left', ...
            'BackgroundColor', get(fig, 'Color'));
    end

    % Store the plot info in the figure's application data
    setappdata(fig, 'plotInfo', info);

    % === Callbacks ===

    function gammaCallback(src, fig)
        g = get(src, 'Value');

        hReadout = findobj(fig, 'Tag', 'powerReadout');
        set(hReadout, 'String', sprintf('%.2f', g));

        pInfo = getappdata(fig, 'plotInfo');

        switch pInfo.mode
            case 'line'
                set(pInfo.hLine, 'YData', applyGamma(pInfo.rawVals, g));

            case 'surf'
                Vg = reshape(applyGamma(pInfo.rawVals, g), ...
                    pInfo.res, pInfo.res);
                set(pInfo.hSurf, 'ZData', Vg, 'CData', Vg);
                maxVg = max(Vg(:));
                if maxVg > 0
                    axR = pInfo.axRange;
                    daspect(pInfo.hAx, [1 1 maxVg / (axR(2) - axR(1))]);
                end
                % Reapply colormap shift to the new gamma-transformed data
                hShift = findobj(fig, 'Tag', 'cmapShiftSlider');
                if ~isempty(hShift)
                    cmapShiftCallback(hShift, fig);
                end

            case 'scatter3'
                vGamma = applyGamma(pInfo.rawVals, g);
                vNorm  = normalizeForDisplay(vGamma);
                set(pInfo.hScatter, 'CData', vNorm, ...
                    'SizeData', 10 * ones(size(vNorm)));
                pInfo.hScatter.AlphaData = vNorm;

            case 'slices'
                for si = 1:numel(pInfo.rawVals)
                    Vg = reshape(applyGamma(pInfo.rawVals{si}, g), ...
                        pInfo.res, pInfo.res);
                    set(pInfo.hImages(si), 'CData', Vg);
                end
        end

        drawnow;
    end

    function cmapShiftCallback(src, fig)
        shiftFrac = get(src, 'Value');

        hReadout = findobj(fig, 'Tag', 'cmapShiftReadout');
        set(hReadout, 'String', sprintf('%.2f', shiftFrac));

        pInfo = getappdata(fig, 'plotInfo');

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

        % Save axes and colorbar positions before projection change
        axPos = get(pInfo.hAx, 'Position');
        hCB = findobj(fig, 'Type', 'ColorBar');
        if ~isempty(hCB)
            cbPos = hCB(1).Position;
        end

        if get(src, 'Value') == 1
            set(pInfo.hAx, 'Projection', 'perspective');
            set(src, 'String', 'Orthographic');
        else
            set(pInfo.hAx, 'Projection', 'orthographic');
            set(src, 'String', 'Perspective');
        end

        % Restore positions (MATLAB may auto-resize on projection change)
        set(pInfo.hAx, 'Position', axPos);
        if ~isempty(hCB)
            hCB(1).Position = cbPos;
        end

        drawnow;
    end

end