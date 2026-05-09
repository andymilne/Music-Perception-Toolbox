%% demo_expTensorPlots.m
%  Visualizes expectation tensor densities in 1 to 4 dimensions for
%  user-specified combinations of r, isRel, and isPer.
%
%  Uses buildExpTens to precompute the density object once per configuration,
%  then passes it to evalExpTens.
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
step_2d = 5;     % e.g., 1 cent per dimension
step_3d = 20;     % e.g., 5 cents per dimension
step_4d = 50;    % e.g., 10 cents per dimension

% === Axis range for non-periodic configurations ===
% For periodic configurations, the range is always [0, period].
% For non-periodic configurations, set the range here.
axMinNonPer = 0;
axMaxNonPer = 2400;

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
            hLine = plot(x, applyTransform(vals, 'off', gamma, eta), 'LineWidth', 1.5);
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
            addPlotControls(fig, info, gamma, eta);

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

            V = reshape(applyTransform(vals, 'off', gamma, eta), res, res);

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
            maxV = max(applyTransform(vals, 'off', gamma, eta));
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
            addPlotControls(fig, info, gamma, eta);

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

                    % Initial render: 'off' mode with max-only normalisation
                    % (matching the redraw path in addPlotControls).
                    M_init = max(vMask);
                    if M_init > 0
                        vNorm = vMask / M_init;
                    else
                        vNorm = vMask;
                    end

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
                addPlotControls(fig, info, gamma, eta);
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
                Vs = reshape(applyTransform(sliceVals, 'off', gamma, eta), res, res);

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
            info.nRows       = nRows;
            info.nCols       = nCols;
            addPlotControls(fig, info, gamma, eta);
    end

    fprintf(' done.\n');
end

fprintf('All plots complete.\n');


%% === Helper functions ===

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
%  For non-surf modes (line, scatter3, slices), the controls are
%  placed in a horizontal row at the bottom of the figure.
%
%  Supported info.mode values:
%    'line'     — updates YData of a line plot
%    'surf'     — updates ZData/CData; includes projection + cmap controls
%    'scatter3' — updates CData and AlphaData of a scatter3 plot
%    'slices'   — updates CData of multiple imagesc subplots

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
        % For 'slices' mode, also include a cmap-shift slider, since
        % the multi-panel grid benefits from clipping low values.
        hasCmap = strcmp(info.mode, 'slices');

        figPos = get(fig, 'Position');
        extraH = 50;
        if hasCmap
            extraH = 80;        % add another row for the cmap slider
        end
        set(fig, 'Position', [figPos(1), figPos(2), figPos(3), ...
                              figPos(4) + extraH]);

        % In slices mode, do an explicit subplot layout so the
        % subplots fill the figure efficiently, with a small bottom
        % reserve for the controls and a small top reserve for the
        % sgtitle. Subplots are placed in row-major order matching
        % their creation by subplot(nRows, nCols, si).
        %
        % We use OuterPosition (which is the bounding rectangle
        % including the axes' title, ticks, and labels) and lock
        % PositionConstraint to 'outerposition'. This way 'axis
        % equal' fits the inner axes inside the bounding box, all
        % subplots align consistently regardless of whether their
        % per-axes title or labels add extra padding, and the
        % xlabels of the bottom row sit safely above the slider.
        if hasCmap
            bottomReserve = 0.14;        % space for control rows
            topReserve    = 0.08;        % space for sgtitle
            leftMargin    = 0.04;
            rightMargin   = 0.02;
            hGap          = 0.01;
            vGap          = 0.02;

            nR = info.nRows;
            nC = info.nCols;

            opW = (1 - leftMargin - rightMargin - (nC - 1) * hGap) / nC;
            opH = (1 - bottomReserve - topReserve - (nR - 1) * vGap) / nR;

            for si = 1:numel(info.hImages)
                axK = get(info.hImages(si), 'Parent');
                if isprop(axK, 'PositionConstraint')
                    axK.PositionConstraint = 'outerposition';
                end
                % Convert linear si to (row, col) in row-major order
                ri = ceil(si / nC) - 1;          % 0 = top row
                ci = mod(si - 1, nC);            % 0 = left column
                x  = leftMargin + ci * (opW + hGap);
                y  = 1 - topReserve - (ri + 1) * opH - ri * vGap;
                set(axK, 'OuterPosition', [x, y, opW, opH]);
            end
        end

        rowH = 0.030;
        y_xform = 0.060;        % bottom row reserved for transform UI
        y_cmap  = 0.020;        % second row for cmap (slices only)
        if ~hasCmap
            y_xform = 0.015;
        end

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

        if hasCmap
            % Per-mode cmap shift memory (was only set up in the surf 
            % branch; needed here too)
            info.cmapShiftOff   = 0;
            info.cmapShiftGamma = 0;
            info.cmapShiftSat   = 0;

            % Cmap label (left-aligned, mirroring the radio column)
            uicontrol(fig, 'Style', 'text', 'String', 'Cmap', ...
                'Units', 'normalized', ...
                'Position', [modeBgX, y_cmap - 0.002, modeBgW, rowH], ...
                'FontSize', 8, 'HorizontalAlignment', 'center', ...
                'Tag', 'cmapShiftLabel', ...
                'BackgroundColor', get(fig, 'Color'));

            % Cmap slider
            uicontrol(fig, 'Style', 'slider', ...
                'Min', 0, 'Max', 0.95, 'Value', 0, ...
                'Units', 'normalized', ...
                'Position', [modeBgX + modeBgW + 0.04, y_cmap, 0.50, rowH], ...
                'Tag', 'cmapShiftSlider', ...
                'SliderStep', [0.005, 0.03], ...
                'Callback', @(src, ~) cmapShiftCallback(src, fig));

            uicontrol(fig, 'Style', 'text', 'String', '0.00', ...
                'Units', 'normalized', ...
                'Position', [modeBgX + modeBgW + 0.55, y_cmap - 0.002, ...
                             0.10, rowH], ...
                'FontSize', 8, ...
                'Tag', 'cmapShiftReadout', ...
                'HorizontalAlignment', 'left', ...
                'BackgroundColor', get(fig, 'Color'));
        end
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
                set(pInfo.hSurf, 'ZData', Vt, 'CData', Vt);
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
            case 'slices'
                for si = 1:numel(pInfo.rawVals)
                    Vt = reshape(applyTransform(pInfo.rawVals{si}, m, g, e), ...
                        pInfo.res, pInfo.res);
                    set(pInfo.hImages(si), 'CData', Vt);
                end
                hShift = findobj(fig, 'Tag', 'cmapShiftSlider');
                if ~isempty(hShift)
                    cmapShiftCallback(hShift, fig);
                end
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

        if strcmp(pInfo.mode, 'slices')
            % Apply the same shift fraction to each imagesc panel using
            % its own (min, max) range.
            for si = 1:numel(pInfo.hImages)
                cdata = get(pInfo.hImages(si), 'CData');
                minC  = min(cdata(:));
                maxC  = max(cdata(:));
                if maxC > minC
                    newLow = minC + shiftFrac * (maxC - minC);
                    set(get(pInfo.hImages(si), 'Parent'), ...
                        'CLim', [newLow, maxC]);
                end
            end
        else
            cdata = get(pInfo.hSurf, 'CData');
            minC  = min(cdata(:));
            maxC  = max(cdata(:));
            if maxC > minC
                newLow = minC + shiftFrac * (maxC - minC);
                set(pInfo.hAx, 'CLim', [newLow, maxC]);
            end
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