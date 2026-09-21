%% demo_maetPlots.m
%  Draws the same seven pitches as a MAET under every combination of
%  the four parameters that define one, and as each of the three
%  methods plotMaet offers for drawing it.
%
%  === The four parameters ===
%
%  The `configs` table below sets out the combinations. What each
%  parameter does, and where it shows in the pictures:
%
%    r       raises the dimensionality, since dim = r - isRel. Going
%            from r = 2 to r = 3 turns a plane into a cube. The
%            density places one kernel per r-tuple, so the number of
%            blobs goes as the number of tuples.
%    isRel   absolute against relative. An absolute density lives at
%            the pitches themselves; a relative one lives at the
%            intervals between them, is transposition-invariant, and
%            costs a dimension. Its kernels are elongated along the
%            all-ones diagonal, which the 'kernels' method shows
%            directly.
%    isPer   whether the space wraps. A periodic density is drawn over
%            one period, and a kernel crossing a face reappears on the
%            other side; a non-periodic one runs off into silence.
%    isExch  unordered against ordered. An ordered density counts each
%            arrangement of a tuple separately and is unsymmetric in
%            its arguments; the unordered one is its symmetrization
%            and so is mirror-symmetric about the diagonal. Each
%            configuration is drawn both ways, adjacent, so the
%            symmetrization is a difference between neighbouring tabs.
%
%  === The drawing ===
%
%  Every plot is drawn by plotMaet, which dispatches on the density's
%  drawn dimensionality and offers three methods -- 'kernels',
%  'points', and 'density' -- described at plotMethod below. This
%  script makes no picture of its own: it builds the densities, sets
%  the options, and frames the result.
%
%  Each parameter block below states the choice made and what the
%  alternatives do.
%
%  Uses: buildMaet, plotMaet (from the Music Perception Toolbox).

%% === User-editable parameters ===
close all
% The multiset to draw, and its weights. The diatonic scale in cents:
% seven pitches keeps the structure legible and r = 4 quick. Empty
% weights means all equal; a weight vector scales each pitch's
% contribution and shows as differing blob heights.
p = [0; 200; 400; 500; 700; 900; 1100];
w = [];

% Kernel width, in the same units as p. At 15 cents the semitone
% spacings of this scale are about seven sigma apart and the blobs
% resolve separately; at 50 they merge into ridges, which shows what
% a listener might confuse rather than where the tuples are.
sigma = 15;

% The period for the periodic configurations, in the same units as p.
period = 1200;

% === Plot configurations ===
% One row per plot: [r, isRel, isPer, isExch], with isRel, isPer, and
% isExch as 0 or 1. What each does is set out in the header; the table
% is ordered so that the differences are adjacent.
%
% Only one to three drawn dimensions can be drawn, dim = r - isRel, so
% r runs to 3 absolute and 4 relative. A four-dimensional density has
% no honest picture: the grid of two-dimensional slices this demo once
% drew for it showed three arbitrary cuts rather than the density.
%
% Each configuration appears twice, unordered then ordered, so that
% the symmetrization is a difference between neighbouring tabs. r = 1
% is included both ways although it has one slot and so nothing to
% order: the two come out identical. Exchangeability is a statement
% about the arrangement of a tuple's elements, and a tuple of one has
% only the one.
%
% Add, remove, or reorder rows to control which plots are produced.
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

% === Grid resolution ===
% The grid plotMaet evaluates. It takes either 'nodes', a count of
% steps across whatever range is drawn, or 'step', a spacing in the
% units of p -- the same request said two ways.
%
% The number that matters is the grid measured against sigma rather
% than against the range: a blob is a few sigma across, so a grid
% coarser than sigma steps straight over it and the density looks as
% though it has peaks missing rather than blurred. Roughly one sample
% per sigma is the least that shows the shape.
%
% Cost goes as the count to the power of the dimensionality. The
% 'kernels' method evaluates no grid and ignores all of this. Empty
% leaves the choice to plotMaet, which asks for 1200 steps at one and
% two dimensions and 120 at three.
nodes_1d = 1200;   % steps across the range, so periodic and
nodes_2d = 1200;   % non-periodic are sampled alike
step_3d  = 10;     % cents per step, so the wider non-periodic cube
                   % is not left at 20 cents and blocky

% === Axis range for non-periodic configurations ===
% Periodic configurations are always drawn over [0, period]. The rest
% are drawn over the range set here, rather than over the extent
% plotMaet would choose from the data, so that every configuration
% shares one frame and can be read against the others. Centred on zero
% because a relative density is symmetric about the unison and this
% shows an interval beside its inversion; absolute densities take the
% same range so the two kinds are comparable.
axMinNonPer = -1200;
axMaxNonPer = 1200;

% === Colour and opacity ===
% plotMaet draws against dark panes, brightens the colour map, and
% fades the low material out. The two settings here control the last
% two of those.
%
% brightenMap is parula's brightening, as MATLAB's brighten: positive
% lifts its low end, which is where most of a density's material sits.
% Two values, [one and two dimensions, three dimensions], because a
% volume rendering accumulates along every ray while a line or a
% surface shows each value once, so the same brightening looks washed
% at three dimensions. Empty takes plotMaet's own default.
%
% alphaFloor2d is where the opacity curve starts, at zero density, for
% the two-dimensional surfaces. At 0 the empty parts of the surface
% are fully transparent, which reads well from directly above; tilted,
% the surface loses its shape there and the blobs float unsupported.
% 1 turns the fading off and draws the surface opaque.
%
% Together they set how far the picture departs from an ordinary
% MATLAB surface. brightenMap = 0 with alphaFloor2d = 1 gives
% unbrightened parula on a solid surface -- MATLAB's own look, and
% probably the more natural one for a two-dimensional density examined
% as a surface under rotation, where relief does the work that opacity
% does from above. The panes stay dark either way; that is plotMaet's
% 'dark', which this script leaves at its default.
brightenMap  = [0.7 0.5];
alphaFloor2d = 0;

% === Figure window style ===
% true docks every figure, so the configurations arrive as tabs of one
% window and can be stepped through; false gives a window each, which
% suits comparing two side by side. Set explicitly rather than left to
% MATLAB's preference, so the demo behaves the same on any machine.
dockFigures = true;

% === Plot method ===
% Passed straight to plotMaet.
%   'kernels' — the model rather than the density: one object per
%               tuple centre, an ellipsoid, an ellipse, or a curve.
%               Shows where the kernels are and what shape they have,
%               which is where the elongation of a relative kernel
%               becomes visible. Evaluates no grid.
%
%               At one dimension each curve is one tuple's own term,
%               its width the kernel's and its height that tuple's
%               weight, so the curves sum to the density. Colour is
%               the density at that centre -- the total, neighbours
%               included -- so two curves of equal height differ in
%               colour where kernels crowd, and a peak can be seen to
%               be one kernel or several.
%   'points'  — the density sampled: one translucent mark per grid
%               node above a threshold. Three dimensions only. At a
%               fine grid it is much the same picture as 'density'
%               and costs more to draw; what differs is that the
%               samples stay discrete, so the grid is visible rather
%               than interpolated away.
%   'density' — the density itself: a line, a translucent surface, or
%               a stack of textured planes.
%
% 'density' is the default here because the demo is about what the
% four parameters do to the density, and the kernels are a step behind
% that. Switch to 'kernels' to see the tuples themselves.
plotMethod = 'density';

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

    % A count at one and two dimensions, a spacing at three.
    if dimC < 3
        if dimC == 1, nodesC = nodes_1d; else, nodesC = nodes_2d; end
        if isempty(nodesC)
            nodesC = localDefaultNodes(dimC);
        end
    elseif isempty(step_3d)
        nodesC = localDefaultNodes(dimC);
    else
        nodesC = round((axMaxC - axMinC) / step_3d);
    end
    resC = max(2, round(nodesC) + 1);

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

    % Empty leaves plotMaet's own colour map alone.
    if isempty(brightenMap)
        mapArgs = {};
    elseif dim < 3
        mapArgs = {'brighten', brightenMap(1)};
    else
        mapArgs = {'brighten', brightenMap(end)};
    end

    % --- The grid: a count at one and two dimensions, a spacing at
    %     three. Empty either way leaves the choice to plotMaet. ---
    if dim < 3
        if dim == 1, nodes = nodes_1d; else, nodes = nodes_2d; end
        if isempty(nodes)
            gridArgs = {};
            res = localDefaultNodes(dim) + 1;
        else
            gridArgs = {'nodes', nodes};
            res = round(nodes) + 1;
        end
    elseif isempty(step_3d)
        gridArgs = {};
        res = localDefaultNodes(dim) + 1;
    else
        gridArgs = {'step', step_3d};
        res = max(2, round((axMax - axMin) / step_3d) + 1);
    end

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
        %  dim = 1: a line, or the kernels that sum to it
        % =============================================================
        case 1
            fig = newDemoFigure(ci, r, dim, dockFigures);
            h1 = plotMaet(dens, 'method', plotMethod, ...
                          'limits', [axMin axMax], gridArgs{:}, ...
                          mapArgs{:});
            xlabel(sprintf('%s 1', axLabel));
            ylabel('Density');
            title(titleStr);
            setDemoTicks(gca, [axMin axMax], 1);

        % =============================================================
        %  dim = 2: a surface, or the kernels' outlines
        % =============================================================
        case 2
            fig = newDemoFigure(ci, r, dim, dockFigures);
            h2 = plotMaet(dens, 'method', plotMethod, ...
                          'limits', [axMin axMax], gridArgs{:}, ...
                          'alphaFloor', alphaFloor2d, mapArgs{:});
            xlabel(sprintf('%s 1', axLabel));
            ylabel(sprintf('%s 2', axLabel));
            title(titleStr);
            setDemoTicks(gca, [axMin axMax], 2);

            % A two-dimensional density is a surface seen from
            % directly above. Tilting it shows as relief what the
            % colour and opacity show only by comparison.
            rotate3d(fig, 'on');

            % A colourbar as a key to the map. plotMaet draws none,
            % and for the density method it is a key to the map alone:
            % a colourbar knows nothing of opacity, so with the
            % surface fading to the panes its low end shows colours
            % the picture never paints.
            colorbar(ancestor(h2, 'axes'));

        % =============================================================
        %  dim = 3: a stack of planes, marks, or ellipsoids
        % =============================================================
        case 3
            fig = newDemoFigure(ci, r, dim, dockFigures);
            h3 = plotMaet(dens, 'method', plotMethod, ...
                          'limits', [axMin axMax], gridArgs{:}, ...
                          'upsample', 2, mapArgs{:});
            xlabel(sprintf('%s 1', axLabel));
            ylabel(sprintf('%s 2', axLabel));
            zlabel(sprintf('%s 3', axLabel));
            title(sprintf('%s — %s', titleStr, plotMethod));
            setDemoTicks(gca, [axMin axMax], 3);

            % These are meant to be turned, so the figure opens ready
            % to. plotMaet leaves the interaction mode alone, that
            % being the caller's to set rather than a plotting
            % function's to impose.
            rotate3d(fig, 'on');

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


function n = localDefaultNodes(dim)
%LOCALDEFAULTNODES  The nodes per axis plotMaet asks for when the step
%   is left to it. Kept here only so that the runtime estimate can
%   report the grid that will actually be evaluated.
    if dim < 3
        n = 1200;
    else
        n = 120;
    end
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
