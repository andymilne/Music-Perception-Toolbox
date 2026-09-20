function h = plotMaet3d(dens, varargin)
%PLOTMAET3D Draw a three-dimensional expectation tensor density.
%
%   plotMaet3d(dens) draws the density as its kernels: one ellipsoid per
%   tuple centre, shaped by the kernel's covariance and coloured by the
%   density there. No grid is evaluated, so the cost is the number of
%   centres rather than the volume, and the result is ordinary geometry
%   that rotates and occludes correctly.
%
%   plotMaet3d(dens, 'method', 'points') draws the density sampled:
%   one mark per grid node above a threshold, coloured and made
%   translucent by the value there. It shows the material between the
%   peaks, which the kernels cannot, and costs only the evaluation and
%   the marks. The opacity is the value itself rather than an
%   extinction, so this is a translucent cloud and not a volume
%   rendering: what a ray accumulates along its length is not what the
%   picture shows.
%
%   The points method has a resolution budget. A mark carries one depth
%   across its whole face, so where two marks overlap the nearer hides
%   the farther outright instead of blending with it, and every such
%   contest reverses when the camera passes to the other side: the same
%   cloud then draws differently from opposite directions, blobs
%   gaining haloes and hard edges from one of them. Marks that only
%   meet cannot do it, which is what the automatic size and step are
%   for. They hold as long as the grid is no finer than the axes can
%   separate, about three points of screen per node, and that depends
%   on the figure's size and on any zoom as much as on the step. A step
%   set by hand can ask for more than the budget allows; the drawing
%   then says so once and is camera-dependent until the figure is
%   enlarged, zoomed, or the step coarsened. The slices method carries
%   its depth per pixel rather than per mark and has no such budget,
%   which is the reason to reach for it rather than a matter of taste.
%
%   plotMaet3d(dens, 'method', 'slices') draws the density itself,
%   as a stack of textured planes square to whichever axis is most
%   nearly square to the view, each carrying the density as its colour
%   and its opacity. A stack is built for each of the three axes and
%   shown one at a time, so the picture changes as a rotation crosses
%   the diagonal rather than when it ends. This is a volume rendering:
%   the value is read as an extinction per unit of path, so what a ray
%   accumulates follows the distance it travels through the material
%   rather than the number of planes that distance is cut into. Not
%   quite: opacity reaches the renderer as eight bits, so a plane
%   fainter than 1/255 rounds away, and material too faint to clear
%   that in one plane is lost rather than accumulated over many. There
%   is one plane per plane of the volume for that reason, and 'step'
%   is what cuts it more finely.
%
%   The three answer different questions. The ellipsoids show where the
%   kernels are and what shape they have, but where kernels overlap
%   they show the kernels and not the sum they make; the points show
%   the density sampled; the slices show the density itself, including
%   everything between the peaks.
%
%   Inputs
%       dens - Density struct from buildMaet (tag 'MaetDensity'), of
%              one attribute and three drawn dimensions.
%
%   Name-value pairs
%       'method'        'ellipsoids' (default), 'points', or 'slices'.
%       'axes'          Target axes. Default: the current axes.
%       'limits'        [lo hi] for all three axes. Default: one
%                       period for a periodic attribute, and otherwise
%                       the centres' own extent with room for the
%                       kernels around them.
%       'kSigma'        Ellipsoids: the level surface drawn, in
%                       standard deviations. Default 1. Larger shows
%                       more of each kernel and hides more behind it.
%       'step'          Points and slices: spacing of the evaluated
%                       volume, in the density's own units. Memory goes
%                       as its cube. Default: the range over 120 for
%                       slices; for points, the spacing that puts the
%                       grid about three points apart on screen, since
%                       a grid finer than the marks drawn on it can
%                       only be shown by marks that overlap, which is
%                       reported once ('mpt:markOverlap').
%       'threshFrac'    Points: nodes below this fraction of the
%                       largest value are left undrawn. Default 0.001.
%       'markerSize'    Points: the mark's area in square points, or
%                       'auto', the default, for marks just wide enough
%                       to meet their neighbours on the grid. Marks
%                       wider than that overlap, and overlapping marks
%                       are what makes the drawing depend on which side
%                       it is seen from. An automatic size is fitted for
%                       an assumed camera rather than the current one,
%                       so that turning the axes changes neither the
%                       marks nor the ink they lay down; it is refitted
%                       when the figure is resized or zoomed.
%       'markScale'     Scales the automatic size. Default 1, which
%                       holds the cloud's brightness roughly steady as
%                       the step changes: ink goes as the assumed
%                       foreshortening squared over the step, so the
%                       assumption rises as its square root. Raising it
%                       gives larger marks and a brighter cloud, and
%                       past the point where the marks meet they
%                       overlap, which shows as haloes on the blobs at
%                       any elevation, worse below the horizontal than
%                       above it. The drawing warns once
%                       ('mpt:markOverlap') when they overlap at the
%                       view being drawn.
%       'upsample'      Slices: resampling within a plane, which sets
%                       how sharp it looks and costs as the square.
%                       Default 1, which leaves the plane at the
%                       volume's own resolution. What it buys is the
%                       resolution a texel is drawn at, so it shows
%                       where a texel covers several pixels -- a large
%                       'step', or a close zoom. At a step of 5 it is
%                       not distinguishable from 1; at 10 it is.
%                       Raising it is expensive in a figure meant to
%                       be turned: at 4, a frame of a rotation costs
%                       about eight times as much and changing stack
%                       about twelve.
%       'alphaPeak'     Points and slices: the opacity the density's
%                       peak reaches -- for points the opacity of the
%                       brightest mark, for slices what a ray through
%                       the tallest blob's centre reaches. Default 1.
%                       For slices this is not the opacity of the
%                       picture, a ray crossing several blobs on its
%                       way across; lower it to see into the cloud.
%       'alphaGamma'    Points and slices: the display curve on the
%                       opacity, as 'colourGamma' is the curve on the
%                       colour. 1 is opacity proportional to density
%                       (for slices, extinction proportional to it);
%                       above that thins the skirts and suppresses the
%                       low blobs, one of a quarter the height going
%                       as (1/4)^gamma. Default 1 for points and 1.75
%                       for slices, the two reading the same number
%                       differently: for points it shapes a mark's own
%                       opacity, for slices an extinction that then
%                       accumulates along a ray.
%       'colourGamma'   All three methods: the display curve on the
%                       colour, below 1 lifting the low material and
%                       above 1 suppressing it. Default 1. The opacity
%                       has its own curve, 'alphaGamma'.
%       'colormap'      Colour map, an M-by-3 matrix. Default
%                       parula(256).
%       'brighten'      Brightening of the colour map, as MATLAB's
%                       brighten: positive lifts its low end, negative
%                       deepens it. In (-1, 1), default 0.5. It
%                       reshapes the map itself, where 'colourGamma'
%                       reshapes the density's reading of it.
%       'view'          [azimuth elevation] in degrees, as MATLAB's
%                       view: the pair a rotated figure reports, so a
%                       view found with the mouse can be read off and
%                       typed back in. Default [20 20].
%       'dark'          Dark panes for the cube, the ground a glow is
%                       read against. Default true.
%
%   Output
%       h    - The graphics object drawn: a patch for 'ellipsoids', a
%              scatter for 'points', or the array of surfaces making
%              up the stack in view for 'slices'. A rotation may bring
%              another stack into view, after which the handles
%              returned are no longer the ones drawn.
%
%   Example
%       pAttr = {[0 200 400 500 700 900 1100].'};
%       specs = flatSpecs(pAttr, 'r', 4, 'rel', true, 'exch', true);
%       dens  = buildMaet(pAttr, [], 'specs', specs, 'sigma', 15, ...
%                         'isPer', true, 'period', 1200);
%       plotMaet3d(dens);                        % the kernels
%       figure; plotMaet3d(dens, 'method', 'slices');   % the density
%
%   The Python mirror is mpt.plot_maet_3d, which offers 'ellipsoids'
%   alone: 'slices' rests on texture-mapped surfaces, which matplotlib
%   has no counterpart for.
%
%   See also BUILDMAET, EVALMAET, MAETCENTRES.

if ~isstruct(dens) || ~isfield(dens, 'tag') ...
        || ~strcmp(dens.tag, 'MaetDensity')
    error('plotMaet3d:badInput', ...
          'Input must be a density struct from buildMaet.');
end

opt = localOptions(varargin{:});

dim = dens.dim;
if dim ~= 3
    error('plotMaet3d:notThreeDimensional', ...
          ['A three-dimensional plot needs a density of three drawn ' ...
           'dimensions; this one has %d.'], dim);
end
if dens.nAttrs ~= 1
    error('plotMaet3d:multiAttribute', ...
          ['plotMaet3d draws one attribute; this density has %d. ' ...
           'Draw one at a time.'], dens.nAttrs);
end

Cc = maetCentres(dens);
C = Cc{1};
isPer = dens.isPer(1);
period = dens.period(1);
sigma = dens.sigma(1);
if isPer
    C = mod(C, period);
end
covK = localKernelCov(dens, dim);
if isempty(opt.limits)
    lims = localLimits(C, covK, isPer, period, opt.kSigma);
else
    lims = sort(opt.limits(:).');
end

% The current axes when none is named, as a plotting function should:
% newplot honours hold and makes a figure if there is none.
if isempty(opt.axes)
    opt.axes = gca;
end
ax = newplot(opt.axes);

% The frame first: the slice stack is chosen from where the camera is,
% so it has to be the camera the plot will be seen from. Chosen before,
% the stack comes out square to whichever axis the default view happened
% to favour, and is then seen edge-on and draws nothing. It is applied
% again afterwards because scatter3, being a high-level call, resets the
% axes' aspect on its way in.
localFrame(ax, lims, opt);

switch opt.method
    case 'ellipsoids'
        h = localDrawEllipsoids(ax, dens, C, covK, isPer, period, opt);
    case 'points'
        h = localDrawPoints(ax, dens, lims, opt);
    case 'slices'
        h = localDrawSlices(ax, dens, lims, sigma, opt);
    otherwise
        error('plotMaet3d:badMethod', ...
              ['method must be ''ellipsoids'', ''points'', or ' ...
               '''slices''.']);
end

localFrame(ax, lims, opt);

end


% -------------------------------------------------------------------------
function opt = localOptions(varargin)
%LOCALOPTIONS  The name-value pairs, with their defaults.
    p = inputParser;
    p.FunctionName = 'plotMaet3d';
    addParameter(p, 'method', 'ellipsoids', @(s) ischar(s) || isstring(s));
    addParameter(p, 'axes', [], @(a) isempty(a) || isgraphics(a, 'axes'));
    addParameter(p, 'limits', [], @(v) isempty(v) || numel(v) == 2);
    addParameter(p, 'kSigma', 1, @(v) isscalar(v) && v > 0);
    addParameter(p, 'step', [], @(v) isempty(v) || (isscalar(v) && v > 0));
    addParameter(p, 'threshFrac', 0.001, @(v) isscalar(v) && v >= 0);
    addParameter(p, 'markerSize', 'auto', ...
                 @(v) localIsAuto(v) || (isscalar(v) && v > 0));
    addParameter(p, 'markScale', 1, ...
                 @(v) isnumeric(v) && isscalar(v) && v > 0);
    addParameter(p, 'upsample', 1, @(v) isscalar(v) && v >= 1);
    addParameter(p, 'alphaPeak', 1, @(v) isscalar(v) && v > 0 && v <= 1);
    addParameter(p, 'alphaGamma', 1, @(v) isscalar(v) && v > 0);
    addParameter(p, 'colourGamma', 1, @(v) isscalar(v) && v > 0);
    addParameter(p, 'colormap', parula(256), @(m) size(m, 2) == 3);
    addParameter(p, 'brighten', 0.5, ...
                 @(v) isnumeric(v) && isscalar(v) && v > -1 && v < 1);
    addParameter(p, 'view', [20 20], @(v) numel(v) == 2);
    addParameter(p, 'dark', true, @(v) islogical(v) || isnumeric(v));
    parse(p, varargin{:});
    opt = p.Results;
    opt.method = char(opt.method);
    % alphaGamma reads differently for the two methods that take it.
    % For points it shapes a mark's own opacity; for slices it shapes
    % an extinction that then accumulates along a ray, so the same
    % number does not make the same picture. Each method therefore has
    % its own default, while a value the caller gives is used as given.
    if strcmpi(opt.method, 'slices') ...
            && any(strcmp('alphaGamma', p.UsingDefaults))
        opt.alphaGamma = 1.75;
    end
    opt.upsample = round(opt.upsample);
    % Applied here, so that every method reads one already-brightened
    % map rather than each brightening its own copy.
    opt.colormap = brighten(opt.colormap, opt.brighten);
end


function covK = localKernelCov(dens, dim)
%LOCALKERNELCOV  The kernel's covariance in the drawn coordinates.
%
%   Spherical in absolute mode. In relative mode the quadratic form is
%   I - J/r over the r - 1 drawn coordinates, whose inverse is I + J,
%   so the kernel is elongated by sqrt(r) along the all-ones diagonal
%   and circular across it.
    if internal.densityHasKernelCov(dens)
        kc = dens.kernelCov;
        if iscell(kc), kc = kc{1}; end
        if isequal(size(kc), [dim dim])
            covK = kc;
            return
        end
        error('plotMaet3d:unsupportedKernelCov', ...
              ['This density carries an anisotropic kernel covariance ' ...
               'plotMaet3d cannot read; draw it with ''slices'', which ' ...
               'takes the density as evaluated.']);
    end
    sigma = dens.sigma(1);
    if dens.isRel(1)
        covK = sigma ^ 2 * (eye(dim) + ones(dim));
    else
        covK = sigma ^ 2 * eye(dim);
    end
end


function lims = localLimits(C, covK, isPer, period, kSigma)
%LOCALLIMITS  The cube drawn in. One period on a periodic attribute;
%   otherwise the centres' own extent, with room for the kernels around
%   them.
    if isPer
        lims = [0 period];
        return
    end
    reach = 3 * max(sqrt(diag(covK))) * max(kSigma, 1);
    lims = [min(C(:)) - reach, max(C(:)) + reach];
end


function h = localDrawEllipsoids(ax, dens, C, covK, isPer, period, opt)
%LOCALDRAWELLIPSOIDS  One mesh per centre, all in a single patch.
%
%   A centre whose kernel crosses a face of a periodic cube is drawn
%   again on the other side, so that the wrap is cut by the face rather
%   than missing from it; the axes clips what falls outside.
    dim = size(C, 1);
    pv = evalMaet(dens, C, 'none', 'verbose', false);
    pv = pv(:);
    L = chol(covK, 'lower') * opt.kSigma;
    [unitV, unitF] = localUnitSphere(16, 10);
    nVert = size(unitV, 1);
    reach = opt.kSigma * sqrt(diag(covK)).';

    verts = cell(1, 0);
    faces = cell(1, 0);
    vals = cell(1, 0);
    nSoFar = 0;
    for j = 1:size(C, 2)
        shifts = cell(1, dim);
        for i = 1:dim
            s = 0;
            if isPer
                if C(i, j) + reach(i) > period, s = [s, -period]; end %#ok<AGROW>
                if C(i, j) - reach(i) < 0,      s = [s,  period]; end %#ok<AGROW>
            end
            shifts{i} = s;
        end
        [S1, S2, S3] = ndgrid(shifts{1}, shifts{2}, shifts{3});
        for k = 1:numel(S1)
            c = C(:, j).' + [S1(k) S2(k) S3(k)];
            verts{end+1} = unitV * L.' + c;           %#ok<AGROW>
            faces{end+1} = unitF + nSoFar;            %#ok<AGROW>
            vals{end+1} = repmat(pv(j), nVert, 1);    %#ok<AGROW>
            nSoFar = nSoFar + nVert;
        end
    end

    shade = (vertcat(vals{:}) / max(pv)) .^ opt.colourGamma;
    nMap = size(opt.colormap, 1);
    rgbV = opt.colormap(min(nMap, max(1, round(shade * (nMap - 1)) + 1)), :);
    h = patch('Parent', ax, 'Vertices', vertcat(verts{:}), ...
              'Faces', vertcat(faces{:}), 'FaceVertexCData', rgbV, ...
              'FaceColor', 'interp', 'EdgeColor', 'none', ...
              'FaceLighting', 'gouraud', 'AmbientStrength', 0.42, ...
              'DiffuseStrength', 0.72, 'SpecularStrength', 0.18, ...
              'SpecularExponent', 12);
    % A light fixed in the data rather than at the camera, so that
    % turning the cube turns the object under a steady illumination.
    light('Parent', ax, 'Style', 'infinite', 'Position', [-0.4 -0.7 0.9]);
end


function h = localDrawPoints(ax, dens, lims, opt)
%LOCALDRAWPOINTS  One translucent mark per grid node worth drawing.
%
%   The colour is mapped rather than given, and the opacity is left to
%   the figure's alphamap rather than taken as it stands. That is how
%   MATLAB's marker transparency is rendered dependably: given
%   truecolour marks and unmapped opacity instead, it is drawn opaque
%   as soon as the figure settles, while looking correct throughout a
%   drag.
    step = opt.step;
    if isempty(step)
        step = localFitStep(ax);
    end
    [V, g] = localVolume(dens, lims, step);
    [Ga, Gb, Gc] = ndgrid(g, g, g);
    keep = V > opt.threshFrac * max(V(:));
    vals = V(keep);
    rel = vals / max(vals);
    shade = rel .^ opt.colourGamma;

    % Opacity read from its own curve rather than from the colour's, so
    % that the low material can be lifted or suppressed without
    % flattening the colours along with it.
    faceAlpha = opt.alphaPeak * rel .^ opt.alphaGamma;
    if localIsAuto(opt.markerSize)
        [markerSize, overlap] = localFitMarkerSize(ax, step, opt.markScale);
        % A new drawing is a new thing to warn about.
        localWarnOverlap('reset');
        localWarnOverlap(opt.markScale, opt.markScale / overlap, ...
                         overlap > 1);
    else
        markerSize = opt.markerSize;
    end

    h = scatter3(ax, Ga(keep), Gb(keep), Gc(keep), markerSize, ...
                 shade, 'filled');
    set(h, 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 'flat', ...
           'AlphaData', faceAlpha);
    % Pinned, so that the opacities are the numbers computed above.
    % Left automatic, the alpha limits stretch to whatever range the
    % data happens to span, and 'alphaPeak' would have nothing to
    % scale: every plot would reach the same opacity at its own peak.
    set(ax, 'ALim', [0 1]);
    colormap(ax, opt.colormap);

    if localIsAuto(opt.markerSize)
        localRegisterRefit(ax, h, step, opt.markScale);
    end
end


function localWarnOverlap(markScale, maxScale, overlapping)
%LOCALWARNOVERLAP  Warn that the marks overlap at the view being drawn.
%
%   localWarnOverlap('reset') rearms it, which every new drawing does:
%   the throttle is there to keep a rotation from warning on every
%   refit, not to let one plot silence the next.
%
%   Raised as a warning with identifier 'mpt:markOverlap', suppressible
%   with warning('off', 'mpt:markOverlap'), rather than printed, so that
%   it does not land in the middle of a script's own output. It is not
%   gated by showHints, which governs the dispatch messages: this one
%   concerns whether the picture can be believed, so it always gets its
%   single showing. A drawing that comes out within the budget rearms
%   it, so a later one that does not is warned about in its turn.
    persistent warned
    if isempty(warned)
        warned = false;
    end
    if nargin == 1 && (ischar(markScale) || isstring(markScale)) ...
            && strcmpi(char(markScale), 'reset')
        warned = false;
        return
    end
    if ~overlapping
        warned = false;
        return
    end
    if warned
        return
    end
    warned = true;
    warning('mpt:markOverlap', ...
        ['The marks overlap at this view. Overlapping marks are drawn ' ...
         'one over the other by depth rather than blended, so the ' ...
         'picture is not the one the opacities describe; it shows as ' ...
         'haloes on the blobs at any elevation, worse below the ' ...
         'horizontal than above it. ''markScale'' is %g here, where ' ...
         'about %.2f would have them clear. Lower it, or draw the ' ...
         'density with ''method'', ''slices'', which composites per ' ...
         'pixel and has no such limit.'], markScale, maxScale);
end


function tf = localIsAuto(v)
%LOCALISAUTO  Whether a marker size was left to be fitted.
    tf = (ischar(v) || isstring(v)) && strcmpi(char(v), 'auto');
end


function [s, overlap] = localFitMarkerSize(ax, step, markScale)
%LOCALFITMARKERSIZE  A mark just wide enough to meet its neighbours.
%
%   Each mark is a screen-aligned disc carrying one depth, its centre's,
%   across its whole face. Where two discs overlap the nearer rejects
%   the farther over the whole overlap rather than only where it is in
%   front of it, and reversing the camera reverses every such contest,
%   so the same cloud is drawn differently from opposite sides. Marks
%   that only meet cannot do it.
    % Both constants are measured rather than derived. A mark renders
    % wider than sqrt(SizeData) points, so the fraction of the spacing
    % it may occupy was found by stepping the size down until the
    % haloes cleared from every direction: two fifths.
    shrink = 0.39;
    % The renderer stops shrinking a mark somewhere below half a point,
    % every smaller size drawing identically, so there is nothing to be
    % gained by asking for less than it will draw.
    floorPts = 0.2;
    % Sized for an assumed foreshortening rather than this camera's, so
    % that turning the axes changes neither the marks nor the ink they
    % lay down. The step is fitted once for the view being drawn; the
    % size has to hold for every view the axes may be turned to.
    spacing = localPointsPerUnit(ax, localMarkFore(step, markScale)) * step;
    s = max(floorPts, (shrink * spacing) ^ 2);

    % Measured against the grid's spacing at this camera, not against
    % the assumed one: above one the marks overlap here and now,
    % whatever they were sized for.
    here = localPointsPerUnit(ax, []) * step;
    overlap = sqrt(s) / max(here, eps);
end


function fore = localMarkFore(step, markScale)
%LOCALMARKFORE  The foreshortening the marks are sized for.
%
%   Ink on screen goes as fore^2 / step: the marks number step^-3 and
%   each covers (fore * step)^2, so holding the cloud's brightness
%   steady as the step changes wants fore proportional to sqrt(step).
%   The constant is measured, at 1.5 for a step of 10.
%
%   The marks clear one another wherever the true foreshortening
%   exceeds what was assumed, and overlap where it does not; markScale
%   trades the one for the other, buying size and light at the cost of
%   overlap over more of the sphere.
    fore = 0.4743 * sqrt(step) * markScale;
end


function step = localFitStep(ax)
%LOCALFITSTEP  A grid the marks drawn on it can resolve.
%
%   A grid finer than the marks can only be shown by marks that overlap,
%   which is what localFitMarkerSize exists to prevent, so the two are
%   settled together: the grid is spaced at the target mark width and
%   the marks are then sized to meet on it. The node count is bounded at
%   both ends, coarse enough to stay legible and fine enough not to
%   exhaust memory.
    targetPts = 3;
    minNodes = 24;
    maxNodes = 200;
    % Looking along an axis, that axis's neighbours fall on one pixel
    % and the spacing goes to nothing, which would ask for a grid finer
    % than any memory. The node count is bounded regardless, so this
    % only stops a view that happens to be near an axis from settling
    % the step for a drawing that will be turned away from it. The
    % reference is the camera-free figure at full foreshortening.
    guard = 0.25;
    k = max(localPointsPerUnit(ax), guard * localPointsPerUnit(ax, 1));
    spacing = targetPts / k;
    span = diff(xlim(ax));
    nodes = min(maxNodes, max(minNodes, round(span / spacing)));
    step = span / nodes;
end


function k = localPointsPerUnit(ax, assumedFore)
%LOCALPOINTSPERUNIT  Screen points per data unit, along the worst axis.
%
%   Given an assumedFore the answer is camera-free: no orientation
%   projects the box wider than its space diagonal, so with a
%   foreshortening supplied rather than measured nothing about the
%   current view enters. Sizing the marks that way makes the size
%   independent of the view -- and with it the ink each mark lays down,
%   which is what a size that tracked the camera was changing as the
%   axes turned. Pass [] for the spacing at the camera as it stands,
%   which is what the step is fitted to and what the marks are checked
%   against.
%
%   Everything is read from the axes as it stands, so a zoom is accounted
%   for like any other change: zooming narrows the limits without moving
%   the camera, and it is the limits that set how far apart the grid
%   falls on screen.
%
%   The plot box is a cube whatever the limits are, so a data unit spans
%   a different screen length on each axis once the ranges differ, and
%   the camera is converted to that box's coordinates before the
%   foreshortening is taken. The projection is orthographic, so an axis
%   of the box spans its scale times sqrt(1 - (v.e)^2), and the closest
%   the grid comes on screen is set by whichever axis fares worst.
    if nargin < 2
        assumedFore = [];
    end
    ranges = [diff(xlim(ax)), diff(ylim(ax)), diff(zlim(ax))];
    ranges = max(ranges(:).', eps);

    if ~isempty(assumedFore)
        box = localAxesPoints(ax);
        % The widest range gives the closest the grid comes on screen,
        % a data unit spanning less of the box the more of it the axis
        % has to cover.
        k = (min(box) / sqrt(3)) * assumedFore / max(ranges);
        return
    end

    v = (campos(ax) - camtarget(ax)) ./ ranges;
    v = v(:) / norm(v);
    up = camup(ax) ./ ranges;
    up = up(:);
    right = cross(up, v);
    if norm(right) < sqrt(eps)
        basis = null(v.');
        right = basis(:, 1);
    end
    right = right / norm(right);
    up = cross(v, right);

    % The box's projected extent, from which its scale follows.
    corners = [0 0 0; 0 0 1; 0 1 0; 0 1 1
               1 0 0; 1 0 1; 1 1 0; 1 1 1];
    projW = max(corners * right) - min(corners * right);
    projH = max(corners * up) - min(corners * up);

    box = localAxesPoints(ax);
    scale = min(box(1) / projW, box(2) / projH);   % points per box unit

    % A data axis spans scale * fore / range points, and what the marks
    % have to fit into is the closest the grid comes on screen: the
    % smallest of the three, not the largest.
    fore = sqrt(max(0, 1 - v .^ 2)).';
    k = min(scale * fore ./ ranges);
end


function wh = localAxesPoints(ax)
%LOCALAXESPOINTS  The axes box in points, whatever its Units are set to.
%
%   getpixelposition rather than the Position property, so that an axes
%   managed by a layout reports the box it is actually drawn in.
    px = getpixelposition(ax, true);
    wh = px(3:4) * 72 / get(groot, 'ScreenPixelsPerInch');
end


function localRegisterRefit(ax, h, step, markScale)
%LOCALREGISTERREFIT  Keep the automatic marker size fitted.
%
%   Two things unfit it: resizing the figure and zooming, each changing
%   how far apart the grid falls on screen while the marks stay the size
%   they were given. Resizing is caught on the figure, so that several
%   axes share one callback; zoom is caught per axes, the limits being
%   the axes' own. Rotation needs no callback: the size is fitted for
%   any camera, so turning the axes cannot unfit it.
%
%   Only the size is refitted. The step would mean evaluating the volume
%   again, which is not something to do during a drag.
    % Every property a moving camera touches, not just 'View': dragging
    % the axes sets CameraPosition and CameraUpVector, and 'View' is
    % derived from them, so a listener on 'View' alone can sit through a
    % whole rotation without firing. CameraViewAngle matters too, since
    % it is what refits the cube in the axes as its projected extent
    % changes.
    % The camera is watched again, though the size no longer depends on
    % it: what does depend on it is whether the marks overlap at the
    % view being looked at, which is what the warning reports.
    listener = addlistener(ax, ...
        {'XLim', 'YLim', 'ZLim', 'CameraPosition', 'CameraUpVector', ...
         'CameraViewAngle', 'View'}, ...
        'PostSet', @(~, ~) localRefit(ax));
    localSetState(ax, 'Points', struct('handle', h, 'step', step, ...
                                       'markScale', markScale, ...
                                       'listener', listener));
    fig = ancestor(ax, 'figure');
    if isempty(fig)
        return
    end
    if ~isappdata(fig, 'plotMaet3dPrevSizeFcn')
        setappdata(fig, 'plotMaet3dPrevSizeFcn', get(fig, 'SizeChangedFcn'));
        set(fig, 'SizeChangedFcn', @localOnResize);
    end
end


function localRefit(ax)
%LOCALREFIT  Resize one axes' marks to the grid as it now projects.
    if ~isgraphics(ax)
        return
    end
    d = localGetState(ax, 'Points');
    if ~isempty(d) && isgraphics(d.handle)
        [sz, overlap] = localFitMarkerSize(ax, d.step, d.markScale);
        d.handle.SizeData = sz;
        localWarnOverlap(d.markScale, d.markScale / overlap, overlap > 1);
    end
end


function localOnResize(fig, evt)
%LOCALONRESIZE  Refit every fitted scatter in the figure, then defer.
    for a = findobj(fig, 'Type', 'axes').'
        localRefit(a);
    end
    % Whatever the figure had before is still the caller's to run.
    prev = getappdata(fig, 'plotMaet3dPrevSizeFcn');
    if isa(prev, 'function_handle')
        prev(fig, evt);
    elseif iscell(prev) && ~isempty(prev)
        feval(prev{1}, fig, evt, prev{2:end});
    elseif (ischar(prev) || isstring(prev)) && strlength(string(prev)) > 0
        evalin('base', char(prev));
    end
end


function h = localDrawSlices(ax, dens, lims, sigma, opt)
%LOCALDRAWSLICES  The volume as a stack of textured planes.
    step = opt.step;
    if isempty(step)
        step = diff(lims) / 120;
    end
    [V, g] = localVolume(dens, lims, step);
    % Extinction per unit of path, fixed so that a ray through the
    % tallest blob's centre reaches alphaPeak: the path integral of
    % (v/vMax)^gamma along the axis of a Gaussian blob of that height is
    % sigma * sqrt(2 pi / gamma).
    stack = struct('map', opt.colormap, 'vMax', max(V(:)), ...
                   'colourGamma', opt.colourGamma, ...
                   'alphaGamma', opt.alphaGamma, ...
                   'extinction', -log(1 - min(opt.alphaPeak, 0.999)) ...
                                 / (sigma * sqrt(2 * pi / opt.alphaGamma)), ...
                   'step', step, 'up', opt.upsample);
    % Depth sorting rejects every plane behind the nearest one, so the
    % stack renders no darker than a single plane however many it holds.
    % Drawing in creation order instead blends them; the planes are
    % created back to front, which is the order that blending wants.
    %
    % This is a property of the axes and not of the stack, and it is
    % wrong for opaque geometry, which needs depth sorting to occlude
    % correctly. What it was is kept and put back when the stack goes,
    % so an axes drawn into again afterwards is not left with it.
    prevSort = ax.SortMethod;
    ax.SortMethod = 'childorder';

    % All three stacks are built before anything is shown. Building the
    % wanted one on demand costs a rebuild in the middle of a turn,
    % which is long enough to see; showing and hiding costs nothing, so
    % the stack changes as the turn crosses the diagonal rather than
    % when the mouse is released. The volume is passed in rather than
    % kept: once the planes carry it, nothing reads it again.
    [stacks, groups] = localBuildAll(ax, V, g, lims, stack);
    groups(1).DeleteFcn = @(~, ~) localSlicesGone(ax, prevSort);

    localSetState(ax, 'Slices', ...
                  struct('stacks', {stacks}, 'groups', groups, ...
                         'axis', 0, 'handles', gobjects(0)));
    localPickStack(ax);

    % The camera moves throughout a turn, so the choice follows it
    % rather than waiting for the turn to end. Which property a camera
    % move sets depends on how it is made: dragging the axes sets
    % CameraPosition and leaves View derived from it, while view() sets
    % View and leaves CameraPosition derived, and a derived property
    % raises nothing. Both are watched for that reason.
    d = localGetState(ax, 'Slices');
    d.listener = addlistener(ax, ...
        {'View', 'CameraPosition', 'CameraTarget', 'CameraUpVector'}, ...
        'PostSet', @(~, ~) localPickStack(ax));
    localSetState(ax, 'Slices', d);
    h = d.handles;
end


function localSlicesGone(ax, prevSort)
%LOCALSLICESGONE  Put the axes back as the stack found it.
    if ~isgraphics(ax, 'axes')
        return
    end
    ax.SortMethod = prevSort;
    localClearState(ax, 'Slices');
end


function [stacks, groups] = localBuildAll(ax, V, g, lims, opt)
%LOCALBUILDALL  Build the stack for each of the three axes.
%
%   Each stack's planes are collected into a group, so that showing or
%   hiding one is a single property rather than one per plane. A stack
%   can hold hundreds of planes, and setting each of them in turn is
%   long enough to be seen as a stall in the middle of a rotation.
    stacks = cell(1, 3);
    groups = gobjects(1, 3);
    for a = 1:3
        groups(a) = hggroup(ax, 'Visible', 'off');
        stacks{a} = localBuildStack(groups(a), ax, V, g, lims, opt, a);
    end
end


function [V, g] = localVolume(dens, lims, step)
%LOCALVOLUME  The density on a cubic grid, a slab of planes at a time,
%   the whole cube of query points being a large array to hold at once.
    % The dispatch messages are throttled per top-level call, and a slab
    % is one, so routing the same way for every slab would announce
    % itself once per slab. Which path the evaluation takes is the
    % toolbox's business rather than the picture's, so they are silenced
    % for the duration and the caller's setting put back afterwards.
    prevHints = mptDefaults('showHints');
    mptDefaults('showHints', false);
    restoreHints = onCleanup(@() mptDefaults('showHints', prevHints)); %#ok<NASGU>

    n = max(1, round(diff(lims) / step));
    g = linspace(lims(1), lims(2), n + 1);
    nG = numel(g);
    V = zeros(nG, nG, nG);
    slabPlanes = 8;
    for first = 1:slabPlanes:nG
        last = min(first + slabPlanes - 1, nG);
        [Ga, Gb, Gc] = ndgrid(g(first:last), g, g);
        v = evalMaet(dens, [Ga(:).'; Gb(:).'; Gc(:).'], 'none', ...
                     'verbose', false);
        V(first:last, :, :) = reshape(v, last - first + 1, nG, nG);
    end
end


function localPickStack(ax)
%LOCALPICKSTACK  Show the stack square to whichever axis is most
%   nearly square to the view, and hide the other two.
%
%   All three stacks exist and are built before anything is shown, so
%   this costs only a change of visibility and can follow the camera
%   through a turn.
%
%   The correction for how obliquely the rays cross the planes, a
%   factor of 1/cos(theta) and at most 1.73, is not applied. A plane's
%   opacity can be changed after it is built, whether in its AlphaData
%   or through the axes' alpha limits, but either costs the renderer a
%   fifth of a second against a frame of two hundredths, so a
%   correction that followed the camera would cost more than the error
%   it removes. The picture is therefore exact square-on to an axis and
%   up to 1.73 times too transparent at the corners.
    if ~isgraphics(ax, 'axes')
        return
    end
    d = localGetState(ax, 'Slices');
    if isempty(d), return, end
    w = ax.CameraPosition - ax.CameraTarget;
    w = w / norm(w);
    [~, a] = max(abs(w));
    if a == d.axis, return, end
    for b = 1:3
        d.groups(b).Visible = localOnOff(b == a);
    end
    d.axis = a;
    d.handles = d.stacks{a};
    localSetState(ax, 'Slices', d);
end


function localSetState(ax, which, state)
%LOCALSETSTATE  Keep a method's state on the axes, under its own key.
%
%   Each method has a key of its own, so that drawing one into an axes
%   that already holds another does not overwrite its state and leave
%   the earlier drawing's listener stranded. Application data rather
%   than UserData, that being the caller's property to use.
    setappdata(ax, ['plotMaet3d' which], state);
end


function state = localGetState(ax, which)
%LOCALGETSTATE  A method's state, or empty if the axes holds none.
    key = ['plotMaet3d' which];
    if isappdata(ax, key)
        state = getappdata(ax, key);
    else
        state = [];
    end
end


function localClearState(ax, which)
%LOCALCLEARSTATE  Forget a method's state.
    key = ['plotMaet3d' which];
    if isappdata(ax, key)
        rmappdata(ax, key);
    end
end


function s = localOnOff(tf)
%LOCALONOFF  'on' or 'off' for a logical.
    if tf
        s = 'on';
    else
        s = 'off';
    end
end


function hs = localBuildStack(parent, ax, V, g, lims, opt, a)
%LOCALBUILDSTACK  One stack of textured planes square to axis A,
%   parented to PARENT and framed by the camera of AX.
%
%   Every plane is given its colour and its opacity in the call that
%   creates it: AlphaData set then is honoured, and a plane made blank
%   and filled in afterwards is not.
%
%   There is one plane per plane of the volume. Cutting more finely
%   than the volume was evaluated would divide the optical depth of the
%   ray between more planes without adding anything to it, and opacity
%   reaches the renderer as eight bits, so the thinner planes round to
%   nothing and the faint material is lost rather than gained. The way
%   to cut the volume more finely is to evaluate it more finely.
    n = numel(g);
    pos = g;
    planeStep = opt.step;
    nMap = size(opt.map, 1);
    hs = gobjects(n, 1);
    % Blending in creation order is only correct back to front, so the
    % planes are created starting from the side away from the camera the
    % stack is built for. A later turn to the other side reverses that,
    % which weights the far material rather than the near in the colour
    % a ray ends up with; the accumulated opacity is a product and so is
    % the same either way, and the difference measures 0.4%, which is
    % not worth reordering the planes for.
    v = ax.CameraPosition - ax.CameraTarget;
    order = 1:n;
    if v(a) < 0
        order = flip(order);
    end
    for m = order
        rel = max(localPlaneOf(V, a, m), 0) / opt.vMax;
        cIdx = min(nMap, max(1, round(rel .^ opt.colourGamma ...
                                      * (nMap - 1)) + 1));
        rgbImg = reshape(opt.map(cIdx(:), :), [size(cIdx), 3]);
        aImg = 1 - exp(-opt.extinction * rel .^ opt.alphaGamma * planeStep);
        if opt.up > 1
            rgbImg = localResample(rgbImg, opt.up);
            aImg = localResample(aImg, opt.up);
        end
        [X, Y, Z] = localPlaneQuad(a, pos(m), lims);
        hs(m) = surface(parent, X, Y, Z, 'FaceColor', 'texturemap', ...
                        'CData', uint8(round(255 * rgbImg)), ...
                        'FaceAlpha', 'texturemap', 'AlphaData', aImg, ...
                        'AlphaDataMapping', 'none', 'EdgeColor', 'none');
    end
end


function [X, Y, Z] = localPlaneQuad(a, at, lims)
%LOCALPLANEQUAD  The quadrilateral of a plane square to axis A.
    lo = lims(1);
    hi = lims(2);
    switch a
        case 1
            X = at * ones(2); Y = [lo hi; lo hi]; Z = [lo lo; hi hi];
        case 2
            Y = at * ones(2); X = [lo hi; lo hi]; Z = [lo lo; hi hi];
        otherwise
            Z = at * ones(2); X = [lo hi; lo hi]; Y = [lo lo; hi hi];
    end
end


function img = localPlaneOf(V, a, i)
%LOCALPLANEOF  Plane I of the volume for stack A, turned so that its
%   rows and columns run along the plotted axes.
    switch a
        case 1
            img = squeeze(V(i, :, :)).';
        case 2
            img = squeeze(V(:, i, :)).';
        otherwise
            img = squeeze(V(:, :, i)).';
    end
end


function out = localResample(img, up)
%LOCALRESAMPLE  Bilinear resampling of a plane's image, by the same grid
%   for the colour and for the opacity so that the two stay registered.
%   Cell centres in and cell centres out, so the image covers the same
%   square it did.
    [r, c, p] = size(img);
    xi = ((1:(c * up)) - 0.5) / up + 0.5;
    yi = ((1:(r * up)) - 0.5) / up + 0.5;
    out = zeros(r * up, c * up, p);
    for k = 1:p
        out(:, :, k) = interp2(img(:, :, k), xi, yi.', 'linear', 0);
    end
end


function [V, F] = localUnitSphere(nLon, nLat)
%LOCALUNITSPHERE  A quadrilateral mesh on the unit sphere, built here
%   rather than taken from the plotting library so that the two
%   languages draw the same mesh.
    lon = (0:(nLon - 1)) * 2 * pi / nLon;
    lat = linspace(0, pi, nLat);
    [T, P] = ndgrid(lon, lat);
    V = [sin(P(:)) .* cos(T(:)), sin(P(:)) .* sin(T(:)), cos(P(:))];
    F = zeros(nLon * (nLat - 1), 4);
    n = 0;
    for i = 1:nLon
        iNext = mod(i, nLon) + 1;
        for j = 1:(nLat - 1)
            n = n + 1;
            F(n, :) = [(i - 1) * nLat + j, (iNext - 1) * nLat + j, ...
                       (iNext - 1) * nLat + j + 1, (i - 1) * nLat + j + 1];
        end
    end
end


function localFrame(ax, lims, opt)
%LOCALFRAME  The cube the density is drawn in.
    view(ax, opt.view);
    set(ax, 'Projection', 'orthographic');
    xlim(ax, lims); ylim(ax, lims); zlim(ax, lims);
    pbaspect(ax, [1 1 1]);
    if opt.dark
        set(ax, 'Color', [0.06 0.06 0.06], ...
                'GridColor', [0.22 0.22 0.22], 'GridAlpha', 1);
    end
    grid(ax, 'on');
end
