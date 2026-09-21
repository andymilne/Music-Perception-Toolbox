function h = plotMaet(dens, varargin)
%PLOTMAET Draw a one-, two-, or three-dimensional expectation tensor density.
%
%   plotMaet(dens) draws a density of one, two, or three drawn
%   dimensions, dispatching on the dimensionality the density carries:
%   dim = r - isRel. Four or more cannot be drawn and is refused.
%
%   Three methods, each named for what it shows rather than for the
%   geometry it uses, since the geometry is what changes with the
%   dimensionality:
%
%   'kernels' (default) draws where the kernels are and what shape they
%   have: an ellipsoid per tuple centre at three dimensions, an ellipse
%   at two, a curve at one, coloured by the density at the centre. No
%   grid is evaluated, so the cost is the number of centres rather than
%   the volume, and the result is ordinary geometry that rotates and
%   occludes correctly. Where kernels overlap it shows the kernels and
%   not the sum they make -- except at one dimension, where each curve
%   is one tuple's own term, its width the kernel's and its height
%   that tuple's weight, so the curves do sum to the density. Colour
%   there is the density at the curve's centre, the total with every
%   neighbour counted, so two curves of equal height differ in colour
%   where kernels crowd.
%
%   'points' draws the density sampled: one translucent mark per grid
%   node above a threshold, coloured and made translucent by the value
%   there. Three drawn dimensions only: below that it samples what
%   'density' already draws whole, a surface carrying every node at
%   once and a line likewise. It shows the material between the peaks,
%   which the kernels cannot.
%
%   At a fine grid it is much the same picture as 'density'. What
%   differs is that the samples stay discrete, so the grid is visible
%   rather than interpolated away, and that a mark's opacity is the
%   value at its own node rather than an extinction accumulated along
%   the ray -- though marks reject one another by depth rather than
%   blending, so what reaches a pixel is the nearest mark and not a
%   sum either.
%
%   It is dearer than 'density', not cheaper. It evaluates the same
%   grid -- the evaluation is around 85 per cent of the cost of either
%   -- and its marks go as the cube of the nodes per axis where a
%   stack's planes go as the first power. Measured on a 121-node grid
%   it gave 20 frames a second against 67, and on a 241-node grid 4
%   against 51.
%
%   At three dimensions 'points' has a resolution budget. A mark
%   carries one depth across its whole face, so where two marks
%   overlap the nearer hides the farther outright instead of blending
%   with it, and every such contest reverses when the camera passes to
%   the other side: the same cloud then draws differently from opposite
%   directions, blobs gaining haloes and hard edges from one of them.
%   Marks that only meet cannot do it, and the automatic size and step
%   keep them apart. They hold as long as the grid is no finer than
%   the axes can separate, about three points of screen per node,
%   which depends on the figure's size and on any zoom as much as on
%   the step. A step set by hand can ask for more than that; the
%   drawing then warns once and is camera-dependent until the figure
%   is enlarged, zoomed, or the step coarsened. Below three
%   dimensions there is no depth for one mark to reject another by, so
%   overlapping marks blend as their opacities describe and there is no
%   budget to keep.
%
%   'density' draws the density itself: a line at one dimension, a
%   textured translucent surface at two, and at three a stack of
%   textured planes square to whichever axis is most nearly square to
%   the view. A stack is built for each of the three axes and shown one
%   at a time, so the picture changes as a rotation crosses the
%   diagonal rather than when it ends. At three dimensions this is a
%   volume rendering: the value is read as an extinction per unit of
%   path, so what a ray accumulates follows the distance it travels
%   through the material rather than the number of planes that distance
%   is cut into -- with one qualification. Opacity reaches the
%   renderer as eight bits, so a plane fainter than 1/255 rounds away,
%   and material too faint to clear that in one plane is lost rather
%   than accumulated over many. There is one plane per plane of the
%   volume for that reason, and 'step' is what cuts it more finely.
%
%   The kernels show the model; the points and the density both show
%   the density, the first as discrete samples and the second as a
%   continuous field.
%
%   Inputs
%       dens - Density struct from buildMaet (tag 'MaetDensity'), of
%              one attribute and one to three drawn dimensions.
%
%   Name-value pairs
%       'method'        'kernels' (default), 'points', or 'density'.
%                       'points' needs two or three drawn dimensions.
%       'axes'          Target axes. Default: the current axes.
%       'limits'        [lo hi] for every drawn axis. Default: one
%                       period for a periodic attribute, and otherwise
%                       the centres' own extent with room for the
%                       kernels around them.
%       'kSigma'        Kernels: the level surface drawn, in standard
%                       deviations. Default 1. Larger shows more of
%                       each kernel and hides more behind it. It sets
%                       the outline at two and three dimensions; at one
%                       the curve is drawn whole and this sets only how
%                       far a periodic kernel has to reach to be drawn
%                       again across a face.
%       'step'          Points and density: spacing of the evaluated
%                       grid, in the density's own units. Memory goes
%                       as its dim-th power. Default for density: the
%                       range over 1200 at one and two dimensions and
%                       over 120 at three, the grid being
%                       dim-dimensional so that the same count per axis
%                       costs wildly different amounts. For points, the
%                       spacing that
%                       puts the grid about three points apart on
%                       screen, since a grid finer than the marks drawn
%                       on it can only be shown by marks that overlap,
%                       which is reported once ('mpt:markOverlap').
%       'nodes'         Points and density: the step named as a count
%                       of steps across the range rather than as a
%                       spacing, the two being the same request --
%                       'nodes', 1200 over a range of 1200 is 'step',
%                       1. The grid then has one more point than this
%                       along each axis, both ends being on it. Give
%                       one or the other, not both.
%       'threshFrac'    Points: nodes below this fraction of the
%                       largest value are left undrawn. Default 0.001.
%       'markerSize'    Points: the mark's area in square points, or
%                       'auto', the default, for marks just wide enough
%                       to meet their neighbours on the grid. At three
%                       dimensions an automatic size is fitted for an
%                       assumed camera rather than the current one, so
%                       that turning the axes changes neither the marks
%                       nor the ink they lay down; it is refitted when
%                       the figure is resized or zoomed.
%       'markScale'     Points: scales the automatic size. Default 1,
%                       which at three dimensions holds the cloud's
%                       brightness roughly steady as the step changes:
%                       ink goes as the assumed foreshortening squared
%                       over the step, so the assumption rises as its
%                       square root. Raising it gives larger marks and
%                       a brighter cloud, and past the point where the
%                       marks meet they overlap, which at three
%                       dimensions shows as haloes on the blobs at any
%                       elevation, worse below the horizontal than
%                       above it. The drawing warns once
%                       ('mpt:markOverlap') when they overlap at the
%                       view being drawn.
%       'upsample'      Density at two and three dimensions: resampling
%                       within a plane, which sets how sharp it looks
%                       and costs as the square. Default 1, which
%                       leaves the plane at the volume's own
%                       resolution. What it buys is the resolution a
%                       texel is drawn at, so it shows where a texel
%                       covers several pixels -- a large 'step', or a
%                       close zoom. At a step of 5 it is not
%                       distinguishable from 1; at 10 it is. Raising it
%                       is expensive in a figure meant to be turned: at
%                       4, a frame of a rotation costs about eight
%                       times as much and changing stack about twelve.
%       'alphaPeak'     Points and density: the opacity the density's
%                       peak reaches -- for points the opacity of the
%                       brightest mark, for the density what a ray
%                       through the tallest blob's centre reaches.
%                       Default 1. At three dimensions this is not the
%                       opacity of the picture, a ray crossing several
%                       blobs on its way across; lower it to see into
%                       the cloud.
%       'alphaFloor'    Points, and density at two dimensions: the
%                       opacity where there is no density, the curve
%                       running from here to 'alphaPeak'. Default 0,
%                       so that empty space is empty. Set it to 1 to
%                       turn the fading off and draw the picture
%                       opaque. A three-dimensional density ignores
%                       it: there the value is read as an extinction
%                       per unit of path, and a floor would be a fog
%                       filling the whole cube.
%       'alphaGamma'    Points and density: the display curve on the
%                       opacity, as 'colourGamma' is the curve on the
%                       colour. 1 is opacity proportional to density
%                       (for a three-dimensional density, extinction
%                       proportional to it); above that thins the
%                       skirts and suppresses the low blobs, one of a
%                       quarter the height going as (1/4)^gamma.
%                       Default 1, except for 'density' at three
%                       dimensions, where it is 1.75. The two read the
%                       same number differently: a mark or a surface is
%                       a single layer, so the opacity computed is the
%                       opacity seen, while a stack shapes an
%                       extinction that compounds over every plane a
%                       ray crosses.
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
%       'view'          Three dimensions: [azimuth elevation] in
%                       degrees, as MATLAB's view, so a view found with
%                       the mouse can be read off and typed back in.
%                       Default [20 20].
%       'view2d'        Two dimensions: the same, default [0 90], which
%                       reads the surface as a map.
%       'dark'          Dark panes, the ground a glow is read against.
%                       Default true, at every dimensionality: the
%                       colour map runs from dark to bright, so
%                       anything coloured for a low density is nearly
%                       black and would be invisible on white.
%
%   Output
%       h    - The graphics object drawn: a patch for 'kernels' at two
%              and three dimensions and an array of lines at one, a
%              scatter for 'points', and for 'density' a line, a
%              surface, or the array of surfaces making up the stack in
%              view. A rotation may bring another stack into view,
%              after which the handles returned are no longer the ones
%              drawn.
%
%   Example
%       pAttr = {[0 200 400 500 700 900 1100].'};
%       specs = flatSpecs(pAttr, 'r', 4, 'rel', true, 'exch', true);
%       dens  = buildMaet(pAttr, [], 'specs', specs, 'sigma', 15, ...
%                         'isPer', true, 'period', 1200);
%       plotMaet(dens);                                  % the kernels
%       figure; plotMaet(dens, 'method', 'density');     % the density
%
%   The Python mirror is mpt.plot_maet. Its 'density' method covers one
%   and two dimensions alone: the three-dimensional one rests on
%   texture-mapped surfaces, which matplotlib has no counterpart for.
%
%   See also BUILDMAET, EVALMAET, MAETCENTRES.

if ~isstruct(dens) || ~isfield(dens, 'tag') ...
        || ~strcmp(dens.tag, 'MaetDensity')
    error('plotMaet:badInput', ...
          'Input must be a density struct from buildMaet.');
end

opt = localOptions(varargin{:});

dim = dens.dim;
% alphaGamma reads differently in the one place where opacity
% accumulates. At one and two dimensions a mark or a surface is a
% single layer, so the opacity computed is the opacity seen; at three
% the 'density' method reads the value as an extinction that compounds
% over every plane a ray crosses, and wants a stiffer curve. A value
% the caller gives is used as given.
if strcmpi(opt.method, 'density') && dim == 3 ...
        && any(strcmp('alphaGamma', opt.usedDefaults))
    opt.alphaGamma = 1.75;
end
if dim < 1 || dim > 3
    error('plotMaet:tooManyDimensions', ...
          ['One to three drawn dimensions can be drawn; this density ' ...
           'has %d.'], dim);
end
if dens.nAttrs ~= 1
    error('plotMaet:multiAttribute', ...
          ['plotMaet draws one attribute; this density has %d. ' ...
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

% 'nodes' is the same request as 'step' in the units a grid is usually
% thought about, so it is turned into a step here and nothing further
% down has to know about it.
if ~isempty(opt.nodes)
    if ~isempty(opt.step)
        error('plotMaet:stepAndNodes', ...
              ['''step'' and ''nodes'' say the same thing two ways, ' ...
               'so only one of them can be given.']);
    end
    opt.step = diff(lims) / round(opt.nodes);
end

% The current axes when none is named: newplot honours hold and makes
% a figure if there is none.
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
localFrame(ax, lims, dim, opt);

switch opt.method
    case 'kernels'
        h = localDrawKernels(ax, dens, C, covK, isPer, period, dim, opt);
    case 'points'
        if dim ~= 3
            error('plotMaet:pointsNeedsThreeDimensions', ...
                  ['''points'' exists because a three-dimensional ' ...
                   'density is hard to draw whole. Below that it only ' ...
                   'samples what ''density'' already draws: at two ' ...
                   'dimensions a surface carries every node at once, ' ...
                   'and at one a line does. This density has %d.'], dim);
        end
        h = localDrawPoints(ax, dens, lims, opt);
    case 'density'
        h = localDrawDensity(ax, dens, lims, sigma, dim, opt);
    otherwise
        error('plotMaet:badMethod', ...
              'method must be ''kernels'', ''points'', or ''density''.');
end

localFrame(ax, lims, dim, opt);

end


% -------------------------------------------------------------------------
function opt = localOptions(varargin)
%LOCALOPTIONS  The name-value pairs, with their defaults.
    p = inputParser;
    p.FunctionName = 'plotMaet';
    addParameter(p, 'method', 'kernels', @(s) ischar(s) || isstring(s));
    addParameter(p, 'axes', [], @(a) isempty(a) || isgraphics(a, 'axes'));
    addParameter(p, 'limits', [], @(v) isempty(v) || numel(v) == 2);
    addParameter(p, 'kSigma', 1, @(v) isscalar(v) && v > 0);
    addParameter(p, 'step', [], @(v) isempty(v) || (isscalar(v) && v > 0));
    addParameter(p, 'nodes', [], @(v) isempty(v) || (isscalar(v) && v >= 1));
    addParameter(p, 'threshFrac', 0.001, @(v) isscalar(v) && v >= 0);
    addParameter(p, 'markerSize', 'auto', ...
                 @(v) localIsAuto(v) || (isscalar(v) && v > 0));
    addParameter(p, 'markScale', 1, ...
                 @(v) isnumeric(v) && isscalar(v) && v > 0);
    addParameter(p, 'upsample', 1, @(v) isscalar(v) && v >= 1);
    addParameter(p, 'alphaPeak', 1, @(v) isscalar(v) && v > 0 && v <= 1);
    addParameter(p, 'alphaFloor', 0, @(v) isscalar(v) && v >= 0 && v <= 1);
    addParameter(p, 'alphaGamma', 1, @(v) isscalar(v) && v > 0);
    addParameter(p, 'colourGamma', 1, @(v) isscalar(v) && v > 0);
    addParameter(p, 'colormap', parula(256), @(m) size(m, 2) == 3);
    addParameter(p, 'brighten', 0.5, ...
                 @(v) isnumeric(v) && isscalar(v) && v > -1 && v < 1);
    addParameter(p, 'view', [20 20], @(v) numel(v) == 2);
    addParameter(p, 'view2d', [0 90], @(v) numel(v) == 2);
    addParameter(p, 'dark', true, @(v) islogical(v) || isnumeric(v));
    parse(p, varargin{:});
    opt = p.Results;
    opt.method = char(opt.method);
    % Which options were left to their defaults, for the few whose
    % default depends on what is being drawn. Resolved by the caller,
    % the drawn dimensionality not being known here.
    opt.usedDefaults = p.UsingDefaults;
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
        error('plotMaet:unsupportedKernelCov', ...
              ['This density carries an anisotropic kernel covariance ' ...
               'plotMaet cannot read; draw it with ''method'', ' ...
               '''density'', which takes the density as evaluated.']);
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


function h = localDrawKernels(ax, dens, C, covK, isPer, period, dim, opt)
%LOCALDRAWKERNELS  The kernels themselves, one per tuple centre.
%
%   An ellipsoid at three dimensions, an ellipse at two, a curve at
%   one. No grid is evaluated in any of them, so the cost is the number
%   of centres rather than the volume.
%
%   A centre whose kernel crosses a face of a periodic box is drawn
%   again on the other side, so that the wrap is cut by the face rather
%   than missing from it; the axes clips what falls outside.
    switch dim
        case 1
            h = localKernelCurves(ax, dens, C, covK, isPer, period, opt);
        case 2
            h = localKernelOutlines(ax, dens, C, covK, isPer, period, ...
                                    localUnitCircle(96), opt);
        otherwise
            h = localKernelEllipsoids(ax, dens, C, covK, isPer, period, opt);
    end
end


function h = localKernelEllipsoids(ax, dens, C, covK, isPer, period, opt)
%LOCALKERNELELLIPSOIDS  One mesh per centre, all in a single patch.
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
        shifts = localWrapShifts(C(:, j), reach, isPer, period, dim);
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


function V = localUnitCircle(n)
%LOCALUNITCIRCLE  Points on the unit circle, built here rather than
%   taken from the plotting library so that the two languages draw the
%   same outline.
    t = (0:(n - 1)).' * 2 * pi / n;
    V = [cos(t), sin(t)];
end


function h = localKernelOutlines(ax, dens, C, covK, isPer, period, unitV, opt)
%LOCALKERNELOUTLINES  One ellipse per centre, all in a single patch.
%
%   The same construction as the ellipsoids, a dimension down: the unit
%   circle carried through the kernel's Cholesky factor, filled and
%   coloured by the density at the centre.
    dim = size(C, 1);
    pv = evalMaet(dens, C, 'none', 'verbose', false);
    pv = pv(:);
    L = chol(covK, 'lower') * opt.kSigma;
    nVert = size(unitV, 1);
    reach = opt.kSigma * sqrt(diag(covK)).';

    verts = cell(1, 0);
    faces = cell(1, 0);
    vals = cell(1, 0);
    nSoFar = 0;
    for j = 1:size(C, 2)
        shifts = localWrapShifts(C(:, j), reach, isPer, period, dim);
        [S1, S2] = ndgrid(shifts{1}, shifts{2});
        for k = 1:numel(S1)
            c = C(:, j).' + [S1(k) S2(k)];
            verts{end+1} = unitV * L.' + c;                    %#ok<AGROW>
            faces{end+1} = (1:nVert) + nSoFar;                 %#ok<AGROW>
            vals{end+1} = repmat(pv(j), nVert, 1);             %#ok<AGROW>
            nSoFar = nSoFar + nVert;
        end
    end

    shade = (vertcat(vals{:}) / max(pv)) .^ opt.colourGamma;
    nMap = size(opt.colormap, 1);
    rgbV = opt.colormap(min(nMap, max(1, round(shade * (nMap - 1)) + 1)), :);
    h = patch('Parent', ax, 'Vertices', vertcat(verts{:}), ...
              'Faces', vertcat(faces{:}), 'FaceVertexCData', rgbV, ...
              'FaceColor', 'interp', 'EdgeColor', 'none');
end


function h = localKernelCurves(ax, dens, C, covK, isPer, period, opt)
%LOCALKERNELCURVES  One curve per centre, summing to the density.
%
%   At one dimension the kernels can be shown as what they are rather
%   than as an outline of where they reach: the density under
%   'normalize', 'none' is sum_j wJ(j) exp(-Q(c_j - x) / 2 sigma^2),
%   so each tuple's own term is a Gaussian of the kernel's width scaled
%   by that tuple's weight, and the curves drawn here add up to the
%   line the 'density' method draws.
    sd = sqrt(covK);
    reach = 4 * sd;
    % wJ is one of the expensive fields, which a lazily built density
    % does not yet carry.
    dens = internal.ensureMaetExpensive(dens);
    wJ = dens.wJ(:);
    pv = evalMaet(dens, C, 'none', 'verbose', false);
    pv = pv(:);
    nMap = size(opt.colormap, 1);
    shade = (pv / max(pv)) .^ opt.colourGamma;
    idx = min(nMap, max(1, round(shade * (nMap - 1)) + 1));

    t = linspace(-reach, reach, 129);
    h = gobjects(0);
    for j = 1:size(C, 2)
        shifts = localWrapShifts(C(:, j), reach, isPer, period, 1);
        for sh = shifts{1}
            x = C(1, j) + sh + t;
            y = wJ(j) * exp(-(t .^ 2) / (2 * covK));
            h(end + 1) = line('Parent', ax, 'XData', x, 'YData', y, ...
                              'Color', opt.colormap(idx(j), :), ...
                              'LineWidth', 1); %#ok<AGROW>
        end
    end
    colormap(ax, opt.colormap);
end


function shifts = localWrapShifts(c, reach, isPer, period, dim)
%LOCALWRAPSHIFTS  The copies of a centre a periodic box calls for.
%
%   Zero always, and one period either way for each coordinate whose
%   kernel reaches past a face.
    shifts = cell(1, dim);
    for i = 1:dim
        sVals = 0;
        if isPer
            if c(i) + reach(min(i, numel(reach))) > period
                sVals = [sVals, -period]; %#ok<AGROW>
            end
            if c(i) - reach(min(i, numel(reach))) < 0
                sVals = [sVals, period];  %#ok<AGROW>
            end
        end
        shifts{i} = sVals;
    end
end


function h = localDrawPoints(ax, dens, lims, opt)
%LOCALDRAWPOINTS  One translucent mark per grid node above the
%   threshold.
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
    [V, g] = localVolume(dens, lims, step, 3);
    [Ga, Gb, Gc] = ndgrid(g, g, g);
    keep = V > opt.threshFrac * max(V(:));
    vals = V(keep);
    rel = vals / max(vals);
    shade = rel .^ opt.colourGamma;

    % Opacity read from its own curve rather than from the colour's, so
    % that the low material can be lifted or suppressed without
    % flattening the colours along with it.
    faceAlpha = localAlphaCurve(rel, opt);
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
         'density with ''method'', ''density'', which composites per ' ...
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
    box = localAxesPoints(ax);
    ranges = [diff(xlim(ax)), diff(ylim(ax)), diff(zlim(ax))];
    ranges = max(ranges(:).', eps);

    if ~isempty(assumedFore)
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
%   again, which would stall a drag.
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
    if ~isappdata(fig, 'plotMaetPrevSizeFcn')
        setappdata(fig, 'plotMaetPrevSizeFcn', get(fig, 'SizeChangedFcn'));
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
    prev = getappdata(fig, 'plotMaetPrevSizeFcn');
    if isa(prev, 'function_handle')
        prev(fig, evt);
    elseif iscell(prev) && ~isempty(prev)
        feval(prev{1}, fig, evt, prev{2:end});
    elseif (ischar(prev) || isstring(prev)) && strlength(string(prev)) > 0
        evalin('base', char(prev));
    end
end


function h = localDrawDensity(ax, dens, lims, sigma, dim, opt)
%LOCALDRAWDENSITY  The density itself: a line, a surface, or a stack.
%
%   At one dimension the density is a curve and there is nothing to see
%   through. At two it is a surface, and at three a stack of planes,
%   both of them taking their opacity from the value so that the low
%   material fades to the panes rather than flooring at the colour
%   map's low colour -- which, painted opaque, would become the ground
%   the density is read against.
    step = opt.step;
    if isempty(step)
        % A node count, not a spacing: the grid is dim-dimensional, so
        % the same count per axis costs wildly different amounts at one
        % and at three. 1200 nodes is a fine grid at one and two
        % dimensions and cheap at both; at three the same would be
        % 1200^3 points, so the count drops to what a cube can carry.
        if dim < 3
            step = diff(lims) / 1200;
        else
            step = diff(lims) / 120;
        end
    end
    switch dim
        case 1
            h = localDensityLine(ax, dens, lims, step, opt);
        case 2
            h = localDensitySurface(ax, dens, lims, step, opt);
        otherwise
            h = localDensityStack(ax, dens, lims, sigma, step, opt);
    end
end


function a = localAlphaCurve(rel, opt)
%LOCALALPHACURVE  Opacity from the density, between floor and peak.
%
%   The curve runs from alphaFloor at nothing to alphaPeak at the
%   density's own peak, so a floor of 1 turns the scaling off and
%   leaves the picture opaque.
    a = opt.alphaFloor + (opt.alphaPeak - opt.alphaFloor) ...
        * rel .^ opt.alphaGamma;
end


function h = localDensityLine(ax, dens, lims, step, opt)
%LOCALDENSITYLINE  The density over its grid.
    [V, g] = localVolume(dens, lims, step, 1);
    h = line('Parent', ax, 'XData', g, 'YData', V, 'LineWidth', 1.5, ...
             'Color', opt.colormap(round(0.75 * size(opt.colormap, 1)), :));
end


function h = localDensitySurface(ax, dens, lims, step, opt)
%LOCALDENSITYSURFACE  The density as one textured, translucent sheet.
%
%   Texture mapping for both the colour and the opacity. FaceColor and
%   FaceAlpha have to agree, and of the pairs that do, only this one
%   renders: per-vertex opacity over a grid this size comes out blank.
    [V, g] = localVolume(dens, lims, step, 2);
    [Ga, Gb] = ndgrid(g, g);
    rel = max(V, 0) / max(max(V(:)), eps);
    h = surface(ax, Ga, Gb, V, ...
                'FaceColor', 'texturemap', 'FaceAlpha', 'texturemap', ...
                'CData', V, ...
                'AlphaData', localAlphaCurve(rel, opt), ...
                'AlphaDataMapping', 'none', 'EdgeColor', 'none');
    set(ax, 'ALim', [0 1]);
    colormap(ax, opt.colormap);
end


function h = localDensityStack(ax, dens, lims, sigma, step, opt)
%LOCALDENSITYSTACK  The volume as a stack of textured planes.
    dim = 3;
    [V, g] = localVolume(dens, lims, step, dim);
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
    groups(1).DeleteFcn = @(~, ~) localDensityGone(ax, prevSort);

    localSetState(ax, 'Density', ...
                  struct('stacks', {stacks}, 'groups', groups, ...
                         'axis', 0, 'handles', gobjects(0)));
    localPickStack(ax);

    % The camera moves throughout a turn, so the choice follows it
    % rather than waiting for the turn to end. Which property a camera
    % move sets depends on how it is made: dragging the axes sets
    % CameraPosition and leaves View derived from it, while view() sets
    % View and leaves CameraPosition derived, and a derived property
    % raises nothing. Both are watched for that reason.
    d = localGetState(ax, 'Density');
    d.listener = addlistener(ax, ...
        {'View', 'CameraPosition', 'CameraTarget', 'CameraUpVector'}, ...
        'PostSet', @(~, ~) localPickStack(ax));
    localSetState(ax, 'Density', d);
    h = d.handles;
end


function localDensityGone(ax, prevSort)
%LOCALDENSITYGONE  Put the axes back as the stack found it.
    if ~isgraphics(ax, 'axes')
        return
    end
    ax.SortMethod = prevSort;
    localClearState(ax, 'Density');
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


function [V, g] = localVolume(dens, lims, step, dim)
%LOCALVOLUME  The density on a grid of DIM dimensions.
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
    switch dim
        case 1
            V = reshape(evalMaet(dens, g, 'none', 'verbose', false), 1, nG);
        case 2
            [Ga, Gb] = ndgrid(g, g);
            v = evalMaet(dens, [Ga(:).'; Gb(:).'], 'none', 'verbose', false);
            V = reshape(v, nG, nG);
        otherwise
            % A slab at a time: the whole cube of query points is a
            % large array to hold at once.
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
    d = localGetState(ax, 'Density');
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
    localSetState(ax, 'Density', d);
end


function localSetState(ax, which, state)
%LOCALSETSTATE  Keep a method's state on the axes, under its own key.
%
%   Each method has a key of its own, so that drawing one into an axes
%   that already holds another does not overwrite its state and leave
%   the earlier drawing's listener stranded. Application data rather
%   than UserData, which is the caller's to use.
    setappdata(ax, ['plotMaet' which], state);
end


function state = localGetState(ax, which)
%LOCALGETSTATE  A method's state, or empty if the axes holds none.
    key = ['plotMaet' which];
    if isappdata(ax, key)
        state = getappdata(ax, key);
    else
        state = [];
    end
end


function localClearState(ax, which)
%LOCALCLEARSTATE  Forget a method's state.
    key = ['plotMaet' which];
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
    % a ray ends up with; the accumulated opacity is a product and so
    % is the same either way, and the difference measures 0.4%.
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


function localFrame(ax, lims, dim, opt)
%LOCALFRAME  The box the density is drawn in.
%
%   Square at two and three dimensions, the drawn coordinates covering
%   the same range as each other; at one the vertical axis is the
%   density's own and is left to the axes.
%
%   Dark panes at every dimensionality, including one: the colour map
%   runs from dark to bright, so a curve coloured for a low density is
%   nearly black and would be invisible against a white ground.
    set(ax, 'Projection', 'orthographic');
    xlim(ax, lims);
    switch dim
        case 1
            localDarkPanes(ax, opt);
        case 2
            ylim(ax, lims);
            pbaspect(ax, [1 1 1]);
            view(ax, opt.view2d);
            localDarkPanes(ax, opt);
        otherwise
            ylim(ax, lims);
            zlim(ax, lims);
            pbaspect(ax, [1 1 1]);
            view(ax, opt.view);
            localDarkPanes(ax, opt);
    end
end


function localDarkPanes(ax, opt)
%LOCALDARKPANES  Dark panes and a dim grid, or the axes' own.
    if opt.dark
        set(ax, 'Color', [0.06 0.06 0.06], ...
                'GridColor', [0.22 0.22 0.22], 'GridAlpha', 1);
    end
    set(ax, 'XGrid', 'on', 'YGrid', 'on', 'ZGrid', 'on');
end
