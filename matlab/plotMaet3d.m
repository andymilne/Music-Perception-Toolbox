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
%   plotMaet3d(dens, 'method', 'slices') draws the density itself,
%   as a stack of textured planes square to whichever axis is most
%   nearly square to the view, each carrying the density as its colour
%   and its opacity. The stack is rebuilt when a rotation turns far
%   enough to want another. This is a volume rendering: the value is
%   read as an extinction per unit of path, so the picture does not
%   depend on how finely the volume is cut.
%
%   The two answer different questions. The ellipsoids show where the
%   kernels are and what shape they have, but where kernels overlap
%   they show the kernels and not the sum they make; the slices show
%   the density, including everything between the peaks.
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
%                       volume, in the density's own units. Default:
%                       the range over 120. Memory goes as its cube.
%       'threshFrac'    Points: nodes below this fraction of the
%                       largest value are left undrawn. Default 0.02.
%       'markerSize'    Points: the mark's area in square points.
%                       Default 10.
%       'upsample'      Slices: resampling within a plane, which sets
%                       how sharp it looks and costs as the square.
%                       Default 4.
%       'depthUpsample' Slices: planes inserted between those the
%                       volume has, which sets how finely the volume is
%                       cut along the view. Default 3. A blob spanning
%                       only three or four planes reads as the slices
%                       it is made of.
%       'alphaPeak'     Slices: what a ray through the tallest blob's
%                       centre reaches. Default 1. This is not the
%                       opacity of the picture, a ray crossing several
%                       blobs on its way across; lower it to see into
%                       the cloud.
%       'alphaGamma'    Slices: 1 is extinction proportional to
%                       density; above that thins the skirts and also
%                       suppresses the low blobs, one of a quarter the
%                       height going as (1/4)^gamma. Default 3.
%       'colourGamma'   Display curve on the colour. Default 0.5.
%       'colormap'      Colour map, an M-by-3 matrix. Default hot(256).
%       'view'          [elevation azimuth] in degrees. Default
%                       [20 -70].
%       'dark'          Dark panes for the cube, the ground a glow is
%                       read against. Default true.
%
%   Output
%       h    - The graphics object drawn: a patch for 'ellipsoids', or
%              the array of surfaces making up the current stack for
%              'slices'.
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
    addParameter(p, 'threshFrac', 0.02, @(v) isscalar(v) && v >= 0);
    addParameter(p, 'markerSize', 10, @(v) isscalar(v) && v > 0);
    addParameter(p, 'upsample', 4, @(v) isscalar(v) && v >= 1);
    addParameter(p, 'depthUpsample', 3, @(v) isscalar(v) && v >= 1);
    addParameter(p, 'alphaPeak', 1, @(v) isscalar(v) && v > 0 && v <= 1);
    addParameter(p, 'alphaGamma', 3, @(v) isscalar(v) && v > 0);
    addParameter(p, 'colourGamma', 0.5, @(v) isscalar(v) && v > 0);
    addParameter(p, 'colormap', hot(256), @(m) size(m, 2) == 3);
    addParameter(p, 'view', [20 -70], @(v) numel(v) == 2);
    addParameter(p, 'dark', true, @(v) islogical(v) || isnumeric(v));
    parse(p, varargin{:});
    opt = p.Results;
    opt.method = char(opt.method);
    opt.upsample = round(opt.upsample);
    opt.depthUpsample = round(opt.depthUpsample);
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
        step = diff(lims) / 120;
    end
    [V, g] = localVolume(dens, lims, step);
    [Ga, Gb, Gc] = ndgrid(g, g, g);
    keep = V > opt.threshFrac * max(V(:));
    vals = V(keep);
    shade = (vals / max(vals)) .^ opt.colourGamma;
    h = scatter3(ax, Ga(keep), Gb(keep), Gc(keep), opt.markerSize, ...
                 shade, 'filled');
    set(h, 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 'flat', ...
           'AlphaData', shade);
    colormap(ax, opt.colormap);
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
                   'step', step, 'up', opt.upsample, ...
                   'depthUp', opt.depthUpsample);
    % The stack's state is kept on the axes rather than on the figure,
    % so that a figure holding several of these does not have one
    % overwrite another's.
    ax.UserData = struct('V', V, 'g', g, 'lims', lims, 'opt', stack, ...
                         'axis', 0, 'handles', gobjects(0));
    localPickStack(ax, []);
    rot = rotate3d(ancestor(ax, 'figure'));
    rot.ActionPostCallback = @localPickStack;
    rot.Enable = 'on';
    h = ax.UserData.handles;
end


function [V, g] = localVolume(dens, lims, step)
%LOCALVOLUME  The density on a cubic grid, a slab of planes at a time,
%   the whole cube of query points being a large array to hold at once.
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


function localPickStack(obj, evd)
%LOCALPICKSTACK  Build the stack square to whichever axis is most
%   nearly square to the view, in place of whichever is there.
%
%   Only one stack exists at a time: three would be several times the
%   texture for no gain, two of them never being seen. The correction
%   for how obliquely the rays cross the planes, a factor of
%   1/cos(theta) and at most 1.73, is not applied, a plane's opacity
%   not being changeable once it is built.
    ax = [];
    if nargin > 1 && ~isempty(evd)
        try
            ax = evd.Axes;                  % from the rotate mode
        catch
            ax = [];
        end
    end
    if isempty(ax) || ~isgraphics(ax, 'axes')
        ax = obj;                           % called directly
    end
    if ~isgraphics(ax, 'axes'), return, end
    d = ax.UserData;
    if ~isstruct(d) || ~isfield(d, 'V'), return, end
    w = ax.CameraPosition - ax.CameraTarget;
    w = w / norm(w);
    [~, a] = max(abs(w));
    if a == d.axis, return, end
    delete(d.handles(isgraphics(d.handles)));
    d.handles = localBuildStack(ax, d.V, d.g, d.lims, d.opt, a);
    d.axis = a;
    ax.UserData = d;
end


function hs = localBuildStack(ax, V, g, lims, opt, a)
%LOCALBUILDSTACK  One stack of textured planes square to axis A.
%
%   Every plane is given its colour and its opacity in the call that
%   creates it: AlphaData set then is honoured, and a plane made blank
%   and filled in afterwards is not.
%
%   Planes are inserted between those the volume has by interpolating
%   along the stack's own axis, and each then stands for a shorter piece
%   of the ray, so its opacity is worked out from that shorter spacing.
    n = numel(g);
    nOut = (n - 1) * opt.depthUp + 1;
    pos = linspace(g(1), g(end), nOut);
    planeStep = opt.step / opt.depthUp;
    nMap = size(opt.map, 1);
    hs = gobjects(nOut, 1);
    for m = 1:nOut
        f = (m - 1) / opt.depthUp + 1;
        i0 = min(floor(f), n - 1);
        wgt = f - i0;
        vImg = (1 - wgt) * localPlaneOf(V, a, i0) ...
               + wgt * localPlaneOf(V, a, i0 + 1);
        rel = max(vImg, 0) / opt.vMax;
        cIdx = min(nMap, max(1, round(rel .^ opt.colourGamma ...
                                      * (nMap - 1)) + 1));
        rgbImg = reshape(opt.map(cIdx(:), :), [size(cIdx), 3]);
        aImg = 1 - exp(-opt.extinction * rel .^ opt.alphaGamma * planeStep);
        if opt.up > 1
            rgbImg = localResample(rgbImg, opt.up);
            aImg = localResample(aImg, opt.up);
        end
        [X, Y, Z] = localPlaneQuad(a, pos(m), lims);
        hs(m) = surface(ax, X, Y, Z, 'FaceColor', 'texturemap', ...
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
    view(ax, [cosd(opt.view(1)) * cosd(opt.view(2)), ...
              cosd(opt.view(1)) * sind(opt.view(2)), sind(opt.view(1))]);
    set(ax, 'Projection', 'orthographic');
    xlim(ax, lims); ylim(ax, lims); zlim(ax, lims);
    pbaspect(ax, [1 1 1]);
    if opt.dark
        set(ax, 'Color', [0.06 0.06 0.06], ...
                'GridColor', [0.22 0.22 0.22], 'GridAlpha', 1);
    end
    grid(ax, 'on');
end
