%% test_bucket_padding.m — the padded bucket lattice in the truncated kernel sum
%
%  The spatial cull pitches a bucket lattice on the tuple centres and,
%  for each query, visits the centres in its own bucket's 3^dim
%  neighbourhood. The lattice carries one empty bucket of margin on
%  every face, so a neighbour of an occupied-region bucket can never
%  fall off it. That is what lets the neighbourhood be computed as
%  arithmetic on lattice strides rather than as a bounds-tested
%  coordinate array, and it puts the whole weight of the arrangement on
%  one claim: a bucket outside the occupied region contributes nothing,
%  whether it is skipped for lying out of bounds or visited and found
%  empty.
%
%  The queries that exercise the margin are the ones whose bucket is
%  clamped to the lattice edge — everything outside the centres'
%  bounding box, which for a density on a scale is most of a plotting
%  grid. What is pinned here is that the culled sum equals a direct sum
%  over the centres inside the truncation ball, for queries far outside
%  the box, on its faces, and within it.
%
%  Mirror of Python tests/test_bucket_padding.py.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

SIGMA  = 15;
KSIGMA = 6;
PERIOD = 1200;
SCALE  = [0 200 400 500 700 900 1100];

rng(0);

% --- the culled sum against a direct sum ---------------------------------

cases = {  {1, false, 'absolute, dim 1'}, ...
           {2, false, 'absolute, dim 2'}, ...
           {3, false, 'absolute, dim 3'}, ...
           {2, true,  'relative, dim 1'}, ...
           {3, true,  'relative, dim 2'}, ...
           {4, true,  'relative, dim 3'}  };

for ci = 1:numel(cases)
    r     = cases{ci}{1};
    isRel = cases{ci}{2};
    label = cases{ci}{3};

    specs = flatSpecs({SCALE(:)}, 'r', r, 'rel', isRel, 'exch', true);
    dens  = buildMaet({SCALE(:)}, [], 'specs', specs, 'sigma', SIGMA, ...
                      'isPer', false, 'period', PERIOD, 'verbose', false);
    Cc = maetCentres(dens);
    C  = Cc{1};
    wJ = linspace(0.5, 1.5, size(C, 2)).';

    X = localProbePoints(C, KSIGMA * SIGMA);
    got = internal.gaussianKernelSum(C, wJ, X, SIGMA, ...
                                     'isRel', isRel, 'r', r, ...
                                     'truncationSigmas', KSIGMA);
    want = localDirectSum(C, wJ, X, SIGMA, KSIGMA, isRel, r);

    results{end+1,1} = ['bucket padding: culled sum matches a direct ' ...
                        'sum (' label ')'];
    results{end,2}   = max(abs(got(:) - want(:))) < 1e-12;
end

% --- degenerate lattices --------------------------------------------------
% A lattice can be one bucket wide, or one bucket in total, and the
% margin has to hold there too.

% Every centre in a single bucket: the lattice is the margin and one cell.
C  = (rand(2, 40) * 10) - 5;
wJ = ones(40, 1);
X  = [-1e4 0 20 1e4; -1e4 0 -20 1e4];
got  = internal.gaussianKernelSum(C, wJ, X, SIGMA, 'truncationSigmas', KSIGMA);
want = localDirectSum(C, wJ, X, SIGMA, KSIGMA, false, 2);
results{end+1,1} = 'bucket padding: one occupied bucket';
results{end,2}   = max(abs(got(:) - want(:))) < 1e-12;

C  = [linspace(0, PERIOD, 13); zeros(1, 13)];
wJ = ones(13, 1);
X  = [(rand(1, 400) * 6000) - 3000; (rand(1, 400) * 600) - 300];
got  = internal.gaussianKernelSum(C, wJ, X, SIGMA, 'truncationSigmas', KSIGMA);
want = localDirectSum(C, wJ, X, SIGMA, KSIGMA, false, 2);
results{end+1,1} = 'bucket padding: centres collinear in one axis';
results{end,2}   = max(abs(got(:) - want(:))) < 1e-12;

C  = repmat(linspace(0, 100, 60), 3, 1);
wJ = ones(60, 1);
X  = 5000 + rand(3, 200) * 1000;
got = internal.gaussianKernelSum(C, wJ, X, SIGMA, 'truncationSigmas', KSIGMA);
results{end+1,1} = 'bucket padding: every query outside the box';
results{end,2}   = all(got(:) == 0);

% --- through the public entry point ---------------------------------------
% The demo case: a grid far wider than the scale it plots, so most of
% its points sit outside the centres' box.

specs = flatSpecs({SCALE(:)}, 'r', 2, 'rel', false, 'exch', true);
dens  = buildMaet({SCALE(:)}, [], 'specs', specs, 'sigma', SIGMA, ...
                  'isPer', false, 'period', PERIOD, 'verbose', false);
g = linspace(-2400, 3600, 200);
[ga, gb] = meshgrid(g, g);
pts = [ga(:).'; gb(:).'];
vTrunc = evalMaet(dens, pts, 'verbose', false);
vExact = evalMaet(dens, pts, 'truncationSigmas', Inf, 'verbose', false);
results{end+1,1} = ['bucket padding: a wide plotting grid agrees with ' ...
                    'the untruncated result'];
results{end,2}   = max(abs(vTrunc(:) - vExact(:))) < 1e-7;


if standalone
    nPass = 0; nFail = 0;
    for i = 1:size(results, 1)
        if results{i,2}
            nPass = nPass + 1;
            fprintf('  PASS  %s\n', results{i,1});
        else
            nFail = nFail + 1;
            fprintf('  FAIL  %s\n', results{i,1});
        end
    end
    fprintf(['\n=== test_bucket_padding: %d passed, %d failed ' ...
             '(of %d) ===\n\n'], nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_bucket_padding:failed', '%d test(s) failed.', nFail);
    end
end


function X = localProbePoints(C, radius)
%LOCALPROBEPOINTS  Queries inside the centres' box, on its faces, and
%   well outside it in every direction. The outside ones are the point
%   of the test: their buckets clamp to the lattice edge, so their
%   neighbourhoods reach into the margin.
    dim  = size(C, 1);
    lo   = min(C, [], 2);
    hi   = max(C, [], 2);
    span = max(hi - lo, 1);

    inside  = lo + rand(dim, 60) .* span;
    onFace  = [repmat(lo, 1, 10), repmat(hi, 1, 10)];
    outside = [lo - (0.01 + 2.99 * rand(dim, 60)) .* span, ...
               hi + (0.01 + 2.99 * rand(dim, 60)) .* span];
    % A few just beyond the truncation radius of the box, the distance
    % at which the answer turns from nonzero to exactly zero.
    justOut = hi + radius * (0.9 + 0.2 * rand(dim, 20));

    X = [inside, onFace, outside, justOut];
end


function v = localDirectSum(C, wJ, X, sigma, kSigma, isRel, r)
%LOCALDIRECTSUM  The quantity the cull is an optimisation of: every
%   centre within the truncation ball of the query, and no other.
    threshold2 = (kSigma * sigma)^2;
    inv2s2     = 1 / (2 * sigma^2);
    nQ = size(X, 2);
    v  = zeros(1, nQ);
    for q = 1:nQ
        D = C - X(:, q);
        if isRel
            qForm = sum(D.^2, 1) - sum(D, 1).^2 / r;
        else
            qForm = sum(D.^2, 1);
        end
        near = qForm <= threshold2;
        v(q) = sum(wJ(near).' .* exp(-qForm(near) * inv2s2));
    end
end
