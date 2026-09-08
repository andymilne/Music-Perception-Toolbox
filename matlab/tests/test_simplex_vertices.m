%% test_simplex_vertices.m — simplexVertices
%
%  Tests for simplexVertices.
%
%  Standalone-runnable; appends to `results` when called from
%  test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end



% (Pairwise distances via local helper pairwiseDistances at end of file,
%  avoiding pdist / Statistics Toolbox.)

% Shapes: simplexVertices(N) returns N x (N-1).
for N = [2 3 4 5 10]
    V = simplexVertices(N);
    results{end+1,1} = sprintf('simplexVertices: shape (N=%d)', N); %#ok<SAGROW>
    results{end,2}   = isequal(size(V), [N, N-1]);
end

% Centroid is at the origin.
for N = [2 3 4 5 7]
    V = simplexVertices(N);
    results{end+1,1} = sprintf('simplexVertices: centroid at origin (N=%d)', N); %#ok<SAGROW>
    results{end,2}   = max(abs(mean(V, 1))) < 1e-12;
end

% Default edge length is 1: all pairwise distances equal 1.
for N = [2 3 4 5 7]
    V = simplexVertices(N);
    D = pairwiseDistances(V);
    results{end+1,1} = sprintf('simplexVertices: unit edge length (N=%d)', N); %#ok<SAGROW>
    results{end,2}   = max(abs(D - 1)) < 1e-12;
end

% Custom edge length scales correctly.
for L = [0.5 2.0 100.0]
    V = simplexVertices(4, L);
    D = pairwiseDistances(V);
    results{end+1,1} = sprintf('simplexVertices: edge length %g', L); %#ok<SAGROW>
    results{end,2}   = max(abs(D - L)) < 1e-10;
end

% N = 2 collapses to a 1-D pair, distance 1, centred at origin.
V = simplexVertices(2);
results{end+1,1} = 'simplexVertices: N=2 collapses to 1-D';
results{end,2}   = isequal(size(V), [2, 1]) ...
                   && abs(V(1) + V(2)) < 1e-12 ...
                   && abs(abs(V(1) - V(2)) - 1) < 1e-12;

% Error paths: N < 2, non-positive edge length.
results{end+1,1} = 'simplexVertices: N=1 errors';
results{end,2}   = throwsError(@() simplexVertices(1));

results{end+1,1} = 'simplexVertices: N=0 errors';
results{end,2}   = throwsError(@() simplexVertices(0));

results{end+1,1} = 'simplexVertices: negative edge length errors';
results{end,2}   = throwsError(@() simplexVertices(3, -1));

results{end+1,1} = 'simplexVertices: zero edge length errors';
results{end,2}   = throwsError(@() simplexVertices(3, 0));


%% ---- Standalone summary ----

% Orientation is pinned: the coordinates themselves, not merely the shape
% they form. A regular simplex is defined only up to rotation, so every
% test above passes under any orientation and none can see a change of
% basis -- which is exactly how the two implementations came to disagree,
% one taking its basis from orth and the other from an SVD, on a matrix
% whose singular vectors are not determined. These are the pinned values,
% shared with the Python twin.
V3ref = [0.5, 0.28867513459481287; ...
         -0.5, 0.28867513459481287; ...
         0, -0.5773502691896257];
V3 = simplexVertices(3);
results{end+1,1} = 'simplexVertices: orientation pinned (N=3)'; %#ok<SAGROW>
results{end,2}   = max(abs(V3(:) - V3ref(:))) < 1e-12;

V4ref = [0.5, 0.28867513459481287, 0.20412414523193154; ...
         -0.5, 0.28867513459481287, 0.20412414523193154; ...
         0, -0.5773502691896257, 0.20412414523193154; ...
         0, 0, -0.6123724356957945];
V4 = simplexVertices(4);
results{end+1,1} = 'simplexVertices: orientation pinned (N=4)'; %#ok<SAGROW>
results{end,2}   = max(abs(V4(:) - V4ref(:))) < 1e-12;

% Two levels sit at +1/2 and -1/2, the first level positive.
V2 = simplexVertices(2);
results{end+1,1} = 'simplexVertices: binary case is +/- 0.5'; %#ok<SAGROW>
results{end,2}   = max(abs(V2(:) - [0.5; -0.5])) < 1e-12;

% Nesting: the first m vertices, on their first m-1 coordinates, are the
% m-simplex, so adding a level extends the coordinates rather than moving
% the levels already there. This is the property the basis is pinned for.
nestOk = true;
for N = [4 5 6]
    V = simplexVertices(N);
    for m = 2:N
        nestOk = nestOk && max(max(abs(V(1:m, 1:m-1) - simplexVertices(m)))) < 1e-12;
    end
end
results{end+1,1} = 'simplexVertices: nesting across N'; %#ok<SAGROW>
results{end,2}   = nestOk;


if standalone
    nPass = sum([results{:, 2}]);
    nFail = size(results, 1) - nPass;
    for i = 1:size(results, 1)
        if results{i, 2}
            fprintf('  PASS  %s\n', results{i, 1});
        else
            fprintf('  FAIL  %s\n', results{i, 1});
        end
    end
    fprintf('\n=== test_simplex_vertices: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_simplex_vertices:failed', '%d test(s) failed.', nFail);
    end
end


%% ---- Local helpers ----


function D = pairwiseDistances(V)
%PAIRWISEDISTANCES Pairwise Euclidean distances between rows of V.
%
%   D = pairwiseDistances(V) returns a column vector of length
%   N*(N-1)/2 containing the pairwise Euclidean distances between
%   the N rows of V, in the order
%   (1,2), (1,3), (2,3), (1,4), (2,4), (3,4), ...
%
%   Computed via the Gram matrix:
%       D2(i,j) = ||v_i||^2 + ||v_j||^2 - 2 * v_i * v_j'
%   so no Statistics Toolbox is required.
    n = size(V, 1);
    sqNorm = sum(V .* V, 2);
    D2 = sqNorm + sqNorm' - 2 * (V * V');
    D2 = max(D2, 0);                       % clamp tiny negatives from FP error
    mask = triu(true(n), 1);               % strict upper triangle
    D = sqrt(D2(mask));
end

