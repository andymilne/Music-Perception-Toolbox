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

