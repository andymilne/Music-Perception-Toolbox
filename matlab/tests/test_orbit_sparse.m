%% test_orbit_sparse.m — Sparse-culled orbit inner product (v2.2)
%
%  Tests for the sparse-orbit fast path in matlab/+mobius/:
%    innerProductOrbitSparse (min-degree elimination on a sparse kernel)
%    and its wiring into maPerAttrInnerMatrix via a size/density gate.
%
%  Mirrors python/tests/test_orbit_sparse.py:
%    - the sparse engine equals the dense innerProductOrbit (value and
%      cancellation ratio) across all shipped arities r = 2..8;
%    - the wired mobius path (which takes the sparse gate on a large,
%      well-separated, non-periodic kernel) agrees with the independent
%      Bulger method under truncation.
%
%  Standalone-runnable; appends to `results` when invoked from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end


%% ---- Engine: sparse equals dense innerProductOrbit, r = 2..8 ----

n = 15;
pX = (linspace(0, 200, n)).';
pY = pX + 2.5;
sigma = 42;
D = pX - pY.';
Kd = exp(-(D .^ 2) / (4 * sigma^2));
Kd(D .^ 2 > 2 * (6 * sigma)^2) = 0;      % truncate at 6 sigma
Ks = sparse(Kd);
wA = (0.5 + 0.06 * (0:n - 1)).';         % 0.50 .. 1.34
wB = (1.3 - 0.05 * (0:n - 1)).';         % 1.30 .. 0.60

engine_ok = true;
ratio_ok = true;
for r = 2:8
    [vd, rd] = mobius.innerProductOrbit(Kd, wA, wB, r, ...
        'prefactor', 1.9, 'returnCancellationRatio', true);
    [vs, rs] = mobius.innerProductOrbitSparse(Ks, wA, wB, r, ...
        'prefactor', 1.9, 'returnCancellationRatio', true);
    if abs(vd - vs) / abs(vd) >= 1e-12
        engine_ok = false;
    end
    if abs(rd - rs) >= 1e-12
        ratio_ok = false;
    end
end
results{end + 1, 1} = 'orbitSparse: engine matches dense innerProductOrbit (r=2..8)';
results{end, 2}   = engine_ok;
results{end + 1, 1} = 'orbitSparse: cancellation ratio matches dense (r=2..8)';
results{end, 2}   = ratio_ok;


%% ---- Wired gate: mobius (sparse) agrees with Bulger, large clustered ----

% 8 well-separated clusters of 60 evenly-spaced slots => 480 slots per
% attribute (Kx*Ky = 230400 >= gate floor) at low density: the sparse
% gate fires for the non-periodic, r>=2 safe submatrix.
mkx = @(k) (600 * k + linspace(0, 70, 60)).';
mkxB = @(k) (600 * k + linspace(0, 68, 60)).';
mky = @(k) (600 * k + 3 + linspace(0, 70, 60)).';
mkyB = @(k) (600 * k + 2 + linspace(0, 68, 60)).';
PxA = cell2mat(arrayfun(mkx,  (0:7).', 'UniformOutput', false));
PxB = cell2mat(arrayfun(mkxB, (0:7).', 'UniformOutput', false));
PyA = cell2mat(arrayfun(mky,  (0:7).', 'UniformOutput', false));
PyB = cell2mat(arrayfun(mkyB, (0:7).', 'UniformOutput', false));

dx = buildExpTens({PxA, PxB}, {[]; []}, [50 50], [2 2], ...
    [false false], [false false], [0 0], 'verbose', false);
dy = buildExpTens({PyA, PyB}, {[]; []}, [50 50], [2 2], ...
    [false false], [false false], [0 0], 'verbose', false);

% Finite truncation (the isolate-defaults baseline pins Inf) so the cull
% and the sparse gate are active: the probe sees a low-density culled
% kernel and the safe submatrix takes the sparse per-pair path.
s_sparse = cosSimExpTens(dx, dy, 'method', 'mobius', ...
    'truncationSigmas', 6, 'verbose', false);
% Inf truncation makes the density probe see a full kernel, so the gate
% stays dormant and the same orbit inner product is computed densely.
% (Bulger is infeasible here -- it materialises the joint centres tensor
%  -- which is exactly why the orbit method exists.) For these clusters,
% 600 apart at sigma = 50, the 6-sigma cull drops only ~exp(-36) mass, so
% the sparse and dense orbit results agree to floating point.
s_dense = cosSimExpTens(dx, dy, 'method', 'mobius', ...
    'truncationSigmas', Inf, 'verbose', false);

results{end + 1, 1} = 'orbitSparse: wired mobius sparse path matches dense orbit (large clustered)';
results{end, 2}   = abs(s_sparse - s_dense) < 1e-5;


%% ---- Standalone reporting ----
if standalone
    fprintf('\n%s\n', repmat('=', 1, 60));
    nPass = 0;
    for i = 1:size(results, 1)
        status = 'FAIL';
        if results{i, 2}
            status = 'pass';
            nPass = nPass + 1;
        end
        fprintf('  [%s] %s\n', status, results{i, 1});
    end
    fprintf('%s\n  %d/%d passed\n', repmat('=', 1, 60), nPass, size(results, 1));
end
