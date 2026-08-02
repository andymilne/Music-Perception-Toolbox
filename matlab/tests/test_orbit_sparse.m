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

% 8 well-separated clusters of 60 evenly-spaced values => 480 values per
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


%% ---- innerProductOrbitSparse: term mass matches the grid engine ----
% The returnTermMass output must equal innerProductOrbitGrid's mass on
% the same (single-node) kernel: both report the prefactored
% max_orb(|term_orb|).
rng(21, 'twister');
tm_n = 14;
tm_pX = sort(200 * rand(tm_n, 1));
tm_pY = tm_pX + (rand(tm_n, 1) - 0.5) * 6;
tm_s = 40.0;
tm_D = tm_pX - tm_pY.';
tm_Kd = exp(-(tm_D.^2) / (4 * tm_s^2));
tm_wA = 0.5 + rand(tm_n, 1);
tm_wB = 0.5 + rand(tm_n, 1);
tm_ok = true;
for tm_r = 2:4
    [~, ~, massG] = mobius.innerProductOrbitGrid( ...
        reshape(tm_Kd, [1, tm_n, tm_n]), tm_wA, tm_wB, tm_r, ...
        'prefactor', 1.3, 'returnCancellationRatio', true);
    [~, ~, massS] = mobius.innerProductOrbitSparse( ...
        sparse(tm_Kd), tm_wA, tm_wB, tm_r, 'prefactor', 1.3, ...
        'returnCancellationRatio', true, 'returnTermMass', true);
    tm_ok = tm_ok && abs(massG(1) - massS) <= 1e-12 * max(massG(1), 1);
end
results{end + 1, 1} = 'orbitSparse: returnTermMass matches innerProductOrbitGrid mass (r=2..4)';
results{end, 2}   = tm_ok;


%% ---- Wired periodic relative sparse path (gate toggled by truncation) ----
% K = 450 values on the circle at sigma = 13, P = 1200: the value kernel
% has Kx*Ky = 202500 >= 200000 entries, and the circular truncation band
% occupies 2*sqrt(2)*k*sigma/P of each row --- 0.18 at k = 6 (below the
% 0.20 density ceiling, so the sparse route fires) but 0.23 at the
% Inf-resolved k = 7.43 (above it, so the gate stays dormant and the
% dense slab route runs). The two agree to the truncation gap
% (~1e-12 measured; 1e-5 asserted). Bulger is infeasible here --- its
% relative pairwise cost grows as K^(2r) --- which is exactly why the
% orbit method exists.
rng(23, 'twister');
rp_K = 450;
rp_P = 1200.0;
rp_sig = 13.0;
rp_p1 = sort(rp_P * rand(rp_K, 1));
rp_p2 = sort(rp_P * rand(rp_K, 1));
rp_w = ones(rp_K, 1);
rp_s6 = cosSimExpTens(rp_p1, rp_w, rp_p2, rp_w, rp_sig, 2, true, true, ...
    rp_P, 'method', 'mobius', 'truncationSigmas', 6, 'verbose', false);
rp_si = cosSimExpTens(rp_p1, rp_w, rp_p2, rp_w, rp_sig, 2, true, true, ...
    rp_P, 'method', 'mobius', 'truncationSigmas', Inf, 'verbose', false);
results{end + 1, 1} = 'orbitSparse: wired rel-per sparse path matches dense slab (gate toggle)';
results{end, 2}   = abs(rp_s6 - rp_si) < 1e-5;


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
