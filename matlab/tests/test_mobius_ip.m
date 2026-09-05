%% test_mobius_ip.m — Orbit IP evaluators and total mass formulae
%
%  Tests for the v3 orbit-IP machinery in matlab/+mobius/:
%    contract, innerProductOrbit, innerProductOrbitGrid,
%    innerProductOrbitPwBatched, totalMassAbs, totalMassRel.
%
%  Mirrors python/tests/test_mobius.py (IP and total-mass sections).
%
%  Standalone-runnable; appends to `results` when invoked from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    % Defaults isolation when run standalone (when invoked from
    % test_mpt.m the outer wrapper has already isolated defaults).
    addpath(fileparts(mfilename('fullpath')));
    addpath(fullfile(fileparts(mfilename('fullpath')), 'reference'));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end


%% ---- Generic contract: hand-checked patterns ----

% T1: simple 2-matrix matmul.  A_ij B_jk -> C_ik
A = reshape(1:15, 5, 3); B = reshape(1:21, 3, 7);
got = reference.contract({A, B}, {[1 2], [2 3]}, [1 3]);
results{end+1, 1} = 'reference.contract: matmul';
results{end, 2}   = isequal(size(got), [5 7]) && max(abs(got(:) - reshape(A*B, [], 1))) < 1e-10;

% T2: trace via contract.
A = reshape(1:20, 4, 5); B = reshape(1:20, 5, 4);
got = reference.contract({A, B}, {[1 2], [2 1]}, []);
results{end+1, 1} = 'reference.contract: trace';
results{end, 2}   = isscalar(got) && abs(got - trace(A*B)) < 1e-10;

% T3: bilinear form u' K v.
u = (1:5)'; K = reshape(1:20, 5, 4); v = (1:4)';
got = reference.contract({u, K, v}, {[1], [1 2], [2]}, []);
results{end+1, 1} = 'reference.contract: bilinear form';
results{end, 2}   = isscalar(got) && abs(got - u' * K * v) < 1e-10;

% T4: shared-keep axis (3 operands sharing the same axis).
A = ones(2, 3); B = ones(2, 4); u = ones(2, 1);
got = reference.contract({A, B, u}, {[1 2], [1 3], [1]}, []);
% sum_{i,j,k} u_i A_ij B_ik with all ones = 2 * 3 * 4 = 24
results{end+1, 1} = 'reference.contract: shared axis across 3 operands';
results{end, 2}   = isscalar(got) && abs(got - 24) < 1e-10;

% T5: orbit-like pattern (axis appears in 1 weight + 2 kernels).
n_A = 5; n_B = 4;
u = (1:n_A)'; v = (1:n_B)'; K = reshape(linspace(0.1, 2, n_A*n_B), n_A, n_B);
got = reference.contract({u, v, v, K, K}, {[1], [2], [3], [1 2], [1 3]}, []);
expected = sum(u .* (K*v) .* (K*v));
results{end+1, 1} = 'reference.contract: orbit-like (axis in 1 weight + 2 K)';
results{end, 2}   = isscalar(got) && abs(got - expected) < 1e-10;


%% ---- innerProductOrbit vs direct enumeration ----

% Direct enumeration computes <T_A, T_B> by summing over all distinct
% ordered r-tuples on each side, the brute-force ground truth.
ok = true; allFinite = true;
seedBase = 42;
for r = 2:3
    for n = [6, 7]
        rng(seedBase + r * 100 + n, 'twister');
        p_A = sort(3000 * rand(n, 1));
        w_A = 0.5 + rand(n, 1);
        p_B = sort(3000 * rand(n, 1));
        w_B = 0.5 + rand(n, 1);
        sigma = 30.0;

        valDirect = v22_directIP(p_A, w_A, p_B, w_B, sigma, r);
        K = exp(-((p_A - p_B').^2) / (4 * sigma^2));
        valOrbit = mobius.innerProductOrbit(K, w_A, w_B, r, ...
            'prefactor', (sigma * sqrt(pi))^r);

        K_A = exp(-((p_A - p_A').^2) / (4 * sigma^2));
        K_B = exp(-((p_B - p_B').^2) / (4 * sigma^2));
        AA = mobius.innerProductOrbit(K_A, w_A, w_A, r, ...
            'prefactor', (sigma * sqrt(pi))^r);
        BB = mobius.innerProductOrbit(K_B, w_B, w_B, r, ...
            'prefactor', (sigma * sqrt(pi))^r);
        geoMean = sqrt(abs(AA * BB));

        if ~isfinite(valOrbit) || ~isfinite(valDirect)
            allFinite = false;
        end
        if geoMean == 0
            continue
        end
        cosErr = abs(valDirect - valOrbit) / geoMean;
        relErr = abs(valDirect - valOrbit) / max(abs(valDirect), 1e-300);

        % Accept either: small cosine-scale error (catastrophic
        % cancellation, but cosine value is fine) or small relative
        % error (healthy regime).
        if cosErr >= 1e-12 && relErr >= 1e-10
            ok = false;
        end
    end
end
results{end+1, 1} = 'mobius.innerProductOrbit: matches direct enumeration (r=2..3, n=6,7)';
results{end, 2}   = ok && allFinite;


%% ---- self-IP positivity at r=2..4 ----

ok = true;
for r = 2:4
    rng(7 + r, 'twister');
    n = 12;
    p = sort(5000 * rand(n, 1));
    w = 0.5 + rand(n, 1);
    sigma = 20.0;
    K = exp(-((p - p').^2) / (4 * sigma^2));
    val = mobius.innerProductOrbit(K, w, w, r, 'prefactor', (sigma * sqrt(pi))^r);
    if val <= 0
        ok = false;
    end
end
results{end+1, 1} = 'mobius.innerProductOrbit: <T,T> > 0 for r=2..4';
results{end, 2}   = ok;


%% ---- innerProductOrbitGrid: zero-shift matches static ----

rng(99, 'twister');
n = 8;
p_A = sort(1500 * rand(n, 1)); w_A = 0.5 + rand(n, 1);
p_B = sort(1500 * rand(n, 1)); w_B = 0.5 + rand(n, 1);
sigma = 30.0; r = 3;
K_static = exp(-((p_A - p_B').^2) / (4 * sigma^2));
K_u = reshape(K_static, [1, n, n]);
valStatic = mobius.innerProductOrbit(K_static, w_A, w_B, r);
valGrid = mobius.innerProductOrbitGrid(K_u, w_A, w_B, r);
results{end+1, 1} = 'mobius.innerProductOrbitGrid: zero-shift matches static';
results{end, 2}   = abs(valGrid(1) - valStatic) < 1e-10 * max(abs(valStatic), 1);


%% ---- innerProductOrbitGrid: each shift matches static at that shift ----

N_u = 5;
uGrid = linspace(-100, 100, N_u);
K_u_multi = zeros(N_u, n, n);
for ui = 1:N_u
    diffsU = (p_A - p_B') - uGrid(ui);
    K_u_multi(ui, :, :) = exp(-(diffsU.^2) / (4 * sigma^2));
end
valGridMulti = mobius.innerProductOrbitGrid(K_u_multi, w_A, w_B, r);
ok = true;
for ui = 1:N_u
    valS = mobius.innerProductOrbit(squeeze(K_u_multi(ui, :, :)), w_A, w_B, r);
    if abs(valGridMulti(ui) - valS) > 1e-10 * max(abs(valS), 1)
        ok = false;
    end
end
results{end+1, 1} = 'mobius.innerProductOrbitGrid: every shift matches static at that shift';
results{end, 2}   = ok;


%% ---- innerProductOrbitPwBatched: shared weights match grid ----

N = 4;
K_g = K_u_multi(1:N, :, :);
w_A_g = repmat(w_A', N, 1);
w_B_g = repmat(w_B', N, 1);
valPw = mobius.innerProductOrbitPwBatched(K_g, w_A_g, w_B_g, r);
results{end+1, 1} = 'mobius.innerProductOrbitPwBatched: shared weights match grid';
results{end, 2}   = max(abs(valPw - valGridMulti(1:N))) < 1e-10;


%% ---- innerProductOrbitPwBatched: independent batches give independent results ----

N = 3;
rng(123, 'twister');
K_g = zeros(N, n, n);
w_A_g = zeros(N, n);
w_B_g = zeros(N, n);
for b = 1:N
    pa = sort(1500 * rand(n, 1));
    pb = sort(1500 * rand(n, 1));
    K_g(b, :, :) = exp(-((pa - pb').^2) / (4 * sigma^2));
    w_A_g(b, :) = 0.5 + rand(1, n);
    w_B_g(b, :) = 0.5 + rand(1, n);
end
valPwIndep = mobius.innerProductOrbitPwBatched(K_g, w_A_g, w_B_g, r);
% Compare each batch to a static call.
ok = true;
for b = 1:N
    Kb = squeeze(K_g(b, :, :));
    valStaticB = mobius.innerProductOrbit(Kb, w_A_g(b, :)', w_B_g(b, :)', r);
    if abs(valPwIndep(b) - valStaticB) > 1e-10 * max(abs(valStaticB), 1)
        ok = false;
    end
end
results{end+1, 1} = 'mobius.innerProductOrbitPwBatched: per-batch result matches static';
results{end, 2}   = ok;


%% ---- totalMassAbs vs direct enumeration ----

ok = true;
for r = 1:4
    rng(11, 'twister');
    n = 6;
    p = sort(1000 * rand(n, 1));
    w = 0.5 + rand(n, 1);
    sigma = 25.0;

    tuples = v22_orderedTuples(n, r);
    directZ = 0.0;
    for ip = 1:size(tuples, 1)
        directZ = directZ + prod(w(tuples(ip, :)));
    end
    directZ = directZ * (sigma * sqrt(2 * pi))^r;

    formulaZ = mobius.totalMassAbs(p, w, sigma, r);
    relErr = abs(directZ - formulaZ) / abs(directZ);
    if relErr >= 1e-12
        ok = false;
    end
end
results{end+1, 1} = 'mobius.totalMassAbs: matches direct enumeration for r=1..4';
results{end, 2}   = ok;


%% ---- totalMassRel: scaling factor relation ----

r = 3; sigma = 25.0;
rng(7, 'twister');
w = 0.5 + rand(5, 1);
Zabs = mobius.totalMassAbs([], w, sigma, r);
Zrel = mobius.totalMassRel([], w, sigma, r);
expectedRatio = 1 / (sigma * sqrt(2 * pi / r));
results{end+1, 1} = 'mobius.totalMassRel: equals Z_abs / (sigma * sqrt(2 pi / r))';
results{end, 2}   = abs(Zrel / Zabs - expectedRatio) < 1e-14;


%% ---- Cancellation ratio is reported when requested ----

rng(4, 'twister');
n = 8; p = sort(1500 * rand(n, 1)); w = 0.5 + rand(n, 1); sigma = 30.0;
K = exp(-((p - p').^2) / (4 * sigma^2));
[val, ratio] = mobius.innerProductOrbit(K, w, w, 3, ...
    'prefactor', (sigma * sqrt(pi))^3, 'returnCancellationRatio', true);
results{end+1, 1} = 'mobius.innerProductOrbit: returns finite cancellation ratio';
results{end, 2}   = isfinite(val) && isfinite(ratio) && ratio >= 0 && ratio <= 1 + 1e-12;


%% ---- standalone summary ----

if standalone
    nPass = sum([results{:, 2}]);
    nFail = size(results, 1) - nPass;
    fprintf('\n=== mobius IP and total-mass tests ===\n\n');
    for i = 1:size(results, 1)
        if results{i, 2}
            fprintf('  PASS  %s\n', results{i, 1});
        else
            fprintf('  FAIL  %s\n', results{i, 1});
        end
    end
    fprintf('\n=== Results: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    % Restore caller's pre-test defaults eagerly
    % (fires the helper's onCleanup destructor before
    % the conditional error below halts execution).
    clear cleanupDefaults
    if nFail > 0
        error('test_mobius_ip:failed', '%d test(s) failed.', nFail);
    end
end
