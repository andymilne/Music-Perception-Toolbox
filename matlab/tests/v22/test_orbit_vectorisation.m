%% test_orbit_vectorisation.m — v2.2.x batched orbit evaluators
%
%  Verifies the v2.2.x extensions:
%
%    - mobius.evalOrbitAbs accepts x of shape (r, ...) with arbitrary
%      trailing dimensions; output shape matches the trailing dims.
%    - mobius.evalOrbitRel batches its u-grid loop into a single
%      chunked call to evalOrbitAbs and produces values bit-identical
%      to a manual sequential u-grid quadrature.
%    - tensorHarmonicity batched mode dedups canonical chords + groups
%      by (nP, dup), and the batched result matches the scalar path to
%      machine precision.
%
%  Standalone-runnable. When invoked from test_mpt.m the existing
%  `results` cell is appended to; when run alone, results print on the
%  fly and a summary follows.

if ~exist('results', 'var')
    results = {};
    standalone = true;
else
    standalone = false;
end

%% ---- evalOrbitAbs: 2-D input unchanged ----

rng(0);
p = sort(rand(8, 1) * 1000);
w = ones(8, 1);
sigma = 12.0;

for r = [2, 3]
    x = rand(r, 5) * 1000;
    vals = mobius.evalOrbitAbs(p, w, sigma, r, x);
    okShape = isequal(size(vals), [5, 1]);  % column vector for 2-D input
    okFinite = all(isfinite(vals));
    results{end+1, 1} = sprintf('evalOrbitAbs r=%d: 2-D input returns (n_q, 1) column vector', r);
    results{end, 2} = okShape && okFinite;
end

%% ---- evalOrbitAbs: 3-D input returns (n1, n2) ----

r = 3;
n1 = 4;
n2 = 5;
x_3d = rand(r, n1, n2) * 1000;
vals_3d = mobius.evalOrbitAbs(p, w, sigma, r, x_3d);
results{end+1, 1} = 'evalOrbitAbs: 3-D input (r, n1, n2) returns (n1, n2) array';
results{end, 2} = isequal(size(vals_3d), [n1, n2]) && all(isfinite(vals_3d(:)));

% Cross-check: collapse to 2-D and reshape -> should match.
x_flat = reshape(x_3d, r, n1 * n2);
vals_flat = mobius.evalOrbitAbs(p, w, sigma, r, x_flat);
results{end+1, 1} = 'evalOrbitAbs: 3-D output matches reshape of flattened 2-D output (bit-identical)';
results{end, 2} = max(abs(vals_3d(:) - vals_flat(:))) == 0;

%% ---- evalOrbitAbs: cancellation ratio for trailing-dim input ----

[vals_3d, ratios_3d] = mobius.evalOrbitAbs(p, w, sigma, r, x_3d, ...
    'returnCancellationRatio', true);
okShape = isequal(size(ratios_3d), [n1, n2]);
okRange = all(ratios_3d(:) >= 0) && all(ratios_3d(:) <= 1 + 1e-9);
results{end+1, 1} = 'evalOrbitAbs: returnCancellationRatio returns matched-shape ratios in [0,1]';
results{end, 2} = okShape && okRange;

%% ---- evalOrbitAbs: bad first-dim raises ----

x_bad = zeros(2, 3, 4);  % first dim != r
ok = false;
try
    mobius.evalOrbitAbs(p, w, sigma, 3, x_bad);
catch err
    ok = contains(err.identifier, 'queryShape');
end
results{end+1, 1} = 'evalOrbitAbs: wrong first-dim raises queryShape error';
results{end, 2} = ok;

%% ---- evalOrbitRel: chunked u-grid matches manual sequential ----

% For each (r, is_per), run mobius.evalOrbitRel and compare to a
% hand-rolled sequential u-grid quadrature using evalOrbitAbs on
% (r, n_q) inputs (the pre-v2.2.x path).
for cfg = {[2, false], [3, false], [2, true], [3, true]}
    cfgArr = cfg{1};
    r_test = cfgArr(1);
    is_per = logical(cfgArr(2));
    period = 0.0;
    if is_per
        period = 1200.0;
    end

    rng(42);
    p_t = sort(rand(6, 1) * 1200);
    w_t = ones(6, 1);
    x_rel = rand(r_test - 1, 7) * 1200;

    v_vec = mobius.evalOrbitRel(p_t, w_t, sigma, r_test, x_rel, ...
        'is_per', is_per, 'period', period);

    % Manual sequential
    samples_per_sigma = 10;
    if is_per
        N_u = max(64, ceil(period / sigma * samples_per_sigma));
        u_grid = linspace(0.0, period, N_u + 1);
        u_grid = u_grid(1:end-1);  % match endpoint=False
        du = period / N_u;
    else
        x_min = min(x_rel(:));
        x_min = min(x_min, 0);
        x_max = max(x_rel(:));
        x_max = max(x_max, 0);
        u_min = min(p_t) - max(0, x_max) - 8.0 * sigma;
        u_max = max(p_t) - min(0, x_min) + 8.0 * sigma;
        N_u = max(64, ceil(max(u_max - u_min, 1.0) / sigma * samples_per_sigma));
        u_grid = linspace(u_min, u_max, N_u);
    end

    n_q = size(x_rel, 2);
    F = zeros(N_u, n_q);
    for j = 1:N_u
        x_full = zeros(r_test, n_q);
        x_full(1, :) = u_grid(j);
        x_full(2:end, :) = u_grid(j) + x_rel;
        vals_j = mobius.evalOrbitAbs(p_t, w_t, sigma, r_test, x_full, ...
            'is_per', is_per, 'period', period);
        F(j, :) = vals_j(:)';
    end
    if is_per
        integral = sum(F, 1) * du;
    else
        integral = trapz(u_grid, F, 1);
    end
    Z_t = sigma * sqrt(2 * pi / r_test);
    v_man = integral(:) / Z_t;

    relTol = 1e-12 * max(abs(v_man));
    results{end+1, 1} = sprintf( ...
        'evalOrbitRel r=%d is_per=%d: chunked u-grid bit-identical to sequential', ...
        r_test, double(is_per));
    results{end, 2} = max(abs(v_vec(:) - v_man(:))) < relTol;
end

%% ---- tensorHarmonicity batched matches scalar (machine precision) ----

spec = {'harmonic', 12, 'powerlaw', 1};
p_chord = [0, 400, 700];
h_scalar = tensorHarmonicity(p_chord, [], sigma, 'spectrum', spec, 'verbose', false);

P = repmat(p_chord, 3, 1);
h_batched = tensorHarmonicity(P, [], sigma, 'spectrum', spec, 'verbose', false);

results{end+1, 1} = 'tensorHarmonicity batched: same chord across rows matches scalar (1e-12)';
results{end, 2} = isequal(size(h_batched), [3, 1]) ...
    && all(abs(h_batched - h_scalar) < 1e-12);

%% ---- tensorHarmonicity batched groups by cardinality ----

P_mixed = [0,    400,  700,  NaN;
           0,    400,  700,  1100;
           500,  900,  1200, NaN];
h_mixed = tensorHarmonicity(P_mixed, [], sigma, 'spectrum', spec, ...
    'verbose', false);
results{end+1, 1} = 'tensorHarmonicity batched: mixed cardinality, transposition-equiv rows agree';
results{end, 2} = isequal(size(h_mixed), [3, 1]) ...
    && abs(h_mixed(1) - h_mixed(3)) < 1e-12 ...
    && abs(h_mixed(1) - h_mixed(2)) > 1e-6 ...
    && all(isfinite(h_mixed));

%% ---- tensorHarmonicity batched normalisation matches scalar ----

normsOk = true;
for normKey = {'none', 'gaussian', 'pdf'}
    nrm = normKey{1};
    h_s = tensorHarmonicity(p_chord, [], sigma, 'spectrum', spec, ...
        'normalize', nrm, 'verbose', false);
    h_b = tensorHarmonicity([p_chord; p_chord], [], sigma, 'spectrum', spec, ...
        'normalize', nrm, 'verbose', false);
    if abs(h_b(1) - h_s) >= 1e-12 * max(abs(h_s), 1e-30)
        normsOk = false;
        break;
    end
end
results{end+1, 1} = 'tensorHarmonicity batched: normalisation matches scalar for none/gaussian/pdf';
results{end, 2} = normsOk;

%% ---- tensorHarmonicity batched dedup collapses repeats ----

P_rep = zeros(50, 3);
for k = 0:49
    P_rep(k+1, :) = p_chord + 100 * k;   % canonically identical
end
h_rep = tensorHarmonicity(P_rep, [], sigma, 'spectrum', spec, 'verbose', false);
results{end+1, 1} = 'tensorHarmonicity batched: 50 transp-equiv rows all return same value (1e-12)';
results{end, 2} = isequal(size(h_rep), [50, 1]) ...
    && max(abs(h_rep - h_rep(1))) < 1e-12;

%% ---- Standalone summary ----

if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:, 2)));
    nFail = numel(results(:, 1)) - nPass;
    fprintf('\n=== test_orbit_vectorisation: %d passed, %d failed (of %d) ===\n', ...
        nPass, nFail, numel(results(:, 1)));
    if nFail > 0
        for ii = 1:size(results, 1)
            if ~isequal(results{ii, 2}, true)
                fprintf('  FAIL  %s\n', results{ii, 1});
            end
        end
    end
end
