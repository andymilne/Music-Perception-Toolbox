function bench_orbit_xlang
%BENCH_ORBIT_XLANG  Cross-language speed benchmark for orbit IP/eval.
%
%   Targeted spot-checks (not a sweep) at the same (r, K, P)
%   configurations as the Python companion
%   ``python/tests/bench_orbit_xlang.py``. Prints CSV-format lines
%   matching the Python output so the two can be concatenated and
%   read into a comparison table.
%
%   Compare median wall times to gauge whether MATLAB is within ~2x
%   of Python (the gating threshold for shipping the v2.2 MATLAB port
%   as-is vs investing in precomputed contraction paths or a sharper
%   greedy heuristic in mobius.contract).
%
%   Run from the matlab/ directory after addpath('+mobius'):
%     >> bench_orbit_xlang
%   then redirect or copy-paste the output and concatenate with the
%   Python output for analysis.

    % --- Configuration (must match Python script) ---
    configs = struct( ...
        'r', {2, 3, 3, 4, 5}, ...
        'K', {8, 8, 12, 8, 6}, ...
        'P', {50, 50, 50, 20, 10});

    N_REPS    = 7;
    N_WARMUP  = 2;

    fprintf('language,bench,r,K,P,t_median_ms\n');
    for ci = 1:numel(configs)
        r = configs(ci).r;
        K = configs(ci).K;
        P = configs(ci).P;

        t_single  = benchOrbitSingle(r, K, N_REPS, N_WARMUP) * 1000;
        t_batched = benchOrbitBatched(r, K, P, N_REPS, N_WARMUP) * 1000;
        t_cossim  = benchCosSimSingleMultisetOrbit(r, K, N_REPS, N_WARMUP) * 1000;

        fprintf('matlab,inner_product_orbit,%d,%d,1,%.4f\n', ...
                r, K, t_single);
        fprintf('matlab,inner_product_orbit_pw_batched,%d,%d,%d,%.4f\n', ...
                r, K, P, t_batched);
        fprintf('matlab,cos_sim_exp_tens_sa_orbit,%d,%d,1,%.4f\n', ...
                r, K, t_cossim);
    end
end


% =========================================================================
%  Helpers
% =========================================================================

function [K, w_a, w_b] = makeKernel(K_x, K_y, seed)
    rng(seed, 'twister');
    p_x = sort(2000 * rand(K_x, 1));
    p_y = sort(2000 * rand(K_y, 1));
    sigma = 30;
    K = exp(-(p_x - p_y.').^2 / (4 * sigma^2));
    w_a = 0.5 + rand(K_x, 1);
    w_b = 0.5 + rand(K_y, 1);
end


function t = timed(fn, nReps, nWarmup)
    for w = 1:nWarmup
        fn();
    end
    times = zeros(nReps, 1);
    for i = 1:nReps
        t0 = tic;
        fn();
        times(i) = toc(t0);
    end
    t = median(times);
end


% =========================================================================
%  Benchmarks
% =========================================================================

function t = benchOrbitSingle(r, K, nReps, nWarmup)
    [K_mat, w_a, w_b] = makeKernel(K, K, r * 7);
    t = timed(@() mobius.innerProductOrbit(K_mat, w_a, w_b, r, ...
                                              'prefactor', 1.0), ...
              nReps, nWarmup);
end


function t = benchOrbitBatched(r, K, P, nReps, nWarmup)
    K_pairs = zeros(P, K, K);
    w_A = zeros(P, K);
    w_B = zeros(P, K);
    for i = 1:P
        [K_mat, w_a, w_b] = makeKernel(K, K, r * 11 + i);
        K_pairs(i, :, :) = K_mat;
        w_A(i, :) = w_a;
        w_B(i, :) = w_b;
    end
    t = timed(@() mobius.innerProductOrbitPwBatched( ...
                K_pairs, w_A, w_B, r, 'prefactor', 1.0), ...
              nReps, nWarmup);
end


function t = benchCosSimSingleMultisetOrbit(r, K, nReps, nWarmup)
    rng(r * 13 + K, 'twister');
    p1 = sort(2000 * rand(K, 1));
    p2 = sort(2000 * rand(K, 1));
    w  = ones(K, 1);
    t = timed(@() cosSimExpTens(p1, w, p2, w, 30, r, false, false, 0, ...
                                  'method', 'mobius', 'verbose', false), ...
              nReps, nWarmup);
end
