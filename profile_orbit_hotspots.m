function profile_orbit_hotspots
%PROFILE_ORBIT_HOTSPOTS  Localise MATLAB orbit slowness with the profiler.
%
%   Runs mobius.innerProductOrbit in a tight loop at three (r, K)
%   configurations and opens the profile viewer. The goal is to confirm
%   (or refute) the hypothesis that the bulk of the time is in
%   mobius.contract's greedy pair-picker and per-merge dispatch
%   (pickBestPair, mergePair, intersect/setdiff/unique) rather than in
%   the actual pagemtimes / arithmetic.
%
%   Typical workflow:
%     >> cd matlab
%     >> profile_orbit_hotspots
%     >> [profiler window opens]
%     >> Sort by Total Time. The top ~10 entries should pinpoint the
%        bottleneck. Send a screenshot or paste the function names &
%        times.
%
%   For r=5 the loop is shorter (~10 iterations) because the call is
%   already slow enough to give a good sample.

    addpath(fileparts(mfilename('fullpath')));   % +mobius access

    configs = struct( ...
        'r', {3, 4, 5}, ...
        'K', {8, 8, 6}, ...
        'reps', {200, 50, 10});

    fprintf('=== Building inputs ===\n');
    inputs = cell(numel(configs), 1);
    for ci = 1:numel(configs)
        cfg = configs(ci);
        rng(cfg.r * 7);
        p_x = sort(2000 * rand(cfg.K, 1));
        p_y = sort(2000 * rand(cfg.K, 1));
        sigma = 30;
        K_mat = exp(-(p_x - p_y.').^2 / (4 * sigma^2));
        inputs{ci} = struct('K', K_mat, ...
                             'w_a', ones(cfg.K, 1), ...
                             'w_b', ones(cfg.K, 1));
    end

    fprintf('=== Profiling ===\n');
    profile clear;
    profile on;

    % Warm up to avoid JIT cost in the first iterations dominating.
    for ci = 1:numel(configs)
        mobius.innerProductOrbit(inputs{ci}.K, inputs{ci}.w_a, ...
                                  inputs{ci}.w_b, configs(ci).r);
    end

    % Hot loop, weighted toward the cheapest config so wall time is reasonable.
    for ci = 1:numel(configs)
        cfg = configs(ci);
        for it = 1:cfg.reps
            mobius.innerProductOrbit(inputs{ci}.K, inputs{ci}.w_a, ...
                                      inputs{ci}.w_b, cfg.r);
        end
    end

    profile off;
    fprintf('Opening profile viewer...\n');
    profile viewer;
end
