%% demo_dispatchAndKernelControls.m
%
%  A tour of v2.2's performance features:
%
%    1. Per-call method dispatch -- Bulger's method vs the Möbius
%       method are picked automatically by the cost-model dispatcher.
%       Forcing either by hand demonstrates the speed-up that 'auto'
%       gets you transparently.
%    2. Kernel truncation -- 'truncationSigmas' skips Gaussian
%       contributions beyond k standard deviations from a centre.
%    3. Single-precision kernel -- 'kernelPrecision','single' casts
%       the kernel matrix to float32 for a ~2x speedup at ~7 sig fig
%       precision.
%    4. Toolbox-wide defaults -- 'mptDefaults' lets all of the above
%       be flipped globally so user code does not need per-call
%       name-value pairs.
%    5. Renyi-2 entropy -- 'method','renyi2' on entropyExpTens for a
%       closed-form alternative to the numerical Shannon path.
%
%  All controls default to v2.1-equivalent behaviour. Opting into
%  them is purely additive.

%% User-adjustable parameters
N_EVENTS = 20;          % source events per density
R        = 3;           % tensor order
SIGMA    = 30.0;        % Gaussian uncertainty (cents)
N_REPEATS = 3;          % repetitions per timing measurement
RNG_SEED  = 0;
PERIOD    = 1200.0;     % nominal range (unused for is_per=false)

%% Setup
rng(RNG_SEED);

p_x = sort(1200 * rand(1, N_EVENTS));
p_y = sort(1200 * rand(1, N_EVENTS));
w_x = 0.5 + rand(1, N_EVENTS);
w_y = 0.5 + rand(1, N_EVENTS);

dens_x = buildExpTens(p_x, w_x, SIGMA, R, false, false, PERIOD);
dens_y = buildExpTens(p_y, w_y, SIGMA, R, false, false, PERIOD);

% A larger source set for the centres-path sections (2-4). MATLAB's
% BLAS is so fast at modest scales that the kernel matmul is in the
% tens of ms range, where the fixed-cost overhead of truncation's
% spatial index and the float32 cast can be comparable to the
% variable-cost savings they buy. N=50 (with 1000 query points
% below) pushes the kernel matmul into the hundreds-of-ms range, so
% the savings dominate and the features show clearly. Section 1
% stays at N=20 because that's already enough to make Bulger's
% method look pathological against the Möbius method.
N_BIG = 50;
p_big = sort(1200 * rand(1, N_BIG));
w_big = 0.5 + rand(1, N_BIG);
dens_big = buildExpTens(p_big, w_big, SIGMA, R, false, false, PERIOD);

timeCall = @(fn) localTimeCall(fn, N_REPEATS);

% Warm-up: flush MATLAB's first-call function resolution and the
% orbit-table disk load out of the timed section. Without this the
% first measurement below would carry ~5-20 ms of one-time cost.
cosSimExpTens(dens_x, dens_y, 'method', 'mobius', 'verbose', false);
cosSimExpTens(dens_x, dens_y, 'method', 'bulger', 'verbose', false);

%% 1. Method dispatch -- Bulger's method vs the Möbius method
fprintf('\n=== 1. Method dispatch (N=%d, r=%d, sigma=%g, abs nonper) ===\n\n', ...
        N_EVENTS, R, SIGMA);

[t_auto,   c_auto]   = timeCall(@() cosSimExpTens(dens_x, dens_y, 'verbose', false));
[t_bulger, c_bulger] = timeCall(@() cosSimExpTens(dens_x, dens_y, 'method', 'bulger', 'verbose', false));
[t_mobius, c_mobius] = timeCall(@() cosSimExpTens(dens_x, dens_y, 'method', 'mobius', 'verbose', false));

fprintf('  method=auto    : %6.1f ms   cosine = %.10f\n', 1000*t_auto,   c_auto);
fprintf('  method=bulger  : %6.1f ms   cosine = %.10f\n', 1000*t_bulger, c_bulger);
fprintf('  method=mobius  : %6.1f ms   cosine = %.10f\n', 1000*t_mobius, c_mobius);
fprintf('  (bulger and mobius agree to %.2e)\n', abs(c_bulger - c_mobius));

%% 2. Kernel truncation
%
%  truncationSigmas affects the centres-array path used by
%  evalExpTens and (when the dispatcher selects it) the centres path
%  of cosSimExpTens. The Möbius method evaluates the same density
%  analytically without a centres matrix, so the truncation control
%  does not apply to it.

fprintf('\n=== 2. Kernel truncation (N=%d, eval at 1000 query points) ===\n\n', N_BIG);

queries = sort(1200 * rand(R, 1000));

[t_no_trunc, v_no_trunc] = timeCall(@() evalExpTens(dens_big, queries, ...
    'method', 'centres', 'truncationSigmas', Inf, 'verbose', false));
[t_k6, v_k6] = timeCall(@() evalExpTens(dens_big, queries, ...
    'method', 'centres', 'truncationSigmas', 6, 'verbose', false));
[t_k4, v_k4] = timeCall(@() evalExpTens(dens_big, queries, ...
    'method', 'centres', 'truncationSigmas', 4, 'verbose', false));

err_k6 = max(abs(v_k6(:) - v_no_trunc(:))) / (max(abs(v_no_trunc(:))) + 1e-30);
err_k4 = max(abs(v_k4(:) - v_no_trunc(:))) / (max(abs(v_no_trunc(:))) + 1e-30);

fprintf('  truncationSigmas=Inf  : %6.1f ms  (reference)\n', 1000*t_no_trunc);
fprintf('  truncationSigmas=6    : %6.1f ms  peak-normalised err = %.2e\n', 1000*t_k6, err_k6);
fprintf('  truncationSigmas=4    : %6.1f ms  peak-normalised err = %.2e\n', 1000*t_k4, err_k4);
fprintf('  (Truncating at k sigmas drops kernel contributions below exp(-k^2/2).\n');
fprintf('   k=6 ~ exp(-18) ~ 1.5e-8; k=4 ~ exp(-8) ~ 3e-4.)\n');

%% 3. Single-precision kernel
fprintf('\n=== 3. kernelPrecision ===\n\n');

[t_double, v_double] = timeCall(@() evalExpTens(dens_big, queries, ...
    'method', 'centres', 'kernelPrecision', 'double', 'verbose', false));
[t_single, v_single] = timeCall(@() evalExpTens(dens_big, queries, ...
    'method', 'centres', 'kernelPrecision', 'single', 'verbose', false));

err_single = max(abs(v_single(:) - v_double(:))) / (max(abs(v_double(:))) + 1e-30);

fprintf('  kernelPrecision=double : %6.1f ms  (reference)\n', 1000*t_double);
fprintf('  kernelPrecision=single : %6.1f ms  peak-normalised err = %.2e\n', 1000*t_single, err_single);
fprintf('  (Speedup is workload- and platform-dependent: on memory-bandwidth-\n');
fprintf('   bound problems the gain is small even at scale. Python typically\n');
fprintf('   sees ~2x at this size; MATLAB with MKL sees less. Precision\n');
fprintf('   retained: ~7 sig figs vs ~15.)\n');

%% 4. Toolbox-wide defaults
fprintf('\n=== 4. Toolbox-wide defaults ===\n\n');

fprintf('  Current defaults:\n');
disp(mptDefaults());
fprintf('  Setting global: truncationSigmas=6, kernelPrecision=single\n');
prev = mptDefaults('truncationSigmas', 6, 'kernelPrecision', 'single');
fprintf('  New defaults:\n');
disp(mptDefaults());

[t_global, ~] = timeCall(@() evalExpTens(dens_big, queries, ...
    'method', 'centres', 'verbose', false));
fprintf('  eval with global defaults active : %6.1f ms\n', 1000*t_global);

% Per-call kwargs always override the global defaults:
[t_override, ~] = timeCall(@() evalExpTens(dens_big, queries, ...
    'method', 'centres', 'truncationSigmas', Inf, ...
    'kernelPrecision', 'double', 'verbose', false));
fprintf('  per-call override back to defaults: %6.1f ms\n', 1000*t_override);

mptDefaults(prev);   % restore via the save/restore idiom
fprintf('  Restored; defaults now:\n');
disp(mptDefaults());

%% 5. Renyi-2 differential entropy
fprintf('\n=== 5. Renyi-2 differential entropy ===\n\n');

[t_shannon, h_shannon] = timeCall(@() entropyExpTens(dens_x, ...
    'method', 'shannon', 'xMin', 0, 'xMax', 1200, ...
    'nPointsPerDim', 100, 'normalize', false, 'verbose', false));
[t_renyi2, h_renyi2] = timeCall(@() entropyExpTens(dens_x, ...
    'method', 'renyi2', 'normalize', false, 'verbose', false));

fprintf('  method=shannon (numerical grid)  : %7.1f ms   H  = %.4f\n', 1000*t_shannon, h_shannon);
fprintf('  method=renyi2  (closed-form)     : %7.1f ms   H2 = %.4f\n', 1000*t_renyi2, h_renyi2);
fprintf('  (Shannon here is a grid-discretised entropy at 100 cells/dim;\n');
fprintf('   Renyi-2 is a continuous differential entropy. Numerical values\n');
fprintf('   are not directly comparable, but ranking behaviour is similar\n');
fprintf('   and Renyi-2 is far cheaper at high r where the grid would OOM.)\n');

fprintf('\n=== Done. See USER_GUIDE.md sec.4 for the full method-selection API. ===\n');


%% --- helpers ---
function [tMedian, result] = localTimeCall(fn, repeats)
    ts = zeros(1, repeats);
    result = [];
    for k = 1:repeats
        t0 = tic;
        result = fn();
        ts(k) = toc(t0);
    end
    tMedian = median(ts);
end
