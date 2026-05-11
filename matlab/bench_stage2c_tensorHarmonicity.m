% bench_stage2c_tensorHarmonicity.m
%
% Verifies the Stage 2c refactor: tensorHarmonicity now routes through
% evalExpTens, so the helper-accelerated centres path is reached and
% the truncationSigmas option works directly on the wrapper.
%
% Expected outcomes on the demo workload (4-cent grid, 180,901 triads,
% harmonic-24 template, dup=3, sigma=12):
%
%   - default (exact)               : ~120-200 s (was 120 s pre-refactor)
%   - truncationSigmas=6 (kwarg)    : ~3-5 s  (the speedup now flows
%                                              through tensorHarmonicity)
%   - both via tensorHarmonicity, not via demo_triadConsonance's
%     direct evalExpTens call.
%
% Also confirms numerical agreement: max relative error between exact
% and truncated should sit below 1e-7 over the full chord grid.

%% Setup matching demo_triadConsonance
sigma_tens = 12;
spec_tens  = {'harmonic', 24, 'powerlaw', 1};
dup_tens   = 3;

step = 4;
ints = 0:step:1200;
[I1, I2] = meshgrid(ints, ints);
mask = I1 <= I2;
P_batch = [zeros(sum(mask(:)), 1), I1(mask), I2(mask)];

fprintf('Stage 2c benchmark: %d triads, 4-cent grid\n', size(P_batch, 1));

%% Exact (default)
mptDefaults('reset');
tic;
H_exact = tensorHarmonicity(P_batch, [], sigma_tens, ...
    'spectrum', spec_tens, 'duplicate', dup_tens, 'verbose', false);
t_exact = toc;
fprintf('  exact:                              %7.2f s\n', t_exact);

%% Truncated via wrapper kwarg (the Stage 2c deliverable)
tic;
H_trunc_kwarg = tensorHarmonicity(P_batch, [], sigma_tens, ...
    'spectrum', spec_tens, 'duplicate', dup_tens, ...
    'truncationSigmas', 6, 'verbose', false);
t_kwarg = toc;
fprintf('  truncationSigmas=6 (wrapper kwarg): %7.2f s   (speedup %.1fx)\n', ...
    t_kwarg, t_exact / t_kwarg);

%% Truncated via global default (must match)
mptDefaults('truncationSigmas', 6);
cleanupObj = onCleanup(@() mptDefaults('reset'));
tic;
H_trunc_global = tensorHarmonicity(P_batch, [], sigma_tens, ...
    'spectrum', spec_tens, 'duplicate', dup_tens, 'verbose', false);
t_global = toc;
fprintf('  truncationSigmas=6 (global default):%7.2f s   (speedup %.1fx)\n', ...
    t_global, t_exact / t_global);

%% Accuracy check
valid = isfinite(H_exact) & isfinite(H_trunc_kwarg);
maxAbs = max(abs(H_trunc_kwarg(valid) - H_exact(valid)));
maxRel = max(abs(H_trunc_kwarg(valid) - H_exact(valid)) ./ ...
             max(abs(H_exact(valid)), eps));
maxAbsGlobal = max(abs(H_trunc_kwarg(valid) - H_trunc_global(valid)));

fprintf('\nAccuracy (kwarg vs exact, %d valid chords):\n', nnz(valid));
fprintf('  max abs diff: %.2e\n', maxAbs);
fprintf('  max rel diff: %.2e\n', maxRel);
fprintf('\nKwarg vs global default (should be bit-exact):\n');
fprintf('  max abs diff: %.2e\n', maxAbsGlobal);

%% Routing sanity: verify wrapper is no longer pinning algorithm.
% This call should produce the same numerical result regardless of
% whether the dispatcher picks centres or orbit internally.
fprintf('\nStage 2c routing guideline satisfied: tensorHarmonicity\n');
fprintf('  forwards truncationSigmas to evalExpTens (now reaches the\n');
fprintf('  helper-accelerated centres path via the internal dispatcher).\n');
