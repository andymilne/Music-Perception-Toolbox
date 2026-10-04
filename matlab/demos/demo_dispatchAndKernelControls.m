%% demo_dispatchAndKernelControls.m
%
%  A tour of the toolbox's performance controls:
%
%    1. Method dispatch -- for flat attributes, Bulger's method and the
%       Möbius method (and, for an ordered attribute, the materialized
%       tuple centres) are chosen automatically by a cost model. Forcing
%       each by hand shows the speed-up that 'auto' gets you, and
%       explainDispatch shows why it chose.
%    2. Nested attributes -- a bound, spectrally enriched attribute is
%       contracted level by level instead of enumerating its nested
%       tuples.
%    3. Sweeps -- the similarity at many translations of a query is
%       computed in one pass by sweepSimMaet (mixture, orbit, or
%       contraction route), not one comparison per offset;
%       sweptSimilarity takes the same routes automatically.
%    4. Kernel truncation -- 'truncationSigmas' skips Gaussian
%       contributions beyond k standard deviations from a centre.
%    5. Single-precision kernel -- 'kernelPrecision','single' casts the
%       kernel matrix to float32, at ~7 significant figures.
%    6. Toolbox-wide defaults -- mptDefaults sets any of the above, and
%       the other defaults, for every later call.
%    7. Entropy estimators -- 'shannon' (a grid), 'differential'
%       (adaptive, continuous), and 'renyi2' (closed form), and where
%       each applies.
%
%  The dispatcher and a kernel truncation at 6 sigma are on by default
%  ('truncationSigmas', 6 drops contributions below exp(-18), about
%  1.5e-8 of a kernel's peak); 'kernelPrecision' defaults to double.
%  Every control can be set per call or toolbox-wide.
%
%  The Python mirror is demo_dispatch_and_kernel_controls.py.

%% User-adjustable parameters
N_EVENTS = 20;          % source events per density
R        = 3;           % tuple size
SIGMA    = 30.0;        % Gaussian uncertainty (cents)
N_REPEATS = 3;          % repetitions per timing measurement
RNG_SEED  = 0;
PERIOD    = 1200.0;     % nominal range (unused for per = false)

%% Setup
rng(RNG_SEED);

p_x = sort(1200 * rand(1, N_EVENTS));
p_y = sort(1200 * rand(1, N_EVENTS));
w_x = 0.5 + rand(1, N_EVENTS);
w_y = 0.5 + rand(1, N_EVENTS);

dens_x = buildMaet(p_x, w_x, SIGMA, R, false, false, PERIOD, 'verbose', false);
dens_y = buildMaet(p_y, w_y, SIGMA, R, false, false, PERIOD, 'verbose', false);

% A larger source set for the centres-path sections (4-6). MATLAB's
% BLAS is so fast at modest scales that the kernel matmul is in the
% tens of ms range, where the fixed-cost overhead of truncation's
% spatial index and the float32 cast can be comparable to the
% variable-cost savings they produce. N=50 (with 1000 query points
% below) pushes the kernel matmul into the hundreds-of-ms range, so
% the savings dominate and the features show clearly. Section 1
% stays at N=20 because that is already enough to make Bulger's
% method look pathological against the Möbius method.
N_BIG = 50;
p_big = sort(1200 * rand(1, N_BIG));
w_big = 0.5 + rand(1, N_BIG);
dens_big = buildMaet(p_big, w_big, SIGMA, R, false, false, PERIOD, 'verbose', false);

timeCall = @(fn) localTimeCall(fn, N_REPEATS);

% Each top-level call announces the route it chose (showHints, Section
% 6). The timings below repeat every call, so the announcements are
% switched off here and explainDispatch reports the choices instead;
% they are restored at the end.
prevHints = mptDefaults('showHints', false);

% Warm-up: flush MATLAB's first-call function resolution and the
% orbit-table disk load out of the timed section. Without this the
% first measurement below would carry ~5-20 ms of one-time cost.
simMaet(dens_x, dens_y, 'method', 'mobius', 'verbose', false);
simMaet(dens_x, dens_y, 'method', 'bulger', 'verbose', false);

%% 1. Method dispatch -- flat attributes
fprintf('\n=== 1. Method dispatch (N=%d, r=%d, sigma=%g, absolute, non-periodic) ===\n\n', ...
        N_EVENTS, R, SIGMA);

[t_auto,   c_auto]   = timeCall(@() simMaet(dens_x, dens_y, 'verbose', false));
[t_bulger, c_bulger] = timeCall(@() simMaet(dens_x, dens_y, 'method', 'bulger', 'verbose', false));
[t_mobius, c_mobius] = timeCall(@() simMaet(dens_x, dens_y, 'method', 'mobius', 'verbose', false));

fprintf('  method=auto    : %6.1f ms   cosine = %.10f\n', 1000*t_auto,   c_auto);
fprintf('  method=bulger  : %6.1f ms   cosine = %.10f\n', 1000*t_bulger, c_bulger);
fprintf('  method=mobius  : %6.1f ms   cosine = %.10f\n', 1000*t_mobius, c_mobius);
fprintf('  (bulger and mobius agree to %.2e)\n', abs(c_bulger - c_mobius));

% explainDispatch reports the choice 'auto' makes, and why, without
% computing anything.
fprintf('\n  explainDispatch(dens_x, dens_y):\n');
explainDispatch(dens_x, dens_y);

%% 2. Nested attributes
%
%  A nested attribute (here, 3 consecutive notes bound into one
%  super-event, each note enriched with 8 partials) has 8^3 = 512 tuples
%  per super-event. Bulger's method enumerates every pair of them; the
%  contraction reduces the nesting level by level, never forming the
%  tuples. 'auto' prices both and takes the contraction.

fprintf('\n=== 2. Nested attributes (3 notes bound, 8 partials each) ===\n\n');

pmNx = localNestedMelody(16, 8, 3);
pmNy = localNestedMelody(16, 8, 3);
densNx = buildMaet(pmNx, 'verbose', false);
densNy = buildMaet(pmNy, 'verbose', false);
simMaet(densNx, densNy, 'verbose', false);              % warm-up

[t_n_auto,   c_n_auto]   = timeCall(@() simMaet(densNx, densNy, 'verbose', false));
[t_n_bulger, c_n_bulger] = timeCall(@() simMaet(densNx, densNy, 'method', 'bulger', 'verbose', false));
fprintf('  method=auto   : %6.1f ms   cosine = %.10f\n', 1000*t_n_auto,   c_n_auto);
fprintf('  method=bulger : %6.1f ms   cosine = %.10f\n', 1000*t_n_bulger, c_n_bulger);
fprintf('\n  explainDispatch(densNx, densNy):\n');
explainDispatch(densNx, densNy);

%% 3. Sweeps: many translations in one pass
%
%  Comparing a query with a context at M translations one offset at a
%  time costs M builds and M comparisons. sweepSimMaet computes all M at
%  once: the 'mixture' route makes one pass over the tuple pairs and
%  evaluates a Gaussian mixture in the offset; the 'orbit' route
%  evaluates the Möbius inner product at the shifted values; the
%  'contract' route (densities with a nested attribute) carries the
%  offsets through the level-by-level contraction. 'auto' picks. The
%  same routes serve sweptSimilarity wherever the context is the
%  same at every translation (see demo_sweptSimilarity).

fprintf('\n=== 3. Sweeps (41 translations of the query) ===\n\n');

mus = -200:10:200;

[t_loop, s_loop] = timeCall(@() localPerOffsetFlat(dens_x, p_y, w_y, SIGMA, R, PERIOD, mus));
fprintf('  flat, r = %d:\n', R);
fprintf('    one offset at a time  : %7.1f ms\n', 1000*t_loop);
for mth = {'auto', 'mixture', 'orbit'}
    [t_sw, s_sw] = timeCall(@() sweepSimMaet(dens_x, dens_y, mus, ...
        'method', mth{1}, 'verbose', false));
    fprintf('    sweep, %-13s : %7.1f ms   max |diff| = %.1e\n', ...
            sprintf('''%s''', mth{1}), 1000*t_sw, max(abs(s_sw(:) - s_loop(:))));
end
fprintf('    (The mixture route enumerates and stores every pair of tuples,\n');
fprintf('     [C(%d, %d) %d!]^2 of them here, so it is the slowest at\n', N_EVENTS, R, R);
fprintf('     this size; ''auto'' prices the routes and avoids it.)\n');

[t_nloop, s_nloop] = timeCall(@() localPerOffsetNested(densNx, pmNy, mus));
[t_nsw,   s_nsw]   = timeCall(@() sweepSimMaet(densNx, densNy, mus, 'verbose', false));
fprintf('  nested (Section 2):\n');
fprintf('    one offset at a time  : %7.1f ms\n', 1000*t_nloop);
fprintf('    sweep, ''auto''         : %7.1f ms   max |diff| = %.1e   (the contraction route)\n', ...
        1000*t_nsw, max(abs(s_nsw(:) - s_nloop(:))));

%% 4. Kernel truncation
%
%  truncationSigmas applies on every route (Bulger's method, the
%  Möbius method, and the centres path alike); it is timed here on the
%  centres path of evalMaet, where the kernel over tuple centres is the
%  dominant cost. The default is 6 sigma; Inf gives the accuracy floor,
%  the finite width (~7.43 sigma) at which the kernel falls to 1e-12 of
%  its peak, rather than an unbounded sum.

fprintf('\n=== 4. Kernel truncation (N=%d, eval at 1000 query points) ===\n\n', N_BIG);

queries = sort(1200 * rand(R, 1000));

[t_no_trunc, v_no_trunc] = timeCall(@() evalMaet(dens_big, queries, ...
    'method', 'centres', 'truncationSigmas', Inf, 'verbose', false));
[t_k6, v_k6] = timeCall(@() evalMaet(dens_big, queries, ...
    'method', 'centres', 'truncationSigmas', 6, 'verbose', false));
[t_k4, v_k4] = timeCall(@() evalMaet(dens_big, queries, ...
    'method', 'centres', 'truncationSigmas', 4, 'verbose', false));

err_k6 = max(abs(v_k6(:) - v_no_trunc(:))) / (max(abs(v_no_trunc(:))) + 1e-30);
err_k4 = max(abs(v_k4(:) - v_no_trunc(:))) / (max(abs(v_no_trunc(:))) + 1e-30);

fprintf('  truncationSigmas=Inf  : %6.1f ms  (accuracy floor, the reference)\n', 1000*t_no_trunc);
fprintf('  truncationSigmas=6    : %6.1f ms  peak-normalized err = %.2e  (the default)\n', 1000*t_k6, err_k6);
fprintf('  truncationSigmas=4    : %6.1f ms  peak-normalized err = %.2e\n', 1000*t_k4, err_k4);
fprintf('  (Truncating at k sigmas drops kernel contributions below exp(-k^2/2).\n');
fprintf('   k=6 ~ exp(-18) ~ 1.5e-8; k=4 ~ exp(-8) ~ 3e-4.)\n');

%% 5. Single-precision kernel
fprintf('\n=== 5. kernelPrecision ===\n\n');

[t_double, v_double] = timeCall(@() evalMaet(dens_big, queries, ...
    'method', 'centres', 'kernelPrecision', 'double', 'verbose', false));
[t_single, v_single] = timeCall(@() evalMaet(dens_big, queries, ...
    'method', 'centres', 'kernelPrecision', 'single', 'verbose', false));

err_single = max(abs(v_single(:) - v_double(:))) / (max(abs(v_double(:))) + 1e-30);

fprintf('  kernelPrecision=double : %6.1f ms  (reference)\n', 1000*t_double);
fprintf('  kernelPrecision=single : %6.1f ms  peak-normalized err = %.2e\n', 1000*t_single, err_single);
fprintf('  (The speed-up depends on the workload and the platform, and is\n');
fprintf('   negligible where the computation is bound by memory bandwidth.\n');
fprintf('   Precision retained: about 7 significant figures, against about 15.)\n');

%% 6. Toolbox-wide defaults
%
%  What each default controls:
%    truncationSigmas     kernel truncation radius (Section 4).
%    kernelPrecision      kernel-matrix arithmetic (Section 5).
%    showHints            console messages: each call's routing decision
%                         and one-time tips. Off in this demo (switched
%                         off at the top of the script).
%    kernelChunkBytes     the memory budget per chunk of a kernel
%                         computation; 'auto' takes half the available
%                         physical memory, or give a byte count.
%    postHocGuards        checks that inspect a route's result and may
%                         recompute it by another route; switch off only
%                         for calibration runs.
%    orbitCostIntercept, relAttrRoute
%                         calibration levers for the cost model, not
%                         part of the public interface.
fprintf('\n=== 6. Toolbox-wide defaults ===\n\n');

fprintf('  Current defaults:\n');
disp(mptDefaults());
fprintf('  Setting global: truncationSigmas=4, kernelPrecision=single\n');
prev = mptDefaults('truncationSigmas', 4, 'kernelPrecision', 'single');
fprintf('  New defaults:\n');
disp(mptDefaults());

[t_global, ~] = timeCall(@() evalMaet(dens_big, queries, ...
    'method', 'centres', 'verbose', false));
fprintf('  eval with global defaults active : %6.1f ms\n', 1000*t_global);

% Per-call arguments always override the global defaults: here to the
% accuracy-floor, double-precision computation.
[t_override, ~] = timeCall(@() evalMaet(dens_big, queries, ...
    'method', 'centres', 'truncationSigmas', Inf, ...
    'kernelPrecision', 'double', 'verbose', false));
fprintf('  per-call override to Inf/double : %6.1f ms\n', 1000*t_override);

mptDefaults(prev);   % restore via the save/restore idiom
fprintf('  Restored; defaults now:\n');
disp(mptDefaults());

%% 7. Entropy estimators, by dimension
%
%  'shannon' is the discrete entropy of the density's masses on a grid
%  of cells; 'differential' is the continuous differential entropy,
%  refined on an adaptive nested grid until it converges to the accuracy
%  truncationSigmas sets; 'renyi2' is the continuous Rényi-2 entropy in
%  closed form, with no grid. The comparison is across the density's
%  dimension, r = 1, 2, 3 (absolute, so the dimension is r): both grids
%  grow as their resolution to the power of the dimension, while the
%  closed form's cost barely changes. At one dimension all three are
%  cheap and Rényi-2 has no speed advantage; it is at two and three
%  that the difference shows.

fprintf('\n=== 7. Entropy estimators, by dimension ===\n\n');

entropyMaet(dens_x, 'method', 'renyi2', 'verbose', false);   % warm-up
refused = false;
fprintf(['  r   ''shannon'' (100 cells/dim)   ''differential'' (adaptive)' ...
         '          ''renyi2'' (closed form)\n']);
for rE = 1:3
    densE = buildMaet(p_x, w_x, SIGMA, rE, false, false, PERIOD, 'verbose', false);
    % Shannon: 100^r cells; at r = 3 a million, some tens of seconds.
    if rE < 3
        [t_sh, h_sh] = timeCall(@() entropyMaet(densE, 'method', 'shannon', ...
            'xMin', 0, 'xMax', 1200, 'nPointsPerDim', 100, 'verbose', false));
        sh = sprintf('%8.1f ms  H = %6.2f', 1000*t_sh, h_sh);
    else
        sh = '  (10^6 cells: not run)';
    end
    % Differential: from two dimensions the adaptive grid needed at the
    % default accuracy (truncationSigmas = 6) can exceed the feasible
    % size, depending on the data; the call then refuses with guidance,
    % and a coarser accuracy converges. At three it is not attempted.
    if rE < 3
        try
            [t_di, h_di] = timeCall(@() entropyMaet(densE, ...
                'method', 'differential', 'verbose', false));
            di = sprintf('%8.1f ms  h = %6.2f', 1000*t_di, h_di);
        catch err
            if ~strcmp(err.identifier, 'entropyMaet:differentialGridLimit')
                rethrow(err);
            end
            refused = true;
            [t_di, h_di] = timeCall(@() entropyMaet(densE, ...
                'method', 'differential', 'truncationSigmas', 4, 'verbose', false));
            di = sprintf('%8.1f ms  h = %6.2f (sigmas = 4)*', 1000*t_di, h_di);
        end
    else
        di = '  (not attempted)';
    end
    [t_r2, h_r2] = timeCall(@() entropyMaet(densE, 'method', 'renyi2', 'verbose', false));
    fprintf('  %d   %-28s%-34s%8.1f ms  h_2 = %6.2f\n', rE, sh, di, 1000*t_r2, h_r2);
end
fprintf('  (Entropies in bits. Shannon is discrete, so its value depends on\n');
fprintf('   the cell size; the two differential entropies are continuous,\n');
fprintf('   with h_2 <= h.)\n');
if refused
    fprintf('  (* The default accuracy was refused, and the value shown is at\n');
    fprintf('   truncationSigmas = 4.)\n');
end

mptDefaults(prevHints);

fprintf('\n=== Done. See USER_GUIDE.md sec. 5 (Method selection; Kernel-evaluation\n');
fprintf('    controls; Toolbox defaults API; Four-method entropy API and adaptive\n');
fprintf('    differential entropy). ===\n');


%% --- helpers ---
function pm = localNestedMelody(nNotes, nPartials, nBound)
%LOCALNESTEDMELODY  A random melody (whole semitones between MIDI 54 and
%   72, in cents) whose pitches are enriched with harmonic partials and
%   bound in groups of consecutive notes: one nested pitch attribute,
%   each super-event an ordered group of notes, each note a multiset of
%   partials.
    p = round((5400 + 1800 * rand(1, nNotes)) / 100) * 100;
    specs = flatSpecs({p}, 'sigma', 15, 'per', false, 'period', 0);
    pm = packPreMaet({p}, [], specs);
    pm = addSpectra(pm, 'harmonic', nPartials, 'powerlaw', 1, ...
                    'attribute', 1, 'units', 1200);
    pm = bindEvents(pm, nBound);
end


function s = localPerOffsetFlat(densX, pY, wY, sigma, r, period, mus)
%LOCALPEROFFSETFLAT  One build and one comparison per offset.
    s = zeros(1, numel(mus));
    for k = 1:numel(mus)
        densQ = buildMaet(pY + mus(k), wY, sigma, r, false, false, period, ...
                          'verbose', false);
        s(k) = simMaet(densX, densQ, 'verbose', false);
    end
end


function s = localPerOffsetNested(densX, pmY, mus)
%LOCALPEROFFSETNESTED  One translation, build and comparison per offset.
    s = zeros(1, numel(mus));
    for k = 1:numel(mus)
        densQ = buildMaet(translateAttributes(pmY, {mus(k)}), 'verbose', false);
        s(k) = simMaet(densX, densQ, 'verbose', false);
    end
end


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
