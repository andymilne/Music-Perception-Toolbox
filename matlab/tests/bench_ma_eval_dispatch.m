%% bench_ma_eval_dispatch.m
%  Calibration harness for the multi-attribute EVAL dispatcher
%  (internal.selectMaEval routing evalExpTens between the factored
%  Möbius evaluator and the joint-centres accumulator).
%
%  Purpose: MEASURE the centres-vs-Möbius crossover on THIS machine, so
%  the dominance constant MA_CENTRES_DOMINANCE in internal.selectMaEval
%  can be calibrated from MATLAB timings rather than inherited from
%  Python (whose 2.0 reflects NumPy's constant factors: JIT, column-major
%  layout, BLAS, and copy-on-write all differ). This is the eval-side
%  twin of bench_ma_dispatch.m (which calibrates the cosine IP dispatch).
%
%  For each shape cell it times method='centres' and method='mobius'
%  through the public evalExpTens, reports which is faster and what the
%  cost model predicted, and prints the op-count ratio orbit/joint at the
%  empirical crossover. Read the last column: wherever 'faster' and
%  'predict' disagree OUTSIDE a noise band, the constant needs revisiting.
%
%  Run from anywhere with the toolbox on the path.

fprintf('\n=== MA eval dispatch calibration (centres vs factored Möbius) ===\n\n');

nQ    = 200;
nReps = 5;
rng(0, 'twister');

% Shape grid spanning the crossover: single- and multi-attribute,
% absolute and relative, small-to-moderate K. Large-K cells omit the
% centres timing (it is infeasible there -- exactly why the model must
% pick Möbius; feasibility is covered by the cost-model test).
%   {label, sigma, r, isRel, isPer, period, K}
grid = {
  {'A1 r2 K6',   30,      2,     false,         false,         0,       6}
  {'A1 r2 K10',  30,      2,     false,         false,         0,       10}
  {'A1 r2 K20',  30,      2,     false,         false,         0,       20}
  {'A1 r3 K6',   30,      3,     false,         false,         0,       6}
  {'A1 r3 K8',   30,      3,     false,         false,         0,       8}
  {'A1 r3 K12',  30,      3,     false,         false,         0,       12}
  {'A2 r2 K5',   [30 25], [2 2], [false false], [false false], [0 0],   5}
  {'A2 r2 K8',   [30 25], [2 2], [false false], [false false], [0 0],   8}
  {'A2 r2 K12',  [30 25], [2 2], [false false], [false false], [0 0],   12}
  {'A2 r3 K6',   [30 25], [3 3], [false false], [false false], [0 0],   6}
  {'A2 r3 K10',  [30 25], [3 3], [false false], [false false], [0 0],   10}
  {'A1 rel r2 K8',  30,   2,     true,          false,         0,       8}
  {'A1 rel r3 K8',  30,   3,     true,          false,         0,       8}
};

fprintf('%-14s %8s %8s %9s %9s %8s %8s %6s\n', ...
    'cell', 'orbit', 'joint', 'cen_ms', 'mob_ms', 'faster', 'predict', 'o/j');
fprintf('%s\n', repmat('-', 1, 82));

% Global warm-up before timing. MATLAB pays one-time costs on the first
% use of each path within a run --- function compilation, +mobius
% package resolution, and (for the Möbius path) loading the orbit /
% contraction-recipe tables from the shipped or disk cache into the
% in-memory cache. The exact contribution of each is not pinned down
% (some survive `clear all` via the on-disk table cache and the OS file
% cache, so it is not purely JIT), but empirically the first timed cell
% over-reports Möbius cost until these are paid. Exercise both paths
% once here, off the clock, so the grid times steady-state code.
% Recalibration should keep this; without it the earliest cells mislead.
warmDens = buildExpTens({100*rand(8,1); 100*rand(8,1)}, {[]; []}, ...
    [30 25], [2 2], [false false], [false false], [0 0], 'verbose', false);
warmX = 100 * rand(warmDens.dim, 50);
evalExpTens(warmDens, warmX, 'method', 'centres', 'verbose', false);
evalExpTens(warmDens, warmX, 'method', 'mobius', 'verbose', false);
warmRel = buildExpTens(100*rand(8,1), ones(8,1), 30, 2, true, false, 0, ...
    'verbose', false);
evalExpTens(warmRel, 100*rand(1, 50), 'method', 'mobius', 'verbose', false);

BELL = [1 2 5 15 52 203 877 4140 21147 115975];
mismatch = 0; tested = 0;

for gi = 1:numel(grid)
    c = grid{gi};
    [label, sig, rv, rel, per, P, K] = c{:};
    A = numel(sig);
    pas = cell(A, 1);
    for a = 1:A, pas{a} = 100 * rand(K, 1); end
    wpas = repmat({[]}, A, 1);
    dens = buildExpTens(pas, wpas, sig, rv, rel, per, P, 'verbose', false);
    xq = 100 * rand(dens.dim, nQ);

    [pred, ~] = internal.selectMaEval(dens, nQ, false);

    % op-counts
    joint = 1; orbit = 0;
    for a = 1:A
        joint = joint * factorial(rv(a)) * nchoosek(K, rv(a));
        orbit = orbit + BELL(rv(a)) * rv(a) * K;
    end

    % time centres (guarded: skip if it would be hopeless)
    tCen = timeMethod(@() evalExpTens(dens, xq, 'method', 'centres', ...
        'verbose', false), nReps);
    tMob = timeMethod(@() evalExpTens(dens, xq, 'method', 'mobius', ...
        'verbose', false), nReps);

    if tCen < tMob, faster = 'centres'; else, faster = 'mobius'; end
    within = abs(tCen - tMob) / max(min(tCen, tMob), 1e-9) < 0.25;
    ok = strcmp(pred, faster) || within;
    tested = tested + 1;
    if ~ok, mismatch = mismatch + 1; end

    fprintf('%-14s %8d %8d %9.2f %9.2f %8s %8s %6.2f  %s\n', ...
        label, orbit, joint, tCen * 1e3, tMob * 1e3, faster, pred, ...
        orbit / joint, tern(ok, '', '<-- MISPICK'));
end

fprintf('%s\n', repmat('-', 1, 82));
fprintf('mispicks outside 25%% noise band: %d / %d\n', mismatch, tested);
fprintf(['\nThis is a quick op-count sanity view; the authoritative ' ...
         'calibration harness is\nbench_ma_eval_calibration.m, whose CSV ' ...
         'fits the MA_COST_* constants in\ninternal/selectMaEval.m. The ' ...
         'o/j column shows the op-count ratio at each\ncell; the ' ...
         'empirical crossover is where ''faster'' flips.\n\n']);

% ---- helpers ----
function t = timeMethod(fn, nReps)
    fn();                       % warm up
    ts = zeros(1, nReps);
    for i = 1:nReps
        tic; fn(); ts(i) = toc;
    end
    t = min(ts);
end

function s = tern(cond, a, b)
    if cond, s = a; else, s = b; end
end
