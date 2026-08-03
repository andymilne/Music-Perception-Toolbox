%% bench_ma_eval_dispatch.m
%  Audit of the multi-attribute EVAL dispatcher: does the shipped cost
%  model pick the arm that is actually faster? internal.selectMaEval
%  routes evalExpTens between the factored Möbius evaluator and the
%  joint-centres accumulator, and this times both arms on a small shape
%  grid spanning the crossover and reports where the pick disagrees with
%  the measurement.
%
%  It does NOT fit anything. That is tools-side work:
%  bench_ma_eval_calibration.m sweeps the grid the MA_COST_* constants
%  in internal/selectMaEval.m are fitted from. This is the quick check
%  that the fit still holds on this machine, and the op-count columns
%  show where the empirical crossover sits.
%
%  Read the report by magnitude, not by the mispick count alone. A model
%  out by 40x that still orders two routes correctly scores no mispick,
%  while one out by 1.01x near a crossover scores one; the predicted and
%  measured times are printed side by side so the continuous quantity is
%  visible behind the pass/fail column.
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

fprintf('%-14s %8s %8s %9s %9s %9s %9s %8s %8s %6s\n', ...
    'cell', 'orbit', 'joint', 'cen_ms', 'mob_ms', 'pred_cen', ...
    'pred_mob', 'faster', 'predict', 'o/j');
fprintf('%s\n', repmat('-', 1, 102));

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

    [pred, ~, predCen, predMob] = internal.selectMaEval(dens, nQ);

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

    fprintf(['%-14s %8d %8d %9.2f %9.2f %9.2f %9.2f %8s %8s %6.2f' ...
             '  %s\n'], label, orbit, joint, tCen * 1e3, tMob * 1e3, ...
        predCen, predMob, faster, pred, orbit / joint, ...
        tern(ok, '', '<-- MISPICK'));
end

fprintf('%s\n', repmat('-', 1, 102));
fprintf('mispicks outside 25%% noise band: %d / %d\n', mismatch, tested);
fprintf(['\nThis audits the shipped fit; it does not produce one. The ' ...
         'calibration harness is\nbench_ma_eval_calibration.m, whose CSV ' ...
         'fits the MA_COST_* constants in\ninternal/selectMaEval.m. ' ...
         'pred_cen and pred_mob are what those constants\npredict, ' ...
         'beside the measured cen_ms and mob_ms, so a near-tie mispick ' ...
         'can be\ntold from a real one. The o/j column is the op-count ' ...
         'ratio; the empirical\ncrossover is where ''faster'' flips. ' ...
         'NaN predictions mark cells the model\ndecides structurally ' ...
         'rather than by pricing.\n\n']);

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
