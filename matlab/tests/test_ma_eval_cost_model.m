%% test_ma_eval_cost_model.m — MA eval dispatch calibration contract
%
%  The MA eval selector (internal.selectMaEval) is a pure cost model with
%  no probe. A per-language group of calibration constants (the MA_COST_*
%  group in internal.selectMaEval) absorbs implementation constant
%  factors, settled once per language by measurement
%  (bench_ma_eval_calibration.m), not rediscovered at runtime. This test
%  guarantees the calibration stays current.
%
%  The contract is ASYMMETRIC. Picking Möbius when centres would be
%  marginally faster costs a fraction of a millisecond (the factored path
%  is flat and fast). Picking centres when Möbius is much faster is the
%  harmful mistake (the centres path materialises the joint tuple set and
%  can be orders slower or exhaust memory). So the test does not demand
%  the strictly faster route always -- it demands the model is never
%  BADLY wrong: whenever it picks centres FOR COST REASONS, centres is
%  within a modest factor of Möbius. Correctness-forced centres picks
%  (feasibility) are exempt from the speed contract.
%
%  Twin of python tests/test_ma_eval_cost_model.py.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults_cm
    cleanupDefaults_cm = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

MAX_TOLERATED_CENTRES_SLOWDOWN = 3.0;

%% ---- Never badly wrong: cost-driven centres picks are not much slower ----
grid = {
  {'A1 abs r2 K6',  30,      2,     false,         false,         0,       6}
  {'A1 abs r2 K20', 30,      2,     false,         false,         0,       20}
  {'A1 abs r3 K8',  30,      3,     false,         false,         0,       8}
  {'A2 abs r2 K5',  [30 25], [2 2], [false false], [false false], [0 0],   5}
  {'A2 abs r2 K12', [30 25], [2 2], [false false], [false false], [0 0],   12}
  {'A2 rel r2 K8',  [30 25], [2 2], [true true],   [false false], [0 0],   8}
};
rng(1, 'twister');
for gi = 1:numel(grid)
    c = grid{gi};
    [label, sig, rv, rel, per, P, K] = c{:};
    A = numel(sig);
    pas = cell(A, 1);
    for a = 1:A, pas{a} = 100 * rand(K, 1); end
    dens = buildExpTens(pas, repmat({[]}, A, 1), sig, rv, rel, per, P, ...
        'verbose', false);
    xq = 100 * rand(dens.dim, 200);
    [chosen, reason] = internal.selectMaEval(dens, 200);

    if strcmp(chosen, 'mobius')
        ok = true;   % failure-safe route; never harmful
    elseif ~startsWith(reason, 'cost model')
        ok = true;   % correctness-forced centres; speed contract N/A
    else
        tCen = localTime(@() evalExpTens(dens, xq, 'method', 'centres', 'verbose', false));
        tMob = localTime(@() evalExpTens(dens, xq, 'method', 'mobius', 'verbose', false));
        ok = (tCen / max(tMob, 1e-9)) <= MAX_TOLERATED_CENTRES_SLOWDOWN;
    end
    results{end+1, 1} = sprintf('cost model never badly wrong: %s', label); %#ok<*AGROW>
    results{end, 2} = ok;
end

%% ---- Above sigma/P, all-image Möbius is the preferred default ----
dens = buildExpTens({100*rand(6,1); 100*rand(6,1)}, {[]; []}, [40 40], ...
    [3 3], [true true], [true true], [1200 1200], 'verbose', false);
[chosen, ~] = internal.selectMaEval(dens, 200);
results{end+1, 1} = 'cost model: rel-per above threshold prefers Möbius';
results{end, 2} = strcmp(chosen, 'mobius');

% and single-image remains available via override
[chosenC, reasonC] = deal('', '');
if true
    % method override path lives in evalExpTens, not the selector; the
    % selector is only asked for 'auto'. Confirm the selector's default
    % here is Möbius (above), and that a large shape stays Möbius (safe).
    densBig = buildExpTens({100*rand(40,1); 100*rand(40,1)}, {[]; []}, ...
        [40 40], [3 3], [true true], [true true], [1200 1200], 'verbose', false);
    [chosenC, reasonC] = internal.selectMaEval(densBig, 200); %#ok<ASGLU>
end
results{end+1, 1} = 'cost model: rel-per stays Möbius at large shape (no OOM)';
results{end, 2} = strcmp(chosenC, 'mobius');

%% ---- Forced-centres infeasible -> raises singleImageInfeasible ----
% Two attributes, r = 11 > feasibility bound forces centres; K = 20
% makes the joint centre set ~10^12 tuples, past the memory budget. Use
% a genuine multi-attribute density so the MA eval selector is exercised
% (the single-attribute vector path routes through the single multiset dispatch, not
% selectMaEval).
densInf = buildExpTens({100*rand(20,1); 100*rand(20,1)}, {[]; []}, ...
    [6 6], [11 11], [false false], [false false], [0 0], 'verbose', false);
ok = false;
try
    internal.selectMaEval(densInf, 200);
catch err
    ok = strcmp(err.identifier, 'mpt:dispatch:singleImageInfeasible');
end
results{end+1, 1} = 'cost model: forced-centres infeasible raises';
results{end, 2} = ok;

% The removed verbose parameter. selectMaEval once took
% (dens, nQ, verbose, truncationSigmas). A caller left on that form would
% hand a logical to relPerSigmaOverPThreshold, which reads false as zero
% and returns the positive-definiteness ceiling in place of the accuracy
% threshold --- a different route, chosen silently. The guard turns that
% into an error, so this pins the loud failure rather than the routing.
densStale = buildExpTens(100*rand(12,1), [], 15, 2, false, false, 0, ...
    'verbose', false);
ok = false;
try
    internal.selectMaEval(densStale, 200, false);
catch err
    ok = strcmp(err.identifier, 'mpt:selectMaEval:staleCallForm');
end
results{end+1, 1} = 'cost model: the removed verbose argument raises';
results{end, 2} = ok;

%% ---- Standalone reporting ----
if standalone
    nFail = 0;
    for ii = 1:size(results, 1)
        if ~isequal(results{ii, 2}, true), nFail = nFail + 1; end
    end
    fprintf('test_ma_eval_cost_model: %d passed, %d failed (of %d)\n', ...
        size(results, 1) - nFail, nFail, size(results, 1));
    for ii = 1:size(results, 1)
        if ~isequal(results{ii, 2}, true)
            fprintf('  FAIL  %s\n', results{ii, 1});
        end
    end
    if exist('cleanupDefaults_cm', 'var'), clear cleanupDefaults_cm; end
end

% ---- helper ----
function t = localTime(fn)
    fn();
    ts = zeros(1, 5);
    for i = 1:5, tic; fn(); ts(i) = toc; end
    t = min(ts);
end
