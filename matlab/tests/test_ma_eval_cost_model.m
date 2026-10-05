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
cm_grid = {
  {'A1 abs r2 K6',  30,      2,     false,         false,         0,       6}
  {'A1 abs r2 K20', 30,      2,     false,         false,         0,       20}
  {'A1 abs r3 K8',  30,      3,     false,         false,         0,       8}
  {'A2 abs r2 K5',  [30 25], [2 2], [false false], [false false], [0 0],   5}
  {'A2 abs r2 K12', [30 25], [2 2], [false false], [false false], [0 0],   12}
  {'A2 rel r2 K8',  [30 25], [2 2], [true true],   [false false], [0 0],   8}
};
rng(1, 'twister');
for gi = 1:numel(cm_grid)
    c = cm_grid{gi};
    [label, sig, rv, rel, per, P, K] = c{:};
    A = numel(sig);
    pas = cell(A, 1);
    for a = 1:A, pas{a} = 100 * rand(K, 1); end
    dens = buildMaet(pas, repmat({[]}, A, 1), sig, rv, rel, per, P, ...
        'verbose', false);
    xq = 100 * rand(dens.dim, 200);
    [chosen, reason] = internal.selectMaEval(dens, 200);

    if strcmp(chosen, 'mobius')
        ok = true;   % failure-safe route; never harmful
    elseif ~startsWith(reason, 'cost model')
        ok = true;   % correctness-forced centres; speed contract N/A
    else
        tCen = localTime(@() evalMaet(dens, xq, 'method', 'centres', 'verbose', false));
        tMob = localTime(@() evalMaet(dens, xq, 'method', 'mobius', 'verbose', false));
        ok = (tCen / max(tMob, 1e-9)) <= MAX_TOLERATED_CENTRES_SLOWDOWN;
    end
    results{end+1, 1} = sprintf('cost model never badly wrong: %s', label); %#ok<*AGROW>
    results{end, 2} = ok;
end

%% ---- Above sigma/P, all-image Möbius is the preferred default ----
dens = buildMaet({100*rand(6,1); 100*rand(6,1)}, {[]; []}, [40 40], ...
    [3 3], [true true], [true true], [1200 1200], 'verbose', false);
[chosen, ~] = internal.selectMaEval(dens, 200);
results{end+1, 1} = 'cost model: rel-per above threshold prefers Möbius';
results{end, 2} = strcmp(chosen, 'mobius');

% and single-image remains available via override
[chosenC, reasonC] = deal('', '');
if true
    % method override path lives in evalMaet, not the selector; the
    % selector is only asked for 'auto'. Confirm the selector's default
    % here is Möbius (above), and that a large shape stays Möbius (safe).
    densBig = buildMaet({100*rand(40,1); 100*rand(40,1)}, {[]; []}, ...
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
densInf = buildMaet({100*rand(20,1); 100*rand(20,1)}, {[]; []}, ...
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
densStale = buildMaet(100*rand(12,1), [], 15, 2, false, false, 0, ...
    'verbose', false);
ok = false;
try
    internal.selectMaEval(densStale, 200, false);
catch err
    ok = strcmp(err.identifier, 'mpt:selectMaEval:staleCallForm');
end
results{end+1, 1} = 'cost model: the removed verbose argument raises';
results{end, 2} = ok;

%% ---- Onsets beside a chord's pitches take the joint-centres path ----
% Onsets at r = 1 beside a chord's pitches at r = 2 take the joint-centres
% path, culled on the onset. The factored Möbius evaluator calls the
% single-multiset evaluator once per event and attribute, so its cost
% grows with the event count however cheap each call is, while the
% joint-centres path meets each query with only the events near it.
% Measured October 2026 on the maintainer's VM (Python): 0.15 against
% 3.9 ms at N = 30, 0.61 against 39 ms at N = 300, and 6.2 against
% 362 ms at N = 3000. Twin of the Python
% test_ma_cost_model_routes_onsets_and_chords_to_the_joint_path.
rng(0, 'twister');
ok = true;
for cmN = [30 300 3000]
    cmOn = sort(100 * rand(1, cmN));
    cmPitch = 1200 * rand(3, cmN);
    densOC = buildMaet({cmOn, cmPitch}, {ones(1, cmN), ones(3, cmN)}, ...
        [0.3 15], [1 2], [false false], [false false], [0 0], ...
        'verbose', false);
    ok = ok && strcmp(internal.selectMaEval(densOC, 200), 'centres');
end
results{end+1, 1} = 'cost model: onsets beside chord pitches take the joint-centres path';
results{end, 2} = ok;

%% ---- The r = 1 rule keeps centres only where the joint set is no larger ----
% At r = 1 on every attribute the joint-centres path holds prod_a K_a
% centres per event and the factored Möbius evaluator sums over sum_a K_a
% values. A single multiset, an event list, or one attribute of many
% values beside scalars keeps centres by rule; several attributes of many
% values go to the cost model, which picks Möbius where the product has
% outgrown the sum. Twin of the Python
% test_r1_rule_keeps_centres_only_where_the_joint_set_is_no_larger.
rng(0, 'twister');
cmRule = 'r = 1, joint set no larger than the values';
cmMk1 = @(Ks, n) buildMaet( ...
    arrayfun(@(k) 100 * rand(k, n), Ks, 'UniformOutput', false), ...
    arrayfun(@(k) ones(k, n), Ks, 'UniformOutput', false), ...
    3 * ones(1, numel(Ks)), ones(1, numel(Ks)), false(1, numel(Ks)), ...
    false(1, numel(Ks)), zeros(1, numel(Ks)), 'verbose', false);
[~, r1] = internal.selectMaEval(cmMk1(12, 1), 200);
[~, r2] = internal.selectMaEval(cmMk1([1 1 1], 500), 200);
[~, r3] = internal.selectMaEval(cmMk1([6 1], 500), 200);
[c4, r4] = internal.selectMaEval(cmMk1([4 4], 200), 200);
[c5, r5] = internal.selectMaEval(cmMk1([50 50], 200), 200);
results{end+1, 1} = 'cost model: r = 1 rule keeps centres where the joint set is no larger';
results{end, 2} = strcmp(r1, cmRule) && strcmp(r2, cmRule) && strcmp(r3, cmRule);
results{end+1, 1} = 'cost model: r = 1 with several many-valued attributes is priced, not ruled';
results{end, 2} = strncmp(r4, 'cost model', 10) && strncmp(r5, 'cost model', 10) ...
    && strcmp(c5, 'mobius');
clear cmRule cmMk1 r1 r2 r3 r4 r5 c4 c5

%% ---- The event-by-event routes scale with the event count ----
% The factored centres route and the factored Möbius evaluator take a
% density event by event, so beyond the per-call setup their cost is N
% times one event's; the joint-centres path (r = [1 2]) holds N times the
% joint centres. Every event holds the same values, so the spreads, and
% with them the culled shares, do not change with N. The setup terms are
% read off the estimates at N = 1 and N = 2.
for cmR = {[2 2], [1 2]}
    cmMk = @(n) buildMaet( ...
        {repmat(linspace(0, 100, 4).', 1, n), repmat(linspace(0, 100, 4).', 1, n)}, ...
        {ones(4, n), ones(4, n)}, [3 3], cmR{1}, [false false], ...
        [false false], [0 0], 'verbose', false);
    [c1, m1] = internal.maEvalCostsMs(cmMk(1), 200);
    [c2, m2] = internal.maEvalCostsMs(cmMk(2), 200);
    [c5, m5] = internal.maEvalCostsMs(cmMk(5), 200);
    % Linear in N: the increment from 1 to 5 events is four times the
    % increment from 1 to 2.
    okM = abs((m5 - m1) - 4 * (m2 - m1)) <= 1e-12 * m5;
    okC = abs((c5 - c1) - 4 * (c2 - c1)) <= 1e-12 * c5;
    results{end+1, 1} = sprintf('cost model: estimates linear in the event count, r = [%d %d]', cmR{1});
    results{end, 2} = okM && okC && m2 > m1 && c2 > c1;
end
clear cmN cmOn cmPitch densOC cmR cmMk c1 c2 c5 m1 m2 m5 okM okC

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
