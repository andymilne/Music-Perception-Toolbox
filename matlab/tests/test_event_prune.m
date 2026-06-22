%% test_event_prune.m
%
%  Regression tests for the density-level live-event prune consumed by
%  the inner-product / total-mass paths: method='renyi2' in
%  entropyExpTens and cosSimExpTens. The prune lives in
%  internal.prunedExpTens and is the MATLAB counterpart of the Python
%  density pruned() method (see python/tests/test_event_prune.py).
%
%  A dead event --- weight zero or NaN on an attribute, so it
%  contributes nothing to any inner product or total mass --- is dropped
%  before the per-attribute IP / mass work. This is the sibling, one
%  level up, of the cell-mass tuple prune in
%  test_cell_mass_zero_weight_prune: that drops zero-weight tuples inside
%  the grid path; this drops whole events before the IP / mass
%  reduction. It is needed because maPerAttrInnerMatrix already prunes
%  per attribute, but weightEvents writes its window factor to only the
%  target attribute, so an event hard-zeroed through one attribute
%  remains present (all-ones) on the others.
%
%  Liveness rule (single predicate: finite and nonzero):
%    * SA: an element is live iff its weight is finite and nonzero.
%    * MA: an event is live iff EVERY attribute has at least one finite,
%      nonzero slot in that event's column (per-attribute factors
%      multiply; an all-zero / all-NaN column kills, a partly-zero
%      column does not).
%
%  Standalone-runnable. When invoked from test_mpt.m the existing
%  `results` cell is appended to; when run alone, a summary follows.

if ~exist('results', 'var')
    results = {};
    standalone_ep = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults_ep
    cleanupDefaults_ep = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone_ep = false;
end

mptDefaults('reset');


%% -----------------------------------------------------------------
%% Liveness rule, observed through internal.prunedExpTens
%% -----------------------------------------------------------------

% SA: live iff finite and nonzero. Zeros kill; finite nonzero (incl.
% negative) survive.
densSA = buildExpTens([60 62 64 66 68], [1 0 0 2 -3], 0.5, 1, ...
                      false, false, 0);
prSA = internal.prunedExpTens(densSA);
results{end+1,1} = 'SA prune drops zero-weight elements, keeps finite nonzero';
results{end,2}   = isequal(prSA.p(:).', [60 66 68]) ...
                   && isequal(prSA.w(:).', [1 2 -3]);

% MA: an all-zero column on the pitch attribute (K=2) kills the event;
% a partly-zero column (one live slot) keeps it.
pPitch = [60 62 64; 67 69 71];
wPitch = [1 0 1; 0 0 1];        % event 2 all-zero -> dead; 1,3 partly -> live
pTime  = [0 1 2];
densMA = buildExpTens({pPitch, pTime}, {wPitch, ones(1,3)}, ...
                      [0.5 0.15], [2 1], ...
                      [false false], [false false], [0 0]);
prMA = internal.prunedExpTens(densMA);
results{end+1,1} = 'MA prune: all-zero column kills, partly-zero column lives';
results{end,2}   = (prMA.N == 2) ...
                   && isequal(prMA.pAttr{1}, [60 64; 67 71]) ...
                   && isequal(prMA.pAttr{2}, [0 2]);

% MA: a live pitch column cannot rescue an event whose time column is
% zero --- the per-attribute factors multiply.
densMA2 = buildExpTens({[60 62 64], [0 1 2]}, ...
                       {ones(1,3), [1 0 1]}, ...
                       [0.5 0.15], [1 1], ...
                       [false false], [false false], [0 0]);
prMA2 = internal.prunedExpTens(densMA2);
results{end+1,1} = 'MA prune: zero on second attribute kills the event';
results{end,2}   = (prMA2.N == 2) ...
                   && isequal(prMA2.pAttr{1}, [60 64]) ...
                   && isequal(prMA2.pAttr{2}, [0 2]);


%% -----------------------------------------------------------------
%% pruned() returns the density unchanged when nothing is dead
%% -----------------------------------------------------------------

densLiveSA = buildExpTens([60 62 64], [1 1 2], 0.5, 1, false, false, 0);
results{end+1,1} = 'SA prune returns density unchanged when all live';
results{end,2}   = isequaln(internal.prunedExpTens(densLiveSA), densLiveSA);

densLiveMA = buildExpTens({[60 62 64], [0 1 2]}, {ones(1,3), ones(1,3)}, ...
                          [0.5 0.15], [1 1], ...
                          [false false], [false false], [0 0]);
results{end+1,1} = 'MA prune returns density unchanged when all live';
results{end,2}   = isequaln(internal.prunedExpTens(densLiveMA), densLiveMA);


%% -----------------------------------------------------------------
%% Numerical invariance: dead events change nothing
%% -----------------------------------------------------------------

% --- SA Rényi-2 ---
pSA = [60 62 64 66 68 70];
wSA = [1.0 0.7 1.3 0.9 1.1 0.5];
deadSA = [2 5];                 % zero these
wSAd = wSA;  wSAd(deadSA) = 0;
keepSA = true(1, numel(pSA));  keepSA(deadSA) = false;

H_sa_with = entropyExpTens(buildExpTens(pSA, wSAd, 0.5, 1, false, false, 0), ...
                           'method', 'renyi2', 'verbose', false);
H_sa_without = entropyExpTens( ...
    buildExpTens(pSA(keepSA), wSA(keepSA), 0.5, 1, false, false, 0), ...
    'method', 'renyi2', 'verbose', false);
results{end+1,1} = 'SA renyi2 invariant to dead events';
results{end,2}   = H_sa_with == H_sa_without;

% --- MA Rényi-2 (cross-attribute kill: zero on pitch, ones on time) ---
pPm = [60 63 66 69 72 62 65 68];
pTm = [0.0 0.3 0.6 0.9 1.2 1.5 1.8 2.1];
wPm = ones(1, 8);
deadMA = [2 5 7];
wPm(deadMA) = 0;
keepMA = true(1, 8);  keepMA(deadMA) = false;

dMA_with = buildExpTens({pPm, pTm}, {wPm, ones(1,8)}, ...
                        [0.5 0.15], [1 1], ...
                        [false false], [false false], [0 0]);
dMA_without = buildExpTens({pPm(keepMA), pTm(keepMA)}, ...
                           {wPm(keepMA), ones(1, nnz(keepMA))}, ...
                           [0.5 0.15], [1 1], ...
                           [false false], [false false], [0 0]);
H_ma_with = entropyExpTens(dMA_with, 'method', 'renyi2', 'verbose', false);
H_ma_without = entropyExpTens(dMA_without, 'method', 'renyi2', 'verbose', false);
results{end+1,1} = 'MA renyi2 invariant to dead events';
results{end,2}   = H_ma_with == H_ma_without;

% --- SA cosine similarity ---
pX = [60 62 64 66 68];
wX = [1 0 1 0 1];               % events 2, 4 dead
keepX = wX ~= 0;
dQ = buildExpTens([61 63 65], [1 1 1], 0.5, 1, false, false, 0);
s_sa_with = cosSimExpTens( ...
    buildExpTens(pX, wX, 0.5, 1, false, false, 0), dQ, 'verbose', false);
s_sa_without = cosSimExpTens( ...
    buildExpTens(pX(keepX), wX(keepX), 0.5, 1, false, false, 0), dQ, ...
    'verbose', false);
results{end+1,1} = 'SA cosine invariant to dead events';
results{end,2}   = s_sa_with == s_sa_without;

% --- MA cosine similarity ---
pXp = [60 63 66 69 72 62];
pXt = [0.0 0.3 0.6 0.9 1.2 1.5];
wXp = ones(1, 6);  wXp([3 5]) = 0;
keepXm = wXp ~= 0;
dQm = buildExpTens({[61 64 67], [0.1 0.4 0.7]}, {ones(1,3), ones(1,3)}, ...
                   [0.5 0.15], [1 1], ...
                   [false false], [false false], [0 0]);
dXm_with = buildExpTens({pXp, pXt}, {wXp, ones(1,6)}, ...
                        [0.5 0.15], [1 1], ...
                        [false false], [false false], [0 0]);
dXm_without = buildExpTens({pXp(keepXm), pXt(keepXm)}, ...
                           {wXp(keepXm), ones(1, nnz(keepXm))}, ...
                           [0.5 0.15], [1 1], ...
                           [false false], [false false], [0 0]);
s_ma_with = cosSimExpTens(dXm_with, dQm, 'verbose', false);
s_ma_without = cosSimExpTens(dXm_without, dQm, 'verbose', false);
results{end+1,1} = 'MA cosine invariant to dead events';
results{end,2}   = s_ma_with == s_ma_without;


%% -----------------------------------------------------------------
%% Realistic weightEvents truncation case (MA): auto vs manual prune
%% -----------------------------------------------------------------

rng(2, 'twister');
N_events = 600;
pitches = 60 + 24 * rand(1, N_events);
times   = (0:N_events-1) * 0.25;
pAttrPre = {pitches, times};
wPre     = {ones(1, N_events), ones(1, N_events)};
groupsPre = [1 2];

mptDefaults('truncationSigmas', 3.0);
centre_qn = times(round(N_events / 2));
% weightEvents deletes the input (time) attribute, leaving a single
% pitch attribute whose column is zeroed for far events.
[p_w, w_w, g_w] = weightEvents(pAttrPre, wPre, 2, 1, centre_qn, 0.0, 'sd', 1.0, 'dropInputAttr', true);

event_w = sum(w_w{1}, 1);
n_kept = sum(event_w > 0);
results{end+1,1} = 'narrow Gaussian keeps only a handful of events';
results{end,2}   = (n_kept > 0) && (n_kept < 40);

H_auto = entropyExpTens(p_w, w_w, 0.5, 1, false, false, 0.0, ...
    'method', 'renyi2', 'verbose', false);
keep = event_w > 0;
p_w_pruned = cellfun(@(p) p(:, keep), p_w, 'UniformOutput', false);
w_w_pruned = cellfun(@(w) w(:, keep), w_w, 'UniformOutput', false);
H_manual = entropyExpTens(p_w_pruned, w_w_pruned, 0.5, 1, ...
    false, false, 0.0, 'method', 'renyi2', 'verbose', false);
results{end+1,1} = 'auto-prune renyi2 matches manual prune (windowed)';
results{end,2}   = H_auto == H_manual;

% Bounded-time soft guard on a large windowed density: correctness is
% anchored by the parity check above; this catches a prune that
% silently stops firing.
rng(3, 'twister');
N_big = 3000;
pitchesBig = 60 + 24 * rand(1, N_big);
timesBig   = (0:N_big-1) * 0.25;
[p_wb, w_wb, g_wb] = weightEvents({pitchesBig, timesBig}, {ones(1, N_big), ones(1, N_big)}, 2, 1, timesBig(round(N_big / 2)), 0.0, 'sd', 1.0, 'dropInputAttr', true);
t0 = tic;
H_big = entropyExpTens(p_wb, w_wb, 0.5, 1, false, false, 0.0, ...
    'method', 'renyi2', 'verbose', false);
elapsed = toc(t0);
results{end+1,1} = 'renyi2 on large windowed density completes in bounded time';
results{end,2}   = isfinite(H_big) && (elapsed < 60);

mptDefaults('reset');


%% -----------------------------------------------------------------
%% Fully dead density
%% -----------------------------------------------------------------

densDeadMA = buildExpTens({[60 62 64], [0 1 2]}, {zeros(1,3), ones(1,3)}, ...
                          [0.5 0.15], [1 1], ...
                          [false false], [false false], [0 0]);
results{end+1,1} = 'all-dead MA prune yields N = 0';
results{end,2}   = (internal.prunedExpTens(densDeadMA).N == 0);
H_dead_ma = entropyExpTens(densDeadMA, 'method', 'renyi2', 'verbose', false);
results{end+1,1} = 'all-dead MA renyi2 is NaN';
results{end,2}   = isnan(H_dead_ma);

% An entirely zero-mass SA density is genuinely degenerate: collision
% entropy of zero mass is undefined, so renyi2 returns NaN (matching the
% MA path and the value a windowed sweep wants at out-of-support centres)
% rather than erroring.
densDeadSA = buildExpTens([60 62 64], [0 0 0], 0.5, 1, false, false, 0);
H_dead_sa = entropyExpTens(densDeadSA, 'method', 'renyi2', 'verbose', false);
results{end+1,1} = 'all-dead SA renyi2 is NaN';
results{end,2}   = isnan(H_dead_sa);


if standalone_ep
    nPass = sum(cell2mat(results(:,2)));
    nTot  = size(results, 1);
    fprintf('\n%s: %d / %d passed.\n', mfilename, nPass, nTot);
    if nPass < nTot
        for k = 1:nTot
            if ~results{k,2}
                fprintf('  FAILED: %s\n', results{k,1});
            end
        end
    end
    clear cleanupDefaults_ep
end
