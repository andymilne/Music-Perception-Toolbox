%% test_cell_mass_zero_weight_prune.m
%
%  Regression tests for the auto-prune of zero-weight tuples in
%  localCellMassesMAAbsolute / localCellMassesSingleMultisetAbsolute inside
%  entropyExpTens.
%
%  The auto-prune mirrors the eval-path prune in evalExpTens. Without
%  it, when weightEvents truncates most events to zero weight under a
%  Gaussian or rectangle window, the cell-mass tensor contraction
%  builds (nJ x nCells) erf-difference matrices over every upstream
%  tuple including the zero-weighted ones, which can OOM on long
%  sequences even though zero-weight tuples contribute exactly zero.
%
%  Standalone-runnable. When invoked from test_mpt.m the existing
%  `results` cell is appended to; when run alone, a summary follows.

if ~exist('results', 'var')
    results = {};
    standalone_cm = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults_cm
    cleanupDefaults_cm = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone_cm = false;
end

mptDefaults('reset');


%% -----------------------------------------------------------------
%% Auto-prune in localCellMassesMAAbsolute
%% -----------------------------------------------------------------

rng(0, 'twister');
N_events = 500;
n_voices = 4;
K_partials = 12;
K = n_voices * K_partials;
pitches = 4000 + 3000 * rand(N_events, n_voices);   % cents
times = (0:N_events-1) * 0.25;                       % 16th-note grid (1 x N row)

pPartials = zeros(N_events, K);
wPartials = zeros(N_events, K);
for n = 1:N_events
    [pAug, wAug] = addSpectra(pitches(n, :), [], 'harmonic', K_partials, ...
                              'powerlaw', 1.0);
    pPartials(n, :) = pAug(:).';
    wPartials(n, :) = wAug(:).';
end

% Pre-MAET inputs: pitch attribute (group 1, K = 48 partials) and time
% attribute (group 2, K = 1).
pAttrPre = {pPartials.', times};
wPre     = {wPartials.', ones(1, N_events)};
groupsPre = [1, 2];

% Apply Gaussian window of sd = 1 at mid-sequence with truncationSigmas = 3.
mptDefaults('truncationSigmas', 3.0);

centre_qn = times(round(N_events / 2));
% weightEvents signature (MATLAB, carrier form):
%   (pAttr, w, inputAttr, targetAttr, centre, shape,
%    'sd', s,  'dropInputAttr', tf)
% inputAttr = 2 (time supplies the per-event scalar values),
% targetAttr = 1 (pitch's weight slot receives the factor).
[p_w, w_w, g_w] = weightEvents(pAttrPre, wPre, 2, 1, centre_qn, 0.0, 'sd', 1.0, 'dropInputAttr', true);

results{end+1,1} = 'weightEvents preserves event count after truncation';
results{end,2}   = (size(p_w{1}, 2) == N_events);

event_w = sum(w_w{1}, 1);
n_kept = sum(event_w > 0);
results{end+1,1} = 'narrow Gaussian keeps only a handful of events';
results{end,2}   = (n_kept > 0) && (n_kept < 30);

% Auto-prune differential entropy.
% entropyExpTens raw MA signature:
%   (pCell, wCell, sigma, r, groups, isRel, isPer, period, ...
%    'method', M, 'base', b, ...)
t0 = tic;
H_auto_diff = entropyExpTens(p_w, w_w, 10.0, 1, ...
    false, false, 0.0, 'method', 'differential', 'base', 2.0, ...
    'verbose', false);
elapsed = toc(t0);
results{end+1,1} = 'differential entropy on truncated sequence is finite';
results{end,2}   = isfinite(H_auto_diff);
results{end+1,1} = 'differential entropy on truncated sequence completes in bounded time';
results{end,2}   = (elapsed < 60);

% Manual prune for parity check.
keep = event_w > 0;
p_w_pruned = cellfun(@(p) p(:, keep), p_w, 'UniformOutput', false);
w_w_pruned = cellfun(@(w) w(:, keep), w_w, 'UniformOutput', false);
H_manual_diff = entropyExpTens(p_w_pruned, w_w_pruned, 10.0, 1, ...
    false, false, 0.0, 'method', 'differential', 'base', 2.0, ...
    'verbose', false);

results{end+1,1} = 'auto-prune differential matches manual prune';
results{end,2}   = abs(H_auto_diff - H_manual_diff) < 1e-8;

% Shannon-on-fixed-grid parity check.
H_auto_shan = entropyExpTens(p_w, w_w, 10.0, 1, ...
    false, false, 0.0, 'method', 'shannon', 'base', 2.0, ...
    'nPointsPerDim', 2001, 'xMin', 3000.0, 'xMax', 13000.0, ...
    'verbose', false);
H_manual_shan = entropyExpTens(p_w_pruned, w_w_pruned, 10.0, 1, ...
    false, false, 0.0, 'method', 'shannon', 'base', 2.0, ...
    'nPointsPerDim', 2001, 'xMin', 3000.0, 'xMax', 13000.0, ...
    'verbose', false);

results{end+1,1} = 'auto-prune shannon matches manual prune';
results{end,2}   = abs(H_auto_shan - H_manual_shan) < 1e-10;

mptDefaults('reset');


if standalone_cm
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
end
