%% test_cell_mass_zero_weight_prune.m — rectWidthFromSupport + cell-mass prune
%
%  Tests for rectWidthFromSupport helper and for the auto-prune of
%  zero-weight tuples in localCellMassesMAAbsolute /
%  localCellMassesSAAbsolute inside entropyExpTens.
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
    % Defaults isolation when run standalone (when invoked from
    % test_mpt.m the outer wrapper has already isolated defaults).
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults_cm
    cleanupDefaults_cm = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone_cm = false;
end

% Reset defaults before testing.
mptDefaults('reset');


%% -----------------------------------------------------------------
%% Helper math
%% -----------------------------------------------------------------

% Total support 0.25 -> width = 0.25 / (2*sqrt(3))
w_rect = rectWidthFromSupport(0.25);
results{end+1,1} = 'rectWidthFromSupport(0.25) = 0.25 / (2*sqrt(3))';
results{end,2}   = abs(w_rect - 0.25/(2*sqrt(3))) < 1e-15;

% Round-trip: full support implied by the returned width must recover the input.
roundtrip_ok = true;
for L = [0.05, 0.1, 0.25, 0.5, 1.0, 7.5]
    w_L = rectWidthFromSupport(L);
    L_implied = 2 * w_L * sqrt(3);
    if abs(L_implied - L) > 1e-12
        roundtrip_ok = false;
        break;
    end
end
results{end+1,1} = 'rectWidthFromSupport: full-support round-trip';
results{end,2}   = roundtrip_ok;

% Non-positive / non-finite inputs error.
threw = false;
try; rectWidthFromSupport(0);    catch; threw = true; end %#ok<NOSEM>
results{end+1,1} = 'rectWidthFromSupport(0) errors';
results{end,2}   = threw;

threw = false;
try; rectWidthFromSupport(-1.0); catch; threw = true; end %#ok<NOSEM>
results{end+1,1} = 'rectWidthFromSupport(-1.0) errors';
results{end,2}   = threw;

threw = false;
try; rectWidthFromSupport(NaN);  catch; threw = true; end %#ok<NOSEM>
results{end+1,1} = 'rectWidthFromSupport(NaN) errors';
results{end,2}   = threw;

threw = false;
try; rectWidthFromSupport(Inf);  catch; threw = true; end %#ok<NOSEM>
results{end+1,1} = 'rectWidthFromSupport(Inf) errors';
results{end,2}   = threw;


%% -----------------------------------------------------------------
%% Auto-prune in localCellMassesMAAbsolute
%% -----------------------------------------------------------------
%
%  Build a 500-event sequence with 4 voices per event and a 12-partial
%  spectrum. Apply a narrow Gaussian window via weightEvents that zeros
%  most events. Differential and shannon entropy with auto-prune must
%  match the manual-prune reference to high precision.

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

% Pre-MAET inputs: 2 attributes (pitch K=48 partials, time K=1 events).
% MATLAB indexing: attribute 1 is pitch, attribute 2 is time, both in
% their own groups (1 and 2 respectively).
pAttrPre = {pPartials.', times};                   % 1xA cell of (K_a x N) matrices
wPre     = {wPartials.', ones(1, N_events)};
groupsPre = [1, 2];

% Apply Gaussian window of sigma=1 at mid-sequence with truncationSigmas=3.
% truncationSigmas hard-zeros events more than 3 widths from the centre.
mptDefaults('truncationSigmas', 3.0);

centre_qn = times(round(N_events / 2));
% weightEvents signature (MATLAB):
%   (pAttr, w, groups, inputAttr, targetAttr, centre, width, shape, ...
%    isPer, period, 'deleteInput', tf)
% inputAttr = 2 (time supplies the per-event scalar values),
% targetAttr = 1 (pitch's weight slot receives the factor).
[p_w, w_w, g_w] = weightEvents(pAttrPre, wPre, groupsPre, ...
    2, 1, centre_qn, 1.0, 0.0, false, 0.0, 'deleteInput', true);

% Confirm weightEvents leaves all events in but zeros most weights.
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
H_auto_diff = entropyExpTens(p_w, w_w, 10.0, 1, g_w, ...
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
H_manual_diff = entropyExpTens(p_w_pruned, w_w_pruned, 10.0, 1, g_w, ...
    false, false, 0.0, 'method', 'differential', 'base', 2.0, ...
    'verbose', false);

results{end+1,1} = 'auto-prune differential matches manual prune';
results{end,2}   = abs(H_auto_diff - H_manual_diff) < 1e-8;

% Shannon-on-fixed-grid parity check.
H_auto_shan = entropyExpTens(p_w, w_w, 10.0, 1, g_w, ...
    false, false, 0.0, 'method', 'shannon', 'base', 2.0, ...
    'nPointsPerDim', 2001, 'xMin', 3000.0, 'xMax', 13000.0, ...
    'verbose', false);
H_manual_shan = entropyExpTens(p_w_pruned, w_w_pruned, 10.0, 1, g_w, ...
    false, false, 0.0, 'method', 'shannon', 'base', 2.0, ...
    'nPointsPerDim', 2001, 'xMin', 3000.0, 'xMax', 13000.0, ...
    'verbose', false);

results{end+1,1} = 'auto-prune shannon matches manual prune';
results{end,2}   = abs(H_auto_shan - H_manual_shan) < 1e-10;

% Restore factory defaults so subsequent tests see a clean state.
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
