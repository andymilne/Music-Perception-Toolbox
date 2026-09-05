%% test_centres_chunking.m — v3 single multiset centres-path n_q chunking
%
%  Regression test. Catches the OOM that surfaced when
%  demo_triadConsonance ran at the 10-cent grid: K=72, r=3, rel,
%  non-periodic, n_q=29161 — the difference tensor would have been
%  2 x 357840 x 29161 = 155 GB unchunked, exceeding MATLAB's
%  default 48 GB array-size cap.
%
%  This regression slipped through the original test suite because
%  every existing routing/parity case used n_q in the 20-50 range,
%  well below the chunking threshold for any reasonable density.
%  Any new fast-path or routing change that touches centres-path
%  evaluation MUST keep this test passing — verifying small-n_q
%  parity alone is not sufficient.
%
%  The test strategy:
%   (a) Build the same density as the demo (K=72, r=3, rel).
%   (b) Call evalExpTens at n_q=1000 — large enough to overflow the
%       difference-tensor allocation under any chunking budget below
%       ~5 GB, but small enough for a routine test run (~few seconds).
%   (c) Verify (i) no OOM, (ii) all outputs finite, (iii) the chunked
%       result equals the concatenation of two manual halves
%       (FP-bit-identical, since per-query output depends only on
%       the q-th query — chunking only changes memory footprint).
%
%  Standalone-runnable. When invoked from test_mpt.m the existing
%  `results` cell is appended to.

if ~exist('results', 'var')
    results = {};
    % Defaults isolation when run standalone (when invoked from
    % test_mpt.m the outer wrapper has already isolated defaults).
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults_cc
    cleanupDefaults_cc = mptTestIsolateDefaults(); %#ok<NASGU>
end

% --- Density: matches demo_triadConsonance ---
%   nJ = 72 * 71 * 70 = 357,840
%   dim = r - 1 = 2 (rel mode)
%   Per-n_q memory in the difference tensor:
%     (dim + 1) * nJ * 8 = 8.59 MB
%   At n_q = 1000: 8.59 GB unchunked — triggers chunking under
%   MATLAB's default budget (memInfo.MaxPossibleArrayBytes * 0.5)
%   on systems with < 17 GB free, and certainly under the 4 GB
%   fallback used when `memory` is unavailable.
cc_K = 72;
cc_p = linspace(0, 1200, cc_K + 1)';
cc_p = cc_p(1:end-1);
cc_w = ones(cc_K, 1);
cc_dens = buildExpTens(cc_p, cc_w, 12, 3, true, false, 1200, ...
    'verbose', false);

rng(42, 'twister');
cc_X = rand(2, 1000) * 1200;

% --- Full call: fast-path with internal n_q chunking ---
cc_vFull = evalExpTens(cc_dens, cc_X, 'method', 'centres', 'verbose', false);

% --- Manual split: two halves, each potentially chunked too ---
cc_n = size(cc_X, 2);
cc_mid = floor(cc_n / 2);
cc_v1 = evalExpTens(cc_dens, cc_X(:, 1:cc_mid), ...
    'method', 'centres', 'verbose', false);
cc_v2 = evalExpTens(cc_dens, cc_X(:, cc_mid + 1:end), ...
    'method', 'centres', 'verbose', false);
cc_vManual = [cc_v1, cc_v2];

% --- Assertions ---
results{end + 1, 1} = 'centres_chunking: completes without OOM (K=72, r=3, rel, n_q=1000)';
results{end, 2}     = numel(cc_vFull) == cc_n;

results{end + 1, 1} = 'centres_chunking: all outputs finite';
results{end, 2}     = all(isfinite(cc_vFull));

results{end + 1, 1} = 'centres_chunking: full call matches manual split (rel-tol 1e-13)';
results{end, 2}     = all(abs(cc_vFull - cc_vManual) <= 1e-13 * abs(cc_vFull));

% Note: the centres n_q chunking is exercised via method='centres' above,
% mirroring the Python chunking test (which forces centres throughout).
% Auto routing is deliberately not asserted here: the cost model routes
% this K=72 r=3 rel non-per shape to the factored Möbius path, so an auto
% call would not exercise the centres chunking path this test targets.
% Auto-vs-centres agreement is a dispatcher concern (test_dispatch_sm_eval,
% test_eval_orbit), not a chunking one.

clear cc_K cc_p cc_w cc_dens cc_X cc_vFull cc_n cc_mid cc_v1 cc_v2 ...
      cc_vManual

% Restore caller's pre-test defaults eagerly when run
% standalone (fires the helper's onCleanup destructor on
% script exit; guarded so we don't clear a like-named
% variable when this file was run from test_mpt.m, where
% the standalone branch was skipped).
if exist('cleanupDefaults_cc', 'var')
    clear cleanupDefaults_cc
end
