%% test_nested_route_scale.m — the nested routes' scales, and a sweep across them
%
%  Partial mirror of the Python tests/test_nested_route_scale.py.
%
%  A nested attribute's inner product can be computed by the materialised
%  tuple centres or by a per-level contraction (a period grid when
%  relative-periodic), and the routes' bare matrices differ by constants
%  known in closed form (internal.nestedRouteScale). The nested combiners
%  of internal.nestedContract divide those constants out, so a sweep whose
%  cross term is priced on its own, and may take a route its self inner
%  products did not, still forms the right ratio. Before that, the
%  one-sided similarity of a three-chord query at inner r = 1 could come
%  out 39 times too small.
%
%  Pinned here: the closed-form factors, and the cadence-shaped sweep
%  against the per-offset similarity. The route split itself is forced
%  only in the Python test, which substitutes the planner.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults_nrs
    cleanupDefaults_nrs = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end
nrs_prevHints = mptDefaults('showHints');
mptDefaults('showHints', false);

% --- the closed-form factors -------------------------------------------
% centres / tau grid = (P / sigma) sqrt(s / (4 pi)) |G|: 39.09 for three
% one-pitch positions at sigma = 0.15, P = 12 -- the factor the cadence
% sweep was missing.
nrs_V = repmat([60; 64; 67], 3, 1);
nrs_spec = struct('tags', repelem(0:2, 3), 'r', [1 3], ...
                  'exch', [true false], 'rel', [0 1]);
nrs_d = buildMaet({nrs_V}, {[]}, 'specs', {nrs_spec}, 'sigma', 0.15, ...
                  'per', true, 'period', 12, 'verbose', false);
nrs_f = internal.nestedRouteScale(nrs_d, 1, 'taugrid') ...
      / internal.nestedRouteScale(nrs_d, 1, 'centres');
nrs_ok = abs(nrs_f - 12 / 0.15 * sqrt(3 / (4 * pi))) <= 1e-12 * nrs_f ...
      && abs(nrs_f - 39.0882) < 1e-4;
nrs_spec2 = struct('tags', repelem(0:2, 3), 'r', [2 3], ...
                   'exch', [true false], 'rel', [0 1]);
nrs_d2 = buildMaet({nrs_V}, {[]}, 'specs', {nrs_spec2}, 'sigma', 0.15, ...
                   'per', true, 'period', 12, 'verbose', false);
nrs_f2 = internal.nestedRouteScale(nrs_d2, 1, 'taugrid') ...
       / internal.nestedRouteScale(nrs_d2, 1, 'centres');
nrs_ok = nrs_ok && abs(nrs_f2 - 12 / 0.15 * sqrt(6 / (4 * pi)) * 8) ...
                   <= 1e-12 * nrs_f2;
% The contraction of an absolute attribute and Bulger's enumeration are
% both |G| times the centres scale.
nrs_ok = nrs_ok && abs(internal.nestedRouteScale(nrs_d2, 1, 'contract') ...
    / internal.nestedRouteScale(nrs_d2, 1, 'centres') - 8) <= 1e-12;
results{end+1, 1} = 'nested route scale: closed-form factors (39.09 at s = 3)';
results{end, 2} = nrs_ok;

% --- a cadence-shaped sweep against the per-offset similarity ------------
% Context: each event is a window of three beats, its chords padded to four
% pitches; nested pitch relative at the outer level and periodic, plus an
% absolute time attribute at the window's last beat. Query: I - V - I.
nrs_chords = {[60 64 67 72], [65 69 72], [62 67 71 65], [60 64 67], ...
              [57 60 64], [62 65 69], [55 59 62 65], [60 64 67 72]};
nrs_K = 4;
nrs_N = numel(nrs_chords) - 2;
nrs_Pc = NaN(3 * nrs_K, nrs_N);
for e = 1:nrs_N
    for j = 1:3
        v = nrs_chords{e + j - 1};
        nrs_Pc((j - 1) * nrs_K + (1:numel(v)), e) = v(:);
    end
end
nrs_Wc = double(~isnan(nrs_Pc));
nrs_tc = 2 + (0:nrs_N - 1);
nrs_Pq = [60; 64; 67; 62; 67; 71; 60; 64; 67];
nrs_pitchC = struct('tags', repelem(0:2, nrs_K), 'r', [1 3], ...
                    'exch', [true false], 'rel', [0 1]);
nrs_pitchQ = struct('tags', repelem(0:2, 3), 'r', [1 3], ...
                    'exch', [true false], 'rel', [0 1]);
nrs_time = struct('r', 1, 'exch', true, 'rel', false);
nrs_args = {'sigma', [0.15 0.1], 'per', [true false], 'period', [12 0], ...
            'verbose', false};
nrs_dc = buildMaet({nrs_Pc, nrs_tc}, {nrs_Wc, ones(1, nrs_N)}, ...
                   'specs', {nrs_pitchC, nrs_time}, nrs_args{:});
nrs_dq = buildMaet({nrs_Pq, 2}, {ones(9, 1), 1}, ...
                   'specs', {nrs_pitchQ, nrs_time}, nrs_args{:});
nrs_mus = nrs_tc - 2;
nrs_off = [zeros(1, nrs_N); nrs_mus];
nrs_ok = true;
for nrs_norm = {'oneSidedDenom', 'cosine'}
    nrs_s = sweepSimMaet(nrs_dc, nrs_dq, nrs_off, ...
                         'normalize', nrs_norm{1}, 'verbose', false);
    nrs_ref = zeros(1, nrs_N);
    for m = 1:nrs_N
        nrs_dqm = buildMaet({nrs_Pq, 2 + nrs_mus(m)}, {ones(9, 1), 1}, ...
                            'specs', {nrs_pitchQ, nrs_time}, nrs_args{:});
        nrs_ref(m) = simMaet(nrs_dc, nrs_dqm, 'normalize', nrs_norm{1}, ...
                             'verbose', false);
    end
    nrs_ok = nrs_ok && max(abs(nrs_s(:).' - nrs_ref)) ...
                       <= 1e-7 * max(abs(nrs_ref)) + 1e-12;
end
results{end+1, 1} = 'nested route scale: cadence sweep equals the per-offset similarity';
results{end, 2} = nrs_ok;

mptDefaults('showHints', nrs_prevHints);

if standalone
    nPass = sum([results{:, 2}]);
    nFail = size(results, 1) - nPass;
    for ii = 1:size(results, 1)
        if results{ii, 2}
            fprintf('  PASS  %s\n', results{ii, 1});
        else
            fprintf('  FAIL  %s\n', results{ii, 1});
        end
    end
    fprintf('\n=== test_nested_route_scale: %d passed, %d failed (of %d) ===\n\n', nPass, nFail, nPass + nFail);
    clear cleanupDefaults_nrs
    if nFail > 0
        error('test_nested_route_scale:failed', '%d test(s) failed.', nFail);
    end
end
