%% test_windowed_similarity_offset.m — windowedSimilarity cross-correlation semantics (offset API)
%
%  Tests for the windowed cross-correlation semantics in
%  windowedSimilarity. Mirror of Python's TestWindowedCrossCorrelation
%  class.
%
%  Standalone-runnable; appends to `results` when called from
%  test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end


% Mirror of the Python TestWindowedCrossCorrelation class. Verifies
% that at offset o the query's effective-space centroid is aligned
% with the window centre, so a peak at offset o means the query
% pattern is present in the context displaced by o from its centroid.

% Helper: build a pitch/time MAET from row vectors of (pitches, times).
% Pitch is absolute, non-periodic, r=1; time is absolute, non-periodic,
% r=1. sigmas: pitch 1.0, time 0.2.
mkPT = @(pvec, tvec) buildExpTens({pvec, tvec}, [], ...
    [1.0, 0.2], [1, 1], [], ...
    [false, false], [false, false], [0.0, 0.0], 'verbose', false);

% Helper: sweep a time offset with spec(size_time, mix_time).
function prof = cc_sweep(q, c, s, m, omin, omax, n)
    offs = linspace(omin, omax, n);
    offsets = zeros(2, n);
    offsets(2, :) = offs;   % pitch offset = 0 (pitch group unwindowed)
    spec = struct('size', [Inf, s], 'mix', [0, m]);
    prof = windowedSimilarity(q, c, spec, offsets, 'verbose', false);
end %#ok<DEFNU>

% 1. Peak at offset equal to single-event context time (mu_q = 0).
q  = mkPT(60, 0);
ct = mkPT(60, 5);
offs_cc = linspace(0, 10, 41);
prof1 = cc_sweep(q, ct, 2.0, 0.0, 0, 10, 41);
[~, ipk] = max(prof1);
results{end+1,1} = 'windowedSimilarity cross-corr: peak at single-event offset';
results{end,2}   = abs(offs_cc(ipk) - 5) < 0.3 && max(prof1) > 0.5;

% 2. Peak invariance across Gaussian window sizes.
szs = [1 2 4 8];
peak_off = zeros(size(szs));
for k = 1:numel(szs)
    pf = cc_sweep(q, ct, szs(k), 0.0, 0, 10, 41);
    [~, ip] = max(pf);
    peak_off(k) = offs_cc(ip);
end
results{end+1,1} = 'windowedSimilarity cross-corr: peak invariant over size (Gaussian)';
results{end,2}   = all(abs(peak_off - 5) < 0.3);

% 3. Peak invariance across (mix) range.
mixes = [0.0 0.25 0.5 0.75 1.0];
peak_off = zeros(size(mixes));
for k = 1:numel(mixes)
    pf = cc_sweep(q, ct, 2.0, mixes(k), 0, 10, 41);
    [~, ip] = max(pf);
    peak_off(k) = offs_cc(ip);
end
results{end+1,1} = 'windowedSimilarity cross-corr: peak invariant over mix';
results{end,2}   = all(abs(peak_off - 5) < 0.3);

% 4. Multi-event query peak at centroid offset.
% Query centroid at t=0.5; context motif centroid at t=5.5; offset 5.0.
q2 = mkPT([60 64], [0 1]);
c2 = mkPT([60 64], [5 6]);
offs_fine = linspace(0, 10, 101);
offsets_fine = zeros(2, 101);
offsets_fine(2, :) = offs_fine;
prof4 = windowedSimilarity(q2, c2, ...
    struct('size', [Inf 2], 'mix', [0 0]), ...
    offsets_fine, 'verbose', false);
[~, ipk] = max(prof4);
results{end+1,1} = 'windowedSimilarity cross-corr: multi-event peak at centroid offset';
results{end,2}   = abs(offs_fine(ipk) - 5.0) < 0.2;

% 5. Two recurrences of the motif produce two equal peaks.
% Motif centroids at 2.5 and 5.5; query centroid at 0.5; peaks at 2.0 and 5.0.
c2b = mkPT([60 64 60 64], [2 3 5 6]);
offs_finer = linspace(0, 10, 201);
offsets_finer = zeros(2, 201);
offsets_finer(2, :) = offs_finer;
prof5 = windowedSimilarity(q2, c2b, ...
    struct('size', [Inf 2], 'mix', [0 0]), ...
    offsets_finer, 'verbose', false);
lm = (prof5(2:end-1) > prof5(1:end-2)) & ...
     (prof5(2:end-1) > prof5(3:end))  & ...
     (prof5(2:end-1) > 0.5 * max(prof5));
pk_idx = find(lm) + 1;
results{end+1,1} = 'windowedSimilarity cross-corr: two motif recurrences -> two peaks';
results{end,2}   = (numel(pk_idx) == 2) ...
                   && all(abs(sort(offs_finer(pk_idx)) - [2.0 5.0]) < 0.15) ...
                   && abs(prof5(pk_idx(1)) - prof5(pk_idx(2))) < 1e-3;

% 6. isRel=true concurrent dyad: time peak invariant under pitch
%    translation of the context.
q_dy  = buildExpTens({[60; 64], 0}, [], ...
                     [1.0, 0.2], [2, 1], [], ...
                     [true, false], [false, false], [0, 0], ...
                     'verbose', false);
c_dy1 = buildExpTens({[60; 64], 5}, [], ...
                     [1.0, 0.2], [2, 1], [], ...
                     [true, false], [false, false], [0, 0], ...
                     'verbose', false);
c_dy2 = buildExpTens({[70; 74], 5}, [], ...
                     [1.0, 0.2], [2, 1], [], ...
                     [true, false], [false, false], [0, 0], ...
                     'verbose', false);
spec_dy = struct('size', [Inf 2], 'mix', [0 0]);
offs_dy = linspace(0, 10, 101);
% dim = (r_pitch - isRel_pitch) + (r_time - isRel_time)
%     = (2 - 1) + (1 - 0) = 2 rows.
offsets_dy = zeros(2, 101);
offsets_dy(2, :) = offs_dy;
p_d1 = windowedSimilarity(q_dy, c_dy1, spec_dy, offsets_dy, 'verbose', false);
p_d2 = windowedSimilarity(q_dy, c_dy2, spec_dy, offsets_dy, 'verbose', false);
[~, i1] = max(p_d1);
[~, i2] = max(p_d2);
results{end+1,1} = ['windowedSimilarity cross-corr: isRel dyad peak ' ...
                    'invariant under pitch translation'];
results{end,2}   = (i1 == i2) && abs(offs_dy(i1) - 5) < 0.3;

% 7. Unwindowed cosSimExpTens on identical densities == 1 (unwindowed
%    path untouched).
d_id = mkPT([60 64 67], [0 1 2]);
s_id = cosSimExpTens(d_id, d_id, 'verbose', false);
results{end+1,1} = 'windowedSimilarity cross-corr: unwindowed cos_sim identical == 1';
results{end,2}   = abs(s_id - 1) < 1e-10;

% 8. Unwindowed cos_sim on distinct MA densities stays in (0, 1).
d_a = mkPT([60 64 67], [0 1 2]);
d_b = mkPT([60 65 67], [0 1 2]);
s_ab = cosSimExpTens(d_a, d_b, 'verbose', false);
results{end+1,1} = 'windowedSimilarity cross-corr: unwindowed cos_sim distinct in (0,1)';
results{end,2}   = s_ab > 0 && s_ab < 1;

% 9. Pitch mismatch suppresses the matched-offset peak.
c_miss  = mkPT(72, 5);
c_match = mkPT(60, 5);
p_miss  = cc_sweep(q, c_miss, 2.0, 0.0, 0, 10, 41);
p_match = cc_sweep(q, c_match, 2.0, 0.0, 0, 10, 41);
results{end+1,1} = 'windowedSimilarity cross-corr: pitch mismatch suppresses peak';
results{end,2}   = max(p_miss) < 0.01 * max(p_match);

% 10. Peak height increases monotonically with window size and approaches
%     1 as size -> Inf.
szs_h = [1 2 8 100];
pk_h  = zeros(size(szs_h));
for k = 1:numel(szs_h)
    pf = cc_sweep(q, ct, szs_h(k), 0.0, 4, 6, 201);
    pk_h(k) = max(pf);
end
results{end+1,1} = 'windowedSimilarity cross-corr: peak height increases with size';
results{end,2}   = all(diff(pk_h) > 0) && pk_h(end) > 0.99;

% --- v2.1 unified dispatch: windowedSimilarity list mode ----

% Build a tiny pair of MaetDensity queries and contexts for list-mode
% testing. We reuse the shape from the earlier section: a single-event
% query against a 4-event context with periodic pitch + non-periodic
% time, sweeping along time only.
qList_a = buildExpTens({62, 0}, [], [0.5 0.1], [1 1], [], ...
    [false false], [true false], [1200 0], 'verbose', false);
qList_b = buildExpTens({64, 0}, [], [0.5 0.1], [1 1], [], ...
    [false false], [true false], [1200 0], 'verbose', false);
cList_a = buildExpTens({[60 62 64 65], [0 1 2 3]}, [], ...
    [0.5 0.1], [1 1], [], [false false], [true false], [1200 0], ...
    'verbose', false);
cList_b = buildExpTens({[60 64 67 71], [0 1 2 3]}, [], ...
    [0.5 0.1], [1 1], [], [false false], [true false], [1200 0], ...
    'verbose', false);
M_list = 9;
offs_list = zeros(2, M_list);
offs_list(2, :) = linspace(-0.5, 3.5, M_list);
spec_list = struct('size', [Inf, 0.3], 'mix', [0, 0]);

% Pairwise mode: equal-length lists give cell of profiles
prof_pair = windowedSimilarity({qList_a, qList_b}, {cList_a, cList_b}, ...
    spec_list, offs_list, 'verbose', false);
results{end+1,1} = 'windowedSimilarity list pairwise: returns 1-by-n cell';
results{end,2}   = iscell(prof_pair) && isequal(size(prof_pair), [1, 2]);

prof_a_scalar = windowedSimilarity(qList_a, cList_a, spec_list, offs_list, ...
    'verbose', false);
prof_b_scalar = windowedSimilarity(qList_b, cList_b, spec_list, offs_list, ...
    'verbose', false);
results{end+1,1} = 'windowedSimilarity list pairwise: matches scalar dispatch element-wise';
results{end,2}   = max(abs(prof_pair{1} - prof_a_scalar)) < 1e-12 ...
                   && max(abs(prof_pair{2} - prof_b_scalar)) < 1e-12;

% Cartesian mode: different-length lists or explicit 'mode' = 'cartesian'
prof_cart = windowedSimilarity({qList_a, qList_b}, {cList_a, cList_b}, ...
    spec_list, offs_list, 'verbose', false, 'mode', 'cartesian');
results{end+1,1} = 'windowedSimilarity list cartesian: returns nQ-by-nC cell';
results{end,2}   = iscell(prof_cart) && isequal(size(prof_cart), [2, 2]);

prof_ab_scalar = windowedSimilarity(qList_a, cList_b, spec_list, offs_list, ...
    'verbose', false);
results{end+1,1} = 'windowedSimilarity list cartesian: matches scalar dispatch (i,j)';
results{end,2}   = max(abs(prof_cart{1, 2} - prof_ab_scalar)) < 1e-12;

% Auto mode: equal lengths -> pairwise
prof_auto_p = windowedSimilarity({qList_a, qList_b}, {cList_a, cList_b}, ...
    spec_list, offs_list, 'verbose', false, 'mode', 'auto');
results{end+1,1} = 'windowedSimilarity list auto: equal lengths -> pairwise cell';
results{end,2}   = iscell(prof_auto_p) && isequal(size(prof_auto_p), [1, 2]);

% Auto mode: unequal lengths -> cartesian
prof_auto_c = windowedSimilarity({qList_a}, {cList_a, cList_b}, ...
    spec_list, offs_list, 'verbose', false, 'mode', 'auto');
results{end+1,1} = 'windowedSimilarity list auto: unequal lengths -> cartesian';
results{end,2}   = iscell(prof_auto_c) && isequal(size(prof_auto_c), [1, 2]);

% Pairwise with mismatched lengths errors
results{end+1,1} = 'windowedSimilarity list pairwise: length mismatch errors';
results{end,2}   = throwsErrorWithId( ...
    @() windowedSimilarity({qList_a, qList_b}, {cList_a}, spec_list, offs_list, ...
        'verbose', false, 'mode', 'bulger'), ...
    'windowedSimilarity:listLengthMismatch');

% Length-1 list returns length-1 cell (Option II)
prof_one = windowedSimilarity({qList_a}, {cList_a}, spec_list, offs_list, ...
    'verbose', false);
results{end+1,1} = 'windowedSimilarity list: length-1 returns length-1 cell';
results{end,2}   = iscell(prof_one) && isequal(size(prof_one), [1, 1]);

% Scalar query + list context (broadcast query)
prof_qScalar = windowedSimilarity(qList_a, {cList_a, cList_b}, ...
    spec_list, offs_list, 'verbose', false);
results{end+1,1} = 'windowedSimilarity list: scalar query + list context';
results{end,2}   = iscell(prof_qScalar) && numel(prof_qScalar) == 2;

% List query + scalar context (broadcast context)
prof_cScalar = windowedSimilarity({qList_a, qList_b}, cList_a, ...
    spec_list, offs_list, 'verbose', false);
results{end+1,1} = 'windowedSimilarity list: list query + scalar context';
results{end,2}   = iscell(prof_cScalar) && numel(prof_cScalar) == 2;

% Per-query reference (cell-of-cells form)
% qList_a has 2 attributes (pitch, time), so each per-query reference
% is a 1-by-2 cell of attribute-dimension vectors.
ref_per_a = {{[62], [0]}, {[64], [0]}};   % length-2 cell of length-2 cells
prof_perRef = windowedSimilarity({qList_a, qList_b}, {cList_a, cList_b}, ...
    spec_list, offs_list, 'verbose', false, ...
    'reference', ref_per_a, 'mode', 'bulger');
prof_a_perRef_scalar = windowedSimilarity(qList_a, cList_a, spec_list, offs_list, ...
    'verbose', false, 'reference', {[62], [0]});
results{end+1,1} = 'windowedSimilarity list: per-query reference forwarded correctly';
results{end,2}   = max(abs(prof_perRef{1} - prof_a_perRef_scalar)) < 1e-12;

% Shared reference (single length-A cell broadcast to all queries)
ref_shared = {[62], [0]};   % length-2 cell of vectors -> shared
prof_sharedRef = windowedSimilarity({qList_a, qList_b}, {cList_a, cList_b}, ...
    spec_list, offs_list, 'verbose', false, ...
    'reference', ref_shared, 'mode', 'bulger');
prof_b_sharedRef_scalar = windowedSimilarity(qList_b, cList_b, spec_list, offs_list, ...
    'verbose', false, 'reference', ref_shared);
results{end+1,1} = 'windowedSimilarity list: shared reference broadcast';
results{end,2}   = max(abs(prof_sharedRef{2} - prof_b_sharedRef_scalar)) < 1e-12;

% Bad mode errors
results{end+1,1} = 'windowedSimilarity: bad mode value errors';
results{end,2}   = throwsErrorWithId( ...
    @() windowedSimilarity(qList_a, cList_a, spec_list, offs_list, ...
        'verbose', false, 'mode', 'bogus'), ...
    'windowedSimilarity:badMode');



%% ---- Standalone summary ----

if standalone
    nPass = sum([results{:, 2}]);
    nFail = size(results, 1) - nPass;
    for i = 1:size(results, 1)
        if results{i, 2}
            fprintf('  PASS  %s\n', results{i, 1});
        else
            fprintf('  FAIL  %s\n', results{i, 1});
        end
    end
    fprintf('\n=== test_windowed_similarity_offset: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_windowed_similarity_offset:failed', '%d test(s) failed.', nFail);
    end
end


