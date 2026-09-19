%% test_maet_core.m — buildMaet / evalMaet / simMaet — single-multiset core
%
%  Tests for single-multiset core.
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



dens = buildMaet([0, 4, 7], [], 0.5, 1, false, true, 12, ...
    'verbose', false);
vals = evalMaet(dens, 0:11, 'verbose', false);
peaks = find(vals > 0.5) - 1;
results{end+1,1} = 'buildMaet/evalMaet: peaks at 0, 4, 7';
results{end,2}   = isequal(peaks, [0, 4, 7]);

s = simMaet([0, 4, 7], [], [0, 4, 7], [], ...
    10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'simMaet: identical = 1';
results{end,2}   = abs(s - 1) < 1e-10;

s = simMaet( ...
    [0, 200, 400, 500, 700, 900, 1100], [], ...
    [0, 400, 700], [], ...
    10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'simMaet: 0 < s < 1';
results{end,2}   = s > 0 && s < 1;

dens = buildMaet([0, 4, 7], [], 0.5, 2, true, true, 12, ...
    'verbose', false);
results{end+1,1} = 'buildMaet: relative tensor dim = 1';
results{end,2}   = dens.dim == 1;

A = [0, 200, 400, 500, 700, 900, 1100;
     0, 200, 400, 500, 700, 900, 1100];
B = [0, 400, 700, NaN, NaN, NaN, NaN;
     0, 300, 700, NaN, NaN, NaN, NaN];
s = simMaet(A, [], B, [], 10, 1, false, true, 1200, ...
    'verbose', false);
results{end+1,1} = 'simMaet batched-raw: output length';
results{end,2}   = numel(s) == 2;
results{end+1,1} = 'simMaet batched-raw: no NaN';
results{end,2}   = all(~isnan(s));
results{end+1,1} = 'simMaet batched-raw: major > minor fit';
results{end,2}   = s(1) > s(2);

% --- v3 unified dispatch: list mode and batched-raw mode ----

% List mode (single-multiset): cell of density structs in, cell of values out
d1 = buildMaet([0, 4, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
d2 = buildMaet([0, 3, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
d3 = buildMaet([0, 4, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
sCell = simMaet({d1, d2}, {d3, d3}, 'verbose', false);
results{end+1,1} = 'simMaet list: returns cell of correct length';
results{end,2}   = iscell(sCell) && numel(sCell) == 2;
sScalar1 = simMaet(d1, d3, 'verbose', false);
sScalar2 = simMaet(d2, d3, 'verbose', false);
results{end+1,1} = 'simMaet list: matches scalar dispatch element-wise';
results{end,2}   = abs(sCell{1} - sScalar1) < 1e-14 ...
                   && abs(sCell{2} - sScalar2) < 1e-14;

% List mode: Option II shape rule (length-1 stays length-1)
sCell1 = simMaet({d1}, {d3}, 'verbose', false);
results{end+1,1} = 'simMaet list: length-1 returns length-1 cell (Option II)';
results{end,2}   = iscell(sCell1) && numel(sCell1) == 1 ...
                   && abs(sCell1{1} - sScalar1) < 1e-14;

% List mode: length mismatch errors
results{end+1,1} = 'simMaet list: length mismatch errors';
results{end,2}   = throwsErrorWithId( ...
    @() simMaet({d1, d2}, {d3}, 'verbose', false), ...
    'simMaet:listLengthMismatch');

% List mode: non-struct entry errors
results{end+1,1} = 'simMaet list: non-struct entry errors';
results{end,2}   = throwsErrorWithId( ...
    @() simMaet({d1, [1, 2, 3]}, {d3, d3}, 'verbose', false), ...
    'simMaet:listNonStruct');

% Batched-raw mode: 2-D matrix dispatch returns vector
A2 = [0, 200, 400, 500, 700, 900, 1100;
      0, 200, 400, 500, 700, 900, 1100];
B2 = [0, 400, 700, NaN, NaN, NaN, NaN;
      0, 300, 700, NaN, NaN, NaN, NaN];
sBatched = simMaet(A2, [], B2, [], 10, 1, false, true, 1200, ...
    'verbose', false);
results{end+1,1} = 'simMaet batched: returns vector of correct length';
results{end,2}   = isnumeric(sBatched) && numel(sBatched) == 2;

% Batched-raw mode: matches scalar dispatch row-by-row
sScalar1 = simMaet(A2(1, :), [], B2(1, ~isnan(B2(1, :))), [], ...
    10, 1, false, true, 1200, 'verbose', false);
sScalar2 = simMaet(A2(2, :), [], B2(2, ~isnan(B2(2, :))), [], ...
    10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'simMaet batched: matches scalar dispatch row-by-row';
results{end,2}   = abs(sBatched(1) - sScalar1) < 1e-12 ...
                   && abs(sBatched(2) - sScalar2) < 1e-12;

% Batched-raw mode: row mismatch errors
A3 = [0, 4, 7; 0, 3, 7; 0, 5, 9];   % 3 rows
results{end+1,1} = 'simMaet batched: row mismatch errors';
results{end,2}   = throwsErrorWithId( ...
    @() simMaet(A3, [], B2, [], 10, 1, false, true, 1200, ...
        'verbose', false), ...
    'simMaet:batchedRowMismatch');

% Row vector still uses scalar single-multiset raw path (backward compatibility)
% Despite being a 1-by-3 matrix, [0 4 7] is a vector and dispatches to
% the existing scalar form, returning a scalar.
sScalarFromRow = simMaet([0, 4, 7], [], [0, 4, 7], [], ...
    10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'simMaet batched: row vector falls through to scalar';
results{end,2}   = isnumeric(sScalarFromRow) && isscalar(sScalarFromRow) ...
                   && abs(sScalarFromRow - 1) < 1e-10;

% spectrum/precision/dedup forwarding (batched-raw only)
spec_fwd = {'harmonic', 12, 'powerlaw', 1};
sBatched_spec = simMaet(A2, [], B2, [], 10, 1, false, true, 1200, ...
    'spectrum', spec_fwd, 'verbose', false);
% 'spectrum' is batched-raw only, so the per-row reference enriches
% with addSpectra first and then compares.
[pA1, wA1] = addSpectra(A2(1, :), [], spec_fwd{:});
[pB1, wB1] = addSpectra(B2(1, ~isnan(B2(1, :))), [], spec_fwd{:});
sRow1_spec = simMaet(pA1, wA1, pB1, wB1, ...
    10, 1, false, true, 1200, 'verbose', false);
[pA2s, wA2s] = addSpectra(A2(2, :), [], spec_fwd{:});
[pB2s, wB2s] = addSpectra(B2(2, ~isnan(B2(2, :))), [], spec_fwd{:});
sRow2_spec = simMaet(pA2s, wA2s, pB2s, wB2s, ...
    10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'simMaet batched: ''spectrum'' matches per-row addSpectra';
results{end,2}   = abs(sBatched_spec(1) - sRow1_spec) < 1e-10 ...
                   && abs(sBatched_spec(2) - sRow2_spec) < 1e-10;

% spectrum kwarg rejected in non-batched modes
results{end+1,1} = 'simMaet scalar: ''spectrum'' kwarg errors';
results{end,2}   = throwsErrorWithId( ...
    @() simMaet([0, 4, 7], [], [0, 4, 7], [], 10, 1, false, true, 1200, ...
        'spectrum', spec_fwd, 'verbose', false), ...
    'simMaet:spectrumNotApplicable');

results{end+1,1} = 'simMaet MA struct: ''spectrum'' kwarg errors';
% Build small MA densities just for this test
densMA_x = buildMaet({[0; 4; 7]}, [], 0.5, 1, false, true, 12, 'verbose', false);
densMA_y = buildMaet({[0; 4; 7]}, [], 0.5, 1, false, true, 12, 'verbose', false);
results{end,2}   = throwsErrorWithId( ...
    @() simMaet(densMA_x, densMA_y, 'spectrum', spec_fwd, 'verbose', false), ...
    'simMaet:spectrumNotApplicable');

% --- Broadcasting in batched-raw mode (v3+) ---
% Reference multiset broadcast against M candidate rows: should match
% the explicit repmat formulation row-by-row.
ref_pitches = [0, 386.31, 701.96];
candidates = [0, 400, 700;
              0, 300, 700;
              0, 300, 600;
              0, 400, 800];
sims_explicit = simMaet(repmat(ref_pitches, 4, 1), [], candidates, [], ...
    10, 1, false, true, 1200, 'verbose', false);

% (a) 1×K row reference broadcast as P1
sims_bcast_row = simMaet(ref_pitches, [], candidates, [], ...
    10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'simMaet batched: 1xK row P1 broadcasts against MxK P2';
results{end,2}   = isequal(size(sims_bcast_row), [4, 1]) && ...
                   max(abs(sims_bcast_row - sims_explicit)) < 1e-12;

% (b) K-by-1 column reference broadcast as P1
sims_bcast_col = simMaet(ref_pitches.', [], candidates, [], ...
    10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'simMaet batched: Kx1 column P1 broadcasts against MxK P2';
results{end,2}   = max(abs(sims_bcast_col - sims_explicit)) < 1e-12;

% (c) Symmetric: P1 matrix, P2 vector reference
sims_bcast_p2 = simMaet(candidates, [], ref_pitches, [], ...
    10, 1, false, true, 1200, 'verbose', false);
% cos sim is symmetric in P1 vs P2 swap, so should equal sims_explicit
results{end+1,1} = 'simMaet batched: P2 vector broadcasts against MxK P1';
results{end,2}   = max(abs(sims_bcast_p2 - sims_explicit)) < 1e-12;

% (d) Broadcast with non-empty weights: W1 vector broadcast in lockstep
ref_w = [1.0, 0.8, 0.6];
sims_w_bcast = simMaet(ref_pitches, ref_w, candidates, [], ...
    10, 1, false, true, 1200, 'verbose', false);
sims_w_explicit = simMaet(repmat(ref_pitches, 4, 1), repmat(ref_w, 4, 1), ...
    candidates, [], 10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'simMaet batched: W1 vector broadcasts alongside P1';
results{end,2}   = max(abs(sims_w_bcast - sims_w_explicit)) < 1e-12;

% (e) Broadcast composes with 'spectrum' kwarg
sims_bcast_spec = simMaet(ref_pitches, [], candidates, [], ...
    10, 1, false, true, 1200, 'spectrum', spec_fwd, 'verbose', false);
sims_explicit_spec = simMaet(repmat(ref_pitches, 4, 1), [], candidates, [], ...
    10, 1, false, true, 1200, 'spectrum', spec_fwd, 'verbose', false);
results{end+1,1} = 'simMaet batched: broadcast composes with ''spectrum''';
results{end,2}   = max(abs(sims_bcast_spec - sims_explicit_spec)) < 1e-12;

% (f) Mismatched row counts (no broadcast possible) errors clearly
results{end+1,1} = 'simMaet batched: mismatched row counts errors';
results{end,2}   = throwsErrorWithId( ...
    @() simMaet(rand(4, 3), [], rand(5, 3), [], ...
        10, 1, false, true, 1200, 'verbose', false), ...
    'simMaet:batchedRowMismatch');

% --- List-mode broadcasting (v3+) ---
% Build a small population of density structs.
dRef = buildMaet([0, 4, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
dC1  = buildMaet([0, 4, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
dC2  = buildMaet([0, 3, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
dC3  = buildMaet([0, 3, 6], [], 0.5, 1, false, true, 12, 'verbose', false);

simExplicit = simMaet({dRef, dRef, dRef}, {dC1, dC2, dC3}, 'verbose', false);

% (a) Right-broadcast: scalar struct vs cell
simBcastR = simMaet(dRef, {dC1, dC2, dC3}, 'verbose', false);
results{end+1,1} = 'simMaet list: scalar vs cell broadcasts (right)';
results{end,2}   = iscell(simBcastR) && numel(simBcastR) == 3 && ...
    abs(simBcastR{1} - simExplicit{1}) < 1e-12 && ...
    abs(simBcastR{2} - simExplicit{2}) < 1e-12 && ...
    abs(simBcastR{3} - simExplicit{3}) < 1e-12;

% (b) Left-broadcast: cell vs scalar struct (symmetric: cosine is symmetric)
simBcastL = simMaet({dC1, dC2, dC3}, dRef, 'verbose', false);
results{end+1,1} = 'simMaet list: cell vs scalar broadcasts (left)';
results{end,2}   = iscell(simBcastL) && numel(simBcastL) == 3 && ...
    abs(simBcastL{1} - simExplicit{1}) < 1e-12 && ...
    abs(simBcastL{2} - simExplicit{2}) < 1e-12 && ...
    abs(simBcastL{3} - simExplicit{3}) < 1e-12;

% (c) Length-1 cell still returns length-1 cell (Option II preserved)
simBcastOne = simMaet(dRef, {dC1}, 'verbose', false);
results{end+1,1} = 'simMaet list: scalar vs length-1 cell returns length-1 cell';
results{end,2}   = iscell(simBcastOne) && numel(simBcastOne) == 1 && ...
    abs(simBcastOne{1} - simExplicit{1}) < 1e-12;

% (d) Cell + cell with mismatched length still errors clearly
results{end+1,1} = 'simMaet list: cell+cell length mismatch errors';
results{end,2}   = throwsErrorWithId( ...
    @() simMaet({dRef, dRef}, {dC1, dC2, dC3}, 'verbose', false), ...
    'simMaet:listLengthMismatch');

% (e) Cell + non-struct, non-cell (e.g. numeric) errors with bad-broadcast id
results{end+1,1} = 'simMaet list: cell vs non-struct other-arg errors';
results{end,2}   = throwsErrorWithId( ...
    @() simMaet({dC1, dC2}, 42, 'verbose', false), ...
    'simMaet:listBadBroadcast');

% --- v3 unified dispatch: evalMaet list and batched-raw modes ----

% List mode: cell of density structs returns cell of value vectors
de1 = buildMaet([0, 4, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
de2 = buildMaet([0, 3, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
xGrid = 0:11;
valsCell = evalMaet({de1, de2}, xGrid, 'verbose', false);
results{end+1,1} = 'evalMaet list: returns cell of correct length';
results{end,2}   = iscell(valsCell) && numel(valsCell) == 2;
vals1 = evalMaet(de1, xGrid, 'verbose', false);
vals2 = evalMaet(de2, xGrid, 'verbose', false);
results{end+1,1} = 'evalMaet list: matches scalar dispatch element-wise';
results{end,2}   = max(abs(valsCell{1}(:) - vals1(:))) < 1e-14 ...
                   && max(abs(valsCell{2}(:) - vals2(:))) < 1e-14;

% List mode: per-density X (cell of vectors of length matching density count)
xCell = {0:11, 0:23};
valsCellPerDens = evalMaet({de1, de2}, xCell, 'verbose', false);
vals1b = evalMaet(de1, xCell{1}, 'verbose', false);
vals2b = evalMaet(de2, xCell{2}, 'verbose', false);
results{end+1,1} = 'evalMaet list: per-density X (cell broadcast disambiguation)';
results{end,2}   = max(abs(valsCellPerDens{1}(:) - vals1b(:))) < 1e-14 ...
                   && max(abs(valsCellPerDens{2}(:) - vals2b(:))) < 1e-14;

% List mode: Option II (length-1 stays length-1)
valsCell1 = evalMaet({de1}, xGrid, 'verbose', false);
results{end+1,1} = 'evalMaet list: length-1 returns length-1 cell';
results{end,2}   = iscell(valsCell1) && numel(valsCell1) == 1;

% List mode: non-struct entry errors
results{end+1,1} = 'evalMaet list: non-struct entry errors';
results{end,2}   = throwsErrorWithId( ...
    @() evalMaet({de1, [1, 2, 3]}, xGrid, 'verbose', false), ...
    'evalMaet:listNonStruct');

% Batched-raw mode: 2-D matrix dispatch returns matrix of values
P_e = [0, 4, 7; 0, 3, 7];   % 2 x 3 matrix (major and minor triads)
xq = 0:11;
valsBatched = evalMaet(P_e, [], 0.5, 1, false, true, 12, xq, ...
    'verbose', false);
results{end+1,1} = 'evalMaet batched: returns matrix of correct shape';
results{end,2}   = isnumeric(valsBatched) && isequal(size(valsBatched), [2, 12]);

% Batched-raw matches scalar dispatch row-by-row
vals_row1 = evalMaet(P_e(1, :), [], 0.5, 1, false, true, 12, xq, ...
    'verbose', false);
vals_row2 = evalMaet(P_e(2, :), [], 0.5, 1, false, true, 12, xq, ...
    'verbose', false);
results{end+1,1} = 'evalMaet batched: matches scalar dispatch row-by-row';
results{end,2}   = max(abs(valsBatched(1, :) - vals_row1(:).')) < 1e-12 ...
                   && max(abs(valsBatched(2, :) - vals_row2(:).')) < 1e-12;

% Batched-raw: NaN-padded rows handled (consistent with batchCosSim convention)
P_e_nan = [0, 4, 7, NaN; 0, 3, 7, NaN];
valsNan = evalMaet(P_e_nan, [], 0.5, 1, false, true, 12, xq, ...
    'verbose', false);
results{end+1,1} = 'evalMaet batched: NaN-padded rows match unpadded rows';
results{end,2}   = max(abs(valsNan(:) - valsBatched(:))) < 1e-14;

% Row vector falls through to scalar single-multiset raw path (backward compatibility)
vals_row_compat = evalMaet([0, 4, 7], [], 0.5, 1, false, true, 12, xq, ...
    'verbose', false);
results{end+1,1} = 'evalMaet batched: row vector falls through to scalar';
results{end,2}   = isnumeric(vals_row_compat) && isvector(vals_row_compat);

% --- Transposition invariance (simMaet fix) ---

B_diat = [0, 200, 400, 500, 700, 900, 1100];

s0 = simMaet([0, 400, 700], [], B_diat, [], ...
    10, 2, true, false, 1200, 'verbose', false);
s1 = simMaet([100, 500, 800], [], B_diat, [], ...
    10, 2, true, false, 1200, 'verbose', false);
results{end+1,1} = 'simMaet: isRel transposition (non-periodic)';
results{end,2}   = abs(s0 - s1) < 1e-14;

s0 = simMaet([0, 400, 700], [], B_diat, [], ...
    10, 2, true, true, 1200, 'verbose', false);
s1 = simMaet([100, 500, 800], [], B_diat, [], ...
    10, 2, true, true, 1200, 'verbose', false);
results{end+1,1} = 'simMaet: isRel transposition (periodic)';
results{end,2}   = abs(s0 - s1) < 1e-14;

s0 = simMaet([0, 400, 700], [], B_diat, [], ...
    10, 3, true, true, 1200, 'verbose', false);
s1 = simMaet([500, 900, 1200], [], B_diat, [], ...
    10, 3, true, true, 1200, 'verbose', false);
results{end+1,1} = 'simMaet: isRel transposition (periodic, r=3)';
results{end,2}   = abs(s0 - s1) < 1e-14;

shifts = [100, 300, 500, 700, 1100];
s_ref = simMaet([0, 400, 700], [], B_diat, [], ...
    10, 2, true, true, 1200, 'verbose', false);
allMatch = true;
for c = shifts
    sc = simMaet([0, 400, 700] + c, [], B_diat, [], ...
        10, 2, true, true, 1200, 'verbose', false);
    if abs(sc - s_ref) >= 1e-14
        allMatch = false;
    end
end
results{end+1,1} = 'simMaet: isRel all shifts (periodic)';
results{end,2}   = allMatch;

s0 = simMaet([0, 400, 700], [], B_diat, [], ...
    10, 1, false, true, 1200, 'verbose', false);
s1 = simMaet([1200, 1600, 1900], [], B_diat, [], ...
    10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'simMaet: isPer octave equivalence';
results{end,2}   = abs(s0 - s1) < 1e-14;

w = [1.0, 0.8, 0.6];
s0 = simMaet([0, 400, 700], w, B_diat, [], ...
    10, 2, true, true, 1200, 'verbose', false);
s1 = simMaet([100, 500, 800], w, B_diat, [], ...
    10, 2, true, true, 1200, 'verbose', false);
results{end+1,1} = 'simMaet: isRel+isPer with weights';
results{end,2}   = abs(s0 - s1) < 1e-14;

A3 = [0, 400, 700; 1200, 1600, 1900; 0, 400, 700];
B3 = repmat(B_diat, 3, 1);
sb = simMaet(A3, [], B3, [], 10, 1, false, true, 1200, ...
    'verbose', false);
results{end+1,1} = 'simMaet batched-raw: octave deduplication';
results{end,2}   = abs(sb(1) - sb(2)) < 1e-14 && ...
                    abs(sb(1) - sb(3)) < 1e-14;


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
    fprintf('\n=== test_maet_core: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_maet_core:failed', '%d test(s) failed.', nFail);
    end
end
