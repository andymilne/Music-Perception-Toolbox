%% test_exp_tens.m — buildExpTens / evalExpTens / cosSimExpTens — single-multiset core
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



dens = buildExpTens([0, 4, 7], [], 0.5, 1, false, true, 12, ...
    'verbose', false);
vals = evalExpTens(dens, 0:11, 'verbose', false);
peaks = find(vals > 0.5) - 1;
results{end+1,1} = 'buildExpTens/evalExpTens: peaks at 0, 4, 7';
results{end,2}   = isequal(peaks, [0, 4, 7]);

s = cosSimExpTens([0, 4, 7], [], [0, 4, 7], [], ...
    10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens: identical = 1';
results{end,2}   = abs(s - 1) < 1e-10;

s = cosSimExpTens( ...
    [0, 200, 400, 500, 700, 900, 1100], [], ...
    [0, 400, 700], [], ...
    10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens: 0 < s < 1';
results{end,2}   = s > 0 && s < 1;

dens = buildExpTens([0, 4, 7], [], 0.5, 2, true, true, 12, ...
    'verbose', false);
results{end+1,1} = 'buildExpTens: relative tensor dim = 1';
results{end,2}   = dens.dim == 1;

A = [0, 200, 400, 500, 700, 900, 1100;
     0, 200, 400, 500, 700, 900, 1100];
B = [0, 400, 700, NaN, NaN, NaN, NaN;
     0, 300, 700, NaN, NaN, NaN, NaN];
s = batchCosSimExpTens(A, B, 10, 1, false, true, 1200, ...
    'verbose', false);
results{end+1,1} = 'batchCosSimExpTens: output length';
results{end,2}   = numel(s) == 2;
results{end+1,1} = 'batchCosSimExpTens: no NaN';
results{end,2}   = all(~isnan(s));
results{end+1,1} = 'batchCosSimExpTens: major > minor fit';
results{end,2}   = s(1) > s(2);

% --- v2.1 unified dispatch: list mode and batched-raw mode ----

% List mode (single-multiset): cell of density structs in, cell of values out
d1 = buildExpTens([0, 4, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
d2 = buildExpTens([0, 3, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
d3 = buildExpTens([0, 4, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
sCell = cosSimExpTens({d1, d2}, {d3, d3}, 'verbose', false);
results{end+1,1} = 'cosSimExpTens list: returns cell of correct length';
results{end,2}   = iscell(sCell) && numel(sCell) == 2;
sScalar1 = cosSimExpTens(d1, d3, 'verbose', false);
sScalar2 = cosSimExpTens(d2, d3, 'verbose', false);
results{end+1,1} = 'cosSimExpTens list: matches scalar dispatch element-wise';
results{end,2}   = abs(sCell{1} - sScalar1) < 1e-14 ...
                   && abs(sCell{2} - sScalar2) < 1e-14;

% List mode: Option II shape rule (length-1 stays length-1)
sCell1 = cosSimExpTens({d1}, {d3}, 'verbose', false);
results{end+1,1} = 'cosSimExpTens list: length-1 returns length-1 cell (Option II)';
results{end,2}   = iscell(sCell1) && numel(sCell1) == 1 ...
                   && abs(sCell1{1} - sScalar1) < 1e-14;

% List mode: length mismatch errors
results{end+1,1} = 'cosSimExpTens list: length mismatch errors';
results{end,2}   = throwsErrorWithId( ...
    @() cosSimExpTens({d1, d2}, {d3}, 'verbose', false), ...
    'cosSimExpTens:listLengthMismatch');

% List mode: non-struct entry errors
results{end+1,1} = 'cosSimExpTens list: non-struct entry errors';
results{end,2}   = throwsErrorWithId( ...
    @() cosSimExpTens({d1, [1, 2, 3]}, {d3, d3}, 'verbose', false), ...
    'cosSimExpTens:listNonStruct');

% Batched-raw mode: 2-D matrix dispatch returns vector
A2 = [0, 200, 400, 500, 700, 900, 1100;
      0, 200, 400, 500, 700, 900, 1100];
B2 = [0, 400, 700, NaN, NaN, NaN, NaN;
      0, 300, 700, NaN, NaN, NaN, NaN];
sBatched = cosSimExpTens(A2, [], B2, [], 10, 1, false, true, 1200, ...
    'verbose', false);
results{end+1,1} = 'cosSimExpTens batched: returns vector of correct length';
results{end,2}   = isnumeric(sBatched) && numel(sBatched) == 2;

% Batched-raw mode: numerically equivalent to batchCosSimExpTens
% (suppress the v2.1 deprecation warning while we make the comparison)
warnState = warning('off', 'batchCosSimExpTens:deprecated');
sBatchOld = batchCosSimExpTens(A2, B2, 10, 1, false, true, 1200, ...
    'verbose', false);
warning(warnState);
results{end+1,1} = 'cosSimExpTens batched: matches batchCosSimExpTens exactly';
results{end,2}   = max(abs(sBatched(:) - sBatchOld(:))) < 1e-14;

% Batched-raw mode: matches scalar dispatch row-by-row
sScalar1 = cosSimExpTens(A2(1, :), [], B2(1, ~isnan(B2(1, :))), [], ...
    10, 1, false, true, 1200, 'verbose', false);
sScalar2 = cosSimExpTens(A2(2, :), [], B2(2, ~isnan(B2(2, :))), [], ...
    10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens batched: matches scalar dispatch row-by-row';
results{end,2}   = abs(sBatched(1) - sScalar1) < 1e-12 ...
                   && abs(sBatched(2) - sScalar2) < 1e-12;

% Batched-raw mode: row mismatch errors
A3 = [0, 4, 7; 0, 3, 7; 0, 5, 9];   % 3 rows
results{end+1,1} = 'cosSimExpTens batched: row mismatch errors';
results{end,2}   = throwsErrorWithId( ...
    @() cosSimExpTens(A3, [], B2, [], 10, 1, false, true, 1200, ...
        'verbose', false), ...
    'cosSimExpTens:batchedRowMismatch');

% Row vector still uses scalar single-multiset raw path (backward compatibility)
% Despite being a 1-by-3 matrix, [0 4 7] is a vector and dispatches to
% the existing scalar form, returning a scalar.
sScalarFromRow = cosSimExpTens([0, 4, 7], [], [0, 4, 7], [], ...
    10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens batched: row vector falls through to scalar';
results{end,2}   = isnumeric(sScalarFromRow) && isscalar(sScalarFromRow) ...
                   && abs(sScalarFromRow - 1) < 1e-10;

% Direct batchCosSimExpTens call now emits a deprecation warning
prevWarnState = warning('on', 'batchCosSimExpTens:deprecated');
lastwarn('', '');  % reset lastwarn so we capture only this call's warning
sDummy = batchCosSimExpTens(A2, B2, 10, 1, false, true, 1200, ...
    'verbose', false); %#ok<NASGU>
[~, lastWarnId] = lastwarn;
warning(prevWarnState);
results{end+1,1} = 'batchCosSimExpTens: emits batchCosSimExpTens:deprecated warning';
results{end,2}   = strcmp(lastWarnId, 'batchCosSimExpTens:deprecated');

% cosSimExpTens batched-raw delegation does NOT re-emit the warning
prevWarnState = warning('on', 'batchCosSimExpTens:deprecated');
lastwarn('', '');
sDummy = cosSimExpTens(A2, [], B2, [], 10, 1, false, true, 1200, ...
    'verbose', false); %#ok<NASGU>
[~, internalWarnId] = lastwarn;
warning(prevWarnState);
results{end+1,1} = 'cosSimExpTens batched: internal call suppresses deprecation';
results{end,2}   = ~strcmp(internalWarnId, 'batchCosSimExpTens:deprecated');

% spectrum/precision/dedup forwarding (batched-raw only)
spec_fwd = {'harmonic', 12, 'powerlaw', 1};
sBatched_spec = cosSimExpTens(A2, [], B2, [], 10, 1, false, true, 1200, ...
    'spectrum', spec_fwd, 'verbose', false);
warnState = warning('off', 'batchCosSimExpTens:deprecated');
sBatchOld_spec = batchCosSimExpTens(A2, B2, 10, 1, false, true, 1200, ...
    'spectrum', spec_fwd, 'verbose', false);
warning(warnState);
results{end+1,1} = 'cosSimExpTens batched: ''spectrum'' forwards to batchCosSimExpTens';
results{end,2}   = max(abs(sBatched_spec(:) - sBatchOld_spec(:))) < 1e-14;

% spectrum kwarg rejected in non-batched modes
results{end+1,1} = 'cosSimExpTens scalar: ''spectrum'' kwarg errors';
results{end,2}   = throwsErrorWithId( ...
    @() cosSimExpTens([0, 4, 7], [], [0, 4, 7], [], 10, 1, false, true, 1200, ...
        'spectrum', spec_fwd, 'verbose', false), ...
    'cosSimExpTens:spectrumNotApplicable');

results{end+1,1} = 'cosSimExpTens MA struct: ''spectrum'' kwarg errors';
% Build small MA densities just for this test
densMA_x = buildExpTens({[0; 4; 7]}, [], 0.5, 1, false, true, 12, 'verbose', false);
densMA_y = buildExpTens({[0; 4; 7]}, [], 0.5, 1, false, true, 12, 'verbose', false);
results{end,2}   = throwsErrorWithId( ...
    @() cosSimExpTens(densMA_x, densMA_y, 'spectrum', spec_fwd, 'verbose', false), ...
    'cosSimExpTens:spectrumNotApplicable');

% --- Broadcasting in batched-raw mode (v2.1.1+) ---
% Reference multiset broadcast against M candidate rows: should match
% the explicit repmat formulation row-by-row.
ref_pitches = [0, 386.31, 701.96];
candidates = [0, 400, 700;
              0, 300, 700;
              0, 300, 600;
              0, 400, 800];
sims_explicit = cosSimExpTens(repmat(ref_pitches, 4, 1), [], candidates, [], ...
    10, 1, false, true, 1200, 'verbose', false);

% (a) 1×K row reference broadcast as P1
sims_bcast_row = cosSimExpTens(ref_pitches, [], candidates, [], ...
    10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens batched: 1xK row P1 broadcasts against MxK P2';
results{end,2}   = isequal(size(sims_bcast_row), [4, 1]) && ...
                   max(abs(sims_bcast_row - sims_explicit)) < 1e-12;

% (b) K-by-1 column reference broadcast as P1
sims_bcast_col = cosSimExpTens(ref_pitches.', [], candidates, [], ...
    10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens batched: Kx1 column P1 broadcasts against MxK P2';
results{end,2}   = max(abs(sims_bcast_col - sims_explicit)) < 1e-12;

% (c) Symmetric: P1 matrix, P2 vector reference
sims_bcast_p2 = cosSimExpTens(candidates, [], ref_pitches, [], ...
    10, 1, false, true, 1200, 'verbose', false);
% cos sim is symmetric in P1 vs P2 swap, so should equal sims_explicit
results{end+1,1} = 'cosSimExpTens batched: P2 vector broadcasts against MxK P1';
results{end,2}   = max(abs(sims_bcast_p2 - sims_explicit)) < 1e-12;

% (d) Broadcast with non-empty weights: W1 vector broadcast in lockstep
ref_w = [1.0, 0.8, 0.6];
sims_w_bcast = cosSimExpTens(ref_pitches, ref_w, candidates, [], ...
    10, 1, false, true, 1200, 'verbose', false);
sims_w_explicit = cosSimExpTens(repmat(ref_pitches, 4, 1), repmat(ref_w, 4, 1), ...
    candidates, [], 10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens batched: W1 vector broadcasts alongside P1';
results{end,2}   = max(abs(sims_w_bcast - sims_w_explicit)) < 1e-12;

% (e) Broadcast composes with 'spectrum' kwarg
sims_bcast_spec = cosSimExpTens(ref_pitches, [], candidates, [], ...
    10, 1, false, true, 1200, 'spectrum', spec_fwd, 'verbose', false);
sims_explicit_spec = cosSimExpTens(repmat(ref_pitches, 4, 1), [], candidates, [], ...
    10, 1, false, true, 1200, 'spectrum', spec_fwd, 'verbose', false);
results{end+1,1} = 'cosSimExpTens batched: broadcast composes with ''spectrum''';
results{end,2}   = max(abs(sims_bcast_spec - sims_explicit_spec)) < 1e-12;

% (f) Mismatched row counts (no broadcast possible) errors clearly
results{end+1,1} = 'cosSimExpTens batched: mismatched row counts errors';
results{end,2}   = throwsErrorWithId( ...
    @() cosSimExpTens(rand(4, 3), [], rand(5, 3), [], ...
        10, 1, false, true, 1200, 'verbose', false), ...
    'cosSimExpTens:batchedRowMismatch');

% --- List-mode broadcasting (v2.1.1+) ---
% Build a small population of density structs.
dRef = buildExpTens([0, 4, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
dC1  = buildExpTens([0, 4, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
dC2  = buildExpTens([0, 3, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
dC3  = buildExpTens([0, 3, 6], [], 0.5, 1, false, true, 12, 'verbose', false);

simExplicit = cosSimExpTens({dRef, dRef, dRef}, {dC1, dC2, dC3}, 'verbose', false);

% (a) Right-broadcast: scalar struct vs cell
simBcastR = cosSimExpTens(dRef, {dC1, dC2, dC3}, 'verbose', false);
results{end+1,1} = 'cosSimExpTens list: scalar vs cell broadcasts (right)';
results{end,2}   = iscell(simBcastR) && numel(simBcastR) == 3 && ...
    abs(simBcastR{1} - simExplicit{1}) < 1e-12 && ...
    abs(simBcastR{2} - simExplicit{2}) < 1e-12 && ...
    abs(simBcastR{3} - simExplicit{3}) < 1e-12;

% (b) Left-broadcast: cell vs scalar struct (symmetric: cosine is symmetric)
simBcastL = cosSimExpTens({dC1, dC2, dC3}, dRef, 'verbose', false);
results{end+1,1} = 'cosSimExpTens list: cell vs scalar broadcasts (left)';
results{end,2}   = iscell(simBcastL) && numel(simBcastL) == 3 && ...
    abs(simBcastL{1} - simExplicit{1}) < 1e-12 && ...
    abs(simBcastL{2} - simExplicit{2}) < 1e-12 && ...
    abs(simBcastL{3} - simExplicit{3}) < 1e-12;

% (c) Length-1 cell still returns length-1 cell (Option II preserved)
simBcastOne = cosSimExpTens(dRef, {dC1}, 'verbose', false);
results{end+1,1} = 'cosSimExpTens list: scalar vs length-1 cell returns length-1 cell';
results{end,2}   = iscell(simBcastOne) && numel(simBcastOne) == 1 && ...
    abs(simBcastOne{1} - simExplicit{1}) < 1e-12;

% (d) Cell + cell with mismatched length still errors clearly
results{end+1,1} = 'cosSimExpTens list: cell+cell length mismatch errors';
results{end,2}   = throwsErrorWithId( ...
    @() cosSimExpTens({dRef, dRef}, {dC1, dC2, dC3}, 'verbose', false), ...
    'cosSimExpTens:listLengthMismatch');

% (e) Cell + non-struct, non-cell (e.g. numeric) errors with bad-broadcast id
results{end+1,1} = 'cosSimExpTens list: cell vs non-struct other-arg errors';
results{end,2}   = throwsErrorWithId( ...
    @() cosSimExpTens({dC1, dC2}, 42, 'verbose', false), ...
    'cosSimExpTens:listBadBroadcast');

% --- v2.1 unified dispatch: evalExpTens list and batched-raw modes ----

% List mode: cell of density structs returns cell of value vectors
de1 = buildExpTens([0, 4, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
de2 = buildExpTens([0, 3, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
xGrid = 0:11;
valsCell = evalExpTens({de1, de2}, xGrid, 'verbose', false);
results{end+1,1} = 'evalExpTens list: returns cell of correct length';
results{end,2}   = iscell(valsCell) && numel(valsCell) == 2;
vals1 = evalExpTens(de1, xGrid, 'verbose', false);
vals2 = evalExpTens(de2, xGrid, 'verbose', false);
results{end+1,1} = 'evalExpTens list: matches scalar dispatch element-wise';
results{end,2}   = max(abs(valsCell{1}(:) - vals1(:))) < 1e-14 ...
                   && max(abs(valsCell{2}(:) - vals2(:))) < 1e-14;

% List mode: per-density X (cell of vectors of length matching density count)
xCell = {0:11, 0:23};
valsCellPerDens = evalExpTens({de1, de2}, xCell, 'verbose', false);
vals1b = evalExpTens(de1, xCell{1}, 'verbose', false);
vals2b = evalExpTens(de2, xCell{2}, 'verbose', false);
results{end+1,1} = 'evalExpTens list: per-density X (cell broadcast disambiguation)';
results{end,2}   = max(abs(valsCellPerDens{1}(:) - vals1b(:))) < 1e-14 ...
                   && max(abs(valsCellPerDens{2}(:) - vals2b(:))) < 1e-14;

% List mode: Option II (length-1 stays length-1)
valsCell1 = evalExpTens({de1}, xGrid, 'verbose', false);
results{end+1,1} = 'evalExpTens list: length-1 returns length-1 cell';
results{end,2}   = iscell(valsCell1) && numel(valsCell1) == 1;

% List mode: non-struct entry errors
results{end+1,1} = 'evalExpTens list: non-struct entry errors';
results{end,2}   = throwsErrorWithId( ...
    @() evalExpTens({de1, [1, 2, 3]}, xGrid, 'verbose', false), ...
    'evalExpTens:listNonStruct');

% Batched-raw mode: 2-D matrix dispatch returns matrix of values
P_e = [0, 4, 7; 0, 3, 7];   % 2 x 3 matrix (major and minor triads)
xq = 0:11;
valsBatched = evalExpTens(P_e, [], 0.5, 1, false, true, 12, xq, ...
    'verbose', false);
results{end+1,1} = 'evalExpTens batched: returns matrix of correct shape';
results{end,2}   = isnumeric(valsBatched) && isequal(size(valsBatched), [2, 12]);

% Batched-raw matches scalar dispatch row-by-row
vals_row1 = evalExpTens(P_e(1, :), [], 0.5, 1, false, true, 12, xq, ...
    'verbose', false);
vals_row2 = evalExpTens(P_e(2, :), [], 0.5, 1, false, true, 12, xq, ...
    'verbose', false);
results{end+1,1} = 'evalExpTens batched: matches scalar dispatch row-by-row';
results{end,2}   = max(abs(valsBatched(1, :) - vals_row1(:).')) < 1e-12 ...
                   && max(abs(valsBatched(2, :) - vals_row2(:).')) < 1e-12;

% Batched-raw: NaN-padded rows handled (consistent with batchCosSim convention)
P_e_nan = [0, 4, 7, NaN; 0, 3, 7, NaN];
valsNan = evalExpTens(P_e_nan, [], 0.5, 1, false, true, 12, xq, ...
    'verbose', false);
results{end+1,1} = 'evalExpTens batched: NaN-padded rows match unpadded rows';
results{end,2}   = max(abs(valsNan(:) - valsBatched(:))) < 1e-14;

% Row vector falls through to scalar single-multiset raw path (backward compatibility)
vals_row_compat = evalExpTens([0, 4, 7], [], 0.5, 1, false, true, 12, xq, ...
    'verbose', false);
results{end+1,1} = 'evalExpTens batched: row vector falls through to scalar';
results{end,2}   = isnumeric(vals_row_compat) && isvector(vals_row_compat);

% --- Transposition invariance (cosSimExpTens fix) ---

B_diat = [0, 200, 400, 500, 700, 900, 1100];

s0 = cosSimExpTens([0, 400, 700], [], B_diat, [], ...
    10, 2, true, false, 1200, 'verbose', false);
s1 = cosSimExpTens([100, 500, 800], [], B_diat, [], ...
    10, 2, true, false, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens: isRel transposition (non-periodic)';
results{end,2}   = abs(s0 - s1) < 1e-14;

s0 = cosSimExpTens([0, 400, 700], [], B_diat, [], ...
    10, 2, true, true, 1200, 'verbose', false);
s1 = cosSimExpTens([100, 500, 800], [], B_diat, [], ...
    10, 2, true, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens: isRel transposition (periodic)';
results{end,2}   = abs(s0 - s1) < 1e-14;

s0 = cosSimExpTens([0, 400, 700], [], B_diat, [], ...
    10, 3, true, true, 1200, 'verbose', false);
s1 = cosSimExpTens([500, 900, 1200], [], B_diat, [], ...
    10, 3, true, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens: isRel transposition (periodic, r=3)';
results{end,2}   = abs(s0 - s1) < 1e-14;

shifts = [100, 300, 500, 700, 1100];
s_ref = cosSimExpTens([0, 400, 700], [], B_diat, [], ...
    10, 2, true, true, 1200, 'verbose', false);
allMatch = true;
for c = shifts
    sc = cosSimExpTens([0, 400, 700] + c, [], B_diat, [], ...
        10, 2, true, true, 1200, 'verbose', false);
    if abs(sc - s_ref) >= 1e-14
        allMatch = false;
    end
end
results{end+1,1} = 'cosSimExpTens: isRel all shifts (periodic)';
results{end,2}   = allMatch;

s0 = cosSimExpTens([0, 400, 700], [], B_diat, [], ...
    10, 1, false, true, 1200, 'verbose', false);
s1 = cosSimExpTens([1200, 1600, 1900], [], B_diat, [], ...
    10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens: isPer octave equivalence';
results{end,2}   = abs(s0 - s1) < 1e-14;

w = [1.0, 0.8, 0.6];
s0 = cosSimExpTens([0, 400, 700], w, B_diat, [], ...
    10, 2, true, true, 1200, 'verbose', false);
s1 = cosSimExpTens([100, 500, 800], w, B_diat, [], ...
    10, 2, true, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens: isRel+isPer with weights';
results{end,2}   = abs(s0 - s1) < 1e-14;

A3 = [0, 400, 700; 1200, 1600, 1900; 0, 400, 700];
B3 = repmat(B_diat, 3, 1);
sb = batchCosSimExpTens(A3, B3, 10, 1, false, true, 1200, ...
    'verbose', false);
results{end+1,1} = 'batchCosSimExpTens: octave deduplication';
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
    fprintf('\n=== test_exp_tens: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_exp_tens:failed', '%d test(s) failed.', nFail);
    end
end
