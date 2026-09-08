%% test_maet.m — Multi-Attribute Expectation Tensor (MAET, v3)
%
%  Tests for Multi-Attribute Expectation Tensor (MAET, v3).
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


% Tests the multi-attribute path of buildExpTens. The single multiset path is covered
% by the Expectation tensors section above; these tests focus on
% MAET-specific behaviours: single multiset-equivalence under the degenerate
% (N=1, A=1) mapping, per-attribute enumeration, weight broadcasting,
% group canonicalisation, NaN padding, and the new error paths.

% -- single multiset-equivalence: MA with (N=1, A=1, K x 1 column w) reproduces single multiset --

p_sm = [0; 400; 700];
w_sm = [1; 0.7; 0.5];
sigma = 10; r_ = 2; isPer_ = true; period_ = 1200;

% --- Lazy/eager parity (v3) ---
%
% buildExpTens defaults to skinny (lazy=true); internal.ensureExpTensExpensive
% populates the per-tuple fields on demand. The eager and ensured-lazy
% paths must produce structurally identical structs.

dens_eager_sm  = buildExpTens(p_sm, w_sm, sigma, r_, true, isPer_, period_, ...
    'lazy', false, 'verbose', false);
dens_skinny_sm = buildExpTens(p_sm, w_sm, sigma, r_, true, isPer_, period_, ...
    'verbose', false);
results{end+1,1} = 'lazy: single multiset skinny default has only cheap fields';
results{end,2}   = ~isfield(dens_skinny_sm, 'Centres') ...
                && ~isfield(dens_skinny_sm, 'U_perm') ...
                && ~isfield(dens_skinny_sm, 'nJ');
results{end+1,1} = 'lazy: single multiset skinny exposes dim';
results{end,2}   = isfield(dens_skinny_sm, 'dim') ...
                && dens_skinny_sm.dim == dens_eager_sm.dim;
dens_filled_sm = internal.ensureExpTensExpensive(dens_skinny_sm);
results{end+1,1} = 'lazy: single multiset ensure -> matches eager Centres';
results{end,2}   = isequal(dens_filled_sm.Centres, dens_eager_sm.Centres);
results{end+1,1} = 'lazy: single multiset ensure -> matches eager wJ';
results{end,2}   = isequal(dens_filled_sm.wJ, dens_eager_sm.wJ);
results{end+1,1} = 'lazy: single multiset ensure -> matches eager U_perm';
results{end,2}   = isequal(dens_filled_sm.U_perm, dens_eager_sm.U_perm);
results{end+1,1} = 'lazy: single multiset ensure idempotent';
dens_twice_sm = internal.ensureExpTensExpensive(dens_filled_sm);
results{end,2}   = isequal(dens_twice_sm, dens_filled_sm);

dens_eager_ma  = buildExpTens({p_sm}, {w_sm}, sigma, r_, true, isPer_, period_, ...
    'lazy', false, 'verbose', false);
dens_skinny_ma = buildExpTens({p_sm}, {w_sm}, sigma, r_, true, isPer_, period_, ...
    'verbose', false);
results{end+1,1} = 'lazy: MA skinny default has only cheap fields';
results{end,2}   = ~isfield(dens_skinny_ma, 'Centres') ...
                && ~isfield(dens_skinny_ma, 'U_perm') ...
                && ~isfield(dens_skinny_ma, 'nJ');
results{end+1,1} = 'lazy: MA skinny exposes dim and dimPerAttr';
results{end,2}   = isfield(dens_skinny_ma, 'dim') ...
                && isfield(dens_skinny_ma, 'dimPerAttr') ...
                && isequal(dens_skinny_ma.dim, dens_eager_ma.dim) ...
                && isequal(dens_skinny_ma.dimPerAttr, dens_eager_ma.dimPerAttr);
dens_filled_ma = internal.ensureExpTensExpensive(dens_skinny_ma);
results{end+1,1} = 'lazy: MA ensure -> matches eager Centres';
results{end,2}   = isequal(dens_filled_ma.Centres, dens_eager_ma.Centres);
results{end+1,1} = 'lazy: MA ensure -> matches eager wJ and wv_comb';
results{end,2}   = isequal(dens_filled_ma.wJ, dens_eager_ma.wJ) ...
                && isequal(dens_filled_ma.wv_comb, dens_eager_ma.wv_comb);
results{end+1,1} = 'lazy: MA ensure idempotent';
dens_twice_ma = internal.ensureExpTensExpensive(dens_filled_ma);
results{end,2}   = isequal(dens_twice_ma, dens_filled_ma);

% Consumers transparently handle skinny input (cosSimExpTens, evalExpTens).
results{end+1,1} = 'lazy: cosSimExpTens accepts skinny dens (single multiset self-similarity = 1)';
s_self = cosSimExpTens(dens_skinny_sm, dens_skinny_sm, 'verbose', false);
results{end,2}   = abs(s_self - 1) < 1e-12;
results{end+1,1} = 'lazy: evalExpTens accepts skinny dens';
v_skinny = evalExpTens(dens_skinny_sm, [0 100 350], 'verbose', false);
v_eager  = evalExpTens(dens_eager_sm,  [0 100 350], 'verbose', false);
results{end,2}   = max(abs(v_skinny - v_eager)) < 1e-12;

for isRel_ = [false, true]
    dens_sm = buildExpTens(p_sm, w_sm, sigma, r_, isRel_, isPer_, period_, ...
        'lazy', false, 'verbose', false);
    dens_ma = buildExpTens({p_sm}, {w_sm}, sigma, r_, isRel_, isPer_, period_, ...
        'lazy', false, 'verbose', false);

    relTag = sprintf(' (isRel=%d)', isRel_);
    results{end+1,1} = ['MAET: single multiset-equivalence tag' relTag];
    results{end,2}   = strcmp(dens_ma.tag, 'MaetDensity'); %#ok<*SAGROW>

    results{end+1,1} = ['MAET: single multiset-equivalence nJ' relTag];
    results{end,2}   = dens_ma.nJ == dens_sm.nJ;

    results{end+1,1} = ['MAET: single multiset-equivalence U_perm' relTag];
    results{end,2}   = isequal(dens_ma.U_perm{1}, dens_sm.U_perm{1});

    results{end+1,1} = ['MAET: single multiset-equivalence V_comb' relTag];
    results{end,2}   = isequal(dens_ma.V_comb{1}, dens_sm.V_comb{1});

    results{end+1,1} = ['MAET: single multiset-equivalence Centres' relTag];
    results{end,2}   = isequal(dens_ma.Centres{1}, dens_sm.Centres{1});

    results{end+1,1} = ['MAET: single multiset-equivalence wJ' relTag];
    results{end,2}   = max(abs(dens_ma.wJ - dens_sm.wJ)) < 1e-12;

    results{end+1,1} = ['MAET: single multiset-equivalence wv_comb' relTag];
    results{end,2}   = max(abs(dens_ma.wv_comb - dens_sm.wv_comb)) < 1e-12;
end

% Dimensionality reduction under isRel=true
results{end+1,1} = 'MAET: Centres dim reduction (isRel=true, r=2)';
dens_ma = buildExpTens({p_sm}, {w_sm}, sigma, 2, true, true, 1200, ...
    'lazy', false, 'verbose', false);
results{end,2}   = isequal(size(dens_ma.Centres{1}), [1, dens_ma.nJ]);

% -- Single-multiset collapse: r = 1, N > 1 pools to one multiset at build --
% A single flat attribute read at r = 1 is one pooled multiset: a tuple is
% a lone value, so the two events {0,4} and {7,10} pool to {0,4,7,10}. The
% build collapses this to the canonical N = 1 form, identical to the
% directly-pooled N = 1 density. r = 2 keeps its event structure (N = 2).
d_r1N2 = buildExpTens({[0 7; 4 10]}, [], 10, 1, false, false, 0, ...
    'verbose', false);
d_pool = buildExpTens([0 4 7 10], [], 10, 1, false, false, 0, ...
    'verbose', false);
d_r2N2 = buildExpTens({[0 7; 4 10]}, [], 10, 2, false, false, 0, ...
    'verbose', false);
results{end+1,1} = 'single-multiset: r=1, N>1 collapses to N=1';
results{end,2}   = d_r1N2.N == 1 && internal.isSingleMultiset(d_r1N2) ...
                && d_r2N2.N == 2 && ~internal.isSingleMultiset(d_r2N2);
Xq = [-10 0 5 12 20];
results{end+1,1} = 'single-multiset: r=1, N>1 eval == pooled';
results{end,2}   = max(abs(evalExpTens(d_r1N2, Xq, 'verbose', false) ...
                       - evalExpTens(d_pool, Xq, 'verbose', false))) < 1e-12;
results{end+1,1} = 'single-multiset: r=1, N>1 cosine with pooled == 1';
results{end,2}   = abs(cosSimExpTens(d_r1N2, d_pool, 'verbose', false) - 1) < 1e-12;
results{end+1,1} = 'single-multiset: r=1, N>1 renyi2 == pooled';
results{end,2}   = abs(entropyExpTens(d_r1N2, 'method', 'renyi2', 'verbose', false) ...
                       - entropyExpTens(d_pool, 'method', 'renyi2', 'verbose', false)) < 1e-12;

% -- Struct basics for pitch + time --

pitchMat = [0 12; 4 15; 7 19];       % 3 x 2
timeMat  = [0 1];                     % 1 x 2
dens = buildExpTens({pitchMat, timeMat}, [], ...
    [10, 0.1], [3, 1], [true, false], [true, false], [1200, 0], ...
    'verbose', false);

results{end+1,1} = 'MAET: struct nAttrs';
results{end,2}   = dens.nAttrs == 2;
results{end+1,1} = 'MAET: struct N';
results{end,2}   = dens.N == 2;
results{end+1,1} = 'MAET: struct r';
results{end,2}   = isequal(dens.r, [3 1]);
results{end+1,1} = 'MAET: struct K';
results{end,2}   = isequal(dens.K, [3 1]);
results{end+1,1} = 'MAET: struct dim';
results{end,2}   = dens.dim == 3;
results{end+1,1} = 'MAET: struct dimPerAttr';
results{end,2}   = isequal(dens.dimPerAttr, [2 1]);

% -- Cartesian product count and event bookkeeping --

pitchMat = [0 12 5; 4 15 9; 7 19 12];   % 3 x 3
timeMat  = [0 1 2];                      % 1 x 3
dens = buildExpTens({pitchMat, timeMat}, [], ...
    [10, 0.1], [3, 1], [true, false], [true, false], [1200, 0], ...
    'lazy', false, 'verbose', false);

results{end+1,1} = 'MAET: nJ = sum of per-event Cartesian products';
results{end,2}   = dens.nJ == 18;   % 3 events * P(3,3)=6 perms * 1 time = 18
results{end+1,1} = 'MAET: nK = sum of per-event Cartesian product (comb)';
results{end,2}   = dens.nK == 3;    % 3 events * C(3,3)=1 comb * 1 time = 3
results{end+1,1} = 'MAET: eventOfJ';
results{end,2}   = isequal(dens.eventOfJ, repelem(1:3, 6));
results{end+1,1} = 'MAET: eventOfK';
results{end,2}   = isequal(dens.eventOfK, 1:3);

% -- Weight broadcasting --

pitchMat = [0 4; 4 8];                   % K=2, N=2
dens = buildExpTens({pitchMat}, [], 10, 2, false, true, 1200, 'verbose', false);
results{end+1,1} = 'MAET: weight [] -> ones';
results{end,2}   = isequal(dens.w{1}, ones(2, 2));

dens = buildExpTens({pitchMat}, 0.5, 10, 2, false, true, 1200, 'verbose', false);
results{end+1,1} = 'MAET: weight scalar top-level';
results{end,2}   = isequal(dens.w{1}, 0.5 * ones(2, 2));

pitchMat = [0 4 5; 4 8 6];               % K=2, N=3
wRow = [0.5, 1.0, 2.0];                  % 1 x N
dens = buildExpTens({pitchMat}, {wRow}, 10, 2, false, true, 1200, 'verbose', false);
results{end+1,1} = 'MAET: weight 1 x N row broadcast';
results{end,2}   = isequal(dens.w{1}, [0.5 1.0 2.0; 0.5 1.0 2.0]);

pitchMat = [0 4 5; 4 8 6; 7 10 9];       % K=3, N=3
wCol = [0.5; 1.0; 2.0];                  % K x 1
dens = buildExpTens({pitchMat}, {wCol}, 10, 2, false, true, 1200, 'verbose', false);
results{end+1,1} = 'MAET: weight K x 1 column broadcast';
results{end,2}   = isequal(dens.w{1}, repmat([0.5; 1.0; 2.0], 1, 3));

pitchMat = [0 4; 4 8];
W = [0.1 0.2; 0.3 0.4];
dens = buildExpTens({pitchMat}, {W}, 10, 2, false, true, 1200, 'verbose', false);
results{end+1,1} = 'MAET: weight K x N full matrix';
results{end,2}   = isequal(dens.w{1}, W);

% -- NaN-padded variable-size events --

pitchMat = [0 0; 4 4; 7 NaN];
timeMat  = [0 1];
dens = buildExpTens({pitchMat, timeMat}, [], ...
    [10, 0.1], [2, 1], [false false], [true false], [1200, 0], ...
    'lazy', false, 'verbose', false);
% Event 1: P(3,2)=6 perms, C(3,2)=3 combs. Event 2: P(2,2)=2, C(2,2)=1.
results{end+1,1} = 'MAET: NaN-padded nJ';
results{end,2}   = dens.nJ == 8;
results{end+1,1} = 'MAET: NaN-padded nK';
results{end,2}   = dens.nK == 4;

% -- Per-tuple weight factorisation --

pitch1 = [0; 4];
time1  = 1.5;
wPitch = [2.0; 3.0];
wTime  = 5.0;
dens = buildExpTens({pitch1, time1}, {wPitch, wTime}, ...
    [10, 0.1], [2, 1], [false false], [true false], [1200, 0], ...
    'lazy', false, 'verbose', false);
% 2 pitch perms, each with weight 2 * 3 * 5 = 30
results{end+1,1} = 'MAET: per-tuple weight factorisation (wJ)';
results{end,2}   = all(abs(dens.wJ - 30) < 1e-12);
results{end+1,1} = 'MAET: per-tuple weight factorisation (wv_comb)';
results{end,2}   = all(abs(dens.wv_comb - 30) < 1e-12);

% -- Error paths --

pitchMat = [0 4];
results{end+1,1} = 'MAET: insufficient values errors';
results{end,2}   = throwsError(@() buildExpTens({[0 0; 4 NaN; NaN NaN]}, [], ...
    10, 2, false, true, 1200, 'verbose', false));

results{end+1,1} = 'MAET: wrong r length errors';
results{end,2}   = throwsError(@() buildExpTens({pitchMat, pitchMat}, [], ...
    [10 10], 1, [false false], [true true], [1200 1200], 'verbose', false));

results{end+1,1} = 'MAET: wrong sigma length errors';
results{end,2}   = throwsError(@() buildExpTens({pitchMat, pitchMat}, [], ...
    10, [1 1], [false false], [true true], [1200 1200], 'verbose', false));

results{end+1,1} = 'MAET: mismatched N errors';
results{end,2}   = throwsError(@() buildExpTens({[0 4], [0 1 2]}, [], ...
    [10 0.1], [1 1], [false false], [true false], [1200 0], 'verbose', false));

results{end+1,1} = 'MAET: wrong positional count errors';
results{end,2}   = throwsError(@() buildExpTens({pitchMat}, [], ...
    10, 1, false, true, 'verbose', false));

% -- isRel + r=1 degenerate warning --

lastwarn('');   % clear the warning buffer
buildExpTens({pitchMat}, [], 10, 1, true, true, 1200, 'verbose', false);
warnMsg = lastwarn;
results{end+1,1} = 'MAET: isRel + r=1 emits degenerate warning';
results{end,2}   = ~isempty(warnMsg) && contains(warnMsg, 'degenerate');

% -- evalExpTens MA path: single multiset-equivalence (isRel=false) --

p_sm_v  = [0; 400; 700];
w_sm_v  = [1; 0.7; 0.5];
sigma_v = 10; r_v = 2; isPer_v = true; period_v = 1200;
xSingleMultiset_abs = [100 500; 300 600];   % dim=2, nQ=2 (absolute r=2)

dens_sm = buildExpTens(p_sm_v, w_sm_v, sigma_v, r_v, false, isPer_v, period_v, ...
    'verbose', false);
vals_sm = evalExpTens(dens_sm, xSingleMultiset_abs, 'verbose', false);

dens_ma = buildExpTens({p_sm_v}, {w_sm_v}, sigma_v, r_v, false, isPer_v, ...
    period_v, 'verbose', false);
vals_ma_cell = evalExpTens(dens_ma, {xSingleMultiset_abs}, 'verbose', false);
vals_ma_mat  = evalExpTens(dens_ma,  xSingleMultiset_abs,  'verbose', false);

results{end+1,1} = 'evalExpTens MA: single multiset-equivalence abs (cell form)';
results{end,2}   = max(abs(vals_ma_cell - vals_sm)) < 1e-12;
results{end+1,1} = 'evalExpTens MA: single multiset-equivalence abs (matrix form)';
results{end,2}   = max(abs(vals_ma_mat - vals_sm)) < 1e-12;

% -- evalExpTens MA path: single multiset-equivalence (isRel=true, r=3) --

r_v = 3;
xSingleMultiset_rel = [400 200; 700 500];    % dim = r-1 = 2, nQ = 2
dens_sm = buildExpTens(p_sm_v, w_sm_v, sigma_v, r_v, true, isPer_v, period_v, ...
    'verbose', false);
vals_sm = evalExpTens(dens_sm, xSingleMultiset_rel, 'verbose', false);

dens_ma = buildExpTens({p_sm_v}, {w_sm_v}, sigma_v, r_v, true, isPer_v, ...
    period_v, 'verbose', false);
vals_ma = evalExpTens(dens_ma, {xSingleMultiset_rel}, 'verbose', false);

results{end+1,1} = 'evalExpTens MA: single multiset-equivalence rel';
results{end,2}   = max(abs(vals_ma - vals_sm)) < 1e-12;

% Normalisation modes
for modeCell = {'gaussian', 'pdf'}
    mode = modeCell{1};
    vals_sm_n = evalExpTens(dens_sm, xSingleMultiset_rel, mode, 'verbose', false);
    vals_ma_n = evalExpTens(dens_ma, {xSingleMultiset_rel}, mode, 'verbose', false);
    results{end+1,1} = ['evalExpTens MA: single multiset-equivalence normalize=' mode]; %#ok<SAGROW>
    results{end,2}   = max(abs(vals_ma_n - vals_sm_n)) < 1e-12;
end

% -- evalExpTens MA: cell form vs matrix form agree --

pitchMat = [0; 4; 7];    % K=3, N=1
timeMat  = 1.0;           % 1 x 1
dens = buildExpTens({pitchMat, timeMat}, [], ...
    [10, 0.1], [2, 1], [false false], [true false], [1200, 0], ...
    'verbose', false);

x_pitch = [0 4; 4 7];    % 2 x 2
x_time  = [1 2];          % 1 x 2
vals_cell = evalExpTens(dens, {x_pitch, x_time}, 'verbose', false);
vals_mat  = evalExpTens(dens, [x_pitch; x_time], 'verbose', false);
results{end+1,1} = 'evalExpTens MA: cell form == matrix form';
results{end,2}   = isequal(vals_cell, vals_mat);

% -- evalExpTens MA: per-group isPer --

pitch1 = 0;   % K=1, N=1
time1  = 0;
dens = buildExpTens({pitch1, time1}, [], ...
    [20, 20], [1, 1], [false false], [true false], [1200, 0], ...
    'verbose', false);
% Pitch periodic: value at pitch=0 vs pitch=1200 should be equal
v_p0    = evalExpTens(dens, {0,    0}, 'verbose', false);
v_p1200 = evalExpTens(dens, {1200, 0}, 'verbose', false);
results{end+1,1} = 'evalExpTens MA: periodic pitch wraps';
results{end,2}   = abs(v_p0 - v_p1200) < 1e-12;
% Time nonperiodic: value at time=0 > time=1200
v_t0    = evalExpTens(dens, {0, 0},    'verbose', false);
v_t1200 = evalExpTens(dens, {0, 1200}, 'verbose', false);
results{end+1,1} = 'evalExpTens MA: nonperiodic time does not wrap';
results{end,2}   = v_t1200 < v_t0;

% -- evalExpTens MA: density positive at a tuple centre --

pitchMat = [0; 4; 7];
timeMat  = 1.0;
dens = buildExpTens({pitchMat, timeMat}, [], ...
    [10, 0.1], [2, 1], [false false], [true false], [1200, 0], ...
    'verbose', false);
v_centre = evalExpTens(dens, {[0; 4], 1.0}, 'verbose', false);
v_far    = evalExpTens(dens, {[600; 800], 50.0}, 'verbose', false);
results{end+1,1} = 'evalExpTens MA: density is positive at tuple centre';
results{end,2}   = v_centre > 0 && v_centre > v_far;

% -- evalExpTens MA: error paths --

results{end+1,1} = 'evalExpTens MA: wrong cell length errors';
results{end,2}   = throwsError(@() evalExpTens(dens, {[0; 4]}, 'verbose', false));

results{end+1,1} = 'evalExpTens MA: wrong per-attr rows errors';
results{end,2}   = throwsError(@() evalExpTens(dens, ...
    {zeros(3,1), zeros(1,1)}, 'verbose', false));

results{end+1,1} = 'evalExpTens MA: wrong total rows (matrix form) errors';
results{end,2}   = throwsError(@() evalExpTens(dens, zeros(5,1), 'verbose', false));

% -- evalExpTens MA raw form: parity with struct path --

% Two-attribute setup: pitch (group 1, periodic mod 12) and time
% (group 2, non-periodic), with three events.
pAttr_er  = {[67 66 64], [5 6 7]};
w_er      = [];
sigma_er  = [0.5, 0.25];
r_er      = [1, 1];
isRel_er  = [false false];
isPer_er  = [true false];
periods_er = [12 0];

Xq_er = [66; 6];   % single query at the penult event

% Path 1: build dens, then eval.
dens_er = buildExpTens(pAttr_er, w_er, sigma_er, r_er, ...
                       isRel_er, isPer_er, periods_er, 'verbose', false);
vals_er_struct = evalExpTens(dens_er, Xq_er, 'verbose', false);

% Path 2: raw MA form (8 positional args + 'verbose').
vals_er_raw = evalExpTens(pAttr_er, w_er, sigma_er, r_er, ...
                          isRel_er, isPer_er, periods_er, Xq_er, ...
                          'verbose', false);

results{end+1,1} = 'evalExpTens MA raw: matches dens-struct path';
results{end,2}   = max(abs(vals_er_raw(:) - vals_er_struct(:))) < 1e-12;

% Multi-query (3 columns), check matrix form parity.
Xq_er_multi = [60 66 72; 5 6 7];
vals_er_multi_struct = evalExpTens(dens_er, Xq_er_multi, 'verbose', false);
vals_er_multi_raw    = evalExpTens(pAttr_er, w_er, sigma_er, r_er, ...
                                   isRel_er, isPer_er, periods_er, Xq_er_multi, ...
                                   'verbose', false);
results{end+1,1} = 'evalExpTens MA raw: multi-query matches struct path';
results{end,2}   = max(abs(vals_er_multi_raw(:) - vals_er_multi_struct(:))) < 1e-12;

% Cell-form X parity (raw MA path must route through localEvalMA, which
% accepts {X_1, ..., X_A} per-attribute cells as well as stacked matrices).
Xq_er_cell = {[60 66 72], [5 6 7]};
vals_er_cell_struct = evalExpTens(dens_er, Xq_er_cell, 'verbose', false);
vals_er_cell_raw    = evalExpTens(pAttr_er, w_er, sigma_er, r_er, ...
                                  isRel_er, isPer_er, periods_er, Xq_er_cell, ...
                                  'verbose', false);
results{end+1,1} = 'evalExpTens MA raw: cell-form X matches struct path';
results{end,2}   = max(abs(vals_er_cell_raw(:) - vals_er_cell_struct(:))) < 1e-12;

% Normalize argument as trailing 9th positional.
vals_er_norm_struct = evalExpTens(dens_er, Xq_er, 'pdf', 'verbose', false);
vals_er_norm_raw    = evalExpTens(pAttr_er, w_er, sigma_er, r_er, ...
                                  isRel_er, isPer_er, periods_er, Xq_er, ...
                                  'pdf', 'verbose', false);
results{end+1,1} = 'evalExpTens MA raw: trailing normalize matches struct path';
results{end,2}   = max(abs(vals_er_norm_raw(:) - vals_er_norm_struct(:))) < 1e-12;

% -- cosSimExpTens MA path: single multiset-equivalence --

p_a_v  = [0; 400; 700];
p_b_v  = [0; 300; 700];
w_a_v  = [1; 0.7; 0.5];
w_b_v  = [1; 0.6; 0.8];

% Absolute (isRel=false), periodic
s_sm = cosSimExpTens(p_a_v, w_a_v, p_b_v, w_b_v, 10, 2, false, true, 1200, ...
    'verbose', false);
da = buildExpTens({p_a_v}, {w_a_v}, 10, 2, false, true, 1200, 'verbose', false);
db = buildExpTens({p_b_v}, {w_b_v}, 10, 2, false, true, 1200, 'verbose', false);
s_ma = cosSimExpTens(da, db, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: single multiset-equivalence abs periodic';
results{end,2}   = abs(s_ma - s_sm) < 1e-12;

% Relative + periodic (uses pairwise-differences formula per attribute)
for r_v = [2, 3]
    s_sm = cosSimExpTens(p_a_v, w_a_v, p_b_v, w_b_v, 10, r_v, true, true, 1200, ...
        'verbose', false);
    da = buildExpTens({p_a_v}, {w_a_v}, 10, r_v, true, true, 1200, 'verbose', false);
    db = buildExpTens({p_b_v}, {w_b_v}, 10, r_v, true, true, 1200, 'verbose', false);
    s_ma = cosSimExpTens(da, db, 'verbose', false);
    results{end+1,1} = sprintf('cosSimExpTens MA: single multiset-equivalence rel periodic r=%d', r_v); %#ok<SAGROW>
    results{end,2}   = abs(s_ma - s_sm) < 1e-12;
end

% Relative + non-periodic
s_sm = cosSimExpTens(p_a_v, w_a_v, p_b_v, w_b_v, 10, 3, true, false, 0, ...
    'verbose', false);
da = buildExpTens({p_a_v}, {w_a_v}, 10, 3, true, false, 0, 'verbose', false);
db = buildExpTens({p_b_v}, {w_b_v}, 10, 3, true, false, 0, 'verbose', false);
s_ma = cosSimExpTens(da, db, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: single multiset-equivalence rel non-periodic';
results{end,2}   = abs(s_ma - s_sm) < 1e-12;

% -- cosSimExpTens MA: self-similarity = 1 --

pitchMA = [0 12; 4 15; 7 19];    % 3 x 2
timeMA  = [0 1];                  % 1 x 2
d = buildExpTens({pitchMA, timeMA}, [], ...
    [10, 0.1], [3, 1], [true, false], [true, false], [1200, 0], ...
    'verbose', false);
s_self = cosSimExpTens(d, d, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: self-similarity = 1';
results{end,2}   = abs(s_self - 1) < 1e-12;

% -- cosSimExpTens MA: symmetry --

pitchA = [0 12; 4 15; 7 19];
timeA  = [0 1];
pitchB = [0 10; 4 13; 7 17];
timeB  = [0 1.2];
da = buildExpTens({pitchA, timeA}, [], ...
    [10, 0.1], [3, 1], [true, false], [true, false], [1200, 0], ...
    'verbose', false);
db = buildExpTens({pitchB, timeB}, [], ...
    [10, 0.1], [3, 1], [true, false], [true, false], [1200, 0], ...
    'verbose', false);
s_ab = cosSimExpTens(da, db, 'verbose', false);
s_ba = cosSimExpTens(db, da, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: symmetry (a,b) == (b,a)';
results{end,2}   = abs(s_ab - s_ba) < 1e-12;

% -- cosSimExpTens MA: isRel transposition invariance --

pitchT  = [0; 400; 700];
pitchTs = pitchT + 137;
timeT   = 1;
d1 = buildExpTens({pitchT,  timeT}, [], ...
    [10, 0.1], [3, 1], [true, false], [true, false], [1200, 0], ...
    'verbose', false);
d2 = buildExpTens({pitchTs, timeT}, [], ...
    [10, 0.1], [3, 1], [true, false], [true, false], [1200, 0], ...
    'verbose', false);
s_trans = cosSimExpTens(d1, d2, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: isRel transposition invariance';
results{end,2}   = abs(s_trans - 1) < 1e-10;

% -- cosSimExpTens MA: raw-args matches struct form --

s_raw = cosSimExpTens({pitchA, timeA}, [], {pitchB, timeB}, [], ...
    [10, 0.1], [3, 1], [true, false], [true, false], [1200, 0], ...
    'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: raw-args == struct form';
results{end,2}   = abs(s_raw - s_ab) < 1e-12;

% -- cosSimExpTens: single multiset raw-args still works (backward compat check) --

s_sm_raw = cosSimExpTens([0 4 7], [], [0 4 7], [], 10, 2, true, true, 1200, ...
    'verbose', false);
results{end+1,1} = 'cosSimExpTens: single multiset raw-args identical = 1';
results{end,2}   = abs(s_sm_raw - 1) < 1e-12;

% -- cosSimExpTens MA: mismatched raw-args kinds error --

results{end+1,1} = 'cosSimExpTens MA: mismatched raw-args kinds error';
results{end,2}   = throwsError(@() cosSimExpTens( ...
    {pitchA, timeA}, [], [0 4 7], [], 10, 2, true, true, 1200, 'verbose', false));

% -- cosSimExpTens: incompatible attribute structure errors --
% Under the unified type there is no single-attribute-vs-multi-attribute type mix to reject; the
% genuine incompatibility is a single-multiset (A=1) density paired with
% a multi-attribute (A=2) density, which cannot share an inner product.

d_single = buildExpTens([0 4 7], [], 10, 2, false, true, 1200, 'verbose', false);
d_multi  = buildExpTens({[0; 4; 7], [0; 1; 2]}, [], [10 10], [2 1], ...
    [false false], [true false], [1200 0], 'verbose', false);
results{end+1,1} = 'cosSimExpTens: single-multiset vs multi-attribute structs error';
results{end,2}   = throwsError(@() cosSimExpTens(d_single, d_multi, 'verbose', false));

% -- cosSimExpTens MA: parameter-mismatch errors --

d_ref = buildExpTens({pitchA}, [], 10, 2, false, true, 1200, 'verbose', false);
% different r
d_r = buildExpTens({pitchA}, [], 10, 3, false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: mismatched r error';
results{end,2}   = throwsError(@() cosSimExpTens(d_ref, d_r, 'verbose', false));
% different sigma
d_s = buildExpTens({pitchA}, [], 20, 2, false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: mismatched sigma error';
results{end,2}   = throwsError(@() cosSimExpTens(d_ref, d_s, 'verbose', false));
% different isRel
d_rel = buildExpTens({pitchA}, [], 10, 2, true, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: mismatched isRel error';
results{end,2}   = throwsError(@() cosSimExpTens(d_ref, d_rel, 'verbose', false));
% different period on periodic group
d_p = buildExpTens({pitchA}, [], 10, 2, false, true, 2400, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: mismatched period error';
results{end,2}   = throwsError(@() cosSimExpTens(d_ref, d_p, 'verbose', false));

% -- entropyExpTens MA: single multiset-equivalence periodic --

p_e = [0; 4; 7];
w_e = [1; 1; 1];
H_sm = entropyExpTens(p_e.', w_e.', 10, 1, false, true, 12, ...
    'nPointsPerDim', 400, 'verbose', false);
H_ma = entropyExpTens({p_e}, {w_e}, 10, 1, false, true, 12, ...
    'nPointsPerDim', 400, 'verbose', false);
results{end+1,1} = 'entropyExpTens MA: single multiset-equivalence periodic';
results{end,2}   = abs(H_ma - H_sm) < 1e-10;

% -- entropyExpTens MA: single multiset-equivalence non-periodic --

H_sm = entropyExpTens(p_e.', w_e.', 10, 1, false, false, 0, ...
    'xMin', -3, 'xMax', 10, 'nPointsPerDim', 400, 'verbose', false);
H_ma = entropyExpTens({p_e}, {w_e}, 10, 1, false, false, 0, ...
    'xMin', -3, 'xMax', 10, 'nPointsPerDim', 400, 'verbose', false);
results{end+1,1} = 'entropyExpTens MA: single multiset-equivalence non-periodic';
results{end,2}   = abs(H_ma - H_sm) < 1e-10;

% -- entropyExpTens MA: uniform pitch near 1 --

p_uniform = (0:11).';
H_u = entropyExpTens({p_uniform}, [], 100, 1, false, true, 12, ...
    'method', 'normalized', 'nPointsPerDim', 400, 'verbose', false);
results{end+1,1} = 'entropyExpTens MA: uniform chromatic near 1';
results{end,2}   = H_u > 0.95;

% -- entropyExpTens MA: concentrated below uniform --

H_one = entropyExpTens({5}, [], 20, 1, false, true, 12, ...
    'nPointsPerDim', 400, 'verbose', false);
H_all = entropyExpTens({p_uniform}, [], 20, 1, false, true, 12, ...
    'nPointsPerDim', 400, 'verbose', false);
results{end+1,1} = 'entropyExpTens MA: concentrated < uniform';
results{end,2}   = H_one < H_all;

% -- entropyExpTens MA: pitch + time runs (dim = 2) --

pitchE = [0 12; 4 15; 7 19];   % 3 x 2
timeE  = [0 1];                 % 1 x 2
densE = buildExpTens({pitchE, timeE}, [], ...
    [20, 0.1], [2, 1], [true, false], [true, false], [1200, 0], ...
    'verbose', false);
results{end+1,1} = 'entropyExpTens MA: dim == 2 (r=2 pitch + r=1 time)';
results{end,2}   = densE.dim == 2;
H_pt = entropyExpTens(densE, ...
    'method', 'normalized', ...
    'xMin', -0.5, 'xMax', 1.5, 'nPointsPerDim', 80, 'verbose', false);
results{end+1,1} = 'entropyExpTens MA: pitch+time H in (0,1)';
results{end,2}   = H_pt > 0 && H_pt < 1;

% -- entropyExpTens MA: grid-limit guard --

results{end+1,1} = 'entropyExpTens MA: grid-limit exceeded errors';
results{end,2}   = throwsError(@() entropyExpTens(densE, ...
    'xMin', 0, 'xMax', 2, 'nPointsPerDim', 20000, 'gridLimit', 1e6, 'verbose', false));

% -- entropyExpTens MA: missing bounds error --

results{end+1,1} = 'entropyExpTens MA: missing non-periodic bounds errors';
results{end,2}   = throwsError(@() entropyExpTens({p_e}, [], 10, 1, ...
    false, false, 0, 'nPointsPerDim', 100, 'verbose', false));

% -- entropyExpTens MA: per-group bounds vector matches scalar --

H_scalar = entropyExpTens(densE, ...
    'xMin', -0.5, 'xMax', 1.5, 'nPointsPerDim', 60, 'verbose', false);
H_vec = entropyExpTens(densE, ...
    'xMin', [NaN, -0.5], 'xMax', [NaN, 1.5], 'nPointsPerDim', 60, 'verbose', false);
results{end+1,1} = 'entropyExpTens MA: per-group bounds vector == scalar';
results{end,2}   = abs(H_scalar - H_vec) < 1e-12;

% -- differenceEvents: moved onto the (pAttr, w, specs) triple (3c-iv);
%    its tests live in tests/test_difference.m. The old groups/cell-order
%    contract and the K=1-only restriction were removed with Commit 3c-iv.

% -- nTupleEntropy: parity after refactor onto differenceEvents --
% A second sanity check that the refactor preserves the n-tuple
% entropy value for a familiar input (whole-tone scale on the
% 12-EDO).
H_whole = nTupleEntropy([0 2 4 6 8 10], 12, 2, 'method', 'shannon');
results{end+1,1} = 'nTupleEntropy: whole-tone n=2 still = 0 after refactor';
results{end,2}   = abs(H_whole) < 1e-12;

% -- bindEvents: now emits nested specs (3c); its tests live in
%    tests/test_bind.m. The old separate-attribute / groups contract
%    was removed with Commit 3c. --

% -- weightEvents: per-event window factor (single-input API, target_attr,
%    drop_input_attr, three-tuple return). Mirrors Python tests/test_maet.py
%    `test_weight_*` for parity. --

% gamma = 0 limit: pure Gaussian with std = width.
p_we = {[60 62 64 67 72]};
[~, w_we, ~] = unpackPreMaet(weightEvents(p_we, [], 1, 1, 64, 0, 'sd', 3, 'dropInputAttr', false));
expected_we = exp(-(([60 62 64 67 72] - 64) .^ 2) ./ (2 * 3 ^ 2));
results{end+1,1} = 'weightEvents: gamma = 0 is pure Gaussian (std = width)';
results{end,2}   = max(abs(w_we{1} - expected_we)) < 1e-12;

% gamma = 1 limit: pure rectangle with half-width = width * sqrt(3).
[~, w_re, ~] = unpackPreMaet(weightEvents({[60 62 64 67 72]}, [], 1, 1, 64, 1, 'sd', 3, 'dropInputAttr', false));
expected_re = double(abs([60 62 64 67 72] - 64) <= 3 * sqrt(3));
results{end+1,1} = 'weightEvents: gamma = 1 is pure rectangle (half-width = width * sqrt(3))';
results{end,2}   = isequal(w_re{1}, expected_re);

% Peak h(0) = 1 throughout the family.
gammas_peak = [0.05 0.1 0.25 0.5 0.75 0.9 0.95];
peak_ok = true;
for gg = gammas_peak
    [~, w_pk, ~] = unpackPreMaet(weightEvents({5}, [], 1, 1, 5, gg, 'sd', 2, 'dropInputAttr', false));
    if abs(w_pk{1} - 1) >= 1e-12
        peak_ok = false; break;
    end
end
results{end+1,1} = 'weightEvents: peak h(0) = 1 for every gamma in (0, 1)';
results{end,2}   = peak_ok;

% Fixed-variance property: total variance is width^2 for every gamma.
y_fv = linspace(-30, 30, 60001);
dy_fv = y_fv(2) - y_fv(1);
width_fv = 4;
gammas_fv = [0 0.1 0.25 0.5 0.75 0.9 1.0];
var_ok = true;
for gg = gammas_fv
    [~, w_fv, ~] = unpackPreMaet(weightEvents({y_fv}, [], 1, 1, 0, gg, 'sd', width_fv, 'dropInputAttr', false));
    h_fv = w_fv{1};
    area = sum(h_fv) * dy_fv;
    variance = sum(y_fv .^ 2 .* h_fv) * dy_fv / area;
    if abs(variance - width_fv ^ 2) >= 5e-3
        var_ok = false; break;
    end
end
results{end+1,1} = 'weightEvents: variance is width^2 for every gamma (fixed-variance family)';
results{end,2}   = var_ok;

% Output is a 3-tuple (pAttrOut, wOut, specsOut).
p_3t = {[1 2], [3 4]};
[p_3t_out, w_3t_out, s_3t_out] = unpackPreMaet(weightEvents(p_3t, [], 1, 1, 1.5, 0, 'sd', 1, 'dropInputAttr', false));
results{end+1,1} = 'weightEvents: returns three-tuple (pAttr, w, specs)';
results{end,2}   = iscell(p_3t_out) && numel(p_3t_out) == 2 && ...
                   iscell(w_3t_out) && numel(w_3t_out) == 2 && ...
                   iscell(s_3t_out) && numel(s_3t_out) == 2 && ...
                   all(cellfun(@isstruct, s_3t_out));

% Non-input, non-target attribute passes its incoming weight through.
p_pt = {[1 2 3], [10 20 30], [100 200 300]};
[~, w_pt, ~] = unpackPreMaet(weightEvents(p_pt, {[], [], 0.5}, 1, 2, 2, 0, 'sd', 1, 'dropInputAttr', false));
results{end+1,1} = 'weightEvents: non-input non-target attribute passes through unchanged';
results{end,2}   = isequal(w_pt{3}, 0.5);

% input ~= target: factor lands on the target attribute's weights, the
% input attribute's are unchanged.
p_int = {[60 64 67], [0 1 2]};        % pitch (target), time (input)
[~, w_int, ~] = unpackPreMaet(weightEvents(p_int, [], 2, 1, 1, 0, 'sd', 1, 'dropInputAttr', false));
expected_int = exp(-(([0 1 2] - 1) .^ 2) ./ 2);
results{end+1,1} = 'weightEvents: input ~= target writes factor to target attribute only';
results{end,2}   = max(abs(w_int{1} - expected_int)) < 1e-12 && isempty(w_int{2});

% Target with K_target > 1: (1, N) factor broadcasts across K_target positions.
p_bc = {[60 64; 62 65; 64 67], [0 1]};   % pitch K=3 (target), time K=1 (input)
w_bc_in = {ones(3, 2), []};
[~, w_bc, ~] = unpackPreMaet(weightEvents(p_bc, w_bc_in, 2, 1, 0, 0, 'sd', 1, 'dropInputAttr', false));
factor_bc = exp(-([0 1] .^ 2) ./ 2);
expected_bc = repmat(factor_bc, 3, 1);
results{end+1,1} = 'weightEvents: (1, N) factor broadcasts across target K_target > 1 positions';
results{end,2}   = isequal(size(w_bc{1}), [3 2]) && ...
                   max(abs(w_bc{1}(:) - expected_bc(:))) < 1e-12;

% Periodic wrap: delta = v - c wrapped to [-P/2, P/2] before h. Raw values intact.
p_per = {[10 11 0 1 2]};
[~, w_per, ~] = unpackPreMaet(weightEvents(p_per, [], 1, 1, 0, 0, 'sd', 2, 'dropInputAttr', false, 'isPer', true, 'period', 12));
expected_per = exp(-([-2 -1 0 1 2] .^ 2) ./ 8);
results{end+1,1} = 'weightEvents: periodic wrap of delta before shape';
results{end,2}   = max(abs(w_per{1} - expected_per)) < 1e-12;
results{end+1,1} = 'weightEvents: input values stay raw (no value mutation)';
results{end,2}   = isequal(p_per{1}, [10 11 0 1 2]);

% Scalar existing weight multiplies in.
[~, w_mul, ~] = unpackPreMaet(weightEvents({[1 2 3]}, 0.5, 1, 1, 2, 0, 'sd', 1, 'dropInputAttr', false));
h_mul = exp(-(([1 2 3] - 2) .^ 2) ./ 2);
results{end+1,1} = 'weightEvents: scalar existing weight multiplies in';
results{end,2}   = max(abs(w_mul{1} - 0.5 .* h_mul)) < 1e-12;

% Sequential composition replaces the old multi-input behaviour: two calls
% on the same target multiply factors. pitch (target), two scaffolding attrs.
p_seq = {[60 64 67], [0 1 2], [0 0.5 1]};
[p_seq1, w_seq1, g_seq1] = unpackPreMaet(weightEvents(p_seq, [], 2, 1, 1, 0, 'sd', 1, 'dropInputAttr', false));
[~, w_seq2, ~] = unpackPreMaet(weightEvents(p_seq1, w_seq1, 3, 1, 0.5, 0, 'sd', 0.5, 'dropInputAttr', false));
h_time_seq = exp(-(([0 1 2] - 1) .^ 2) ./ 2);
h_beat_seq = exp(-(([0 0.5 1] - 0.5) .^ 2) ./ 0.5);
results{end+1,1} = 'weightEvents: sequential composition multiplies factors into target';
results{end,2}   = max(abs(w_seq2{1} - h_time_seq .* h_beat_seq)) < 1e-12;

% drop_input_attr=true drops the input attribute.
p_del = {[60 64 67], [0 1 2]};
[p_del_out, w_del_out, g_del_out] = unpackPreMaet(weightEvents(p_del, [], 2, 1, 1, 0, 'sd', 1, 'dropInputAttr', true));
results{end+1,1} = 'weightEvents: drop_input_attr=true drops the input attribute';
results{end,2}   = numel(p_del_out) == 1 && numel(w_del_out) == 1 && ...
                   isequal(p_del_out{1}, [60 64 67]);

% drop_input_attr with inputAttr > targetAttr: input attribute's value,
% weight, and spec are dropped; the target keeps its output index.
% Input = attr 2, target = attr 1 (input after target).
p_gc = {[1 2], [3 4], [5 6]};
[p_gc_out, w_gc_out, s_gc_out] = unpackPreMaet(weightEvents(p_gc, [], 2, 1, 3.5, 0, 'sd', 1, 'dropInputAttr', true));
factor_gc = exp(-(([3 4] - 3.5) .^ 2) ./ 2);   % from input attr 2 values
results{end+1,1} = 'weightEvents: drop_input_attr (input after target) keeps target index';
results{end,2}   = numel(p_gc_out) == 2 && numel(w_gc_out) == 2 && ...
                   numel(s_gc_out) == 2 && ...
                   isequal(p_gc_out{1}, [1 2]) && isequal(p_gc_out{2}, [5 6]) && ...
                   max(abs(w_gc_out{1} - factor_gc)) < 1e-12;

% drop_input_attr with inputAttr < targetAttr: input attribute is dropped
% and the target shifts down one output index, carrying the factor.
% Input = attr 1, target = attr 3 (input before target).
p_gk = {[1 2], [3 4], [5 6]};
[p_gk_out, w_gk_out, s_gk_out] = unpackPreMaet(weightEvents(p_gk, [], 1, 3, 1.5, 0, 'sd', 1, 'dropInputAttr', true));
factor_gk = exp(-(([1 2] - 1.5) .^ 2) ./ 2);   % from input attr 1 values
results{end+1,1} = 'weightEvents: drop_input_attr (input before target) shifts target index';
results{end,2}   = numel(p_gk_out) == 2 && numel(w_gk_out) == 2 && ...
                   numel(s_gk_out) == 2 && ...
                   isequal(p_gk_out{1}, [3 4]) && isequal(p_gk_out{2}, [5 6]) && ...
                   max(abs(w_gk_out{2} - factor_gk)) < 1e-12;

% --- sd / width API: exactly one must be supplied; both convertible. ---
results{end+1,1} = 'weightEvents: no sd and no width errors (sdWidthXor)';
results{end,2}   = throwsErrorWithId( ...
    @() weightEvents({[1 2]}, [], 1, 1, 1, 0, 'dropInputAttr', false), ...
    'weightEvents:sdWidthXor');

results{end+1,1} = 'weightEvents: both sd and width errors (sdWidthXor)';
results{end,2}   = throwsErrorWithId( ...
    @() weightEvents({[1 2]}, [], 1, 1, 1, 0, 'sd', 1, 'width', 1, 'dropInputAttr', false), ...
    'weightEvents:sdWidthXor');

% sd and width produce the same output when paired by sd = width / (2*sqrt(3)).
sd_xy   = 1.0;
wid_xy  = sd_xy * 2 * sqrt(3);
[~, w_sd_xy,  ~] = unpackPreMaet(weightEvents({linspace(-5, 5, 21)}, [], 1, 1, 0, 1, 'sd', sd_xy, 'dropInputAttr', false));
[~, w_wid_xy, ~] = unpackPreMaet(weightEvents({linspace(-5, 5, 21)}, [], 1, 1, 0, 1, 'width', wid_xy, 'dropInputAttr', false));
results{end+1,1} = 'weightEvents: sd and width yield identical output under conversion';
results{end,2}   = max(abs(w_sd_xy{1} - w_wid_xy{1})) < 1e-12;

% width form: rect of full support L covers half-open [-L/2, L/2); the
% lower edge -L/2 is kept, the upper edge +L/2 is excluded, events just
% past are zeroed (a closed interval over-counts).
L_rect = 1.0;
eps_rect = 1e-6;
t_rect = [-L_rect/2, -L_rect/4, 0, L_rect/4, L_rect/2, L_rect/2 + eps_rect];
[~, w_rect, ~] = unpackPreMaet(weightEvents({t_rect}, [], 1, 1, 0, 1, 'width', L_rect, 'dropInputAttr', false));
results{end+1,1} = 'weightEvents: width gives half-open rectangle of total support width';
results{end,2}   = all(w_rect{1}(1:4) == 1) && w_rect{1}(5) == 0 && w_rect{1}(6) == 0;

% #20: half-open rect window keeps exactly N pulses for full support N on a
% unit grid (a closed interval would give 1, 3, 3, 5, 5 for widths 1..5).
t_grid = 0:8;                       % IOI = 1
rectCounts = zeros(1, 5);
for Wn = 1:5
    [~, w_g, ~] = unpackPreMaet(weightEvents({t_grid, t_grid}, [], 2, 1, 4, 1, ...
                               'width', Wn, 'dropInputAttr', false));
    rectCounts(Wn) = nnz(w_g{1});
end
results{end+1,1} = 'weightEvents: half-open rect width N keeps N pulses (on-pulse centre)';
results{end,2}   = isequal(rectCounts, [1 2 3 4 5]);

% Between-pulse centre keeps one pulse (lower edge), not zero or two.
[~, w_bp, ~] = unpackPreMaet(weightEvents({t_grid, t_grid}, [], 2, 1, 3.5, 1, ...
                            'width', 1, 'dropInputAttr', false));
results{end+1,1} = 'weightEvents: half-open rect between-pulse centre keeps 1 pulse';
results{end,2}   = (nnz(w_bp{1}) == 1);

% #21: an out-of-support rectangular window gives a zero-mass density, and
% renyi2 returns NaN rather than erroring.
[pa_z, wa_z, ~] = unpackPreMaet(weightEvents({[60 62 64], [0 1 2]}, [], 2, 1, 100, 1, ...
                               'width', 1, 'dropInputAttr', false));
dens_z = buildExpTens(pa_z, wa_z, [1 1], [1 1], [false false], ...
                      [false false], [0 0]);
H_z = entropyExpTens(dens_z, 'method', 'renyi2', 'verbose', false);
results{end+1,1} = 'weightEvents+renyi2: out-of-support window yields NaN';
results{end,2}   = isnan(H_z);

% Error cases (per-validation IDs in MATLAB).
results{end+1,1} = 'weightEvents: zero sd errors (badSd id)';
results{end,2}   = throwsErrorWithId( ...
    @() weightEvents({[1 2]}, [], 1, 1, 1, 0.5, 'sd', 0, 'dropInputAttr', false), ...
    'weightEvents:badSd');

results{end+1,1} = 'weightEvents: negative sd errors (badSd id)';
results{end,2}   = throwsErrorWithId( ...
    @() weightEvents({[1 2]}, [], 1, 1, 1, 0.5, 'sd', -1, 'dropInputAttr', false), ...
    'weightEvents:badSd');

results{end+1,1} = 'weightEvents: shape > 1 errors (badShape id)';
results{end,2}   = throwsErrorWithId( ...
    @() weightEvents({[1 2]}, [], 1, 1, 1, 1.5, 'sd', 1, 'dropInputAttr', false), ...
    'weightEvents:badShape');

results{end+1,1} = 'weightEvents: shape < 0 errors (badShape id)';
results{end,2}   = throwsErrorWithId( ...
    @() weightEvents({[1 2]}, [], 1, 1, 1, -0.1, 'sd', 1, 'dropInputAttr', false), ...
    'weightEvents:badShape');

results{end+1,1} = 'weightEvents: inputAttr out of range errors (badInputAttr id)';
results{end,2}   = throwsErrorWithId( ...
    @() weightEvents({[1 2]}, [], 2, 1, 1, 0, 'sd', 1, 'dropInputAttr', false), ...
    'weightEvents:badInputAttr');

results{end+1,1} = 'weightEvents: input K > 1 errors (inputAttrNotK1 id)';
results{end,2}   = throwsErrorWithId( ...
    @() weightEvents({[1 2; 3 4]}, [], 1, 1, 1, 0, 'sd', 1, 'dropInputAttr', false), ...
    'weightEvents:inputAttrNotK1');

results{end+1,1} = 'weightEvents: drop_input_attr=true with input==target errors';
results{end,2}   = throwsErrorWithId( ...
    @() weightEvents({[1 2]}, [], 1, 1, 1, 0, 'sd', 1, 'dropInputAttr', true), ...
    'weightEvents:dropInputAttrIncoherent');

results{end+1,1} = 'weightEvents: isPer=true with period=0 errors (badPeriod id)';
results{end,2}   = throwsErrorWithId( ...
    @() weightEvents({[1 2]}, [], 1, 1, 1, 0, 'sd', 1, 'dropInputAttr', false, 'isPer', true, 'period', 0), ...
    'weightEvents:badPeriod');

% T \circ W centre-shift commutation: T then W with centre c equals W with
% centre c - mu then T (T leaves weights unchanged). Use an intermediate
% gamma so the convolution branch is exercised.
p_tw = {[60 62 64]};
mu_tw = 5; c_tw = 64; width_tw = 3; gamma_tw = 0.3;
pm_after_t = translateAttributes(p_tw, [], {mu_tw});
p_after_t = pm_after_t.pAttr;
[~, w_after_t, ~] = unpackPreMaet(weightEvents(p_after_t, [], 1, 1, c_tw, gamma_tw, 'sd', width_tw, 'dropInputAttr', false));
[~, w_first, ~]   = unpackPreMaet(weightEvents(p_tw, [], 1, 1, c_tw - mu_tw, gamma_tw, 'sd', width_tw, 'dropInputAttr', false));
results{end+1,1} = 'weightEvents: T \circ W centre-shift commutation';
results{end,2}   = max(abs(w_first{1} - w_after_t{1})) < 1e-12;

% translateAttributes moved onto the (pAttr, w, specs) triple (3c-iv-d);
% its tests now live in tests/test_translate.m. The old groups / isRel /
% isPer / period positional contract has been removed.

% -- cosSimExpTens raw-MA scalar-vs-list mode --

p_ref     = {transformAttributes([60 62 64 65 67 69 71], [], {'midi', 'cents'}), 0:6};
p_qry     = {transformAttributes([60 64 67], [], {'midi', 'cents'}),             0:2};
sigma_ma  = [50 0.3];
r_ma      = [1 1];
groups_ma = [1 2];
isRel_ma  = [false false];
isPer_ma  = [true  false];
period_ma = [1200 0];

% (a) Scalar dispatch unchanged.
s_scalar = cosSimExpTens(p_ref, [], p_qry, [], ...
    sigma_ma, r_ma, isRel_ma, isPer_ma, period_ma, ...
    'verbose', false);
results{end+1,1} = 'cosSimExpTens raw-MA scalar dispatch returns numeric scalar';
results{end,2}   = isnumeric(s_scalar) && isscalar(s_scalar) && isfinite(s_scalar);

% (b) Scalar-vs-list broadcast: matrix-form translateAttributes feed.
offs_rma  = {[-100 0 100 200], [0 1 2 1]};
pm_qry_swept = translateAttributes(p_qry, [], offs_rma);
qry_swept = pm_qry_swept.pAttr;
s_list = cosSimExpTens(p_ref, [], qry_swept, [], ...
    sigma_ma, r_ma, isRel_ma, isPer_ma, period_ma, ...
    'verbose', false);
results{end+1,1} = 'cosSimExpTens raw-MA list returns 1-by-M cell';
results{end,2}   = iscell(s_list) && numel(s_list) == 4 ...
                   && all(cellfun(@(x) isnumeric(x) && isscalar(x) && isfinite(x), ...
                                  s_list));

% (c) Floating-point parity with manual build loop.
dens_ref = buildExpTens(p_ref, [], sigma_ma, r_ma, ...
    isRel_ma, isPer_ma, period_ma, 'verbose', false);
s_manual = zeros(1, numel(qry_swept));
for m = 1:numel(qry_swept)
    dens_q = buildExpTens(qry_swept{m}, [], sigma_ma, r_ma, ...
        isRel_ma, isPer_ma, period_ma, 'verbose', false);
    s_manual(m) = cosSimExpTens(dens_ref, dens_q, 'verbose', false);
end
s_list_num = cell2mat(s_list);
results{end+1,1} = 'cosSimExpTens raw-MA list parity with manual buildExpTens loop';
results{end,2}   = max(abs(s_list_num - s_manual)) < 1e-12;

% (d) Operand order symmetric.
s_rev = cosSimExpTens(qry_swept, [], p_ref, [], ...
    sigma_ma, r_ma, isRel_ma, isPer_ma, period_ma, ...
    'verbose', false);
s_rev_num = cell2mat(s_rev);
results{end+1,1} = 'cosSimExpTens raw-MA list symmetric in operand order';
results{end,2}   = max(abs(s_list_num - s_rev_num)) < 1e-12;

% (e) List-vs-list rejected.
pm_qry_swept_2 = translateAttributes(p_qry, [], {[0 100], [0 0]});
pm_ref_swept   = translateAttributes(p_ref, [], {[0 50], [0 0]});
qry_swept_2 = pm_qry_swept_2.pAttr;
ref_swept   = pm_ref_swept.pAttr;
results{end+1,1} = 'cosSimExpTens raw-MA list-vs-list rejected';
results{end,2}   = throwsErrorWithId( ...
    @() cosSimExpTens(ref_swept, [], qry_swept_2, [], ...
        sigma_ma, r_ma, isRel_ma, isPer_ma, period_ma, ...
        'verbose', false), ...
    'cosSimExpTens:listVsListNotSupported');

% (f) Self-sweep peaks at zero offset.
offs_self = {[-200 -100 0 100 200], [0 0 0 0 0]};
pm_ref_self = translateAttributes(p_ref, [], offs_self);
ref_self  = pm_ref_self.pAttr;
s_self    = cosSimExpTens(p_ref, [], ref_self, [], ...
    sigma_ma, r_ma, isRel_ma, isPer_ma, period_ma, ...
    'verbose', false);
s_self_num = cell2mat(s_self);
[~, iMax]  = max(s_self_num);
results{end+1,1} = 'cosSimExpTens raw-MA list peaks at self-match (offset 0)';
results{end,2}   = iMax == 3 && abs(s_self_num(3) - 1) < 1e-9;


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
    fprintf('\n=== test_maet: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_maet:failed', '%d test(s) failed.', nFail);
    end
end
