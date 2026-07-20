%% test_maet.m — Multi-Attribute Expectation Tensor (MAET, v2.1.0)
%
%  Tests for Multi-Attribute Expectation Tensor (MAET, v2.1.0).
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

% --- Lazy/eager parity (v2.2) ---
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
results{end+1,1} = 'MAET: insufficient slots errors';
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

% -- differenceEvents: moved onto the (pAttr, w, specs) carrier (3c-iv);
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
%    delete_input, three-tuple return). Mirrors Python tests/test_maet.py
%    `test_weight_*` for parity. --

% gamma = 0 limit: pure Gaussian with std = width.
p_we = {[60 62 64 67 72]};
[~, w_we, ~] = weightEvents(p_we, [], 1, 1, 64, 0, 'sd', 3, 'dropInputAttr', false);
expected_we = exp(-(([60 62 64 67 72] - 64) .^ 2) ./ (2 * 3 ^ 2));
results{end+1,1} = 'weightEvents: gamma = 0 is pure Gaussian (std = width)';
results{end,2}   = max(abs(w_we{1} - expected_we)) < 1e-12;

% gamma = 1 limit: pure rectangle with half-width = width * sqrt(3).
[~, w_re, ~] = weightEvents({[60 62 64 67 72]}, [], 1, 1, 64, 1, 'sd', 3, 'dropInputAttr', false);
expected_re = double(abs([60 62 64 67 72] - 64) <= 3 * sqrt(3));
results{end+1,1} = 'weightEvents: gamma = 1 is pure rectangle (half-width = width * sqrt(3))';
results{end,2}   = isequal(w_re{1}, expected_re);

% Peak h(0) = 1 throughout the family.
gammas_peak = [0.05 0.1 0.25 0.5 0.75 0.9 0.95];
peak_ok = true;
for gg = gammas_peak
    [~, w_pk, ~] = weightEvents({5}, [], 1, 1, 5, gg, 'sd', 2, 'dropInputAttr', false);
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
    [~, w_fv, ~] = weightEvents({y_fv}, [], 1, 1, 0, gg, 'sd', width_fv, 'dropInputAttr', false);
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
[p_3t_out, w_3t_out, s_3t_out] = weightEvents(p_3t, [], 1, 1, 1.5, 0, 'sd', 1, 'dropInputAttr', false);
results{end+1,1} = 'weightEvents: returns three-tuple (pAttr, w, specs)';
results{end,2}   = iscell(p_3t_out) && numel(p_3t_out) == 2 && ...
                   iscell(w_3t_out) && numel(w_3t_out) == 2 && ...
                   iscell(s_3t_out) && numel(s_3t_out) == 2 && ...
                   all(cellfun(@isstruct, s_3t_out));

% Non-input, non-target attribute passes its incoming weight through.
p_pt = {[1 2 3], [10 20 30], [100 200 300]};
[~, w_pt, ~] = weightEvents(p_pt, {[], [], 0.5}, 1, 2, 2, 0, 'sd', 1, 'dropInputAttr', false);
results{end+1,1} = 'weightEvents: non-input non-target attribute passes through unchanged';
results{end,2}   = isequal(w_pt{3}, 0.5);

% input ~= target: factor lands on target slot, input slot unchanged.
p_int = {[60 64 67], [0 1 2]};        % pitch (target), time (input)
[~, w_int, ~] = weightEvents(p_int, [], 2, 1, 1, 0, 'sd', 1, 'dropInputAttr', false);
expected_int = exp(-(([0 1 2] - 1) .^ 2) ./ 2);
results{end+1,1} = 'weightEvents: input ~= target writes factor to target slot only';
results{end,2}   = max(abs(w_int{1} - expected_int)) < 1e-12 && isempty(w_int{2});

% Target with K_target > 1: (1, N) factor broadcasts across K_target slots.
p_bc = {[60 64; 62 65; 64 67], [0 1]};   % pitch K=3 (target), time K=1 (input)
w_bc_in = {ones(3, 2), []};
[~, w_bc, ~] = weightEvents(p_bc, w_bc_in, 2, 1, 0, 0, 'sd', 1, 'dropInputAttr', false);
factor_bc = exp(-([0 1] .^ 2) ./ 2);
expected_bc = repmat(factor_bc, 3, 1);
results{end+1,1} = 'weightEvents: (1, N) factor broadcasts across target K_target > 1 slots';
results{end,2}   = isequal(size(w_bc{1}), [3 2]) && ...
                   max(abs(w_bc{1}(:) - expected_bc(:))) < 1e-12;

% Periodic wrap: delta = v - c wrapped to [-P/2, P/2] before h. Raw values intact.
p_per = {[10 11 0 1 2]};
[~, w_per, ~] = weightEvents(p_per, [], 1, 1, 0, 0, 'sd', 2, 'dropInputAttr', false, 'isPer', true, 'period', 12);
expected_per = exp(-([-2 -1 0 1 2] .^ 2) ./ 8);
results{end+1,1} = 'weightEvents: periodic wrap of delta before shape';
results{end,2}   = max(abs(w_per{1} - expected_per)) < 1e-12;
results{end+1,1} = 'weightEvents: input values stay raw (no value mutation)';
results{end,2}   = isequal(p_per{1}, [10 11 0 1 2]);

% Scalar existing weight multiplies in.
[~, w_mul, ~] = weightEvents({[1 2 3]}, 0.5, 1, 1, 2, 0, 'sd', 1, 'dropInputAttr', false);
h_mul = exp(-(([1 2 3] - 2) .^ 2) ./ 2);
results{end+1,1} = 'weightEvents: scalar existing weight multiplies in';
results{end,2}   = max(abs(w_mul{1} - 0.5 .* h_mul)) < 1e-12;

% Sequential composition replaces the old multi-input behaviour: two calls
% on the same target multiply factors. pitch (target), two scaffolding attrs.
p_seq = {[60 64 67], [0 1 2], [0 0.5 1]};
[p_seq1, w_seq1, g_seq1] = weightEvents(p_seq, [], 2, 1, 1, 0, 'sd', 1, 'dropInputAttr', false);
[~, w_seq2, ~] = weightEvents(p_seq1, w_seq1, 3, 1, 0.5, 0, 'sd', 0.5, 'dropInputAttr', false);
h_time_seq = exp(-(([0 1 2] - 1) .^ 2) ./ 2);
h_beat_seq = exp(-(([0 0.5 1] - 0.5) .^ 2) ./ 0.5);
results{end+1,1} = 'weightEvents: sequential composition multiplies factors into target';
results{end,2}   = max(abs(w_seq2{1} - h_time_seq .* h_beat_seq)) < 1e-12;

% delete_input=true drops the input attribute.
p_del = {[60 64 67], [0 1 2]};
[p_del_out, w_del_out, g_del_out] = weightEvents(p_del, [], 2, 1, 1, 0, 'sd', 1, 'dropInputAttr', true);
results{end+1,1} = 'weightEvents: delete_input=true drops the input attribute';
results{end,2}   = numel(p_del_out) == 1 && numel(w_del_out) == 1 && ...
                   isequal(p_del_out{1}, [60 64 67]);

% delete_input with inputAttr > targetAttr: input attribute's value,
% weight, and spec are dropped; the target keeps its output index.
% Input = attr 2, target = attr 1 (input after target).
p_gc = {[1 2], [3 4], [5 6]};
[p_gc_out, w_gc_out, s_gc_out] = weightEvents(p_gc, [], 2, 1, 3.5, 0, 'sd', 1, 'dropInputAttr', true);
factor_gc = exp(-(([3 4] - 3.5) .^ 2) ./ 2);   % from input attr 2 values
results{end+1,1} = 'weightEvents: delete_input (input after target) keeps target index';
results{end,2}   = numel(p_gc_out) == 2 && numel(w_gc_out) == 2 && ...
                   numel(s_gc_out) == 2 && ...
                   isequal(p_gc_out{1}, [1 2]) && isequal(p_gc_out{2}, [5 6]) && ...
                   max(abs(w_gc_out{1} - factor_gc)) < 1e-12;

% delete_input with inputAttr < targetAttr: input attribute is dropped
% and the target shifts down one output index, carrying the factor.
% Input = attr 1, target = attr 3 (input before target).
p_gk = {[1 2], [3 4], [5 6]};
[p_gk_out, w_gk_out, s_gk_out] = weightEvents(p_gk, [], 1, 3, 1.5, 0, 'sd', 1, 'dropInputAttr', true);
factor_gk = exp(-(([1 2] - 1.5) .^ 2) ./ 2);   % from input attr 1 values
results{end+1,1} = 'weightEvents: delete_input (input before target) shifts target index';
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
[~, w_sd_xy,  ~] = weightEvents({linspace(-5, 5, 21)}, [], 1, 1, 0, 1, 'sd', sd_xy, 'dropInputAttr', false);
[~, w_wid_xy, ~] = weightEvents({linspace(-5, 5, 21)}, [], 1, 1, 0, 1, 'width', wid_xy, 'dropInputAttr', false);
results{end+1,1} = 'weightEvents: sd and width yield identical output under conversion';
results{end,2}   = max(abs(w_sd_xy{1} - w_wid_xy{1})) < 1e-12;

% width form: rect of full support L covers half-open [-L/2, L/2); the
% lower edge -L/2 is kept, the upper edge +L/2 is excluded, events just
% past are zeroed (a closed interval over-counts).
L_rect = 1.0;
eps_rect = 1e-6;
t_rect = [-L_rect/2, -L_rect/4, 0, L_rect/4, L_rect/2, L_rect/2 + eps_rect];
[~, w_rect, ~] = weightEvents({t_rect}, [], 1, 1, 0, 1, 'width', L_rect, 'dropInputAttr', false);
results{end+1,1} = 'weightEvents: width gives half-open rectangle of total support width';
results{end,2}   = all(w_rect{1}(1:4) == 1) && w_rect{1}(5) == 0 && w_rect{1}(6) == 0;

% #20: half-open rect window keeps exactly N pulses for full support N on a
% unit grid (a closed interval would give 1, 3, 3, 5, 5 for widths 1..5).
t_grid = 0:8;                       % IOI = 1
rectCounts = zeros(1, 5);
for Wn = 1:5
    [~, w_g, ~] = weightEvents({t_grid, t_grid}, [], 2, 1, 4, 1, ...
                               'width', Wn, 'dropInputAttr', false);
    rectCounts(Wn) = nnz(w_g{1});
end
results{end+1,1} = 'weightEvents: half-open rect width N keeps N pulses (on-pulse centre)';
results{end,2}   = isequal(rectCounts, [1 2 3 4 5]);

% Between-pulse centre keeps one pulse (lower edge), not zero or two.
[~, w_bp, ~] = weightEvents({t_grid, t_grid}, [], 2, 1, 3.5, 1, ...
                            'width', 1, 'dropInputAttr', false);
results{end+1,1} = 'weightEvents: half-open rect between-pulse centre keeps 1 pulse';
results{end,2}   = (nnz(w_bp{1}) == 1);

% #21: an out-of-support rectangular window gives a zero-mass density, and
% renyi2 returns NaN rather than erroring.
[pa_z, wa_z, ~] = weightEvents({[60 62 64], [0 1 2]}, [], 2, 1, 100, 1, ...
                               'width', 1, 'dropInputAttr', false);
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

results{end+1,1} = 'weightEvents: delete_input=true with input==target errors';
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
p_after_t = translateAttributes(p_tw, [], {mu_tw});
[~, w_after_t, ~] = weightEvents(p_after_t, [], 1, 1, c_tw, gamma_tw, 'sd', width_tw, 'dropInputAttr', false);
[~, w_first, ~]   = weightEvents(p_tw, [], 1, 1, c_tw - mu_tw, gamma_tw, 'sd', width_tw, 'dropInputAttr', false);
results{end+1,1} = 'weightEvents: T \circ W centre-shift commutation';
results{end,2}   = max(abs(w_first{1} - w_after_t{1})) < 1e-12;

% translateAttributes moved onto the (pAttr, w, specs) carrier (3c-iv-d);
% its tests now live in tests/test_translate.m. The old groups / isRel /
% isPer / period positional contract has been removed.

% -- cosSimExpTens raw-MA scalar-vs-list mode --

p_ref     = {convertPitch([60 62 64 65 67 69 71], 'midi', 'cents'), 0:6};
p_qry     = {convertPitch([60 64 67], 'midi', 'cents'),             0:2};
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
qry_swept = translateAttributes(p_qry, [], offs_rma);
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
qry_swept_2 = translateAttributes(p_qry, [], {[0 100], [0 0]});
ref_swept   = translateAttributes(p_ref, [], {[0 50], [0 0]});
results{end+1,1} = 'cosSimExpTens raw-MA list-vs-list rejected';
results{end,2}   = throwsErrorWithId( ...
    @() cosSimExpTens(ref_swept, [], qry_swept_2, [], ...
        sigma_ma, r_ma, isRel_ma, isPer_ma, period_ma, ...
        'verbose', false), ...
    'cosSimExpTens:listVsListNotSupported');

% (f) Self-sweep peaks at zero offset.
offs_self = {[-200 -100 0 100 200], [0 0 0 0 0]};
ref_self  = translateAttributes(p_ref, [], offs_self);
s_self    = cosSimExpTens(p_ref, [], ref_self, [], ...
    sigma_ma, r_ma, isRel_ma, isPer_ma, period_ma, ...
    'verbose', false);
s_self_num = cell2mat(s_self);
[~, iMax]  = max(s_self_num);
results{end+1,1} = 'cosSimExpTens raw-MA list peaks at self-match (offset 0)';
results{end,2}   = iMax == 3 && abs(s_self_num(3) - 1) < 1e-9;

% -- windowTensor: basic construction --

pitch_w = [60 62 64 65];    % 1 x 4 events
time_w  = [0  1  2  3];
dens_w = buildExpTens({pitch_w, time_w}, [], ...
    [10 0.1], [1 1], ...
    [false false], [true false], [1200 0], ...
    'lazy', false, 'verbose', false);

spec_w = struct();
spec_w.size = [Inf, 1];
spec_w.mix  = [0, 0];
spec_w.centre = {zeros(1, 1), 0.5};
wmd_w = windowTensor(dens_w, spec_w);
results{end+1,1} = 'windowTensor: returns tagged WindowedMaetDensity';
results{end,2}   = strcmp(wmd_w.tag, 'WindowedMaetDensity');

% -- windowTensor: wide window, centred at context mean, gives
%    cos_sim ~= 1 --
% Under v2.1.0 cross-correlation semantics the query is translated so
% that its effective-space mean moves onto the window centre, so a
% centred window at the context's own mean is the correct analogue of
% the no-window case.

t_mean_w = mean(time_w);
spec_wide = struct('size', [Inf, 1e6], 'mix', [0, 0], ...
                   'centre', {{zeros(1, 1), t_mean_w}});
wmd_wide = windowTensor(dens_w, spec_wide);
s_wide = internal.windowedInnerProduct(dens_w, wmd_wide, false);
s_self = cosSimExpTens(dens_w, dens_w, 'verbose', false);
results{end+1,1} = 'windowTensor: wide centred window == unwindowed self-sim';
results{end,2}   = abs(s_wide - s_self) < 1e-3;

% -- windowTensor: infinite size on all groups == identity --

spec_inf = struct('size', [Inf, Inf], 'mix', [0, 0]);
wmd_inf = windowTensor(dens_w, spec_inf);
s_inf = internal.windowedInnerProduct(dens_w, wmd_inf, false);
results{end+1,1} = 'windowTensor: all-Inf size == identity (s ~= 1)';
results{end,2}   = abs(s_inf - 1) < 1e-6;

% -- windowTensor: narrow window reduces cos_sim --

spec_narrow = struct('size', [Inf, 0.2], 'mix', [0, 0], ...
                     'centre', {{zeros(1, 1), 0}});
wmd_narrow = windowTensor(dens_w, spec_narrow);
s_narrow = internal.windowedInnerProduct(dens_w, wmd_narrow, false);
results{end+1,1} = 'windowTensor: narrow window reduces cos_sim';
results{end,2}   = s_narrow < 0.5;

% -- windowTensor: rectangular window on 1-D time works --

spec_rect = struct('size', [Inf, 0.5], 'mix', [0, 1], ...
                   'centre', {{zeros(1, 1), 1.0}});
wmd_rect = windowTensor(dens_w, spec_rect);
s_rect = internal.windowedInnerProduct(dens_w, wmd_rect, false);
results{end+1,1} = 'windowTensor: rectangular 1-D time works (finite, 0<s<1)';
results{end,2}   = isfinite(s_rect) && s_rect > 0 && s_rect < 1;

% -- windowTensor: raised-rectangular window on 1-D time works --

spec_raised = struct('size', [Inf, 0.5], 'mix', [0, 0.5], ...
                     'centre', {{zeros(1, 1), 1.0}});
wmd_raised = windowTensor(dens_w, spec_raised);
s_raised = internal.windowedInnerProduct(dens_w, wmd_raised, false);
results{end+1,1} = 'windowTensor: raised-rectangular 1-D time works';
results{end,2}   = isfinite(s_raised) && s_raised > 0 && s_raised < 1;

% -- windowTensor: multi-D relative Gaussian works --

pitchMR = [60 62; 64 65; 67 69];   % 3 slots, 2 events
dens_mr = buildExpTens({pitchMR}, [], 10, 3, ...
    true, true, 1200, 'verbose', false);
spec_mr_gauss = struct('size', 1, 'mix', 0, ...
                       'centre', {{[50; 100]}});
wmd_mr = windowTensor(dens_mr, spec_mr_gauss);
s_mr = internal.windowedInnerProduct(dens_mr, wmd_mr, false);
results{end+1,1} = 'windowTensor: multi-D rel Gaussian window works';
results{end,2}   = isfinite(s_mr) && s_mr >= 0 && s_mr <= 1;

% -- windowTensor: multi-D relative rectangular raises --

spec_mr_rect = struct('size', 1, 'mix', 1, 'centre', {{[50; 100]}});
wmd_mr_rect = windowTensor(dens_mr, spec_mr_rect);
results{end+1,1} = 'windowTensor: multi-D rel rectangular errors';
results{end,2}   = throwsError(@() internal.windowedInnerProduct(dens_mr, wmd_mr_rect, false));

% -- windowTensor: multi-D relative raised-rect raises --

spec_mr_rr = struct('size', 1, 'mix', 0.5, 'centre', {{[50; 100]}});
wmd_mr_rr = windowTensor(dens_mr, spec_mr_rr);
results{end+1,1} = 'windowTensor: multi-D rel raised-rect errors';
results{end,2}   = throwsError(@() internal.windowedInnerProduct(dens_mr, wmd_mr_rr, false));

% -- windowTensor: entropy on rect-windowed multi-D rel works --

H_mr_rect = entropyExpTens(wmd_mr_rect, 'nPointsPerDim', 20, 'verbose', false);
results{end+1,1} = 'windowTensor: entropy on rect-windowed multi-D rel runs';
results{end,2}   = isfinite(H_mr_rect);

% -- windowTensor: narrower window yields lower entropy --

H_base = entropyExpTens(dens_w, 'xMin', [0, -1], 'xMax', [1200, 4], ...
                         'nPointsPerDim', 60, 'verbose', false);
spec_ew = struct('size', [Inf, 0.3], 'mix', [0, 0], ...
                 'centre', {{zeros(1, 1), 1.0}});
wmd_ew = windowTensor(dens_w, spec_ew);
H_narrow = entropyExpTens(wmd_ew, 'xMin', [0, -1], 'xMax', [1200, 4], ...
                           'nPointsPerDim', 60, 'verbose', false);
results{end+1,1} = 'windowTensor: narrower window => lower entropy';
results{end,2}   = H_narrow < H_base;

% -- windowedTensorSimilarity: profile peaks at matching event offset --

pitch_narrow = [60 62 64 65];
time_narrow  = [0  1  2  3];
ctx_narrow = buildExpTens({pitch_narrow, time_narrow}, [], ...
    [0.5 0.1], [1 1], [false false], [true false], [1200 0], ...
    'verbose', false);

% Fixed single-event query at pitch 62, time 0 (centroid at t=0).
q_sw = buildExpTens({62, 0}, [], ...
    [0.5 0.1], [1 1], [false false], [true false], [1200 0], ...
    'verbose', false);

M_sweep = 21;
offs_sw = linspace(-0.5, 3.5, M_sweep);
offsets_sw = zeros(2, M_sweep);
offsets_sw(2, :) = offs_sw;
spec_sw = struct('size', [Inf, 0.3], 'mix', [0, 0]);
profile = windowedTensorSimilarity(ctx_narrow, q_sw, spec_sw, offsets_sw, ...
    'verbose', false);
[~, peak_idx] = max(profile);
peak_off = offs_sw(peak_idx);
% Query centroid is at t=0, so offset 1 corresponds to the pitch-62
% context event at absolute t=1.
results{end+1,1} = 'windowedTensorSimilarity: profile peaks at matching event offset';
results{end,2}   = abs(peak_off - 1.0) < 0.3;

% -- windowedTensorSimilarity: returns length-M profile --

offsets_vec = zeros(2, 7);
offsets_vec(2, :) = linspace(0, 1, 7);
spec_lm = struct('size', [Inf, 0.5], 'mix', [0, 0]);
prof_lm = windowedTensorSimilarity(dens_w, dens_w, spec_lm, offsets_vec, 'verbose', false);
results{end+1,1} = 'windowedTensorSimilarity: output is 1 x M';
results{end,2}   = isequal(size(prof_lm), [1, 7]);

% -- windowedTensorSimilarity: truncationSigmas / kernelPrecision threaded --
% v2.2.x: replaces the v2.2.0 mptDefaults stop-gap. Explicit Inf
% truncation + double precision must produce identical results to
% the default call; tight finite truncation must match the default
% to numerical precision.
prof_default_thread = windowedTensorSimilarity(dens_w, dens_w, spec_lm, ...
    offsets_vec, 'verbose', false);
prof_inf_thread = windowedTensorSimilarity(dens_w, dens_w, spec_lm, ...
    offsets_vec, 'truncationSigmas', Inf, ...
    'kernelPrecision', 'double', 'verbose', false);
results{end+1,1} = 'windowedTensorSimilarity: explicit Inf/double matches default';
results{end,2}   = isequal(prof_default_thread, prof_inf_thread);

prof_trunc_thread = windowedTensorSimilarity(dens_w, dens_w, spec_lm, ...
    offsets_vec, 'truncationSigmas', 6, 'verbose', false);
results{end+1,1} = 'windowedTensorSimilarity: truncationSigmas=6 matches default to 1e-12';
results{end,2}   = all(abs(prof_default_thread - prof_trunc_thread) < 1e-12);

clear prof_default_thread prof_inf_thread prof_trunc_thread

% -- windowedTensorSimilarity: reference=[] (default) matches omitted reference --
%
% Explicit empty reference must reproduce the default path byte-for-byte.
q_ref     = dens_w;
ctx_ref   = dens_w;
offs_ref  = zeros(2, 11);
offs_ref(2, :) = linspace(-0.5, 1.5, 11);
spec_ref  = struct('size', [Inf, 0.3], 'mix', [0, 0]);
prof_default  = windowedTensorSimilarity(ctx_ref, q_ref, spec_ref, offs_ref, ...
                               'verbose', false);
prof_explicit = windowedTensorSimilarity(ctx_ref, q_ref, spec_ref, offs_ref, ...
                               'reference', [], 'verbose', false);
results{end+1,1} = 'windowedTensorSimilarity: reference=[] == default';
results{end,2}   = max(abs(prof_default - prof_explicit)) < 1e-12;

% -- windowedTensorSimilarity: supplied reference shifts the profile --
%
% Set the time-attribute reference to (default + 0.2 s); the resulting
% profile at offset o must equal the default profile at offset o + 0.2
% (for offsets where both fall on the sweep grid).
muA_pitch = mean(q_ref.Centres{1}, 2);
muA_time  = mean(q_ref.Centres{2}, 2);
ref_shift = { muA_pitch, muA_time + 0.2 };
M_sh      = 21;
offs_sh   = zeros(2, M_sh);
off_t_sh  = linspace(-1.0, 3.0, M_sh);
offs_sh(2, :) = off_t_sh;
prof_d  = windowedTensorSimilarity(ctx_ref, q_ref, spec_ref, offs_sh, ...
                         'verbose', false);
prof_sh = windowedTensorSimilarity(ctx_ref, q_ref, spec_ref, offs_sh, ...
                         'reference', ref_shift, 'verbose', false);
% Check: prof_sh(m) should equal prof_d at offset off_t_sh(m) + 0.2
ok_shift = true;
for m_idx = 1:M_sh
    target = off_t_sh(m_idx) + 0.2;
    [dmin, jj] = min(abs(off_t_sh - target));
    if dmin < 1e-9
        if abs(prof_sh(m_idx) - prof_d(jj)) > 1e-10
            ok_shift = false;
            break;
        end
    end
end
results{end+1,1} = 'windowedTensorSimilarity: reference shifts profile by offset';
results{end,2}   = ok_shift;

% -- windowedTensorSimilarity: bad reference shape errors --
results{end+1,1} = 'windowedTensorSimilarity: reference wrong cell count errors';
results{end,2}   = throwsError(@() windowedTensorSimilarity(ctx_ref, q_ref, ...
    spec_ref, offs_ref, 'reference', {muA_pitch}, 'verbose', false));

results{end+1,1} = 'windowedTensorSimilarity: reference wrong length errors';
results{end,2}   = throwsError(@() windowedTensorSimilarity(ctx_ref, q_ref, ...
    spec_ref, offs_ref, 'reference', {muA_pitch, [0; 0]}, 'verbose', false));

% -- windowedTensorSimilarity: periodic windowing (wrapped-Gaussian) --
%
% For periodic groups, the window is the wrapped Gaussian (or wrapped
% rect-conv-Gaussian for mix > 0): the sum of line-case window
% functions at all periodic images of the centre. The toolbox sums
% these adaptively until the latest image-pair's contribution falls
% below 1e-12 of the running maximum. The
% windowedTensorSimilarity:periodicWindowApprox warning of pre-v2.2 has
% been removed because there is no longer an approximation to warn
% about. See User Guide §3.1 "Post-tensor windowing".

offs_off = [zeros(1, 5); linspace(0, 1, 5)];

% (a) A periodic windowed group must not emit
% windowedTensorSimilarity:periodicWindowApprox (the warning class has
% been removed).
spec_small = struct('size', [5, 0.3], 'mix', [0, 0]);
W = warning('error', 'windowedTensorSimilarity:periodicWindowApprox');
no_warn_periodic = true;
try
    windowedTensorSimilarity(dens_w, dens_w, spec_small, offs_off, 'verbose', false);
catch ME
    no_warn_periodic = ~strcmp(ME.identifier, ...
        'windowedTensorSimilarity:periodicWindowApprox');
end
warning(W);
results{end+1,1} = 'windowedTensorSimilarity: no periodic-approx warning (v2.2)';
results{end,2}   = no_warn_periodic;

% (b) eval_exp_tens on a windowed periodic density returns identical
% values at periodic-equivalent query points (X, X+P, X-P): under the
% pre-v2.2 line-case window this was broken; the wrapped Gaussian
% restores periodic-equivalence to FP precision.
P_test = 12;
sigma_test = 1;
pitches_test = [3, 7];
dens_per = buildExpTens({pitches_test}, [], sigma_test, 1, false, true, P_test, 'verbose', false);
spec_eval = struct('size', 3, 'mix', 0, 'centre', {{2}});
wmd_eval = windowTensor(dens_per, spec_eval);
v0      = evalExpTens(wmd_eval, 0.5);
v_plus  = evalExpTens(wmd_eval, 0.5 + P_test);
v_minus = evalExpTens(wmd_eval, 0.5 - P_test);
tol_eval = 1e-10 * max(abs([v0, v_plus, v_minus]));
results{end+1,1} = 'evalExpTens: periodic representative equivalence';
results{end,2}   = abs(v0 - v_plus) <= tol_eval && abs(v0 - v_minus) <= tol_eval;

% -- windowTensor: shape-validation errors --

results{end+1,1} = 'windowTensor: bad size length errors';
results{end,2}   = throwsError(@() windowTensor(dens_w, ...
    struct('size', [1 1 1], 'mix', [0 0])));

results{end+1,1} = 'windowTensor: mix out of range errors';
results{end,2}   = throwsError(@() windowTensor(dens_w, ...
    struct('size', [1 1], 'mix', [0 1.5])));

results{end+1,1} = 'windowTensor: wrong centre length errors';
bad_spec = struct('size', [1 1], 'mix', [0 0]);
bad_spec.centre = {zeros(1,1)};   % length 1 cell, need A = 2
results{end,2}   = throwsError(@() windowTensor(dens_w, bad_spec));

% -- windowTensor: scalar centre broadcasting --
% A size-1 centre input (numeric scalar, 1x1 array, or single-element
% cell containing a scalar) broadcasts to fill every per-attribute
% slot uniformly. Mirrors the Python window_tensor behaviour.

% Build a small MA density: A=2, both attributes r=1, separate groups.
% dim_per_attr = [1, 1], dim_total = 2.
% For these tests we use the existing dens_w (pitch r=1 + time r=1).
ref_centre_cell = {3.0, 3.0};   % equivalent uniform centre per attribute
spec_ref = struct('size', [1, 0.5], 'mix', [0, 0], 'centre', {ref_centre_cell});
wmd_ref = windowTensor(dens_w, spec_ref);
cos_ref = internal.windowedInnerProduct(dens_w, wmd_ref, false);

% Numeric scalar.
spec_scalar = struct('size', [1, 0.5], 'mix', [0, 0], 'centre', 3.0);
wmd_s = windowTensor(dens_w, spec_scalar);
cos_s  = internal.windowedInnerProduct(dens_w, wmd_s, false);
results{end+1,1} = 'windowTensor: numeric scalar centre broadcasts';
results{end,2}   = isequal(wmd_s.centre, wmd_ref.centre) && cos_s == cos_ref;

% 1x1 array.
spec_1x1 = struct('size', [1, 0.5], 'mix', [0, 0], 'centre', [3.0]);
wmd_1x1 = windowTensor(dens_w, spec_1x1);
cos_1x1 = internal.windowedInnerProduct(dens_w, wmd_1x1, false);
results{end+1,1} = 'windowTensor: 1x1 array centre broadcasts';
results{end,2}   = isequal(wmd_1x1.centre, wmd_ref.centre) && cos_1x1 == cos_ref;

% Length-1 cell does NOT broadcast — preserves the pre-fix contract
% that a wrong-length cell raises. This complements the existing test
% 'windowTensor: wrong centre length errors' above.
results{end+1,1} = 'windowTensor: length-1 cell on A>1 still errors (no broadcast)';
results{end,2}   = throwsError(@() windowTensor(dens_w, ...
    struct('size', [1, 0.5], 'mix', [0, 0], 'centre', {{3.0}})));

% Zero scalar.
spec_zero = struct('size', [1, 0.5], 'mix', [0, 0], 'centre', 0);
wmd_z = windowTensor(dens_w, spec_zero);
results{end+1,1} = 'windowTensor: zero scalar centre broadcasts';
results{end,2}   = isequal(wmd_z.centre{1}, 0) && isequal(wmd_z.centre{2}, 0);

% Negative scalar.
spec_neg = struct('size', [1, 0.5], 'mix', [0, 0], 'centre', -7.5);
wmd_n = windowTensor(dens_w, spec_neg);
results{end+1,1} = 'windowTensor: negative scalar centre broadcasts';
results{end,2}   = isequal(wmd_n.centre{1}, -7.5) && isequal(wmd_n.centre{2}, -7.5);

% Multi-D scalar broadcasting: r=3 absolute single attribute, d_a = 3.
% A scalar should fill all three slots.
pitchMA3 = [60 62 64; 67 69 71; 72 74 76];   % 3 slots, 3 events
dens_ma3 = buildExpTens({pitchMA3}, [], 10, 3, ...
    false, false, 0, 'verbose', false);
spec_sc3 = struct('size', 1.5, 'mix', 0.5, 'centre', 5.0);
wmd_sc3 = windowTensor(dens_ma3, spec_sc3);
results{end+1,1} = 'windowTensor: scalar broadcast on r=3 absolute (d_a=3)';
results{end,2}   = isequal(wmd_sc3.centre{1}, [5.0; 5.0; 5.0]);

% Compare scalar broadcast vs explicit uniform cell — must be byte-equal.
spec_ref3 = struct('size', 1.5, 'mix', 0.5, 'centre', {{[5.0; 5.0; 5.0]}});
wmd_ref3 = windowTensor(dens_ma3, spec_ref3);
results{end+1,1} = 'windowTensor: scalar broadcast == explicit uniform cell';
results{end,2}   = isequal(wmd_sc3.centre, wmd_ref3.centre) && ...
    internal.windowedInnerProduct(dens_ma3, wmd_sc3, false) == ...
    internal.windowedInnerProduct(dens_ma3, wmd_ref3, false);

% Per-attribute scalar list NOT broadcast. With A=2 and dim_total=2, the
% length-2 numeric vector [5; 10] is a valid flat-form input — and is
% accepted as such, NOT as per-attribute scalar broadcast. This is the
% intended behaviour: the scalar-broadcast bypass is reserved for
% size-1 inputs only.
spec_flat2 = struct('size', [1, 0.5], 'mix', [0, 0], 'centre', [5; 10]);
wmd_f2 = windowTensor(dens_w, spec_flat2);
results{end+1,1} = 'windowTensor: length-2 vector parsed as flat form, not broadcast';
results{end,2}   = isequal(wmd_f2.centre{1}, 5) && isequal(wmd_f2.centre{2}, 10);

% Wrong-length still errors informatively.
results{end+1,1} = 'windowTensor: wrong-length flat input errors';
results{end,2}   = throwsError(@() windowTensor(dens_ma3, ...
    struct('size', 1.5, 'mix', 0.5, 'centre', [1; 2; 3; 4])));

% -- cosSimExpTens windowed: within-attribute centre symmetrisation --
% The MAET density is symmetric under permutations of components within
% each attribute's effective coordinates (JMM windowing-theorem remark
% on within-attribute symmetry of the windowed integral). The integral
% therefore depends on the within-attribute centre components only
% through their multiset. The toolbox uses a perm-comb summation form
% internally; the v2.1 fix detects non-uniform within-attribute centres
% and averages the IP over their within-attribute permutations to
% restore framework-correct behaviour.
%
% This block mirrors the Python tests/test_windowed_within_attr_
% symmetrisation.py: 5 logical tests, expanded by parametrisation to
% 42 result rows, sharing the local helper functions
% directWindowedCosineSingleMultiset and toolboxWindowedCosineSingleMultiset defined at the
% end of this file.

sigma_sym = 30.0;
K_sym = 6;
sm_grid = {[5.0, 0.0], [3.0, 1.0], [4.0, 0.5]};
cp_grid = {'uniform', 'non_uniform_small_spread', 'non_uniform_large_spread'};
ORTH_FLOOR_SYM = 1e-6;
RTOL_SYM = 1e-9;

% --- 1. Toolbox cosine matches direct enumeration (perm-perm) ---
% Mirrors test_toolbox_matches_direct_for_non_uniform_centre. 2 × 3 × 3 = 18 rows.
for r_sym = [2, 3]
    for sm_idx = 1:numel(sm_grid)
        size_v = sm_grid{sm_idx}(1);
        mix_v  = sm_grid{sm_idx}(2);
        for cp_idx = 1:numel(cp_grid)
            cp = cp_grid{cp_idx};
            rng(7919*r_sym + 31*sm_idx + cp_idx);
            p_a = (rand(K_sym, 1) * 2 - 1) * 200;
            w_a = 0.5 + rand(K_sym, 1);
            p_b = (rand(K_sym, 1) * 2 - 1) * 200;
            w_b = 0.5 + rand(K_sym, 1);
            switch cp
                case 'uniform'
                    offset_vec = repmat(50.0, r_sym, 1);
                case 'non_uniform_small_spread'
                    offset_vec = linspace(-30.0, 30.0, r_sym).';
                case 'non_uniform_large_spread'
                    offset_vec = linspace(-100.0, 100.0, r_sym).';
            end
            cos_d = directWindowedCosineSingleMultiset(p_a, w_a, p_b, w_b, ...
                sigma_sym, r_sym, offset_vec, size_v, mix_v);
            cos_t = toolboxWindowedCosineSingleMultiset(p_a, w_a, p_b, w_b, ...
                sigma_sym, r_sym, offset_vec, size_v, mix_v);
            results{end+1,1} = sprintf( ...
                'symmetrisation: toolbox==direct r=%d (s,m)=(%g,%g) %s', ...
                r_sym, size_v, mix_v, cp);
            if abs(cos_d) < ORTH_FLOOR_SYM && abs(cos_t) < ORTH_FLOOR_SYM
                results{end,2} = true;
            else
                results{end,2} = abs(cos_t - cos_d) <= ...
                    RTOL_SYM * max(abs(cos_d), abs(cos_t));
            end
        end
    end
end

% --- 2. Cosine invariant under context-side input reordering ---
% Mirrors test_windowed_cosine_invariant_under_context_reorder.
% 2 × 3 × 3 = 18 rows; each row checks reversed AND shuffled orderings.
for r_sym = [2, 3]
    for sm_idx = 1:numel(sm_grid)
        size_v = sm_grid{sm_idx}(1);
        mix_v  = sm_grid{sm_idx}(2);
        for cp_idx = 1:numel(cp_grid)
            cp = cp_grid{cp_idx};
            rng(13591*r_sym + 41*sm_idx + 17*cp_idx + 23);
            p_a = (rand(K_sym, 1) * 2 - 1) * 200;
            w_a = 0.5 + rand(K_sym, 1);
            p_b = (rand(K_sym, 1) * 2 - 1) * 200;
            w_b = 0.5 + rand(K_sym, 1);
            switch cp
                case 'uniform'
                    offset_vec = repmat(30.0, r_sym, 1);
                case 'non_uniform_small_spread'
                    offset_vec = linspace(-25.0, 25.0, r_sym).';
                case 'non_uniform_large_spread'
                    offset_vec = linspace(-100.0, 100.0, r_sym).';
            end
            cos_orig = toolboxWindowedCosineSingleMultiset(p_a, w_a, p_b, w_b, ...
                sigma_sym, r_sym, offset_vec, size_v, mix_v);
            % Reverse ordering of the context-side values.
            cos_rev = toolboxWindowedCosineSingleMultiset(p_a, w_a, ...
                p_b(end:-1:1), w_b(end:-1:1), ...
                sigma_sym, r_sym, offset_vec, size_v, mix_v);
            % Random shuffle.
            perm_b = randperm(K_sym);
            cos_shuf = toolboxWindowedCosineSingleMultiset(p_a, w_a, ...
                p_b(perm_b), w_b(perm_b), ...
                sigma_sym, r_sym, offset_vec, size_v, mix_v);
            results{end+1,1} = sprintf( ...
                'symmetrisation: context reorder invariant r=%d (s,m)=(%g,%g) %s', ...
                r_sym, size_v, mix_v, cp);
            if abs(cos_orig) < ORTH_FLOOR_SYM
                results{end,2} = abs(cos_rev) < ORTH_FLOOR_SYM ...
                    && abs(cos_shuf) < ORTH_FLOOR_SYM;
            else
                ok_rev = abs(cos_rev - cos_orig) <= ...
                    RTOL_SYM * max(abs(cos_orig), abs(cos_rev));
                ok_shuf = abs(cos_shuf - cos_orig) <= ...
                    RTOL_SYM * max(abs(cos_orig), abs(cos_shuf));
                results{end,2} = ok_rev && ok_shuf;
            end
        end
    end
end

% --- 3. Cosine invariant under permutation of centre entries ---
% Mirrors test_windowed_cosine_invariant_under_centre_permutation.
% 2 × 2 = 4 rows; each row checks all r_sym! permutations.
sm_grid_3 = {[5.0, 0.0], [3.0, 1.0]};
for r_sym = [2, 3]
    for sm_idx = 1:numel(sm_grid_3)
        size_v = sm_grid_3{sm_idx}(1);
        mix_v  = sm_grid_3{sm_idx}(2);
        rng(50261*r_sym + 91*sm_idx + 7);
        p_a = (rand(K_sym, 1) * 2 - 1) * 200;
        w_a = 0.5 + rand(K_sym, 1);
        p_b = (rand(K_sym, 1) * 2 - 1) * 200;
        w_b = 0.5 + rand(K_sym, 1);
        offset_vec = linspace(-50.0, 50.0, r_sym).';
        cos_orig = toolboxWindowedCosineSingleMultiset(p_a, w_a, p_b, w_b, ...
            sigma_sym, r_sym, offset_vec, size_v, mix_v);
        all_perms_3 = perms(1:r_sym);
        ok_all = true;
        for ip_row = 1:size(all_perms_3, 1)
            offset_perm = offset_vec(all_perms_3(ip_row, :));
            cos_pi = toolboxWindowedCosineSingleMultiset(p_a, w_a, p_b, w_b, ...
                sigma_sym, r_sym, offset_perm, size_v, mix_v);
            if abs(cos_orig) < ORTH_FLOOR_SYM
                if abs(cos_pi) >= ORTH_FLOOR_SYM
                    ok_all = false; break;
                end
            else
                if abs(cos_pi - cos_orig) > ...
                        RTOL_SYM * max(abs(cos_orig), abs(cos_pi))
                    ok_all = false; break;
                end
            end
        end
        results{end+1,1} = sprintf( ...
            'symmetrisation: centre permutation invariant r=%d (s,m)=(%g,%g)', ...
            r_sym, size_v, mix_v);
        results{end,2} = ok_all;
    end
end

% --- 4. Multi-attribute case: per-attribute symmetrisation independence ---
% Mirrors test_multi_attribute_within_attribute_symmetrisation. 1 row,
% inner check covering attr-0-reversed AND attr-1-reversed.
rng(0);
K_a4 = 4; K_b4 = 4;
p0_x = (rand(K_a4, 1) * 2 - 1) * 200; w0_x = 0.5 + rand(K_a4, 1);
p1_x = (rand(K_b4, 1) * 2 - 1) * 200; w1_x = 0.5 + rand(K_b4, 1);
p0_y = (rand(K_a4, 1) * 2 - 1) * 200; w0_y = 0.5 + rand(K_a4, 1);
p1_y = (rand(K_b4, 1) * 2 - 1) * 200; w1_y = 0.5 + rand(K_b4, 1);
dens_x_4 = buildExpTens({p0_x, p1_x}, {w0_x, w1_x}, [30.0 30.0], [2 2], ...
    [false false], [false false], [0.0 0.0], 'verbose', false);
dens_y_4 = buildExpTens({p0_y, p1_y}, {w0_y, w1_y}, [30.0 30.0], [2 2], ...
    [false false], [false false], [0.0 0.0], 'verbose', false);
centre_a0 = [10; 50];   centre_a1 = [-30; 20];
spec_orig_4 = struct('size', 5.0, 'mix', 0.0, ...
    'centre', {{centre_a0, centre_a1}});
spec_a0_rev_4 = struct('size', 5.0, 'mix', 0.0, ...
    'centre', {{centre_a0(end:-1:1), centre_a1}});
spec_a1_rev_4 = struct('size', 5.0, 'mix', 0.0, ...
    'centre', {{centre_a0, centre_a1(end:-1:1)}});
cos_orig_4 = internal.windowedInnerProduct(dens_x_4, ...
    windowTensor(dens_y_4, spec_orig_4), false);
cos_a0_rev = internal.windowedInnerProduct(dens_x_4, ...
    windowTensor(dens_y_4, spec_a0_rev_4), false);
cos_a1_rev = internal.windowedInnerProduct(dens_x_4, ...
    windowTensor(dens_y_4, spec_a1_rev_4), false);
if abs(cos_orig_4) < ORTH_FLOOR_SYM
    ok_a0 = abs(cos_a0_rev) < ORTH_FLOOR_SYM;
    ok_a1 = abs(cos_a1_rev) < ORTH_FLOOR_SYM;
else
    ok_a0 = abs(cos_a0_rev - cos_orig_4) <= ...
        RTOL_SYM * max(abs(cos_orig_4), abs(cos_a0_rev));
    ok_a1 = abs(cos_a1_rev - cos_orig_4) <= ...
        RTOL_SYM * max(abs(cos_orig_4), abs(cos_a1_rev));
end
results{end+1,1} = 'symmetrisation: multi-attribute per-attribute symmetry';
results{end,2} = ok_a0 && ok_a1;

% --- 5. Uniform-c regression: byte-identical to v2.1 fast path ---
% Mirrors test_uniform_centre_byte_identical_to_old_path. 1 row.
rng(0);
K_5 = 5;
p_a5 = (rand(K_5, 1) * 2 - 1) * 200; w_a5 = 0.5 + rand(K_5, 1);
p_b5 = (rand(K_5, 1) * 2 - 1) * 200; w_b5 = 0.5 + rand(K_5, 1);
offset_unif = repmat(25.0, 2, 1);
cos_t_5 = toolboxWindowedCosineSingleMultiset(p_a5, w_a5, p_b5, w_b5, ...
    30.0, 2, offset_unif, 5.0, 0.0);
cos_d_5 = directWindowedCosineSingleMultiset(p_a5, w_a5, p_b5, w_b5, ...
    30.0, 2, offset_unif, 5.0, 0.0);
results{end+1,1} = 'symmetrisation: uniform centre toolbox==direct (regression)';
results{end,2} = abs(cos_t_5 - cos_d_5) <= 1e-12 * max(abs(cos_d_5), abs(cos_t_5));


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


%% ---- Local helpers (windowed symmetrisation reference) ----

function f = perAxisF(mu_shift, a_rect, b_conv, sigma_pair)
%PERAXISF  Closed-form per-axis windowed factor for the rect-conv-
%Gaussian family. Mirrors localWindowedFactorisable's per-axis form.
    sigma_t = sqrt(sigma_pair^2 + b_conv^2);
    if a_rect == 0 && b_conv > 0
        f = (b_conv / sigma_t) * exp(-mu_shift^2 / (2 * sigma_t^2));
    elseif b_conv == 0 && a_rect > 0
        d = sigma_t * sqrt(2);
        f = 0.5 * (erf((mu_shift + a_rect) / d) ...
                  - erf((mu_shift - a_rect) / d));
    else
        d = sigma_t * sqrt(2);
        num = erf((mu_shift + a_rect) / d) ...
            - erf((mu_shift - a_rect) / d);
        f = num / (2 * erf(a_rect / (b_conv * sqrt(2))));
    end
end

function P = orderedPermutations(K, r)
%ORDEREDPERMUTATIONS  All ordered r-tuples drawn without replacement
%from {1, ..., K}. Output is M-by-r where M = K!/(K-r)!.
    if r == 0
        P = zeros(1, 0);
        return;
    end
    if r > K
        P = zeros(0, r);
        return;
    end
    combos = nchoosek(1:K, r);                 % nchoosek(K,r) x r
    nC = size(combos, 1);
    P = zeros(nC * factorial(r), r);
    row = 1;
    for i = 1:nC
        ps = perms(combos(i, :));               % r! x r
        nP = size(ps, 1);
        P(row : row + nP - 1, :) = ps;
        row = row + nP;
    end
end

function ip = directWindowedCrossCorr(p_a, w_a, p_b, w_b, sigma, r, ...
        offset_vec, mu_q, a_rect, b_conv)
%DIRECTWINDOWEDCROSSCORR  Framework-correct (perm-perm) reference for
%the windowed cross-correlation IP, in the toolbox's translated form
%(kernel_shift = offset - mu_q, F-centre = (mu_q + offset)/2).
    K_a = numel(p_a); K_b = numel(p_b);
    sigma_pair = sigma / sqrt(2);
    Tperm_a = orderedPermutations(K_a, r);
    Tperm_b = orderedPermutations(K_b, r);
    total = 0;
    for ia = 1:size(Tperm_a, 1)
        ta = Tperm_a(ia, :);
        for ib = 1:size(Tperm_b, 1)
            tb = Tperm_b(ib, :);
            contrib = 1;
            for l = 1:r
                d = (p_a(ta(l)) - p_b(tb(l))) ...
                  + (offset_vec(l) - mu_q);
                K_l = exp(-d^2 / (4 * sigma^2));
                f_centre = (mu_q + offset_vec(l)) / 2;
                mid = (p_a(ta(l)) + p_b(tb(l))) / 2;
                F_l = perAxisF(mid - f_centre, a_rect, b_conv, sigma_pair);
                contrib = contrib * K_l * F_l ...
                        * w_a(ta(l)) * w_b(tb(l));
            end
            total = total + contrib;
        end
    end
    ip = total * (sigma * sqrt(pi))^r;
end

function ip = directUnwindowedAbs(p, w, sigma, r)
%DIRECTUNWINDOWEDABS  Direct (perm-perm) enumeration of the unwindowed
%absolute non-periodic IP, used for self-norms in the cosine.
    K = numel(p);
    Tperm = orderedPermutations(K, r);
    total = 0;
    for ia = 1:size(Tperm, 1)
        ta = Tperm(ia, :);
        for ib = 1:size(Tperm, 1)
            tb = Tperm(ib, :);
            contrib = 1;
            for l = 1:r
                d = p(ta(l)) - p(tb(l));
                K_l = exp(-d^2 / (4 * sigma^2));
                contrib = contrib * K_l * w(ta(l)) * w(tb(l));
            end
            total = total + contrib;
        end
    end
    ip = total * (sigma * sqrt(pi))^r;
end

function c = directWindowedCosineSingleMultiset(p_a, w_a, p_b, w_b, sigma, r, ...
        offset_vec, size_v, mix_v)
%DIRECTWINDOWEDCOSINESINGLEMULTISET  Framework-correct windowed similarity for the single multiset
%case, via direct perm-perm enumeration of the cross-correlation IP,
%under normaliser (i): divide by <f_a, f_a> (the query's unwindowed
%self inner product), not by sqrt(<f_a, f_a> * <f_b, f_b>).
    a_rect = size_v * sigma * sqrt(3 * mix_v);
    b_conv = size_v * sigma * sqrt(1 - mix_v);
    mu_q = mean(p_a);
    ip_xy = directWindowedCrossCorr(p_a, w_a, p_b, w_b, sigma, r, ...
        offset_vec, mu_q, a_rect, b_conv);
    % Normaliser (i): divide by the query's own unwindowed self inner
    % product. f_a is the unwindowed query here (the toolbox passes
    % dens_q = built from p_a).
    ip_qq = directUnwindowedAbs(p_a, w_a, sigma, r);
    c = ip_xy / ip_qq;
end

function c = toolboxWindowedCosineSingleMultiset(p_a, w_a, p_b, w_b, sigma, r, ...
        offset_vec, size_v, mix_v)
%TOOLBOXWINDOWEDCOSINESINGLEMULTISET  Toolbox-API windowed cosine for the single multiset
%case, used as the path under test in the symmetrisation suite.
    Pa = p_a(:);  Wa = w_a(:);
    Pb = p_b(:);  Wb = w_b(:);
    dens_q = buildExpTens({Pa}, {Wa}, sigma, r, ...
        false, false, 0, 'verbose', false);
    dens_c = buildExpTens({Pb}, {Wb}, sigma, r, ...
        false, false, 0, 'verbose', false);
    spec = struct('size', size_v, 'mix', mix_v, ...
        'centre', {{offset_vec(:)}});
    wmd = windowTensor(dens_c, spec);
    c = internal.windowedInnerProduct(dens_q, wmd, false);
end

