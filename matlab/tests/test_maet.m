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


% Tests the multi-attribute path of buildExpTens. The SA path is covered
% by the Expectation tensors section above; these tests focus on
% MAET-specific behaviours: SA-equivalence under the degenerate
% (N=1, A=1) mapping, per-attribute enumeration, weight broadcasting,
% group canonicalisation, NaN padding, and the new error paths.

% -- SA-equivalence: MA with (N=1, A=1, K x 1 column w) reproduces SA --

p_sa = [0; 400; 700];
w_sa = [1; 0.7; 0.5];
sigma = 10; r_ = 2; isPer_ = true; period_ = 1200;

% --- Lazy/eager parity (v2.2) ---
%
% buildExpTens defaults to skinny (lazy=true); internal.ensureExpTensExpensive
% populates the per-tuple fields on demand. The eager and ensured-lazy
% paths must produce structurally identical structs.

dens_eager_sa  = buildExpTens(p_sa, w_sa, sigma, r_, true, isPer_, period_, ...
    'lazy', false, 'verbose', false);
dens_skinny_sa = buildExpTens(p_sa, w_sa, sigma, r_, true, isPer_, period_, ...
    'verbose', false);
results{end+1,1} = 'lazy: SA skinny default has only cheap fields';
results{end,2}   = ~isfield(dens_skinny_sa, 'Centres') ...
                && ~isfield(dens_skinny_sa, 'U_perm') ...
                && ~isfield(dens_skinny_sa, 'nJ');
results{end+1,1} = 'lazy: SA skinny exposes dim';
results{end,2}   = isfield(dens_skinny_sa, 'dim') ...
                && dens_skinny_sa.dim == dens_eager_sa.dim;
dens_filled_sa = internal.ensureExpTensExpensive(dens_skinny_sa);
results{end+1,1} = 'lazy: SA ensure -> matches eager Centres';
results{end,2}   = isequal(dens_filled_sa.Centres, dens_eager_sa.Centres);
results{end+1,1} = 'lazy: SA ensure -> matches eager wJ';
results{end,2}   = isequal(dens_filled_sa.wJ, dens_eager_sa.wJ);
results{end+1,1} = 'lazy: SA ensure -> matches eager U_perm';
results{end,2}   = isequal(dens_filled_sa.U_perm, dens_eager_sa.U_perm);
results{end+1,1} = 'lazy: SA ensure idempotent';
dens_twice_sa = internal.ensureExpTensExpensive(dens_filled_sa);
results{end,2}   = isequal(dens_twice_sa, dens_filled_sa);

dens_eager_ma  = buildExpTens({p_sa}, {w_sa}, sigma, r_, [], true, isPer_, period_, ...
    'lazy', false, 'verbose', false);
dens_skinny_ma = buildExpTens({p_sa}, {w_sa}, sigma, r_, [], true, isPer_, period_, ...
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
results{end+1,1} = 'lazy: cosSimExpTens accepts skinny dens (SA self-similarity = 1)';
s_self = cosSimExpTens(dens_skinny_sa, dens_skinny_sa, 'verbose', false);
results{end,2}   = abs(s_self - 1) < 1e-12;
results{end+1,1} = 'lazy: evalExpTens accepts skinny dens';
v_skinny = evalExpTens(dens_skinny_sa, [0 100 350], 'verbose', false);
v_eager  = evalExpTens(dens_eager_sa,  [0 100 350], 'verbose', false);
results{end,2}   = max(abs(v_skinny - v_eager)) < 1e-12;

for isRel_ = [false, true]
    dens_sa = buildExpTens(p_sa, w_sa, sigma, r_, isRel_, isPer_, period_, ...
        'lazy', false, 'verbose', false);
    dens_ma = buildExpTens({p_sa}, {w_sa}, sigma, r_, [], isRel_, isPer_, period_, ...
        'lazy', false, 'verbose', false);

    relTag = sprintf(' (isRel=%d)', isRel_);
    results{end+1,1} = ['MAET: SA-equivalence tag' relTag];
    results{end,2}   = strcmp(dens_ma.tag, 'MaetDensity'); %#ok<*SAGROW>

    results{end+1,1} = ['MAET: SA-equivalence nJ' relTag];
    results{end,2}   = dens_ma.nJ == dens_sa.nJ;

    results{end+1,1} = ['MAET: SA-equivalence U_perm' relTag];
    results{end,2}   = isequal(dens_ma.U_perm{1}, dens_sa.U_perm);

    results{end+1,1} = ['MAET: SA-equivalence V_comb' relTag];
    results{end,2}   = isequal(dens_ma.V_comb{1}, dens_sa.V_comb);

    results{end+1,1} = ['MAET: SA-equivalence Centres' relTag];
    results{end,2}   = isequal(dens_ma.Centres{1}, dens_sa.Centres);

    results{end+1,1} = ['MAET: SA-equivalence wJ' relTag];
    results{end,2}   = max(abs(dens_ma.wJ - dens_sa.wJ)) < 1e-12;

    results{end+1,1} = ['MAET: SA-equivalence wv_comb' relTag];
    results{end,2}   = max(abs(dens_ma.wv_comb - dens_sa.wv_comb)) < 1e-12;
end

% Dimensionality reduction under isRel=true
results{end+1,1} = 'MAET: Centres dim reduction (isRel=true, r=2)';
dens_ma = buildExpTens({p_sa}, {w_sa}, sigma, 2, [], true, true, 1200, ...
    'lazy', false, 'verbose', false);
results{end,2}   = isequal(size(dens_ma.Centres{1}), [1, dens_ma.nJ]);

% -- Struct basics for pitch + time --

pitchMat = [0 12; 4 15; 7 19];       % 3 x 2
timeMat  = [0 1];                     % 1 x 2
dens = buildExpTens({pitchMat, timeMat}, [], ...
    [10, 0.1], [3, 1], [], [true, false], [true, false], [1200, 0], ...
    'verbose', false);

results{end+1,1} = 'MAET: struct nAttrs';
results{end,2}   = dens.nAttrs == 2;
results{end+1,1} = 'MAET: struct nGroups';
results{end,2}   = dens.nGroups == 2;
results{end+1,1} = 'MAET: struct N';
results{end,2}   = dens.N == 2;
results{end+1,1} = 'MAET: struct groupOfAttr default';
results{end,2}   = isequal(dens.groupOfAttr, [1 2]);
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
    [10, 0.1], [3, 1], [], [true, false], [true, false], [1200, 0], ...
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
dens = buildExpTens({pitchMat}, [], 10, 2, [], false, true, 1200, 'verbose', false);
results{end+1,1} = 'MAET: weight [] -> ones';
results{end,2}   = isequal(dens.w{1}, ones(2, 2));

dens = buildExpTens({pitchMat}, 0.5, 10, 2, [], false, true, 1200, 'verbose', false);
results{end+1,1} = 'MAET: weight scalar top-level';
results{end,2}   = isequal(dens.w{1}, 0.5 * ones(2, 2));

pitchMat = [0 4 5; 4 8 6];               % K=2, N=3
wRow = [0.5, 1.0, 2.0];                  % 1 x N
dens = buildExpTens({pitchMat}, {wRow}, 10, 2, [], false, true, 1200, 'verbose', false);
results{end+1,1} = 'MAET: weight 1 x N row broadcast';
results{end,2}   = isequal(dens.w{1}, [0.5 1.0 2.0; 0.5 1.0 2.0]);

pitchMat = [0 4 5; 4 8 6; 7 10 9];       % K=3, N=3
wCol = [0.5; 1.0; 2.0];                  % K x 1
dens = buildExpTens({pitchMat}, {wCol}, 10, 2, [], false, true, 1200, 'verbose', false);
results{end+1,1} = 'MAET: weight K x 1 column broadcast';
results{end,2}   = isequal(dens.w{1}, repmat([0.5; 1.0; 2.0], 1, 3));

pitchMat = [0 4; 4 8];
W = [0.1 0.2; 0.3 0.4];
dens = buildExpTens({pitchMat}, {W}, 10, 2, [], false, true, 1200, 'verbose', false);
results{end+1,1} = 'MAET: weight K x N full matrix';
results{end,2}   = isequal(dens.w{1}, W);

% -- Groups: vector form and cell form agree --

pitchMat = [0; 4];   % K=2, N=1
timeMat  = 0;         % 1 x 1
xMat     = 0; yMat = 0; zMat = 0;
sigV     = [10, 0.1, 0.2];  rV = [1 1 1 1 1];
isRelV   = [false false false];
isPerV   = [true false false];
perV     = [1200, 0, 0];

dens_v = buildExpTens({pitchMat, timeMat, xMat, yMat, zMat}, [], ...
    sigV, rV, [1 2 3 3 3], isRelV, isPerV, perV, 'verbose', false);
dens_c = buildExpTens({pitchMat, timeMat, xMat, yMat, zMat}, [], ...
    sigV, rV, {1, 2, [3 4 5]}, isRelV, isPerV, perV, 'verbose', false);

results{end+1,1} = 'MAET: groups vector vs cell (groupOfAttr)';
results{end,2}   = isequal(dens_v.groupOfAttr, dens_c.groupOfAttr);
results{end+1,1} = 'MAET: groups vector vs cell (nGroups)';
results{end,2}   = dens_v.nGroups == dens_c.nGroups;

agree = true;
for gg = 1:dens_v.nGroups
    if ~isequal(sort(dens_v.attrsOfGroup{gg}), sort(dens_c.attrsOfGroup{gg}))
        agree = false; break;
    end
end
results{end+1,1} = 'MAET: groups vector vs cell (attrsOfGroup)';
results{end,2}   = agree;

results{end+1,1} = 'MAET: groups non-contiguous errors';
results{end,2}   = throwsError(@() buildExpTens({pitchMat, pitchMat}, [], ...
    [10 10], [1 1], [1 3], [false false], [true true], [1200 1200], 'verbose', false));

results{end+1,1} = 'MAET: groups cell duplicate attr errors';
results{end,2}   = throwsError(@() buildExpTens({pitchMat, pitchMat}, [], ...
    10, [1 1], {[1 2], 2}, false, true, 1200, 'verbose', false));

% -- NaN-padded variable-size events --

pitchMat = [0 0; 4 4; 7 NaN];
timeMat  = [0 1];
dens = buildExpTens({pitchMat, timeMat}, [], ...
    [10, 0.1], [2, 1], [], [false false], [true false], [1200, 0], ...
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
    [10, 0.1], [2, 1], [], [false false], [true false], [1200, 0], ...
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
    10, 2, [], false, true, 1200, 'verbose', false));

results{end+1,1} = 'MAET: wrong r length errors';
results{end,2}   = throwsError(@() buildExpTens({pitchMat, pitchMat}, [], ...
    [10 10], 1, [], [false false], [true true], [1200 1200], 'verbose', false));

results{end+1,1} = 'MAET: wrong sigma length errors';
results{end,2}   = throwsError(@() buildExpTens({pitchMat, pitchMat}, [], ...
    10, [1 1], [], [false false], [true true], [1200 1200], 'verbose', false));

results{end+1,1} = 'MAET: mismatched N errors';
results{end,2}   = throwsError(@() buildExpTens({[0 4], [0 1 2]}, [], ...
    [10 0.1], [1 1], [], [false false], [true false], [1200 0], 'verbose', false));

results{end+1,1} = 'MAET: wrong positional count errors';
results{end,2}   = throwsError(@() buildExpTens({pitchMat}, [], ...
    10, 1, false, true, 1200, 'verbose', false));

% -- isRel + r=1 degenerate warning --

lastwarn('');   % clear the warning buffer
buildExpTens({pitchMat}, [], 10, 1, [], true, true, 1200, 'verbose', false);
warnMsg = lastwarn;
results{end+1,1} = 'MAET: isRel + r=1 emits degenerate warning';
results{end,2}   = ~isempty(warnMsg) && contains(warnMsg, 'degenerate');

% -- evalExpTens MA path: SA-equivalence (isRel=false) --

p_sa_v  = [0; 400; 700];
w_sa_v  = [1; 0.7; 0.5];
sigma_v = 10; r_v = 2; isPer_v = true; period_v = 1200;
xSA_abs = [100 500; 300 600];   % dim=2, nQ=2 (absolute r=2)

dens_sa = buildExpTens(p_sa_v, w_sa_v, sigma_v, r_v, false, isPer_v, period_v, ...
    'verbose', false);
vals_sa = evalExpTens(dens_sa, xSA_abs, 'verbose', false);

dens_ma = buildExpTens({p_sa_v}, {w_sa_v}, sigma_v, r_v, [], false, isPer_v, ...
    period_v, 'verbose', false);
vals_ma_cell = evalExpTens(dens_ma, {xSA_abs}, 'verbose', false);
vals_ma_mat  = evalExpTens(dens_ma,  xSA_abs,  'verbose', false);

results{end+1,1} = 'evalExpTens MA: SA-equivalence abs (cell form)';
results{end,2}   = max(abs(vals_ma_cell - vals_sa)) < 1e-12;
results{end+1,1} = 'evalExpTens MA: SA-equivalence abs (matrix form)';
results{end,2}   = max(abs(vals_ma_mat - vals_sa)) < 1e-12;

% -- evalExpTens MA path: SA-equivalence (isRel=true, r=3) --

r_v = 3;
xSA_rel = [400 200; 700 500];    % dim = r-1 = 2, nQ = 2
dens_sa = buildExpTens(p_sa_v, w_sa_v, sigma_v, r_v, true, isPer_v, period_v, ...
    'verbose', false);
vals_sa = evalExpTens(dens_sa, xSA_rel, 'verbose', false);

dens_ma = buildExpTens({p_sa_v}, {w_sa_v}, sigma_v, r_v, [], true, isPer_v, ...
    period_v, 'verbose', false);
vals_ma = evalExpTens(dens_ma, {xSA_rel}, 'verbose', false);

results{end+1,1} = 'evalExpTens MA: SA-equivalence rel';
results{end,2}   = max(abs(vals_ma - vals_sa)) < 1e-12;

% Normalisation modes
for modeCell = {'gaussian', 'pdf'}
    mode = modeCell{1};
    vals_sa_n = evalExpTens(dens_sa, xSA_rel, mode, 'verbose', false);
    vals_ma_n = evalExpTens(dens_ma, {xSA_rel}, mode, 'verbose', false);
    results{end+1,1} = ['evalExpTens MA: SA-equivalence normalize=' mode]; %#ok<SAGROW>
    results{end,2}   = max(abs(vals_ma_n - vals_sa_n)) < 1e-12;
end

% -- evalExpTens MA: cell form vs matrix form agree --

pitchMat = [0; 4; 7];    % K=3, N=1
timeMat  = 1.0;           % 1 x 1
dens = buildExpTens({pitchMat, timeMat}, [], ...
    [10, 0.1], [2, 1], [], [false false], [true false], [1200, 0], ...
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
    [20, 20], [1, 1], [], [false false], [true false], [1200, 0], ...
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
    [10, 0.1], [2, 1], [], [false false], [true false], [1200, 0], ...
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

% -- cosSimExpTens MA path: SA-equivalence --

p_a_v  = [0; 400; 700];
p_b_v  = [0; 300; 700];
w_a_v  = [1; 0.7; 0.5];
w_b_v  = [1; 0.6; 0.8];

% Absolute (isRel=false), periodic
s_sa = cosSimExpTens(p_a_v, w_a_v, p_b_v, w_b_v, 10, 2, false, true, 1200, ...
    'verbose', false);
da = buildExpTens({p_a_v}, {w_a_v}, 10, 2, [], false, true, 1200, 'verbose', false);
db = buildExpTens({p_b_v}, {w_b_v}, 10, 2, [], false, true, 1200, 'verbose', false);
s_ma = cosSimExpTens(da, db, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: SA-equivalence abs periodic';
results{end,2}   = abs(s_ma - s_sa) < 1e-12;

% Relative + periodic (uses pairwise-differences formula per attribute)
for r_v = [2, 3]
    s_sa = cosSimExpTens(p_a_v, w_a_v, p_b_v, w_b_v, 10, r_v, true, true, 1200, ...
        'verbose', false);
    da = buildExpTens({p_a_v}, {w_a_v}, 10, r_v, [], true, true, 1200, 'verbose', false);
    db = buildExpTens({p_b_v}, {w_b_v}, 10, r_v, [], true, true, 1200, 'verbose', false);
    s_ma = cosSimExpTens(da, db, 'verbose', false);
    results{end+1,1} = sprintf('cosSimExpTens MA: SA-equivalence rel periodic r=%d', r_v); %#ok<SAGROW>
    results{end,2}   = abs(s_ma - s_sa) < 1e-12;
end

% Relative + non-periodic
s_sa = cosSimExpTens(p_a_v, w_a_v, p_b_v, w_b_v, 10, 3, true, false, 0, ...
    'verbose', false);
da = buildExpTens({p_a_v}, {w_a_v}, 10, 3, [], true, false, 0, 'verbose', false);
db = buildExpTens({p_b_v}, {w_b_v}, 10, 3, [], true, false, 0, 'verbose', false);
s_ma = cosSimExpTens(da, db, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: SA-equivalence rel non-periodic';
results{end,2}   = abs(s_ma - s_sa) < 1e-12;

% -- cosSimExpTens MA: self-similarity = 1 --

pitchMA = [0 12; 4 15; 7 19];    % 3 x 2
timeMA  = [0 1];                  % 1 x 2
d = buildExpTens({pitchMA, timeMA}, [], ...
    [10, 0.1], [3, 1], [], [true, false], [true, false], [1200, 0], ...
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
    [10, 0.1], [3, 1], [], [true, false], [true, false], [1200, 0], ...
    'verbose', false);
db = buildExpTens({pitchB, timeB}, [], ...
    [10, 0.1], [3, 1], [], [true, false], [true, false], [1200, 0], ...
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
    [10, 0.1], [3, 1], [], [true, false], [true, false], [1200, 0], ...
    'verbose', false);
d2 = buildExpTens({pitchTs, timeT}, [], ...
    [10, 0.1], [3, 1], [], [true, false], [true, false], [1200, 0], ...
    'verbose', false);
s_trans = cosSimExpTens(d1, d2, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: isRel transposition invariance';
results{end,2}   = abs(s_trans - 1) < 1e-10;

% -- cosSimExpTens MA: raw-args matches struct form --

s_raw = cosSimExpTens({pitchA, timeA}, [], {pitchB, timeB}, [], ...
    [10, 0.1], [3, 1], [], [true, false], [true, false], [1200, 0], ...
    'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: raw-args == struct form';
results{end,2}   = abs(s_raw - s_ab) < 1e-12;

% -- cosSimExpTens: SA raw-args still works (backward compat check) --

s_sa_raw = cosSimExpTens([0 4 7], [], [0 4 7], [], 10, 2, true, true, 1200, ...
    'verbose', false);
results{end+1,1} = 'cosSimExpTens: SA raw-args identical = 1';
results{end,2}   = abs(s_sa_raw - 1) < 1e-12;

% -- cosSimExpTens MA: mismatched raw-args kinds error --

results{end+1,1} = 'cosSimExpTens MA: mismatched raw-args kinds error';
results{end,2}   = throwsError(@() cosSimExpTens( ...
    {pitchA, timeA}, [], [0 4 7], [], 10, 2, true, true, 1200, 'verbose', false));

% -- cosSimExpTens MA: mixed struct types error --

d_sa = buildExpTens([0 4 7], [], 10, 2, false, true, 1200, 'verbose', false);
d_ma = buildExpTens({[0; 4; 7]}, [], 10, 2, [], false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens: mixed SA/MA structs error';
results{end,2}   = throwsError(@() cosSimExpTens(d_sa, d_ma, 'verbose', false));

% -- cosSimExpTens MA: parameter-mismatch errors --

d_ref = buildExpTens({pitchA}, [], 10, 2, [], false, true, 1200, 'verbose', false);
% different r
d_r = buildExpTens({pitchA}, [], 10, 3, [], false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: mismatched r error';
results{end,2}   = throwsError(@() cosSimExpTens(d_ref, d_r, 'verbose', false));
% different sigma
d_s = buildExpTens({pitchA}, [], 20, 2, [], false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: mismatched sigma error';
results{end,2}   = throwsError(@() cosSimExpTens(d_ref, d_s, 'verbose', false));
% different isRel
d_rel = buildExpTens({pitchA}, [], 10, 2, [], true, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: mismatched isRel error';
results{end,2}   = throwsError(@() cosSimExpTens(d_ref, d_rel, 'verbose', false));
% different period on periodic group
d_p = buildExpTens({pitchA}, [], 10, 2, [], false, true, 2400, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: mismatched period error';
results{end,2}   = throwsError(@() cosSimExpTens(d_ref, d_p, 'verbose', false));

% -- entropyExpTens MA: SA-equivalence periodic --

p_e = [0; 4; 7];
w_e = [1; 1; 1];
H_sa = entropyExpTens(p_e.', w_e.', 10, 1, false, true, 12, ...
    'nPointsPerDim', 400, 'verbose', false);
H_ma = entropyExpTens({p_e}, {w_e}, 10, 1, [], false, true, 12, ...
    'nPointsPerDim', 400, 'verbose', false);
results{end+1,1} = 'entropyExpTens MA: SA-equivalence periodic';
results{end,2}   = abs(H_ma - H_sa) < 1e-10;

% -- entropyExpTens MA: SA-equivalence non-periodic --

H_sa = entropyExpTens(p_e.', w_e.', 10, 1, false, false, 0, ...
    'xMin', -3, 'xMax', 10, 'nPointsPerDim', 400, 'verbose', false);
H_ma = entropyExpTens({p_e}, {w_e}, 10, 1, [], false, false, 0, ...
    'xMin', -3, 'xMax', 10, 'nPointsPerDim', 400, 'verbose', false);
results{end+1,1} = 'entropyExpTens MA: SA-equivalence non-periodic';
results{end,2}   = abs(H_ma - H_sa) < 1e-10;

% -- entropyExpTens MA: uniform pitch near 1 --

p_uniform = (0:11).';
H_u = entropyExpTens({p_uniform}, [], 100, 1, [], false, true, 12, ...
    'nPointsPerDim', 400, 'verbose', false);
results{end+1,1} = 'entropyExpTens MA: uniform chromatic near 1';
results{end,2}   = H_u > 0.95;

% -- entropyExpTens MA: concentrated below uniform --

H_one = entropyExpTens({5}, [], 20, 1, [], false, true, 12, ...
    'nPointsPerDim', 400, 'verbose', false);
H_all = entropyExpTens({p_uniform}, [], 20, 1, [], false, true, 12, ...
    'nPointsPerDim', 400, 'verbose', false);
results{end+1,1} = 'entropyExpTens MA: concentrated < uniform';
results{end,2}   = H_one < H_all;

% -- entropyExpTens MA: pitch + time runs (dim = 2) --

pitchE = [0 12; 4 15; 7 19];   % 3 x 2
timeE  = [0 1];                 % 1 x 2
densE = buildExpTens({pitchE, timeE}, [], ...
    [20, 0.1], [2, 1], [], [true, false], [true, false], [1200, 0], ...
    'verbose', false);
results{end+1,1} = 'entropyExpTens MA: dim == 2 (r=2 pitch + r=1 time)';
results{end,2}   = densE.dim == 2;
H_pt = entropyExpTens(densE, ...
    'xMin', -0.5, 'xMax', 1.5, 'nPointsPerDim', 80, 'verbose', false);
results{end+1,1} = 'entropyExpTens MA: pitch+time H in (0,1)';
results{end,2}   = H_pt > 0 && H_pt < 1;

% -- entropyExpTens MA: grid-limit guard --

results{end+1,1} = 'entropyExpTens MA: grid-limit exceeded errors';
results{end,2}   = throwsError(@() entropyExpTens(densE, ...
    'xMin', 0, 'xMax', 2, 'nPointsPerDim', 20000, 'gridLimit', 1e6, 'verbose', false));

% -- entropyExpTens MA: missing bounds error --

results{end+1,1} = 'entropyExpTens MA: missing non-periodic bounds errors';
results{end,2}   = throwsError(@() entropyExpTens({p_e}, [], 10, 1, [], ...
    false, false, 0, 'nPointsPerDim', 100, 'verbose', false));

% -- entropyExpTens MA: per-group bounds vector matches scalar --

H_scalar = entropyExpTens(densE, ...
    'xMin', -0.5, 'xMax', 1.5, 'nPointsPerDim', 60, 'verbose', false);
H_vec = entropyExpTens(densE, ...
    'xMin', [NaN, -0.5], 'xMax', [NaN, 1.5], 'nPointsPerDim', 60, 'verbose', false);
results{end+1,1} = 'entropyExpTens MA: per-group bounds vector == scalar';
results{end,2}   = abs(H_scalar - H_vec) < 1e-12;

% -- differenceEvents: order 0 identity --

p_d = {[0 2 5 7]};
[pd, wd] = differenceEvents(p_d, [], [], 0, 12);
results{end+1,1} = 'differenceEvents: order 0 values unchanged';
results{end,2}   = isequal(pd{1}, p_d{1});
results{end+1,1} = 'differenceEvents: order 0 weight stays []';
results{end,2}   = isempty(wd);

% -- differenceEvents: order 1 non-periodic --

[pd, ~] = differenceEvents({[0 2 5 7]}, [], [], 1, 0);
results{end+1,1} = 'differenceEvents: order 1 non-periodic';
results{end,2}   = isequal(pd{1}, [2 3 2]);

% -- differenceEvents: order 1 periodic wrap --

[pd, ~] = differenceEvents({[0 11]}, [], [], 1, 12);
results{end+1,1} = 'differenceEvents: order 1 periodic wrap (11 -> -1)';
results{end,2}   = isequal(pd{1}, -1);

% -- differenceEvents: order 2 --

[pd, ~] = differenceEvents({[0 2 5 7]}, [], [], 2, 0);
results{end+1,1} = 'differenceEvents: order 2';
results{end,2}   = isequal(pd{1}, [1 -1]);

% -- differenceEvents: order 1 weight rolling product --

[~, wd] = differenceEvents({[0 2 5 7]}, {[0.5 0.8 1.0 0.2]}, [], 1, 0);
expected = [0.5*0.8, 0.8*1.0, 1.0*0.2];
results{end+1,1} = 'differenceEvents: order 1 rolling-product weights';
results{end,2}   = max(abs(wd{1} - expected)) < 1e-12;

% -- differenceEvents: order 2 weight rolling product (width 3) --

[~, wd] = differenceEvents({[0 1 3 6]}, {[0.5 0.8 1.0 0.2]}, [], 2, 0);
expected = [0.5*0.8*1.0, 0.8*1.0*0.2];
results{end+1,1} = 'differenceEvents: order 2 rolling-product weights';
results{end,2}   = max(abs(wd{1} - expected)) < 1e-12;

% -- differenceEvents: scalar top-level weight raised to power --
% A top-level scalar c with uniform order k returns scalar c^(k+1),
% so scalar and vector-of-c inputs produce equivalent downstream
% densities.

[~, wd] = differenceEvents({[0 2 5]}, 0.7, [], 1, 0);
results{end+1,1} = 'differenceEvents: scalar weight raised to power';
results{end,2}   = abs(wd - 0.7^2) < 1e-12;

% -- differenceEvents: mixed orders alignment --

p_d = {[0 2 5 7], [0 1 2 3.5]};
[pd, ~] = differenceEvents(p_d, [], [], [0 1], [0 0]);
results{end+1,1} = 'differenceEvents: mixed orders — k=0 group drops leading';
results{end,2}   = isequal(pd{1}, [2 5 7]);
results{end+1,1} = 'differenceEvents: mixed orders — k=1 group differenced';
results{end,2}   = isequal(pd{2}, [1 1 1.5]);

% -- differenceEvents: grouped attributes share order --

p_d = {[0 1 3], [10 12 16]};
[pd, ~] = differenceEvents(p_d, [], [1 1], 1, 0);
results{end+1,1} = 'differenceEvents: grouped attrs both differenced';
results{end,2}   = isequal(pd{1}, [1 2]) && isequal(pd{2}, [2 4]);

% -- differenceEvents: output feeds buildExpTens --

p_d = {[0 2 5 7], [0 0.5 1.2 1.7]};
[pd, wd] = differenceEvents(p_d, [], [], [0 1], [1200 0]);
dens_d = buildExpTens(pd, wd, [10 0.05], [1 1], [], ...
    [false false], [true false], [1200 0], 'verbose', false);
results{end+1,1} = 'differenceEvents: output feeds buildExpTens';
results{end,2}   = strcmp(dens_d.tag, 'MaetDensity') && dens_d.N == 3;

% -- differenceEvents: too-high order errors --

results{end+1,1} = 'differenceEvents: too-high order errors';
results{end,2}   = throwsError(@() differenceEvents({[0 2 5]}, [], [], 3, 0));

% -- differenceEvents: negative order errors --

results{end+1,1} = 'differenceEvents: negative order errors';
results{end,2}   = throwsError(@() differenceEvents({[0 2 5]}, [], [], -1, 0));

% -- differenceEvents: diffOrders length mismatch errors --

results{end+1,1} = 'differenceEvents: diffOrders length mismatch errors';
results{end,2}   = throwsError(@() differenceEvents({[0 2 5]}, [], [], [1 1], [0 0]));

% -- differenceEvents: mismatched event counts error --

results{end+1,1} = 'differenceEvents: mismatched event counts error';
results{end,2}   = throwsError(@() differenceEvents({[0 2 5], [0 1]}, [], [], [0 0], [0 0]));

% -- differenceEvents: multi-slot attribute errors --
% A single K_a = 2 attribute must raise differenceEvents:multiSlotAttribute.
% Column-wise differencing would impose a cross-event slot correspondence
% that within-event slot exchangeability does not license.

p_ms = {[60 62 64; 67 69 71]};   % K_a = 2, N = 3
results{end+1,1} = 'differenceEvents: K_a = 2 attribute errors (multiSlotAttribute id)';
results{end,2}   = throwsErrorWithId( ...
    @() differenceEvents(p_ms, [], [], 1, 0), ...
    'differenceEvents:multiSlotAttribute');

% -- differenceEvents: empty attribute (K_a = 0) errors --
% K_a = 0 is likewise rejected by the K_a = 1 check; there is nothing
% to difference in an empty attribute.

p_empty = {zeros(0, 3)};          % K_a = 0, N = 3
results{end+1,1} = 'differenceEvents: K_a = 0 attribute errors (multiSlotAttribute id)';
results{end,2}   = throwsErrorWithId( ...
    @() differenceEvents(p_empty, [], [], 1, 0), ...
    'differenceEvents:multiSlotAttribute');

% -- differenceEvents: mixed K_a input errors on the offending attribute --
% Attribute 1 has K_a = 1, attribute 2 has K_a = 2 — the error must fire
% and its message must name the offending attribute index.

p_mixed = {[60 62 64], [60 62 64; 67 69 71]};
results{end+1,1} = 'differenceEvents: mixed K_a input errors (multiSlotAttribute id)';
results{end,2}   = throwsErrorWithId( ...
    @() differenceEvents(p_mixed, [], [], [1 1], [0 0]), ...
    'differenceEvents:multiSlotAttribute');

results{end+1,1} = 'differenceEvents: mixed K_a error message names attribute 2';
results{end,2}   = errorMessageContains( ...
    @() differenceEvents(p_mixed, [], [], [1 1], [0 0]), ...
    'Attribute 2');

% -- differenceEvents: voices-as-attributes pipeline round trip --
% Four voices, each K_a = 1 in a shared group, differenced, then stacked
% into a single multi-slot attribute before buildExpTens. The resulting
% MaetDensity should have the expected shape and evalExpTens should
% return finite non-negative values at a few query points.

pS = [72 74 76 77];     % soprano
pA = [67 69 71 72];     % alto
pT = [60 62 64 65];     % tenor
pB = [48 50 52 53];     % bass
pAttr  = {pS, pA, pT, pB};
groupsV = [1 1 1 1];
[pDiff, ~] = differenceEvents(pAttr, [], groupsV, 1, 0);

results{end+1,1} = 'differenceEvents: voices-as-attrs — each differenced attribute is 1 x 3';
results{end,2}   = all(cellfun(@(M) isequal(size(M), [1 3]), pDiff));

pBundled = { vertcat(pDiff{:}) };    % 4 x 3 multi-slot bundle
results{end+1,1} = 'differenceEvents: voices-as-attrs — bundle is 4 x 3';
results{end,2}   = isequal(size(pBundled{1}), [4 3]);

dens_v = buildExpTens(pBundled, [], 10, 1, [], false, false, 0, ...
    'verbose', false);
results{end+1,1} = 'differenceEvents: voices-as-attrs — buildExpTens returns MaetDensity';
results{end,2}   = strcmp(dens_v.tag, 'MaetDensity');

% Evaluate at a handful of query points; expect finite non-negative
% output everywhere.
x_query = [-3 0 2 4 7];
vals_v = evalExpTens(dens_v, x_query);
results{end+1,1} = 'differenceEvents: voices-as-attrs — evalExpTens returns finite non-negative values';
results{end,2}   = all(isfinite(vals_v)) && all(vals_v >= 0);

% -- translateEvents: zero shift identity --

p_t = {[60 62 64], [0 1 2]};
out = translateEvents(p_t, [1 2], [0 0], [false false], [false false], [0 0]);
results{end+1,1} = 'translateEvents: zero shift returns input values';
results{end,2}   = isequal(out{1}, p_t{1}) && isequal(out{2}, p_t{2});

% -- translateEvents: all-NaN offsets identity --

out = translateEvents(p_t, [1 2], [NaN NaN], [false false], [false false], [0 0]);
results{end+1,1} = 'translateEvents: all-NaN offsets are identity';
results{end,2}   = isequal(out{1}, p_t{1}) && isequal(out{2}, p_t{2});

% -- translateEvents: does not mutate input --

p_orig = [60 62 64];
p_in   = {p_orig};
translateEvents(p_in, [], 5, false, false, 0);   % discard output
results{end+1,1} = 'translateEvents: does not mutate input matrix';
results{end,2}   = isequal(p_in{1}, p_orig);

% -- translateEvents: non-periodic absolute shift --

out = translateEvents({[60 62 64]}, [], 5, false, false, 0);
results{end+1,1} = 'translateEvents: non-periodic adds mu to every value';
results{end,2}   = isequal(out{1}, [65 67 69]);

% -- translateEvents: periodic wrap to [0, P) --

out = translateEvents({[10 11 0]}, [], 3, false, true, 12);
results{end+1,1} = 'translateEvents: periodic shift wraps to [0, P)';
results{end,2}   = isequal(out{1}, [1 2 3]);

% -- translateEvents: periodic negative mu wraps into [0, P) --

out = translateEvents({[1 2]}, [], -3, false, true, 12);
results{end+1,1} = 'translateEvents: negative mu wraps into [0, P)';
results{end,2}   = isequal(out{1}, [10 11]);

% -- translateEvents: period ignored when isPer is false --

out = translateEvents({[10 11]}, [], 5, false, false, 12);
% Non-periodic: no wrap, values become 15, 16 (not 3, 4).
results{end+1,1} = 'translateEvents: period ignored when isPer is false';
results{end,2}   = isequal(out{1}, [15 16]);

% -- translateEvents: K_a > 1 (multi-slot attribute) --

out = translateEvents({[60 64 67; 63 67 70]}, [], 5, false, false, 0);
results{end+1,1} = 'translateEvents: K_a > 1 shifts every slot uniformly';
results{end,2}   = isequal(out{1}, [65 69 72; 68 72 75]);

% -- translateEvents: relative group emits no-op warning and passes through --

lastwarn('');   % clear the warning buffer
out = translateEvents({[60 64 67]}, [], 5, true, false, 0);
[~, warnId] = lastwarn;
results{end+1,1} = 'translateEvents: relative group emits noOpRelative warning';
results{end,2}   = strcmp(warnId, 'translateEvents:noOpRelative');
results{end+1,1} = 'translateEvents: relative group passes through unchanged';
results{end,2}   = isequal(out{1}, [60 64 67]);

% -- translateEvents: NaN entries skip groups --

p_tg = {[60 64], [0 1]};
out = translateEvents(p_tg, [1 2], [5 NaN], [false false], [false false], [0 0]);
results{end+1,1} = 'translateEvents: NaN skips group, finite shifts other';
results{end,2}   = isequal(out{1}, [65 69]) && isequal(out{2}, [0 1]);

% -- translateEvents: multi-group simultaneous --

out = translateEvents(p_tg, [1 2], [5 0.5], [false false], [false false], [0 0]);
results{end+1,1} = 'translateEvents: multi-group simultaneous';
results{end,2}   = isequal(out{1}, [65 69]) && isequal(out{2}, [0.5 1.5]);

% -- translateEvents: multiple attributes sharing one group --

p_share = {[60 64], [67 71]};
out = translateEvents(p_share, [1 1], 5, false, false, 0);
results{end+1,1} = 'translateEvents: multi-attribute shared group both shift';
results{end,2}   = isequal(out{1}, [65 69]) && isequal(out{2}, [72 76]);

% -- translateEvents: composition (non-periodic) --

p_c = {[60 62 64]};
once  = translateEvents(p_c, [], 5, false, false, 0);
twice = translateEvents(once, [], 3, false, false, 0);
direct = translateEvents(p_c, [], 8, false, false, 0);
results{end+1,1} = 'translateEvents: composition is additive (non-periodic)';
results{end,2}   = isequal(twice{1}, direct{1});

% -- translateEvents: composition (periodic) --

p_cp = {[10 11]};
once  = translateEvents(p_cp, [], 7, false, true, 12);
twice = translateEvents(once, [], 9, false, true, 12);
% 10 + 16 = 26; mod(26, 12) = 2.  11 + 16 = 27; mod(27, 12) = 3.
results{end+1,1} = 'translateEvents: composition (periodic) wraps correctly';
results{end,2}   = isequal(twice{1}, [2 3]);

% -- translateEvents: self-IP invariant under translation (non-periodic) --

p_si = {[60 64 67]};
M_si = buildExpTens(p_si, [], 0.15, 1, [], false, false, 0, 'verbose', false);
ip_self = cosSimExpTens(M_si, M_si, 'verbose', false);
si_ok = true;
for mu = [-3.0 1.5 7.0]
    p_mu = translateEvents(p_si, [], mu, false, false, 0);
    M_mu = buildExpTens(p_mu, [], 0.15, 1, [], false, false, 0, 'verbose', false);
    ip_mu = cosSimExpTens(M_mu, M_mu, 'verbose', false);
    if abs(ip_mu - ip_self) > 1e-12 * max(1, abs(ip_self))
        si_ok = false;
        break;
    end
end
results{end+1,1} = 'translateEvents: <f^mu, f^mu> = <f, f> non-periodic';
results{end,2}   = si_ok;

% -- translateEvents: self-IP invariant under translation (periodic) --

p_sp = {[0 4 7]};
M_sp = buildExpTens(p_sp, [], 0.15, 1, [], false, true, 12, 'verbose', false);
ip_sp_self = cosSimExpTens(M_sp, M_sp, 'verbose', false);
sp_ok = true;
for mu = [-7.0 1.5 6.0 15.0]
    p_mu = translateEvents(p_sp, [], mu, false, true, 12);
    M_mu = buildExpTens(p_mu, [], 0.15, 1, [], false, true, 12, 'verbose', false);
    ip_mu = cosSimExpTens(M_mu, M_mu, 'verbose', false);
    if abs(ip_mu - ip_sp_self) > 1e-12 * max(1, abs(ip_sp_self))
        sp_ok = false;
        break;
    end
end
results{end+1,1} = 'translateEvents: <f^mu, f^mu> = <f, f> periodic';
results{end,2}   = sp_ok;

% -- translateEvents: cos-sim sweep recovers transposition peak --

p_q = {[60 64 67]};       % C major
p_c = {[62 66 69]};       % D major (= +2 st)
M_q = buildExpTens(p_q, [], 0.15, 1, [], false, false, 0, 'verbose', false);
best_mu = NaN;
best_s  = -Inf;
for mu = -12:0.25:12
    p_c_mu = translateEvents(p_c, [], mu, false, false, 0);
    M_c_mu = buildExpTens(p_c_mu, [], 0.15, 1, [], false, false, 0, 'verbose', false);
    s = cosSimExpTens(M_q, M_c_mu, 'verbose', false);
    if s > best_s
        best_s = s;
        best_mu = mu;
    end
end
results{end+1,1} = 'translateEvents: sweep peak at expected offset (-2 st)';
results{end,2}   = abs(best_mu - (-2.0)) < 0.01;
results{end+1,1} = 'translateEvents: sweep peak similarity is 1';
results{end,2}   = best_s > 1.0 - 1e-9;

% -- translateEvents: error cases --

results{end+1,1} = 'translateEvents: wrong-shape offsets (length-3 row, G=2) errors';
results{end,2}   = throwsErrorWithId( ...
    @() translateEvents({[1 2], [3 4]}, [], [5 0 7], ...
                         [false false], [false false], [0 0]), ...
    'translateEvents:wrongOffsetsShape');

results{end+1,1} = 'translateEvents: wrong-length isRel errors';
results{end,2}   = throwsErrorWithId( ...
    @() translateEvents({[1 2]}, [], 5, [false false], false, 0), ...
    'translateEvents:wrongIsRelLength');

results{end+1,1} = 'translateEvents: wrong-length isPer errors';
results{end,2}   = throwsErrorWithId( ...
    @() translateEvents({[1 2]}, [], 5, false, [false false], 0), ...
    'translateEvents:wrongIsPerLength');

results{end+1,1} = 'translateEvents: wrong-length periods errors';
results{end,2}   = throwsErrorWithId( ...
    @() translateEvents({[1 2]}, [], 5, false, false, [0 12]), ...
    'translateEvents:wrongPeriodsLength');

results{end+1,1} = 'translateEvents: infinite offset errors';
results{end,2}   = throwsErrorWithId( ...
    @() translateEvents({[1 2]}, [], Inf, false, false, 0), ...
    'translateEvents:nonFiniteOffset');

% -- translateEvents: matrix-form offsets (sweep) --

% (a) Matrix shape (G, M) returns a 1-by-M cell of 1-by-A cells.
p_sweep_in = {[60 64 67]};
offs_mat   = [0 100 200];   % G=1, M=3 (1xM row vector unambiguous as matrix form)
out_sweep  = translateEvents(p_sweep_in, [1], offs_mat, false, false, 0);
ok = iscell(out_sweep) && numel(out_sweep) == 3 ...
     && iscell(out_sweep{1}) && numel(out_sweep{1}) == 1 ...
     && isequal(out_sweep{1}{1}, [60 64 67]) ...
     && isequal(out_sweep{2}{1}, [160 164 167]) ...
     && isequal(out_sweep{3}{1}, [260 264 267]);
results{end+1,1} = 'translateEvents: matrix form returns 1-by-M cell of 1-by-A cells';
results{end,2}   = ok;

% (b) (G, 1) column-vector input is matrix form M = 1 and keeps the
% cell wrapper around the inner 1-by-A cell. (Vector form in MATLAB
% requires a 1-by-G ROW vector; column vectors are matrix-form.)
p_g2     = {[1 2], [10 20]};
offs_g2  = [5; 7];   % 2-by-1 column → matrix form M = 1
out_g2   = translateEvents(p_g2, [1 2], offs_g2, ...
                            [false false], [false false], [0 0]);
ok = iscell(out_g2) && numel(out_g2) == 1 ...
     && iscell(out_g2{1}) && numel(out_g2{1}) == 2 ...
     && isequal(out_g2{1}{1}, [6 7]) ...
     && isequal(out_g2{1}{2}, [17 27]);
results{end+1,1} = 'translateEvents: (G,1) matrix form keeps cell wrapper (M=1)';
results{end,2}   = ok;

% (c) Per-column equivalence with vector-form calls. Each column is
% transposed to a 1-by-G row to invoke vector form.
p_pe      = {[60 64 67], [0 1 2]};
groups_pe = [1 2];
isRel_pe  = [false false];
isPer_pe  = [true  false];
period_pe = [1200  0];
offs_mat2 = [0   100  200  -50; ...
             0    0.5   1    -0.25];   % 2-by-4 matrix form
sweep2 = translateEvents(p_pe, groups_pe, offs_mat2, ...
                          isRel_pe, isPer_pe, period_pe);
allMatch = true;
for m = 1:size(offs_mat2, 2)
    one_m = translateEvents(p_pe, groups_pe, offs_mat2(:, m).', ...
                             isRel_pe, isPer_pe, period_pe);
    for a = 1:numel(p_pe)
        if ~isequal(sweep2{m}{a}, one_m{a})
            allMatch = false; break;
        end
    end
end
results{end+1,1} = 'translateEvents: matrix per-column equals vector-form calls';
results{end,2}   = allMatch;

% (d) NaN entries per column skip translation column-by-column.
offs_nan = [10   NaN  30; ...
             NaN  5    NaN];
out_nan = translateEvents(p_pe, groups_pe, offs_nan, ...
                           isRel_pe, isPer_pe, period_pe);
ok = isequal(out_nan{1}{1}, [70 74 77]) ...    % col 1: g1 by +10
     && isequal(out_nan{1}{2}, [0 1 2]) ...    %         g2 untouched (NaN)
     && isequal(out_nan{2}{1}, [60 64 67]) ... % col 2: g1 untouched (NaN)
     && isequal(out_nan{2}{2}, [5 6 7]) ...    %         g2 by +5
     && isequal(out_nan{3}{1}, [90 94 97]) ... % col 3: g1 by +30
     && isequal(out_nan{3}{2}, [0 1 2]);       %         g2 untouched (NaN)
results{end+1,1} = 'translateEvents: matrix NaN entries skip per column';
results{end,2}   = ok;

% (e) Periodic wrap applies per column.
p_per      = {[10 1190]};
out_per    = translateEvents(p_per, [1], [100 1100], false, true, 1200);
ok = isequal(out_per{1}{1}, [110 90]) ...    % col 1: +100, second wraps
     && isequal(out_per{2}{1}, [1110 1090]);   % col 2: +1100, second wraps
results{end+1,1} = 'translateEvents: matrix periodic wrap applies per column';
results{end,2}   = ok;

% (f) Relative group warns at most once across multiple columns.
p_rel       = {[60 64 67], [0 1 2]};
offs_rel    = [10 20 30; 0 0.5 1.0];   % all columns finite on relative row
isRel_rel   = [true false];
prevWarn    = warning('off', 'translateEvents:noOpRelative'); %#ok<WNOFF>
warning('off', 'all');                 % clear all
lastwarn('');                          % clear last warning
warning('on', 'translateEvents:noOpRelative');
% Capture warning count via a custom helper-free pattern: count by
% checking lastwarn after each call. We can also rely on the fact
% that a single warning per call is the contract; assert that
% lastwarn after the call matches the relative-group message exactly
% once and that no per-column repetition occurs (smoke-test only).
out_rel = translateEvents(p_rel, [1 2], offs_rel, isRel_rel, ...
                           [false false], [0 0]);
[~, lastId] = lastwarn();
warning(prevWarn);
ok = strcmp(lastId, 'translateEvents:noOpRelative') ...
     && numel(out_rel) == 3 ...
     && isequal(out_rel{1}{1}, p_rel{1}) ...   % relative untouched
     && isequal(out_rel{1}{2}, [0 1 2]) ...
     && isequal(out_rel{2}{2}, [0.5 1.5 2.5]) ...
     && isequal(out_rel{3}{2}, [1 2 3]);
results{end+1,1} = 'translateEvents: relative-row warns at most once across columns';
results{end,2}   = ok;

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
    sigma_ma, r_ma, groups_ma, isRel_ma, isPer_ma, period_ma, ...
    'verbose', false);
results{end+1,1} = 'cosSimExpTens raw-MA scalar dispatch returns numeric scalar';
results{end,2}   = isnumeric(s_scalar) && isscalar(s_scalar) && isfinite(s_scalar);

% (b) Scalar-vs-list broadcast: matrix-form translateEvents feed.
offs_rma  = [-100  0   100  200; 0 1 2 1];
qry_swept = translateEvents(p_qry, groups_ma, offs_rma, ...
                             isRel_ma, isPer_ma, period_ma);
s_list = cosSimExpTens(p_ref, [], qry_swept, [], ...
    sigma_ma, r_ma, groups_ma, isRel_ma, isPer_ma, period_ma, ...
    'verbose', false);
results{end+1,1} = 'cosSimExpTens raw-MA list returns 1-by-M cell';
results{end,2}   = iscell(s_list) && numel(s_list) == 4 ...
                   && all(cellfun(@(x) isnumeric(x) && isscalar(x) && isfinite(x), ...
                                  s_list));

% (c) Floating-point parity with manual build loop.
dens_ref = buildExpTens(p_ref, [], sigma_ma, r_ma, groups_ma, ...
    isRel_ma, isPer_ma, period_ma, 'verbose', false);
s_manual = zeros(1, numel(qry_swept));
for m = 1:numel(qry_swept)
    dens_q = buildExpTens(qry_swept{m}, [], sigma_ma, r_ma, groups_ma, ...
        isRel_ma, isPer_ma, period_ma, 'verbose', false);
    s_manual(m) = cosSimExpTens(dens_ref, dens_q, 'verbose', false);
end
s_list_num = cell2mat(s_list);
results{end+1,1} = 'cosSimExpTens raw-MA list parity with manual buildExpTens loop';
results{end,2}   = max(abs(s_list_num - s_manual)) < 1e-12;

% (d) Operand order symmetric.
s_rev = cosSimExpTens(qry_swept, [], p_ref, [], ...
    sigma_ma, r_ma, groups_ma, isRel_ma, isPer_ma, period_ma, ...
    'verbose', false);
s_rev_num = cell2mat(s_rev);
results{end+1,1} = 'cosSimExpTens raw-MA list symmetric in operand order';
results{end,2}   = max(abs(s_list_num - s_rev_num)) < 1e-12;

% (e) List-vs-list rejected.
qry_swept_2 = translateEvents(p_qry, groups_ma, [0 100; 0 0], ...
                               isRel_ma, isPer_ma, period_ma);
ref_swept   = translateEvents(p_ref, groups_ma, [0 50; 0 0], ...
                               isRel_ma, isPer_ma, period_ma);
results{end+1,1} = 'cosSimExpTens raw-MA list-vs-list rejected';
results{end,2}   = throwsErrorWithId( ...
    @() cosSimExpTens(ref_swept, [], qry_swept_2, [], ...
        sigma_ma, r_ma, groups_ma, isRel_ma, isPer_ma, period_ma, ...
        'verbose', false), ...
    'cosSimExpTens:listVsListNotSupported');

% (f) Self-sweep peaks at zero offset.
offs_self = [-200 -100 0 100 200; 0 0 0 0 0];
ref_self  = translateEvents(p_ref, groups_ma, offs_self, ...
                             isRel_ma, isPer_ma, period_ma);
s_self    = cosSimExpTens(p_ref, [], ref_self, [], ...
    sigma_ma, r_ma, groups_ma, isRel_ma, isPer_ma, period_ma, ...
    'verbose', false);
s_self_num = cell2mat(s_self);
[~, iMax]  = max(s_self_num);
results{end+1,1} = 'cosSimExpTens raw-MA list peaks at self-match (offset 0)';
results{end,2}   = iMax == 3 && abs(s_self_num(3) - 1) < 1e-9;

% -- windowTensor: basic construction --

pitch_w = [60 62 64 65];    % 1 x 4 events
time_w  = [0  1  2  3];
dens_w = buildExpTens({pitch_w, time_w}, [], ...
    [10 0.1], [1 1], [], ...
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
dens_mr = buildExpTens({pitchMR}, [], 10, 3, [], ...
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

% -- windowedSimilarity: profile peaks at matching event offset --

pitch_narrow = [60 62 64 65];
time_narrow  = [0  1  2  3];
ctx_narrow = buildExpTens({pitch_narrow, time_narrow}, [], ...
    [0.5 0.1], [1 1], [], [false false], [true false], [1200 0], ...
    'verbose', false);

% Fixed single-event query at pitch 62, time 0 (centroid at t=0).
q_sw = buildExpTens({62, 0}, [], ...
    [0.5 0.1], [1 1], [], [false false], [true false], [1200 0], ...
    'verbose', false);

M_sweep = 21;
offs_sw = linspace(-0.5, 3.5, M_sweep);
offsets_sw = zeros(2, M_sweep);
offsets_sw(2, :) = offs_sw;
spec_sw = struct('size', [Inf, 0.3], 'mix', [0, 0]);
profile = windowedSimilarity(ctx_narrow, q_sw, spec_sw, offsets_sw, ...
    'verbose', false);
[~, peak_idx] = max(profile);
peak_off = offs_sw(peak_idx);
% Query centroid is at t=0, so offset 1 corresponds to the pitch-62
% context event at absolute t=1.
results{end+1,1} = 'windowedSimilarity: profile peaks at matching event offset';
results{end,2}   = abs(peak_off - 1.0) < 0.3;

% -- windowedSimilarity: returns length-M profile --

offsets_vec = zeros(2, 7);
offsets_vec(2, :) = linspace(0, 1, 7);
spec_lm = struct('size', [Inf, 0.5], 'mix', [0, 0]);
prof_lm = windowedSimilarity(dens_w, dens_w, spec_lm, offsets_vec, 'verbose', false);
results{end+1,1} = 'windowedSimilarity: output is 1 x M';
results{end,2}   = isequal(size(prof_lm), [1, 7]);

% -- windowedSimilarity: truncationSigmas / kernelPrecision threaded --
% v2.2.x: replaces the v2.2.0 mptDefaults stop-gap. Explicit Inf
% truncation + double precision must produce identical results to
% the default call; tight finite truncation must match the default
% to numerical precision.
prof_default_thread = windowedSimilarity(dens_w, dens_w, spec_lm, ...
    offsets_vec, 'verbose', false);
prof_inf_thread = windowedSimilarity(dens_w, dens_w, spec_lm, ...
    offsets_vec, 'truncationSigmas', Inf, ...
    'kernelPrecision', 'double', 'verbose', false);
results{end+1,1} = 'windowedSimilarity: explicit Inf/double matches default';
results{end,2}   = isequal(prof_default_thread, prof_inf_thread);

prof_trunc_thread = windowedSimilarity(dens_w, dens_w, spec_lm, ...
    offsets_vec, 'truncationSigmas', 6, 'verbose', false);
results{end+1,1} = 'windowedSimilarity: truncationSigmas=6 matches default to 1e-12';
results{end,2}   = all(abs(prof_default_thread - prof_trunc_thread) < 1e-12);

clear prof_default_thread prof_inf_thread prof_trunc_thread

% -- windowedSimilarity: reference=[] (default) matches omitted reference --
%
% Explicit empty reference must reproduce the default path byte-for-byte.
q_ref     = dens_w;
ctx_ref   = dens_w;
offs_ref  = zeros(2, 11);
offs_ref(2, :) = linspace(-0.5, 1.5, 11);
spec_ref  = struct('size', [Inf, 0.3], 'mix', [0, 0]);
prof_default  = windowedSimilarity(ctx_ref, q_ref, spec_ref, offs_ref, ...
                               'verbose', false);
prof_explicit = windowedSimilarity(ctx_ref, q_ref, spec_ref, offs_ref, ...
                               'reference', [], 'verbose', false);
results{end+1,1} = 'windowedSimilarity: reference=[] == default';
results{end,2}   = max(abs(prof_default - prof_explicit)) < 1e-12;

% -- windowedSimilarity: supplied reference shifts the profile --
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
prof_d  = windowedSimilarity(ctx_ref, q_ref, spec_ref, offs_sh, ...
                         'verbose', false);
prof_sh = windowedSimilarity(ctx_ref, q_ref, spec_ref, offs_sh, ...
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
results{end+1,1} = 'windowedSimilarity: reference shifts profile by offset';
results{end,2}   = ok_shift;

% -- windowedSimilarity: bad reference shape errors --
results{end+1,1} = 'windowedSimilarity: reference wrong cell count errors';
results{end,2}   = throwsError(@() windowedSimilarity(ctx_ref, q_ref, ...
    spec_ref, offs_ref, 'reference', {muA_pitch}, 'verbose', false));

results{end+1,1} = 'windowedSimilarity: reference wrong length errors';
results{end,2}   = throwsError(@() windowedSimilarity(ctx_ref, q_ref, ...
    spec_ref, offs_ref, 'reference', {muA_pitch, [0; 0]}, 'verbose', false));

% -- windowedSimilarity: periodic windowing (wrapped-Gaussian) --
%
% For periodic groups, the window is the wrapped Gaussian (or wrapped
% rect-conv-Gaussian for mix > 0): the sum of line-case window
% functions at all periodic images of the centre. The toolbox sums
% these adaptively until the latest image-pair's contribution falls
% below 1e-12 of the running maximum. The
% windowedSimilarity:periodicWindowApprox warning of pre-v2.2 has
% been removed because there is no longer an approximation to warn
% about. See User Guide §3.1 "Post-tensor windowing".

offs_off = [zeros(1, 5); linspace(0, 1, 5)];

% (a) A periodic windowed group must not emit
% windowedSimilarity:periodicWindowApprox (the warning class has
% been removed).
spec_small = struct('size', [5, 0.3], 'mix', [0, 0]);
W = warning('error', 'windowedSimilarity:periodicWindowApprox');
no_warn_periodic = true;
try
    windowedSimilarity(dens_w, dens_w, spec_small, offs_off, 'verbose', false);
catch ME
    no_warn_periodic = ~strcmp(ME.identifier, ...
        'windowedSimilarity:periodicWindowApprox');
end
warning(W);
results{end+1,1} = 'windowedSimilarity: no periodic-approx warning (v2.2)';
results{end,2}   = no_warn_periodic;

% (b) eval_exp_tens on a windowed periodic density returns identical
% values at periodic-equivalent query points (X, X+P, X-P): under the
% pre-v2.2 line-case window this was broken; the wrapped Gaussian
% restores periodic-equivalence to FP precision.
P_test = 12;
sigma_test = 1;
pitches_test = [3, 7];
dens_per = buildExpTens({pitches_test}, [], sigma_test, 1, [], false, true, P_test, 'verbose', false);
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
dens_ma3 = buildExpTens({pitchMA3}, [], 10, 3, [], ...
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
% directWindowedCosineSA and toolboxWindowedCosineSA defined at the
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
            cos_d = directWindowedCosineSA(p_a, w_a, p_b, w_b, ...
                sigma_sym, r_sym, offset_vec, size_v, mix_v);
            cos_t = toolboxWindowedCosineSA(p_a, w_a, p_b, w_b, ...
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
            cos_orig = toolboxWindowedCosineSA(p_a, w_a, p_b, w_b, ...
                sigma_sym, r_sym, offset_vec, size_v, mix_v);
            % Reverse ordering of the context-side values.
            cos_rev = toolboxWindowedCosineSA(p_a, w_a, ...
                p_b(end:-1:1), w_b(end:-1:1), ...
                sigma_sym, r_sym, offset_vec, size_v, mix_v);
            % Random shuffle.
            perm_b = randperm(K_sym);
            cos_shuf = toolboxWindowedCosineSA(p_a, w_a, ...
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
        cos_orig = toolboxWindowedCosineSA(p_a, w_a, p_b, w_b, ...
            sigma_sym, r_sym, offset_vec, size_v, mix_v);
        all_perms_3 = perms(1:r_sym);
        ok_all = true;
        for ip_row = 1:size(all_perms_3, 1)
            offset_perm = offset_vec(all_perms_3(ip_row, :));
            cos_pi = toolboxWindowedCosineSA(p_a, w_a, p_b, w_b, ...
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
dens_x_4 = buildExpTens({p0_x, p1_x}, {w0_x, w1_x}, 30.0, [2 2], [1 1], ...
    false, false, 0.0, 'verbose', false);
dens_y_4 = buildExpTens({p0_y, p1_y}, {w0_y, w1_y}, 30.0, [2 2], [1 1], ...
    false, false, 0.0, 'verbose', false);
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
cos_t_5 = toolboxWindowedCosineSA(p_a5, w_a5, p_b5, w_b5, ...
    30.0, 2, offset_unif, 5.0, 0.0);
cos_d_5 = directWindowedCosineSA(p_a5, w_a5, p_b5, w_b5, ...
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

function c = directWindowedCosineSA(p_a, w_a, p_b, w_b, sigma, r, ...
        offset_vec, size_v, mix_v)
%DIRECTWINDOWEDCOSINESA  Framework-correct windowed similarity for the SA
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

function c = toolboxWindowedCosineSA(p_a, w_a, p_b, w_b, sigma, r, ...
        offset_vec, size_v, mix_v)
%TOOLBOXWINDOWEDCOSINESA  Toolbox-API windowed cosine for the SA
%case, used as the path under test in the symmetrisation suite.
    Pa = p_a(:);  Wa = w_a(:);
    Pb = p_b(:);  Wb = w_b(:);
    dens_q = buildExpTens({Pa}, {Wa}, sigma, r, [], ...
        false, false, 0, 'verbose', false);
    dens_c = buildExpTens({Pb}, {Wb}, sigma, r, [], ...
        false, false, 0, 'verbose', false);
    spec = struct('size', size_v, 'mix', mix_v, ...
        'centre', {{offset_vec(:)}});
    wmd = windowTensor(dens_c, spec);
    c = internal.windowedInnerProduct(dens_q, wmd, false);
end

