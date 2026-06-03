%% test_sym.m — per-attribute [sym] (symmetrisation) flag
%
%  Mirror of the Python tests/test_sym.py. Covers the sym-flag
%  specification §10:
%
%   * r = 1: the flag is vacuous; isSym true/false coincide (SA + MA).
%   * r = K: isSym = false deposits the single ordered tuple; isSym =
%     true deposits the full S_K orbit (K! tuples).
%   * 1 < r < K: isSym = false is the de-reflected isSym = true density,
%     verified by the exact orbit-sum relation at r = 2.
%   * OPT-completeness: isSym = false with isRel = true reaches the
%     ordered transposition-invariant spaces (an ordered interval is
%     distinguished from its inversion; isSym = true cannot).
%   * Cross-cardinality comparability at fixed r; doubling reweights
%     without equalising (anti-C).
%   * Default value is symmetric; ordered self-similarity is 1.
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


% ---------------------------------------------------------------------
%  r = 1: the flag is vacuous
% ---------------------------------------------------------------------

p4 = [0 4 7 11];
xg = linspace(-3, 14, 60);
d_sym = buildExpTens(p4, [], 1, 1, false, false, 0, true,  'verbose', false);
d_ord = buildExpTens(p4, [], 1, 1, false, false, 0, false, 'verbose', false);
v_sym = evalExpTens(d_sym, xg, 'verbose', false);
v_ord = evalExpTens(d_ord, xg, 'verbose', false);
results{end+1,1} = 'sym: r=1 vacuous (SA eval coincides)';
results{end,2}   = max(abs(v_sym(:) - v_ord(:))) < 1e-12;

Pma = {[0 4 7]};   % one attribute, 1 slot, 3 events
xm = linspace(-3, 12, 50);
dms = buildExpTens(Pma, [], 1, 1, false, false, 0, true,  'verbose', false);
dmo = buildExpTens(Pma, [], 1, 1, false, false, 0, false, 'verbose', false);
vms = evalExpTens(dms, xm, 'verbose', false);
vmo = evalExpTens(dmo, xm, 'verbose', false);
results{end+1,1} = 'sym: r=1 vacuous (MA eval coincides)';
results{end,2}   = max(abs(vms(:) - vmo(:))) < 1e-12;


% ---------------------------------------------------------------------
%  Centre counts: ordered C(K,r) vs symmetric r! * C(K,r)
% ---------------------------------------------------------------------

K = 4;                       % p4 has four values
rTests   = [1 2 3 4];
expOrd   = [4 6 4 1];        % C(4,r)
expSym   = [4 12 24 24];     % r! * C(4,r)
okCounts = true;
for ii = 1:numel(rTests)
    rr = rTests(ii);
    do_ = internal.ensureExpTensExpensive( ...
        buildExpTens(p4, [], 1, rr, false, false, 0, false, 'verbose', false));
    ds_ = internal.ensureExpTensExpensive( ...
        buildExpTens(p4, [], 1, rr, false, false, 0, true,  'verbose', false));
    okCounts = okCounts ...
        && size(do_.U_perm, 2) == expOrd(ii) ...
        && size(ds_.U_perm, 2) == expSym(ii);
end
results{end+1,1} = 'sym: SA u_perm column counts (ordered vs symmetric)';
results{end,2}   = okCounts;

% r = K: single ordered tuple in listed order.
pUns = [3 1 8];
dK = internal.ensureExpTensExpensive( ...
    buildExpTens(pUns, [], 1, 3, false, false, 0, false, 'verbose', false));
results{end+1,1} = 'sym: r=K single ordered tuple in listed order';
results{end,2}   = size(dK.U_perm, 2) == 1 ...
                && isequal(dK.U_perm(:).', [3 1 8]);

% MA centre counts (one event, 3 slots, r = 2).
PmaC = {[0; 4; 7]};
dmo2 = internal.ensureExpTensExpensive( ...
    buildExpTens(PmaC, [], 1, 2, false, false, 0, false, 'verbose', false));
dms2 = internal.ensureExpTensExpensive( ...
    buildExpTens(PmaC, [], 1, 2, false, false, 0, true,  'verbose', false));
results{end+1,1} = 'sym: MA u_perm column counts (ordered vs symmetric)';
results{end,2}   = size(dmo2.U_perm{1}, 2) == 3 ...   % C(3,2)
                && size(dms2.U_perm{1}, 2) == 6;       % x2!


% ---------------------------------------------------------------------
%  1 < r < K: the de-reflected density relation (r = 2)
% ---------------------------------------------------------------------

p3 = [0 4 7];
do2 = buildExpTens(p3, [], 1.3, 2, false, false, 0, false, 'verbose', false);
ds2 = buildExpTens(p3, [], 1.3, 2, false, false, 0, true,  'verbose', false);
rng(0);
q = -2 + 11 * rand(2, 25);
qsw = q([2 1], :);
ev_sym = evalExpTens(ds2, q,   'verbose', false);
ev_ord = evalExpTens(do2, q,   'verbose', false);
ev_swp = evalExpTens(do2, qsw, 'verbose', false);
results{end+1,1} = 'sym: r=2 symmetric = ordered + reflection';
results{end,2}   = max(abs(ev_sym(:) - (ev_ord(:) + ev_swp(:)))) < 1e-11;

% Order sensitivity: ascending vs descending.
asc  = [0 4 7];
desc = [7 3 0];
s_sym = cosSimExpTens( ...
    buildExpTens(asc,  [], 50, 2, false, false, 0, true, 'verbose', false), ...
    buildExpTens(desc, [], 50, 2, false, false, 0, true, 'verbose', false), ...
    'verbose', false);
s_ord = cosSimExpTens( ...
    buildExpTens(asc,  [], 50, 2, false, false, 0, false, 'verbose', false), ...
    buildExpTens(desc, [], 50, 2, false, false, 0, false, 'verbose', false), ...
    'verbose', false);
results{end+1,1} = 'sym: ordered distinguishes order, symmetric does not';
results{end,2}   = s_ord < s_sym - 1e-4;

% Large-K routing: at a cardinality where the orbit (Möbius) path would
% otherwise be selected, an ordered cosine must still be forced onto the
% centres path and stay distinct from the symmetric reading.
ascL  = 0:7;
descL = 7:-1:0;
sL_sym = cosSimExpTens( ...
    buildExpTens(ascL,  [], 50, 2, false, false, 0, true, 'verbose', false), ...
    buildExpTens(descL, [], 50, 2, false, false, 0, true, 'verbose', false), ...
    'verbose', false);
sL_ord = cosSimExpTens( ...
    buildExpTens(ascL,  [], 50, 2, false, false, 0, false, 'verbose', false), ...
    buildExpTens(descL, [], 50, 2, false, false, 0, false, 'verbose', false), ...
    'verbose', false);
results{end+1,1} = 'sym: large-K ordered cosine not symmetrised (SA routing)';
results{end,2}   = abs(sL_sym - 1) < 1e-9 && sL_ord < 1 - 1e-4;

ML  = {(0:5).'};        % 6 slots, 1 event
MLr = {(5:-1:0).'};
mL_sym = cosSimExpTens(ML, [], MLr, [], 50, 2, false, false, 0, true,  'verbose', false);
mL_ord = cosSimExpTens(ML, [], MLr, [], 50, 2, false, false, 0, false, 'verbose', false);
results{end+1,1} = 'sym: large-K ordered cosine not symmetrised (MA routing)';
results{end,2}   = abs(mL_sym - 1) < 1e-9 && mL_ord < 1 - 1e-4;


% ---------------------------------------------------------------------
%  OPT-completeness: ordered transposition-invariant reach
% ---------------------------------------------------------------------

dRel = buildExpTens(p3, [], 1, 3, true, false, 0, false, 'verbose', false);
results{end+1,1} = 'sym: isRel drops dim by one (independent of isSym)';
results{end,2}   = dRel.dim == 2;

up   = [0 4];     % ordered interval +4
down = [0 -4];    % ordered interval -4
s_sym_oi = cosSimExpTens( ...
    buildExpTens(up,   [], 1, 2, true, false, 0, true, 'verbose', false), ...
    buildExpTens(down, [], 1, 2, true, false, 0, true, 'verbose', false), ...
    'verbose', false);
s_ord_oi = cosSimExpTens( ...
    buildExpTens(up,   [], 1, 2, true, false, 0, false, 'verbose', false), ...
    buildExpTens(down, [], 1, 2, true, false, 0, false, 'verbose', false), ...
    'verbose', false);
results{end+1,1} = 'sym: ordered interval vs inversion (sym=1 same, sym=0 differ)';
results{end,2}   = abs(s_sym_oi - 1) < 1e-9 && s_ord_oi < 0.5;


% ---------------------------------------------------------------------
%  Cross-cardinality comparability and anti-C
% ---------------------------------------------------------------------

triad   = [0 400 700];
seventh = [0 400 700 1000];
s_cc = cosSimExpTens( ...
    buildExpTens(triad,   [], 30, 2, false, false, 0, false, 'verbose', false), ...
    buildExpTens(seventh, [], 30, 2, false, false, 0, false, 'verbose', false), ...
    'verbose', false);
results{end+1,1} = 'sym: triad vs seventh well posed (0 < s < 1, finite)';
results{end,2}   = isfinite(s_cc) && s_cc > 0 && s_cc < 1;

doubled = [0 0 400 700];   % doubled root
s_db = cosSimExpTens( ...
    buildExpTens(triad,   [], 30, 2, false, false, 0, false, 'verbose', false), ...
    buildExpTens(doubled, [], 30, 2, false, false, 0, false, 'verbose', false), ...
    'verbose', false);
results{end+1,1} = 'sym: doubling reweights without equalising (anti-C)';
results{end,2}   = isfinite(s_db) && s_db < 1 - 1e-6 && s_db > 0.5;


% ---------------------------------------------------------------------
%  Default value and self-similarity
% ---------------------------------------------------------------------

dDef = internal.ensureExpTensExpensive( ...
    buildExpTens(p3, [], 1, 2, false, false, 0, 'verbose', false));
dSym = internal.ensureExpTensExpensive( ...
    buildExpTens(p3, [], 1, 2, false, false, 0, true, 'verbose', false));
results{end+1,1} = 'sym: default is symmetric';
results{end,2}   = size(dDef.U_perm, 2) == size(dSym.U_perm, 2) ...
                && all(logical(dDef.isSym(:)));

dSelf = buildExpTens(p3, [], 30, 2, false, false, 0, false, 'verbose', false);
results{end+1,1} = 'sym: ordered self-similarity is 1';
results{end,2}   = abs(cosSimExpTens(dSelf, dSelf, 'verbose', false) - 1) < 1e-9;


% ---------------------------------------------------------------------
%  Batched (2-D) input rejects ordered at r > 1 (parity with Python)
% ---------------------------------------------------------------------

P2 = [0 4 7; 7 4 0];
results{end+1,1} = 'sym: batched cosSim rejects isSym=false at r>1';
results{end,2}   = errorMessageContains( ...
    @() cosSimExpTens(P2, [], P2, [], 30, 2, false, false, 0, false, ...
                      'verbose', false), 'ordered');
results{end+1,1} = 'sym: batched eval rejects isSym=false at r>1';
results{end,2}   = errorMessageContains( ...
    @() evalExpTens(P2, [], 30, 2, false, false, 0, false, [0;4], ...
                    'verbose', false), 'ordered');
results{end+1,1} = 'sym: renyi2 rejects isSym=false at r>1';
results{end,2}   = errorMessageContains( ...
    @() entropyExpTens(p3, [], 1, 2, false, false, 0, false, ...
                       'method', 'renyi2'), 'ordered');


% ---------------------------------------------------------------------
%  Standalone summary
% ---------------------------------------------------------------------

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
    fprintf('\n=== test_sym: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_sym:failed', '%d test(s) failed.', nFail);
    end
end
