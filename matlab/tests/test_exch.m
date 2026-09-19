%% test_exch.m — per-attribute [exch] (symmetrisation) flag
%
%  Mirror of the Python tests/test_exch.py. Covers the exch-flag
%  specification §10:
%
%   * r = 1: the flag is vacuous; isExch true/false coincide (single-multiset + MA).
%   * r = K: isExch = false deposits the single ordered tuple; isExch =
%     true deposits the full S_K orbit (K! tuples).
%   * 1 < r < K: isExch = false is the de-reflected isExch = true density,
%     verified by the exact orbit-sum relation at r = 2.
%   * OPT-completeness: isExch = false with isRel = true reaches the
%     ordered transposition-invariant spaces (an ordered interval is
%     distinguished from its inversion; isExch = true cannot) --- both the
%     line R^{n-1} and, with isPer = true, the torus T^{n-1}. The three
%     Sym/Ord confirmations are also checked under isPer = true.
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
d_exch = buildMaet(p4, [], 1, 1, false, false, 0, true,  'verbose', false);
d_ord = buildMaet(p4, [], 1, 1, false, false, 0, false, 'verbose', false);
v_exch = evalMaet(d_exch, xg, 'verbose', false);
v_ord = evalMaet(d_ord, xg, 'verbose', false);
results{end+1,1} = 'exch: r=1 vacuous (single-multiset eval coincides)';
results{end,2}   = max(abs(v_exch(:) - v_ord(:))) < 1e-12;

Pma = {[0 4 7]};   % one attribute, 1 value, 3 events
xm = linspace(-3, 12, 50);
dms = buildMaet(Pma, [], 1, 1, false, false, 0, true,  'verbose', false);
dmo = buildMaet(Pma, [], 1, 1, false, false, 0, false, 'verbose', false);
vms = evalMaet(dms, xm, 'verbose', false);
vmo = evalMaet(dmo, xm, 'verbose', false);
results{end+1,1} = 'exch: r=1 vacuous (MA eval coincides)';
results{end,2}   = max(abs(vms(:) - vmo(:))) < 1e-12;


% ---------------------------------------------------------------------
%  Centre counts: ordered C(K,r) vs symmetric r! * C(K,r)
% ---------------------------------------------------------------------

K = 4;                       % p4 has four values
rTests   = [1 2 3 4];
expOrd   = [4 6 4 1];        % C(4,r)
expExch   = [4 12 24 24];     % r! * C(4,r)
okCounts = true;
for ii = 1:numel(rTests)
    rr = rTests(ii);
    do_ = internal.ensureMaetExpensive( ...
        buildMaet(p4, [], 1, rr, false, false, 0, false, 'verbose', false));
    ds_ = internal.ensureMaetExpensive( ...
        buildMaet(p4, [], 1, rr, false, false, 0, true,  'verbose', false));
    okCounts = okCounts ...
        && size(do_.U_perm{1}, 2) == expOrd(ii) ...
        && size(ds_.U_perm{1}, 2) == expExch(ii);
end
results{end+1,1} = 'exch: single-multiset u_perm column counts (ordered vs symmetric)';
results{end,2}   = okCounts;

% r = K: single ordered tuple in listed order.
pUns = [3 1 8];
dK = internal.ensureMaetExpensive( ...
    buildMaet(pUns, [], 1, 3, false, false, 0, false, 'verbose', false));
results{end+1,1} = 'exch: r=K single ordered tuple in listed order';
results{end,2}   = size(dK.U_perm{1}, 2) == 1 ...
                && isequal(dK.U_perm{1}(:).', [3 1 8]);

% MA centre counts (one event, 3 values, r = 2).
PmaC = {[0; 4; 7]};
dmo2 = internal.ensureMaetExpensive( ...
    buildMaet(PmaC, [], 1, 2, false, false, 0, false, 'verbose', false));
dms2 = internal.ensureMaetExpensive( ...
    buildMaet(PmaC, [], 1, 2, false, false, 0, true,  'verbose', false));
results{end+1,1} = 'exch: MA u_perm column counts (ordered vs symmetric)';
results{end,2}   = size(dmo2.U_perm{1}, 2) == 3 ...   % C(3,2)
                && size(dms2.U_perm{1}, 2) == 6;       % x2!


% ---------------------------------------------------------------------
%  1 < r < K: the de-reflected density relation (r = 2)
% ---------------------------------------------------------------------

p3 = [0 4 7];
do2 = buildMaet(p3, [], 1.3, 2, false, false, 0, false, 'verbose', false);
ds2 = buildMaet(p3, [], 1.3, 2, false, false, 0, true,  'verbose', false);
rng(0);
q = -2 + 11 * rand(2, 25);
qsw = q([2 1], :);
ev_exch = evalMaet(ds2, q,   'verbose', false);
ev_ord = evalMaet(do2, q,   'verbose', false);
ev_swp = evalMaet(do2, qsw, 'verbose', false);
results{end+1,1} = 'exch: r=2 symmetric = ordered + reflection';
results{end,2}   = max(abs(ev_exch(:) - (ev_ord(:) + ev_swp(:)))) < 1e-11;

% Order sensitivity: ascending vs descending.
asc  = [0 4 7];
desc = [7 3 0];
s_exch = simMaet( ...
    buildMaet(asc,  [], 50, 2, false, false, 0, true, 'verbose', false), ...
    buildMaet(desc, [], 50, 2, false, false, 0, true, 'verbose', false), ...
    'verbose', false);
s_ord = simMaet( ...
    buildMaet(asc,  [], 50, 2, false, false, 0, false, 'verbose', false), ...
    buildMaet(desc, [], 50, 2, false, false, 0, false, 'verbose', false), ...
    'verbose', false);
results{end+1,1} = 'exch: ordered distinguishes order, symmetric does not';
results{end,2}   = s_ord < s_exch - 1e-4;

% Large-K routing: at a cardinality where the orbit (Möbius) path would
% otherwise be selected, an ordered cosine must still be forced onto the
% centres path and stay distinct from the symmetric reading.
ascL  = 0:7;
descL = 7:-1:0;
sL_exch = simMaet( ...
    buildMaet(ascL,  [], 50, 2, false, false, 0, true, 'verbose', false), ...
    buildMaet(descL, [], 50, 2, false, false, 0, true, 'verbose', false), ...
    'verbose', false);
sL_ord = simMaet( ...
    buildMaet(ascL,  [], 50, 2, false, false, 0, false, 'verbose', false), ...
    buildMaet(descL, [], 50, 2, false, false, 0, false, 'verbose', false), ...
    'verbose', false);
results{end+1,1} = 'exch: large-K ordered cosine not symmetrised (single-multiset routing)';
results{end,2}   = abs(sL_exch - 1) < 1e-9 && sL_ord < 1 - 1e-4;

ML  = {(0:5).'};        % 6 values, 1 event
MLr = {(5:-1:0).'};
mL_exch = simMaet(ML, [], MLr, [], 50, 2, false, false, 0, true,  'verbose', false);
mL_ord = simMaet(ML, [], MLr, [], 50, 2, false, false, 0, false, 'verbose', false);
results{end+1,1} = 'exch: large-K ordered cosine not symmetrised (MA routing)';
results{end,2}   = abs(mL_exch - 1) < 1e-9 && mL_ord < 1 - 1e-4;


% ---------------------------------------------------------------------
%  OPT-completeness: ordered transposition-invariant reach
% ---------------------------------------------------------------------

dRel = buildMaet(p3, [], 1, 3, true, false, 0, false, 'verbose', false);
results{end+1,1} = 'exch: isRel drops dim by one (independent of isExch)';
results{end,2}   = dRel.dim == 2;

up   = [0 4];     % ordered interval +4
down = [0 -4];    % ordered interval -4
s_exch_oi = simMaet( ...
    buildMaet(up,   [], 1, 2, true, false, 0, true, 'verbose', false), ...
    buildMaet(down, [], 1, 2, true, false, 0, true, 'verbose', false), ...
    'verbose', false);
s_ord_oi = simMaet( ...
    buildMaet(up,   [], 1, 2, true, false, 0, false, 'verbose', false), ...
    buildMaet(down, [], 1, 2, true, false, 0, false, 'verbose', false), ...
    'verbose', false);
results{end+1,1} = 'exch: ordered interval vs inversion (exch=1 same, exch=0 differ)';
results{end,2}   = abs(s_exch_oi - 1) < 1e-9 && s_ord_oi < 0.5;


% ---------------------------------------------------------------------
%  Periodic + ordered (isPer = true with isExch = false): the second of
%  the two ordered transposition-invariant spaces (the torus T^{n-1}),
%  and the isPer = true arm of the r-sweep confirmations.
% ---------------------------------------------------------------------

Pp = 12;

% isRel drops one dimension on the torus too (T^{n-1}).
dRelP = buildMaet(p3, [], 1, 2, true, true, Pp, false, 'verbose', false);
results{end+1,1} = 'exch: periodic isRel drops dim by one (T^{n-1})';
results{end,2}   = dRelP.dim == 1;

% Ordered relative on a period-12 torus distinguishes +4 from -4 (==+8);
% symmetric symmetrises the pair so the two orbits {4, 8} coincide.
upP   = [0 4];     % +4
downP = [0 -4];    % -4 == +8 (mod 12)
s_exch_poi = simMaet( ...
    buildMaet(upP,   [], 0.5, 2, true, true, Pp, true, 'verbose', false), ...
    buildMaet(downP, [], 0.5, 2, true, true, Pp, true, 'verbose', false), ...
    'verbose', false);
s_ord_poi = simMaet( ...
    buildMaet(upP,   [], 0.5, 2, true, true, Pp, false, 'verbose', false), ...
    buildMaet(downP, [], 0.5, 2, true, true, Pp, false, 'verbose', false), ...
    'verbose', false);
results{end+1,1} = 'exch: periodic ordered interval vs inversion (T^{n-1})';
results{end,2}   = abs(s_exch_poi - 1) < 1e-9 && s_ord_poi < 0.5;

% isPer wraps the value axis in the ordered path: [0 4] equals [0 16]
% (16 == 4 mod 12) when periodic, but is distinct when not.
s_wrap = simMaet( ...
    buildMaet([0 4],  [], 0.5, 2, false, true, Pp, false, 'verbose', false), ...
    buildMaet([0 16], [], 0.5, 2, false, true, Pp, false, 'verbose', false), ...
    'verbose', false);
s_nowrap = simMaet( ...
    buildMaet([0 4],  [], 0.5, 2, false, false, 0, false, 'verbose', false), ...
    buildMaet([0 16], [], 0.5, 2, false, false, 0, false, 'verbose', false), ...
    'verbose', false);
results{end+1,1} = 'exch: periodic wrapping active in ordered mode';
results{end,2}   = abs(s_wrap - 1) < 1e-9 && s_nowrap < 0.5;

% r = K under isPer: ordered deposits one whole tuple, symmetric the
% full S_K orbit (K! = 6). Periodicity does not change the orbit count.
dKpo = internal.ensureMaetExpensive( ...
    buildMaet(p3, [], 1, 3, false, true, Pp, false, 'verbose', false));
dKps = internal.ensureMaetExpensive( ...
    buildMaet(p3, [], 1, 3, false, true, Pp, true, 'verbose', false));
results{end+1,1} = 'exch: periodic r = K single ordered tuple vs S_K orbit';
results{end,2}   = size(dKpo.U_perm{1}, 2) == 1 && size(dKps.U_perm{1}, 2) == 6;

% r = 1 under isPer: the flag is vacuous, so ordered and symmetric
% periodic densities are identical pointwise.
xgp   = linspace(-3, 14, 60);
d1po  = buildMaet(p3, [], 1, 1, false, true, Pp, false, 'verbose', false);
d1ps  = buildMaet(p3, [], 1, 1, false, true, Pp, true,  'verbose', false);
v1po  = evalMaet(d1po, xgp, 'verbose', false);
v1ps  = evalMaet(d1ps, xgp, 'verbose', false);
results{end+1,1} = 'exch: periodic r = 1 ordered and symmetric coincide';
results{end,2}   = max(abs(v1po(:) - v1ps(:))) < 1e-12;

% Self-similarity of a periodic ordered relative density is exactly 1.
dSelfP = buildMaet(p3, [], 30, 2, true, true, Pp, false, 'verbose', false);
results{end+1,1} = 'exch: periodic ordered relative self-similarity is 1';
results{end,2}   = abs(simMaet(dSelfP, dSelfP, 'verbose', false) - 1) < 1e-9;


% ---------------------------------------------------------------------
%  Cross-cardinality comparability and anti-C
% ---------------------------------------------------------------------

triad   = [0 400 700];
seventh = [0 400 700 1000];
s_cc = simMaet( ...
    buildMaet(triad,   [], 30, 2, false, false, 0, false, 'verbose', false), ...
    buildMaet(seventh, [], 30, 2, false, false, 0, false, 'verbose', false), ...
    'verbose', false);
results{end+1,1} = 'exch: triad vs seventh well posed (0 < s < 1, finite)';
results{end,2}   = isfinite(s_cc) && s_cc > 0 && s_cc < 1;

doubled = [0 0 400 700];   % doubled root
s_db = simMaet( ...
    buildMaet(triad,   [], 30, 2, false, false, 0, false, 'verbose', false), ...
    buildMaet(doubled, [], 30, 2, false, false, 0, false, 'verbose', false), ...
    'verbose', false);
results{end+1,1} = 'exch: doubling reweights without equalising (anti-C)';
results{end,2}   = isfinite(s_db) && s_db < 1 - 1e-6 && s_db > 0.5;


% ---------------------------------------------------------------------
%  Default value and self-similarity
% ---------------------------------------------------------------------

dDef = internal.ensureMaetExpensive( ...
    buildMaet(p3, [], 1, 2, false, false, 0, 'verbose', false));
dExch = internal.ensureMaetExpensive( ...
    buildMaet(p3, [], 1, 2, false, false, 0, true, 'verbose', false));
results{end+1,1} = 'exch: default is symmetric';
results{end,2}   = size(dDef.U_perm{1}, 2) == size(dExch.U_perm{1}, 2) ...
                && all(logical(dDef.isExch(:)));

dSelf = buildMaet(p3, [], 30, 2, false, false, 0, false, 'verbose', false);
results{end+1,1} = 'exch: ordered self-similarity is 1';
results{end,2}   = abs(simMaet(dSelf, dSelf, 'verbose', false) - 1) < 1e-9;


% ---------------------------------------------------------------------
%  Batched (2-D) input rejects ordered at r > 1 (parity with Python)
% ---------------------------------------------------------------------

P2 = [0 4 7; 7 4 0];
results{end+1,1} = 'exch: batched cosSim rejects isExch=false at r>1';
results{end,2}   = errorMessageContains( ...
    @() simMaet(P2, [], P2, [], 30, 2, false, false, 0, false, ...
                      'verbose', false), 'ordered');
results{end+1,1} = 'exch: batched eval rejects isExch=false at r>1';
results{end,2}   = errorMessageContains( ...
    @() evalMaet(P2, [], 30, 2, false, false, 0, false, [0;4], ...
                    'verbose', false), 'ordered');


% ---------------------------------------------------------------------
%  Ordered renyi2 at r > 1 computes (no orbit); golden vs Python
% ---------------------------------------------------------------------

% single-multiset raw form: ordered absolute r=2, sigma=1, base 2 (default).
hOrdSingleMultiset = entropyMaet(p3, [], 1, 2, false, false, 0, false, ...
                        'method', 'renyi2', 'verbose', false);
results{end+1,1} = 'exch: renyi2 single-multiset ordered r2 finite';
results{end,2}   = isfinite(hOrdSingleMultiset);
results{end+1,1} = 'exch: renyi2 single-multiset ordered r2 matches Python golden';
results{end,2}   = abs(hOrdSingleMultiset - 5.120408605589989) < 1e-6;

% r = 1 ordered is exempt ([exch] vacuous) and must still compute.
hOrdR1 = entropyMaet(p3, [], 1, 1, false, false, 0, false, ...
                        'method', 'renyi2', 'verbose', false);
results{end+1,1} = 'exch: renyi2 single-multiset ordered r1 finite';
results{end,2}   = isfinite(hOrdR1);

% MA flat-ordered attribute (positional form): r=2 absolute, sigma=2.
dOrdMA = buildMaet({[0 4 7 11].'}, [], 2, 2, false, false, 0, false, ...
                      'verbose', false);
hOrdMA = entropyMaet(dOrdMA, 'method', 'renyi2', 'verbose', false);
results{end+1,1} = 'exch: renyi2 MA ordered r2 matches Python golden';
results{end,2}   = abs(hOrdMA - 7.136351706681117) < 1e-6;


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
    fprintf('\n=== test_exch: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_exch:failed', '%d test(s) failed.', nFail);
    end
end
