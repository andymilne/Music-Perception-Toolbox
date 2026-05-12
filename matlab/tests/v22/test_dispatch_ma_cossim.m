%% test_dispatch_ma_cossim.m — v2.2 MA orbit dispatch in cosSimExpTens
%
%  Tests for the MA-side method dispatch added in v2.2 (Commit 6c).
%  Covers:
%    - Method kwarg validation: bad string raises informative error.
%    - Orbit and pairwise paths agree to numerical tolerance on healthy
%      regimes (r >= 3, abs and rel modes, periodic and non-periodic).
%    - Auto routing: rel groups always pairwise; r_max < 3 always
%      pairwise; r_max >= 3 abs routes to orbit.
%    - sigma/period > 0.03 in rel+per groups falls back to pairwise
%      with a warning.
%    - K_{a,n}-vs-r margin: ragged K (NaN-padded events) handled
%      transparently via zero-pad inside the orbit wrapper.
%    - Skinny dens flows through the dispatcher without forcing eager
%      build on the orbit branch.
%    - Self-similarity remains 1 along the orbit branch.
%
%  Standalone-runnable. When invoked from test_mpt.m the existing
%  `results` cell is appended to; when run alone, results print on the
%  fly and a summary follows.

if ~exist('results', 'var')
    results = {};
    standalone = true;
else
    standalone = false;
end

%% ---- Method kwarg validation ----

% A trivially valid MA setup (single attribute) so the dispatch site
% reaches the method-validation branch.
pAttr = {[0; 4; 7]};
wA    = {ones(3, 1)};
sigma = 30;  rVec = 2;  groups = 1;
isRel = false;  isPer = false;  period = 0;

ok_badMethod = false;
try
    cosSimExpTens(pAttr, wA, pAttr, wA, sigma, rVec, groups, ...
        isRel, isPer, period, 'method', 'bogus', 'verbose', false);
catch ME
    ok_badMethod = strcmp(ME.identifier, 'cosSimExpTens:badMethod');
end
results{end+1,1} = 'dispatch.MA cossim: bad method string raises cosSimExpTens:badMethod';
results{end,2}   = ok_badMethod;

%% ---- r_max >= 3 abs nonper: auto routes to orbit; agrees with pairwise ----

rng(51, 'twister');
% Two attributes, r=3 each, abs nonper, modest K.
N1 = 4;  K_pa = 6;  K_pb = 6;
PxA = sort(2000 * rand(K_pa, N1));   WxA = ones(K_pa, N1);
PxB = sort(1500 * rand(K_pb, N1));   WxB = ones(K_pb, N1);
PyA = sort(2000 * rand(K_pa, N1));   WyA = ones(K_pa, N1);
PyB = sort(1500 * rand(K_pb, N1));   WyB = ones(K_pb, N1);

dx = buildExpTens({PxA, PxB}, {WxA, WxB}, [30 30], [3 3], [1 2], ...
    [false false], [false false], [0 0], 'verbose', false);
dy = buildExpTens({PyA, PyB}, {WyA, WyB}, [30 30], [3 3], [1 2], ...
    [false false], [false false], [0 0], 'verbose', false);

s_auto    = cosSimExpTens(dx, dy, 'verbose', false);
s_orbit   = cosSimExpTens(dx, dy, 'method', 'orbit', 'verbose', false);
s_pwise   = cosSimExpTens(dx, dy, 'method', 'pairwise', 'verbose', false);

results{end+1,1} = 'dispatch.MA cossim: r=3 abs nonper auto matches pairwise (1e-10)';
results{end,2}   = abs(s_auto - s_pwise) < 1e-10;
results{end+1,1} = 'dispatch.MA cossim: r=3 abs nonper orbit matches pairwise (1e-8)';
results{end,2}   = abs(s_orbit - s_pwise) < 1e-8;

%% ---- r_max = 4 abs periodic: orbit/pairwise agree ----

rng(53, 'twister');
period_p = 1200;
PxP = sort(period_p * rand(7, 3));   WxP = ones(7, 3);
PyP = sort(period_p * rand(7, 3));   WyP = ones(7, 3);
dxp = buildExpTens({PxP}, {WxP}, 30, 4, 1, false, true, period_p, ...
    'verbose', false);
dyp = buildExpTens({PyP}, {WyP}, 30, 4, 1, false, true, period_p, ...
    'verbose', false);
s_orb_per = cosSimExpTens(dxp, dyp, 'method', 'orbit', 'verbose', false);
s_pwi_per = cosSimExpTens(dxp, dyp, 'method', 'pairwise', 'verbose', false);
results{end+1,1} = 'dispatch.MA cossim: r=4 abs per orbit matches pairwise (1e-8)';
results{end,2}   = abs(s_orb_per - s_pwi_per) < 1e-8;

%% ---- Rel groups: auto routes to pairwise (no warning) ----

rng(55, 'twister');
period_r = 1200;
PxR = sort(period_r * rand(8, 3));   WxR = ones(8, 3);
PyR = sort(period_r * rand(8, 3));   WyR = ones(8, 3);
% sigma/period = 30/1200 = 0.025, just under 0.03 threshold -> no warning,
% but rel-group rule still routes to pairwise.
dxR = buildExpTens({PxR}, {WxR}, 30, 3, 1, true, true, period_r, ...
    'verbose', false);
dyR = buildExpTens({PyR}, {WyR}, 30, 3, 1, true, true, period_r, ...
    'verbose', false);
% Auto and pairwise should give identical answers (auto picks pairwise).
s_auto_rel  = cosSimExpTens(dxR, dyR, 'verbose', false);
s_pwise_rel = cosSimExpTens(dxR, dyR, 'method', 'pairwise', 'verbose', false);
results{end+1,1} = 'dispatch.MA cossim: rel-group auto = pairwise (exact)';
results{end,2}   = isequal(s_auto_rel, s_pwise_rel);

%% ---- Explicit orbit on rel groups: runs un-vectorised, agrees with pairwise ----

s_orb_rel = cosSimExpTens(dxR, dyR, 'method', 'orbit', 'verbose', false);
results{end+1,1} = 'dispatch.MA cossim: explicit orbit on rel groups matches pairwise (1e-6)';
results{end,2}   = abs(s_orb_rel - s_pwise_rel) < 1e-6;

%% ---- sigma/period > 0.03 fallback warning ----

rng(57, 'twister');
period_w = 100;
PxW = sort(period_w * rand(8, 2));   WxW = ones(8, 2);
PyW = sort(period_w * rand(8, 2));   WyW = ones(8, 2);
dxW = buildExpTens({PxW}, {WxW}, 30, 3, 1, true, true, period_w, ...
    'verbose', false);  % sigma/P = 0.30 -> well above 0.03
dyW = buildExpTens({PyW}, {WyW}, 30, 3, 1, true, true, period_w, ...
    'verbose', false);

w_state = warning('on', 'cosSimExpTens:orbitSigmaOverPFallback');
lastwarn('');
s_warn = cosSimExpTens(dxW, dyW, 'verbose', true);  %#ok<NASGU>
[~, lastID] = lastwarn;
warning(w_state);
results{end+1,1} = 'dispatch.MA cossim: sigma/P > 0.03 in rel+per emits fallback warning';
results{end,2}   = strcmp(lastID, 'cosSimExpTens:orbitSigmaOverPFallback');

%% ---- r_max < 3 auto routes to pairwise ----

% Same setup as the r=3 abs case but r=2 throughout.
dx2 = buildExpTens({PxA, PxB}, {WxA, WxB}, [30 30], [2 2], [1 2], ...
    [false false], [false false], [0 0], 'verbose', false);
dy2 = buildExpTens({PyA, PyB}, {WyA, WyB}, [30 30], [2 2], [1 2], ...
    [false false], [false false], [0 0], 'verbose', false);
s_auto2  = cosSimExpTens(dx2, dy2, 'verbose', false);
s_pwise2 = cosSimExpTens(dx2, dy2, 'method', 'pairwise', 'verbose', false);
results{end+1,1} = 'dispatch.MA cossim: r=2 auto = pairwise (exact)';
results{end,2}   = isequal(s_auto2, s_pwise2);

% ---- r_max > _ORBIT_R_MAX_SHIPPED auto routes to pairwise ----
% v2.2.0 verified this at r=7 by running cosSimExpTens end-to-end and
% comparing auto vs forced pairwise. After Phase 5 extended shipped
% tables to r=2..8, the natural boundary test would be r=9 — but
% pairwise at r=9 builds a K!/(K-r)! ordered-tuple tensor that exceeds
% feasible test memory for any K large enough to expose the dispatch
% decision (K >= 9). The dispatcher decision itself is covered by the
% Python unit test test_ma_dispatcher_routes_pairwise_when_r_too_large
% in test_ma_orbit.py, which calls _select_ma_inner_product_method
% directly and avoids the workload cost.

%% ---- Self-similarity = 1 on orbit branch ----

s_self = cosSimExpTens(dx, dx, 'method', 'orbit', 'verbose', false);
results{end+1,1} = 'dispatch.MA cossim: self-similarity on orbit branch = 1 (1e-12)';
results{end,2}   = abs(s_self - 1) < 1e-12;

%% ---- Skinny dens transparency: orbit branch never forces ensure ----

% Build skinny; check no Centres/U_perm before dispatch; run orbit.
dens_skinnyX = buildExpTens({PxA, PxB}, {WxA, WxB}, [30 30], [3 3], [1 2], ...
    [false false], [false false], [0 0], 'verbose', false);
dens_skinnyY = buildExpTens({PyA, PyB}, {WyA, WyB}, [30 30], [3 3], [1 2], ...
    [false false], [false false], [0 0], 'verbose', false);
results{end+1,1} = 'dispatch.MA cossim: skinny dens has no U_perm before dispatch';
results{end,2}   = ~isfield(dens_skinnyX, 'U_perm');
s_skinny = cosSimExpTens(dens_skinnyX, dens_skinnyY, ...
    'method', 'orbit', 'verbose', false);
results{end+1,1} = 'dispatch.MA cossim: orbit on skinny dens matches eager-built orbit';
results{end,2}   = abs(s_skinny - s_orbit) < 1e-12;

%% ---- Ragged K (NaN-padded events): orbit handles via zero-pad ----

% One attribute with two events of different K (3 and 5 non-NaN slots).
P_ragX = [10 100; 30 200; 50 300; NaN 400; NaN 500];   % (5, 2)
W_ragX = [1 1; 1 1; 1 1; NaN 1; NaN 1];                 % (5, 2)
P_ragY = [20 110; 40 220; 60 330; NaN 440; NaN 550];
W_ragY = [1 1; 1 1; 1 1; NaN 1; NaN 1];
dx_rag = buildExpTens({P_ragX}, {W_ragX}, 30, 3, 1, false, false, 0, ...
    'verbose', false);
dy_rag = buildExpTens({P_ragY}, {W_ragY}, 30, 3, 1, false, false, 0, ...
    'verbose', false);
s_rag_orbit = cosSimExpTens(dx_rag, dy_rag, 'method', 'orbit', ...
    'verbose', false);
s_rag_pwise = cosSimExpTens(dx_rag, dy_rag, 'method', 'pairwise', ...
    'verbose', false);
results{end+1,1} = 'dispatch.MA cossim: ragged K orbit matches pairwise (1e-8)';
results{end,2}   = abs(s_rag_orbit - s_rag_pwise) < 1e-8;

%% ---- Standalone summary ----

if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:,2)));
    nFail = numel(results(:,1)) - nPass;
    fprintf('\n=== test_dispatch_ma_cossim: %d passed, %d failed (of %d) ===\n', ...
        nPass, nFail, numel(results(:,1)));
    if nFail > 0
        for ii = 1:size(results, 1)
            if ~isequal(results{ii, 2}, true)
                fprintf('  FAIL  %s\n', results{ii, 1});
            end
        end
    end
end
