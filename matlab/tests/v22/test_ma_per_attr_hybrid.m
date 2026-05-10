%% test_ma_per_attr_hybrid.m — v2.2-dev safe/unsafe hybrid in MA per-attr IP
%
%  Tests for the hybrid safe/unsafe partition in
%  mobius.maPerAttrInnerMatrix (added when we addressed the Python/MATLAB
%  ragged-K parity gap). Strategy:
%    - Each event gets classified as "safe" if K_eff - r >= 2, "unsafe"
%      otherwise (matches _ORBIT_K_MINUS_R_MIN).
%    - Safe-vs-safe pairs flow through the vectorised batched orbit
%      with zero-pad within the safe group.
%    - Pairs involving any unsafe event flow through
%      mobius.innerProductDirectAbsSA (direct r-tuple enumeration; no
%      Möbius alternating sum, so no cancellation).
%
%  Tests:
%    - innerProductDirectAbsSA standalone correctness (matches a
%      hand-rolled centres-array IP).
%    - All-safe ragged: hybrid produces same matrix as the prior
%      zero-pad-everything approach (since safe group covers all events).
%    - All-unsafe (every event has K_eff = r): hybrid equals an
%      explicit per-pair direct-enum reference.
%    - Mixed safe/unsafe: cosSimExpTens with method='orbit' agrees with
%      method='pairwise' to high precision (the hybrid's correctness
%      claim).
%    - r=1 ragged: zero-pad path unchanged, still matches pairwise.
%    - Edge: K_eff = r exactly (single ordered tuple per event,
%      direct-enum trivially exact).
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
else
    standalone = false;
end

%% ---- innerProductDirectAbsSA correctness ----

% Compare direct enumeration to a hand-rolled centres-array IP for a
% small abs-mode SA case. The hand-rolled version uses buildExpTens to
% generate ordered tuples + weight products and computes the IP via
% explicit kernel matmul.
rng(81, 'twister');
p_x = sort(2000 * rand(5, 1));
w_x = ones(5, 1);
p_y = sort(2000 * rand(5, 1));
w_y = ones(5, 1);
sigma = 30;

ip_direct = mobius.innerProductDirectAbsSA(p_x, w_x, p_y, w_y, ...
    sigma, 3, false, 0);

% Hand-rolled reference via buildExpTens centres
densX = buildExpTens(p_x, w_x, sigma, 3, false, false, 0, ...
    'lazy', false, 'verbose', false);
densY = buildExpTens(p_y, w_y, sigma, 3, false, false, 0, ...
    'lazy', false, 'verbose', false);
diffs_ref = reshape(densX.U_perm, 3, densX.nJ, 1) ...
          - reshape(densY.U_perm, 3, 1, densY.nJ);
Q_ref = reshape(sum(diffs_ref.^2, 1), densX.nJ, densY.nJ);
ip_ref = (sigma * sqrt(pi))^3 * (densX.wJ * exp(-Q_ref / (4*sigma^2)) * densY.wJ.');

results{end+1,1} = 'mobius.innerProductDirectAbsSA: matches centres-array IP (1e-12)';
results{end,2}   = abs(ip_direct - ip_ref) < 1e-12 * abs(ip_ref);

% NaN-tolerance: NaN entries are dropped per side.
p_x_nan = [p_x; NaN; NaN];   w_x_nan = [w_x; NaN; NaN];
ip_direct_nan = mobius.innerProductDirectAbsSA(p_x_nan, w_x_nan, p_y, w_y, ...
    sigma, 3, false, 0);
results{end+1,1} = 'mobius.innerProductDirectAbsSA: NaN-padded input dropped per side';
results{end,2}   = abs(ip_direct_nan - ip_direct) < 1e-12 * abs(ip_direct);

% K_eff < r returns 0 by convention.
ip_zero = mobius.innerProductDirectAbsSA([0; 1], [1; 1], [0; 1], [1; 1], ...
    sigma, 3, false, 0);
results{end+1,1} = 'mobius.innerProductDirectAbsSA: K_eff < r returns 0';
results{end,2}   = ip_zero == 0;

%% ---- All-safe ragged: hybrid equals safe-only orbit ----

% Two events, both with K_eff = 6 (well above r+2 = 5); the hybrid
% should not invoke the unsafe branch at all.
rng(83, 'twister');
P_safe = sort(2000 * rand(6, 4));   W_safe = ones(6, 4);   % all K_eff = 6
sigma = 30; r = 3;
I_hybrid_safe = mobius.maPerAttrInnerMatrix(P_safe, W_safe, P_safe, W_safe, ...
    sigma, r, false, false, 0);

% Reference: a single buildExpTens-style direct computation per pair
% (gives the gold-standard IP; no Möbius cancellation since we're
% summing positive terms).
N_safe = 4;
I_ref_safe = zeros(N_safe, N_safe);
for nx = 1:N_safe
    for ny = 1:N_safe
        I_ref_safe(nx, ny) = mobius.innerProductDirectAbsSA( ...
            P_safe(:, nx), W_safe(:, nx), P_safe(:, ny), W_safe(:, ny), ...
            sigma, r, false, 0);
    end
end
results{end+1,1} = 'maPerAttrInnerMatrix all-safe ragged: hybrid matches direct-enum reference (1e-10)';
results{end,2}   = max(abs(I_hybrid_safe(:) - I_ref_safe(:))) < 1e-10 * max(abs(I_ref_safe(:)));

%% ---- All-unsafe: hybrid uses direct enum on every pair ----

% Two events, both with K_eff = 3 (= r, so K_eff - r = 0 < 2 -> unsafe).
P_unsafe = [0 100; 4 200; 7 300];   % (3, 2), all events K_eff = 3
W_unsafe = ones(3, 2);
I_hybrid_unsafe = mobius.maPerAttrInnerMatrix( ...
    P_unsafe, W_unsafe, P_unsafe, W_unsafe, sigma, r, false, false, 0);

% Reference: every pair via direct enumeration.
I_ref_unsafe = zeros(2, 2);
for nx = 1:2
    for ny = 1:2
        I_ref_unsafe(nx, ny) = mobius.innerProductDirectAbsSA( ...
            P_unsafe(:, nx), W_unsafe(:, nx), P_unsafe(:, ny), W_unsafe(:, ny), ...
            sigma, r, false, 0);
    end
end
results{end+1,1} = 'maPerAttrInnerMatrix all-unsafe: hybrid matches direct-enum (exact)';
results{end,2}   = isequal(I_hybrid_unsafe, I_ref_unsafe);

%% ---- Mixed safe/unsafe: cosSim orbit equals pairwise ----

% This is the killer test: the hybrid claim is that for any ragged-K
% input, the orbit method produces an answer equal to pairwise (the
% gold standard) within tolerance. We exercise that on a deliberately
% mixed case.

% Three events: K_eff = 3 (unsafe), 6 (safe), 4 (unsafe). r = 3.
P_mix = [10  100  500;
         30  200  600;
         50  300  700;
         NaN 400  800;
         NaN 500  NaN;
         NaN 600  NaN];   % (6, 3)
W_mix = [1  1  1;
         1  1  1;
         1  1  1;
         NaN  1  1;
         NaN  1  NaN;
         NaN  1  NaN];

dx = buildExpTens({P_mix}, {W_mix}, 30, 3, 1, ...
    false, false, 0, 'verbose', false);
dy = buildExpTens({P_mix}, {W_mix}, 30, 3, 1, ...
    false, false, 0, 'verbose', false);
s_orbit = cosSimExpTens(dx, dy, 'method', 'orbit', 'verbose', false);
s_pwise = cosSimExpTens(dx, dy, 'method', 'pairwise', 'verbose', false);

results{end+1,1} = 'cosSimExpTens mixed safe/unsafe MA orbit matches pairwise (1e-8)';
results{end,2}   = abs(s_orbit - s_pwise) < 1e-8;

%% ---- r=1 ragged still matches pairwise ----

% At r=1 there is no orbit alternating sum, so zero-pad-everything is
% used and unchanged in the hybrid rewrite.
P_r1 = [10 100 200; 30 NaN 400; 50 NaN NaN];
W_r1 = [1 1 1; 1 NaN 1; 1 NaN NaN];
dx_r1 = buildExpTens({P_r1}, {W_r1}, 30, 1, 1, ...
    false, false, 0, 'verbose', false);
dy_r1 = buildExpTens({P_r1}, {W_r1}, 30, 1, 1, ...
    false, false, 0, 'verbose', false);
s_r1_orbit = cosSimExpTens(dx_r1, dy_r1, 'method', 'orbit', 'verbose', false);
s_r1_pw    = cosSimExpTens(dx_r1, dy_r1, 'method', 'pairwise', 'verbose', false);
results{end+1,1} = 'cosSimExpTens r=1 ragged: orbit matches pairwise (1e-10)';
results{end,2}   = abs(s_r1_orbit - s_r1_pw) < 1e-10;

%% ---- Standalone summary ----

if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:,2)));
    nFail = numel(results(:,1)) - nPass;
    fprintf('\n=== test_ma_per_attr_hybrid: %d passed, %d failed (of %d) ===\n', ...
        nPass, nFail, numel(results(:,1)));
    if nFail > 0
        for ii = 1:size(results, 1)
            if ~isequal(results{ii, 2}, true)
                fprintf('  FAIL  %s\n', results{ii, 1});
            end
        end
    end
end
