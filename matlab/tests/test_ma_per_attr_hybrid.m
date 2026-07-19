%% test_ma_per_attr_hybrid.m — v2.2-dev safe/unsafe hybrid in MA per-attr IP
%
%  Tests for the hybrid safe/unsafe partition in
%  mobius.maPerAttrInnerMatrix (added when we addressed the Python/MATLAB
%  ragged-K parity gap). Strategy:
%    - Each event gets classified as "safe" if K_eff - r >= 2, "unsafe"
%      otherwise (matches _ORBIT_K_MINUS_R_MIN).
%    - Safe-vs-safe pairs flow through the vectorised batched Möbius method
%      with zero-pad within the safe group.
%    - Pairs involving any unsafe event flow through
%      mobius.innerProductDirectAbsSingleMultiset (direct r-tuple enumeration; no
%      Möbius alternating sum, so no cancellation).
%
%  Tests:
%    - innerProductDirectAbsSingleMultiset standalone correctness (matches a
%      hand-rolled centres-array IP).
%    - All-safe ragged: hybrid produces same matrix as the prior
%      zero-pad-everything approach (since safe group covers all events).
%    - All-unsafe (every event has K_eff = r): hybrid equals an
%      explicit per-pair direct-enum reference.
%    - Mixed safe/unsafe: cosSimExpTens with method='mobius' agrees with
%      method='bulger' to high precision (the hybrid's correctness
%      claim).
%    - r=1 ragged: zero-pad path unchanged, still matches Bulger.
%    - Edge: K_eff = r exactly (single ordered tuple per event,
%      direct-enum trivially exact).
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    % Defaults isolation when run standalone (when invoked from
    % test_mpt.m the outer wrapper has already isolated defaults).
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

%% ---- innerProductDirectAbsSingleMultiset correctness ----

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

ip_direct = mobius.innerProductDirectAbsSingleMultiset(p_x, w_x, p_y, w_y, ...
    sigma, 3, false, 0);

% Hand-rolled reference via buildExpTens centres
densX = buildExpTens(p_x, w_x, sigma, 3, false, false, 0, ...
    'lazy', false, 'verbose', false);
densY = buildExpTens(p_y, w_y, sigma, 3, false, false, 0, ...
    'lazy', false, 'verbose', false);
diffs_ref = reshape(densX.U_perm{1}, 3, densX.nJ, 1) ...
          - reshape(densY.U_perm{1}, 3, 1, densY.nJ);
Q_ref = reshape(sum(diffs_ref.^2, 1), densX.nJ, densY.nJ);
ip_ref = (sigma * sqrt(pi))^3 * (densX.wJ * exp(-Q_ref / (4*sigma^2)) * densY.wJ.');

results{end+1,1} = 'mobius.innerProductDirectAbsSingleMultiset: matches centres-array IP (1e-12)';
results{end,2}   = abs(ip_direct - ip_ref) < 1e-12 * abs(ip_ref);

% NaN-tolerance: NaN entries are dropped per side.
p_x_nan = [p_x; NaN; NaN];   w_x_nan = [w_x; NaN; NaN];
ip_direct_nan = mobius.innerProductDirectAbsSingleMultiset(p_x_nan, w_x_nan, p_y, w_y, ...
    sigma, 3, false, 0);
results{end+1,1} = 'mobius.innerProductDirectAbsSingleMultiset: NaN-padded input dropped per side';
results{end,2}   = abs(ip_direct_nan - ip_direct) < 1e-12 * abs(ip_direct);

% K_eff < r returns 0 by convention.
ip_zero = mobius.innerProductDirectAbsSingleMultiset([0; 1], [1; 1], [0; 1], [1; 1], ...
    sigma, 3, false, 0);
results{end+1,1} = 'mobius.innerProductDirectAbsSingleMultiset: K_eff < r returns 0';
results{end,2}   = ip_zero == 0;

%% ---- All-safe ragged: hybrid equals safe-only Möbius ----

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
        I_ref_safe(nx, ny) = mobius.innerProductDirectAbsSingleMultiset( ...
            P_safe(:, nx), W_safe(:, nx), P_safe(:, ny), W_safe(:, ny), ...
            sigma, r, false, 0);
    end
end
results{end+1,1} = 'maPerAttrInnerMatrix all-safe ragged: hybrid matches direct-enum reference (1e-10)';
results{end,2}   = max(abs(I_hybrid_safe(:) - I_ref_safe(:))) < 1e-10 * max(abs(I_ref_safe(:)));

%% ---- All-unsafe: hybrid uses direct enum on every pair ----

% Compared at 1e-13 rtol (near bit-parity). Under the default (Inf)
% truncation, which now resolves to the 1e-12 accuracy floor, the
% hybrid and the direct-enum reference can drop marginally different
% far-tail contributions and diverge at ~1e-12, above this tolerance.
% Widen the floor to 1e-300 (effectively exhaustive) so the two are
% compared exactly, then restore.
hyb_prevEps = internal.accuracyFloor('setEps', 1e-300);

% Two events, both with K_eff = 3 (= r, so K_eff - r = 0 < 2 -> unsafe).
P_unsafe = [0 100; 4 200; 7 300];   % (3, 2), all events K_eff = 3
W_unsafe = ones(3, 2);
I_hybrid_unsafe = mobius.maPerAttrInnerMatrix( ...
    P_unsafe, W_unsafe, P_unsafe, W_unsafe, sigma, r, false, false, 0);

% Reference: every pair via direct enumeration.
I_ref_unsafe = zeros(2, 2);
for nx = 1:2
    for ny = 1:2
        I_ref_unsafe(nx, ny) = mobius.innerProductDirectAbsSingleMultiset( ...
            P_unsafe(:, nx), W_unsafe(:, nx), P_unsafe(:, ny), W_unsafe(:, ny), ...
            sigma, r, false, 0);
    end
end
% v2.2.0 used a per-pair MATLAB double-loop; v2.2.x replaces it with
% a single vectorised tensor contraction per (K_eff_x, K_eff_y)
% sub-block (K-grouped batched direct enum). The two produce
% mathematically identical results but accumulate Q sums in a
% different order, so individual entries can differ by ~1 ULP.
results{end+1,1} = 'maPerAttrInnerMatrix all-unsafe: hybrid matches direct-enum (1e-13 rtol)';
results{end,2}   = all(abs(I_hybrid_unsafe(:) - I_ref_unsafe(:)) <= ...
                        1e-13 * abs(I_ref_unsafe(:)) + 1e-12);

% Restore the accuracy floor (paired with the setEps above).
internal.accuracyFloor('setEps', hyb_prevEps);

% --- v2.2.x: K-grouped batched direct-enum primitive correctness ---
% The localBatchedDirectEnumAbsSingleMultiset local function (not exported) is
% exercised via the all-unsafe and mixed-K paths above. Here we test
% the variable-K_eff Möbius-vs-Bulger equivalence explicitly: a
% density with events at multiple K_eff values must produce the same
% cosine under the Möbius method as under Bulger.
rng(7, 'twister');
N_kg = 12;
K_max_kg = 6;
r_kg = 3;
sigma_kg = 30.0;
P_kg = nan(K_max_kg, N_kg);
W_kg = nan(K_max_kg, N_kg);
K_dist = [3 3 3 3 4 4 4 4 6 6 6 6];  % 4 events each at K_eff 3, 4, 6
for n = 1:N_kg
    K_eff_n = K_dist(n);
    P_kg(1:K_eff_n, n) = 1200 * rand(K_eff_n, 1);
    W_kg(1:K_eff_n, n) = 1;
end
dens_kg = buildExpTens({P_kg}, {W_kg}, sigma_kg, r_kg, ...
    false, false, 0, 'verbose', false);
s_orbit_kg = cosSimExpTens(dens_kg, dens_kg, 'method', 'mobius', ...
    'verbose', false);
s_pw_kg = cosSimExpTens(dens_kg, dens_kg, 'method', 'bulger', ...
    'verbose', false);
results{end+1,1} = 'maPerAttrInnerMatrix v2.2.x: K-grouped Möbius matches Bulger (1e-12)';
results{end,2}   = abs(s_orbit_kg - s_pw_kg) < 1e-12;

clear P_unsafe W_unsafe I_hybrid_unsafe I_ref_unsafe ...
      N_kg K_max_kg r_kg sigma_kg P_kg W_kg K_dist K_eff_n n dens_kg ...
      s_orbit_kg s_pw_kg

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

dx = buildExpTens({P_mix}, {W_mix}, 30, 3, ...
    false, false, 0, 'verbose', false);
dy = buildExpTens({P_mix}, {W_mix}, 30, 3, ...
    false, false, 0, 'verbose', false);
s_orbit = cosSimExpTens(dx, dy, 'method', 'mobius', 'verbose', false);
s_pwise = cosSimExpTens(dx, dy, 'method', 'bulger', 'verbose', false);

results{end+1,1} = 'cosSimExpTens mixed safe/unsafe MA Möbius matches Bulger (1e-8)';
results{end,2}   = abs(s_orbit - s_pwise) < 1e-8;

%% ---- r=1 ragged still matches pairwise ----

% At r=1 there is no orbit alternating sum, so zero-pad-everything is
% used and unchanged in the hybrid rewrite.
P_r1 = [10 100 200; 30 NaN 400; 50 NaN NaN];
W_r1 = [1 1 1; 1 NaN 1; 1 NaN NaN];
dx_r1 = buildExpTens({P_r1}, {W_r1}, 30, 1, ...
    false, false, 0, 'verbose', false);
dy_r1 = buildExpTens({P_r1}, {W_r1}, 30, 1, ...
    false, false, 0, 'verbose', false);
s_r1_orbit = cosSimExpTens(dx_r1, dy_r1, 'method', 'mobius', 'verbose', false);
s_r1_pw    = cosSimExpTens(dx_r1, dy_r1, 'method', 'bulger', 'verbose', false);
results{end+1,1} = 'cosSimExpTens r=1 ragged: Möbius matches Bulger (1e-10)';
results{end,2}   = abs(s_r1_orbit - s_r1_pw) < 1e-10;

%% ---- Standalone summary ----

if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:,2)));
    nFail = numel(results(:,1)) - nPass;
    fprintf('\n=== test_ma_per_attr_hybrid: %d passed, %d failed (of %d) ===\n', ...
        nPass, nFail, numel(results(:,1)));
    % Restore caller's pre-test defaults eagerly
    % (fires the helper's onCleanup destructor before
    % the conditional error below halts execution).
    clear cleanupDefaults
    if nFail > 0
        for ii = 1:size(results, 1)
            if ~isequal(results{ii, 2}, true)
                fprintf('  FAIL  %s\n', results{ii, 1});
            end
        end
    end
end
