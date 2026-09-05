%% test_ma_per_attr_hybrid.m — ragged-event handling in MA per-attr IP
%
%  Tests for ragged (NaN-padded) events in mobius.maPerAttrInnerMatrix.
%  Every event takes the vectorised batched Mobius route with
%  zero-weight padding; accuracy is governed by truncationSigmas, so no
%  size-based partition is applied. The direct r-tuple enumeration
%  reference.innerProductDirectAbsSingleMultiset (tests/reference; no
%  Mobius alternating sum) is the comparison point.
%
%  Tests:
%    - innerProductDirectAbsSingleMultiset standalone correctness (matches a
%      hand-rolled centres-array IP).
%    - Ragged events well above r: the batched matrix equals the per-pair
%      direct-enum reference.
%    - Every event at K_eff = r: the batched matrix equals the per-pair
%      direct-enum reference.
%    - Mixed K_eff: cosSimExpTens with method='mobius' agrees with
%      method='bulger' to high precision.
%    - r=1 ragged: zero-pad path unchanged, still matches Bulger.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    % Defaults isolation when run standalone (when invoked from
    % test_mpt.m the outer wrapper has already isolated defaults).
    addpath(fileparts(mfilename('fullpath')));
    addpath(fullfile(fileparts(mfilename('fullpath')), 'reference'));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

%% ---- innerProductDirectAbsSingleMultiset correctness ----

% Compare direct enumeration to a hand-rolled centres-array IP for a
% small abs-mode single-multiset case. The hand-rolled version uses buildExpTens to
% generate ordered tuples + weight products and computes the IP via
% explicit kernel matmul.
rng(81, 'twister');
p_x = sort(2000 * rand(5, 1));
w_x = ones(5, 1);
p_y = sort(2000 * rand(5, 1));
w_y = ones(5, 1);
sigma = 30;

ip_direct = reference.innerProductDirectAbsSingleMultiset(p_x, w_x, p_y, w_y, ...
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

results{end+1,1} = 'reference.innerProductDirectAbsSingleMultiset: matches centres-array IP (1e-12)';
results{end,2}   = abs(ip_direct - ip_ref) < 1e-12 * abs(ip_ref);

% NaN-tolerance: NaN entries are dropped per side.
p_x_nan = [p_x; NaN; NaN];   w_x_nan = [w_x; NaN; NaN];
ip_direct_nan = reference.innerProductDirectAbsSingleMultiset(p_x_nan, w_x_nan, p_y, w_y, ...
    sigma, 3, false, 0);
results{end+1,1} = 'reference.innerProductDirectAbsSingleMultiset: NaN-padded input dropped per side';
results{end,2}   = abs(ip_direct_nan - ip_direct) < 1e-12 * abs(ip_direct);

% K_eff < r returns 0 by convention.
ip_zero = reference.innerProductDirectAbsSingleMultiset([0; 1], [1; 1], [0; 1], [1; 1], ...
    sigma, 3, false, 0);
results{end+1,1} = 'reference.innerProductDirectAbsSingleMultiset: K_eff < r returns 0';
results{end,2}   = ip_zero == 0;

%% ---- Ragged events well above r: batched matrix equals direct enum ----

% Four events, all with K_eff = 6 (well above r = 3).
rng(83, 'twister');
P_big = sort(2000 * rand(6, 4));   W_big = ones(6, 4);   % all K_eff = 6
sigma = 30; r = 3;
I_big = mobius.maPerAttrInnerMatrix(P_big, W_big, P_big, W_big, ...
    sigma, r, false, false, 0);

% Reference: a single buildExpTens-style direct computation per pair
% (gives the gold-standard IP; no Möbius cancellation since we're
% summing positive terms).
N_big = 4;
I_ref_big = zeros(N_big, N_big);
for nx = 1:N_big
    for ny = 1:N_big
        I_ref_big(nx, ny) = reference.innerProductDirectAbsSingleMultiset( ...
            P_big(:, nx), W_big(:, nx), P_big(:, ny), W_big(:, ny), ...
            sigma, r, false, 0);
    end
end
results{end+1,1} = 'maPerAttrInnerMatrix K_eff well above r: batched Mobius matches direct-enum reference (1e-10)';
results{end,2}   = max(abs(I_big(:) - I_ref_big(:))) < 1e-10 * max(abs(I_ref_big(:)));

%% ---- Every event at K_eff = r: batched matrix equals direct enum ----

% Compared at 1e-13 rtol (near bit-parity). Under the default (Inf)
% truncation, which now resolves to the 1e-12 accuracy floor, the
% batched matrix and the direct-enum reference can drop marginally
% different far-tail contributions and diverge at ~1e-12, above this
% tolerance. Widen the floor to 1e-300 (effectively exhaustive) so the
% two are compared exactly, then restore.
hyb_prevEps = internal.accuracyFloor('setEps', 1e-300);

% Two events, both with K_eff = 3 (= r).
P_kr = [0 100; 4 200; 7 300];   % (3, 2), all events K_eff = 3
W_kr = ones(3, 2);
I_kr = mobius.maPerAttrInnerMatrix( ...
    P_kr, W_kr, P_kr, W_kr, sigma, r, false, false, 0);

% Reference: every pair via direct enumeration.
I_ref_kr = zeros(2, 2);
for nx = 1:2
    for ny = 1:2
        I_ref_kr(nx, ny) = reference.innerProductDirectAbsSingleMultiset( ...
            P_kr(:, nx), W_kr(:, nx), P_kr(:, ny), W_kr(:, ny), ...
            sigma, r, false, 0);
    end
end
% The batched Mobius route and the enumeration accumulate their sums
% in a different order, so individual entries can differ by ~1 ULP.
results{end+1,1} = 'maPerAttrInnerMatrix K_eff = r: batched Mobius matches direct-enum on the value scale';
% Judge by absolute error on the scale the inner product lives on:
% entries span many orders of magnitude, so a relative tolerance would
% be dominated by near-zero cross terms.
results{end,2}   = all(abs(I_kr(:) - I_ref_kr(:)) <= ...
                        1e-13 * max(abs(I_ref_kr(:))));

% Restore the accuracy floor (paired with the setEps above).
internal.accuracyFloor('setEps', hyb_prevEps);

% --- Variable-K_eff Mobius-vs-Bulger equivalence through the public
% API: a density with events at several K_eff values must produce the
% same cosine under the Mobius method as under Bulger's.
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
results{end+1,1} = 'maPerAttrInnerMatrix ragged K_eff: Möbius matches Bulger (1e-12)';
results{end,2}   = abs(s_orbit_kg - s_pw_kg) < 1e-12;

clear P_kr W_kr I_kr I_ref_kr ...
      N_kg K_max_kg r_kg sigma_kg P_kg W_kg K_dist K_eff_n n dens_kg ...
      s_orbit_kg s_pw_kg

%% ---- Mixed K_eff: cosSim orbit equals pairwise ----

% For any ragged-K input the orbit method must produce an answer equal
% to pairwise (the gold standard) within tolerance; a deliberately
% mixed case.

% Three events: K_eff = 3, 6, 4. r = 3.
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

results{end+1,1} = 'cosSimExpTens mixed K_eff MA Möbius matches Bulger (1e-8)';
results{end,2}   = abs(s_orbit - s_pwise) < 1e-8;

%% ---- r=1 ragged still matches pairwise ----

% At r=1 there is no orbit alternating sum; the zero-pad kernel sum is
% used.
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
