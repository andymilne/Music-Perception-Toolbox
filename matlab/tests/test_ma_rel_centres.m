%% test_ma_rel_centres.m
%  Tests the tuple-centres route for relative attributes in the
%  multi-attribute Möbius path (mobius.maRelAttrPrefersCentres,
%  mobius.closedFormAttrCentres, mobius.closedFormAttrMatrixFrom, and
%  the routing inside cosSimExpTens's MA Möbius orchestrator).
%
%  Tests:
%    - rel-per small K: method='mobius' (centres route) agrees with
%      method='bulger' at the cosine level (the two compute the same
%      minimum-image measure up to a cancelling per-attribute
%      constant; sigma/P here is far below the measure threshold).
%    - rel-nonper: likewise (the non-periodic closed form is exact).
%    - ragged rel (NaN-padded events): likewise.
%    - Unit: the centres matrix and the translation-grid matrix agree
%      entrywise up to a single constant factor (the dropped
%      per-attribute prefactor), verified via the spread of their
%      entrywise ratio.
%    - Predicate: prefers centres at small K, defers to the grid at
%      large K and above the sigma/P measure threshold.
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

rng(7);

% --- Shared workload builder: scalar onset + K-pitch chord ---
makeMa = @(N, K, r, isRelP, isPerP, seed) localMakeMa(N, K, r, ...
    isRelP, isPerP, seed);

% --- rel-per small K: mobius (centres) vs bulger ---
dx = makeMa(30, 4, 2, true, true, 11);
dy = makeMa(30, 4, 2, true, true, 22);
sBul = cosSimExpTens(dx, dy, 'method', 'bulger', 'verbose', false);
sMob = cosSimExpTens(dx, dy, 'method', 'mobius', 'verbose', false);
results{end+1,1} = 'MA rel-per centres route: mobius matches bulger (1e-8)';
results{end,2}   = abs(sMob - sBul) < 1e-8;

% --- rel-nonper: mobius (centres) vs bulger ---
dx = makeMa(30, 4, 2, true, false, 13);
dy = makeMa(30, 4, 2, true, false, 24);
sBul = cosSimExpTens(dx, dy, 'method', 'bulger', 'verbose', false);
sMob = cosSimExpTens(dx, dy, 'method', 'mobius', 'verbose', false);
results{end+1,1} = 'MA rel-nonper centres route: mobius matches bulger (1e-8)';
results{end,2}   = abs(sMob - sBul) < 1e-8;

% --- ragged rel-per ---
dxR = localMakeMaRagged(24, 4, 2, true, true, 15);
dyR = localMakeMaRagged(24, 4, 2, true, true, 26);
sBul = cosSimExpTens(dxR, dyR, 'method', 'bulger', 'verbose', false);
sMob = cosSimExpTens(dxR, dyR, 'method', 'mobius', 'verbose', false);
results{end+1,1} = 'MA rel-per centres route ragged: mobius matches bulger (1e-8)';
results{end,2}   = abs(sMob - sBul) < 1e-8;

% --- Unit: centres matrix == grid matrix up to one constant ---
% The centres and grid paths agree up to a constant factor only when
% both are computed exactly; at the default (Inf) truncation, now the
% 1e-12 accuracy floor, the grid path's kernel truncates marginally
% and perturbs the ratio's constancy past 1e-6. Widen the floor to
% 1e-300 for this unit check, then restore.
mrc_prevEps = internal.accuracyFloor('setEps', 1e-300);
dx = makeMa(8, 4, 2, true, true, 17);
dy = makeMa(8, 4, 2, true, true, 28);
a = 2;   % the pitch attribute (attribute 1 is the scalar onset)
cxB = mobius.closedFormAttrCentres(dx, a);
cyB = mobius.closedFormAttrCentres(dy, a);
I_centres = mobius.closedFormAttrMatrixFrom(cxB, cyB);
I_grid = mobius.maPerAttrInnerMatrix( ...
    dx.pAttr{a}, dx.w{a}, dy.pAttr{a}, dy.w{a}, ...
    dx.sigma(a), dx.r(a), true, true, dx.period(a));
% The entrywise ratio is only meaningful where I_grid carries signal.
% The matrix spans some twenty orders of magnitude -- the Moebius
% alternating sum cancels to the noise floor in a few cells -- and the
% ratio at a cancellation-noise cell is arbitrary, so it must not be
% allowed to set the spread. Restrict the constancy check to entries
% above a relative magnitude floor, and check the discarded entries are
% negligible against the matrix scale rather than ignoring them.
mrc_scale = max(abs(I_grid(:)));
mrc_live  = abs(I_grid) > 1e-8 * mrc_scale;
ratio = I_centres(mrc_live) ./ I_grid(mrc_live);
results{end+1,1} = 'MA rel centres matrix: constant ratio to grid matrix (1e-6)';
results{end,2}   = (max(ratio(:)) / min(ratio(:)) - 1) < 1e-6;

% The proportionality itself, measured against the matrix scale so no
% single near-zero cell dominates: fit the constant by least squares and
% bound the residual.
mrc_c = (I_centres(:).' * I_grid(:)) / (I_grid(:).' * I_grid(:));
mrc_resid = max(abs(I_centres(:) - mrc_c * I_grid(:))) ...
            / (abs(mrc_c) * mrc_scale);
results{end+1,1} = 'MA rel centres matrix: scale-relative residual (1e-10)';
results{end,2}   = mrc_resid < 1e-10;

% Restore the accuracy floor (paired with the setEps above).
internal.accuracyFloor('setEps', mrc_prevEps);

% --- Predicate behaviour ---
PxSmall = rand(4, 10) * 1200;
% The centres/grid crossover at r = 2 with these parameters (N_u = 1666
% grid nodes) sits at K = 42: (2*C(K,2))^2 vs 1666*K^2. K = 60 sits
% comfortably on the grid side, robust to modest node-count changes.
PxLarge = rand(60, 10) * 1200;
results{end+1,1} = 'maRelAttrPrefersCentres: small K -> centres';
results{end,2}   = mobius.maRelAttrPrefersCentres( ...
    PxSmall, PxSmall, 6, 2, true, true, 1200);
results{end+1,1} = 'maRelAttrPrefersCentres: large K -> grid';
results{end,2}   = ~mobius.maRelAttrPrefersCentres( ...
    PxLarge, PxLarge, 6, 2, true, true, 1200);
results{end+1,1} = 'maRelAttrPrefersCentres: above sigma/P threshold -> grid';
results{end,2}   = ~mobius.maRelAttrPrefersCentres( ...
    PxSmall, PxSmall, 100, 2, true, true, 1200);

if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:, 2)));
    nFail = size(results, 1) - nPass;
    fprintf('test_ma_rel_centres: %d passed, %d failed\n', nPass, nFail);
    if nFail > 0
        for ii = 1:size(results, 1)
            if ~isequal(results{ii, 2}, true)
                fprintf('  FAIL  %s\n', results{ii, 1});
            end
        end
    end
end


function dens = localMakeMa(N, K, r, isRelP, isPerP, seed)
    rng(seed);
    pitches = rand(K, N) * 1200;
    onsets  = (0:N-1) * 250 + randn(1, N) * 10;
    dens = buildExpTens({onsets; pitches}, {[]; []}, [15, 6], [1, r], ...
        [false, isRelP], [false, isPerP], [4000, 1200], ...
        'verbose', false);
end


function dens = localMakeMaRagged(N, K, r, isRelP, isPerP, seed)
    rng(seed);
    pitches = rand(K, N) * 1200;
    pitches(K, 1:3:N) = NaN;   % every third event loses its last slot
    onsets  = (0:N-1) * 250 + randn(1, N) * 10;
    dens = buildExpTens({onsets; pitches}, {[]; []}, [15, 6], [1, r], ...
        [false, isRelP], [false, isPerP], [4000, 1200], ...
        'verbose', false);
end
