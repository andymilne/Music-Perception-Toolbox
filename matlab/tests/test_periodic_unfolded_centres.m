%% test_periodic_unfolded_centres.m — periodic image sums on unfolded coords
%
%  A family of periodic routines build their result by summing Gaussian
%  images across the period: the differential/Shannon grid cell mass
%  (localPhiDiffAxisPeriodic in entropyExpTens), the wrapped-window factor
%  (localWrappedWindowFactor1D in evalExpTens), and the windowed
%  inner-product image sum (localPeriodicImageSumContribution in
%  internal.windowedInnerProduct). Each must reduce its input coordinate
%  modulo the period first. Otherwise a coordinate many periods from the
%  canonical [0, period) window --- for example an absolute spectral
%  partial thousands of cents above the grid --- never contributes, and
%  the routine returns a degenerate, input-independent value. These tests
%  lock in the modulo reduction.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end


% --- Site 1: periodic differential entropy on unfolded (absolute) centres.
% Centres several periods above the [0, period) grid must give the same
% result as the folded centres, and must vary with content (the bug
% returned the same constant for every input).

pA = [4000, 5000, 6400];   % absolute cents, all > period
pB = [4300, 5100, 6200];
wts = [1, 1, 1];
hUnfolded = entropyExpTens(pA, wts, 15, 1, false, true, 1200, ...
    'method', 'differential', 'verbose', false);
hFolded   = entropyExpTens(mod(pA, 1200), wts, 15, 1, false, true, 1200, ...
    'method', 'differential', 'verbose', false);
results{end+1,1} = 'periodic differential: unfolded centres match folded';
results{end,2}   = abs(hUnfolded - hFolded) < 1e-9 && hUnfolded > 1;

hB = entropyExpTens(pB, wts, 15, 1, false, true, 1200, ...
    'method', 'differential', 'verbose', false);
results{end+1,1} = 'periodic differential: varies with content (not degenerate)';
results{end,2}   = abs(hUnfolded - hB) > 1e-3;

% At non-negligible sigma/period the periodic differential entropy must
% track the minimum-image density (Eq. 1), not the wrapped normal.
cMi = [0, 400, 700, 1100];  wMi = ones(1, 4);  sMi = 300;  % sigma/period = 0.25
G = 200000;  xs = (0:G-1) * (1200 / G);
dMin = xs - cMi.';  dMin = dMin - 1200 * round(dMin / 1200);
fMin = sum(exp(-dMin.^2 / (2 * sMi^2)), 1);
fWrap = zeros(1, G);
for n = -6:6
    fWrap = fWrap + sum(exp(-(xs - cMi.' - n * 1200).^2 / (2 * sMi^2)), 1);
end
dx = 1200 / G;
hMin  = -sum((fMin  / sum(fMin)  ) .* log(fMin  / (sum(fMin)  * dx) + 1e-300)) ;
hWrap = -sum((fWrap / sum(fWrap) ) .* log(fWrap / (sum(fWrap) * dx) + 1e-300)) ;
hTbMi = entropyExpTens(cMi, wMi, sMi, 1, false, true, 1200, ...
    'method', 'differential', 'base', exp(1), 'verbose', false);
results{end+1,1} = 'periodic differential: tracks minimum-image (not wrapped normal)';
results{end,2}   = abs(hMin - hWrap) > 1e-4 ...
                   && abs(hTbMi - hMin) < abs(hTbMi - hWrap) ...
                   && abs(hTbMi - hMin) < 1e-3;

% --- Site 2: wrapped-window factor in evalExpTens. A windowed periodic
% density must return identical values at periodic-equivalent query points
% even when the shift is many periods (the near images underflow to 0, so
% an unfolded sum would terminate before reaching the dominant image).

P_test = 12;
dens_per = buildExpTens({[3, 7]}, [], 1, 1, false, true, P_test, 'verbose', false);
spec_eval = struct('size', 3, 'mix', 0, 'centre', {{2}});
wmd_eval = windowTensor(dens_per, spec_eval);
v0     = evalExpTens(wmd_eval, 0.5);
v_far  = evalExpTens(wmd_eval, 0.5 + 50 * P_test);   % 50 periods away
results{end+1,1} = 'evalExpTens: periodic window equivalence at large shift';
results{end,2}   = abs(v0 - v_far) <= 1e-10 * max(abs([v0, v_far])) && v0 > 0;

% --- Site 3: windowed inner-product image sum. A periodic windowed
% attribute whose window centre is shifted by whole periods must give the
% same windowed inner product (the per-pair midpoint-minus-centre offset
% is many periods otherwise).

pitch_w = [60, 62, 64, 65];
time_w  = [0, 1, 2, 3];
dens_w = buildExpTens({pitch_w, time_w}, [], [10, 0.1], [1, 1], ...
    [false, false], [true, false], [1200, 0], 'lazy', false, 'verbose', false);
spec_near = struct('size', [3, Inf], 'mix', [0, 0], 'centre', {{62, 0}});
spec_far  = struct('size', [3, Inf], 'mix', [0, 0], 'centre', {{62 + 24000, 0}}); % +20 P
s_near = internal.windowedInnerProduct(dens_w, windowTensor(dens_w, spec_near), false);
s_far  = internal.windowedInnerProduct(dens_w, windowTensor(dens_w, spec_far), false);
results{end+1,1} = 'windowedInnerProduct: periodic window centre-shift invariance';
results{end,2}   = abs(s_near - s_far) < 1e-9 * max(1, abs(s_near));


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
    fprintf('\n=== test_periodic_unfolded_centres: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_periodic_unfolded_centres:failed', '%d test(s) failed.', nFail);
    end
end
