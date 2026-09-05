%% test_periodic_unfolded_centres.m — periodic image sums on unfolded coords
%
%  The differential/Shannon grid cell mass (localPhiDiffAxisPeriodic in
%  entropyExpTens) builds its result by summing Gaussian images across
%  the period, and must reduce its input coordinate modulo the period
%  first. Otherwise a coordinate many periods from the canonical
%  [0, period) window --- for example an absolute spectral partial
%  thousands of cents above the grid --- never contributes, and the
%  routine returns a degenerate, input-independent value. These tests
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
% track the density the wrap declares: the wrapped normal under the
% default 'full-image', the minimum-image density (Eq. 1) under
% 'single-image'.
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
hTbFull = entropyExpTens(cMi, wMi, sMi, 1, false, true, 1200, ...
    'method', 'differential', 'base', exp(1), 'verbose', false);
absPerWarn = warning('off', 'buildExpTens:absPerSingleImage');
dSingleMi = buildExpTens({cMi(:)}, {wMi(:)}, sMi, 1, false, true, 1200, ...
    'wrap', {'single-image'}, 'verbose', false);
warning(absPerWarn);
hTbSingle = entropyExpTens(dSingleMi, 'method', 'differential', ...
    'base', exp(1), 'verbose', false);
results{end+1,1} = 'periodic differential: default wrap tracks the wrapped normal';
results{end,2}   = abs(hMin - hWrap) > 1e-4 ...
                   && abs(hTbFull - hWrap) < abs(hTbFull - hMin) ...
                   && abs(hTbFull - hWrap) < 1e-3;
results{end+1,1} = 'periodic differential: single-image wrap tracks minimum-image';
results{end,2}   = abs(hTbSingle - hMin) < abs(hTbSingle - hWrap) ...
                   && abs(hTbSingle - hMin) < 1e-3;


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
