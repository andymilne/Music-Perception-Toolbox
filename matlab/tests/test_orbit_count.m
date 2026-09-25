%% test_orbit_count.m — |Omega_r| without the orbit table
%
%  Cost models need only |Omega_r|, so pricing a route must never build
%  an orbit table (hours at r = 9). mobius.orbitCount reads the count
%  from a closed table, checked here against the shipped tables' lengths
%  for r <= 8, and the sweep chooser prices a swept r = 9 attribute
%  without building. Mirrors python/tests/test_orbit_count.py (which
%  also checks the closed table against an independent Burnside count).

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

okShipped = true;
for r = 2:8
    okShipped = okShipped && ...
        mobius.orbitCount(r) == numel(mobius.getOrbitTable(r));
end
results{end+1,1} = 'mobius.orbitCount: equals the shipped table length for r=2..8';
results{end,2}   = okShipped;

results{end+1,1} = 'mobius.orbitCount: closed values for r=9..12';
results{end,2}   = isequal(arrayfun(@mobius.orbitCount, 9:12), ...
                           [9945, 34207, 119369, 429250]);

results{end+1,1} = 'mobius.orbitCount: r=1 and r=13 raise';
results{end,2}   = throwsErrorWithId(@() mobius.orbitCount(1), ...
                       'mobius:orbitCount:rTooSmall') ...
                && throwsErrorWithId(@() mobius.orbitCount(13), ...
                       'mobius:orbitCount:rTooLarge');

% A swept r = 9 attribute that both routes admit: the chooser prices
% the orbit route, which must not build (or even look up) a table. A
% user cache directory that cannot hold a table makes any build visible
% as an r = 9 table appearing there.
prevCache = getenv('MPT_CACHE_DIR');
tmpCache = tempname;
mkdir(tmpCache);
setenv('MPT_CACHE_DIR', tmpCache);
restoreCache = onCleanup(@() setenv('MPT_CACHE_DIR', prevCache)); %#ok<NASGU>
rng(9);
dX = buildMaet({randn(9, 1) * 3}, [], 0.9, 9, false, false, NaN, true, ...
               'verbose', false);
dY = buildMaet({randn(9, 1) * 3}, [], 0.9, 9, false, false, NaN, true, ...
               'verbose', false);
off = linspace(-2, 2, 5);
t0 = tic;
vals = sweepSimMaet(dX, dY, off, 'verbose', false);
elapsed = toc(t0);
results{end+1,1} = 'sweep: pricing a swept r=9 attribute builds no orbit table';
results{end,2}   = all(isfinite(vals)) && elapsed < 600 ...
                   && ~isfile(fullfile(tmpCache, 'orbit_r9.mat'));
clear restoreCache

%% ---- Standalone summary ----

if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:,2)));
    nFail = numel(results(:,1)) - nPass;
    fprintf('\n=== test_orbit_count: %d passed, %d failed (of %d) ===\n', ...
        nPass, nFail, numel(results(:,1)));
    clear cleanupDefaults
    if nFail > 0
        for ii = 1:size(results, 1)
            if ~isequal(results{ii, 2}, true)
                fprintf('  FAIL  %s\n', results{ii, 1});
            end
        end
    end
end
