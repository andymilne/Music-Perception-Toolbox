%% test_mobius_orbit_table.m — Orbit table builder, cache, and shipped tables
%
%  Tests for the v2.2 orbit-table machinery in matlab/+mobius/:
%    buildOrbitTable, getOrbitTable, buildAndSavePrebuiltTables.
%
%  Mirrors python/tests/test_mobius.py (orbit-table section).
%
%  Standalone-runnable. When invoked from test_mpt.m the existing
%  `results` cell is appended to; when run alone, results print on the
%  fly and a summary follows.

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

% Bell numbers B_0..B_6 for the weights-sum-to-B_r^2 cross-check.
% Computed inline rather than imported, to keep this file self-contained.
B = [1 1 2 5 15 52 203 877 4140];

%% ---- Orbit count per r matches expected ----

% r = 2..8: 4, 10, 33, 92, 306, 948, 3210. Source: Möbius-Bulger orbit
% decomposition; r=7, r=8 counts measured at table-build time in v2.2.
expectedOrbits = [4 10 33 92 306 948 3210];
ok = true;
for r = 2:8
    T = mobius.getOrbitTable(r);
    if numel(T) ~= expectedOrbits(r - 1)
        ok = false;
        break
    end
end
results{end+1, 1} = 'mobius.getOrbitTable: orbit count for r=2..8 matches expected';
results{end, 2}   = ok;

%% ---- Orbit weights sum to B_r^2 ----

ok = true;
for r = 2:8
    T = mobius.getOrbitTable(r);
    sumW = sum([T.weight]);
    if sumW ~= B(r + 1)^2
        ok = false;
        break
    end
end
results{end+1, 1} = 'mobius.getOrbitTable: orbit weights sum to B_r^2 for r=2..8';
results{end, 2}   = ok;

%% ---- Per-orbit block-size consistency ----

% For every orbit, the block-size profiles m_A and m_B sum to r.
ok = true;
for r = 2:8
    T = mobius.getOrbitTable(r);
    for k = 1:numel(T)
        if sum(T(k).m_A) ~= r || sum(T(k).m_B) ~= r
            ok = false;
            break
        end
    end
    if ~ok; break; end
end
results{end+1, 1} = 'mobius.getOrbitTable: every orbit has sum(m_A) = sum(m_B) = r';
results{end, 2}   = ok;

%% ---- Edge consistency: row/column sums match block sizes ----

% For every orbit, the multiplicity matrix reconstructed from edges has
% row sums equal to m_A and column sums equal to m_B.
ok = true;
for r = 2:8
    T = mobius.getOrbitTable(r);
    for k = 1:numel(T)
        E = T(k).edges;
        Mrebuild = zeros(T(k).qA, T(k).qB);
        for e = 1:size(E, 1)
            Mrebuild(E(e, 1), E(e, 2)) = E(e, 3);
        end
        if any(sum(Mrebuild, 2)' ~= T(k).m_A) || any(sum(Mrebuild, 1) ~= T(k).m_B)
            ok = false;
            break
        end
    end
    if ~ok; break; end
end
results{end+1, 1} = 'mobius.getOrbitTable: edge row/col sums match m_A/m_B';
results{end, 2}   = ok;

%% ---- Möbius coefficient consistency ----

% Each orbit's mu equals mu(m_A) * mu(m_B), since both are functions of
% block-size profiles only.
ok = true;
for r = 2:8
    T = mobius.getOrbitTable(r);
    for k = 1:numel(T)
        expectedMu = mobius.mobiusForBlocksizes(T(k).m_A) ...
                   * mobius.mobiusForBlocksizes(T(k).m_B);
        if T(k).mu ~= expectedMu
            ok = false;
            break
        end
    end
    if ~ok; break; end
end
results{end+1, 1} = 'mobius.getOrbitTable: orbit mu = mu(m_A) * mu(m_B)';
results{end, 2}   = ok;

%% ---- buildOrbitTable matches getOrbitTable for shipped r ----

% Direct build (bypassing cache) must agree with cached table on the
% structural fields. Ordering may differ — cache returns the shipped
% table verbatim while build follows the live algorithm — so we compare
% by canonical-key sorting rather than by element-wise position.
ok = true;
for r = 2:5  % r=6 builds in ~7s under Octave; skip for runtime
    Tcache = mobius.getOrbitTable(r);
    Tbuild = mobius.buildOrbitTable(r);
    if numel(Tcache) ~= numel(Tbuild)
        ok = false; break
    end
    % Build sorted canonical-key strings on both sides.
    keysC = cell(numel(Tcache), 1);
    keysB = cell(numel(Tbuild), 1);
    for k = 1:numel(Tcache)
        keysC{k} = sprintf('w=%d mu=%d mA=%s mB=%s E=%s', ...
            Tcache(k).weight, Tcache(k).mu, ...
            mat2str(Tcache(k).m_A), mat2str(Tcache(k).m_B), ...
            mat2str(sortrows(Tcache(k).edges)));
        keysB{k} = sprintf('w=%d mu=%d mA=%s mB=%s E=%s', ...
            Tbuild(k).weight, Tbuild(k).mu, ...
            mat2str(Tbuild(k).m_A), mat2str(Tbuild(k).m_B), ...
            mat2str(sortrows(Tbuild(k).edges)));
    end
    if ~isequal(sort(keysC), sort(keysB))
        ok = false; break
    end
end
results{end+1, 1} = 'mobius.buildOrbitTable: direct build matches shipped table (r=2..5)';
results{end, 2}   = ok;

%% ---- In-memory cache ----

% Two consecutive calls return identical data; a deep `isequal` is the
% strongest practical check.
T1 = mobius.getOrbitTable(3);
T2 = mobius.getOrbitTable(3);
results{end+1, 1} = 'mobius.getOrbitTable: in-memory cache returns identical data';
results{end, 2}   = isequal(T1, T2);

%% ---- Range guards ----

threwLow = false;
try
    mobius.getOrbitTable(1);
catch
    threwLow = true;
end
results{end+1, 1} = 'mobius.getOrbitTable: r=1 raises an error';
results{end, 2}   = threwLow;

threwHigh = false;
try
    mobius.getOrbitTable(13);
catch
    threwHigh = true;
end
results{end+1, 1} = 'mobius.getOrbitTable: r=13 (above hard cap) raises an error';
results{end, 2}   = threwHigh;

%% ---- Shipped tables present at expected location ----

shippedDir = fullfile( ...
    fileparts(which('mobius.getOrbitTable')), '_orbit_tables');
ok = true;
for r = 2:8
    p = fullfile(shippedDir, sprintf('orbit_r%d.mat', r));
    if ~isfile(p)
        ok = false; break
    end
end
results{end+1, 1} = 'mobius._orbit_tables: orbit_r{2..8}.mat present';
results{end, 2}   = ok;

%% ---- standalone summary ----

if standalone
    nPass = sum([results{:, 2}]);
    nFail = size(results, 1) - nPass;
    fprintf('\n=== mobius orbit-table tests ===\n\n');
    for i = 1:size(results, 1)
        if results{i, 2}
            fprintf('  PASS  %s\n', results{i, 1});
        else
            fprintf('  FAIL  %s\n', results{i, 1});
        end
    end
    fprintf('\n=== Results: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    % Restore caller's pre-test defaults eagerly
    % (fires the helper's onCleanup destructor before
    % the conditional error below halts execution).
    clear cleanupDefaults
    if nFail > 0
        error('test_mobius_orbit_table:failed', '%d test(s) failed.', nFail);
    end
end
