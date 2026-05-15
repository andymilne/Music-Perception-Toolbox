%% test_mobius_combinatorics.m — Combinatorial primitives in +mobius/
%
%  Tests for the v2.2 Mobius-Bulger orbit-decomposition combinatorial
%  primitives in matlab/+mobius/:
%    integerPartitions, autSize, mobiusForBlocksizes,
%    labelledPairsRealisingM, enumerateContingencyTables, canonicalForm.
%
%  Mirrors python/tests/test_mobius.py (combinatorial section).
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

%% ---- mobius.integerPartitions ----

% Number of partitions of r matches the partition function p(r) (OEIS A000041).
expected = [1 1 2 3 5 7 11 15 22 30];
ok = true;
for r = 0:9
    if numel(mobius.integerPartitions(r)) ~= expected(r + 1)
        ok = false; break
    end
end
results{end+1, 1} = 'mobius.integerPartitions: count matches p(r) for r=0..9';
results{end, 2}   = ok;

% Every yielded partition is weakly decreasing.
ok = true;
for r = 1:8
    P = mobius.integerPartitions(r);
    for k = 1:numel(P)
        if any(diff(P{k}) > 0)
            ok = false; break
        end
    end
    if ~ok; break; end
end
results{end+1, 1} = 'mobius.integerPartitions: weakly decreasing';
results{end, 2}   = ok;

% Every yielded partition sums to r.
ok = true;
for r = 1:8
    P = mobius.integerPartitions(r);
    for k = 1:numel(P)
        if sum(P{k}) ~= r
            ok = false; break
        end
    end
    if ~ok; break; end
end
results{end+1, 1} = 'mobius.integerPartitions: parts sum to r';
results{end, 2}   = ok;

% Empty partition: integerPartitions(0) yields exactly one empty partition.
P0 = mobius.integerPartitions(0);
results{end+1, 1} = 'mobius.integerPartitions: r=0 yields one empty partition';
results{end, 2}   = numel(P0) == 1 && isempty(P0{1});

%% ---- mobius.autSize ----

results{end+1, 1} = 'mobius.autSize: empty partition = 1';
results{end, 2}   = mobius.autSize([]) == 1;

results{end+1, 1} = 'mobius.autSize: single block = 1';
results{end, 2}   = mobius.autSize(3) == 1;

results{end+1, 1} = 'mobius.autSize: [2 2] = 2';
results{end, 2}   = mobius.autSize([2 2]) == 2;

results{end+1, 1} = 'mobius.autSize: [3 3 2 1 1] = 4';
results{end, 2}   = mobius.autSize([3 3 2 1 1]) == 4;  % 2! * 1! * 2!

results{end+1, 1} = 'mobius.autSize: [1 1 1 1] = 24';
results{end, 2}   = mobius.autSize([1 1 1 1]) == 24;  % 4!

%% ---- mobius.mobiusForBlocksizes ----

% Per-block coefficients (-1)^(m-1) (m-1)! for m=1..6.
expected = [1 -1 2 -6 24 -120];
ok = true;
for m = 1:6
    if mobius.mobiusForBlocksizes(m) ~= expected(m)
        ok = false; break
    end
end
results{end+1, 1} = 'mobius.mobiusForBlocksizes: single-block table';
results{end, 2}   = ok;

% Multi-block: product across blocks.
results{end+1, 1} = 'mobius.mobiusForBlocksizes: [2 1] = -1';
results{end, 2}   = mobius.mobiusForBlocksizes([2 1]) == -1;     % -1 * 1

results{end+1, 1} = 'mobius.mobiusForBlocksizes: [3 2 1] = -2';
results{end, 2}   = mobius.mobiusForBlocksizes([3 2 1]) == -2;   % 2 * -1 * 1

results{end+1, 1} = 'mobius.mobiusForBlocksizes: [2 2] = 1';
results{end, 2}   = mobius.mobiusForBlocksizes([2 2]) == 1;      % -1 * -1

%% ---- mobius.labelledPairsRealisingM ----

% Worked example: M = [2 0; 0 2] -> 4! / (2! 2!) = 6.
results{end+1, 1} = 'mobius.labelledPairsRealisingM: [2 0; 0 2] = 6';
results{end, 2}   = mobius.labelledPairsRealisingM([2 0; 0 2]) == 6;

% Identity diagonal: M = I_r -> r! / 1^r = r!.
ok = true;
for r = 1:6
    if mobius.labelledPairsRealisingM(eye(r)) ~= factorial(r)
        ok = false; break
    end
end
results{end+1, 1} = 'mobius.labelledPairsRealisingM: identity = r! for r=1..6';
results{end, 2}   = ok;

% Single-cell: M = [r] -> r!/r! = 1 (trivial).
ok = true;
for r = 1:6
    if mobius.labelledPairsRealisingM(r) ~= 1
        ok = false; break
    end
end
results{end+1, 1} = 'mobius.labelledPairsRealisingM: 1x1 [r] = 1';
results{end, 2}   = ok;

%% ---- mobius.enumerateContingencyTables ----

% 0-1 tables with all-1 margins of length r equals r!.
ok = true;
for r = 1:6
    margins = ones(1, r);
    T = mobius.enumerateContingencyTables(margins, margins);
    if numel(T) ~= factorial(r)
        ok = false; break
    end
end
results{end+1, 1} = 'mobius.enumerateContingencyTables: r! tables with all-1 margins';
results{end, 2}   = ok;

% Margins respected for non-trivial input.
T = mobius.enumerateContingencyTables([3 2 1], [2 2 2]);
ok = true;
for k = 1:numel(T)
    M = T{k};
    if any(sum(M, 2)' ~= [3 2 1]) || any(sum(M, 1) ~= [2 2 2])
        ok = false; break
    end
end
results{end+1, 1} = 'mobius.enumerateContingencyTables: margins respected ([3 2 1] vs [2 2 2])';
results{end, 2}   = ok;

% Mismatched margins: returns empty.
T = mobius.enumerateContingencyTables([1 1], [3]);
results{end+1, 1} = 'mobius.enumerateContingencyTables: mismatched margins -> {}';
results{end, 2}   = isempty(T);

% Trivial 0x0 case: one empty table.
T = mobius.enumerateContingencyTables([], []);
results{end+1, 1} = 'mobius.enumerateContingencyTables: 0x0 -> one empty table';
results{end, 2}   = numel(T) == 1 && isequal(size(T{1}), [0 0]);

%% ---- mobius.canonicalForm ----

% Idempotent: canonicalising a canonical form returns the same thing.
M = [2 1 0; 0 2 1; 1 0 2];
rs = [3 3 3]; cs = [3 3 3];
[r1, c1, M1] = mobius.canonicalForm(M, rs, cs);
[r2, c2, M2] = mobius.canonicalForm(M1, r1, c1);
results{end+1, 1} = 'mobius.canonicalForm: idempotent';
results{end, 2}   = isequal(r1, r2) && isequal(c1, c2) && isequal(M1, M2);

% Row permutation invariant: swapping two rows of equal block size
% gives the same canonical form.
M = [2 1 0; 1 2 0; 0 0 3];
rs = [3 3 3]; cs = [3 3 3];
[r1, c1, M1] = mobius.canonicalForm(M, rs, cs);
M_swap = M([2 1 3], :);  % swap rows 1 and 2 (both size 3)
[r2, c2, M2] = mobius.canonicalForm(M_swap, rs, cs);
results{end+1, 1} = 'mobius.canonicalForm: row permutation invariant within size group';
results{end, 2}   = isequal(r1, r2) && isequal(c1, c2) && isequal(M1, M2);

% Column permutation invariant.
M = [2 1 0; 1 0 2; 0 2 1];
rs = [3 3 3]; cs = [3 3 3];
[r1, c1, M1] = mobius.canonicalForm(M, rs, cs);
M_swap = M(:, [3 1 2]);
[r2, c2, M2] = mobius.canonicalForm(M_swap, rs, cs);
results{end+1, 1} = 'mobius.canonicalForm: column permutation invariant within size group';
results{end, 2}   = isequal(r1, r2) && isequal(c1, c2) && isequal(M1, M2);

% Output sums consistent with input sums.
M = [2 0 1; 0 2 0; 0 0 1];
rs = [3 2 1]; cs = [2 2 2];
[rsCanon, csCanon, MCanon] = mobius.canonicalForm(M, rs, cs);
results{end+1, 1} = 'mobius.canonicalForm: row sums of MCanon == rsCanon';
results{end, 2}   = isequal(sum(MCanon, 2)', rsCanon);

results{end+1, 1} = 'mobius.canonicalForm: col sums of MCanon == csCanon';
results{end, 2}   = isequal(sum(MCanon, 1), csCanon);

%% ---- standalone summary ----

if standalone
    nPass = sum([results{:, 2}]);
    nFail = size(results, 1) - nPass;
    fprintf('\n=== mobius combinatorics tests ===\n\n');
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
        error('test_mobius_combinatorics:failed', '%d test(s) failed.', nFail);
    end
end
