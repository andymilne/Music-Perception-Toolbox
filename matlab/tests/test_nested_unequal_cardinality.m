%% test_nested_unequal_cardinality.m — nested contraction at r > 1 with
%  unequal inner cardinalities, plus the absolute-non-periodic [per] fix.
%
%  Guards two defects fixed together (mirror of the Python
%  test_nested_unequal_cardinality.py):
%
%   1. The tree contraction walked a single recipe for both axes of the
%      rectangular leaf kernel, so two densities whose nested cardinalities
%      differed (a 4-pitch prototype against an 8-pitch window, say) had the
%      Y axis mis-indexed and the cross inner product collapsed to ~0 while
%      self-similarity stayed 1. The contraction now threads one recipe per
%      side. (A MATLAB-specific facet: the tags were also mis-oriented as a
%      row, giving a degenerate one-slot recipe; oriented to a column now.)
%   2. Absolute-mode periodicity was inferred from a finite period rather
%      than the density's [per] flag, wrongly wrapping a non-periodic
%      attribute that happened to carry a finite period.
%
%  Cross-checks: the contraction against the exact Bulger enumeration within
%  MATLAB, and against independent brute-force goldens computed in Python.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

ATOL = 1e-9;     % contract-vs-bulger within MATLAB
GTOL = 1e-6;     % cross-language vs the Python brute-force goldens


% --- 1. Absolute, unequal cardinality: contract == bulger == Python brute ---
X = ndens([1 3 7 9],            [0 0 1 1],   2, 2, true, false, [], false, 1e9, 0.5);
Y = ndens([1.2 2.8 5.1 6.7 9.3 10.8], [0 0 0 1 1 1], 2, 2, true, false, [], false, 1e9, 0.5);
cC = cosSimExpTens(X, Y, 'method', 'contract', 'verbose', false);
cB = cosSimExpTens(X, Y, 'method', 'bulger',   'verbose', false);
results{end+1, 1} = 'nested-uneq: abs 2v3 r_in=2 contract==bulger==golden'; %#ok<*SAGROW>
results{end, 2}   = abs(cC - cB) < ATOL ...
                 && abs(cC - 0.2551553211) < GTOL ...
                 && abs(cosSimExpTens(X, X, 'method', 'contract', 'verbose', false) - 1) < ATOL;

X = ndens([1 3 5 7 9 11],     [0 0 0 1 1 1],   3, 2, true, false, [], false, 1e9, 0.5);
Y = ndens([1 3 5 2 7 9 11 8], [0 0 0 0 1 1 1 1], 3, 2, true, false, [], false, 1e9, 0.5);
cC = cosSimExpTens(X, Y, 'method', 'contract', 'verbose', false);
cB = cosSimExpTens(X, Y, 'method', 'bulger',   'verbose', false);
results{end+1, 1} = 'nested-uneq: abs 3v4 r_in=3 contract==bulger==golden';
results{end, 2}   = abs(cC - cB) < ATOL && abs(cC - 0.4809068849) < GTOL;


% --- 2. Reported regression: r=2 unequal, relative-periodic ---
%  4-pitch prototype vs the same chords doubled to 8 pitches. Was exactly 0
%  for every cross-comparison; must be nonzero, symmetric, self-similar 1.
P = 12; sg = 0.15;
proto4  = [0 4 7 0   7 11 2 7   0 4 7 0];
tags4   = [0 0 0 0   1 1 1 1    2 2 2 2];
win8    = [0 4 7 0 0 4 7 0   7 11 2 7 7 11 2 7   0 4 7 0 0 4 7 0];
tags8   = [0 0 0 0 0 0 0 0   1 1 1 1 1 1 1 1     2 2 2 2 2 2 2 2];
A = ndens(proto4, tags4, 2, 3, true, false, 1, true, P, sg);
B = ndens(win8,   tags8, 2, 3, true, false, 1, true, P, sg);
xy = cosSimExpTens(A, B, 'method', 'contract', 'verbose', false);
yx = cosSimExpTens(B, A, 'method', 'contract', 'verbose', false);
results{end+1, 1} = 'nested-uneq: r=2 relper nonzero, symmetric, self=1';
results{end, 2}   = xy > 1e-3 && abs(xy - yx) < ATOL ...
                 && abs(cosSimExpTens(A, A, 'method', 'contract', 'verbose', false) - 1) < ATOL ...
                 && abs(cosSimExpTens(B, B, 'method', 'contract', 'verbose', false) - 1) < ATOL;


% --- 3. Relative-periodic, unequal cardinality: contract == bulger ---
X = ndens([0 4 7 2 5 9 4 7 11],    [0 0 0 1 1 1 2 2 2],     2, 3, true, false, 1, true, 12, 0.2);
Y = ndens([0 4 7 0 2 5 9 2 4 7 11 4], [0 0 0 0 1 1 1 1 2 2 2 2], 2, 3, true, false, 1, true, 12, 0.2);
cC = cosSimExpTens(X, Y, 'method', 'contract', 'verbose', false);
cB = cosSimExpTens(X, Y, 'method', 'bulger',   'verbose', false);
results{end+1, 1} = 'nested-uneq: relper 3v4 contract==bulger';
results{end, 2}   = abs(cC - cB) < ATOL;


% --- 4. Equal cardinality unchanged (X==Y path is the original walk) ---
X = ndens([0 4 7 2 5 9 4 7 11], [0 0 0 1 1 1 2 2 2], 2, 3, true, false, 1, true, 12, 0.2);
Y = ndens([0 3 7 2 5 9 4 7 10], [0 0 0 1 1 1 2 2 2], 2, 3, true, false, 1, true, 12, 0.2);
cC = cosSimExpTens(X, Y, 'method', 'contract', 'verbose', false);
cB = cosSimExpTens(X, Y, 'method', 'bulger',   'verbose', false);
results{end+1, 1} = 'nested-uneq: equal-card contract==bulger, self=1';
results{end, 2}   = abs(cC - cB) < ATOL ...
                 && abs(cosSimExpTens(X, X, 'method', 'contract', 'verbose', false) - 1) < ATOL;


% --- 5. Absolute non-periodic with a finite stored period must not wrap ---
X = ndens([1 3 7 9],            [0 0 1 1],   2, 2, true, false, [], false, 2.0, 0.5);
Y = ndens([1.2 2.8 5.1 6.7 9.3 10.8], [0 0 0 1 1 1], 2, 2, true, false, [], false, 2.0, 0.5);
cSmall = cosSimExpTens(X, Y, 'method', 'contract', 'verbose', false);
results{end+1, 1} = 'nested-uneq: abs non-periodic finite period does not wrap';
results{end, 2}   = abs(cSmall - 0.2551553211) < GTOL;


% --- Standalone summary ---
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
    fprintf('\n=== test_nested_unequal_cardinality: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_nested_unequal_cardinality:failed', '%d test(s) failed.', nFail);
    end
end


% ----------------------------------------------------------------------
function d = ndens(p, tags, rIn, rOut, symIn, symOut, relOut, isPer, period, sigma)
    if isempty(relOut)
        sp = struct('tags', tags, 'r', [rIn rOut], 'sym', [symIn symOut]);
    else
        sp = struct('tags', tags, 'r', [rIn rOut], 'sym', [symIn symOut], ...
                    'rel', [0 relOut]);
    end
    d = buildExpTens({p(:)}, [], sigma, 1, false, isPer, period, ...
                     'nested', {sp}, 'verbose', false);
end
