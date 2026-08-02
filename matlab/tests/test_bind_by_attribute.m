%% test_bind_by_attribute.m — run-length (bind-by-attribute) binding
%
%  Mirror of Python tests/test_bind_by_attribute.py (structural + self-sim
%  + large-tuple-size ragged smoke). bindEvents(..., 'groupBy', a) gathers
%  consecutive events sharing a constant value on attribute a into one
%  ragged super-event (NaN-padded to the max group size, padded positions at
%  zero weight); the outer tuple size defaults to the smallest group size.

if ~exist('results', 'var')
    results = {};
    standalone = true;
else
    standalone = false;
end
tol = 1e-9;

% --- structure: groups 3,2,1,4 -> Lmax=4, nPrime=4, rOuter=min=1 ----------
pitch = [60 64 67 62 65 60 59 62 67 71];
onset = [0 0 0 1 1 2 3 3 3 3];
[pb, wb, sp] = bindEvents({pitch, onset}, [], [], 'groupBy', 2);
okShape = isequal(size(pb{1}), [4, 4]);
okR     = isequal(sp{1}.r, [1 1]);
okPad   = isequal(sum(isnan(pb{1}), 1), [1 2 3 0]);
okWzero = all(wb{1}(isnan(pb{1})) == 0);
results(end+1, :) = {'run-length structure (sizes 3/2/1/4)', ...
    okShape && okR && okPad && okWzero}; %#ok<SAGROW>

% --- rOuter defaults to smallest group size ------------------------------
onset2 = [0 0 0 0 0 0 1 1 1 1 2 2 2 2 2];   % sizes 6,4,5 -> min 4
pit2 = 60:74;
[~, ~, sp2] = bindEvents({pit2, onset2}, [], [], 'groupBy', 2);
results(end+1, :) = {'rOuter defaults to min group size', sp2{1}.r(2) == 4}; %#ok<SAGROW>

% --- self-similarity = 1 on a ragged density -----------------------------
[pb3, wb3, sp3] = bindEvents({pit2, onset2}, [], [], 'groupBy', 2, 'rOuter', 4, 'symOuter', true);
d3 = buildExpTens(pb3, wb3, 'sigma', [30 0.01], 'isPer', [false false], 'period', [0 0], 'specs', sp3, 'verbose', false);
s3 = cosSimExpTens(d3, d3, 'verbose', false);
results(end+1, :) = {'ragged self-similarity == 1', abs(s3 - 1) < tol}; %#ok<SAGROW>

% --- large-tuple-size ragged smoke (orbit path carries it) ---------------------
onset4 = [zeros(1,7), ones(1,6), 2*ones(1,8)];   % sizes 7,6,8 -> min 6
pit4 = 60 + (0:numel(onset4)-1);
[pb4, wb4, sp4] = bindEvents({pit4, onset4}, [], [], 'groupBy', 2, 'rOuter', 6, 'symOuter', true);
d4 = buildExpTens(pb4, wb4, 'sigma', [30 0.01], 'isPer', [false false], 'period', [0 0], 'specs', sp4, 'verbose', false);
s4 = cosSimExpTens(d4, d4, 'verbose', false);
results(end+1, :) = {'large-tuple-size ragged self-similarity == 1', abs(s4 - 1) < tol}; %#ok<SAGROW>

% --- consecutive runs, not global grouping -------------------------------
[pbc, ~, ~] = bindEvents({[60 61 62 63], [0 0 1 0]}, [], [], 'groupBy', 2);
results(end+1, :) = {'consecutive runs (0,0,1,0 -> 3 groups)', size(pbc{1}, 2) == 3}; %#ok<SAGROW>

% --- validation ----------------------------------------------------------
threw = false;
try
    bindEvents({[60 61 62], [0 0 1]}, [], 2, 'groupBy', 2);   % bindOrders + groupBy
catch
    threw = true;
end
results(end+1, :) = {'bindOrders + groupBy errors', threw}; %#ok<SAGROW>

threwK = false;
try
    bindEvents({[60 62; 64 66; 67 69].', [0 0 1]}, [], [], 'groupBy', 1);  % K~=1
catch
    threwK = true;
end
results(end+1, :) = {'groupBy K~=1 errors', threwK}; %#ok<SAGROW>

if standalone
    nPass = sum([results{:, 2}]);
    fprintf('test_bind_by_attribute: %d/%d passed\n', nPass, size(results, 1));
    for i = 1:size(results, 1)
        if ~results{i, 2}; fprintf('  FAIL: %s\n', results{i, 1}); end
    end
end
