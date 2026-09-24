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
[pb, wb, sp] = unpackPreMaet(bindEvents({pitch, onset}, [], [], 'groupBy', 2));
okShape = isequal(size(pb{1}), [4, 4]);
okR     = isequal(sp{1}.r, [1 1]);
okPad   = isequal(sum(isnan(pb{1}), 1), [1 2 3 0]);
okWzero = all(wb{1}(isnan(pb{1})) == 0);
results(end+1, :) = {'run-length structure (sizes 3/2/1/4)', ...
    okShape && okR && okPad && okWzero}; %#ok<SAGROW>

% --- rOuter defaults to smallest group size ------------------------------
onset2 = [0 0 0 0 0 0 1 1 1 1 2 2 2 2 2];   % sizes 6,4,5 -> min 4
pit2 = 60:74;
[~, ~, sp2] = unpackPreMaet(bindEvents({pit2, onset2}, [], [], 'groupBy', 2));
results(end+1, :) = {'rOuter defaults to min group size', sp2{1}.r(2) == 4}; %#ok<SAGROW>

% --- self-similarity = 1 on a ragged density -----------------------------
[pb3, wb3, sp3] = unpackPreMaet(bindEvents({pit2, onset2}, [], [], 'groupBy', 2, 'rOuter', 4, 'exchOuter', true));
d3 = buildMaet(pb3, wb3, 'sigma', [30 0.01], 'isPer', [false false], 'period', [0 0], 'specs', sp3, 'verbose', false);
s3 = simMaet(d3, d3, 'verbose', false);
results(end+1, :) = {'ragged self-similarity == 1', abs(s3 - 1) < tol}; %#ok<SAGROW>

% --- large-tuple-size ragged smoke (orbit path carries it) ---------------------
onset4 = [zeros(1,7), ones(1,6), 2*ones(1,8)];   % sizes 7,6,8 -> min 6
pit4 = 60 + (0:numel(onset4)-1);
[pb4, wb4, sp4] = unpackPreMaet(bindEvents({pit4, onset4}, [], [], 'groupBy', 2, 'rOuter', 6, 'exchOuter', true));
d4 = buildMaet(pb4, wb4, 'sigma', [30 0.01], 'isPer', [false false], 'period', [0 0], 'specs', sp4, 'verbose', false);
s4 = simMaet(d4, d4, 'verbose', false);
results(end+1, :) = {'large-tuple-size ragged self-similarity == 1', abs(s4 - 1) < tol}; %#ok<SAGROW>

% --- consecutive runs, not global grouping -------------------------------
[pbc, ~, ~] = unpackPreMaet(bindEvents({[60 61 62 63], [0 0 1 0]}, [], [], 'groupBy', 2));
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

% --- groupBy by name, and the carried kernel geometry ---------------------
% Run-length binding regroups values without touching them, so an
% attribute's sigma (and periodicity) belong to the nested spec just as they
% did to the flat one; naming the grouping attribute is the same selection
% bindAttributes and selectPreMaet accept.
bnPitch = [60 64 67 62 65];
bnChord = [0 0 0 1 1];
bnSpecs = flatSpecs({bnPitch, bnChord}, 'sigma', [0.5 0.25], ...
                    'isPer', [true false], 'period', [12 0], ...
                    'name', {'pitch', 'chord'});
[pbName, wbName, spName] = unpackPreMaet(bindEvents({bnPitch, bnChord}, [], [], ...
    'groupBy', 'chord', 'specs', bnSpecs, 'rOuter', 2));
[pbIdx, ~, ~] = unpackPreMaet(bindEvents({bnPitch, bnChord}, [], [], ...
    'groupBy', 2, 'specs', bnSpecs, 'rOuter', 2));
okSame = isequaln(pbName{1}, pbIdx{1});
okSigma = isfield(spName{1}, 'sigma') && abs(spName{1}.sigma - 0.5) < tol ...
    && isfield(spName{2}, 'sigma') && abs(spName{2}.sigma - 0.25) < tol;
okPer = isfield(spName{1}, 'isPer') && spName{1}.isPer ...
    && abs(spName{1}.period - 12) < tol;
okBuild = true;
try
    % The spec carries its own sigma, so the density builds without one.
    buildMaet(pbName, wbName, 'specs', spName, 'verbose', false);
catch
    okBuild = false;
end
results(end+1, :) = {'groupBy by name, sigma/isPer/period carried', ...
    okSame && okSigma && okPer && okBuild}; %#ok<SAGROW>

% --- an inner attribute of K > 1 gets its weight on every value row -------
% A weight given per event applies to each of that event's values, so the
% bound weight matrix has Lmax * K_a rows, as the bound value matrix does.
bwCoords = [60 64 67 62; 0 1 2 3];          % K = 2
bwChord  = [0 0 1 1];
bwW      = [1 0.5 2 0.25];
[pbW, wbW, spW] = unpackPreMaet(bindEvents({bwCoords, bwChord}, ...
    {bwW, ones(1, 4)}, [], 'groupBy', 2, 'rOuter', 2));
okWshape = isequal(size(wbW{1}), size(pbW{1}));
okWrows = max(abs(wbW{1}(1:2, :) - [1 2; 1 2]), [], 'all') < tol ...
    && max(abs(wbW{1}(3:4, :) - [0.5 0.25; 0.5 0.25]), [], 'all') < tol;
okWbuild = true;
try
    buildMaet(pbW, wbW, 'sigma', [1 1], 'isPer', [false false], ...
              'period', [0 0], 'specs', spW, 'verbose', false);
catch
    okWbuild = false;
end
results(end+1, :) = {'groupBy weights have one row per value', ...
    okWshape && okWrows && okWbuild}; %#ok<SAGROW>

if standalone
    nPass = sum([results{:, 2}]);
    fprintf('test_bind_by_attribute: %d/%d passed\n', nPass, size(results, 1));
    for i = 1:size(results, 1)
        if ~results{i, 2}; fprintf('  FAIL: %s\n', results{i, 1}); end
    end
end
