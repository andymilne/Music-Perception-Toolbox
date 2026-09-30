%% test_swept_sweep_values.m — the sweep values returned by the swept functions
%
%  sweptSimilarity (third output), sweptEntropy and sweptMass (second
%  output) return the sweep values, listed or generated, as a 1 x A cell:
%  the axes of the profile. Mirror of Python's
%  tests/test_swept_sweep_values.py.
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

claveV = [0 3 6 10 12];
tV   = [claveV, claveV + 16];
pV   = 100 * [60 62 64 67 64 67 69 71 74 71];
ctxV = {pV, tV};
qryV = {pV(1:3), tV(1:3)};
geomV = {[20 0.5], [1 1], [false false], [true false], [1200 0]};
winV = struct('shape', 'gaussian', 'sd', 4);

% Listed values come back as given.
sL = 0:0.5:28;
[mL, svL] = sweptMass(ctxV, [], geomV{:}, 'sweep', {2, sL}, ...
    'window', {2, winV}, 'drop', 2, 'region', {1, [-50 50]}, ...
    'normalize', 'total');
results(end+1, :) = {'sweep values: listed values are returned as given', ...
    iscell(svL) && numel(svL) == 2 && isempty(svL{1}) ...
    && isequal(svL{2}(:)', sL) && numel(mL) == numel(sL)}; %#ok<*SAGROW>

% Generated values: the default start and stop, with a given step, match
% the same list given explicitly.
[mG, svG] = sweptMass(ctxV, [], geomV{:}, 'sweep', 2, 'step', 0.5, ...
    'window', {2, winV}, 'drop', 2, 'region', {1, [-50 50]}, ...
    'normalize', 'total');
results(end+1, :) = {'sweep values: generated values match the explicit list', ...
    max(abs(svG{2}(:)' - sL)) < 1e-12 && max(abs(mG - mL)) < 1e-12};

% Fully defaulted: one value per point of the profile.
[hD, svD] = sweptEntropy(ctxV, [], geomV{:}, 'sweep', 2, ...
    'window', {2, winV}, 'drop', 2, 'method', 'renyi2');
results(end+1, :) = {'sweep values: defaulted values are the axes of the profile', ...
    numel(svD{2}) == numel(hD) && svD{2}(1) == 0 && svD{2}(end) <= 28};

% sweptSimilarity: under 'query' the sweep values equal the offsets;
% under 'both' they differ from them by the query's middle.
[~, muQ, svQ] = sweptSimilarity(ctxV, [], qryV, [], geomV{:}, 'sweep', 2);
[~, muB, svB] = sweptSimilarity(ctxV, [], qryV, [], geomV{:}, 'sweep', 2, ...
    'align', {2, 'both'});
results(end+1, :) = {'sweep values: equal the offsets under query, not under both', ...
    max(abs(svQ{2} - muQ{2})) < 1e-12 ...
    && max(abs(svB{2} - muB{2} - mean(tV(1:3)))) < 1e-12};

% 'independent': the pair {windowValues, queryValues}.
[sI, ~, svI] = sweptSimilarity(ctxV, [], qryV, [], geomV{:}, ...
    'sweep', {2, {[0 16], [0 8 16]}}, 'align', {2, 'independent'}, ...
    'window', {2, {'rect', 8}});
results(end+1, :) = {'sweep values: independent gives the window and query lists', ...
    iscell(svI{2}) && isequal(svI{2}{1}(:)', [0 16]) ...
    && isequal(svI{2}{2}(:)', [0 8 16]) && isequal(size(sI), [2 3])};

% A pure rectangle placed alone: the default sweep values sample each
% piece between breakpoints (each event's value plus or minus half the
% width) just inside both ends, and each value holds throughout its piece.
wR = 4;
[mR, svR] = sweptMass(ctxV, [], geomV{:}, 'sweep', 2, ...
    'window', {2, {'rect', wR}}, 'drop', 2, 'region', {1, [-50 50]}, ...
    'normalize', 'total');
sR = svR{2}(:)';
brk = unique([tV - wR / 2, tV + wR / 2]);
brk = brk(brk > min(tV) & brk < max(tV));
innerR = reshape(sR(2:end - 1), 2, []);
edgesR = [min(tV), brk, max(tV)];
midsR = (edgesR(1:end - 1) + edgesR(2:end)) / 2;
mMid = sweptMass(ctxV, [], geomV{:}, 'sweep', {2, midsR}, ...
    'window', {2, {'rect', wR}}, 'drop', 2, 'region', {1, [-50 50]}, ...
    'normalize', 'total');
[~, svU] = sweptMass(ctxV, [], geomV{:}, 'sweep', 2, 'step', 1, ...
    'window', {2, {'rect', wR}}, 'drop', 2);
results(end+1, :) = {'sweep values: a rectangle takes its pieces', ...
    sR(1) == min(tV) && sR(end) == max(tV) ...
    && max(abs(mean(innerR, 1) - brk)) < 1e-9 ...
    && max(abs(mR(1:2:end) - mMid)) < 1e-12 ...
    && max(abs(mR(2:2:end) - mMid)) < 1e-12 ...
    && max(abs(diff(svU{2}) - 1)) < 1e-12};

if standalone
    nPass = sum([results{:, 2}]);
    fprintf('\n=== test_swept_sweep_values: %d passed, %d failed (of %d) ===\n', ...
            nPass, size(results, 1) - nPass, size(results, 1));
end
