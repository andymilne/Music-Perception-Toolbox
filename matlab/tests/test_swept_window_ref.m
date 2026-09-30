%% test_swept_window_ref.m — a rectangle's reference value ('ref')
%
%  A window's reference value is the point placed at each sweep value. For
%  a rectangle it is its centre by default; 'ref', 'start' places its start
%  there, so it covers [s, s + width), and 'ref', 'end' its end, covering
%  [s - width, s). Either is the centred window at a shifted sweep value,
%  and a rectangle's default sweep values (its pieces) shift with it.
%  Mirror of Python's tests/test_swept_window_ref.py.
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

claveR = [0 3 6 10 12];
tR   = [claveR, claveR + 16];
pR   = 100 * [60 62 64 67 64 67 69 71 74 71];
ctxR = {pR, tR};
qryR = {pR(1:3), tR(1:3)};
geomR = {[20 0.5], [1 1], [false false], [true false], [1200 0]};
WR = 10;
SR = 0:0.5:20;
massR = @(win, varargin) sweptMass(ctxR, [], geomR{:}, 'window', {2, win}, ...
    'drop', 2, varargin{:});
% The G-major triad's presence in the window: the one-sided similarity of
% the notes in it with the triad, a count of the notes on its tones in
% units of the triad's three.
gMajR = {100 * [67 71 74], zeros(1, 3)};
countG = @(win, varargin) sweptSimilarity(ctxR, [], gMajR, [], geomR{:}, ...
    'align', {2, 'window'}, 'window', {2, win}, 'drop', 2, ...
    'normalize', 'oneSidedDenom', varargin{:});
shareR = @(win, s) countG(win, 'sweep', {2, s});

% Start and end are the centred window at a shifted sweep value.
simR = @(win, s) sweptSimilarity(ctxR, [], qryR, [], geomR{:}, ...
    'sweep', {2, s}, 'align', {2, 'window'}, 'window', {2, win}, 'drop', 2);
results(end+1, :) = {'window ref: start and end are the centred window shifted', ...
    max(abs(shareR({'rect', 'width', WR, 'ref', 'start'}, SR) ...
        - shareR({'rect', 'width', WR}, SR + WR / 2))) < 1e-14 ...
    && max(abs(shareR({'rect', 'width', WR, 'ref', 'end'}, SR) ...
        - shareR({'rect', 'width', WR}, SR - WR / 2))) < 1e-14 ...
    && isequal(shareR(struct('shape', 'rect', 'width', WR, 'ref', 'centre'), SR), ...
        shareR({'rect', 'width', WR}, SR)) ...
    && max(abs(simR({'rect', 'width', WR, 'ref', 'start'}, SR) ...
        - simR({'rect', 'width', WR}, SR + WR / 2))) < 1e-14}; %#ok<SAGROW>

% Half-open: the lower edge included and the upper not, wherever the
% reference lies. With width 3, onsets 3 and 6 are one width apart.
countR = @(win, s) massR(win, 'sweep', {2, s});
results(end+1, :) = {'window ref: half-open edges follow the rectangle', ...
    abs(countR({'rect', 'width', 3, 'ref', 'start'}, 3) - 1) < 1e-12 ...
    && abs(countR({'rect', 'width', 3, 'ref', 'end'}, 6) - 1) < 1e-12 ...
    && abs(countR({'rect', 'width', 3, 'ref', 'start', 'edges', 'closed'}, 3) - 2) < 1e-12 ...
    && abs(countR({'rect', 'width', 3, 'ref', 'end', 'edges', 'closed'}, 6) - 2) < 1e-12}; %#ok<SAGROW>

% The default sweep values, a rectangle's pieces, shift with the reference.
[mD, ~, svD] = countG({'rect', 'width', WR, 'ref', 'start'}, 'sweep', 2);
sD = svD{2};
% Both ends are breakpoints here (the first and last onsets), so each keeps
% its own value and the piece beside it is sampled just inside it.
brkD = mean(reshape(sD(3:end - 2), 2, []), 1);
wantD = unique([tR - WR, tR]);
wantD = wantD(wantD > min(tR) & wantD < max(tR));
results(end+1, :) = {'window ref: the default pieces shift with the reference', ...
    sD(1) == min(tR) && sD(end) == max(tR) ...
    && sD(2) - sD(1) > 0 && sD(2) - sD(1) < 1e-4 ...
    && sD(end) - sD(end - 1) > 0 && sD(end) - sD(end - 1) < 1e-4 ...
    && numel(brkD) == numel(wantD) && max(abs(brkD - wantD)) < 1e-9 ...
    && max(abs(mD - shareR({'rect', 'width', WR}, sD + WR / 2))) < 1e-14}; %#ok<SAGROW>

% An end of the range on a breakpoint keeps its own value: a window
% starting at 0 holds C, D, and E, and one starting just above it D, E,
% and G (the C leaves and the G at onset 10 enters), so the G-major count
% rises from one note to two, and no line joins the two values.
mE = countG({'rect', 'width', WR, 'ref', 'start'}, 'sweep', 2);
results(end+1, :) = {'window ref: an end on a breakpoint keeps its own value', ...
    abs(mE(1) - shareR({'rect', 'width', WR, 'ref', 'start'}, 0)) < 1e-12 ...
    && abs(mE(2) - shareR({'rect', 'width', WR, 'ref', 'start'}, 1)) < 1e-12 ...
    && abs(mE(1) - mE(2)) > 0.1 && abs(mE(2) - mE(3)) < 1e-12}; %#ok<SAGROW>

% Refusals: a start or end needs a rectangle of given width.
okRef = true;
badRef = {{'gaussian', 'sd', 2, 'ref', 'start'}, 'sweptMass:windowRef'; ...
          {'rect', 'width', WR, 'ref', 'middle'}, 'sweptMass:windowRef'};
for k = 1:size(badRef, 1)
    try
        massR(badRef{k, 1}, 'sweep', {2, SR});
        okRef = false;
    catch e
        okRef = okRef && strcmp(e.identifier, badRef{k, 2});
    end
end
try
    sweptSimilarity(ctxR, [], qryR, [], geomR{:}, 'sweep', {2, SR}, ...
        'align', {2, 'both'}, 'window', {2, {'rect', 'ref', 'start'}});
    okRef = false;
catch e
    okRef = okRef && strcmp(e.identifier, 'sweptSimilarity:windowRef');
end
results(end+1, :) = {'window ref: refusals', okRef}; %#ok<SAGROW>

if standalone
    nPass = sum([results{:, 2}]);
    fprintf('\n=== test_swept_window_ref: %d passed, %d failed (of %d) ===\n', ...
            nPass, size(results, 1) - nPass, size(results, 1));
end
