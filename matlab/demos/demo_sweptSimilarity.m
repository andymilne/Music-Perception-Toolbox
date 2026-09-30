%% demo_translateSweep.m
% A sliding comparison by attribute translation: a query translated in
% pitch and time across a reference, with the similarity read at every
% offset.
%
% Scenario: a 3-note motif (C E G) hidden inside a 7-note melody
% (D E F C E G A, one note per second). The motif appears exactly at
% reference times 3, 4, 5. At each (pitch transposition, time shift)
% offset the query is translated and compared with the reference by the
% cosine similarity of a 2-attribute MAET (pitch periodic at the octave;
% time absolute and not periodic). The profile should peak at
% (0 cents, 3 s), where the query lands on the embedded C-E-G, and at
% (1200 cents, 3 s) by octave periodicity. Both sequences start at time
% 0, so a time offset is the time from the start of the reference to the
% start of the query (User Guide §7.3.4).
%
%   1. The inputs, as two pre-MAETs.
%   2. The whole profile in one call: windowedSimilarity with an
%      'offsets' map naming both attributes and no window.
%   3. The same call with a window in time, which travels with the query
%      and restricts the comparison to the reference's notes near it.
%   4. The two profiles plotted.
%   5. What the call in 2 computes: sweepSimMaet on the two built
%      densities, one pass over the tuple pairs and one evaluation per
%      offset, with no translated copy of the query built.
%   6. The same profile offset by offset, for transparency:
%      translateAttributes builds the translated copies, simMaet compares
%      them in its list mode, and an explicit loop over the offsets does
%      the same one comparison at a time.
%
% See also WINDOWEDSIMILARITY, SWEEPSIMMAET, TRANSLATEATTRIBUTES,
% SIMMAET, BUILDMAET.
%
% The Python mirror is demo_translate_sweep.py.

clear; close all;

% Keep the dispatcher's per-call announcements out of the printed output
% (showHints gates only those); restored at the end.
prevDefaults = mptDefaults('showHints', false);

%% ===================================================================
%  1. The reference melody and the query motif, as pre-MAETs
%  ===================================================================

fprintf('=== 1. Inputs ===\n');

% Reference: D-E-F-C-E-G-A at one note per second. The query C-E-G
% appears exactly at times 3, 4, 5.
refMidi  = [62 64 65 60 64 67 69];
refPitch = transformAttributes(refMidi, [], {'midi', 'cents'});
refTime  = 0:6;
refPAttr = {refPitch, refTime};

% Query: C-E-G, one note per second, starting at time 0 as the
% reference does.
qryMidi  = [60 64 67];
qryPitch = transformAttributes(qryMidi, [], {'midi', 'cents'});
qryTime  = 0:2;
qryPAttr = {qryPitch, qryTime};

% Per-attribute geometry, carried by both pre-MAETs' specs: pitch
% (attribute 1) is periodic at the octave; time (attribute 2) is absolute
% and not periodic.
sigma = [50, 0.3];            % cents, seconds
specs = flatSpecs(refPAttr, 'name', {'pitch', 'time'}, 'sigma', sigma, ...
                  'isPer', [true, false], 'period', [1200, 0]);
pmRef = packPreMaet(refPAttr, [], specs);
pmQry = packPreMaet(qryPAttr, [], specs);

% The offsets: pitch 0-1200 cents in 100-cent steps (the peak at 0
% recurs at 1200 because pitch is periodic), time -1 to 5 s in 0.25 s
% steps.
pitchGrid = 0:100:1200;
timeGrid  = -1:0.25:5;

fprintf('  reference: D-E-F-C-E-G-A, one note per second\n');
fprintf('  query    : C-E-G, one note per second\n');
fprintf('  (the motif appears exactly at reference times 3, 4, 5)\n');
fprintf('  sigma    : %.0f cents (pitch) / %.2f s (time)\n', sigma(1), sigma(2));
fprintf('  offsets  : %d pitch x %d time\n\n', numel(pitchGrid), numel(timeGrid));

%% ===================================================================
%  2. The whole profile in one call
%  ===================================================================

fprintf('=== 2. windowedSimilarity with an offsets map ===\n');

% An offsets map {a, offsets; ...} translates each named attribute of
% the query by every combination of its offsets and compares; an
% attribute is windowed only if 'contextWindow' names it, and here none
% is, so this is attribute translation and nothing else. The output is
% indexed by the offsets, one dimension per attribute: (pitch, time).
S = windowedSimilarity(pmRef, pmQry, [], ...
    'offsets', {1, pitchGrid; 2, timeGrid}, 'normalize', 'cosine');

[sMax, iLin] = max(S(:));
[iP, iT] = ind2sub(size(S), iLin);
fprintf('  cosine similarity surface: %d x %d (pitch x time)\n', size(S, 1), size(S, 2));
fprintf('  max similarity %.4f at pitch shift %.0f c, time shift %.2f s\n', ...
        sMax, pitchGrid(iP), timeGrid(iT));
fprintf('  (expected: 0 cents, 3.00 s --- the embedded C-E-G)\n\n');

%% ===================================================================
%  3. Adding a window in time
%  ===================================================================

fprintf('=== 3. The same call with a window in time ===\n');

% Without a window the query is compared with the whole reference, so
% even at the match the reference's other four notes lower the cosine.
% A window on time, named in 'contextWindow', travels with the query
% (centred on its position, the offset plus its mean onset) and weights
% the reference's events by their distance from it, so the comparison
% is local. A rectangle 3 s wide spans the query's three notes; at the
% match it keeps exactly the embedded C-E-G.
SWin = windowedSimilarity(pmRef, pmQry, [], ...
    'offsets', {1, pitchGrid; 2, timeGrid}, ...
    'contextWindow', {2, struct('shape', 'rect', 'width', 3)}, ...
    'normalize', 'cosine');

[sMaxW, iLinW] = max(SWin(:));
[iPW, iTW] = ind2sub(size(SWin), iLinW);
fprintf('  max similarity %.4f at pitch shift %.0f c, time shift %.2f s\n', ...
        sMaxW, pitchGrid(iPW), timeGrid(iTW));
fprintf('  (%.4f without the window, where the reference''s other\n', sMax);
fprintf('   four notes dilute the match)\n\n');

%% ===================================================================
%  4. The two profiles
%  ===================================================================

fprintf('=== 4. Plot ===\n');

figure('Name', 'demo\_translateSweep', 'Position', [100 100 1200 500], 'Color', 'w');
surfs = {S, SWin};
names = {'no window', 'time window 3 s wide'};
for k = 1:2
    subplot(1, 2, k);
    imagesc(pitchGrid, timeGrid, surfs{k}.', [0 1]);
    axis xy;
    xlabel('Pitch transposition (cents)');
    if k == 1, ylabel('Time shift (s)'); end
    title(sprintf('Cosine similarity, %s', names{k}));
    set(gca, 'XTick', 0:200:1200, 'YTick', -1:1:5);
    hold on;
    % The expected peaks: the query on the embedded C-E-G at (0 c, 3 s),
    % and again at (1200 c, 3 s) by octave periodicity.
    plot([0 1200], [3 3], 'rx', 'MarkerSize', 12, 'LineWidth', 1.5);
    hold off;
    colorbar;
end
sgtitle('Reference: D-E-F-C-E-G-A; query: C-E-G');
fprintf('  Red x marks the expected peaks at (0 c, 3 s) and (1200 c, 3 s).\n\n');

%% ===================================================================
%  5. What the call in 2 computes: sweepSimMaet
%  ===================================================================

fprintf('=== 5. sweepSimMaet (one pass, no translated copies) ===\n');
fprintf('  A uniform translation of the query enters the inner product only\n');
fprintf('  through the offset, so the whole profile is one pass over the\n');
fprintf('  tuple pairs and then one evaluation per offset. The pitch\n');
fprintf('  attribute is periodic, which the mixture route refuses; under\n');
fprintf('  ''method'', ''auto'' the orbit route carries the sweep instead\n');
fprintf('  (the wrapped kernel absorbs the periodicity).\n');

densRef = buildMaet(pmRef, 'verbose', false);
densQry = buildMaet(pmQry, 'verbose', false);
% One column per combination of offsets: the A x M form, pitch running
% fastest, as in the output of Section 2.
[Pm, Tm] = ndgrid(pitchGrid, timeGrid);
offsetsAM = [Pm(:).'; Tm(:).'];
SSweep = reshape(sweepSimMaet(densRef, densQry, offsetsAM, 'verbose', false), size(Pm));
dSweep = max(abs(S(:) - SSweep(:)));
fprintf('  max |S - SSweep| = %.2e\n\n', dSweep);
assert(dSweep < 1e-10, 'sweepSimMaet disagrees with windowedSimilarity.');

%% ===================================================================
%  6. Offset by offset, for transparency
%  ===================================================================

fprintf('=== 6. translateAttributes + simMaet, and an explicit loop ===\n');

% translateAttributes builds the M translated copies: its offsets are a
% 1-by-A cell whose entries are 1-by-M rows, one candidate shift per
% column, column m of every entry together defining the m-th copy.
% The result is one pre-MAET holding all M copies on one geometry.
pmSwept = translateAttributes(pmQry, {Pm(:).', Tm(:).'});
fprintf('  %d translated copies of the query\n', numel(pmSwept.pAttr));

% simMaet compares the reference with every copy in its list mode; it
% computes every copy with Bulger's method where the sweep in 5 took the
% orbit route, so the two agree to the truncation floor rather than to
% the last digit.
SList = reshape(cell2mat(simMaet(pmRef, pmSwept, 'verbose', false)), ...
                size(Pm));

% And one offset at a time: translate the query by one (pitch, time)
% pair, build it, and compare it with the reference.
SLoop = zeros(size(Pm));
for m = 1:numel(Pm)
    densQ = buildMaet(translateAttributes(pmQry, {Pm(m), Tm(m)}), ...
                      'verbose', false);
    SLoop(m) = simMaet(densRef, densQ, 'verbose', false);
end

dList = max(abs(S(:) - SList(:)));
dLoop = max(abs(SList(:) - SLoop(:)));
fprintf('  max |S - SList|     = %.2e\n', dList);
fprintf('  max |SList - SLoop| = %.2e\n', dLoop);
assert(dList < 1e-8 && dLoop < 1e-12, 'The offset-by-offset routes disagree.');

mptDefaults(prevDefaults);
fprintf('\n=== Demo complete ===\n');
