%% demo_sweptSimilarity.m
% sweptSimilarity in depth: what each setting finds, and why.
%
% One melody and one four-note query are used throughout. The melody
% holds four statements related to the query in different ways,
% separated by bars of filler:
%
%     bar 1   (1) the query exactly               C D E G
%     bar 3   (2) the query transposed a fifth    G A B D
%     bar 5   (3) the query's pitches, reordered  E G C D
%     bar 7   (4) the query's rhythm, other notes F A Bb F
%
% All four share the query's rhythm (a crotchet then two quavers); the
% filler bars are plain crotchets. Each section below asks the same
% question -- where does the query occur? -- with a different setting,
% and each setting finds a different subset of the four statements.
%
%    1. The inputs, as pre-MAETs.
%    2. Translation in time (the canonical call, 'sweep', a): finds (1).
%    3. Translation in pitch and time: finds (1) and (2), each in its key.
%    4. 'align' 'both': a local comparison; finds (1), scored against
%       only the music around it.
%    5. 'align' 'window', time dropped: pitch content without position;
%       finds (1) and (3).
%    6. A window on time, time dropped, and translation in pitch: which
%       transposition each bar holds; finds (1), (2), and (3), and shows
%       why cosine suits it.
%    7. 'align' 'window', time relative (bound onsets): rhythm without
%       position; finds (1) to (4).
%    8. 'align' 'window', time absolute: two voices compared in place.
%    9. 'align' 'independent': a correlogram of a drifting lag.
%   10. What the calls compute: sweepSimMaet in one pass, against
%       translated copies compared one by one.
%
% See also SWEPTSIMILARITY, SWEEPSIMMAET, TRANSLATEATTRIBUTES,
% BINDEVENTS, SIMMAET.
%
% The Python mirror is demo_swept_similarity.py.

clear; close all;

% Keep the dispatcher's per-call announcements out of the printed output
% (showHints gates only those); restored at the end.
prevDefaults = mptDefaults('showHints', false);

%% ===================================================================
%  1. The inputs
%  ===================================================================

fprintf('=== 1. Inputs ===\n');

% The query: C D E G in a crotchet-quaver-quaver rhythm, written from
% beat 0.
queryMidi = [60 62 64 67];
rhythm    = [0 1 1.5 2];

% Each statement: four notes in the query's rhythm from beat t0, and a
% closing F at t0 + 3. Each filler bar: plain crotchets, F A F A.
statement = @(m, t0) [m, 65; t0 + rhythm, t0 + 3];
filler    = @(t0) [65 69 65 69; t0 + (0:3)];
notes = [filler(0),  statement(queryMidi, 4), ...          % (1) bar 1
         filler(8),  statement([67 69 71 74], 12), ...     % (2) bar 3
         filler(16), statement([64 67 60 62], 20), ...     % (3) bar 5
         filler(24), statement([65 69 70 65], 28)];        % (4) bar 7
midi  = notes(1, :);
onset = notes(2, :);

% The melody is in 12-TET, so pitch is in semitones, as MIDI note
% numbers: periodic at the octave with sigma = 0.2 semitones. Onset (in
% beats) is absolute with sigma = 0.1 beats.
sigma = [0.2, 0.1];
pCtx  = {midi, onset};
specs = flatSpecs(pCtx, 'name', {'pitch', 'onset'}, 'sigma', sigma, ...
                  'isPer', [true, false], 'period', [12, 0]);
melody = packPreMaet(pCtx, [], specs);
query  = packPreMaet({queryMidi, rhythm}, [], specs);
showPreMaet(query);
fprintf(['  melody: %d notes in 8 bars; the statements start at ' ...
         'beats 4, 12, 20, 28\n\n'], numel(midi));

bars       = 0:7;
statements = [1 4; 3 12; 5 20; 7 28];     % bar, start beat

%% ===================================================================
%  2. Translation in time: the canonical call
%  ===================================================================

fprintf('=== 2. Translation in time (''sweep'', 2) ===\n');

% Naming the attribute alone asks for the default sweep values: every
% offset at which query and melody overlap, from -2 to 31 beats. The
% default step is the largest whole fraction of the spacing the notes
% lie on (half a beat) that is no more than half the width of the
% profile's peaks, sigma * sqrt(2) / 2 at tuple size 1: 0.0625 beats, an
% eighth of that spacing, so every exact match falls on the grid. The
% query is translated by each offset and compared with the whole melody,
% in pitch and onset jointly, in one pass through sweepSimMaet. The
% offsets come back as mu2, a cell with one entry per attribute, here
% mu2{2}; both sequences are written from beat 0, so an offset is the
% beat at which the query starts.
[S2, mu2] = sweptSimilarity(melody, query, 'sweep', 2);
fprintf('  %d offsets, %.2f to %.2f beats\n', numel(mu2{2}), ...
        mu2{2}(1), mu2{2}(end));
% The matches are the profile's local maxima (islocalmax) of at least 0.5.
for i = find(islocalmax(S2) & S2 >= 0.5)
    fprintf('  match at beat %5.2f: %.3f\n', mu2{2}(i), S2(i));
end
fprintf(['  Only (1): pitch is compared as it stands, so (2) is at ' ...
         'the wrong\n']);
fprintf('  pitch and (3) in the wrong order; (4) shares only the rhythm.\n\n');

%% ===================================================================
%  3. Translation in pitch and time
%  ===================================================================

fprintf('=== 3. Translation in pitch and time (''sweep'', [1 2]) ===\n');

% Sweeping pitch as well translates the query to every transposition at
% every time offset: a (pitch, time) surface. Pitch being periodic, the
% transpositions cover one octave; the notes lie on a semitone grid, so
% the step is an eighth of a semitone, the largest whole fraction no
% more than half the width of the peaks in pitch.
%
% The query is written from C4 (MIDI 60), so adding its root to each
% pitch offset gives the pitch the root lands on, naming each match's key.
[S3, mu3] = sweptSimilarity(melody, query, 'sweep', [1 2]);
fprintf('  surface %d transpositions x %d offsets\n', size(S3, 1), size(S3, 2));
maxS3 = max(S3, [], 1);
root = queryMidi(1);                                % C4, MIDI 60
keyNames = {'C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'};
for j = find(islocalmax(maxS3) & maxS3 >= 0.9)
    [~, i] = max(S3(:, j));
    key = keyNames{mod(round(root + mu3{1}(i)), 12) + 1};
    fprintf(['  match at beat %5.2f, transposed %2.0f semitones ' ...
             '(root %s): %.3f\n'], mu3{2}(j), mu3{1}(i), key, maxS3(j));
end
fprintf(['  (1) untransposed, in C, and (2) a fifth (7 semitones) up, ' ...
         'in G.\n\n']);

%% ===================================================================
%  4. 'align' 'both': a local comparison
%  ===================================================================

fprintf('=== 4. ''align'' ''both'': a local comparison ===\n');

% Under 'both' the query is translated as in section 2, and a window on
% the melody is aligned with it, so each placement compares the query
% with only the music around it. What this changes is the normalization.
% Cosine divides by the norms of both sides. Under 'query' the melody's
% norm is that of the whole melody, so even the exact statement scores
% only 1/3 (its four notes against the melody's 36, sqrt(4/36)), a value
% that falls as the piece grows. Under 'both' it is the norm of what the
% window keeps: a score of 1 means the window holds the query and nothing
% else, and unmatched notes nearby count against the match. The window's
% width sets how near is nearby.
%
% The window is aligned at the query's middle, the default of queryRef
% under 'both'. The offsets, mu4{2}, are still the beats at which the query
% starts, so the peaks read as in section 2. queryRef, the point of the
% query at which the window is aligned, matters mainly for a window that
% extends to one side only: 'exponentialBefore' aligned at the query's
% last onset ('queryRef', {2, 2}) weights the music leading up to where
% the query ends, the most recent most heavily. Each value then depends
% only on music heard by the time the query ends, so it is easily read:
% against the sweep values, the moments at which the query ends (mu + 2),
% the profile says how closely what has just been heard matches the
% query, as a listener could judge it at that moment.
[S4, mu4] = sweptSimilarity(melody, query, 'sweep', 2, ...
                            'normalize', 'cosine');
[s, i] = max(S4);
fprintf('  ''query'', whole melody  : %.3f at beat %.2f\n', s, mu4{2}(i));
for width = [3 4 6 8]
    [S4, mu4] = sweptSimilarity(melody, query, 'sweep', 2, ...
        'align', {2, 'both'}, 'window', {2, {'rect', 'width', width}}, ...
        'normalize', 'cosine');
    [s, i] = max(S4);
    fprintf('  ''both'', window %d beats : %.3f at beat %.2f\n', width, ...
            s, mu4{2}(i));
end
fprintf('  A 3-beat window keeps the statement alone; 4 beats take in its\n');
fprintf('  closing F, and 6 and 8 the filler either side.\n\n');

%% ===================================================================
%  5. 'align' 'window', time dropped: content without position
%  ===================================================================

fprintf('=== 5. ''align'' ''window'', onset dropped ===\n');

% The query is not translated. A window one bar wide (a half-open
% rectangle, so bars tile without overlap) steps through the melody, and
% onset is marginalized after the window has weighted the events: the
% query is compared with the pitches each bar contains, whatever their
% order or rhythm.
barCentres = 4 * bars + 2;
S5 = sweptSimilarity(melody, query, 'sweep', {2, barCentres}, ...
                     'align', {2, 'window'}, ...
                     'window', {2, {'rect', 'width', 4}}, ...
                     'drop', 2);
fprintf('  bar: %s\n', sprintf('%5d  ', bars));
fprintf('  sim: %s\n', sprintf('%5.2f  ', S5));
fprintf('  (1) and (3) hold the query''s pitches; (2) shares two of them,\n');
fprintf('  G and D, so it scores about half.\n\n');

%% ===================================================================
%  6. A window on time, translation in pitch: which transposition each
%     bar holds
%  ===================================================================

fprintf('=== 6. Window on onset (dropped), translation in pitch ===\n');

% The window can do what translation cannot: localize on an attribute
% that is not compared. Each bar is windowed on onset, onset is dropped,
% and the query is translated in pitch: every transposition in every bar,
% a (transposition, bar) surface. Within a bar only the pitches count, so
% this finds (1) and (3) untransposed and (2) at 7 semitones together.
% Nothing on the query side could do it: with onset dropped, the query
% has no position for a window to select.
%
% The normalization matters here. The default divides by the query's own
% self-overlap, so a bar that repeats two of a transposition's pitches
% scores as highly as one holding all four: every filler bar, F A F A,
% scores 1 at 5 semitones (F G A C), matching F and A twice each. Cosine
% divides by the windowed bar's own norm as well, so repetition no longer
% stands in for the missing pitches. That local denominator, the norm of
% what the window keeps, is also a reason to window the context on a
% translated attribute, where a window on the query would otherwise do
% much the same (sections 4 and 9).
transp = 0:11;
norms = {'oneSidedDenom', 'cosine'};
S6 = cell(1, 2);
for k = 1:2
    S6{k} = sweptSimilarity(melody, query, ...
        'sweep', {1, transp; 2, barCentres}, ...
        'align', {1, 'query'; 2, 'window'}, ...
        'window', {2, {'rect', 'width', 4}}, ...
        'drop', 2, 'normalize', norms{k});
end
fprintf(['  surface %d transpositions x %d bars; best transposition in ' ...
         'each bar:\n'], size(S6{2}, 1), size(S6{2}, 2));
fprintf('  bar          : %s\n', sprintf('%5d  ', bars));
rowNames = {'default     ', 'cosine      '};
for k = 1:2
    [bestS6, iBest6] = max(S6{k}, [], 1);
    fprintf('  %s : %s\n', rowNames{k}, sprintf('%5.0f  ', transp(iBest6)));
    fprintf('                 %s\n', sprintf('%5.2f  ', bestS6));
end
fprintf('  Under the default every bar scores 1 somewhere; under cosine the\n');
fprintf('  statements (about 0.89; their closing F, absent from the query,\n');
fprintf('  costs the rest) stand clear of the fillers (0.71).\n\n');

%% ===================================================================
%  7. 'align' 'window', time relative: rhythm without position
%  ===================================================================

fprintf('=== 7. ''align'' ''window'', onset relative (bound onsets) ===\n');

% Relative mode compares each tuple only up to a common translation:
% through its values relative to the lowest. For onsets, a tuple needs
% several of them, so consecutive onsets are bound into 4-note
% super-events (bindEvents) with the outer level relative ('relOuter'):
% each super-event is then compared by its onsets measured from its
% first, its rhythm. Pitch is left out here, to ask about rhythm alone.
tSpecs = flatSpecs({onset}, 'name', {'onset'}, 'sigma', 0.1, ...
                   'isPer', false, 'period', 0);
boundMel = bindEvents(packPreMaet({onset}, [], tSpecs), 4, 'relOuter', true);
boundQry = bindEvents(packPreMaet({rhythm}, [], tSpecs), 4, 'relOuter', true);
showPreMaet(boundQry);

% The window still weights each super-event by onset time, the value the
% pre-MAET holds (its four onsets reduced to one, their mean, the default
% of 'locate'), before the density is built; the comparison then uses
% only the spacing. Translation would change nothing on a relative
% attribute, so 'window' is the role.
S7 = sweptSimilarity(boundMel, boundQry, 'sweep', {1, barCentres}, ...
                     'align', {1, 'window'}, ...
                     'window', {1, {'rect', 'width', 4}});
fprintf('  bar: %s\n', sprintf('%5d  ', bars));
fprintf('  sim: %s\n', sprintf('%5.2f  ', S7));
fprintf('  Every statement has the query''s rhythm, so all four are found;\n');
fprintf('  the filler bars, in plain crotchets, are not.\n\n');

%% ===================================================================
%  8. 'align' 'window', time absolute: in place
%  ===================================================================

fprintf(['=== 8. ''align'' ''window'', onset absolute: two voices ' ...
         'in place ===\n']);

% A second voice doubles the melody for four bars, then moves a major
% third above it. Comparing the two in place -- the second voice as the
% query, not translated, onset compared as it stands -- with windows that
% tile the piece shows where their similarity comes from. Under the
% default normalization the profile is linear in the window, so the
% bars' contributions sum to the similarity of the whole voices.
midiB = midi;
midiB(onset >= 16) = midiB(onset >= 16) + 4;
voiceB = packPreMaet({midiB, onset}, [], specs);
S8 = sweptSimilarity(melody, voiceB, 'sweep', {2, barCentres}, ...
                     'align', {2, 'window'}, ...
                     'window', {2, {'rect', 'width', 4}});
whole = simMaet(melody, voiceB, 'normalize', 'oneSidedDenom', ...
                'verbose', false);
fprintf('  bar: %s\n', sprintf('%5d  ', bars));
fprintf('  sim: %s\n', sprintf('%5.3f  ', S8));
fprintf('  sum over bars %.6f; whole voices %.6f\n\n', sum(S8), whole);
assert(abs(sum(S8) - whole) < 1e-10);

%% ===================================================================
%  9. 'align' 'independent': a correlogram
%  ===================================================================

fprintf('=== 9. ''align'' ''independent'': a drifting lag ===\n');

% Two parts play the same four-note ostinato, the second slightly faster,
% so it runs ever further ahead of the first (phasing). A window steps
% through the first part while the second is translated through a range
% of lags at each window position: every combination, a correlogram.
% queryRef 0 makes the query's sweep values the lags themselves. The
% window here is far narrower than the query, the whole second part, so
% each window position resolves the lag locally. Windowing the second
% part instead, around the same region, and translating it would give
% nearly the same correlogram: on a translated attribute, a window on the
% context and one on the query differ only in whether a near miss is
% weighted where the context's event lies or where the query's does.
ostMidi = [60 64 67 71];
nRep = 16;
tA = 0.5 * (0:4 * nRep - 1);
mA = repmat(ostMidi, 1, nRep);
tB = tA * 0.985;                        % 1.5% faster
partA = packPreMaet({mA, tA}, [], specs);
partB = packPreMaet({mA, tB}, [], specs);
winPos = 2:2:30;
lags = -0.2:0.01:0.8;
S9 = sweptSimilarity(partA, partB, 'sweep', {2, {winPos, lags}}, ...
                     'align', {2, 'independent'}, ...
                     'window', {2, {'rect', 'width', 4}}, 'queryRef', {2, 0});
[~, iBest] = max(S9, [], 2);
bestLag = lags(iBest);
fprintf('  correlogram %d window positions x %d lags\n', ...
        size(S9, 1), size(S9, 2));
fprintf('  best lag rises with position, as the second part runs ahead:\n');
fprintf('  window: %s\n', sprintf('%5.1f ', winPos(1:3:end)));
fprintf('  lag   : %s\n', sprintf('%5.2f ', bestLag(1:3:end)));
fprintf('  (the drift is 1.5%% of the time: %.2f to %.2f beats)\n\n', ...
        0.015 * winPos(1), 0.015 * winPos(end));

%% ===================================================================
%  10. What the calls compute
%  ===================================================================

fprintf('=== 10. sweepSimMaet in one pass, against copies one by one ===\n');

% Section 3 translated the query to every (pitch, time) pair. Translating
% every value of an attribute by the same amount changes the inner
% product only through that amount, so sweepSimMaet computes all the
% offsets from the two densities in one pass, with no translated copy of
% the query built. The same profile, on a coarser grid, three ways; the
% second and third are given sweptSimilarity's default normalization,
% 'oneSidedDenom', explicitly, since theirs is 'cosine'.
pGrid = 0:11;
tGrid = -2:0.5:31;
tic;
SWs = sweptSimilarity(melody, query, 'sweep', {1, pGrid; 2, tGrid});
tWs = toc;

[PP, TT] = ndgrid(pGrid, tGrid);
tic;
densMel = buildMaet(melody, 'verbose', false);
densQry = buildMaet(query, 'verbose', false);
SSw = reshape(sweepSimMaet(densMel, densQry, [PP(:).'; TT(:).'], ...
              'normalize', 'oneSidedDenom', 'verbose', false), size(PP));
tSw = toc;

tic;
SLoop = zeros(size(PP));
for m = 1:numel(PP)
    qM = translateAttributes(query, {PP(m), TT(m)});
    SLoop(m) = simMaet(densMel, buildMaet(qM, 'verbose', false), ...
                       'normalize', 'oneSidedDenom', 'verbose', false);
end
tLoop = toc;

fprintf('  %d offsets\n', numel(PP));
fprintf('  sweptSimilarity : %7.3f s\n', tWs);
fprintf('  sweepSimMaet       : %7.3f s   max diff %.1e\n', tSw, ...
        max(abs(SSw(:) - SWs(:))));
fprintf('  copy by copy       : %7.3f s   max diff %.1e\n', tLoop, ...
        max(abs(SLoop(:) - SWs(:))));
assert(max(abs(SSw(:) - SWs(:))) < 1e-10);
assert(max(abs(SLoop(:) - SWs(:))) < 1e-8);
fprintf(['  Where a window changes with each sweep value (''both'', ' ...
         '''window''),\n']);
fprintf('  there is no shared context to reuse, and each sweep value is one\n');
fprintf('  simMaet call.\n\n');

%% ===================================================================
%  Figures
%  ===================================================================

fprintf('=== Figures ===\n');
noteNames = {'C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'};
colours = lines(5);
labels = {'(1) exact', '(2) transposed', '(3) reordered', '(4) rhythm only'};

% Figure 1: the melody, the statements coloured, and what sections 2, 3,
% 5, and 7 each find, on one time axis.
figure('Name', 'demo\_sweptSimilarity: statements', ...
       'Position', [100 100 900 900], 'Color', 'w');
tl = tiledlayout(6, 1, 'TileSpacing', 'compact');
ax = nexttile(tl, [2 1]);
scatter(onset, midi, 40, [0.5 0.5 0.5]);
hold on;
for k = 1:4
    t0 = statements(k, 2);
    sel = onset >= t0 & onset < t0 + 2.5;
    scatter(onset(sel), midi(sel), 40, colours(k + 1, :), 'filled');
    text(t0, 77, labels{k}, 'Color', colours(k + 1, :), 'FontSize', 8);
end
for n = 1:numel(midi)
    text(onset(n), midi(n) + 1, noteNames{mod(midi(n), 12) + 1}, ...
         'HorizontalAlignment', 'center', 'FontSize', 6);
end
hold off;
ylim([57 80]);
ylabel('MIDI pitch');
title('The melody: four statements related to the query C D E G');
axs = gobjects(4, 1);
axs(1) = nexttile(tl); plot(mu2{2}, S2); ylabel('2. time');
axs(2) = nexttile(tl); plot(mu3{2}, maxS3); ylabel({'3. pitch', '+ time'});
axs(3) = nexttile(tl); bar(barCentres, S5, 0.9, 'FaceColor', [0.6 0.6 0.6]);
ylabel('5. content');
axs(4) = nexttile(tl); bar(barCentres, S7, 0.9, 'FaceColor', [0.6 0.6 0.6]);
ylabel('7. rhythm');
xlabel('beat (offset of the query, or bar)');
linkaxes([ax; axs], 'x');
for a = axs.'
    xline(a, statements(:, 2), 'Color', [0.85 0.85 0.85]);
end

% Figure 2: the (pitch, time) surface of section 3.
figure('Name', 'demo\_sweptSimilarity: surface', ...
       'Position', [120 120 1000 400], 'Color', 'w');
imagesc(mu3{2}, mu3{1}, S3);
axis xy;
xlabel('time offset (beats)');
ylabel('transposition (semitones)');
title('3. Translation in pitch and time: (1) at (4, 0), (2) at (12, 7)');
colorbar;

% Figure 3: the (transposition, bar) surfaces of section 6, under the
% default normalization and under cosine.
figure('Name', 'demo\_sweptSimilarity: transposition in each bar', ...
       'Position', [140 140 1200 400], 'Color', 'w');
titles6 = {'default', 'cosine'};
for k = 1:2
    subplot(1, 2, k);
    imagesc(bars, transp, S6{k}, [0 1]);
    axis xy;
    xlabel('bar');
    if k == 1, ylabel('transposition (semitones)'); end
    title(sprintf('6. Transposition in each bar: %s', titles6{k}));
    colorbar;
end

% Figure 4: the two voices' similarity, bar by bar (section 8), and the
% correlogram of section 9.
figure('Name', 'demo\_sweptSimilarity: in place and independent', ...
       'Position', [160 160 1200 400], 'Color', 'w');
subplot(1, 2, 1);
bar(bars, S8, 'FaceColor', [0.6 0.6 0.6]);
xlabel('bar');
ylabel('contribution to the similarity');
title(sprintf('8. In place: bars sum to %.3f', whole));
subplot(1, 2, 2);
imagesc(winPos, lags, S9.');
axis xy;
hold on;
plot(winPos, bestLag, 'w.-');
hold off;
xlabel('window position (beats)');
ylabel('lag of the second part (beats)');
title('9. ''align'' ''independent'': the best lag drifts');
colorbar;

mptDefaults(prevDefaults);
fprintf('  four figures drawn\n');
fprintf('\n=== Demo complete ===\n');
