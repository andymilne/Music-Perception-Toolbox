%% test_score.m — readScore and eventsFromScore on the shared fixtures
%
%  tests/data holds a format-1 MIDI file and a MusicXML score (plain and
%  compressed) that the Python suite reads too; both assert the same
%  note tables. Mirror of Python tests/test_score.py.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults_sc
    cleanupDefaults_sc = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

sc_data = fullfile(fileparts(mfilename('fullpath')), 'data');

% The fixtures: a 3/4 MIDI file at 120 bpm switching to 60 bpm at beat 4,
% with a melody track and a chord track that uses running status and
% leaves two notes open; a MusicXML score with a soprano (tie, rest,
% grace note, dynamics) and a piano part (chord, backup into a second
% voice, tempo change).
sc_midi = struct( ...
    'onsetBeats',      [0 0 0 0 1 2 2 2 4], ...
    'onsetSeconds',    [0 0 0 0 0.5 1 1 1 2], ...
    'durationBeats',   [1 2 2 2 0.5 2 2 2 1], ...
    'durationSeconds', [0.5 1 1 1 0.25 1 1 1 1], ...
    'pitch',           [60 48 52 55 64 67 53 57 72], ...
    'velocity',        [96 100 100 100 80 112 100 100 64], ...
    'part',            [1 2 2 2 1 1 2 2 1], ...
    'channel',         [1 2 2 2 1 1 2 2 1], ...
    'measure',         [1 1 1 1 1 1 1 1 2]);
sc_xml = struct( ...
    'onsetBeats',      [0 0 0 0 0 1 1 2 2 3 4], ...
    'onsetSeconds',    [0 0 0 0 0 0.5 0.5 1 1 1.5 2.5], ...
    'durationBeats',   [1 1 2 2 2 0.5 1 2 1 3 2], ...
    'durationSeconds', [0.5 0.5 1 1 1 0.25 0.5 1.5 0.5 3 2], ...
    'pitch',           [60 36 48 52 55 64 43 67 41 53 70], ...
    'velocity',        [90 90 90 90 90 90 90 90 90 90 54], ...
    'part',            [1 2 2 2 2 1 2 1 2 2 1], ...
    'channel',         [1 2 1 1 1 1 2 1 2 1 1], ...
    'measure',         [1 1 1 1 1 1 1 1 1 2 2]);

t = readScore(fullfile(sc_data, 'score_small.mid'));
results{end+1, 1} = 'score: MIDI note table'; %#ok<*SAGROW>
results{end, 2}   = strcmp(t.source, 'midi') && isequal(t.partNames, {'Melody', 'Chords'}) ...
                    && scTableMatches(t, sc_midi);

for sc_name = {'score_small.musicxml', 'score_small.mxl'}
    t = readScore(fullfile(sc_data, sc_name{1}));
    results{end+1, 1} = sprintf('score: MusicXML note table (%s)', sc_name{1});
    results{end, 2}   = strcmp(t.source, 'musicxml') && isequal(t.partNames, {'Soprano', 'Piano'}) ...
                        && scTableMatches(t, sc_xml);
end

results{end+1, 1} = 'score: unknown extension errors';
results{end, 2}   = throwsErrorWithId(@() readScore('score.abc'), 'readScore:extension');

[p, w, sp] = eventsFromScore(fullfile(sc_data, 'score_small.mid'));
results{end+1, 1} = 'score: events bind chords by default';
results{end, 2}   = strcmp(sp{1}.name, 'pitch') && strcmp(sp{2}.name, 'onset') ...
                    && isequal(size(p{1}), [4 4]) && isequal(size(p{2}), [1 4]) ...
                    && isequal(p{1}(:, 1).', [60 48 52 55]) && all(isnan(p{1}(2:end, 2))) ...
                    && max(abs(p{2} - [0 0.5 1 2])) < 1e-12 ...
                    && max(abs(w{1}(:, 1).' - [96 100 100 100] / 127)) < 1e-12 ...
                    && all(w{1}(2:end, 2) == 0) && isequal(w{2}, ones(1, 4));

[p, w, sp] = eventsFromScore(fullfile(sc_data, 'score_small.musicxml'), ...
                             'attributes', {'pitch', 'onset', 'duration'}, 'pitch', 'cents', ...
                             'time', 'beats', 'weights', 'ones', 'chords', 'separate', 'parts', 2);
results{end+1, 1} = 'score: events options (cents, beats, ones, separate, parts)';
results{end, 2}   = isempty(w) && numel(sp) == 3 && strcmp(sp{3}.name, 'duration') ...
                    && all(cellfun(@(m) isequal(size(m), [1 7]), p)) ...
                    && max(abs(p{1} - [3600 4800 5200 5500 4300 4100 5300])) < 1e-9 ...
                    && max(abs(p{2} - [0 0 0 0 1 2 3])) < 1e-12 ...
                    && max(abs(p{3} - [1 2 2 2 1 1 3])) < 1e-12;
[~, w2, ~] = eventsFromScore(fullfile(sc_data, 'score_small.musicxml'), ...
                             'weights', 'duration', 'chords', 'separate', 'time', 'beats');
results{end+1, 1} = 'score: duration weights';
results{end, 2}   = max(abs(w2{1} - sc_xml.durationBeats)) < 1e-12;

t = readScore(fullfile(sc_data, 'score_small.mid'));
[p, ~, ~] = eventsFromScore(t, 'chords', 'bind', 'chordTolerance', 0.6, 'time', 'seconds', 'weights', 'ones');
results{end+1, 1} = 'score: events from a table with chord tolerance';
results{end, 2}   = isequal(size(p{1}), [5 3]);

sc_path = fullfile(sc_data, 'score_small.mid');
results{end+1, 1} = 'score: bad arguments';
results{end, 2}   = throwsErrorWithId(@() eventsFromScore(sc_path, 'attributes', {'pitch', 'colour'}), 'eventsFromScore:attribute') ...
                    && throwsErrorWithId(@() eventsFromScore(sc_path, 'time', 'ticks'), 'eventsFromScore:time') ...
                    && throwsErrorWithId(@() eventsFromScore(sc_path, 'chords', 'merge'), 'eventsFromScore:chords');

[p, w, sp] = eventsFromScore(sc_path, 'parts', 2);
sp{1}.r = 2;
d = buildExpTens(p, w, 'specs', sp, 'sigma', [1 0.2], 'isPer', [true false], ...
                 'period', [12 0], 'verbose', false);
results{end+1, 1} = 'score: carrier feeds the pipeline (pitch-class dyads of the chord track)';
results{end, 2}   = abs(cosSimExpTens(d, d, 'verbose', false) - 1) < 1e-9;

if standalone
    nPass = sum([results{:, 2}]);
    nFail = size(results, 1) - nPass;
    for ii = 1:size(results, 1)
        if results{ii, 2}
            fprintf('  PASS  %s\n', results{ii, 1});
        else
            fprintf('  FAIL  %s\n', results{ii, 1});
        end
    end
    fprintf('\n=== test_score: %d passed, %d failed (of %d) ===\n\n', nPass, nFail, nPass + nFail);
    clear cleanupDefaults_sc
    if nFail > 0
        error('test_score:failed', '%d test(s) failed.', nFail);
    end
end


function ok = scTableMatches(t, expect)
    ok = true;
    f = fieldnames(expect);
    for i = 1:numel(f)
        v = t.(f{i});
        ok = ok && numel(v) == numel(expect.(f{i})) ...
             && max(abs(v(:).' - expect.(f{i}))) < 1e-12;
    end
end
