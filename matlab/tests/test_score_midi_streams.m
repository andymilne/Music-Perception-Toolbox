%% test_score_midi_streams.m — sustain, sostenuto, pitch bend, loudness
%
%  readScore resolves the MIDI controller streams that change a note's own
%  columns: sustain and sostenuto into soundingDuration, pitch bend into
%  pitch, and channel volume and expression into weight. The fixtures here
%  are written by hand so that each rule is isolated.
%
%  Mirror of Python tests/test_score_midi_streams.py.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
else
    standalone = false;
end

ms_tpq = 480;
ms_q = ms_tpq;
ms_tol = 1e-9;

% --- sustain -------------------------------------------------------------
t = scRead({ {0, scCc(0, 64, 127)}, {0, scOn(0, 60)}, ...
             {ms_q, scOff(0, 60)}, {3 * ms_q, scCc(0, 64, 0)} }, 4 * ms_q, ms_tpq);
results(end+1, :) = {'midi streams: sustain extends only the sounding duration', ...
    abs(t.durationBeats(1) - 1) < ms_tol ...
    && abs(t.soundingDurationBeats(1) - 3) < ms_tol}; %#ok<*SAGROW>

t = scRead({ {0, scOn(0, 60)}, {ms_q, scOff(0, 60)}, ...
             {2 * ms_q, scCc(0, 64, 127)}, {3 * ms_q, scCc(0, 64, 0)} }, 4 * ms_q, ms_tpq);
results(end+1, :) = {'midi streams: a pedal pressed after the note-off does not hold it', ...
    abs(t.soundingDurationBeats(1) - 1) < ms_tol};

t = scRead({ {0, scCc(0, 64, 127)}, {0, scOn(0, 60)}, {ms_q, scOff(0, 60)} }, ...
           8 * ms_q, ms_tpq);
results(end+1, :) = {'midi streams: a pedal never released holds to the end', ...
    abs(t.soundingDurationBeats(1) - 8) < ms_tol};

t = scRead({ {0, scCc(0, 64, 63)}, {0, scOn(0, 60)}, {ms_q, scOff(0, 60)} }, ...
           8 * ms_q, ms_tpq);
results(end+1, :) = {'midi streams: half-pedalling counts as up', ...
    abs(t.soundingDurationBeats(1) - 1) < ms_tol};

% --- the re-strike rule --------------------------------------------------
t = scRead({ {0, scCc(0, 64, 127)}, {0, scOn(0, 60)}, {ms_q, scOff(0, 60)}, ...
             {2 * ms_q, scOn(0, 60)}, {3 * ms_q, scOff(0, 60)}, ...
             {4 * ms_q, scCc(0, 64, 0)} }, 4 * ms_q, ms_tpq);
results(end+1, :) = {'midi streams: a re-strike on the same channel damps the tail', ...
    abs(t.soundingDurationBeats(1) - 2) < ms_tol};

% Two channels may be two instruments, so the pitches overlap.
t = scRead({ {0, scCc(0, 64, 127)}, {0, scCc(1, 64, 127)}, {0, scOn(0, 60)}, ...
             {ms_q, scOff(0, 60)}, {2 * ms_q, scOn(1, 60)}, ...
             {3 * ms_q, scOff(1, 60)}, {4 * ms_q, scCc(0, 64, 0)}, ...
             {4 * ms_q, scCc(1, 64, 0)} }, 4 * ms_q, ms_tpq);
onCh1 = t(t.channel == 1, :);
results(end+1, :) = {'midi streams: a re-strike on another channel does not', ...
    abs(onCh1.soundingDurationBeats(1) - 4) < ms_tol};

% --- sostenuto -----------------------------------------------------------
t = scRead({ {0, scOn(0, 60)}, {ms_q / 2, scCc(0, 66, 127)}, ...
             {ms_q, scOff(0, 60)}, {ms_q, scOn(0, 64)}, ...
             {2 * ms_q, scOff(0, 64)}, {4 * ms_q, scCc(0, 66, 0)} }, ...
           4 * ms_q, ms_tpq);
held  = t(t.noteNumber == 60, :);
later = t(t.noteNumber == 64, :);
results(end+1, :) = {'midi streams: sostenuto holds only what was down when pressed', ...
    abs(held.soundingDurationBeats(1) - 4) < ms_tol ...
    && abs(later.soundingDurationBeats(1) - 1) < ms_tol};

% --- pitch bend ----------------------------------------------------------
t = scRead([scRpn(0, 0, 0, 2), { {0, scBend(0, 2048)}, {0, scOn(0, 60)}, ...
            {ms_q, scOff(0, 60)} }], ms_q, ms_tpq);
results(end+1, :) = {'midi streams: bend lands in pitch and leaves noteNumber alone', ...
    abs(t.pitch(1) - 60.5) < ms_tol && t.noteNumber(1) == 60};

t = scRead([scRpn(0, 0, 0, 12), { {0, scBend(0, 4096)}, {0, scOn(0, 60)}, ...
            {ms_q, scOff(0, 60)} }], ms_q, ms_tpq);
results(end+1, :) = {'midi streams: RPN 0 sets the bend range', ...
    abs(t.pitch(1) - 66) < ms_tol};

% An MPE Configuration Message on channel 1 opens a lower zone, whose
% member channels take the 48-semitone default.
t = scRead([scRpn(0, 0, 6, 3), { {0, scBend(1, round(8192 / 48))}, ...
            {0, scOn(1, 60)}, {ms_q, scOff(1, 60)} }], ms_q, ms_tpq);
results(end+1, :) = {'midi streams: an MPE zone gives members the 48-semitone default', ...
    abs(t.pitch(1) - 61) < 0.01};

msWarn = warning('off', 'readScore:bendRange');
msRestore = onCleanup(@() warning(msWarn));
lastwarn('');
scRead({ {0, scBend(0, 2048)}, {0, scOn(0, 60)}, {ms_q, scOff(0, 60)} }, ...
       ms_q, ms_tpq);
[~, msId] = lastwarn;
results(end+1, :) = {'midi streams: bend without a declared range warns', ...
    strcmp(msId, 'readScore:bendRange')};

% The first note of a channel may be preceded by nothing, so a bend sent
% just after it is read, provided no other note intervenes.
t = scRead({ {0, scOn(0, 60)}, {1, scBend(0, 4096)}, {ms_q, scOff(0, 60)} }, ...
           ms_q, ms_tpq);
results(end+1, :) = {'midi streams: a bend just after a channel''s first note-on applies', ...
    abs(t.pitch(1) - 61) < ms_tol};
clear msRestore

% --- loudness controllers ------------------------------------------------
t = scRead({ {0, scCc(0, 7, 64)}, {0, scCc(0, 11, 127)}, ...
             {0, scOn(0, 60, 100)}, {ms_q, scOff(0, 60)} }, ms_q, ms_tpq);
results(end+1, :) = {'midi streams: volume and expression fold into weight', ...
    abs(t.weight(1) - (100 / 127) * (64 / 127) ^ 2) < ms_tol ...
    && t.velocity(1) == 100};

t = scRead({ {0, scOn(0, 60, 64)}, {ms_q, scOff(0, 60)} }, ms_q, ms_tpq);
results(end+1, :) = {'midi streams: weight is velocity alone where no controller is sent', ...
    abs(t.weight(1) - 64 / 127) < ms_tol};

% --- program change ------------------------------------------------------
% Program change selects the instrument sound on a channel; the column
% carries whichever is in force when the note starts.
t = scRead({ {0, [192 40]}, {0, scOn(0, 60)}, {ms_q, scOff(0, 60)}, ...
             {ms_q, [192 73]}, {2 * ms_q, scOn(0, 62)}, ...
             {3 * ms_q, scOff(0, 62)} }, 3 * ms_q, ms_tpq);
results(end+1, :) = {'midi streams: the program in force at the onset', ...
    isequal(t.program(:).', [40 73])};

t = scRead({ {0, scOn(0, 60)}, {ms_q, scOff(0, 60)} }, ms_q, ms_tpq);
results(end+1, :) = {'midi streams: no program change reads as zero', ...
    t.program(1) == 0};

t = scRead({ {0, [192 40]}, {0, [193 73]}, {0, scOn(0, 60)}, ...
             {0, scOn(1, 72)}, {ms_q, scOff(0, 60)}, {ms_q, scOff(1, 72)} }, ...
           ms_q, ms_tpq);
results(end+1, :) = {'midi streams: program is per channel', ...
    isequal(t.program(t.channel == 1), 40) ...
    && isequal(t.program(t.channel == 2), 73)};

if standalone
    nPass = sum([results{:, 2}]);
    fprintf('\n=== test_score_midi_streams: %d passed, %d failed (of %d) ===\n', ...
            nPass, size(results, 1) - nPass, size(results, 1));
end


% --- a minimal Standard MIDI File writer ---------------------------------

function t = scRead(events, endTick, tpq)
    path = [tempname(), '.mid'];
    cleanup = onCleanup(@() delete(path)); %#ok<NASGU>
    body = scTrack(events, endTick);
    head = [double('MThd'), 0 0 0 6, 0 1, 0 1, ...
            floor(tpq / 256), mod(tpq, 256)];
    fid = fopen(path, 'w');
    fwrite(fid, [head, body], 'uint8');
    fclose(fid);
    t = readScore(path);
end

function bytes = scTrack(events, endTick)
    body = [];
    prev = 0;
    for i = 1:numel(events)
        tick = events{i}{1};
        body = [body, scVlq(tick - prev), events{i}{2}]; %#ok<AGROW>
        prev = tick;
    end
    body = [body, scVlq(max(0, endTick - prev)), 255, 47, 0];
    n = numel(body);
    bytes = [double('MTrk'), ...
             floor(n / 16777216), mod(floor(n / 65536), 256), ...
             mod(floor(n / 256), 256), mod(n, 256), body];
end

function out = scVlq(n)
    out = mod(n, 128);
    n = floor(n / 128);
    while n > 0
        out = [mod(n, 128) + 128, out]; %#ok<AGROW>
        n = floor(n / 128);
    end
end

function b = scOn(ch, note, vel)
    if nargin < 3; vel = 100; end
    b = [144 + ch, note, vel];
end

function b = scOff(ch, note)
    b = [128 + ch, note, 0];
end

function b = scCc(ch, number, value)
    b = [176 + ch, number, value];
end

function b = scBend(ch, value)
    raw = value + 8192;
    b = [224 + ch, mod(raw, 128), floor(raw / 128)];
end

function ev = scRpn(ch, msb, lsb, data)
    ev = { {0, scCc(ch, 101, msb)}, {0, scCc(ch, 100, lsb)}, ...
           {0, scCc(ch, 6, data)} };
end
