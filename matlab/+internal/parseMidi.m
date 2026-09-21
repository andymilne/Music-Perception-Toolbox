function raw = parseMidi(path)
%PARSEMIDI  Standard MIDI File (format 0 or 1) to note rows.
%
%   raw = internal.parseMidi(path) returns a struct with .rows (M x 13),
%   .columns (1 x 13 cellstr naming them), .partNames (1 x P cell), and
%   .source = 'midi'. Sustain and sostenuto are resolved into the
%   sounding durations, pitch bend into pitch, and channel volume and
%   expression into weight. Twin of the Python mpt.score._parse_midi; see
%   readScore for the conventions.

    fid = fopen(path, 'r', 'ieee-be');
    if fid < 0
        error('readScore:open', 'Cannot open %s.', path);
    end
    data = fread(fid, inf, 'uint8=>double').';
    fclose(fid);
    if numel(data) < 14 || ~isequal(char(data(1:4)), 'MThd')
        error('readScore:midi', 'Not a Standard MIDI File (missing MThd header).');
    end
    hlen = localU32(data, 5);
    fmt = localU16(data, 9);
    ntrk = localU16(data, 11);
    division = localU16(data, 13);
    if division >= 32768
        error('readScore:midi', ['SMPTE time division is not supported; use ' ...
                                 'a file with ticks-per-quarter-note timing.']);
    end
    tpq = division;
    if fmt ~= 0 && fmt ~= 1
        error('readScore:midi', 'MIDI format %d is not supported (0 or 1).', fmt);
    end
    pos = 9 + hlen;
    tracks = cell(1, ntrk);
    for t = 1:ntrk
        if ~isequal(char(data(pos:pos + 3)), 'MTrk')
            error('readScore:midi', 'Malformed MIDI file (missing MTrk chunk).');
        end
        len = localU32(data, pos + 4);
        tracks{t} = data(pos + 8:pos + 7 + len);
        pos = pos + 8 + len;
    end

    % Pass 1: tempo and time-signature maps from every track.
    tempoMap = zeros(0, 2);      % tick, microseconds per quarter
    sigMap = zeros(0, 2);        % tick, quarter notes per bar
    trackEvents = cell(1, ntrk);
    trackEnd = zeros(1, ntrk);
    trackNames = cell(1, ntrk);
    ctrlAll = zeros(0, 5);       % tick, channel, kind, d1, d2
    fileEnd = 0;
    for t = 1:ntrk
        [ev, ctrl, tempos, sigs, name, endTick] = localTrackEvents(tracks{t});
        trackEvents{t} = ev;
        trackEnd(t) = endTick;
        tempoMap = [tempoMap; tempos]; %#ok<AGROW>
        sigMap = [sigMap; sigs]; %#ok<AGROW>
        trackNames{t} = name;
        ctrlAll = [ctrlAll; ctrl]; %#ok<AGROW>
        fileEnd = max(fileEnd, endTick);
    end
    % A channel is a property of the file, not of a track, so controller
    % state is gathered across tracks and read per channel.
    if ~isempty(ctrlAll)
        ctrlAll = sortrows(ctrlAll, 1);
    end
    streams = localControllerStreams(ctrlAll);
    tempoMap = sortrows(tempoMap, 1);
    sigMap = sortrows(sigMap, 1);
    if isempty(tempoMap) || tempoMap(1, 1) > 0
        tempoMap = [0 500000; tempoMap];
    end
    if isempty(sigMap) || sigMap(1, 1) > 0
        sigMap = [0 4; sigMap];
    end

    % Every note-on tick per (channel, note number), for the re-strike
    % rule that truncates a pedal-sustained tail, and per channel, for
    % the pitch-bend fallback.
    strikes = cell(1, 16 * 128);
    channelOns = cell(1, 16);
    for t = 1:ntrk
        ev = trackEvents{t};
        for i = 1:size(ev, 1)
            if bitand(ev(i, 2), 240) == 144 && ev(i, 4) > 0
                ch = bitand(ev(i, 2), 15);
                key = ch * 128 + ev(i, 3) + 1;
                strikes{key}(end + 1) = ev(i, 1); %#ok<AGROW>
                channelOns{ch + 1}(end + 1) = ev(i, 1); %#ok<AGROW>
            end
        end
    end
    for k = 1:numel(strikes)
        if ~isempty(strikes{k}); strikes{k} = sort(strikes{k}); end
    end
    for k = 1:16
        if ~isempty(channelOns{k}); channelOns{k} = sort(channelOns{k}); end
    end

    rows = zeros(0, 13);
    partNames = {};
    partIndex = 0;
    for t = 1:ntrk
        ev = trackEvents{t};          % tick status d1 d2
        openKeys = zeros(0, 2);       % channel, pitch
        openVals = zeros(0, 2);       % tick, velocity
        trackRows = zeros(0, 5);      % t0 t1 pitch vel ch
        for i = 1:size(ev, 1)
            tick = ev(i, 1); status = ev(i, 2); d1 = ev(i, 3); d2 = ev(i, 4);
            kind = bitand(status, 240);
            ch = bitand(status, 15);
            isOn = kind == 144 && d2 > 0;
            isOff = kind == 128 || (kind == 144 && d2 == 0);
            if isOn || isOff
                k = find(openKeys(:, 1) == ch & openKeys(:, 2) == d1, 1);
                if ~isempty(k)
                    trackRows(end + 1, :) = [openVals(k, 1), tick, d1, openVals(k, 2), ch]; %#ok<AGROW>
                    openKeys(k, :) = [];
                    openVals(k, :) = [];
                end
                if isOn
                    openKeys(end + 1, :) = [ch, d1]; %#ok<AGROW>
                    openVals(end + 1, :) = [tick, d2]; %#ok<AGROW>
                end
            end
        end
        for k = 1:size(openKeys, 1)
            trackRows(end + 1, :) = [openVals(k, 1), max(trackEnd(t), openVals(k, 1)), ...
                                     openKeys(k, 2), openVals(k, 2), openKeys(k, 1)]; %#ok<AGROW>
        end
        if isempty(trackRows)
            continue;
        end
        partIndex = partIndex + 1;
        if isempty(trackNames{t})
            partNames{end + 1} = sprintf('track %d', t); %#ok<AGROW>
        else
            partNames{end + 1} = trackNames{t}; %#ok<AGROW>
        end
        for i = 1:size(trackRows, 1)
            t0 = trackRows(i, 1); t1 = trackRows(i, 2);
            note = trackRows(i, 3); vel = trackRows(i, 4);
            ch = trackRows(i, 5);
            s0 = localSecondsAt(t0, tempoMap, tpq);
            s1 = localSecondsAt(t1, tempoMap, tpq);
            tEnd = localSoundingEnd(t0, t1, ch, note, streams, strikes, fileEnd);
            sEnd = localSecondsAt(tEnd, tempoMap, tpq);
            bend = localBendSemitones(ch, t0, streams, channelOns{ch + 1});
            vol = localStateAt(streams.volume{ch + 1}, t0, 127);
            expr = localStateAt(streams.expression{ch + 1}, t0, 127);
            rows(end + 1, :) = [t0 / tpq, s0, (t1 - t0) / tpq, s1 - s0, ...
                                (tEnd - t0) / tpq, sEnd - s0, ...
                                note + bend, note, vel, ...
                                (vel / 127) * (vol / 127) ^ 2 * (expr / 127) ^ 2, ...
                                partIndex, ch + 1, ...
                                localMeasureAt(t0, sigMap, tpq)]; %#ok<AGROW>
        end
    end
    raw = struct('rows', rows, 'columns', {internal.midiColumns()}, ...
                 'partNames', {partNames}, 'source', 'midi');
end


function v = localU32(data, i)
    v = data(i) * 16777216 + data(i + 1) * 65536 + data(i + 2) * 256 + data(i + 3);
end

function v = localU16(data, i)
    v = data(i) * 256 + data(i + 1);
end

function [value, pos] = localVarlen(data, pos)
    value = 0;
    while true
        b = data(pos);
        pos = pos + 1;
        value = value * 128 + bitand(b, 127);
        if b < 128
            return;
        end
    end
end

function s = localSecondsAt(tick, tempoMap, tpq)
    s = 0;
    prevTick = tempoMap(1, 1); prevUs = tempoMap(1, 2);
    for i = 2:size(tempoMap, 1)
        if tempoMap(i, 1) >= tick
            break;
        end
        s = s + (tempoMap(i, 1) - prevTick) / tpq * prevUs * 1e-6;
        prevTick = tempoMap(i, 1); prevUs = tempoMap(i, 2);
    end
    s = s + (tick - prevTick) / tpq * prevUs * 1e-6;
end

function m = localMeasureAt(tick, sigMap, tpq)
    m = 1;
    prevTick = sigMap(1, 1); prevQ = sigMap(1, 2);
    for i = 2:size(sigMap, 1)
        if sigMap(i, 1) >= tick
            break;
        end
        m = m + floor((sigMap(i, 1) - prevTick) / tpq / prevQ);
        prevTick = sigMap(i, 1); prevQ = sigMap(i, 2);
    end
    m = m + floor((tick - prevTick) / tpq / prevQ);
end

function [events, ctrl, tempos, sigs, name, endTick] = localTrackEvents(tdata)
    pos = 1;
    tick = 0;
    status = -1;
    events = zeros(0, 4);
    ctrl = zeros(0, 5);
    tempos = zeros(0, 2);
    sigs = zeros(0, 2);
    name = '';
    n = numel(tdata);
    while pos <= n
        [delta, pos] = localVarlen(tdata, pos);
        tick = tick + delta;
        b = tdata(pos);
        if b == 255                                   % meta
            mtype = tdata(pos + 1);
            [len, pos2] = localVarlen(tdata, pos + 2);
            payload = tdata(pos2:pos2 + len - 1);
            pos = pos2 + len;
            if mtype == 81 && len == 3
                tempos(end + 1, :) = [tick, payload(1) * 65536 + payload(2) * 256 + payload(3)]; %#ok<AGROW>
            elseif mtype == 88 && len >= 2
                sigs(end + 1, :) = [tick, payload(1) * 4 / 2 ^ payload(2)]; %#ok<AGROW>
            elseif mtype == 3 && isempty(name)
                name = strtrim(char(payload(payload ~= 0)));
            elseif mtype == 47
                endTick = tick;
                return;
            end
            continue;
        end
        if b == 240 || b == 247                       % sysex
            [len, pos2] = localVarlen(tdata, pos + 1);
            pos = pos2 + len;
            continue;
        end
        if b >= 128
            status = b;
            pos = pos + 1;
        end
        if status < 0
            error('readScore:midi', 'Malformed MIDI track (data byte before status).');
        end
        kind = bitand(status, 240);
        if kind == 192 || kind == 208
            d1 = tdata(pos); d2 = 0;
            pos = pos + 1;
        else
            d1 = tdata(pos); d2 = tdata(pos + 1);
            pos = pos + 2;
        end
        if kind == 128 || kind == 144
            events(end + 1, :) = [tick, status, d1, d2]; %#ok<AGROW>
        elseif kind == 176 || kind == 224
            ctrl(end + 1, :) = [tick, bitand(status, 15), kind, d1, d2]; %#ok<AGROW>
        end
    end
    endTick = tick;
end


function streams = localControllerStreams(ctrl)
    % Per-channel controller state, as step functions of tick. A value
    % holds until the next message on that channel, so each stream is an
    % n x 2 [tick value] list of changes read back by localStateAt.
    names = {'volume', 'expression', 'sustain', 'sostenuto', 'bend'};
    for i = 1:numel(names)
        streams.(names{i}) = repmat({zeros(0, 2)}, 1, 16);
    end
    bendRange = nan(1, 16);
    rpn = repmat([127 127], 16, 1);
    mpeMember = false(1, 16);
    sawBend = false;
    sawRange = false;

    for i = 1:size(ctrl, 1)
        tick = ctrl(i, 1); c = ctrl(i, 2) + 1;
        kind = ctrl(i, 3); d1 = ctrl(i, 4); d2 = ctrl(i, 5);
        if kind == 224
            streams.bend{c}(end + 1, :) = [tick, d2 * 128 + d1 - 8192]; %#ok<AGROW>
            sawBend = true;
        elseif d1 == 7
            streams.volume{c}(end + 1, :) = [tick, d2]; %#ok<AGROW>
        elseif d1 == 11
            streams.expression{c}(end + 1, :) = [tick, d2]; %#ok<AGROW>
        elseif d1 == 64
            streams.sustain{c}(end + 1, :) = [tick, d2 >= 64]; %#ok<AGROW>
        elseif d1 == 66
            streams.sostenuto{c}(end + 1, :) = [tick, d2 >= 64]; %#ok<AGROW>
        elseif d1 == 101
            rpn(c, 1) = d2;
        elseif d1 == 100
            rpn(c, 2) = d2;
        elseif d1 == 6
            if isequal(rpn(c, :), [0 0])
                bendRange(c) = d2;
                sawRange = true;
            elseif isequal(rpn(c, :), [0 6]) && any(c == [1 16]) && d2 > 0
                % An MPE Configuration Message: channel 1 opens a lower
                % zone, channel 16 an upper zone, over d2 member channels.
                if c == 1
                    members = 2:(d2 + 1);
                else
                    members = (16 - d2):15;
                end
                members = members(members >= 1 & members <= 16);
                mpeMember(members) = true;
                sawRange = true;
            end
        elseif d1 == 38 && isequal(rpn(c, :), [0 0])
            if isnan(bendRange(c)); bendRange(c) = 0; end
            bendRange(c) = bendRange(c) + d2 / 100;
        end
    end

    if sawBend && ~sawRange
        warning('readScore:bendRange', ...
                ['This file bends pitch but never sets a pitch-bend range ' ...
                 '(RPN 0) or an MPE zone, so the 2-semitone default is ' ...
                 'assumed; a file tuned for the 48-semitone MPE range will ' ...
                 'read 24 times too flat or sharp.']);
    end

    unset = isnan(bendRange);
    bendRange(unset & mpeMember) = 48;
    bendRange(unset & ~mpeMember) = 2;
    streams.bendRange = bendRange;
end


function v = localStateAt(changes, tick, default)
    % The value in force at tick: the last change at or before it.
    if isempty(changes)
        v = default;
        return;
    end
    k = find(changes(:, 1) <= tick, 1, 'last');
    if isempty(k)
        v = default;
    else
        v = changes(k, 2);
    end
end


function t = localNextRelease(changes, tick, endTick)
    % The first tick at or after tick where the pedal comes up.
    k = find(changes(:, 1) >= tick & changes(:, 2) == 0, 1);
    if isempty(k)
        t = endTick;
    else
        t = changes(k, 1);
    end
end


function tEnd = localSoundingEnd(t0, t1, ch, note, streams, strikes, endTick)
    % When the note stops sounding, given the pedals and the re-strikes.
    tEnd = t1;
    sustain = streams.sustain{ch + 1};
    if localStateAt(sustain, t1, 0)
        tEnd = max(tEnd, localNextRelease(sustain, t1, endTick));
    end
    % Sostenuto holds only what was already down when it was pressed.
    sos = streams.sostenuto{ch + 1};
    for i = 1:size(sos, 1)
        if sos(i, 2) && sos(i, 1) >= t0 && sos(i, 1) < t1 ...
                && localStateAt(sos, t1, 0)
            tEnd = max(tEnd, localNextRelease(sos, t1, endTick));
            break;
        end
    end
    % The same pitch struck again on the same channel damps what is left
    % of this one; a different channel may be a different instrument, so
    % it does not.
    later = strikes{ch * 128 + note + 1};
    k = find(later >= t1, 1);
    if ~isempty(k) && later(k) < tEnd
        tEnd = later(k);
    end
end


function semis = localBendSemitones(ch, t0, streams, channelOns)
    % The bend in force at a note's onset, in semitones. An exporter may
    % send a note's bend just before its note-on, or at the same tick,
    % and within a tick the ordering carries no meaning; both are covered
    % by reading the last bend at or before the onset tick. Where a
    % channel has no bend before its first note, the first bend after
    % that note is used, provided no further note-on intervenes.
    changes = streams.bend{ch + 1};
    if isempty(changes)
        semis = 0;
        return;
    end
    k = find(changes(:, 1) <= t0, 1, 'last');
    if ~isempty(k)
        raw = changes(k, 2);
    else
        nxt = channelOns(channelOns > t0);
        if ~isempty(nxt) && changes(1, 1) >= nxt(1)
            semis = 0;
            return;
        end
        raw = changes(1, 2);
    end
    semis = raw / 8192 * streams.bendRange(ch + 1);
end
