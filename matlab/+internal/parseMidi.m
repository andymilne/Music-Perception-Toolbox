function raw = parseMidi(path)
%PARSEMIDI  Standard MIDI File (format 0 or 1) to note rows.
%
%   raw = internal.parseMidi(path) returns a struct with .rows (M x 9:
%   onsetBeats onsetSeconds durationBeats durationSeconds pitch velocity
%   part channel measure), .partNames (1 x P cell), .source = 'midi'.
%   Twin of the Python mpt.score._parse_midi; see readScore for the
%   conventions.

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
    for t = 1:ntrk
        [ev, tempos, sigs, name, endTick] = localTrackEvents(tracks{t});
        trackEvents{t} = ev;
        trackEnd(t) = endTick;
        tempoMap = [tempoMap; tempos]; %#ok<AGROW>
        sigMap = [sigMap; sigs]; %#ok<AGROW>
        trackNames{t} = name;
    end
    tempoMap = sortrows(tempoMap, 1);
    sigMap = sortrows(sigMap, 1);
    if isempty(tempoMap) || tempoMap(1, 1) > 0
        tempoMap = [0 500000; tempoMap];
    end
    if isempty(sigMap) || sigMap(1, 1) > 0
        sigMap = [0 4; sigMap];
    end

    rows = zeros(0, 9);
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
            s0 = localSecondsAt(t0, tempoMap, tpq);
            s1 = localSecondsAt(t1, tempoMap, tpq);
            rows(end + 1, :) = [t0 / tpq, s0, (t1 - t0) / tpq, s1 - s0, ...
                                trackRows(i, 3), trackRows(i, 4), partIndex, ...
                                trackRows(i, 5) + 1, localMeasureAt(t0, sigMap, tpq)]; %#ok<AGROW>
        end
    end
    raw = struct('rows', rows, 'partNames', {partNames}, 'source', 'midi');
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

function [events, tempos, sigs, name, endTick] = localTrackEvents(tdata)
    pos = 1;
    tick = 0;
    status = -1;
    events = zeros(0, 4);
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
        end
    end
    endTick = tick;
end
