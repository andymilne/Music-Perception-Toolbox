function raw = parseMusicXml(txt)
%PARSEMUSICXML  MusicXML (partwise or timewise) text to note rows.
%
%   raw = internal.parseMusicXml(txt) returns a struct with .rows (M x 9:
%   onsetBeats onsetSeconds durationBeats durationSeconds pitch velocity
%   part voice measure), .partNames (1 x P cell), .source = 'musicxml'.
%   Twin of the Python mpt.score._parse_musicxml; see readScore for the
%   conventions.

    root = internal.parseXmlMini(txt);
    if strcmp(root.name, 'score-timewise')
        root = localTimewiseToPartwise(root);
    end
    if ~strcmp(root.name, 'score-partwise')
        error('readScore:xml', 'Not a MusicXML score (root element ''%s'').', root.name);
    end

    % Part names from the part-list.
    ids = {}; names = {};
    pl = localChild(root, 'part-list');
    if ~isempty(pl)
        sps = localChildren(pl, 'score-part');
        for i = 1:numel(sps)
            ids{end + 1} = localAttr(sps{i}, 'id', ''); %#ok<AGROW>
            names{end + 1} = localText(localChild(sps{i}, 'part-name'), ''); %#ok<AGROW>
        end
    end

    parts = localChildren(root, 'part');
    % Pass 1: tempo changes (a change applies to the whole score).
    tempoChanges = zeros(0, 2);
    for p = 1:numel(parts)
        [~, tempos] = localWalkPart(parts{p});
        tempoChanges = [tempoChanges; tempos]; %#ok<AGROW>
    end
    tempoChanges = sortrows(tempoChanges, 1);
    if isempty(tempoChanges) || tempoChanges(1, 1) > 0
        tempoChanges = [0 120; tempoChanges];
    end

    rows = zeros(0, 9);
    partNames = cell(1, numel(parts));
    for p = 1:numel(parts)
        notes = localWalkPart(parts{p});    % onset dur midi vel voice measure
        pid = localAttr(parts{p}, 'id', '');
        k = find(strcmp(ids, pid), 1);
        nm = '';
        if ~isempty(k)
            nm = names{k};
        end
        if isempty(nm)
            nm = sprintf('part %d', p);
        end
        partNames{p} = nm;
        for i = 1:size(notes, 1)
            o = notes(i, 1); dq = notes(i, 2);
            s0 = localSecondsAt(o, tempoChanges);
            s1 = localSecondsAt(o + dq, tempoChanges);
            rows(end + 1, :) = [o, s0, dq, s1 - s0, notes(i, 3), notes(i, 4), ...
                                p, notes(i, 5), notes(i, 6)]; %#ok<AGROW>
        end
    end
    raw = struct('rows', rows, 'partNames', {partNames}, 'source', 'musicxml');
end


% ---------------------------------------------------------------------
%  Tree helpers
% ---------------------------------------------------------------------

function c = localChild(node, name)
    c = [];
    if isempty(node), return; end
    for i = 1:numel(node.children)
        if strcmp(node.children{i}.name, name)
            c = node.children{i};
            return;
        end
    end
end

function cs = localChildren(node, name)
    cs = {};
    if isempty(node), return; end
    for i = 1:numel(node.children)
        if strcmp(node.children{i}.name, name)
            cs{end + 1} = node.children{i}; %#ok<AGROW>
        end
    end
end

function v = localAttr(node, name, default)
    v = default;
    if ~isempty(node) && isfield(node.attrs, name)
        v = node.attrs.(name);
    end
end

function t = localText(node, default)
    t = default;
    if ~isempty(node) && ~isempty(node.text)
        t = node.text;
    end
end

function found = localDescendants(node, name)
    % All descendants named `name`, depth first.
    found = {};
    for i = 1:numel(node.children)
        ch = node.children{i};
        if strcmp(ch.name, name)
            found{end + 1} = ch; %#ok<AGROW>
        end
        found = [found, localDescendants(ch, name)]; %#ok<AGROW>
    end
end


% ---------------------------------------------------------------------
%  Part walk
% ---------------------------------------------------------------------

function [notes, tempos] = localWalkPart(part)
    % notes: M x 6 (onsetQ durQ midi velocity voice measure); tempos: T x 2
    % (posQ bpm). Positions in quarter notes.
    divisions = 1;
    pos = 0;
    notes = zeros(0, 6);
    tempos = zeros(0, 2);
    tieKeys = zeros(0, 2);        % voice, midi
    tieIdx = zeros(0, 1);
    measures = localChildren(part, 'measure');
    for mi = 1:numel(measures)
        measure = measures{mi};
        measureStart = pos;
        measureNo = str2double(localAttr(measure, 'number', num2str(mi)));
        if isnan(measureNo), measureNo = mi; end
        lastOnset = pos;
        for ei = 1:numel(measure.children)
            el = measure.children{ei};
            switch el.name
                case 'attributes'
                    d = localText(localChild(el, 'divisions'), '');
                    if ~isempty(d), divisions = str2double(d); end
                case {'direction', 'sound'}
                    snds = localDescendants(el, 'sound');
                    if strcmp(el.name, 'sound'), snds = [{el}, snds]; end
                    for k = 1:numel(snds)
                        tp = localAttr(snds{k}, 'tempo', '');
                        if ~isempty(tp)
                            tempos(end + 1, :) = [pos, str2double(tp)]; %#ok<AGROW>
                        end
                    end
                    mets = localDescendants(el, 'metronome');
                    for k = 1:numel(mets)
                        unit = localText(localChild(mets{k}, 'beat-unit'), 'quarter');
                        pm = str2double(localText(localChild(mets{k}, 'per-minute'), ''));
                        if ~isnan(pm)
                            bpm = pm * localQuartersPerUnit(unit);
                            if ~isempty(localChild(mets{k}, 'beat-unit-dot'))
                                bpm = bpm * 1.5;
                            end
                            tempos(end + 1, :) = [pos, bpm]; %#ok<AGROW>
                        end
                    end
                case 'backup'
                    pos = pos - str2double(localText(localChild(el, 'duration'), '0')) / divisions;
                case 'forward'
                    pos = pos + str2double(localText(localChild(el, 'duration'), '0')) / divisions;
                case 'note'
                    isChord = ~isempty(localChild(el, 'chord'));
                    isGrace = ~isempty(localChild(el, 'grace'));
                    durTxt = localText(localChild(el, 'duration'), '');
                    if isempty(durTxt), durQ = 0; else, durQ = str2double(durTxt) / divisions; end
                    if isChord, onset = lastOnset; else, onset = pos; end
                    voice = str2double(localText(localChild(el, 'voice'), '1'));
                    if isnan(voice), voice = 1; end
                    pitchEl = localChild(el, 'pitch');
                    isRest = ~isempty(localChild(el, 'rest'));
                    if ~isGrace && ~isempty(pitchEl) && ~isRest
                        midi = localMidiFromPitch(pitchEl);
                        dyn = localAttr(el, 'dynamics', '');
                        if isempty(dyn), vel = 90; else, vel = str2double(dyn) * 0.9; end
                        vel = min(127, max(0, vel));
                        ties = localChildren(el, 'tie');
                        tieStart = false; tieStop = false;
                        for k = 1:numel(ties)
                            tt = localAttr(ties{k}, 'type', '');
                            tieStart = tieStart || strcmp(tt, 'start');
                            tieStop = tieStop || strcmp(tt, 'stop');
                        end
                        k = find(tieKeys(:, 1) == voice & tieKeys(:, 2) == midi, 1);
                        if tieStop && ~isempty(k)
                            idx = tieIdx(k);
                            notes(idx, 2) = notes(idx, 2) + durQ;
                            if ~tieStart
                                tieKeys(k, :) = []; tieIdx(k) = [];
                            end
                        else
                            notes(end + 1, :) = [onset, durQ, midi, vel, voice, measureNo]; %#ok<AGROW>
                            if tieStart
                                tieKeys(end + 1, :) = [voice, midi]; %#ok<AGROW>
                                tieIdx(end + 1, 1) = size(notes, 1); %#ok<AGROW>
                            end
                        end
                    end
                    if ~isChord && ~isGrace
                        lastOnset = pos;
                        pos = pos + durQ;
                    end
            end
        end
        if pos < measureStart
            pos = measureStart;
        end
    end
end

function q = localQuartersPerUnit(unit)
    switch unit
        case 'whole',   q = 4;
        case 'half',    q = 2;
        case 'eighth',  q = 0.5;
        case '16th',    q = 0.25;
        case '32nd',    q = 0.125;
        case '64th',    q = 0.0625;
        otherwise,      q = 1;
    end
end

function midi = localMidiFromPitch(pitchEl)
    step = upper(localText(localChild(pitchEl, 'step'), 'C'));
    alter = str2double(localText(localChild(pitchEl, 'alter'), '0'));
    if isnan(alter), alter = 0; end
    octave = str2double(localText(localChild(pitchEl, 'octave'), '4'));
    if isnan(octave), octave = 4; end
    semis = struct('C', 0, 'D', 2, 'E', 4, 'F', 5, 'G', 7, 'A', 9, 'B', 11);
    if isfield(semis, step), st = semis.(step); else, st = 0; end
    midi = 12 * (octave + 1) + st + alter;
end

function s = localSecondsAt(q, tempoChanges)
    s = 0;
    prevQ = tempoChanges(1, 1); prevTempo = tempoChanges(1, 2);
    for i = 2:size(tempoChanges, 1)
        if tempoChanges(i, 1) >= q
            break;
        end
        s = s + (tempoChanges(i, 1) - prevQ) * 60 / prevTempo;
        prevQ = tempoChanges(i, 1); prevTempo = tempoChanges(i, 2);
    end
    s = s + (q - prevQ) * 60 / prevTempo;
end

function new = localTimewiseToPartwise(root)
    % Regroup a timewise score (measures containing parts) as partwise.
    new = struct('name', 'score-partwise', 'attrs', root.attrs, 'children', {{}}, 'text', '');
    partIds = {};
    partNodes = {};
    for i = 1:numel(root.children)
        ch = root.children{i};
        if ~strcmp(ch.name, 'measure')
            new.children{end + 1} = ch;
            continue;
        end
        for j = 1:numel(ch.children)
            pt = ch.children{j};
            if ~strcmp(pt.name, 'part'), continue; end
            pid = localAttr(pt, 'id', '');
            k = find(strcmp(partIds, pid), 1);
            if isempty(k)
                partIds{end + 1} = pid; %#ok<AGROW>
                partNodes{end + 1} = struct('name', 'part', 'attrs', pt.attrs, 'children', {{}}, 'text', ''); %#ok<AGROW>
                k = numel(partIds);
            end
            m = struct('name', 'measure', 'attrs', ch.attrs, 'children', {pt.children}, 'text', '');
            partNodes{k}.children{end + 1} = m;
        end
    end
    new.children = [new.children, partNodes];
end
