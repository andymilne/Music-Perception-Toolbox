function notes = readScore(path)
%READSCORE  Parse a MIDI or MusicXML file into an event table.
%
%   notes = readScore(path)
%
%   Parses a Standard MIDI File (format 0 or 1; .mid, .midi) or a MusicXML
%   score (.musicxml, .xml, partwise or timewise; compressed .mxl) into an
%   EVENT TABLE: a MATLAB table with one row per sounding note, sorted by
%   onset, then part, then pitch.
%
%   Columns carried by both sources
%       onsetBeats, onsetSeconds        onset in quarter notes / seconds
%       durationBeats, durationSeconds  duration in quarter notes / seconds
%       pitch                           in MIDI note numbers (A4 = 69), and
%                                       not an integer where the file bends
%                                       or writes a microtone
%       velocity                        MIDI velocity (0-127)
%       part                            categorical; the part names are its
%                                       categories, in part order
%       measure                         1-based bar number
%
%   A MIDI file adds
%       channel                         MIDI channel (1-16)
%       noteNumber                      the note number as recorded, which
%                                       is what note identity rests on
%       program                         the program change in force on that
%                                       channel at the note's onset, 0
%                                       where none was sent, which selects
%                                       the instrument sound
%       weight                          velocity with the loudness
%                                       controllers folded in
%       soundingDurationBeats,          duration with the pedals resolved
%       soundingDurationSeconds
%
%   A MusicXML score adds
%       voice                           1-based voice within its part
%       staff                           1-based; a part written on more
%                                       than one staff, as a keyboard part
%                                       is, says which each note is on
%       fermata                         logical
%       staccato, accent, tenuto        logical articulations
%
%   The boolean marks are not mutually exclusive -- a note may be both
%   staccato and accented -- so each is its own column, and a merged tied
%   note carries a mark any of its segments carries.
%
%   A column is present only where the source carries the information, so
%   channel and voice are never the same column and never stand in for one
%   another. notes.Properties.Description is 'midi' or 'musicxml'.
%
%   The table is an ordinary MATLAB table, so rows are selected with
%   MATLAB's own indexing -- notes(notes.part == "Soprano", :) -- and no
%   toolbox function is needed to read or filter it. preMaetFromScore turns
%   it into the (pAttr, wAttr, specs) of buildMaet. Both parsers are
%   self-contained (no toolbox or Java dependency) and mirror the Python
%   mpt.read_score, which reads the same files to the same table.
%
%   Conventions
%     - A beat is a quarter note (MIDI ticks per quarter note; MusicXML
%       divisions per quarter note), whatever the time signature.
%     - Seconds follow the tempo map: every MIDI set_tempo event, every
%       MusicXML <sound tempo> or metronome direction (120 quarter notes
%       per minute where a file gives none).
%     - MusicXML pitches are converted from step, alter, and octave;
%       unpitched notes and rests are not notes.
%     - A MIDI note-on with velocity 0 is a note-off. Notes left open at
%       the end of a track are closed there.
%     - MusicXML tied notes are merged into one note (the start carries the
%       summed duration); grace notes carry no duration and are skipped;
%       <chord/> notes share the preceding note's onset.
%     - MusicXML velocity is the note's dynamics attribute (a percentage of
%       forte, forte being 90), 90 where absent.
%
%   MIDI controller streams
%     The three that change a note's own columns are resolved at read;
%     every other controller is out of scope, since a value sampled at the
%     onset would misrepresent a ramp inside a held note.
%
%     - Sustain and sostenuto give soundingDuration beside the recorded
%       duration, so either can feed an analysis. A note whose note-off
%       falls while sustain (CC64, at or above 64) is down sounds until the
%       pedal comes up, or to the end of the file where it never does;
%       sostenuto (CC66) holds only what was already down when it was
%       pressed; and the same note number struck again on the same channel
%       damps what is left of the first, while the same pitch on another
%       channel does not, since two channels may be two instruments. Pedal
%       state and the damping rule are both per channel, and a channel
%       belongs to the file rather than to a track.
%     - Pitch bend is resolved into pitch, which is therefore not an
%       integer: bend is how microtonal music is carried in MIDI, in the
%       one-channel-per-note idiom and under MPE alike. The range is 2
%       semitones unless RPN 0 sets it, or an MPE Configuration Message
%       (RPN 6 on channel 1 or 16) opens a zone, whose member channels take
%       the 48-semitone MPE default; a file that bends without declaring a
%       range raises readScore:bendRange. A note takes the last bend at or
%       before its onset tick, so a bend sent immediately before a note-on,
%       or at the same tick in either file order, tunes it; where a channel
%       has no earlier bend at all, the first bend after that note is used
%       provided no further note-on intervenes. Under MPE the bend
%       continues through the note as a slide, and the resolved value is
%       the pitch at onset.
%     - Channel volume (CC7) and expression (CC11) fold into weight:
%
%           weight = (velocity / 127) * (cc7 / 127)^2 * (cc11 / 127)^2
%
%       so that a passage played down by expression is not weighted as
%       though it were at full strength. velocity keeps the value as
%       recorded.
%
%       The two factors rest on different grounds, and the formula is a
%       hybrid. The squares are MIDI's specified default response for both
%       controllers, an attenuation of 40*log10(cc/127) dB; the two are
%       cascaded gain stages, so in dB they add and in amplitude they
%       multiply. The velocity factor is linear because MIDI specifies no
%       velocity-to-amplitude curve -- it is instrument-dependent -- and
%       because taking it linearly makes weight equal to the toolbox's
%       'weights', 'velocity' weighting on any file that sends no
%       controller, which is nearly all of them. So weight is that
%       weighting corrected by the channel's specified gain, and not an
%       estimate of sounding amplitude. A different velocity curve is one
%       transformation of the column away.
%
%   See also PREMAETFROMSCORE, BUILDMAET, TRANSFORMATTRIBUTES.

    path = char(path);
    [~, ~, ext] = fileparts(path);
    ext = lower(ext);
    switch ext
        case {'.mid', '.midi', '.smf', '.kar'}
            raw = internal.parseMidi(path);
        case '.mxl'
            raw = internal.parseMusicXml(localUnzipMxl(path));
        case {'.musicxml', '.xml'}
            raw = internal.parseMusicXml(fileread(path));
        otherwise
            error('readScore:extension', ...
                  ['Unrecognised score file extension ''%s'': expected ' ...
                   '.mid, .midi, .musicxml, .xml, or .mxl.'], ext);
    end
    notes = localFinishTable(raw);
end


function notes = localFinishTable(raw)
    cols = raw.columns;
    rows = raw.rows;
    if isempty(rows)
        rows = zeros(0, numel(cols));
    end
    iOnset = find(strcmp(cols, 'onsetBeats'), 1);
    iPart  = find(strcmp(cols, 'part'), 1);
    iPitch = find(strcmp(cols, 'pitch'), 1);
    [~, order] = sortrows(rows(:, [iOnset iPart iPitch]));
    rows = rows(order, :);

    partNames = localPartCategories(raw.partNames, rows(:, iPart));
    notes = array2table(rows, 'VariableNames', cols);
    notes.part = localPartColumn(rows(:, iPart), partNames);
    counts = {'noteNumber', 'channel', 'program', 'voice', 'staff', 'measure'};
    for i = 1:numel(counts)
        if any(strcmp(cols, counts{i}))
            notes.(counts{i}) = round(notes.(counts{i}));
        end
    end
    flags = {'fermata', 'staccato', 'accent', 'tenuto'};
    for i = 1:numel(flags)
        if any(strcmp(cols, flags{i}))
            notes.(flags{i}) = logical(notes.(flags{i}));
        end
    end
    notes.Properties.Description = raw.source;
end


function col = localPartColumn(partIndex, partNames)
    if isempty(partNames)
        col = categorical(partIndex);
    else
        col = categorical(partIndex, 1:numel(partNames), partNames);
    end
end


function names = localPartCategories(partNames, partIndex)
    % Part names as a unique, non-empty category list, one per part. The
    % parser supplies one name per part, but a score may leave a part
    % unnamed or repeat a name, and categories have to be distinct.
    nParts = numel(partNames);
    if ~isempty(partIndex)
        nParts = max(nParts, max(partIndex));
    end
    names = cell(1, nParts);
    for i = 1:nParts
        if i <= numel(partNames)
            nm = strtrim(char(partNames{i}));
        else
            nm = '';
        end
        if isempty(nm)
            nm = sprintf('Part %d', i);
        end
        base = nm;
        k = 1;
        while any(strcmp(nm, names(1:i-1)))
            k = k + 1;
            nm = sprintf('%s (%d)', base, k);
        end
        names{i} = nm;
    end
end


function txt = localUnzipMxl(path)
    tmp = tempname();
    mkdir(tmp);
    cleanup = onCleanup(@() rmdir(tmp, 's')); %#ok<NASGU>
    files = unzip(path, tmp);
    rootName = '';
    cont = fullfile(tmp, 'META-INF', 'container.xml');
    if exist(cont, 'file') == 2
        c = fileread(cont);
        tok = regexp(c, '<rootfile[^>]*full-path\s*=\s*"([^"]+)"', 'tokens', 'once');
        if ~isempty(tok)
            rootName = fullfile(tmp, tok{1});
        end
    end
    if isempty(rootName)
        for i = 1:numel(files)
            [~, nm, ex] = fileparts(files{i});
            if any(strcmpi(ex, {'.xml', '.musicxml'})) && ~strcmpi(nm, 'container')
                rootName = files{i};
                break;
            end
        end
    end
    if isempty(rootName)
        error('readScore:mxl', 'No MusicXML document inside the .mxl.');
    end
    txt = fileread(rootName);
end
