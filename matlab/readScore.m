function notes = readScore(path)
%READSCORE  Parse a MIDI or MusicXML file into a note table.
%
%   notes = readScore(path)
%
%   Parses a Standard MIDI File (format 0 or 1; .mid, .midi) or a MusicXML
%   score (.musicxml, .xml, partwise or timewise; compressed .mxl) into a
%   NOTE TABLE: a struct of N x 1 column vectors, one row per sounding
%   note, sorted by onset, then part, then pitch:
%
%       .onsetBeats, .onsetSeconds        onset in quarter notes / seconds
%       .durationBeats, .durationSeconds  duration in quarter notes / seconds
%       .pitch                            MIDI note number (A4 = 69)
%       .velocity                         MIDI velocity (0-127)
%       .part                             1-based part (MIDI: track with notes)
%       .channel                          MIDI channel (1-16); MusicXML voice
%       .measure                          1-based bar number
%       .fermata                          1 where a MusicXML note carries a
%                                         fermata (a merged tied note counts
%                                         if any segment does), else 0; MIDI
%                                         has no fermatas, so always 0
%       .partNames                        1 x P cell of part names
%       .source                           'midi' or 'musicxml'
%
%   eventsFromScore turns the table into the (pAttr, w, specs) carrier of
%   buildExpTens. Both parsers are self-contained (no toolbox or Java
%   dependency) and mirror the Python mpt.read_score, which reads the same
%   files to the same table.
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
%   See also EVENTSFROMSCORE, BUILDEXPTENS, TRANSFORMATTRIBUTES.

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
    % raw.rows is M x 10: onsetBeats onsetSeconds durationBeats
    % durationSeconds pitch velocity part channel measure fermata.
    rows = raw.rows;
    if isempty(rows)
        rows = zeros(0, 10);
    end
    [~, order] = sortrows(rows(:, [1 7 5]));
    rows = rows(order, :);
    notes = struct();
    notes.onsetBeats      = rows(:, 1);
    notes.onsetSeconds    = rows(:, 2);
    notes.durationBeats   = rows(:, 3);
    notes.durationSeconds = rows(:, 4);
    notes.pitch           = rows(:, 5);
    notes.velocity        = rows(:, 6);
    notes.part            = rows(:, 7);
    notes.channel         = rows(:, 8);
    notes.measure         = rows(:, 9);
    notes.fermata         = rows(:, 10);
    notes.partNames       = raw.partNames;
    notes.source          = raw.source;
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
