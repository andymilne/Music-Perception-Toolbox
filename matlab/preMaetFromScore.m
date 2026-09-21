function pm = preMaetFromScore(source, nvArgs)
%PREMAETFROMSCORE  Build a pre-MAET from a score.
%
%   PM = preMaetFromScore(source, ...)
%
%   source is a file path (parsed with readScore: MIDI, MusicXML, or .mxl)
%   or an event table from readScore. The output is the pre-MAET that
%   buildMaet and the pre-MAET preprocessors consume.
%
%   Name-value pairs
%       'attributes'     - cell of names from {'pitch', 'onset', 'duration',
%                          'soundingDuration', 'velocity', 'weight',
%                          'noteNumber', 'part', 'measure', 'fermata'}, in
%                          order (default {'pitch', 'onset'}). The last four
%                          need a column the source carries, and raise where
%                          it does not.
%       'pitch'          - pitch scale: 'midi' (default), 'cents', 'hz',
%                          'octave', or any pitch scale of transformAttributes.
%       'time'           - 'seconds' (default) or 'beats' (quarter notes)
%                          for onsets and durations.
%       'weights'        - 'velocity' (default; velocity / 127), 'ones',
%                          'duration' (in the chosen time unit), or 'weight'
%                          (the table's weight column, which folds channel
%                          volume and expression into the velocity).
%       'parts'          - [] (all), the 1-based parts to keep, or a cell
%                          or string array of part names.
%       'chords'         - 'bind' (default) gathers notes that start together
%                          (within 'chordTolerance', in the chosen time unit)
%                          into one event whose pitch, duration, velocity,
%                          and part attributes carry one value per note
%                          (K = largest chord size, NaN-padded), so that a
%                          pitch attribute at r = 2 reads the chords' dyads;
%                          'separate' makes every note its own event (K = 1
%                          throughout).
%       'chordTolerance' - onset tolerance for binding (default 0).
%       'names'          - name the specs after the attributes (default true).
%
%   Output
%       pm - Pre-MAET: pAttr is a 1 x A cell of K_a x N value matrices;
%            wAttr a 1 x A cell of K_a x N weight matrices (NaN-padded
%            slots carry weight 0), or [] under 'ones'; specs a 1 x A cell
%            of flat specs, named after the attributes.
%
%   See also PREMAET, READSCORE, BUILDMAET, TRANSFORMATTRIBUTES,
%            FLATSPECS.

    arguments
        source
        nvArgs.attributes = {'pitch', 'onset'}
        nvArgs.pitch (1,:) char = 'midi'
        nvArgs.time (1,:) char = 'seconds'
        nvArgs.weights (1,:) char = 'velocity'
        nvArgs.parts = []
        nvArgs.chords (1,:) char = 'bind'
        nvArgs.chordTolerance (1,1) double = 0
        nvArgs.names (1,1) logical = true
    end

    if ischar(source) || isstring(source)
        notes = readScore(char(source));
    else
        notes = source;
    end
    attributes = cellfun(@(a) lower(char(a)), cellstr(nvArgs.attributes), 'UniformOutput', false);
    allowed = {'pitch', 'onset', 'duration', 'soundingDuration', 'velocity', ...
               'weight', 'noteNumber', 'part', 'measure', 'fermata'};
    for i = 1:numel(attributes)
        if ~any(strcmp(attributes{i}, allowed))
            error('preMaetFromScore:attribute', ...
                  'Unknown attribute ''%s''; choose from %s.', attributes{i}, strjoin(allowed, ', '));
        end
    end
    if ~any(strcmp(nvArgs.time, {'seconds', 'beats'}))
        error('preMaetFromScore:time', 'time must be ''seconds'' or ''beats''.');
    end
    if ~any(strcmp(nvArgs.weights, {'velocity', 'ones', 'duration', 'weight'}))
        error('preMaetFromScore:weights', ...
              'weights must be ''velocity'', ''ones'', ''duration'', or ''weight''.');
    end
    if ~any(strcmp(nvArgs.chords, {'bind', 'separate'}))
        error('preMaetFromScore:chords', 'chords must be ''bind'' or ''separate''.');
    end

    if ~istable(notes)
        error('preMaetFromScore:source', ...
              ['source must be a file path or an event table from ' ...
               'readScore; got %s.'], class(notes));
    end
    vars = notes.Properties.VariableNames;
    needs = {'soundingDuration', 'soundingDurationBeats'; ...
             'weight',           'weight'; ...
             'noteNumber',       'noteNumber'; ...
             'fermata',          'fermata'};
    for i = 1:size(needs, 1)
        if any(strcmp(needs{i, 1}, attributes)) && ~any(strcmp(vars, needs{i, 2}))
            error('preMaetFromScore:missingColumn', ...
                  ['The table has no ''%s'' column, so ''%s'' cannot be an ' ...
                   'attribute; this source does not carry it.'], ...
                  needs{i, 2}, needs{i, 1});
        end
    end
    if strcmp(nvArgs.weights, 'weight') && ~any(strcmp(vars, 'weight'))
        error('preMaetFromScore:noWeight', ...
              ['weights ''weight'' needs a ''weight'' column, which this ' ...
               'source does not carry.']);
    end

    partCodes = double(notes.part);
    keep = true(height(notes), 1);
    if ~isempty(nvArgs.parts)
        wanted = nvArgs.parts;
        if isnumeric(wanted)
            keep = keep & ismember(partCodes, wanted(:));
        else
            keep = keep & ismember(cellstr(notes.part), cellstr(wanted));
        end
    end
    if strcmp(nvArgs.time, 'seconds')
        onset = notes.onsetSeconds(keep); dur = notes.durationSeconds(keep);
        soundingName = 'soundingDurationSeconds';
    else
        onset = notes.onsetBeats(keep); dur = notes.durationBeats(keep);
        soundingName = 'soundingDurationBeats';
    end
    midi = notes.pitch(keep);
    vel = notes.velocity(keep);
    part = partCodes(keep);
    measure = double(notes.measure(keep));
    optional = @(name) localOptional(notes, name, keep);
    sounding = optional(soundingName);
    noteNumber = optional('noteNumber');
    weightCol = optional('weight');
    fermata = optional('fermata');
    nNotes = numel(midi);

    if strcmpi(nvArgs.pitch, 'midi')
        pitchVals = midi;
    else
        pitchVals = transformAttributes(midi, [], {'midi', nvArgs.pitch});
    end
    perNote = struct('pitch', pitchVals, 'onset', onset, 'duration', dur, ...
                     'soundingDuration', sounding, 'velocity', vel, ...
                     'weight', weightCol, 'noteNumber', noteNumber, ...
                     'part', part, 'measure', measure, 'fermata', fermata);
    switch nvArgs.weights
        case 'velocity', wNote = vel / 127;
        case 'duration', wNote = dur;
        case 'weight',   wNote = weightCol;
        otherwise,       wNote = [];
    end

    % Group notes into events.
    % A gridded table already says which rows share an event, so its grid
    % position is the key and the onset tolerance does not apply.
    if strcmp(nvArgs.chords, 'separate') || nNotes == 0
        groups = num2cell(1:nNotes);
    elseif any(strcmp(vars, 'gridIndex'))
        key = double(notes.gridIndex(keep));
        groups = {};
        for i = 1:nNotes
            if ~isempty(groups) && key(i) == key(groups{end}(1))
                groups{end}(end + 1) = i;
            else
                groups{end + 1} = i; %#ok<AGROW>
            end
        end
    else
        [~, order] = sort(onset);
        groups = {};
        for i = order(:).'
            if ~isempty(groups) && onset(i) - onset(groups{end}(1)) <= nvArgs.chordTolerance
                groups{end}(end + 1) = i;
            else
                groups{end + 1} = i; %#ok<AGROW>
            end
        end
    end
    N = numel(groups);
    K = 1;
    for n = 1:N
        K = max(K, numel(groups{n}));
    end

    A = numel(attributes);
    pAttr = cell(1, A);
    wList = cell(1, A);
    for a = 1:A
        vals = perNote.(attributes{a});
        if strcmp(attributes{a}, 'onset') || K == 1
            M = nan(1, N);
            W = zeros(1, N);
            for n = 1:N
                g = groups{n};
                M(1, n) = vals(g(1));
                if strcmp(attributes{a}, 'onset') || isempty(wNote)
                    W(1, n) = 1;
                else
                    W(1, n) = wNote(g(1));
                end
            end
        else
            M = nan(K, N);
            W = zeros(K, N);
            for n = 1:N
                g = groups{n};
                M(1:numel(g), n) = vals(g);
                if isempty(wNote)
                    W(1:numel(g), n) = 1;
                else
                    W(1:numel(g), n) = wNote(g);
                end
            end
        end
        % A slot with no value carries no weight, whether it is padding
        % or an empty grid point whose weight column is itself missing.
        W(isnan(M)) = 0;
        pAttr{a} = M;
        wList{a} = W;
    end
    if isempty(wNote)
        w = [];
    else
        w = wList;
    end
    if nvArgs.names
        specs = flatSpecs(pAttr, 'name', attributes);
    else
        specs = flatSpecs(pAttr);
    end
    % A score determines the periodicity of its attributes and not their
    % kernel widths. Pitches, onsets, durations, velocities, parts, bars
    % and fermatas are all read as they are written -- absolute, on an
    % unbounded axis -- so [per] = 0 and the period is inert; octave
    % equivalence is an equivalence the analyst imposes, not one the score
    % states. Sigma is left unset rather than defaulted, because there is
    % no width a score implies: buildMaet will then name the attribute
    % that still needs one.
    for a = 1:numel(specs)
        specs{a}.isPer = false;
        specs{a}.period = 0;
    end

    pm = preMaet(pAttr, w, specs);
end


function v = localOptional(notes, name, keep)
    % A column the source may not carry reads as zeros, so that an
    % attribute the caller did not ask for costs nothing; asking for one
    % that is absent is refused earlier, by name.
    if any(strcmp(notes.Properties.VariableNames, name))
        v = double(notes.(name)(keep));
    else
        v = zeros(sum(keep), 1);
    end
end
