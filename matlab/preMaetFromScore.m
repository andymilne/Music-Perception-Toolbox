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
%                          it does not. On a gridded table 'onset' reads the
%                          grid's onset, the event there being the grid
%                          point rather than any one note.
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
%       'roles'          - struct mapping a categorical column to how it
%                          reaches the pre-MAET: 'separateAttributes',
%                          'orderedMultiset', 'simplex', or 'drop'. A
%                          column with no entry is not encoded.
%
%                          The first two are STRUCTURAL: the level is
%                          realized as which attribute you are in, or as
%                          which position, so the binding of value to
%                          level is carried by the layout. They gather an
%                          event's rows into one event holding one slot
%                          per level, which needs 'chords', 'bind' and an
%                          event that holds exactly one row per level;
%                          events that do not are dropped, with a warning
%                          naming the count. Only one category may be
%                          structural, since a structural category
%                          individuates the values sounding together and
%                          two of them give an attribute set that can
%                          never be fully populated. A structural category
%                          splits every listed attribute except the
%                          event-level ones ('onset'), whose value belongs
%                          to the event rather than to the note.
%
%                          'simplex' is a VALUE: the level becomes the
%                          coordinates of a vertex of a unit-edge regular
%                          simplex, carried as its own attribute read
%                          whole, and the binding of value to level is the
%                          tensor product of the two attributes at the
%                          event. Each concurrently-sounding note is then
%                          its own event, so it needs 'chords',
%                          'separate' -- unless a structural category is
%                          also given, in which case the simplex is tagged
%                          within each of its slots.
%
%                          A caution about the no-role reading. With
%                          'chords', 'bind' and no role, an event holds the
%                          chord as an unordered multiset on every
%                          attribute. Where two attributes describe the
%                          SAME notes -- a pitch-class attribute and a
%                          pitch-height one, say -- their product then
%                          pairs every value of one with every value of
%                          the other, including the soprano's pitch class
%                          with the bass's height, and matching rewards
%                          combinations the chord does not contain.
%                          Binding a note's attributes to each other needs
%                          one event per note ('chords', 'separate'),
%                          which is what the JMM article's voice-agnostic
%                          encoding does.
%       'groupBy'        - the column whose equal values in consecutive
%                          rows make one event. The default is the grid
%                          position where the table has been gridded, and
%                          otherwise the onset within 'chordTolerance'. An
%                          event is a contiguous run, not every row
%                          sharing a value, so a bar number that comes
%                          round again after a repeat gives two events
%                          rather than one.
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
        nvArgs.roles = struct()
        nvArgs.groupBy = ''
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

    % How a categorical column reaches the pre-MAET. The first two roles
    % are structural -- the level is realized as which attribute you are
    % in, or as which position -- and gather a group's rows into one
    % event. The third is a value, carried alongside the note's own
    % attributes.
    allRoles = {'separateAttributes', 'orderedMultiset', 'simplex', 'drop'};
    structuralColumn = '';
    structuralRole = '';
    simplexColumns = {};
    roleNames = fieldnames(nvArgs.roles);
    for i = 1:numel(roleNames)
        column = roleNames{i};
        role = char(nvArgs.roles.(column));
        if ~any(strcmp(role, allRoles))
            error('preMaetFromScore:unknownRole', ...
                  'roles.%s: unknown role ''%s''; choose from %s.', ...
                  column, role, strjoin(allRoles, ', '));
        end
        if ~any(strcmp(notes.Properties.VariableNames, column))
            error('preMaetFromScore:unknownColumn', ...
                  'roles names column ''%s'', which the table does not have.', ...
                  column);
        end
        if strcmp(role, 'drop'); continue; end
        if ~iscategorical(notes.(column))
            error('preMaetFromScore:roleKind', ...
                  ['roles.%s: a role needs a categorical column, and %s ' ...
                   'is %s.'], column, column, class(notes.(column)));
        end
        if strcmp(role, 'simplex')
            simplexColumns{end + 1} = column; %#ok<AGROW>
            continue;
        end
        if ~isempty(structuralColumn)
            error('preMaetFromScore:twoStructural', ...
                  ['roles.%s: only one category may be structural against ' ...
                   'a given value set, and ''%s'' already is. A structural ' ...
                   'category individuates the values sounding together; two ' ...
                   'of them give an attribute set that can never be fully ' ...
                   'populated. Either keep ''%s'' structural and give ' ...
                   '''%s'' the ''simplex'' role, which tags it within each ' ...
                   'slot, or give ''%s'' the ''simplex'' role too, which ' ...
                   'yields one event per note and additive partial credit ' ...
                   'across levels.'], column, structuralColumn, ...
                  structuralColumn, column, structuralColumn);
        end
        structuralColumn = column;
        structuralRole = role;
    end
    if ~isempty(structuralColumn) && ~strcmp(nvArgs.chords, 'bind')
        error('preMaetFromScore:structuralNeedsBind', ...
              ['roles.%s is structural, which gathers the rows of an event ' ...
               'into one; that needs ''chords'', ''bind''.'], structuralColumn);
    end
    if ~isempty(simplexColumns) && isempty(structuralColumn) ...
            && strcmp(nvArgs.chords, 'bind')
        error('preMaetFromScore:simplexNeedsSeparate', ...
              ['The ''simplex'' role carries the level as a value at each ' ...
               'event, so each concurrently-sounding note is its own event; ' ...
               'that needs ''chords'', ''separate'', or a structural ' ...
               'category to tag within.']);
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
        unit = 'Seconds'; other = 'Beats';
    else
        unit = 'Beats'; other = 'Seconds';
    end
    % On a gridded table the event is the grid point, so its onset is the
    % grid's, not the onset of whichever note happens to be in the first
    % slot. The note's own onset stays in the table for selection.
    onsetName = ['onset', unit];
    if any(strcmp(vars, ['gridOnset', unit]))
        onsetName = ['gridOnset', unit];
    elseif any(strcmp(vars, ['gridOnset', other]))
        error('preMaetFromScore:gridUnit', ...
              ['The table was gridded over %s, so time ''%s'' has no grid ' ...
               'onset to read; grid over %s or convert with that unit.'], ...
              lower(other), nvArgs.time, nvArgs.time);
    end
    onset = notes.(onsetName)(keep);
    dur = notes.(['duration', unit])(keep);
    soundingName = ['soundingDuration', unit];
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

    % Group notes into events. An explicit key, or a gridded table's own
    % grid position, already says which rows share an event, and the
    % onset tolerance then does not apply.
    if ~isempty(nvArgs.groupBy) && ~any(strcmp(vars, char(nvArgs.groupBy)))
        error('preMaetFromScore:unknownColumn', ...
              'groupBy names column ''%s'', which the table does not have.', ...
              char(nvArgs.groupBy));
    end
    if ~isempty(nvArgs.groupBy)
        keyColumn = char(nvArgs.groupBy);
    elseif any(strcmp(vars, 'gridIndex'))
        keyColumn = 'gridIndex';
    else
        keyColumn = '';
    end
    if strcmp(nvArgs.chords, 'separate') || nNotes == 0
        groups = num2cell(1:nNotes);
    elseif ~isempty(keyColumn)
        key = notes.(keyColumn)(keep);
        groups = {};
        for i = 1:nNotes
            if ~isempty(groups) && isequaln(key(i), key(groups{end}(1)))
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

    % A structural category puts one of its levels in each slot of every
    % event, so an event must hold exactly one row per level. One that
    % does not is dropped, with a count: an analyst may well accept
    % losing a few events to use the encoding.
    slots = [];
    levels = {};
    if ~isempty(structuralColumn)
        levels = categories(notes.(structuralColumn)).';
        codes = double(notes.(structuralColumn));
        codes = codes(keep);
        nLevels = numel(levels);
        slots = zeros(numel(groups), nLevels);
        keptGroups = cell(1, numel(groups));
        nKept = 0;
        lost = 0;
        for n = 1:numel(groups)
            g = groups{n};
            row = zeros(1, nLevels);
            ok = true;
            for j = 1:numel(g)
                c = codes(g(j));
                if isnan(c) || c < 1 || row(c) > 0
                    ok = false;
                    break;
                end
                row(c) = g(j);
            end
            if ~ok || any(row == 0)
                lost = lost + 1;
                continue;
            end
            nKept = nKept + 1;
            slots(nKept, :) = row;
            keptGroups{nKept} = g;
        end
        slots = slots(1:nKept, :);
        groups = keptGroups(1:nKept);
        if lost > 0
            if strcmp(keyColumn, 'gridIndex')
                gridded = '; on a gridded table that breaks the uniform time index';
            else
                gridded = '';
            end
            warning('preMaetFromScore:incompleteEvents', ...
                    ['%d of %d events do not hold exactly one %s per level, ' ...
                     'so they are dropped%s. A structural category fills ' ...
                     'every slot of every event.'], ...
                    lost, lost + nKept, structuralColumn, gridded);
        end
    end

    N = numel(groups);
    K = 1;
    for n = 1:N
        K = max(K, numel(groups{n}));
    end

    pAttr = {};
    wList = {};
    specR = zeros(1, 0);
    specExch = false(1, 0);
    specNames = {};

    for a = 1:numel(attributes)
        name = attributes{a};
        vals = perNote.(name);
        % The notes gathered into one event share an onset and a bar,
        % so a structural category does not split either: one
        % attribute per level carrying the same number would raise
        % that factor to the power of the level count.
        isEventLevel = any(strcmp(name, {'onset', 'measure'}));
        if isempty(structuralColumn)
            if isEventLevel || K == 1
                M = nan(1, N);
                W = zeros(1, N);
                for n = 1:N
                    g = groups{n};
                    M(1, n) = vals(g(1));
                    if isEventLevel || isempty(wNote)
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
            [pAttr, wList, specR, specExch, specNames] = localAddAttr( ...
                pAttr, wList, specR, specExch, specNames, M, W, 1, true, name);
        elseif isEventLevel
            M = nan(1, N);
            for n = 1:N
                M(1, n) = vals(slots(n, 1));
            end
            [pAttr, wList, specR, specExch, specNames] = localAddAttr( ...
                pAttr, wList, specR, specExch, specNames, M, ones(1, N), ...
                1, true, name);
        elseif strcmp(structuralRole, 'separateAttributes')
            for v = 1:numel(levels)
                M = nan(1, N);
                W = ones(1, N);
                for n = 1:N
                    M(1, n) = vals(slots(n, v));
                    if ~isempty(wNote); W(1, n) = wNote(slots(n, v)); end
                end
                [pAttr, wList, specR, specExch, specNames] = localAddAttr( ...
                    pAttr, wList, specR, specExch, specNames, M, W, 1, true, ...
                    sprintf('%s_%s', name, levels{v}));
            end
        else
            V = numel(levels);
            M = nan(V, N);
            W = ones(V, N);
            for n = 1:N
                for v = 1:V
                    M(v, n) = vals(slots(n, v));
                    if ~isempty(wNote); W(v, n) = wNote(slots(n, v)); end
                end
            end
            [pAttr, wList, specR, specExch, specNames] = localAddAttr( ...
                pAttr, wList, specR, specExch, specNames, M, W, V, false, name);
        end
    end

    % Simplex-coded categories. The coordinates of one level form one
    % ordered value read whole, so the attribute's tuple size is their
    % number and its weights are one: the note's own weight is carried by
    % its other attributes, and the attributes multiply.
    for i = 1:numel(simplexColumns)
        column = simplexColumns{i};
        vertices = simplexVertices(numel(categories(notes.(column))));
        d = size(vertices, 2);
        codesS = double(notes.(column));
        codesS = codesS(keep);
        if isempty(structuralColumn)
            M = nan(d, N);
            for n = 1:N
                c = codesS(groups{n}(1));
                if ~isnan(c); M(:, n) = vertices(c, :).'; end
            end
            [pAttr, wList, specR, specExch, specNames] = localAddAttr( ...
                pAttr, wList, specR, specExch, specNames, M, ones(d, N), ...
                d, false, column);
        else
            for v = 1:numel(levels)
                M = nan(d, N);
                for n = 1:N
                    c = codesS(slots(n, v));
                    if ~isnan(c); M(:, n) = vertices(c, :).'; end
                end
                [pAttr, wList, specR, specExch, specNames] = localAddAttr( ...
                    pAttr, wList, specR, specExch, specNames, M, ones(d, N), ...
                    d, false, sprintf('%s_%s', column, levels{v}));
            end
        end
    end

    if isempty(wNote)
        w = [];
    else
        w = wList;
    end
    if nvArgs.names
        specs = flatSpecs(pAttr, 'r', specR, 'exch', specExch, ...
                          'name', specNames);
    else
        specs = flatSpecs(pAttr, 'r', specR, 'exch', specExch);
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


function [pAttr, wList, specR, specExch, specNames] = localAddAttr( ...
        pAttr, wList, specR, specExch, specNames, M, W, r, exch, name)
    % A slot with no value carries no weight, whether it is padding or an
    % empty grid point whose weight column is itself missing.
    W(isnan(M)) = 0;
    pAttr{end + 1} = M;
    wList{end + 1} = W;
    specR(end + 1) = r;
    specExch(end + 1) = exch;
    specNames{end + 1} = name;
end
