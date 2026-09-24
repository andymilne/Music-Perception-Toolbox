function pm = preMaetFromAttrTable(T, nvArgs)
%PREMAETFROMATTRTABLE  Build a pre-MAET from an attribute table.
%
%   PM = preMaetFromAttrTable(T, ...)
%
%   T is an attribute table, as readScore returns and gridAttrTable passes on.
%   A score file is read first, with readScore; converting reads a table
%   and nothing else. The output is the pre-MAET that buildMaet and the
%   pre-MAET preprocessors consume.
%
%   Name-value pairs
%       'attributes'     - required: a cell of one entry per attribute, in
%                          order. An entry is a struct carrying its column
%                          in a 'column' field together with the
%                          attribute's own parameters: 'name', 'sigma',
%                          'r', 'exch', 'rel', 'isPer', 'period'. The ten
%                          names 'pitch', 'onset', 'duration',
%                          'soundingDuration', 'velocity', 'weight',
%                          'noteNumber', 'part', 'measure', and 'fermata'
%                          get the score-specific treatment (the pitch
%                          scale, beats against seconds, the grid's
%                          onset); any other column of the table is read
%                          as it stands, so a table that never saw a
%                          score converts too. A categorical column is
%                          refused here and belongs to 'roles', its
%                          levels not being values on a line. Listing one column twice gives two
%                          attributes of the same values, read under
%                          different parameters, which is how pitch class
%                          and pitch height are taken from one pitch column.
%
%                            preMaetFromAttrTable(T, 'attributes', { ...
%                              struct('column','pitch','name','pitchClass', ...
%                                     'sigma',0.5,'isPer',true,'period',12), ...
%                              struct('column','pitch','name','pitchHeight', ...
%                                     'sigma',8), ...
%                              struct('column','onset','sigma',0.5)}, ...
%                              'time', 'beats')
%
%                          Of the ten, the last four need a column the
%                          source carries, and raise where it does not. On a
%                          gridded table 'onset' reads the grid's onset, the
%                          event there being the grid point rather than any
%                          one note.
%       'pitch'          - pitch scale: 'midi' (default), 'cents', 'hz',
%                          'octave', or any pitch scale of transformAttributes.
%       'time'           - 'seconds' (default) or 'beats' (quarter notes)
%                          for onsets and durations.
%       'weights'        - 'velocity' (default; velocity / 127), 'ones',
%                          'duration' (in the chosen time unit), or 'weight'
%                          (the table's weight column, which folds channel
%                          volume and expression into the velocity).
%       'parts'          - [] (all), the 1-based parts to keep, or a cell
%                          or string array of part names. A convenience
%                          for the common case; selecting rows of the
%                          table before converting is the more general
%                          route, and reaches any column.
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
%                          column with no entry is not encoded. A value
%                          may instead be a struct carrying the role in a
%                          'role' field together with the parameters of the
%                          attribute the role creates, which is how a
%                          simplex-coded category is given its own sigma:
%                          struct('role', 'simplex', 'sigma', 0.2).
%
%                          Where a role fixes 'r' or 'exch' and a value is
%                          supplied too: under 'orderedMultiset' the
%                          supplied value is taken and a warning names what
%                          the role implies, the role having only arranged
%                          existing values into slots; under 'simplex' it
%                          is refused, the role having replaced the level
%                          with coordinates that denote a vertex only read
%                          whole and in order.
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
%            The pre-MAET is complete: every attribute carries the
%            parameters its density needs, so it is ready for buildMaet
%            without anything being set on the specs afterwards. The
%            conversion fills in only what follows from the data or from
%            another argument -- the values, r and exch under a structural
%            role, and reading a value as written for rel and isPer -- and
%            asks for the rest: sigma always, and r and exch where an
%            attribute holds more than one value at an event and no role
%            has fixed them.
%
%   See also PACKPREMAET, READSCORE, BUILDMAET, TRANSFORMATTRIBUTES,
%            FLATSPECS.

    arguments
        T table
        nvArgs.attributes = {}
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

    notes = T;
    if isempty(nvArgs.attributes)
        error('preMaetFromAttrTable:noAttributes', ...
              ['''attributes'' is required: a pre-MAET is its attributes, ' ...
               'and each carries parameters a score cannot supply. Write ' ...
               '{struct(''column'', ''pitch'', ''sigma'', 0.5), ...}.']);
    end
    [attributes, suppliedSpecs] = localEntries(nvArgs.attributes, 'attributes');
    known = {'pitch', 'onset', 'duration', 'soundingDuration', 'velocity', ...
             'weight', 'noteNumber', 'part', 'measure', 'fermata'};
    % The ten known names get the score-specific treatment -- the pitch
    % scale, beats against seconds, the grid's onset. Any other column of
    % the table is read as it stands, so a table that never saw a score
    % converts too. A known name is matched without regard to case and
    % then taken in its canonical spelling.
    extra = {};
    for i = 1:numel(attributes)
        k = find(strcmpi(attributes{i}, known), 1);
        if ~isempty(k)
            attributes{i} = known{k};
            continue;
        end
        if ~any(strcmp(notes.Properties.VariableNames, attributes{i}))
            error('preMaetFromAttrTable:attribute', ...
                  ['Unknown attribute ''%s'': it is neither one of the ' ...
                   'score attributes (%s) nor a column of the table.'], ...
                  attributes{i}, strjoin(known, ', '));
        end
        if iscategorical(notes.(attributes{i})) ...
                || iscellstr(notes.(attributes{i})) ...
                || isstring(notes.(attributes{i}))
            error('preMaetFromAttrTable:categoricalAttribute', ...
                  ['Attribute ''%s'' reads a categorical column, whose ' ...
                   'levels are not values on a line. Give it to ''roles'' ' ...
                   'instead, which says how a category reaches the ' ...
                   'pre-MAET.'], attributes{i});
        end
        extra{end + 1} = attributes{i}; %#ok<AGROW>
    end
    if ~any(strcmp(nvArgs.time, {'seconds', 'beats'}))
        error('preMaetFromAttrTable:time', 'time must be ''seconds'' or ''beats''.');
    end
    if ~any(strcmp(nvArgs.weights, {'velocity', 'ones', 'duration', 'weight'}))
        error('preMaetFromAttrTable:weights', ...
              'weights must be ''velocity'', ''ones'', ''duration'', or ''weight''.');
    end
    if ~any(strcmp(nvArgs.chords, {'bind', 'separate'}))
        error('preMaetFromAttrTable:chords', 'chords must be ''bind'' or ''separate''.');
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
    roleSpecs = struct();
    for i = 1:numel(roleNames)
        column = roleNames{i};
        [roleCell, roleSpecCell] = localEntries({nvArgs.roles.(column)}, 'roles');
        roleSpecs.(column) = roleSpecCell{1};
        k = find(strcmpi(roleCell{1}, allRoles), 1);
        if ~isempty(k); role = allRoles{k}; else; role = roleCell{1}; end
        if ~any(strcmp(role, allRoles))
            error('preMaetFromAttrTable:unknownRole', ...
                  'roles.%s: unknown role ''%s''; choose from %s.', ...
                  column, role, strjoin(allRoles, ', '));
        end
        if ~any(strcmp(notes.Properties.VariableNames, column))
            error('preMaetFromAttrTable:unknownColumn', ...
                  'roles names column ''%s'', which the table does not have.', ...
                  column);
        end
        if strcmp(role, 'drop'); continue; end
        if ~iscategorical(notes.(column))
            error('preMaetFromAttrTable:roleKind', ...
                  ['roles.%s: a role needs a categorical column, and %s ' ...
                   'is %s.'], column, column, class(notes.(column)));
        end
        if strcmp(role, 'simplex')
            simplexColumns{end + 1} = column; %#ok<AGROW>
            continue;
        end
        if ~isempty(structuralColumn)
            error('preMaetFromAttrTable:twoStructural', ...
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
        error('preMaetFromAttrTable:structuralNeedsBind', ...
              ['roles.%s is structural, which gathers the rows of an event ' ...
               'into one; that needs ''chords'', ''bind''.'], structuralColumn);
    end
    if ~isempty(simplexColumns) && isempty(structuralColumn) ...
            && strcmp(nvArgs.chords, 'bind')
        error('preMaetFromAttrTable:simplexNeedsSeparate', ...
              ['The ''simplex'' role carries the level as a value at each ' ...
               'event, so each concurrently-sounding note is its own event; ' ...
               'that needs ''chords'', ''separate'', or a structural ' ...
               'category to tag within.']);
    end

    vars = notes.Properties.VariableNames;
    if strcmp(nvArgs.time, 'seconds')
        unit = 'Seconds'; other = 'Beats';
    else
        unit = 'Beats'; other = 'Seconds';
    end
    % Of the ten names, these need a column the source carries. A table
    % that never saw a score carries few of them, and asks for none of
    % them, so each is read only where something names it.
    needs = {'soundingDuration', ['soundingDuration', unit]; ...
             'duration',         ['duration', unit]; ...
             'weight',           'weight'; ...
             'noteNumber',       'noteNumber'; ...
             'fermata',          'fermata'; ...
             'velocity',         'velocity'; ...
             'part',             'part'; ...
             'measure',          'measure'};
    for i = 1:size(needs, 1)
        if any(strcmp(needs{i, 1}, attributes)) && ~any(strcmp(vars, needs{i, 2}))
            error('preMaetFromAttrTable:missingColumn', ...
                  ['The table has no ''%s'' column, so ''%s'' cannot be an ' ...
                   'attribute; this source does not carry it.'], ...
                  needs{i, 2}, needs{i, 1});
        end
    end
    policies = {'weight', 'weight'; 'velocity', 'velocity'; ...
                'duration', ['duration', unit]};
    for i = 1:size(policies, 1)
        if strcmp(nvArgs.weights, policies{i, 1}) ...
                && ~any(strcmp(vars, policies{i, 2}))
            error('preMaetFromAttrTable:noWeight', ...
                  ['weights ''%s'' needs a ''%s'' column, which this ' ...
                   'source does not carry.'], policies{i, 1}, policies{i, 2});
        end
    end
    hasPart = any(strcmp(vars, 'part'));
    if ~isempty(nvArgs.parts) && ~hasPart
        error('preMaetFromAttrTable:missingColumn', ...
              ['''parts'' selects by part, and this source has no ''part'' ' ...
               'column. Selecting rows of the table before converting is ' ...
               'the more general route, and reaches any column.']);
    end

    if hasPart; partCodes = double(notes.part); else; partCodes = []; end
    keep = true(height(notes), 1);
    if ~isempty(nvArgs.parts)
        wanted = nvArgs.parts;
        if isnumeric(wanted)
            keep = keep & ismember(partCodes, wanted(:));
        else
            keep = keep & ismember(cellstr(notes.part), cellstr(wanted));
        end
    end
    % On a gridded table the event is the grid point, so its onset is the
    % grid's, not the onset of whichever note happens to be in the first
    % slot. The note's own onset stays in the table for selection.
    onsetName = ['onset', unit];
    if any(strcmp(vars, ['gridOnset', unit]))
        onsetName = ['gridOnset', unit];
    elseif any(strcmp(vars, ['gridOnset', other]))
        error('preMaetFromAttrTable:gridUnit', ...
              ['The table was gridded over %s, so time ''%s'' has no grid ' ...
               'onset to read; grid over %s or convert with that unit.'], ...
              lower(other), nvArgs.time, nvArgs.time);
    end
    optional = @(name) localOptional(notes, name, keep);
    onset = notes.(onsetName)(keep);
    dur = optional(['duration', unit]);
    soundingName = ['soundingDuration', unit];
    midi = notes.pitch(keep);
    vel = optional('velocity');
    if hasPart; part = partCodes(keep); else; part = zeros(sum(keep), 1); end
    measure = optional('measure');
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
    for i = 1:numel(extra)
        perNote.(extra{i}) = localOptional(notes, extra{i}, keep);
    end
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
        error('preMaetFromAttrTable:unknownColumn', ...
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
            warning('preMaetFromAttrTable:incompleteEvents', ...
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
    specSupplied = {};
    specFixed = {};

    for a = 1:numel(attributes)
        name = attributes{a};
        supplied = suppliedSpecs{a};
        if isfield(supplied, 'name'); base = supplied.name; else; base = name; end
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
                pAttr, wList, specR, specExch, specNames, M, W, 1, true, base);
            specSupplied{end + 1} = supplied; specFixed{end + 1} = '';
        elseif isEventLevel
            M = nan(1, N);
            for n = 1:N
                M(1, n) = vals(slots(n, 1));
            end
            [pAttr, wList, specR, specExch, specNames] = localAddAttr( ...
                pAttr, wList, specR, specExch, specNames, M, ones(1, N), ...
                1, true, base);
            specSupplied{end + 1} = supplied; specFixed{end + 1} = '';
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
                    sprintf('%s_%s', base, levels{v}));
                specSupplied{end + 1} = supplied; specFixed{end + 1} = '';
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
                pAttr, wList, specR, specExch, specNames, M, W, V, false, base);
            specSupplied{end + 1} = supplied;
            specFixed{end + 1} = 'orderedMultiset';
        end
    end

    % Simplex-coded categories. The coordinates of one level form one
    % ordered value read whole, so the attribute's tuple size is their
    % number and its weights are one: the note's own weight is carried by
    % its other attributes, and the attributes multiply.
    for i = 1:numel(simplexColumns)
        column = simplexColumns{i};
        supplied = roleSpecs.(column);
        if isfield(supplied, 'name'); base = supplied.name; else; base = column; end
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
                d, false, base);
            specSupplied{end + 1} = supplied; specFixed{end + 1} = 'simplex';
        else
            for v = 1:numel(levels)
                M = nan(d, N);
                for n = 1:N
                    c = codesS(slots(n, v));
                    if ~isnan(c); M(:, n) = vertices(c, :).'; end
                end
                [pAttr, wList, specR, specExch, specNames] = localAddAttr( ...
                    pAttr, wList, specR, specExch, specNames, M, ones(d, N), ...
                    d, false, sprintf('%s_%s', base, levels{v}));
                specSupplied{end + 1} = supplied;
                specFixed{end + 1} = 'simplex';
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
    for a = 1:numel(specs)
        specs{a} = localMergeSpec(specs{a}, specSupplied{a}, ...
                                  specFixed{a}, size(pAttr{a}, 1));
    end

    pm = packPreMaet(pAttr, w, specs);
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


function [subjects, supplied] = localEntries(entries, what)
    % Split each 'attributes' or 'roles' entry into its subject and the
    % per-attribute parameters it carries. A plain name is the subject
    % with no parameters, which is the form that predates the
    % spec-carrying one.
    fields = {'name', 'sigma', 'r', 'exch', 'rel', 'isPer', 'period'};
    if ~iscell(entries); entries = cellstr(entries); end
    subjects = cell(1, numel(entries));
    supplied = cell(1, numel(entries));
    if strcmp(what, 'attributes'); key = 'column'; else; key = 'role'; end
    for i = 1:numel(entries)
        e = entries{i};
        if ischar(e) || isstring(e)
            if strcmp(what, 'attributes')
                error('preMaetFromAttrTable:badEntry', ...
                      ['attributes.%s: an attribute is given as a struct ' ...
                       'of its column and its parameters, not as a bare ' ...
                       'name, since a name carries no sigma. Write ' ...
                       'struct(''column'', ''%s'', ''sigma'', ...).'], ...
                      char(e), char(e));
            end
            subjects{i} = char(e);
            supplied{i} = struct();
            continue;
        end
        if ~isstruct(e) || ~isscalar(e)
            error('preMaetFromAttrTable:badEntry', ...
                  ['An %s entry must be a name or a scalar struct; got ' ...
                   '%s.'], what, class(e));
        end
        if ~isfield(e, key)
            error('preMaetFromAttrTable:badEntry', ...
                  ['An %s entry given as a struct needs a ''%s'' field; ' ...
                   'got %s.'], what, key, strjoin(fieldnames(e).', ', '));
        end
        subjects{i} = char(e.(key));
        e = rmfield(e, key);
        unknown = setdiff(fieldnames(e).', fields);
        if ~isempty(unknown)
            error('preMaetFromAttrTable:badEntry', ...
                  '%s.%s: unknown parameter(s) %s; choose from %s.', ...
                  what, subjects{i}, strjoin(unknown, ', '), ...
                  strjoin(fields, ', '));
        end
        if isfield(e, 'isPer') && e.isPer && ~isfield(e, 'period')
            error('preMaetFromAttrTable:badEntry', ...
                  '%s.%s: isPer is set, so it needs a period.', ...
                  what, subjects{i});
        end
        supplied{i} = e;
    end
end


function spec = localMergeSpec(spec, supplied, fixed, K)
    % Fold one attribute's supplied parameters into the spec the
    % conversion built, and settle any conflict with a role.
    %
    % A role fixes r and exch for the attributes it governs. Under
    % 'orderedMultiset' the role only arranges existing values into
    % slots, so how many are drawn from them and whether their order
    % counts remain the analyst's questions and a supplied value wins,
    % with a warning. Under 'simplex' the role replaces the level with
    % the coordinates of a simplex vertex, which denote a vertex only
    % read whole and in order, so a supplied value is refused.
    roleR = spec.r;
    roleExch = spec.exch;
    hasR = isfield(supplied, 'r');
    hasExch = isfield(supplied, 'exch');

    switch fixed
      case 'simplex'
        if hasR && double(supplied.r) ~= roleR
            error('preMaetFromAttrTable:simplexTupleSize', ...
                  ['%s: the ''simplex'' role carries the level as the %d ' ...
                   'coordinates of a simplex vertex, read whole, so r ' ...
                   'must be %d and not %d. A tuple of some of a point''s ' ...
                   'coordinates is not a point. To compare runs of levels ' ...
                   'rather than one level, nest: bindAttributes then ' ...
                   'bindEvents, giving an outer tuple size over positions ' ...
                   '(r = (%d, 2) for pairs of levels in order). The ' ...
                   'grammar analysis of the JMM online supplement is the ' ...
                   'worked case.'], ...
                  spec.name, roleR, roleR, double(supplied.r), roleR);
        end
        if hasExch && supplied.exch
            error('preMaetFromAttrTable:simplexOrder', ...
                  ['%s: the ''simplex'' role''s coordinates are read in ' ...
                   'order, so exch must be false; permuting them gives a ' ...
                   'point that is not a vertex.'], spec.name);
        end
      case 'orderedMultiset'
        if hasR && double(supplied.r) ~= roleR
            warning('preMaetFromAttrTable:tupleSizeOverride', ...
                    ['%s: the ''orderedMultiset'' role fills %d slots, so ' ...
                     'it implies r = %d; taking the supplied r = %d, which ' ...
                     'reads tuples of %d of those slots.'], ...
                    spec.name, roleR, roleR, double(supplied.r), ...
                    double(supplied.r));
            spec.r = double(supplied.r);
        end
        if hasExch && logical(supplied.exch) ~= roleExch
            warning('preMaetFromAttrTable:orderOverride', ...
                    ['%s: the ''orderedMultiset'' role binds each value to ' ...
                     'its slot, so it implies exch = false; taking the ' ...
                     'supplied exch = true, which reads the slots as an ' ...
                     'unordered multiset and leaves nothing downstream ' ...
                     'reading the binding.'], spec.name);
            spec.exch = logical(supplied.exch);
        end
      otherwise
        if hasR; spec.r = double(supplied.r); end
        if hasExch; spec.exch = logical(supplied.exch); end
    end

    % A score reads its values as they are written -- absolute, on an
    % unbounded axis -- so rel and isPer are false unless the analyst
    % says otherwise; octave equivalence is an equivalence imposed, not
    % one the score states.
    if isfield(supplied, 'rel'); spec.rel = logical(supplied.rel); end
    if isfield(supplied, 'isPer')
        spec.isPer = logical(supplied.isPer);
    else
        spec.isPer = false;
    end
    if isfield(supplied, 'period')
        spec.period = double(supplied.period);
    else
        spec.period = 0;
    end
    if ~isfield(supplied, 'sigma')
        error('preMaetFromAttrTable:noSigma', ...
              ['%s: no sigma. A score fixes what the values are and not ' ...
               'how tolerant a match is, so every attribute needs one; ' ...
               'there is no width to default to, sigma = 0 being a real ' ...
               'and degenerate choice rather than an absence.'], spec.name);
    end
    spec.sigma = double(supplied.sigma);

    % r and exch are claims about what an attribute's values mean, not
    % transformations with an off position, so neither has an identity to
    % default to. Where the attribute holds one value per event both are
    % determined -- r = 1, and exch says nothing -- and neither need be
    % given; where a role fixes them it has already answered for the
    % analyst.
    if K > 1 && isempty(fixed)
        missing = {};
        if ~hasR; missing{end + 1} = 'r'; end
        if ~hasExch; missing{end + 1} = 'exch'; end
        if ~isempty(missing)
            error('preMaetFromAttrTable:noTupleSize', ...
                  ['%s: this attribute holds %d values at an event, so it ' ...
                   'needs %s. r says how many of them a tuple takes, and ' ...
                   'exch whether their order signifies; neither follows ' ...
                   'from the score.'], spec.name, K, strjoin(missing, ' and '));
        end
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
