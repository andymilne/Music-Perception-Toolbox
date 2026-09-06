function [pAttr, w, specs] = eventsFromScore(source, nvArgs)
%EVENTSFROMSCORE  Build the (pAttr, w, specs) carrier from a score.
%
%   [pAttr, w, specs] = eventsFromScore(source, ...)
%
%   source is a file path (parsed with readScore: MIDI, MusicXML, or .mxl)
%   or a note table from readScore. The output is the carrier that
%   buildExpTens and the pre-MAET preprocessors consume.
%
%   Name-value pairs
%       'attributes'     - cell of names from {'pitch', 'onset', 'duration',
%                          'velocity', 'part', 'measure'}, in order
%                          (default {'pitch', 'onset'}).
%       'pitch'          - pitch scale: 'midi' (default), 'cents', 'hz',
%                          'octave', or any pitch scale of transformAttributes.
%       'time'           - 'seconds' (default) or 'beats' (quarter notes)
%                          for onsets and durations.
%       'weights'        - 'velocity' (default; velocity / 127), 'ones', or
%                          'duration' (in the chosen time unit).
%       'parts'          - [] (all) or the 1-based parts to keep.
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
%   Outputs
%       pAttr - 1 x A cell of K_a x N value matrices.
%       w     - 1 x A cell of K_a x N weight matrices (NaN-padded slots
%               carry weight 0), or [] under 'ones'.
%       specs - 1 x A cell of flat specs, named after the attributes.
%
%   See also READSCORE, BUILDEXPTENS, TRANSFORMATTRIBUTES, FLATSPECS.

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
    allowed = {'pitch', 'onset', 'duration', 'velocity', 'part', 'measure'};
    for i = 1:numel(attributes)
        if ~any(strcmp(attributes{i}, allowed))
            error('eventsFromScore:attribute', ...
                  'Unknown attribute ''%s''; choose from %s.', attributes{i}, strjoin(allowed, ', '));
        end
    end
    if ~any(strcmp(nvArgs.time, {'seconds', 'beats'}))
        error('eventsFromScore:time', 'time must be ''seconds'' or ''beats''.');
    end
    if ~any(strcmp(nvArgs.weights, {'velocity', 'ones', 'duration'}))
        error('eventsFromScore:weights', 'weights must be ''velocity'', ''ones'', or ''duration''.');
    end
    if ~any(strcmp(nvArgs.chords, {'bind', 'separate'}))
        error('eventsFromScore:chords', 'chords must be ''bind'' or ''separate''.');
    end

    keep = true(numel(notes.pitch), 1);
    if ~isempty(nvArgs.parts)
        keep = keep & ismember(notes.part(:), nvArgs.parts(:));
    end
    if strcmp(nvArgs.time, 'seconds')
        onset = notes.onsetSeconds(keep); dur = notes.durationSeconds(keep);
    else
        onset = notes.onsetBeats(keep); dur = notes.durationBeats(keep);
    end
    midi = notes.pitch(keep);
    vel = notes.velocity(keep);
    part = notes.part(keep);
    measure = notes.measure(keep);
    nNotes = numel(midi);

    if strcmpi(nvArgs.pitch, 'midi')
        pitchVals = midi;
    else
        pitchVals = transformAttributes(midi, [], {'midi', nvArgs.pitch});
    end
    perNote = struct('pitch', pitchVals, 'onset', onset, 'duration', dur, ...
                     'velocity', vel, 'part', part, 'measure', measure);
    switch nvArgs.weights
        case 'velocity', wNote = vel / 127;
        case 'duration', wNote = dur;
        otherwise,       wNote = [];
    end

    % Group notes into events.
    if strcmp(nvArgs.chords, 'separate') || nNotes == 0
        groups = num2cell(1:nNotes);
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
end
