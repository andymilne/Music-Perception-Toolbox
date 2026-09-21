function G = gridEvents(T, step, nvArgs)
%GRIDEVENTS  Sample an event table on a regular grid.
%
%   G = gridEvents(T, step)
%   G = gridEvents(T, step, Name, Value, ...)
%
%   Each grid point opens a slice of the chosen length, and a note belongs
%   to every slice it overlaps, so a held note replicates across the slices
%   it spans and a note shorter than the step still lands in one. A slice
%   with nothing sounding becomes one row whose note columns are all
%   missing. Those empty rows are kept: they hold the place that makes the
%   event index a uniform index of time, which is what binding and
%   differencing read. Dropping them afterwards is one selection away --
%   G(~isnan(G.noteId), :).
%
%   Name-value pairs
%     'time'      - 'beats' (default) or 'seconds': which time base the
%                   grid runs over. A metrical grid presupposes a beat
%                   map, which a score has and a bare performance may not.
%     'duration'  - 'duration' (default) or 'soundingDuration': which
%                   duration defines occupancy, the recorded one or the
%                   one with the pedals resolved.
%     'weights'   - what a slice takes from a note that overlaps it.
%                   'coverage' (default) takes the fraction of the slice
%                   the note fills: how the span is filled. This is the
%                   weighting of Analysis 1.3 of the JMM article, 'the
%                   fraction of the eighth each note sounds'. 'presence'
%                   takes the note's full weight in every slice it appears
%                   in at all, however briefly: which notes are here,
%                   rather than how much of the span each occupies. It is
%                   a membership reading: each slice records the set of
%                   what occurs in it, whatever the step. It separates
%                   from coverage as the step grows relative to the notes
%                   -- at a bar-length step, say, a slice holds the set of
%                   what occurs in that bar, where coverage would hold a
%                   duration-weighted profile of it.
%                   'item' takes the fraction of the note in the slice, so
%                   that its weight is distributed over the slices it
%                   spans and it counts once in total; for an attribute
%                   constant over the note this gives a density identical
%                   to the ungridded one at r = 1, the kernel being linear
%                   in weight.
%
%                   Coverage and presence differ only where a note does
%                   not fill a slice, so on a grid at or finer than the
%                   shortest note they agree. For what is sounding at a
%                   given moment, the instrument is coverage on a fine
%                   grid: an instant has no duration, and coverage
%                   approaches the momentary reading as the step shrinks.
%     'limits'    - [lo hi], the half-open span the grid covers. The
%                   default runs from 0 to the last note's end.
%
%   The result carries T's columns, with duration still meaning the note's
%   own duration and the slice length being a property of the grid, plus
%
%       gridIndex                       1-based position of the grid point
%       gridOnsetBeats or               the grid point's time, named for
%       gridOnsetSeconds                the unit gridded over
%       noteId                          1-based row of T, NaN on an empty
%                                       point, so that the grid collapses
%                                       back to the note table by grouping
%                                       and nothing is lost
%       weight                          under the chosen policy; where T
%                                       carries no weight of its own,
%                                       every note weighs one
%
%   A logical column becomes double, MATLAB having no missing logical and
%   an empty grid point having no value to give it. G.Properties.UserData
%   records the granularity, the step, and the time base.
%
%   Example
%       T = readScore('score.mid');
%       G = gridEvents(T, 0.25);            % sixteenth notes
%       pm = preMaetFromScore(G, 'attributes', {'pitch'}, ...
%                             'weights', 'weight', 'time', 'beats');
%
%   See also READSCORE, PREMAETFROMSCORE.

    arguments
        T table
        step (1,1) double
        nvArgs.time (1,:) char = 'beats'
        nvArgs.duration (1,:) char = 'duration'
        nvArgs.weights (1,:) char = 'coverage'
        nvArgs.limits double = []
    end

    tol = 1e-9;   % occupancy is an overlap with the slice, so a note
                  % ending exactly where a slice begins does not occupy it
    if ~any(strcmp(nvArgs.time, {'beats', 'seconds'}))
        error('gridEvents:time', 'time must be ''beats'' or ''seconds''.');
    end
    if ~any(strcmp(nvArgs.duration, {'duration', 'soundingDuration'}))
        error('gridEvents:duration', ...
              'duration must be ''duration'' or ''soundingDuration''.');
    end
    if ~any(strcmp(nvArgs.weights, {'coverage', 'presence', 'item'}))
        error('gridEvents:weights', ...
              'weights must be ''coverage'', ''presence'', or ''item''.');
    end
    if ~isfinite(step) || step <= 0
        error('gridEvents:step', 'step must be a positive, finite number.');
    end

    unit = [upper(nvArgs.time(1)), nvArgs.time(2:end)];
    onsetCol = ['onset', unit];
    durCol   = [nvArgs.duration, unit];
    vars = T.Properties.VariableNames;
    for name = {onsetCol, durCol}
        if ~any(strcmp(vars, name{1}))
            error('gridEvents:missingColumn', ...
                  ['The table has no ''%s'' column, so it cannot be ' ...
                   'gridded over %s; a metrical grid needs a beat map, ' ...
                   'which a bare performance may not have.'], ...
                  name{1}, nvArgs.time);
        end
    end

    onset  = double(T.(onsetCol));
    finish = onset + double(T.(durCol));
    if isempty(nvArgs.limits)
        lo = 0;
        if isempty(finish); hi = 0; else; hi = max(finish); end
    else
        if numel(nvArgs.limits) ~= 2 || ~(nvArgs.limits(2) > nvArgs.limits(1))
            error('gridEvents:limits', 'limits must be [lo hi] and increasing.');
        end
        lo = nvArgs.limits(1); hi = nvArgs.limits(2);
    end
    if hi > lo
        nPoints = ceil((hi - lo) / step - tol);
    else
        nPoints = 0;
    end
    times = lo + step * (0:nPoints - 1);

    % --- Occupancy ------------------------------------------------------
    gridOf = zeros(0, 1);
    noteOf = zeros(0, 1);
    overlapOf = zeros(0, 1);
    for i = 1:nPoints
        g = times(i);
        overlap = min(finish, g + step) - max(onset, g);
        occ = find(overlap > tol);
        if isempty(occ)
            gridOf(end + 1, 1) = i; %#ok<AGROW>
            noteOf(end + 1, 1) = 0; %#ok<AGROW>
            overlapOf(end + 1, 1) = NaN; %#ok<AGROW>
        else
            gridOf(end + 1:end + numel(occ), 1) = i; %#ok<AGROW>
            noteOf(end + 1:end + numel(occ), 1) = occ(:); %#ok<AGROW>
            overlapOf(end + 1:end + numel(occ), 1) = overlap(occ); %#ok<AGROW>
        end
    end
    live = noteOf > 0;
    src = noteOf;
    src(~live) = 1;

    G = T(src, :);
    G = localBlankEmptyRows(G, live);

    % --- Weights --------------------------------------------------------
    if any(strcmp(vars, 'weight'))
        base = double(T.weight);
    else
        base = ones(height(T), 1);
    end
    noteDur = double(T.(durCol));
    weight = nan(numel(noteOf), 1);
    switch nvArgs.weights
        case 'presence'
            weight(live) = base(noteOf(live));
        case 'coverage'
            weight(live) = base(noteOf(live)) .* overlapOf(live) / step;
        otherwise
            weight(live) = base(noteOf(live)) .* overlapOf(live) ...
                           ./ noteDur(noteOf(live));
    end

    noteId = noteOf;
    noteId(~live) = NaN;
    G.gridIndex = gridOf;
    G.(['gridOnset', unit]) = reshape(times(gridOf), [], 1);
    G.noteId = noteId;
    G.weight = weight;
    G.Properties.UserData = struct('granularity', 'grid', ...
                                   'gridStep', step, ...
                                   'gridTime', nvArgs.time);
end


function G = localBlankEmptyRows(G, live)
    % An empty grid point has no note, so every note column is missing
    % there. A logical column becomes double first, MATLAB having no
    % missing logical; it does so whether or not this grid has an empty
    % point, so that the result's types do not depend on its content.
    vars = G.Properties.VariableNames;
    for k = 1:numel(vars)
        col = G.(vars{k});
        if islogical(col)
            col = double(col);
            G.(vars{k}) = col;
        end
        if all(live); continue; end
        if isnumeric(col)
            col(~live) = NaN;
        elseif iscategorical(col)
            col(~live) = missing;
        elseif iscellstr(col) %#ok<ISCLSTR>
            col(~live) = {''};
        elseif isstring(col)
            col(~live) = missing;
        else
            continue
        end
        G.(vars{k}) = col;
    end
end
