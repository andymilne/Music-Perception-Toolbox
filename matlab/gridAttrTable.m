function G = gridAttrTable(T, step, nvArgs)
%GRIDATTRTABLE  Sample an attribute table on a regular grid.
%
%   G = gridAttrTable(T, step)
%   G = gridAttrTable(T, step, Name, Value, ...)
%
%   The grid is a series of time points spaced step apart. Each point
%   opens a SLICE, reaching from it up to the next, and a note
%   occupies every slice it sounds in, so a held note occupies several and
%   a note shorter than the step still occupies one. A note that ends
%   exactly where a slice begins does not occupy it.
%
%   The points are equally spaced in the unit the grid steps in -- in
%   beats for a beat grid. Under a changing tempo that is not equal
%   spacing in seconds, so the slices last different amounts of clock
%   time.
%
%   A slice with nothing sounding becomes one row whose note columns are
%   all missing. Those empty rows are kept: they hold the place that makes
%   the event index a uniform index of time, which is what binding and
%   differencing read. Dropping them afterwards is one selection away --
%   G(~isnan(G.noteId), :).
%
%   An already-gridded table regrids at a coarser step, and gridding
%   composes: gridAttrTable(gridAttrTable(T, fine), coarse) equals
%   gridAttrTable(T, coarse). This is how a weighting the toolbox does not
%   itself offer is reached -- weight the fine slices as the analysis
%   requires, then coarsen, and the weights carry. The coarse step must be
%   a whole multiple of the fine one, the time base must be the one the
%   table was gridded over, 'weights' must name the policy the table's
%   weights already carry, since that is what fixes how they combine, and
%   'limits' is refused, the span being the one the table covers.
%
%   A grid does not refine. Coarsening combines the fine slices, which is
%   determined; refining would have to divide a slice's weight among finer
%   ones, which is not -- and a weighting the caller wrote into the grid is
%   exactly what would have to be divided. The route is to grid the notes
%   again, gridAttrTable(ungridAttrTable(G), finer), which says plainly
%   that the grid is being left behind.
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
%                   the note fills: how the slice is filled. This is the
%                   weighting of Analysis 1.3 of the JMM article, 'the
%                   fraction of the eighth each note sounds'. 'presence'
%                   takes the note's full weight in every slice it appears
%                   in at all, however briefly: which notes are here,
%                   rather than how much of the slice each occupies. It is
%                   a membership reading: each slice records the set of
%                   what occurs in it, whatever the step. It separates
%                   from coverage as the step grows relative to the notes
%                   -- at a bar-length step, say, a slice holds the set of
%                   what occurs in that bar, where coverage would hold a
%                   duration-weighted profile of it.
%                   'item' takes the fraction of the note in the slice, so
%                   that its weight is distributed over the slices it
%                   covers and it counts once in total; for an attribute
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
%       gridOnsetBeats and              the grid point's time in each
%       gridOnsetSeconds                unit. The grid steps in the chosen
%                                       one; the other is interpolated
%                                       from the table's note samples (one
%                                       pair per onset, one per note end)
%                                       and continued at the nearest rate
%                                       beyond the first and last, so that
%                                       metrical slices can be read on a
%                                       clock -- a sigma in milliseconds
%                                       over a grid of sixteenths. That is
%                                       exact wherever the tempo is
%                                       constant across the bracketing
%                                       samples and approximate only
%                                       across a tempo change inside a
%                                       gap. Only the stepped unit is
%                                       present where the source carries
%                                       no second time base.
%       noteId                          1-based row of T, NaN on an empty
%                                       point, so that the grid collapses
%                                       back to the table it came from and
%                                       nothing is lost
%       weight                          under the chosen policy; where T
%                                       carries no weight of its own,
%                                       every note weighs one
%
%   A logical column becomes double, MATLAB having no missing logical and
%   an empty grid point having no value to give it. G.Properties.UserData
%   records the granularity, the step, and the time base.
%
%   ungridAttrTable is the inverse, collapsing the grid back to the table
%   it came from.
%
%   Example
%       T = readScore('score.mid');
%       G = gridAttrTable(T, 0.25);            % sixteenth notes
%       pm = preMaetFromAttrTable(G, 'attributes', {'pitch'}, ...
%                             'weights', 'weight', 'time', 'beats');
%
%   See also UNGRIDATTRTABLE, READSCORE, PREMAETFROMATTRTABLE.

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
        error('gridAttrTable:time', 'time must be ''beats'' or ''seconds''.');
    end
    if ~any(strcmp(nvArgs.duration, {'duration', 'soundingDuration'}))
        error('gridAttrTable:duration', ...
              'duration must be ''duration'' or ''soundingDuration''.');
    end
    if ~any(strcmp(nvArgs.weights, {'coverage', 'presence', 'item'}))
        error('gridAttrTable:weights', ...
              'weights must be ''coverage'', ''presence'', or ''item''.');
    end
    if ~isfinite(step) || step <= 0
        error('gridAttrTable:step', 'step must be a positive, finite number.');
    end

    ud = T.Properties.UserData;
    if isstruct(ud) && isfield(ud, 'granularity') && ...
            strcmp(ud.granularity, 'grid')
        G = localCoarsen(T, step, nvArgs, tol);
        return
    end

    unit = [upper(nvArgs.time(1)), nvArgs.time(2:end)];
    onsetCol = ['onset', unit];
    durCol   = [nvArgs.duration, unit];
    vars = T.Properties.VariableNames;
    for name = {onsetCol, durCol}
        if ~any(strcmp(vars, name{1}))
            error('gridAttrTable:missingColumn', ...
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
            error('gridAttrTable:limits', 'limits must be [lo hi] and increasing.');
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
    if strcmp(nvArgs.time, 'beats'); other = 'seconds'; else; other = 'beats'; end
    otherTimes = localOtherUnit(T, nvArgs.time, other, times);
    if ~isempty(otherTimes)
        otherUnit = [upper(other(1)), other(2:end)];
        G.(['gridOnset', otherUnit]) = reshape(otherTimes(gridOf), [], 1);
    end
    G.noteId = noteId;
    G.weight = weight;
    G.Properties.UserData = struct('granularity', 'grid', ...
                                   'gridStep', step, ...
                                   'gridTime', nvArgs.time, ...
                                   'gridWeights', nvArgs.weights);
end



function G = localCoarsen(T, step, nvArgs, tol)
    %localCoarsen Regrid an already-gridded table at a coarser step.
    %
    % A gridded row records that a note sounds in a slice, with a weight;
    % the note's own onset and duration are carried through unchanged, so
    % reading them again would re-expand the note once per row it already
    % has. Coarsening instead composes: the fine slices falling in one
    % coarse slice are combined, note by note, and each policy combines in
    % the way that makes gridding the grid agree with gridding the notes.
    %
    % 'coverage' sums and rescales by fine / coarse, coverage being a
    % fraction of the slice and the slice having grown. 'item' sums, its
    % normalization by the note's length making it additive already.
    % 'presence' takes the maximum, an indicator over the coarse slice
    % holding wherever it holds over a fine one.
    ud = T.Properties.UserData;
    fineStep = double(ud.gridStep);
    if isfield(ud, 'gridTime'); fineTime = ud.gridTime; else; fineTime = nvArgs.time; end
    if ~strcmp(nvArgs.time, fineTime)
        error('gridAttrTable:regridTime', ...
              ['This table was gridded over %s, so it cannot be regridded ' ...
               'over %s; ungrid it first to change the time base.'], ...
              fineTime, nvArgs.time);
    end
    if isfield(ud, 'gridWeights') && ~strcmp(nvArgs.weights, ud.gridWeights)
        error('gridAttrTable:regridWeights', ...
              ['This table''s weights are ''%s'', which is how they ' ...
               'combine; regridding it under ''%s'' would read them as ' ...
               'something they are not.'], ud.gridWeights, nvArgs.weights);
    end
    if ~isempty(nvArgs.limits)
        error('gridAttrTable:regridLimits', ...
              ['limits cannot be given when regridding: the span is the ' ...
               'one the table already covers.']);
    end
    ratio = step / fineStep;
    if ratio < 1 - tol
        error('gridAttrTable:regridFiner', ...
              ['step %g is finer than this table''s step %g. Coarsening ' ...
               'combines the fine slices, which is determined; refining ' ...
               'would have to divide a slice''s weight among finer ones, ' ...
               'which is not. Grid the notes again instead: ' ...
               'gridAttrTable(ungridAttrTable(G), %g).'], ...
              step, fineStep, step);
    end
    if abs(ratio - round(ratio)) > tol
        error('gridAttrTable:regridStep', ...
              ['step %g is not a whole multiple of this table''s step %g, ' ...
               'so the coarse slices would not align with the fine ones.'], ...
              step, fineStep);
    end
    ratio = round(ratio);

    unit = [upper(nvArgs.time(1)), nvArgs.time(2:end)];
    onsetCol = ['gridOnset', unit];
    fineIdx = double(T.gridIndex);
    if isempty(fineIdx); nFine = 0; else; nFine = max(fineIdx); end
    nCoarse = ceil(nFine / ratio);
    if ratio == 1 || nCoarse == 0
        G = T;
        ud.gridStep = step;
        ud.gridWeights = nvArgs.weights;
        G.Properties.UserData = ud;
        return
    end

    lo = min(T.(onsetCol));
    coarseOf = floor((fineIdx - 1) / ratio) + 1;
    coarseTime = lo + step * (0:nCoarse - 1).';

    if strcmp(nvArgs.time, 'beats'); other = 'seconds'; else; other = 'beats'; end
    otherCol = ['gridOnset', upper(other(1)), other(2:end)];
    hasOther = any(strcmp(T.Properties.VariableNames, otherCol));
    coarseOther = [];
    if hasOther
        % The coarse point is the first fine point of its slice, so it
        % already has a time in the other unit and none is interpolated.
        coarseOther = nan(nCoarse, 1);
        for c = 1:nCoarse
            k = find(fineIdx == (c - 1) * ratio + 1, 1);
            if ~isempty(k); coarseOther(c) = T.(otherCol)(k); end
        end
    end

    live = ~isnan(double(T.noteId));
    srcRow = find(live);
    [pairs, ~, bin] = unique([coarseOf(live), double(T.noteId(live))], ...
                             'rows', 'stable');
    nPairs = size(pairs, 1);
    w = double(T.weight(live));
    switch nvArgs.weights
        case 'presence'
            combined = accumarray(bin, w, [nPairs 1], @max);
        case 'coverage'
            combined = accumarray(bin, w, [nPairs 1]) * (fineStep / step);
        otherwise
            combined = accumarray(bin, w, [nPairs 1]);
    end
    firstRow = accumarray(bin, srcRow, [nPairs 1], @min);

    % A coarse slice holding no note keeps its place as one blank row.
    filled = false(nCoarse, 1);
    filled(pairs(:, 1)) = true;
    blanks = find(~filled);
    nBlank = numel(blanks);
    rowsOf    = [firstRow;     ones(nBlank, 1)];
    coarseIdx = [pairs(:, 1);  blanks];
    noteOut   = [pairs(:, 2);  nan(nBlank, 1)];
    wOut      = [combined;     nan(nBlank, 1)];
    isLive    = [true(nPairs, 1); false(nBlank, 1)];

    [~, order] = sortrows([coarseIdx, noteOut]);
    rowsOf = rowsOf(order); coarseIdx = coarseIdx(order);
    noteOut = noteOut(order); wOut = wOut(order); isLive = isLive(order);

    G = T(rowsOf, :);
    G = localBlankEmptyRows(G, isLive);
    G.gridIndex = coarseIdx;
    G.(onsetCol) = coarseTime(coarseIdx);
    if hasOther; G.(otherCol) = coarseOther(coarseIdx); end
    G.noteId = noteOut;
    G.weight = wOut;
    ud.gridStep = step;
    ud.gridWeights = nvArgs.weights;
    G.Properties.UserData = ud;
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


function out = localOtherUnit(T, time, other, times)
    % The grid points' times in the unit the grid did not step in.
    %
    % The grid steps in one unit, but its points have a time in both, and
    % an analysis may want metrical slices read on a clock. The table
    % carries the correspondence only as samples, one pair per note onset
    % and one per note end, so the map is interpolated linearly between
    % them and continued at the nearest rate beyond the first and last.
    % Empty where the source carries only one time base, a bare
    % performance having no beat map to read.
    out = [];
    have = ['onset', upper(time(1)), time(2:end)];
    want = ['onset', upper(other(1)), other(2:end)];
    haveDur = ['duration', upper(time(1)), time(2:end)];
    wantDur = ['duration', upper(other(1)), other(2:end)];
    cols = T.Properties.VariableNames;
    if ~all(ismember({have, want, haveDur, wantDur}, cols)); return; end

    x = [T.(have); T.(have) + T.(haveDur)];
    y = [T.(want); T.(want) + T.(wantDur)];
    good = isfinite(x) & isfinite(y);
    x = x(good);
    y = y(good);
    if isempty(x); return; end
    [x, order] = sort(x);
    y = y(order);
    [x, first] = unique(x);
    y = y(first);
    if numel(x) == 1
        out = repmat(y(1), size(times));
        return;
    end

    out = interp1(x, y, times, 'linear');
    % interp1 leaves NaN outside the samples; continue the end rates
    % instead, so that a slice before the first note or after the last
    % still has a time rather than none.
    loRate = (y(2) - y(1)) / (x(2) - x(1));
    hiRate = (y(end) - y(end - 1)) / (x(end) - x(end - 1));
    below = times < x(1);
    above = times > x(end);
    out(below) = y(1) + loRate * (times(below) - x(1));
    out(above) = y(end) + hiRate * (times(above) - x(end));
end
