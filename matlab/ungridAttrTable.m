function T = ungridAttrTable(G)
%UNGRIDATTRTABLE  Collapse a gridded attribute table back to its source.
%
%   T = ungridAttrTable(G)
%
%   The inverse of gridAttrTable. noteId names the row of the source each
%   grid row came from, so keeping the first row of each and removing the
%   columns the grid wrote returns the source table.
%
%   Which columns those are is the toolbox's business rather than the
%   caller's, and it is not a fixed list: the grid ADDS weight to a table
%   that had none and OVERWRITES the weight of one that did, so a caller
%   comparing the two tables' columns would keep a weight whose values
%   are no longer the note's but the slice's.
%
%   Three things do not come back. A note that 'limits' cut out of the
%   grid's span is gone, which is a truncation the caller asked for. The
%   source's own weight column, where it had one, is gone: the grid
%   folded it into a per-slice weight, so read the source again if you
%   need it. And a logical column comes back as double, MATLAB having no
%   missing logical for the grid to have given an empty point; Python's
%   nullable boolean round-trips, which is an idiomatic difference and
%   not a difference in the construction.
%
%   Rows and columns may go before ungridding: the empty points, a run
%   of slices, a column. A note whose first slice was dropped comes back
%   from its next, since the source's columns repeat unchanged across its
%   slices; a note whose slices were all dropped does not come back at
%   all, having been selected away. Where a kept column takes more than
%   one value within a note -- which only a column the caller added can
%   do -- the first slice's value is taken, with a warning.
%
%   See also GRIDATTRTABLE, READSCORE, PREMAETFROMATTRTABLE.

    arguments
        G table
    end

    if ~any(strcmp(G.Properties.VariableNames, 'noteId'))
        error('ungridAttrTable:notGridded', ...
              ['The table has no ''noteId'' column, so it did not come ' ...
               'from gridAttrTable and there is no grid to undo.']);
    end

    live = G(~isnan(G.noteId), :);
    [~, first] = unique(live.noteId);
    T = live(first, :);

    written = {'gridIndex', 'gridOnsetBeats', 'gridOnsetSeconds', ...
               'noteId', 'weight'};
    kept = setdiff(live.Properties.VariableNames, written, 'stable');
    T = removevars(T, intersect(T.Properties.VariableNames, written));
    T.Properties.RowNames = {};

    % A column the grid wrote is gone; a column of the source repeats
    % unchanged across a note's slices. One the caller added may not: a
    % per-slice quantity has no single value to collapse to, and the
    % first slice's is taken.
    if ~isempty(kept) && height(live) > 0
        grp = findgroups(live.noteId);
        [~, firstOfGroup] = unique(grp);
        varying = {};
        for i = 1:numel(kept)
            if localVaries(live.(kept{i}), grp, firstOfGroup)
                varying{end + 1} = kept{i}; %#ok<AGROW>
            end
        end
        if ~isempty(varying)
            warning('ungridAttrTable:variesWithinNote', ...
                    ['%s vary within a note, so they are not properties ' ...
                     'of the note the grid came from; the first slice''s ' ...
                     'value is taken. A per-slice quantity is lost by ' ...
                     'ungridding, which is what ungridding means.'], ...
                    strjoin(varying, ', '));
        end
    end

    meta = G.Properties.UserData;
    if isstruct(meta)
        for f = {'granularity', 'gridStep', 'gridTime', 'gridWeights'}
            if isfield(meta, f{1}); meta = rmfield(meta, f{1}); end
        end
        T.Properties.UserData = meta;
    end
end


function tf = localVaries(v, grp, firstOfGroup)
    % Whether a column takes more than one value within some note, with
    % missing counted as equal to missing.
    tf = false;
    f = v(firstOfGroup(grp), :);
    if isnumeric(v) || islogical(v) || iscategorical(v) || isstring(v) ...
            || isduration(v) || isdatetime(v)
        same = (v == f);
        if any(ismissing(v)) || any(ismissing(f))
            same = same | (ismissing(v) & ismissing(f));
        end
        tf = ~all(same(:));
    elseif iscellstr(v)
        tf = ~all(strcmp(v, f));
    end
    % A type with no comparison says nothing rather than guessing.
end
