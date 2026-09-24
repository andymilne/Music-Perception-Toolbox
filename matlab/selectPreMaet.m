function pm = selectPreMaet(pAttr, wAttr, nvArgs)
%SELECTPREMAET  Keep a selection of a pre-MAET's attributes and events.
%
%   pm = selectPreMaet(pm, 'attributes', ..., 'events', ...)
%   pm = selectPreMaet(pAttr, wAttr, ...)
%
%   Pre-MAET preprocessing that knows nothing of where the pre-MAET came
%   from: it reads the two levels every pre-MAET has, its attributes and
%   its events, and nothing else. That is the point of it -- a filter on
%   the object itself, as against selecting rows of the table it may have
%   been built from, which is MATLAB's own job.
%
%   Name-value pairs
%     'attributes' - the attributes to keep: indices, names as the specs
%                    carry them, or a logical mask of length A. [] keeps
%                    all.
%     'events'     - the events to keep: indices or a logical mask of
%                    length N. [] keeps all. A predicate is applied by
%                    the caller, which reads the values it wants from
%                    pAttr and passes the mask.
%     'specs'      - the attribute specifications; [] synthesises flat
%                    ones.
%
%   The kept attributes and events come back in the order given.
%   Attributes keep their tuple sizes and flags, so a selection cannot
%   change what an attribute means; an attribute holding several
%   coordinates of one value moves whole, because it is one attribute and
%   not several.
%
%   Selecting events may leave an attribute with no value at some kept
%   event. That is allowed and means what it says: the event contributes
%   nothing on that attribute while keeping its place in the sequence.
%
%   See also PACKPREMAET, BUILDMAET, BINDEVENTS, DIFFERENCEEVENTS,
%            WEIGHTEVENTS.

    arguments
        pAttr
        wAttr = []
        nvArgs.attributes = []
        nvArgs.events = []
        nvArgs.specs = []
    end

    specs = nvArgs.specs;
    if internal.isPreMaet(pAttr)
        pm0 = pAttr;
        pAttr = pm0.pAttr;
        if isempty(specs); specs = pm0.specs; end
        if nargin < 2 || isempty(wAttr); wAttr = pm0.wAttr; end
    end
    if ~iscell(pAttr)
        error('selectPreMaet:badPAttr', ...
              'pAttr must be a cell of per-attribute value matrices.');
    end
    A = numel(pAttr);
    if A == 0
        error('selectPreMaet:noAttributes', ...
              'pAttr must contain at least one attribute.');
    end
    N = size(pAttr{1}, 2);
    if isempty(specs)
        specs = flatSpecs(pAttr);
    elseif numel(specs) ~= A
        error('selectPreMaet:specsLength', ...
              'specs must be a length-A (%d) cell, one per attribute.', A);
    end

    names = cell(1, A);
    for a = 1:A
        if isfield(specs{a}, 'name'); names{a} = specs{a}.name; end
    end
    aKeep = localIndices(nvArgs.attributes, A, 'attribute', names);
    if isempty(aKeep)
        error('selectPreMaet:emptySelection', ...
              'The selection keeps no attribute; a pre-MAET has at least one.');
    end
    nKeep = localIndices(nvArgs.events, N, 'event', {});

    pOut = cell(1, numel(aKeep));
    wOut = [];
    if ~isempty(wAttr); wOut = cell(1, numel(aKeep)); end
    specsOut = cell(1, numel(aKeep));
    for i = 1:numel(aKeep)
        a = aKeep(i);
        pOut{i} = pAttr{a}(:, nKeep);
        if ~isempty(wAttr); wOut{i} = wAttr{a}(:, nKeep); end
        specsOut{i} = specs{a};
    end
    pm = packPreMaet(pOut, wOut, specsOut);
end


function idx = localIndices(selector, n, what, names)
    % Resolve a selector to an index vector over n positions.
    if isempty(selector) && ~islogical(selector)
        idx = 1:n;
        return;
    end
    if islogical(selector)
        if numel(selector) ~= n
            error('selectPreMaet:maskLength', ...
                  'A logical %s mask must have length %d; got %d.', ...
                  what, n, numel(selector));
        end
        idx = find(selector(:).');
        return;
    end
    if ischar(selector) || isstring(selector) || iscell(selector)
        selector = cellstr(selector);
        idx = zeros(1, numel(selector));
        for i = 1:numel(selector)
            k = find(strcmp(names, selector{i}), 1);
            if isempty(k)
                error('selectPreMaet:unknownName', ...
                      'No %s is named ''%s''.', what, selector{i});
            end
            idx(i) = k;
        end
        return;
    end
    idx = double(selector(:).');
    if any(idx < 1) || any(idx > n) || any(idx ~= round(idx))
        error('selectPreMaet:outOfRange', ...
              '%s index out of range for %d %ss.', ...
              [upper(what(1)), what(2:end)], n, what);
    end
end
