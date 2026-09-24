function pm = separateAttributes(pAttr, wAttr, attribute, nvArgs)
%SEPARATEATTRIBUTES  Split one attribute into one attribute per slot.
%
%   PM = separateAttributes(PM0, attribute, ...)
%   PM = separateAttributes(pAttr, wAttr, attribute, ...)
%
%   The inverse of bindAttributes, and the operation by which the
%   conversion's two structural roles differ: under 'orderedMultiset'
%   slot k is level k, so splitting that attribute slot by slot gives
%   what the 'separateAttributes' role builds from the table directly.
%
%   Each output attribute holds one row of the input and carries its
%   kernel parameters; r is 1 and exch says nothing, both being
%   determined by there being one value per event.
%
%   attribute is the attribute to split, as an index or a name.
%
%   Name-value pairs
%     'names' - names for the parts, one per slot. The default suffixes
%               the source's name with the 1-based slot position.
%     'specs' - the attribute specifications; [] synthesises flat ones.
%
%   See also BINDATTRIBUTES, SELECTPREMAET, BUILDMAET.

    arguments
        pAttr
        wAttr = []
        attribute = []
        nvArgs.names = {}
        nvArgs.specs = []
    end

    specs = nvArgs.specs;
    if internal.isPreMaet(pAttr)
        pm0 = pAttr;
        if nargin >= 2 && ~isempty(wAttr) && isempty(attribute)
            attribute = wAttr;
        end
        pAttr = pm0.pAttr;
        if isempty(specs); specs = pm0.specs; end
        wAttr = pm0.wAttr;
    end
    if ~iscell(pAttr)
        error('separateAttributes:badPAttr', ...
              'pAttr must be a cell of per-attribute value matrices.');
    end
    A = numel(pAttr);
    if A == 0
        error('separateAttributes:noAttributes', ...
              'pAttr must contain at least one attribute.');
    end
    if isempty(specs); specs = flatSpecs(pAttr); end
    if numel(specs) ~= A
        error('separateAttributes:specsLength', ...
              'specs must be a length-A (%d) cell, one per attribute.', A);
    end
    if isempty(attribute)
        error('separateAttributes:noSelection', ...
              'separateAttributes needs the attribute to split.');
    end

    names = cell(1, A);
    for a = 1:A
        if isfield(specs{a}, 'name'); names{a} = specs{a}.name; end
    end
    idx = internal.attrIndices(attribute, A, names, 'separateAttributes');
    if numel(idx) ~= 1
        error('separateAttributes:oneAttribute', ...
              'separateAttributes splits one attribute; got %d.', numel(idx));
    end
    at = idx(1);
    K = size(pAttr{at}, 1);
    if K < 2
        error('separateAttributes:nothingToSeparate', ...
              ['That attribute holds one value at an event, so there is ' ...
               'nothing to separate.']);
    end

    source = specs{at};
    if isfield(source, 'name') && ~isempty(source.name)
        base = source.name;
    else
        base = sprintf('a_%d', at);
    end
    partNames = nvArgs.names;
    if isempty(partNames)
        partNames = cell(1, K);
        for k = 1:K
            partNames{k} = sprintf('%s_%d', base, k);
        end
    else
        partNames = cellstr(partNames);
        if numel(partNames) ~= K
            error('separateAttributes:namesLength', ...
                  'names must have one entry per slot (%d); got %d.', ...
                  K, numel(partNames));
        end
    end

    carried = {'sigma', 'rel', 'isPer', 'period'};
    pOut = {};
    wOut = {};
    specsOut = {};
    for a = 1:A
        if a ~= at
            pOut{end + 1} = pAttr{a}; %#ok<AGROW>
            specsOut{end + 1} = specs{a}; %#ok<AGROW>
            if ~isempty(wAttr); wOut{end + 1} = wAttr{a}; end %#ok<AGROW>
            continue;
        end
        for k = 1:K
            spec = struct('name', partNames{k}, 'r', 1, 'exch', true);
            for f = 1:numel(carried)
                if isfield(source, carried{f})
                    spec.(carried{f}) = source.(carried{f});
                end
            end
            pOut{end + 1} = pAttr{at}(k, :); %#ok<AGROW>
            specsOut{end + 1} = spec; %#ok<AGROW>
            if ~isempty(wAttr); wOut{end + 1} = wAttr{at}(k, :); end %#ok<AGROW>
        end
    end
    % No weights in means no weights out: an empty cell is a
    % length-zero list of them, which is a different thing.
    if isempty(wAttr); wOut = []; end
    pm = packPreMaet(pOut, wOut, specsOut);
end
