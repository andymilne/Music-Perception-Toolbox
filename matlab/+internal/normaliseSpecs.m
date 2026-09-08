function [rVec, isRelVec, isSymVec, nestedList, names, specKernel] = ...
        normaliseSpecs(specs, A)
    %NORMALISESPECS  Unpack a per-attribute specs cell into geometry.
    %   specs is the canonical home for level-structured geometry (§6.4):
    %   a cell of per-attribute structs. A flat attribute is a one-level
    %   spec struct('r',.,'rel',.,'sym',.,'name',.) (scalar r, bool
    %   rel/sym); a nested attribute carries a 'tags' field plus per-level
    %   vectors. The presence of 'tags' is the flat-vs-nested discriminant.
    %   A spec may also carry the attribute's scalar kernel geometry
    %   (sigma, isPer, period), which is not level-structured but is part
    %   of the pre-MAET as Milne (2026, Def. 2.6) defines it. Each is
    %   optional in the pre-MAET and compulsory at the tensor: an operator
    %   that cannot carry one forward writes NA (NaN) rather than a stale
    %   or invented value, and buildExpTens refuses an NA it is not given
    %   explicitly.
    %   Returns rVec/isRelVec/isSymVec (placeholders for nested entries,
    %   overridden by the nested machinery), nestedList (cell, [] = flat),
    %   names (cell of attribute names, [] = unnamed), and specKernel (a
    %   struct of three 1 x A cells: the spec's value, [] where the field
    %   is absent, NaN where it is NA).
    if ~iscell(specs)
        error('buildExpTens:specsType', ...
              'specs must be a cell array of per-attribute spec structs.');
    end
    if numel(specs) ~= A
        error('buildExpTens:specsLength', ...
              'specs must have length %d (one per attribute), got %d.', ...
              A, numel(specs));
    end
    rVec = zeros(1, A); isRelVec = false(1, A); isSymVec = true(1, A);
    nestedList = cell(1, A); names = cell(1, A);
    specKernel = struct('sigma', {cell(1, A)}, 'isPer', {cell(1, A)}, ...
                        'period', {cell(1, A)});
    kFields = {'sigma', 'isPer', 'period'};
    kAliases = {'sigma', 'is_per', 'period'};
    for a = 1:A
        for kf = 1:numel(kFields)
            if isfield(specs{a}, kFields{kf})
                specKernel.(kFields{kf}){a} = specs{a}.(kFields{kf});
            elseif isfield(specs{a}, kAliases{kf})
                specKernel.(kFields{kf}){a} = specs{a}.(kAliases{kf});
            end
        end
    end
    for a = 1:A
        s = specs{a};
        if ~isstruct(s)
            error('buildExpTens:specsEntry', 'specs{%d} must be a struct.', a);
        end
        if isfield(s, 'name')
            names{a} = s.name;
        else
            names{a} = [];
        end
        if isfield(s, 'tags')
            nestedList{a} = s;       % nested machinery derives r/rel/sym
            rVec(a)     = 1;         % placeholder -> prod(level r)
            isRelVec(a) = false;     % placeholder -> resolved projection
            isSymVec(a) = true;      % placeholder -> per-level sym
        else
            if ~isfield(s, 'r')
                error('buildExpTens:specsFlatR', ...
                      'specs{%d} (flat) must have an ''r'' field.', a);
            end
            rA = s.r(:).';
            if numel(rA) ~= 1
                error('buildExpTens:specsFlatRVec', ...
                      ['specs{%d}: ''r'' is a multi-element vector but the ' ...
                       'spec has no ''tags'' field. A per-level ''r'' denotes ' ...
                       'a nested spec, which must also carry ''tags'' (the ' ...
                       'value-to-level map).'], a);
            end
            nestedList{a} = [];
            rVec(a) = double(rA(1));
            if isfield(s, 'rel'), isRelVec(a) = logical(s.rel); end
            if isfield(s, 'sym'), isSymVec(a) = logical(s.sym); end
        end
    end
end
