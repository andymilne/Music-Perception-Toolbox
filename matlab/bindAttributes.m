function pm = bindAttributes(pAttr, wAttr, attributes, nvArgs)
%BINDATTRIBUTES  Gather several attributes into one whose tuple holds them all.
%
%   PM = bindAttributes(PM0, attributes, ...)
%   PM = bindAttributes(pAttr, wAttr, attributes, ...)
%
%   The attribute-axis counterpart of bindEvents, which binds along the
%   event axis. Where three columns carry the three coordinates of one
%   position, or the coordinates of a simplex-coded level, they are three
%   attributes of a pre-MAET and their product pairs each with every
%   other; binding them makes them one attribute whose value at an event
%   is the tuple of all of them, read together.
%
%   The bound attribute takes the place of the first of its inputs; the
%   attributes not listed keep their order around it.
%
%   attributes are the attributes to bind, as indices or names, in the
%   order their values are to be read. That order is what 'exch', false
%   makes significant.
%
%   Name-value pairs
%     'name'   - the bound attribute's name. Required, since no input's
%                name describes the result.
%     'r'      - how many of the bound tuple's values a tuple takes.
%                Required: it is a claim about what the values mean and
%                has no identity. r = sum(K_a) reads the whole tuple as
%                one object, which is what a coordinate vector wants.
%     'exch'   - whether the bound values are exchangeable. Required, and
%                normally false, the point of binding being that position
%                signifies.
%     'sigma', 'rel', 'isPer', 'period'
%              - the bound attribute's kernel parameters. Each is
%                inherited where every input agrees on it and required
%                where they differ, since there is no reading of a
%                periodic value bound to a non-periodic one that the call
%                has not chosen.
%     'specs'  - the attribute specifications; [] synthesises flat ones.
%
%   See also SEPARATEATTRIBUTES, BINDEVENTS, SELECTPREMAET, BUILDMAET.

    arguments
        pAttr
        wAttr = []
        attributes = []
        nvArgs.name = ''
        nvArgs.r = []
        nvArgs.exch = []
        nvArgs.sigma = []
        nvArgs.rel = []
        nvArgs.isPer = []
        nvArgs.period = []
        nvArgs.specs = []
    end

    specs = nvArgs.specs;
    if internal.isPreMaet(pAttr)
        pm0 = pAttr;
        % The whole-pre-MAET form shifts the positional arguments back.
        if nargin >= 2 && ~isempty(wAttr) && isempty(attributes)
            attributes = wAttr;
        end
        pAttr = pm0.pAttr;
        if isempty(specs); specs = pm0.specs; end
        wAttr = pm0.wAttr;
    end
    if ~iscell(pAttr)
        error('bindAttributes:badPAttr', ...
              'pAttr must be a cell of per-attribute value matrices.');
    end
    A = numel(pAttr);
    if A == 0
        error('bindAttributes:noAttributes', ...
              'pAttr must contain at least one attribute.');
    end
    if isempty(specs); specs = flatSpecs(pAttr); end
    if numel(specs) ~= A
        error('bindAttributes:specsLength', ...
              'specs must be a length-A (%d) cell, one per attribute.', A);
    end
    if isempty(attributes)
        error('bindAttributes:noSelection', ...
              ['bindAttributes needs the attributes to bind; binding is a ' ...
               'choice about which values belong to one another.']);
    end

    names = cell(1, A);
    for a = 1:A
        if isfield(specs{a}, 'name'); names{a} = specs{a}.name; end
    end
    idx = internal.attrIndices(attributes, A, names, 'bindAttributes');
    if numel(idx) < 2
        error('bindAttributes:oneAttribute', ...
              ['Binding needs at least two attributes; one attribute is ' ...
               'already its own tuple.']);
    end
    if numel(unique(idx)) ~= numel(idx)
        error('bindAttributes:repeated', ...
              'An attribute cannot be bound to itself; list each once.');
    end
    if isempty(nvArgs.name)
        error('bindAttributes:noName', ...
              ['bindAttributes needs a name: none of the bound attributes'' ' ...
               'names describes the result.']);
    end
    if isempty(nvArgs.r) || isempty(nvArgs.exch)
        missing = {};
        if isempty(nvArgs.r); missing{end + 1} = 'r'; end
        if isempty(nvArgs.exch); missing{end + 1} = 'exch'; end
        K = 0;
        for i = 1:numel(idx); K = K + size(pAttr{idx(i)}, 1); end
        error('bindAttributes:noTupleSize', ...
              ['bindAttributes needs %s. The bound attribute holds %d ' ...
               'values at an event: r says how many of them a tuple takes, ' ...
               'and exch whether their order signifies. Binding to read a ' ...
               'coordinate vector whole takes r = sum of the inputs'' K and ' ...
               'exch = false.'], strjoin(missing, ' and '), K);
    end

    spec = struct('name', nvArgs.name, 'r', double(nvArgs.r), ...
                  'exch', logical(nvArgs.exch));
    fields = {'sigma', 'rel', 'isPer', 'period'};
    given = {nvArgs.sigma, nvArgs.rel, nvArgs.isPer, nvArgs.period};
    for f = 1:numel(fields)
        if ~isempty(given{f})
            spec.(fields{f}) = given{f};
            continue;
        end
        [value, agreed] = localInherit(specs, idx, fields{f});
        if ~agreed
            error('bindAttributes:disagree', ...
                  ['The bound attributes disagree on %s, so the bound one ' ...
                   'has no value to inherit; give it in the call.'], fields{f});
        end
        if ~isempty(value); spec.(fields{f}) = value; end
    end
    if isfield(spec, 'isPer') && spec.isPer ...
            && (~isfield(spec, 'period') || spec.period == 0)
        error('bindAttributes:noPeriod', ...
              '%s: isPer is set, so it needs a period.', spec.name);
    end

    boundP = vertcat(pAttr{idx});
    boundW = [];
    if ~isempty(wAttr); boundW = vertcat(wAttr{idx}); end

    pOut = {};
    wOut = {};
    specsOut = {};
    for a = 1:A
        if a == idx(1)
            pOut{end + 1} = boundP; %#ok<AGROW>
            specsOut{end + 1} = spec; %#ok<AGROW>
            if ~isempty(wAttr); wOut{end + 1} = boundW; end %#ok<AGROW>
        elseif ~any(a == idx)
            pOut{end + 1} = pAttr{a}; %#ok<AGROW>
            specsOut{end + 1} = specs{a}; %#ok<AGROW>
            if ~isempty(wAttr); wOut{end + 1} = wAttr{a}; end %#ok<AGROW>
        end
    end
    % No weights in means no weights out: an empty cell is a
    % length-zero list of them, which is a different thing.
    if isempty(wAttr); wOut = []; end
    pm = packPreMaet(pOut, wOut, specsOut);
end


function [value, agreed] = localInherit(specs, idx, field)
    % The inputs' shared value of one field, and whether they share one.
    value = [];
    agreed = true;
    have = false;
    for i = 1:numel(idx)
        s = specs{idx(i)};
        if ~isfield(s, field)
            if have; agreed = false; return; end
            continue;
        end
        if ~have
            value = s.(field);
            have = true;
        elseif ~isequal(value, s.(field))
            agreed = false;
            return;
        end
    end
end
