function idx = attrIndices(selector, A, names, caller)
%ATTRINDICES  Resolve an attribute selector to an index vector.
%
%   IDX = internal.attrIndices(SELECTOR, A, NAMES, CALLER) takes indices,
%   names as the specs carry them, or a logical mask of length A, and
%   returns the 1-based positions in the order given.

    if islogical(selector)
        if numel(selector) ~= A
            error([caller ':maskLength'], ...
                  'A logical attribute mask must have length %d; got %d.', ...
                  A, numel(selector));
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
                error([caller ':unknownName'], ...
                      'No attribute is named ''%s''.', selector{i});
            end
            idx(i) = k;
        end
        return;
    end
    idx = double(selector(:).');
    if any(idx < 1) || any(idx > A) || any(idx ~= round(idx))
        error([caller ':outOfRange'], ...
              'Attribute index out of range for %d attributes.', A);
    end
end
