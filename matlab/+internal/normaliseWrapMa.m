function out = normaliseWrapMa(wrap, nAttrs)
%NORMALISEWRAPMA  Validate a multi-attribute ``wrap`` argument.
%
%   OUT = INTERNAL.NORMALISEWRAPMA(WRAP, NATTRS) returns a
%   (nAttrs, 1) cell array of wrap strings for the multi-attribute
%   build path. Accepts:
%
%     - [] (empty)                    -> all 'full-image'
%     - scalar char / string          -> broadcast to every attribute
%     - cellstr / string of length A  -> per-attribute; each entry
%                                        validated against
%                                        {'full-image', 'single-image'}
%     - anything else                 -> error
%
%   Mirror of Python mpt._tensor.build._normalise_wrap_ma.
    values = {'full-image', 'single-image'};
    if isempty(wrap)
        out = repmat({'full-image'}, nAttrs, 1);
        return
    end
    if (ischar(wrap) || (isstring(wrap) && isscalar(wrap)))
        s = char(wrap);
        if ~any(strcmp(s, values))
            error('buildExpTens:invalidWrap', ...
                  'wrap must be ''full-image'' or ''single-image''; got ''%s''.', ...
                  s);
        end
        out = repmat({s}, nAttrs, 1);
        return
    end
    if iscellstr(wrap) || isstring(wrap)
        arr = cellstr(wrap);
        arr = arr(:);
        if numel(arr) ~= nAttrs
            error('buildExpTens:invalidWrap', ...
                  ['wrap array length (%d) does not match number ' ...
                   'of attributes (%d).'], numel(arr), nAttrs);
        end
        for a = 1:nAttrs
            if ~any(strcmp(arr{a}, values))
                error('buildExpTens:invalidWrap', ...
                      ['wrap entries must each be ''full-image'' ' ...
                       'or ''single-image''; got ''%s'' at index %d.'], ...
                      arr{a}, a);
            end
        end
        out = arr;
        return
    end
    error('buildExpTens:invalidWrap', ...
          ['wrap for the multi-attribute path must be a string, a ' ...
           'cell array / string array of length nAttrs, [], or omitted.']);
end
