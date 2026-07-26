function out = normaliseWrapScalar(wrap)
%NORMALISEWRAPSCALAR  Validate a scalar ``wrap`` argument.
%
%   OUT = INTERNAL.NORMALISEWRAPSCALAR(WRAP) returns the wrap string
%   for the single-attribute build path. Accepts:
%
%     - [] (empty)          -> 'full-image' (the default)
%     - char / string       -> validated against {'full-image',
%                              'single-image'}
%     - anything else       -> error
%
%   Mirror of Python mpt._tensor.build._normalise_wrap_scalar.
    if isempty(wrap)
        out = 'full-image';
        return
    end
    if ischar(wrap) || (isstring(wrap) && isscalar(wrap))
        s = char(wrap);
        if ~any(strcmp(s, {'full-image', 'single-image'}))
            error('buildExpTens:invalidWrap', ...
                  'wrap must be ''full-image'' or ''single-image''; got ''%s''.', ...
                  s);
        end
        out = s;
        return
    end
    error('buildExpTens:invalidWrap', ...
          ['wrap for the single-multiset path must be a string ' ...
           '(''full-image'' or ''single-image''), [], or omitted.']);
end
