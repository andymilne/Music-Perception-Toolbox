function tf = axisIsRel(specs, isRel, a)
%AXISISREL  Outer relative flag for an axis: from specs when nested, else isRel.
    if ~isempty(specs)
        sp = specs{a};
        if isstruct(sp) && isfield(sp, 'rel')
            rel = sp.rel;
            if isscalar(rel), tf = logical(rel); else, tf = logical(rel(end)); end
            return;
        end
        tf = false; return;
    end
    if a <= numel(isRel), tf = logical(isRel(a)); else, tf = false; end
end
