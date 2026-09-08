function tf = preMaetSame(a, b)
%PREMAETSAME  True when two pre-MAETs are the same pre-MAET.
%   Values and weights compare with NaN equal to NaN; specs compare field
%   by field, on the fields both carry.
    tf = false;
    if ~internal.isPreMaet(a) || ~internal.isPreMaet(b)
        return;
    end
    if numel(a.pAttr) ~= numel(b.pAttr)
        return;
    end
    for k = 1:numel(a.pAttr)
        if ~isequaln(a.pAttr{k}, b.pAttr{k})
            return;
        end
    end
    if ~isequaln(a.wAttr, b.wAttr)
        return;
    end
    if isempty(a.specs) ~= isempty(b.specs)
        return;
    end
    for k = 1:numel(a.specs)
        f = fieldnames(a.specs{k});
        if ~isequal(sort(f), sort(fieldnames(b.specs{k})))
            return;
        end
        for j = 1:numel(f)
            va = a.specs{k}.(f{j});
            vb = b.specs{k}.(f{j});
            if ischar(va) || ischar(vb)
                if ~isequal(va, vb)
                    return;
                end
            elseif ~isequaln(double(va(:)'), double(vb(:)'))
                return;
            end
        end
    end
    tf = true;
end
