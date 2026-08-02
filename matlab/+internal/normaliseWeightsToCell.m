function wCell = normaliseWeightsToCell(w, A)
%NORMALISEWEIGHTSTOCELL  Coerce w to a 1 x A cell, preserving entries.
    wCell = cell(1, A);
    if isempty(w) && ~iscell(w)
        for a = 1:A
            wCell{a} = [];
        end
        return;
    end
    if isnumeric(w) && isscalar(w)
        sw = double(w);
        for a = 1:A
            wCell{a} = sw;
        end
        return;
    end
    if iscell(w)
        if numel(w) ~= A
            error('mptWindowing:badWeightCellLength', ...
                  'w cell must have length A = %d; got %d.', A, numel(w));
        end
        for a = 1:A
            wCell{a} = w{a};
        end
        return;
    end
    error('mptWindowing:badWeightType', ...
          ['w must be [], a scalar, or a 1 x A cell of scalar/' ...
           '(1,N)/(K_a,N) entries.']);
end
