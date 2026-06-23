function [gamma, sd] = resolveWindowStruct(s, qExtent, axis)
%RESOLVEWINDOWSTRUCT  (gamma, sd) from a struct('shape',..,'width'/'sd',..);
%   an empty struct defaults to a rectangle of qExtent (pass NaN to forbid).
    if isempty(s)
        if ~(qExtent > 0)
            error('mptWindowing:zeroExtent', ...
                ['axis %d: no window given and no query extent to size a ' ...
                 'default from; supply width or sd.'], axis);
        end
        gamma = 1.0; sd = qExtent / (2 * sqrt(3)); return;
    end
    if isfield(s, 'shape'), gamma = internal.resolveShape(s.shape); else, gamma = 1.0; end
    hasSd = isfield(s, 'sd'); hasW = isfield(s, 'width');
    if hasSd == hasW
        error('mptWindowing:sdWidthXor', ...
            'axis %d: give exactly one of width or sd in contextWindow.', axis);
    end
    if hasSd, sd = s.sd; else, sd = s.width / (2 * sqrt(3)); end
    if ~(sd > 0)
        error('mptWindowing:badWidth', 'axis %d: window width/sd must be > 0.', axis);
    end
end
