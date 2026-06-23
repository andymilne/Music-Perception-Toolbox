function [gamma, sd] = singleWindow(cw, qExtent, axis)
%SINGLEWINDOW  (gamma, sd) from a {shape, width} tuple; width defaults to the
%   query extent qExtent (pass NaN when there is no query, forcing explicit).
    shapeRaw = cw{1}; width = cw{2};
    if isempty(width), width = qExtent; end
    if ~(width > 0)
        error('mptWindowing:badWidth', ...
            ['window width on axis %d is zero or undefined; pass an explicit ' ...
             'width (there is no query extent to size it from).'], axis);
    end
    if isempty(shapeRaw), gamma = 1.0; else, gamma = internal.resolveShape(shapeRaw); end
    sd = width / (2 * sqrt(3));
end
