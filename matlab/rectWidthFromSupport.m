function width = rectWidthFromSupport(totalSupport)
%RECTWIDTHFROMSUPPORT  Convert a desired full rect-window support into width.
%
%   width = rectWidthFromSupport(totalSupport) returns the width
%   argument to pass to weightEvents to obtain a rectangle (shape = 1)
%   of the given full support.
%
%   The window family in weightEvents is parameterised so that total
%   variance equals width^2 for every shape value in [0, 1]. For a
%   pure rectangle (shape = 1) the indicator is on
%   abs(delta) <= width * sqrt(3), so the full support is
%   L = 2 * width * sqrt(3), and the variance of a uniform on
%   [-L/2, L/2] is L^2 / 12; equating that to width^2 gives
%   width = L / (2 * sqrt(3)).
%
%   Use this when the natural specification is the full support of the
%   rectangle. For example, a rect window covering one 16th-note grid
%   step in quarter-note units has totalSupport = 0.25, hence
%   width = 0.25 / (2*sqrt(3)) ~= 0.0722.
%
%   For a pure Gaussian (shape = 0), width is already the standard
%   deviation, so no conversion is needed.
%
%   See also: weightEvents.

    if ~isnumeric(totalSupport) || ~isscalar(totalSupport) ...
            || ~isfinite(totalSupport) || totalSupport <= 0
        error('rectWidthFromSupport:badInput', ...
              'totalSupport must be a positive finite scalar; got %s.', ...
              mat2str(totalSupport));
    end
    width = double(totalSupport) / (2.0 * sqrt(3.0));
end
