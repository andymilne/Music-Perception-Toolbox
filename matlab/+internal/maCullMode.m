function mode = maCullMode(newMode)
%MACULLMODE  Testing lever for culling on the joint-centres path.
%
%   MODE = INTERNAL.MACULLMODE() returns the current setting: 'auto'
%   (the default) culls where internal.maCullPlan's estimate favours it,
%   while 'never' and 'always' force the choice. PREV =
%   INTERNAL.MACULLMODE(NEWMODE) sets it and returns the previous
%   setting. Not part of the public interface: the tests use it to
%   compare culled with dense evaluation, which must agree to rounding.
%   'clear all' restores the default.
%
%   Twin of the Python module constant _MA_CULL_MODE in
%   mpt/_tensor/eval.py.
%
%   See also INTERNAL.MACULLPLAN, EVALMAET.

    persistent current
    if isempty(current)
        current = 'auto';
    end
    mode = current;
    if nargin > 0
        newMode = char(newMode);
        if ~any(strcmp(newMode, {'auto', 'never', 'always'}))
            error('internal:maCullMode', ...
                  'maCullMode must be ''auto'', ''never'', or ''always''.');
        end
        current = newMode;
    end
end
